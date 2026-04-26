"""
Experiment 2: VMIB Compression-Prediction Tradeoff
===================================================
Trains VMIB models at varying lambda (compression strength),
plots the real information plane trajectory I(Z;X^{1:M}) vs I(Z;Y).

Uses:
  - KL(q(z|x) || N(0,I)) as upper bound on I(Z; X^{1:M})
  - H(Y) - CE(Y|Z) as lower bound on I(Z; Y)
Also tracks per-modality retention I(Z; X^(i)) via probing.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, get_loaders, train_vmib, evaluate, save_results,
    MODALITY_NAMES, DEVICE
)


def estimate_modality_retention(model, loader, modality_idx, input_dim,
                                 latent_dim=32, epochs=80):
    """Estimate I(Z; X^(i)) via probing: train a decoder Z -> X^(i).

    Uses reconstruction MSE as a proxy:
        I(Z; X^(i)) ~ 0.5 * D * log(var(X^(i))) - 0.5 * D * log(MSE)
    where D is the dimension. This gives a relative ordering.

    More practically, we report the R^2 between reconstructed and true X^(i).
    """
    model.eval()
    # Collect Z and X^(i)
    all_z, all_x = [], []
    with torch.no_grad():
        for xs, y in loader:
            xs = [x.to(DEVICE) for x in xs]
            _, mu, _, _ = model(xs)
            all_z.append(mu.cpu())
            all_x.append(xs[modality_idx].cpu())

    Z = torch.cat(all_z)
    X = torch.cat(all_x)

    # Train a simple linear decoder Z -> X^(i)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, input_dim),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    dataset = TensorDataset(Z.to(DEVICE), X.to(DEVICE))
    probe_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_mse = float("inf")
    for epoch in range(epochs):
        decoder.train()
        for zb, xb in probe_loader:
            x_hat = decoder(zb)
            loss = F.mse_loss(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        decoder.eval()
        with torch.no_grad():
            x_hat = decoder(Z.to(DEVICE))
            mse = F.mse_loss(x_hat, X.to(DEVICE)).item()
            if mse < best_mse:
                best_mse = mse

    # R^2 = 1 - MSE / Var(X)
    x_var = X.var(dim=0).mean().item()
    r2 = max(0, 1.0 - best_mse / (x_var + 1e-10))

    # MI proxy: 0.5 * D * log(Var(X) / MSE), clamped at 0
    D = input_dim
    mi_proxy = max(0, 0.5 * D * np.log(max(x_var, 1e-10) / max(best_mse, 1e-10)))

    return {"r2": r2, "mse": best_mse, "x_var": x_var, "mi_proxy": mi_proxy}


def main():
    print("=" * 70)
    print("Experiment 2: VMIB Compression-Prediction Tradeoff")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_data, test_data, input_dims, num_classes, _ = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    # H(Y) from training labels
    y_train = train_data.labels.numpy()
    _, counts = np.unique(y_train, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))
    print(f"H(Y) = {H_Y:.4f} nats")

    # Sweep lambda values
    lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = {"H_Y": H_Y, "lambda_values": lambda_values, "sweeps": []}

    for lam in lambda_values:
        print(f"\n--- lambda = {lam} ---")
        model, history = train_vmib(
            input_dims, num_classes, train_loader, test_loader,
            lambda_kl=lam, gamma_consist=0.0, hidden_dim=256,
            latent_dim=32, epochs=100, lr=1e-3, verbose=False
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, lam)
        train_metrics_eval = evaluate(model, train_loader, lam)

        # I(Z; X^{1:M}) upper bound = KL
        I_ZX_upper = test_metrics["kl"]
        # I(Z; Y) lower bound = H(Y) - CE
        I_ZY_lower = max(0, H_Y - test_metrics["ce"])

        print(f"  I(Z;X) <= {I_ZX_upper:.4f} | I(Z;Y) >= {I_ZY_lower:.4f} | "
              f"Acc={test_metrics['acc']:.4f} | AUC={test_metrics['auc']:.4f}")

        # Per-modality retention probing
        modality_retention = {}
        for i in range(3):
            ret = estimate_modality_retention(
                model, test_loader, i, input_dims[i],
                latent_dim=32, epochs=80
            )
            modality_retention[MODALITY_NAMES[i]] = ret
            print(f"  I(Z;{MODALITY_NAMES[i]}): R²={ret['r2']:.4f}, "
                  f"MI_proxy={ret['mi_proxy']:.2f}")

        sweep_result = {
            "lambda": lam,
            "I_ZX_upper": I_ZX_upper,
            "I_ZY_lower": I_ZY_lower,
            "test_acc": test_metrics["acc"],
            "test_auc": test_metrics["auc"],
            "test_ce": test_metrics["ce"],
            "train_acc": train_metrics_eval["acc"],
            "pred_entropy": test_metrics["pred_entropy"],
            "modality_retention": {
                name: {"r2": ret["r2"], "mi_proxy": ret["mi_proxy"]}
                for name, ret in modality_retention.items()
            },
        }
        results["sweeps"].append(sweep_result)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INFORMATION PLANE TRAJECTORY")
    print("=" * 70)
    print(f"{'Lambda':>10} | {'I(Z;X) UB':>10} | {'I(Z;Y) LB':>10} | {'Acc':>7} | {'AUC':>7}")
    print("-" * 55)
    for s in results["sweeps"]:
        print(f"{s['lambda']:>10.4f} | {s['I_ZX_upper']:>10.4f} | "
              f"{s['I_ZY_lower']:>10.4f} | {s['test_acc']:>7.4f} | {s['test_auc']:>7.4f}")

    save_results(results, "exp2_vmib_tradeoff.json")
    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    main()
