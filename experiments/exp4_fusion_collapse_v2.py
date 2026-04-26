"""
Experiment 4 (v2): Fusion Collapse Diagnostics
================================================
Refined version:
  - Uses AUC-based G_i (more stable than CE-based)
  - Tests multiple lambda values to show collapse progression
  - Compares standard vs balanced training (modality dropout)
  - Computes balance index from G_i distribution
  - Multiple seeds for robustness

G_i (AUC-based) = AUC(all modalities) - AUC(modality i ablated)
  Interpretation: how much predictive performance depends on modality i
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, get_loaders, VMIBModel, vmib_loss, evaluate, save_results,
    MODALITY_NAMES, NUM_MODALITIES, DEVICE
)


def train_model(input_dims, num_classes, train_loader, lambda_kl=0.01,
                modality_dropout=0.0, epochs=150, lr=1e-3, seed=42):
    """Train a VMIB model, optionally with modality dropout."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VMIBModel(input_dims, hidden_dim=256, latent_dim=32,
                      num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for xs, y in train_loader:
            xs = [x.to(DEVICE) for x in xs]
            y = y.to(DEVICE)

            mask = None
            if modality_dropout > 0:
                mask = [torch.rand(1).item() > modality_dropout
                        for _ in range(NUM_MODALITIES)]
                if not any(mask):
                    mask[np.random.randint(NUM_MODALITIES)] = True

            logits, mu, logvar, z = model(xs, modality_mask=mask)
            loss, _, _ = vmib_loss(logits, y, mu, logvar, lambda_kl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    return model


def compute_gaps(model, test_loader, lambda_kl):
    """Compute AUC-based modality gaps G_i."""
    # Full evaluation
    full = evaluate(model, test_loader, lambda_kl)
    full_auc = full["auc"]
    full_acc = full["acc"]

    gaps = {}
    for i in range(NUM_MODALITIES):
        mask = [True] * NUM_MODALITIES
        mask[i] = False
        ablated = evaluate(model, test_loader, lambda_kl, modality_mask=mask)
        G_i = max(0, full_auc - ablated["auc"])
        gaps[MODALITY_NAMES[i]] = {
            "G_i_auc": G_i,
            "auc_full": full_auc,
            "auc_ablated": ablated["auc"],
        }

    return gaps, full_auc, full_acc


def balance_index(gaps):
    """Normalized entropy of G_i distribution. 1.0 = perfectly balanced."""
    G_vals = np.array([g["G_i_auc"] for g in gaps.values()])
    total = G_vals.sum()
    if total < 1e-8:
        return 1.0  # if all gaps are 0, model doesn't rely on any one modality
    G_norm = G_vals / total
    ent = -np.sum(G_norm * np.log(G_norm + 1e-10))
    return ent / np.log(len(G_vals))


def main():
    print("=" * 70)
    print("Experiment 4 (v2): Fusion Collapse Diagnostics")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_data, test_data, input_dims, num_classes, _ = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    results = {"experiments": []}
    seeds = [42, 123, 456]

    # ── A: Standard training at different lambda values ───────────────
    print("\n=== Part A: Standard Training (varying lambda) ===")
    lambda_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

    for lam in lambda_values:
        all_gaps = {name: [] for name in MODALITY_NAMES}
        all_aucs = []
        all_balance = []

        for seed in seeds:
            model = train_model(input_dims, num_classes, train_loader,
                                lambda_kl=lam, modality_dropout=0.0,
                                epochs=120, seed=seed)
            gaps, full_auc, full_acc = compute_gaps(model, test_loader, lam)
            bi = balance_index(gaps)

            for name in MODALITY_NAMES:
                all_gaps[name].append(gaps[name]["G_i_auc"])
            all_aucs.append(full_auc)
            all_balance.append(bi)

        mean_gaps = {name: np.mean(all_gaps[name]) for name in MODALITY_NAMES}
        std_gaps = {name: np.std(all_gaps[name]) for name in MODALITY_NAMES}

        result = {
            "type": "standard",
            "lambda": lam,
            "modality_dropout": 0.0,
            "full_auc_mean": np.mean(all_aucs),
            "full_auc_std": np.std(all_aucs),
            "balance_mean": np.mean(all_balance),
            "balance_std": np.std(all_balance),
            "gaps_mean": mean_gaps,
            "gaps_std": std_gaps,
        }
        results["experiments"].append(result)

        gap_str = ", ".join(f"{name}={mean_gaps[name]:.4f}±{std_gaps[name]:.4f}"
                           for name in MODALITY_NAMES)
        print(f"  lambda={lam:.3f}: AUC={np.mean(all_aucs):.4f} | "
              f"Balance={np.mean(all_balance):.3f} | G=[{gap_str}]")

    # ── B: Modality dropout training ──────────────────────────────────
    print("\n=== Part B: Modality Dropout Training (lambda=0.01) ===")
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

    for dp in dropout_rates:
        all_gaps = {name: [] for name in MODALITY_NAMES}
        all_aucs = []
        all_balance = []

        for seed in seeds:
            model = train_model(input_dims, num_classes, train_loader,
                                lambda_kl=0.01, modality_dropout=dp,
                                epochs=120, seed=seed)
            gaps, full_auc, _ = compute_gaps(model, test_loader, 0.01)
            bi = balance_index(gaps)

            for name in MODALITY_NAMES:
                all_gaps[name].append(gaps[name]["G_i_auc"])
            all_aucs.append(full_auc)
            all_balance.append(bi)

        mean_gaps = {name: np.mean(all_gaps[name]) for name in MODALITY_NAMES}
        std_gaps = {name: np.std(all_gaps[name]) for name in MODALITY_NAMES}

        result = {
            "type": "modality_dropout",
            "lambda": 0.01,
            "modality_dropout": dp,
            "full_auc_mean": np.mean(all_aucs),
            "full_auc_std": np.std(all_aucs),
            "balance_mean": np.mean(all_balance),
            "balance_std": np.std(all_balance),
            "gaps_mean": mean_gaps,
            "gaps_std": std_gaps,
        }
        results["experiments"].append(result)

        gap_str = ", ".join(f"{name}={mean_gaps[name]:.4f}" for name in MODALITY_NAMES)
        print(f"  dropout={dp:.1f}: AUC={np.mean(all_aucs):.4f} | "
              f"Balance={np.mean(all_balance):.3f} | G=[{gap_str}]")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FUSION COLLAPSE SUMMARY")
    print("=" * 70)
    print("\nKey findings:")

    # Find most collapsed and most balanced configs
    std_results = [r for r in results["experiments"] if r["type"] == "standard"]
    dp_results = [r for r in results["experiments"] if r["type"] == "modality_dropout"]

    most_collapsed = min(std_results, key=lambda r: r["balance_mean"])
    most_balanced_std = max(std_results, key=lambda r: r["balance_mean"])
    most_balanced_dp = max(dp_results, key=lambda r: r["balance_mean"])

    print(f"  Most collapsed: lambda={most_collapsed['lambda']}, "
          f"balance={most_collapsed['balance_mean']:.3f}")
    print(f"  Most balanced (standard): lambda={most_balanced_std['lambda']}, "
          f"balance={most_balanced_std['balance_mean']:.3f}")
    print(f"  Most balanced (dropout): dropout={most_balanced_dp['modality_dropout']}, "
          f"balance={most_balanced_dp['balance_mean']:.3f}")

    save_results(results, "exp4_fusion_collapse_v2.json")
    print("\nExperiment 4 (v2) complete.")


if __name__ == "__main__":
    main()
