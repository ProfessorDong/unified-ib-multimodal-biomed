"""
Experiment 0: Synthetic Synergy via Interaction Model
=====================================================
Generates data from a genuinely synergistic model where Y depends on
the product X^(1)*X^(2), creating predictive information accessible
only through joint observation.

Model:
  X^(1) ~ N(0,1), X^(2) ~ N(0,1) independent
  Y | X ~ Bernoulli(sigma(beta_1*X^(1) + beta_2*X^(2) + alpha*X^(1)*X^(2)))

When alpha=0: additive model, S_12 <= 0 (no synergy)
When alpha>0: interaction term creates genuine synergy, S_12 > 0

MI estimation uses a single masked MLP to ensure consistent bounds:
  I(X^(S);Y) ~ H(Y) - CE_model(Y|X^(S))
where CE is evaluated from the SAME model on all subsets.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_results, DEVICE


class MaskedMLP(nn.Module):
    """MLP that accepts modality masks for consistent MI estimation."""

    def __init__(self, input_dim=2, hidden_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(0).float().to(x.device)
        return self.net(x)


def generate_data(n, beta1, beta2, alpha, seed=42):
    """Generate synthetic data from interaction model."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    logit = beta1 * x1 + beta2 * x2 + alpha * x1 * x2
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob)
    X = np.column_stack([x1, x2]).astype(np.float32)
    return X, y.astype(np.int64)


def train_masked_model(X_train, y_train, hidden_dim=64, epochs=200,
                       lr=1e-3, dropout_prob=0.3):
    """Train a masked MLP with modality dropout for consistent MI estimation."""
    model = MaskedMLP(input_dim=2, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            # Random modality dropout during training
            mask = torch.ones(2, device=DEVICE)
            if np.random.rand() < dropout_prob:
                drop_idx = np.random.randint(2)
                mask[drop_idx] = 0.0

            logits = model(xb, mask=mask)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def estimate_mi_consistent(model, X_test, y_test):
    """Estimate MI for all subsets using the SAME masked model.

    Returns dict with MI estimates for X1, X2, and X1+X2.
    """
    model.eval()
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    y_t = torch.LongTensor(y_test).to(DEVICE)

    # H(Y)
    _, counts = np.unique(y_test, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))

    results = {}
    subsets = {
        "X1": torch.tensor([1.0, 0.0]),
        "X2": torch.tensor([0.0, 1.0]),
        "X1+X2": torch.tensor([1.0, 1.0]),
    }

    with torch.no_grad():
        for name, mask in subsets.items():
            logits = model(X_t, mask=mask)
            ce = F.cross_entropy(logits, y_t).item()
            mi = max(0, H_Y - ce)
            results[name] = {"mi": mi, "ce": ce}

    results["H_Y"] = H_Y
    return results


def main():
    print("=" * 70)
    print("Experiment 0: Synthetic Synergy via Interaction Model")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    N_train, N_test = 5000, 2000
    beta1, beta2 = 0.5, 0.5  # weak individual signals
    alpha_values = np.linspace(0, 3.0, 13)
    seeds = [42, 123, 456, 789, 1024]

    results = {
        "model": "Y ~ Bernoulli(sigma(beta1*X1 + beta2*X2 + alpha*X1*X2))",
        "beta1": beta1, "beta2": beta2,
        "N_train": N_train, "N_test": N_test,
        "alpha_values": alpha_values.tolist(),
        "sweeps": [],
    }

    print(f"\nModel: Y ~ Bernoulli(sigma({beta1}*X1 + {beta2}*X2 + alpha*X1*X2))")
    print(f"Samples: {N_train} train, {N_test} test")
    print(f"Seeds: {seeds}")
    print(f"\n{'alpha':>6} | {'I(X1;Y)':>8} | {'I(X2;Y)':>8} | {'I(X1,X2;Y)':>11} | {'S_12':>8}")
    print("-" * 55)

    for alpha in alpha_values:
        all_I1, all_I2, all_I12, all_S = [], [], [], []

        for seed in seeds:
            X_train, y_train = generate_data(N_train, beta1, beta2, alpha, seed=seed)
            X_test, y_test = generate_data(N_test, beta1, beta2, alpha, seed=seed + 10000)

            model = train_masked_model(X_train, y_train, hidden_dim=64,
                                        epochs=200, dropout_prob=0.3)
            mi = estimate_mi_consistent(model, X_test, y_test)

            I1 = mi["X1"]["mi"]
            I2 = mi["X2"]["mi"]
            I12 = mi["X1+X2"]["mi"]
            S = I12 - I1 - I2

            all_I1.append(I1)
            all_I2.append(I2)
            all_I12.append(I12)
            all_S.append(S)

        sweep = {
            "alpha": float(alpha),
            "I_X1_Y": {"mean": np.mean(all_I1), "std": np.std(all_I1)},
            "I_X2_Y": {"mean": np.mean(all_I2), "std": np.std(all_I2)},
            "I_X1X2_Y": {"mean": np.mean(all_I12), "std": np.std(all_I12)},
            "S_12": {"mean": np.mean(all_S), "std": np.std(all_S)},
        }
        results["sweeps"].append(sweep)

        print(f"{alpha:>6.2f} | {np.mean(all_I1):>8.4f} | {np.mean(all_I2):>8.4f} | "
              f"{np.mean(all_I12):>11.4f} | {np.mean(all_S):>+8.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    s_at_0 = results["sweeps"][0]["S_12"]["mean"]
    s_at_max = results["sweeps"][-1]["S_12"]["mean"]
    print(f"S_12 at alpha=0: {s_at_0:+.4f} (should be <= 0, additive/redundant)")
    print(f"S_12 at alpha={alpha_values[-1]:.1f}: {s_at_max:+.4f} (should be > 0, synergistic)")

    # Find alpha where S crosses zero
    for sweep in results["sweeps"]:
        if sweep["S_12"]["mean"] > 0:
            print(f"S_12 first positive at alpha={sweep['alpha']:.2f}")
            break

    save_results(results, "exp0_synthetic_synergy.json")
    print("\nExperiment 0 complete.")


if __name__ == "__main__":
    main()
