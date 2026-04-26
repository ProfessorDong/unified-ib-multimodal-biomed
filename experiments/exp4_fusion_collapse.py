"""
Experiment 4: Fusion Collapse Diagnostics
==========================================
Computes the modality-conditioned predictive gap G_i for trained models.
Shows that naively trained models exhibit fusion collapse (G_i concentrated
in one modality), while VMIB-regularized models distribute reliance.

G_i = I(Z; Y) - I(Z; Y | X^(i))

Estimated as:
  G_i ≈ AUC(full) - AUC(without modality i)
  (normalized by AUC(full) for interpretability)

Also computes information-theoretic G_i via:
  G_i = [H(Y) - CE(Y|Z)] - [H(Y) - CE(Y|Z, ablating modality i)]
      = CE(Y|Z, ablated) - CE(Y|Z, full)
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, get_loaders, train_vmib, evaluate, save_results,
    VMIBModel, MODALITY_NAMES, NUM_MODALITIES, DEVICE
)


def compute_predictive_gaps(model, test_loader, lambda_kl, H_Y):
    """Compute G_i for each modality.

    G_i = I(Z;Y) - I(Z;Y|X^(i)) = CE(Y|Z, ablate i) - CE(Y|Z, full)
    """
    # Full evaluation
    full_eval = evaluate(model, test_loader, lambda_kl)
    I_ZY_full = max(0, H_Y - full_eval["ce"])

    gaps = {}
    for i in range(NUM_MODALITIES):
        # Ablate modality i (set to zero)
        mask = [True] * NUM_MODALITIES
        mask[i] = False
        ablated_eval = evaluate(model, test_loader, lambda_kl, modality_mask=mask)
        I_ZY_ablated = max(0, H_Y - ablated_eval["ce"])

        G_i = max(0, I_ZY_full - I_ZY_ablated)

        gaps[MODALITY_NAMES[i]] = {
            "G_i": G_i,
            "I_ZY_full": I_ZY_full,
            "I_ZY_ablated": I_ZY_ablated,
            "auc_full": full_eval["auc"],
            "auc_ablated": ablated_eval["auc"],
            "auc_drop": full_eval["auc"] - ablated_eval["auc"],
            "ce_full": full_eval["ce"],
            "ce_ablated": ablated_eval["ce"],
        }

    return gaps, full_eval


def main():
    print("=" * 70)
    print("Experiment 4: Fusion Collapse Diagnostics")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_data, test_data, input_dims, num_classes, _ = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    # H(Y)
    y_train = train_data.labels.numpy()
    _, counts = np.unique(y_train, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))
    print(f"H(Y) = {H_Y:.4f} nats")

    results = {"H_Y": H_Y, "models": {}}

    # ── Model 1: Standard (no IB regularization) ─────────────────────
    # This should be more prone to fusion collapse
    print("\n--- Training Standard Model (lambda=0, no regularization) ---")
    model_standard, _ = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.0, gamma_consist=0.0, hidden_dim=256,
        latent_dim=32, epochs=120, lr=1e-3,
    )
    gaps_std, full_std = compute_predictive_gaps(
        model_standard, test_loader, 0.0, H_Y
    )
    results["models"]["standard"] = {
        "description": "No IB regularization (lambda=0)",
        "gaps": gaps_std,
        "full_auc": full_std["auc"],
        "full_acc": full_std["acc"],
    }

    print("\nStandard model G_i:")
    for name, g in gaps_std.items():
        print(f"  G_{name}: {g['G_i']:.4f} | AUC drop: {g['auc_drop']:+.4f}")

    # ── Model 2: Weak VMIB ───────────────────────────────────────────
    print("\n--- Training Weak VMIB (lambda=0.001) ---")
    model_weak, _ = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.001, gamma_consist=0.0, hidden_dim=256,
        latent_dim=32, epochs=120, lr=1e-3,
    )
    gaps_weak, full_weak = compute_predictive_gaps(
        model_weak, test_loader, 0.001, H_Y
    )
    results["models"]["weak_vmib"] = {
        "description": "Weak IB regularization (lambda=0.001)",
        "gaps": gaps_weak,
        "full_auc": full_weak["auc"],
        "full_acc": full_weak["acc"],
    }

    print("\nWeak VMIB G_i:")
    for name, g in gaps_weak.items():
        print(f"  G_{name}: {g['G_i']:.4f} | AUC drop: {g['auc_drop']:+.4f}")

    # ── Model 3: Strong VMIB ─────────────────────────────────────────
    print("\n--- Training Strong VMIB (lambda=0.01) ---")
    model_strong, _ = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.01, gamma_consist=0.0, hidden_dim=256,
        latent_dim=32, epochs=120, lr=1e-3,
    )
    gaps_strong, full_strong = compute_predictive_gaps(
        model_strong, test_loader, 0.01, H_Y
    )
    results["models"]["strong_vmib"] = {
        "description": "Strong IB regularization (lambda=0.01)",
        "gaps": gaps_strong,
        "full_auc": full_strong["auc"],
        "full_acc": full_strong["acc"],
    }

    print("\nStrong VMIB G_i:")
    for name, g in gaps_strong.items():
        print(f"  G_{name}: {g['G_i']:.4f} | AUC drop: {g['auc_drop']:+.4f}")

    # ── Model 4: VMIB + Consistency ───────────────────────────────────
    print("\n--- Training VMIB + Consistency (lambda=0.01, gamma=0.5) ---")
    model_consist, _ = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.01, gamma_consist=0.5, missing_prob=0.4,
        hidden_dim=256, latent_dim=32, epochs=120, lr=1e-3,
    )
    gaps_consist, full_consist = compute_predictive_gaps(
        model_consist, test_loader, 0.01, H_Y
    )
    results["models"]["vmib_consistency"] = {
        "description": "VMIB + consistency penalty (lambda=0.01, gamma=0.5)",
        "gaps": gaps_consist,
        "full_auc": full_consist["auc"],
        "full_acc": full_consist["acc"],
    }

    print("\nVMIB + Consistency G_i:")
    for name, g in gaps_consist.items():
        print(f"  G_{name}: {g['G_i']:.4f} | AUC drop: {g['auc_drop']:+.4f}")

    # ── Fusion Collapse Index ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FUSION COLLAPSE INDEX")
    print("=" * 70)
    print("(Higher = more balanced; 1.0 = perfectly balanced across modalities)")
    print("(Computed as normalized entropy of G_i distribution)\n")

    for model_name, model_data in results["models"].items():
        G_values = np.array([g["G_i"] for g in model_data["gaps"].values()])
        G_total = G_values.sum()
        if G_total > 0:
            G_norm = G_values / G_total
            # Normalized entropy: H(G_norm) / log(M)
            entropy = -np.sum(G_norm * np.log(G_norm + 1e-10))
            balance_index = entropy / np.log(NUM_MODALITIES)
        else:
            balance_index = 0.0

        results["models"][model_name]["balance_index"] = balance_index
        G_str = ", ".join(f"{MODALITY_NAMES[i]}={G_values[i]:.4f}"
                         for i in range(NUM_MODALITIES))
        print(f"  {model_name:>20}: balance={balance_index:.4f} | G=[{G_str}]")

    save_results(results, "exp4_fusion_collapse.json")
    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
