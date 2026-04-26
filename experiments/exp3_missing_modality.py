"""
Experiment 3: Missing-Modality Robustness
==========================================
Trains VMIB with vs. without KL consistency penalty.
Systematically drops modalities and measures AUC degradation
against the theoretical information loss I_miss(S).

Compares:
  - VMIB + consistency penalty (gamma > 0, modality dropout during training)
  - VMIB standard (gamma = 0, no modality dropout)
"""

import sys
import numpy as np
import torch
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, get_loaders, train_vmib, evaluate,
    estimate_mi_classification, save_results,
    MODALITY_NAMES, NUM_MODALITIES, DEVICE
)


def generate_modality_subsets():
    """Generate all non-empty subsets of modalities with masks."""
    subsets = []
    indices = list(range(NUM_MODALITIES))

    # Single modalities
    for i in indices:
        mask = [False] * NUM_MODALITIES
        mask[i] = True
        subsets.append({
            "name": MODALITY_NAMES[i],
            "mask": mask,
            "missing": [MODALITY_NAMES[j] for j in indices if j != i],
            "num_present": 1,
        })

    # Pairs
    for i, j in combinations(indices, 2):
        mask = [False] * NUM_MODALITIES
        mask[i] = True
        mask[j] = True
        missing_idx = [k for k in indices if k not in (i, j)][0]
        subsets.append({
            "name": f"{MODALITY_NAMES[i]}+{MODALITY_NAMES[j]}",
            "mask": mask,
            "missing": [MODALITY_NAMES[missing_idx]],
            "num_present": 2,
        })

    # All three
    subsets.append({
        "name": "All",
        "mask": [True] * NUM_MODALITIES,
        "missing": [],
        "num_present": 3,
    })

    return subsets


def main():
    print("=" * 70)
    print("Experiment 3: Missing-Modality Robustness")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_data, test_data, input_dims, num_classes, _ = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    # ── Step 1: Estimate I_miss(S) for each modality subset ──────────
    print("\n--- Estimating Information Loss I_miss(S) ---")
    # We need I(X^(S); Y) for each subset S to compute
    #   I_miss(S) = I(X^(1:M); Y) - I(X^(S); Y)

    X_trains = [train_data.modalities[i].numpy() for i in range(3)]
    X_tests = [test_data.modalities[i].numpy() for i in range(3)]
    y_train = train_data.labels.numpy()
    y_test = test_data.labels.numpy()

    # Full MI
    X_train_all = np.concatenate(X_trains, axis=1)
    X_test_all = np.concatenate(X_tests, axis=1)
    full_mi_res = estimate_mi_classification(
        X_train_all, y_train, X_test_all, y_test,
        num_classes, hidden_dim=512, epochs=150
    )
    I_full = full_mi_res["mi_lower_bound"]
    print(f"I(X^(1:3); Y) >= {I_full:.4f}")

    subsets = generate_modality_subsets()
    info_losses = {}

    for subset in subsets:
        if subset["name"] == "All":
            info_losses["All"] = 0.0
            continue

        present_indices = [i for i, m in enumerate(subset["mask"]) if m]
        X_tr = np.concatenate([X_trains[i] for i in present_indices], axis=1)
        X_te = np.concatenate([X_tests[i] for i in present_indices], axis=1)

        mi_res = estimate_mi_classification(
            X_tr, y_train, X_te, y_test,
            num_classes, hidden_dim=256, epochs=150
        )
        I_S = mi_res["mi_lower_bound"]
        I_miss = max(0, I_full - I_S)
        info_losses[subset["name"]] = I_miss
        print(f"  {subset['name']}: I(X^(S);Y) >= {I_S:.4f}, "
              f"I_miss = {I_miss:.4f}")

    # ── Step 2: Train VMIB with consistency ───────────────────────────
    print("\n--- Training VMIB + Consistency ---")
    model_consist, hist_consist = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.01, gamma_consist=0.5, missing_prob=0.4,
        hidden_dim=256, latent_dim=32, epochs=120, lr=1e-3,
    )

    # ── Step 3: Train VMIB standard (no consistency) ──────────────────
    print("\n--- Training VMIB Standard ---")
    model_standard, hist_standard = train_vmib(
        input_dims, num_classes, train_loader, test_loader,
        lambda_kl=0.01, gamma_consist=0.0, missing_prob=0.0,
        hidden_dim=256, latent_dim=32, epochs=120, lr=1e-3,
    )

    # ── Step 4: Evaluate with missing modalities ──────────────────────
    print("\n--- Evaluating Missing-Modality Robustness ---")
    results = {
        "I_full": I_full,
        "info_losses": info_losses,
        "consistency_model": [],
        "standard_model": [],
    }

    print(f"\n{'Subset':>25} | {'I_miss':>7} | "
          f"{'AUC(consist)':>12} | {'AUC(standard)':>13}")
    print("-" * 70)

    for subset in subsets:
        mask = subset["mask"]

        eval_consist = evaluate(model_consist, test_loader, 0.01, modality_mask=mask)
        eval_standard = evaluate(model_standard, test_loader, 0.01, modality_mask=mask)

        I_miss = info_losses[subset["name"]]

        results["consistency_model"].append({
            "subset": subset["name"],
            "mask": mask,
            "missing": subset["missing"],
            "I_miss": I_miss,
            "auc": eval_consist["auc"],
            "acc": eval_consist["acc"],
            "pred_entropy": eval_consist["pred_entropy"],
        })
        results["standard_model"].append({
            "subset": subset["name"],
            "mask": mask,
            "missing": subset["missing"],
            "I_miss": I_miss,
            "auc": eval_standard["auc"],
            "acc": eval_standard["acc"],
            "pred_entropy": eval_standard["pred_entropy"],
        })

        print(f"{subset['name']:>25} | {I_miss:>7.4f} | "
              f"{eval_consist['auc']:>12.4f} | {eval_standard['auc']:>13.4f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEGRADATION SUMMARY")
    print("=" * 70)

    full_auc_c = [r for r in results["consistency_model"] if r["subset"] == "All"][0]["auc"]
    full_auc_s = [r for r in results["standard_model"] if r["subset"] == "All"][0]["auc"]
    print(f"Full AUC: consistency={full_auc_c:.4f}, standard={full_auc_s:.4f}")

    for rc, rs in zip(results["consistency_model"], results["standard_model"]):
        if rc["subset"] == "All":
            continue
        delta_c = full_auc_c - rc["auc"]
        delta_s = full_auc_s - rs["auc"]
        print(f"  {rc['subset']:>25}: "
              f"dAUC(consist)={delta_c:+.4f}, dAUC(standard)={delta_s:+.4f}, "
              f"I_miss={rc['I_miss']:.4f}")

    save_results(results, "exp3_missing_modality.json")
    print("\nExperiment 3 complete.")


if __name__ == "__main__":
    main()
