"""
Experiment 1: Information Decomposition Across Real Modalities
==============================================================
Estimates I(X^(i); Y) for each modality individually, for each pair,
and for all three together. Computes redundancy/synergy proxies S_ij.

Uses classification-based MI lower bounds:
    I(X;Y) >= H(Y) - min_classifier CE(Y|X)
"""

import sys
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, estimate_mi_classification, save_results,
    MODALITY_NAMES, DEVICE
)


def main():
    print("=" * 70)
    print("Experiment 1: Information Decomposition Across Real Modalities")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load data
    train_data, test_data, input_dims, num_classes, scalers = load_brca_data()

    # Extract raw numpy arrays
    X_trains = [train_data.modalities[i].numpy() for i in range(3)]
    X_tests = [test_data.modalities[i].numpy() for i in range(3)]
    y_train = train_data.labels.numpy()
    y_test = test_data.labels.numpy()

    results = {"individual": {}, "pairs": {}, "triple": {}, "synergy": {}}

    # ── Individual modality MI: I(X^(i); Y) ─────────────────────────
    print("\n--- Individual Modality MI ---")
    for i in range(3):
        name = MODALITY_NAMES[i]
        print(f"\nEstimating I({name}; Y)...")
        res = estimate_mi_classification(
            X_trains[i], y_train, X_tests[i], y_test,
            num_classes, hidden_dim=256, epochs=150, lr=1e-3
        )
        results["individual"][name] = res
        print(f"  I({name}; Y) >= {res['mi_lower_bound']:.4f} nats "
              f"| Acc={res['accuracy']:.4f} | AUC={res['auc']:.4f}")

    # ── Pairwise MI: I(X^(i), X^(j); Y) ─────────────────────────────
    print("\n--- Pairwise Modality MI ---")
    for i, j in combinations(range(3), 2):
        name_i, name_j = MODALITY_NAMES[i], MODALITY_NAMES[j]
        pair_name = f"{name_i}+{name_j}"
        X_train_pair = np.concatenate([X_trains[i], X_trains[j]], axis=1)
        X_test_pair = np.concatenate([X_tests[i], X_tests[j]], axis=1)

        print(f"\nEstimating I({pair_name}; Y)...")
        res = estimate_mi_classification(
            X_train_pair, y_train, X_test_pair, y_test,
            num_classes, hidden_dim=256, epochs=150, lr=1e-3
        )
        results["pairs"][pair_name] = res
        print(f"  I({pair_name}; Y) >= {res['mi_lower_bound']:.4f} nats "
              f"| Acc={res['accuracy']:.4f} | AUC={res['auc']:.4f}")

    # ── Triple MI: I(X^(1), X^(2), X^(3); Y) ────────────────────────
    print("\n--- All Three Modalities ---")
    X_train_all = np.concatenate(X_trains, axis=1)
    X_test_all = np.concatenate(X_tests, axis=1)

    res = estimate_mi_classification(
        X_train_all, y_train, X_test_all, y_test,
        num_classes, hidden_dim=512, epochs=150, lr=1e-3
    )
    results["triple"]["all"] = res
    print(f"  I(X^(1:3); Y) >= {res['mi_lower_bound']:.4f} nats "
          f"| Acc={res['accuracy']:.4f} | AUC={res['auc']:.4f}")

    # ── Synergy/Redundancy Proxies ────────────────────────────────────
    print("\n--- Synergy/Redundancy Proxies ---")
    I_all = results["triple"]["all"]["mi_lower_bound"]

    for i, j in combinations(range(3), 2):
        name_i, name_j = MODALITY_NAMES[i], MODALITY_NAMES[j]
        pair_name = f"{name_i}+{name_j}"
        I_i = results["individual"][name_i]["mi_lower_bound"]
        I_j = results["individual"][name_j]["mi_lower_bound"]
        I_ij = results["pairs"][pair_name]["mi_lower_bound"]

        S_ij = I_ij - I_i - I_j
        R_ij = -S_ij

        results["synergy"][pair_name] = {
            "S_ij": S_ij,
            "R_ij": R_ij,
            "I_i": I_i,
            "I_j": I_j,
            "I_ij": I_ij,
            "interpretation": "synergy-dominated" if S_ij > 0.01
                              else "redundancy-dominated" if S_ij < -0.01
                              else "approximately additive"
        }
        print(f"  {pair_name}: S_ij={S_ij:+.4f} ({results['synergy'][pair_name]['interpretation']})")

    # Also compute incremental contributions: I(X^(i); Y | X^(j)) = I(X^(i),X^(j); Y) - I(X^(j); Y)
    print("\n--- Incremental Contributions ---")
    results["incremental"] = {}
    for i, j in combinations(range(3), 2):
        name_i, name_j = MODALITY_NAMES[i], MODALITY_NAMES[j]
        pair_name = f"{name_i}+{name_j}"
        I_ij = results["pairs"][pair_name]["mi_lower_bound"]
        I_j = results["individual"][name_j]["mi_lower_bound"]
        I_i = results["individual"][name_i]["mi_lower_bound"]

        incr_i_given_j = I_ij - I_j
        incr_j_given_i = I_ij - I_i

        results["incremental"][f"{name_i}|{name_j}"] = incr_i_given_j
        results["incremental"][f"{name_j}|{name_i}"] = incr_j_given_i
        print(f"  I({name_i}; Y | {name_j}) >= {incr_i_given_j:.4f}")
        print(f"  I({name_j}; Y | {name_i}) >= {incr_j_given_i:.4f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    H_Y = results["individual"][MODALITY_NAMES[0]]["H_Y"]
    print(f"H(Y) = {H_Y:.4f} nats")
    for name in MODALITY_NAMES:
        mi = results["individual"][name]["mi_lower_bound"]
        print(f"I({name}; Y) >= {mi:.4f} nats ({mi/H_Y*100:.1f}% of H(Y))")
    print(f"I(X^(1:3); Y)  >= {I_all:.4f} nats ({I_all/H_Y*100:.1f}% of H(Y))")

    save_results(results, "exp1_information_decomposition.json")
    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
