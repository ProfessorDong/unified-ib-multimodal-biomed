"""
TCGA-GBMLGG Synergy Experiment
================================
Demonstrates synergy (S_ij > 0) between demographic and molecular modalities
for glioma histological subtype classification (6 classes).

Key insight: binary targets with strong individual modalities mathematically
preclude synergy (I_clin + I_mol > H(Y) when H(Y) = ln(2)). Multiclass
targets have higher H(Y), providing room for synergistic interactions.

Modality 1 (Demographic): age, sex
Modality 2 (Molecular): key gene mutations + CNV features

Target: 6-class histological subtype (GBM, AASTR, AOAST, ASTR, OAST, ODG)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    estimate_mi_classification, save_results,
    MultiOmicsDataset, get_loaders, train_vmib, evaluate,
    DEVICE
)

MCAT_CSV = Path(__file__).parent / "data" / "MCAT" / "dataset_csv"


def prepare_gbmlgg_data():
    """Prepare TCGA-GBMLGG demographic + molecular modalities for subtype prediction."""
    import zipfile
    csv_path = MCAT_CSV / "tcga_gbmlgg_all_clean.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(MCAT_CSV / "tcga_gbmlgg_all_clean.csv.zip", "r") as z:
            z.extractall(MCAT_CSV)

    df = pd.read_csv(csv_path, low_memory=False)

    # Target: histological subtype (6 classes)
    subtype_map = {s: i for i, s in enumerate(sorted(df["oncotree_code"].unique()))}
    labels = df["oncotree_code"].map(subtype_map).values
    subtype_names = sorted(df["oncotree_code"].unique())

    # --- Modality 1: Demographic ---
    demographic = pd.DataFrame({
        "age": df["age"].values,
        "is_female": df["is_female"].values,
    })

    # --- Modality 2: Molecular ---
    mutation_genes = ["ATRX", "CIC", "EGFR", "FLG", "FUBP1", "HMCN1", "IDH1",
                      "MUC16", "NF1", "PIK3CA", "PIK3R1", "PTEN", "RYR2", "TP53", "TTN"]
    mutations = df[mutation_genes].copy()
    cnv_cols = [c for c in df.columns if "_cnv" in c]
    cnv = df[cnv_cols].copy()
    molecular = pd.concat([mutations.reset_index(drop=True),
                           cnv.reset_index(drop=True)], axis=1)

    print(f"GBMLGG dataset: {len(df)} samples, {len(subtype_names)} subtypes")
    print(f"  Subtypes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Names: {subtype_names}")
    print(f"  Demographic features: {demographic.shape[1]}")
    print(f"  Molecular features: {molecular.shape[1]}")

    X_demo = demographic.values.astype(np.float32)
    X_mol = molecular.values.astype(np.float32)

    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=42
    )

    scaler_demo = StandardScaler()
    X_demo_tr = scaler_demo.fit_transform(X_demo[idx_train])
    X_demo_te = scaler_demo.transform(X_demo[idx_test])

    scaler_mol = StandardScaler()
    X_mol_tr = scaler_mol.fit_transform(X_mol[idx_train])
    X_mol_te = scaler_mol.transform(X_mol[idx_test])

    y_train = labels[idx_train]
    y_test = labels[idx_test]

    return (X_demo_tr, X_demo_te, X_mol_tr, X_mol_te,
            y_train, y_test, demographic.shape[1], molecular.shape[1],
            subtype_names)


def main():
    print("=" * 70)
    print("TCGA-GBMLGG: Demographic + Molecular Synergy Experiment")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    (X_demo_tr, X_demo_te, X_mol_tr, X_mol_te,
     y_train, y_test, d_demo, d_mol, subtype_names) = prepare_gbmlgg_data()

    num_classes = len(np.unique(y_train))
    _, counts = np.unique(y_train, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))
    print(f"\nH(Y) = {H_Y:.4f} nats ({num_classes} classes)")

    results = {"dataset": "TCGA-GBMLGG", "task": "6-class histological subtype",
               "H_Y": H_Y, "num_classes": num_classes, "subtype_names": subtype_names,
               "n_train": len(y_train), "n_test": len(y_test),
               "d_demographic": d_demo, "d_molecular": d_mol}

    # ── MI Estimation (multiple seeds for robustness) ─────────
    print("\n--- MI Estimation (5 seeds) ---")
    seeds = [42, 123, 456, 789, 1024]
    all_I_demo, all_I_mol, all_I_both, all_S = [], [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        r_d = estimate_mi_classification(X_demo_tr, y_train, X_demo_te, y_test,
                                          num_classes, hidden_dim=64, epochs=200)
        r_m = estimate_mi_classification(X_mol_tr, y_train, X_mol_te, y_test,
                                          num_classes, hidden_dim=128, epochs=200)

        X_both_tr = np.concatenate([X_demo_tr, X_mol_tr], axis=1)
        X_both_te = np.concatenate([X_demo_te, X_mol_te], axis=1)
        r_b = estimate_mi_classification(X_both_tr, y_train, X_both_te, y_test,
                                          num_classes, hidden_dim=128, epochs=200)

        I_d, I_m, I_b = r_d["mi_lower_bound"], r_m["mi_lower_bound"], r_b["mi_lower_bound"]
        S = I_b - I_d - I_m
        all_I_demo.append(I_d)
        all_I_mol.append(I_m)
        all_I_both.append(I_b)
        all_S.append(S)

        print(f"  Seed {seed}: I_demo={I_d:.4f}, I_mol={I_m:.4f}, "
              f"I_both={I_b:.4f}, S={S:+.4f} | "
              f"Acc: d={r_d['accuracy']:.3f}, m={r_m['accuracy']:.3f}, b={r_b['accuracy']:.3f}")

    I_demo_mean = np.mean(all_I_demo)
    I_mol_mean = np.mean(all_I_mol)
    I_both_mean = np.mean(all_I_both)
    S_mean = np.mean(all_S)
    S_std = np.std(all_S)

    results["mi_estimation"] = {
        "I_demographic": {"mean": I_demo_mean, "std": np.std(all_I_demo),
                          "pct_HY": I_demo_mean / H_Y * 100},
        "I_molecular": {"mean": I_mol_mean, "std": np.std(all_I_mol),
                        "pct_HY": I_mol_mean / H_Y * 100},
        "I_joint": {"mean": I_both_mean, "std": np.std(all_I_both),
                    "pct_HY": I_both_mean / H_Y * 100},
        "synergy": {"mean": S_mean, "std": S_std,
                    "n_positive": sum(1 for s in all_S if s > 0),
                    "values": all_S},
    }

    print(f"\n  Mean: I_demo={I_demo_mean:.4f} ({I_demo_mean/H_Y*100:.1f}% H(Y)), "
          f"I_mol={I_mol_mean:.4f} ({I_mol_mean/H_Y*100:.1f}% H(Y)), "
          f"I_both={I_both_mean:.4f} ({I_both_mean/H_Y*100:.1f}% H(Y))")
    print(f"  Synergy S: {S_mean:+.4f} +/- {S_std:.4f} "
          f"(positive in {sum(1 for s in all_S if s > 0)}/{len(seeds)} seeds)")
    interpretation = ("synergy-dominated" if S_mean > 0.01
                      else "redundancy-dominated" if S_mean < -0.01
                      else "approximately additive")
    print(f"  Interpretation: {interpretation}")

    # ── Incremental contributions ─────────────────────────────
    print(f"\n--- Incremental Contributions ---")
    print(f"  I(Demo; Y | Mol) >= {I_both_mean - I_mol_mean:.4f} nats "
          f"(demographic adds to molecular)")
    print(f"  I(Mol; Y | Demo) >= {I_both_mean - I_demo_mean:.4f} nats "
          f"(molecular adds to demographic)")

    # ── VMIB Fusion ───────────────────────────────────────────
    print("\n--- VMIB Fusion ---")
    input_dims = [d_demo, d_mol]
    train_data = MultiOmicsDataset([X_demo_tr, X_mol_tr], y_train)
    test_data = MultiOmicsDataset([X_demo_te, X_mol_te], y_test)
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    model, _ = train_vmib(input_dims, num_classes, train_loader, test_loader,
                           lambda_kl=0.01, hidden_dim=128, latent_dim=16,
                           epochs=120, lr=1e-3)

    eval_fused = evaluate(model, test_loader, 0.01)
    eval_demo = evaluate(model, test_loader, 0.01, modality_mask=[True, False])
    eval_mol = evaluate(model, test_loader, 0.01, modality_mask=[False, True])

    G_demo = max(0, eval_fused["acc"] - eval_demo["acc"])
    G_mol = max(0, eval_fused["acc"] - eval_mol["acc"])

    results["vmib"] = {
        "fused_acc": eval_fused["acc"],
        "demo_only_acc": eval_demo["acc"],
        "mol_only_acc": eval_mol["acc"],
        "G_demographic": G_demo,
        "G_molecular": G_mol,
    }

    print(f"\n  Fused Acc:       {eval_fused['acc']:.4f}")
    print(f"  Demographic only: {eval_demo['acc']:.4f}")
    print(f"  Molecular only:   {eval_mol['acc']:.4f}")
    print(f"  G_demo={G_demo:.4f}, G_mol={G_mol:.4f}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: TCGA-GBMLGG, {num_classes}-class subtype, {len(y_train)+len(y_test)} samples")
    print(f"H(Y) = {H_Y:.4f} nats")
    print(f"I(Demographic; Y)           = {I_demo_mean:.4f} +/- {np.std(all_I_demo):.4f} nats")
    print(f"I(Molecular; Y)             = {I_mol_mean:.4f} +/- {np.std(all_I_mol):.4f} nats")
    print(f"I(Demographic+Molecular; Y) = {I_both_mean:.4f} +/- {np.std(all_I_both):.4f} nats")
    print(f"Synergy proxy S             = {S_mean:+.4f} +/- {S_std:.4f} ({interpretation})")
    print(f"Fused accuracy: {eval_fused['acc']:.4f}")

    save_results(results, "exp_gbmlgg_synergy.json")
    print("\nGBMLGG synergy experiment complete.")


if __name__ == "__main__":
    main()
