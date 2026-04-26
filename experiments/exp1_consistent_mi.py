"""
Experiment 1 (v3): MI Estimation with 5-Fold CV and Bootstrap CIs
==================================================================
Uses separate classifiers per subset (same architecture, hyperparameters)
with 5-fold stratified CV and bootstrap CIs.

Key caveats (reported in paper):
  - These are lower bounds; tightness varies with input dimension
  - Differences of bounds are not bounds on differences
  - Synergy sign claims are supported by consistency across folds, not
    by theoretical guarantee
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_results, SimpleClassifier, DEVICE

DATA_DIR = Path(__file__).parent / "data" / "MOGONET" / "BRCA"
MCAT_CSV = Path(__file__).parent / "data" / "MCAT" / "dataset_csv"


def estimate_mi_cv(X, y, num_classes, hidden_dim=256, epochs=150, lr=1e-3):
    """Estimate MI lower bound with val-based early stopping.

    Uses 80/20 stratified split within X for early stopping,
    evaluates on full X as test.
    """
    H_Y_vals = []
    _, counts = np.unique(y, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))

    # Split for early stopping
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(sss.split(X, y))

    model = SimpleClassifier(X.shape[1], num_classes, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.FloatTensor(X[tr_idx]).to(DEVICE)
    y_tr = torch.LongTensor(y[tr_idx]).to(DEVICE)
    X_val = torch.FloatTensor(X[val_idx]).to(DEVICE)
    y_val = torch.LongTensor(y[val_idx]).to(DEVICE)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_ce = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_ce = F.cross_entropy(model(X_val), y_val).item()
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluate on full held-out test set (passed from CV outer loop)
    model.load_state_dict(best_state)
    return model, H_Y


def evaluate_model(model, X_test, y_test, H_Y):
    """Evaluate a trained model and return MI, acc, AUC."""
    model.eval()
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    y_t = torch.LongTensor(y_test).to(DEVICE)

    with torch.no_grad():
        logits = model(X_t)
        ce = F.cross_entropy(logits, y_t).item()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    mi = max(0, H_Y - ce)
    return {"mi": mi, "ce": ce, "acc": acc, "auc": auc}


def bootstrap_mi(model, X_test, y_test, H_Y, n_bootstrap=1000, seed=42):
    """Bootstrap CI for MI from a single model on test data."""
    model.eval()
    rng = np.random.RandomState(seed)
    n = len(y_test)
    X_t = torch.FloatTensor(X_test).to(DEVICE)

    with torch.no_grad():
        logits = model(X_t).cpu()

    y_t = torch.LongTensor(y_test)
    mis = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        ce = F.cross_entropy(logits[idx], y_t[idx]).item()
        mis.append(max(0, H_Y - ce))

    return {
        "mean": float(np.mean(mis)),
        "std": float(np.std(mis)),
        "ci_lower": float(np.percentile(mis, 2.5)),
        "ci_upper": float(np.percentile(mis, 97.5)),
    }


def load_brca_all():
    modalities = []
    for i in range(1, 4):
        tr = pd.read_csv(DATA_DIR / f"{i}_tr.csv", header=None).values
        te = pd.read_csv(DATA_DIR / f"{i}_te.csv", header=None).values
        modalities.append(np.vstack([tr, te]))
    ltr = pd.read_csv(DATA_DIR / "labels_tr.csv", header=None).values.ravel().astype(int)
    lte = pd.read_csv(DATA_DIR / "labels_te.csv", header=None).values.ravel().astype(int)
    return modalities, np.concatenate([ltr, lte])


def load_gbmlgg_all():
    import zipfile
    csv_path = MCAT_CSV / "tcga_gbmlgg_all_clean.csv"
    if not csv_path.exists():
        with zipfile.ZipFile(MCAT_CSV / "tcga_gbmlgg_all_clean.csv.zip", "r") as z:
            z.extractall(MCAT_CSV)
    df = pd.read_csv(csv_path, low_memory=False)
    subtype_map = {s: i for i, s in enumerate(sorted(df["oncotree_code"].unique()))}
    labels = df["oncotree_code"].map(subtype_map).values
    demo = pd.DataFrame({"age": df["age"].values, "is_female": df["is_female"].values}).values.astype(np.float32)
    mut_genes = ["ATRX","CIC","EGFR","FLG","FUBP1","HMCN1","IDH1","MUC16","NF1","PIK3CA","PIK3R1","PTEN","RYR2","TP53","TTN"]
    cnv_cols = [c for c in df.columns if "_cnv" in c]
    mol = pd.concat([df[mut_genes].reset_index(drop=True), df[cnv_cols].reset_index(drop=True)], axis=1).values.astype(np.float32)
    return [demo, mol], labels, sorted(df["oncotree_code"].unique())


def run_dataset(name, modalities_all, labels, mod_names, n_folds=5):
    print(f"\n{'=' * 70}")
    print(f"{name}: MI Estimation ({n_folds}-fold CV + Bootstrap CIs)")
    print(f"{'=' * 70}")

    num_mod = len(modalities_all)
    num_classes = len(np.unique(labels))
    _, counts = np.unique(labels, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))
    dims = [m.shape[1] for m in modalities_all]
    print(f"N={len(labels)}, Classes={num_classes}, H(Y)={H_Y:.4f}")
    print(f"Modalities: {mod_names}, dims: {dims}")

    # All non-empty subsets
    all_subsets = []
    for r in range(1, num_mod + 1):
        for s in combinations(range(num_mod), r):
            all_subsets.append(tuple(sorted(s)))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = {s: [] for s in all_subsets}
    last_fold_models = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        torch.manual_seed(fold * 100 + 42)
        np.random.seed(fold * 100 + 42)

        y_train, y_test = labels[train_idx], labels[test_idx]

        for subset in all_subsets:
            # Build concatenated features for this subset
            mods_tr = [modalities_all[i][train_idx] for i in subset]
            mods_te = [modalities_all[i][test_idx] for i in subset]

            X_train = np.concatenate(mods_tr, axis=1)
            X_test = np.concatenate(mods_te, axis=1)

            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Determine hidden dim based on input size
            hdim = 256 if X_train.shape[1] > 100 else 128

            model, _ = estimate_mi_cv(X_train, y_train, num_classes,
                                       hidden_dim=hdim, epochs=150)
            ev = evaluate_model(model, X_test, y_test, H_Y)
            fold_results[subset].append(ev)

            if fold == n_folds - 1:
                last_fold_models[subset] = (model, X_test, y_test)

            sname = "+".join(mod_names[i] for i in subset)
            print(f"  I({sname};Y) = {ev['mi']:.4f} (Acc={ev['acc']:.3f}, AUC={ev['auc']:.3f})")

    # Aggregate
    print(f"\n{'=' * 70}")
    print(f"AGGREGATED ({n_folds}-fold CV)")
    print(f"{'=' * 70}")

    agg = {}
    for subset in all_subsets:
        mis = [r["mi"] for r in fold_results[subset]]
        accs = [r["acc"] for r in fold_results[subset]]
        aucs = [r["auc"] for r in fold_results[subset]]
        sname = "+".join(mod_names[i] for i in subset)

        # Bootstrap CI from last fold
        boot = {}
        if subset in last_fold_models:
            m, xt, yt = last_fold_models[subset]
            boot = bootstrap_mi(m, xt, yt, H_Y, n_bootstrap=1000)

        agg[subset] = {
            "name": sname,
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "mi_folds": mis,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "boot_ci": boot,
        }
        ci_str = f" 95%CI=[{boot.get('ci_lower',0):.3f}, {boot.get('ci_upper',0):.3f}]" if boot else ""
        print(f"  I({sname};Y) = {np.mean(mis):.4f} +/- {np.std(mis):.4f}"
              f"  ({np.mean(mis)/H_Y*100:.1f}% H(Y)){ci_str}"
              f"  Acc={np.mean(accs):.3f} AUC={np.mean(aucs):.3f}")

    # DPI
    full_key = tuple(range(num_mod))
    full_mi = agg[full_key]["mi_mean"]
    print(f"\n--- DPI Check ---")
    dpi_ok = True
    for s in all_subsets:
        if s != full_key:
            smi = agg[s]["mi_mean"]
            ok = full_mi >= smi - 0.02
            if not ok: dpi_ok = False
            sname = "+".join(mod_names[i] for i in s)
            print(f"  I(all)={full_mi:.4f} >= I({sname})={smi:.4f}: {'OK' if ok else 'VIOLATED'}")
    print(f"  Overall: {'PASS' if dpi_ok else 'FAIL (estimation artifact)'}")

    # Synergy
    print(f"\n--- Synergy (per-fold consistency) ---")
    synergy = {}
    for i, j in combinations(range(num_mod), 2):
        pair = (i, j)
        s_folds = []
        for f in range(n_folds):
            s_val = fold_results[pair][f]["mi"] - fold_results[(i,)][f]["mi"] - fold_results[(j,)][f]["mi"]
            s_folds.append(s_val)
        sname = f"{mod_names[i]}+{mod_names[j]}"
        synergy[sname] = {
            "mean": float(np.mean(s_folds)), "std": float(np.std(s_folds)),
            "folds": s_folds,
            "n_negative": sum(1 for v in s_folds if v < 0),
        }
        print(f"  S({sname}) = {np.mean(s_folds):+.4f} +/- {np.std(s_folds):.4f}"
              f"  (negative in {synergy[sname]['n_negative']}/{n_folds} folds)")

    # Incremental
    print(f"\n--- Incremental Contributions ---")
    incremental = {}
    for i, j in combinations(range(num_mod), 2):
        pair = (i, j)
        ic_folds_i = [fold_results[pair][f]["mi"] - fold_results[(j,)][f]["mi"] for f in range(n_folds)]
        ic_folds_j = [fold_results[pair][f]["mi"] - fold_results[(i,)][f]["mi"] for f in range(n_folds)]
        incremental[f"{mod_names[i]}|{mod_names[j]}"] = {"mean": float(np.mean(ic_folds_i)), "std": float(np.std(ic_folds_i))}
        incremental[f"{mod_names[j]}|{mod_names[i]}"] = {"mean": float(np.mean(ic_folds_j)), "std": float(np.std(ic_folds_j))}
        print(f"  I({mod_names[i]};Y|{mod_names[j]}) = {np.mean(ic_folds_i):+.4f} +/- {np.std(ic_folds_i):.4f}")
        print(f"  I({mod_names[j]};Y|{mod_names[i]}) = {np.mean(ic_folds_j):+.4f} +/- {np.std(ic_folds_j):.4f}")

    return {
        "dataset": name, "H_Y": H_Y, "num_classes": num_classes,
        "n_folds": n_folds, "n_samples": len(labels),
        "aggregated": {str(k): v for k, v in agg.items()},
        "synergy": synergy, "incremental": incremental, "dpi_ok": dpi_ok,
    }


def main():
    print("MI Estimation with 5-Fold CV and Bootstrap CIs")
    print(f"Device: {DEVICE}\n")

    # BRCA
    brca_mods, brca_labels = load_brca_all()
    brca = run_dataset("TCGA-BRCA", brca_mods, brca_labels,
                        ["mRNA", "Methylation", "miRNA"])

    # GBMLGG
    gbmlgg_mods, gbmlgg_labels, _ = load_gbmlgg_all()
    gbmlgg = run_dataset("TCGA-GBMLGG", gbmlgg_mods, gbmlgg_labels,
                          ["Demographic", "Molecular"])

    save_results({"brca": brca, "gbmlgg": gbmlgg}, "exp1_consistent_mi.json")
    print("\nAll done.")


if __name__ == "__main__":
    main()
