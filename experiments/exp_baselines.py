"""
Baseline Comparison Experiment
===============================
Trains and evaluates baseline models for TCGA-BRCA:
  1. Best unimodal MLP (one per modality, report best)
  2. Concatenation MLP (no variational bottleneck)
  3. VMIB (our framework)

All evaluated with 5-fold stratified cross-validation.
Reports mean +/- std AUC across folds.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, MultiOmicsDataset, get_loaders,
    VMIBModel, vmib_loss, ConcatMLP,
    SimpleClassifier, MODALITY_NAMES, DEVICE, save_results
)


def train_simple_mlp(X_train, y_train, X_test, y_test, num_classes,
                     hidden_dim=256, epochs=150, lr=1e-3):
    """Train a simple MLP and return test AUC."""
    model = SimpleClassifier(X_train.shape[1], num_classes, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = y_test
    with torch.no_grad():
        logits = model(X_te)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(y_te, preds)
    try:
        auc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return acc, auc


def train_concat_mlp(modalities_train, y_train, modalities_test, y_test,
                     num_classes, hidden_dim=256, epochs=150, lr=1e-3):
    """Train a concatenation MLP (no bottleneck)."""
    X_train = np.concatenate(modalities_train, axis=1)
    X_test = np.concatenate(modalities_test, axis=1)
    return train_simple_mlp(X_train, y_train, X_test, y_test, num_classes,
                             hidden_dim, epochs, lr)


def train_vmib_model(input_dims, num_classes, train_loader, test_loader,
                     lambda_kl=0.01, epochs=150, lr=1e-3, seed=42):
    """Train VMIB and return test AUC."""
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
            logits, mu, logvar, z = model(xs)
            loss, _, _ = vmib_loss(logits, y, mu, logvar, lambda_kl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    all_probs, all_labels = [], []
    all_preds = []
    with torch.no_grad():
        for xs, y in test_loader:
            xs = [x.to(DEVICE) for x in xs]
            logits, _, _, _ = model(xs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return acc, auc


def main():
    print("=" * 70)
    print("Baseline Comparison: 5-Fold Stratified CV")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load all data (we'll do our own CV splits)
    train_data, test_data, input_dims, num_classes, scalers = load_brca_data()

    # Combine train and test for CV
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    base = Path(__file__).parent / "data" / "MOGONET" / "BRCA"
    modalities_all = []
    for i in range(1, 4):
        tr = pd.read_csv(base / f"{i}_tr.csv", header=None).values
        te = pd.read_csv(base / f"{i}_te.csv", header=None).values
        modalities_all.append(np.vstack([tr, te]))

    labels_tr = pd.read_csv(base / "labels_tr.csv", header=None).values.ravel().astype(int)
    labels_te = pd.read_csv(base / "labels_te.csv", header=None).values.ravel().astype(int)
    y_all = np.concatenate([labels_tr, labels_te])

    print(f"Total samples: {len(y_all)}, Classes: {num_classes}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {name: {"acc": [], "auc": []} for name in
               ["Best Unimodal", "Concat MLP", "VMIB"]}

    for fold, (train_idx, test_idx) in enumerate(skf.split(y_all, y_all)):
        print(f"\n--- Fold {fold+1}/5 ---")

        # Split and standardize
        mods_train, mods_test = [], []
        for m in modalities_all:
            scaler = StandardScaler()
            mods_train.append(scaler.fit_transform(m[train_idx]))
            mods_test.append(scaler.transform(m[test_idx]))
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # 1. Best Unimodal MLP
        best_uni_auc = 0
        best_uni_acc = 0
        for i in range(3):
            acc, auc = train_simple_mlp(mods_train[i], y_train,
                                         mods_test[i], y_test, num_classes)
            if auc > best_uni_auc:
                best_uni_auc = auc
                best_uni_acc = acc
        results["Best Unimodal"]["acc"].append(best_uni_acc)
        results["Best Unimodal"]["auc"].append(best_uni_auc)
        print(f"  Best Unimodal: AUC={best_uni_auc:.4f}")

        # 2. Concat MLP
        acc, auc = train_concat_mlp(mods_train, y_train, mods_test, y_test,
                                     num_classes)
        results["Concat MLP"]["acc"].append(acc)
        results["Concat MLP"]["auc"].append(auc)
        print(f"  Concat MLP:    AUC={auc:.4f}")

        # 3. VMIB
        td = MultiOmicsDataset(mods_train, y_train)
        ted = MultiOmicsDataset(mods_test, y_test)
        tl, tel = get_loaders(td, ted, batch_size=64)
        acc, auc = train_vmib_model(input_dims, num_classes, tl, tel,
                                     lambda_kl=0.01, epochs=150, seed=fold)
        results["VMIB"]["acc"].append(acc)
        results["VMIB"]["auc"].append(auc)
        print(f"  VMIB:          AUC={auc:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("5-FOLD CV RESULTS")
    print("=" * 70)
    print(f"{'Model':>15} | {'AUC (mean +/- std)':>20} | {'Acc (mean +/- std)':>20}")
    print("-" * 60)
    for name in ["Best Unimodal", "Concat MLP", "VMIB"]:
        aucs = results[name]["auc"]
        accs = results[name]["acc"]
        print(f"{name:>15} | {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
              f"      | {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    save_results(results, "exp_baselines.json")
    print("\nBaseline comparison complete.")


if __name__ == "__main__":
    main()
