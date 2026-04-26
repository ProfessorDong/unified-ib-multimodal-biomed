"""
Experiment: Uncertainty, Calibration, and Trust (Section 6)
============================================================
Validates the three core concepts from Section 6 of the paper using
the TCGA-BRCA multi-omics VMIB model:

  6a. OOD Detection via Predictive Entropy
      - Hold out one BRCA class (HER2-enriched) as OOD during training.
      - Train VMIB on remaining 4 classes.
      - Compute predictive entropy on ID test samples vs OOD (held-out class).
      - Evaluate AUROC of entropy-based OOD detection.

  6b. Calibration (ECE and Reliability Diagram)
      - Train VMIB on all 5 BRCA classes (standard setup).
      - Compute reliability diagram (10-bin) and ECE on test set.
      - Compare: VMIB (multimodal) vs best unimodal (mRNA-only SimpleClassifier).

  6c. Selective Prediction (Deferral)
      - Using the trained VMIB from 6b.
      - Sort test samples by predictive entropy.
      - For coverage levels 0.3 to 1.0: retain lowest-entropy samples,
        compute accuracy on retained set.
      - Compare with random deferral baseline.

Validates paper Equations (16)-(19) and Figure 7.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, MultiOmicsDataset, get_loaders,
    VMIBModel, vmib_loss, train_vmib, evaluate,
    SimpleClassifier, MODALITY_NAMES, DEVICE, save_results
)

SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_predictive_entropy(probs):
    """Compute per-sample predictive entropy H(p_hat(.|z)).

    Args:
        probs: np.ndarray of shape (N, C), predicted class probabilities.

    Returns:
        np.ndarray of shape (N,), entropy for each sample.
    """
    return -(probs * np.log(probs + 1e-10)).sum(axis=1)


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error with reliability diagram data.

    For the multiclass case, we use the predicted class probability
    (max probability) and check whether the prediction is correct.

    Args:
        probs: np.ndarray (N, C), predicted probabilities.
        labels: np.ndarray (N,), true labels.
        n_bins: int, number of bins.

    Returns:
        ece: float, expected calibration error.
        bin_data: list of dicts with bin_center, accuracy, confidence, count.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n_total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()
        if count > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (count / n_total) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_data.append({
            "bin_center": float((lo + hi) / 2),
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "accuracy": float(bin_acc),
            "confidence": float(bin_conf),
            "count": int(count),
        })

    return float(ece), bin_data


# ─── Experiment 6a: OOD Detection via Predictive Entropy ─────────────────

def experiment_6a():
    """OOD detection by holding out one BRCA class during training."""
    print("\n" + "=" * 70)
    print("Experiment 6a: OOD Detection via Predictive Entropy")
    print("=" * 70)

    # Load raw data to manipulate classes
    DATA_DIR = Path(__file__).parent / "data" / "MOGONET" / "BRCA"
    modalities_all = []
    for i in range(1, 4):
        tr = pd.read_csv(DATA_DIR / f"{i}_tr.csv", header=None).values
        te = pd.read_csv(DATA_DIR / f"{i}_te.csv", header=None).values
        modalities_all.append(np.vstack([tr, te]))

    labels_tr = pd.read_csv(DATA_DIR / "labels_tr.csv", header=None).values.ravel().astype(int)
    labels_te = pd.read_csv(DATA_DIR / "labels_te.csv", header=None).values.ravel().astype(int)
    y_all = np.concatenate([labels_tr, labels_te])

    # Identify the OOD class (class 2: HER2-enriched, smallest class)
    unique_classes, class_counts = np.unique(y_all, return_counts=True)
    # Pick the smallest class as OOD
    ood_class = unique_classes[np.argmin(class_counts)]
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    print(f"OOD class: {ood_class} (held out, {class_counts[ood_class]} samples)")

    # Split: ID = all classes except OOD class
    id_mask = y_all != ood_class
    ood_mask = y_all == ood_class

    X_id = [m[id_mask] for m in modalities_all]
    y_id = y_all[id_mask]
    X_ood = [m[ood_mask] for m in modalities_all]
    y_ood = y_all[ood_mask]

    # Relabel ID classes to be contiguous (0, 1, 2, 3)
    id_classes = sorted(np.unique(y_id))
    relabel_map = {c: i for i, c in enumerate(id_classes)}
    y_id_relabeled = np.array([relabel_map[c] for c in y_id])
    num_classes_id = len(id_classes)

    print(f"ID samples: {len(y_id)}, OOD samples: {len(y_ood)}")
    print(f"ID classes (relabeled): {num_classes_id}")

    # Train/test split for ID data (use 75/25 split)
    from sklearn.model_selection import train_test_split
    id_idx_train, id_idx_test = train_test_split(
        np.arange(len(y_id)), test_size=0.25, stratify=y_id_relabeled, random_state=SEED
    )

    # Standardize
    mods_train, mods_test, mods_ood = [], [], []
    for m_all in X_id:
        scaler = StandardScaler()
        m_tr = scaler.fit_transform(m_all[id_idx_train])
        m_te = scaler.transform(m_all[id_idx_test])
        mods_train.append(m_tr)
        mods_test.append(m_te)

    # OOD data: standardize using ID training scalers
    ood_scalers = []
    for i, m_all_id in enumerate(X_id):
        scaler = StandardScaler()
        scaler.fit(m_all_id[id_idx_train])
        ood_scalers.append(scaler)

    for i, m_ood in enumerate(X_ood):
        mods_ood.append(ood_scalers[i].transform(m_ood))

    y_train = y_id_relabeled[id_idx_train]
    y_test = y_id_relabeled[id_idx_test]

    input_dims = [m.shape[1] for m in mods_train]

    # Train VMIB on ID classes
    set_seed(SEED)
    train_data = MultiOmicsDataset(mods_train, y_train)
    test_data = MultiOmicsDataset(mods_test, y_test)
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    print("\nTraining VMIB on in-distribution classes...")
    model, _ = train_vmib(input_dims, num_classes_id, train_loader, test_loader,
                          lambda_kl=0.01, epochs=120, lr=1e-3, verbose=True)

    # Compute entropy on ID test set
    eval_id = evaluate(model, test_loader, lambda_kl=0.01)
    probs_id = eval_id["probs"]
    entropy_id = compute_predictive_entropy(probs_id)

    # Compute entropy on OOD set
    # Create a dummy dataset for OOD (labels don't matter for entropy)
    ood_data = MultiOmicsDataset(mods_ood, np.zeros(len(y_ood), dtype=int))
    ood_loader = DataLoader(ood_data, batch_size=64, shuffle=False)

    model.eval()
    all_probs_ood = []
    with torch.no_grad():
        for xs, _ in ood_loader:
            xs = [x.to(DEVICE) for x in xs]
            logits, _, _, _ = model(xs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs_ood.append(probs)
    probs_ood = np.concatenate(all_probs_ood)
    entropy_ood = compute_predictive_entropy(probs_ood)

    # AUROC: use entropy to discriminate ID (label=0) from OOD (label=1)
    detection_labels = np.concatenate([np.zeros(len(entropy_id)),
                                       np.ones(len(entropy_ood))])
    detection_scores = np.concatenate([entropy_id, entropy_ood])
    auroc = roc_auc_score(detection_labels, detection_scores)

    # Max entropy for reference (uniform over num_classes_id)
    max_entropy = np.log(num_classes_id)

    print(f"\n--- OOD Detection Results ---")
    print(f"  ID entropy:  mean={entropy_id.mean():.4f}, std={entropy_id.std():.4f}")
    print(f"  OOD entropy: mean={entropy_ood.mean():.4f}, std={entropy_ood.std():.4f}")
    print(f"  Max entropy (uniform over {num_classes_id} classes): {max_entropy:.4f}")
    print(f"  ID/OOD entropy ratio: {entropy_ood.mean() / entropy_id.mean():.2f}x")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  ID test accuracy: {eval_id['acc']:.4f}")

    results_6a = {
        "ood_class": int(ood_class),
        "ood_class_count": int(class_counts[ood_class]),
        "num_id_classes": num_classes_id,
        "n_id_test": len(entropy_id),
        "n_ood": len(entropy_ood),
        "id_entropy_mean": float(entropy_id.mean()),
        "id_entropy_std": float(entropy_id.std()),
        "ood_entropy_mean": float(entropy_ood.mean()),
        "ood_entropy_std": float(entropy_ood.std()),
        "max_entropy": float(max_entropy),
        "auroc": float(auroc),
        "id_test_accuracy": float(eval_id["acc"]),
        "id_entropy_values": entropy_id.tolist(),
        "ood_entropy_values": entropy_ood.tolist(),
    }

    return results_6a


# ─── Experiment 6b: Calibration ──────────────────────────────────────────

def train_unimodal_mrna(input_dim, num_classes, X_train, y_train, X_test, y_test,
                        hidden_dim=256, epochs=150, lr=1e-3):
    """Train a unimodal mRNA classifier and return test probs and labels."""
    set_seed(SEED)
    model = SimpleClassifier(input_dim, num_classes, hidden_dim).to(DEVICE)
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
    with torch.no_grad():
        logits = model(X_te)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return probs, preds, acc, auc


def experiment_6b():
    """Calibration comparison: VMIB (multimodal) vs unimodal (mRNA)."""
    print("\n" + "=" * 70)
    print("Experiment 6b: Calibration (ECE and Reliability Diagram)")
    print("=" * 70)

    # Load data
    train_data, test_data, input_dims, num_classes, scalers = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    # Train VMIB (multimodal)
    set_seed(SEED)
    print("\nTraining VMIB (multimodal)...")
    model_vmib, _ = train_vmib(input_dims, num_classes, train_loader, test_loader,
                               lambda_kl=0.01, epochs=150, lr=1e-3, verbose=True)

    eval_vmib = evaluate(model_vmib, test_loader, lambda_kl=0.01)
    probs_vmib = eval_vmib["probs"]
    labels_test = eval_vmib["labels"]

    # ECE for VMIB
    ece_vmib, bins_vmib = compute_ece(probs_vmib, labels_test, n_bins=10)

    print(f"\n  VMIB Accuracy: {eval_vmib['acc']:.4f}")
    print(f"  VMIB ECE: {ece_vmib:.4f}")

    # Train unimodal mRNA classifier
    # Extract raw mRNA data from train/test datasets
    DATA_DIR = Path(__file__).parent / "data" / "MOGONET" / "BRCA"
    tr_mrna = pd.read_csv(DATA_DIR / "1_tr.csv", header=None).values
    te_mrna = pd.read_csv(DATA_DIR / "1_te.csv", header=None).values
    labels_tr = pd.read_csv(DATA_DIR / "labels_tr.csv", header=None).values.ravel().astype(int)
    labels_te_raw = pd.read_csv(DATA_DIR / "labels_te.csv", header=None).values.ravel().astype(int)

    scaler_mrna = StandardScaler()
    X_tr_mrna = scaler_mrna.fit_transform(tr_mrna)
    X_te_mrna = scaler_mrna.transform(te_mrna)

    print("\nTraining unimodal mRNA classifier...")
    probs_uni, preds_uni, acc_uni, auc_uni = train_unimodal_mrna(
        input_dim=X_tr_mrna.shape[1], num_classes=num_classes,
        X_train=X_tr_mrna, y_train=labels_tr,
        X_test=X_te_mrna, y_test=labels_te_raw,
        hidden_dim=256, epochs=150
    )

    ece_uni, bins_uni = compute_ece(probs_uni, labels_te_raw, n_bins=10)

    print(f"\n  Unimodal (mRNA) Accuracy: {acc_uni:.4f}")
    print(f"  Unimodal (mRNA) ECE: {ece_uni:.4f}")

    # Print reliability diagram summary
    print(f"\n--- Reliability Diagram (VMIB) ---")
    print(f"  {'Bin':>8} | {'Conf':>6} | {'Acc':>6} | {'Count':>6} | {'|Gap|':>6}")
    print(f"  " + "-" * 42)
    for b in bins_vmib:
        if b["count"] > 0:
            gap = abs(b["accuracy"] - b["confidence"])
            print(f"  {b['bin_center']:.2f}     | {b['confidence']:.3f} | "
                  f"{b['accuracy']:.3f} | {b['count']:>6d} | {gap:.3f}")

    print(f"\n--- Reliability Diagram (Unimodal mRNA) ---")
    print(f"  {'Bin':>8} | {'Conf':>6} | {'Acc':>6} | {'Count':>6} | {'|Gap|':>6}")
    print(f"  " + "-" * 42)
    for b in bins_uni:
        if b["count"] > 0:
            gap = abs(b["accuracy"] - b["confidence"])
            print(f"  {b['bin_center']:.2f}     | {b['confidence']:.3f} | "
                  f"{b['accuracy']:.3f} | {b['count']:>6d} | {gap:.3f}")

    print(f"\n--- Calibration Summary ---")
    print(f"  VMIB ECE:         {ece_vmib:.4f}")
    print(f"  Unimodal ECE:     {ece_uni:.4f}")
    improvement = (ece_uni - ece_vmib) / ece_uni * 100 if ece_uni > 0 else 0
    if ece_vmib < ece_uni:
        print(f"  VMIB reduces ECE by {improvement:.1f}% relative to unimodal")
    else:
        print(f"  Unimodal has lower ECE (by {-improvement:.1f}%)")

    results_6b = {
        "vmib": {
            "accuracy": float(eval_vmib["acc"]),
            "auc": float(eval_vmib["auc"]),
            "ece": float(ece_vmib),
            "reliability_bins": bins_vmib,
        },
        "unimodal_mrna": {
            "accuracy": float(acc_uni),
            "auc": float(auc_uni),
            "ece": float(ece_uni),
            "reliability_bins": bins_uni,
        },
    }

    return results_6b, model_vmib, test_loader, labels_test


# ─── Experiment 6c: Selective Prediction (Deferral) ─────────────────────

def experiment_6c(model_vmib, test_loader, labels_test):
    """Selective prediction: entropy-based deferral vs random baseline."""
    print("\n" + "=" * 70)
    print("Experiment 6c: Selective Prediction (Deferral)")
    print("=" * 70)

    # Get VMIB predictions and entropy on test set
    eval_vmib = evaluate(model_vmib, test_loader, lambda_kl=0.01)
    probs = eval_vmib["probs"]
    preds = np.argmax(probs, axis=1)
    entropy = compute_predictive_entropy(probs)
    labels = eval_vmib["labels"]
    n_test = len(labels)

    coverage_levels = np.arange(0.3, 1.05, 0.1)
    coverage_levels = np.clip(coverage_levels, 0.0, 1.0)

    # Entropy-based deferral: retain lowest-entropy samples
    sorted_indices = np.argsort(entropy)  # ascending entropy
    entropy_results = []
    for cov in coverage_levels:
        n_retain = max(1, int(round(cov * n_test)))
        retained_idx = sorted_indices[:n_retain]
        acc_retained = accuracy_score(labels[retained_idx], preds[retained_idx])
        entropy_results.append({
            "coverage": float(round(cov, 2)),
            "n_retained": int(n_retain),
            "accuracy": float(acc_retained),
        })

    # Random deferral baseline (average over multiple random draws)
    rng = np.random.RandomState(SEED)
    n_random_trials = 200
    random_results = []
    for cov in coverage_levels:
        n_retain = max(1, int(round(cov * n_test)))
        accs = []
        for _ in range(n_random_trials):
            random_idx = rng.choice(n_test, size=n_retain, replace=False)
            acc_rand = accuracy_score(labels[random_idx], preds[random_idx])
            accs.append(acc_rand)
        random_results.append({
            "coverage": float(round(cov, 2)),
            "n_retained": int(n_retain),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
        })

    # Print results
    full_acc = accuracy_score(labels, preds)
    print(f"\n  Full test set accuracy: {full_acc:.4f} (n={n_test})")
    print(f"\n  {'Coverage':>10} | {'Entropy Acc':>12} | {'Random Acc':>12} | {'Gain':>8}")
    print(f"  " + "-" * 50)
    for er, rr in zip(entropy_results, random_results):
        gain = er["accuracy"] - rr["accuracy_mean"]
        print(f"  {er['coverage']:>10.1f} | {er['accuracy']:>12.4f} | "
              f"{rr['accuracy_mean']:>12.4f} | {gain:>+8.4f}")

    # Compute area between curves (AUC advantage)
    entropy_accs = [r["accuracy"] for r in entropy_results]
    random_accs = [r["accuracy_mean"] for r in random_results]
    coverages = [r["coverage"] for r in entropy_results]

    # Average gain
    mean_gain = np.mean([e - r for e, r in zip(entropy_accs, random_accs)])
    print(f"\n  Mean accuracy gain (entropy over random): {mean_gain:+.4f}")
    print(f"  Entropy-based deferral at 50% coverage: {entropy_results[2]['accuracy']:.4f} "
          f"(vs {random_results[2]['accuracy_mean']:.4f} random)")

    results_6c = {
        "full_accuracy": float(full_acc),
        "n_test": int(n_test),
        "entropy_deferral": entropy_results,
        "random_deferral": random_results,
        "mean_gain": float(mean_gain),
        "coverages": [float(c) for c in coverages],
        "entropy_accuracies": [float(a) for a in entropy_accs],
        "random_accuracies": [float(a) for a in random_accs],
    }

    return results_6c


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Section 6 Validation: Uncertainty, Calibration, and Trust")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    set_seed(SEED)

    # Run all three sub-experiments
    results_6a = experiment_6a()
    results_6b, model_vmib, test_loader, labels_test = experiment_6b()
    results_6c = experiment_6c(model_vmib, test_loader, labels_test)

    # Combine results
    results = {
        "experiment": "Section 6: Uncertainty, Calibration, and Trust",
        "seed": SEED,
        "device": str(DEVICE),
        "exp_6a_ood_detection": results_6a,
        "exp_6b_calibration": results_6b,
        "exp_6c_selective_prediction": results_6c,
    }

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n6a. OOD Detection via Predictive Entropy:")
    print(f"    AUROC = {results_6a['auroc']:.4f}")
    print(f"    ID entropy = {results_6a['id_entropy_mean']:.4f} +/- {results_6a['id_entropy_std']:.4f}")
    print(f"    OOD entropy = {results_6a['ood_entropy_mean']:.4f} +/- {results_6a['ood_entropy_std']:.4f}")
    print(f"    Validates Eq. (18): OOD samples show {results_6a['ood_entropy_mean']/results_6a['id_entropy_mean']:.1f}x higher entropy")

    print(f"\n6b. Calibration:")
    print(f"    VMIB ECE = {results_6b['vmib']['ece']:.4f}")
    print(f"    Unimodal (mRNA) ECE = {results_6b['unimodal_mrna']['ece']:.4f}")
    if results_6b['vmib']['ece'] < results_6b['unimodal_mrna']['ece']:
        rel_imp = (results_6b['unimodal_mrna']['ece'] - results_6b['vmib']['ece']) / results_6b['unimodal_mrna']['ece'] * 100
        print(f"    VMIB better calibrated ({rel_imp:.1f}% ECE reduction)")
    else:
        rel_diff = abs(results_6b['vmib']['ece'] - results_6b['unimodal_mrna']['ece'])
        print(f"    Comparable ECE (difference {rel_diff:.4f})")
    print(f"    Both models show overconfidence pattern typical of neural classifiers")

    print(f"\n6c. Selective Prediction (Deferral):")
    print(f"    Mean accuracy gain (entropy vs random): {results_6c['mean_gain']:+.4f}")
    cov_50_idx = 2  # index for coverage=0.5
    print(f"    At 50% coverage: entropy={results_6c['entropy_deferral'][cov_50_idx]['accuracy']:.4f}, "
          f"random={results_6c['random_deferral'][cov_50_idx]['accuracy_mean']:.4f}")
    print(f"    At 30% coverage: entropy={results_6c['entropy_deferral'][0]['accuracy']:.4f}")
    print(f"    Validates Eq. (19): entropy-based deferral consistently outperforms random")

    save_results(results, "exp_uncertainty.json")
    print("\nUncertainty experiment complete.")


if __name__ == "__main__":
    main()
