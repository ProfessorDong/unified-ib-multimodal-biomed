"""
Shared utilities for VMIB experiments on TCGA-BRCA multi-omics data.
Provides data loading, VMIB model, MI estimation, and training helpers.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import json
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "MOGONET" / "BRCA"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODALITY_NAMES = ["mRNA", "Methylation", "miRNA"]
NUM_MODALITIES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data Loading ─────────────────────────────────────────────────────────
class MultiOmicsDataset(Dataset):
    """TCGA-BRCA multi-omics dataset from MOGONET."""

    def __init__(self, modalities, labels):
        """
        Args:
            modalities: list of np.ndarray, one per modality
            labels: np.ndarray of integer labels
        """
        self.modalities = [torch.FloatTensor(m) for m in modalities]
        self.labels = torch.LongTensor(labels)
        self.n_samples = len(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [m[idx] for m in self.modalities], self.labels[idx]


def load_brca_data():
    """Load and standardize TCGA-BRCA multi-omics data.

    Returns:
        train_data, test_data: MultiOmicsDataset instances
        input_dims: list of int, feature dimensions per modality
        num_classes: int
        scalers: list of fitted StandardScaler instances
    """
    modalities_train, modalities_test = [], []
    scalers = []
    input_dims = []

    for i in range(1, 4):
        tr = pd.read_csv(DATA_DIR / f"{i}_tr.csv", header=None).values
        te = pd.read_csv(DATA_DIR / f"{i}_te.csv", header=None).values
        scaler = StandardScaler()
        tr = scaler.fit_transform(tr)
        te = scaler.transform(te)
        modalities_train.append(tr)
        modalities_test.append(te)
        scalers.append(scaler)
        input_dims.append(tr.shape[1])

    labels_tr = pd.read_csv(DATA_DIR / "labels_tr.csv", header=None).values.ravel().astype(int)
    labels_te = pd.read_csv(DATA_DIR / "labels_te.csv", header=None).values.ravel().astype(int)
    num_classes = len(np.unique(labels_tr))

    train_data = MultiOmicsDataset(modalities_train, labels_tr)
    test_data = MultiOmicsDataset(modalities_test, labels_te)

    return train_data, test_data, input_dims, num_classes, scalers


def get_loaders(train_data, test_data, batch_size=64):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ─── Model Components ─────────────────────────────────────────────────────
class ModalityEncoder(nn.Module):
    """Per-modality encoder with two hidden layers."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class VMIBModel(nn.Module):
    """Variational Multimodal Information Bottleneck model.

    Architecture:
        Per-modality encoders -> concatenation -> variational bottleneck (mu, logvar) -> predictor
    """

    def __init__(self, input_dims, hidden_dim=256, latent_dim=32, num_classes=5):
        super().__init__()
        self.num_modalities = len(input_dims)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Per-modality encoders
        self.encoders = nn.ModuleList([
            ModalityEncoder(d, hidden_dim) for d in input_dims
        ])

        # Fusion -> variational bottleneck
        fusion_dim = self.num_modalities * hidden_dim
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, xs, modality_mask=None):
        """Encode modalities with optional masking.

        Args:
            xs: list of tensors, one per modality
            modality_mask: list of bool or None. True = present, False = zeroed out.
        """
        features = []
        for i, (enc, x) in enumerate(zip(self.encoders, xs)):
            if modality_mask is not None and not modality_mask[i]:
                features.append(torch.zeros(x.size(0), self.hidden_dim, device=x.device))
            else:
                features.append(enc(x))
        fused = torch.cat(features, dim=-1)
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, xs, modality_mask=None):
        mu, logvar = self.encode(xs, modality_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.predictor(z)
        return logits, mu, logvar, z


# ─── Loss Functions ───────────────────────────────────────────────────────
def vmib_loss(logits, y, mu, logvar, lambda_kl):
    """VMIB loss = cross-entropy + lambda * KL(q(z|x) || N(0,I))."""
    ce = F.cross_entropy(logits, y)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return ce + lambda_kl * kl, ce, kl


def consistency_loss(mu_full, logvar_full, mu_partial, logvar_partial):
    """Forward KL: KL(q(z|x^{1:M}) || q(z|x^{S})).

    Encourages partial-observation encoder to cover the full encoder's support.
    """
    var_full = logvar_full.exp()
    var_partial = logvar_partial.exp()
    kl = 0.5 * (
        logvar_partial - logvar_full
        + (var_full + (mu_full - mu_partial).pow(2)) / var_partial
        - 1
    ).sum(dim=-1).mean()
    return kl


# ─── Training Helpers ─────────────────────────────────────────────────────
def train_vmib_epoch(model, loader, optimizer, lambda_kl, gamma_consist=0.0,
                     missing_prob=0.0):
    """Train one epoch, optionally with consistency penalty and modality dropout.

    Args:
        gamma_consist: weight for consistency loss (0 = disabled)
        missing_prob: probability of dropping each modality during consistency training
    """
    model.train()
    total_loss, total_ce, total_kl, total_consist = 0, 0, 0, 0
    n = 0

    for xs, y in loader:
        xs = [x.to(DEVICE) for x in xs]
        y = y.to(DEVICE)
        bs = y.size(0)

        # Full forward pass
        logits, mu, logvar, z = model(xs)
        loss, ce, kl = vmib_loss(logits, y, mu, logvar, lambda_kl)

        # Consistency penalty: sample random modality subsets
        consist = torch.tensor(0.0, device=DEVICE)
        if gamma_consist > 0 and missing_prob > 0:
            for _ in range(2):  # 2 random subsets per batch
                mask = [torch.rand(1).item() > missing_prob for _ in range(NUM_MODALITIES)]
                if not any(mask):  # ensure at least one modality
                    mask[np.random.randint(NUM_MODALITIES)] = True
                mu_part, logvar_part = model.encode(xs, modality_mask=mask)
                consist = consist + consistency_loss(mu.detach(), logvar.detach(),
                                                     mu_part, logvar_part)
            consist = consist / 2.0
            loss = loss + gamma_consist * consist

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_kl += kl.item() * bs
        total_consist += consist.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kl": total_kl / n,
        "consist": total_consist / n,
    }


@torch.no_grad()
def evaluate(model, loader, lambda_kl, modality_mask=None):
    """Evaluate model, optionally with modality mask."""
    model.eval()
    all_logits, all_labels, all_mu, all_logvar, all_z = [], [], [], [], []
    total_ce, total_kl = 0, 0
    n = 0

    for xs, y in loader:
        xs = [x.to(DEVICE) for x in xs]
        y = y.to(DEVICE)
        bs = y.size(0)

        logits, mu, logvar, z = model(xs, modality_mask=modality_mask)
        _, ce, kl = vmib_loss(logits, y, mu, logvar, lambda_kl)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())
        all_z.append(z.cpu())
        total_ce += ce.item() * bs
        total_kl += kl.item() * bs
        n += bs

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    mu = torch.cat(all_mu)
    logvar = torch.cat(all_logvar)
    z = torch.cat(all_z)

    probs = F.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    labels_np = labels.numpy()

    acc = accuracy_score(labels_np, preds)
    try:
        auc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # Predictive entropy
    pred_entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1).mean()

    return {
        "acc": acc,
        "auc": auc,
        "ce": total_ce / n,
        "kl": total_kl / n,
        "pred_entropy": pred_entropy,
        "mu": mu,
        "logvar": logvar,
        "z": z,
        "probs": probs,
        "labels": labels_np,
    }


def train_vmib(input_dims, num_classes, train_loader, test_loader,
               lambda_kl=0.01, gamma_consist=0.0, missing_prob=0.3,
               hidden_dim=256, latent_dim=32, epochs=100, lr=1e-3,
               verbose=True):
    """Full VMIB training loop. Returns trained model and history."""
    model = VMIBModel(input_dims, hidden_dim, latent_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train": [], "test": []}

    for epoch in range(epochs):
        train_metrics = train_vmib_epoch(model, train_loader, optimizer, lambda_kl,
                                          gamma_consist, missing_prob)
        test_metrics = evaluate(model, test_loader, lambda_kl)
        scheduler.step()

        history["train"].append(train_metrics)
        history["test"].append({k: v for k, v in test_metrics.items()
                                if isinstance(v, (int, float))})

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | CE={train_metrics['ce']:.4f} "
                  f"KL={train_metrics['kl']:.4f} | Test Acc={test_metrics['acc']:.4f} "
                  f"AUC={test_metrics['auc']:.4f}")

    return model, history


# ─── MI Estimation via Classification ─────────────────────────────────────
class SimpleClassifier(nn.Module):
    """Simple MLP classifier for MI lower-bound estimation."""

    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def estimate_mi_classification(X_train, y_train, X_test, y_test, num_classes,
                                hidden_dim=256, epochs=100, lr=1e-3):
    """Estimate I(X;Y) lower bound via classifier cross-entropy.

    I(X;Y) = H(Y) - H(Y|X) >= H(Y) - CE_classifier(Y|X)

    Uses a train/validation split for epoch selection (early stopping on
    validation CE), then evaluates the final MI bound on the held-out test set.
    This avoids test-set leakage that would inflate the lower bound.

    Returns dict with mi_lower_bound, accuracy, auc, H_Y.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    # H(Y) from empirical distribution
    _, counts = np.unique(y_train, return_counts=True)
    p_y = counts / counts.sum()
    H_Y = -np.sum(p_y * np.log(p_y + 1e-10))

    # Split training data into train (80%) and validation (20%) for early stopping
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim, num_classes, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)
    X_te_t = torch.FloatTensor(X_test).to(DEVICE)
    y_te_t = torch.LongTensor(y_test).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_ce = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Early stopping on validation set (NOT test set)
        model.eval()
        with torch.no_grad():
            val_ce = F.cross_entropy(model(X_val_t), y_val_t).item()
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model and evaluate on held-out test set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        test_ce = F.cross_entropy(logits, y_te_t).item()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = y_te_t.cpu().numpy()

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    mi_lower = H_Y - test_ce

    return {
        "mi_lower_bound": max(0, mi_lower),  # MI is non-negative
        "H_Y": H_Y,
        "test_ce": test_ce,
        "val_ce": best_val_ce,
        "accuracy": acc,
        "auc": auc,
    }


def save_results(results, filename):
    """Save results dict to JSON."""
    filepath = RESULTS_DIR / filename
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {filepath}")


# ─── Consistent MI Estimation via Single Masked Model ─────────────────────
def evaluate_all_subsets(model, test_loader, num_modalities, lambda_kl, H_Y):
    """Evaluate a SINGLE trained model on all 2^M - 1 non-empty modality subsets.

    Returns a dict mapping subset tuple -> {mi, ce, acc, auc}.
    Using one model for all subsets ensures consistent bound tightness,
    making differences of MI estimates meaningful.
    """
    from itertools import combinations

    indices = list(range(num_modalities))
    results = {}

    for r in range(1, num_modalities + 1):
        for subset in combinations(indices, r):
            mask = [False] * num_modalities
            for i in subset:
                mask[i] = True

            ev = evaluate(model, test_loader, lambda_kl, modality_mask=mask)
            mi = max(0, H_Y - ev["ce"])

            subset_key = tuple(sorted(subset))
            results[subset_key] = {
                "mi": mi,
                "ce": ev["ce"],
                "acc": ev["acc"],
                "auc": ev["auc"],
                "pred_entropy": ev["pred_entropy"],
            }

    return results


def bootstrap_ci(model, test_data, num_modalities, lambda_kl, H_Y,
                 n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence intervals for MI estimates from all subsets.

    Resamples the test set n_bootstrap times and recomputes CE (and thus MI)
    for each subset on each resample. Returns per-subset mean, std, and CI.
    """
    from itertools import combinations

    rng = np.random.RandomState(seed)
    n = len(test_data)
    indices_list = list(range(num_modalities))

    # Collect all subsets
    all_subsets = []
    for r in range(1, num_modalities + 1):
        for subset in combinations(indices_list, r):
            all_subsets.append(tuple(sorted(subset)))

    # Pre-compute model outputs on full test set for each mask
    model.eval()
    subset_logits = {}
    all_labels = test_data.labels.numpy()

    for subset in all_subsets:
        mask = [False] * num_modalities
        for i in subset:
            mask[i] = True

        all_logits_list = []
        loader = DataLoader(test_data, batch_size=64, shuffle=False)
        with torch.no_grad():
            for xs, y in loader:
                xs = [x.to(DEVICE) for x in xs]
                logits, _, _, _ = model(xs, modality_mask=mask)
                all_logits_list.append(logits.cpu())
        subset_logits[subset] = torch.cat(all_logits_list)

    # Bootstrap
    boot_mi = {s: [] for s in all_subsets}
    alpha = (1 - ci) / 2

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = torch.LongTensor(all_labels[idx])

        for subset in all_subsets:
            logits_boot = subset_logits[subset][idx]
            ce = F.cross_entropy(logits_boot, y_boot).item()
            mi = max(0, H_Y - ce)
            boot_mi[subset].append(mi)

    results = {}
    for subset in all_subsets:
        vals = np.array(boot_mi[subset])
        results[subset] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "ci_lower": np.percentile(vals, 100 * alpha),
            "ci_upper": np.percentile(vals, 100 * (1 - alpha)),
        }

    return results


class ConcatMLP(nn.Module):
    """Simple concatenation MLP baseline (no variational bottleneck)."""

    def __init__(self, input_dims, hidden_dim=256, num_classes=5):
        super().__init__()
        total_dim = sum(input_dims)
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, xs):
        x = torch.cat(xs, dim=-1)
        return self.net(x)
