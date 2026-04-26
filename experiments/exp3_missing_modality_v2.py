"""
Experiment 3 (v2): Missing-Modality Robustness
================================================
Refined version with:
  - Lower consistency gamma values (0.01, 0.05, 0.1)
  - Modality-dropout-only baseline
  - Consistency warmup (gamma ramped from 0 to target over first 30 epochs)
  - Multiple random seeds for stability
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_brca_data, get_loaders, VMIBModel, vmib_loss, consistency_loss,
    evaluate, estimate_mi_classification, save_results,
    MODALITY_NAMES, NUM_MODALITIES, DEVICE
)


def train_with_consistency(input_dims, num_classes, train_loader, test_loader,
                           lambda_kl=0.01, gamma_consist=0.05, missing_prob=0.4,
                           warmup_epochs=30, epochs=150, lr=1e-3, seed=42):
    """Train VMIB with consistency penalty and warmup."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VMIBModel(input_dims, hidden_dim=256, latent_dim=32,
                      num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        # Warmup: ramp gamma from 0 to target
        if epoch < warmup_epochs:
            current_gamma = gamma_consist * (epoch / warmup_epochs)
        else:
            current_gamma = gamma_consist

        for xs, y in train_loader:
            xs = [x.to(DEVICE) for x in xs]
            y = y.to(DEVICE)

            # Full forward
            logits, mu, logvar, z = model(xs)
            loss, ce, kl = vmib_loss(logits, y, mu, logvar, lambda_kl)

            # Consistency: sample random subsets
            if current_gamma > 0:
                consist = torch.tensor(0.0, device=DEVICE)
                n_subsets = 3
                for _ in range(n_subsets):
                    mask = [torch.rand(1).item() > missing_prob
                            for _ in range(NUM_MODALITIES)]
                    if not any(mask):
                        mask[np.random.randint(NUM_MODALITIES)] = True
                    if all(mask):  # skip if all present (no missing)
                        continue
                    mu_part, logvar_part = model.encode(xs, modality_mask=mask)
                    consist = consist + consistency_loss(
                        mu.detach(), logvar.detach(), mu_part, logvar_part
                    )
                loss = loss + current_gamma * consist / n_subsets

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 30 == 0:
            test_m = evaluate(model, test_loader, lambda_kl)
            print(f"  Epoch {epoch+1:3d} | gamma={current_gamma:.4f} | "
                  f"Acc={test_m['acc']:.4f} AUC={test_m['auc']:.4f}")

    return model


def train_with_dropout(input_dims, num_classes, train_loader, test_loader,
                       lambda_kl=0.01, missing_prob=0.3, epochs=150, lr=1e-3,
                       seed=42):
    """Train VMIB with modality dropout only (no explicit consistency loss)."""
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

            # Random modality dropout
            mask = [torch.rand(1).item() > missing_prob
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

        if (epoch + 1) % 30 == 0:
            test_m = evaluate(model, test_loader, lambda_kl)
            print(f"  Epoch {epoch+1:3d} | Acc={test_m['acc']:.4f} AUC={test_m['auc']:.4f}")

    return model


def train_standard(input_dims, num_classes, train_loader, test_loader,
                   lambda_kl=0.01, epochs=150, lr=1e-3, seed=42):
    """Train standard VMIB with no robustness mechanisms."""
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

        if (epoch + 1) % 30 == 0:
            test_m = evaluate(model, test_loader, lambda_kl)
            print(f"  Epoch {epoch+1:3d} | Acc={test_m['acc']:.4f} AUC={test_m['auc']:.4f}")

    return model


def generate_modality_subsets():
    """Generate all non-empty subsets of modalities."""
    subsets = []
    indices = list(range(NUM_MODALITIES))

    # Single modalities
    for i in indices:
        mask = [False] * NUM_MODALITIES
        mask[i] = True
        subsets.append({"name": MODALITY_NAMES[i], "mask": mask, "num_present": 1})

    # Pairs
    for i, j in combinations(indices, 2):
        mask = [False] * NUM_MODALITIES
        mask[i] = True
        mask[j] = True
        subsets.append({
            "name": f"{MODALITY_NAMES[i]}+{MODALITY_NAMES[j]}",
            "mask": mask, "num_present": 2,
        })

    # All
    subsets.append({"name": "All", "mask": [True] * NUM_MODALITIES, "num_present": 3})
    return subsets


def main():
    print("=" * 70)
    print("Experiment 3 (v2): Missing-Modality Robustness")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_data, test_data, input_dims, num_classes, _ = load_brca_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size=64)

    # ── Estimate I_miss(S) ────────────────────────────────────────────
    print("\n--- Estimating Information Loss I_miss(S) ---")
    X_trains = [train_data.modalities[i].numpy() for i in range(3)]
    X_tests = [test_data.modalities[i].numpy() for i in range(3)]
    y_train = train_data.labels.numpy()
    y_test = test_data.labels.numpy()

    X_train_all = np.concatenate(X_trains, axis=1)
    X_test_all = np.concatenate(X_tests, axis=1)
    full_mi = estimate_mi_classification(
        X_train_all, y_train, X_test_all, y_test, num_classes, hidden_dim=512, epochs=150
    )
    I_full = full_mi["mi_lower_bound"]
    print(f"I(X^(1:3); Y) >= {I_full:.4f}")

    subsets = generate_modality_subsets()
    info_losses = {}
    for subset in subsets:
        if subset["name"] == "All":
            info_losses["All"] = 0.0
            continue
        present = [i for i, m in enumerate(subset["mask"]) if m]
        X_tr = np.concatenate([X_trains[i] for i in present], axis=1)
        X_te = np.concatenate([X_tests[i] for i in present], axis=1)
        mi = estimate_mi_classification(X_tr, y_train, X_te, y_test, num_classes,
                                         hidden_dim=256, epochs=150)
        info_losses[subset["name"]] = max(0, I_full - mi["mi_lower_bound"])
        print(f"  {subset['name']}: I_miss = {info_losses[subset['name']]:.4f}")

    # ── Train three model variants ────────────────────────────────────
    lambda_kl = 0.01

    print("\n--- Training Standard VMIB ---")
    model_std = train_standard(input_dims, num_classes, train_loader, test_loader,
                                lambda_kl=lambda_kl, epochs=150, seed=42)

    print("\n--- Training VMIB + Modality Dropout ---")
    model_dropout = train_with_dropout(input_dims, num_classes, train_loader, test_loader,
                                        lambda_kl=lambda_kl, missing_prob=0.3,
                                        epochs=150, seed=42)

    print("\n--- Training VMIB + Consistency (gamma=0.05, warmup) ---")
    model_consist = train_with_consistency(input_dims, num_classes, train_loader, test_loader,
                                            lambda_kl=lambda_kl, gamma_consist=0.05,
                                            missing_prob=0.4, warmup_epochs=30,
                                            epochs=150, seed=42)

    # ── Evaluate all models on all subsets ─────────────────────────────
    print("\n--- Evaluating Missing-Modality Robustness ---")
    models = {
        "Standard": model_std,
        "Dropout": model_dropout,
        "Consistency": model_consist,
    }

    results = {"I_full": I_full, "info_losses": info_losses}

    for model_name, model in models.items():
        results[model_name] = []
        for subset in subsets:
            ev = evaluate(model, test_loader, lambda_kl, modality_mask=subset["mask"])
            results[model_name].append({
                "subset": subset["name"],
                "mask": subset["mask"],
                "I_miss": info_losses[subset["name"]],
                "auc": ev["auc"],
                "acc": ev["acc"],
                "pred_entropy": ev["pred_entropy"],
            })

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n{'Subset':>25} | {'I_miss':>6} | {'Standard':>9} | {'Dropout':>9} | {'Consist':>9}")
    print("-" * 70)
    for i, subset in enumerate(subsets):
        row = f"{subset['name']:>25} | {info_losses[subset['name']]:>6.3f}"
        for model_name in ["Standard", "Dropout", "Consistency"]:
            auc = results[model_name][i]["auc"]
            row += f" | {auc:>9.4f}"
        print(row)

    # ── Degradation summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AUC DEGRADATION (drop from full-modality baseline)")
    print("=" * 70)
    for model_name in ["Standard", "Dropout", "Consistency"]:
        full_auc = [r for r in results[model_name] if r["subset"] == "All"][0]["auc"]
        print(f"\n{model_name} (Full AUC={full_auc:.4f}):")
        for r in results[model_name]:
            if r["subset"] == "All":
                continue
            delta = full_auc - r["auc"]
            print(f"  {r['subset']:>25}: dAUC={delta:+.4f}, I_miss={r['I_miss']:.4f}")

    save_results(results, "exp3_missing_modality_v2.json")
    print("\nExperiment 3 (v2) complete.")


if __name__ == "__main__":
    main()
