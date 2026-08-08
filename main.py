"""
Main Experiment: Detailed Study of Label Smoothing on the UCI Bank Marketing Dataset.

Full-Batch Gradient Descent (Full Dataset Processing per epoch without mini-batches).
Evaluates how varying the Label Smoothing parameter (epsilon = 0.0, 0.15, 0.30, 0.45)
impacts:
1. Raw Logit Score Distribution (pre-softmax activations)
2. Logit Margins (z_correct - z_other)
3. Precision, Recall, F1, Accuracy, AUC-ROC, and ECE
4. Training & Validation Loss curves and Confusion Matrices
"""

import sys
import argparse
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data import get_bank_marketing_data
from src.model import TabularMLP
from src.losses import LabelSmoothingCrossEntropy
from src.calibration import compute_metrics, compute_threshold_sweep
from src.visualize import (
    plot_label_smoothing_study,
    plot_training_and_evaluation_metrics,
    plot_threshold_table,
    plot_threshold_histograms
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_data, val_data, criterion, optimizer, epochs, device, patience=25):
    """
    Full-Batch Training with Early Stopping (Patience) and Best Checkpoint Restoration.
    """
    history = {"train_loss": [], "val_loss": []}
    X_train, y_train = train_data[0].to(device), train_data[1].to(device)
    X_val, y_val = val_data[0].to(device), val_data[1].to(device)

    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # 1. Forward & Backward on full training set
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 2. Validation loss on full validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)

        train_loss_val = float(loss.item())
        val_loss_val = float(val_loss.item())
        history["train_loss"].append(train_loss_val)
        history["val_loss"].append(val_loss_val)

        # Early Stopping Check
        if val_loss_val < best_val_loss - 1e-5:
            best_val_loss = val_loss_val
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == epochs or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f} (Best Val: {best_val_loss:.4f} @ Ep {best_epoch:03d})")

        if patience > 0 and patience_counter >= patience:
            print(f"    --> [Early Stopping] Triggered at Epoch {epoch:03d} (Patience={patience}). Restoring Best Weights from Epoch {best_epoch:03d} (Val Loss: {best_val_loss:.4f}).")
            break

    # Restore best checkpoint
    model.load_state_dict(best_weights)
    return history


def evaluate_model(model, test_data, device):
    """
    Full-Batch Evaluation on the entire test set.
    """
    model.eval()
    X_test, y_test = test_data[0].to(device), test_data[1].to(device)

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=-1)

    logits_arr = np.nan_to_num(logits.cpu().numpy(), nan=0.0)
    probs_arr = np.nan_to_num(probs.cpu().numpy(), nan=0.5)
    targets_arr = y_test.cpu().numpy()

    metrics = compute_metrics(probs_arr, targets_arr)
    
    # Safe AUC-ROC Calculation
    try:
        if len(np.unique(targets_arr)) > 1:
            auc_val = roc_auc_score(targets_arr, probs_arr[:, 1]) * 100.0
            metrics["AUC_ROC"] = float(auc_val) if not np.isnan(auc_val) else 50.0
        else:
            metrics["AUC_ROC"] = 50.0
    except Exception:
        metrics["AUC_ROC"] = 50.0

    return logits_arr, probs_arr, targets_arr, metrics


def main():
    parser = argparse.ArgumentParser(description="Label Smoothing Study on Bank Marketing Dataset (Full-Batch Training)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs per model (default: 500)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience in epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--balance_method",
        type=str,
        default="oversample",
        choices=["oversample", "weights", "none"],
        help="Imbalance handling: 'oversample' (balances train to 50/50), 'weights' (cost-sensitive loss), or 'none'"
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.0, 0.15, 0.30, 0.45],
        help="List of epsilon values to evaluate: e.g. 0.0 0.15 0.30 0.45"
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128, 64, 32],
        help="Hidden layer dimensions for the MLP (default: [128, 64, 32])"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print(f"  LABEL SMOOTHING BENCHMARK - FULL-BATCH TRAINING (NO MINI-BATCHES)")
    print(f"  Device: {device} | Max Epochs: {args.epochs} | Patience: {args.patience}")
    print(f"  Architecture: {args.hidden_dims} | Balancing Strategy: {args.balance_method.upper()}")
    print(f"  Epsilon values: {args.epsilons}")
    print("=" * 80 + "\n")

    # 1. Load Bank Marketing Dataset (Full Tensors with Balancing)
    train_data, val_data, test_data, in_features, class_weights = get_bank_marketing_data(
        balance_method=args.balance_method,
        random_state=args.seed
    )

    # 2. Experiment Matrix with different Epsilon values
    epsilon_experiments = []
    for eps in args.epsilons:
        if eps == 0.0:
            name = "Baseline (ε = 0.00)"
        else:
            name = f"Smoothing (ε = {eps:.2f})"
        epsilon_experiments.append({"name": name, "eps": eps})

    results_collection = {}
    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) if args.balance_method == "weights" else None

    for exp in epsilon_experiments:
        name = exp["name"]
        eps = exp["eps"]
        print(f"\n---> Training: {name}...")

        model = TabularMLP(
            in_features=in_features,
            hidden_dims=args.hidden_dims,
            dropout=0.1
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Architecture: {args.hidden_dims} | Trainable Params: {total_params:,}")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if eps == 0.0:
            criterion = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            criterion = LabelSmoothingCrossEntropy(epsilon=eps, weight=loss_weights)

        history = train_model(model, train_data, val_data, criterion, optimizer, args.epochs, device, patience=args.patience)
        logits, probs, targets, metrics = evaluate_model(model, test_data, device)

        results_collection[name] = {
            "logits": logits,
            "probs": probs,
            "targets": targets,
            "metrics": metrics,
            "history": history,
            "epsilon": eps
        }
        print(
            f"  Results: Acc={metrics['Accuracy']:.2f}% | "
            f"Precision={metrics['Precision']:.2f}% | "
            f"Recall={metrics['Recall']:.2f}% | "
            f"F1={metrics['F1_Score']:.2f}% | "
            f"AUC={metrics['AUC_ROC']:.2f}% | "
            f"ECE={metrics['ECE']:.4f}"
        )

    # 3. Formatted Summary Table
    print("\n" + "=" * 105)
    print(f"{'Experiment':<24} | {'Accuracy':<10} | {'Precision':<11} | {'Recall':<9} | {'F1-Score':<10} | {'AUC-ROC':<9} | {'ECE':<8}")
    print("-" * 105)
    for name, data in results_collection.items():
        m = data["metrics"]
        print(
            f"{name:<24} | "
            f"{m['Accuracy']:>8.2f}% | "
            f"{m['Precision']:>9.2f}% | "
            f"{m['Recall']:>7.2f}% | "
            f"{m['F1_Score']:>8.2f}% | "
            f"{m['AUC_ROC']:>7.2f}% | "
            f"{m['ECE']:>8.4f}"
        )
    print("=" * 105)

    # 4. Decision Threshold Sweep Analysis (Focusing on the 4th Algorithm: Smoothing ε = 0.45)
    fourth_name = list(results_collection.keys())[-1]
    fourth_probs = results_collection[fourth_name]["probs"]
    fourth_targets = results_collection[fourth_name]["targets"]
    sweep_thresholds = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
    sweep_results = compute_threshold_sweep(fourth_probs, fourth_targets, sweep_thresholds)

    print("\n" + "=" * 115)
    print(f"  TABELA DE IMPACTO DO CUTOFF - 4º ALGORITMO: {fourth_name.upper()}")
    print("=" * 105)
    print(f"{'Cutoff (T)':<14} | {'Recall (Captação)':<18} | {'Clientes Contatados':<24} | {'Convertidos':<14} | {'Taxa de Conversão':<18}")
    print("-" * 105)
    for row in sweep_results:
        t_str = f"P >= {row['threshold']:.3f}"
        print(
            f"{t_str:<14} | "
            f"{row['recall']:>11.2f}%    | "
            f"{row['pos_predicted']:>7,} ({row['contact_rate']:>5.1f}% base)   | "
            f"{row['true_positives']:>8,}      | "
            f"{row['conversion_rate']:>12.2f}%"
        )
    print("=" * 105)

    # 5. Generate Visual Comparisons
    # Study 1: Logits and Margins Distribution
    plot_label_smoothing_study(results_collection, save_path="results/label_smoothing_study.png")
    
    # Study 2: Training & Validation Loss curves, ROC-AUC and Confusion Matrices
    plot_training_and_evaluation_metrics(results_collection, save_path="results/training_and_metrics.png")

    # Study 3: Clean Visual Table Figure (Cutoff, Recall, Contatados, Convertidos, Tx. Conversão)
    plot_threshold_table(sweep_results, experiment_name=fourth_name, save_path="results/threshold_table.png")
    
    # Study 4: Threshold Probability Histograms
    fourth_logits = results_collection[fourth_name]["logits"]
    plot_threshold_histograms(fourth_logits, fourth_targets, sweep_results, experiment_name=fourth_name, save_path="results/threshold_histograms.png")
    
    print(f"\nAll studies completed successfully! Check the 'results/' folder:\n  1. results/label_smoothing_study.png\n  2. results/training_and_metrics.png\n  3. results/threshold_table.png\n  4. results/threshold_histograms.png\n")


if __name__ == "__main__":
    main()
