import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict


def plot_label_smoothing_study(
    experiments_data: Dict[str, Dict],
    save_path: str = "results/label_smoothing_study.png"
):
    """
    Plots a multi-panel visual study analyzing the exact impact of Label Smoothing:
    1. Raw Logit Scores Distribution (Pre-Softmax network activations)
    2. Logit Margin Distribution (z_correct - z_other)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    names = list(experiments_data.keys())
    n = len(names)
    
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8.5))
    colors = sns.color_palette("tab10", n)

    for i, name in enumerate(names):
        data = experiments_data[name]
        targets = data["targets"]
        logits = data["logits"]

        # Row 1: Raw Logit Scores Distribution (Before Softmax)
        ax_raw = axes[0, i]
        all_raw_logits = logits.flatten()
        logit_mean = np.mean(all_raw_logits)
        logit_std = np.std(all_raw_logits)

        sns.histplot(
            all_raw_logits,
            bins=30,
            kde=True,
            ax=ax_raw,
            color=colors[i],
            edgecolor="black",
            alpha=0.6
        )
        ax_raw.set_title(
            f"{name}\nRaw Logits (Pre-Softmax)\nStd: {logit_std:.2f} | Range: [{np.min(all_raw_logits):.1f}, {np.max(all_raw_logits):.1f}]",
            fontsize=11,
            fontweight="bold"
        )
        ax_raw.set_xlabel("Raw Score / Logit ($z$)", fontsize=11)
        ax_raw.set_ylabel("Frequency", fontsize=11)

        # Row 2: Logit Difference (z_correct - z_other)
        ax_logit = axes[1, i]
        correct_logits = logits[np.arange(len(targets)), targets]
        other_logits = logits[np.arange(len(targets)), 1 - targets]
        logit_diff = correct_logits - other_logits

        sns.histplot(
            logit_diff,
            bins=30,
            kde=True,
            ax=ax_logit,
            color=colors[i],
            edgecolor="black",
            alpha=0.6
        )
        ax_logit.set_title(f"Logit Margins ($z_{{correct}} - z_{{other}}$)", fontsize=11, fontweight="bold")
        ax_logit.set_xlabel("Margin Value", fontsize=11)
        ax_logit.set_ylabel("Count", fontsize=11)

    plt.suptitle("Detailed Analysis of Label Smoothing on Bank Marketing Dataset", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: '{save_path}'")


def plot_training_and_evaluation_metrics(
    experiments_data: Dict[str, Dict],
    save_path: str = "results/training_and_metrics.png"
):
    """
    Plots training dynamics and evaluation metrics:
    1. Top Left: Training & Validation Loss curves over epochs for all models.
    2. Top Right: ROC Curves & AUC comparison on Test set.
    3. Bottom Row: Confusion Matrices for each epsilon model on Test set.
    """
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    names = list(experiments_data.keys())
    n = len(names)
    colors = sns.color_palette("tab10", n)

    fig = plt.figure(figsize=(5 * n, 10))
    gs = fig.add_gridspec(2, n, height_ratios=[1.2, 1.0])

    # Panel 1: Training & Validation Loss Curves (Top Left)
    ax_loss = fig.add_subplot(gs[0, : n // 2])
    for i, name in enumerate(names):
        history = experiments_data[name]["history"]
        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs_range, history["train_loss"], label=f"{name} (Train)", color=colors[i], linewidth=2.0)
        ax_loss.plot(epochs_range, history["val_loss"], label=f"{name} (Val)", color=colors[i], linestyle="--", linewidth=1.8, alpha=0.8)

    ax_loss.set_title("Training and Validation Loss Curves", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: ROC Curves & AUC (Top Right)
    ax_roc = fig.add_subplot(gs[0, n // 2 :])
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")

    for i, name in enumerate(names):
        probs = np.nan_to_num(experiments_data[name]["probs"], nan=0.5)
        targets = experiments_data[name]["targets"]
        try:
            fpr, tpr, _ = roc_curve(targets, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            if np.isnan(roc_auc):
                roc_auc = 0.50
        except Exception:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.50
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=colors[i], linewidth=2.0)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.legend(loc="lower right", fontsize=9.5)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Confusion Matrices (Bottom Row)
    class_labels = ["No Deposit (0)", "Deposit (1)"]

    for i, name in enumerate(names):
        ax_cm = fig.add_subplot(gs[1, i])
        probs = experiments_data[name]["probs"]
        targets = experiments_data[name]["targets"]
        preds = np.argmax(probs, axis=1)
        
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

        # Annotate with count and percentage
        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]}\n({cm_norm[r, c]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["Pred: 0", "Pred: 1"],
            yticklabels=["True: 0", "True: 1"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax_cm.set_title(f"Confusion Matrix\n{name}", fontsize=11, fontweight="bold")

    plt.suptitle("Model Dynamics, ROC Curves and Confusion Matrices (Bank Marketing Test Set)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves, ROC and Confusion Matrices saved to: '{save_path}'")


def plot_threshold_table(
    sweep_results: list,
    experiment_name: str = "Smoothing (ε = 0.45)",
    save_path: str = "results/threshold_table.png"
):
    """
    Renders and saves a clean, publication-ready visual table figure with:
      - Cutoff (T)
      - Recall (Taxa de Captação)
      - Clientes Contatados (Nº e % da base)
      - Clientes Convertidos (Verdadeiros Positivos)
      - Taxa de Conversão (%)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    headers = [
        "Cutoff (T)",
        "Recall (Captação)",
        "Clientes Contatados",
        "Convertidos",
        "Taxa de Conversão"
    ]

    cell_data = []
    for row in sweep_results:
        t_val = f"P >= {row['threshold']:.3f}"
        rec = f"{row['recall']:.2f}%"
        contacted = f"{row['pos_predicted']:,}  ({row['contact_rate']:.1f}% da base)"
        converted = f"{row['true_positives']:,}"
        conversion = f"{row['conversion_rate']:.2f}%"
        cell_data.append([t_val, rec, contacted, converted, conversion])

    num_rows = len(cell_data)
    fig_height = max(4.0, 1.4 + num_rows * 0.48)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")

    # Titles
    fig.text(0.5, 0.94, "Impacto do Cutoff no Desempenho e Conversão de Campanha", 
             ha="center", va="top", fontsize=14, fontweight="bold", color="#0F172A")
    fig.text(0.5, 0.88, f"Modelo: {experiment_name} | Avaliação no Conjunto de Teste (N = {sweep_results[0]['total_samples']:,})", 
             ha="center", va="top", fontsize=11, color="#475569")

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.76]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style Header and Zebra Rows
    header_bg = "#1E3A8A"   # Navy Blue
    header_fg = "#FFFFFF"
    zebra_even = "#F1F5F9"  # Slate Light
    zebra_odd = "#FFFFFF"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(1.0)
        if row_idx == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color=header_fg, fontweight="bold", fontsize=11.5)
            cell.set_height(0.12)
        else:
            bg_color = zebra_even if row_idx % 2 == 0 else zebra_odd
            cell.set_facecolor(bg_color)
            cell.set_text_props(color="#0F172A", fontsize=10.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Threshold table figure saved to: '{save_path}'")


def plot_threshold_histograms(
    probs: np.ndarray,
    targets: np.ndarray,
    sweep_results: list,
    experiment_name: str = "Smoothing (ε = 0.45)",
    save_path: str = "results/threshold_histograms.png"
):
    """
    Plots a grid of histograms for the predicted probabilities of the entire test dataset (all 9,043 instances).
    For each cutoff threshold, the bars >= cutoff are highlighted using the exact model color (tab10 index 3),
    while the bars < cutoff remain muted.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    
    # Extract probabilities for the whole test dataset
    all_probs = probs[:, 1] if probs.ndim > 1 else probs
    total_samples = len(all_probs)
    n_class_0 = int((targets == 0).sum())
    n_class_1 = int((targets == 1).sum())
    
    # Range of the entire test dataset
    p_min = float(np.min(all_probs))
    p_max = float(np.max(all_probs))
    
    # Use the exact same color palette as the Label Smoothing Study (tab10 blue)
    palette = sns.color_palette("tab10", 4)
    active_color = palette[0]     # Blue color
    inactive_color = "#94A3B8"   # Slate gray for P < Cutoff
    
    n_plots = len(sweep_results)
    cols = 2
    rows = (n_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.6 * rows))
    axes = axes.flatten()
    
    # Dynamic binning spanning the actual distribution range of the entire test set
    bins = np.linspace(p_min - 0.005, p_max + 0.005, 35)
    counts, bin_edges = np.histogram(all_probs, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    for i, row in enumerate(sweep_results):
        ax = axes[i]
        t = row["threshold"]
        
        # Color array: Model's tab10 color if >= threshold, else Slate Gray
        bar_colors = [active_color if c >= t else inactive_color for c in bin_centers]
        
        ax.bar(
            bin_centers, 
            counts, 
            width=bin_widths, 
            color=bar_colors, 
            edgecolor="black", 
            linewidth=0.6, 
            alpha=0.7
        )
        
        # Add KDE curve over the histogram
        sns.kdeplot(
            all_probs, 
            ax=ax, 
            color="#1E293B", 
            linewidth=1.5, 
            bw_adjust=0.8
        )
        
        # Vertical line for cutoff
        ax.axvline(t, color="#1E293B", linestyle="--", linewidth=1.8, label=rf"Cutoff ($T = {t:.3f}$)")
        
        # Info box with full dataset context
        info_text = (
            f"Recall: {row['recall']:.2f}%\n"
            f"Tx. Conversão: {row['conversion_rate']:.2f}%\n"
            f"Contatados: {row['pos_predicted']:,} / {total_samples:,} ({row['contact_rate']:.1f}%)\n"
            f"Convertidos: {row['true_positives']:,} / {n_class_1:,}"
        )
        ax.text(
            0.96, 0.94, 
            info_text,
            transform=ax.transAxes,
            fontsize=9.2,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#CBD5E1')
        )
        
        ax.set_title(rf"Cutoff $P \geq {t:.3f}$  |  Faixa Real: [{p_min:.3f}, {p_max:.3f}]", fontweight="bold", fontsize=11)
        ax.set_xlabel("Probabilidade Predita P(Depósito = Sim)", fontsize=9.5)
        ax.set_ylabel("Frequência / Densidade", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(p_min - 0.015, p_max + 0.015)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.suptitle(
        f"Distribuição Completa de Probabilidades no Conjunto de Teste (N = {total_samples:,} | {n_class_0:,} Não / {n_class_1:,} Sim)\nModelo: {experiment_name}",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Threshold histograms saved to: '{save_path}'")
