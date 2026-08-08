import numpy as np
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """
    Computes classification and uncertainty calibration metrics:
    - Accuracy (%)
    - Precision (%)
    - Recall (%)
    - F1-Score (%)
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Average Confidence
    - Brier Score
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(conf_in_bin - acc_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

    # Brier Score for binary classification
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    brier_score = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    # Precision, Recall and F1 for positive class (1 = deposit subscribed / yes)
    precision = precision_score(targets, predictions, pos_label=1, average="binary", zero_division=0) * 100.0
    recall = recall_score(targets, predictions, pos_label=1, average="binary", zero_division=0) * 100.0
    f1 = f1_score(targets, predictions, pos_label=1, average="binary", zero_division=0) * 100.0

    return {
        "Accuracy": np.mean(accuracies) * 100.0,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ECE": ece,
        "MCE": mce,
        "Avg_Confidence": np.mean(confidences),
        "Brier_Score": brier_score
    }


def compute_threshold_sweep(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: list = None
):
    """
    Computes Recall, Conversion Rate, Contact Rate, TP count across varying cutoff thresholds.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
    results = []
    pos_probs = probabilities[:, 1]
    
    for t in thresholds:
        preds = (pos_probs >= t).astype(int)
        prec = precision_score(targets, preds, pos_label=1, average="binary", zero_division=0) * 100.0
        rec = recall_score(targets, preds, pos_label=1, average="binary", zero_division=0) * 100.0
        f1 = f1_score(targets, preds, pos_label=1, average="binary", zero_division=0) * 100.0
        acc = np.mean(preds == targets) * 100.0
        pos_predicted = int(np.sum(preds == 1))
        total_samples = len(targets)
        contact_rate = (pos_predicted / total_samples) * 100.0 if total_samples > 0 else 0.0
        tp_count = int(np.sum((preds == 1) & (targets == 1)))
        
        results.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "accuracy": float(acc),
            "conversion_rate": float(prec),
            "contact_rate": float(contact_rate),
            "pos_predicted": pos_predicted,
            "true_positives": tp_count,
            "total_samples": total_samples
        })
    return results
