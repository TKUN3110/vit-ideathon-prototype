"""
Clinical Evaluation Metrics for ICU Readmission Prediction.
Calculates AUROC, AUPRC, Brier Score, Sensitivity, and Specificity.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def compute_clinical_metrics(
    y_true: List[float],
    y_pred_probs: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes comprehensive clinical validation metrics.
    """
    y_t = np.array(y_true, dtype=np.float32)
    y_p = np.array(y_pred_probs, dtype=np.float32)

    if len(y_t) == 0:
        return {"loss": 0.0, "auroc": 0.5, "auprc": 0.0, "brier": 0.0, "accuracy": 0.0}

    # Brier Score (Probability calibration error: lower is better)
    brier = float(brier_score_loss(y_t, y_p))

    # AUROC & AUPRC (Safe handling for single-class batches)
    if len(np.unique(y_t)) > 1:
        try:
            auroc = float(roc_auc_score(y_t, y_p))
        except Exception:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_t, y_p))
        except Exception:
            auprc = float(np.mean(y_t))
    else:
        auroc = 0.5
        auprc = float(np.mean(y_t))

    # Binary metrics at decision threshold
    y_pred_bin = (y_p >= threshold).astype(int)
    acc = float(accuracy_score(y_t, y_pred_bin))

    # Sensitivity (Recall) and Specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred_bin, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        sensitivity = 0.0
        specificity = 0.0

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
    }


class ClinicalMetricsTracker:
    """Accumulates batch predictions and targets during evaluation loops."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.y_true: List[float] = []
        self.y_pred: List[float] = []
        self.total_loss: float = 0.0
        self.num_batches: int = 0
        self.num_samples: int = 0

    def update(self, targets: List[float], probs: List[float], batch_loss: float, batch_size: int) -> None:
        self.y_true.extend(targets)
        self.y_pred.extend(probs)
        self.total_loss += batch_loss * batch_size
        self.num_batches += 1
        self.num_samples += batch_size

    def compute(self) -> Dict[str, float]:
        metrics = compute_clinical_metrics(self.y_true, self.y_pred)
        avg_loss = (self.total_loss / self.num_samples) if self.num_samples > 0 else 0.0
        metrics["loss"] = round(avg_loss, 4)
        metrics["num_samples"] = self.num_samples
        return metrics
