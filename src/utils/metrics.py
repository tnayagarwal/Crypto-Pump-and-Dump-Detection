"""Evaluation metrics for anomaly detection model validation."""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true: list, y_pred: list, y_prob: list = None) -> dict:
    """
    Compute classification metrics for anomaly detection evaluation.

    Args:
        y_true: Ground truth binary labels [0 = normal, 1 = anomaly]
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for anomaly class (optional, for AUC)

    Returns:
        Dictionary with precision, recall, f1, and optionally roc_auc
    """
    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics
