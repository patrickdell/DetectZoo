"""Evaluation metrics for detection tasks."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def compute_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute standard binary-classification metrics.

    Parameters:
        labels: Ground-truth binary labels (``1`` = AI, ``0`` = human).
        scores: Detector scores (higher → more likely AI).
        threshold: Decision threshold for deriving predicted labels.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``auroc``, and ``avg_precision``.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for compute_metrics. "
            "Install it with: pip install detectzoo[eval]"
        ) from exc

    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= threshold).astype(int)

    results: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds, zero_division=0)),
    }

    if len(np.unique(labels_arr)) > 1:
        results["auroc"] = float(roc_auc_score(labels_arr, scores_arr))
        results["avg_precision"] = float(average_precision_score(labels_arr, scores_arr))
    else:
        results["auroc"] = float("nan")
        results["avg_precision"] = float("nan")

    return results
