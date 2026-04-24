"""Evaluation metrics for detection tasks.

Threshold-free metrics:
    * ``roc_auc``, ``pr_auc``, ``avg_precision``

Threshold-dependent metrics (use ``threshold``):
    * ``accuracy``, ``precision``, ``recall``, ``f1``, ``tpr``, ``fpr``

Note that ``tpr`` equals ``recall`` (sensitivity); both are reported
for convenience — ``recall`` for the PR vocabulary, ``tpr`` for the
ROC vocabulary.
"""

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
        ``tpr``, ``fpr``, ``roc_auc``, ``pr_auc``, and ``avg_precision``.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            auc,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
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

    # Use confusion_matrix with fixed label order to stay safe even if
    # one class is absent at this threshold.
    cm = confusion_matrix(labels_arr, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    results: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds, zero_division=0)),
        "tpr": tpr_val,
        "fpr": fpr_val,
    }

    if len(np.unique(labels_arr)) > 1:
        results["roc_auc"] = float(roc_auc_score(labels_arr, scores_arr))

        precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
        results["pr_auc"] = float(auc(recall, precision))

        results["avg_precision"] = float(average_precision_score(labels_arr, scores_arr))
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")
        results["avg_precision"] = float("nan")

    return results
