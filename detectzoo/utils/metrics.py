"""Evaluation metrics for detection tasks.

Threshold-free metrics:
    * ``roc_auc``, ``pr_auc``, ``avg_precision``, ``eer``

Threshold-dependent metrics (use ``threshold``):
    * ``accuracy``, ``precision``, ``recall``, ``f1``, ``tpr``, ``fpr``

Note that ``tpr`` equals ``recall`` (sensitivity); both are reported
for convenience — ``recall`` for the PR vocabulary, ``tpr`` for the
ROC vocabulary.

``eer`` (Equal Error Rate) is the operating-point error where the false
acceptance rate equals the false rejection rate; standard for audio
anti-spoofing benchmarks (ASVspoof, In-The-Wild, …).
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

    Samples with NaN or infinite scores are omitted from metric computation.

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

    finite = np.isfinite(scores_arr)
    labels_fit = labels_arr[finite]
    scores_fit = scores_arr[finite]

    # Threshold metrics only where score is finite (NaN/inf would break ROC/PR sklearn calls).
    if labels_fit.size == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "avg_precision": float("nan"),
            "eer": float("nan"),
        }

    preds = (scores_fit >= threshold).astype(int)

    # Use confusion_matrix with fixed label order to stay safe even if
    # one class is absent at this threshold.
    cm = confusion_matrix(labels_fit, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    results: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_fit, preds)),
        "precision": float(precision_score(labels_fit, preds, zero_division=0)),
        "recall": float(recall_score(labels_fit, preds, zero_division=0)),
        "f1": float(f1_score(labels_fit, preds, zero_division=0)),
        "tpr": tpr_val,
        "fpr": fpr_val,
    }

    if len(np.unique(labels_fit)) > 1:
        results["roc_auc"] = float(roc_auc_score(labels_fit, scores_fit))

        precision, recall, _ = precision_recall_curve(labels_fit, scores_fit)
        results["pr_auc"] = float(auc(recall, precision))

        results["avg_precision"] = float(average_precision_score(labels_fit, scores_fit))
        results["eer"] = _compute_eer(labels_fit, scores_fit)
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")
        results["avg_precision"] = float("nan")
        results["eer"] = float("nan")

    return results


def _compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate (FAR == FRR), linearly interpolated on the ROC curve.

    Convention: ``labels == 1`` is the positive class (AI / spoof) and
    higher ``scores`` correspond to that class — matching DetectZoo's
    "higher score ⇒ more likely AI" output convention.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the index where the gap (fpr - fnr) changes sign, then linearly
    # interpolate between the two surrounding points to get sub-bin precision.
    diff = fpr - fnr
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if sign_change.size == 0:
        # No crossover (e.g. degenerate scores): fall back to the closest point.
        idx = int(np.argmin(np.abs(diff)))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    i = int(sign_change[0])
    # Linear interpolation between (fpr[i], fnr[i]) and (fpr[i+1], fnr[i+1]).
    d0, d1 = diff[i], diff[i + 1]
    if d1 == d0:
        t = 0.0
    else:
        t = -d0 / (d1 - d0)
    fpr_eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
    fnr_eer = fnr[i] + t * (fnr[i + 1] - fnr[i])
    return float((fpr_eer + fnr_eer) / 2.0)
