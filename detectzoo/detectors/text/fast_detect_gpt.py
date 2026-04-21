"""Fast-DetectGPT — perturbation-free probability-curvature detector.

Reference:
    Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection
    of Machine-Generated Text via Conditional Probability Curvature",
    ICLR 2024.

Instead of generating explicit perturbations (expensive), Fast-DetectGPT
estimates the curvature of the log-probability surface by comparing
observed token log-probs against the *expected* log-prob under the
model's own conditional distribution (which equals the negative entropy).

    score_i = log p(x_i | x_{<i}) + H(p(· | x_{<i}))

Averaged over positions and normalised by standard deviation, this
gives a z-score that separates human from machine text.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("fast_detectgpt")
class FastDetectGPTDetector(BaseTextDetector):
    """Fast-DetectGPT detector (no perturbation model needed).

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        threshold: Decision threshold on the curvature score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)

    @torch.no_grad()
    def _curvature_score(self, text: str) -> tuple[float, dict]:
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]   # [1, T-1, V]
        shift_labels = ids[:, 1:]          # [1, T-1]

        if shift_labels.ndim == shift_logits.ndim - 1:
            shift_labels = shift_labels.unsqueeze(-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)

        # log p(x_i | x_{<i})  — observed log-likelihood per position
        log_likelihood = log_probs.gather(dim=-1, index=shift_labels).squeeze(-1)  # [1, T-1]

        # E_p[log p] per position (= negative entropy)
        mean_ref = (probs * log_probs).sum(dim=-1)  # [1, T-1]

        # Var_p[log p] per position
        var_ref = (probs * log_probs.square()).sum(dim=-1) - mean_ref.square()  # [1, T-1]

        # Analytic sampling discrepancy: (sum(LL) - sum(mean)) / sqrt(sum(var))
        ll_sum = log_likelihood.sum(dim=-1)       # [1]
        mean_sum = mean_ref.sum(dim=-1)            # [1]
        std_sum = var_ref.sum(dim=-1).clamp(min=1e-10).sqrt()  # [1]

        score = float((ll_sum - mean_sum) / std_sum)

        return score, {
            "mean_log_prob": float(log_likelihood.mean()),
            "mean_entropy": float(-mean_ref.mean()),
            "sum_variance": float(var_ref.sum()),
        }

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        score, meta = self._curvature_score(text)
        return self._make_result(score, **meta)
