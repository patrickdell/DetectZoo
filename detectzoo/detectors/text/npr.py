"""NPR — Normalized Perturbation Rank detector.

Reference:
    Su et al., "DetectLLM: Leveraging Log Rank Information for
    Zero-Shot Detection of Machine-Generated Text", EMNLP 2023.

Like DetectGPT, NPR generates perturbations of the input text, but
instead of comparing log-probabilities it compares *log-ranks*.

    NPR(x) = mean(LogRank(x̃)) / LogRank(x)

where x̃ are perturbations of x produced by a mask-filling model.
For machine text the original has much lower log-rank than its
perturbations, so the ratio is > 1.

The masking/fill pipeline is identical to DetectGPT's so we inherit it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("npr")
class NPRDetector(DetectGPTDetector):
    """Normalized Perturbation Rank detector.

    Inherits the T5 perturbation pipeline from :class:`DetectGPTDetector`
    and replaces the log-prob-based scoring with a log-rank ratio, as
    specified in DetectLLM.

    Parameters:
        model_name: Scoring model (causal LM, default ``"EleutherAI/gpt-neo-2.7B"``).
        perturbation_model: Mask-filling model (default ``"t5-3b"``).
        n_perturbations: Number of perturbations to generate.
        pct_words_masked: Fraction of words targeted for masking.
        span_length: Words per masked span (default 2).
        buffer_size: Minimum gap between spans (default 1).
        mask_top_p: top-p for the T5 sampler (default 1.0).
        threshold: Decision threshold (default 1.0 — neutral ratio).
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-2.7B",
        perturbation_model: str = "t5-3b",
        n_perturbations: int = 10,
        pct_words_masked: float = 0.3,
        span_length: int = 2,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        threshold: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            perturbation_model=perturbation_model,
            n_perturbations=n_perturbations,
            pct_words_masked=pct_words_masked,
            span_length=span_length,
            buffer_size=buffer_size,
            mask_top_p=mask_top_p,
            threshold=threshold,
            device=device,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Log-rank computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mean_log_rank(self, text: str) -> float | None:
        if not text.strip():
            return None
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        if shift_labels.numel() == 0:
            return None
        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        ranks = sorted_indices.argsort(dim=-1)
        token_ranks = ranks.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # +1 so rank is 1-indexed
        log_ranks = torch.log(token_ranks.float() + 1)
        return float(log_ranks.mean())

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        original_lr = self._mean_log_rank(text)
        if original_lr is None or abs(original_lr) < 1e-8:
            return self._make_result(0.0, reason="degenerate input")

        perturbed_lrs: list[float] = []
        for _ in range(self.n_perturbations):
            perturbed = self._perturb_once(text)
            if perturbed and perturbed.strip():
                lr = self._mean_log_rank(perturbed)
                if lr is not None:
                    perturbed_lrs.append(lr)

        if len(perturbed_lrs) < 2:
            return self._make_result(0.0, reason="insufficient perturbations")

        mean_pert_lr = float(np.mean(perturbed_lrs))
        score = mean_pert_lr / original_lr

        return self._make_result(
            score,
            original_log_rank=original_lr,
            mean_perturbed_log_rank=mean_pert_lr,
            n_perturbations_used=len(perturbed_lrs),
        )
