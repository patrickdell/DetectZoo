"""LRR — Log-Likelihood Ratio detector.

Reference:
    Su et al., "DetectLLM: Leveraging Log Rank Information for
    Zero-Shot Detection of Machine-Generated Text", EMNLP 2023.

Combines two zero-shot signals — average log-likelihood and average
log-rank — into a single score:

    LRR(x) = −LL(x) / LogRank(x)

where ``LL`` is the mean token log-probability and ``LogRank`` is the
mean log(rank) of each token.  The ratio amplifies the detection
signal compared to either metric alone.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("lrr")
class LRRDetector(BaseTextDetector):
    """Log-Likelihood Ratio detector.

    Score = −LL / LogRank.  Higher score → more likely AI.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        threshold: Decision boundary.
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
    def _token_log_ranks(self, text: str) -> torch.Tensor:
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        ranks = sorted_indices.argsort(dim=-1)
        token_ranks = ranks.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return torch.log(token_ranks.float() + 1).squeeze(0)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        ll = self._mean_log_prob(text)
        log_rank = float(self._token_log_ranks(text).mean())

        if abs(log_rank) < 1e-8:
            score = 0.0
        else:
            score = -ll / log_rank

        return self._make_result(score, log_likelihood=ll, log_rank=log_rank)
