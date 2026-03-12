"""Rank baseline detector.

Reference:
    Gehrmann et al., "GLTR: Statistical Detection and
    Visualization of Generated Text", ACL 2019.

Scores text by the average *raw* rank of each observed token in the
model's predicted distribution (without a log transform).  Machine-
generated text tends to pick top-ranked tokens, yielding a low average
rank.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("rank")
class RankDetector(BaseTextDetector):
    """Detect AI text via average raw token rank.

    Score = −mean(rank(x_i)).  Lower rank (model predicted well) →
    more negative mean → *higher* negated score → more likely AI.
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
    def _token_ranks(self, text: str) -> torch.Tensor:
        """Return 1-indexed ranks for each observed token (shape ``[T-1]``)."""
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        ranks = sorted_indices.argsort(dim=-1)
        token_ranks = ranks.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return (token_ranks.float() + 1).squeeze(0)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        ranks = self._token_ranks(text)
        avg_rank = float(ranks.mean())
        score = -avg_rank
        return self._make_result(score, avg_rank=avg_rank)
