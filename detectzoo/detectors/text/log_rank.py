"""Log-Rank baseline detector.

Reference:
    Gehrmann et al., "GLTR: Statistical Detection and
    Visualization of Generated Text", ACL 2019.

Instead of raw log-probabilities this detector uses the *rank* of each
observed token in the model's predicted distribution.  Machine text tends
to pick high-probability (low-rank) tokens, resulting in lower average
log-rank.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("log_rank")
class LogRankDetector(BaseTextDetector):
    """Detect AI text via average log-rank of observed tokens.

    Score = −mean(log(rank(x_i))).  Higher score → more likely AI.
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
        # +1 so rank starts at 1 (avoids log(0))
        log_ranks = torch.log(token_ranks.float() + 1)
        return log_ranks.squeeze(0)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        log_ranks = self._token_log_ranks(text)
        avg_log_rank = float(log_ranks.mean())
        score = -avg_log_rank
        return self._make_result(score, avg_log_rank=avg_log_rank)
