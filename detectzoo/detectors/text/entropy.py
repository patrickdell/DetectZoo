"""Entropy-based detector.

Measures the average predictive entropy of the model's next-token
distribution.  When the model is confident (low entropy) the text is
more likely to be machine-generated.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("entropy")
class EntropyDetector(BaseTextDetector):
    """Detect AI text via average next-token entropy.

    Score = −mean(H(p(· | x_{<i}))).  Lower entropy → higher score → more
    likely AI.
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
    def _token_entropies(self, text: str) -> torch.Tensor:
        logits, _ = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.squeeze(0)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        entropies = self._token_entropies(text)
        avg_entropy = float(entropies.mean())
        score = -avg_entropy
        return self._make_result(score, avg_entropy=avg_entropy)
