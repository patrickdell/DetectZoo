"""BiScope — bidirectional cross-entropy zero-shot detector.

Reference:
    Guo et al., "BiScope: AI-generated Text Detection by Checking
    Memorization of Preceding Tokens", NeurIPS 2024.

BiScope exploits two kinds of information in a causal LM's logits:

1. **Forward CE (FCE):** Standard next-token cross-entropy — how well
   the model predicts the *next* token from position i's logits.
2. **Backward CE (BCE):** How well the model's logits at position i
   encode the token at position i *itself* (memorisation signal).

Human text → poor next-token prediction, strong memorisation.
Machine text → good prediction, weak memorisation.

This implementation computes per-token FCE and BCE and derives
statistical features (mean, max, min, std) over multiple suffix
ranges, returning the mean FCE − mean BCE as the primary score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("biscope")
class BiScopeDetector(BaseTextDetector):
    """BiScope bidirectional cross-entropy detector.

    Parameters:
        model_name: HuggingFace causal LM for detection
            (default ``"gpt2"``).
        n_segments: Number of segments for multi-point feature
            extraction (default ``10``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        n_segments: int = 10,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold,
                         device=device, **kwargs)
        self.n_segments = n_segments

    @torch.no_grad()
    def _compute_biscope_losses(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-token FCE and BCE loss arrays."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        logits = self.model(**enc).logits  # [1, T, V]
        ids = enc["input_ids"]             # [1, T]
        T = ids.shape[1]

        if T < 3:
            return np.array([0.0]), np.array([0.0])

        targets = ids.squeeze(0)  # [T]
        log_probs = F.log_softmax(logits.squeeze(0), dim=-1)  # [T, V]

        # FCE: logits at position i-1, target at position i (standard next-token)
        fce = -log_probs[:-1].gather(1, targets[1:].unsqueeze(1)).squeeze(1)
        # BCE: logits at position i, target at position i (memorisation)
        bce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Align: skip position 0 for BCE to match FCE length
        bce = bce[1:]

        return fce.cpu().numpy(), bce.cpu().numpy()

    def _extract_features(self, fce: np.ndarray, bce: np.ndarray) -> np.ndarray:
        """Extract multi-point statistical features from loss arrays."""
        features: list[float] = []
        n = len(fce)
        for p in range(1, self.n_segments):
            split = n * p // self.n_segments
            fce_suffix = fce[split:]
            bce_suffix = bce[split:]
            if len(fce_suffix) == 0:
                features.extend([0.0] * 8)
                continue
            features.extend([
                float(np.mean(fce_suffix)), float(np.max(fce_suffix)),
                float(np.min(fce_suffix)), float(np.std(fce_suffix)),
                float(np.mean(bce_suffix)), float(np.max(bce_suffix)),
                float(np.min(bce_suffix)), float(np.std(bce_suffix)),
            ])
        return np.array(features)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        fce, bce = self._compute_biscope_losses(text)

        if len(fce) < 2:
            return self._make_result(0.0, reason="text too short")

        features = self._extract_features(fce, bce)

        # Primary zero-shot score: difference in mean FCE and mean BCE
        # Machine text: low FCE (good prediction), high BCE (weak memorisation)
        # → FCE - BCE is negative for machine, positive for human
        # Negate so higher → more likely AI
        mean_fce = float(np.mean(fce))
        mean_bce = float(np.mean(bce))
        score = -(mean_fce - mean_bce)

        return self._make_result(
            score,
            mean_fce=mean_fce,
            mean_bce=mean_bce,
            feature_dim=len(features),
        )
