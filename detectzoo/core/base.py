"""Base classes for all DetectZoo detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Union

import torch


@dataclass
class DetectionResult:
    """Standardized output from any detector.

    Attributes:
        score: Detection score. Higher values indicate higher likelihood
            that the input is AI-generated.
        label: Binary label — ``"ai"`` or ``"human"``.
        confidence: Confidence in the assigned label (0–1).
        metadata: Arbitrary extra information from the detector.
    """

    score: float
    label: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(score={self.score:.4f}, label='{self.label}', "
            f"confidence={self.confidence:.4f})"
        )


class BaseDetector(ABC):
    """Abstract base class that every detector must inherit from.

    Subclasses must set ``name`` and ``modality`` and implement
    :meth:`predict`.
    """

    name: str = ""
    modality: str = ""  # "text", "image", or "audio"

    def __init__(self, threshold: float = 0.5, device: str = "cpu", **kwargs: Any) -> None:
        self.threshold = threshold
        self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, input_data: Any) -> DetectionResult:
        """Run detection on a single input and return a :class:`DetectionResult`."""

    def predict_batch(
        self, inputs: Union[Sequence[Any], List[Any]], **kwargs: Any
    ) -> list[DetectionResult]:
        """Run detection on a batch of inputs.

        The default implementation loops over inputs; subclasses may
        override for vectorised inference.
        """
        return [self.predict(inp, **kwargs) for inp in inputs]

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str) -> "BaseDetector":
        """Move detector (and its models) to *device*."""
        self._device = torch.device(device)
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                attr.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_result(self, score: float, **extra_metadata: Any) -> DetectionResult:
        """Build a :class:`DetectionResult` from a raw score."""
        label = "ai" if score >= self.threshold else "human"
        confidence = abs(score - self.threshold) / max(abs(score) + abs(self.threshold), 1e-8)
        confidence = min(confidence, 1.0)
        return DetectionResult(
            score=score,
            label=label,
            confidence=confidence,
            metadata=extra_metadata,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', modality='{self.modality}')"
