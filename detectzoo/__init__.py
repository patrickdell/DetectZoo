"""DetectZoo: A unified toolkit for detecting AI-generated content."""

import detectzoo.detectors.audio  # noqa: F401
import detectzoo.detectors.image  # noqa: F401
import detectzoo.detectors.text  # noqa: F401
from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import list_detectors, load_detector

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "load_detector",
    "list_detectors",
]
