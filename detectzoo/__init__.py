"""DetectZoo: A unified toolkit for detecting AI-generated content."""

from detectzoo.utils.hf_quiet import configure_hf_quiet

configure_hf_quiet()

import detectzoo.datasets.audio  # noqa: F401
import detectzoo.datasets.image  # noqa: F401
import detectzoo.datasets.text  # noqa: F401
import detectzoo.detectors.audio  # noqa: F401
import detectzoo.detectors.image  # noqa: F401
import detectzoo.detectors.text  # noqa: F401
from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import list_datasets, list_detectors, load_dataset, load_detector

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "load_dataset",
    "load_detector",
    "list_datasets",
    "list_detectors",
]
