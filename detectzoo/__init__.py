"""DetectZoo: A unified toolkit for detecting AI-generated content."""

import importlib
import warnings

from detectzoo.utils.hf_quiet import configure_hf_quiet

configure_hf_quiet()


def _try_import(module: str) -> None:
    """Import a modality subpackage, warning (not failing) if optional deps are missing.

    This keeps pure-audio or pure-text workflows usable even when optional
    image/text/audio extras are not installed.
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        warnings.warn(
            f"detectzoo: skipped loading '{module}' ({exc}). "
            "Install the corresponding optional extra to enable it "
            "(e.g. `pip install detectzoo[audio]`).",
            stacklevel=2,
        )


for _mod in (
    "detectzoo.datasets.audio",
    "detectzoo.datasets.image",
    "detectzoo.datasets.text",
    "detectzoo.detectors.audio",
    "detectzoo.detectors.image",
    "detectzoo.detectors.text",
):
    _try_import(_mod)

from detectzoo.core.base import BaseDetector, DetectionResult  # noqa: E402
from detectzoo.core.registry import (  # noqa: E402
    list_datasets,
    list_detectors,
    load_dataset,
    load_detector,
)

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "load_dataset",
    "load_detector",
    "list_datasets",
    "list_detectors",
]
