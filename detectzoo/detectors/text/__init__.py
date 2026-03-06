"""Text-modality detectors for identifying LLM-generated text."""

from detectzoo.detectors.text.binoculars import BinocularsDetector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector

__all__ = [
    "BinocularsDetector",
    "DetectGPTDetector",
    "FastDetectGPTDetector",
]
