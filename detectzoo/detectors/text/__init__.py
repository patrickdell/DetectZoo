"""Text-modality detectors for identifying LLM-generated text."""

from detectzoo.detectors.text.binoculars import BinocularsDetector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.detectors.text.entropy import EntropyDetector
from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector
from detectzoo.detectors.text.log_likelihood import LogLikelihoodDetector
from detectzoo.detectors.text.log_rank import LogRankDetector

__all__ = [
    "BinocularsDetector",
    "DetectGPTDetector",
    "EntropyDetector",
    "FastDetectGPTDetector",
    "LogLikelihoodDetector",
    "LogRankDetector",
]
