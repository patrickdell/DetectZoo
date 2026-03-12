"""Text-modality detectors for identifying LLM-generated text."""

from detectzoo.detectors.text.binoculars import BinocularsDetector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.detectors.text.dna_gpt import DNAGPTDetector
from detectzoo.detectors.text.entropy import EntropyDetector
from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector
from detectzoo.detectors.text.log_likelihood import LogLikelihoodDetector
from detectzoo.detectors.text.log_rank import LogRankDetector
from detectzoo.detectors.text.lrr import LRRDetector
from detectzoo.detectors.text.npr import NPRDetector

__all__ = [
    "BinocularsDetector",
    "DetectGPTDetector",
    "DNAGPTDetector",
    "EntropyDetector",
    "FastDetectGPTDetector",
    "LogLikelihoodDetector",
    "LogRankDetector",
    "LRRDetector",
    "NPRDetector",
]
