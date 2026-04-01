"""Text-modality detectors for identifying LLM-generated text."""

from detectzoo.detectors.text.binoculars import BinocularsDetector
from detectzoo.detectors.text.coco import CoCoDetector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.detectors.text.dna_gpt import DNAGPTDetector
from detectzoo.detectors.text.entropy import EntropyDetector
from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector
from detectzoo.detectors.text.imbd import ImBDDetector
from detectzoo.detectors.text.lastde import LastdeDetector, LastdePPDetector
from detectzoo.detectors.text.log_likelihood import LogLikelihoodDetector
from detectzoo.detectors.text.log_rank import LogRankDetector
from detectzoo.detectors.text.lrr import LRRDetector
from detectzoo.detectors.text.npr import NPRDetector
from detectzoo.detectors.text.radar import RADARDetector
from detectzoo.detectors.text.rank import RankDetector
from detectzoo.detectors.text.revise_detect import ReviseDetector

__all__ = [
    "BinocularsDetector",
    "CoCoDetector",
    "DetectGPTDetector",
    "DNAGPTDetector",
    "EntropyDetector",
    "FastDetectGPTDetector",
    "ImBDDetector",
    "LastdeDetector",
    "LastdePPDetector",
    "LogLikelihoodDetector",
    "LogRankDetector",
    "LRRDetector",
    "NPRDetector",
    "RADARDetector",
    "RankDetector",
    "ReviseDetector",
]
