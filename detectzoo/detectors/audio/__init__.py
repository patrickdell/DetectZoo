"""Audio-modality detectors for identifying synthetic / deepfake speech."""

from detectzoo.detectors.audio.aasist import AASISTDetector
from detectzoo.detectors.audio.rawnet2 import RawNet2Detector

__all__ = [
    "AASISTDetector",
    "RawNet2Detector",
]
