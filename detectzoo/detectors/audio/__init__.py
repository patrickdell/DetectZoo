"""Audio-modality detectors for identifying synthetic / deepfake speech."""

from detectzoo.detectors.audio.aasist import AASISTDetector
from detectzoo.detectors.audio.ast_asvspoof.detector import ASTASVspoofDetector
from detectzoo.detectors.audio.melody_wav2vec.detector import MelodyWav2VecDetector
from detectzoo.detectors.audio.rawgat_st import RawGATSTDetector
from detectzoo.detectors.audio.rawnet2 import RawNet2Detector
from detectzoo.detectors.audio.restssdnet import ResTSSDNetDetector
from detectzoo.detectors.audio.samo import SAMODetector
from detectzoo.detectors.audio.whisper_mesonet.detector import WhisperMesoNetDetector
from detectzoo.detectors.audio.xlsr_sls.detector import XLSRSLSDetector

__all__ = [
    "AASISTDetector",
    "ASTASVspoofDetector",
    "MelodyWav2VecDetector",
    "RawGATSTDetector",
    "RawNet2Detector",
    "ResTSSDNetDetector",
    "SAMODetector",
    "WhisperMesoNetDetector",
    "XLSRSLSDetector",
]
