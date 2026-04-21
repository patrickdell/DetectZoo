"""Image-modality detectors for identifying AI-generated images."""

from detectzoo.detectors.image.aeroblade import AerobladeDetector
from detectzoo.detectors.image.aide import AIDEDetector
from detectzoo.detectors.image.cnnspot import CNNSpotDetector
from detectzoo.detectors.image.cospy import CoSpyDetector
from detectzoo.detectors.image.d3 import D3Detector
from detectzoo.detectors.image.fatformer import FatFormerDetector
from detectzoo.detectors.image.lgrad import LGradDetector
from detectzoo.detectors.image.npr_deepfake import NPRDeepfakeDetector
from detectzoo.detectors.image.patchcraft import PatchCraftDetector
from detectzoo.detectors.image.safe import SAFEDetector
from detectzoo.detectors.image.univfd import UnivFDDetector
from detectzoo.detectors.image.c2p_clip import C2PCLIPDetector

__all__ = [
    "AerobladeDetector",
    "AIDEDetector",
    "CNNSpotDetector",
    "CoSpyDetector",
    "D3Detector",
    "FatFormerDetector",
    "LGradDetector",
    "NPRDeepfakeDetector",
    "PatchCraftDetector",
    "SAFEDetector",
    "UnivFDDetector",
    "C2PCLIPDetector",
]
