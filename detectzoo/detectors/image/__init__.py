"""Image-modality detectors for identifying AI-generated images."""

from detectzoo.detectors.image.aeroblade import AerobladeDetector
from detectzoo.detectors.image.cnnspot import CNNSpotDetector
from detectzoo.detectors.image.npr_deepfake import NPRDeepfakeDetector

__all__ = [
    "AerobladeDetector",
    "CNNSpotDetector",
    "NPRDeepfakeDetector",
]
