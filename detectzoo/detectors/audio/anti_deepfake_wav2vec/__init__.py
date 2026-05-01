"""AntiDeepfake (Wav2Vec2-Large frontend) detector package.

NII Yamagishi-Lab post-trained Wav2Vec2-Large + adaptive-avg-pool +
binary FC classifier, distributed at
``nii-yamagishilab/wav2vec-large-anti-deepfake`` on HuggingFace Hub.
"""

from detectzoo.detectors.audio.anti_deepfake_wav2vec.detector import (
    AntiDeepfakeWav2VecDetector,
)

__all__ = ["AntiDeepfakeWav2VecDetector"]
