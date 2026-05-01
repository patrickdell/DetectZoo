"""AntiDeepfake (HuBERT-XLarge frontend) detector package.

NII Yamagishi-Lab post-trained HuBERT-XLarge + adaptive-avg-pool +
binary FC classifier, distributed at
``nii-yamagishilab/hubert-xlarge-anti-deepfake`` on HuggingFace Hub.
"""

from detectzoo.detectors.audio.anti_deepfake_hubert.detector import (
    AntiDeepfakeHuBERTDetector,
)

__all__ = ["AntiDeepfakeHuBERTDetector"]
