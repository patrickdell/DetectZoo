"""AntiDeepfake (XLS-R-2B frontend) detector package.

NII Yamagishi-Lab post-trained Wav2Vec2 XLS-R-2B + adaptive-avg-pool +
binary FC classifier, distributed at
``nii-yamagishilab/xls-r-2b-anti-deepfake`` on HuggingFace Hub.

.. warning::

    ~2 B parameters; ~8.65 GB checkpoint. See
    :class:`AntiDeepfakeXLSR2BDetector` for memory requirements.
"""

from detectzoo.detectors.audio.anti_deepfake_xlsr2b.detector import (
    AntiDeepfakeXLSR2BDetector,
)

__all__ = ["AntiDeepfakeXLSR2BDetector"]
