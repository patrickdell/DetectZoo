"""CNNDetection ResNet-50 inference.

Reference:
    Wang et al., "CNN-Generated Images Are Surprisingly Easy to Spot...For Now", CVPR 2020.

The key idea: CNN-based generators leave systematic artifacts in synthesized images
that differ from real photographs.  A ResNet trained for real vs. CNN-generated
binary classification learns those traces.

Upstream: https://github.com/PeterWang512/CNNDetection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.image.resnet50_binary import build_resnet50_binary, load_pytorch_checkpoint
from detectzoo.utils.io import load_image

_CKPT_URL = "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"
_CKPT_NAME = "blur_jpg_prob0.5.pth"


@register_detector("cnnspot", aliases=["cnn_spot"])
class CNNSpotDetector(BaseDetector):
    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            self._weight_path = get_cache_dir("cnnspot", cache_dir) / _CKPT_NAME
            download_file(_CKPT_URL, self._weight_path)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._model = build_resnet50_binary()
        self._model.load_state_dict(load_pytorch_checkpoint(self._weight_path, self._device)["model"])
        self._model.to(self._device).eval()

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode") and hasattr(input_data, "convert"):
            return input_data.convert("RGB")
        path = Path(str(input_data))
        if path.is_file():
            return load_image(path)
        raise TypeError(
            "Expected a PIL Image or a path to an image file; got "
            f"{type(input_data).__name__}."
        )

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        x = self._transform(self._normalize_input(input_data)).unsqueeze(0).to(self._device)
        score = self._model(x).sigmoid().item()
        return self._make_result(score)
