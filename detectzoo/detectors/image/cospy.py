"""CO-SPY — Combining Semantic and Pixel Features (CVPR 2025).

Reference:
    Cheng et al., "CO-SPY: Combining Semantic and Pixel Features to Detect
    Synthetic Images by AI", CVPR 2025.
    https://arxiv.org/abs/2503.18286

The key idea: combines two complementary detection signals: (1) *semantic*
features from a frozen SigLIP encoder that capture high-level inconsistencies,
and (2) *artifact* features from a VAE reconstruction-error pipeline that expose
pixel-level generator traces.  A calibrated linear fusion layer merges both
branch predictions into a single detection score.

Upstream: https://github.com/Megum1/CO-SPY
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.io import load_image

_HF_REPO = "ruojiruoli/Co-Spy-Pretrained-Weights"
_HF_ZIP = "sd-v1_4.zip"


def _conv3x3(in_p: int, out_p: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_p, out_p, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_p: int, out_p: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_p, out_p, 1, stride=stride, bias=False)


class _ArtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, planes: int, stride: int = 1, ds: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1, self.bn1 = _conv1x1(inp, planes), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = _conv3x3(planes, planes, stride), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = _conv1x1(planes, planes * 4), nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _VAEReconEncoder(nn.Module):
    """Artifact branch: VAE reconstruction residual → truncated ResNet-50."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(_ArtBottleneck, 64, 3)
        self.layer2 = self._make_layer(_ArtBottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ds = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds = nn.Sequential(_conv1x1(self.inplanes, planes * block.expansion, stride),
                               nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(x).latent_dist.mean
        return self.vae.decode(latent).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = (x - self._reconstruct(x)) / 7.0 * 100.0
        out = self.maxpool(self.relu(self.bn1(self.conv1(residual))))
        out = self.layer2(self.layer1(out))
        return self.avgpool(out).flatten(1)


class _CoSpyFusion(nn.Module):
    """Full CO-SPY fusion detector (semantic + artifact + calibrated fc)."""

    def __init__(self) -> None:
        super().__init__()
        import open_clip
        from diffusers import StableDiffusionPipeline

        # --- Semantic branch: SigLIP ViT-SO400M-14-384 ---
        sig, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
        sig.requires_grad_(False)
        self.clip = sig
        self.sem_fc = nn.Linear(1152, 1)

        # --- Artifact branch: SD-v1.4 VAE recon → ResNet ---
        vae = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").vae
        vae.requires_grad_(False)
        self.artifact_encoder = _VAEReconEncoder(vae)
        self.art_fc = nn.Linear(512, 1)

        # --- Fusion ---
        self.fusion_fc = nn.Linear(2, 1)

        # --- Transforms (applied inside forward per-branch) ---
        self.sem_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.art_resize = transforms.Resize(224, antialias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sem = self.sem_norm(x)
        x_art = self.art_resize(x)  # artifact branch gets no normalization

        sem_pred = self.sem_fc(self.clip.encode_image(x_sem))
        art_pred = self.art_fc(self.artifact_encoder(x_art))
        return self.fusion_fc(torch.cat([sem_pred, art_pred], dim=1))


def _load_cospy_weights(model: _CoSpyFusion, cache: Path) -> None:
    """Load the three-part CO-SPY pretrained weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import zipfile, tempfile

    zip_path = Path(hf_hub_download(repo_id=_HF_REPO, filename=_HF_ZIP, repo_type="model"))

    extract_dir = cache / "sd-v1_4"
    if not (extract_dir / "fusion_weights.pth").is_file():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir.parent)

    sem_path = extract_dir / "semantic_weights.pth"
    art_path = extract_dir / "artifact_weights.pth"
    fus_path = extract_dir / "fusion_weights.pth"

    if sem_path.is_file():
        w = torch.load(sem_path, map_location="cpu", weights_only=False)
        model.sem_fc.weight.data = w["fc.weight"]
        model.sem_fc.bias.data = w["fc.bias"]

    if art_path.is_file():
        art_w = torch.load(art_path, map_location="cpu", weights_only=False)
        art_enc_state = {k: v for k, v in art_w.items() if not k.startswith("fc.")}
        fc_state = {k[3:]: v for k, v in art_w.items() if k.startswith("fc.")}
        if art_enc_state:
            model.artifact_encoder.load_state_dict(art_enc_state, strict=False)
        if fc_state:
            model.art_fc.load_state_dict(fc_state, strict=True)

    if fus_path.is_file():
        fus_w = torch.load(fus_path, map_location="cpu", weights_only=False)
        fc_w = {k.replace("fc.", ""): v for k, v in fus_w.items() if k.startswith("fc.")}
        if fc_w:
            model.fusion_fc.load_state_dict(fc_w, strict=True)


@register_detector("cospy", aliases=["co_spy", "cospy_cvpr2025"])
class CoSpyDetector(BaseDetector):
    """CO-SPY fusion detector (Cheng et al., CVPR 2025).

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Directory containing ``semantic_weights.pth``, ``artifact_weights.pth``,
        and ``fusion_weights.pth``.  Auto-downloaded from HuggingFace when omitted.
    threshold : float
        Decision boundary (default 0.5).
    device : str
        Torch device string.
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``).
    """

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self._model = _CoSpyFusion()

        cache = get_cache_dir("cospy", cache_dir)
        if checkpoint_dir is not None:
            cdir = Path(checkpoint_dir).expanduser().resolve()
            sem = torch.load(cdir / "semantic_weights.pth", map_location="cpu", weights_only=False)
            self._model.sem_fc.weight.data = sem["fc.weight"]
            self._model.sem_fc.bias.data = sem["fc.bias"]
            art = torch.load(cdir / "artifact_weights.pth", map_location="cpu", weights_only=False)
            art_enc = {k: v for k, v in art.items() if not k.startswith("fc.")}
            art_fc = {k[3:]: v for k, v in art.items() if k.startswith("fc.")}
            if art_enc:
                self._model.artifact_encoder.load_state_dict(art_enc, strict=False)
            if art_fc:
                self._model.art_fc.load_state_dict(art_fc, strict=True)
            fus = torch.load(cdir / "fusion_weights.pth", map_location="cpu", weights_only=False)
            fus_fc = {k.replace("fc.", ""): v for k, v in fus.items() if k.startswith("fc.")}
            if fus_fc:
                self._model.fusion_fc.load_state_dict(fus_fc, strict=True)
        else:
            _load_cospy_weights(self._model, cache)

        self._model.to(self._device).eval()

        self._transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
        ])  

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------    

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        img = self._normalize_input(input_data)
        x = self._transform(img).unsqueeze(0).to(self._device)
        score = self._model(x).sigmoid().item()
        return self._make_result(float(score))
