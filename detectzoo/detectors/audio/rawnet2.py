"""RawNet2 — End-to-end anti-spoofing from raw waveforms.

Reference:
    Tak et al., "End-to-end anti-spoofing with RawNet2", ICASSP 2021.
    https://arxiv.org/abs/2011.01108

GitHub:  https://github.com/eurecom-asp/rawnet2-antispoofing
         https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2
Weights: https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip

Key idea:
    RawNet2 operates directly on raw waveforms via a fixed sinc filter
    front-end, followed by 6 residual blocks with filter-wise feature map
    scaling (FMS) attention, a GRU layer, and a linear classifier.
    No hand-crafted features needed.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

# ---------------------------------------------------------------------------
# Constants — dims verified from official pretrained checkpoint
# ---------------------------------------------------------------------------
_CKPT_URL  = "https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip"
_CKPT_NAME = "pre_trained_DF_RawNet2.pth"
_CKPT_ZIP  = "pre_trained_DF_RawNet2.zip"

_SAMPLE_RATE     = 16_000
_MAX_SAMPLES     = 64_600

_NB_SINC_FILTERS = 20
_SINC_FILTER_LEN = 1024
_NB_FILTS        = [20, 20, 20, 128, 128, 128, 128]
_GRU_NODE        = 1024
_NB_FC_NODE      = 1024
_NB_CLASSES      = 2


# ---------------------------------------------------------------------------
# Residual block  (FMS is top-level per official asvspoof-challenge/2021 code)
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1],
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1],
                               kernel_size=3, padding=1, bias=False)
        self.mp    = nn.MaxPool1d(3)
        self.selu  = nn.SELU(inplace=True)
        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv1d(
                nb_filts[0], nb_filts[1], kernel_size=1, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            x = self.selu(self.bn1(x))
        x = self.conv1(x)
        x = self.selu(self.bn2(x))
        x = self.conv2(x)
        if hasattr(self, "conv_downsample"):
            identity = self.conv_downsample(identity)
        x = x + identity
        return self.mp(x)


# ---------------------------------------------------------------------------
# Full RawNet2 model
# ---------------------------------------------------------------------------
class _RawNet2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # front-end (fixed conv, matches checkpoint key "first_conv")
        self.first_conv = nn.Conv1d(
            1, _NB_SINC_FILTERS,
            kernel_size=_SINC_FILTER_LEN,
            stride=1,
            padding=_SINC_FILTER_LEN // 2,
            bias=False,
        )
        self.first_bn = nn.BatchNorm1d(_NB_SINC_FILTERS)
        self.selu     = nn.SELU(inplace=True)
        self.sig      = nn.Sigmoid()

        # residual blocks
        self.block0 = nn.Sequential(_ResBlock([_NB_FILTS[0], _NB_FILTS[1]], first=True))
        self.block1 = nn.Sequential(_ResBlock([_NB_FILTS[1], _NB_FILTS[2]]))
        self.block2 = nn.Sequential(_ResBlock([_NB_FILTS[2], _NB_FILTS[3]]))
        self.block3 = nn.Sequential(_ResBlock([_NB_FILTS[3], _NB_FILTS[4]]))
        self.block4 = nn.Sequential(_ResBlock([_NB_FILTS[4], _NB_FILTS[5]]))
        self.block5 = nn.Sequential(_ResBlock([_NB_FILTS[5], _NB_FILTS[6]]))

        # FMS attention — top-level, one per block (matches checkpoint keys)
        self.avgpool       = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = nn.Linear(_NB_FILTS[1], _NB_FILTS[1])
        self.fc_attention1 = nn.Linear(_NB_FILTS[2], _NB_FILTS[2])
        self.fc_attention2 = nn.Linear(_NB_FILTS[3], _NB_FILTS[3])
        self.fc_attention3 = nn.Linear(_NB_FILTS[4], _NB_FILTS[4])
        self.fc_attention4 = nn.Linear(_NB_FILTS[5], _NB_FILTS[5])
        self.fc_attention5 = nn.Linear(_NB_FILTS[6], _NB_FILTS[6])

        # back-end
        self.bn_before_gru = nn.BatchNorm1d(_NB_FILTS[-1])
        self.gru = nn.GRU(
            input_size=_NB_FILTS[-1],
            hidden_size=_GRU_NODE,
            num_layers=1,
            batch_first=True,
        )
        self.fc1_gru = nn.Linear(_GRU_NODE, _NB_FC_NODE)
        self.fc2_gru = nn.Linear(_NB_FC_NODE, _NB_CLASSES)

    def _fms(self, x: torch.Tensor, fc: nn.Linear) -> torch.Tensor:
        """Filter-wise feature Map Scaling (FMS) attention."""
        y = self.avgpool(x).view(x.size(0), -1)
        y = self.sig(fc(y)).view(x.size(0), -1, 1)
        return x * y + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] raw waveform at 16 kHz."""
        B, T = x.shape
        x = x.view(B, 1, T)

        x = self.first_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.selu(self.first_bn(x))

        x = self._fms(self.block0(x), self.fc_attention0)
        x = self._fms(self.block1(x), self.fc_attention1)
        x = self._fms(self.block2(x), self.fc_attention2)
        x = self._fms(self.block3(x), self.fc_attention3)
        x = self._fms(self.block4(x), self.fc_attention4)
        x = self._fms(self.block5(x), self.fc_attention5)

        x = self.selu(self.bn_before_gru(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        return self.fc2_gru(x)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(data.T.astype(np.float32))
        if sr != target_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    T = wav.shape[-1]
    if T < length:
        wav = wav.repeat(1, math.ceil(length / T))
    return wav[:, :length]


# ---------------------------------------------------------------------------
# DetectZoo detector wrapper
# ---------------------------------------------------------------------------

@register_detector("rawnet2", aliases=["rawnet2_audio"])
class RawNet2Detector(BaseDetector):
    """RawNet2 audio deepfake detector (Tak et al., ICASSP 2021).

    Detects AI-generated (spoofed) speech directly from raw waveforms using
    a fixed sinc filter front-end, 6 residual blocks with filter-wise feature
    map scaling (FMS) attention, and a GRU back-end classifier.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local ``.pth`` file. When omitted the official pretrained weights
        are downloaded and cached automatically.
    threshold : float
        Score threshold for ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("rawnet2")
    >>> result = det.predict("path/to/audio.wav")
    >>> print(result.label, result.score)
    """

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("rawnet2", cache_dir)
            self._weight_path = cache / _CKPT_NAME
            if not self._weight_path.exists():
                zip_path = cache / _CKPT_ZIP
                download_file(_CKPT_URL, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    pth_names = [n for n in zf.namelist() if n.endswith(".pth")]
                    if not pth_names:
                        raise RuntimeError("No .pth found inside RawNet2 zip")
                    with zf.open(pth_names[0]) as f:
                        self._weight_path.write_bytes(f.read())

        self._model = _RawNet2Model()
        self._load_weights()
        self._model.to(self._device).eval()

    def _load_weights(self) -> None:
        state = torch.load(
            self._weight_path, map_location="cpu", weights_only=False
        )
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE).squeeze(0)
        wav = _pad_or_trim(wav.unsqueeze(0), _MAX_SAMPLES).squeeze(0)
        return wav

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Predict whether audio is AI-generated (spoofed).

        Parameters
        ----------
        input_data
            Audio file path, numpy array, or torch tensor (16 kHz mono).

        Returns
        -------
        DetectionResult
            score=P(ai), label='ai'/'human', confidence in [0,1].
        """
        wav    = self._normalize_input(input_data).unsqueeze(0).to(self._device)
        logits = self._model(wav)
        probs  = torch.softmax(logits, dim=-1)

        score_ai = float(probs[0, 1])

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 0]),
            score_spoof=float(probs[0, 1]),
            logit_bonafide=float(logits[0, 0]),
            logit_spoof=float(logits[0, 1]),
        )
