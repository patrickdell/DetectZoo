"""DetectZoo wrapper: XLSR-Mamba (Fairseq XLSR + Mamba classifier).

Reference:
    Xiao & Das, "XLSR-Mamba: A Dual-Column Bidirectional State Space Model for
    Spoofing Attack Detection", IEEE SPL / arXiv:2411.10027.
    https://arxiv.org/abs/2411.10027

Code & weights:
    https://github.com/swagshaw/XLSR-Mamba
    Hugging Face: ``AustinXiao/XLSR-Mamba-LA``, ``AustinXiao/XLSR-Mamba-DF``
    XLSR-300M backbone: https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt

**Optional dependencies** (install ``detectzoo[xlsr_mamba]``): ``fairseq``,
``mamba-ssm``, ``causal-conv1d``, ``safetensors``, ``huggingface_hub``, plus
``librosa``/``soundfile``/``torchaudio`` for I/O. ``fairseq`` often requires a
Linux/macOS-friendly environment; CUDA is recommended for ``mamba-ssm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

_SAMPLE_RATE = 16_000
# ASVspoof-style eval crop in upstream ``data_utils.Dataset_eval``
_CUT_SAMPLES = 66_800
_XLSR_URL = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
_XLSR_NAME = "xlsr2_300m.pt"

_HF_REPO = {
    "la": "AustinXiao/XLSR-Mamba-LA",
    "df": "AustinXiao/XLSR-Mamba-DF",
}


def _optional_imports() -> None:
    try:
        import fairseq  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "XLSR-Mamba requires `fairseq`. Install the XLSR-Mamba stack, e.g. "
            "`pip install detectzoo[xlsr_mamba]` and follow Fairseq install "
            "notes in the XLSR-Mamba README."
        ) from e
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "XLSR-Mamba requires `mamba-ssm` (and typically `causal-conv1d`). "
            "Install `pip install detectzoo[xlsr_mamba]` on a CUDA-capable setup."
        ) from e


def _pad_waveform(x: np.ndarray, max_len: int) -> np.ndarray:
    """Match ``utils.pad`` in the upstream repo."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> np.ndarray:
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.mean(dim=0)
        return wav.numpy().astype(np.float32)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        if sr != target_sr:
            import torchaudio

            t = torch.from_numpy(wav).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            wav = t.squeeze(0).numpy()
        return wav


@register_detector("xlsr_mamba", aliases=["xls_mamba"])
class XLSRMambaDetector(BaseDetector):
    """XLSR-Mamba spoofing countermeasure (logits = bonafide vs spoof).

    Parameters
    ----------
    variant
        ``\"la\"`` (ASVspoof2021 LA checkpoint) or ``\"df\"`` (DF checkpoint).
    checkpoint_path
        Optional path to ``model.safetensors``. When omitted, weights are
        downloaded from Hugging Face.
    xlsr_path
        Optional path to ``xlsr2_300m.pt``. When omitted, it is downloaded from
        Meta/Fairseq (large file).
    cut_samples
        Waveform length after padding/trim (default matches upstream LA/DF eval).
    threshold, device, cache_dir
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
    """

    modality = "audio"

    def __init__(
        self,
        variant: str = "la",
        checkpoint_path: Optional[Union[str, Path]] = None,
        xlsr_path: Optional[Union[str, Path]] = None,
        *,
        cut_samples: int = _CUT_SAMPLES,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        v = variant.strip().lower()
        if v not in _HF_REPO:
            raise ValueError(f"variant must be 'la' or 'df', got {variant!r}")
        self._variant = v
        self._cut = cut_samples

        _optional_imports()

        from huggingface_hub import hf_hub_download

        from detectzoo.detectors.audio.xlsr_mamba import core as xlsr_core

        root = get_cache_dir("xlsr_mamba", cache_dir)

        if xlsr_path is not None:
            xp = Path(xlsr_path).expanduser().resolve()
        else:
            xp = root / _XLSR_NAME
            if not xp.is_file():
                download_file(_XLSR_URL, xp)

        if checkpoint_path is not None:
            sp = Path(checkpoint_path).expanduser().resolve()
            if not sp.is_file():
                raise FileNotFoundError(sp)
        else:
            sp = Path(
                hf_hub_download(
                    repo_id=_HF_REPO[self._variant],
                    filename="model.safetensors",
                    cache_dir=str(root / "hf"),
                )
            )

        dev = torch.device(device)
        self._model = xlsr_core.build_model(dev, str(xp))
        xlsr_core.load_safetensors_weights(self._model, str(sp))
        self._model.to(dev)

    def _normalize_input(self, input_data: Any) -> np.ndarray:
        if isinstance(input_data, torch.Tensor):
            wav = input_data.detach().cpu().float().numpy()
            if wav.ndim == 2:
                wav = wav.mean(axis=0)
        elif isinstance(input_data, np.ndarray):
            wav = input_data.astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=0)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE)
        wav = _pad_waveform(wav, self._cut)
        return wav

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return spoof probability (treat as AI/deepfake score)."""
        wav = self._normalize_input(input_data)
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self._device)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)
        score_ai = float(probs[0, 1].item())

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 0].item()),
            score_spoof=float(probs[0, 1].item()),
            logit_bonafide=float(logits[0, 0].item()),
            logit_spoof=float(logits[0, 1].item()),
            variant=self._variant,
        )
