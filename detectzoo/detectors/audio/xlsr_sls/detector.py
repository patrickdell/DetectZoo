"""DetectZoo wrapper: XLSR-SLS — XLS-R 300M frontend + Self-supervised
Linear-classifier (attentive-stat-pool) backend for spoof detection.

Frontend
--------
``facebook/wav2vec2-xls-r-300m`` loaded with :class:`Wav2Vec2Model` (NOT
``Wav2Vec2ForCTC``). XLS-R 300M is a 24-layer transformer pre-trained on
~436k hours of multilingual speech (128 languages) — its self-supervised
representations transfer significantly better across recording conditions
and TTS systems than spectrogram or 2019-LA-only encoders.

    Babu et al., "XLS-R: Self-supervised Cross-lingual Speech
    Representation Learning at Scale", Interspeech 2022.
    https://arxiv.org/abs/2111.09296

Backend
-------
A lightweight head consisting of:

    1. **Attentive statistics pooling** (Okabe et al., 2018) over the last
       hidden state of the XLS-R encoder: a single-head attention computes
       a soft alignment over time, yielding a weighted mean and standard
       deviation of the frame embeddings.
    2. **Linear classifier** mapping the (2 * hidden_dim,) pooled vector
       to 2 logits (bonafide vs spoof).

Both components are intentionally simple — the discriminative power
comes almost entirely from the pre-trained XLS-R features.

Score convention
----------------
Higher score => more likely AI / spoof, matching the rest of DetectZoo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Default XLS-R frontend (pure transformers, no fairseq).
_DEFAULT_FRONTEND_NAME = "facebook/wav2vec2-xls-r-300m"

# XLS-R 300M / XLS-R always operates at 16 kHz.
_SAMPLE_RATE = 16_000

# Fixed 4-second window (64 000 samples @ 16 kHz). This matches the
# windowing used by most published wav2vec2-based anti-spoofing systems
# (Tak et al., 2022; "Automatic Speaker Verification Spoofing and
# Deepfake Detection Using Wav2Vec 2.0 and Data Augmentation").
_INPUT_LENGTH = 64_000

# TODO: upload the trained XLSR-SLS classifier-head weights to HuggingFace
# Hub and replace this placeholder URL with the resolved blob URL, e.g.
#     https://huggingface.co/<org>/xlsr-sls-asvspoof2019/resolve/main/weights.pth
# Until then the auto-download will fail and the detector will raise a
# clear FileNotFoundError pointing the user at `checkpoint_path=`.
_PLACEHOLDER_WEIGHTS_URL = (
    "https://huggingface.co/PLACEHOLDER/xlsr-sls-asvspoof2019/resolve/main/weights.pth"
)
_CKPT_NAME = "xlsr_sls_weights.pth"


# ---------------------------------------------------------------------------
# Audio I/O helpers — mirror the conventions used by the other audio
# detectors in DetectZoo (ast_asvspoof, melody_wav2vec, ...).
# ---------------------------------------------------------------------------

def _load_audio_to_numpy(
    path: Union[str, Path], target_sr: int = _SAMPLE_RATE
) -> np.ndarray:
    """Load an audio file -> mono float32 numpy array at ``target_sr``."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = data.astype(np.float32)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1, keepdims=True)
        wav = wav[:, 0]
        if sr != target_sr:
            import torchaudio

            wav_t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, target_sr
            )
            wav = wav_t.squeeze(0).numpy().astype(np.float32)
        return wav


def _pad_or_trim_numpy(wav: np.ndarray, length: int) -> np.ndarray:
    """Tile-and-crop padding to ``length`` samples (matches the upstream
    convention for short utterances in ASVspoof-style training pipelines)."""
    T = int(wav.shape[-1])
    if T >= length:
        return wav[:length].astype(np.float32, copy=False)
    num_repeats = int(math.ceil(length / max(T, 1)))
    return np.tile(wav, num_repeats)[:length].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Backend: Attentive Statistics Pooling + Linear classifier
# ---------------------------------------------------------------------------

class AttentiveStatsPooling(nn.Module):
    """Single-head attentive statistics pooling (Okabe et al., 2018).

    Maps a sequence of frame embeddings ``x`` of shape ``(B, T, D)`` to a
    pooled vector of shape ``(B, 2 * D)`` by computing a soft attention
    weighting over time and returning the weighted mean concatenated with
    the weighted standard deviation.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) with 1 == valid, 0 == padding.
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)

        mean = (weights * x).sum(dim=1)  # (B, D)
        var = (weights * (x - mean.unsqueeze(1)) ** 2).sum(dim=1).clamp(min=1e-9)
        std = var.sqrt()
        return torch.cat([mean, std], dim=-1)  # (B, 2D)


class XLSRSLSModel(nn.Module):
    """XLS-R frontend + attentive-stat-pool + linear binary classifier."""

    def __init__(
        self,
        frontend_name: str = _DEFAULT_FRONTEND_NAME,
        cache_dir: Optional[str] = None,
        num_classes: int = 2,
        attention_dim: int = 128,
        freeze_frontend: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "XLSRSLSDetector requires `transformers`. Install with:\n"
                "  pip install detectzoo[xlsr_sls]\n"
                "or:\n"
                "  pip install 'transformers>=4.30' torchaudio soundfile"
            ) from exc

        self.frontend = Wav2Vec2Model.from_pretrained(
            frontend_name, cache_dir=cache_dir
        )
        if freeze_frontend:
            for p in self.frontend.parameters():
                p.requires_grad = False

        hidden_dim = int(self.frontend.config.hidden_size)
        self.pool = AttentiveStatsPooling(hidden_dim, attention_dim=attention_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.frontend(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state  # (B, T, D)

        pool_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # The frontend may downsample the time axis; align by simple
            # ratio interpolation rather than depending on a private
            # `_get_feat_extract_output_lengths` API.
            t_in = int(attention_mask.shape[-1])
            t_out = int(hidden.shape[1])
            if t_in == t_out:
                pool_mask = attention_mask
            else:
                # Take every k-th element of the input attention mask.
                idx = torch.linspace(
                    0, t_in - 1, steps=t_out, device=attention_mask.device
                ).long()
                pool_mask = attention_mask[:, idx]

        pooled = self.pool(hidden, mask=pool_mask)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

@register_detector(
    "xlsr_sls",
    aliases=["xlsr-sls", "xlsr", "xls-r-sls", "xlsr_sls_aasist"],
)
class XLSRSLSDetector(BaseDetector):
    """XLS-R 300M frontend + attentive-stat-pool linear-classifier head.

    Parameters
    ----------
    frontend_name : str, optional
        HuggingFace Hub model id of the wav2vec2/XLS-R frontend. Defaults
        to ``"facebook/wav2vec2-xls-r-300m"``. Other compatible frontends
        (e.g. ``"facebook/wav2vec2-xls-r-1b"``) also work but require
        weights trained against a matching hidden size.
    checkpoint_path : str or Path, optional
        Path to a ``state_dict`` containing the *full* model weights
        (XLS-R encoder + pooling + classifier). When omitted, the
        detector tries:

        1. Cached ``<cache_dir>/xlsr_sls_weights.pth``.
        2. A best-effort download from a placeholder HF URL (currently
           a TODO — see the module-level comment); failure is expected
           until the trained weights are uploaded.

        If both paths fall through, a :class:`FileNotFoundError` is
        raised with a clear message instructing the user to either
        provide ``checkpoint_path`` or wait for the weights to be
        published on HuggingFace.
    attention_dim : int
        Hidden size of the single-head attention used inside the
        attentive statistics pooling layer. Default 128.
    freeze_frontend : bool
        If ``True``, the XLS-R encoder weights are frozen at inference
        time. This has no effect on the forward pass output but reduces
        ``state_dict`` size when the user re-saves the model. Default
        ``False`` (just keeps the loaded checkpoint intact).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        ``cache_dir`` is also forwarded to ``transformers.from_pretrained``
        so the XLS-R download lands inside DetectZoo's cache tree
        (``<cache_dir>/xlsr_sls/...``).

    Notes
    -----
    Inputs are resampled to 16 kHz mono and pad/tile-trimmed to a fixed
    4-second window (64 000 samples) — matching the convention used by
    most wav2vec2-based anti-spoofing systems in the literature. The
    feature extractor is intentionally hand-rolled (zero-mean, unit-
    variance with eps-clipped denominator) rather than going through
    ``Wav2Vec2FeatureExtractor`` so this detector has no soft dependency
    on the HF ``processor`` for inference.

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("xlsr_sls",
    ...                     checkpoint_path="weights/xlsr_sls.pth")
    >>> res = det.predict("sample.wav")
    >>> print(res.label, res.score)
    """

    modality = "audio"

    def __init__(
        self,
        frontend_name: str = _DEFAULT_FRONTEND_NAME,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        attention_dim: int = 128,
        freeze_frontend: bool = False,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.frontend_name = frontend_name
        self._cache_root = get_cache_dir("xlsr_sls", cache_dir)

        _LOGGER.info(
            "Loading XLSR-SLS frontend from HuggingFace Hub: %s (cache=%s)",
            frontend_name,
            self._cache_root,
        )
        self._model = XLSRSLSModel(
            frontend_name=frontend_name,
            cache_dir=str(self._cache_root),
            num_classes=2,
            attention_dim=attention_dim,
            freeze_frontend=freeze_frontend,
        )

        weight_path = self._resolve_weights(checkpoint_path, self._cache_root)
        self._load_weights(weight_path)

        self._model.to(self._device).eval()
        self._sample_rate = _SAMPLE_RATE

        # Conventional ASVspoof-style class indices. The trained weights
        # are expected to follow {0: bonafide, 1: spoof}; if a future
        # checkpoint flips this, plumb a `spoof_label_index` kwarg the
        # same way the AST/Melody wrappers do.
        self._spoof_idx = 1
        self._bonafide_idx = 0

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _resolve_weights(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        cache: Path,
    ) -> Optional[Path]:
        """Locate the XLSR-SLS classifier-head ``state_dict``.

        Returns ``None`` only when ``checkpoint_path`` is omitted *and*
        no usable cached / downloadable weights are found — in which
        case :meth:`_load_weights` raises a helpful FileNotFoundError.
        """
        if checkpoint_path is not None:
            p = Path(checkpoint_path).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"checkpoint_path does not exist: {p}")
            return p

        cached = cache / _CKPT_NAME
        if cached.is_file():
            _LOGGER.info("Reusing cached XLSR-SLS weights at %s", cached)
            return cached

        # Best-effort download from the placeholder URL. This is expected
        # to fail until the trained checkpoint is published — we swallow
        # the error and fall through to the FileNotFoundError below so
        # the user gets a clean, actionable message rather than a stack
        # trace from urllib.
        try:
            _LOGGER.info(
                "Attempting to download XLSR-SLS weights from %s",
                _PLACEHOLDER_WEIGHTS_URL,
            )
            downloaded = download_file(_PLACEHOLDER_WEIGHTS_URL, cached)
            if downloaded.is_file() and downloaded.stat().st_size > 0:
                return downloaded
        except Exception as e:
            _LOGGER.warning(
                "XLSR-SLS auto-download failed (expected — placeholder URL): %s",
                e,
            )
            # Remove any zero-byte stub left behind by a half-finished
            # urlretrieve so the next launch retries cleanly.
            try:
                if cached.exists() and cached.stat().st_size == 0:
                    cached.unlink()
            except OSError:
                pass

        return None

    def _load_weights(self, weight_path: Optional[Path]) -> None:
        if weight_path is None:
            raise FileNotFoundError(
                "XLSR-SLS pretrained weights were not found.\n"
                "The trained classifier-head + frontend `state_dict` has not\n"
                "yet been uploaded to HuggingFace Hub. To use this detector:\n"
                f"  - place the weights at {self._cache_root / _CKPT_NAME}, or\n"
                "  - pass them via `XLSRSLSDetector(checkpoint_path=...)`,\n"
                "  - or wait until the upstream HF upload is published; the\n"
                "    placeholder URL inside the wrapper is\n"
                f"      {_PLACEHOLDER_WEIGHTS_URL}\n"
                "    (TODO: replace once the canonical model id is available)."
            )

        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        state = {k.replace("module.", ""): v for k, v in state.items()}
        result = self._model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            _LOGGER.warning(
                "XLSR-SLS checkpoint key mismatch — EER may be degraded.\n"
                "  missing   : %s\n  unexpected: %s",
                result.missing_keys,
                result.unexpected_keys,
            )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _normalize_input(self, input_data: Any) -> np.ndarray:
        """Accept path / numpy / tensor → mono float32 numpy at 16 kHz,
        padded / trimmed to ``_INPUT_LENGTH`` samples."""
        if isinstance(input_data, np.ndarray):
            wav = input_data.astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)
        elif isinstance(input_data, torch.Tensor):
            t = input_data.detach().to(torch.float32).cpu()
            if t.dim() == 2:
                t = t.mean(dim=0) if t.shape[0] < t.shape[1] else t.mean(dim=1)
            wav = t.numpy().astype(np.float32)
        else:
            wav = _load_audio_to_numpy(input_data, _SAMPLE_RATE)
        return _pad_or_trim_numpy(wav, _INPUT_LENGTH)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI probability for a single audio input.

        Parameters
        ----------
        input_data
            Audio file path, 1-D / 2-D numpy array, or torch tensor.
            Multi-channel inputs are downmixed to mono and resampled to
            16 kHz; arbitrary lengths are padded or trimmed to a fixed
            4-second window (64 000 samples).
        """
        wav = self._normalize_input(input_data)

        # Hand-rolled zero-mean / unit-variance normalisation matching
        # the default `Wav2Vec2FeatureExtractor(do_normalize=True)`
        # behaviour — keeps this detector free of a hard dependency on
        # the HF processor at inference time.
        m = float(wav.mean())
        s = float(wav.std())
        wav_norm = (wav - m) / max(s, 1e-7)

        x = torch.from_numpy(wav_norm).float().unsqueeze(0).to(self._device)
        logits = self._model(x).view(-1)
        probs = torch.softmax(logits, dim=-1)
        score_ai = float(probs[self._spoof_idx].item())
        score_human = float(probs[self._bonafide_idx].item())

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_human,
            logit_spoof=float(logits[self._spoof_idx].item()),
            logit_bonafide=float(logits[self._bonafide_idx].item()),
            frontend_name=self.frontend_name,
        )
