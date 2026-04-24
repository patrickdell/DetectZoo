"""Vendored Whisper AudioEncoder + log-mel utilities.

This file is a minimal, inference-only port of the relevant pieces of

    https://github.com/piotrkawa/deepfake-whisper-features/
        blob/main/src/models/whisper_main.py

(which is itself derived from
 https://github.com/openai/whisper/blob/main/whisper/model.py).

Only the audio encoder side is kept — the text decoder is not required for
Whisper-MesoNet's feature-extraction pipeline. Module / parameter names are
preserved *exactly* so that pretrained checkpoints trained against the
upstream class (``Whisper(dims).encoder``) load cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Audio hyper-parameters (tiny.en — fixed by the paper / upstream code)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480_000 samples per 30 s chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH       # 3000 STFT frames

# tiny.en encoder dimensions (from openai-whisper's released tiny.en.pt)
TINY_EN_N_MELS = 80
TINY_EN_N_AUDIO_CTX = 1500
TINY_EN_N_AUDIO_STATE = 384
TINY_EN_N_AUDIO_HEAD = 6
TINY_EN_N_AUDIO_LAYER = 4


# ---------------------------------------------------------------------------
# Dimension dataclass (kept so ckpts that pickle a ModelDimensions still load)
# ---------------------------------------------------------------------------
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int = 0
    n_text_ctx: int = 0
    n_text_state: int = 0
    n_text_head: int = 0
    n_text_layer: int = 0


def tiny_en_dims() -> ModelDimensions:
    """Return the tiny.en encoder-only dimensions used by the paper."""
    return ModelDimensions(
        n_mels=TINY_EN_N_MELS,
        n_audio_ctx=TINY_EN_N_AUDIO_CTX,
        n_audio_state=TINY_EN_N_AUDIO_STATE,
        n_audio_head=TINY_EN_N_AUDIO_HEAD,
        n_audio_layer=TINY_EN_N_AUDIO_LAYER,
    )


# ---------------------------------------------------------------------------
# Log-Mel spectrogram (matches upstream exactly, minus the .npz asset file)
# ---------------------------------------------------------------------------
_MEL_FB_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _mel_filters(device: torch.device, n_mels: int = N_MELS) -> torch.Tensor:
    """Mel filterbank matrix identical to ``librosa.filters.mel(sr=16000,
    n_fft=400, n_mels=80)`` — which is what openai-whisper ships as
    ``assets/mel_filters.npz``. Computed on demand via ``librosa`` or
    fallen back to a pure-NumPy re-implementation so no binary asset is
    required.
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    key = (device, n_mels)
    if key in _MEL_FB_CACHE:
        return _MEL_FB_CACHE[key]
    try:
        import librosa  # type: ignore

        fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
    except Exception:
        fb = _mel_filterbank_numpy(SAMPLE_RATE, N_FFT, n_mels)
    fb_t = torch.from_numpy(np.asarray(fb, dtype=np.float32)).to(device)
    _MEL_FB_CACHE[key] = fb_t
    return fb_t


def _mel_filterbank_numpy(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Pure-NumPy Slaney-style mel filterbank (matches librosa default)."""
    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmin, fmax = 0.0, sr / 2.0
    mmin, mmax = hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax))
    m_pts = np.linspace(mmin, mmax, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, c):
            fb[i, k] = (k - l) / max(1, (c - l))
        for k in range(c, r):
            fb[i, k] = (r - k) / max(1, (r - c))
        # Slaney-style peak normalization
        enorm = 2.0 / max(1e-8, f_pts[i + 2] - f_pts[i])
        fb[i] *= enorm
    return fb


def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = N_MELS) -> torch.Tensor:
    """Compute a log-Mel spectrogram identical to upstream / openai-whisper.

    Expects a 1-D tensor of 16 kHz samples (already padded/trimmed to 30 s).
    """
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = _mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


# ---------------------------------------------------------------------------
# Whisper AudioEncoder (module/parameter names preserved for state_dict load)
# ---------------------------------------------------------------------------
class _LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class _Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class _Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def _sinusoids(length: int, channels: int, max_timescale: float = 10_000.0) -> Tensor:
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = _Linear(n_state, n_state)
        self.key = _Linear(n_state, n_state, bias=False)
        self.value = _Linear(n_state, n_state)
        self.out = _Linear(n_state, n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.out(self._qkv(q, k, v, mask))

    def _qkv(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = _LayerNorm(n_state)
        # Kept ``cross_attn`` attributes absent — encoder blocks only.
        self.cross_attn = None
        self.cross_attn_ln = None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            _Linear(n_state, n_mlp), nn.GELU(), _Linear(n_mlp, n_state)
        )
        self.mlp_ln = _LayerNorm(n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
    ) -> None:
        super().__init__()
        self.conv1 = _Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = _Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", _sinusoids(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = _LayerNorm(n_state)

    def forward(self, x: Tensor) -> Tensor:
        """x : ``(batch, n_mels, n_ctx)`` log-Mel spectrogram."""
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class Whisper(nn.Module):
    """Encoder-only Whisper wrapper, matching upstream attribute layout.

    The upstream ``Whisper`` class holds both encoder and decoder, but the
    deepfake-whisper-features training only exercises ``self.encoder`` so we
    can safely drop the decoder here. The ``encoder.*`` state-dict keys still
    match the published checkpoints.
    """

    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
        )

    def forward(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)
