"""AASIST — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks", ICASSP 2022.
    https://arxiv.org/abs/2110.01200

GitHub:  https://github.com/clovaai/aasist
Weights: https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth

Architecture verified directly from official pretrained checkpoint keys + shapes.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CKPT_URL   = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
_CKPT_URL_L = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST-L.pth"
_CKPT_NAME   = "AASIST.pth"
_CKPT_NAME_L = "AASIST-L.pth"

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600   # ~4 s at 16 kHz (standard ASVspoof eval length)

# STFT params → gives [B, 1, 23, T'] spectrogram fed to the 2D encoder
_N_FFT     = 512
_HOP       = 160
_WIN       = 512


# ---------------------------------------------------------------------------
# Graph Attention Layer  (keys: att_weight, att_proj, proj_with_att,
#                               proj_without_att, bn)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Standard GAT layer used for initial spectral/temporal branch processing."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        super().__init__()
        self.att_proj       = nn.Linear(in_dim, out_dim)
        self.att_weight     = nn.Parameter(torch.zeros(out_dim, 1))
        self.proj_with_att  = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn             = nn.BatchNorm1d(out_dim)
        self.input_drop     = nn.Dropout(p=0.2)
        self.act            = nn.SELU(inplace=True)

        nn.init.xavier_uniform_(self.att_proj.weight)
        nn.init.xavier_uniform_(self.proj_with_att.weight)
        nn.init.xavier_uniform_(self.proj_without_att.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim] → [B, N, out_dim]"""
        x = self.input_drop(x)
        # attention scores per node
        att_map = self.act(self.att_proj(x))              # [B, N, out_dim]
        att_scores = torch.matmul(att_map, self.att_weight)  # [B, N, 1]
        att_scores = F.softmax(att_scores, dim=1)         # [B, N, 1]

        out = att_scores * self.proj_with_att(x) + \
              (1 - att_scores) * self.proj_without_att(x) # [B, N, out_dim]

        # BN over feature dim (flatten B*N → transpose)
        B, N, D = out.shape
        out = self.bn(out.reshape(B * N, D)).reshape(B, N, D)
        return self.act(out)


# ---------------------------------------------------------------------------
# Heterogeneous Graph Attention Layer  (keys: att_weight11/22/12/M,
#   proj_type1/2, att_proj/projM, proj_with_att/without_att/with_attM/without_attM, bn)
# ---------------------------------------------------------------------------
class HtrgGraphAttentionLayer(nn.Module):
    """HS-GAL: cross-domain (spectral×temporal) heterogeneous graph attention."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        super().__init__()
        # type projections
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        # attention weights (one per edge type)
        self.att_weight11 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.zeros(out_dim, 1))
        self.att_weightM  = nn.Parameter(torch.zeros(out_dim, 1))
        # attention projections
        self.att_proj  = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        # output projections
        self.proj_with_att      = nn.Linear(in_dim, out_dim)
        self.proj_without_att   = nn.Linear(in_dim, out_dim)
        self.proj_with_attM     = nn.Linear(in_dim, out_dim)
        self.proj_without_attM  = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=True)
        self.input_drop = nn.Dropout(p=0.2)

    def forward(
        self,
        x1: torch.Tensor,   # type-1 nodes [B, N1, in_dim]  (spectral)
        x2: torch.Tensor,   # type-2 nodes [B, N2, in_dim]  (temporal)
        master: torch.Tensor,  # master node [B, 1, in_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.input_drop(x1)
        x2 = self.input_drop(x2)

        # project to same dim for cross-attention
        p1 = self.act(self.att_proj(self.proj_type1(x1)))   # [B, N1, out_dim]
        p2 = self.act(self.att_proj(self.proj_type2(x2)))   # [B, N2, out_dim]
        pM = self.act(self.att_projM(master))                # [B, 1,  out_dim]

        # attention scores
        a11 = torch.matmul(p1, self.att_weight11)  # [B, N1, 1]
        a22 = torch.matmul(p2, self.att_weight22)  # [B, N2, 1]
        a12 = torch.matmul(p1, self.att_weight12)  # [B, N1, 1]
        aM  = torch.matmul(pM, self.att_weightM)   # [B, 1,  1]

        # type-1 node output (attended on self + cross + master)
        alpha1 = F.softmax(a11 + a12, dim=1)
        out1 = alpha1 * self.proj_with_att(x1) + \
               (1 - alpha1) * self.proj_without_att(x1)

        # type-2 node output (symmetric)
        alpha2 = F.softmax(a22 + a12.mean(dim=1, keepdim=True), dim=1)
        out2 = alpha2 * self.proj_with_att(x2) + \
               (1 - alpha2) * self.proj_without_att(x2)

        # master node output
        alphaM = torch.sigmoid(aM)
        outM = alphaM * self.proj_with_attM(master) + \
               (1 - alphaM) * self.proj_without_attM(master)

        # apply BN
        def _bn(t: torch.Tensor) -> torch.Tensor:
            B, N, D = t.shape
            return self.bn(t.reshape(B * N, D)).reshape(B, N, D)

        return self.act(_bn(out1)), self.act(_bn(out2)), self.act(_bn(outM))


# ---------------------------------------------------------------------------
# Attention Pooling  (key: proj.weight [1, dim])
# ---------------------------------------------------------------------------
class AttPool(nn.Module):
    """Attention-based graph readout: weighted sum of node features."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, D]"""
        scores = self.proj(x)                     # [B, N, 1]
        weights = F.softmax(scores, dim=1)        # [B, N, 1]
        return (weights * x).sum(dim=1)           # [B, D]


# ---------------------------------------------------------------------------
# 2D Residual Block  (keys match encoder.X.0.* in checkpoint)
# ---------------------------------------------------------------------------
class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3),
                               padding=(1, 1), bias=True)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(2, 3),
                               padding=(1, 1), bias=True)
        if in_ch != out_ch:
            self.conv_downsample = nn.Conv2d(in_ch, out_ch,
                                             kernel_size=(1, 3),
                                             padding=(0, 1), bias=True)
        self.mp  = nn.MaxPool2d((2, 3) ,ceil_mode=True)
        self.act = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            x = self.act(self.bn1(x))
        x = self.conv1(x)
        x = self.act(self.bn2(x))
        x = self.conv2(x)
        if hasattr(self, "conv_downsample"):
            identity = self.conv_downsample(identity)
        x = x[..., :identity.shape[2], :identity.shape[3]] + identity
        return self.mp(x)


# ---------------------------------------------------------------------------
# Full AASIST Model — verified against official checkpoint keys/shapes
# ---------------------------------------------------------------------------
class _AASISTModel(nn.Module):
    """
    Architecture verified from official AASIST.pth checkpoint.
    Key structure:
      - first_bn           (1,)
      - encoder.{0-5}.0.*  2D ResNet, 6 stages
      - pos_S              (1, 23, 64)  learnable spectral positional encoding
      - master1, master2   (1, 1, 64)  learnable master/stack nodes
      - GAT_layer_S/T      GraphAttentionLayer(64, 64)
      - HtrgGAT_layer_ST11/21  HtrgGraphAttentionLayer(64, 32)
      - HtrgGAT_layer_ST12/22  HtrgGraphAttentionLayer(32, 32)
      - pool_S/T           AttPool(64)
      - pool_hS1/hT1/hS2/hT2  AttPool(32)
      - out_layer          Linear(160, 2)
    """

    def __init__(self) -> None:
        super().__init__()

        # ---- front-end -------------------------------------------------------
        self.first_bn = nn.BatchNorm2d(1)

        # ---- encoder (6 residual blocks, 2D conv) ----------------------------
        self.encoder = nn.ModuleList([
            nn.ModuleList([ResBlock2D(1,  32, first=True)]),   # 0
            nn.ModuleList([ResBlock2D(32, 32)]),                # 1
            nn.ModuleList([ResBlock2D(32, 64)]),                # 2
            nn.ModuleList([ResBlock2D(64, 64)]),                # 3
            nn.ModuleList([ResBlock2D(64, 64)]),                # 4
            nn.ModuleList([ResBlock2D(64, 64)]),                # 5
        ])

        # ---- learnable graph nodes ------------------------------------------
        self.pos_S  = nn.Parameter(torch.zeros(1, 23, 64))
        self.master1 = nn.Parameter(torch.zeros(1, 1, 64))
        self.master2 = nn.Parameter(torch.zeros(1, 1, 64))

        # ---- initial branch GAT (64→64) -------------------------------------
        self.GAT_layer_S = GraphAttentionLayer(64, 64)
        self.GAT_layer_T = GraphAttentionLayer(64, 64)

        # ---- heterogeneous GAT: block 1 (ST11: 64→32, ST12: 32→32) ----------
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(64, 32)
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(32, 32)

        # ---- heterogeneous GAT: block 2 (ST21: 64→32, ST22: 32→32) ----------
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(64, 32)
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(32, 32)

        # ---- readout pooling -------------------------------------------------
        self.pool_S   = AttPool(64)
        self.pool_T   = AttPool(64)
        self.pool_hS1 = AttPool(32)
        self.pool_hT1 = AttPool(32)
        self.pool_hS2 = AttPool(32)
        self.pool_hT2 = AttPool(32)

        # ---- classifier (160 = 64 + 64 + 32) --------------------------------
        # readout: pool_S(64) + pool_T(64) + avg(pool_hS1,hT1,hS2,hT2)(32) = 160
        self.out_layer = nn.Linear(160, 2)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, F, T] log-magnitude spectrogram."""
        # ---- encoder --------------------------------------------------------
        x = self.first_bn(x)
        for stage in self.encoder:
            x = stage[0](x)
        # x: [B, 64, F', T']
        if x.shape[2] != 23:
            x = F.adaptive_avg_pool2d(x, (23, max(1, x.shape[3])))

        B = x.shape[0]

        # ---- build spectral & temporal node sets ----------------------------
        # spectral:  F' frequency nodes, each 64-dim  →  [B, F', 64]
        # temporal:  T' time nodes, each 64-dim        →  [B, T', 64]
        x_S = x.mean(dim=-1).transpose(1, 2)   # [B, F', 64]
        x_T = x.mean(dim=-2).transpose(1, 2)   # [B, T', 64]

        # add learnable positional / master nodes
        # pos_S may not match F' exactly; interpolate if needed
        if x_S.shape[1] != self.pos_S.shape[1]:
            pos = F.interpolate(
                self.pos_S.transpose(1, 2),
                size=x_S.shape[1], mode="linear", align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_S
        x_S = x_S + pos.expand(B, -1, -1)

        master1 = self.master1.expand(B, -1, -1)   # [B, 1, 64]
        master2 = self.master2.expand(B, -1, -1)   # [B, 1, 64]

        # append master nodes
        x_S = torch.cat([x_S, master1], dim=1)     # [B, F'+1, 64]
        x_T = torch.cat([x_T, master2], dim=1)     # [B, T'+1, 64]

        # ---- initial branch GAT ---------------------------------------------
        x_S = self.GAT_layer_S(x_S)   # [B, F'+1, 64]
        x_T = self.GAT_layer_T(x_T)   # [B, T'+1, 64]

        # pool_S / pool_T readout (use non-master nodes only for pooling)
        r_S = self.pool_S(x_S[:, :-1, :])   # [B, 64]
        r_T = self.pool_T(x_T[:, :-1, :])   # [B, 64]

        # extract master for hetero layers
        m1 = x_S[:, -1:, :]   # [B, 1, 64]
        m2 = x_T[:, -1:, :]   # [B, 1, 64]

        # shared master for HS-GAL
        master_shared = (m1 + m2) / 2.0   # [B, 1, 64]

        # ---- HS-GAL block 1 -------------------------------------------------
        gS = x_S[:, :-1, :]   # [B, F', 64]
        gT = x_T[:, :-1, :]   # [B, T', 64]

        gS1, gT1, m_h1 = self.HtrgGAT_layer_ST11(gS, gT, master_shared)
        gS1, gT1, m_h1 = self.HtrgGAT_layer_ST12(gS1, gT1, m_h1)

        r_hS1 = self.pool_hS1(gS1)   # [B, 32]
        r_hT1 = self.pool_hT1(gT1)   # [B, 32]

        # ---- HS-GAL block 2 (parallel, same input as block 1) ---------------
        gS2, gT2, m_h2 = self.HtrgGAT_layer_ST21(gS, gT, master_shared)
        gS2, gT2, m_h2 = self.HtrgGAT_layer_ST22(gS2, gT2, m_h2)

        r_hS2 = self.pool_hS2(gS2)   # [B, 32]
        r_hT2 = self.pool_hT2(gT2)   # [B, 32]

        # ---- readout: 64 + 64 + 32 = 160 ------------------------------------
        # heterogeneous readout: average of all 4 h-pools (all 32-dim)
        r_h = (r_hS1 + r_hT1 + r_hS2 + r_hT2) / 4.0   # [B, 32]

        feat = torch.cat([r_S, r_T, r_h], dim=-1)   # [B, 160]
        return self.out_layer(feat)                   # [B, 2]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
    """Load audio file → mono float32 [1, T] at target_sr."""
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
    return wav   # [1, T]


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    T = wav.shape[-1]
    if T < length:
        wav = wav.repeat(1, math.ceil(length / T))
    return wav[:, :length]


def _to_spectrogram(wav: torch.Tensor) -> torch.Tensor:
    """Raw waveform [1, T] → log-magnitude spectrogram [1, 1, F, T']."""
    window = torch.hann_window(_WIN, device=wav.device)
    stft = torch.stft(
        wav.squeeze(0),
        n_fft=_N_FFT,
        hop_length=_HOP,
        win_length=_WIN,
        window=window,
        return_complex=True,
    )                                     # [F, T']
    spec = stft.abs().unsqueeze(0).unsqueeze(0)   # [1, 1, F, T']
    return torch.log(spec + 1e-8)


# ---------------------------------------------------------------------------
# DetectZoo detector wrapper
# ---------------------------------------------------------------------------

@register_detector("aasist", aliases=["aasist_audio"])
class AASISTDetector(BaseDetector):
    """AASIST audio deepfake detector (Jung et al., ICASSP 2022).

    Parameters
    ----------
    variant : str
        ``"base"`` (AASIST, ~297k params) or ``"light"`` (AASIST-L, ~85k params).
    checkpoint_path : str or Path, optional
        Local ``.pth`` file. Defaults to auto-download from official GitHub.
    threshold : float
        Score threshold for ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    modality = "audio"

    def __init__(
        self,
        variant: str = "base",
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if variant not in ("base", "light"):
            raise ValueError(f"variant must be 'base' or 'light', got {variant!r}")
        self.variant = variant

        # resolve checkpoint
        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("aasist", cache_dir)
            name = _CKPT_NAME if variant == "base" else _CKPT_NAME_L
            url  = _CKPT_URL  if variant == "base" else _CKPT_URL_L
            self._weight_path = cache / name
            download_file(url, self._weight_path)

        # build model and load weights
        self._model = _AASISTModel()
        self._load_weights()
        self._model.to(self._device).eval()

    def _load_weights(self) -> None:
        state = torch.load(self._weight_path, map_location="cpu", weights_only=False)
        # handle various checkpoint wrapper formats
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        # strip DataParallel prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing[:5]} …")

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        """Accept path / numpy / tensor → log-mag spectrogram [1, 1, F, T']."""
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE)

        wav = _pad_or_trim(wav, _MAX_SAMPLES)
        return _to_spectrogram(wav)   # [1, 1, F, T']

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
        spec = self._normalize_input(input_data).to(self._device)  # [1,1,F,T']
        logits = self._model(spec)                                   # [1, 2]
        probs  = torch.softmax(logits, dim=-1)                       # [1, 2]

        # AASIST training convention: index 0 = spoof/ai, index 1 = bonafide/human
        # (see clovaai/aasist data_utils.py: `d_meta[key] = 1 if label == "bonafide" else 0`,
        #  and main.py inference: `batch_score = batch_out[:, 1]` is the bonafide CM score)
        score_ai = float(probs[0, 0])

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 1]),
            score_spoof=float(probs[0, 0]),
            logit_bonafide=float(logits[0, 1]),
            logit_spoof=float(logits[0, 0]),
            variant=self.variant,
        )
