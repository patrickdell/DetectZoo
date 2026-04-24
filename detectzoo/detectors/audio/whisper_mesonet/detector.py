"""DetectZoo wrapper: Whisper-MesoNet (Kawa et al., Interspeech 2023).

Reference
---------
Kawa, Plata, Syga, "Improved DeepFake Detection Using Whisper Features",
Interspeech 2023.  https://arxiv.org/abs/2306.01428

Code & weights
--------------
- Upstream: https://github.com/piotrkawa/deepfake-whisper-features
- Pretrained classifier checkpoints (trained on ASVspoof 2021 DF, reported on
  In-The-Wild) are distributed via the author's Google Drive folder:
  https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch
  The relevant file for this detector is ``whisper+mesonet/ckpt.pth`` (or the
  equivalent inside ``all_models/`` for the vanilla Whisper-MesoNet variant).

Key paper numbers
-----------------
    Whisper (tiny.en) MesoNet on DeepFakes In-The-Wild :  EER ≈ 0.33

The checkpoint contains the **full** ``WhisperMesoNet`` state dict — both the
Whisper tiny.en encoder and the MesoInception-4 classifier — so no extra
Whisper download is required.

Score convention
----------------
Upstream trains with ``BCEWithLogitsLoss`` and the label mapping
``1 == bonafide, 0 == spoof`` (see ``SimpleAudioFakeDataset`` in upstream).
Consequently a *high* sigmoid value indicates **bonafide**. This wrapper
therefore returns ``score_ai = 1 - sigmoid(logit)`` so that, consistent with
the rest of DetectZoo, higher score ⇒ more likely AI/deepfake.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.logger import get_logger

from detectzoo.detectors.audio.whisper_mesonet.meso_inception import WhisperMesoNet
from detectzoo.detectors.audio.whisper_mesonet.whisper_encoder import N_SAMPLES, SAMPLE_RATE

_LOGGER = get_logger(__name__)

_CKPT_NAME = "whisper_mesonet_ckpt.pth"
# Google-Drive folder that hosts the paper's pretrained checkpoints.
_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch"
)


# ---------------------------------------------------------------------------
# Audio I/O helpers (match upstream ``apply_preprocessing`` except for the
# optional sox silence-trim, which is skipped here because sox is not always
# available at inference time)
# ---------------------------------------------------------------------------

def _load_audio(path: Union[str, Path], target_sr: int = SAMPLE_RATE) -> torch.Tensor:
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
    # Upstream keeps only the first channel (``waveform[:1, ...]``) — we do the
    # same to match training-time preprocessing exactly.
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Tile-and-crop padding to ``length`` samples (matches upstream ``apply_pad``)."""
    wav = wav.squeeze(0)
    T = wav.shape[-1]
    if T >= length:
        return wav[:length]
    num_repeats = int(math.ceil(length / T))
    return wav.repeat(num_repeats)[:length]


def _try_gdown_folder(dest_dir: Path) -> Optional[Path]:
    """Best-effort download of the upstream GDrive folder via ``gdown``.

    Returns the path to the first ``ckpt.pth`` found under ``dest_dir``
    after the download, or ``None`` if gdown is unavailable or the download
    fails.
    """
    try:
        import gdown  # type: ignore
    except ImportError:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        _LOGGER.info("Attempting `gdown` folder download from %s", _GDRIVE_FOLDER)
        gdown.download_folder(
            url=_GDRIVE_FOLDER,
            output=str(dest_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        _LOGGER.warning("gdown folder download failed: %s", e)
        return None
    # Prefer the pure whisper+mesonet variant over the fine-tuned w/MFCC one.
    candidates = sorted(dest_dir.rglob("ckpt.pth"))
    for c in candidates:
        if "whisper" in str(c).lower() and "meso" in str(c).lower():
            return c
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

@register_detector("whisper_mesonet", aliases=["whisper-mesonet", "whispermesonet"])
class WhisperMesoNetDetector(BaseDetector):
    """Whisper-MesoNet audio deepfake detector.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a ``ckpt.pth`` saved by the upstream ``train_models.py``
        (a full ``WhisperMesoNet`` ``state_dict``). When omitted the detector
        tries, in order:

        1. The cached copy under ``<cache_dir>/whisper_mesonet/ckpt.pth``.
        2. An automatic download of the upstream Google Drive folder via
           ``gdown`` (install with ``pip install detectzoo[whisper_mesonet]``).

        If both fall through, a :class:`FileNotFoundError` is raised with a
        message pointing at the upstream Google Drive folder.
    fc1_dim : int
        Bottleneck size of MesoInception-4 (``fc1``). Default ``1024`` matches
        the published config.
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("whisper_mesonet",
    ...                     checkpoint_path="weights/whisper_mesonet.pth")
    >>> res = det.predict("sample.wav")
    >>> print(res.label, res.score)
    """

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        fc1_dim: int = 1024,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        root = get_cache_dir("whisper_mesonet", cache_dir)
        self._weight_path = self._resolve_weights(checkpoint_path, root)

        self._model = WhisperMesoNet(
            freeze_encoder=True,
            input_channels=1,
            fc1_dim=fc1_dim,
            num_classes=1,
        )
        self._load_weights()
        self._model.to(self._device).eval()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _resolve_weights(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        cache: Path,
    ) -> Path:
        if checkpoint_path is not None:
            p = Path(checkpoint_path).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"checkpoint_path does not exist: {p}")
            return p

        cached = cache / _CKPT_NAME
        if cached.is_file():
            return cached

        # Try gdown auto-download of the upstream folder.
        downloaded = _try_gdown_folder(cache / "upstream_gdrive")
        if downloaded is not None and downloaded.is_file():
            return downloaded

        raise FileNotFoundError(
            "Whisper-MesoNet pretrained weights were not found. "
            "The paper's checkpoints are hosted only on Google Drive:\n  "
            f"{_GDRIVE_FOLDER}\n"
            "Download `whisper+mesonet/ckpt.pth` (or any `ckpt.pth` inside "
            "`all_models/`) and either:\n"
            f"  - place it at {cached}, or\n"
            "  - pass it via `WhisperMesoNetDetector(checkpoint_path=...)`,\n"
            "  - or install `pip install detectzoo[whisper_mesonet]` so gdown "
            "can auto-download the folder."
        )

    def _load_weights(self) -> None:
        state = torch.load(self._weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        state = {k.replace("module.", ""): v for k, v in state.items()}
        result = self._model.load_state_dict(state, strict=False)
        # The upstream ckpt does not store ``positional_embedding`` buffers in
        # some saves; they are deterministic sinusoids regenerated at __init__
        # time and are safe to ignore. Any *other* missing / unexpected keys
        # indicate a structural mismatch that WILL degrade EER.
        ignored_missing = {
            k for k in result.missing_keys
            if k.endswith("positional_embedding")
        }
        missing = [k for k in result.missing_keys if k not in ignored_missing]
        if missing or result.unexpected_keys:
            _LOGGER.warning(
                "Whisper-MesoNet checkpoint key mismatch — EER may be degraded.\n"
                "  missing   : %s\n  unexpected: %s",
                missing,
                result.unexpected_keys,
            )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav[:1]
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav[:1]
        else:
            wav = _load_audio(input_data, SAMPLE_RATE)
        return _pad_or_trim(wav, N_SAMPLES)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return spoof probability (treat as AI/deepfake score).

        Parameters
        ----------
        input_data
            Audio file path, 1-D/2-D numpy array, or torch tensor
            (16 kHz mono; arbitrary length — resampled and padded/trimmed
            to 30 s internally).
        """
        wav = self._normalize_input(input_data).to(self._device)
        x = wav.view(1, -1)  # (batch=1, 480_000)
        logit = self._model(x).view(-1)
        p_bonafide = float(torch.sigmoid(logit[0]).item())
        p_spoof = 1.0 - p_bonafide

        return self._make_result(
            p_spoof,
            score_bonafide=p_bonafide,
            score_spoof=p_spoof,
            logit=float(logit[0].item()),
        )
