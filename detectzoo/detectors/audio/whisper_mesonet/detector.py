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
  The relevant file for this detector is ``all_models/whisper_mesonet/weights.pth``
  (file id ``19LPAA1-nFkxlR6FztoWBaA_8vGwgXPiG``). The auto-download path
  pulls *only* that single 31 MB file rather than the whole ~1 GB folder.

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

_CKPT_NAME = "whisper_mesonet_weights.pth"
# Google-Drive folder that hosts the paper's pretrained checkpoints.
_GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch"
)
# Single-file id of `all_models/whisper_mesonet/weights.pth` inside that folder.
# Captured from the upstream GDrive listing — pulling just this one file avoids
# the ~1 GB / rate-limit-prone full-folder download.
_GDRIVE_FILE_ID = "19LPAA1-nFkxlR6FztoWBaA_8vGwgXPiG"


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


def _find_legacy_cached_weights(cache_root: Path) -> Optional[Path]:
    """Return a previously-downloaded plain ``whisper_mesonet/weights.pth``
    under ``cache_root`` (e.g. left over from an older DetectZoo full-folder
    download).

    Only the *exact* folder name ``whisper_mesonet`` is matched — sibling
    variants like ``whisper_mesonet_finetuned``, ``whisper_lfcc_mesonet``,
    ``whisper_mfcc_mesonet`` or ``whisper+mesonet`` use a different
    ``input_channels`` and would fail to load.
    """
    for hit in cache_root.rglob("weights.pth"):
        if hit.parent.name == "whisper_mesonet":
            return hit
    return None


def _try_gdown_single_file(dest: Path) -> Optional[Path]:
    """Best-effort download of the single upstream `weights.pth` via ``gdown``.

    Returns ``dest`` on success, ``None`` if ``gdown`` is unavailable or the
    download fails. Uses the file-id form so it never has to enumerate the
    surrounding ~1 GB folder.
    """
    try:
        import gdown  # type: ignore
    except ImportError:
        _LOGGER.warning(
            "`gdown` is not installed; cannot auto-download Whisper-MesoNet "
            "weights. Install with `pip install detectzoo[whisper_mesonet]`."
        )
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _LOGGER.info(
            "Downloading Whisper-MesoNet weights from Google Drive "
            "(file id %s) to %s",
            _GDRIVE_FILE_ID,
            dest,
        )
        gdown.download(
            id=_GDRIVE_FILE_ID,
            output=str(dest),
            quiet=False,
        )
    except Exception as e:
        _LOGGER.warning("gdown single-file download failed: %s", e)
        return None
    return dest if dest.is_file() else None


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

@register_detector("whisper_mesonet", aliases=["whisper-mesonet", "whispermesonet"])
class WhisperMesoNetDetector(BaseDetector):
    """Whisper-MesoNet audio deepfake detector.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a ``weights.pth`` saved by the upstream ``train_models.py``
        (a full ``WhisperMesoNet`` ``state_dict``). When omitted the detector
        tries, in order:

        1. The cached copy under ``<cache_dir>/whisper_mesonet_weights.pth``.
        2. Any pre-existing ``whisper_mesonet/weights.pth`` left under
           ``<cache_dir>`` by an older DetectZoo full-folder download.
        3. A single-file ``gdown`` pull of the upstream
           ``all_models/whisper_mesonet/weights.pth`` (install with
           ``pip install detectzoo[whisper_mesonet]``).

        If all three fall through, a :class:`FileNotFoundError` is raised
        with a message pointing at the upstream Google Drive folder.
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

        # Reuse any pre-existing weights left over from older DetectZoo
        # versions (which downloaded the whole upstream folder).
        legacy = _find_legacy_cached_weights(cache)
        if legacy is not None and legacy.is_file():
            _LOGGER.info("Reusing pre-existing Whisper-MesoNet weights at %s", legacy)
            return legacy

        # Single-file gdown of the published `weights.pth`.
        downloaded = _try_gdown_single_file(cached)
        if downloaded is not None and downloaded.is_file():
            return downloaded

        raise FileNotFoundError(
            "Whisper-MesoNet pretrained weights were not found. "
            "The paper's checkpoints are hosted only on Google Drive:\n  "
            f"{_GDRIVE_FOLDER}\n"
            "Download `all_models/whisper_mesonet/weights.pth` and either:\n"
            f"  - place it at {cached}, or\n"
            "  - pass it via `WhisperMesoNetDetector(checkpoint_path=...)`,\n"
            "  - or install `pip install detectzoo[whisper_mesonet]` so gdown "
            "can auto-download the file (id "
            f"{_GDRIVE_FILE_ID})."
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
