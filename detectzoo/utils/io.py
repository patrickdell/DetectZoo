"""I/O helpers for loading text, images, and audio."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def load_text(source: Union[str, Path]) -> str:
    """Load text from a file path or return the string directly.

    If *source* is an existing file, its contents are read; otherwise
    *source* is treated as raw text.
    """
    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return str(source)


def load_image(source: Union[str, Path]):
    """Load an image as a PIL ``Image`` in RGB mode.

    Parameters:
        source: Path to an image file.

    Returns:
        A ``PIL.Image.Image`` in RGB.
    """
    from PIL import Image

    return Image.open(source).convert("RGB")


def load_audio(
    source: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *target_sr*.

    Attempts to use ``torchaudio`` first, falling back to ``librosa``.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D numpy
        array (mono, float32).
    """
    path = str(source)

    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.mean(dim=0).numpy()
        return waveform, target_sr
    except ImportError:
        pass

    import librosa

    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform, sr
