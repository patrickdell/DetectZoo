"""WaveFake — neural vocoder deepfake audio (LJSpeech / JSUT conditioning).

Reference:
    Frank & Schönherr, "WaveFake: A Data Set to Facilitate Audio Deepfake
    Detection", NeurIPS 2021 Datasets and Benchmarks Track.
    https://arxiv.org/abs/2111.02813

Zenodo (generated / fake utterances only):
    https://zenodo.org/records/4904579  (DOI 10.5281/zenodo.4904579)

Bonafide waveforms are **not** in the Zenodo archive; obtain LJSpeech and/or
JSUT separately and pass ``real_paths`` (see the `RUB-SysSec/WaveFake`_ repo).

.. _RUB-SysSec/WaveFake: https://github.com/RUB-SysSec/WaveFake
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_ZENODO_ZIP = (
    "https://zenodo.org/records/4904579/files/generated_audio.zip?download=1"
)
_WAV_EXT = frozenset({".wav", ".WAV"})


def _parse_generator_tag(stem: str) -> str | None:
    """Return WF* tag from stems like ``LJ001-0001_WF3`` or ``None``."""
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[-1]
    u = suffix.upper()
    if u.startswith("WF") and u[2:].isdigit():
        return u
    return None


def _infer_corpus_from_stem(stem: str) -> str:
    s = stem.upper()
    if s.startswith("LJ"):
        return "ljspeech"
    if "BASIC" in s or s.startswith("TR"):
        return "jsut"
    return "unknown"


def _infer_corpus_from_path(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if any("ljspeech" in x or "lj-speech" in x for x in parts):
        return "ljspeech"
    if any("jsut" in x for x in parts):
        return "jsut"
    return "unknown"


def _resolve_generated_root(user_root: Path) -> Path:
    u = user_root.resolve()
    ga = u / "generated_audio"
    if ga.is_dir():
        return ga
    return u


def _collect_wavs(
    root: Path,
    *,
    label: int,
    class_name: str,
    require_wf_tag: bool,
    generators: frozenset[str] | None,
) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _WAV_EXT:
            continue
        stem = path.stem
        gen = _parse_generator_tag(stem)
        if require_wf_tag and gen is None:
            continue
        if generators is not None and gen is not None and gen not in generators:
            continue
        corpus = _infer_corpus_from_stem(stem)
        meta = {
            "modality": "audio",
            "class": class_name,
            "corpus": corpus,
            "generator": gen or "",
        }
        items.append(DatasetItem(data=str(path), label=label, metadata=meta))
    return items


def _collect_real_roots(real_paths: Sequence[str | Path]) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for rp in real_paths:
        root = Path(rp).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"WaveFake: real audio directory not found: {root}")
        corpus_hint = _infer_corpus_from_path(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in _WAV_EXT:
                continue
            items.append(
                DatasetItem(
                    data=str(path),
                    label=0,
                    metadata={
                        "modality": "audio",
                        "class": "bonafide",
                        "corpus": corpus_hint,
                        "generator": "",
                    },
                )
            )
    return items


@register_dataset("wavefake", aliases=["wave_fake", "wavefake_dataset"])
class WaveFakeDataset(BaseDataset):
    """WaveFake binary detection dataset (bonafide vs generated).

    The official Zenodo release contains **generated** clips only (named with a
    ``_WF*`` vocoder suffix, e.g. ``LJ001-0001_WF1.wav``). Matching **bonafide**
    audio lives in the original LJSpeech / JSUT corpora; pass those directories
    via ``real_paths`` to build a full labelled set (as in the upstream
    training scripts).

    Parameters
    ----------
    path
        Local directory: either the extracted ``generated_audio`` folder, or a
        parent that contains ``generated_audio/``.
    real_paths
        Optional list of directories with real ``.wav`` files (e.g.
        ``LJSpeech-1.1/wavs``). When omitted, only generated samples are loaded
        (all label ``1``).
    download
        If ``True`` and ``path`` is ``None``, download ``generated_audio.zip``
        from Zenodo record 4904579 into ``cache_dir`` and extract it. The
        archive is large (~16 GB).
    cache_dir
        Root cache directory (default ``.detectzoo_data``).
    generators
        If set, keep only files whose stem ends with one of these tags (e.g.
        ``\"WF1\"``, ``\"WF7\"``). Comparison is case-insensitive.
    fake_only
        If ``True``, ignore ``real_paths`` and load only generated audio.
    max_samples
        Optional cap (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "wavefake"
    modality = "audio"

    def __init__(
        self,
        path: str | Path | None = None,
        real_paths: Sequence[str | Path] | None = None,
        *,
        download: bool = True,
        cache_dir: str | Path | None = None,
        generators: Sequence[str] | None = None,
        fake_only: bool = False,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path) if path is not None else None
        self.real_paths: Optional[Sequence[str | Path]] = None if fake_only else real_paths
        self.download = download
        self.cache_dir = cache_dir
        self.generators = (
            frozenset(g.strip().upper() for g in generators) if generators else None
        )

    def _ensure_generated_root(self) -> Path:
        if self.path is not None:
            return _resolve_generated_root(self.path)

        if not self.download:
            raise ValueError(
                "WaveFakeDataset: `path` is None and download=False — "
                "provide a local `path` to generated audio."
            )

        from detectzoo.datasets._download import download_and_extract_zip, get_cache_dir

        dest = get_cache_dir("wavefake", self.cache_dir)
        download_and_extract_zip(_ZENODO_ZIP, dest, force=False)
        return _resolve_generated_root(dest)

    def _load_all(self) -> List[DatasetItem]:
        gen_root = self._ensure_generated_root()
        fake_items = _collect_wavs(
            gen_root,
            label=1,
            class_name="fake",
            require_wf_tag=True,
            generators=self.generators,
        )
        if not fake_items:
            raise FileNotFoundError(
                f"WaveFake: no generated WAVs with _WF* suffix under {gen_root}"
            )

        items: List[DatasetItem] = list(fake_items)
        if self.real_paths:
            items.extend(_collect_real_roots(self.real_paths))
        return items
