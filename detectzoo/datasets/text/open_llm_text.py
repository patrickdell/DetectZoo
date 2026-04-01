"""OpenLLMText – Multi-LLM text detection dataset.

Reference:
    Chen et al., "Token Prediction as Implicit Classification to Identify
    LLM-Generated Text", EMNLP 2023.
    https://arxiv.org/abs/2311.08723

GitHub: https://github.com/MarkChenYutian/T5-Sentinel-public
Data hosted on Zenodo: https://zenodo.org/records/8285326
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

from detectzoo.datasets.base import BaseDataset, DatasetItem

_ZENODO_BASE = "https://zenodo.org/records/8285326/files"

_SOURCES: dict[str, tuple[str, str, int]] = {
    "gpt2": ("GPT2.zip", "gpt2-output", 1),
    "chatgpt": ("ChatGPT.zip", "open-gpt-text", 1),
    "llama": ("LLaMA.zip", "open-llama-text", 1),
    "palm": ("PaLM.zip", "open-palm-text", 1),
    "human": ("Human.zip", "open-web-text", 0),
}

_SPLIT_FILES = {
    "train": "train-dirty.jsonl",
    "test": "test-dirty.jsonl",
    "valid": "valid-dirty.jsonl",
}


class OpenLLMTextDataset(BaseDataset):
    """OpenLLMText dataset for multi-source LLM text detection.

    Contains ~340,000 text samples written by humans and several LLMs
    (GPT-2, ChatGPT/GPT-3.5, PaLM, LLaMA), enabling both binary
    detection and source attribution.

    When *path* is omitted the ZIP archives are downloaded automatically
    from `Zenodo <https://zenodo.org/records/8285326>`_ and cached under
    ``.detectzoo_data/open_llm_text/``.

    Parameters
    ----------
    path : str or Path, optional
        Local root directory that contains per-source subdirectories
        (``gpt2-output/``, ``open-gpt-text/``, etc.).  When *None* the
        data is downloaded from Zenodo.
    sources : sequence of str, optional
        Filter to specific sources: ``"gpt2"``, ``"chatgpt"``,
        ``"llama"``, ``"palm"``, ``"human"``.  *None* loads all.
    splits : sequence of str, optional
        Filter to specific splits: ``"train"``, ``"test"``,
        ``"valid"``.  *None* loads all.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "open_llm_text"
    modality = "text"

    def __init__(
        self,
        path: str | Path | None = None,
        sources: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.sources = {s.lower() for s in sources} if sources else None
        self.splits = set(splits) if splits else None
        self.cache_dir = cache_dir
        self._items: Optional[List[DatasetItem]] = None

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_and_extract_zip, get_cache_dir

        data_dir = get_cache_dir("open_llm_text", self.cache_dir)
        requested = _SOURCES.items()
        if self.sources:
            requested = [(k, v) for k, v in _SOURCES.items() if k in self.sources]
        for _key, (zip_name, _subdir, _label) in requested:
            url = f"{_ZENODO_BASE}/{zip_name}?download=1"
            download_and_extract_zip(url, data_dir / _subdir)
        return data_dir

    def _iter_source_dirs(self, data_dir: Path):
        """Yield ``(source_key, directory, label)`` for requested sources."""
        for key, (_zip, subdir, label) in _SOURCES.items():
            if self.sources and key not in self.sources:
                continue
            d = data_dir / subdir
            if d.is_dir():
                yield key, d, label

    def _load_jsonl(
        self,
        fp: Path,
        source: str,
        label: int,
        split_name: str,
    ) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                items.append(DatasetItem(
                    data=row.get("text", row.get("string", "")),
                    label=label,
                    metadata={"source": source, "split": split_name},
                ))
        return items

    def load(self) -> List[DatasetItem]:
        if self._items is not None:
            return self._items

        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for source_key, source_dir, label in self._iter_source_dirs(data_dir):
            for split_name, filename in _SPLIT_FILES.items():
                if self.splits and split_name not in self.splits:
                    continue
                fp = source_dir / filename
                if fp.exists():
                    items.extend(self._load_jsonl(fp, source_key, label, split_name))

        self._items = items
        return self._items
