"""CHEAT – Detecting CHatGPT-writtEn AbsTracts.

Reference:
    Yu et al., "CHEAT: A Large-scale Dataset for Detecting CHatGPT-writtEn
    AbsTracts", IEEE Transactions on Big Data 2025.
    https://arxiv.org/abs/2304.12008

GitHub: https://github.com/botianzhe/CHEAT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

from detectzoo.datasets.base import BaseDataset, DatasetItem

_GITHUB_RAW = "https://raw.githubusercontent.com/botianzhe/CHEAT/main/data"

_FILES: dict[str, tuple[str, int]] = {
    "ieee-init.jsonl": ("human", 0),
    "ieee-chatgpt-generation.jsonl": ("generation", 1),
    "ieee-chatgpt-polish.jsonl": ("polish", 1),
    "ieee-chatgpt-fusion.jsonl": ("fusion", 1),
}


class CHEATDataset(BaseDataset):
    """CHEAT dataset for detecting ChatGPT-written academic abstracts.

    Contains 35,304 synthetic abstracts in three categories:

    * **generation** – first-pass ChatGPT output
    * **polish** – ChatGPT-refined versions of human abstracts
    * **fusion** – human / ChatGPT hybrid abstracts

    When *path* is omitted the JSONL files are downloaded automatically
    from `GitHub <https://github.com/botianzhe/CHEAT>`_ and cached
    under ``.detectzoo_data/cheat/``.

    Parameters
    ----------
    path : str or Path, optional
        Local directory containing the JSONL files.  When *None* the
        data is downloaded from GitHub.
    categories : sequence of str, optional
        Filter to specific categories (``"generation"``, ``"polish"``,
        ``"fusion"``).  *None* loads all including the human baseline.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "cheat"
    modality = "text"

    CATEGORIES = ("generation", "polish", "fusion")

    def __init__(
        self,
        path: str | Path | None = None,
        categories: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.categories = set(categories) if categories else None
        self.cache_dir = cache_dir
        self._items: Optional[List[DatasetItem]] = None

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_file, get_cache_dir

        data_dir = get_cache_dir("cheat", self.cache_dir)
        for filename in _FILES:
            download_file(f"{_GITHUB_RAW}/{filename}", data_dir / filename)
        return data_dir

    def _load_jsonl(self, fp: Path, category: str, label: int) -> List[DatasetItem]:
        if self.categories and category not in self.categories and label != 0:
            return []
        items: List[DatasetItem] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                row: dict[str, Any] = json.loads(line)
                items.append(DatasetItem(
                    data=row["abstract"],
                    label=label,
                    metadata={
                        "category": category,
                        "title": row.get("title", ""),
                    },
                ))
        return items

    def load(self) -> List[DatasetItem]:
        if self._items is not None:
            return self._items

        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for filename, (category, label) in _FILES.items():
            fp = data_dir / filename
            if fp.exists():
                items.extend(self._load_jsonl(fp, category, label))

        self._items = items
        return self._items
