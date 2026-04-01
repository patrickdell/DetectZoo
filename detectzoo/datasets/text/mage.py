"""MAGE – Machine-generated Text Detection in the Wild.

Reference:
    Li et al., "MAGE: Machine-generated Text Detection in the Wild", ACL 2024.
    https://arxiv.org/abs/2305.13242

HuggingFace: ``yaful/MAGE``
GitHub: https://github.com/yafuly/MAGE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from detectzoo.datasets.base import BaseDataset, DatasetItem


class MAGEDataset(BaseDataset):
    """MAGE dataset for machine-generated text detection in the wild.

    A comprehensive testbed gathering texts from diverse human writings
    and multiple LLMs for in-distribution and out-of-distribution
    evaluation of text detectors.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file path.  When *None* the dataset is loaded
        from HuggingFace (``yaful/MAGE``).
    sources : sequence of str, optional
        Filter to specific sources / generators.  *None* loads all.
    split : str
        HuggingFace split to use (default ``"train"``).
    text_column : str
        Column name for the text content (default ``"text"``).
    label_column : str
        Column name for the binary label (default ``"label"``).
    """

    name = "mage"
    modality = "text"

    def __init__(
        self,
        path: str | Path | None = None,
        sources: Sequence[str] | None = None,
        split: str = "train",
        text_column: str = "text",
        label_column: str = "label",
        **kwargs: Any,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.sources = {s.lower() for s in sources} if sources else None
        self.split = split
        self.text_column = text_column
        self.label_column = label_column
        self._items: Optional[List[DatasetItem]] = None

    @staticmethod
    def _flip_label(raw_label: int) -> int:
        """MAGE uses 0=machine, 1=human; DetectZoo uses 0=human, 1=AI."""
        return 1 - raw_label

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        ds = load_dataset("yaful/MAGE", split=self.split)
        items: List[DatasetItem] = []
        for row in ds:
            source = row.get("src", "unknown")
            if self.sources and str(source).lower() not in self.sources:
                continue
            items.append(DatasetItem(
                data=row[self.text_column],
                label=self._flip_label(int(row[self.label_column])),
                metadata={"source": source},
            ))
        return items

    def _load_from_local(self) -> List[DatasetItem]:
        import csv
        import json

        items: List[DatasetItem] = []
        files = sorted(self.path.iterdir()) if self.path.is_dir() else [self.path]  # type: ignore[union-attr]
        for fp in files:
            if fp.suffix == ".csv":
                with open(fp, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        source = row.get("source", row.get("model", "unknown"))
                        if self.sources and source.lower() not in self.sources:
                            continue
                        items.append(DatasetItem(
                            data=row[self.text_column],
                            label=self._flip_label(int(row[self.label_column])),
                            metadata={"source": source},
                        ))
            elif fp.suffix in (".json", ".jsonl"):
                with open(fp, encoding="utf-8") as fh:
                    if fp.suffix == ".json":
                        data = json.load(fh)
                    else:
                        data = [json.loads(line) for line in fh]
                for row in data:
                    source = row.get("source", row.get("model", "unknown"))
                    if self.sources and str(source).lower() not in self.sources:
                        continue
                    items.append(DatasetItem(
                        data=row.get(self.text_column, ""),
                        label=self._flip_label(int(row.get(self.label_column, 0))),
                        metadata={"source": source},
                    ))
        return items

    def load(self) -> List[DatasetItem]:
        if self._items is not None:
            return self._items
        if self.path is not None:
            self._items = self._load_from_local()
        else:
            self._items = self._load_from_huggingface()
        return self._items
