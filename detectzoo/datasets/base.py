"""Base dataset interface for detection benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence


@dataclass
class DatasetItem:
    """A single labelled sample.

    Attributes:
        data: The raw content — text string, image path, or audio path.
        label: Ground-truth label (``1`` for AI, ``0`` for human).
        metadata: Optional provenance information.
    """

    data: Any
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base for detection datasets.

    Subclass and implement :meth:`_load_all` to produce a list of
    :class:`DatasetItem` instances.  The public :meth:`load` method
    handles caching and ``max_samples`` truncation automatically.

    Parameters
    ----------
    max_samples : int, optional
        Cap the number of samples returned by :meth:`load`.
        Useful for debugging and quick experiments.
    """

    name: str = ""
    modality: str = ""

    def __init__(self, *, max_samples: int | None = None) -> None:
        self.max_samples = max_samples
        self._items: Optional[List[DatasetItem]] = None

    @abstractmethod
    def _load_all(self) -> List[DatasetItem]:
        """Load all samples (before ``max_samples`` truncation)."""

    def load(self) -> List[DatasetItem]:
        """Load samples, applying ``max_samples`` truncation if set."""
        if self._items is not None:
            return self._items
        items = self._load_all()
        if self.max_samples is not None:
            items = items[: self.max_samples]
        self._items = items
        return self._items

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.load())

    def __len__(self) -> int:
        return len(self.load())

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        real_dir: str | Path,
        fake_dir: str | Path,
        extensions: Sequence[str] | None = None,
    ) -> "SimpleDirectoryDataset":
        """Create a dataset from two directories (``real/`` and ``fake/``).

        Useful for image and audio benchmarks that organise data by class
        folder.
        """
        return SimpleDirectoryDataset(
            real_dir=Path(real_dir),
            fake_dir=Path(fake_dir),
            extensions=extensions,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        text_column: str = "text",
        label_column: str = "label",
    ) -> "CSVDataset":
        """Create a text dataset from a CSV file."""
        return CSVDataset(
            csv_path=Path(csv_path),
            text_column=text_column,
            label_column=label_column,
        )


class SimpleDirectoryDataset(BaseDataset):
    """Dataset backed by two directories: one for real, one for fake."""

    name = "directory"

    def __init__(
        self,
        real_dir: Path,
        fake_dir: Path,
        extensions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.extensions = extensions

    def _list_files(self, directory: Path) -> List[Path]:
        files = sorted(directory.iterdir())
        if self.extensions:
            exts = {e.lower().lstrip(".") for e in self.extensions}
            files = [f for f in files if f.suffix.lower().lstrip(".") in exts]
        return [f for f in files if f.is_file()]

    def _load_all(self) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        for path in self._list_files(self.real_dir):
            items.append(DatasetItem(data=str(path), label=0, metadata={"source": "real"}))
        for path in self._list_files(self.fake_dir):
            items.append(DatasetItem(data=str(path), label=1, metadata={"source": "fake"}))
        return items


class CSVDataset(BaseDataset):
    """Dataset backed by a CSV file with text and label columns."""

    name = "csv"
    modality = "text"

    def __init__(
        self,
        csv_path: Path,
        text_column: str = "text",
        label_column: str = "label",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column

    def _load_all(self) -> List[DatasetItem]:
        import csv

        items: List[DatasetItem] = []
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                items.append(
                    DatasetItem(
                        data=row[self.text_column],
                        label=int(row[self.label_column]),
                    )
                )
        return items
