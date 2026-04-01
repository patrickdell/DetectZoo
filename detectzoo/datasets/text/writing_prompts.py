"""WritingPrompts – Creative story generation dataset.

WritingPrompts is a large dataset of human-written stories paired with
writing prompts, originally collected from Reddit's r/WritingPrompts.
It is used as a source corpus in LLM-generated text detection research
where model-generated stories are compared against human-written ones.

Reference:
    Fan et al., "Hierarchical Neural Story Generation", ACL 2018.
    https://arxiv.org/abs/1805.04833

HuggingFace: ``euclaise/writingprompts``
GitHub: https://github.com/facebookresearch/fairseq/blob/main/examples/stories/README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from detectzoo.datasets.base import BaseDataset, DatasetItem


class WritingPromptsDataset(BaseDataset):
    """WritingPrompts dataset for text detection benchmarking.

    Contains ~303k human-written short stories from Reddit's
    r/WritingPrompts, each paired with its writing prompt.

    Each item's ``data`` field contains the story text (``label=0``,
    human-written).  The prompt is available in
    ``metadata["prompt"]``.

    This is a *source* corpus — it only contains human text.  Use it
    together with LLM-generated stories (``label=1``) to build a
    detection benchmark.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file.  When *None* the dataset is loaded
        from HuggingFace (``euclaise/writingprompts``).
    split : str
        HuggingFace split (default ``"test"``).  Also ``"train"``
        (273k rows) and ``"validation"`` (15.6k rows).
    max_samples : int, optional
        Cap the number of samples loaded (useful for quick experiments).
    """

    name = "writing_prompts"
    modality = "text"

    def __init__(
        self,
        path: str | Path | None = None,
        split: str = "test",
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.split = split
        self.max_samples = max_samples
        self._items: Optional[List[DatasetItem]] = None

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        ds = load_dataset("euclaise/writingprompts", split=self.split)
        items: List[DatasetItem] = []
        for row in ds:
            items.append(DatasetItem(
                data=row.get("story", ""),
                label=0,
                metadata={
                    "source": "human",
                    "prompt": row.get("prompt", ""),
                },
            ))
            if self.max_samples and len(items) >= self.max_samples:
                break
        return items

    def _load_from_local(self) -> List[DatasetItem]:
        import json

        assert self.path is not None
        items: List[DatasetItem] = []
        files = sorted(self.path.glob("*.jsonl")) if self.path.is_dir() else [self.path]
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    row: dict[str, Any] = json.loads(line)
                    items.append(DatasetItem(
                        data=row.get("story", row.get("text", "")),
                        label=int(row.get("label", 0)),
                        metadata={
                            "source": row.get("source", "human"),
                            "prompt": row.get("prompt", ""),
                        },
                    ))
                    if self.max_samples and len(items) >= self.max_samples:
                        return items
        return items

    def load(self) -> List[DatasetItem]:
        if self._items is not None:
            return self._items
        if self.path is not None:
            self._items = self._load_from_local()
        else:
            self._items = self._load_from_huggingface()
        return self._items
