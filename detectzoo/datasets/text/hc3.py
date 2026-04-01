"""HC3 – Human ChatGPT Comparison Corpus.

Reference:
    Guo et al., "How Close is ChatGPT to Human Experts? Comparison Corpus,
    Evaluation, and Detection", 2023.
    https://arxiv.org/abs/2301.07597

HuggingFace: ``Hello-SimpleAI/HC3``
GitHub: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from detectzoo.datasets.base import BaseDataset, DatasetItem


class HC3Dataset(BaseDataset):
    """HC3 dataset for human vs. ChatGPT text detection.

    The corpus contains paired human-expert and ChatGPT answers across
    multiple domains (open-domain QA, finance, medicine, law, psychology).

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file path. When *None* the dataset is streamed
        from HuggingFace (``Hello-SimpleAI/HC3``).
    subsets : sequence of str, optional
        Domain subsets to load (e.g. ``["finance", "medicine"]``).
        *None* loads all available subsets.
    split : str
        HuggingFace split to use (default ``"train"``).
    """

    name = "hc3"
    modality = "text"

    SUBSETS = ("all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")

    def __init__(
        self,
        path: str | Path | None = None,
        subsets: Sequence[str] | None = None,
        split: str = "train",
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.subsets = list(subsets) if subsets else ["all"]
        self.split = split
        self._items: Optional[List[DatasetItem]] = None

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        items: List[DatasetItem] = []
        for subset in self.subsets:
            ds = load_dataset("json", data_files=f"hf://datasets/Hello-SimpleAI/HC3/{subset}.jsonl")[self.split]
            for row in ds:
                question = row.get("question", "")
                for answer in row.get("human_answers", []):
                    items.append(DatasetItem(
                        data=answer,
                        label=0,
                        metadata={"source": "human", "question": question, "subset": subset},
                    ))
                for answer in row.get("chatgpt_answers", []):
                    items.append(DatasetItem(
                        data=answer,
                        label=1,
                        metadata={"source": "chatgpt", "question": question, "subset": subset},
                    ))
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
                    question = row.get("question", "")
                    for answer in row.get("human_answers", []):
                        items.append(DatasetItem(
                            data=answer,
                            label=0,
                            metadata={"source": "human", "question": question},
                        ))
                    for answer in row.get("chatgpt_answers", []):
                        items.append(DatasetItem(
                            data=answer,
                            label=1,
                            metadata={"source": "chatgpt", "question": question},
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
