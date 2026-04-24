"""RAID – Robust AI-generated Text Detection Benchmark.

Reference:
    Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of
    Machine-Generated Text Detectors", ACL 2024.
    https://aclanthology.org/2024.acl-long.674.pdf

HuggingFace: ``liamdugan/raid``
GitHub    : https://github.com/liamdugan/raid
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("raid")
class RAIDDataset(BaseDataset):
    """RAID dataset for robust machine-generated text detection.

    The largest & most comprehensive detection benchmark: over 10 million
    documents spanning 11 LLMs, 11 genres, 4 decoding strategies, and 12
    adversarial attacks.  Each row contains one *generation* (either a
    human reference or an LLM output under a specific configuration).

    When *path* is omitted the CSV files are streamed from HuggingFace
    (``liamdugan/raid``) via the ``datasets`` library.  To work offline,
    download ``train.csv`` / ``test.csv`` / ``extra.csv`` from the
    HuggingFace repo and point *path* to the containing directory.

    Parameters
    ----------
    path : str or Path, optional
        Local directory with ``train.csv``/``test.csv``/``extra.csv`` or
        a direct path to one such CSV file.  When *None* the dataset is
        loaded from HuggingFace.
    split : str
        Which RAID split to load.  One of ``"train"`` (labeled,
        ~802 M w/o adversarial), ``"test"`` (leaderboard split — the
        ``model`` column is still provided, but detector outputs are
        typically evaluated on the official server), or ``"extra"``
        (labeled, code + Czech + German).  Default ``"train"``.
    models : sequence of str, optional
        Filter to specific generators (e.g. ``["chatgpt", "gpt4"]``).
        Use ``"human"`` to keep the human baseline.  *None* loads all.
    domains : sequence of str, optional
        Filter to specific genres (e.g. ``["news", "books"]``).
    attacks : sequence of str, optional
        Filter to specific adversarial attacks.  Pass ``["none"]`` to
        get the non-adversarial subset (recommended for initial
        experiments).  *None* loads all.
    decoding : sequence of str, optional
        Filter to specific decoding strategies (``"greedy"`` or
        ``"sampling"``).
    repetition_penalty : sequence of str, optional
        Filter to specific repetition-penalty flags (``"yes"`` / ``"no"``).
    include_human : bool
        When ``models`` is provided, still keep the human reference
        samples (label=0).  Default ``True``.
    max_samples : int, optional
        Cap the number of samples loaded (useful for quick experiments).
    """

    name = "raid"
    modality = "text"

    info = (
        "RAID (Robust AI-generated text Detection benchmark)\n"
        "===================================================\n"
        "The largest shared benchmark for machine-generated text\n"
        "detectors: >10M documents spanning 11 LLMs, 11 genres,\n"
        "4 decoding strategies, and 12 adversarial attacks.\n"
        "\n"
        "Paper  : Dugan et al., 'RAID: A Shared Benchmark for Robust\n"
        "         Evaluation of Machine-Generated Text Detectors',\n"
        "         ACL 2024.\n"
        "\n"
        "Models\n"
        "------\n"
        "  chatgpt, gpt4, gpt3, gpt2, llama-chat, mistral, mistral-chat,\n"
        "  mpt, mpt-chat, cohere, cohere-chat, human\n"
        "\n"
        "Domains\n"
        "-------\n"
        "  abstracts, books, code, czech, german, news, poetry, recipes,\n"
        "  reddit, reviews, wiki\n"
        "\n"
        "Attacks\n"
        "-------\n"
        "  none, homoglyph, number, article_deletion, insert_paragraphs,\n"
        "  perplexity_misspelling, upper_lower, whitespace,\n"
        "  zero_width_space, synonym, paraphrase, alternative_spelling\n"
        "\n"
        "Splits\n"
        "------\n"
        "  train – labeled, 8 domains (English), 802 MB (non-adv) / 11.8 GB (w/ adv)\n"
        "  test  – leaderboard split (labels hidden server-side, but the\n"
        "          model column is provided for offline analysis)\n"
        "  extra – labeled, 3 extra domains (code, czech, german)\n"
        "\n"
        "Labels: DetectZoo normalises to 0 = human, 1 = AI (any model).\n"
        "        The full model name is preserved in metadata['model'].\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Start with non-adversarial training data for a single model:\n"
        "  RAIDDataset(split='train', models=['chatgpt'],\n"
        "              attacks=['none'], decoding=['greedy'])\n"
        "Full adversarial sweep (16.7 GB!) — use max_samples or streaming:\n"
        "  RAIDDataset(split='train', max_samples=20000)\n"
    )

    MODELS = (
        "chatgpt", "gpt4", "gpt3", "gpt2",
        "llama-chat", "mistral", "mistral-chat",
        "mpt", "mpt-chat", "cohere", "cohere-chat", "human",
    )
    DOMAINS = (
        "abstracts", "books", "code", "czech", "german",
        "news", "poetry", "recipes", "reddit", "reviews", "wiki",
    )
    ATTACKS = (
        "none", "homoglyph", "number", "article_deletion",
        "insert_paragraphs", "perplexity_misspelling", "upper_lower",
        "whitespace", "zero_width_space", "synonym", "paraphrase",
        "alternative_spelling",
    )
    SPLITS = ("train", "test", "extra")

    _HF_REPO = "liamdugan/raid"
    _SPLIT_TO_FILE = {"train": "train.csv", "test": "test.csv", "extra": "extra.csv"}

    def __init__(
        self,
        path: str | Path | None = None,
        split: str = "train",
        models: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        attacks: Sequence[str] | None = None,
        decoding: Sequence[str] | None = None,
        repetition_penalty: Sequence[str] | None = None,
        include_human: bool = True,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)
        if split not in self._SPLIT_TO_FILE:
            raise ValueError(
                f"Unknown RAID split '{split}'. Valid: {list(self._SPLIT_TO_FILE)}"
            )
        self.path = Path(path) if path is not None else None
        self.split = split
        self.models = {m.lower() for m in models} if models else None
        self.domains = {d.lower() for d in domains} if domains else None
        self.attacks = {a.lower() for a in attacks} if attacks else None
        self.decoding = {d.lower() for d in decoding} if decoding else None
        self.repetition_penalty = (
            {r.lower() for r in repetition_penalty} if repetition_penalty else None
        )
        self.include_human = include_human

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _keep(self, row: dict[str, Any]) -> bool:
        model = str(row.get("model", "")).lower()
        domain = str(row.get("domain", "")).lower()
        attack = str(row.get("attack", "")).lower()
        decoding = str(row.get("decoding", "")).lower()
        rep = str(row.get("repetition_penalty", "")).lower()

        if self.models is not None:
            if model not in self.models:
                if not (self.include_human and model == "human"):
                    return False
        if self.domains is not None and domain not in self.domains:
            return False
        if self.attacks is not None and attack not in self.attacks:
            return False
        if self.decoding is not None and decoding not in self.decoding:
            return False
        if self.repetition_penalty is not None and rep not in self.repetition_penalty:
            return False
        return True

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> DatasetItem:
        model = str(row.get("model", "")).lower()
        label = 0 if model == "human" else 1
        return DatasetItem(
            data=row.get("generation", ""),
            label=label,
            metadata={
                "id": row.get("id", ""),
                "model": model,
                "domain": row.get("domain", ""),
                "attack": row.get("attack", ""),
                "decoding": row.get("decoding", ""),
                "repetition_penalty": row.get("repetition_penalty", ""),
                "source_id": row.get("source_id", ""),
                "adv_source_id": row.get("adv_source_id", ""),
                "title": row.get("title", ""),
                "prompt": row.get("prompt", ""),
            },
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _iter_rows_huggingface(self) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(
            self._HF_REPO,
            data_files={self.split: self._SPLIT_TO_FILE[self.split]},
            split=self.split,
            streaming=True,
        )
        for row in ds:
            yield row

    def _iter_rows_local(self) -> Iterable[dict[str, Any]]:
        assert self.path is not None
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            target = self.path / self._SPLIT_TO_FILE[self.split]
            files = [target] if target.exists() else sorted(self.path.glob("*.csv"))
        else:
            raise FileNotFoundError(f"RAID path does not exist: {self.path}")

        for fp in files:
            with open(fp, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield row

    def _load_all(self) -> List[DatasetItem]:
        iterator = (
            self._iter_rows_local() if self.path is not None else self._iter_rows_huggingface()
        )
        items: List[DatasetItem] = []
        for row in iterator:
            if not self._keep(row):
                continue
            items.append(self._row_to_item(row))
            if self.max_samples is not None and len(items) >= self.max_samples:
                break
        return items
