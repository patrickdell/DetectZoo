"""Dataset abstractions for AI-content detection benchmarks."""

from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.datasets.text import (
    CHEATDataset,
    HC3Dataset,
    HC3PlusDataset,
    MAGEDataset,
    OpenLLMTextDataset,
    WritingPromptsDataset,
    XSumDataset,
)

__all__ = [
    "BaseDataset",
    "DatasetItem",
    "CHEATDataset",
    "HC3Dataset",
    "HC3PlusDataset",
    "MAGEDataset",
    "OpenLLMTextDataset",
    "WritingPromptsDataset",
    "XSumDataset",
]
