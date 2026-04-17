"""Dataset abstractions for AI-content detection benchmarks."""

from detectzoo.datasets.audio import ASVspoof2019Dataset, FoRDataset, WaveFakeDataset
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.datasets.image import (
    AIGCDetectDataset,
    CNNDetectionDataset,
    DRCT2MDataset,
    SelfSynthesisDataset,
)
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
    "ASVspoof2019Dataset",
    "FoRDataset",
    "WaveFakeDataset",
    "AIGCDetectDataset",
    "CNNDetectionDataset",
    "DRCT2MDataset",
    "SelfSynthesisDataset",
    "CHEATDataset",
    "HC3Dataset",
    "HC3PlusDataset",
    "MAGEDataset",
    "OpenLLMTextDataset",
    "WritingPromptsDataset",
    "XSumDataset",
]
