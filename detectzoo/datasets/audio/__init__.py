"""Audio-modality datasets for AI-generated audio detection."""

from detectzoo.datasets.audio.asvspoof2019 import ASVspoof2019Dataset
from detectzoo.datasets.audio.for_dataset import FoRDataset
from detectzoo.datasets.audio.wavefake import WaveFakeDataset

__all__ = ["ASVspoof2019Dataset", "FoRDataset", "WaveFakeDataset"]
