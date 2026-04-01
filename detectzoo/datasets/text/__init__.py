"""Text-modality datasets for LLM-generated text detection."""

from detectzoo.datasets.text.cheat import CHEATDataset
from detectzoo.datasets.text.hc3 import HC3Dataset
from detectzoo.datasets.text.hc3_plus import HC3PlusDataset
from detectzoo.datasets.text.mage import MAGEDataset
from detectzoo.datasets.text.open_llm_text import OpenLLMTextDataset
from detectzoo.datasets.text.writing_prompts import WritingPromptsDataset
from detectzoo.datasets.text.xsum import XSumDataset

__all__ = [
    "CHEATDataset",
    "HC3Dataset",
    "HC3PlusDataset",
    "MAGEDataset",
    "OpenLLMTextDataset",
    "WritingPromptsDataset",
    "XSumDataset",
]
