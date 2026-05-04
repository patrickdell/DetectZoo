"""RoBERTa OpenAI Detector — supervised GPT-2 output detection.

Reference:
    Solaiman et al., "Release strategies and the social impacts
    of language models." arXiv 2024.


Uses the pre-trained OpenAI GPT-2 output detector models hosted on
HuggingFace.  Two variants are available:

* **base** — ``openai-community/roberta-base-openai-detector`` (125 M params)
* **large** — ``openai-community/roberta-large-openai-detector`` (355 M params)

The models are RoBERTa-based sequence classifiers fine-tuned to
distinguish WebText (human) from 1.5B-parameter GPT-2 outputs.
Label 0 = *Real* (human), label 1 = *Fake* (AI-generated).
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_VARIANTS: dict[str, str] = {
    "base": "openai-community/roberta-base-openai-detector",
    "large": "openai-community/roberta-large-openai-detector",
}


class _RobertaOpenAIBase(BaseTextDetector):
    """Shared implementation for both RoBERTa OpenAI Detector variants.

    Parameters:
        threshold: Decision boundary on the AI-class probability.
        max_length: Maximum token length for the tokenizer.
        device: ``"cpu"`` or ``"cuda"``.
    """

    modality = "text"
    _variant: str = "base"

    def __init__(
        self,
        threshold: float = 0.5,
        max_length: int = 512,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        model_name = _VARIANTS[self._variant]
        super().__init__(
            model_name=model_name,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self._cls_model: torch.nn.Module | None = None
        self._cls_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Model loading — sequence-classification head
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading RoBERTa OpenAI Detector (%s) '%s' …", self._variant, self.model_name)
        self._cls_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        ).to(self._device)
        self._cls_model.eval()

    @property
    def cls_model(self) -> torch.nn.Module:
        if self._cls_model is None:
            self._load_model()
        return self._cls_model  # type: ignore[return-value]

    @property
    def cls_tokenizer(self):
        if self._cls_tokenizer is None:
            self._load_model()
        return self._cls_tokenizer

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.cls_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self._device)

        logits = self.cls_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        return self._make_result(
            fake_prob,
            real_prob=real_prob,
            fake_prob=fake_prob,
            variant=self._variant,
        )


# ------------------------------------------------------------------
# Registered variants
# ------------------------------------------------------------------


@register_detector("roberta_base", aliases=["roberta_openai_base"])
class RobertaBaseDetector(_RobertaOpenAIBase):
    """RoBERTa Base OpenAI Detector (125 M parameters).

    Uses ``openai-community/roberta-base-openai-detector``.
    """

    _variant = "base"


@register_detector("roberta_large", aliases=["roberta_openai_large"])
class RobertaLargeDetector(_RobertaOpenAIBase):
    """RoBERTa Large OpenAI Detector (355 M parameters).

    Uses ``openai-community/roberta-large-openai-detector``.
    """

    _variant = "large"
