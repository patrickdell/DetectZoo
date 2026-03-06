"""Binoculars detector — cross-perplexity ratio from two LMs.

Reference:
    Artidoro Pagnoni et al., "Binoculars: Zero-Shot Detection of
    LLM-Generated Text", 2024.

The detector uses an *observer* and a *performer* model.  For human
text both models have roughly equal perplexity, yielding a ratio close
to 1.  For machine text the performer (closer to the generating model)
has lower perplexity, pushing the ratio down.

    B(x) = PPL_observer(x) / PPL_performer(x)

Low Binoculars score → likely AI.  We negate the ratio so that
*higher* score → more likely AI, matching the rest of DetectZoo.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("binoculars")
class BinocularsDetector(BaseTextDetector):
    """Binoculars detector (two-model perplexity ratio).

    Parameters:
        observer_model: Observer LM name (default ``"gpt2"``).
        performer_model: Performer LM name (default ``"gpt2-medium"``).
        threshold: Decision threshold (on the negated ratio).
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=observer_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.performer_model_name = performer_model
        self._performer_model: torch.nn.Module | None = None
        self._performer_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def performer_model(self) -> torch.nn.Module:
        if self._performer_model is None:
            self._load_performer()
        return self._performer_model  # type: ignore[return-value]

    @property
    def performer_tokenizer(self):
        if self._performer_tokenizer is None:
            self._load_performer()
        return self._performer_tokenizer

    def _load_performer(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading performer model '%s' …", self.performer_model_name)
        self._performer_tokenizer = AutoTokenizer.from_pretrained(self.performer_model_name)
        if self._performer_tokenizer.pad_token is None:
            self._performer_tokenizer.pad_token = self._performer_tokenizer.eos_token
        self._performer_model = AutoModelForCausalLM.from_pretrained(
            self.performer_model_name
        ).to(self._device)
        self._performer_model.eval()

    # ------------------------------------------------------------------
    # Perplexity helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ppl(self, text: str, model, tokenizer) -> float:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        logits = model(**enc).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        return math.exp(float(loss))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        ppl_observer = self._ppl(text, self.model, self.tokenizer)
        ppl_performer = self._ppl(text, self.performer_model, self.performer_tokenizer)

        ratio = ppl_observer / max(ppl_performer, 1e-8)

        # Negate so higher → more likely AI (when performer has lower PPL,
        # ratio > 1, and negated ratio < 0 for human; for machine text
        # ratio < 1, negated > 0).
        score = -math.log(ratio)

        return self._make_result(
            score,
            ppl_observer=ppl_observer,
            ppl_performer=ppl_performer,
            ratio=ratio,
        )
