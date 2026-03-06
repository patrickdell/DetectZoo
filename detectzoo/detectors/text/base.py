"""Shared base class and helpers for text detectors that use causal LMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector
from detectzoo.utils.io import load_text
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTextDetector(BaseDetector):
    """Base for text detectors backed by a HuggingFace causal language model.

    Handles lazy model/tokenizer loading and provides common log-probability
    utilities.
    """

    modality = "text"

    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = 0.5,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self._model: torch.nn.Module | None = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Log-probability utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_logits(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise *text* and return ``(logits, input_ids)``."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        logits = self.model(**enc).logits
        return logits, enc["input_ids"]

    def _token_log_probs(self, text: str) -> torch.Tensor:
        """Return per-token log-probabilities (shape ``[T-1]``)."""
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_lp.squeeze(0)

    def _mean_log_prob(self, text: str) -> float:
        """Average log-probability of *text* under the model."""
        return float(self._token_log_probs(text).mean())

    # ------------------------------------------------------------------
    # Text normalisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_input(input_data: Any) -> str:
        """Accept a file path or raw string and return text."""
        return load_text(input_data)
