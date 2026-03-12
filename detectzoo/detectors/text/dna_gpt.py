"""DNA-GPT — Divergent N-Gram Analysis for GPT-generated text.

Reference:
    Yang et al., "DNA-GPT: Divergent N-Gram Analysis with LLM Probability
    for Training-Free Detection of GPT-Generated Text", ICLR 2024.

The core idea: truncate the text at a midpoint, use a language model to
re-generate the continuation multiple times, then compare the
log-probability of the *original* full text against the
log-probabilities of the re-generated versions.

Machine text was drawn from a model distribution, so its continuation
is hard to reproduce exactly — the original will have notably higher
log-prob than the re-generations.  For human text the gap is smaller.

    DNA-GPT(x) = log p(x) − mean_i( log p(x̃_i) )

where x̃_i are re-generated versions with different continuations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("dna_gpt")
class DNAGPTDetector(BaseTextDetector):
    """DNA-GPT detector (truncation + re-generation).

    A *single* causal LM is used both for scoring (log-prob) and for
    re-generating continuations.  You may optionally specify a
    separate ``regen_model`` for generation.

    Parameters:
        model_name: Scoring model (default ``"gpt2"``).
        regen_model: Model used to regenerate continuations.  Defaults
            to the same as *model_name*.
        truncate_ratio: Fraction of the text (by words) kept as the
            prompt before re-generation (default ``0.5``).
        n_regens: Number of re-generated continuations.
        max_new_tokens: Maximum tokens to generate per continuation.
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        regen_model: str | None = None,
        truncate_ratio: float = 0.5,
        n_regens: int = 10,
        max_new_tokens: int = 200,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.regen_model_name = regen_model or model_name
        self.truncate_ratio = truncate_ratio
        self.n_regens = n_regens
        self.max_new_tokens = max_new_tokens
        self._regen_model: torch.nn.Module | None = None
        self._regen_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Regen model (may be the same as scoring model)
    # ------------------------------------------------------------------

    @property
    def regen_model(self) -> torch.nn.Module:
        if self.regen_model_name == self.model_name:
            return self.model
        if self._regen_model is None:
            self._load_regen_model()
        return self._regen_model  # type: ignore[return-value]

    @property
    def regen_tokenizer(self):
        if self.regen_model_name == self.model_name:
            return self.tokenizer
        if self._regen_tokenizer is None:
            self._load_regen_model()
        return self._regen_tokenizer

    def _load_regen_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading regen model '%s' …", self.regen_model_name)
        self._regen_tokenizer = AutoTokenizer.from_pretrained(self.regen_model_name)
        if self._regen_tokenizer.pad_token is None:
            self._regen_tokenizer.pad_token = self._regen_tokenizer.eos_token
        self._regen_model = AutoModelForCausalLM.from_pretrained(
            self.regen_model_name
        ).to(self._device)
        self._regen_model.eval()

    # ------------------------------------------------------------------
    # Regeneration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _regenerate(self, prompt: str, target_word_count: int) -> str:
        """Generate a continuation from *prompt* of roughly *target_word_count* words."""
        enc = self.regen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length,
        ).to(self._device)
        out = self.regen_model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=0.96,
            temperature=1.0,
            pad_token_id=self.regen_tokenizer.pad_token_id,
        )
        full = self.regen_tokenizer.decode(out[0], skip_special_tokens=True)
        words = full.split()[:len(prompt.split()) + target_word_count]
        return " ".join(words)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        words = text.split()
        if len(words) < 6:
            return self._make_result(0.0, reason="text too short")

        split_idx = max(3, int(len(words) * self.truncate_ratio))
        prompt = " ".join(words[:split_idx])
        continuation_word_count = len(words) - split_idx

        original_ll = self._mean_log_prob(text)

        regen_lls: list[float] = []
        for _ in range(self.n_regens):
            regen_text = self._regenerate(prompt, continuation_word_count)
            if regen_text.strip():
                regen_lls.append(self._mean_log_prob(regen_text))

        if not regen_lls:
            return self._make_result(0.0, reason="regeneration failed")

        mean_regen_ll = float(np.mean(regen_lls))
        score = original_ll - mean_regen_ll

        return self._make_result(
            score,
            original_ll=original_ll,
            mean_regen_ll=mean_regen_ll,
            n_regens_used=len(regen_lls),
        )
