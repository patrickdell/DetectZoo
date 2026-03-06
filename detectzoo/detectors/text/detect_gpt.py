"""DetectGPT — perturbation-based LLM-text detector.

Reference:
    Eric Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text
    Detection using Probability Curvature", ICML 2023.

The key idea: machine-generated text sits near local maxima of the
log-probability surface of the source model.  By comparing the original
log-prob to those of minor perturbations (produced by a mask-filling
model such as T5) one obtains a curvature score that separates human
from machine text.
"""

from __future__ import annotations

import random
import re
from typing import Any, List

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("detectgpt")
class DetectGPTDetector(BaseTextDetector):
    """DetectGPT detector using perturbation-based probability curvature.

    Parameters:
        model_name: Scoring model (causal LM, default ``"gpt2"``).
        perturbation_model: Mask-filling model (default ``"t5-small"``).
        n_perturbations: Number of perturbations to generate.
        mask_pct: Fraction of tokens to mask for perturbation.
        threshold: Decision threshold on the z-score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        perturbation_model: str = "t5-small",
        n_perturbations: int = 25,
        mask_pct: float = 0.15,
        threshold: float = 0.0,
        device: str = "cpu",
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.perturbation_model_name = perturbation_model
        self.n_perturbations = n_perturbations
        self.mask_pct = mask_pct
        self._pmodel: torch.nn.Module | None = None
        self._ptokenizer: Any = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Perturbation model loading
    # ------------------------------------------------------------------

    @property
    def perturbation_model(self) -> torch.nn.Module:
        if self._pmodel is None:
            self._load_perturbation_model()
        return self._pmodel  # type: ignore[return-value]

    @property
    def perturbation_tokenizer(self):
        if self._ptokenizer is None:
            self._load_perturbation_model()
        return self._ptokenizer

    def _load_perturbation_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading perturbation model '%s' …", self.perturbation_model_name)
        self._ptokenizer = AutoTokenizer.from_pretrained(self.perturbation_model_name)
        self._pmodel = AutoModelForSeq2SeqLM.from_pretrained(
            self.perturbation_model_name
        ).to(self._device)
        self._pmodel.eval()

    # ------------------------------------------------------------------
    # Perturbation generation
    # ------------------------------------------------------------------

    def _mask_text(self, text: str) -> str:
        """Randomly replace spans of words with T5 sentinel tokens."""
        words = text.split()
        n_words = len(words)
        if n_words < 4:
            return text

        n_masks = max(1, int(n_words * self.mask_pct))
        positions = sorted(random.sample(range(n_words), min(n_masks, n_words)))

        spans: List[List[int]] = []
        cur: List[int] = [positions[0]]
        for pos in positions[1:]:
            if pos == cur[-1] + 1 and random.random() < 0.5:
                cur.append(pos)
            else:
                spans.append(cur)
                cur = [pos]
        spans.append(cur)

        result = list(words)
        sentinel_id = 0
        for span in reversed(spans):
            sentinel = f"<extra_id_{sentinel_id}>"
            result[span[0] : span[-1] + 1] = [sentinel]
            sentinel_id += 1

        return " ".join(result)

    @torch.no_grad()
    def _fill_masks(self, masked_text: str) -> str:
        """Use T5 to fill sentinel tokens and reconstruct the text."""
        enc = self.perturbation_tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        output_ids = self.perturbation_model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.96,
            temperature=1.0,
        )
        decoded = self.perturbation_tokenizer.decode(output_ids[0], skip_special_tokens=False)

        fills: dict[int, str] = {}
        for m in re.finditer(r"<extra_id_(\d+)>\s*(.*?)(?=<extra_id_|\Z)", decoded):
            fills[int(m.group(1))] = m.group(2).strip()

        result = masked_text
        for sid in sorted(fills, reverse=True):
            result = result.replace(f"<extra_id_{sid}>", fills.get(sid, ""))
        return result.strip()

    def _perturb(self, text: str) -> str:
        masked = self._mask_text(text)
        return self._fill_masks(masked)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        original_ll = self._mean_log_prob(text)

        perturbation_lls = []
        for _ in range(self.n_perturbations):
            perturbed = self._perturb(text)
            if perturbed.strip():
                perturbation_lls.append(self._mean_log_prob(perturbed))

        if len(perturbation_lls) < 2:
            return self._make_result(0.0, reason="insufficient perturbations")

        mean_pert = float(np.mean(perturbation_lls))
        std_pert = float(np.std(perturbation_lls))

        if std_pert < 1e-8:
            score = 0.0
        else:
            score = (original_ll - mean_pert) / std_pert

        return self._make_result(
            score,
            original_ll=original_ll,
            mean_perturbation_ll=mean_pert,
            std_perturbation_ll=std_pert,
            n_perturbations_used=len(perturbation_lls),
        )
