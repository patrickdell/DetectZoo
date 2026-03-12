"""NPR — Normalized Perturbation Rank detector.

Reference:
    Su et al., "DetectLLM: Leveraging Log Rank Information for
    Zero-Shot Detection of Machine-Generated Text", EMNLP 2023.

Like DetectGPT, NPR generates perturbations of the input text, but
instead of comparing log-probabilities it compares *log-ranks*.

    NPR(x) = mean(LogRank(x̃)) / LogRank(x)

where x̃ are perturbations of x produced by a mask-filling model.
For machine text the original has much lower log-rank than its
perturbations, so the ratio is large.
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


@register_detector("npr")
class NPRDetector(BaseTextDetector):
    """Normalized Perturbation Rank detector.

    Parameters:
        model_name: Scoring model (causal LM, default ``"gpt2"``).
        perturbation_model: Mask-filling model (default ``"t5-small"``).
        n_perturbations: Number of perturbations to generate.
        mask_pct: Fraction of tokens to mask.
        threshold: Decision threshold.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        perturbation_model: str = "t5-small",
        n_perturbations: int = 25,
        mask_pct: float = 0.15,
        threshold: float = 1.0,
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
    # Perturbation model
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
    # Log-rank computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mean_log_rank(self, text: str) -> float | None:
        if not text.strip():
            return None
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        ranks = sorted_indices.argsort(dim=-1)
        token_ranks = ranks.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_ranks = torch.log(token_ranks.float() + 1)
        return float(log_ranks.mean())

    # ------------------------------------------------------------------
    # Perturbation (shared logic with DetectGPT)
    # ------------------------------------------------------------------

    def _mask_text(self, text: str) -> str:
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
            result[span[0] : span[-1] + 1] = [f"<extra_id_{sentinel_id}>"]
            sentinel_id += 1
        return " ".join(result)

    @torch.no_grad()
    def _fill_masks(self, masked_text: str) -> str:
        enc = self.perturbation_tokenizer(
            masked_text, return_tensors="pt", truncation=True, max_length=self.max_length,
        ).to(self._device)
        output_ids = self.perturbation_model.generate(
            **enc, max_new_tokens=256, do_sample=True, top_p=0.96, temperature=1.0,
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
        return self._fill_masks(self._mask_text(text))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        original_lr = self._mean_log_rank(text)
        if original_lr is None or abs(original_lr) < 1e-8:
            return self._make_result(0.0, reason="degenerate input")

        perturbed_lrs: list[float] = []
        for _ in range(self.n_perturbations):
            perturbed = self._perturb(text)
            lr = self._mean_log_rank(perturbed)
            if lr is not None:
                perturbed_lrs.append(lr)

        if len(perturbed_lrs) < 2:
            return self._make_result(0.0, reason="insufficient perturbations")

        mean_pert_lr = float(np.mean(perturbed_lrs))
        score = mean_pert_lr / original_lr

        return self._make_result(
            score,
            original_log_rank=original_lr,
            mean_perturbed_log_rank=mean_pert_lr,
            n_perturbations_used=len(perturbed_lrs),
        )
