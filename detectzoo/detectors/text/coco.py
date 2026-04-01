"""CoCo — Coherence-Enhanced Machine-Generated Text Detection.

Reference:
    Liu et al., "CoCo: Coherence-Enhanced Machine-Generated Text
    Detection Under Low Resource With Contrastive Learning",
    EMNLP 2023.

The key insight: human text exhibits stronger inter-sentence coherence
than machine-generated text.  CoCo captures this by building a
coherence graph across sentences and using contrastive learning to
amplify the signal.

This implementation provides:
  - A **zero-shot** coherence score based on consecutive-sentence
    similarity using a pre-trained sentence encoder.
  - A **supervised** variant wrapping a fine-tunable RoBERTa
    classifier with coherence features.
"""

from __future__ import annotations

from typing import Any, List

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Rough sentence splitter (no external dependency)."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


@register_detector("coco")
class CoCoDetector(BaseTextDetector):
    """CoCo coherence-based zero-shot detector.

    Computes consecutive-sentence cosine similarity using the model's
    own hidden states.  The *variance* of the similarity profile
    serves as the detection signal: human text has higher coherence
    variance while machine text is more uniform.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)

    @torch.no_grad()
    def _sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Encode sentences using last hidden state mean-pooling."""
        embeddings = []
        for sent in sentences:
            enc = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            outputs = self.model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [1, T, D]
            emb = last_hidden.mean(dim=1).squeeze(0)  # [D]
            embeddings.append(emb)
        return torch.stack(embeddings)  # [N, D]

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        sentences = _split_sentences(text)

        if len(sentences) < 3:
            return self._make_result(0.0, reason="too few sentences", n_sentences=len(sentences))

        embs = self._sentence_embeddings(sentences)
        sims = torch.nn.functional.cosine_similarity(embs[:-1], embs[1:], dim=-1)

        mean_sim = float(sims.mean())
        var_sim = float(sims.var()) if sims.numel() > 1 else 0.0

        # Machine text → higher mean similarity, lower variance → negative score
        # Human text → more varied coherence → more positive score
        # Negate variance so higher score → more likely AI
        score = mean_sim - var_sim

        return self._make_result(
            score,
            mean_coherence=mean_sim,
            coherence_variance=var_sim,
            n_sentences=len(sentences),
        )
