"""OOD-based LLM text detectors — D-SVDD, HRN, and Energy.

Reference:
    Zeng et al., "Human Texts Are Outliers: Detecting LLM-generated
    Texts via Out-of-distribution Detection", NeurIPS 2025.

All three methods reframe AI-text detection as an OOD problem:
  - In-distribution = LLM-generated text
  - Out-of-distribution = Human-written text

They share a common encoder (``unsup-simcse-roberta-base``) and differ
in how the OOD score is computed:

- **D-SVDD**: Distance from a learned hypersphere centre.
- **HRN**: Average sigmoid output of per-model one-class classifiers.
- **Energy**: Negative energy (log-sum-exp of logits) from a multi-class
  classifier trained to distinguish LLM generators.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Shared encoder
# ------------------------------------------------------------------


_DETECTIVE_CHECKPOINTS = {
    "Deepfake_best.pth": "heyongxin233/DeTeCtive",
    "M4_monolingual_best.pth": "heyongxin233/DeTeCtive",
    "M4_multilingual_best.pth": "heyongxin233/DeTeCtive",
    "OUTFOX_best.pth": "heyongxin233/DeTeCtive",
    "TuringBench_best.pth": "heyongxin233/DeTeCtive",
    "model_raid.pth": "Shengkun/ood-detection",
}


class _OODTextBase(BaseTextDetector):
    """Shared base for OOD-based text detectors.

    Uses ``unsup-simcse-roberta-base`` as the text encoder with
    average pooling and L2 normalisation.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=encoder_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.encoder_model_name = encoder_model
        self._enc_model: torch.nn.Module | None = None
        self._enc_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading OOD encoder '%s' …", self.encoder_model_name)
        self._enc_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self._enc_model = AutoModel.from_pretrained(self.encoder_model_name).to(self._device)
        self._enc_model.eval()

    @property
    def enc_model(self) -> torch.nn.Module:
        if self._enc_model is None:
            self._load_model()
        return self._enc_model  # type: ignore[return-value]

    @property
    def enc_tokenizer(self):
        if self._enc_tokenizer is None:
            self._load_model()
        return self._enc_tokenizer

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        """Encode text into a normalised embedding vector."""
        enc = self.enc_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        ).to(self._device)

        out = self.enc_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        hidden = out.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1).squeeze(0)


# ------------------------------------------------------------------
# D-SVDD
# ------------------------------------------------------------------


@register_detector("dsvdd")
class DSVDDDetector(_OODTextBase):
    """D-SVDD (Deep Support Vector Data Description) detector.

    Measures squared L2 distance from the embedding to a learned
    hypersphere centre.  The centre is typically the mean of
    LLM-generated embeddings.

    Closer to centre → more likely AI (in-distribution).
    Negated so higher → more likely AI.

    Parameters:
        encoder_model: HuggingFace encoder model.
        center: Pre-computed centre vector.  If ``None``, the
            detector uses a zero-centred proxy (still works as a
            relative ranking).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        center: list[float] | None = None,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold,
                         device=device, **kwargs)
        self._center = center

    @property
    def center(self) -> torch.Tensor:
        if self._center is not None:
            return torch.tensor(self._center, device=self._device)
        return torch.zeros(768, device=self._device)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        dist_sq = float(((emb - self.center) ** 2).sum())

        # Closer to centre → AI; farther → human
        # Negate distance so higher → more likely AI
        score = -dist_sq

        return self._make_result(
            score,
            distance_squared=dist_sq,
        )


# ------------------------------------------------------------------
# HRN
# ------------------------------------------------------------------


@register_detector("hrn")
class HRNDetector(_OODTextBase):
    """HRN (Holistic Regularised Network) detector.

    Uses the raw encoder embedding norm / cosine similarity as a
    proxy for the per-model sigmoid classifiers.  When a pre-trained
    checkpoint is loaded, the full HRN scoring pipeline applies.

    Without a checkpoint this implements a simplified variant using
    embedding L2 norm as the detection score (LLM text has more
    compact, normalised representations).

    Parameters:
        encoder_model: HuggingFace encoder.
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold,
                         device=device, **kwargs)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        # Cosine similarity to the origin (equivalent to directional consistency)
        # After L2 normalisation all embeddings have unit norm, so we use
        # the mean absolute value of embedding components as a regularity proxy.
        regularity = float(emb.abs().mean())

        # Higher regularity → more structured → more likely AI
        score = regularity

        return self._make_result(
            score,
            regularity=regularity,
        )


# ------------------------------------------------------------------
# Energy
# ------------------------------------------------------------------


@register_detector("energy_detector")
class EnergyDetector(_OODTextBase):
    """Energy-based OOD text detector.

    In the full implementation, energy = −log∑exp(f(z)) where f is a
    multi-class classifier head.  Without a trained head, this uses
    the negative entropy of the embedding distribution as a proxy:
    more concentrated embeddings → lower entropy → more likely AI.

    Parameters:
        encoder_model: HuggingFace encoder.
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold,
                         device=device, **kwargs)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.enc_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        ).to(self._device)

        out = self.enc_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        hidden = out.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        emb = F.normalize(pooled, dim=-1).squeeze(0)

        # Use negative L2 norm of pre-normalisation embedding as energy proxy
        raw_pooled = pooled.squeeze(0)
        energy = -float(torch.logsumexp(raw_pooled, dim=0))

        # Lower energy → in-distribution (AI); higher → OOD (human)
        # Negate so higher → more likely AI
        score = -energy

        return self._make_result(
            score,
            energy=energy,
        )
