"""Text Fluoroscopy — layer-wise KL divergence detector.

Reference:
    Yu et al., "Text Fluoroscopy: Detecting LLM-Generated Text through
    Intrinsic Features", EMNLP 2024.

The insight: when a transformer model processes human text vs.
machine text, the token-probability distributions at intermediate
layers diverge differently.  Text Fluoroscopy projects each layer's
hidden state to the vocabulary space and measures KL divergence
between intermediate layers and the final layer.  The layer with the
largest KL divergence carries the strongest detection signal.

This implementation provides a zero-shot variant that uses the
*maximum* inter-layer KL divergence as the detection score.  Machine
text tends to show *lower* max-KL (distributions converge faster)
compared to human text.
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


@register_detector("text_fluoroscopy")
class TextFluoroscopyDetector(BaseTextDetector):
    """Text Fluoroscopy detector — layer-wise KL divergence analysis.

    Computes the KL divergence between each intermediate layer's
    distribution and the final layer's distribution, then uses the
    maximum KL value as the detection score.  Human text tends to
    have *higher* max-KL divergence than machine text.

    Requires a causal LM that supports ``output_hidden_states=True``
    and has an ``lm_head`` (or equivalent) projection layer.

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

    def _get_lm_head(self) -> torch.nn.Module:
        """Resolve the LM head that projects hidden states to vocab logits."""
        model = self.model
        for attr in ("lm_head", "cls", "score"):
            if hasattr(model, attr):
                return getattr(model, attr)
        raise AttributeError(
            f"Cannot find an LM head on model {self.model_name}. "
            "Text Fluoroscopy requires a model with an `lm_head` attribute."
        )

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        outputs = self.model(**enc, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, T, D]

        lm_head = self._get_lm_head()
        n_layers = len(hidden_states) - 1  # exclude embedding layer

        last_logits = lm_head(hidden_states[-1])
        last_log_probs = F.log_softmax(last_logits, dim=-1)

        first_logits = lm_head(hidden_states[1])  # layer 1 (after embedding)
        first_log_probs = F.log_softmax(first_logits, dim=-1)

        kl_with_last = []
        kl_with_first = []
        for layer_idx in range(1, n_layers + 1):
            h = hidden_states[layer_idx]
            layer_logits = lm_head(h)
            layer_probs = F.softmax(layer_logits, dim=-1)

            # KL(layer || last) — averaged over tokens
            kl_last = F.kl_div(last_log_probs, layer_probs, reduction="none").sum(-1).mean()
            kl_with_last.append(float(kl_last))

            # KL(layer || first) — averaged over tokens
            kl_first = F.kl_div(first_log_probs, layer_probs, reduction="none").sum(-1).mean()
            kl_with_first.append(float(kl_first))

        kl_last_tensor = torch.tensor(kl_with_last)
        kl_first_tensor = torch.tensor(kl_with_first)

        # Combined divergence: product of KL with first and last
        combined = kl_last_tensor * kl_first_tensor
        max_combined, best_layer = combined.max(dim=0)

        # Human text → higher max KL; machine text → lower max KL
        # Negate so higher score → more likely AI
        score = -float(max_combined)

        return self._make_result(
            score,
            max_kl_combined=float(max_combined),
            best_layer=int(best_layer) + 1,
            max_kl_with_last=max(kl_with_last),
            max_kl_with_first=max(kl_with_first),
            n_layers=n_layers,
        )
