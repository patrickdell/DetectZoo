"""Fairseq XLSR front-end + Mamba classifier stack (XLSR-Mamba).

Upstream layout: https://github.com/swagshaw/XLSR-Mamba/blob/main/model.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import fairseq
import torch
import torch.nn as nn

from detectzoo.detectors.audio.xlsr_mamba.mamba_blocks import MixerModel


@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class SSLModel(nn.Module):
    """Loads Meta XLSR-300M via Fairseq (``xlsr2_300m.pt``)."""

    def __init__(self, device: torch.device, cp_path: str) -> None:
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> tuple[torch.Tensor, Any]:
        if next(self.model.parameters()).device != input_data.device or next(
            self.model.parameters()
        ).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        out = self.model(input_tmp, mask=False, features_only=True)
        emb = out["x"]
        layerresult = out["layer_results"]
        return emb, layerresult


class XLSRMambaNet(nn.Module):
    """Same module layout / state-dict keys as upstream ``Model``."""

    def __init__(
        self,
        device: torch.device,
        xlsr_ckpt: str,
        emb_size: int = 144,
        num_encoders: int = 12,
    ) -> None:
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(device, xlsr_ckpt)
        self.LL = nn.Linear(1024, emb_size)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(d_model=emb_size, n_layer=num_encoders // 2)
        self.conformer = MixerModel(
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            ssm_cfg=self.config.ssm_cfg,
            rms_norm=self.config.rms_norm,
            residual_in_fp32=self.config.residual_in_fp32,
            fused_add_norm=self.config.fused_add_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ssl_feat, _ = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        return self.conformer(x)


def build_model(
    device: torch.device,
    xlsr_ckpt: str,
    emb_size: int = 144,
    num_encoders: int = 12,
) -> XLSRMambaNet:
    return XLSRMambaNet(device, xlsr_ckpt, emb_size, num_encoders)


def load_safetensors_weights(model: nn.Module, path: str) -> None:
    from safetensors.torch import load_file

    state = load_file(path)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
