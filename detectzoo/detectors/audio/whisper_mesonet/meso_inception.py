"""MesoInception-4 backbone + WhisperMesoNet composition.

Mirrors

    https://github.com/piotrkawa/deepfake-whisper-features/
        blob/main/src/models/meso_net.py
    https://github.com/piotrkawa/deepfake-whisper-features/
        blob/main/src/models/whisper_meso_net.py

Module names (including the upstream typo ``Incption*``) are preserved so
that the pretrained ``ckpt.pth`` published by the authors loads cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from detectzoo.detectors.audio.whisper_mesonet.whisper_encoder import (
    Whisper,
    log_mel_spectrogram,
    tiny_en_dims,
)


class MesoInception4(nn.Module):
    """MesoInception-4 classifier (Afchar et al., 2018) for spectrogram input.

    Parameters
    ----------
    num_classes
        Output logits — upstream uses 1 (binary with BCE-with-logits, higher
        sigmoid ⇒ bonafide).
    input_channels
        Number of channels in the input 4-D tensor (Whisper-only == 1).
    fc1_dim
        Size of the adaptive-pool bottleneck before ``fc1`` (upstream == 1024).
    """

    def __init__(
        self,
        *,
        num_classes: int = 1,
        input_channels: int = 1,
        fc1_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.fc1_dim = fc1_dim
        self.num_classes = num_classes

        # --- Inception layer 1 (note the intentional upstream typo "Incption")
        self.Incption1_conv1 = nn.Conv2d(input_channels, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(input_channels, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # --- Inception layer 2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # --- Post-inception convolutional tower
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        # --- Classifier head
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.fc1_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

    # ------------------------------------------------------------------
    # Inception blocks — concatenate dilated-conv branches
    # ------------------------------------------------------------------
    def _inception1(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Incption1_conv1(x)
        x2 = self.Incption1_conv2_2(self.Incption1_conv2_1(x))
        x3 = self.Incption1_conv3_2(self.Incption1_conv3_1(x))
        x4 = self.Incption1_conv4_2(self.Incption1_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        return y

    def _inception2(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Incption2_conv1(x)
        x2 = self.Incption2_conv2_2(self.Incption2_conv2_1(x))
        x3 = self.Incption2_conv3_2(self.Incption2_conv3_1(x))
        x4 = self.Incption2_conv4_2(self.Incption2_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        return y

    # ------------------------------------------------------------------
    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self._inception1(x)
        x = self._inception2(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compute_embedding(x)


class WhisperMesoNet(MesoInception4):
    """Whisper-MesoNet classifier (Kawa et al., Interspeech 2023).

    Pipeline
    --------
    waveform (B, T=480_000 @16 kHz)
        → log-mel spectrogram (B, 80, 3000)
        → Whisper tiny.en encoder (B, 1500, 384)
        → permute + unsqueeze + tile ⇒ (B, 1, 384, 3000) image tensor
        → MesoInception-4 ⇒ 1 logit  (sigmoid ≈ P[bonafide])
    """

    def __init__(
        self,
        *,
        freeze_encoder: bool = True,
        input_channels: int = 1,
        fc1_dim: int = 1024,
        num_classes: int = 1,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            fc1_dim=fc1_dim,
        )
        self.whisper_model = Whisper(tiny_en_dims())
        if freeze_encoder:
            for p in self.whisper_model.parameters():
                p.requires_grad = False

    def compute_whisper_features(self, x: torch.Tensor) -> torch.Tensor:
        specs = torch.stack([log_mel_spectrogram(sample) for sample in x])
        x = self.whisper_model(specs)          # (B, 1500, 384)
        x = x.permute(0, 2, 1)                  # (B, 384, 1500)
        x = x.unsqueeze(1)                      # (B, 1, 384, 1500)
        x = x.repeat((1, 1, 1, 2))              # (B, 1, 384, 3000)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compute_whisper_features(x)
        return self._compute_embedding(x)
