"""Blur-image-only, spatial A1 estimator producing a three-kernel map."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class LocalBlurEstimator(nn.Module):
    """Predict [B,3,ceil(H/4),ceil(W/4)] logits from blurred RGB [0,1]."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = block(3, 32, stride=2)
        self.enc2 = block(32, 64, stride=2)
        self.enc3 = block(64, 96, stride=2)
        self.enc4 = block(96, 128, stride=2)
        self.dec3 = block(128 + 96, 96)
        self.dec2 = block(96 + 64, 64)
        self.head = nn.Conv2d(64, 3, 1)

    def forward(self, blur_01: torch.Tensor) -> torch.Tensor:
        if blur_01.ndim != 4 or blur_01.shape[1] != 3:
            raise ValueError("Expected blurred RGB [B,3,H,W]")
        x1 = self.stem(blur_01)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        up3 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        up3 = self.dec3(torch.cat((up3, x3), dim=1))
        up2 = F.interpolate(up3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec2(torch.cat((up2, x2), dim=1)))


@torch.no_grad()
def predict_local_probabilities(model: LocalBlurEstimator, blur_01: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(blur_01).all() or blur_01.min() < 0 or blur_01.max() > 1:
        raise ValueError("Expected finite blurred RGB in [0,1]")
    return model(blur_01).softmax(dim=1)
