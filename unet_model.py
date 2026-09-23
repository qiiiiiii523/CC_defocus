"""Small RGB-to-RGB U-Net used as a deterministic restoration baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions with ReLU, preserving spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class SmallUNet(nn.Module):
    """Four-level U-Net: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 3.

    The decoder uses bilinear interpolation to the exact skip-connection size,
    so the output retains the input height and width even for odd dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(3, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.encoder4 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.decoder4 = DoubleConv(256 + 128, 128)
        self.decoder3 = DoubleConv(128 + 64, 64)
        self.decoder2 = DoubleConv(64 + 32, 32)
        self.decoder1 = DoubleConv(32 + 16, 16)
        self.output = nn.Conv2d(16, 3, kernel_size=1)

    @staticmethod
    def _upsample_and_join(deeper: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deeper = F.interpolate(
            deeper, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((skip, deeper), dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, got shape {tuple(image.shape)}")
        if min(image.shape[-2:]) < 16:
            raise ValueError("Input height and width must each be at least 16 pixels")

        e1 = self.encoder1(image)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(self._upsample_and_join(center, e4))
        d3 = self.decoder3(self._upsample_and_join(d4, e3))
        d2 = self.decoder2(self._upsample_and_join(d3, e2))
        d1 = self.decoder1(self._upsample_and_join(d2, e1))
        return torch.sigmoid(self.output(d1))
