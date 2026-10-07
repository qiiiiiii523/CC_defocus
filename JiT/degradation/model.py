"""Small image-only estimator of the three CNSeg synthetic blur settings."""

from __future__ import annotations

import torch
from torch import nn

if __package__:
    from .psf import LEVELS
else:
    from psf import LEVELS


class BlurEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in (24, 48, 96, 128):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, len(LEVELS)))

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        if blur.ndim != 4 or blur.shape[1] != 3:
            raise ValueError("BlurEstimator expects RGB BCHW images")
        return self.classifier(self.features(blur))


def predicted_sigma(logits: torch.Tensor) -> torch.Tensor:
    sigma_values = logits.new_tensor([entry[1] for entry in LEVELS])
    return (logits.softmax(dim=1) * sigma_values[None, :]).sum(dim=1)
