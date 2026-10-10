"""Spatially varying synthetic Gaussian fields with known per-pixel PSF mix."""

from __future__ import annotations

import torch


def sample_spatial_weights(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return smooth, genuinely varying [B,3,h,w] probability fields.

    Three random severity centres create soft Voronoi-like regions; no global
    label is broadcast over the image. A small random bias varies their areas.
    The same three Gaussian components are used by ``spatial_psf``.
    """
    if batch < 1 or min(height, width) < 4:
        raise ValueError("Invalid spatial field dimensions")
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((yy, xx), dim=0)[None, None]
    centres = torch.rand(batch, 3, 2, 1, 1, device=device, generator=generator) * 1.6 - 0.8
    radii = 0.5 + torch.rand(batch, 3, 1, 1, device=device, generator=generator) * 0.25
    logits = -((coordinates - centres).square().sum(dim=2)) / (2 * radii.square())
    logits = logits + 0.3 * torch.randn(batch, 3, 1, 1, device=device, generator=generator)
    return (logits * 3).softmax(dim=1)
