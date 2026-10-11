"""Spatially varying three-kernel PSF mixture for local A1 and A2/A3.

At each output location (y, x), the three probabilities select a mixture of
the *full-image* Gaussian convolutions. This is a local output-PSF model, not
an optical measurement or a patchwise copy of a global severity score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .psf import LEVELS, gaussian_blur
except ImportError:
    from psf import LEVELS, gaussian_blur


def spatial_probabilities(probabilities: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize nonnegative [B,3,h,w] mixture weights to image resolution."""
    if probabilities.ndim != 4 or probabilities.shape[1] != len(LEVELS):
        raise ValueError("Expected local probabilities [B,3,h,w]")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("Local probabilities must be finite and nonnegative")
    if bool((probabilities.sum(dim=1) <= 0).any()):
        raise ValueError("Local probabilities must have positive channel sum")
    if probabilities.shape[-2:] != size:
        probabilities = F.interpolate(probabilities, size=size, mode="bilinear", align_corners=False)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def reblur_spatial_mixture(clear_01: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Apply the local output-PSF field to an RGB image in [0,1]."""
    if clear_01.ndim != 4 or clear_01.shape[1] != 3:
        raise ValueError("Expected RGB image [B,3,H,W]")
    if clear_01.shape[0] != probabilities.shape[0]:
        raise ValueError("Image and PSF map batch sizes differ")
    weights = spatial_probabilities(probabilities, clear_01.shape[-2:])
    reblurred = torch.zeros_like(clear_01)
    for level, (_, sigma, kernel_size) in enumerate(LEVELS):
        reblurred = reblurred + weights[:, level : level + 1] * gaussian_blur(clear_01, sigma, kernel_size)
    return reblurred


def psf_consistency_loss(
    restored_01: torch.Tensor,
    observed_blur_01: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier loss between reblurred restoration and observed blur.

    The caller decides whether to freeze/detach estimated PSF probabilities.
    A2 supplies the map only here; A3 also supplies the same map to JiT's
    condition adapter. Inference must derive the map from observed blur only.
    """
    if restored_01.shape != observed_blur_01.shape:
        raise ValueError("Restored and observed images must have identical BCHW shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reconstruction = reblur_spatial_mixture(restored_01, probabilities)
    residual = torch.sqrt((reconstruction - observed_blur_01).square() + epsilon**2) - epsilon
    if valid_mask is None:
        return residual.mean()
    if valid_mask.shape != (restored_01.shape[0], 1, *restored_01.shape[-2:]):
        raise ValueError("valid_mask must have shape [B,1,H,W]")
    if not torch.isfinite(valid_mask).all() or bool((valid_mask < 0).any()):
        raise ValueError("valid_mask must be finite and nonnegative")
    denominator = valid_mask.sum() * restored_01.shape[1]
    if denominator <= 0:
        raise ValueError("valid_mask contains no valid pixels")
    return (residual * valid_mask).sum() / denominator
