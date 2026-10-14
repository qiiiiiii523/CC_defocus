"""Differentiable Gaussian PSFs matching the CNSeg synthetic blur settings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Order is shared by the manifest loader, estimator output, and PSF mixture.
LEVELS = (("light", 1.0, 7), ("medium", 2.0, 13), ("heavy", 3.5, 23))


def gaussian_blur(image: torch.Tensor, sigma: torch.Tensor | float, kernel_size: int) -> torch.Tensor:
    """Blur BCHW RGB images with a normalized, per-image, separable PSF.

    Reflect padding agrees with ``numpy.pad(..., mode='reflect')`` used by the
    original CNSeg preparation code. The function retains gradients to both
    ``image`` and a tensor-valued ``sigma``.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected RGB BCHW tensor, got {tuple(image.shape)}")
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be positive and odd")
    radius = kernel_size // 2
    if min(image.shape[-2:]) <= radius:
        raise ValueError("Image is too small for reflect padding")
    batch, channels, height, width = image.shape
    sigma_tensor = torch.as_tensor(sigma, device=image.device, dtype=image.dtype)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.expand(batch)
    if sigma_tensor.shape != (batch,) or bool(torch.any(sigma_tensor <= 0)):
        raise ValueError("sigma must contain one positive value per image")

    positions = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(positions[None, :] ** 2) / (2 * sigma_tensor[:, None] ** 2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    repeated = kernel.repeat_interleave(channels, dim=0)
    work = image.reshape(1, batch * channels, height, width)
    work = F.conv2d(
        F.pad(work, (radius, radius, 0, 0), mode="reflect"),
        repeated[:, None, None, :],
        groups=batch * channels,
    )
    work = F.conv2d(
        F.pad(work, (0, 0, radius, radius), mode="reflect"),
        repeated[:, None, :, None],
        groups=batch * channels,
    )
    return work.reshape(batch, channels, height, width)


def blur_by_level(image: torch.Tensor, level_ids: torch.Tensor) -> torch.Tensor:
    """Apply each manifest's discrete Gaussian setting to its source image."""
    if level_ids.shape != (image.shape[0],):
        raise ValueError("level_ids must have one value per image")
    result = torch.empty_like(image)
    for level_id, (_, sigma, kernel_size) in enumerate(LEVELS):
        indices = torch.nonzero(level_ids == level_id, as_tuple=True)[0]
        if indices.numel():
            result[indices] = gaussian_blur(image[indices], sigma, kernel_size)
    if bool(torch.any((level_ids < 0) | (level_ids >= len(LEVELS)))):
        raise ValueError("Unknown blur level")
    return result


def reblur_mixture(image: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Reblur with predicted preset-kernel mixture weights for A2/A3.

    No ground-truth severity is used here. Gradients reach the restored image
    and the predicted mixture weights.
    """
    if probabilities.shape != (image.shape[0], len(LEVELS)):
        raise ValueError("probabilities must have shape [batch, 3]")
    result = torch.zeros_like(image)
    for level_id, (_, sigma, kernel_size) in enumerate(LEVELS):
        result = result + probabilities[:, level_id, None, None, None] * gaussian_blur(
            image, sigma, kernel_size
        )
    return result
