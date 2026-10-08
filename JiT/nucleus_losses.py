"""Simple, mask based nucleus region and boundary losses.

The functions in this module deliberately use ordinary L1 reconstruction
errors.  They are intended for the K0 interface check and can be reused by a
training loop without depending on a particular restoration backbone.
"""

from __future__ import annotations

from typing import Dict

import torch


def _mask_3d(instance_mask: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """Return an instance mask with shape ``[B, H, W]``."""
    if instance_mask.ndim == 4:
        if instance_mask.shape[1] != 1:
            raise ValueError("instance_mask with four dimensions must have one channel")
        instance_mask = instance_mask[:, 0]
    if instance_mask.ndim != 3:
        raise ValueError("instance_mask must have shape [B,H,W] or [B,1,H,W]")
    if reference is not None and tuple(instance_mask.shape) != tuple(reference.shape[0:1] + reference.shape[2:]):
        raise ValueError("instance_mask spatial shape must match prediction")
    return instance_mask


def instance_boundary_mask(instance_mask: torch.Tensor) -> torch.Tensor:
    """Build a one-pixel inner boundary mask from integer instance labels.

    A foreground pixel is on the boundary when one of its four-neighbors has
    a different instance label, including background label zero.  For two
    touching instances, pixels from both instances are retained.
    """
    labels = _mask_3d(instance_mask)
    foreground = labels > 0
    boundary = torch.zeros_like(foreground, dtype=torch.bool)
    boundary[:, 1:, :] |= foreground[:, 1:, :] & (labels[:, 1:, :] != labels[:, :-1, :])
    boundary[:, :-1, :] |= foreground[:, :-1, :] & (labels[:, :-1, :] != labels[:, 1:, :])
    boundary[:, :, 1:] |= foreground[:, :, 1:] & (labels[:, :, 1:] != labels[:, :, :-1])
    boundary[:, :, :-1] |= foreground[:, :, :-1] & (labels[:, :, :-1] != labels[:, :, 1:])
    return boundary


def _pixel_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have the same shape [B,C,H,W]")
    return torch.abs(prediction - target).mean(dim=1)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    region_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over a boolean ``[B,H,W]`` region.

    Empty masks return a differentiable zero instead of NaN, which keeps a
    batch with no nuclei valid while making the absence explicit to callers.
    """
    error = _pixel_l1(prediction, target)
    mask = region_mask.to(device=error.device, dtype=torch.bool)
    if mask.shape != error.shape:
        raise ValueError("region_mask must have shape [B,H,W]")
    count = mask.sum()
    if int(count.item()) == 0:
        return prediction.sum() * 0.0
    return error[mask].mean()


def nucleus_region_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    instance_mask: torch.Tensor,
) -> torch.Tensor:
    """Ordinary L1 reconstruction loss inside all annotated nuclei."""
    labels = _mask_3d(instance_mask, prediction)
    return masked_l1_loss(prediction, target, labels > 0)


def nucleus_boundary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    instance_mask: torch.Tensor,
) -> torch.Tensor:
    """Ordinary L1 reconstruction loss on the one-pixel nucleus boundary."""
    labels = _mask_3d(instance_mask, prediction)
    return masked_l1_loss(prediction, target, instance_boundary_mask(labels))


def nucleus_supervision_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    instance_mask: torch.Tensor,
    *,
    region_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return region, boundary, and weighted total losses for K0."""
    if region_weight < 0 or boundary_weight < 0:
        raise ValueError("loss weights must be non-negative")
    region = nucleus_region_loss(prediction, target, instance_mask)
    boundary = nucleus_boundary_loss(prediction, target, instance_mask)
    total = region_weight * region + boundary_weight * boundary
    return {"total": total, "region": region, "boundary": boundary}

