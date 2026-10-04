"""Pixel supervision for JiT's sampled clean-endpoint prediction."""

import math

import torch


def charbonnier_loss(prediction, target, *, eps=1e-3):
    """Mean smooth L1 error, measured on the public RGB [0, 1] scale.

    Inputs are JiT images on the [-1, 1] scale. The affine range conversion
    is applied to the difference without clamping the prediction, preserving
    gradients for out-of-range predictions. Arithmetic is explicitly FP32.
    This loss has no flow-time weighting.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Charbonnier eps must be finite and positive")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")
    difference = (prediction.float() - target.float()) * 0.5
    return torch.sqrt(difference.square() + eps ** 2).mean()
