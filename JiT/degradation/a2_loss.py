"""A2 loss bridge: freeze the local field, retain gradients to restoration."""

import torch

from .spatial_psf import psf_consistency_loss


def local_psf_loss(prediction_jit, observed_jit, probabilities, *, is_identity=None, epsilon=1e-3):
    """Reblur JiT's clean-endpoint prediction on the public RGB scale.

    Identity-replaced training samples keep the original supervised objective
    but do not receive a conflicting nonzero-blur consistency constraint.
    No clamp/detach is applied to the restoration, including out-of-range values.
    """
    if prediction_jit.shape != observed_jit.shape:
        raise ValueError("Prediction and observed blur must have identical BCHW shape")
    if is_identity is not None:
        if is_identity.shape != (prediction_jit.shape[0],):
            raise ValueError("is_identity must have shape [B]")
        eligible = ~is_identity.to(device=prediction_jit.device, dtype=torch.bool)
        if not bool(eligible.any()):
            return prediction_jit.float().sum() * 0.0
        prediction_jit = prediction_jit[eligible]
        observed_jit = observed_jit[eligible]
        probabilities = probabilities[eligible]

    # The frozen estimator has its own no_grad scope in Denoiser. This loss
    # must remain outside that scope and outside restoration BF16 arithmetic.
    with torch.amp.autocast(device_type=prediction_jit.device.type, enabled=False):
        return psf_consistency_loss(
            (prediction_jit.float() + 1.0) * 0.5,
            (observed_jit.float() + 1.0) * 0.5,
            probabilities.detach().float(),
            epsilon=epsilon,
        )
