"""Check local A1/A3 token alignment and shared field with A2/A3 reblur."""

from __future__ import annotations

import torch

from local_condition import LocalDegradationCondition, predict_local_from_jit_blur
from local_model import LocalBlurEstimator
from spatial_psf import psf_consistency_loss, reblur_spatial_mixture


def main() -> None:
    torch.manual_seed(20261010)
    estimator = LocalBlurEstimator().eval().requires_grad_(False)
    blur_jit = torch.rand(2, 3, 64, 80) * 2 - 1
    probabilities = predict_local_from_jit_blur(blur_jit, estimator)
    assert probabilities.shape == (2, 3, 16, 20)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2, 16, 20), atol=1e-6, rtol=1e-6)
    adapter = LocalDegradationCondition(96)
    condition = adapter(probabilities, (4, 5))
    assert condition.shape == (2, 20, 96)
    torch.testing.assert_close(condition, torch.zeros_like(condition))
    condition.square().mean().add(condition.mean()).backward()
    assert adapter.gate.grad is not None and torch.isfinite(adapter.gate.grad)
    restored = torch.rand(2, 3, 64, 80, requires_grad=True)
    observed = reblur_spatial_mixture(restored.detach(), probabilities).detach()
    loss = psf_consistency_loss(restored, observed * 0.9, probabilities.detach())
    loss.backward()
    assert restored.grad is not None and restored.grad.abs().sum() > 0
    print("Local A1/A3 token condition and A2/A3 shared PSF field: PASS")


if __name__ == "__main__":
    main()
