"""Check A1's predicted-condition shape and zero-initialized A0 behavior."""

from __future__ import annotations

import torch

from condition import DegradationCondition, predict_from_jit_blur
from model import BlurEstimator
from psf import reblur_mixture


def main() -> None:
    torch.manual_seed(20261004)
    estimator = BlurEstimator().eval()
    for parameter in estimator.parameters():
        parameter.requires_grad_(False)
    blur_jit = torch.rand(2, 3, 64, 64) * 2 - 1
    probabilities = predict_from_jit_blur(blur_jit, estimator)
    assert probabilities.shape == (2, 3)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    adapter = DegradationCondition(hidden_size=768)
    condition = adapter(probabilities)
    assert condition.shape == (2, 768)
    torch.testing.assert_close(condition, torch.zeros_like(condition))
    loss = (condition - 0.1).square().mean()
    loss.backward()
    if adapter.gate.grad is None or not torch.isfinite(adapter.gate.grad):
        raise AssertionError("A1 adapter gate has no valid gradient")
    restored = torch.rand(2, 3, 64, 64, requires_grad=True)
    reblurred = reblur_mixture(restored, probabilities)
    assert reblurred.shape == restored.shape
    print("Predicted A1 condition and A2/A3 PSF interface: PASS")


if __name__ == "__main__":
    main()
