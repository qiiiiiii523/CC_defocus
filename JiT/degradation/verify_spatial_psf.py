"""Check local A2/A3 reblurring, fixed baseline, boundaries, and gradients."""

from __future__ import annotations

import torch

from psf import reblur_mixture
from spatial_psf import psf_consistency_loss, reblur_spatial_mixture, spatial_probabilities


def main() -> None:
    torch.manual_seed(20261010)
    clear = torch.rand(2, 3, 64, 80)
    fixed = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    fixed_map = fixed[:, :, None, None].expand(-1, -1, 8, 10)
    torch.testing.assert_close(reblur_spatial_mixture(clear, fixed_map), reblur_mixture(clear, fixed), atol=1e-6, rtol=1e-6)

    logits = torch.randn(2, 3, 8, 10, requires_grad=True)
    local = logits.softmax(dim=1)
    resized = spatial_probabilities(local, clear.shape[-2:])
    assert resized.shape == (2, 3, 64, 80)
    torch.testing.assert_close(resized.sum(dim=1), torch.ones(2, 64, 80), atol=1e-6, rtol=1e-6)
    assert float(resized[:, 0].std()) > 0.01, "The spatial map must actually vary"

    observed = reblur_spatial_mixture(clear, local).detach()
    fixed_light = torch.zeros_like(local)
    fixed_light[:, 0] = 1
    correct_loss = psf_consistency_loss(clear, observed, local)
    fixed_loss = psf_consistency_loss(clear, observed, fixed_light)
    assert float(correct_loss) < 1e-6 and float(fixed_loss) > float(correct_loss) + 1e-4

    restored = (clear + 0.03).clamp(0, 1).detach().requires_grad_(True)
    loss = psf_consistency_loss(restored, observed, local)
    loss.backward()
    assert restored.grad is not None and torch.isfinite(restored.grad).all() and restored.grad.abs().sum() > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    print("Local PSF / A2-A3 consistency: fixed-map agreement, spatial variation, baseline, shape, and both gradients PASS")


if __name__ == "__main__":
    main()
