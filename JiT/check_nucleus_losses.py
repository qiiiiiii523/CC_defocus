"""Numerical and gradient checks for the ordinary K0 nucleus losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nucleus_losses import instance_boundary_mask, nucleus_supervision_loss


def _loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return nucleus_supervision_loss(prediction, target, mask)["total"]


def _finite_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    index: tuple[int, int, int, int],
    epsilon: float = 1e-4,
) -> float:
    plus = prediction.detach().clone()
    minus = prediction.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    return float((_loss(plus, target, mask) - _loss(minus, target, mask)) / (2 * epsilon))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../results/loss_checks/nucleus_loss_checks.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Two touching instances plus one small instance exercise both ordinary
    # foreground boundaries and the boundary between adjacent nuclei.
    mask = torch.zeros((1, 7, 9), dtype=torch.int64)
    mask[0, 1:5, 1:4] = 1
    mask[0, 1:5, 4:7] = 2
    mask[0, 5:7, 7:9] = 3
    target = torch.zeros((1, 1, 7, 9), dtype=torch.float32)
    prediction = torch.full_like(target, 0.25, requires_grad=True)

    boundary = instance_boundary_mask(mask)
    foreground = mask > 0
    interior = foreground & ~boundary
    outside = ~foreground
    losses = nucleus_supervision_loss(prediction, target, mask)
    losses["total"].backward()
    gradient = prediction.grad.detach()

    # Zero-error check.
    zero_prediction = target.clone().requires_grad_(True)
    zero_losses = nucleus_supervision_loss(zero_prediction, target, mask)

    boundary_index = tuple(int(value) for value in torch.nonzero(boundary[0], as_tuple=False)[0])
    boundary_pixel = (0, 0, boundary_index[0], boundary_index[1])
    finite_difference = _finite_difference(prediction, target, mask, boundary_pixel)
    analytic_gradient = float(gradient[boundary_pixel])

    checks = {
        "zero_total_is_zero": abs(float(zero_losses["total"].detach())) < 1e-12,
        "region_count_positive": int(foreground.sum()) > 0,
        "boundary_count_positive": int(boundary.sum()) > 0,
        "interior_count_positive": int(interior.sum()) > 0,
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "boundary_gradient_nonzero": abs(analytic_gradient) > 0,
        "outside_gradient_zero": bool(torch.allclose(gradient[0, 0][outside[0]], torch.zeros_like(gradient[0, 0][outside[0]]))),
        "finite_difference_matches": abs(finite_difference - analytic_gradient) < 2e-3,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "mask_counts": {
            "foreground": int(foreground.sum()),
            "boundary": int(boundary.sum()),
            "interior": int(interior.sum()),
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "finite_difference": finite_difference,
        "analytic_gradient": analytic_gradient,
        "gradient_min": float(gradient.min()),
        "gradient_max": float(gradient.max()),
        "definition": "ordinary mean L1 reconstruction error over nucleus pixels and one-pixel inner instance boundaries",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
