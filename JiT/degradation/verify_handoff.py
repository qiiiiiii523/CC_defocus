"""Check the trained estimator's A1/A2 interfaces on one real blurred image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from condition import DegradationCondition, predict_from_jit_blur
from inference import load_estimator, predict_probabilities
from psf import reblur_mixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    estimator = load_estimator(args.checkpoint, args.device)
    with Image.open(args.image) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    blur_01 = torch.from_numpy(array).permute(2, 0, 1)[None].float().div_(255).to(args.device)
    blur_jit = blur_01 * 2 - 1
    with torch.no_grad():
        probabilities = predict_from_jit_blur(blur_jit, estimator)
        torch.testing.assert_close(probabilities, predict_probabilities(estimator, blur_01))
    assert probabilities.shape == (1, 3) and torch.isfinite(probabilities).all()
    adapter = DegradationCondition(hidden_size=768).to(args.device)
    condition = adapter(probabilities)
    torch.testing.assert_close(condition, torch.zeros_like(condition))
    restored = blur_01.detach().clone().requires_grad_(True)
    reblurred = reblur_mixture(restored, probabilities)
    assert reblurred.shape == restored.shape and torch.isfinite(reblurred).all()
    reblurred.square().mean().backward()
    if restored.grad is None or not torch.isfinite(restored.grad).all():
        raise AssertionError("No finite reblur gradient to restored image")
    print("Trained checkpoint -> predicted A1 condition -> A2/A3 reblur: PASS")


if __name__ == "__main__":
    main()
