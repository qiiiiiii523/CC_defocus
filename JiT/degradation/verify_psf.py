"""Numerical and gradient checks for the CNSeg PSF and future A2/A3 loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from psf import LEVELS, gaussian_blur, reblur_mixture


def numpy_reference(image: np.ndarray, sigma: float, size: int) -> np.ndarray:
    radius = size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    work = image.astype(np.float32)
    padded = np.pad(work, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    horizontal = np.empty_like(work)
    for x in range(work.shape[1]):
        horizontal[:, x, :] = np.tensordot(padded[:, x : x + size, :], kernel, axes=([1], [0]))
    padded = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    blurred = np.empty_like(work)
    for y in range(work.shape[0]):
        blurred[y, :, :] = np.tensordot(padded[y : y + size, :, :], kernel, axes=([0], [0]))
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("RGB")).copy()[:64, :64]
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float().div_(255)
    for name, sigma, size in LEVELS:
        expected = numpy_reference(image, sigma, size)
        actual = gaussian_blur(tensor, sigma, size).mul(255).round().clamp(0, 255)
        actual_image = actual[0].permute(1, 2, 0).byte().numpy()
        max_error = int(np.abs(expected.astype(np.int16) - actual_image.astype(np.int16)).max())
        print(f"{name}: max_uint8_difference={max_error}")
        if max_error > 1:
            raise AssertionError(f"PSF differs from source implementation at {name}")

    restored = tensor.clone().requires_grad_()
    logits = torch.tensor([[0.2, 0.3, -0.4]], requires_grad=True)
    probabilities = logits.softmax(dim=1)
    reblurred = reblur_mixture(restored, probabilities)
    assert reblurred.shape == restored.shape
    loss = (reblurred - tensor * 0.75).square().mean()
    loss.backward()
    for name, gradient in (("restored", restored.grad), ("condition", logits.grad)):
        if gradient is None or not torch.isfinite(gradient).all() or float(gradient.abs().sum()) == 0:
            raise AssertionError(f"No valid gradient reaches {name}")
    print("PSF shape, source agreement and both gradients: PASS")


if __name__ == "__main__":
    main()
