"""Load C's trained CNSeg blur estimator and predict from a blurred RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from .model import BlurEstimator
    from .psf import LEVELS
else:
    from model import BlurEstimator
    from psf import LEVELS


def load_estimator(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> BlurEstimator:
    """Return a frozen estimator; the checkpoint contains weights and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(tuple(row) for row in checkpoint["levels"]) != LEVELS:
        raise ValueError("Checkpoint blur settings differ from this code")
    model = BlurEstimator()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def predict_probabilities(model: BlurEstimator, blur_01: torch.Tensor) -> torch.Tensor:
    """Predict [light, medium, heavy] probabilities from RGB BCHW in [0, 1]."""
    if blur_01.ndim != 4 or blur_01.shape[1] != 3:
        raise ValueError("Expected RGB BCHW tensor")
    if not torch.isfinite(blur_01).all() or blur_01.min() < 0 or blur_01.max() > 1:
        raise ValueError("Expected finite image values in [0, 1]")
    return model(blur_01.to(next(model.parameters()).device)).softmax(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model = load_estimator(args.checkpoint, args.device)
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    blur = torch.from_numpy(image).permute(2, 0, 1)[None].float().div_(255)
    probabilities = predict_probabilities(model, blur)
    output = {
        "image": str(args.image),
        "predicted_severity": LEVELS[int(probabilities.argmax(1).item())][0],
        "predicted_sigma": float((probabilities * probabilities.new_tensor([row[1] for row in LEVELS])).sum().item()),
        "probabilities": {name: float(probabilities[0, i].item()) for i, (name, _, _) in enumerate(LEVELS)},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
