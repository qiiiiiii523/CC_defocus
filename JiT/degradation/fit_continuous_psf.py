"""Fit a continuous Gaussian blur strength to fixed paired 3DHistech images.

The fitted strength is a *proxy*, not a measured optical PSF. Keep train and
fixed validation outputs separate; never use the latter for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from psf import gaussian_blur


SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
COARSE = (0.0, 1.0, 2.0, 3.5)


def read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32).div_(255)


def blur_at(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return image
    kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
    return gaussian_blur(image, sigma, kernel_size)


def fit_pair(clear: torch.Tensor, blurred: torch.Tensor) -> dict[str, float | str]:
    if clear.shape != blurred.shape:
        raise ValueError(f"Shape mismatch: {tuple(clear.shape)} vs {tuple(blurred.shape)}")
    losses = {sigma: float((blur_at(clear, sigma) - blurred).square().mean().item()) for sigma in SIGMAS}
    best_sigma = min(SIGMAS, key=lambda sigma: losses[sigma])
    coarse_sorted = sorted(COARSE, key=lambda sigma: losses[sigma])
    coarse_best, coarse_second = coarse_sorted[:2]
    coarse_gap = (losses[coarse_second] - losses[coarse_best]) / max(losses[coarse_second], 1e-12)
    return {
        "fit_sigma": best_sigma,
        "target_sigma": float(np.clip(best_sigma, 1.0, 3.5)),
        "best_setting": "identity" if coarse_best == 0 else {1.0: "light", 2.0: "medium", 3.5: "heavy"}[coarse_best],
        "mse_best": losses[best_sigma],
        "mse_identity": losses[0.0],
        "coarse_gap_relative": coarse_gap,
        "ambiguous_coarse_2pct": int(coarse_gap < 0.02),
        "fit_at_grid_edge": int(best_sigma == SIGMAS[-1]),
        "improvement_over_identity": (losses[0.0] - losses[best_sigma]) / max(losses[0.0], 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    root = args.project_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rows: list[dict] = []
    with args.manifest.open("r", encoding="utf-8") as handle, torch.no_grad():
        for index, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("quality_status") != "pass":
                continue
            clear = read_rgb(root / record["clear_path"], device)
            blurred = read_rgb(root / record["blur_path"], device)
            rows.append({"sample_id": record["sample_id"], "group_id": record["group_id"], **fit_pair(clear, blurred)})
            if index % 100 == 0:
                print(json.dumps({"fitted": len(rows), "manifest_lines": index}), flush=True)
    if not rows:
        raise ValueError("No usable manifest rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "samples": len(rows),
        "ambiguous_coarse_2pct": sum(int(row["ambiguous_coarse_2pct"]) for row in rows),
        "fit_at_grid_edge": sum(int(row["fit_at_grid_edge"]) for row in rows),
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
