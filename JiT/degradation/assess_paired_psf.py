"""Check whether the three CNSeg Gaussian kernels explain paired 3DHistech blur."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from psf import LEVELS, gaussian_blur


def read_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32).div_(255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle, torch.no_grad():
        for line in handle:
            if args.max_samples and len(rows) >= args.max_samples:
                break
            record = json.loads(line)
            blur = read_image(args.project_root / record["blur_path"], device)
            clear = read_image(args.project_root / record["clear_path"], device)
            if clear.shape != blur.shape:
                raise ValueError(f"Shape mismatch: {record['sample_id']}")
            losses = [float((clear - blur).square().mean().item())]
            for _, sigma, kernel_size in LEVELS:
                synthetic = gaussian_blur(clear, sigma, kernel_size)
                losses.append(float((synthetic - blur).square().mean().item()))
            best_index = int(np.argmin(losses))
            rows.append({
                "sample_id": record["sample_id"],
                "best_setting": ("identity", "light", "medium", "heavy")[best_index],
                "mse_identity": losses[0],
                "mse_light": losses[1],
                "mse_medium": losses[2],
                "mse_heavy": losses[3],
                "best_over_identity_percent": round(100 * (1 - min(losses) / max(losses[0], 1e-12)), 3),
            })
    if not rows:
        raise ValueError("No records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "samples": len(rows),
        "best_settings": dict(Counter(row["best_setting"] for row in rows)),
        "median_improvement_over_identity_percent": float(np.median([row["best_over_identity_percent"] for row in rows])),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
