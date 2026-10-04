"""Export image-only degradation predictions for a fixed paired manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference import load_estimator, predict_probabilities
from psf import LEVELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model = load_estimator(args.checkpoint, args.device)
    sigma_values = [row[1] for row in LEVELS]
    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            blur_path = args.project_root / record["blur_path"]
            with Image.open(blur_path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
            blur = torch.from_numpy(image).permute(2, 0, 1)[None].float().div_(255)
            probabilities = predict_probabilities(model, blur)[0].cpu().tolist()
            prediction = int(np.argmax(probabilities))
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "blur_path": record["blur_path"],
                    "predicted_severity": LEVELS[prediction][0],
                    "predicted_sigma": round(sum(p * s for p, s in zip(probabilities, sigma_values)), 6),
                    **{f"p_{name}": round(probabilities[i], 6) for i, (name, _, _) in enumerate(LEVELS)},
                }
            )
    if not rows:
        raise ValueError("Manifest contains no samples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"samples": len(rows), "predicted_levels": dict(Counter(row["predicted_severity"] for row in rows)), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
