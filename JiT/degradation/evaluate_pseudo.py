"""Evaluate a frozen estimator once against coarse paired-PSF fit labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_3dh_pseudo import PairedPseudoDataset
from inference import load_estimator
from train_estimator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    data = PairedPseudoDataset(args.project_root, args.manifest, args.labels, "val")
    model = load_estimator(args.checkpoint, args.device)
    metrics, rows = evaluate(model, DataLoader(data, batch_size=16, shuffle=False, num_workers=2), torch.device(args.device), 256, 0, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "pseudo_validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
