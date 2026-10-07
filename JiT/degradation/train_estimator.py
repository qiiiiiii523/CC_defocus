"""Train C's lightweight blur estimator on the fixed CNSeg synthetic split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CNSegSigmaDataset
from data_3dh import PairedClearSyntheticDataset
from data_3dh_pseudo import PairedPseudoDataset
from model import BlurEstimator, predicted_sigma
from psf import LEVELS, blur_by_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--data-kind", choices=("cnseg", "paired-clear", "paired-pseudo"), default="cnseg")
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--val-labels", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--selection", choices=("accuracy", "accuracy_then_sigma"), default="accuracy")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Nonzero only for smoke tests")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Nonzero only for smoke tests")
    return parser.parse_args()


def make_blurred_crop(
    clear: torch.Tensor, level_ids: torch.Tensor, crop_size: int, training: bool
) -> torch.Tensor:
    # Blur the whole source image first; cropping before convolution would
    # create artificial PSF boundaries inside the original image.
    blurred = blur_by_level(clear, level_ids).clamp(0, 1).mul(255).round().div(255)
    height, width = blurred.shape[-2:]
    if crop_size > min(height, width):
        raise ValueError(f"crop_size={crop_size} exceeds image size {(height, width)}")
    crops = []
    for image in blurred:
        top = int(torch.randint(height - crop_size + 1, ()).item()) if training else (height - crop_size) // 2
        left = int(torch.randint(width - crop_size + 1, ()).item()) if training else (width - crop_size) // 2
        crops.append(image[:, top : top + crop_size, left : left + crop_size])
    return torch.stack(crops)


def evaluate(
    model: BlurEstimator,
    loader: DataLoader,
    device: torch.device,
    crop_size: int,
    max_batches: int,
    synthetic: bool,
) -> tuple[dict, list[dict]]:
    model.eval()
    counts = Counter()
    confusion = [[0] * len(LEVELS) for _ in LEVELS]
    rows: list[dict] = []
    with torch.no_grad():
        for batch_no, (clear, labels, sample_ids) in enumerate(loader):
            if max_batches and batch_no >= max_batches:
                break
            clear = clear.to(device)
            labels = labels.to(device)
            blur = make_blurred_crop(clear, labels, crop_size, training=False) if synthetic else clear
            logits = model(blur)
            predictions = logits.argmax(dim=1)
            probabilities = logits.softmax(dim=1)
            sigma_predictions = predicted_sigma(logits)
            sigma_truth = logits.new_tensor([row[1] for row in LEVELS])[labels]
            counts["samples"] += len(labels)
            counts["correct"] += int((predictions == labels).sum().item())
            counts["sigma_error"] += float((sigma_predictions - sigma_truth).abs().sum().item())
            counts["fixed_medium_error"] += float((sigma_truth - LEVELS[1][1]).abs().sum().item())
            counts["fixed_light_error"] += float((sigma_truth - LEVELS[0][1]).abs().sum().item())
            counts["loss"] += float(F.cross_entropy(logits, labels, reduction="sum").item())
            for index, sample_id in enumerate(sample_ids):
                truth = int(labels[index].item())
                guess = int(predictions[index].item())
                confusion[truth][guess] += 1
                row = {
                    "sample_id": sample_id,
                    "true_severity": LEVELS[truth][0],
                    "true_sigma": LEVELS[truth][1],
                    "predicted_severity": LEVELS[guess][0],
                    "predicted_sigma": round(float(sigma_predictions[index].item()), 6),
                }
                row.update({f"p_{name}": round(float(probabilities[index, i].item()), 6) for i, (name, _, _) in enumerate(LEVELS)})
                rows.append(row)
    if not counts["samples"]:
        raise ValueError("Validation loader produced no samples")
    size = counts["samples"]
    metrics = {
        "samples": size,
        "cross_entropy": counts["loss"] / size,
        "accuracy": counts["correct"] / size,
        "sigma_mae": counts["sigma_error"] / size,
        "fixed_medium_accuracy": sum(1 for row in rows if row["true_severity"] == "medium") / size,
        "fixed_medium_sigma_mae": counts["fixed_medium_error"] / size,
        "fixed_light_accuracy": sum(1 for row in rows if row["true_severity"] == "light") / size,
        "fixed_light_sigma_mae": counts["fixed_light_error"] / size,
        "confusion_matrix_true_rows_predicted_columns": confusion,
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 1:
        raise ValueError("epochs, batch-size and crop-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is assigned")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    root = args.project_root.resolve()
    manifest = args.manifest.resolve()
    val_manifest = args.val_manifest.resolve() if args.val_manifest is not None else manifest
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = args.data_kind != "paired-pseudo"
    if args.data_kind == "paired-pseudo":
        if args.train_labels is None or args.val_labels is None:
            raise ValueError("paired-pseudo requires --train-labels and --val-labels")
        train_data = PairedPseudoDataset(root, manifest, args.train_labels, "train")
        val_data = PairedPseudoDataset(root, val_manifest, args.val_labels, "val")
    else:
        dataset_class = CNSegSigmaDataset if args.data_kind == "cnseg" else PairedClearSyntheticDataset
        train_data = dataset_class(root, manifest, "train")
        val_data = dataset_class(root, val_manifest, "val")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = BlurEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    partial = bool(args.max_train_batches or args.max_val_batches)
    history = []
    best_accuracy = -1.0
    best_sigma_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_no, (clear, labels, _) in enumerate(train_loader):
            if args.max_train_batches and batch_no >= args.max_train_batches:
                break
            clear = clear.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                blur = make_blurred_crop(clear, labels, args.crop_size, training=True) if synthetic else clear
            optimizer.zero_grad(set_to_none=True)
            logits = model(blur)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
        if not seen:
            raise ValueError("Training loader produced no samples")
        metrics, predictions = evaluate(model, val_loader, device, args.crop_size, args.max_val_batches, synthetic)
        metrics.update({"epoch": epoch, "train_loss": loss_sum / seen, "train_samples_seen": seen})
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        better = metrics["accuracy"] > best_accuracy
        if args.selection == "accuracy_then_sigma" and metrics["accuracy"] == best_accuracy:
            better = metrics["sigma_mae"] < best_sigma_mae
        if better:
            best_accuracy = metrics["accuracy"]
            best_sigma_mae = metrics["sigma_mae"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "levels": LEVELS,
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "val_manifest_sha256": hashlib.sha256(val_manifest.read_bytes()).hexdigest(),
                    "train_labels_sha256": hashlib.sha256(args.train_labels.read_bytes()).hexdigest() if args.train_labels else None,
                    "val_labels_sha256": hashlib.sha256(args.val_labels.read_bytes()).hexdigest() if args.val_labels else None,
                    "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "validation": metrics,
                },
                output_dir / "estimator-best.pt",
            )
            with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
    summary = {
        "status": "SMOKE_TEST_ONLY" if partial else "FULL_VALIDATION",
        "device": str(device),
        "data_kind": args.data_kind,
        "train_manifest_records": len(train_data),
        "val_manifest_records": len(val_data),
        "history": history,
        "best_accuracy": best_accuracy,
        "best_sigma_mae_at_selected_accuracy": best_sigma_mae,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
