"""Train the A1 blur estimator on continuous, confidence-aware 3DH proxy labels.

Use ``cv`` only on the fixed 2,000-pair training set to choose an epoch count.
Then run ``final`` once on all 2,000 pairs and ``evaluate`` once on the fixed
300-pair validation set. This script never treats fitted Gaussian strength as
ground-truth optical PSF.
"""

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import BlurEstimator, predicted_sigma
from psf import LEVELS


SIGMAS = (1.0, 2.0, 3.5)


def severity(sigma: float) -> int:
    return 0 if sigma < 1.5 else (1 if sigma < 2.75 else 2)


def soft_levels(sigma: torch.Tensor) -> torch.Tensor:
    """Interpolate a continuous proxy target over the existing three A1 bins."""
    left = (2.0 - sigma).clamp(0, 1)
    right = ((sigma - 2.0) / 1.5).clamp(0, 1)
    middle = 1 - left - right
    return torch.stack((left, middle, right), dim=1)


def read_samples(manifest: Path, labels: Path) -> list[dict]:
    records = [json.loads(line) for line in manifest.open("r", encoding="utf-8")]
    with labels.open("r", newline="", encoding="utf-8") as handle:
        labels_by_id = {row["sample_id"]: row for row in csv.DictReader(handle)}
    samples = []
    for record in records:
        if record.get("quality_status") != "pass":
            continue
        sample_id = record["sample_id"]
        if sample_id not in labels_by_id:
            raise ValueError(f"Missing proxy fit for {sample_id}")
        label = labels_by_id[sample_id]
        if label["group_id"] != record["group_id"]:
            raise ValueError(f"Group mismatch for {sample_id}")
        samples.append({
            "sample_id": sample_id,
            "group_id": record["group_id"],
            "blur_path": record["blur_path"],
            "sigma": float(label["target_sigma"]),
            "fit_sigma": float(label["fit_sigma"]),
            "gap": float(label["coarse_gap_relative"]),
        })
    if len(samples) != len(labels_by_id) or len({r["sample_id"] for r in samples}) != len(samples):
        raise ValueError("Manifest and fitted labels are not one-to-one")
    return samples


class BlurPairs(Dataset):
    def __init__(self, root: Path, samples: list[dict], augment: bool):
        self.root, self.samples, self.augment = root, samples, augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        with Image.open(self.root / row["blur_path"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        image = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
        return image, torch.tensor(row["sigma"], dtype=torch.float32), torch.tensor(row["gap"], dtype=torch.float32), row["sample_id"]


def train_one_epoch(model, loader, optimizer, device, bin_weights):
    model.train()
    loss_sum = 0.0
    seen = 0
    for image, sigma, gap, _ in loader:
        image, sigma, gap = image.to(device), sigma.to(device), gap.to(device)
        logits = model(image)
        estimate = predicted_sigma(logits)
        regression = F.huber_loss(estimate, sigma, delta=0.35, reduction="none")
        soft_ce = -(soft_levels(sigma) * F.log_softmax(logits, dim=1)).sum(dim=1)
        confidence = (gap / 0.05).clamp(0.25, 1.0)
        bins = torch.where(sigma < 1.5, 0, torch.where(sigma < 2.75, 1, 2))
        weight = confidence * bin_weights[bins]
        loss = ((regression + 0.25 * soft_ce) * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(sigma)
        seen += len(sigma)
    return loss_sum / seen


def evaluate(model, loader, device, constant_sigma: float):
    model.eval()
    rows = []
    with torch.no_grad():
        for image, sigma, gap, ids in loader:
            probabilities = model(image.to(device)).softmax(dim=1).cpu()
            estimates = (probabilities * torch.tensor(SIGMAS)).sum(dim=1)
            for index, sample_id in enumerate(ids):
                target = float(sigma[index])
                estimate = float(estimates[index])
                rows.append({
                    "sample_id": sample_id,
                    "target_sigma": target,
                    "fit_gap_relative": float(gap[index]),
                    "predicted_sigma": estimate,
                    "true_bin": severity(target),
                    "predicted_bin": severity(estimate),
                    **{f"p_{LEVELS[j][0]}": float(probabilities[index, j]) for j in range(3)},
                })
    errors = np.asarray([abs(r["predicted_sigma"] - r["target_sigma"]) for r in rows])
    constant_errors = np.asarray([abs(constant_sigma - r["target_sigma"]) for r in rows])
    categories = np.asarray([r["true_bin"] for r in rows])
    per_bin = {LEVELS[b][0]: {"n": int((categories == b).sum()),
                                  "mae": float(errors[categories == b].mean()) if (categories == b).any() else None,
                                  "constant_mae": float(constant_errors[categories == b].mean()) if (categories == b).any() else None}
               for b in range(3)}
    observed = [b for b in range(3) if (categories == b).any()]
    confusion = [[sum(r["true_bin"] == i and r["predicted_bin"] == j for r in rows) for j in range(3)] for i in range(3)]
    return {
        "samples": len(rows),
        "sigma_mae": float(errors.mean()),
        "constant_sigma_mae": float(constant_errors.mean()),
        "balanced_sigma_mae": float(np.mean([per_bin[LEVELS[b][0]]["mae"] for b in observed])),
        "constant_balanced_sigma_mae": float(np.mean([per_bin[LEVELS[b][0]]["constant_mae"] for b in observed])),
        "bin_accuracy": float(np.mean([r["true_bin"] == r["predicted_bin"] for r in rows])),
        "predicted_counts": dict(Counter(LEVELS[r["predicted_bin"]][0] for r in rows)),
        "per_bin": per_bin,
        "confusion_true_rows_predicted_columns": confusion,
    }, rows


def weights_for(samples: list[dict], device: torch.device):
    counts = Counter(severity(r["sigma"]) for r in samples)
    weights = np.asarray([1 / np.sqrt(max(counts[b], 1)) for b in range(3)], dtype=np.float32)
    weights /= weights.mean()
    return torch.from_numpy(weights).to(device)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(root, samples, args, augment):
    return DataLoader(BlurPairs(root, samples, augment), batch_size=args.batch_size,
                      shuffle=augment, num_workers=args.num_workers, pin_memory=args.device == "cuda")


def train_fold(root, train_rows, val_rows, args, fold_name):
    seed_all(args.seed)
    device = torch.device(args.device)
    model = BlurEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = loader(root, train_rows, args, True)
    val_loader = loader(root, val_rows, args, False)
    constant = float(np.median([r["sigma"] for r in train_rows]))
    bin_weights = weights_for(train_rows, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, bin_weights)
        metrics, _ = evaluate(model, val_loader, device, constant)
        row = {"fold": fold_name, "epoch": epoch, "train_loss": train_loss, "constant_sigma": constant, **metrics}
        print(json.dumps(row), flush=True)
        history.append(row)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cv", "final", "evaluate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--constant-sigma", type=float)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.manifest, args.labels)
    if args.mode == "cv":
        groups = sorted({r["group_id"] for r in samples})
        if len(groups) < 2:
            raise ValueError("CV needs at least two source-image groups")
        histories = []
        for group in groups:
            train_rows = [r for r in samples if r["group_id"] != group]
            val_rows = [r for r in samples if r["group_id"] == group]
            histories.extend(train_fold(root, train_rows, val_rows, args, group))
        ranked = []
        for epoch in range(1, args.epochs + 1):
            current = [r for r in histories if r["epoch"] == epoch]
            ranked.append((float(np.mean([r["balanced_sigma_mae"] / r["constant_balanced_sigma_mae"] for r in current])), epoch))
        score, chosen_epoch = min(ranked)
        result = {"selection": "training-only leave-one-group-out mean balanced MAE / train-median baseline", "chosen_epoch": chosen_epoch,
                  "chosen_relative_score": score, "folds": groups, "history": histories}
        (args.output_dir / "cv_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "history"}), flush=True)
    elif args.mode == "final":
        if args.checkpoint:
            raise ValueError("final trains from scratch; do not pass a checkpoint")
        seed_all(args.seed)
        device = torch.device(args.device)
        model = BlurEstimator().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = loader(root, samples, args, True)
        bin_weights = weights_for(samples, device)
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, bin_weights)
            print(json.dumps({"epoch": epoch, "train_loss": loss}), flush=True)
        checkpoint = args.output_dir / "estimator-continuous.pt"
        torch.save({"model": model.state_dict(), "epoch": args.epochs, "levels": LEVELS,
                    "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    "proxy_labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                    "source": "continuous Gaussian proxy, confidence weighted; fixed validation excluded from training and selection"}, checkpoint)
        result = {"checkpoint": str(checkpoint), "train_samples": len(samples), "train_median_sigma": float(np.median([r["sigma"] for r in samples]))}
        (args.output_dir / "final_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    else:
        if args.checkpoint is None or args.constant_sigma is None:
            raise ValueError("evaluate needs --checkpoint and --constant-sigma from fixed training data")
        device = torch.device(args.device)
        model = BlurEstimator().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
        metrics, predictions = evaluate(model, loader(root, samples, args, False), device, args.constant_sigma)
        (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
