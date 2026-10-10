"""Train and assess a blur-only local A1 estimator on spatially synthetic 3DH images.

Model selection uses leave-one-source-group-out folds *inside* fixed debug_train.
The fixed debug_val is evaluated once after the final epoch count is frozen.
Synthetic PSF fields are known labels; real 3DH PSF remains unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from local_model import LocalBlurEstimator
from local_synthetic import sample_spatial_weights
from model import BlurEstimator
from psf import LEVELS
from spatial_psf import reblur_spatial_mixture


def read_records(manifest: Path, expected_split: str) -> list[dict]:
    records = [json.loads(line) for line in manifest.open("r", encoding="utf-8")]
    if not records or any(r["split"] != expected_split or r.get("quality_status") != "pass" for r in records):
        raise ValueError(f"Manifest must contain only pass {expected_split} records: {manifest}")
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    return records


class ClearImages(Dataset):
    def __init__(self, root: Path, records: list[dict]) -> None:
        self.root, self.records = root, records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        with Image.open(self.root / row["clear_path"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        clear = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return clear, row["sample_id"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(root: Path, records: list[dict], args, training: bool):
    return DataLoader(ClearImages(root, records), batch_size=args.batch_size,
                      shuffle=training, num_workers=args.num_workers, pin_memory=args.device == "cuda")


def sigma_field(probabilities: torch.Tensor) -> torch.Tensor:
    sigmas = probabilities.new_tensor([r[1] for r in LEVELS])[None, :, None, None]
    return (probabilities * sigmas).sum(dim=1)


def map_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def synthetic_batch(clear: torch.Tensor, generator: torch.Generator):
    height, width = clear.shape[-2:]
    truth = sample_spatial_weights(len(clear), (height + 3) // 4, (width + 3) // 4,
                                   device=clear.device, generator=generator)
    with torch.no_grad():
        blurred = reblur_spatial_mixture(clear, truth).clamp(0, 1)
    return blurred, truth


def train_epoch(model, loader, optimizer, device, epoch_seed: int,
                reblur_weight: float = 0.0, edge_weight: float = 0.0):
    model.train()
    generator = map_generator(device, epoch_seed)
    total, count = 0.0, 0
    for clear, _ in loader:
        clear = clear.to(device, non_blocking=True)
        blurred, truth = synthetic_batch(clear, generator)
        logits = model(blurred)
        if logits.shape != truth.shape:
            raise ValueError(f"Model map {tuple(logits.shape)} != target {tuple(truth.shape)}")
        predicted = logits.softmax(dim=1)
        pixel_ce = -(truth * F.log_softmax(logits, dim=1)).sum(dim=1)
        pixel_severity = F.smooth_l1_loss(sigma_field(predicted), sigma_field(truth),
                                          beta=0.25, reduction="none")
        if edge_weight:
            horizontal = (clear[..., 1:] - clear[..., :-1]).abs().mean(dim=1, keepdim=True)
            vertical = (clear[..., 1:, :] - clear[..., :-1, :]).abs().mean(dim=1, keepdim=True)
            energy = F.pad(horizontal, (0, 1)) + F.pad(vertical, (0, 0, 0, 1))
            energy = F.adaptive_avg_pool2d(energy, truth.shape[-2:])[:, 0]
            importance = 1 + edge_weight * energy / energy.mean(dim=(1, 2), keepdim=True).clamp_min(1e-5)
            importance = importance / importance.mean(dim=(1, 2), keepdim=True)
            cross_entropy = (pixel_ce * importance).mean()
            severity_loss = (pixel_severity * importance).mean()
        else:
            cross_entropy = pixel_ce.mean()
            severity_loss = pixel_severity.mean()
        loss = cross_entropy + severity_loss
        if reblur_weight:
            loss = loss + reblur_weight * F.mse_loss(reblur_spatial_mixture(clear, predicted), blurred)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(clear)
        count += len(clear)
    return total / count


def heatmap(sigma: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    values = F.interpolate(sigma[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
    scaled = ((values.clamp(1, 3.5) - 1) / 2.5 * 255).byte().cpu().numpy()
    pixels = np.stack((scaled, np.zeros_like(scaled), 255 - scaled), axis=-1)
    return Image.fromarray(pixels, "RGB")


def save_example(path: Path, clear: torch.Tensor, blurred: torch.Tensor,
                 truth: torch.Tensor, predicted: torch.Tensor) -> None:
    height, width = clear.shape[-2:]
    images = [Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
              for x in (clear, blurred)]
    images.extend((heatmap(sigma_field(truth[None])[0], (height, width)),
                   heatmap(sigma_field(predicted[None])[0], (height, width))))
    board = Image.new("RGB", (width * 4, height))
    for index, image in enumerate(images):
        board.paste(image, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


@torch.no_grad()
def evaluate(model, loader, device, *, seed: int, global_model=None, examples: Path | None = None):
    model.eval()
    if global_model is not None:
        global_model.eval()
    generator = map_generator(device, seed)
    totals = {"pixels": 0, "images": 0, "sigma_mae": 0.0, "probability_mae": 0.0,
              "fixed_light_sigma_mae": 0.0, "global_sigma_mae": 0.0,
              "reblur_mse": 0.0, "fixed_light_reblur_mse": 0.0, "global_reblur_mse": 0.0,
              "spatial_std": 0.0, "truth_spatial_std": 0.0}
    example_count = 0
    fixed = None
    for clear, ids in loader:
        clear = clear.to(device, non_blocking=True)
        blurred, truth = synthetic_batch(clear, generator)
        prediction = model(blurred).softmax(dim=1)
        if fixed is None or fixed.shape[0] != len(clear) or fixed.shape[-2:] != truth.shape[-2:]:
            fixed = torch.zeros_like(truth)
            fixed[:, 0] = 1
        truth_sigma, pred_sigma = sigma_field(truth), sigma_field(prediction)
        fixed_sigma = sigma_field(fixed)
        global_map = None
        if global_model is not None:
            global_map = global_model(blurred).softmax(dim=1)[:, :, None, None].expand_as(truth)
        n_pixels = truth_sigma.numel()
        totals["pixels"] += n_pixels
        totals["images"] += len(clear)
        totals["sigma_mae"] += (pred_sigma - truth_sigma).abs().sum().item()
        totals["probability_mae"] += (prediction - truth).abs().sum().item() / 3
        totals["fixed_light_sigma_mae"] += (fixed_sigma - truth_sigma).abs().sum().item()
        totals["spatial_std"] += pred_sigma.flatten(1).std(dim=1).sum().item()
        totals["truth_spatial_std"] += truth_sigma.flatten(1).std(dim=1).sum().item()
        if global_map is not None:
            totals["global_sigma_mae"] += (sigma_field(global_map) - truth_sigma).abs().sum().item()
        totals["reblur_mse"] += (reblur_spatial_mixture(clear, prediction) - blurred).square().sum().item()
        totals["fixed_light_reblur_mse"] += (reblur_spatial_mixture(clear, fixed) - blurred).square().sum().item()
        if global_map is not None:
            totals["global_reblur_mse"] += (reblur_spatial_mixture(clear, global_map) - blurred).square().sum().item()
        if examples is not None:
            for index, sample_id in enumerate(ids):
                if example_count >= 6:
                    break
                save_example(examples / f"{example_count:02d}_{sample_id}.png", clear[index], blurred[index],
                             truth[index], prediction[index])
                example_count += 1
    pixels, images = totals["pixels"], totals["images"]
    if not pixels:
        raise ValueError("Empty validation loader")
    result = {"images": images, "sigma_mae": totals["sigma_mae"] / pixels,
              "probability_mae": totals["probability_mae"] / pixels,
              "fixed_light_sigma_mae": totals["fixed_light_sigma_mae"] / pixels,
              "predicted_spatial_std": totals["spatial_std"] / images,
              "truth_spatial_std": totals["truth_spatial_std"] / images,
              "reblur_mse": totals["reblur_mse"] / (pixels * 3),
              "fixed_light_reblur_mse": totals["fixed_light_reblur_mse"] / (pixels * 3)}
    if global_model is not None:
        result.update(global_sigma_mae=totals["global_sigma_mae"] / pixels,
                      global_reblur_mse=totals["global_reblur_mse"] / (pixels * 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cv", "final", "evaluate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--global-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reblur-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    device = torch.device(args.device)
    root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_records(args.train_manifest, "train")
    groups = sorted({r["group_id"] for r in train_records})
    if args.mode == "cv":
        if len(groups) < 2:
            raise ValueError("At least two source groups required for internal validation")
        histories = []
        for group in groups:
            seed_all(args.seed)
            training = [r for r in train_records if r["group_id"] != group]
            validation = [r for r in train_records if r["group_id"] == group]
            model = LocalBlurEstimator().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            train_loader = make_loader(root, training, args, True)
            val_loader = make_loader(root, validation, args, False)
            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(model, train_loader, optimizer, device, args.seed + epoch,
                                         args.reblur_weight, args.edge_weight)
                metrics = evaluate(model, val_loader, device, seed=9471)
                row = {"fold": group, "epoch": epoch, "train_loss": train_loss, **metrics}
                histories.append(row)
                print(json.dumps(row), flush=True)
        scores = [(float(np.mean([r["sigma_mae"] for r in histories if r["epoch"] == epoch])), epoch)
                  for epoch in range(1, args.epochs + 1)]
        score, chosen_epoch = min(scores)
        result = {"chosen_epoch": chosen_epoch, "mean_internal_sigma_mae": score,
                  "selection": "leave-one-original-group-out synthetic sigma MAE; fixed debug_val unused",
                  "groups": groups, "history": histories}
        (args.output_dir / "cv_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    elif args.mode == "final":
        if args.checkpoint:
            raise ValueError("Final training starts from scratch")
        seed_all(args.seed)
        model = LocalBlurEstimator().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = make_loader(root, train_records, args, True)
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device, args.seed + epoch,
                               args.reblur_weight, args.edge_weight)
            history.append({"epoch": epoch, "train_loss": loss})
            print(json.dumps(history[-1]), flush=True)
        checkpoint = args.output_dir / "local-estimator.pt"
        torch.save({"model": model.state_dict(), "epoch": args.epochs, "levels": LEVELS,
                    "map_resolution": "ceil(H/4) x ceil(W/4)", "output": "three local PSF mixture logits",
                    "manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
                    "source": "spatially varying synthetic Gaussian mixtures on fixed 3DH train clear images",
                    "reblur_weight": args.reblur_weight, "edge_weight": args.edge_weight}, checkpoint)
        (args.output_dir / "final_summary.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "train_images": len(train_records), "epochs": args.epochs,
            "history": history}, indent=2), encoding="utf-8")
    else:
        if args.val_manifest is None or args.checkpoint is None:
            raise ValueError("Evaluate requires --val-manifest and --checkpoint")
        val_records = read_records(args.val_manifest, "val")
        train_paths = {r["clear_path"] for r in train_records}
        if any(r["clear_path"] in train_paths for r in val_records):
            raise ValueError("Fixed validation overlaps training clear images")
        model = LocalBlurEstimator().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["model"])
        global_model = None
        if args.global_checkpoint is not None:
            global_model = BlurEstimator().to(device)
            global_model.load_state_dict(torch.load(args.global_checkpoint, map_location=device, weights_only=True)["model"])
        metrics = evaluate(model, make_loader(root, val_records, args, False), device, seed=9471,
                           global_model=global_model, examples=args.output_dir / "examples")
        (args.output_dir / "synthetic_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
