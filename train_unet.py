"""Train the small U-Net on fixed 3DHistech pairs.

Use --overfit-eight first to check that the pipeline can learn eight fixed
training pairs. A separate invocation starts a fresh model for all 2,000 pairs;
--resume continues a previously interrupted invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common_io import PROJECT_ROOT, load_manifest
from unet_data import PairedImages
from unet_model import MODEL_VERSION, SmallUNet


DEFAULT_TRAIN = PROJECT_ROOT / "prepared/manifests/debug_train.jsonl"
DEFAULT_VAL = PROJECT_ROOT / "prepared/manifests/debug_val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--overfit-eight", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue from the latest training state; --epochs is the total target")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: SmallUNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for blur, clear in loader:
            blur = blur.to(device)
            clear = clear.to(device)
            prediction = model(blur)
            loss = criterion(prediction, clear)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = blur.shape[0]
            total_loss += loss.item() * batch_size
            count += batch_size
    return total_loss / count


def main() -> None:
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else (200 if args.overfit_eight else 30)
    if epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise SystemExit("epochs, batch-size and lr must be positive")
    device = choose_device(args.device)
    set_seed(args.seed)

    train_records = load_manifest(args.train_manifest, expected_split="train")
    expected_train = 2000
    if len(train_records) != expected_train:
        raise SystemExit(
            f"Expected {expected_train} debug_train records, found {len(train_records)}"
        )
    if args.overfit_eight:
        train_records = train_records[:8]
        val_loader = None
    else:
        val_records = load_manifest(args.val_manifest, expected_split="val")
        if len(val_records) != 300:
            raise SystemExit(f"Expected 300 debug_val records, found {len(val_records)}")
        val_loader = DataLoader(
            PairedImages(val_records), batch_size=args.batch_size, shuffle=False,
            num_workers=0,
        )

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairedImages(train_records), batch_size=args.batch_size, shuffle=True,
        generator=shuffle_generator, num_workers=0,
    )
    model = SmallUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mode = "overfit8" if args.overfit_eight else "debug"
    checkpoint = args.checkpoint or PROJECT_ROOT / f"checkpoints/unet_{mode}_{MODEL_VERSION}.pt"
    latest_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_last{checkpoint.suffix}")
    log_path = args.log or PROJECT_ROOT / f"runs/unet_{mode}_{MODEL_VERSION}.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"mode={mode} train_pairs={len(train_records)} epochs={epochs} "
        f"batch_size={args.batch_size} device={device}", flush=True
    )
    best = float("inf")
    start_epoch = 1
    if args.resume:
        if not latest_checkpoint.is_file():
            raise SystemExit(f"No resumable training state: {latest_checkpoint}")
        state = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "model_version": MODEL_VERSION,
            "mode": mode,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": None if args.overfit_eight else str(args.val_manifest.resolve()),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(f"Resume mismatch for {key}: saved={state.get(key)!r}, current={value!r}")
        if not checkpoint.is_file():
            raise SystemExit(f"Best model checkpoint is missing: {checkpoint}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        shuffle_generator.set_state(state["shuffle_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        best = float(state["best_l1"])
        start_epoch = int(state["epoch"]) + 1
        if not log_path.is_file():
            raise SystemExit(f"Cannot resume without training log: {log_path}")
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        logged_epoch = int(rows[-1]["epoch"]) if rows else 0
        if logged_epoch == start_epoch - 2:
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=("epoch", "train_l1", "val_l1")).writerow(state["last_record"])
        elif logged_epoch != start_epoch - 1:
            raise SystemExit("Training log and latest checkpoint disagree on the last epoch")
        if epochs < start_epoch:
            raise SystemExit(f"Target --epochs={epochs} is below completed epoch {start_epoch - 1}")
        print(f"Resuming after epoch {start_epoch - 1}; best_L1={best:.6f}", flush=True)

    with log_path.open("a" if args.resume else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_l1", "val_l1"))
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = (
                run_epoch(model, val_loader, criterion, device, None)
                if val_loader is not None else None
            )
            record = {
                "epoch": epoch,
                "train_l1": f"{train_loss:.8f}",
                "val_l1": "" if val_loss is None else f"{val_loss:.8f}",
            }
            score = train_loss if val_loss is None else val_loss
            if score < best:
                best = score
                best_state = {
                    "model_state": model.state_dict(),
                    "model_version": MODEL_VERSION,
                    "epoch": epoch,
                    "seed": args.seed,
                    "mode": mode,
                    "best_l1": best,
                }
                temporary_best = checkpoint.with_name(checkpoint.name + ".tmp")
                torch.save(best_state, temporary_best)
                temporary_best.replace(checkpoint)
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_version": MODEL_VERSION,
                "mode": mode,
                "epoch": epoch,
                "best_l1": best,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "train_manifest": str(args.train_manifest.resolve()),
                "val_manifest": None if args.overfit_eight else str(args.val_manifest.resolve()),
                "shuffle_state": shuffle_generator.get_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "last_record": record,
            }
            temporary_checkpoint = latest_checkpoint.with_name(latest_checkpoint.name + ".tmp")
            torch.save(state, temporary_checkpoint)
            temporary_checkpoint.replace(latest_checkpoint)
            writer.writerow(record)
            handle.flush()
            print(
                f"epoch={epoch:03d} train_L1={train_loss:.6f} "
                + ("" if val_loss is None else f"val_L1={val_loss:.6f} ")
                + f"best={best:.6f}", flush=True
            )

    config = {
        "mode": mode,
        "train_manifest": args.train_manifest.name,
        "val_manifest": None if args.overfit_eight else args.val_manifest.name,
        "train_pairs": len(train_records),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "loss": "L1",
        "model_version": MODEL_VERSION,
        "best_checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "torch_version": torch.__version__,
    }
    config_path = log_path.with_suffix(".json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {checkpoint}\nLog: {log_path}")


if __name__ == "__main__":
    main()
