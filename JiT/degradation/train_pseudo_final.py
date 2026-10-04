"""Fit the final 3DHistech pseudo-PSF estimator on all 2,000 fixed training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_3dh_pseudo import PairedPseudoDataset
from model import BlurEstimator
from psf import LEVELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Fixed in advance by training-only internal group validation")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = PairedPseudoDataset(args.project_root, args.manifest, args.labels, "train")
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=2)
    model = BlurEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for blur, label, _sample_id in loader:
            blur, label = blur.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(blur), label)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(label)
            count += len(label)
        row = {"epoch": epoch, "train_loss": loss_sum / count, "samples": count}
        history.append(row)
        print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "levels": LEVELS,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "pseudo_labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "source": "3DHistech real blurred training images; coarse labels fitted from paired clear images",
        "selection": "epoch count selected on training-only held-out group 10140064; fixed A1 validation not used",
    }, args.output_dir / "estimator-final.pt")
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "status": "FULL_TRAINING_NO_VALIDATION_SELECTION",
        "train_records": len(data),
        "device": str(device),
        "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
