"""Save U-Net predictions with the shared outputs/{model}/{sample_id}.png protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from common_io import PROJECT_ROOT, load_manifest, load_pair, save_prediction
from unet_model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "prepared/manifests/debug_val.jsonl",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--model-name", default="unet")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only for inspecting the first few training samples")
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and (args.limit < 1 or args.split != "train"):
        raise SystemExit("--limit is only supported for positive counts on --split train")
    if args.preview_count < 0:
        raise SystemExit("--preview-count must be nonnegative")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot access a GPU")
    device = torch.device(device_name)

    records = load_manifest(args.manifest, expected_split=args.split)
    if args.split == "val" and len(records) != 300:
        raise SystemExit(f"Expected 300 debug_val records, found {len(records)}")
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise SystemExit("Manifest contains no records")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preview_dir = PROJECT_ROOT / "prepared/previews" / args.model_name

    with torch.no_grad():
        for index, record in enumerate(records, 1):
            blur, clear = load_pair(record)
            input_tensor = torch.from_numpy(
                np.ascontiguousarray(blur.transpose(2, 0, 1))
            ).unsqueeze(0).to(device)
            prediction = model(input_tensor)[0].permute(1, 2, 0).cpu().numpy()
            output_path = save_prediction(
                prediction, model_name=args.model_name, sample_id=record.sample_id,
                expected_hw=clear.shape[:2],
            )
            if index <= args.preview_count:
                preview_dir.mkdir(parents=True, exist_ok=True)
                panels = [
                    np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
                    for image in (blur, prediction, clear)
                ]
                comparison = np.concatenate(panels, axis=1)
                Image.fromarray(comparison, mode="RGB").save(
                    preview_dir / f"{record.sample_id}.png"
                )
            if index % 50 == 0 or index == len(records):
                print(f"Saved {index}/{len(records)} predictions; latest={output_path}", flush=True)

    print("Preview panels, left to right: blur | U-Net | clear")


if __name__ == "__main__":
    main()
