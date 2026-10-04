"""Create a fixed validation sheet of predicted blur severities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data import CNSegSigmaDataset
from data_3dh import PairedClearSyntheticDataset
from model import BlurEstimator, predicted_sigma
from psf import LEVELS
from train_estimator import make_blurred_crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-kind", choices=("cnseg", "paired-clear"), default="cnseg")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-level", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.per_level < 1:
        raise ValueError("per-level must be positive")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    dataset_class = CNSegSigmaDataset if args.data_kind == "cnseg" else PairedClearSyntheticDataset
    dataset = dataset_class(args.project_root, args.manifest, "val")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BlurEstimator().to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    chosen: dict[int, list[int]] = {i: [] for i in range(len(LEVELS))}
    for index, (_, level_id, _) in enumerate(dataset.records):
        if len(chosen[level_id]) < args.per_level:
            chosen[level_id].append(index)
        if all(len(group) == args.per_level for group in chosen.values()):
            break
    if not all(len(group) == args.per_level for group in chosen.values()):
        raise ValueError("Validation manifest has too few records for the requested sheet")

    tile_size = args.crop_size
    label_height = 48
    sheet = Image.new(
        "RGB",
        (2 * tile_size * args.per_level, (tile_size + label_height) * len(LEVELS)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    with torch.no_grad():
        for row, (name, sigma, _) in enumerate(LEVELS):
            for column, index in enumerate(chosen[row]):
                clear, level_id, sample_id = dataset[index]
                source = clear[None].to(device)
                labels = torch.tensor([level_id], device=device)
                blur = make_blurred_crop(source, labels, tile_size, training=False)
                top = (source.shape[-2] - tile_size) // 2
                left = (source.shape[-1] - tile_size) // 2
                clear_crop = source[:, :, top : top + tile_size, left : left + tile_size]
                logits = model(blur)
                predicted_id = int(logits.argmax(dim=1).item())
                estimated_sigma = float(predicted_sigma(logits).item())
                x = column * 2 * tile_size
                y = row * (tile_size + label_height)
                for offset, image in ((0, blur), (tile_size, clear_crop)):
                    array = image[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
                    sheet.paste(Image.fromarray(np.asarray(array)), (x + offset, y))
                draw.text((x + 4, y + tile_size + 2), f"{sample_id[:30]} | true={name} ({sigma})", fill="black", font=font)
                draw.text(
                    (x + 4, y + tile_size + 23),
                    f"pred={LEVELS[predicted_id][0]} | sigma={estimated_sigma:.2f} | blur / clear",
                    fill="black",
                    font=font,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
