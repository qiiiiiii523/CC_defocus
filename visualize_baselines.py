"""Create matched six-panel previews for the fixed 3DHistech validation IDs.

Requires Pillow. The first ten records in debug_val.jsonl are used by default.
Missing DGNO outputs are shown as clearly labeled placeholders, so the same
command can be rerun after outputs/dgno is populated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required: python -m pip install Pillow") from exc


PANEL_LABELS = ("Blur input", "U-Net", "RFN", "DGNO-Face", "MPT", "Clear label")
MODEL_DIRS = ("unet", "rfn", "dgno", "mpt")
HEADER_HEIGHT = 36
FOOTER_HEIGHT = 28
PANEL_GAP = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=10,
                        help="Use the first N frozen validation IDs (default: 10).")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="Optional UTF-8 file with one fixed sample_id per line; overrides --count.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any DGNO image is missing instead of showing a placeholder.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or Path(sample_id).name != sample_id:
                raise ValueError(f"Invalid sample_id at line {line_number}: {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            if row.get("split") != "val":
                raise ValueError(f"Non-validation record at line {line_number}: {sample_id}")
            if not row.get("blur_path") or not row.get("clear_path"):
                raise ValueError(f"Missing image path at line {line_number}: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("Validation manifest is empty")
    return rows


def select_records(rows: list[dict], count: int, ids_file: Path | None) -> list[dict]:
    if ids_file is None:
        if count < 1 or count > len(rows):
            raise ValueError(f"--count must be between 1 and {len(rows)}")
        return rows[:count]
    wanted = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("--ids-file must contain unique, nonempty sample IDs")
    by_id = {row["sample_id"]: row for row in rows}
    unknown = [sample_id for sample_id in wanted if sample_id not in by_id]
    if unknown:
        raise ValueError(f"IDs not in validation manifest: {unknown[:5]}")
    return [by_id[sample_id] for sample_id in wanted]


def read_rgb(path: Path, expected_size: tuple[int, int] | None = None) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if expected_size is not None and image.size != expected_size:
        raise ValueError(f"Size mismatch: {path} is {image.size}, expected {expected_size}")
    return image


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                  label: str, fill: str, font: ImageFont.ImageFont) -> None:
    bounds = draw.textbbox((0, 0), label, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2 - bounds[1]
    draw.text((x, y), label, fill=fill, font=font)


def make_panel(images: list[Image.Image | None], sample_id: str) -> Image.Image:
    size = images[0].size
    width, height = size
    canvas_width = len(images) * width + (len(images) - 1) * PANEL_GAP
    canvas = Image.new("RGB", (canvas_width, HEADER_HEIGHT + height + FOOTER_HEIGHT), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(zip(PANEL_LABELS, images)):
        x = index * (width + PANEL_GAP)
        centered_text(draw, (x, 0, x + width, HEADER_HEIGHT), label, "#171717", font)
        if image is None:
            draw.rectangle((x, HEADER_HEIGHT, x + width - 1, HEADER_HEIGHT + height - 1),
                           fill="#454b54")
            centered_text(draw, (x, HEADER_HEIGHT, x + width, HEADER_HEIGHT + height),
                          "DGNO output pending", "#ffffff", font)
        else:
            canvas.paste(image, (x, HEADER_HEIGHT))
    centered_text(draw, (0, HEADER_HEIGHT + height, canvas_width,
                         HEADER_HEIGHT + height + FOOTER_HEIGHT),
                  f"sample_id: {sample_id}", "#171717", font)
    return canvas


def main() -> None:
    args = arguments()
    root = args.root.resolve()
    manifest = args.manifest or root / "prepared/manifests/debug_val.jsonl"
    output_dir = args.output_dir or root / "prepared/previews/baseline_comparison"
    rows = select_records(load_records(manifest), args.count, args.ids_file)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_dgno = []

    for row in rows:
        sample_id = row["sample_id"]
        blur = read_rgb(root / row["blur_path"])
        clear = read_rgb(root / row["clear_path"], blur.size)
        model_images = []
        for model_name in MODEL_DIRS:
            path = root / "outputs" / model_name / f"{sample_id}.png"
            if model_name == "dgno" and not path.is_file() and not args.strict:
                model_images.append(None)
                missing_dgno.append(sample_id)
            else:
                model_images.append(read_rgb(path, blur.size))
        panel = make_panel([blur, *model_images, clear], sample_id)
        destination = output_dir / f"{sample_id}.png"
        panel.save(destination, format="PNG")
        print(destination)

    print(f"Created {len(rows)} matched six-column previews in {output_dir}")
    if missing_dgno:
        print(f"DGNO pending for {len(missing_dgno)} selected IDs. "
              "After copying outputs/dgno, rerun the same command to fill that column.")


if __name__ == "__main__":
    main()
