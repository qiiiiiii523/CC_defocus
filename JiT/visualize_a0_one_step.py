"""Build blur / clear / one-step prediction panels from existing PNG outputs.

No model inference or training. Requires Pillow, not PyTorch or a GPU.
Relative image paths in the manifest are resolved against --root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_EXPERIMENT = "jit_a0_scratch_full66976_charb1_e20_sampling_ema2_heun_s1"


def resolve(root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_records(manifest: Path) -> list[dict]:
    records = []
    seen = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if (not isinstance(sid, str) or not sid or sid in {".", ".."}
                or any(c in sid for c in '/\\:') or sid in seen):
            raise ValueError(f"Invalid or duplicate sample_id at line {number}: {sid!r}")
        if row.get("split") != "val" or row.get("quality_status") != "pass":
            raise ValueError(f"Expected a pass/val record at line {number}: {sid}")
        if not row.get("blur_path") or not row.get("clear_path"):
            raise ValueError(f"Missing paired paths at line {number}: {sid}")
        seen.add(sid)
        records.append(row)
    if not records:
        raise ValueError("Empty validation manifest")
    return records


def select_records(records: list[dict], count: int, ids_file: Path | None) -> list[dict]:
    if ids_file is not None:
        ids = [s.strip() for s in ids_file.read_text(encoding="utf-8-sig").splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("IDs file must contain unique nonempty IDs")
        by_id = {r["sample_id"]: r for r in records}
        missing = [sid for sid in ids if sid not in by_id]
        if missing:
            raise ValueError(f"IDs not in manifest: {missing[:5]}")
        return [by_id[sid] for sid in ids]
    if count == 0:
        return records
    if count < 1 or count > len(records):
        raise ValueError(f"--count must be 0 (all), or 1..{len(records)}")
    indices = [0] if count == 1 else [i * (len(records) - 1) // (count - 1) for i in range(count)]
    return [records[i] for i in indices]


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_panel(images: list[Image.Image], sid: str, scale: int) -> Image.Image:
    w, h = images[0].size
    w, h = w * scale, h * scale
    gap, header, footer = 8, 36, 30
    canvas = Image.new("RGB", (3 * w + 2 * gap, header + h + footer), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (image, label) in enumerate(zip(images, ("Blur input", "Clear reference", "JiT: 1 step / EMA2"))):
        left = i * (w + gap)
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text((left + (w - bounds[2] + bounds[0]) // 2, 10), label, fill="#171717", font=font)
        # Nearest-neighbor magnification only. Never smooth or resize mismatched inputs.
        display = image if scale == 1 else image.resize((w, h), Image.Resampling.NEAREST)
        canvas.paste(display, (left, header))
    draw.text((8, header + h + 8), f"sample_id: {sid} | display scale: {scale}x", fill="#171717", font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--manifest", type=Path, default=Path("prepared/manifests/debug_val.jsonl"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs") / DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/a0_full_e20_one_step_panels"))
    parser.add_argument("--count", type=int, default=20, help="Evenly spaced manifest records; 0 means all.")
    parser.add_argument("--ids-file", type=Path, help="One sample_id per line; overrides --count.")
    parser.add_argument("--scale", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = resolve(root, args.manifest)
    prediction_dir = resolve(root, args.prediction_dir)
    output_dir = resolve(root, args.output_dir)
    selected = select_records(load_records(manifest), args.count,
                              resolve(root, args.ids_file) if args.ids_file else None)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be new or empty; choose another --output-dir: {output_dir}")
    paths = []
    # Check every selected input before producing panels; missing predictions are errors.
    for row in selected:
        sid = row["sample_id"]
        trio = [resolve(root, row["blur_path"]), resolve(root, row["clear_path"]),
                prediction_dir / f"{sid}.png"]
        sizes = []
        for path in trio:
            if not path.is_file():
                raise FileNotFoundError(f"Missing image for {sid}: {path}")
            with Image.open(path) as image:
                sizes.append(image.size)
                image.verify()
        if len(set(sizes)) != 1:
            raise ValueError(f"Image dimensions differ for {sid}: {sizes}; no automatic resizing")
        paths.append((sid, trio))
    output_dir.mkdir(parents=True, exist_ok=True)
    for sid, trio in paths:
        destination = output_dir / f"{sid}_panel.png"
        make_panel([read_rgb(p) for p in trio], sid, args.scale).save(destination)
        print(destination, flush=True)
    metadata = {
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "prediction_dir": str(prediction_dir),
        "count": len(paths), "sample_ids": [sid for sid, _ in paths],
        "columns": ["blur", "clear reference", "existing one-step EMA2 prediction"],
        "display_scale": args.scale, "display_resampling": "nearest",
        "note": "Only visualizes existing outputs. Does not verify checkpoint provenance, infer, or train.",
    }
    (output_dir / "panel_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(paths)} panels in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
