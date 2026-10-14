"""Generate fixed synthetic PNG pairs and their true native probability fields."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--train-manifest", type=Path)
    p.add_argument("--val-manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--train-count", type=int, default=5000, help="0 = all source training pairs")
    p.add_argument("--val-count", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--field-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-group-overlap", action="store_true",
                   help="Explicitly label this as a mechanism experiment if source groups overlap")
    return p


def main(args):
    import numpy as np
    import torch
    from PIL import Image
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from common_io import load_manifest, read_rgb_image
    from a1_synthetic_data import sha256_file, validate_truth
    source_dir = Path(__file__).resolve().parents[1] / "experiments/a1_oracle_synthetic/source"
    sys.path.insert(0, str(source_dir))
    from local_synthetic import sample_spatial_weights
    from spatial_psf import reblur_spatial_mixture
    from psf import LEVELS
    train_path = args.train_manifest or root / "prepared/manifests/3dh_full_train.jsonl"
    val_path = args.val_manifest or root / "prepared/manifests/debug_val.jsonl"
    out = (args.output_dir or root / "experiments/a1_oracle_synthetic/data_v1").resolve()
    if not out.is_relative_to(root):
        raise ValueError("Place generated data under project root for portable manifests")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory; not overwriting {out}")
    if args.train_count < 0 or args.val_count < 1 or args.field_size < 4 or args.image_size < 32:
        raise ValueError("Invalid sample count or spatial size")
    sets = {"train": load_manifest(train_path, root=root, expected_split="train"),
            "val": load_manifest(val_path, root=root, expected_split="val")}
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        records = sorted(sets[split], key=lambda r: r.sample_id)
        if count > len(records):
            raise ValueError(f"Requested {count} {split} images, only {len(records)} available")
        if count:
            indices = np.linspace(0, len(records) - 1, count, dtype=int)
            records = [records[i] for i in indices]
        if not records:
            raise ValueError(f"No {split} source images")
        sets[split] = records
    train_ids = {r.sample_id for r in sets["train"]}
    train_clear = {r.clear_path.resolve() for r in sets["train"]}
    if any(r.sample_id in train_ids or r.clear_path.resolve() in train_clear for r in sets["val"]):
        raise ValueError("Source train/val IDs or clear images overlap")
    groups = {s: {r.sample_id.split("_")[0] for r in rs} for s, rs in sets.items()}
    overlap = sorted(groups["train"] & groups["val"])
    if overlap and not args.allow_group_overlap:
        raise ValueError(f"Source groups overlap: {overlap}; use independent source groups or explicitly --allow-group-overlap for a mechanism experiment")
    config = {"status": "generating", "generation_kind": "known_local_gaussian_mixture",
              "seed": args.seed, "field_size": args.field_size, "image_size": args.image_size,
              "channel_order": ["light", "medium", "heavy"], "levels": LEVELS,
              "interpolation": "bilinear_align_corners_false_then_channel_normalization",
              "blur_storage": "PNG_round_to_nearest_uint8", "device": args.device,
              "train_count": len(sets["train"]), "val_count": len(sets["val"]),
              "source_group_overlap": overlap, "allow_group_overlap": args.allow_group_overlap,
              "group_id_rule": "sample_id prefix before first underscore",
              "source_manifest_sha256": {"train": sha256_file(train_path), "val": sha256_file(val_path)},
              "source_code_sha256": {n: sha256_file(source_dir / n) for n in
                                      ("local_synthetic.py", "spatial_psf.py", "psf.py")}}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "generation_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device(args.device)
    for split, records in sets.items():
        for folder in ("blur", "truth", "manifests"):
            (out / folder / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for index, record in enumerate(records):
            array = read_rgb_image(record.clear_path, sample_id=record.sample_id, role="source clear")
            if array.shape != (args.image_size, args.image_size, 3):
                raise ValueError(f"{record.sample_id} has {array.shape}; prepare exact square patches without silently resizing")
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{split}:{record.sample_id}".encode()).digest()[:8], "little") & ((1 << 63) - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                field = sample_spatial_weights(1, args.field_size, args.field_size,
                                               device=device, generator=generator)
                clear = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                blurred = reblur_spatial_mixture(clear, field)
            probabilities = validate_truth(field[0].cpu().numpy().astype(np.float32))
            truth = out / "truth" / split / f"{record.sample_id}.npz"
            np.savez_compressed(truth, probabilities=probabilities,
                                channel_order=np.array("light,medium,heavy"), generator_seed=np.array(seed))
            blur = out / "blur" / split / record.clear_path.name
            image = np.rint(blurred[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(image).save(blur, format="PNG")
            rows.append({"sample_id": record.sample_id, "source_sample_id": record.sample_id,
                         "group_id": record.sample_id.split("_")[0], "split": split,
                         "quality_status": "pass", "generation_kind": "known_local_gaussian_mixture",
                         "clear_path": record.clear_path.relative_to(root).as_posix(),
                         "blur_path": blur.relative_to(root).as_posix(),
                         "truth_probability_path": truth.relative_to(root).as_posix(),
                         "generator_seed": seed, "truth_sha256": sha256_file(truth),
                         "blur_sha256": sha256_file(blur), "clear_sha256": sha256_file(record.clear_path)})
            if index % 100 == 0 or index + 1 == len(records):
                print(f"{split}: {index + 1}/{len(records)}", flush=True)
        manifest = out / "manifests" / f"synthetic_{split}.jsonl"
        manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    config["manifest_sha256"] = {s: sha256_file(out / "manifests" / f"synthetic_{s}.jsonl") for s in sets}
    config["status"] = "complete"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {config_path}", flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
