"""Reconstruct sampled saved synthetic PNGs using their SAVED true fields."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path


def quantization_comparison(observed, reconstructed, max_mae_levels=0.05):
    import numpy as np
    if observed.dtype != np.uint8 or reconstructed.dtype != np.uint8:
        raise ValueError("Compare RGB uint8 PNG values, not float images")
    if observed.shape != reconstructed.shape:
        raise ValueError("Reconstructed PNG shape differs from stored PNG")
    difference = np.abs(observed.astype(np.int16) - reconstructed.astype(np.int16))
    maximum, mae = int(difference.max()), float(difference.mean())
    exact = maximum == 0
    passed = maximum <= 1 and mae <= max_mae_levels
    return {"max_abs_uint8": maximum, "mae_uint8": mae,
            "changed_fraction": float((difference > 0).mean()),
            "over_one_level_fraction": float((difference > 1).mean()),
            "mse_01": float(np.mean((difference.astype(np.float64) / 255) ** 2)),
            "status": "exact" if exact else "rounding_tolerance" if passed else "mismatch",
            "passed": passed}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--generation-config", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-split", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p


def main(args):
    import numpy as np
    import torch
    from PIL import Image
    from a1_synthetic_data import read_truth, sha256_file
    from common_io import read_rgb_image
    root = args.data_root.resolve()
    if args.samples_per_split < 1:
        raise ValueError("samples-per-split must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("Use a new check output directory")
    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise ValueError("Data generation is not complete")
    source = args.project_root.resolve() / "experiments/a1_oracle_synthetic/source"
    code_checks = {name: sha256_file(source / name) == expected
                   for name, expected in config["source_code_sha256"].items()}
    if not all(code_checks.values()):
        (out / "summary.json").write_text(json.dumps({"status": "failed", "reason": "generation code changed",
                                                      "source_code_hash_checks": code_checks}, indent=2), encoding="utf-8")
        raise ValueError("Use the same PSF source files as the original generation")
    sys.path.insert(0, str(source))
    from spatial_psf import reblur_spatial_mixture
    rows, errors = [], []
    device = torch.device(args.device)
    for split, manifest in (("train", args.train_manifest), ("val", args.val_manifest)):
        if sha256_file(manifest) != config["manifest_sha256"][split]:
            raise ValueError(f"{split} manifest changed since generation")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        indices = np.linspace(0, len(records) - 1, min(args.samples_per_split, len(records)), dtype=int)
        for index in indices:
            record = records[int(index)]
            row = {"split": split, "sample_id": record["sample_id"], "hashes_verified": False}
            try:
                paths = {}
                for role, key in (("clear", "clear_path"), ("blur", "blur_path"), ("truth", "truth_probability_path")):
                    path = Path(record[key])
                    path = (path if path.is_absolute() else root / path).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(f"{role} outside data root")
                    if sha256_file(path) != record[role + "_sha256"]:
                        raise ValueError(f"{role} content hash changed")
                    paths[role] = path
                row["hashes_verified"] = True
                clear = read_rgb_image(paths["clear"], sample_id=record["sample_id"], role="clear")
                with Image.open(paths["blur"]) as image:
                    observed = np.asarray(image.convert("RGB"), dtype=np.uint8)
                probabilities = read_truth(paths["truth"])
                with torch.no_grad():
                    clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                    field = torch.from_numpy(probabilities).unsqueeze(0).to(device)
                    reconstructed_float = reblur_spatial_mixture(clear_tensor, field)
                reconstructed = np.rint(reconstructed_float[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                row.update(quantization_comparison(observed, reconstructed))
                if not row["passed"]:
                    errors.append({"sample_id": record["sample_id"], "reason": "pixel mismatch"})
                # Save only the first two examples per split for easy inspection.
                if int(index) in set(int(v) for v in indices[:2]):
                    panel = np.concatenate([np.rint(clear * 255).astype(np.uint8), observed, reconstructed,
                                            np.minimum(np.abs(observed.astype(np.int16)-reconstructed.astype(np.int16))*16, 255).astype(np.uint8)], axis=1)
                    Image.fromarray(panel).save(out / f"{split}_{record['sample_id']}_clear_stored_rebuilt_diff16.png")
            except Exception as exc:
                row.update(status="error", passed=False)
                errors.append({"sample_id": record["sample_id"], "reason": str(exc)})
            rows.append(row)
            print(split, record["sample_id"], row["status"], flush=True)
    columns = ("split", "sample_id", "hashes_verified", "status", "passed", "max_abs_uint8", "mae_uint8",
               "changed_fraction", "over_one_level_fraction", "mse_01")
    with (out / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    summary = {"status": "complete" if not errors else "failed", "checked_count": len(rows),
               "all_passed": not errors, "source_code_hash_checks": code_checks,
               "generation_config_sha256": sha256_file(args.generation_config),
               "threshold": {"max_abs_uint8": 1, "max_mae_uint8": 0.05},
               "exact_count": sum(r["status"] == "exact" for r in rows),
               "rounding_tolerance_count": sum(r["status"] == "rounding_tolerance" for r in rows),
               "errors": errors,
               "scope": "sampled native fields reconstruct stored pairs; not proof of every downstream transform"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reblur summary: {out / 'summary.json'}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        main(arguments)
    except Exception as exc:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = arguments.output_dir / "summary.json"
        if not summary_path.exists():
            summary_path.write_text(json.dumps({"status": "failed", "all_passed": False,
                                                "reason": str(exc)}, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
        raise
