"""Run official DGNO weights through the CC_defocus common I/O protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DGNO_ROOT = PROJECT_ROOT / "third_party" / "DGNO"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DGNO_ROOT) not in sys.path:
    sys.path.insert(0, str(DGNO_ROOT))

from common_io import ImageProtocolError, load_manifest, load_pair, save_prediction
from basicsr.models.archs.dgno_eval_arch import DGNO_eval


VARIANTS = {
    "face": {
        "config": DGNO_ROOT / "options" / "DGNO_3DHistech_Face.yml",
        "weights": PROJECT_ROOT / "weights" / "dgno" / "DGNO_3DHistech_Face.pth",
    },
    "cell": {
        "config": DGNO_ROOT / "options" / "DGNO_3DHistech_Cell.yml",
        "weights": PROJECT_ROOT / "weights" / "dgno" / "DGNO_3DHistech_Cell.pth",
    },
}
DEFAULT_MANIFEST = PROJECT_ROOT / "prepared" / "manifests" / "debug_val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="face")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default="dgno")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_revision() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_model(variant: str, weights_path: Path, device: torch.device) -> torch.nn.Module:
    config_path = VARIANTS[variant]["config"]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    network_options = dict(config["network_g"])
    network_options.pop("type", None)
    if network_options.get("variant") != variant:
        raise RuntimeError(
            f"Config variant mismatch: expected {variant!r}, "
            f"found {network_options.get('variant')!r}"
        )

    model = DGNO_eval(**network_options)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise RuntimeError("Checkpoint must be a dictionary containing the 'params' key")
    model.load_state_dict(checkpoint["params"], strict=True)
    return model.to(device).eval()


def write_records(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    fields = ("sample_id", "status", "elapsed_seconds", "output_path", "error")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    if not torch.cuda.is_available():
        raise SystemExit("DGNO requires an NVIDIA CUDA GPU, but torch.cuda.is_available() is false")

    weights_path = (args.weights or VARIANTS[args.variant]["weights"]).resolve()
    if not weights_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {weights_path}. Run "
            f"python scripts/dgno/download_weights.py --variant {args.variant}"
        )

    device = torch.device(args.device)
    records = load_manifest(
        args.manifest,
        root=PROJECT_ROOT,
        expected_split="val",
        required_quality_status="pass",
    )
    if args.limit is not None:
        records = records[: args.limit]

    model = load_model(args.variant, weights_path, device)
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    rows: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        torch.cuda.synchronize(device)
        item_started = time.perf_counter()
        row: dict[str, object] = {
            "sample_id": record.sample_id,
            "status": "failed",
            "elapsed_seconds": "",
            "output_path": "",
            "error": "",
        }
        try:
            blur, clear = load_pair(record)
            height, width = blur.shape[:2]
            if height % 8 or width % 8:
                raise ImageProtocolError(
                    f"DGNO requires height and width divisible by 8, got {(height, width)}"
                )
            input_tensor = torch.from_numpy(
                np.ascontiguousarray(blur.transpose(2, 0, 1))
            ).unsqueeze(0).to(device=device)
            with torch.inference_mode():
                restored = model(input_tensor)
                if isinstance(restored, (list, tuple)):
                    restored = restored[-1]
                restored = restored.clamp_(0.0, 1.0)
            torch.cuda.synchronize(device)
            prediction = (
                restored.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            )
            output_path = save_prediction(
                prediction,
                model_name=args.model_name,
                sample_id=record.sample_id,
                expected_hw=clear.shape[:2],
                output_root=args.output_root,
            )
            row["status"] = "ok"
            row["output_path"] = output_path.relative_to(PROJECT_ROOT).as_posix()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            row["elapsed_seconds"] = round(time.perf_counter() - item_started, 6)
            rows.append(row)
            print(f"[{index}/{len(records)}] {record.sample_id}: {row['status']}")

    result_dir = args.results_root / args.model_name
    record_path = result_dir / "inference_records.csv"
    write_records(record_path, rows)
    failed = [row for row in rows if row["status"] != "ok"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "variant": args.variant,
        "project_revision": project_revision(),
        "official_upstream_commit": (DGNO_ROOT / "UPSTREAM_COMMIT").read_text(
            encoding="utf-8"
        ).strip(),
        "manifest": args.manifest.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "weights": weights_path.name,
        "weights_sha256": file_sha256(weights_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "requested_count": len(records),
        "successful_count": len(rows) - len(failed),
        "failed_count": len(failed),
        "total_seconds": round(time.perf_counter() - run_started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "record_file": record_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(f"Inference completed with {len(failed)} failed sample(s)")


if __name__ == "__main__":
    main()
