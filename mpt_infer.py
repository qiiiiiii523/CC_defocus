"""Run the official MPT checkpoint on the frozen project manifest.

This is an I/O and device adapter only. It imports the official MPT model
without changing its architecture and saves predictions through common_io.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common_io import PROJECT_ROOT, load_manifest, read_rgb_image, save_prediction


DEFAULT_MANIFEST = PROJECT_ROOT / "prepared/manifests/debug_val.jsonl"
DEFAULT_MPT_REPO = PROJECT_ROOT.parent / "external/MPT-CataBlur"
DEFAULT_CHECKPOINT = PROJECT_ROOT.parent / "weights/mpt/official/3dhistech.pytorch"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RUN_LOG = PROJECT_ROOT / "results/mpt/inference_run.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mpt-repo", type=Path, default=DEFAULT_MPT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_model(mpt_repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(mpt_repo.resolve()))
    from models.MPT import MPT

    raw_state = torch.load(checkpoint, map_location="cpu")
    state = {}
    for key, value in raw_state.items():
        if key.startswith("module.Network."):
            key = key[len("module.Network.") :]
        elif key.startswith("Network."):
            key = key[len("Network.") :]
        state[key] = value

    model = MPT()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    device = choose_device(args.device)
    records = load_manifest(
        args.manifest, root=PROJECT_ROOT, expected_split="val"
    )
    selected = records if args.limit is None else records[: args.limit]
    model = load_model(args.mpt_repo, args.checkpoint, device)

    per_image = []
    started = time.perf_counter()
    with torch.inference_mode():
        for record in selected:
            image = read_rgb_image(
                record.blur_path, sample_id=record.sample_id, role="blur"
            )
            tensor = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            ).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            item_started = time.perf_counter()
            result = model(tensor)["result"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - item_started

            prediction = result[0].detach().cpu().numpy().transpose(1, 2, 0)
            output_path = save_prediction(
                prediction,
                model_name="mpt",
                sample_id=record.sample_id,
                expected_hw=image.shape[:2],
                output_root=args.output_root,
            )
            per_image.append(
                {
                    "sample_id": record.sample_id,
                    "seconds": elapsed,
                    "shape": list(prediction.shape),
                    "raw_min": float(prediction.min()),
                    "raw_max": float(prediction.max()),
                    "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
                }
            )
            print(f"{record.sample_id}: {elapsed:.3f}s -> {output_path}")

    total = time.perf_counter() - started
    report = {
        "model": "MPT",
        "result_type": "official 3DHistech pretrained weight inference",
        "mpt_commit": "3078ea354b547f4c403527f12c3f55a436bddaff",
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "torch": torch.__version__,
        "sample_count": len(per_image),
        "total_seconds": total,
        "mean_model_seconds": float(np.mean([x["seconds"] for x in per_image])),
        "successful": [x["sample_id"] for x in per_image],
        "failed": [],
        "per_image": per_image,
    }
    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    args.run_log.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Run log: {args.run_log}")


if __name__ == "__main__":
    main()
