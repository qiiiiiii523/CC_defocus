"""One-shot real-pair audit of local PSF predictions against fixed/global fields.

The local/global estimators see only the blurred image. The paired clear image
is used solely *after predictions are frozen* to compare reblur explanations;
this pixel discrepancy is not a true optical-PSF error or JiT recovery score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from local_model import LocalBlurEstimator
from model import BlurEstimator
from psf import LEVELS
from spatial_psf import reblur_spatial_mixture


def read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32).div_(255)


def sigma_map(probabilities: torch.Tensor) -> torch.Tensor:
    levels = probabilities.new_tensor([row[1] for row in LEVELS])[None, :, None, None]
    return (probabilities * levels).sum(dim=1)


def interior_mse(candidate: torch.Tensor, target: torch.Tensor, border: int = 12) -> float:
    if min(candidate.shape[-2:]) <= 2 * border:
        raise ValueError("Image too small for interior comparison")
    return float((candidate[..., border:-border, border:-border] - target[..., border:-border, border:-border]).square().mean())


def picture(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


def save_case(path: Path, blur: torch.Tensor, local_reblur: torch.Tensor,
              fixed_reblur: torch.Tensor, local_sigma: torch.Tensor) -> None:
    height, width = blur.shape[-2:]
    severity = ((local_sigma.detach().clamp(1, 3.5) - 1) / 2.5 * 255).byte().cpu().numpy()
    heat = np.stack((severity, np.zeros_like(severity), 255 - severity), axis=-1)
    panels = [picture(x) for x in (blur, local_reblur, fixed_reblur)] + [
        Image.fromarray(heat, "RGB").resize((width, height), Image.Resampling.BILINEAR)
    ]
    board = Image.new("RGB", (4 * width, height))
    for index, panel in enumerate(panels):
        board.paste(panel, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    root = args.project_root.resolve()
    local = LocalBlurEstimator().to(device).eval()
    local.load_state_dict(torch.load(args.local_checkpoint, map_location=device, weights_only=True)["model"])
    global_model = BlurEstimator().to(device).eval()
    global_model.load_state_dict(torch.load(args.global_checkpoint, map_location=device, weights_only=True)["model"])
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.manifest.open("r", encoding="utf-8")]
    if not records or any(row["split"] != "val" or row.get("quality_status") != "pass" for row in records):
        raise ValueError("Only the fixed pass validation manifest is supported")
    with torch.no_grad():
        for index, record in enumerate(records, 1):
            observed = read_rgb(root / record["blur_path"], device)
            clear = read_rgb(root / record["clear_path"], device)
            if observed.shape != clear.shape:
                raise ValueError(f"Shape mismatch: {record['sample_id']}")
            probabilities = local(observed).softmax(dim=1)
            global_probs = global_model(observed).softmax(dim=1)[:, :, None, None]
            fixed_probs = torch.zeros_like(global_probs)
            fixed_probs[:, 0] = 1
            local_reblur = reblur_spatial_mixture(clear, probabilities)
            global_reblur = reblur_spatial_mixture(clear, global_probs)
            fixed_reblur = reblur_spatial_mixture(clear, fixed_probs)
            local_sigma = sigma_map(probabilities)[0]
            rows.append({
                "sample_id": record["sample_id"],
                "mse_local": interior_mse(local_reblur, observed),
                "mse_global": interior_mse(global_reblur, observed),
                "mse_fixed_light": interior_mse(fixed_reblur, observed),
                "mse_identity": interior_mse(clear, observed),
                "predicted_sigma_mean": float(local_sigma.mean()),
                "predicted_sigma_spatial_std": float(local_sigma.std()),
            })
            if index % 50 == 0:
                print(json.dumps({"evaluated": index}), flush=True)
    with (args.output_dir / "real_paired_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    local_errors = np.asarray([row["mse_local"] for row in rows])
    fixed_errors = np.asarray([row["mse_fixed_light"] for row in rows])
    global_errors = np.asarray([row["mse_global"] for row in rows])
    summary = {
        "samples": len(rows),
        "mean_interior_mse_local": float(local_errors.mean()),
        "mean_interior_mse_global": float(global_errors.mean()),
        "mean_interior_mse_fixed_light": float(fixed_errors.mean()),
        "local_better_than_global_count": int((local_errors < global_errors).sum()),
        "local_better_than_fixed_count": int((local_errors < fixed_errors).sum()),
        "mean_local_spatial_sigma_std": float(np.mean([row["predicted_sigma_spatial_std"] for row in rows])),
        "interpretation": "Paired-clear offline reblur fit only; not measured local PSF or restoration PSNR",
    }
    (args.output_dir / "real_paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked = sorted(rows, key=lambda row: row["mse_fixed_light"] - row["mse_local"])
    cases = [("failure", row) for row in ranked[:3]] + [("success", row) for row in ranked[-3:]]
    by_id = {record["sample_id"]: record for record in records}
    with torch.no_grad():
        for kind, row in cases:
            record = by_id[row["sample_id"]]
            observed = read_rgb(root / record["blur_path"], device)
            clear = read_rgb(root / record["clear_path"], device)
            probabilities = local(observed).softmax(dim=1)
            fixed_probs = torch.zeros(1, 3, 1, 1, device=device)
            fixed_probs[:, 0] = 1
            save_case(args.output_dir / "examples" / f"{kind}_{row['sample_id']}.png", observed[0],
                      reblur_spatial_mixture(clear, probabilities)[0],
                      reblur_spatial_mixture(clear, fixed_probs)[0],
                      sigma_map(probabilities)[0])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
