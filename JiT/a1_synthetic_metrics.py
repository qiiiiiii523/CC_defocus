"""Independent synthetic evaluation with the shared RGB/LPIPS definitions."""
from __future__ import annotations
import csv
import json


def run_metrics(args):
    import numpy as np
    import torch
    import lpips
    import util.misc as misc
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from common_io import load_manifest, read_rgb_image
    from evaluate import _compute_lpips, _require_valid_range, _require_finite_metric
    if not args.run_metrics:
        return
    if not misc.is_main_process():
        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        return
    records = load_manifest(args.val_manifest, root=args.data_root, expected_split="val")
    if len(records) != args.expected_val_pairs:
        raise ValueError("Synthetic evaluation count mismatch")
    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    for record in records:
        prediction = read_rgb_image(args.prediction_root / args.model_name / f"{record.sample_id}.png",
                                    sample_id=record.sample_id, role="prediction")
        clear = read_rgb_image(record.clear_path, sample_id=record.sample_id, role="clear")
        if prediction.shape != clear.shape:
            raise ValueError(f"Prediction size mismatch: {record.sample_id}")
        for image, role in ((prediction, "prediction"), (clear, "reference")):
            _require_valid_range(image, sample_id=record.sample_id, role=role)
        values = {"psnr": float(peak_signal_noise_ratio(clear, prediction, data_range=1)),
                  "ssim": float(structural_similarity(clear, prediction, data_range=1, channel_axis=-1)),
                  "lpips": _compute_lpips(metric, device, prediction, clear)}
        for name, value in values.items():
            _require_finite_metric(record.sample_id, name, value)
        rows.append({"sample_id": record.sample_id, **values})
    summary = {"model_name": args.model_name, "evaluation_domain": args.evaluation_domain,
               "condition_source": args.condition_source,
               "manifest": str(args.val_manifest), "evaluated_count": len(rows),
               "generation_config_sha256": args.generation_config_sha256,
               "sampling": {"weights": args.eval_weights, "method": args.sampling_method,
                            "steps": args.num_sampling_steps, "seed": args.eval_seed},
               "metrics": {name: {"mean": float(np.mean([r[name] for r in rows])),
                                   "std": float(np.std([r[name] for r in rows], ddof=0))}
                           for name in ("psnr", "ssim", "lpips")},
               "settings": {"rgb": True, "data_range": 1, "crop_border": 0, "resize": False,
                            "ssim_channel_axis": -1, "lpips_backbone": "alex",
                            "lpips_input_range": [-1, 1], "lpips_normalize": False,
                            "std_ddof": 0, "lpips_device": str(device)}}
    directory = args.results_root / args.model_name
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "per_image_metrics.csv"
    pending = csv_path.with_suffix(".csv.tmp")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "psnr", "ssim", "lpips"))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(csv_path)
    json_path = directory / "summary.json"
    pending = json_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(json_path)
    marker = args.output_dir / "final_eval_config.json"
    marker.write_text(json.dumps({"status": "complete", **summary["sampling"],
                                  "evaluated_count": len(rows)}, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic metrics: {json_path}")
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
