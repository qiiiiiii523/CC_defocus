"""Evaluate saved CNSeg center-crop predictions, not the 3DHistech 300-pair protocol."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from dataset_cnseg import CNSegPairedDataset
from nucleus_losses import instance_boundary_mask


def evaluate_cnseg(args):
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    dataset = CNSegPairedDataset(args.val_manifest, split="val", image_size=args.img_size,
                                training=False, root=Path(__file__).resolve().parents[1])
    metric = lpips.LPIPS(net="alex").to("cuda").eval()
    rows = []
    for item in dataset:
        path = args.prediction_root / args.model_name / (item["sample_id"] + ".png")
        with Image.open(path) as image:
            pred = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        target = ((item["clear"]+1)/2).permute(1,2,0).numpy()
        blur = ((item["blur"]+1)/2).permute(1,2,0).numpy()
        if pred.shape != target.shape:
            raise ValueError(f"Prediction geometry differs: {path}")
        labels = item["instance_mask"]
        regions = dict(nucleus_region=labels.numpy()>0,
                       nucleus_boundary=instance_boundary_mask(labels[None])[0].numpy(),
                       background=labels.numpy()==0)
        row = dict(sample_id=item["sample_id"])
        for prefix, array in (("", pred), ("blur_", blur)):
            row[prefix+"psnr"] = float(peak_signal_noise_ratio(target, array, data_range=1))
            row[prefix+"ssim"] = float(structural_similarity(target, array, data_range=1, channel_axis=-1))
            p = torch.from_numpy(array.transpose(2,0,1).copy())[None].cuda()*2-1
            t = item["clear"][None].cuda()
            with torch.no_grad():
                row[prefix+"lpips"] = float(metric(p,t,normalize=False).item())
            current_error = np.abs(array-target).mean(axis=-1)
            for name, mask in regions.items():
                row[prefix+name+"_mae"] = float(current_error[mask].mean()) if mask.any() else None
        row.update(nucleus_pixels=int(regions["nucleus_region"].sum()),
                   boundary_pixels=int(regions["nucleus_boundary"].sum()))
        rows.append(row)
    out = args.results_root / args.model_name
    out.mkdir(parents=True, exist_ok=True)
    with (out/"per_image_metrics.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries={}
    for key in rows[0]:
        if key == "sample_id":
            continue
        values=[r[key] for r in rows if r[key] is not None]
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite CNSeg metric {key}")
        summaries[key]=dict(mean=float(np.mean(values)) if values else None, valid_count=len(values))
    (out/"summary.json").write_text(json.dumps(dict(model_name=args.model_name,
        evaluated_count=len(rows), geometry="center crop; Gaussian blur before crop; no resize",
        nucleus_definition="RGB MAE in instance foreground / inner 4-neighbor boundary, not segmentation accuracy",
        metrics=summaries),indent=2),encoding="utf-8")
    print(f"CNSeg K0 metrics: {out}")
