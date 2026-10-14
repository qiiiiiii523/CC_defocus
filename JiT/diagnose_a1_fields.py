"""Read-only comparison of S1's frozen V3 fields with saved synthetic truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", help="Original three-arm run tag")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--data-root", type=Path)
    p.add_argument("--estimator", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="0: full validation manifest")
    p.add_argument("--examples", type=int, default=8)
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_panel(path, clear, blur, masks, truth, pred, truth16, pred16, sample_id):
    import numpy as np
    from PIL import Image, ImageDraw
    size, label, columns = 256, 24, 6
    canvas = Image.new("RGB", (size * columns, (size + label) * 4 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    def cell(array, x, y, text):
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
        canvas.paste(image.resize((size, size), Image.Resampling.NEAREST), (x * size, y * (size + label) + label))
        draw.text((x * size + 4, y * (size + label) + 4), text, fill="black")
    def probability(a):
        return np.stack((a, .25 + .5 * a, 1 - a), axis=-1)
    def error(a):
        return np.stack((a, np.zeros_like(a), np.zeros_like(a)), axis=-1)
    mask_rgb = np.zeros_like(clear)
    mask_rgb[masks["tissue"]] = (.3, .8, .3)
    mask_rgb[masks["background"]] = (.3, .5, 1)
    mask_rgb[masks["uncertain"]] = (.6, .6, .6)
    cell(clear, 0, 0, "Clear RGB (mask source)")
    cell(blur, 1, 0, "Synthetic blur (estimator input)")
    cell(mask_rgb, 2, 0, "Green=tissue Blue=background Gray=other")
    draw.text((3 * size + 4, label + 12), "All maps use fixed scales [0,1].\nProbability: blue(0) -> yellow(1).\nAbsolute error: black(0) -> red(1).\n16x16 maps enlarged with nearest.\nMasks are heuristic, not labels.", fill="black")
    for c, name in enumerate(("light", "medium", "heavy")):
        for x, (a, text, is_error) in enumerate(((truth[c], "Truth 64", False), (pred[c], "V3 64", False),
                (abs(truth[c] - pred[c]), "Abs error 64", True), (truth16[c], "Truth 16", False),
                (pred16[c], "V3 16", False), (abs(truth16[c] - pred16[c]), "Abs error 16", True))):
            cell(error(a) if is_error else probability(a), x, c + 1, name + " / " + text)
    draw.text((4, canvas.height - 24), sample_id, fill="black")
    canvas.save(path)


def make_report(summary, image_rows, out):
    stats = summary["metrics"]
    def value(res, region, channel, key):
        s = stats.get(res, {}).get(region, {}).get(channel)
        v = s[key]["mean"] if s else None
        return "NA" if v is None else f"{v:.6f}"
    lines = ["# V3 预测场与合成真值诊断", "",
             f"原实验：{summary['source_tag']}；验证样本：{summary['evaluated_count']}；状态：complete。", "",
             "这是估计器概率场检查，不运行 JiT 恢复、不训练，也不评价 PSNR/SSIM/LPIPS。", "",
             "## 1. 三个通道", "",
             "误差单位为混合权重（0～1）。bias=预测－真值；正值表示该通道偏高。各项为每图统计的等权均值。", "",
             "| 分辨率 | 通道 | MAE↓ | bias | 预测空间std | 真值空间std | 空间相关性↑ | 空间MAE收益↑ |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for res in ("native64", "token16"):
        for c in ("light", "medium", "heavy"):
            vals = [value(res, "all", c, k) for k in ("mae", "bias", "pred_spatial_std", "truth_spatial_std", "spatial_corr", "spatial_mae_gain")]
            lines.append("| " + " | ".join([res, c] + vals) + " |")
    lines += ["", "空间相关性在每图每通道内计算，常量场记为 NA，不伪造为0。空间MAE收益=把本图预测替换为其空间均值后的MAE－原预测MAE，正值表示预测的空间起伏降低了误差；这不是恢复指标。区域CSV中的均值与相关性均在对应区域内按面积加权计算。", "",
              "## 2. 组织/背景（颜色阈值辅助分区）", "",
              "| 区域 | 有效图片数 | 平均区域占比（全部图片） | token16轻/中/重MAE |",
              "|---|---:|---:|---|"]
    for region in ("tissue", "background"):
        n = sum(r[region + "_fraction"] > 0 for r in image_rows)
        fraction = sum(r[region + "_fraction"] for r in image_rows) / len(image_rows)
        lines.append(f"| {region} | {n} | {fraction:.2%} | " + " / ".join(value("token16", region, c, "mae") for c in ("light", "medium", "heavy")) + " |")
    lines += ["", "分区来自配对清晰图：background=minRGB≥0.90且HSV饱和度≤0.10；tissue=平均RGB≤0.85或饱和度≥0.15；其余记uncertain。mask按区域面积占比聚合到64/16网格。它不是组织分割真值，需结合 examples/ 拼图检查。", "",
              "纯背景/低纹理区的退化可能从模糊RGB中不可辨识，背景误差大不能单独说明估计器错误；组织区误差也需结合纹理和图像内容解释。", "",
              "## 3. 64→16 的信息变化", "",
              "resize_metrics.csv逐图记录预测/真值的空间std比值和64→16→64往返MAE。复现生产缩放算子bilinear、align_corners=False、无antialias、不额外归一化；诊断统一FP32，不模拟JiT BF16量化、MLP或门控。", "",
              "| 通道 | 预测std比值(16/64) | 真值std比值(16/64) | 预测往返MAE | 真值往返MAE | std比值有效图片数(预测/真值) |",
              "|---|---:|---:|---:|---:|---:|"]
    for channel, entries in summary["resize_metrics"].items():
        values = [entries[k]["mean"] for k in ("pred_std_ratio", "truth_std_ratio", "pred_roundtrip_mae", "truth_roundtrip_mae")]
        counts = str(entries["pred_std_ratio"]["valid_images"]) + "/" + str(entries["truth_std_ratio"]["valid_images"])
        lines.append("| " + channel + " | " + " | ".join("NA" if v is None else f"{v:.6f}" for v in values) + " | " + counts + " |")
    lines += ["", "std比值接近1、往返MAE较小，支持该类平滑场在此缩放中改变较小；比值明显降低、往返MAE较大，说明场受到更大改变。不能仅因token16的预测MAE下降就说信息保留得更好，平滑也会降低误差。", "",
              "## 4. 怎样结合 S0/S1/S2 解读", "",
              "- native64就偏离真值：优先查看具体通道、组织区域及空间标定，给估计器同学这些证据。",
              "- 原场接近真值但缩放明显损失：再检查条件分辨率/表示是否适合该信息。",
              "- token16预测已经接近真值，但S1仍弱于S2：不能仅归咎于估计器，应检查误差类型、训练适应和模型敏感性。",
              "- 高相关性不等于数值准确；低MAE也可能来自场本来接近恒定。需结合bias、空间std、均值对照共同解读。", "",
              "本报告不自动判定责任，不设置任意“合格线”。这里只知道合成混合场真值，不能据此声称验证了真实模糊的物理退化程度。",
              "验证来源group：" + ", ".join(summary["validation_groups"]) + "；同来源切块有相关性，不等于同等数量的独立病例。", "",
              "## 5. 文件", "",
              "summary.json：完整通道/区域聚合统计、有效相关性数量和来源哈希。",
              "per_field_metrics.csv：逐图×分辨率×区域×通道的统计。",
              "per_image.csv：逐图总体与组织MAE、区域占比、样例排序。",
              "resize_metrics.csv：逐图缩放信息变化。",
              "examples/：误差较大、较小及中位附近样例，共享色标拼图和对应数值NPZ。", "",
              "所有评价样本均按原验证清单读取；检查了清晰图、模糊图、真值NPZ哈希。估计器文件SHA与原S1配置一致。没有使用真实验证集作为合成真值。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from a1_synthetic_data import SyntheticPairedDataset, read_truth, sha256_file, validate_truth
    from a1_field_metrics import (MASK_RULE, aggregate_rows, compression_rows, field_rows, region_masks)
    from common_io import load_pair
    from degradation.local_condition import predict_local_from_jit_blur
    from degradation.local_inference import load_local_estimator
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag) or args.limit < 0 or args.examples < 0:
        raise ValueError("Invalid run tag, limit or examples")
    root = args.project_root.resolve()
    cfg_path = root / "checkpoints" / f"jit_a1_exp_s1_{args.tag}" / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("degradation_mode") != "predicted_local" or cfg.get("model") != "JiT-B/16" or cfg.get("img_size") != 256:
        raise ValueError("Expected the original 256px JiT-B/16 S1 predicted_local experiment")
    data_root = (args.data_root or Path(cfg["data_root"])).resolve()
    generation = Path(cfg["generation_config"])
    config = json.loads(generation.read_text(encoding="utf-8"))
    if config.get("status") != "complete" or config["field_size"] != 64 or config["image_size"] != 256:
        raise ValueError("Expected complete 256px synthetic data with native 64px fields")
    if config["channel_order"] != ["light", "medium", "heavy"]:
        raise ValueError("Incorrect generation channel order")
    if sha256_file(generation) != cfg["generation_config_sha256"]:
        raise ValueError("Generation config changed since S1 training")
    manifest = Path(cfg["val_manifest"])
    if sha256_file(manifest) != config["manifest_sha256"]["val"] or sha256_file(manifest) != cfg["val_manifest_sha256"]:
        raise ValueError("Validation manifest changed since generation/training")
    estimator_path = (args.estimator or Path(cfg["degradation_checkpoint"])).resolve()
    estimator_sha = sha256_file(estimator_path)
    if not cfg.get("degradation_checkpoint_sha256") or estimator_sha != cfg["degradation_checkpoint_sha256"]:
        raise ValueError("Estimator file SHA differs from the frozen estimator used by S1")
    dataset = SyntheticPairedDataset(manifest, split="val", training=False, data_root=data_root)
    if len(dataset) != config["val_count"]:
        raise ValueError("Validation count differs from generation config")
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    device = torch.device(args.device)
    estimator = load_local_estimator(estimator_path).to(device).eval().requires_grad_(False)
    rows, resize_rows, image_rows, cache = [], [], [], []
    for n, index in enumerate(indices):
        record = dataset.records[int(index)]
        blur, clear = load_pair(record)
        if blur.shape != (256, 256, 3):
            raise ValueError("Exact 256px paired inputs required; no image resize is allowed")
        truth = read_truth(dataset.truth_paths[record.sample_id])
        if truth.shape != (3, 64, 64):
            raise ValueError("Expected native [3,64,64] saved field")
        # Same RGB normalization, FP32 estimator, and prediction helper as Denoiser.
        blur_jit = torch.from_numpy(blur.transpose(2, 0, 1).copy()).unsqueeze(0).to(device).mul(2).sub(1)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
            pt = predict_local_from_jit_blur(blur_jit.float(), estimator)
            tt = torch.from_numpy(truth).unsqueeze(0).to(device)
            p16 = F.interpolate(pt, size=(16, 16), mode="bilinear", align_corners=False)
            t16 = F.interpolate(tt, size=(16, 16), mode="bilinear", align_corners=False)
            p64 = F.interpolate(p16, size=(64, 64), mode="bilinear", align_corners=False)
            t64 = F.interpolate(t16, size=(64, 64), mode="bilinear", align_corners=False)
        pred, pred16, truth16, rebuilt_p, rebuilt_t = [x[0].cpu().numpy() for x in (pt, p16, t16, p64, t64)]
        validate_truth(pred)
        if pred.shape != truth.shape:
            raise ValueError("Estimator and native truth grids differ")
        masks = region_masks(clear)
        r64 = field_rows(pred, truth, masks, sample_id=record.sample_id, resolution="native64")
        r16 = field_rows(pred16, truth16, masks, sample_id=record.sample_id, resolution="token16")
        rows.extend(r64 + r16)
        resize_rows.extend(compression_rows(pred, truth, pred16, truth16, rebuilt_p, rebuilt_t, sample_id=record.sample_id))
        image_row = {"sample_id": record.sample_id,
                     **{r + "_fraction": float(masks[r].mean()) for r in ("tissue", "background", "uncertain")}}
        for res, entries in (("native64", r64), ("token16", r16)):
            for region in ("all", "tissue", "background"):
                selected = [r["mae"] for r in entries if r["region"] == region]
                image_row[f"{res}_{region}_mae"] = float(np.mean(selected)) if selected else None
        image_rows.append(image_row)
        cache.append((record, pred, truth, pred16, truth16))
        if n % 25 == 0 or n + 1 == count:
            print(f"Validation {n+1}/{count}: {record.sample_id}", flush=True)
    out = args.output_dir.resolve()
    write_csv(out / "per_field_metrics.csv", rows)
    write_csv(out / "per_image.csv", image_rows)
    write_csv(out / "resize_metrics.csv", resize_rows)
    # Select on tissue MAE when present, otherwise whole-image MAE; retain IDs in summary.
    ranking = sorted(range(count), key=lambda i: (image_rows[i]["token16_tissue_mae"]
                      if image_rows[i]["token16_tissue_mae"] is not None else image_rows[i]["token16_all_mae"], image_rows[i]["sample_id"]))
    candidates = ranking[-4:][::-1] + ranking[:2] + ranking[max(0, count//2-1):count//2+1] + ranking
    selected = list(dict.fromkeys(candidates))[:min(args.examples, count)]
    example_ids = []
    for i in selected:
        record, pred, truth, pred16, truth16 = cache[i]
        blur, clear = load_pair(record)
        name = f"{i:03d}_" + hashlib.sha256(record.sample_id.encode()).hexdigest()[:12]
        folder = out / "examples"
        folder.mkdir(exist_ok=True)
        save_panel(folder / (name + ".png"), clear, blur, region_masks(clear), truth, pred, truth16, pred16, record.sample_id)
        np.savez_compressed(folder / (name + ".npz"), sample_id=np.array(record.sample_id),
                            channel_order=np.array("light,medium,heavy"), pred64=pred, truth64=truth,
                            pred16=pred16, truth16=truth16)
        example_ids.append({"sample_id": record.sample_id, "file_stem": name})
    summary = {"status": "complete", "source_tag": args.tag, "evaluated_count": count,
               "full_validation_count": len(dataset), "all_validation": count == len(dataset),
               "generation_config_sha256": sha256_file(generation), "val_manifest_sha256": sha256_file(manifest),
               "s1_run_config_sha256": sha256_file(cfg_path), "estimator_sha256": estimator_sha,
               "estimator_matches_s1": True, "all_pair_and_truth_hashes_verified": True,
               "estimator_precision": "FP32, eval, frozen; same predict_local_from_jit_blur helper",
               "resize": {"native_grid": [64,64], "token_grid": [16,16], "mode": "bilinear",
                          "align_corners": False, "antialias": False, "renormalization": False,
                          "precision": "FP32 adapter-input fields; not a BF16 encoder/output intervention"},
               "mask_rule": MASK_RULE, "aggregation": "equal-image mean; region pixels weighted by block-area fraction; null correlations excluded with counts",
               "validation_groups": sorted({dataset.records[int(i)].sample_id.split('_')[0] for i in indices}),
               "metrics": aggregate_rows(rows), "resize_metrics": {}, "examples": example_ids,
               "source_files_sha256": {name: sha256_file(Path(__file__).parent / name) for name in (
                   "diagnose_a1_fields.py", "a1_field_metrics.py", "degradation/local_condition.py", "degradation/local_model.py")},
               "limitations": ["Heuristic tissue/background, not annotations", "Background blur may be unidentifiable",
                               "Synthetic mixture weights, not real physical degradation labels", "No JiT restoration or within-model causal intervention"]}
    for channel in ("light", "medium", "heavy"):
        entries = [r for r in resize_rows if r["channel"] == channel]
        summary["resize_metrics"][channel] = {}
        for key in entries[0].keys() - {"sample_id", "channel"}:
            values = [r[key] for r in entries if r[key] is not None]
            summary["resize_metrics"][channel][key] = {
                "mean": float(np.mean(values)) if values else None, "valid_images": len(values)}
    write_json(out / "summary.json", summary)
    make_report(summary, image_rows, out)
    print(f"Complete: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    arguments = parser().parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ("launcher.log", "launcher.pid") for p in output.iterdir()):
        raise SystemExit("Use a fresh output directory; existing results are never overwritten")
    write_json(output / "request.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()})
    try:
        main(arguments)
    except Exception as exc:
        write_json(output / "summary.json", {"status": "failed", "reason": str(exc)})
        raise
