"""Run saved-field reconstruction and three NON-EMA evaluations; never train."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ARMS = ("s0", "s1", "s2")
MODES = {"s0": "none", "s1": "predicted_local", "s2": "oracle_local"}


def build_eval_command(cfg, checkpoint, out, name):
    fields = ("model", "img_size", "init_mode", "class_num", "attn_dropout", "proj_dropout",
              "P_mean", "P_std", "noise_scale", "t_eps", "label_drop_prob", "lambda_pix",
              "charbonnier_eps", "ema_decay1", "ema_decay2", "identity_ratio", "sampling_method",
              "num_sampling_steps", "cfg", "interval_min", "interval_max", "eval_seed",
              "eval_batch_size", "num_workers", "seed", "data_root", "generation_config",
              "train_manifest", "val_manifest", "degradation_mode")
    command = [sys.executable, "-u", str(Path(__file__).with_name("main_a1_exp.py")),
               "--eval_only", "--resume", str(checkpoint), "--eval_weights", "model", "--run_metrics"]
    for key in fields:
        if cfg.get(key) is not None:
            command.extend(["--" + key, str(cfg[key])])
    if cfg.get("amp_bf16"):
        command.append("--amp_bf16")
    command.extend(["--model_name", name, "--output_dir", str(out / "eval_state" / name),
                    "--prediction_root", str(out / "outputs"), "--results_root", str(out / "model_results")])
    return command


def read_metrics(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapped = {r["sample_id"]: {k: float(r[k]) for k in ("psnr", "ssim", "lpips")} for r in rows}
    if len(mapped) != len(rows) or not rows or not all(math.isfinite(v) for row in mapped.values() for v in row.values()):
        raise ValueError(f"Duplicate IDs, empty or nonfinite metrics: {path}")
    return mapped


def paired_difference(left, right):
    if left.keys() != right.keys():
        raise ValueError("Cannot compare different sample sets")
    result = {}
    for metric in ("psnr", "ssim", "lpips"):
        values = [left[s][metric] - right[s][metric] for s in sorted(left)]
        result[metric] = {"delta": math.fsum(values) / len(values),
                          "wins": sum((d > 0 if metric != "lpips" else d < 0) for d in values),
                          "losses": sum((d < 0 if metric != "lpips" else d > 0) for d in values)}
    return result


def main(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("Invalid original run tag")
    if len(set(args.gpus)) != 3 or not all(re.fullmatch(r"[0-9]+", gpu) for gpu in args.gpus):
        raise ValueError("Provide three distinct GPU indices, e.g. --gpus 0 1 2")
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(root / "reports"):
        raise ValueError("Checks must write to a new folder under project reports/")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "check_summary.json").exists() or (out / "evaluation_commands.json").exists():
        raise FileExistsError("Use a new check output directory")
    configs, checkpoints, original_dirs = {}, {}, {}
    for arm in ARMS:
        name = f"jit_a1_exp_{arm}_{args.tag}"
        run = root / "checkpoints" / name
        cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        checkpoint = run / "checkpoint-last.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if cfg["degradation_mode"] != MODES[arm] or not cfg.get("synthetic_experiment"):
            raise ValueError(f"Unexpected source mode: {arm}")
        marker = json.loads((run / "final_eval_config.json").read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("weights") != "ema2":
            raise ValueError(f"Original EMA2 evaluation incomplete: {arm}")
        if cfg["sampling_method"] != "euler" or cfg["num_sampling_steps"] != 1:
            raise ValueError("This check expects the original Euler-one-step protocol")
        configs[arm], checkpoints[arm] = cfg, checkpoint
        original_dirs[arm] = root / "results" / name
        reference = out / "original_ema2" / arm
        reference.mkdir(parents=True)
        for filename in ("summary.json", "per_image_metrics.csv"):
            shutil.copyfile(original_dirs[arm] / filename, reference / filename)
        shutil.copyfile(run / "run_config.json", reference / "run_config.json")
    for key in ("generation_config_sha256", "train_manifest_sha256", "val_manifest_sha256",
                "a0_checkpoint_sha256", "eval_seed", "amp_bf16", "eval_batch_size"):
        if len({json.dumps(configs[a][key], sort_keys=True) for a in ARMS}) != 1:
            raise ValueError(f"Original runs differ in {key}")
    cfg = configs["s0"]
    reblur_command = [sys.executable, "-u", str(Path(__file__).with_name("check_a1_reblur.py")),
                     "--project-root", str(root), "--data-root", cfg["data_root"],
                     "--generation-config", cfg["generation_config"],
                     "--train-manifest", cfg["train_manifest"], "--val-manifest", cfg["val_manifest"],
                     "--output-dir", str(out / "reblur"), "--samples-per-split", str(args.samples_per_split)]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    print("Check 1: reconstruct saved fields, then check 2: parallel model evaluation", flush=True)
    with (out / "reblur.log").open("w", encoding="utf-8") as log:
        reblur_returncode = subprocess.run(reblur_command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    # Keep the two checks independent: even a reconstruction mismatch is reported
    # alongside all three evaluations rather than silently cancelling them.
    processes, handles, commands, names, launch_errors = {}, {}, {}, {}, {}
    try:
        for arm, gpu in zip(ARMS, args.gpus):
            name = f"jit_a1_exp_{arm}_{args.tag}_modelcheck_{out.name}"
            command = build_eval_command(configs[arm], checkpoints[arm], out, name)
            names[arm], commands[arm] = name, command
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            handles[arm] = (out / f"{arm}_model.log").open("w", encoding="utf-8")
            try:
                processes[arm] = subprocess.Popen(command, cwd=root / "JiT", env=env,
                                                  stdout=handles[arm], stderr=subprocess.STDOUT)
            except OSError as exc:
                processes[arm] = None
                launch_errors[arm] = str(exc)
            print(f"{arm}: GPU {gpu}, eval_only/model; {out / (arm + '_model.log')}", flush=True)
        (out / "evaluation_commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        codes = {arm: process.wait() if process is not None else 1 for arm, process in processes.items()}
    finally:
        for handle in handles.values(): handle.close()
    report = {"status": "complete" if reblur_returncode == 0 and all(c == 0 for c in codes.values()) else "failed",
              "source_tag": args.tag, "reblur_returncode": reblur_returncode,
              "model_evaluation_returncodes": codes, "launch_errors": launch_errors,
              "output_dir": str(out)}
    results, metric_rows = {}, {}
    for arm in ARMS:
        if codes[arm] != 0:
            continue
        for weights, directory in (("ema2", out / "original_ema2" / arm),
                                    ("model", out / "model_results" / names[arm])):
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if (summary["generation_config_sha256"] != configs[arm]["generation_config_sha256"]
                    or summary["condition_source"] != configs[arm]["condition_source"]
                    or summary["sampling"] != {"weights": weights, "method": "euler", "steps": 1, "seed": configs[arm]["eval_seed"]}):
                raise ValueError(f"Unexpected evaluation protocol: {arm}/{weights}")
            if weights == "model" and summary["settings"] != results[arm + "_ema2"]["settings"]:
                raise ValueError(f"Metric settings changed: {arm}")
            rows = read_metrics(directory / "per_image_metrics.csv")
            if len(rows) != configs[arm]["val_pairs"]:
                raise ValueError(f"Incomplete evaluation: {arm}/{weights}")
            for key in ("psnr", "ssim", "lpips"):
                if abs(math.fsum(r[key] for r in rows.values()) / len(rows) - summary["metrics"][key]["mean"]) > 1e-9:
                    raise ValueError("CSV and summary disagree")
            results[arm + "_" + weights] = summary
            metric_rows[arm + "_" + weights] = rows
    report["results"] = {name: s["metrics"] for name, s in results.items()}
    if len(results) == 6:
        pairs = {}
        for weights in ("ema2", "model"):
            for left, right in (("s1", "s0"), ("s2", "s0"), ("s2", "s1")):
                pairs[f"{left}-{right}_{weights}"] = paired_difference(metric_rows[left + "_" + weights], metric_rows[right + "_" + weights])
        for arm in ARMS:
            pairs[arm + "_model-ema2"] = paired_difference(metric_rows[arm + "_model"], metric_rows[arm + "_ema2"])
        report["paired_comparisons"] = pairs
    text = ["# 真值配对与非EMA评价检查", "", f"原实验：{args.tag}；状态：{report['status']}", "",
            f"合成复现退出码：{reblur_returncode}（详情见reblur/summary.json）", "",
            "| 组别 | 权重 | PSNR | SSIM | LPIPS |", "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        for weights in ("ema2", "model"):
            key = arm + "_" + weights
            if key in results:
                m = results[key]["metrics"]
                text.append(f"| {arm.upper()} | {weights} | {m['psnr']['mean']:.6f} | {m['ssim']['mean']:.6f} | {m['lpips']['mean']:.6f} |")
    text.extend(["", "请比较同一权重口径下S2−S0的变化，不能只看S2自身是否提高。", "两项均为已有数据/权重的只读检查，没有重新训练。"])
    (out / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "check_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generation = Path(cfg["generation_config"])
    shutil.copyfile(generation, out / "generation_config.json")
    print(f"Finished: {out / 'comparison.md'}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", nargs=3, default=["0", "1", "2"])
    p.add_argument("--samples-per-split", type=int, default=16)
    main(p.parse_args())
