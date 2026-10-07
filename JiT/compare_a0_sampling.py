"""Sequential, evaluation-only A0 sampling comparison on the GPU server.

This runner uses only the standard library. Child evaluations require the JiT
environment. It never trains, updates weights, or modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


JIT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = JIT_ROOT.parent


def get_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weights', choices=('model', 'ema1', 'ema2'), default='ema2')
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 10, 20, 50])
    parser.add_argument('--method', choices=('heun', 'euler'), default='heun')
    parser.add_argument('--manifest', type=Path,
                        default=PROJECT_ROOT / 'prepared/manifests/debug_val.jsonl')
    parser.add_argument('--experiment_name', default=None,
                        help='Default: checkpoint parent directory name; change this to rerun safely.')
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp_bf16', action='store_true')
    parser.add_argument('--model', default='JiT-B/16')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--noise_scale', type=float, default=1.0)
    parser.add_argument('--t_eps', type=float, default=0.05,
                        help='Keep the original denominator floor; this sweep does not change it.')
    return parser


def checkpoint_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main(args):
    checkpoint = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    entry = JIT_ROOT / 'main_restoration.py'
    for path in (checkpoint, manifest, entry):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gpu < 0 or args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid GPU, batch size or worker count')
    if not args.steps or len(set(args.steps)) != len(args.steps) or any(s < 1 for s in args.steps):
        raise ValueError('steps must be distinct positive integers')
    if not math.isfinite(args.noise_scale) or args.noise_scale <= 0:
        raise ValueError('noise_scale must be finite and positive')
    if not math.isfinite(args.t_eps) or args.t_eps <= 0:
        raise ValueError('t_eps must be finite and positive')
    experiment = args.experiment_name or checkpoint.parent.name
    if not experiment or experiment in ('.', '..') or '/' in experiment or '\\' in experiment:
        raise ValueError('experiment_name must be a single safe path component')

    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines()
               if line.strip()]
    if len(records) != 300 or any(r.get('split') != 'val' for r in records):
        raise ValueError('The shared metric evaluator requires exactly 300 validation records')
    if len({r['sample_id'] for r in records}) != 300:
        raise ValueError('Validation sample IDs are not unique')

    prefix = f'{experiment}_sampling_{args.weights}_{args.method}'
    report = PROJECT_ROOT / 'reports' / prefix
    names = {step: f'{prefix}_s{step}' for step in args.steps}
    # Refuse reuse even for an empty previous output directory: no silent mixing.
    targets = [report]
    for name in names.values():
        targets.extend((PROJECT_ROOT / 'outputs' / name, PROJECT_ROOT / 'results' / name))
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError('Existing comparison paths; use a new --experiment_name:\n' + '\n'.join(existing))

    signature = checkpoint_signature(checkpoint)
    report.mkdir(parents=True)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_signature=signature,
                    manifest=str(manifest),
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    gpu=args.gpu, weights=args.weights, steps=args.steps,
                    sampling_method=args.method, seed=args.seed,
                    eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
                    amp_bf16=args.amp_bf16, model=args.model, img_size=args.img_size,
                    noise_scale=args.noise_scale, t_eps=args.t_eps,
                    status='running', completed=[],
                    note='No training. Same per-sample starting noise. For one step, JiT uses its final Euler step regardless of the method label. Elapsed time includes loading, compilation, sampling, saving and metrics.')
    write_json(report / 'comparison_config.json', metadata)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    rows = []
    settings = None
    try:
        for step in args.steps:
            if checkpoint_signature(checkpoint) != signature:
                raise RuntimeError('Checkpoint changed during comparison; do not train into this checkpoint path')
            name = names[step]
            command = [sys.executable, '-u', str(entry), '--eval_only',
                       '--resume', str(checkpoint), '--val_manifest', str(manifest),
                       '--expected_val_pairs', '300', '--eval_weights', args.weights,
                       '--model', args.model, '--img_size', str(args.img_size),
                       '--noise_scale', str(args.noise_scale), '--t_eps', str(args.t_eps),
                       '--sampling_method', args.method, '--num_sampling_steps', str(step),
                       '--seed', str(args.seed), '--eval_seed', str(args.seed),
                       '--eval_batch_size', str(args.eval_batch_size),
                       '--num_workers', str(args.num_workers), '--model_name', name,
                       '--output_dir', str(report / f'runtime_s{step}'),
                       '--prediction_root', str(PROJECT_ROOT / 'outputs'),
                       '--results_root', str(PROJECT_ROOT / 'results'), '--run_metrics']
            if args.amp_bf16:
                command.append('--amp_bf16')
            metadata['active_step'] = step
            write_json(report / 'comparison_config.json', metadata)
            print(f'Start {step} steps; weights={args.weights}; GPU={args.gpu}; log={report / f"steps_{step}.log"}', flush=True)
            started = time.perf_counter()
            with (report / f'steps_{step}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=JIT_ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f'Evaluation failed at {step} steps (exit {result.returncode}); inspect steps_{step}.log')
            if checkpoint_signature(checkpoint) != signature:
                raise RuntimeError('Checkpoint changed while evaluation was running')
            summary_path = PROJECT_ROOT / 'results' / name / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary['evaluated_count'] != 300:
                raise RuntimeError('Evaluation did not finish all 300 pairs')
            if settings is None:
                settings = summary['settings']
            elif summary['settings'] != settings:
                raise RuntimeError('Metric settings differ between runs')
            row = dict(steps=step, method=args.method, weights=args.weights,
                       model_name=name, evaluated_count=300,
                       total_wall_seconds=round(time.perf_counter()-started, 3))
            for metric in ('psnr', 'ssim', 'lpips'):
                for statistic in ('mean', 'std'):
                    value = float(summary['metrics'][metric][statistic])
                    if not math.isfinite(value):
                        raise RuntimeError(f'Non-finite {metric} {statistic}')
                    row[f'{metric}_{statistic}'] = value
            rows.append(row)
            with (report / 'sampling_summary.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata['completed'].append(step)
            write_json(report / 'comparison_config.json', metadata)
            print(f'Finished {step}: PSNR={row["psnr_mean"]:.4f} SSIM={row["ssim_mean"]:.5f} LPIPS={row["lpips_mean"]:.5f}', flush=True)
        metadata['status'] = 'complete'
        metadata.pop('active_step', None)
        metadata['metric_settings'] = settings
        write_json(report / 'comparison_config.json', metadata)
    except BaseException as exc:
        metadata['status'] = 'failed_or_interrupted'
        metadata['error'] = str(exc)
        write_json(report / 'comparison_config.json', metadata)
        raise
    print(f'All sampling evaluations complete: {report}', flush=True)


if __name__ == '__main__':
    main(get_parser().parse_args())
