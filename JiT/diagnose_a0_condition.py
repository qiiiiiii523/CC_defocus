"""Read-only A0 condition/ODE diagnostic; no training or optimizer updates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=None)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--weights', choices=('model', 'ema1', 'ema2'), default='ema2')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--sample_ids', nargs='+', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=None, help='Default: saved sampling steps, otherwise 50.')
    p.add_argument('--times', nargs='+', type=float, default=(0., .1, .3, .5, .8, .95))
    p.add_argument('--amp_bf16', action='store_true')
    p.add_argument('--enable_compile', action='store_true', help='Off by default to avoid compile warmup.')
    p.add_argument('--output_dir', type=Path, required=True)
    return p


DEFAULTS = dict(model='JiT-B/16', img_size=256, class_num=1000,
                attn_dropout=0., proj_dropout=0., label_drop_prob=0.,
                P_mean=-.8, P_std=.8, t_eps=.05, noise_scale=1.,
                lambda_pix=1., charbonnier_eps=.001,
                ema_decay1=.9999, ema_decay2=.9996, sampling_method='heun',
                num_sampling_steps=50, cfg=1., interval_min=0., interval_max=1.)


@contextmanager
def condition_off(encoder, enabled, torch):
    """Disable only the condition feature, not the RGB state or time pathway."""
    handle = None
    if enabled:
        handle = encoder.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_indices(records, args):
    if args.sample_ids:
        lookup = {r.sample_id: i for i, r in enumerate(records)}
        if len(set(args.sample_ids)) != len(args.sample_ids):
            raise ValueError('sample_ids must be distinct')
        missing = set(args.sample_ids) - set(lookup)
        if missing:
            raise ValueError(f'Unknown sample IDs: {sorted(missing)}')
        indices = [lookup[s] for s in args.sample_ids]
    else:
        if not 2 <= args.num_samples <= len(records):
            raise ValueError('num_samples must be between 2 and manifest size')
        indices = [i * (len(records) - 1) // (args.num_samples - 1)
                   for i in range(args.num_samples)]
    if len(indices) < 2:
        raise ValueError('At least two distinct samples are needed for shuffled conditions')
    return indices


def main(args):
    # Read-only inference by default. Disabling compile does not change weights.
    if not args.enable_compile:
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    jit_root = Path(__file__).resolve().parent
    project_root = jit_root.parent
    sys.path.insert(0, str(jit_root))
    sys.path.insert(1, str(project_root))
    from dataset_restoration import A0PairedDataset
    from denoiser import Denoiser
    from common_io import save_prediction

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the GPU server or --device cpu with dependencies installed')
    if args.amp_bf16 and args.device != 'cuda':
        raise ValueError('amp_bf16 is only supported here with --device cuda')
    if args.amp_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU does not support BF16')
    if not args.times or any(not 0 <= t < 1 for t in args.times):
        raise ValueError('Diagnostic times must be in [0,1); never use t=1')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory is not empty; choose a new diagnostic directory')

    # Only load trusted project checkpoints: weights_only=False permits saved argparse args.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args')
    saved = saved if isinstance(saved, dict) else vars(saved) if saved is not None else {}
    config = {k: saved.get(k, v) for k, v in DEFAULTS.items()}
    if args.steps is not None:
        config['num_sampling_steps'] = args.steps
    if config['num_sampling_steps'] < 1:
        raise ValueError('Sampling steps must be positive')
    model = Denoiser(SimpleNamespace(**config))
    key = {'model': 'model', 'ema1': 'model_ema1', 'ema2': 'model_ema2'}[args.weights]
    if key not in checkpoint:
        raise KeyError(f'Checkpoint is missing {key}')
    model.load_state_dict(checkpoint[key], strict=True)
    del checkpoint
    device = torch.device(args.device)
    model.to(device).eval()
    model.requires_grad_(False)
    torch.manual_seed(args.seed)
    manifest = args.manifest or project_root / 'prepared/manifests/debug_val.jsonl'
    if not manifest.is_absolute():
        manifest = project_root / manifest
    manifest = manifest.resolve()
    dataset = A0PairedDataset(manifest, split=args.split, image_size=config['img_size'],
                              training=False, identity_ratio=0., root=project_root)
    indices = select_indices(dataset.records, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.net.blur_condition_encoder
    gate = float(encoder.gate.detach().cpu())
    print(f'Diagnostic only: weights={args.weights}, gate={gate:.6f}, samples={len(indices)}, device={device}')

    def autocast():
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=args.amp_bf16)

    def to_rgb(tensor):
        return (tensor.float() + 1.) / 2.

    def error(prediction, target):
        # Unclamped FP32 values: do not let saving-time clipping hide changes.
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Non-finite diagnostic output')
        delta = to_rgb(prediction) - to_rgb(target)
        return float(delta.abs().mean()), float(delta.square().mean())

    def delta_stats(a, b):
        delta = (to_rgb(a) - to_rgb(b)).abs()
        return float(delta.mean()), float(delta.max())

    def save(tensor, variant, sample_id):
        array = to_rgb(tensor)[0].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
        save_prediction(np.asarray(array, dtype=np.float32), model_name=variant,
                        sample_id=sample_id, expected_hw=(config['img_size'],)*2,
                        output_root=args.output_dir / 'images')

    generation_rows, endpoint_rows = [], []
    with torch.inference_mode():
        for position, index in enumerate(indices):
            item = dataset[index]
            donor_index = indices[(position + 1) % len(indices)]
            donor = dataset[donor_index]
            sample_id, donor_id = item['sample_id'], donor['sample_id']
            blur = item['blur'].unsqueeze(0).to(device)
            clear = item['clear'].unsqueeze(0).to(device)
            wrong = donor['blur'].unsqueeze(0).to(device)
            digest = hashlib.sha256(f'{args.seed}:{sample_id}'.encode()).digest()
            seed = int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            noise = torch.randn(blur.shape, generator=generator, dtype=torch.float32).to(device)
            # RGB gray=0.5 -> JiT zero; RGB white=1 -> JiT one. Neither is branch removal.
            cases = {'correct': blur, 'shuffled': wrong, 'gray': torch.zeros_like(blur),
                     'white': torch.ones_like(blur), 'branch_off': blur}
            restored = {}
            for variant, condition in cases.items():
                print(f'{position+1}/{len(indices)} {sample_id}: ODE {variant}', flush=True)
                with condition_off(encoder, variant == 'branch_off', torch), autocast():
                    prediction = model.generate(condition, noise=noise.clone()).float()
                restored[variant] = prediction
                mae, mse = error(prediction, clear)
                difference, max_difference = delta_stats(prediction, restored['correct'])
                generation_rows.append(dict(sample_id=sample_id, donor_id=donor_id,
                    variant=variant, seed=seed, output_mae_vs_correct=difference,
                    output_max_abs_vs_correct=max_difference, endpoint_mae_vs_clear=mae,
                    endpoint_mse_vs_clear=mse,
                    outside_rgb01_fraction=float(((to_rgb(prediction)<0)|(to_rgb(prediction)>1)).float().mean())))
                save(prediction, variant, sample_id)

            # Teacher-forced, fixed state: x is used ONLY for this offline diagnostic.
            # All cases see the same z_t; this isolates instantaneous condition response.
            labels = model._null_labels(1, device)
            for time in args.times:
                t = torch.tensor([time], device=device, dtype=torch.float32)
                state = time * clear + (1-time) * config['noise_scale'] * noise
                endpoints = {}
                for variant, condition in cases.items():
                    with condition_off(encoder, variant == 'branch_off', torch), autocast():
                        prediction = model.net(state, t, labels, blur=condition, degradation=None).float()
                    endpoints[variant] = prediction
                    mae, mse = error(prediction, clear)
                    difference, max_difference = delta_stats(prediction, endpoints['correct'])
                    endpoint_rows.append(dict(sample_id=sample_id, donor_id=donor_id,
                        time=time, variant=variant, seed=seed,
                        endpoint_mae_vs_correct=difference,
                        endpoint_max_abs_vs_correct=max_difference,
                        endpoint_mae_vs_clear=mae, endpoint_mse_vs_clear=mse))

            save(blur, 'blur', sample_id)
            save(clear, 'clear', sample_id)
            save(wrong, 'donor_blur', sample_id)
            tiles = [('blur', blur), ('clear_reference', clear), ('donor_blur', wrong)] + list(restored.items())
            size, caption = config['img_size'], 24
            panel = Image.new('RGB', (size*4, (size+caption)*2), 'white')
            draw = ImageDraw.Draw(panel)
            for tile, (label, tensor) in enumerate(tiles):
                array = to_rgb(tensor)[0].clamp(0,1).permute(1,2,0).cpu().numpy()
                image = Image.fromarray(np.rint(array*255).astype(np.uint8))
                x, y = (tile%4)*size, (tile//4)*(size+caption)
                draw.text((x+5, y+4), label, fill='black')
                panel.paste(image, (x,y+caption))
            panel_dir = args.output_dir / 'panels'
            panel_dir.mkdir(exist_ok=True)
            panel.save(panel_dir / f'{sample_id}_panel.png')

    write_csv(args.output_dir / 'generation_metrics.csv', generation_rows)
    write_csv(args.output_dir / 'fixed_state_metrics.csv', endpoint_rows)
    averages = {}
    for variant in cases:
        subset = [r for r in generation_rows if r['variant']==variant]
        averages[variant] = {key: sum(r[key] for r in subset)/len(subset)
                            for key in ('output_mae_vs_correct', 'endpoint_mae_vs_clear',
                                        'endpoint_mse_vs_clear', 'outside_rgb01_fraction')}
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), weights=args.weights,
        init_mode=saved.get('init_mode', 'unknown'), model_config=config,
        manifest=str(Path(manifest).resolve()),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        sample_ids=[dataset.records[i].sample_id for i in indices], seed=args.seed,
        times=args.times, device=str(device), amp_bf16=args.amp_bf16,
        compile_enabled=args.enable_compile, gate=gate,
        generation_means=averages,
        warning='Diagnostic subset, not formal 300-pair evaluation. Constant/shuffled inputs are out-of-distribution. Fixed states use clear references and are not deployment inputs.')
    (args.output_dir / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished. Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main(parser().parse_args())
