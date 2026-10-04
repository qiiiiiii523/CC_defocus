"""Train and restore with JiT A0 on the fixed 3DHistech manifests.

This is intentionally separate from ``main_jit.py``, which remains the
official ImageNet class-conditional entry point.
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter


JIT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = JIT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.misc as misc
from dataset_restoration import A0PairedDataset
from denoiser import Denoiser
from engine_restoration import restore_fixed_validation, train_one_epoch_a0


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("JiT A0 paired restoration")

    # Architecture and official JiT flow settings.
    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--class_num", default=1000, type=int)
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--label_drop_prob", default=0.0, type=float)

    # Optimization.
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=5e-5, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--lr_schedule", default="constant")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)
    parser.add_argument("--amp_bf16", action="store_true")
    parser.add_argument("--identity_ratio", default=0.10, type=float)
    parser.add_argument(
        "--lambda_pix", default=1.0, type=float,
        help="Charbonnier pixel-loss weight; 0 reproduces the flow-only objective.",
    )
    parser.add_argument(
        "--charbonnier_eps", default=1e-3, type=float,
        help="Positive Charbonnier smoothing constant on the RGB [0,1] scale.",
    )
    parser.add_argument(
        "--overfit_samples",
        default=0,
        type=int,
        help=(
            "Use a deterministic subset of the fixed 2,000-pair training set "
            "for the A0-O overfit check; 0 uses all pairs, otherwise use 1-32."
        ),
    )

    # Sampling and fixed validation restoration.
    parser.add_argument("--sampling_method", default="heun", choices=("euler", "heun"))
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=1.0, type=float)
    parser.add_argument("--interval_min", default=0.0, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--eval_freq",
        default=0,
        type=int,
        help="Restore all 300 validation pairs every N epochs; 0 means final epoch only.",
    )
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument(
        "--eval_seed",
        default=0,
        type=int,
        help="Fixed per-sample validation noise seed, independent of batch/GPU count.",
    )
    parser.add_argument("--model_name", default="jit_a0")
    parser.add_argument("--run_metrics", action="store_true")

    # Fixed manifests and output paths.
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=PROJECT_ROOT / "prepared" / "manifests" / "debug_train.jsonl",
    )
    parser.add_argument(
        "--val_manifest",
        type=Path,
        default=PROJECT_ROOT / "prepared" / "manifests" / "debug_val.jsonl",
    )
    parser.add_argument(
        "--prediction_root", type=Path, default=PROJECT_ROOT / "outputs"
    )
    parser.add_argument(
        "--results_root", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "jit_a0"
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=JIT_ROOT / "pretrained" / "jit-b-16" / "checkpoint-last.pth",
        help="Official JiT checkpoint used to initialize A0.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="A0 checkpoint file or directory containing checkpoint-last.pth.",
    )
    parser.add_argument("--save_last_freq", default=5, type=int)
    parser.add_argument("--log_freq", default=20, type=int)

    # Runtime and distributed training.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def _checkpoint_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path / "checkpoint-last.pth" if path.is_dir() else path


def _initialize_ema(model_without_ddp) -> None:
    model_without_ddp.ema_params1 = [
        parameter.detach().clone() for parameter in model_without_ddp.parameters()
    ]
    model_without_ddp.ema_params2 = [
        parameter.detach().clone() for parameter in model_without_ddp.parameters()
    ]


def _load_ema(model_without_ddp, checkpoint, device) -> None:
    for key in ("model_ema1", "model_ema2"):
        if key not in checkpoint:
            raise KeyError(f"A0 resume checkpoint is missing {key}")
    model_without_ddp.ema_params1 = [
        checkpoint["model_ema1"][name].to(device=device)
        for name, _parameter in model_without_ddp.named_parameters()
    ]
    model_without_ddp.ema_params2 = [
        checkpoint["model_ema2"][name].to(device=device)
        for name, _parameter in model_without_ddp.named_parameters()
    ]


def _validation_subset(dataset) -> Subset:
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    return Subset(dataset, list(range(rank, len(dataset), world_size)))


def _run_public_metrics(args) -> None:
    from evaluate import evaluate

    prediction_dir = args.prediction_root / args.model_name
    results_dir = args.results_root / args.model_name
    csv_path, summary_path = evaluate(
        model_name=args.model_name,
        manifest_path=args.val_manifest,
        prediction_dir=prediction_dir,
        results_dir=results_dir,
        use_blur_as_prediction=False,
    )
    print(f"Per-image metrics: {csv_path}")
    print(f"Summary: {summary_path}")


def _run_metrics_and_sync(args) -> None:
    if misc.is_main_process() and args.run_metrics:
        _run_public_metrics(args)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def main(args) -> None:
    if not math.isfinite(args.lambda_pix) or args.lambda_pix < 0:
        raise ValueError("--lambda_pix must be finite and non-negative")
    if not math.isfinite(args.charbonnier_eps) or args.charbonnier_eps <= 0:
        raise ValueError("--charbonnier_eps must be finite and positive")
    misc.init_distributed_mode(args)
    if args.device != "cuda":
        raise ValueError("A0 training entry is designed for the GPU server; use --device cuda")
    if args.num_sampling_steps < 1:
        raise ValueError("--num_sampling_steps must be at least 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("--batch_size and --eval_batch_size must be positive")
    if args.save_last_freq < 1 or args.log_freq < 1:
        raise ValueError("--save_last_freq and --log_freq must be positive")
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_writer = (
        SummaryWriter(log_dir=args.output_dir)
        if misc.is_main_process() and not args.eval_only
        else None
    )

    train_dataset = None
    train_loader = None
    train_sampler = None
    if not args.eval_only:
        full_train_dataset = A0PairedDataset(
            args.train_manifest,
            split="train",
            image_size=args.img_size,
            training=True,
            identity_ratio=args.identity_ratio,
            root=PROJECT_ROOT,
        )
        if len(full_train_dataset) != 2000:
            raise ValueError(
                "The fixed training manifest must contain exactly 2000 pairs "
                f"before any overfit subset is selected; got {len(full_train_dataset)}"
            )
        if args.overfit_samples < 0 or args.overfit_samples > 32:
            raise ValueError("--overfit_samples must be 0 or an integer from 1 to 32")
        if args.overfit_samples:
            subset_indices = np.linspace(
                0,
                len(full_train_dataset) - 1,
                num=args.overfit_samples,
                dtype=int,
            ).tolist()
            train_dataset = Subset(full_train_dataset, subset_indices)
        else:
            train_dataset = full_train_dataset
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=misc.get_world_size(),
                rank=misc.get_rank(),
                shuffle=True,
            )
            if args.distributed
            else RandomSampler(train_dataset)
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=not bool(args.overfit_samples),
        )

    val_dataset = A0PairedDataset(
        args.val_manifest,
        split="val",
        image_size=args.img_size,
        training=False,
        identity_ratio=0.0,
        root=PROJECT_ROOT,
    )
    if len(val_dataset) != 300:
        raise ValueError(
            f"Fixed validation manifest must contain 300 pairs, got {len(val_dataset)}"
        )
    if misc.get_world_size() > len(val_dataset):
        raise ValueError(
            f"world_size={misc.get_world_size()} exceeds the 300 validation samples"
        )
    val_loader = DataLoader(
        _validation_subset(val_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = Denoiser(args)
    resume_path = _checkpoint_file(args.resume)
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"A0 resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        print(f"Loaded A0 model checkpoint: {resume_path}")
    else:
        if not args.pretrained.is_file():
            raise FileNotFoundError(f"Official JiT checkpoint does not exist: {args.pretrained}")
        official_checkpoint = torch.load(
            args.pretrained, map_location="cpu", weights_only=False
        )
        model.load_official_state_dict(official_checkpoint["model"])
        del official_checkpoint
        print(f"Initialized A0 from official JiT checkpoint: {args.pretrained}")

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = None
    if not args.eval_only:
        effective_batch_size = args.batch_size * misc.get_world_size()
        if args.lr is None:
            args.lr = args.blr * effective_batch_size / 256
        param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95)
        )

    if resume_checkpoint is not None:
        _load_ema(model_without_ddp, resume_checkpoint, device)
        if not args.eval_only:
            if "optimizer" not in resume_checkpoint or "epoch" not in resume_checkpoint:
                raise KeyError("A0 resume checkpoint is missing optimizer or epoch")
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            args.start_epoch = int(resume_checkpoint["epoch"]) + 1
        del resume_checkpoint
    else:
        _initialize_ema(model_without_ddp)

    if args.eval_only:
        if resume_path is None:
            raise ValueError("--eval_only requires --resume with a trained A0 checkpoint")
        restore_fixed_validation(
            model_without_ddp,
            val_loader,
            device,
            model_name=args.model_name,
            output_root=args.prediction_root,
            use_ema=True,
            expected_total=300,
            amp_bf16=args.amp_bf16,
            eval_seed=args.eval_seed,
        )
        _run_metrics_and_sync(args)
        return

    print(f"A0 effective train pairs: {len(train_dataset)}")
    if args.overfit_samples:
        print("A0-O overfit mode is active; results are not formal A0 metrics")
    print(f"A0 fixed validation pairs: {len(val_dataset)}")
    print(f"Identity-pair probability: {args.identity_ratio}")
    print(
        f"A0 objective: flow + {args.lambda_pix:g} * Charbonnier; "
        f"eps={args.charbonnier_eps:g}; pixel error measured on RGB [0,1]"
    )
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch_a0(
            model,
            model_without_ddp,
            train_loader,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
        )

        should_save = epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs
        if should_save:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )

        should_restore = epoch + 1 == args.epochs or (
            args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0
        )
        if should_restore:
            restore_fixed_validation(
                model_without_ddp,
                val_loader,
                device,
                model_name=args.model_name,
                output_root=args.prediction_root,
                use_ema=True,
                expected_total=300,
                amp_bf16=args.amp_bf16,
                eval_seed=args.eval_seed,
            )
            _run_metrics_and_sync(args)

        if log_writer is not None:
            log_writer.flush()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"A0 training time: {elapsed}")
    if log_writer is not None:
        log_writer.close()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
