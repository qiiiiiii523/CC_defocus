"""Train and restore with JiT A0 on paired 3DHistech manifests.

This is intentionally separate from ``main_jit.py``, which remains the
official ImageNet class-conditional entry point.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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
if str(JIT_ROOT) not in sys.path:
    sys.path.insert(0, str(JIT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.misc as misc
from dataset_restoration import A0PairedDataset
from dataset_cnseg import CNSegPairedDataset
from denoiser import Denoiser
from engine_restoration import restore_fixed_validation, train_one_epoch_a0


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("JiT A0 paired restoration")
    parser.add_argument("--dataset_mode", choices=("3dhistech", "cnseg"), default="3dhistech")
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Start a NEW experiment from A0 model weights; reset optimizer, EMA and epoch.")
    parser.add_argument("--init_weights", choices=("model", "ema1", "ema2"), default="model")
    parser.add_argument("--lambda_nucleus", type=float, default=0.0)
    parser.add_argument("--nucleus_region_weight", type=float, default=1.0)
    parser.add_argument("--nucleus_boundary_weight", type=float, default=1.0)

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
            "Use a deterministic subset of the training manifest "
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
        "--eval_weights", default="ema1", choices=("model", "ema1", "ema2"),
        help="Weight source for restoration only; does not change training or checkpoints.",
    )
    parser.add_argument(
        "--eval_freq",
        default=0,
        type=int,
        help="Restore the entire validation manifest every N epochs; 0 means final epoch only.",
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
        "--expected_train_pairs", default=2000, type=int,
        help="Expected full training manifest size before subsetting; 0 accepts any nonempty size.",
    )
    parser.add_argument(
        "--expected_val_pairs", default=300, type=int,
        help="Expected validation manifest size; 0 accepts any nonempty size.",
    )
    parser.add_argument(
        "--init_mode", default=None, choices=("pretrained", "scratch"),
        help="Fresh runs default to pretrained; resumes inherit the saved initialization mode.",
    )
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


def _resolve_init_mode(requested, checkpoint=None):
    saved_args = checkpoint.get("args") if checkpoint is not None else None
    saved = (saved_args.get("init_mode") if isinstance(saved_args, dict)
             else getattr(saved_args, "init_mode", None))
    if checkpoint is not None:
        # Legacy A0 checkpoints predate scratch support and used official weights.
        saved = saved or "pretrained"
        if requested is not None and requested != saved:
            raise ValueError(f"Resume init_mode mismatch: requested {requested}, saved {saved}")
        return saved
    return requested or "pretrained"


def _initialize_fresh_model(model, args):
    if args.init_mode == "scratch":
        # Denoiser construction already applies the official random initialization.
        # Copy values, not parameters: the RGB branches remain independently trainable.
        model.net.initialize_blur_condition_from_state_embedder()
        print("Initialized A0 from scratch; no official checkpoint was read")
    else:
        if not args.pretrained.is_file():
            raise FileNotFoundError(f"Official JiT checkpoint does not exist: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        model.load_official_state_dict(checkpoint["model"])
        print(f"Initialized A0 from official JiT checkpoint: {args.pretrained}")


def _check_pair_counts(actual, expected, split):
    if expected < 0:
        raise ValueError(f"Expected {split} pair count must be non-negative")
    if actual < 1 or (expected and actual != expected):
        raise ValueError(f"{split} manifest has {actual} pairs; expected {expected or 'a nonempty manifest'}")


def _check_train_val_overlap(train_records, val_records):
    # This catches exact duplicate IDs or reused images, not adjacent crops/slide leakage.
    ids = {record.sample_id for record in train_records}
    paths = {path.resolve() for record in train_records
             for path in (record.blur_path, record.clear_path)}
    groups = {getattr(record, "group_id", None) for record in train_records} - {None}
    for record in val_records:
        if getattr(record, "group_id", None) in groups or record.sample_id in ids or any(
            path.resolve() in paths for path in (record.blur_path, record.clear_path)
        ):
            raise ValueError(f"Train/validation overlap: {record.sample_id}")


def _manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    if args.dataset_mode == "cnseg":
        from evaluate_k0 import evaluate_cnseg
        evaluate_cnseg(args)
        return
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
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("Use --init_checkpoint for a new experiment OR --resume for continuation, not both")
    for field in ("lambda_nucleus", "nucleus_region_weight", "nucleus_boundary_weight"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if args.lambda_nucleus > 0 and args.dataset_mode != "cnseg":
        raise ValueError("K0 requires CNSeg images and corresponding instance masks")
    dataset_class = CNSegPairedDataset if args.dataset_mode == "cnseg" else A0PairedDataset
    if args.init_checkpoint is not None and args.eval_only:
        raise ValueError("Evaluation uses --resume; --init_checkpoint is for NEW training experiments")
    if args.init_checkpoint is not None and args.start_epoch != 0:
        raise ValueError("A new K0 experiment must start at epoch 0")
    if args.init_checkpoint is not None and args.output_dir.resolve() == _checkpoint_file(args.init_checkpoint).resolve().parent:
        raise ValueError("K0 output_dir must not overwrite the A0 initialization directory")
    resume_path = _checkpoint_file(args.resume)
    if args.eval_only and resume_path is None:
        raise ValueError("--eval_only requires --resume with a trained A0 checkpoint")
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"A0 resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    args.init_mode = _resolve_init_mode(args.init_mode, resume_checkpoint)
    if args.init_checkpoint is not None:
        start_checkpoint = torch.load(_checkpoint_file(args.init_checkpoint), map_location="cpu", weights_only=False)
        args.init_mode = _resolve_init_mode(None, start_checkpoint)
    if args.init_mode == "scratch":
        if args.model_name == "jit_a0":
            args.model_name = "jit_a0_scratch"
        if args.output_dir == PROJECT_ROOT / "checkpoints" / "jit_a0":
            args.output_dir = PROJECT_ROOT / "checkpoints" / "jit_a0_scratch"
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
        full_train_dataset = dataset_class(
            args.train_manifest,
            split="train",
            image_size=args.img_size,
            training=True,
            identity_ratio=args.identity_ratio,
            root=PROJECT_ROOT,
        )
        _check_pair_counts(len(full_train_dataset), args.expected_train_pairs, "train")
        if args.overfit_samples < 0 or args.overfit_samples > 32:
            raise ValueError("--overfit_samples must be 0 or an integer from 1 to 32")
        if args.overfit_samples:
            if args.overfit_samples > len(full_train_dataset):
                raise ValueError("--overfit_samples exceeds training manifest size")
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
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty; reduce batch size or use overfit mode")

    val_dataset = dataset_class(
        args.val_manifest,
        split="val",
        image_size=args.img_size,
        training=False,
        identity_ratio=0.0,
        root=PROJECT_ROOT,
    )
    _check_pair_counts(len(val_dataset), args.expected_val_pairs, "val")
    if not args.eval_only:
        _check_train_val_overlap(full_train_dataset.records, val_dataset.records)
    if misc.get_world_size() > len(val_dataset):
        raise ValueError(
            f"world_size={misc.get_world_size()} exceeds {len(val_dataset)} validation samples"
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
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        print(f"Loaded A0 model checkpoint: {resume_path}")
    elif args.init_checkpoint is not None:
        key = {"model": "model", "ema1": "model_ema1", "ema2": "model_ema2"}[args.init_weights]
        model.load_state_dict(start_checkpoint[key], strict=True)
        args.start_epoch = 0
        args.global_step = 0
        print(f"K0 new-run initialization: {args.init_checkpoint}, {key}; optimizer/EMA/epoch reset")
        del start_checkpoint
    else:
        _initialize_fresh_model(model, args)

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
            args.lr = (1e-4 if args.init_mode == "scratch"
                       else args.blr * effective_batch_size / 256)
            if args.init_mode == "scratch":
                print("Scratch initial LR defaults to 1e-4; this is a trial setting, not a validated optimum")
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
            saved_args = resume_checkpoint.get("args")
            for field in ("lambda_pix", "charbonnier_eps", "identity_ratio",
                          "ema_decay1", "ema_decay2", "dataset_mode", "lambda_nucleus",
                          "nucleus_region_weight", "nucleus_boundary_weight"):
                saved_value = (saved_args.get(field) if isinstance(saved_args, dict)
                               else getattr(saved_args, field, None))
                if saved_value is not None and saved_value != getattr(args, field):
                    raise ValueError(f"Resume {field} mismatch: saved {saved_value}; repeat the original configuration")
            args.global_step = (saved_args.get("global_step", args.start_epoch * len(train_loader))
                                if isinstance(saved_args, dict) else
                                getattr(saved_args, "global_step", args.start_epoch * len(train_loader)))
            previous_hash = (saved_args.get("train_manifest_sha256") if isinstance(saved_args, dict)
                             else getattr(saved_args, "train_manifest_sha256", None))
            if previous_hash and previous_hash != _manifest_sha256(args.train_manifest):
                raise ValueError("Resume training manifest differs; do not change training data mid-run")
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
            weight_source=args.eval_weights,
            expected_total=len(val_dataset),
            amp_bf16=args.amp_bf16,
            eval_seed=args.eval_seed,
        )
        _run_metrics_and_sync(args)
        return

    args.global_step = getattr(args, "global_step", 0)
    args.train_manifest_sha256 = _manifest_sha256(args.train_manifest)
    args.val_manifest_sha256 = _manifest_sha256(args.val_manifest)
    if misc.is_main_process():
        report = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        report.update(full_train_pairs=len(full_train_dataset),
                      effective_train_pairs=len(train_dataset), val_pairs=len(val_dataset),
                      condition_init="checkpoint_preserved" if (resume_path or args.init_checkpoint) else "copy_state_patch_embedder",
                      gate_init=None if (resume_path or args.init_checkpoint) else 0,
                      initialization_source=str(resume_path or args.init_checkpoint) if (resume_path or args.init_checkpoint) else args.init_mode)
        (args.output_dir / "run_config.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"A0 initialization mode: {args.init_mode}; all trainable parameters are optimized")
    print(f"A0 effective train pairs: {len(train_dataset)}")
    if args.overfit_samples:
        print("A0-O overfit mode is active; results are not formal A0 metrics")
    print(f"A0 validation pairs: {len(val_dataset)}")
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
        args.global_step += len(train_loader)
        print(f"A0 global_step: {args.global_step}")

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
                weight_source=args.eval_weights,
                expected_total=len(val_dataset),
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
