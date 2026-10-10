"""Single-GPU scratch training with accumulation, validation and resume."""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PairedManifest
from flow_matching import make_training_state, sample
from model import DEFAULT_MODEL_ID, SD35ScratchRestorer
from runtime_utils import (autocast_context, capture_rng, check_resume, load_checkpoint,
                           manifest_digest, normalize_config, prune_checkpoints, restore_rng, save_checkpoint, seed_all)
from vae_utils import decode, encode, load_frozen_vae


def train_epoch(model, vae, optimizer, loader, device, cfg, progress, on_update, resume_rng=None):
    """Update only after each accumulation group, including a partial final group."""
    model.train()
    epoch = progress["epoch"]
    cursor = progress["next_batch"]
    iterator = iter(loader)
    for _ in range(cursor):
        next(iterator)
    if resume_rng is not None:
        restore_rng(resume_rng)
    optimizer.zero_grad(set_to_none=True)
    for first in range(cursor, len(loader), cfg["gradient_accumulation_steps"]):
        batches = list(itertools.islice(iterator, cfg["gradient_accumulation_steps"]))
        total_samples = sum(batch["blur"].shape[0] for batch in batches)
        total_loss = 0.0
        for batch in batches:
            zb = encode(vae, batch["blur"].to(device=device, dtype=torch.float32)).float()
            zc = encode(vae, batch["clear"].to(device=device, dtype=torch.float32)).float()
            xt, cond, t, target = make_training_state(zb, zc, cfg["flow_condition_noise_alpha"])
            with autocast_context(device, cfg["bf16"]):
                prediction = model(xt, cond, t)
            loss = (prediction.float() - target.float()).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; optimizer update cancelled")
            weight = batch["blur"].shape[0] / total_samples
            (loss * weight).backward()
            total_loss += loss.detach().item() * weight
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress["step"] += 1
        consumed = first + len(batches)
        progress["epoch"] = epoch + 1 if consumed == len(loader) else epoch
        progress["next_batch"] = 0 if consumed == len(loader) else consumed
        on_update(progress, total_loss, float(grad_norm))
        if progress["step"] >= cfg.get("max_train_steps", math.inf):
            break


@torch.no_grad()
def validate(model, vae, dataset, device, cfg):
    """Online mean per-image PSNR; official PSNR/SSIM/LPIPS remain in evaluate.py."""
    training = model.training
    rng = capture_rng()
    model.eval()
    scores = []
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    try:
        for i in range(min(len(dataset), cfg["validation_samples"])):
            pair = dataset[i]
            blur = pair["blur"].unsqueeze(0).to(device=device, dtype=torch.float32)
            target = pair["clear"].unsqueeze(0).to(device=device, dtype=torch.float32)
            latent = encode(vae, blur).float()
            prediction = decode(vae, sample(model, latent, steps=cfg["sampling_steps"],
                                alpha=cfg["flow_condition_noise_alpha"], generator=generator,
                                bf16=cfg["bf16"]))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite validation prediction")
            mse = ((prediction.float() - target) / 2).square().mean().item()
            scores.append(-10 * math.log10(max(mse, 1e-12)))
    finally:
        model.train(training)
        restore_rng(rng)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = normalize_config(json.loads(args.config.read_text(encoding="utf-8")))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for full SD3.5 training")
    if cfg["bf16"] and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16; set bf16=false explicitly")
    run_training(args, cfg, torch.device("cuda"))


def run_training(args, cfg, device):
    """Shared orchestration; CPU is used only by reduced-model regression tests."""
    root = args.repo_root.resolve()
    manifest = root / cfg["train_manifest"]
    ds = PairedManifest(manifest, root, cfg["resolution"], train=True,
                        identity_ratio=cfg["identity_ratio"], max_samples=cfg["overfit_samples"],
                        seed=cfg["seed"], augment=cfg["augment_train"])
    val_ds = None
    if cfg["validation_every"]:
        val_ds = PairedManifest(root / cfg["val_manifest"], root, cfg["resolution"], train=False)
    digest = manifest_digest(manifest)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    model_id = args.model_id or (checkpoint["model_id"] if checkpoint else DEFAULT_MODEL_ID)
    if checkpoint:
        check_resume(checkpoint, cfg, digest, model_id)
    seed_all(cfg["seed"])
    model = SD35ScratchRestorer(model_id, transformer_config=(
        checkpoint["transformer_config"] if checkpoint else None))
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint["model"]
    model.to(device=device, dtype=torch.float32)
    if cfg["gradient_checkpointing"]:
        model.transformer.enable_gradient_checkpointing()
    vae = load_frozen_vae(model_id, device, torch.float32)
    if cfg.get("optimizer", "adamw").lower() == "sgd":
        # Useful for a tiny 32 GB smoke/overfit run: SGD has no Adam moment
        # buffers. Formal training keeps the default AdamW setting.
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg["weight_decay"], foreach=False)
    progress = {"step": 0, "epoch": 0, "next_batch": 0}
    resume_rng = None
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = dict(checkpoint["progress"])
        resume_rng = checkpoint["rng"]
        restore_rng(resume_rng)
        del checkpoint
    batches_per_epoch = math.ceil(len(ds) / cfg["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / cfg["gradient_accumulation_steps"])
    if not 0 <= progress["next_batch"] < batches_per_epoch:
        raise ValueError("Invalid checkpoint data cursor")
    if progress["next_batch"] % cfg["gradient_accumulation_steps"]:
        raise ValueError("Checkpoint cursor is not on an optimizer boundary")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"samples={len(ds)} effective_batch={cfg['batch_size'] * cfg['gradient_accumulation_steps']} "
          f"updates_per_epoch={updates_per_epoch} bf16={cfg['bf16']} "
          f"gradient_checkpointing={cfg['gradient_checkpointing']}", flush=True)
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def on_update(state, loss, grad_norm):
        step = state["step"]
        metrics = {"step": step, "epoch": state["epoch"], "next_batch": state["next_batch"],
                   "loss": loss, "grad_norm": grad_norm}
        if cfg["validation_every"] and step % cfg["validation_every"] == 0:
            metrics["online_val_psnr"] = validate(model, vae, val_ds, device, cfg)
        if step == 1 or step % cfg["log_every"] == 0 or "online_val_psnr" in metrics:
            print(json.dumps(metrics), flush=True)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
        if cfg["checkpoint_every"] and step % cfg["checkpoint_every"] == 0:
            save_checkpoint(out / f"checkpoint-{step}.pt", model, optimizer, cfg, model_id, state, digest)
            prune_checkpoints(out, cfg["keep_checkpoints"])

    while (progress["step"] < cfg.get("max_train_steps", math.inf)
           and progress["epoch"] < cfg.get("num_epochs", math.inf)):
        epoch = progress["epoch"]
        ds.set_epoch(epoch)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
        train_epoch(model, vae, optimizer, loader, device, cfg, progress, on_update, resume_rng)
        resume_rng = None
    save_checkpoint(out / "checkpoint-final.pt", model, optimizer, cfg, model_id, progress, digest)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    print(f"finished updates={progress['step']} epochs={progress['epoch']} "
          f"seconds={time.time()-started:.1f} peak_GiB={peak:.2f}", flush=True)


if __name__ == "__main__":
    main()
