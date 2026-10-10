"""Configuration, reproducible RNG and portable checkpoint helpers."""
import hashlib
import json
import math
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


DEFAULTS = {
    "resolution": 256, "batch_size": 1, "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5, "weight_decay": 0.01, "max_grad_norm": 1.0,
    "seed": 20261010, "bf16": True, "gradient_checkpointing": True,
    "flow_condition_noise_alpha": 0.15, "sampling_steps": 20,
    "num_workers": 2, "overfit_samples": 0, "identity_ratio": 0.0,
    "augment_train": True, "checkpoint_every": 500, "log_every": 10,
    "validation_every": 500, "validation_samples": 8,
    "keep_checkpoints": 2,
}
RESUME_KEYS = (
    "resolution", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "max_grad_norm", "seed", "bf16", "gradient_checkpointing",
    "flow_condition_noise_alpha", "overfit_samples", "identity_ratio", "augment_train",
)


def normalize_config(raw):
    cfg = dict(DEFAULTS, **raw)
    for key in ("resolution", "batch_size", "gradient_accumulation_steps", "sampling_steps",
                "log_every", "validation_samples"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if cfg["resolution"] % 16:
        raise ValueError("resolution must be divisible by VAE scale 8 * Transformer patch size 2")
    for key in ("num_workers", "overfit_samples", "checkpoint_every", "validation_every", "seed", "keep_checkpoints"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(key + " must be a nonnegative integer")
    for key in ("bf16", "gradient_checkpointing", "augment_train"):
        if type(cfg[key]) is not bool:
            raise ValueError(key + " must be boolean")
    for key in ("learning_rate", "weight_decay", "max_grad_norm", "flow_condition_noise_alpha", "identity_ratio"):
        if not isinstance(cfg[key], (int, float)) or isinstance(cfg[key], bool) or not math.isfinite(cfg[key]):
            raise ValueError(key + " must be finite")
    if cfg["learning_rate"] <= 0 or cfg["max_grad_norm"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("learning_rate/max_grad_norm must be positive and weight_decay nonnegative")
    if cfg["flow_condition_noise_alpha"] < 0 or not 0 <= cfg["identity_ratio"] <= 1:
        raise ValueError("Invalid noise alpha or identity ratio")
    for key in ("max_train_steps", "num_epochs"):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(key + " must be a positive integer when specified")
    if "max_train_steps" not in cfg and "num_epochs" not in cfg:
        raise ValueError("Specify max_train_steps (optimizer updates) or num_epochs")
    if not cfg.get("train_manifest"):
        raise ValueError("train_manifest is required")
    if cfg["validation_every"] and not cfg.get("val_manifest"):
        raise ValueError("val_manifest is required when validation_every is enabled")
    return cfg


def autocast_context(device, bf16):
    # Keep model parameters and accumulated gradients in FP32.
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16 else nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def load_checkpoint(path):
    # CPU load avoids a second full state dict on the training GPU.
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(path, model, optimizer, cfg, model_id, progress, digest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    payload = {
        "format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": cfg, "model_id": str(model_id),
        "transformer_config": json.loads(json.dumps(dict(model.transformer.config))),
        "step": progress["step"], "progress": dict(progress),
        "rng": capture_rng(), "train_manifest_sha256": digest,
    }
    torch.save(payload, pending)
    pending.replace(path)


def check_resume(checkpoint, cfg, digest, model_id):
    if checkpoint.get("format_version") != 2:
        raise ValueError("This checkpoint has weights only; exact training resume requires a v2 checkpoint")
    for key in RESUME_KEYS:
        if checkpoint["config"][key] != cfg[key]:
            raise ValueError("Resume config differs: " + key)
    if checkpoint["train_manifest_sha256"] != digest:
        raise ValueError("Training manifest changed since checkpoint")
    if checkpoint["model_id"] != str(model_id):
        raise ValueError("Resume model_id differs; use the same VAE source")


def prune_checkpoints(directory, keep):
    """Retain newest numeric checkpoints; never touch final or arbitrary files."""
    if not keep:
        return
    files = []
    for path in Path(directory).glob("checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d+)\.pt", path.name)
        if match and path.is_file():
            files.append((int(match.group(1)), path))
    for _, path in sorted(files, reverse=True)[keep:]:
        path.unlink()
