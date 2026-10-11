"""Load the frozen local A1 estimator without importing reblur/loss tools."""

import hashlib
from pathlib import Path

import torch

from .local_model import LocalBlurEstimator


LOCAL_LEVELS = (("light", 1.0, 7), ("medium", 2.0, 13), ("heavy", 3.5, 23))
# C's delivered v3 uses the same channel definition but omits levels metadata.
# Recognize this exact artifact rather than guessing for every metadata-free file.
LOCAL_V3_SHA256 = "04cb85d6d59016e4a7e3cf2f83ce607bb771d0556c8bdb96c7fe6682178adc0b"


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_levels(checkpoint, checkpoint_sha256=None) -> None:
    if "levels" in checkpoint:
        levels = checkpoint["levels"]
        if (not isinstance(levels, (tuple, list))
                or any(not isinstance(row, (tuple, list)) for row in levels)
                or tuple(tuple(row) for row in levels) != LOCAL_LEVELS):
            raise ValueError("Local checkpoint channels must be light/medium/heavy with the agreed kernels")
    elif checkpoint_sha256 != LOCAL_V3_SHA256:
        raise ValueError("Local checkpoint has no levels metadata and is not C's verified v3 artifact")


def load_local_estimator(path: str | Path) -> LocalBlurEstimator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _validate_local_levels(
        checkpoint, _checkpoint_sha256(path) if "levels" not in checkpoint else None
    )
    model = LocalBlurEstimator()
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Local estimator checkpoint contains nonfinite weights")
    return model.eval().requires_grad_(False)
