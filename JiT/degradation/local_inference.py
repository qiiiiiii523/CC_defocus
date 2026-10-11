"""Load the frozen local A1 estimator without importing reblur/loss tools."""

from pathlib import Path

import torch

from .local_model import LocalBlurEstimator


LOCAL_LEVELS = (("light", 1.0, 7), ("medium", 2.0, 13), ("heavy", 3.5, 23))


def load_local_estimator(path: str | Path) -> LocalBlurEstimator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(tuple(row) for row in checkpoint["levels"]) != LOCAL_LEVELS:
        raise ValueError("Local checkpoint channels must be light/medium/heavy with the agreed kernels")
    model = LocalBlurEstimator()
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Local estimator checkpoint contains nonfinite weights")
    return model.eval().requires_grad_(False)
