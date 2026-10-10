"""Spatial A1 token condition that shares its PSF field with local A2/A3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .local_model import LocalBlurEstimator
except ImportError:
    from local_model import LocalBlurEstimator


class LocalDegradationCondition(nn.Module):
    """Map [B,3,h,w] probabilities to [B,grid_h*grid_w,hidden_size].

    Resize with bilinear interpolation to JiT's actual image-token grid. The
    zero-initialized gate keeps the initial A1/A3 token path equal to A0.
    JiT's integration owner determines the token grid and injection point.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(3, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, probabilities: torch.Tensor, token_grid: tuple[int, int]) -> torch.Tensor:
        if probabilities.ndim != 4 or probabilities.shape[1] != 3:
            raise ValueError("Expected local probabilities [B,3,h,w]")
        if len(token_grid) != 2 or min(token_grid) < 1:
            raise ValueError("token_grid must be a positive (height,width) pair")
        field = F.interpolate(probabilities, size=token_grid, mode="bilinear", align_corners=False)
        tokens = field.permute(0, 2, 3, 1).reshape(probabilities.shape[0], -1, 3)
        return self.gate * self.embedding(tokens)


def predict_local_from_jit_blur(blur_jit: torch.Tensor, estimator: LocalBlurEstimator) -> torch.Tensor:
    """Predict a local field from JiT's blurred-image condition only."""
    if blur_jit.ndim != 4 or blur_jit.shape[1] != 3:
        raise ValueError("Expected JiT blurred RGB [B,3,H,W]")
    blur_01 = ((blur_jit + 1) * 0.5).clamp(0, 1)
    return estimator(blur_01).softmax(dim=1)
