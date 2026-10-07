"""A1 condition contract between the blur estimator and JiT's token width."""

from __future__ import annotations

import torch
from torch import nn

if __package__:
    from .model import BlurEstimator
    from .psf import LEVELS
else:
    from model import BlurEstimator
    from psf import LEVELS


class DegradationCondition(nn.Module):
    """Encode three predicted PSF weights as one residual JiT condition vector.

    The zero gate makes the initial A1 forward path identical to A0 after its
    weights are loaded. JiT integration will add this vector to its existing
    time/class condition; it does not replace the blurred-image input.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(len(LEVELS), hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != len(LEVELS):
            raise ValueError("Expected [batch, 3] predicted PSF probabilities")
        return self.gate * self.embedding(probabilities)


def predict_from_jit_blur(blur_jit: torch.Tensor, estimator: BlurEstimator) -> torch.Tensor:
    """Return predicted weights from JiT's [-1,1] RGB blur condition only.

    This function must never receive clear targets or ground-truth PSF labels.
    The estimator is frozen during the first A1 comparison to isolate the
    condition adapter; the caller may explicitly choose otherwise later.
    """
    if blur_jit.ndim != 4 or blur_jit.shape[1] != 3:
        raise ValueError("Expected RGB BCHW JiT blur tensor")
    blur_01 = (blur_jit + 1.0) * 0.5
    return estimator(blur_01).softmax(dim=1)
