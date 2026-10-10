"""Shared FP32 blur-plus-noise flow path for training and Euler inference."""
import math
import torch
from runtime_utils import autocast_context


def make_training_state(z_blur, z_clear, alpha=0.15):
    if not math.isfinite(alpha) or alpha < 0 or z_blur.shape != z_clear.shape:
        raise ValueError("Require equal latent shapes and finite nonnegative alpha")
    z_blur, z_clear = z_blur.float(), z_clear.float()
    z_source = z_blur + alpha * torch.randn_like(z_blur)
    sigma = torch.rand((z_blur.shape[0],), device=z_blur.device, dtype=torch.float32)
    view = sigma.view(-1, 1, 1, 1)
    x_t = (1.0 - view) * z_clear + view * z_source
    return x_t, z_blur, sigma * 1000.0, z_source - z_clear


@torch.no_grad()
def sample(model, z_blur, steps=20, alpha=0.15, generator=None, bf16=False):
    if type(steps) is not int or steps < 1 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("steps must be positive and alpha finite and nonnegative")
    z_blur = z_blur.float()
    noise = torch.randn(z_blur.shape, device=z_blur.device, dtype=torch.float32, generator=generator)
    z = z_blur + alpha * noise
    for i in range(steps):
        t = torch.full((z.shape[0],), 1000.0 * (1.0 - i / steps), device=z.device, dtype=torch.float32)
        with autocast_context(z.device, bf16):
            velocity = model(z, z_blur, t)
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("Non-finite flow velocity")
        z = z - velocity.float() / steps
    if not torch.isfinite(z).all():
        raise FloatingPointError("Non-finite sampled latent")
    return z
