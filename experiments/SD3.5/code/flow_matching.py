"""Flow-matching helpers used by the scratch training entry point."""
import torch


def make_training_state(z_blur, z_clear, alpha=0.15):
    noise = torch.randn_like(z_blur)
    z_source = z_blur + alpha * noise
    sigma = torch.rand((z_blur.shape[0],), device=z_blur.device, dtype=z_blur.dtype)
    view = sigma.view(-1, 1, 1, 1)
    x_t = (1.0 - view) * z_clear + view * z_source
    target = z_source - z_clear
    return x_t, z_blur, sigma * 1000.0, target
