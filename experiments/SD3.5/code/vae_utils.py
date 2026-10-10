"""Frozen, deterministic SD3 VAE encoding with model-specific scaling."""
import torch


def load_frozen_vae(model_id, device, dtype=torch.float32):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    # Honor force_upcast even if the Transformer uses mixed precision.
    if getattr(vae.config, "force_upcast", True):
        dtype = torch.float32
    vae = vae.to(device=device, dtype=dtype).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, image):
    image = image.to(dtype=next(vae.parameters()).dtype)
    z = vae.encode(image).latent_dist.mode()
    return (z - (getattr(vae.config, "shift_factor", None) or 0.0)) * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, latent):
    latent = latent.to(dtype=next(vae.parameters()).dtype)
    z = latent / vae.config.scaling_factor + (getattr(vae.config, "shift_factor", None) or 0.0)
    return vae.decode(z).sample.clamp(-1, 1)
