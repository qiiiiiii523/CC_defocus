import torch
from diffusers import AutoencoderKL


def load_frozen_vae(model_id, device, dtype):
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').to(device, dtype=dtype)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, image):
    z = vae.encode(image).latent_dist.mode()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, latent):
    z = latent / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(z).sample.clamp(-1, 1)
