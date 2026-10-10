"""Inference entry point. It accepts blur images only; clear images are never read."""
import argparse, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
from model import SD35ScratchRestorer
from vae_utils import load_frozen_vae, encode, decode


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True)
    p.add_argument('--blur', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model-id', default='stabilityai/stable-diffusion-3.5-medium'); p.add_argument('--steps', type=int, default=20)
    a = p.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    m = SD35ScratchRestorer(a.model_id).to(device, dtype=dtype); m.load_state_dict(torch.load(a.checkpoint, map_location=device)['model']); m.eval()
    vae = load_frozen_vae(a.model_id, device, dtype)
    x = resize(to_tensor(Image.open(a.blur).convert('RGB')), [256, 256]).mul(2).sub(1).unsqueeze(0).to(device, dtype=dtype)
    zc = encode(vae, x); z = zc + 0.15 * torch.randn_like(zc)
    with torch.no_grad():
        for i in range(a.steps):
            t = torch.full((1,), 1000 * (1 - i / a.steps), device=device, dtype=dtype)
            v = m(z, zc, t); z = z - v / a.steps
    to_pil_image((decode(vae, z)[0].float() + 1) / 2).save(a.out)


if __name__ == '__main__': main()
