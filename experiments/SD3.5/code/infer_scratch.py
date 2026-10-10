"""Single-image or manifest inference; only blur pixels enter the model."""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import PROJECT_ROOT
from common_io import load_manifest, read_rgb_image, save_prediction
from flow_matching import sample
from model import DEFAULT_MODEL_ID, SD35ScratchRestorer
from runtime_utils import load_checkpoint
from vae_utils import decode, encode, load_frozen_vae


@torch.no_grad()
def restore_image(model, vae, blur, device, resolution, steps, alpha, seed, bf16):
    if blur.shape[:2] != (resolution, resolution):
        raise ValueError(f"Expected original {resolution}x{resolution} image; resizing is disabled")
    x = torch.from_numpy(blur.transpose(2, 0, 1).copy()).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32) * 2 - 1
    latent = encode(vae, x).float()
    generator = torch.Generator(device=device).manual_seed(seed)
    restored = sample(model, latent, steps=steps, alpha=alpha, generator=generator, bf16=bf16)
    image = (decode(vae, restored)[0].float() + 1) / 2
    result = image.permute(1, 2, 0).cpu().numpy()
    if result.shape != blur.shape or not np.isfinite(result).all():
        raise ValueError("Invalid restoration shape or values")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--blur", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Single-image PNG output")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-name", default="sd35_scratch")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp32", action="store_true", help="Disable Transformer BF16 autocast")
    args = parser.parse_args()
    if args.blur and not args.out:
        parser.error("--out is required with --blur")
    if args.manifest and args.out:
        parser.error("Use --output-root for manifest inference")
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {})
    model_id = args.model_id or ckpt.get("model_id", DEFAULT_MODEL_ID)
    if ckpt.get("model_id") and model_id != ckpt["model_id"]:
        raise ValueError("model-id must match the checkpoint VAE source")
    steps = args.steps if args.steps is not None else cfg.get("sampling_steps", 20)
    seed = args.seed if args.seed is not None else cfg.get("seed", 20261010)
    resolution = cfg.get("resolution", 256)
    alpha = cfg.get("flow_condition_noise_alpha", 0.15)
    if steps < 1 or seed < 0:
        parser.error("steps must be positive and seed nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(cfg.get("bf16", True) and not args.fp32 and device.type == "cuda")
    if bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support BF16; use --fp32")
    model = SD35ScratchRestorer(model_id, transformer_config=ckpt.get("transformer_config"))
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    model.to(device=device, dtype=torch.float32).eval()
    vae = load_frozen_vae(model_id, device, torch.float32)

    def predict(path, sample_id):
        blur = read_rgb_image(path, sample_id=sample_id, role="blur")
        result = restore_image(model, vae, blur, device, resolution, steps, alpha, seed, bf16)
        return result, blur.shape[:2]

    if args.blur:
        result, hw = predict(args.blur, args.blur.stem)
        if args.out.suffix.lower() != ".png":
            parser.error("--out must be a lossless PNG")
        # Use the common validation/quantization protocol for single outputs too.
        destination = args.out.resolve()
        path = save_prediction(result, model_name=destination.parent.name,
                               sample_id=destination.stem, expected_hw=hw,
                               output_root=destination.parent.parent)
        print(f"saved={path}", flush=True)
    else:
        records = load_manifest(args.manifest, root=args.repo_root, expected_split=args.split)
        if not records:
            raise ValueError("Manifest is empty")
        for index, record in enumerate(records, 1):
            result, hw = predict(record.blur_path, record.sample_id)
            save_prediction(result, model_name=args.model_name, sample_id=record.sample_id,
                            expected_hw=hw, output_root=args.output_root or args.repo_root / "outputs")
            print(f"saved {index}/{len(records)} {record.sample_id}", flush=True)


if __name__ == "__main__":
    main()
