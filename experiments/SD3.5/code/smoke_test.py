"""Real blur/clear -> frozen VAE -> Transformer -> optimizer -> sampler smoke test."""
import argparse
import json

import torch

from data import PairedManifest, PROJECT_ROOT
from flow_matching import make_training_state, sample
from model import DEFAULT_MODEL_ID, SD35ScratchRestorer
from runtime_utils import autocast_context, seed_all
from vae_utils import decode, encode, load_frozen_vae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest", default="prepared/manifests/debug_train.jsonl")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the full SD3.5 smoke test")
    bf16 = not args.fp32
    if bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support BF16; use --fp32")
    seed_all(20261010)
    device = torch.device("cuda")
    pair = PairedManifest(args.manifest, args.repo_root, max_samples=1, augment=False)[0]
    # Keep the trainable Transformer in BF16 for the 32 GB smoke card.  The
    # smoke test only checks gradients and one parameter update; AdamW state is
    # intentionally not allocated here because it can exceed the card limit.
    model_dtype = torch.bfloat16 if bf16 else torch.float32
    model = SD35ScratchRestorer(args.model_id).to(device=device, dtype=model_dtype).train()
    model.transformer.enable_gradient_checkpointing()
    vae = load_frozen_vae(args.model_id, device, torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    blur = encode(vae, pair["blur"].unsqueeze(0).to(device)).float()
    clear = encode(vae, pair["clear"].unsqueeze(0).to(device)).float()
    state, condition, time, target = make_training_state(blur, clear)
    with autocast_context(device, bf16):
        prediction = model(state, condition, time)
    loss = (prediction.float() - target).square().mean()
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite loss")
    loss.backward()
    for name, module in (("transformer", model.transformer), ("condition_stem", model.condition_stem)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError("Missing/non-finite gradients: " + name)
        if not any(torch.count_nonzero(g) for g in gradients):
            raise AssertionError("Zero gradients: " + name)
    if any(p.requires_grad or p.grad is not None for p in vae.parameters()):
        raise AssertionError("VAE must remain frozen")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del prediction, loss
    model.eval()
    with torch.no_grad(), autocast_context(device, bf16):
        a = model(state, condition, time)
        b = model(state, torch.zeros_like(condition), time)
    if torch.allclose(a, b):
        raise AssertionError("Condition path is inactive")
    image = decode(vae, sample(model, blur, steps=2, bf16=bf16))
    if image.shape != (1, 3, 256, 256) or not torch.isfinite(image).all():
        raise AssertionError("Invalid decoded prediction")
    print(json.dumps({"status": "PASS", "bf16": bf16, "vae_dtype": str(blur.dtype),
                      "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


if __name__ == "__main__":
    main()
