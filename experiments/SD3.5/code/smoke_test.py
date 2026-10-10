"""GPU smoke test: construction, forward, backward and condition sensitivity."""
import argparse
import torch
from model import SD35ScratchRestorer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-id', default='stabilityai/stable-diffusion-3.5-medium')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('GPU is required for this smoke test; no claim of success was made.')
    device = torch.device('cuda')
    model = SD35ScratchRestorer(args.model_id).to(device, dtype=torch.bfloat16)
    model.train()
    x = torch.randn(1, model.in_channels, 32, 32, device=device, dtype=torch.bfloat16)
    c = torch.randn_like(x)
    t = torch.full((1,), 500.0, device=device, dtype=torch.bfloat16)
    y = model(x, c, t)
    loss = y.float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(loss).item() and any(g is not None and torch.isfinite(g).all() for g in grads)
    y2 = model(x, torch.zeros_like(c), t).detach()
    assert not torch.allclose(y.detach(), y2), 'condition path appears inactive'
    print('PASS: forward, backward, finite loss/gradients, condition sensitivity')


if __name__ == '__main__':
    main()
