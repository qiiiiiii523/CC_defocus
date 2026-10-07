"""CPU plumbing checks; run on the server: python -m unittest test_a1_integration.

No training/full JiT forward: a tiny stand-in isolates condition and loader logic.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from degradation.condition import DegradationCondition
from denoiser import Denoiser
from main_restoration import _resolve_degradation_mode


class TinyNet(nn.Module):
    def __init__(self, degradation_conditioning=False, **kwargs):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.2))
        self.degradation_condition = (
            DegradationCondition(3) if degradation_conditioning else None
        )

    def forward(self, z, t, labels, blur=None, degradation=None):
        output = blur * self.bias
        if degradation is not None:
            output = output + self.degradation_condition(degradation)[:, :, None, None]
        return output


def make_model(mode):
    args = SimpleNamespace(
        model="tiny", img_size=32, class_num=1000, attn_dropout=0., proj_dropout=0.,
        label_drop_prob=0., P_mean=-.8, P_std=.8, t_eps=.05, noise_scale=1.,
        lambda_pix=1., charbonnier_eps=.001, ema_decay1=.9999, ema_decay2=.9996,
        sampling_method="euler", num_sampling_steps=1, cfg=1.,
        interval_min=0., interval_max=1., degradation_mode=mode,
    )
    with patch.dict("denoiser.JiT_models", {"tiny": TinyNet}):
        return Denoiser(args)


class IntegrationTests(unittest.TestCase):
    def test_resume_mode(self):
        self.assertEqual(_resolve_degradation_mode(None), "none")
        self.assertEqual(_resolve_degradation_mode(None, {"args": {"degradation_mode": "predicted"}}),
                         "predicted")
        with self.assertRaises(ValueError):
            _resolve_degradation_mode("none", {"args": {"degradation_mode": "fixed"}})

    def test_zero_gate_preserves_a0_and_one_step(self):
        base, fixed = make_model("none"), make_model("fixed")
        fixed.load_a0_state_dict(base.state_dict())
        blur, noise = torch.rand(2, 3, 32, 32) * 2 - 1, torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(base.generate(blur, noise=noise),
                                   fixed.generate(blur, noise=noise), rtol=0, atol=0)
        expected = fixed.net(noise, torch.zeros(2), fixed._null_labels(2, noise.device),
                             blur=blur, degradation=fixed._degradation_probabilities(blur))
        torch.testing.assert_close(fixed.generate(blur, noise=noise), expected)
        fixed(blur, blur).backward()
        self.assertIsNotNone(fixed.net.degradation_condition.gate.grad)

    def test_frozen_estimator_and_sampling_cache(self):
        model = make_model("predicted").train()
        estimator = model.degradation_estimator
        self.assertFalse(estimator.training)
        self.assertTrue(all(not p.requires_grad for p in estimator.parameters()))
        buffers = {key: value.clone() for key, value in estimator.named_buffers()}
        blur = torch.rand(2, 3, 32, 32) * 2 - 1
        probabilities = model._degradation_probabilities(blur)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
        model(blur, blur).backward()
        self.assertTrue(all(p.grad is None for p in estimator.parameters()))
        for key, value in estimator.named_buffers():
            torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
        model.steps, model.method = 3, "heun"
        with patch.object(estimator, "forward", wraps=estimator.forward) as call:
            model.generate(blur)
            self.assertEqual(call.call_count, 1)
        restored = make_model("predicted")
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_a0_loader_rejects_missing_backbone(self):
        with self.assertRaises(RuntimeError):
            make_model("fixed").load_a0_state_dict({})


if __name__ == "__main__":
    unittest.main()
