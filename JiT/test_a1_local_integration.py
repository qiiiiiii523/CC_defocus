"""Local A1 checks: python -m unittest test_a1_integration test_a1_local_integration.

The small real JiT uses CPU attention and disables compilation in tests only.
This checks wiring, not the full B/16 GPU training environment or metrics.
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from degradation.local_condition import LocalDegradationCondition
from degradation.local_inference import LOCAL_LEVELS, load_local_estimator
from degradation.local_model import LocalBlurEstimator
from denoiser import Denoiser
from main_restoration import get_args_parser, _resolve_degradation_mode, _initialize_ema, _load_ema
from model_jit import JiT
from test_a1_integration import make_model


def small_jit(**kwargs):
    return JiT(patch_size=8, hidden_size=32, depth=2, num_heads=4,
               bottleneck_dim=16, in_context_len=4, in_context_start=1, **kwargs)


def local_model(mode):
    args = get_args_parser().parse_args(["--degradation_mode", mode, "--img_size", "32",
                                        "--sampling_method", "euler", "--num_sampling_steps", "1"])
    args.model = "small"
    with patch.dict("denoiser.JiT_models", {"small": small_jit}):
        model = Denoiser(args)
    # A restored A0 has trained output weights; a zero output head would mask
    # conditioning errors and prevent the new gate from learning in this check.
    with torch.no_grad():
        nn.init.normal_(model.net.final_layer.linear.weight, std=.02)
    return model


class LocalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.attention = patch("model_jit.scaled_dot_product_attention",
                               side_effect=F.scaled_dot_product_attention)
        self.attention.start()
        self.addCleanup(self.attention.stop)

    def test_cli_and_resume_keep_global_modes(self):
        for mode in ("none", "fixed", "predicted", "fixed_local", "predicted_local"):
            self.assertEqual(get_args_parser().parse_args(["--degradation_mode", mode]).degradation_mode, mode)
            self.assertEqual(_resolve_degradation_mode(None, {"args": {"degradation_mode": mode}}), mode)
        with self.assertRaises(ValueError):
            _resolve_degradation_mode("predicted", {"args": {"degradation_mode": "predicted_local"}})

    def test_local_grid_order_and_odd_image_shape(self):
        adapter = LocalDegradationCondition(3)
        adapter.embedding = nn.Identity()
        adapter.gate.data.fill_(1)
        field = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [0., 0.]], [[0., 0.], [1., 0.]]]])
        expected = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
        torch.testing.assert_close(adapter(field, (2, 2)), expected)
        self.assertEqual(LocalBlurEstimator()(torch.rand(1, 3, 33, 35)).shape, (1, 3, 9, 9))

    def test_actual_jit_zero_gate_preserves_a0_and_learns(self):
        base = local_model("none")
        blur, noise = torch.rand(1, 3, 32, 32) * 2 - 1, torch.randn(1, 3, 32, 32)
        for mode in ("fixed_local", "predicted_local"):
            model = local_model(mode)
            model.load_a0_state_dict(base.state_dict())
            torch.testing.assert_close(base.generate(blur, noise=noise), model.generate(blur, noise=noise),
                                       rtol=0, atol=0)
            loss = model(torch.rand_like(blur), blur)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gate_grad = model.net.local_degradation_condition.gate.grad
            self.assertIsNotNone(gate_grad)
            self.assertGreater(gate_grad.abs().item(), 0)
            self.assertIsNotNone(model.net.final_layer.linear.weight.grad)

    def test_injection_reaches_only_aligned_image_tokens(self):
        model = local_model("fixed_local")
        net = model.net
        field = torch.zeros(1, 3, 4, 4)
        field[:, 0, :, :2], field[:, 2, :, 2:] = 1., 1.
        captures = []
        handle = net.blocks[0].register_forward_pre_hook(lambda module, args: captures.append(args[0].detach().clone()))
        self.addCleanup(handle.remove)
        z, blur = torch.randn(1, 3, 32, 32), torch.rand(1, 3, 32, 32) * 2 - 1
        labels = model._null_labels(1, z.device)
        net(z, torch.zeros(1), labels, blur=blur, degradation=field)
        net.local_degradation_condition.gate.data.fill_(1.)
        net(z, torch.zeros(1), labels, blur=blur, degradation=field)
        expected = net.local_degradation_condition(field, (4, 4)).detach()
        self.assertEqual(captures[0].shape, (1, 16, 32))
        torch.testing.assert_close(captures[1] - captures[0], expected, atol=1e-6, rtol=1e-5)
        self.assertGreater((expected[:, 0] - expected[:, 3]).abs().max().item(), 0)

    def test_delivered_weights_blur_only_freezing_and_cache(self):
        model = local_model("predicted_local")
        model.load_degradation_estimator(Path(__file__).parent / "degradation/weights/local-estimator.pt")
        model.train()
        estimator = model.degradation_estimator
        self.assertFalse(estimator.training)
        self.assertTrue(all(not p.requires_grad for p in estimator.parameters()))
        blur = torch.rand(1, 3, 32, 32) * 2 - 1
        with patch.object(estimator, "forward", wraps=estimator.forward) as call:
            probabilities = model._degradation_probabilities(blur)
            torch.testing.assert_close(call.call_args.args[0], (blur + 1) * .5)
        self.assertEqual(probabilities.shape, (1, 3, 8, 8))
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1, 8, 8))
        self.assertFalse(probabilities.requires_grad)
        model(blur, blur).backward()
        self.assertTrue(all(p.grad is None for p in estimator.parameters()))
        model.steps, model.method = 3, "heun"
        with patch.object(estimator, "forward", wraps=estimator.forward) as call:
            self.assertTrue(torch.isfinite(model.generate(blur)).all())
            self.assertEqual(call.call_count, 1)

    def test_local_checkpoint_and_ema_roundtrip(self):
        model = local_model("predicted_local")
        _initialize_ema(model)
        with torch.no_grad():
            next(model.degradation_estimator.parameters()).add_(.1)
        model.update_ema()
        ema1 = dict(zip((name for name, _ in model.named_parameters()), model.ema_params1))
        ema2 = dict(zip((name for name, _ in model.named_parameters()), model.ema_params2))
        for name, value in model.degradation_estimator.named_parameters():
            torch.testing.assert_close(ema2["degradation_estimator." + name], value, rtol=0, atol=0)
        restored = local_model("predicted_local")
        restored.load_state_dict(model.state_dict(), strict=True)
        _load_ema(restored, {"model_ema1": ema1, "model_ema2": ema2}, torch.device("cpu"))
        torch.testing.assert_close(restored.ema_params2[-1], model.ema_params2[-1])
        with self.assertRaises(ValueError):
            restored.load_a0_state_dict(model.state_dict())
        with self.assertRaises(RuntimeError):
            restored.load_a0_state_dict({})

    def test_loader_rejects_wrong_channel_order_and_global_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"levels": LOCAL_LEVELS[::-1], "model": LocalBlurEstimator().state_dict()}, path)
            with self.assertRaises(ValueError):
                load_local_estimator(path)
            torch.save({"levels": LOCAL_LEVELS, "model": make_model("predicted").degradation_estimator.state_dict()}, path)
            with self.assertRaises(RuntimeError):
                load_local_estimator(path)

    def test_loader_accepts_delivered_v3_and_rejects_unknown_missing_metadata(self):
        path = Path(__file__).parent / "degradation/weights/local-estimator-v3.pt"
        estimator = load_local_estimator(path)
        self.assertFalse(estimator.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in estimator.parameters()))
        model = local_model("predicted_local")
        model.load_degradation_estimator(path)
        probabilities = model._degradation_probabilities(torch.rand(1, 3, 32, 32) * 2 - 1)
        self.assertEqual(probabilities.shape, (1, 3, 8, 8))
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1, 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.pt"
            torch.save({"model": estimator.state_dict()}, unknown)
            with self.assertRaisesRegex(ValueError, "no levels metadata"):
                load_local_estimator(unknown)


if __name__ == "__main__":
    unittest.main()
