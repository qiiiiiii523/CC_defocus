"""SERVER-only small-model oracle routing checks; not run on the local PC."""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_a1_local_integration import local_model
from main_restoration import _initialize_ema, _load_ema


class OracleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.attention = patch("model_jit.scaled_dot_product_attention", side_effect=F.scaled_dot_product_attention)
        self.attention.start()
        self.addCleanup(self.attention.stop)

    def test_a0_loading_zero_gate_gradient_and_sampling(self):
        base = local_model("none")
        model = local_model("oracle_local")
        model.load_a0_state_dict(base.state_dict())
        self.assertIsNone(model.degradation_estimator)
        blur = torch.rand(1, 3, 32, 32) * 2 - 1
        truth = torch.zeros(1, 3, 8, 8); truth[:, 0] = 1
        noise = torch.randn_like(blur)
        torch.testing.assert_close(base.generate(blur, noise=noise), model.generate(blur, degradation=truth, noise=noise), rtol=0, atol=0)
        model(torch.rand_like(blur), blur, degradation=truth).backward()
        self.assertGreater(model.net.local_degradation_condition.gate.grad.abs().item(), 0)
        with self.assertRaises(ValueError):
            model.generate(blur, noise=noise)
        with self.assertRaises(ValueError):
            model.generate(blur, degradation=truth * 0, noise=noise)
        with self.assertRaises(ValueError):
            base.generate(blur, degradation=truth, noise=noise)

    def test_oracle_state_and_ema_roundtrip(self):
        model = local_model("oracle_local")
        _initialize_ema(model)
        replica = local_model("oracle_local")
        replica.load_state_dict(model.state_dict(), strict=True)
        checkpoint = {"model_ema1": model.state_dict(), "model_ema2": model.state_dict()}
        _load_ema(replica, checkpoint, torch.device("cpu"))
        for left, right in zip(model.ema_params2, replica.ema_params2):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
