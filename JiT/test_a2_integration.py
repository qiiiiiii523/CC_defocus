"""Server-only A2 tensor/gradient checks; not executed on the local machine."""
import unittest
from unittest.mock import patch

import torch

from degradation.a2_loss import local_psf_loss
from degradation.psf import gaussian_blur
from degradation.spatial_psf import psf_consistency_loss, reblur_spatial_mixture
from test_a1_integration import make_model


class A2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constant_medium_field_matches_known_gaussian(self):
        image = torch.rand(1, 3, 32, 32)
        field = torch.zeros(1, 3, 8, 8)
        field[:, 1] = 1
        torch.testing.assert_close(reblur_spatial_mixture(image, field), gaussian_blur(image, 2., 13))

    def test_no_injection_frozen_map_and_weighted_objective(self):
        model = make_model("psf_local")
        model.lambda_psf = .1
        field = torch.full((2, 3, 8, 8), 1 / 3, requires_grad=True)
        blur = torch.rand(2, 3, 32, 32) * 2 - 1
        with patch.object(model, "_degradation_probabilities", return_value=field), \
                patch.object(model.net, "forward", wraps=model.net.forward) as net:
            losses = model(torch.rand_like(blur), blur, return_loss_components=True,
                           is_identity=torch.zeros(2, dtype=torch.bool))
        self.assertIsNone(net.call_args.kwargs["degradation"])
        self.assertIsNone(model.net.degradation_condition)
        torch.testing.assert_close(losses["loss_psf_weighted"], .1 * losses["loss_psf"])
        torch.testing.assert_close(losses["loss"], losses["loss_flow"] + losses["loss_pix_weighted"] + losses["loss_psf_weighted"])
        losses["loss_psf"].backward()
        self.assertGreater(model.net.bias.grad.abs().item(), 0)
        self.assertIsNone(field.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.degradation_estimator.parameters()))

    def test_identity_samples_are_excluded_and_all_identity_is_zero(self):
        prediction = torch.rand(2, 3, 32, 32, requires_grad=True)
        blur = torch.rand_like(prediction)
        field = torch.full((2, 3, 8, 8), 1 / 3)
        loss = local_psf_loss(prediction, blur, field, is_identity=torch.tensor([True, False]))
        expected = psf_consistency_loss((prediction[1:] + 1) * .5, (blur[1:] + 1) * .5, field[1:])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(prediction.grad[0], torch.zeros_like(prediction.grad[0]))
        prediction.grad = None
        zero = local_psf_loss(prediction, blur, field, is_identity=torch.ones(2, dtype=torch.bool))
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_out_of_range_prediction_still_gets_gradient(self):
        prediction = torch.full((1, 3, 32, 32), 3., requires_grad=True)
        field = torch.full((1, 3, 8, 8), 1 / 3, requires_grad=True)
        local_psf_loss(prediction, torch.zeros_like(prediction), field).backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertIsNone(field.grad)

    def test_a0_loading_inference_and_checkpoint_roundtrip(self):
        base, model = make_model("none"), make_model("psf_local")
        model.load_a0_state_dict(base.state_dict())
        blur, noise = torch.rand(2, 3, 32, 32) * 2 - 1, torch.randn(2, 3, 32, 32)
        with patch.object(model.degradation_estimator, "forward", side_effect=AssertionError("A2 inference invoked estimator")):
            torch.testing.assert_close(model.generate(blur, noise=noise), base.generate(blur, noise=noise), rtol=0, atol=0)
        restored = make_model("psf_local")
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(all(not p.requires_grad for p in restored.degradation_estimator.parameters()))


if __name__ == "__main__":
    unittest.main()
