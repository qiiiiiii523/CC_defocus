"""Pure tests for check metrics and eval-only command routing; no Torch."""
import unittest
from pathlib import Path
import numpy as np
from check_a1_reblur import quantization_comparison
from run_a1_exp_checks import build_eval_command, paired_difference


class CheckTests(unittest.TestCase):
    def test_exact_rounding_and_mismatch_are_distinct(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertEqual(quantization_comparison(a, a)["status"], "exact")
        b = a.copy(); b[0, 0, 0] = 1
        self.assertEqual(quantization_comparison(a, b)["status"], "rounding_tolerance")
        b[0, 0, 0] = 2
        self.assertEqual(quantization_comparison(a, b)["status"], "mismatch")
        self.assertFalse(quantization_comparison(a, np.ones_like(a))["passed"])

    def test_commands_only_evaluate_and_preserve_original_outputs(self):
        checkpoint = Path('/project/checkpoints/original/checkpoint-last.pth')
        out = Path('/project/reports/checks')
        cfg = dict(model='JiT-B/16', degradation_mode='oracle_local', amp_bf16=True,
                   sampling_method='euler', num_sampling_steps=1, init_mode='scratch')
        cmd = build_eval_command(cfg, checkpoint, out, 's2_new')
        self.assertIn('--eval_only', cmd)
        self.assertEqual(cmd[cmd.index('--resume') + 1], str(checkpoint))
        self.assertEqual(cmd[cmd.index('--eval_weights') + 1], 'model')
        self.assertNotIn('--a0_checkpoint', cmd)
        self.assertNotIn('--degradation_checkpoint', cmd)
        self.assertNotEqual(cmd[cmd.index('--output_dir') + 1], str(checkpoint.parent))
        self.assertTrue(cmd[cmd.index('--prediction_root') + 1].startswith(str(out)))

    def test_pairing_checks_ids_and_metric_directions(self):
        a = {'x': dict(psnr=32., ssim=.9, lpips=.1)}
        b = {'x': dict(psnr=31., ssim=.8, lpips=.2)}
        result = paired_difference(a, b)
        self.assertEqual(result['psnr']['wins'], 1)
        self.assertEqual(result['lpips']['wins'], 1)
        self.assertLess(result['lpips']['delta'], 0)
        with self.assertRaises(ValueError):
            paired_difference(a, {'other': b['x']})


if __name__ == '__main__':
    unittest.main()
