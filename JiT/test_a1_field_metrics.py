"""Pure NumPy/PIL checks; no Torch, model inference or GPU probes."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from a1_field_metrics import (aggregate_rows, area_weights, compression_rows,
                              field_rows, region_masks, weighted_channel_stats)
from diagnose_a1_fields import make_report, save_panel, write_json


class FieldMetricTests(unittest.TestCase):
    def test_constant_bias_and_spatial_pattern_are_distinguished(self):
        truth = np.tile(np.linspace(.1, .7, 4), (4, 1))
        stats = weighted_channel_stats(truth + .15, truth, np.ones((4, 4)))
        self.assertAlmostEqual(stats["bias"], .15)
        self.assertAlmostEqual(stats["rmse"], .15)
        self.assertAlmostEqual(stats["centered_rmse"], 0)
        self.assertAlmostEqual(stats["spatial_corr"], 1)
        same = weighted_channel_stats(truth, truth, np.ones((4, 4)))
        self.assertEqual(same["mae"], 0)
        self.assertGreater(same["spatial_mae_gain"], 0)
        wrong = weighted_channel_stats(truth[:, ::-1], truth, np.ones((4, 4)))
        self.assertAlmostEqual(wrong["spatial_corr"], -1)
        self.assertLess(wrong["spatial_mae_gain"], 0)

    def test_constant_correlation_is_undefined_and_empty_region_is_excluded(self):
        a = np.ones((4, 4)) * .3
        self.assertIsNone(weighted_channel_stats(a, a, np.ones_like(a))["spatial_corr"])
        self.assertIsNone(weighted_channel_stats(a, a, np.zeros_like(a)))
        p = np.full((3, 4, 4), 1 / 3, dtype=np.float32)
        masks = region_masks(np.ones((16, 16, 3)))
        rows = field_rows(p, p, masks, sample_id="a", resolution="native64")
        self.assertNotIn("tissue", {r["region"] for r in rows})
        summary = aggregate_rows(rows)
        self.assertEqual(summary["native64"]["background"]["light"]["image_count"], 1)
        self.assertIsNone(summary["native64"]["all"]["light"]["spatial_corr"]["mean"])
        self.assertEqual(summary["native64"]["all"]["light"]["spatial_corr"]["valid_images"], 0)

    def test_region_occupancy_preserves_fraction_and_partial_cell_weights(self):
        clear = np.ones((16, 16, 3), dtype=np.float32)
        clear[:, :5] = [.5, .2, .6]
        clear[:, 5:7] = [.88, .88, .88]
        masks = region_masks(clear)
        self.assertFalse((masks["tissue"] & masks["background"]).any())
        self.assertAlmostEqual(masks["tissue"].mean(), 5 / 16)
        self.assertAlmostEqual(masks["uncertain"].mean(), 2 / 16)
        weights = area_weights(masks["tissue"], (4, 4))
        self.assertAlmostEqual(weights.mean(), 5 / 16)
        self.assertAlmostEqual(weights[0, 1], .25)
        stats = weighted_channel_stats(np.ones((4, 4)), np.zeros((4, 4)), weights)
        self.assertEqual(stats["mae"], 1)

    def test_roundtrip_retention_does_not_mislabel_constant_as_zero_ratio(self):
        native = np.full((3, 4, 4), 1 / 3, dtype=np.float32)
        grid = native[:, :2, :2]
        rows = compression_rows(native, native, grid, grid, native, native, sample_id="x")
        self.assertIsNone(rows[0]["pred_std_ratio"])
        self.assertEqual(rows[0]["truth_roundtrip_mae"], 0)

    def test_shared_scale_panel_and_numeric_fields_need_no_model(self):
        p = np.zeros((3, 64, 64), dtype=np.float32); p[0] = 1
        clear = np.ones((256, 256, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.png"
            save_panel(path, clear, clear, region_masks(clear), p, p, p[:, ::4, ::4],
                       p[:, ::4, ::4], "test_sample")
            with Image.open(path) as im:
                self.assertEqual(im.size, (1536, 1152))

    def test_diagnostic_resize_contract_matches_production(self):
        here = Path(__file__).parent
        production = here / "degradation/local_condition.py"
        if not production.is_file():
            self.skipTest("Production source contract checked after copying into JiT")
        source = ast.parse(production.read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(source) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "interpolate"]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["mode"]), "bilinear")
        self.assertIs(ast.literal_eval(keywords["align_corners"]), False)
        self.assertNotIn("antialias", keywords)
        diagnostic = ast.parse((here / "diagnose_a1_fields.py").read_text(encoding="utf-8"))
        resized = [n for n in ast.walk(diagnostic) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "interpolate"]
        self.assertEqual(len(resized), 4)
        for call in resized:
            kws = {kw.arg: kw.value for kw in call.keywords}
            self.assertEqual(ast.literal_eval(kws["mode"]), "bilinear")
            self.assertIs(ast.literal_eval(kws["align_corners"]), False)
            self.assertNotIn("antialias", kws)

    def test_report_handles_missing_tissue_and_constant_correlations(self):
        pred = np.full((3, 64, 64), 1/3, dtype=np.float32)
        masks = region_masks(np.ones((256, 256, 3)))
        rows = field_rows(pred, pred, masks, sample_id="x", resolution="native64")
        rows += field_rows(pred[:, ::4, ::4], pred[:, ::4, ::4], masks,
                           sample_id="x", resolution="token16")
        summary = {"source_tag": "test", "evaluated_count": 1, "metrics": aggregate_rows(rows),
                   "resize_metrics": {c: {k: {"mean": None if "ratio" in k else 0., "valid_images": 0 if "ratio" in k else 1}
                                          for k in ("pred_std_ratio", "truth_std_ratio", "pred_roundtrip_mae", "truth_roundtrip_mae")}
                                      for c in ("light", "medium", "heavy")},
                   "validation_groups": ["x"]}
        images = [{"tissue_fraction": 0., "background_fraction": 1.}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            make_report(summary, images, out)
            write_json(out / "summary.json", summary)
            text = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("| tissue | 0 |", text)
            self.assertIn("NA", text)


if __name__ == "__main__":
    unittest.main()
