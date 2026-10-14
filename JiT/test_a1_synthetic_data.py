"""Pure data/protocol tests; no Torch imports or model execution."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from a1_synthetic_data import (SyntheticPairedDataset, horizontal_flip,
                               read_truth, sha256_file, validate_truth)


class SyntheticDataTests(unittest.TestCase):
    def test_probability_contract_and_channel_order(self):
        good = np.zeros((3, 4, 4), dtype=np.float32)
        good[0] = 1
        validate_truth(good)
        for bad in (good * 0, good.astype(np.float64), good[:2], good - 0.1,
                    np.full_like(good, np.nan)):
            with self.assertRaises(ValueError):
                validate_truth(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.npz"
            np.savez(path, probabilities=good, channel_order=np.array("heavy,medium,light"))
            with self.assertRaises(ValueError):
                read_truth(path)

    def test_horizontal_flip_aligns_native_field_and_rgb(self):
        blur = np.arange(8 * 8 * 3).reshape(8, 8, 3)
        clear = blur + 7
        field = np.zeros((3, 4, 4), dtype=np.float32)
        field[0, :, :2], field[2, :, 2:] = 1, 1
        b, c, f = horizontal_flip(blur, clear, field)
        np.testing.assert_array_equal(b[:, 0], blur[:, -1])
        np.testing.assert_array_equal(c[:, 0], clear[:, -1])
        self.assertTrue((f[2, :, :2] == 1).all())
        for original, restored in zip((blur, clear, field), horizontal_flip(b, c, f)):
            np.testing.assert_array_equal(original, restored)

    def test_dataset_rejects_changed_truth_and_identity_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blur").mkdir(); (root / "clear").mkdir()
            for role in ("blur", "clear"):
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / role / "sample.png")
            field = np.zeros((3, 8, 8), dtype=np.float32); field[0] = 1
            truth = root / "truth.npz"
            np.savez(truth, probabilities=field, channel_order=np.array("light,medium,heavy"))
            row = {"sample_id": "sample", "split": "train", "quality_status": "pass",
                   "generation_kind": "known_local_gaussian_mixture",
                   "blur_path": "blur/sample.png", "clear_path": "clear/sample.png",
                   "truth_probability_path": "truth.npz", "truth_sha256": sha256_file(truth),
                   "blur_sha256": sha256_file(root / "blur/sample.png"),
                   "clear_sha256": sha256_file(root / "clear/sample.png")}
            manifest = root / "train.jsonl"; manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(len(SyntheticPairedDataset(manifest, split="train", training=True, root=root)), 1)
            with self.assertRaises(ValueError):
                SyntheticPairedDataset(manifest, split="train", training=True, root=root, identity_ratio=.1)
            field[:] = 1 / 3
            np.savez(truth, probabilities=field, channel_order=np.array("light,medium,heavy"))
            with self.assertRaises(ValueError):
                SyntheticPairedDataset(manifest, split="train", training=True, root=root)


if __name__ == "__main__":
    unittest.main()
