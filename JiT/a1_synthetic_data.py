"""Known-field experiment data; no model imports until tensor conversion."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common_io import load_manifest, load_pair


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_truth(field):
    field = np.asarray(field)
    if field.ndim != 3 or field.shape[0] != 3 or min(field.shape[1:]) < 1:
        raise ValueError("Truth must be [3,h,w], light/medium/heavy")
    if field.dtype != np.float32 or not np.isfinite(field).all():
        raise ValueError("Truth must be finite float32")
    if field.min() < 0 or field.max() > 1:
        raise ValueError("Truth probabilities must be in [0,1]")
    if not np.allclose(field.sum(axis=0), 1, atol=1e-5, rtol=0):
        raise ValueError("Truth channels must sum to one")
    return field


def read_truth(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["channel_order"].item()) != "light,medium,heavy":
            raise ValueError("Incorrect oracle channel order")
        return validate_truth(data["probabilities"].copy())


def horizontal_flip(blur, clear, truth):
    """Flip exactly the same spatial direction at the two resolutions."""
    return blur[:, ::-1].copy(), clear[:, ::-1].copy(), truth[:, :, ::-1].copy()


class SyntheticPairedDataset:
    """All arms read the SAME PNG pairs and actual generation fields.

    Inputs must already be image_size square. There is no resize or crop:
    this preserves the exact field used for synthesis. Only a paired horizontal
    flip is applied during training, as in the existing 256px patch task.
    """
    def __init__(self, manifest_path, *, split, image_size=256, training,
                 identity_ratio=0, root=PROJECT_ROOT, data_root=None):
        if identity_ratio != 0:
            raise ValueError("Synthetic oracle protocol requires identity_ratio=0")
        self.root = Path(data_root if data_root is not None else root).resolve()
        path = Path(manifest_path)
        path = path if path.is_absolute() else self.root / path
        self.records = load_manifest(path, root=self.root, expected_split=split)
        if not self.records:
            raise ValueError("Synthetic manifest is empty")
        self.image_size, self.training = image_size, training
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
        self.truth_paths = {}
        records_by_id = {r.sample_id: r for r in self.records}
        for row in raw:
            if row.get("generation_kind") != "known_local_gaussian_mixture":
                raise ValueError("This entry only accepts generated synthetic manifests")
            truth = Path(row["truth_probability_path"])
            truth = truth if truth.is_absolute() else self.root / truth
            if not truth.is_file():
                raise FileNotFoundError(truth)
            if not truth.resolve().is_relative_to(self.root):
                raise ValueError("Truth resolves outside the explicitly selected data root")
            if sha256_file(truth) != row["truth_sha256"]:
                raise ValueError(f"Truth file changed: {row['sample_id']}")
            record = records_by_id[row["sample_id"]]
            for role in ("blur", "clear"):
                if sha256_file(getattr(record, role + "_path")) != row[role + "_sha256"]:
                    raise ValueError(f"{role} image changed: {record.sample_id}")
            self.truth_paths[row["sample_id"]] = truth

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        blur, clear = load_pair(record)
        if blur.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"{record.sample_id}: exact square source patches required")
        truth = read_truth(self.truth_paths[record.sample_id])
        if self.training and bool(torch.rand(()) < 0.5):
            blur, clear, truth = horizontal_flip(blur, clear, truth)
        return {
            "sample_id": record.sample_id,
            "blur": torch.from_numpy(blur.transpose(2, 0, 1).copy()).mul(2).sub(1),
            "clear": torch.from_numpy(clear.transpose(2, 0, 1).copy()).mul(2).sub(1),
            "truth_probabilities": torch.from_numpy(truth.copy()),
            "is_identity": torch.tensor(False),
        }
