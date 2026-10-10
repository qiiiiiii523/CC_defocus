"""Paired data; augmentation is reproducible across workers and resumed epochs."""
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common_io import load_manifest, load_pair


class PairedManifest(Dataset):
    def __init__(self, manifest, repo_root, size=256, train=True,
                 identity_ratio=0.0, max_samples=0, seed=20261010, augment=True):
        if size <= 0 or size % 16:
            raise ValueError("resolution must be a positive multiple of 16")
        if not 0 <= identity_ratio <= 1 or (not train and identity_ratio):
            raise ValueError("identity_ratio must be in [0,1] and zero for validation")
        if max_samples < 0:
            raise ValueError("max_samples must be nonnegative")
        self.records = load_manifest(manifest, root=repo_root,
                                     expected_split="train" if train else "val")
        if not self.records:
            raise ValueError("Manifest is empty")
        if max_samples:
            if max_samples > len(self.records):
                raise ValueError("overfit_samples exceeds manifest length")
            self.records = self.records[:max_samples]
        self.size, self.train, self.identity_ratio = size, train, identity_ratio
        self.seed, self.epoch, self.augment = seed, 0, augment

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        blur, clear = load_pair(record)
        blur = torch.from_numpy(blur.transpose(2, 0, 1).copy())
        clear = torch.from_numpy(clear.transpose(2, 0, 1).copy())
        height, width = blur.shape[-2:]
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + i)
        if self.train:
            if min(height, width) < self.size:
                raise ValueError("Image is smaller than training crop: " + record.sample_id)
            if self.augment:
                top = int(torch.randint(height - self.size + 1, (), generator=generator))
                left = int(torch.randint(width - self.size + 1, (), generator=generator))
            else:
                top, left = (height - self.size) // 2, (width - self.size) // 2
            blur = blur[:, top:top+self.size, left:left+self.size]
            clear = clear[:, top:top+self.size, left:left+self.size]
            if self.augment and torch.rand((), generator=generator) < 0.5:
                blur, clear = blur.flip(-1), clear.flip(-1)
            if self.identity_ratio and torch.rand((), generator=generator) < self.identity_ratio:
                blur = clear.clone()
        elif (height, width) != (self.size, self.size):
            raise ValueError("Validation images must retain their original size; define tiling first")
        return {"blur": blur.contiguous() * 2 - 1,
                "clear": clear.contiguous() * 2 - 1,
                "sample_id": record.sample_id}
