"""K0 CNSeg pairs: explicit Gaussian/reflect synthesis, aligned instance masks.

Validation uses a documented center crop, never silent resizing. Blur is
generated on the full image before cropping to avoid crop-edge synthesis artifacts.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def gaussian_blur(clear, degradation):
    if degradation.get("kernel_type") != "gaussian" or degradation.get("padding") != "reflect":
        raise ValueError("K0 currently supports only declared Gaussian kernels with reflect padding")
    size, sigma = degradation["kernel_size"], degradation["sigma"]
    if not isinstance(size, int) or size < 1 or size % 2 != 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid Gaussian kernel size/sigma")
    radius = size // 2
    if min(clear.shape[-2:]) <= radius:
        raise ValueError("Image too small for declared reflect kernel")
    grid = torch.arange(size, dtype=clear.dtype, device=clear.device) - radius
    kernel = torch.exp(-grid.square() / (2 * sigma ** 2))
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / kernel.sum()
    channels = clear.shape[0]
    return F.conv2d(F.pad(clear[None], (radius,) * 4, mode="reflect"),
                    kernel.expand(channels, 1, size, size), groups=channels)[0]


class CNSegPairedDataset(Dataset):
    def __init__(self, manifest_path, *, split, image_size=256, training,
                 identity_ratio=0.0, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self.image_size, self.training, self.identity_ratio = image_size, training, identity_ratio
        if image_size < 1 or not 0 <= identity_ratio <= 1 or (not training and identity_ratio):
            raise ValueError("Invalid CNSeg crop/identity settings")
        path = Path(manifest_path)
        if not path.is_absolute():
            path = self.root / path
        self.raw, self.records = [], []
        seen = set()
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw["split"] != split or raw.get("quality_status") != "pass":
                    continue
                sid = raw["sample_id"]
                if not sid or Path(sid).name != sid or "/" in sid or "\\" in sid or sid in seen:
                    raise ValueError(f"Invalid/duplicate CNSeg sample_id: {sid}")
                seen.add(sid)
                clear = self._path(raw["clear_path"])
                mask = self._path(raw["instance_mask_path"])
                for file in (clear, mask):
                    if not file.is_file():
                        raise FileNotFoundError(f"CNSeg {sid}: missing {file}; request B's data handoff")
                self.raw.append(raw)
                self.records.append(SimpleNamespace(sample_id=sid, clear_path=clear,
                                                    blur_path=clear, mask_path=mask, group_id=raw["group_id"]))
        if not self.records:
            raise ValueError(f"No passing CNSeg samples for split={split}")
        # Check group/image leakage against all other splits, including test.
        own_groups = {r["group_id"] for r in self.raw}
        own_images = {r.clear_path for r in self.records}
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw["split"] != split and (raw["group_id"] in own_groups or self._path(raw["clear_path"]) in own_images):
                    raise ValueError(f"CNSeg split leakage: {raw['sample_id']}")

    def _path(self, value):
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("CNSeg path escapes project root")
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record, raw = self.records[index], self.raw[index]
        with Image.open(record.clear_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(record.mask_path) as image:
            labels = np.array(image)
        if labels.ndim != 2 or labels.shape != array.shape[:2] or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError(f"CNSeg {record.sample_id}: mask must be aligned nonnegative integer instance IDs")
        clear = torch.from_numpy(array.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(labels.astype(np.int64))
        blur = gaussian_blur(clear, raw["degradation"])
        height, width = mask.shape
        size = self.image_size
        if min(height, width) < size:
            raise ValueError(f"CNSeg {record.sample_id}: image smaller than crop {size}; no resizing is performed")
        if self.training:
            top = int(torch.randint(height-size+1, ()).item())
            left = int(torch.randint(width-size+1, ()).item())
        else:
            top, left = (height-size)//2, (width-size)//2
        clear, blur = (x[:, top:top+size, left:left+size] for x in (clear, blur))
        mask = mask[top:top+size, left:left+size]
        if self.training and torch.rand(()) < .5:
            clear, blur, mask = (torch.flip(x, (-1,)) for x in (clear, blur, mask))
        identity = self.training and bool(torch.rand(()) < self.identity_ratio)
        if identity:
            blur = clear.clone()
        return dict(blur=blur*2-1, clear=clear*2-1, instance_mask=mask,
                    sample_id=record.sample_id, is_identity=torch.tensor(identity))
