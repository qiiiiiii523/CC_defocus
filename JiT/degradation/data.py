"""Read the fixed CNSeg split and synthetic labels without requiring masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from psf import LEVELS


class CNSegSigmaDataset(Dataset):
    def __init__(self, project_root: str | Path, manifest_path: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(project_root).resolve()
        self.records: list[tuple[Path, int, str]] = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split or record["quality_status"] != "pass":
                    continue
                degradation = record["degradation"]
                level = degradation["severity"]
                level_id = next((i for i, row in enumerate(LEVELS) if row[0] == level), None)
                if level_id is None:
                    raise ValueError(f"Unknown severity in {record['sample_id']}")
                _, sigma, kernel_size = LEVELS[level_id]
                if degradation["kernel_type"] != "gaussian" or degradation["padding"] != "reflect":
                    raise ValueError(f"Unsupported PSF in {record['sample_id']}")
                if float(degradation["sigma"]) != sigma or int(degradation["kernel_size"]) != kernel_size:
                    raise ValueError(f"Manifest PSF does not match {record['sample_id']}")
                self.records.append((self.root / record["clear_path"], level_id, record["sample_id"]))
        if not self.records:
            raise ValueError(f"No pass records for {split} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, level_id, sample_id = self.records[index]
        with Image.open(path) as image:
            image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        clear = torch.from_numpy(image_array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        return clear, level_id, sample_id
