"""Build three synthetic PSF labels from the fixed 3DHistech clear-image split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from psf import LEVELS


class PairedClearSyntheticDataset(Dataset):
    def __init__(self, project_root: str | Path, manifest_path: str | Path, split: str) -> None:
        self.root = Path(project_root).resolve()
        self.records: list[tuple[Path, int, str]] = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split or record.get("quality_status") != "pass":
                    continue
                clear_path = self.root / record["clear_path"]
                for level_id, (name, _, _) in enumerate(LEVELS):
                    self.records.append((clear_path, level_id, f"{record['sample_id']}__{name}"))
        if not self.records:
            raise ValueError(f"No pass records for {split} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, level_id, sample_id = self.records[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255), level_id, sample_id
