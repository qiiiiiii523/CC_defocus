"""Train on real 3DHistech blur using coarse labels fitted from paired images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LEVEL_ID = {"identity": 0, "light": 0, "medium": 1, "heavy": 2}


class PairedPseudoDataset(Dataset):
    def __init__(self, project_root: str | Path, manifest_path: str | Path, labels_path: str | Path, split: str) -> None:
        root = Path(project_root).resolve()
        with Path(labels_path).open("r", newline="", encoding="utf-8") as handle:
            labels = {row["sample_id"]: LEVEL_ID[row["best_setting"]] for row in csv.DictReader(handle)}
        self.records: list[tuple[Path, int, str]] = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split or record.get("quality_status") != "pass":
                    continue
                sample_id = record["sample_id"]
                if sample_id not in labels:
                    raise ValueError(f"Missing pseudo label: {sample_id}")
                self.records.append((root / record["blur_path"], labels[sample_id], sample_id))
        if not self.records or len(self.records) != len(labels):
            raise ValueError(f"Manifest and pseudo-label CSV differ for {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, level_id, sample_id = self.records[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255), level_id, sample_id
