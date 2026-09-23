"""PyTorch adapter for the shared 3DHistech manifest and image protocol."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from common_io import SampleRecord, load_pair


class PairedImages(Dataset):
    def __init__(self, records: list[SampleRecord]) -> None:
        if not records:
            raise ValueError("The training/evaluation manifest is empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        blur, clear = load_pair(self.records[index])
        # common_io returns HWC RGB float32 [0,1]; PyTorch expects CHW.
        blur_tensor = torch.from_numpy(blur.transpose(2, 0, 1).copy())
        clear_tensor = torch.from_numpy(clear.transpose(2, 0, 1).copy())
        return blur_tensor, clear_tensor
