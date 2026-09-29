"""Paired manifest dataset for JiT A0 restoration.

The public project boundary is RGB float32 in [0, 1].  This dataset reuses
``common_io`` for manifest validation and image decoding, applies only paired
spatial augmentation, and converts both images to JiT's [-1, 1] range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common_io import SampleRecord, load_manifest, load_pair  # noqa: E402


class A0PairedDataset(Dataset):
    """Fixed blur/clear pairs with synchronized crop and flip.

    Training items may be converted to clear-to-same-clear identity pairs with
    probability ``identity_ratio``.  Validation never changes pair identity or
    image geometry because predictions must match the public reference size.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        image_size: int = 256,
        training: bool,
        identity_ratio: float = 0.0,
        root: str | Path = PROJECT_ROOT,
    ) -> None:
        if not 0.0 <= identity_ratio <= 1.0:
            raise ValueError(
                f"identity_ratio must be in [0, 1], got {identity_ratio}"
            )
        if not training and identity_ratio != 0.0:
            raise ValueError("identity_ratio must be 0 for validation")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")

        self.root = Path(root).resolve()
        self.image_size = image_size
        self.training = training
        self.identity_ratio = identity_ratio
        self.records: list[SampleRecord] = load_manifest(
            manifest_path,
            root=self.root,
            expected_split=split,
            required_quality_status="pass",
        )
        if not self.records:
            raise ValueError(f"No usable samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        blur_array, clear_array = load_pair(record)
        blur = _hwc_to_chw_tensor(blur_array)
        clear = _hwc_to_chw_tensor(clear_array)

        if self.training:
            blur, clear = _paired_random_crop(blur, clear, self.image_size)
            if torch.rand(()) < 0.5:
                blur = torch.flip(blur, dims=(-1,))
                clear = torch.flip(clear, dims=(-1,))
        else:
            actual_hw = tuple(blur.shape[-2:])
            expected_hw = (self.image_size, self.image_size)
            if actual_hw != expected_hw:
                raise ValueError(
                    f"Validation sample {record.sample_id} has size {actual_hw}, "
                    f"but JiT requires {expected_hw}. A tiling policy must be "
                    "defined before formal evaluation; validation images are "
                    "never silently resized or cropped."
                )

        is_identity = False
        if self.training and self.identity_ratio > 0.0:
            is_identity = bool(torch.rand(()) < self.identity_ratio)
            if is_identity:
                blur = clear.clone()

        return {
            "blur": blur.mul(2.0).sub(1.0),
            "clear": clear.mul(2.0).sub(1.0),
            "sample_id": record.sample_id,
            "is_identity": torch.tensor(is_identity, dtype=torch.bool),
        }


def _hwc_to_chw_tensor(array) -> torch.Tensor:
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).to(torch.float32)


def _paired_random_crop(
    blur: torch.Tensor,
    clear: torch.Tensor,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if blur.shape != clear.shape:
        raise ValueError(
            f"Paired tensors must have identical shapes, got "
            f"blur={tuple(blur.shape)} and clear={tuple(clear.shape)}"
        )
    height, width = blur.shape[-2:]
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Pair is smaller than requested crop {crop_size}: {(height, width)}"
        )
    if height == crop_size and width == crop_size:
        return blur, clear

    top = int(torch.randint(0, height - crop_size + 1, ()).item())
    left = int(torch.randint(0, width - crop_size + 1, ()).item())
    slices = (..., slice(top, top + crop_size), slice(left, left + crop_size))
    return blur[slices].contiguous(), clear[slices].contiguous()


__all__ = ["A0PairedDataset", "PROJECT_ROOT"]
