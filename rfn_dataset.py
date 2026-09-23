"""RFN paired-image reader backed by prepared/manifests/rfn.jsonl.

The official train/val/test split is read directly from the manifest. Samples
are returned as PyTorch tensors without resizing, cropping, or augmentation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


VALID_SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parent


class RFNReadError(RuntimeError):
    """Raised when an RFN sample image cannot be loaded."""


class RFNDataset(Dataset):
    """DataLoader-ready RFN dataset for one official split.

    Each item is ``(y, x, sample_id, split)``. ``y`` is the 3D defocused input
    and ``x`` is its same-name 3D_label clear reference. Both are contiguous
    ``torch.float32`` tensors shaped ``[3, H, W]`` with values in ``[0, 1]``.
    No resize, crop, mean/std normalization, or augmentation is performed.

    Parameters
    ----------
    root:
        Path used to resolve the manifest's relative image paths.
    split:
        Official split exposed through ``__len__`` and ``__getitem__``.
    manifest_path:
        Optional manifest override. Relative values are resolved under ``root``.
    quality_status:
        Records to expose. The default is only ``"pass"``. Pass ``None`` to
        expose all quality states.
    """

    def __init__(
        self,
        root: str | Path = PROJECT_ROOT,
        split: str = "train",
        manifest_path: str | Path | None = None,
        quality_status: str | set[str] | tuple[str, ...] | None = "pass",
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.root = Path(root).resolve()
        self.split = split
        if manifest_path is None:
            self.manifest_path = self.root / "prepared" / "manifests" / "rfn.jsonl"
        else:
            candidate = Path(manifest_path)
            self.manifest_path = candidate if candidate.is_absolute() else self.root / candidate

        if isinstance(quality_status, str):
            self.quality_statuses: set[str] | None = {quality_status}
        elif quality_status is None:
            self.quality_statuses = None
        else:
            self.quality_statuses = set(quality_status)

        self._records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._records_by_id: dict[str, dict[str, Any]] = {}
        self._filtered_out_by_id: dict[str, str] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"RFN manifest does not exist: {self.manifest_path}")

        seen_all: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in RFN manifest at {self.manifest_path}:{line_number}: {exc}"
                    ) from exc

                required = {
                    "sample_id",
                    "source_dataset",
                    "split",
                    "quality_status",
                    "blur_path",
                    "clear_path",
                }
                missing = required - record.keys()
                if missing:
                    raise ValueError(
                        f"RFN manifest line {line_number} is missing fields: {sorted(missing)}"
                    )
                if record["source_dataset"] != "RFN":
                    raise ValueError(
                        f"Unexpected source_dataset at line {line_number}: {record['source_dataset']!r}"
                    )
                if record["split"] not in VALID_SPLITS:
                    raise ValueError(
                        f"Unexpected RFN split at line {line_number}: {record['split']!r}"
                    )

                sample_id = record["sample_id"]
                if sample_id in seen_all:
                    raise ValueError(f"Duplicate RFN sample_id at line {line_number}: {sample_id}")
                seen_all.add(sample_id)

                status = record["quality_status"]
                if self.quality_statuses is not None and status not in self.quality_statuses:
                    self._filtered_out_by_id[sample_id] = status
                    continue
                self._records_by_id[sample_id] = record
                self._records_by_split[record["split"]].append(record)

    def __len__(self) -> int:
        return len(self._records_by_split[self.split])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Load one item from the configured official split."""
        return self._load_record(self._records_by_split[self.split][index])

    def split_size(self, split: str) -> int:
        """Return the number of exposed records in one official split."""
        self._validate_split(split)
        return len(self._records_by_split[split])

    def sample_ids(self, split: str) -> tuple[str, ...]:
        """Return sample IDs in their reproducible manifest order."""
        self._validate_split(split)
        return tuple(record["sample_id"] for record in self._records_by_split[split])

    def iter_split(self, split: str) -> Iterator[tuple[torch.Tensor, torch.Tensor, str, str]]:
        """Yield ``(y, x, sample_id, split)`` in manifest order."""
        self._validate_split(split)
        for record in self._records_by_split[split]:
            yield self._load_record(record)

    def get(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Load a named sample that belongs to the configured official split."""
        record = self._records_by_id.get(sample_id)
        if record is None:
            if sample_id in self._filtered_out_by_id:
                status = self._filtered_out_by_id[sample_id]
                raise KeyError(
                    f"RFN sample_id {sample_id!r} has quality_status={status!r} and is filtered out"
                )
            raise KeyError(f"RFN sample_id not found in {self.manifest_path}: {sample_id!r}")
        if record["split"] != self.split:
            raise KeyError(
                f"RFN sample_id {sample_id!r} belongs to split={record['split']!r}, "
                f"not this dataset split={self.split!r}"
            )
        return self._load_record(record)

    def _validate_split(self, split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    def _load_record(self, record: dict[str, Any]):
        sample_id = record["sample_id"]
        blur_path = self._resolve_data_path(record["blur_path"], sample_id, "blur_path")
        clear_path = self._resolve_data_path(record["clear_path"], sample_id, "clear_path")
        y = _rgb_image_to_tensor(self._load_rgb(blur_path, sample_id, "blur_path"))
        x = _rgb_image_to_tensor(self._load_rgb(clear_path, sample_id, "clear_path"))
        return y, x, sample_id, record["split"]

    def _resolve_data_path(self, value: str, sample_id: str, field: str) -> Path:
        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RFNReadError(
                f"RFN sample {sample_id!r} has {field} outside root {self.root}: {path}"
            ) from exc
        return path

    @staticmethod
    def _load_rgb(path: Path, sample_id: str, field: str) -> Image.Image:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                return rgb
        except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
            raise RFNReadError(
                f"Failed to read RFN sample {sample_id!r} {field} at {path}: {exc}"
            ) from exc


def _rgb_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB Pillow image to CHW float32 in [0, 1]."""
    width, height = image.size
    buffer = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return (
        buffer.reshape(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


__all__ = [
    "RFNDataset",
    "RFNReadError",
    "VALID_SPLITS",
]
