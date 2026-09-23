"""PyTorch Tensor reader for CCCID V2 AI multi-focus sequences.

This dataset is intended for external inference and multi-focus experiments. It
does not create a clear target, invent train/val/test splits, or modify source
images and manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_FOCUS_LEVELS = 11


class CCCIDReadError(RuntimeError):
    """Raised when one image in a CCCID focus sequence cannot be loaded."""


class CCCIDDataset(Dataset):
    """Read CCCID V2 AI sequences as ``[11, 3, H, W]`` float tensors.

    Each item is ``(focus_stack, sample_id, source_category)``. The eleven RGB
    images follow the manifest's ``focus_paths`` order (focus levels 0 through
    10). Pixel values are converted to ``torch.float32`` in ``[0, 1]`` without
    resizing, cropping, mean/std normalization, or augmentation.

    Parameters
    ----------
    root:
        Project root used to resolve relative paths in the manifest.
    manifest_path:
        Optional manifest override. Relative values are resolved under ``root``.
    quality_status:
        Quality state to expose. The default is only ``"pass"``.
    split:
        Manifest pool to expose. The default is only ``"external_pool"``.
    """

    def __init__(
        self,
        root: str | Path = PROJECT_ROOT,
        manifest_path: str | Path | None = None,
        quality_status: str = "pass",
        split: str = "external_pool",
    ) -> None:
        self.root = Path(root).resolve()
        if manifest_path is None:
            self.manifest_path = self.root / "prepared" / "manifests" / "cccid.jsonl"
        else:
            candidate = Path(manifest_path)
            self.manifest_path = candidate if candidate.is_absolute() else self.root / candidate
        self.quality_status = quality_status
        self.split = split
        self._records: list[dict[str, Any]] = []
        self._records_by_id: dict[str, dict[str, Any]] = {}
        self._filtered_out_by_id: dict[str, tuple[str, str]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"CCCID manifest does not exist: {self.manifest_path}")

        seen: set[str] = set()
        required = {
            "sample_id",
            "source_dataset",
            "split",
            "quality_status",
            "focus_paths",
            "source_category",
        }
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in CCCID manifest at {self.manifest_path}:{line_number}: {exc}"
                    ) from exc

                missing = required - record.keys()
                if missing:
                    raise ValueError(
                        f"CCCID manifest line {line_number} is missing fields: {sorted(missing)}"
                    )
                if record["source_dataset"] != "CCCID-V2-AI":
                    raise ValueError(
                        f"Unexpected source_dataset at line {line_number}: "
                        f"{record['source_dataset']!r}"
                    )
                sample_id = record["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate CCCID sample_id at line {line_number}: {sample_id}")
                seen.add(sample_id)
                if len(record["focus_paths"]) != EXPECTED_FOCUS_LEVELS:
                    raise ValueError(
                        f"CCCID sample {sample_id!r} has {len(record['focus_paths'])} focus paths; "
                        f"expected {EXPECTED_FOCUS_LEVELS}"
                    )

                status = record["quality_status"]
                record_split = record["split"]
                if status != self.quality_status or record_split != self.split:
                    self._filtered_out_by_id[sample_id] = (status, record_split)
                    continue
                self._records.append(record)
                self._records_by_id[sample_id] = record

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        """Load one sequence by its reproducible manifest index."""
        return self._load_record(self._records[index])

    def get(self, sample_id: str) -> tuple[torch.Tensor, str, str]:
        """Load one sequence by its exact manifest ``sample_id``."""
        record = self._records_by_id.get(sample_id)
        if record is None:
            if sample_id in self._filtered_out_by_id:
                status, split = self._filtered_out_by_id[sample_id]
                raise KeyError(
                    f"CCCID sample_id {sample_id!r} is filtered out: "
                    f"quality_status={status!r}, split={split!r}"
                )
            raise KeyError(f"CCCID sample_id not found in {self.manifest_path}: {sample_id!r}")
        return self._load_record(record)

    def sample_ids(self) -> tuple[str, ...]:
        """Return exposed sequence IDs in reproducible manifest order."""
        return tuple(record["sample_id"] for record in self._records)

    def _load_record(self, record: dict[str, Any]) -> tuple[torch.Tensor, str, str]:
        sample_id = record["sample_id"]
        tensors = []
        expected_shape = None
        for focus_level, relative_path in enumerate(record["focus_paths"]):
            path = self._resolve_data_path(relative_path, sample_id, focus_level)
            tensor = self._load_rgb_tensor(path, sample_id, focus_level)
            if expected_shape is None:
                expected_shape = tensor.shape
            elif tensor.shape != expected_shape:
                raise CCCIDReadError(
                    f"CCCID sequence {sample_id!r} focus {focus_level} at {path} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)}"
                )
            tensors.append(tensor)
        focus_stack = torch.stack(tensors, dim=0)
        return focus_stack, sample_id, record["source_category"]

    def _resolve_data_path(self, value: str, sample_id: str, focus_level: int) -> Path:
        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise CCCIDReadError(
                f"CCCID sequence {sample_id!r} focus {focus_level} path is outside "
                f"project root {self.root}: {path}"
            ) from exc
        return path

    @staticmethod
    def _load_rgb_tensor(path: Path, sample_id: str, focus_level: int) -> torch.Tensor:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                width, height = rgb.size
                buffer = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return (
                buffer.reshape(height, width, 3)
                .permute(2, 0, 1)
                .contiguous()
                .to(dtype=torch.float32)
                .div_(255.0)
            )
        except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
            raise CCCIDReadError(
                f"Failed to read CCCID sequence {sample_id!r} focus {focus_level} at {path}: {exc}"
            ) from exc


__all__ = ["CCCIDDataset", "CCCIDReadError"]
