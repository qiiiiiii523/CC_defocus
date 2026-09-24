"""Paired 3D/3D_label adapter for the author's three-input RFN generator.

All generator inputs are derived from the blurry 3D image. No external masks
or domain-normalized images are required.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common_io import SampleRecord, load_manifest, read_rgb_image


ROOT = Path(__file__).resolve().parent
SHAPE = (256, 256, 3)
SCALE_NUM = 3
PATCH_SIZE = 64
PATCH_STEP = 32


@dataclass(frozen=True)
class RFNConfig:
    img_shape: tuple[int, int, int] = SHAPE
    channels: int = 3
    gf: int = 32
    denseblocks: int = 2
    scale_num: int = SCALE_NUM


def records(path: str, split: str) -> list[SampleRecord]:
    if Path(path).name == "rfn.jsonl":
        # The canonical manifest contains all three official splits.
        manifest = (ROOT / path).resolve()
        result = []
        seen = set()
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != split or row["quality_status"] != "pass":
                    continue
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate {split} sample ID: {sample_id}")
                seen.add(sample_id)
                blur = (ROOT / row["blur_path"]).resolve()
                clear = (ROOT / row["clear_path"]).resolve()
                if not blur.is_relative_to(ROOT) or not clear.is_relative_to(ROOT):
                    raise ValueError(f"Path outside project root: {sample_id}")
                if blur.stem != sample_id or clear.stem != sample_id or not blur.is_file() or not clear.is_file():
                    raise FileNotFoundError(f"Invalid 3D pair for {sample_id}: {blur}, {clear}")
                result.append(SampleRecord(sample_id, split, blur, clear))
    else:
        result = load_manifest(path, root=ROOT, expected_split=split)
    if not result:
        raise ValueError(f"Empty {split} manifest: {path}")
    return result


def _local_patches(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select three distinct high-contrast 64 px crops using only the blur.

    Local grayscale variation favors cells and nuclei over blank slide
    background. This deterministic selection replaces the paper's separately
    generated kernel masks; it is part of this paired-data adaptation.
    """
    gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    candidates = []
    for y in range(0, SHAPE[0] - PATCH_SIZE + 1, PATCH_STEP):
        for x in range(0, SHAPE[1] - PATCH_SIZE + 1, PATCH_STEP):
            tile = gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            score = float(tile.std())
            candidates.append((score, y, x))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = []
    for _, y, x in candidates:
        if all(abs(x - old_x) >= PATCH_SIZE or abs(y - old_y) >= PATCH_SIZE
               for old_y, old_x in chosen):
            chosen.append((y, x))
            if len(chosen) == SCALE_NUM:
                break
    if len(chosen) != SCALE_NUM:
        raise RuntimeError("Could not select three local patches")

    image_u8 = np.rint(image * 255.0).astype(np.uint8)
    patches = np.zeros((SCALE_NUM,) + SHAPE, dtype=np.float32)
    positions = np.zeros((SCALE_NUM, 5), dtype=np.float32)
    for i, (y, x) in enumerate(chosen):
        crop = Image.fromarray(image_u8[y:y + PATCH_SIZE, x:x + PATCH_SIZE], mode="RGB")
        patch = crop.resize((SHAPE[1], SHAPE[0]), resample=Image.Resampling.BILINEAR)
        patches[i] = np.asarray(patch, dtype=np.float32) / 127.5 - 1.0
        positions[i] = (1, x, y, PATCH_SIZE, PATCH_SIZE)
    return patches, positions


def make_inputs(record: SampleRecord) -> list[np.ndarray]:
    """Return all three RFN inputs from one blurry RGB image."""
    image = read_rgb_image(record.blur_path, sample_id=record.sample_id, role="blur")
    if tuple(image.shape) != SHAPE:
        raise ValueError(f"{record.sample_id}: expected {SHAPE}, got {image.shape}")
    patches, positions = _local_patches(image)
    return [image.astype(np.float32) * 2.0 - 1.0, patches, positions]


def load_target(record: SampleRecord) -> np.ndarray:
    target = read_rgb_image(record.clear_path, sample_id=record.sample_id, role="clear")
    if tuple(target.shape) != SHAPE:
        raise ValueError(f"{record.sample_id}: expected target {SHAPE}, got {target.shape}")
    return target.astype(np.float32) * 2.0 - 1.0
