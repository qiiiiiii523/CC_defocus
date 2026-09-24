"""Manifest and image adapter for the author's three-input RFN generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from common_io import SampleRecord, load_manifest, read_rgb_image


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "prepared" / "images" / "3DHistech"
KERNEL_ROOT = IMAGE_ROOT / "3D_to_3D_kernel"
SHAPE = (256, 256, 3)
SCALE_NUM = 3


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


def check_kernel_masks(items: list[SampleRecord]) -> None:
    if not KERNEL_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing {KERNEL_ROOT}. Extract 3D_to_3D_kernel.zip into {IMAGE_ROOT} first."
        )
    missing = [r.sample_id for r in items if not (KERNEL_ROOT / f"{r.sample_id}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} kernel masks; example ID: {missing[0]}")


def make_inputs(record: SampleRecord) -> list[np.ndarray]:
    """Return the three inputs consumed by upstream build_multi_scale_v8.

    The original loader creates six patches then passes only the first three to
    the generator. Those three are the largest kernel-mask contours. The
    position tensor is retained although upstream v8 does not use its values.
    """
    import cv2

    image = read_rgb_image(record.blur_path, sample_id=record.sample_id, role="blur")
    if tuple(image.shape) != SHAPE:
        raise ValueError(f"{record.sample_id}: expected {SHAPE}, got {image.shape}")
    mask_path = KERNEL_ROOT / f"{record.sample_id}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != SHAPE[:2]:
        raise ValueError(f"{record.sample_id}: missing or invalid kernel mask: {mask_path}")
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = sorted((cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3], reverse=True)
    image_u8 = np.rint(image * 255).astype(np.uint8)
    padded = cv2.copyMakeBorder(image_u8, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    patches = np.zeros((SCALE_NUM,) + SHAPE, dtype=np.float32)
    positions = np.zeros((SCALE_NUM, 5), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(rects[:SCALE_NUM]):
        cx, cy = x + w // 2, y + h // 2
        left, top = cx - 32 + 64, cy - 32 + 64
        patch = padded[top:top + 64, left:left + 64]
        patches[i] = cv2.resize(patch, (256, 256)).astype(np.float32) / 127.5 - 1.0
        positions[i] = (1, left, top, 64, 64)
    return [image.astype(np.float32) * 2.0 - 1.0, patches, positions]


def load_target(record: SampleRecord) -> np.ndarray:
    target = read_rgb_image(record.clear_path, sample_id=record.sample_id, role="clear")
    if tuple(target.shape) != SHAPE:
        raise ValueError(f"{record.sample_id}: expected target {SHAPE}, got {target.shape}")
    return target.astype(np.float32) * 2.0 - 1.0
