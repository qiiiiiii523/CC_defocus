"""Framework-independent input/output protocol for 3DHistech restoration.

The public boundary uses RGB NumPy arrays in HWC layout with float32 values
in [0, 1]. Model-specific branches are responsible for converting between
this representation and their own PyTorch or TensorFlow tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VALID_SPLITS = {"train", "val", "test"}
REQUIRED_FIELDS = {
    "sample_id",
    "split",
    "quality_status",
    "blur_path",
    "clear_path",
}


class ImageProtocolError(RuntimeError):
    """Raised when a sample violates the public image protocol."""


@dataclass(frozen=True)
class SampleRecord:
    """One paired sample resolved from a JSONL manifest."""

    sample_id: str
    split: str
    blur_path: Path
    clear_path: Path


def load_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
    expected_split: str | None = None,
    required_quality_status: str = "pass",
) -> list[SampleRecord]:
    """Load and validate paired records from a JSONL manifest."""
    project_root = Path(root).resolve()
    path = _resolve_under_root(project_root, manifest_path, "manifest")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    if expected_split is not None and expected_split not in VALID_SPLITS:
        raise ValueError(
            f"expected_split must be one of {sorted(VALID_SPLITS)}, "
            f"got {expected_split!r}"
        )

    records: list[SampleRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImageProtocolError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            missing = REQUIRED_FIELDS - raw.keys()
            if missing:
                raise ImageProtocolError(
                    f"Manifest record at {path}:{line_number} is missing fields: "
                    f"{sorted(missing)}"
                )

            sample_id = raw["sample_id"]
            split = raw["split"]
            quality_status = raw["quality_status"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ImageProtocolError(
                    f"Invalid sample_id at {path}:{line_number}: {sample_id!r}"
                )
            if sample_id in seen_ids:
                raise ImageProtocolError(
                    f"Duplicate sample_id at {path}:{line_number}: {sample_id}"
                )
            if split not in VALID_SPLITS:
                raise ImageProtocolError(
                    f"Invalid split for sample_id={sample_id}: {split!r}"
                )
            if expected_split is not None and split != expected_split:
                raise ImageProtocolError(
                    f"Unexpected split for sample_id={sample_id}: "
                    f"expected {expected_split!r}, got {split!r}"
                )
            if quality_status != required_quality_status:
                raise ImageProtocolError(
                    f"Unexpected quality_status for sample_id={sample_id}: "
                    f"expected {required_quality_status!r}, got {quality_status!r}"
                )

            blur_path = _resolve_under_root(
                project_root, raw["blur_path"], f"blur_path for sample_id={sample_id}"
            )
            clear_path = _resolve_under_root(
                project_root, raw["clear_path"], f"clear_path for sample_id={sample_id}"
            )
            if blur_path.name != clear_path.name:
                raise ImageProtocolError(
                    f"Paired filenames differ for sample_id={sample_id}: "
                    f"blur={blur_path}, clear={clear_path}"
                )
            if blur_path.stem != sample_id or clear_path.stem != sample_id:
                raise ImageProtocolError(
                    f"Paired paths do not match sample_id={sample_id}: "
                    f"blur={blur_path}, clear={clear_path}"
                )
            if not blur_path.is_file():
                raise FileNotFoundError(
                    f"Missing blur image for sample_id={sample_id}: {blur_path}"
                )
            if not clear_path.is_file():
                raise FileNotFoundError(
                    f"Missing clear image for sample_id={sample_id}: {clear_path}"
                )

            seen_ids.add(sample_id)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    split=split,
                    blur_path=blur_path,
                    clear_path=clear_path,
                )
            )

    return records


def read_rgb_image(
    path: str | Path,
    *,
    sample_id: str,
    role: str,
) -> np.ndarray:
    """Read an image as an HWC RGB float32 array in [0, 1]."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
    except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
        raise ImageProtocolError(
            f"Failed to decode {role} image for sample_id={sample_id}: "
            f"path={image_path}, error={exc}"
        ) from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageProtocolError(
            f"Invalid {role} image shape for sample_id={sample_id}: "
            f"path={image_path}, shape={array.shape}"
        )
    if not np.isfinite(array).all():
        raise ImageProtocolError(
            f"Non-finite pixels in {role} image for sample_id={sample_id}: "
            f"path={image_path}"
        )
    return array


def load_pair(record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one blur/clear pair and require identical HWC shapes."""
    blur = read_rgb_image(
        record.blur_path, sample_id=record.sample_id, role="blur"
    )
    clear = read_rgb_image(
        record.clear_path, sample_id=record.sample_id, role="clear"
    )
    if blur.shape != clear.shape:
        raise ImageProtocolError(
            f"Blur/clear size mismatch for sample_id={record.sample_id}: "
            f"blur_path={record.blur_path}, blur_shape={blur.shape}, "
            f"clear_path={record.clear_path}, clear_shape={clear.shape}"
        )
    return blur, clear


def save_prediction(
    prediction: np.ndarray,
    *,
    model_name: str,
    sample_id: str,
    expected_hw: tuple[int, int],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Validate and save one prediction as an RGB uint8 lossless PNG."""
    _validate_path_component(model_name, "model_name")
    _validate_path_component(sample_id, "sample_id")

    array = np.asarray(prediction)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageProtocolError(
            f"Invalid prediction shape for sample_id={sample_id}: "
            f"expected HWC RGB, got {array.shape}"
        )
    if tuple(array.shape[:2]) != tuple(expected_hw):
        raise ImageProtocolError(
            f"Prediction size mismatch for sample_id={sample_id}: "
            f"expected_hw={tuple(expected_hw)}, actual_hw={tuple(array.shape[:2])}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ImageProtocolError(
            f"Invalid prediction dtype for sample_id={sample_id}: "
            f"expected floating-point [0, 1], got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ImageProtocolError(
            f"Prediction contains NaN or Inf for sample_id={sample_id}"
        )

    rgb_uint8 = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path = Path(output_root) / model_name / f"{sample_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgb_uint8, mode="RGB").save(output_path, format="PNG")
    except (OSError, ValueError) as exc:
        raise ImageProtocolError(
            f"Failed to save prediction for sample_id={sample_id}: "
            f"path={output_path}, error={exc}"
        ) from exc
    return output_path


def _resolve_under_root(root: Path, value: str | Path, field: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ImageProtocolError(
            f"{field} resolves outside project root {root}: {resolved}"
        ) from exc
    return resolved


def _validate_path_component(value: str, field: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Invalid {field}: {value!r}")


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "ImageProtocolError",
    "SampleRecord",
    "load_manifest",
    "load_pair",
    "read_rgb_image",
    "save_prediction",
]
