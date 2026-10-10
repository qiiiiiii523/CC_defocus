"""Evaluate one model on the fixed 3DHistech debug validation manifest.

All models are evaluated from their saved RGB PNG outputs. PSNR and SSIM use
float32 images in [0, 1]. LPIPS uses the same images mapped once to [-1, 1]
and calls the fixed AlexNet metric with ``normalize=False``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common_io import ImageProtocolError, PROJECT_ROOT, load_manifest, read_rgb_image


DEFAULT_MANIFEST = PROJECT_ROOT / "prepared" / "manifests" / "debug_val.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_EXPECTED_COUNT = 300
METRIC_NAMES = ("psnr", "ssim", "lpips")


class EvaluationError(RuntimeError):
    """Raised when a formal evaluation cannot be completed."""


def _portable_path(path: Path) -> str:
    """Return a repository-relative path without leaking a local absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model directory name under outputs/ and results/.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Directory containing <sample_id>.png (default: outputs/<model-name>).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Result directory (default: results/<model-name>).",
    )
    parser.add_argument(
        "--use-blur-as-prediction",
        action="store_true",
        help="Use manifest blur_path for the no-restoration baseline.",
    )
    return parser.parse_args()


def _validate_model_name(model_name: str) -> None:
    if not model_name or Path(model_name).name != model_name or model_name in {".", ".."}:
        raise EvaluationError(f"Invalid model name: {model_name!r}")


def _require_valid_range(image: np.ndarray, *, sample_id: str, role: str) -> None:
    if image.dtype != np.float32:
        raise EvaluationError(
            f"Expected float32 {role} image for sample_id={sample_id}, got {image.dtype}"
        )
    if not np.isfinite(image).all():
        raise EvaluationError(
            f"Non-finite pixels in {role} image for sample_id={sample_id}"
        )
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 1.0:
        raise EvaluationError(
            f"Out-of-range {role} image for sample_id={sample_id}: "
            f"min={minimum}, max={maximum}, expected [0, 1]"
        )


def _require_finite_metric(sample_id: str, metric_name: str, value: float) -> None:
    if not math.isfinite(value):
        detail = " (the two images may be identical)" if metric_name == "psnr" else ""
        raise EvaluationError(
            f"Non-finite {metric_name.upper()} for sample_id={sample_id}: "
            f"value={value!r}{detail}. Formal summary was not written."
        )


def _compute_lpips(
    lpips_model: torch.nn.Module,
    device: torch.device,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    def to_lpips_tensor(image: np.ndarray):
        chw = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor_01 = torch.from_numpy(chw).unsqueeze(0).to(device=device)
        return tensor_01.mul(2.0).sub(1.0)

    prediction_m11 = to_lpips_tensor(prediction)
    reference_m11 = to_lpips_tensor(reference)
    with torch.no_grad():
        value = lpips_model(
            prediction_m11,
            reference_m11,
            normalize=False,
        )
    return float(value.squeeze().item())


def evaluate(
    *,
    model_name: str,
    manifest_path: Path,
    prediction_dir: Path,
    results_dir: Path,
    use_blur_as_prediction: bool,
) -> tuple[Path, Path]:
    """Run a strict all-or-nothing evaluation and write CSV/JSON results."""
    _validate_model_name(model_name)
    records = load_manifest(
        manifest_path,
        root=PROJECT_ROOT,
        expected_split="val",
        required_quality_status="pass",
    )
    if len(records) != DEFAULT_EXPECTED_COUNT:
        raise EvaluationError(
            f"Manifest count mismatch: expected {DEFAULT_EXPECTED_COUNT}, "
            f"found {len(records)} "
            f"in {manifest_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows: list[dict[str, str | float]] = []
    for record in records:
        sample_id = record.sample_id
        prediction_path = (
            record.blur_path
            if use_blur_as_prediction
            else prediction_dir / f"{sample_id}.png"
        )
        if not prediction_path.is_file():
            raise EvaluationError(
                f"Missing prediction for sample_id={sample_id}: {prediction_path}. "
                "Formal summary was not written."
            )

        prediction = read_rgb_image(
            prediction_path,
            sample_id=sample_id,
            role="prediction",
        )
        reference = read_rgb_image(
            record.clear_path,
            sample_id=sample_id,
            role="clear reference",
        )
        if prediction.shape != reference.shape:
            raise EvaluationError(
                f"Image size mismatch for sample_id={sample_id}: "
                f"prediction_shape={prediction.shape}, "
                f"reference_shape={reference.shape}. Formal summary was not written."
            )
        _require_valid_range(prediction, sample_id=sample_id, role="prediction")
        _require_valid_range(reference, sample_id=sample_id, role="clear reference")

        try:
            psnr_value = float(
                peak_signal_noise_ratio(reference, prediction, data_range=1.0)
            )
            ssim_value = float(
                structural_similarity(
                    reference,
                    prediction,
                    data_range=1.0,
                    channel_axis=-1,
                )
            )
            lpips_value = _compute_lpips(
                lpips_model,
                device,
                prediction,
                reference,
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(
                f"Metric computation failed for sample_id={sample_id}: {exc}. "
                "Formal summary was not written."
            ) from exc

        values = {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
        }
        for metric_name, value in values.items():
            _require_finite_metric(sample_id, metric_name, value)
        rows.append({"sample_id": sample_id, **values})

    if len(rows) != len(records) or len(rows) != DEFAULT_EXPECTED_COUNT:
        raise EvaluationError(
            f"Successful evaluation count mismatch: expected {DEFAULT_EXPECTED_COUNT}, "
            f"evaluated {len(rows)}. Formal summary was not written."
        )

    summary = {
        "model_name": model_name,
        "prediction_source": (
            "manifest_blur_path" if use_blur_as_prediction else "model_output"
        ),
        "manifest": _portable_path(manifest_path),
        "prediction_dir": (
            None if use_blur_as_prediction else _portable_path(prediction_dir)
        ),
        "evaluated_count": len(rows),
        "metrics": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "std": float(np.std([row[name] for row in rows], ddof=0)),
            }
            for name in METRIC_NAMES
        },
        "settings": {
            "rgb": True,
            "data_range": 1.0,
            "crop_border": 0,
            "resize": False,
            "ssim_channel_axis": -1,
            "lpips_backbone": "alex",
            "lpips_input_range": [-1.0, 1.0],
            "lpips_normalize": False,
            "std_ddof": 0,
            "lpips_device": str(device),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_image_metrics.csv"
    csv_pending = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)
    csv_pending.replace(csv_path)

    summary_path = results_dir / "summary.json"
    summary_pending = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with summary_pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    summary_pending.replace(summary_path)
    return csv_path, summary_path


def main() -> None:
    args = parse_args()
    prediction_dir = (
        args.prediction_dir
        if args.prediction_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.model_name
    )
    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else DEFAULT_RESULTS_ROOT / args.model_name
    )
    try:
        csv_path, summary_path = evaluate(
            model_name=args.model_name,
            manifest_path=args.manifest,
            prediction_dir=prediction_dir,
            results_dir=results_dir,
            use_blur_as_prediction=args.use_blur_as_prediction,
        )
    except (EvaluationError, ImageProtocolError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    print(f"Evaluated {DEFAULT_EXPECTED_COUNT} samples for model={args.model_name}")
    print(f"Per-image metrics: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
