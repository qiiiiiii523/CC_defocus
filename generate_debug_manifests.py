"""Generate fixed RFN debug manifests from the audited full manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "prepared" / "manifests" / "rfn.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "prepared" / "manifests"
DEFAULT_SEED = 20260923
DEBUG_SIZES = {"train": 2000, "val": 300}
REQUIRED_FIELDS = {
    "sample_id",
    "source_dataset",
    "split",
    "quality_status",
    "blur_path",
    "clear_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Source record at {path}:{line_number} is missing fields: "
                    f"{sorted(missing)}"
                )
            sample_id = record["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id at {path}:{line_number}: {sample_id}")
            seen_ids.add(sample_id)
            records.append(record)
    return records


def select_records(
    records: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}

    for split, count in DEBUG_SIZES.items():
        candidates = sorted(
            (
                record
                for record in records
                if record["source_dataset"] == "RFN"
                and record["split"] == split
                and record["quality_status"] == "pass"
            ),
            key=lambda record: record["sample_id"],
        )
        if len(candidates) < count:
            raise ValueError(
                f"Not enough pass records in official {split}: "
                f"need {count}, found {len(candidates)}"
            )
        selected[split] = sorted(
            rng.sample(candidates, count), key=lambda record: record["sample_id"]
        )

    return selected


def validate_selected(
    selected: dict[str, list[dict[str, Any]]], project_root: Path
) -> None:
    all_ids: set[str] = set()
    for split, expected_count in DEBUG_SIZES.items():
        records = selected[split]
        if len(records) != expected_count:
            raise ValueError(
                f"Unexpected {split} selection size: {len(records)} != {expected_count}"
            )

        for record in records:
            sample_id = record["sample_id"]
            if sample_id in all_ids:
                raise ValueError(f"Duplicate selected sample_id: {sample_id}")
            all_ids.add(sample_id)

            if record["split"] != split:
                raise ValueError(
                    f"Selected {sample_id} has split={record['split']!r}, expected {split!r}"
                )
            if record["quality_status"] != "pass":
                raise ValueError(
                    f"Selected {sample_id} has quality_status="
                    f"{record['quality_status']!r}"
                )

            blur_path = project_root / record["blur_path"]
            clear_path = project_root / record["clear_path"]
            if blur_path.stem != sample_id or clear_path.stem != sample_id:
                raise ValueError(f"Path pairing does not match sample_id: {sample_id}")
            if blur_path.name != clear_path.name:
                raise ValueError(f"Blur/clear filenames do not match: {sample_id}")
            if not blur_path.is_file():
                raise FileNotFoundError(f"Missing blur image for {sample_id}: {blur_path}")
            if not clear_path.is_file():
                raise FileNotFoundError(f"Missing clear image for {sample_id}: {clear_path}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending.replace(path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    records = load_manifest(source)
    selected = select_records(records, args.seed)
    validate_selected(selected, PROJECT_ROOT)

    outputs = {
        "train": output_dir / "debug_train.jsonl",
        "val": output_dir / "debug_val.jsonl",
    }
    for split, path in outputs.items():
        write_jsonl(path, selected[split])
        print(f"Wrote {len(selected[split]):,} official {split} pass records: {path}")
    print(f"Seed: {args.seed}")
    print("Test records were not selected or modified.")


if __name__ == "__main__":
    main()
