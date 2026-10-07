"""Create a group-held-out estimator split only from A1's fixed training pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--holdout-group", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.open("r", encoding="utf-8")]
    if any(record["split"] != "train" for record in records):
        raise ValueError("Internal split must come only from the fixed training manifest")
    with args.labels.open("r", newline="", encoding="utf-8") as handle:
        label_reader = csv.DictReader(handle)
        fieldnames = label_reader.fieldnames
        labels = {row["sample_id"]: row for row in label_reader}
    if not fieldnames or len(labels) != len(records):
        raise ValueError("Labels and manifest must contain the same fixed training samples")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        chosen = [record for record in records if (record["group_id"] == args.holdout_group) == (split == "val")]
        if not chosen:
            raise ValueError(f"Empty internal {split} split")
        with (output_dir / f"internal_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in chosen:
                copied = dict(record)
                copied["split"] = split
                handle.write(json.dumps(copied, ensure_ascii=False) + "\n")
        with (output_dir / f"internal_{split}_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in chosen:
                writer.writerow(labels[record["sample_id"]])
        print(f"internal {split}: {len(chosen)} records")


if __name__ == "__main__":
    main()
