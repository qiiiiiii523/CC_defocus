"""Check that every fixed A1 training/validation pair exists on this machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(root: Path, manifest: Path, expected_split: str) -> tuple[int, int, list[str]]:
    records = 0
    missing = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records += 1
            if record["split"] != expected_split:
                raise ValueError(f"{manifest}: unexpected split {record['split']}")
            for field in ("blur_path", "clear_path"):
                path = root / record[field]
                if not path.is_file():
                    missing.append(str(path))
    return records, len(missing), missing[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    args = parser.parse_args()
    for name, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        records, count, examples = inspect(args.project_root, path, name)
        print(f"{name}: {records} pairs, {count} missing files")
        for example in examples:
            print(f"  MISSING {example}")
        if count:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
