"""Extract official RFN train/pass and val/pass without changing any split."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from common_io import load_manifest, load_pair


def prepare(root, source=None, check_images=False):
    root = Path(root).resolve()
    source = Path(source) if source else root / "prepared/manifests/rfn.jsonl"
    if not source.is_absolute():
        source = root / source
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = {split: [] for split in ("train", "val", "test")}
    seen = set()
    groups = {split: set() for split in selected}
    for row in rows:
        if row["source_dataset"] != "RFN" or row["quality_status"] != "pass":
            continue
        split = row["split"]
        if split not in selected:
            raise ValueError("Invalid split: " + str(split))
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id: " + row["sample_id"])
        seen.add(row["sample_id"])
        selected[split].append(row)
        if row.get("group_id"):
            groups[split].add(row["group_id"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"group_id overlap between {a} and {b}")
    outputs = {}
    pending = []
    try:
        for split in ("train", "val"):
            if not selected[split]:
                raise ValueError("Empty RFN/pass split: " + split)
            destination = root / f"prepared/manifests/full_{split}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(".jsonl.tmp")
            pending.append(temp)
            temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in selected[split]), encoding="utf-8")
            records = load_manifest(temp, root=root, expected_split=split)
            if check_images:
                dimensions = Counter()
                for index, record in enumerate(records, 1):
                    blur, _ = load_pair(record)
                    if blur.shape[:2] != (256, 256):
                        raise ValueError("Expected 256x256: " + record.sample_id)
                    dimensions[blur.shape] += 1
                    if index % 1000 == 0:
                        print(f"decoded {split} {index}/{len(records)}", flush=True)
                print(f"decoded {split} dimensions={dict(dimensions)}", flush=True)
            outputs[split] = destination
        for split, destination in outputs.items():
            destination.with_suffix(".jsonl.tmp").replace(destination)
    finally:
        for temp in pending:
            if temp.exists():
                temp.unlink()
    counts = {s: len(r) for s, r in selected.items()}
    print(json.dumps({"counts": counts, "outputs": {s: str(p) for s, p in outputs.items()}},
                     ensure_ascii=False), flush=True)
    return counts, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check-images", action="store_true", help="Decode every selected pair before training")
    args = parser.parse_args()
    prepare(args.repo_root, args.source, args.check_images)


if __name__ == "__main__":
    main()
