"""Extract only images referenced by a manifest from 3D/3D_label ZIP files."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "prepared" / "manifests" / "debug_val.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "prepared" / "images" / "3DHistech"


def read_sample_ids(manifest_path: Path) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid sample_id at line {line_number}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id at line {line_number}: {sample_id}")
            if record.get("quality_status") != "pass":
                raise ValueError(f"Non-pass record at line {line_number}: {sample_id}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return sample_ids


def extract_role(
    archive_path: Path,
    *,
    archive_dir: str,
    output_dir: Path,
    sample_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        available = set(archive.namelist())
        expected = {sample_id: f"{archive_dir}/{sample_id}.png" for sample_id in sample_ids}
        missing = [name for name in expected.values() if name not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} manifest image(s) missing from {archive_path}: {preview}"
            )

        for index, sample_id in enumerate(sample_ids, start=1):
            member = expected[sample_id]
            destination = output_dir / f"{sample_id}.png"
            pending = destination.with_suffix(".png.tmp")
            try:
                with archive.open(member) as source, pending.open("wb") as target:
                    shutil.copyfileobj(source, target)
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
            if index % 50 == 0 or index == len(sample_ids):
                print(f"{archive_dir}: {index}/{len(sample_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blur-zip", type=Path, required=True)
    parser.add_argument("--clear-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optionally create a tar containing only the extracted manifest images.",
    )
    args = parser.parse_args()

    sample_ids = read_sample_ids(args.manifest)
    extract_role(
        args.blur_zip,
        archive_dir="3D",
        output_dir=args.output_root / "3D",
        sample_ids=sample_ids,
    )
    extract_role(
        args.clear_zip,
        archive_dir="3D_label",
        output_dir=args.output_root / "3D_label",
        sample_ids=sample_ids,
    )
    print(f"Extracted {len(sample_ids)} paired samples into {args.output_root}")

    if args.archive is not None:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        pending = args.archive.with_suffix(args.archive.suffix + ".tmp")
        try:
            with tarfile.open(pending, mode="w") as archive:
                for role in ("3D", "3D_label"):
                    for sample_id in sample_ids:
                        source = args.output_root / role / f"{sample_id}.png"
                        archive.add(
                            source,
                            arcname=(
                                Path("prepared")
                                / "images"
                                / "3DHistech"
                                / role
                                / source.name
                            ).as_posix(),
                        )
            pending.replace(args.archive)
        finally:
            pending.unlink(missing_ok=True)
        print(f"Created manifest-only archive: {args.archive}")


if __name__ == "__main__":
    main()
