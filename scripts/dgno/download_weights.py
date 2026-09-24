"""Download an official DGNO 3DHistech checkpoint and print its SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILENAMES = {
    "face": "DGNO_3DHistech_Face.pth",
    "cell": "DGNO_3DHistech_Cell.pth",
}
BASE_URL = "https://huggingface.co/Duane245/DGNO/resolve/main/pretrained"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(FILENAMES), default="face")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "weights" / "dgno"
    )
    args = parser.parse_args()

    filename = FILENAMES[args.variant]
    output_path = args.output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_suffix(output_path.suffix + ".tmp")
    url = f"{BASE_URL}/{filename}?download=true"

    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "CC_defocus/1.0"})
    try:
        with urllib.request.urlopen(request) as response, pending.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        pending.replace(output_path)
    finally:
        pending.unlink(missing_ok=True)

    print(f"Saved: {output_path}")
    print(f"SHA-256: {sha256(output_path)}")


if __name__ == "__main__":
    main()
