"""Run a trained RFN generator on fixed validation IDs and save common PNGs."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from common_io import SampleRecord, save_prediction
from rfn_adapter import RFNConfig, ROOT, make_inputs, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--manifest", default="prepared/manifests/debug_val.jsonl")
    p.add_argument("--input", help="Single blurry PNG or directory of blurry PNGs; no clear images needed")
    p.add_argument("--limit", type=int, default=0, help="Use 5-10 only for smoke checks; omit for all IDs")
    args = p.parse_args()
    if args.input:
        path = (ROOT / args.input).resolve()
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(path.glob("*.png"))
        else:
            p.error(f"No such input image or directory: {path}")
        if not paths:
            p.error(f"No PNG images in {path}")
        items = [SampleRecord(file.stem, "val", file, file) for file in paths]
    else:
        items = records(args.manifest, "val")
    if args.limit < 0:
        p.error("limit must be nonnegative")
    if args.limit:
        items = items[:args.limit]
    weights = (ROOT / args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    import tensorflow as tf

    sys.path.insert(0, str(ROOT / "rfn_author"))
    from mmodels.multi_scale import build_multi_scale_v8

    model = build_multi_scale_v8(RFNConfig())
    model.load_weights(str(weights))
    started = time.perf_counter()
    for i, record in enumerate(items, 1):
        x = [np.expand_dims(t, axis=0) for t in make_inputs(record)]
        result = model.predict_on_batch(x)[0]
        prediction = np.clip((result + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        save_prediction(prediction, model_name="rfn", sample_id=record.sample_id,
                        expected_hw=(256, 256))
        if i % 25 == 0 or i == len(items):
            print(f"{i}/{len(items)} saved; elapsed {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
