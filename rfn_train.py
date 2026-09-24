"""Train the unchanged upstream RFN v8 generator on paired 3D images.

This is supervised-only training, not the paper's full hybrid DNN/RFN scheme.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from rfn_adapter import RFNConfig, ROOT, check_kernel_masks, load_target, make_inputs, records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", default="prepared/manifests/debug_train.jsonl")
    p.add_argument("--val-manifest", default="prepared/manifests/debug_val.jsonl")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--run-name", default="rfn_debug")
    p.add_argument("--dry-run", action="store_true", help="Check manifests and masks without importing TensorFlow")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError("run-name must be a single path component")
    train = records(args.train_manifest, "train")
    val = records(args.val_manifest, "val")
    if set(r.sample_id for r in train) & set(r.sample_id for r in val):
        raise ValueError("Training and validation manifests overlap")
    check_kernel_masks(train + val)
    print(f"Validated {len(train)} train and {len(val)} validation samples with kernel masks")
    if args.dry_run:
        return

    import tensorflow as tf

    sys.path.insert(0, str(ROOT / "rfn_author"))
    from mmodels.multi_scale import build_multi_scale_v8

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    class Batches(tf.keras.utils.Sequence):
        def __init__(self, items, shuffle):
            self.items = list(items)
            self.shuffle = shuffle
            self.order = np.arange(len(items))
            self.rng = np.random.default_rng(args.seed)
            self.on_epoch_end()

        def __len__(self):
            return (len(self.items) + args.batch_size - 1) // args.batch_size

        def __getitem__(self, index):
            chosen = [self.items[i] for i in self.order[index * args.batch_size:(index + 1) * args.batch_size]]
            data = [make_inputs(r) for r in chosen]
            x = [np.stack([row[j] for row in data]) for j in range(3)]
            y = np.stack([load_target(r) for r in chosen])
            return x, y

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

    model = build_multi_scale_v8(RFNConfig())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="mae")
    output = ROOT / "checkpoints" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": "RFN v8 generator; paired supervised MAE only",
        "upstream": "ShenghuaCheng/Cytopathology-image-refocusing",
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_count": len(train),
        "val_count": len(val),
        "input": "raw 3D RGB with author-provided 3D_to_3D kernel masks",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "tensorflow": tf.__version__,
    }
    (output / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output / "best.weights.h5"), save_weights_only=True,
        monitor="val_loss", mode="min", save_best_only=True,
    )
    history = model.fit(Batches(train, True), validation_data=Batches(val, False),
                        epochs=args.epochs, callbacks=[checkpoint], workers=1, use_multiprocessing=False)
    model.save_weights(str(output / "last.weights.h5"))
    (output / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    print(f"Weights: {output / 'best.weights.h5'}")


if __name__ == "__main__":
    main()

