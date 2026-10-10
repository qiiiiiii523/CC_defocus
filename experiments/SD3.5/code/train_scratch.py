"""Training entry-point scaffold for the SD3.5 scratch experiment.

This deliberately stops before data loading until the captain wires the
repository's existing paired loader. It prevents accidental use of a new split
or a clear reference during inference. Run smoke_test.py first.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'resolved_config.json').write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(
        'Training scaffold is wired for the SD3.5 scratch design but is not GPU-verified. '
        'Run smoke_test.py, then connect the repository paired loader before formal training.')


if __name__ == '__main__':
    main()
