#!/usr/bin/env bash
set -euo pipefail
tag="${1:?Usage: bash JiT/run_a1_exp_checks.sh ORIGINAL_RUN_TAG}"
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid tag' >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
out="$root/reports/a1_exp_checks_${tag}_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$out" ]] || { echo "Output exists: $out" >&2; exit 2; }
mkdir -p "$out"
printf '%s\n' "$out" > "$root/reports/a1_exp_checks_latest.txt"
cd "$root"
nohup python -u JiT/run_a1_exp_checks.py "$tag" --project-root "$root" \
  --output-dir "$out" --gpus 0 1 2 \
  > "$out/launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$out/launcher.pid"
printf 'Checks: %s\nLog: %s\n' "$out" "$out/launcher.log"
