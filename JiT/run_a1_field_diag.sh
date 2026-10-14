#!/usr/bin/env bash
set -euo pipefail
tag="${1:-20261014_114501}"
gpu="${2:-0}"
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ && "$gpu" =~ ^[0-9]+$ ]] || { echo 'Invalid tag or GPU' >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
out="$root/reports/a1_field_diag_${tag}_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$out" ]] || { echo "Output exists: $out" >&2; exit 2; }
mkdir -p "$out"
cd "$root"
nohup env CUDA_VISIBLE_DEVICES="$gpu" python -u JiT/diagnose_a1_fields.py "$tag" \
  --project-root "$root" --output-dir "$out" --device cuda \
  > "$out/launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$out/launcher.pid"
printf '%s\n' "$out" > "$root/reports/a1_field_diag_latest.txt"
printf 'Diagnostic: %s\nLog: %s\n' "$out" "$out/launcher.log"
