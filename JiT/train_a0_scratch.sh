#!/usr/bin/env bash
# Activate jit-a0 on the GPU server before invoking this script.
# No model download or environment installation is performed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gpu="${1:-0}"
stage="${2:-overfit16}"
case "$stage" in
  overfit16) subset=16; epochs=100; batch=4; identity=0; schedule=constant ;;
  debug2000) subset=0; epochs=200; batch=8; identity=0.10; schedule=cosine ;;
  *) echo "Usage: bash train_a0_scratch.sh [GPU] [overfit16|debug2000]" >&2; exit 2 ;;
esac
name="jit_a0_scratch_${stage}_charb1"
output="../checkpoints/$name"
if [[ -e "$output/checkpoint-last.pth" || -e "$output/train.log" ]]; then
  echo "Existing run found: $output. Use --resume manually or choose a new run name." >&2
  exit 1
fi
mkdir -p -- "$output"
echo "Scratch A0: GPU=$gpu stage=$stage; log=$output/train.log"
CUDA_VISIBLE_DEVICES="$gpu" python -u main_restoration.py \
  --init_mode scratch --expected_train_pairs 2000 --expected_val_pairs 300 \
  --overfit_samples "$subset" --epochs "$epochs" --batch_size "$batch" \
  --lr 1e-4 --warmup_epochs 5 --lr_schedule "$schedule" --min_lr 1e-6 \
  --clip_grad 1.0 --identity_ratio "$identity" \
  --lambda_pix 1.0 --charbonnier_eps 1e-3 --amp_bf16 \
  --seed 0 --eval_seed 0 --eval_weights model --num_workers 4 \
  --eval_freq 20 --save_last_freq 10 --model_name "$name" \
  --output_dir "$output" --run_metrics > "$output/train.log" 2>&1
