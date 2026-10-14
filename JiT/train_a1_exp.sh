#!/usr/bin/env bash
# Same A0 EMA2, same fixed synthetic pairs, independent S0/S1/S2 runs.
set -euo pipefail
arm="${1:-s0}"
gpu="${2:-0}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
data_root="${A1_DATA_ROOT:-/home/qht/大创/CC_defocus}"
data_dir="${A1_EXP_DATA_DIR:-$data_root/experiments/a1_oracle_synthetic/data_2000}"
a0="${A0_CKPT:-$data_root/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth}"
estimator="${LOCAL_ESTIMATOR:-$data_root/JiT/degradation/weights/local-estimator-v3.pt}"
tag="${A1_EXP_TAG:-$(date +%Y%m%d_%H%M%S)}"
epochs="${A1_EXP_EPOCHS:-10}"
warmup="${A1_EXP_WARMUP:-2}"
case "$arm" in
  s0) mode=none ;;
  s1) mode=predicted_local ;;
  s2) mode=oracle_local ;;
  *) echo 'Usage: bash JiT/train_a1_exp.sh {s0|s1|s2} GPU_ID' >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one integer' >&2; exit 2; }
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid experiment tag' >&2; exit 2; }
[[ "$(git -C "$code_root" branch --show-current)" == jit-a1-exp ]] || {
  echo 'Select jit-a1-exp first' >&2; exit 2;
}
for file in "$a0" "$data_dir/generation_config.json" \
  "$data_dir/manifests/synthetic_train.jsonl" "$data_dir/manifests/synthetic_val.jsonl"; do
  [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 2; }
done
if [[ "$arm" == s1 && ! -f "$estimator" ]]; then
  echo "Missing: $estimator" >&2; exit 2
fi
name="jit_a1_exp_${arm}_${tag}"
run_dir="$code_root/checkpoints/$name"
[[ ! -e "$run_dir" ]] || { echo "Already exists: $run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
git -C "$code_root" rev-parse HEAD > "$run_dir/git_head.txt"
git -C "$code_root" status --short --branch > "$run_dir/git_status.txt"
args=(
  --model JiT-B/16 --img_size 256 --init_mode scratch
  --a0_checkpoint "$a0" --a0_weights ema2 --degradation_mode "$mode"
  --data_root "$data_root" --generation_config "$data_dir/generation_config.json"
  --train_manifest "$data_dir/manifests/synthetic_train.jsonl"
  --val_manifest "$data_dir/manifests/synthetic_val.jsonl"
  --epochs "$epochs" --batch_size 8 --lr 1e-4 --warmup_epochs "$warmup"
  --min_lr 1e-6 --lr_schedule cosine --weight_decay 0 --clip_grad 1
  --identity_ratio 0 --lambda_pix 1 --charbonnier_eps 1e-3
  --ema_decay1 0.9999 --ema_decay2 0.9996
  --P_mean -0.8 --P_std 0.8 --noise_scale 1 --t_eps 0.05
  --label_drop_prob 0 --attn_dropout 0 --proj_dropout 0
  --sampling_method euler --num_sampling_steps 1
  --cfg 1 --interval_min 0 --interval_max 1
  --eval_weights ema2 --eval_seed 0 --eval_batch_size 4 --eval_freq 0
  --seed 0 --num_workers 4 --amp_bf16 --run_metrics --save_last_freq 1 --log_freq 20
  --model_name "$name" --output_dir "$run_dir"
  --prediction_root "$code_root/outputs" --results_root "$code_root/results"
)
[[ "$arm" != s1 ]] || args+=(--degradation_checkpoint "$estimator")
cd "$code_root/JiT"
nohup env CUDA_VISIBLE_DEVICES="$gpu" python -u main_a1_exp.py "${args[@]}" \
  > "$run_dir/train.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/train.pid"
printf 'Arm: %s\nRun: %s\nLog: %s\nResults: %s\n' \
  "$arm" "$name" "$run_dir/train.log" "$code_root/results/$name/summary.json"
