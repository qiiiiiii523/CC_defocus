# JiT A2：局部重模糊一致性损失接入

日期：2026-10-11。代码分支 `jit-a2` 从 `jit-a1` 的 `bdef0dc` 创建。
复用的是代码与严格的 A0 初始化、冻结估计器、EMA、记录和评价入口。
训练始终从固定 A0 checkpoint 的 EMA2 开始，不从已训练的 A1 权重继续。

## A2 与 A1 的区别

| 实验模式 | JiT 额外退化条件 | 重模糊损失 |
|---|---|---|
| `none`：A0-continued | 无 | 无 |
| `predicted_local`：A1 | 局部概率图注入图像 token | 无 |
| `psf_local`：A2 | 无；两个退化条件适配器均不构造 | 有，由 `lambda_psf` 控制 |

原 RGB 模糊图条件保留；A2 的 `psf_local` 仅表示预测用于 loss 的局部 PSF 场，
不是把它送给恢复网络。此分支保留 A0/A1 模式，未启用 A3 混合模式。

```text
实际模糊输入 → 冻结的 3.0 估计器 → 局部轻/中/重概率图 ─┐
                                                         ↓
JiT 预测清晰终点图 → 三种完整图像高斯卷积 → 按位置概率混合
                                                         ↓
                                             与实际模糊输入比较
```

目标：`loss = loss_flow + lambda_pix * loss_charb + lambda_psf * loss_psf`。
一致性项使用 C 提供的 Charbonnier 重模糊误差，`psf_eps` 默认 `1e-3`。
不是对当前噪声状态 `z_t` 做重模糊，而是对本次训练 forward 的 `x_pred` 做重模糊。

## 接入文件与边界

- `degradation/spatial_psf.py`：从用户的 3.0 交接包原样引入。
- `degradation/psf.py`：沿用已有版本，与交接包一致；轻/中/重为 σ=1.0/2.0/3.5、K=7/13/23，reflect 边界。
- `degradation/a2_loss.py`：把 JiT 的 `[-1,1]` 图像转换为公共 RGB 尺度，在 FP32 中计算一致性损失。
- `denoiser.py`：冻结局部估计器、A2 不注入退化图、汇总原损失和 PSF 损失；推理跳过估计器。
- `engine_restoration.py`：传入恒等样本标志，记录原始和加权 PSF 损失。
- `main_restoration.py`：A2 模式、系数、恢复兼容性和配置记录。

估计器始终只读取本批实际 blur，不读取清晰目标或真实 PSF 标签，保持 eval、
requires_grad=False、no_grad 和 FP32。loss 使用已冻结的概率图，梯度只回到恢复网络。
恢复图不 clamp、不 detach，也不把整个重模糊 loss 放进 no_grad；超出 [0,1] 的
恢复预测仍保留梯度，由原像素目标及一致性目标约束。

原训练有 10% 恒等替换：用清晰图作为输入、目标仍是清晰图。此时三种核都不是
零模糊，直接施加重模糊 loss 会与恒等目标冲突。本接入只对非恒等样本计算 PSF loss，
恒等样本仍完整参与 Flow/Charbonnier。若一个 batch 全是恒等样本，PSF 项为可微的 0。

C 的算子是三种预设高斯核的局部混合近似，不是真实显微镜 PSF 测量；其离线拟合
改善不能直接证明 A2 恢复收益。`lambda_psf=0.1` 在下面仅作为首个训练候选，尚未验证
为最佳系数。检查原始/加权 PSF loss 与主损失量级，再用明确记录的验证对照选择系数。

## 服务器上传与切换

沿用原 `/home/qht/大创/CC_defocus` 目录和已有数据，不创建额外 worktree，
不添加 `--data_root`，不修改公共清单路径协议。

同步本分支变更的以下文件即可：

```text
JiT/denoiser.py
JiT/main_restoration.py
JiT/engine_restoration.py
JiT/degradation/a2_loss.py
JiT/degradation/spatial_psf.py
JiT/test_a2_entry.py
JiT/test_a2_integration.py
JiT/A2_INTEGRATION.md
```

服务器必须已有本次 A1-v3 所需的 `local_model.py`、`local_condition.py` 和
支持 3.0 的 `local_inference.py`。不要用交接包里的旧全局 `model.py` 替换它们。

权重仍使用同一份（已存在则不再上传）：

`/home/qht/大创/CC_defocus/JiT/degradation/weights/local-estimator-v3.pt`

SHA-256：`04cb85d6d59016e4a7e3cf2f83ce607bb771d0556c8bdb96c7fe6682178adc0b`。
权重被 Git 忽略，推送代码不会携带它。原 A0 checkpoint、图片和清单继续复用。
通过 Git 同步时先推送本地 `jit-a2`，服务器 fetch 后切换该分支；手动同步时，
先在服务器从 `jit-a1` 创建 `jit-a2`，再把上述文件按目录结构上传。

## 新训练：从同一个 A0 EMA2 起点开始

激活原来的 `jit-a0` 环境，在 Bash 中执行。A2 的初始参数和额外预算与 A1 对齐；
只增加 PSF 损失。不要加载 `a0_v1_continued_e10` 或 A1 实验 checkpoint 作为新起点。

```bash
cd "/home/qht/大创/CC_defocus/JiT"
ROOT="/home/qht/大创/CC_defocus"
RUN_NAME="jit_a2_local_v3_lpsf01_e10_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/checkpoints/$RUN_NAME"
mkdir -p "$RUN_DIR"

COMMON_ARGS=(
  --model "JiT-B/16" --img_size 256
  --train_manifest "$ROOT/prepared/manifests/3dh_full_train.jsonl" --expected_train_pairs 66976
  --val_manifest "$ROOT/prepared/manifests/debug_val.jsonl" --expected_val_pairs 300
  --epochs 10 --batch_size 8 --lr 1e-4 --lr_schedule cosine
  --warmup_epochs 2 --min_lr 1e-6 --weight_decay 0 --clip_grad 1
  --lambda_pix 1 --charbonnier_eps 1e-3 --identity_ratio 0.10
  --lambda_psf 0.1 --psf_eps 1e-3
  --ema_decay1 0.9999 --ema_decay2 0.9996
  --eval_weights ema2 --sampling_method euler --num_sampling_steps 1
  --noise_scale 1 --t_eps 0.05 --seed 0 --eval_seed 0
  --eval_batch_size 4 --amp_bf16 --num_workers 4 --run_metrics
  --prediction_root "$ROOT/outputs" --results_root "$ROOT/results"
  --model_name "$RUN_NAME" --output_dir "$RUN_DIR"
)

nohup python -u main_restoration.py "${COMMON_ARGS[@]}" \
  --a0_checkpoint "$ROOT/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth" \
  --a0_weights ema2 \
  --degradation_mode psf_local \
  --degradation_checkpoint "$ROOT/JiT/degradation/weights/local-estimator-v3.pt" \
  > "$RUN_DIR/train.log" 2>&1 &
echo $! > "$RUN_DIR/train.pid"
echo "实验名：$RUN_NAME"
echo "日志：$RUN_DIR/train.log"
```

`psf_local` 新实验必须显式指定 `--lambda_psf`；0 可用于零权重接线对照。
把 `--lambda_psf` 设为非零同时选择 `predicted_local` 会拒绝，避免误把 A3 当作 A2。

每个新系数使用新实验名和目录；不要覆盖 A0、A1 或已有 A2 结果。
结果与先前一致，训练结束用 EMA2/一步 Euler/seed0 自动生成验证图并评价 PSNR/SSIM/LPIPS。

```bash
tail -f "$RUN_DIR/train.log"
# 训练完成后查看指标
cat "$ROOT/results/$RUN_NAME/summary.json"
```

新增日志键为 `loss_psf`、`loss_psf_weighted`；TensorBoard 使用 `a2/train_...`。
配置记录 `lambda_psf`、`psf_eps`、`degradation_injection=none`、恒等排除策略，
以及 A0、估计器和训练/验证清单的哈希。`lambda_psf=0` 时记录 PSF 未启用。

## 中断恢复与评价

同一终端的变量和 `COMMON_ARGS` 保留时，去掉 A0 起点和外部估计器路径：

```bash
nohup python -u main_restoration.py "${COMMON_ARGS[@]}" \
  --resume "$RUN_DIR/checkpoint-last.pth" \
  >> "$RUN_DIR/train.log" 2>&1 &
echo $! > "$RUN_DIR/train.pid"
```

重新登录后按原实验名重新设置变量和公共参数，不要生成新的 RUN_NAME。
checkpoint 内保存冻结估计器及其 EMA；恢复自动继承 `psf_local`。
`lambda_psf` 和 `psf_eps` 可省略以继承保存值，也可显式传入完全相同的值；
更换系数或模式会拒绝。`--epochs 10` 表示本轮总共 10 轮，不是再增加 10 轮。
仅评价时在恢复命令中增加 `--eval_only`，保留统一评价参数。

## 检查状态

本机只做语法、配置和接线检查，不检查环境/GPU、不运行模型、不启动训练。
无模型检查命令：

```bash
python -m unittest test_a2_entry test_scratch_entry
```

服务器另提供真实 Tensor/梯度及兼容性检查，本次未执行：

```bash
python -m unittest test_a2_integration test_a1_integration
```

这些检查覆盖已知高斯核重模糊、PSF 梯度、冻结场、无退化注入、恒等排除、
超范围恢复梯度、A0 初始化、推理不调用估计器和状态恢复；仍不能替代实际恢复训练。
公平比较 A0-continued、A1 和 A2 时，需同一 A0 EMA2 起点、相同数据、额外预算及评价设置。
