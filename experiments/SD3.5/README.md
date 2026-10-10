# SD3.5 scratch 去模糊训练

Transformer 使用官方 SD3.5 Medium 配置随机初始化，不加载 Transformer 预训练权重；官方 VAE 冻结。模糊潜变量通过条件卷积进入 Transformer，训练条件 Flow Matching，推理只使用模糊图。

训练与推理均保持参数和潜变量 FP32，Transformer 使用 BF16 autocast，VAE 使用 FP32。配置中的梯度累积、梯度检查点、训练样本限制和验证均实际生效。`max_train_steps` 指优化器更新次数；`num_epochs` 指完整遍历训练清单的轮数，同时设置时先达到任一上限即结束。尾部不足一个累积组也会更新，损失按实际样本数加权。

本地使用真实的缩小版 Diffusers SD3 Transformer 验证算法和入口逻辑；完整 SD3.5 Medium、官方 VAE、CUDA 显存及真实小样本过拟合仍必须在服务器验收。正式训练是否收敛取决于数据和超参数，示例配置不构成效果保证。

## 环境

建议独立 Python 3.10–3.12 环境。本地 CPU 验证使用 Python 3.12、Torch 2.8.0+cpu、Diffusers 0.35.2、Transformers 4.56.2。服务器先安装与其驱动匹配的 Torch 2.8.0 / torchvision 0.23.0 CUDA 构建，再安装本目录依赖，避免误用 CPU 构建。确认能访问 `stabilityai/stable-diffusion-3.5-medium` 的 Transformer 配置和 VAE 权重，或通过 `--model-id` 使用包含 `transformer/` 配置与 `vae/` 权重的本地目录。

```bash
cd '/home/qht/大创/CC_defocus'
python -m pip install -r experiments/SD3.5/requirements.txt
python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

这是单 GPU 入口；不通过 torchrun 启动多个独立进程。显存以真实 smoke_test 的峰值为准，不能把此前未启用精度/检查点的估计当作实测值。

## 全量数据准备

`--repo-root` 是 `/home/qht/大创/CC_defocus`，不是图像目录。清单中的相对路径已经包含 `prepared/images/3DHistech/`。需要原有 `prepared/manifests/rfn.jsonl` 和固定 debug 清单。

```bash
python experiments/SD3.5/code/prepare_full_manifests.py \
  --repo-root '/home/qht/大创/CC_defocus' --check-images
```

此命令保留现有划分，仅选 RFN/pass，生成 `full_train.jsonl` 和 `full_val.jsonl`，不写入测试清单，不更改 debug 清单。当前清单应得到 train=66,976、val=9,088、test=18,909。路径、配对和跨 split 的 group_id 均检查；`--check-images` 还逐图解码并检查原始 256×256 尺寸。生成的全量清单是可再生文件，不需要提交 Git。

## 先验收，再正式训练

先跑真实配对图像的编码、前向、反向、优化器更新、条件敏感性和解码：

```bash
python experiments/SD3.5/code/smoke_test.py \
  --repo-root '/home/qht/大创/CC_defocus'
```

再用固定 8 对训练样本做过拟合。该配置关闭随机裁剪/翻转；在线 val 是独立验证集，不能用它代替训练样本的过拟合判定。

```bash
python experiments/SD3.5/code/train_scratch.py \
  --config experiments/SD3.5/configs/scratch_overfit8.example.json \
  --repo-root '/home/qht/大创/CC_defocus' \
  --output-dir runs/sd35-overfit8
```

检查训练损失下降，并对这 8 个 sample_id 的模糊图调用单图推理，和对应清晰图比较。无法学习或条件无响应时停止验收，定位原因；不要直接启动全量训练。

通过后，调整训练轮数/学习率，启动全量训练：

```bash
python experiments/SD3.5/code/train_scratch.py \
  --config experiments/SD3.5/configs/scratch_full.example.json \
  --repo-root '/home/qht/大创/CC_defocus' \
  --output-dir runs/sd35-full
```

全量配置以 10 轮作为可修改的起点；batch=1、累积=4 时每轮 16,744 次优化器更新。在线监控固定 debug_val 的前 16 对 PSNR，并非完整验证指标。训练日志在 `metrics.jsonl`。

## 保存与续训

结束时始终保存 `checkpoint-final.pt`。周期 checkpoint 保存模型、AdamW、随机数、Transformer 配置、VAE 来源、清单哈希及 epoch/batch 游标，使用临时文件完成后原子替换。默认保留最近 2 个数字命名的周期 checkpoint；`keep_checkpoints=0` 保留全部，final 不被清理。FP32 全模型和 AdamW checkpoint 很大，应为模型参数与优化器状态预留磁盘和内存。

```bash
python experiments/SD3.5/code/train_scratch.py \
  --config experiments/SD3.5/configs/scratch_full.example.json \
  --repo-root '/home/qht/大创/CC_defocus' \
  --output-dir runs/sd35-full \
  --resume runs/sd35-full/checkpoint-2000.pt
```

从实际存在的最近 checkpoint 恢复；周期保存之间尚未保存的更新无法恢复。旧版仅模型权重的 checkpoint 可以用于推理，但不能用于精确续训。续训可以延长训练步数/轮数或调整日志、验证、保存间隔；不允许静默改变 batch、累积数、学习率、训练清单等训练约定。固定种子和按 epoch/index 生成的增强保证 worker 预取不改变样本；同设备环境下恢复模型 RNG。跨 GPU 型号/算子实现不承诺逐位一致。

## 推理与公共评价

推理读取 checkpoint 中的分辨率、alpha、采样步数和模型配置，不缩放原图。FP32 潜变量积分和 BF16 autocast 共用 `flow_matching.sample`，固定种子保证同输入结果可复现。

```bash
python experiments/SD3.5/code/infer_scratch.py \
  --checkpoint runs/sd35-full/checkpoint-final.pt \
  --manifest prepared/manifests/debug_val.jsonl \
  --repo-root '/home/qht/大创/CC_defocus' \
  --model-name sd35_scratch
python evaluate.py --model-name sd35_scratch
```

公共评价继续使用固定 300 对，全图 RGB、无缩放裁边、PSNR/SSIM/LPIPS-Alex。生成全量 val 的图像可以传入 `full_val.jsonl`，但公共 evaluate.py 固定要求 300 对，不能直接把该清单传入来声称完成全量评价。

单图推理：

```bash
python experiments/SD3.5/code/infer_scratch.py \
  --checkpoint runs/sd35-full/checkpoint-final.pt \
  --blur prepared/images/3DHistech/3D/<sample_id>.png \
  --out outputs/sd35_scratch/<sample_id>.png
```

## 本地回归测试

```bash
python -m pip install pytest==8.4.2
python -m pytest experiments/SD3.5/tests -q
```

测试不下载官方模型，使用真实 Diffusers 缩小版 Transformer、真实小型 VAE 和确定性 VAE 替身，验证 BF16 梯度/采样、累积尾批次、checkpoint 恢复、结束保存、数据划分、增强一致性、固定种子推理和可学习性。命令行推理测试将清晰参考文件设为不可解码，确认模型只读取模糊像素。它们不替代服务器的完整模型验收。

当前固定清单 SHA256（本次没有改写清单）：

- debug_train：`C4EA702E559A97027C7476154F6565AE6848CCF3D39C155270BB7624636D640B`
- debug_val：`6CE1474E51C1814191EA01734A98375A6B50261D0B6FEDC74A58693745BB821C`

`reports/preimplementation_review.json` 是历史审查记录，不能当作当前实现的验收报告。
