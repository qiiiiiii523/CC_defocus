# 小型 U-Net 基线

使用 `3D` 模糊 RGB 图恢复同名 `3D_label` 清晰参考。模型为四级 U-Net：编码通道 16/32/64/128，瓶颈 256，解码通道 128/64/32/16；每级两次 3×3 卷积与 ReLU，上采样用双线性插值，跳跃连接拼接。最终 1×1 卷积预测对输入图的修正量，输出为 `clip(模糊输入 + 修正量, 0, 1)`。该卷积以零初始化，训练前输出恰好等于输入图，适合模糊图与参考图本来较接近的数据。训练仍只用 L1 损失，不使用 PSF、核损失或 flow。

当前架构版本为 `residual_v1`。旧版直接生成整图的检查点与本版不兼容；推理脚本会拒绝加载旧检查点。切换到本版后须重新运行八对自检及调试集训练，不要把两版指标混在同一结果行。

在已安装 PyTorch、NumPy、Pillow 的环境中，从项目根目录运行。模型训练和批量推理建议在服务器 GPU 上执行；本地可先检查命令行及少量样本。

## 1. 八对样本自检

```bash
python train_unet.py --overfit-eight
python predict_unet.py --checkpoint checkpoints/unet_overfit8_residual_v1.pt --manifest prepared/manifests/debug_train.jsonl --split train --limit 8 --model-name unet_overfit8_residual_v1 --preview-count 8
```

`runs/unet_overfit8_residual_v1.csv` 记录每轮训练 L1；应与八对模糊输入的 L1 基线比较（当前固定清单约为 `0.0197`），并查看 `prepared/previews/unet_overfit8_residual_v1/` 中的“模糊｜输出｜清晰”拼图。这里不调用验证集评价程序，也不把该权重用于正式训练。

## 2. 调试集训练与统一评价

以下命令会**重新初始化** U-Net，在 2,000 对训练图上训练，并用 300 对验证图的 L1 选择检查点。可按 GPU 资源调整批量大小、轮数和学习率，并在实验记录中写明。

```bash
python train_unet.py --epochs 30 --batch-size 4 --lr 0.001 --seed 42
python predict_unet.py --checkpoint checkpoints/unet_debug_residual_v1.pt --model-name unet
python evaluate.py --model-name unet
```

训练日志在 `runs/unet_debug_residual_v1.csv`，配置在 `runs/unet_debug_residual_v1.json`，权重在 `checkpoints/unet_debug_residual_v1.pt`。预测图写入 `outputs/unet/`；前十组对照图写入 `prepared/previews/unet/`，从左到右依次为模糊图、恢复图、清晰参考。`evaluate.py` 负责 PSNR、SSIM、LPIPS，模型代码不另算指标。

训练还会每轮保存 `checkpoints/unet_debug_residual_v1_last.pt`，其中包含最新模型、优化器和随机状态。`unet_debug_residual_v1.pt` 仍是验证 L1 最佳权重，供 `predict_unet.py` 使用。若训练中断，使用与首次训练相同的参数和文件路径，加上 `--resume`；`--epochs` 表示最终总轮数，不是额外轮数。例如先计划训练到 200 轮：

```bash
python train_unet.py --epochs 200 --batch-size 16 --lr 0.001 --seed 42 --device cuda --checkpoint checkpoints/unet_debug_residual_v1_b16_e200.pt --log runs/unet_debug_residual_v1_b16_e200.csv
```

如果在第 80 轮之后中断，用下面的命令从第 81 轮接着训练至第 200 轮：

```bash
python train_unet.py --epochs 200 --batch-size 16 --lr 0.001 --seed 42 --device cuda --checkpoint checkpoints/unet_debug_residual_v1_b16_e200.pt --log runs/unet_debug_residual_v1_b16_e200.csv --resume
```

对应的续训状态文件是 `checkpoints/unet_debug_residual_v1_b16_e200_last.pt`。若之后决定训练至第 250 轮，只需把续训命令的 `--epochs` 改成 `250`，其余参数和路径保持一致。旧版检查点只有最佳模型权重，没有优化器状态，不能用于 `--resume`；需要用更新后的脚本开始一次新训练。

以上结果仅是 2,000/300 子集上的调试基线。正式比较应另行使用官方完整训练集训练及完整验证集评价；本分支的训练脚本目前特意限制为固定调试清单，避免误把测试集混入训练。
