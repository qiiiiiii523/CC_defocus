# 小型 U-Net 基线

使用 `3D` 模糊 RGB 图恢复同名 `3D_label` 清晰参考。模型为四级 U-Net：编码通道 16/32/64/128，瓶颈 256，解码通道 128/64/32/16；每级两次 3×3 卷积与 ReLU，上采样用双线性插值，跳跃连接拼接，最终 1×1 卷积加 Sigmoid 输出 RGB `[0,1]`。训练只用 L1 损失，不使用 PSF、核损失或 flow。

在已安装 PyTorch、NumPy、Pillow 的环境中，从项目根目录运行。模型训练和批量推理建议在服务器 GPU 上执行；本地可先检查命令行及少量样本。

## 1. 八对样本自检

```bash
python train_unet.py --overfit-eight
python predict_unet.py --checkpoint checkpoints/unet_overfit8.pt --manifest prepared/manifests/debug_train.jsonl --split train --limit 8 --model-name unet_overfit8 --preview-count 8
```

`runs/unet_overfit8.csv` 记录每轮训练 L1；应观察到同一批八对图的损失明显下降，并查看 `prepared/previews/unet_overfit8/` 中的“模糊｜输出｜清晰”拼图。这里不调用验证集评价程序，也不把该权重用于正式训练。

## 2. 调试集训练与统一评价

以下命令会**重新初始化** U-Net，在 2,000 对训练图上训练，并用 300 对验证图的 L1 选择检查点。可按 GPU 资源调整批量大小、轮数和学习率，并在实验记录中写明。

```bash
python train_unet.py --epochs 30 --batch-size 4 --lr 0.001 --seed 42
python predict_unet.py --checkpoint checkpoints/unet_debug.pt --model-name unet
python evaluate.py --model-name unet
```

训练日志在 `runs/unet_debug.csv`，配置在 `runs/unet_debug.json`，权重在 `checkpoints/unet_debug.pt`。预测图写入 `outputs/unet/`；前十组对照图写入 `prepared/previews/unet/`，从左到右依次为模糊图、恢复图、清晰参考。`evaluate.py` 负责 PSNR、SSIM、LPIPS，模型代码不另算指标。

以上结果仅是 2,000/300 子集上的调试基线。正式比较应另行使用官方完整训练集训练及完整验证集评价；本分支的训练脚本目前特意限制为固定调试清单，避免误把测试集混入训练。
