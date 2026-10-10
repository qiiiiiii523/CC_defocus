# SD3.5 Medium VAE 编解码结构检查

- 日期：2026-10-04
- 评估设备：NVIDIA GeForce RTX 4080 SUPER
- 推理设备：CUDA
- 精度：FP16
- 样本数：12（固定 CNSeg 验证集，已冻结）
- VAE 权重：`stabilityai/stable-diffusion-3.5-medium` 官方 `vae` 子目录
- 评估性质：VAE encode-decode 结构检查，不代表 SD3.5 去模糊性能

## 汇总指标

| 指标 | 12 样本均值 |
|---|---:|
| PSNR | 38.7862 dB |
| SSIM | 0.9681 |
| LPIPS | 0.00771 |
| 核区域 MAE | 0.01339 |
| 核边界 MAE | 0.01356 |

- VAE 加载时间：0.190 秒（云端本次运行记录）
- 峰值 CUDA 显存：0.779 GiB
- LPIPS 使用官方 torchvision AlexNet 权重完成，12 个样本均有有效值。

## 输出

- `results/vae_reconstruction/vae_eval_summary.json`
- `results/vae_reconstruction/*_reconstruction.png`
- `previews/vae_reconstruction_comparison.jpg`（原图与 VAE 重建并排，含逐样本指标）
- `previews/vae_reconstruction_comparisons/*_comparison.jpg`（12 张独立对比图）

原始运行使用的 VAE 权重先下载到云卡本地目录，再从本地目录加载，避免镜像端重复检查受限仓库。
