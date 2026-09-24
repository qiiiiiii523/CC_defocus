# DGNO 基线实验提交报告（成员 B）

## 1. 实验目标与范围

本次工作对应“本周基线实验三人分工”中的成员 B：在固定验证子集上复现 DGNO 的图像去模糊推理，适配项目公共数据与输出接口，并使用项目统一评价程序报告结果。

本次采用 DGNO 作者发布的 3DHistech DGNO-Face 预训练权重，属于**作者预训练权重推理（开箱测试）**。本周任务不要求重新训练 DGNO；本次也未改动模型核心结构，未加入 PSF 条件或细胞核结构约束。

## 2. 数据与评价协议

- 数据集：3DHistech 模糊—清晰配对图像。
- 清单：`prepared/manifests/debug_val.jsonl`，固定验证调试子集 300 对；不重新抽样，不使用测试集。
- 推理输入：模糊图；参考目标：对应清晰图。
- 输出：每张预测以 RGB PNG 保存，尺寸与对应清晰参考一致，输出目录为 `outputs/dgno/`。
- 指标：调用项目根目录的统一 `evaluate.py`，计算全图 RGB PSNR、SSIM、LPIPS；不缩放、不裁剪边界。LPIPS 使用 AlexNet 实现，输入按协议映射到 `[-1,1]`。
- 指标报告格式：均值 ± 标准差；样本数为 300。

## 3. 环境与模型

- 运行平台：AutoDL 远程服务器，NVIDIA GeForce RTX 4090 D（24 GB）。
- Python：3.10.21。
- PyTorch：2.4.1+cu121；torchvision：0.19.1+cu121。
- 关键依赖：mamba-ssm 2.2.2、transformers 4.44.2、ninja 1.13.2、setuptools 75.8.0。
- 模型：DGNO-Face，作者提供的 3DHistech 预训练权重 `DGNO_3DHistech_Face.pth`。
- 权重 SHA-256：`aa4eb5a4f9539a6cda7026520f4bf891391b414dfb9489bb7f6adba12102993f`。
- 上游 DGNO 源码版本：`318fef62de081878a6498f12f5441acf5e83685d`。
- 项目代码版本：`3d8b722`（`baseline/dgno`）。

完整依赖快照、权重获取说明与复现步骤见 `environment/dgno-freeze.txt` 和 `docs/dgno_reproduction.md`。数据及大型权重未提交到 Git；权重和实验输出按项目忽略规则保存在服务器。

## 4. 运行过程

1. 检查 AutoDL GPU 与 PyTorch CUDA 可用性。
2. 用 5 张固定验证样本进行冒烟测试，确认模型可加载、输入输出接口正常。
3. 对固定清单中的 300 张图像完整推理。
4. 检查推理记录及输出数量，确认所有结果可由统一评价程序读取。
5. 运行统一评价程序，生成逐图和汇总指标。

主要运行入口：

```bash
python scripts/dgno/infer.py --variant face --limit 5
python scripts/dgno/infer.py --variant face
python evaluate.py --model-name dgno
```

## 5. 实验结果

| 项目 | 结果 |
|---|---:|
| 固定验证样本数 | 300 对 |
| 推理成功 / 失败 | 300 / 0 |
| PSNR | 33.7761 ± 1.9486 dB |
| SSIM | 0.8906 ± 0.0290 |
| LPIPS（AlexNet） | 0.1003 ± 0.0264 |
| 批量推理总耗时 | 37.46 秒 |
| 峰值 GPU 已分配显存 | 322,715,136 字节（约 308 MiB） |

耗时和显存为本次 AutoDL RTX 4090 D 运行记录；耗时会受服务器状态及数据读取缓存影响。指标为 300 张图像的均值和标准差，非单张结果。

## 6. 交付文件

服务器项目目录：`/root/autodl-tmp/CC_defocus`

- 恢复图：`outputs/dgno/{sample_id}.png`，共 300 张。
- 实验环境与硬件、代码版本、权重校验值、总体成功/失败和资源记录：`results/dgno/inference_summary.json`。
- 逐图推理耗时及失败记录：`results/dgno/inference_records.csv`。
- 逐图 PSNR、SSIM、LPIPS：`results/dgno/per_image_metrics.csv`。
- 指标汇总：`results/dgno/summary.json`。
- 可复现说明：`docs/dgno_reproduction.md`。

代码、环境说明和推理入口已提交至 GitHub `baseline/dgno` 分支，最新提交为 `3d8b722`。原始数据、预训练权重及全量恢复图未推送到 GitHub。

## 7. 结论与限制

DGNO-Face 作者权重已在本项目固定接口上完成 300 对验证调试样本推理，样本全部成功，统一评价流程正常。本结果可作为本周运行可行性和初步表现记录。

该结果来自固定的 `debug_val` 调试子集，不是官方完整验证集或测试集结果；模型使用作者预训练权重，与其他方法若采用不同训练规模或权重来源，不能据此宣称严格公平排名，也不能作为论文最终结论。后续正式比较应按团队统一方案确定训练数据、训练预算及评价集，再进行完整验证和测试。

## 8. 复现注意事项

- 在项目根目录按 `docs/dgno_reproduction.md` 准备环境、3DHistech 图像和权重。
- 运行前确保 `debug_val.jsonl` 路径下的 300 对图像均可读取。
- 先运行 5 张冒烟测试，再运行完整推理及统一评价。
- 结果生成在远程服务器；由于 `outputs/`、`results/`、`weights/` 和数据目录被 `.gitignore` 排除，提交到 GitHub 的是代码、清单/配置和复现文档，不包含大型数据与结果图。
