# MPT 本地复现记录

## 复现范围

- 成员 C 原 RFN 职责替换为 MPT 作者 3DHistech 预训练权重推理。
- 只增加 manifest、device 与输入输出适配，不修改 MPT 核心网络。
- 固定使用 `prepared/manifests/debug_val.jsonl`，不得重新抽样。
- 输出固定为 `outputs/mpt/{sample_id}.png`。
- 正式指标只由仓库根目录的 `evaluate.py` 产生。

## 官方来源

- 仓库：`https://github.com/PieceZhang/MPT-CataBlur`
- commit：`3078ea354b547f4c403527f12c3f55a436bddaff`
- 权重包：官方 README 的 `Weights.zip`
- ZIP SHA-256：`1D54BF642D02F2A45D70E4563FDBD3D95145B0BECA2070C87F76EEC10801EABB`
- 3DHistech checkpoint：`3dhistech.pytorch`
- checkpoint SHA-256：`6E3DBD3BAAE97413AF2AC265C9B007033910DD6AF2B26730C71368B899EF455E`

## 本地目录约定

- 官方代码：仓库同级的 `external/MPT-CataBlur`
- 官方权重：仓库同级的 `weights/mpt/official/3dhistech.pytorch`
- 独立环境：由使用者自行指定，不放入 Git
- 数据入口：`prepared/images/3DHistech` 内仅为冻结的 300 对
  `debug_val` 建立同盘硬链接；不复制原图，也不改变 manifest。

## 兼容性说明

- 本机为 Intel Arc 140T，没有 NVIDIA CUDA。
- 官方 `run_inference.py` 无条件调用 `.cuda()`，不能直接在本机运行。
- `mpt_infer.py` 使用官方 `MPT` 类并允许 CPU/CUDA 设备选择。
- 官方 checkpoint 的键带 `Network.` 前缀；适配器只在加载时移除该包装前缀，并执行严格权重匹配。
- 不采用官方脚本的 `[:1024, :1280]` 裁剪；统一评价要求输出与参考保持原尺寸。

## 固定流程

1. 先运行 5 张：`python mpt_infer.py --limit 5 --device cpu`
2. 检查运行日志、RGB、尺寸、范围、颜色及文件名。
3. 小样本通过后运行全部 300 张：`python mpt_infer.py --device <cpu|cuda>`
4. 全部输出完成后运行：`python evaluate.py --model-name mpt`

在小样本检查通过前，不运行完整 300 张；完整输出不足 300 张时，不运行正式统一评价。

## 本次运行结果

- 设备：CPU（Intel Arc 不参与 PyTorch 2.0 CPU 推理）
- PyTorch：`2.0.0+cpu`
- 成功/失败：`300 / 0`
- 总推理时间：`998.519 s`
- 平均模型推理时间：`3.315 s/张`
- 输出数量与固定 manifest：`300 / 300`，ID 集合完全一致
- PSNR：`34.0113309694 ± 1.8349148226`
- SSIM：`0.8904043251 ± 0.0285487403`
- LPIPS：`0.1040103884 ± 0.0290206126`
- 逐图指标：`results/mpt/per_image_metrics.csv`
- 统一汇总：`results/mpt/summary.json`
- 完整推理日志：`results/mpt/inference_full300_cpu.json`

以上指标由 `main` 的 `evaluate.py` 直接生成，设置为全图 RGB、无裁边、
无缩放、LPIPS AlexNet。结果类型为“作者 3DHistech 预训练权重开箱推理”，
不应描述为统一训练预算下的最终公平排行榜。
