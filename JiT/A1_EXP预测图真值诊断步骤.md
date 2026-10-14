# V3预测图与合成真值：服务器诊断步骤

服务器：`/home/qht/大创/CC_defocus`；分支：`jit-a1-exp`；原三组实验标签：`20261014_114501`。

本次只运行冻结估计器，检查已有合成验证集300张；不训练JiT，不重新生成模糊图，不修改原权重、数据和评价结果。只需一张卡。

## 1. 更新代码并启动

在原来能运行实验的终端中执行：

```bash
cd "/home/qht/大创/CC_defocus"
git pull --ff-only
bash JiT/run_a1_field_diag.sh 20261014_114501 0
```

最后的`0`是GPU编号，可改为`1`或`2`。默认后台运行，检查完整300张验证图，并保存8张代表样例。不用切分支、重装环境或重新上传权重。

脚本从原S1的`checkpoints/jit_a1_exp_s1_20261014_114501/run_config.json`读取合成清单、生成配置和估计器路径，验证估计器文件SHA与当时训练一致。不要删除这个配置文件；如权重文件移动了，可按第5节显式指定新路径。

## 2. 查看进度

```bash
DIAG="$(cat reports/a1_field_diag_latest.txt)"
tail -f "$DIAG/launcher.log"
```

出现`Complete: .../report.md`表示完成。Ctrl+C只退出日志查看，后台诊断继续。若有报错，保留日志，先不要启动训练。

## 3. 查看结果

```bash
cat "$DIAG/report.md"
```

输出在新的`reports/a1_field_diag_.../`目录：

| 文件 | 用途 |
|---|---|
| `report.md` | 可读结果：通道、组织/背景、空间响应、缩放变化 |
| `summary.json` | 完整统计、有效样本数、来源哈希与限制 |
| `per_field_metrics.csv` | 逐图、分辨率、区域、通道统计 |
| `per_image.csv` | 逐图总体/组织误差与区域占比 |
| `resize_metrics.csv` | 64→16→64往返MAE、空间std保留情况 |
| `examples/` | 8张共享色标拼图及对应预测/真值NPZ |
| `launcher.log` | 日志 |

若换了终端，先重新执行`cd`和`DIAG=...`再查看或打包。

## 4. 打包发回

```bash
DIAG="$(cat reports/a1_field_diag_latest.txt)"
tar -czf a1_field_diag.tar.gz -C "$(dirname "$DIAG")" "$(basename "$DIAG")"
```

下载项目根目录的`a1_field_diag.tar.gz`即可。它包含数值、样例和日志，不包含模型权重或整套数据。队友也可以直接查看`report.md`与`examples/`。

## 5. 仅在估计器文件移动时指定新路径

仍需是同一份V3权重；程序会核对SHA。下面只是替代第1节的启动方式，正常情况不用执行：

```bash
DIAG="$PWD/reports/a1_field_diag_manual_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python -u JiT/diagnose_a1_fields.py 20261014_114501 \
  --estimator "/替换为实际目录/local-estimator-v3.pt" --output-dir "$DIAG"
```

这条自定义命令前台运行；完成后可用相同`DIAG`变量打包。它不更新`a1_field_diag_latest.txt`，请保留当前变量。

## 6. 检查口径与能说明什么

1. 估计器只读取模糊RGB，和S1使用相同归一化/推理函数；冻结、eval、FP32、关闭autocast。清晰图仅用于颜色阈值分区。
2. 概率通道固定为light/medium/heavy。它们是软混合权重，不把argmax分类准确率当主要指标。
3. 两套分辨率均计算MAE、RMSE、bias、空间std、每图空间相关性。常量场的相关性记为空并报告有效数，不强行记0。
4. 比较原预测图和“本图预测的空间均值”：原图误差更低，才支持其空间起伏有助于匹配真值。这个收益不是恢复PSNR收益。
5. 概率场64→16用生产模块相同的bilinear、align_corners=False，不用area pooling，不额外归一化；统计的是FP32适配器输入场，不是在重跑BF16编码器或干预恢复输出。mask的area fraction仅用于区域统计，不是条件场缩放方式。
6. 记录64→16→64往返变化；16网格MAE下降可能只是平滑，不能单独证明信息保留良好。
7. 组织/背景按清晰RGB阈值粗分，不是人工标注。平坦背景的混合权重可能不可从图像识别，不能仅凭背景误差大归责估计器。
8. 结论只涉及这套合成混合场；不能直接代表真实模糊的物理参数，也不能替代同权重下的条件干预。

本机仅执行语法和NumPy/PIL测试，不运行Torch、模型推理或GPU检查。实际诊断数值需要服务器运行后才能分析。
