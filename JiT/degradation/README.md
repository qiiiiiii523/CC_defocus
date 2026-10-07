# C：退化估计与重新模糊

本目录从已有 CNSeg 原图与二状态清单读取 **pass** 样本，不更改原图。清单中的三档全局高斯参数为 light `(sigma=1.0, K=7)`、medium `(2.0, 13)`、heavy `(3.5, 23)`，均为 reflect 边界。每张原图的三档记录继承相同 train/val/test 划分。

`train_estimator.py` 先对整张原图合成失焦，再裁切图块。估计器只接收模糊图，输出三档概率；连续 `sigma` 估计是这些概率的加权平均。验证同时报告固定预测 medium 的基线。`test` 划分不参与训练或调参。

正式训练后，`preview_predictions.py` 可从验证集各选三例生成“模糊图／清晰参考”对照，并标出真档位与预测档位。训练脚本另写出逐条预测的 CSV 和准确率、sigma 误差；小批量试运行会明确标注 `SMOKE_TEST_ONLY`。

`psf.py` 提供可微的三档高斯模糊，以及使用**预测概率**的核混合重新模糊算子。`verify_psf.py` 检查它与原始 NumPy 处理程序在整数图像上的差异、输出尺寸，以及图像和混合权重两侧的梯度。这是 A2/A3 的接口检查，尚不代表已完成 A2/A3 训练。

`condition.py` 定义 A1 的输入约定：从 JiT 使用的 `[-1,1]` 模糊图取得三档**预测**概率，通过零门控适配器映射到 JiT 的条件向量。零门控保证载入 A0 权重时初始行为不变。`verify_interfaces.py` 检查形状与门控行为；真正将此适配器加入 JiT 主网络和训练 A1，仍需取得 A0 检查点后验证。

`inference.py` 提供权重加载和单张模糊图预测；`predict_manifest.py` 可以只读取固定清单中的模糊图，导出逐张预测概率。具体接线位置与交付边界见 `A1_HANDOFF.md`。

为检查 CNSeg→3DHistech 的域差异，`data_3dh.py` 可从固定 3DHistech 清晰图清单生成同样三档合成失焦。`train_estimator.py --data-kind paired-clear --manifest prepared/manifests/debug_train.jsonl --val-manifest prepared/manifests/debug_val.jsonl` 训练独立的同域对照权重。`assess_paired_psf.py` 只比较配对图的预设高斯核拟合误差，结果不能称为真实 PSF 标签。`check_training_data.py` 在 A1 开训前检查固定清单路径是否齐全。

在仓库根目录执行：

```bash
/root/miniconda3/bin/python JiT/degradation/verify_psf.py --image data/CNSeg/PatchSeg/train-images/0000_0.jpg
/root/miniconda3/bin/python JiT/degradation/verify_interfaces.py
/root/miniconda3/bin/python JiT/degradation/train_estimator.py \
  --project-root . \
  --manifest data/CNSeg/cnseg_binary_20260930.jsonl \
  --output-dir runs/degradation_estimator --device cuda
```

没有 GPU 时可加 `--device cpu --epochs 1 --max-train-batches 2 --max-val-batches 1 --num-workers 0` 做流程检查。这样生成的报告会标为 `SMOKE_TEST_ONLY`，不能当作全量训练结果。

本目录的训练权重和输出保存在 Git 忽略的 `runs/` 下。A1 完整训练还需要 A0 检查点；把这里预测的概率接入 JiT 并在固定 3DHistech 数据上评价后，才算 A1 完成。CNSeg 的合成全局参数不能直接解释为 3DHistech 真实图的局部 PSF 真值。
