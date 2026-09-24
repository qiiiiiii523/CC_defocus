# RFN 监督重聚焦适配（成员 C）

本分支保留作者 `build_multi_scale_v8` 三输入生成器的网络代码，并接入项目的固定样本清单、RGB 范围和统一评价。训练只读取原始 `3D` 模糊图与对应 `3D_label` 清晰图，使用 MAE 损失从零训练。推理时只需输入一张 `3D` 模糊图；代码从这张图自动裁出生成器所需的三个局部图块，再输出一张清晰图。

这是**RFN 网络结构在本项目配对数据上的监督训练适配**。作者完整方法还包含 DNN 域归一化、核区域辅助图、反向生成器、判别器、循环及注意力损失和第二阶段联合训练。本分支的权重与结果必须标注为“本项目从头训练，配对监督适配”，不能写成作者权重或完整论文复现。

## 来源与改动

作者代码：<https://github.com/ShenghuaCheng/Cytopathology-image-refocusing>，MIT License，`rfn_author/` 保留需要的原始 `mmodels/multi_scale.py`、`mmodels/keras_tools.py` 和 LICENSE；具体版本记录在 `rfn_author/UPSTREAM_COMMIT.txt`。网络代码本身未修改。`rfn_adapter.py` 负责清单读取、`[0,1]` RGB 到 `[-1,1]` 的转换，以及从模糊图自动选择三个高局部对比度图块。作者原来用另外生成的核区域图选择图块；本分支为满足“只输入模糊图”的任务要求，改用确定性的图像内部选择规则。这个预处理差异可能影响结果，不能视为完全相同的论文实现。

## 环境和数据

建议在独立环境中使用 Python 3.10、TensorFlow 2.10.x、NumPy 1.x 和 Pillow。作者原仓库声明 Python 3.6、TensorFlow/Keras 1.13.1；当前适配使用 `tf.keras`，实际 GPU 兼容性仍须在你的机器上验证。

```powershell
py -3.10 -m venv .venv-rfn
.\.venv-rfn\Scripts\Activate.ps1
python -m pip install -r requirements-rfn.txt
```

```text
prepared/images/3DHistech/3D/*.png
prepared/images/3DHistech/3D_label/*.png
```

**不需要解压 `3D_to_3D_kernel.zip`，也不读取 `3D_to_3D` 图像。**

`debug_train.jsonl`、`debug_val.jsonl` 已由 main 固定。完整清单是 `prepared/manifests/rfn.jsonl`，脚本按官方 `train`/`val` 划分筛选 `pass` 记录，不把测试集用于训练。数据、权重和全部输出都留在本地，不提交 Git。

## 运行

先检查清单中的模糊和清晰图片：

```powershell
python rfn_train.py --dry-run
```

用调试子集做一次可运行性检查：

```powershell
python rfn_train.py --epochs 1 --batch-size 1 --run-name rfn_debug
python rfn_predict.py --weights checkpoints/rfn_debug/best.weights.h5 --limit 10
```

检查 10 张图的颜色、尺寸与核结构后，对全部固定验证子集输出并评分：

```powershell
python rfn_predict.py --weights checkpoints/rfn_debug/best.weights.h5
python evaluate.py --model-name rfn
```

对单独的模糊 PNG 进行推理，不需要清晰参考图：

```powershell
python rfn_predict.py --weights checkpoints/rfn_debug/best.weights.h5 --input path/to/blur.png
```

调试无误后，单独从零进行完整训练，并让 RFN 与 DGNO 在同一批 300 张验证图上评分：

```powershell
python rfn_train.py --train-manifest prepared/manifests/rfn.jsonl --val-manifest prepared/manifests/debug_val.jsonl --epochs 30 --run-name rfn_full
python rfn_predict.py --weights checkpoints/rfn_full/best.weights.h5
python evaluate.py --model-name rfn
```

`epochs 30` 只是启动示例，实际轮数应根据验证损失、显存、耗时及训练预算确定，并完整记录。`best.weights.h5` 按验证 MAE 选取；公共 PSNR、SSIM、LPIPS 只由 `evaluate.py` 计算。若已有 `outputs/rfn/` 中其他权重的结果，应先移走或清理后再评分，以免把不同运行结果混在一起。

## 当前验证边界

已用当前机器的 Python 检查调试清单和完整训练清单，分别核对了 2,000/300 与 66,976/300 条配对记录。当前环境没有 TensorFlow，因此尚未运行网络构建和训练。首次启动时若出现兼容错误，应保存最小报错和版本信息，再做兼容性修正；不要悄悄替换核心网络结构。
