# RFN 监督重聚焦适配（成员 C）

本分支保留作者 `build_multi_scale_v8` 三输入生成器的网络代码，并接入项目的固定样本清单、RGB 范围和统一评价。训练以原始 `3D` 为输入、`3D_label` 为目标，使用 MAE 损失从零训练。它是**RFN 生成器的监督部分复现**；作者完整方法还包含 DNN 域归一化、反向生成器、判别器、循环及注意力损失和第二阶段联合训练。本分支的权重与结果必须标注为“本项目从头训练，监督部分”，不能写成作者权重或完整论文复现。

## 来源与改动

作者代码：<https://github.com/ShenghuaCheng/Cytopathology-image-refocusing>，MIT License，`rfn_author/` 保留需要的原始 `mmodels/multi_scale.py`、`mmodels/keras_tools.py` 和 LICENSE。请在提交前记录 `rfn_author/UPSTREAM_COMMIT.txt` 中的具体提交。网络代码本身未修改。`rfn_adapter.py` 负责清单读取、`[0,1]` RGB 到 `[-1,1]` 的转换，以及局部图块。作者加载器从每幅图的核区域辅助图中按面积取前三个局部图块，网络接受整图、图块、位置三个输入；本适配保留这三个输入。作者的局部辅助图由 `3D_to_3D` 域生成，本分支将其位置用于原始 `3D` 图块裁取，因此需人工核对两域图像的空间配准。这一输入域差异会影响与作者论文结果的对应程度。

## 环境和数据

建议在独立环境中使用 Python 3.10、TensorFlow 2.10.x、NumPy 1.x、Pillow 和 OpenCV。作者原仓库声明 Python 3.6、TensorFlow/Keras 1.13.1；当前适配使用 `tf.keras`，实际 GPU 兼容性仍须在你的机器上验证。先完成 CPU 的 1～2 步小样本检查，再安排完整训练。

```powershell
py -3.10 -m venv .venv-rfn
.\.venv-rfn\Scripts\Activate.ps1
python -m pip install -r requirements-rfn.txt
```

从 `D:\大创\data\Cervical Cytopathology Refocusing Dataset\3D_to_3D_kernel.zip` 解压到 `D:\大创\prepared\images\3DHistech\`，最终目录应为：

```text
prepared/images/3DHistech/3D/*.png
prepared/images/3DHistech/3D_label/*.png
prepared/images/3DHistech/3D_to_3D_kernel/*.png
```

`debug_train.jsonl`、`debug_val.jsonl` 已由 main 固定。完整清单是 `prepared/manifests/rfn.jsonl`，脚本按官方 `train`/`val` 划分筛选 `pass` 记录，不把测试集用于训练。数据、权重和全部输出都留在本地，不提交 Git。

## 运行

先检查清单及核辅助图：

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

调试无误后，单独从零进行完整训练，并让 RFN 与 DGNO 在同一批 300 张验证图上评分：

```powershell
python rfn_train.py --train-manifest prepared/manifests/rfn.jsonl --val-manifest prepared/manifests/debug_val.jsonl --epochs 30 --run-name rfn_full
python rfn_predict.py --weights checkpoints/rfn_full/best.weights.h5
python evaluate.py --model-name rfn
```

`epochs 30` 只是启动示例，实际轮数应根据验证损失、显存、耗时及训练预算确定，并完整记录。`best.weights.h5` 按验证 MAE 选取；公共 PSNR、SSIM、LPIPS 只由 `evaluate.py` 计算。若已有 `outputs/rfn/` 中其他权重的结果，应先移走或清理后再评分，以免把不同运行结果混在一起。

## 当前验证边界

已用当前机器的 Python 运行调试清单和完整训练清单的 `--dry-run`，分别核对了 2,000/300 与 66,976/300 条记录及所需核辅助图。脚本和作者网络文件已通过语法解析。当前环境没有 TensorFlow 或 OpenCV，因此尚未运行网络构建、训练、图块提取或推理。首次启动时若出现兼容错误，应保存最小报错和版本信息，再做兼容性修正；不要悄悄替换核心网络结构。
