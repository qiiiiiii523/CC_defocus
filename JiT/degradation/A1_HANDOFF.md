# 成员 C → JiT A1 训练交接

成员 C 已完成退化估计代码、预测接口、固定数据清单核对和检查报告；**尚未改 JiT A0 主干，也未训练 A1**。GitHub 的 `a1-degradation` 分支只放代码和文档。权重、预测 CSV 和示例图在单独的本地交接包中，由用户发给接收方；原始图像不进 GitHub。

**2026-10-07 更新：**新增连续强度拟合与重训权重 `estimator-continuous.pt`，作为当前 A1 预测条件的优先探索版本。它在同一套新近似标签上优于旧权重与固定强度，但尚未通过 JiT 恢复效果验证。数据、指标与限制见 `CONTINUOUS_V1_RESULTS.md`。

## 接收方先检查数据和权重

A1 固定使用 3DHistech `prepared/manifests/debug_train.jsonl` 的 2,000 对和 `debug_val.jsonl` 的 300 对。清单只有路径，不包含图像。接收方已有原始图像即可，无需从用户重复接收，但训练服务器上必须能按清单找到每条 `blur_path` 与 `clear_path`：

```bash
python JiT/degradation/check_training_data.py \
  --project-root . \
  --train-manifest prepared/manifests/debug_train.jsonl \
  --val-manifest prepared/manifests/debug_val.jsonl
```

旧交接包有两份权重；新交接包另含改进权重：

- `estimator-continuous.pt`：**当前 A1 探索性对照优先使用**。连续近似强度标签、可信度加权和训练集内部分组验证后重训；在固定 300 对的新近似标签上强度 MAE `0.152`、三档正确 `267/300`。仍非真实 PSF 标注或去模糊收益证明。
- `estimator_a1_3dh_pseudo.pt`：旧版对照权重。旧近似标签准确率 `83.0%`，低于当时固定预测轻档的 `85.7%`；旧版结果口径与新标签结果不同，不直接比较。
- `estimator_cnseg_synthetic.pt`：CNSeg 第二版 `pass/exclude` 清单的三档合成失焦权重；合成验证集 `912/912` 正确，但跨到固定 3DHistech 图像时 300 张全判轻档。只作为合成训练对照，不建议直接作为 A1 的主要条件。

接收方自行把权重放在其训练服务器上的私有路径，运行时传入实际路径；**不要将 `.pt` 推到 GitHub**。

## 输入输出约定

- 估计器只读模糊 RGB 图：`float32 [B,3,H,W]`，范围 `[0,1]`；不读清晰目标或近似标签。输出三个 logit，`softmax` 后为 `[B,3]` 概率，顺序固定 `light, medium, heavy`。
- 三个预设全局高斯核为 `(σ=1.0,K=7)`、`(2.0,13)`、`(3.5,23)`，`reflect` 边界。概率加权 `sigma` 只是这三个预设值的摘要，**不是 3DHistech 的真实局部 PSF**。
- JiT A0 的 `blur` 是 `[-1,1]`。`predict_from_jit_blur` 将它转换成 `[0,1]` 再用冻结估计器输出概率。`DegradationCondition(hidden_size)` 把概率编码成 `[B,hidden_size]` 的残差向量，零门控初始化使 A1 初始行为等于 A0。
- `reblur_mixture(restored_01, probabilities)` 是将来 A2/A3 可用的可微重新模糊；图像和比较目标需统一在 `[0,1]`。它通过数值和梯度检查，未做 A2/A3 训练。

在仓库根目录运行一次权重与接口检查：

```bash
python JiT/degradation/inference.py \
  --checkpoint /path/to/estimator-continuous.pt \
  --image prepared/images/3DHistech/3D/10140071_10164_34804.png \
  --device cuda

python JiT/degradation/verify_handoff.py \
  --checkpoint /path/to/estimator-continuous.pt \
  --image prepared/images/3DHistech/3D/10140071_10164_34804.png \
  --device cuda
```

从 Python 调用：

```python
from JiT.degradation.inference import load_estimator, predict_probabilities

estimator = load_estimator("/path/to/estimator-continuous.pt", "cuda")
probabilities = predict_probabilities(estimator, blur_01)  # [B,3]
```

## 接收方还需完成的 JiT A1 工作

当前 `JiT/model_jit.py` 的 `JiT.forward(x, t, y, blur=None, degradation=None)` 已预留 `degradation` 参数，但 A0 遇到非空值会抛 `NotImplementedError`；`JiT/denoiser.py` 的训练和采样路径同样尚未接线。接收方应在自己的 A1 分支：

1. 从 A 训练好的 **A0 检查点**出发，保留 A0 原有的模糊图空间条件；增加 `DegradationCondition(hidden_size)`。在现有 `c = t_emb + y_emb` 后加入预测概率的残差条件。原始 ImageNet 类别嵌入不是模糊等级。
2. 在 `Denoiser.forward` 和 `generate` 中都只由输入 `blur` 预测概率。首次对照冻结估计器；采样前预测一次，并将同一个条件传给全部 ODE 步。训练和推理不得读取清晰目标或拟合标签来构造条件。
3. 用同一 A0 起点、固定 2,000/300 清单、相同图像处理、预算和评价程序做 A0/A1 对照。即使新估计器在近似标签上有提升，仍需增加**固定条件**对照，以判断恢复收益是否来自预测内容，而非仅增加了参数。

旧版结果见 `RESULTS.md`，改进版见 `CONTINUOUS_V1_RESULTS.md`。A1 接线、训练和恢复图评价完成前，不能声称 A1 已完成或退化条件已经带来收益。
