# 3DHistech 公共数据与评价接口

本目录提供三种恢复模型共同使用的数据清单、图像输入输出协议和评价程序。模型分支只需要实现模型推理，并将结果按统一格式保存。

## 文件说明

- `generate_debug_manifests.py`：从完整 `rfn.jsonl` 中固定抽取 `debug_train` 2000 对和 `debug_val` 300 对。只使用官方 `train`、`val` 中状态为 `pass` 的记录，固定种子为 `20260923`，不使用测试集。
- `prepared/manifests/debug_train.jsonl`：模型调试训练清单。
- `prepared/manifests/debug_val.jsonl`：统一推理和评价清单。
- `common_io.py`：加载清单、读取 RGB 配对图像、检查尺寸并保存统一格式的预测 PNG。
- `evaluate.py`：从模型输出 PNG 计算逐图及汇总的 PSNR、SSIM 和 LPIPS。

原始数据和模型输出不提交到 Git。每个人在本地准备 `prepared/images/3DHistech/`，并保持清单中的相对路径不变。


## 模型分支调用公共接口

公共接口使用 `HWC RGB float32 [0,1]` NumPy 数组。模型分支负责转换为自己的 PyTorch 或 TensorFlow tensor，并在推理后转回公共格式。

```python
from common_io import load_manifest, load_pair, save_prediction

records = load_manifest(
    "prepared/manifests/debug_val.jsonl",
    expected_split="val",
)

for record in records:
    blur, clear = load_pair(record)
    prediction = run_model(blur)
    save_prediction(
        prediction,
        model_name="unet",
        sample_id=record.sample_id,
        expected_hw=clear.shape[:2],
    )
```

模型输出统一保存为：

```text
outputs/{model_name}/{sample_id}.png
```

`prediction` 必须是 `HWC RGB` 浮点 NumPy 数组，并且尺寸与清晰参考一致。公共保存函数会检查通道、尺寸、数据类型和 NaN/Inf，再保存为无损 RGB PNG。

## 统一评价

模型完成全部 300 个 `debug_val` 输出后运行：

```bash
python evaluate.py --model-name unet
```

其他模型只需替换名称，例如 `dgno` 或 `rfn`。

评价程序只读取固定的 `debug_val.jsonl`，不缩放、不裁边。PSNR 和 SSIM 使用全图 RGB `[0,1]`，LPIPS 使用 AlexNet 并将输入转换到 `[-1,1]`。

结果保存为：

```text
results/{model_name}/per_image_metrics.csv
results/{model_name}/summary.json
```

模型分支不得复制或修改评价逻辑，应直接调用 `main` 中的公共接口和评价程序。

## DGNO 基线（成员 B）

`baseline/dgno` 分支已固定官方 DGNO 源码版本，并提供适配公共清单与输出格式的推理入口。默认使用 3DHistech 的 DGNO-Face 官方预训练权重。

```bash
conda env create -f environment/dgno.yml
conda activate dgno
pip install mamba-ssm==2.2.2 --no-build-isolation
cd third_party/DGNO && python setup.py develop --no_cuda_ext && cd ../..
python scripts/dgno/download_weights.py --variant face
python scripts/dgno/infer.py --variant face --limit 5
python scripts/dgno/infer.py --variant face
python evaluate.py --model-name dgno
```

完整的数据路径、服务器准备、实验记录和 Cell 变体命令见 [DGNO 基线复现说明](docs/dgno_reproduction.md)。
