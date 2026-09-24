# DGNO 基线复现说明

## 固定方案

- 数据：`prepared/manifests/debug_val.jsonl`，共 300 对，只做推理和统一评价。
- 默认模型：DGNO-Face（3DHistech 官方预训练权重）。论文中它在 3DHistech 上的 PSNR 为 34.02 dB，略高于 DGNO-Cell 的 34.00 dB。
- 输出：`outputs/dgno/{sample_id}.png`。
- 评价：只调用项目根目录的 `evaluate.py`，不使用 DGNO 仓库自带指标作为最终结果。
- 官方代码固定版本：见 `third_party/DGNO/UPSTREAM_COMMIT`。
- Python：统一环境使用 3.10。官方依赖注释记录的测试环境是 3.8，README 示例为 3.9；本项目公共接口使用了 Python 3.10 的类型语法，因此统一提升到 3.10，其余官方包版本保持不变。

## 服务器准备

当前实测服务器没有可用 GPU，也没有 Python，因此应先切换到带 NVIDIA GPU 的实例。项目建议放在 `/root/autodl-tmp/CC_defocus`，数据与环境也放在数据盘，避免占用系统盘。

```bash
cd /root/autodl-tmp/CC_defocus
conda env create -f environment/dgno.yml
conda activate dgno
pip install setuptools==75.8.0 ninja==1.13.2 transformers==4.44.2
pip install mamba-ssm==2.2.2 --no-build-isolation
```

DGNO 推理脚本会直接从 `third_party/DGNO` 导入官方源码，不需要运行其 `setup.py develop`。官方裁剪仓库未包含三个可选扩展的源码，强行执行可编辑安装会失败，但这些扩展不参与本项目的 DGNO 推理。

当前 AutoDL 服务器不能直接访问 GitHub Release，因此实际安装使用了上传到数据盘的官方预编译 wheel：

```bash
pip install --no-deps /root/autodl-tmp/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

该 wheel 的 SHA-256 为 `6b082468a6abb6f6bc50c99263f17c6c7f5a2e8f6b275ed7998b81fb25279229`。`environment/dgno-freeze.txt` 记录本次成功导入 DGNO 后的完整包版本；它是实验快照，不是跨机器安装入口。

安装后检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import mamba_ssm; print('mamba_ssm ok')"
```

## 权重

默认只下载 3DHistech Face 权重：

```bash
python scripts/dgno/download_weights.py --variant face
```

权重会保存到 `weights/dgno/DGNO_3DHistech_Face.pth`。脚本同时输出 SHA-256，应把该值保留在实验记录中。权重和数据均被 `.gitignore` 排除，不提交 Git。

## 数据

把 `3D.zip` 和 `3D_label.zip` 上传到服务器后，分别解压到：

```text
prepared/images/3DHistech/3D/
prepared/images/3DHistech/3D_label/
```

这里必须解压，因为公共清单逐张读取 PNG；仅把 ZIP 放在服务器上不能开始推理。解压后先确认清单可读：

```bash
python -c "from common_io import load_manifest; print(len(load_manifest('prepared/manifests/debug_val.jsonl', expected_split='val')))"
```

## 推理与评价

先用 5 张图做冒烟测试：

```bash
python scripts/dgno/infer.py --variant face --limit 5
```

确认 `results/dgno/inference_summary.json` 中 `failed_count` 为 0 后，清空测试输出并运行完整 300 张：

```bash
rm -rf outputs/dgno results/dgno
python scripts/dgno/infer.py --variant face
python evaluate.py --model-name dgno
```

最终保留：

- `results/dgno/inference_summary.json`：代码版本、权重哈希、环境、GPU、总耗时、峰值显存和成功/失败数。
- `results/dgno/inference_records.csv`：逐图耗时及失败原因。
- `results/dgno/per_image_metrics.csv`：逐图 PSNR、SSIM、LPIPS。
- `results/dgno/summary.json`：统一指标均值与标准差。

若需要补跑 Cell，使用 `--variant cell --model-name dgno_cell`，避免覆盖 Face 的输出和结果。

## 来源

- 论文：https://arxiv.org/abs/2605.23282
- 官方代码：https://github.com/Duane245/DGNO
- 官方权重：https://huggingface.co/Duane245/DGNO

DGNO 官方代码采用 Academic Public License，仅限非商业学术研究；本分支保留了原始 `LICENSE`。
