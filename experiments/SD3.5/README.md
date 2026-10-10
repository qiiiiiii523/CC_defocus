# SD3.5 scratch 训练交接包

这是成员 B 交给队长的独立交接目录。它只服务于“SD3.5 Medium Transformer 随机初始化、官方 VAE 冻结、模糊图条件恢复”的新版本，不覆盖 JiT A0，也不覆盖旧 ControlNet/B2 实验。

## 结论先看

- **现在不需要租卡**：整理、审查和交接材料已可在本地完成。
- **队长训练前需要租卡**：先做单 batch 冒烟测试，再做 4–8 对过拟合测试，最后决定是否启动 2000/300 训练或更大规模训练。
- **当前包不是已经训练好的模型**：本包提供固定清单、统一评价程序、已有负面结果和新方案说明；新 scratch 主干仍需在 GPU 上实现/验证后训练。

## 本次方案

1. 不使用 ControlNet。
2. SD3.5 Transformer 不加载预训练权重，按官方配置随机初始化。
3. 使用官方预训练 VAE，并冻结 VAE 参数。
4. 将模糊图编码为潜变量，作为空间条件；当前生成状态与模糊潜变量通过一个可训练的 3×3 条件卷积融合后送入 Transformer。
5. 训练目标为潜空间条件 Flow Matching：从“模糊潜变量+噪声”走向清晰潜变量。
6. 推理时只读取模糊图，不能读取清晰参考图。
7. 评价沿用公共 `code/evaluate.py`，固定 RGB、无裁剪缩放、PSNR/SSIM/LPIPS-Alex。

## 目录

- `configs/`：scratch 训练示例配置。
- `code/model.py`：随机初始化 Transformer 和模糊潜变量条件融合。
- `code/flow_matching.py`：训练状态与目标构造。
- `code/smoke_test.py`：GPU 前向、反向和条件敏感性检查。
- `code/train_scratch.py`：训练入口骨架，正式训练前需接入仓库已有配对读取器。
- `docs/交接清单.md`：队长执行顺序和验收项。
- `reports/`：本方案审查记录和必要背景说明。

仓库根目录已经有公共 `common_io.py`、`evaluate.py` 和固定清单；本分支不重复复制。固定清单继续使用仓库现有的 `prepared/manifests/debug_train.jsonl` 与 `prepared/manifests/debug_val.jsonl`。

代码是未训练版本。`train_scratch.py` 会先保存配置并明确提示队长接入现有配对读取器，避免误用清晰参考或重新划分数据；不能把它当作已经跑通的训练结果。

## 固定数据校验

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `debug_train.jsonl` | 2000 | `EF5AECFCABC40590AD4348CF0DD8EB9B310B90B9D946A878C7561923909D195B` |
| `debug_val.jsonl` | 300 | `D5D7EC0863008D71A13C173AE064EBB3C25B71291FB2BF5578D0BC09E07D411B` |

不要重新划分、排序或混用清单。`rfn.jsonl` 是合并清单，不能直接替代上述两个固定清单。

## 队长建议执行顺序

1. 在有 GPU 的环境安装 `requirements.txt`，确认 `diffusers`、`torch` 和官方 SD3.5 Medium 权限/权重可用。
2. 先只加载 VAE 和 Transformer 配置，确认 **没有调用 Transformer.from_pretrained**；Transformer 必须随机初始化。
3. 跑单 batch 前向、反向和条件敏感性检查：损失有限、Transformer 有梯度、VAE 无梯度。
4. 用 4–8 对样本做小样本过拟合；若不能下降或恢复图不随模糊输入改变，先修实现，不启动正式训练。
5. 通过后用固定 2000/300 清单训练和评价，与 JiT A0 使用同一 `evaluate.py`、同一 300 对验证集。
6. 记录 checkpoint、配置、随机种子、训练/推理耗时、显存、PSNR、SSIM、LPIPS 和结构预览。

## 资源估计

未在本机完成 GPU 运行验证。SD3.5 Medium 随机主干训练预计至少需要 48 GB 显存并启用 BF16、gradient checkpointing、梯度累积；80 GB 更稳妥。16–24 GB 不建议直接尝试正式训练。实际显存以队长的单 batch 冒烟结果为准。

## 目前已验证与未验证

已验证：固定清单及哈希、公共评价协议、旧 B2 结果归档、官方 VAE 小样本结构检查、现有代码审查。

未验证：新的随机初始化 Transformer 条件通路、训练反向、推理采样、GPU 显存和正式指标。不要把本包写成“scratch 已训练成功”。

## 与 JiT 的边界

本实验用于检验 SD3.5 潜空间路线是否能学会模糊到清晰的对应关系。SD3.5 与 JiT 的架构和表示空间不同，指标只能作为同一数据与评价协议下的工程比较，不能单独归因于某一个结构因素。
