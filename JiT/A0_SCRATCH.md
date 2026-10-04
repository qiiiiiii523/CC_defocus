# JiT A0 从头训练路线

分支：`jit-a0-scratch`，从 `jit-a0` 创建。保留 A0 架构、Flow + 可调
Charbonnier、恒等样本机制与采样过程；不增加 PSF、核掩膜或 A1 条件。
本地不安装训练环境，不启动训练。运行验证必须在 GPU 服务器完成。

## 初始化与恢复

- `--init_mode scratch`：构建网络，使用官方架构的随机初始化，再把状态
  patch 编码器的随机参数复制到模糊条件编码器。不共享参数；adapter 和
  gate 保持现有初始化，gate 从 0 开始。不会读取 `--pretrained`，即使
  默认官方权重文件不存在也不影响 scratch 初始化。
- `--init_mode pretrained`：仍使用官方权重初始化，行为与旧 A0 相同。
- 不指定初始化模式：新实验默认 pretrained，resume 则继承 checkpoint
  中记录的模式。旧 checkpoint 没有此字段时按 pretrained 处理。
- `--resume`：严格加载已有模型、EMA、优化器、epoch及已记录的更新步数。
  不重新复制条件编码器。显式指定与已保存模式冲突的 init_mode 会报错。
  训练恢复需重复原来的损失、identity、EMA等配置；不允许偷偷更换训练清单。
  resume 仍需复用原学习率调度、batch、总预算等配置，不是只给一个路径就够。

所有可训练参数交给 AdamW，未冻结主干。固定 null 类别保留，但 scratch
里的该 embedding 也是随机初始化，不是“预训练自然图像表征”。官方零
输出头与 adaLN 初始化保留；门控/分支梯度可能在早期逐步激活，需看日志
与实际恢复图，不能只依据 loss 下降判断条件有效。

## 数据规模与边界

默认仍读取 `prepared/manifests/debug_train.jsonl`（2000对）及
`debug_val.jsonl`（300对）。小样本模式只是从2000对取16/32对训练，验证
仍是300对，并不自动生成训练集上的过拟合评价。

完整训练时指定新的 JSONL `--train_manifest`，并设置
`--expected_train_pairs 0`（接受任意非空数量），或提供明确的期望数量。
验证规模同理使用 `--expected_val_pairs`，默认300。

`prepared/images/3DHistech/3D_label_train.txt`已检查到前几行为 PNG 文件名，
不是当前所需 JSONL 配对协议，因此不能直接传给 `--train_manifest`。
本次不自动转换或重划数据。后续需确认清晰/模糊对应路径、quality_status、
split，并转换成公共 JSONL 格式。

复用数据集原有同步裁剪/翻转、路径、质量及文件检查。入口新增训练/验证
sample_id 和图像路径交叉检查，但这不能识别同一切片的相邻裁块泄漏。
完整数据需人工审查按患者/切片/视野分组的划分，不能默认原300对仍独立。

## GPU 服务器启动（本次没有执行）

上传修改后的 `main_restoration.py`、`model_jit.py`、
`engine_restoration.py`和 `train_a0_scratch.sh`，保留其余完整 JiT 文件。

```bash
cd /home/qht/大创/CC_defocus/JiT
conda activate jit-a0
bash train_a0_scratch.sh 0 overfit16
# 小样本检查之后，另开2000对实验：
bash train_a0_scratch.sh 0 debug2000
```

脚本前台运行、输出写入独立 train.log；用另一个终端 tail -f 查看。
不是自动先后启动两组。脚本拒绝覆盖已有日志/checkpoint。
可在服务器自行用 nohup 启动脚本；本地不会执行它。

默认试验 LR=1e-4、warmup=5、clip_grad=1；它们是稳定性小试的起点，
不是已验证最优配置。小样本100 epoch、batch4、identity0；2000对200
epoch、batch8、identity10%。损失保持 lambda_pix=1，便于保留现有
公式，但 Charb 退化原因尚未定位，该值不是优选结论。手动命令可以
用 lambda_pix=0/0.1 做独立对照，务必换实验名。

脚本初期用普通权重验证（eval_weights=model），避免把慢EMA的滞后
混入初始化检查。之后用现有权重对比脚本分别评价普通/EMA1/EMA2。
比较预训练与scratch时应使用相同损失、划分、identity和推理权重类型；
同时报告步数、耗时及收敛趋势。不同LR/裁剪等配置不属于仅初始化单变量对照。

## 实验记录

`run_config.json`保存启动配置、初始化来源、数据数量及清单SHA256。
checkpoint中原有 `args`同时保存 init_mode、清单指纹和 global_step。
每epoch打印普通模型条件 gate与累计更新步数；TensorBoard记录gate。
不更改模型checkpoint键，不更改损失公式、不自动加载完整数据。

默认 scratch 名称和保存目录与旧 jit_a0 区分；若手动显式指定旧目录，
仍需自行避免覆盖旧结果。
