# jit-a1-exp：2000/300快速排查，原目录完成三组实验

更新：2026-10-14。服务器直接使用 `/home/qht/大创/CC_defocus`，已切换 jit-a1-exp；不再克隆新目录。本轮先做2000张训练/300张验证的快速机制排查，三组都做，共用一次生成的合成数据、同一原A0 EMA2起点和训练预算。

## 1. 更新代码并固定公共路径

在原训练环境、同一终端执行：

```bash
cd "/home/qht/大创/CC_defocus"
git pull --ff-only
conda activate jit-a0
ROOT="/home/qht/大创/CC_defocus"
export A1_DATA_ROOT="$ROOT"
export A0_CKPT="$ROOT/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth"
export LOCAL_ESTIMATOR="$ROOT/JiT/degradation/weights/local-estimator-v3.pt"
unset A1_EXP_DATA_DIR A1_EXP_EPOCHS A1_EXP_WARMUP A1_EXP_TAG
export A1_EXP_DATA_DIR="$ROOT/experiments/a1_oracle_synthetic/data_2000"
```

不覆盖原权重或结果。若git pull因本地代码修改发生冲突，保留并处理这些修改，不使用reset --hard或强制覆盖。本地已有的生成代码现已包含在分支里，不需要另外上传交接包；V3权重仍是原目录中已有的二进制文件。

## 2. 重新执行代码功能测试

你之前的17项测试中有4项报错：三项是小模型CPU测试输入与CUDA RoPE张量不一致，另一项引用了服务器不存在的旧local-estimator.pt。更新修复仅限测试：显式把小模型的RoPE张量移到CPU，并改为使用现有local-estimator-v3.pt。正式训练网络和RoPE实现未改变。

```bash
cd "$ROOT/JiT"
python -m unittest test_a1_synthetic_data test_a1_oracle_integration test_a1_integration test_a1_local_integration
cd "$ROOT"
```

应看到OK再继续。如果仍报错，把完整输出发回；本机没有运行Torch模型，不能把静态修复等同于服务器模型测试已通过。这些测试检查接线、梯度、冻结和EMA，不检验实验分数。

## 3. 一次性生成三组共用的合成数据

如果data_2000已经生成完成，直接进入下一节，不要重新生成。若尚未生成，执行：

```bash
mkdir -p "$ROOT/experiments/a1_oracle_synthetic"
nohup env CUDA_VISIBLE_DEVICES=0 python -u JiT/generate_a1_synthetic.py \
  --project-root "$ROOT" --train-count 2000 --val-count 300 \
  --train-manifest "$ROOT/prepared/manifests/debug_train.jsonl" \
  --val-manifest "$ROOT/prepared/manifests/debug_val.jsonl" \
  --allow-group-overlap --device cuda \
  > "$ROOT/experiments/a1_oracle_synthetic/generation_2000.log" 2>&1 < /dev/null &
```

```bash
tail -f "$ROOT/experiments/a1_oracle_synthetic/generation_2000.log"
```

Ctrl+C只退出日志查看，生成进程继续。等待日志最后出现Complete，然后确认：

```bash
python -c 'import json; c=json.load(open("experiments/a1_oracle_synthetic/data_2000/generation_config.json")); print(c["status"], c["train_count"], c["val_count"]); assert c["status"]=="complete"'
```

本命令使用debug_train.jsonl的2000张清晰源图和debug_val.jsonl的300张源图，生成固定的新合成模糊图。已核对本地两份清单的数量和split；服务器以实际清单及生成配置为准。使用debug清单比从全量另抽2000张更贴近队友交接的源图范围，但她的实际清单哈希尚未提供，不能仅凭文件名认定全部生成协议一致。不能用旧真实模糊图的成绩代替S0/S1。

```text
/home/qht/大创/CC_defocus/experiments/a1_oracle_synthetic/data_2000/
  generation_config.json
  blur/train/、blur/val/
  truth/train/、truth/val/
  manifests/synthetic_train.jsonl
  manifests/synthetic_val.jsonl
```

代码拒绝覆盖非空data_2000。生成中途失败时先分析日志；确需重生成可用新的output-dir，再把A1_EXP_DATA_DIR设为新目录，三组同时使用它。不要删旧数据或强行重跑覆盖。

## 4. 固定三组实验标签

```bash
cd "$ROOT"
export A1_EXP_TAG="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/checkpoints"
printf '%s\n' "$A1_EXP_TAG" > "$ROOT/checkpoints/a1_exp_three_arm_tag.txt"
echo "$A1_EXP_TAG"
```

三组都沿用这个tag，启动S2时不要重新生成tag。若重新登录，恢复公共路径和标签：

```bash
ROOT="/home/qht/大创/CC_defocus"
cd "$ROOT"
export A1_DATA_ROOT="$ROOT"
export A1_EXP_TAG="$(cat "$ROOT/checkpoints/a1_exp_three_arm_tag.txt")"
```

若本轮用了自定义A1_EXP_DATA_DIR、A0_CKPT或LOCAL_ESTIMATOR，重登录后也恢复其原值。

## 5. 先用GPU 0/1启动S0和S1

```bash
cd "$ROOT"
bash JiT/train_a1_exp.sh s0 0
bash JiT/train_a1_exp.sh s1 1
```

脚本自动后台启动，打印实验名和日志位置。

| 组别 | 条件 | 本轮安排 |
|---|---|---|
| S0 | 只有合成模糊RGB，关闭局部分支 | GPU 0先运行 |
| S1 | 同一RGB＋冻结V3的预测概率图 | GPU 1同时运行 |
| S2 | 同一RGB＋实际生成真值概率图 | S0完成后用GPU 0运行 |

```bash
tail -f "$ROOT/checkpoints/jit_a1_exp_s0_${A1_EXP_TAG}/train.log"
```

查看S1只需把s0换成s1。Ctrl+C只退出日志查看。

## 6. S0完成后，GPU 0运行S2

S0最终评价完成且训练进程已结束后，再启动S2；不要在GPU 0同时运行S0和S2。S0日志最后应输出训练时间，且下面的记录为complete：

```bash
cat "$ROOT/checkpoints/jit_a1_exp_s0_${A1_EXP_TAG}/final_eval_config.json"
```

保持原tag、数据和配置执行：

```bash
cd "$ROOT"
bash JiT/train_a1_exp.sh s2 0
```

S2本轮一定做，不取决于S0/S1的分数；这里只是将第三组安排到空闲卡。

## 7. 三组共同配置及生成规则

| 项目 | 设置 |
|---|---|
| 起点 | 原始A0 checkpoint的EMA2，三组相同 |
| 训练状态 | 新建optimizer与EMA，追加10轮；不resume现有A1 |
| 模型 | JiT-B/16，256×256，A0主干继续学习 |
| 学习率 | 1e-4、warmup2、cosine、最低1e-6 |
| batch / 随机种子 | 8 / 0 |
| 损失 | Flow＋Charb，lambda_pix=1，epsilon=1e-3 |
| identity_ratio | 0；三核没有零模糊，避免错误真值标签 |
| 评价 | EMA2、Euler一步、seed0、BF16；仅最后一轮评价 |
| S1/S2融合 | 同一局部MLP、零初始化门控和图像token注入 |

本轮每组2000张、batch8、10轮，共2500次更新。三组必须使用相同样本和更新次数，实际数量以generation_config和run_config为准。这是快速筛查：若无收益，还需结合损失、门控与梯度考虑适配训练是否充分，不能直接排除融合能力。此前全量data_v1及其结果保留，不与本轮混合比较。以后扩大规模时需为三组一起换固定数据和预算。

生成函数来自交接包。light/medium/heavy对应高斯核(sigma1,K7)/(sigma2,K13)/(sigma3.5,K23)，reflect边界。64×64原生概率场缩放到256×256后混合三幅高斯模糊结果；真值原场保存为float32 NPZ。RGB PNG有uint8量化，配置明确记录。条件仍按64→16×16→MLP→gate注入，不更改JiT主干或新增A2/A3损失。

S1从当前实际模糊RGB预测，估计器冻结、eval且FP32。V3 SHA256应为04cb85d6d59016e4a7e3cf2f83ce607bb771d0556c8bdb96c7fe6682178adc0b。S2不经过估计器，直接读取实际生成场；训练时图像和真值同步水平翻转，不进行未定义的裁剪或缩放。所有文件及清单记录SHA256，开始训练时验证。

allow-group-overlap明确将沿用原来源划分的实验标为机制实验，来源组交集会保存。若有独立来源组清单，传给生成入口并取消此选项。当前300张不等于300个独立病例，合成实验也不是实拍PSF真值评价。

## 8. 查看三个结果

```bash
cat "$ROOT/results/jit_a1_exp_s0_${A1_EXP_TAG}/summary.json"
cat "$ROOT/results/jit_a1_exp_s1_${A1_EXP_TAG}/summary.json"
cat "$ROOT/results/jit_a1_exp_s2_${A1_EXP_TAG}/summary.json"
```

三组的final_eval_config.json都应为complete。summary应标明synthetic_known_gaussian_mixture。checkpoint目录还保存run_config、git版本、日志及condition_diagnostics.jsonl，后者记录gate、局部梯度及局部/RGB注入RMS。

| 对照 | 回答的问题 |
|---|---|
| S2－S0 | 正确生成条件通过当前模型是否带来净收益 |
| S1－S0 | 当前V3预测条件是否带来净收益 |
| S2－S1 | 同一融合结构下，预测条件与真值条件相差多少 |

S2明显好于S0而S1无收益时，优先检查预测图与真值的差距及表达。S1/S2均有收益、真实任务无收益时，进一步查域差异。两者均无收益时，还需考虑条件价值、空间压缩、优化和预算，不能单独归责估计器。S2相比S0多了适配器容量，有收益后可对S2做真值/错配/均值/关闭干预确认条件内容的作用。

## 9. 三组完成后的tar命令

```bash
cd "$ROOT"
tar --exclude='*.pth' --exclude='*.pt' --exclude='events.out.tfevents.*' \
  -czf "a1_exp_three_${A1_EXP_TAG}.tar.gz" \
  checkpoints/jit_a1_exp_{s0,s1,s2}_"${A1_EXP_TAG}" \
  results/jit_a1_exp_{s0,s1,s2}_"${A1_EXP_TAG}" \
  experiments/a1_oracle_synthetic/data_2000/generation_config.json \
  experiments/a1_oracle_synthetic/data_2000/manifests \
  JiT/train_a1_exp.sh JiT/main_a1_exp.py
```

下载这个tar.gz用于三组对比分析。无需打包权重或全部合成图。如果用了自定义数据目录，把最后的generation_config和manifests路径改为对应目录。

## 10. 可选：少量数据检查或中断恢复

若想先验证全流程，另用output-dir生成8张训练/2张验证，然后给三组统一设置A1_EXP_DATA_DIR、A1_EXP_EPOCHS=1、A1_EXP_WARMUP=0及单独smoke标签。它只检查功能，不能混入正式成绩；正式三组开始前unset这些覆盖变量。

中断恢复使用main_a1_exp.py的resume，恢复各组自己的checkpoint，去掉a0_checkpoint。S0模式为none、S1为predicted_local、S2为oracle_local；S1恢复不再传外部degradation_checkpoint，使用checkpoint中的冻结估计器。保持原数据、generation_config、总epochs=10及其他配置，epochs不是再追加10轮。下面以S2为例：

```bash
cd "$ROOT/JiT"
CUDA_VISIBLE_DEVICES=0 python -u main_a1_exp.py \
  --resume "$ROOT/checkpoints/jit_a1_exp_s2_${A1_EXP_TAG}/checkpoint-last.pth" \
  --degradation_mode oracle_local --init_mode scratch --data_root "$ROOT" \
  --generation_config "$ROOT/experiments/a1_oracle_synthetic/data_2000/generation_config.json" \
  --train_manifest "$ROOT/experiments/a1_oracle_synthetic/data_2000/manifests/synthetic_train.jsonl" \
  --val_manifest "$ROOT/experiments/a1_oracle_synthetic/data_2000/manifests/synthetic_val.jsonl" \
  --epochs 10 --amp_bf16 --run_metrics \
  --model_name "jit_a1_exp_s2_${A1_EXP_TAG}" \
  --output_dir "$ROOT/checkpoints/jit_a1_exp_s2_${A1_EXP_TAG}" \
  --prediction_root "$ROOT/outputs" --results_root "$ROOT/results"
```

如果只重做评价，加eval_only。恢复沿用原checkpoint格式，没有额外保存随机状态，不能声称与不中断训练逐位一致。S2推理必须有对应合成真值，不能用于未知真实模糊图部署。
