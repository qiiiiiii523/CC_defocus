# jit-a1-exp：合成真值 S0/S2 实验

日期：2026-10-14。本实验诊断正确局部条件经过当前融合是否有恢复收益；结果属于合成机制实验，不替代真实300对的正式成绩。

## 1. 本分支的功能

从 jit-a1 的 bdef0dc 创建 jit-a1-exp，复用原 A0/JiT 主干、RGB条件、Flow＋Charb、局部MLP和零门控。新增固定合成数据生成、真值读取及独立评价。正式真实数据入口仍保留；oracle_local只允许从 main_a1_exp.py 的合成数据入口启动。

| 组别 | 共同输入 | 局部条件 | 起点 |
|---|---|---|---|
| S0 / s0 / none | 固定合成模糊RGB | 关闭 | 原A0 EMA2 |
| S2 / s2 / oracle_local | 同一合成模糊RGB | 生成时实际使用的概率场 | 同一原A0 EMA2 |
| S1 / s1 / predicted_local，可选 | 同一合成模糊RGB | 冻结V3预测场 | 同一原A0 EMA2 |

S0保留RGB条件但没有局部支路。S2和可选S1使用完全相同的局部MLP、门控和图像token注入位置。V3权重不用于S0/S2。

三组新建optimizer和EMA，不恢复之前A1训练状态，不冻结A0主干。共同追加10轮，batch8、LR1e-4、warmup2、cosine、最低1e-6、Flow＋Charb（lambda_pix=1）、EMA2、BF16、Euler一步、评价seed0。identity_ratio统一为0，三核没有零模糊，避免制造错误真值。没有A2/A3重模糊训练损失。

生成代码是交接包中 local_synthetic.py / spatial_psf.py / psf.py 的原样副本。核为light=(sigma1,K7)、medium=(sigma2,K13)、heavy=(sigma3.5,K23)，reflect padding。64×64原生概率场双线性缩放到256×256后混合三幅整图高斯模糊；实际生成的64×64场保存为float32 NPZ。JiT保持原来64→16×16→MLP→gate注入。模糊图保存为RGB PNG，因此生成配置记录了uint8量化。

## 2. 本地提交与服务器克隆

代码必须先推送到远程，服务器才能克隆该分支。本次修改只包含本实验代码，不包含权重、生成数据、原evaluate.py及已有文档修改。若Codex已提交并推送，直接进入服务器步骤；若推送未成功，本机执行：

```powershell
git -c safe.directory=D:/大创 -C "D:\大创" push -u origin jit-a1-exp
```

服务器激活之前训练A0的环境，使用新的代码目录，复用原数据目录，无需复制图片或权重：

```bash
conda activate jit-a0
CODE_ROOT="/home/qht/大创/CC_defocus-a1-exp"
DATA_ROOT="/home/qht/大创/CC_defocus"
git clone --branch jit-a1-exp --single-branch https://github.com/qiiiiiii523/CC_defocus.git "$CODE_ROOT"
cd "$CODE_ROOT"
export A1_DATA_ROOT="$DATA_ROOT"
```

若这个新代码目录已存在且就是本分支的克隆，不再clone；进入目录执行git pull --ff-only即可。不要向正在训练的目录强制切换。也可在原项目目录切换分支训练：脚本会自动使用其所在代码目录；DATA_ROOT仍明确指向原数据根目录。

新入口的data_root只用于本合成实验，允许代码在新克隆、数据留在原目录；没有修改common_io的原项目路径约束，不需要worktree或图片软链接。

## 3. 服务器代码功能检查

本机仅完成语法和不导入Torch的数据协议测试，没有检查GPU/环境或运行模型。下面的模型测试在服务器执行；它们覆盖oracle接线、零门控、梯度和EMA，以及原有A1模式回归。

```bash
cd "$CODE_ROOT/JiT"
python -m unittest test_a1_synthetic_data test_a1_oracle_integration test_a1_integration test_a1_local_integration
cd "$CODE_ROOT"
```

## 4. 一次性生成固定合成数据

沿用原准备好的清晰图和train/val清单。train-count=0表示使用原清单全部66,976张清晰源图，val-count=300固定验证300张。若先做小规模机制筛选，可以把train-count改为5000；随后S0/S2必须共用这份固定数据，不能与真实旧分数直接混比。

```bash
mkdir -p "$DATA_ROOT/experiments/a1_oracle_synthetic"
nohup env CUDA_VISIBLE_DEVICES=0 python -u JiT/generate_a1_synthetic.py \
  --project-root "$DATA_ROOT" --train-count 0 --val-count 300 \
  --allow-group-overlap --device cuda \
  > "$DATA_ROOT/experiments/a1_oracle_synthetic/generation_v1.log" 2>&1 < /dev/null &
```

```bash
tail -f "$DATA_ROOT/experiments/a1_oracle_synthetic/generation_v1.log"
```

Ctrl+C只退出日志查看。日志最后应打印Complete及generation_config.json路径。生成代码拒绝覆盖非空data_v1；失败或想换生成规则时使用新的output-dir，并通过A1_EXP_DATA_DIR指向新数据，不删除或覆盖旧结果。

默认生成到原数据目录：

```text
CC_defocus/experiments/a1_oracle_synthetic/data_v1/
  generation_config.json
  blur/train/与blur/val/
  truth/train/与truth/val/
  manifests/synthetic_train.jsonl
  manifests/synthetic_val.jsonl
```

清晰图只引用原路径。每条清单记录实际真值文件、生成种子、source_sample_id及清晰/模糊/真值SHA256。开始训练时核对manifest和内容哈希。代码拒绝训练/验证重复ID或重复清晰路径。

allow-group-overlap是显式承认沿用原清单可能共享来源组，用于机制实验；生成配置会记录交集。若有独立来源组清单，传入train-manifest/val-manifest并省略这个参数。当前300张不能称为300个独立病例或新的独立测试集。仅提供概率场生成函数并不能独立证明V3训练时调用的分辨率等配置完全一致；64×64是本实验明确固定的设置。

## 5. 确认数据完成后，GPU 0/1分别训练S0/S2

```bash
cd "$CODE_ROOT"
export A1_DATA_ROOT="$DATA_ROOT"
export A1_EXP_TAG="$(date +%Y%m%d_%H%M%S)"
bash JiT/train_a1_exp.sh s0 0
bash JiT/train_a1_exp.sh s2 1
```

脚本自动后台运行，两组使用同一tag。每次打印准确实验名、日志及结果路径。两组主干均来自：

```text
/home/qht/大创/CC_defocus/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth
```

默认10轮的实际更新次数依所选合成训练样本数量而定；使用全部66,976对时共83,720次。训练只有最后一轮评价，两组一致。从构造模型之后重新固定训练随机数种子，避免新增适配器/估计器的初始化消耗改变后续随机序列；这是同协议控制，不承诺不同GPU完全逐位相同。

## 6. 结果怎么看和怎么打包

权重与配置位于新代码目录的checkpoints/<实验名>/；图片在outputs/<实验名>/；指标在results/<实验名>/。

```bash
cat "$CODE_ROOT/results/jit_a1_exp_s0_${A1_EXP_TAG}/summary.json"
cat "$CODE_ROOT/results/jit_a1_exp_s2_${A1_EXP_TAG}/summary.json"
```

确认两组checkpoints里的final_eval_config.json均为complete，summary中的evaluation_domain是synthetic_known_gaussian_mixture。condition_diagnostics.jsonl记录局部gate、梯度范数、局部/RGB残差RMS及其比值。零门控第一步MLP梯度可以为零，gate随后学习；负gate不代表接入错误。

两组完成后，在同一终端保留A1_EXP_TAG，执行：

```bash
cd "$CODE_ROOT"
tar --exclude='*.pth' --exclude='*.pt' --exclude='events.out.tfevents.*' \
  -czf "a1_exp_s0_s2_${A1_EXP_TAG}.tar.gz" \
  "checkpoints/jit_a1_exp_s0_${A1_EXP_TAG}" "checkpoints/jit_a1_exp_s2_${A1_EXP_TAG}" \
  "results/jit_a1_exp_s0_${A1_EXP_TAG}" "results/jit_a1_exp_s2_${A1_EXP_TAG}" \
  JiT/train_a1_exp.sh JiT/main_a1_exp.py
```

另取generation_config.json一并发回；它在原数据目录。不要把整套blur/truth或大权重打包。重新登录后，用实际目录后缀恢复A1_EXP_TAG。

S2优于S0支持当前融合在这个合成任务上能利用真值，但新增适配器容量仍是因素之一；可再对S2权重做真值/错配/均值/关闭干预验证信息作用。S2不优于S0时，仍要考虑条件价值、空间压缩、优化和预算，不能直接判定估计器有问题。

## 7. 需要时再补S1

S0/S2的结果值得继续时，再用同一dataset和同一tag启动S1；不要重新生成数据：

```bash
bash JiT/train_a1_exp.sh s1 0
```

S1读取原数据目录里的JiT/degradation/weights/local-estimator-v3.pt。权重SHA应为04cb85d6d59016e4a7e3cf2f83ce607bb771d0556c8bdb96c7fe6682178adc0b；冻结且FP32预测，S1/S2适配器初始值保持相同。S0/S2均无需该权重。

## 8. 可选：少量数据先检查全流程

不想一次生成全量时，先生成8张训练/2张验证到独立data_smoke，然后每组运行1轮只验证功能，不能把这次指标当正式结果。

```bash
CUDA_VISIBLE_DEVICES=0 python JiT/generate_a1_synthetic.py \
  --project-root "$DATA_ROOT" --output-dir "$DATA_ROOT/experiments/a1_oracle_synthetic/data_smoke" \
  --train-count 8 --val-count 2 --allow-group-overlap --device cuda
export A1_EXP_DATA_DIR="$DATA_ROOT/experiments/a1_oracle_synthetic/data_smoke"
export A1_EXP_EPOCHS=1 A1_EXP_WARMUP=0 A1_EXP_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
bash JiT/train_a1_exp.sh s0 0
bash JiT/train_a1_exp.sh s2 1
```

正式实验前恢复默认值并生成新tag：

```bash
unset A1_EXP_DATA_DIR A1_EXP_EPOCHS A1_EXP_WARMUP A1_EXP_TAG
```

## 9. 中断恢复与单独评价

使用main_a1_exp.py的resume参数，恢复对应S0或S2自己的checkpoint。保留原generation_config、清单、模式及训练配置，去掉a0_checkpoint；epochs表示总轮数。下面用S2演示，保持原A1_EXP_TAG：

```bash
cd "$CODE_ROOT/JiT"
CUDA_VISIBLE_DEVICES=1 python -u main_a1_exp.py \
  --resume "$CODE_ROOT/checkpoints/jit_a1_exp_s2_${A1_EXP_TAG}/checkpoint-last.pth" \
  --degradation_mode oracle_local --init_mode scratch --data_root "$DATA_ROOT" \
  --generation_config "$DATA_ROOT/experiments/a1_oracle_synthetic/data_v1/generation_config.json" \
  --train_manifest "$DATA_ROOT/experiments/a1_oracle_synthetic/data_v1/manifests/synthetic_train.jsonl" \
  --val_manifest "$DATA_ROOT/experiments/a1_oracle_synthetic/data_v1/manifests/synthetic_val.jsonl" \
  --epochs 10 --amp_bf16 --run_metrics \
  --model_name "jit_a1_exp_s2_${A1_EXP_TAG}" \
  --output_dir "$CODE_ROOT/checkpoints/jit_a1_exp_s2_${A1_EXP_TAG}" \
  --prediction_root "$CODE_ROOT/outputs" --results_root "$CODE_ROOT/results"
```

默认参数与启动脚本一致；若曾改参数，必须同步重复。此入口沿用已有checkpoint格式恢复optimizer/EMA/轮数，没有新增随机状态快照，不能声称中断恢复与不中断训练逐位一致。若只重新生成评价，在该命令加eval_only即可；S2评价必须使用对应合成真值，不能用于真实图部署。
