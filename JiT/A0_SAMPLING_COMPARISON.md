# A0 固定checkpoint采样步数对照

本次只新增compare_a0_sampling.py；没有修改JiT、Denoiser、训练loss、
t_eps或checkpoint。没有本地启动推理/训练，运行功能须在服务器验证。

## 实验协议

默认EMA2、Heun、1/10/20/50步、固定300对验证清单、eval_seed=0。
每组调用现有main_restoration.py的eval_only入口。初始噪声沿用
engine_restoration.py按sample_id与seed生成的实现，不因步数改变。
各步数均单独加载同一checkpoint，保存到独立outputs/results目录。

1步时generate没有中间Heun步骤，只执行最终Euler从0到1；其输出等于
t=0的清晰终点预测（有浮点舍入误差）。它是诊断对照，不代表完整流的
多步积分已被验证或单步恢复必然更好。

保持模型参数、权重来源、输入处理、BF16选择、噪声尺度、t_eps与评分
程序相同。cfg不构成这次变量，A0保留固定null标签。晚期分母截断仍为
0.05：本次不同时调整数值方案，以免混淆步数的影响。

## 服务器命令

只需把新增Python脚本上传到完整项目的JiT目录；激活现有环境。

```bash
conda activate jit-a0
CUDA_VISIBLE_DEVICES=0 python -u /home/qht/大创/CC_defocus/JiT/compare_a0_sampling.py \
  --gpu 0 \
  --checkpoint /home/qht/大创/CC_defocus/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth \
  --weights ema2 --method heun --steps 1 10 20 50 \
  --amp_bf16 --eval_batch_size 4 --num_workers 4 --seed 0
```

runner仅需Python标准库，但子进程会运行JiT，必须使用服务器的深度学习
环境和GPU。不要在缺依赖的本地启动。运行期间不要续训并覆盖该checkpoint。

## 输出

名称前缀默认是
`jit_a0_scratch_full66976_charb1_e20_sampling_ema2_heun`：

- outputs/<前缀>_s1、_s10、_s20、_s50：各300张恢复图。
- results/<前缀>_sN：逐图评分与公共summary.json。
- reports/<前缀>/sampling_summary.csv：四组汇总均值、标准差。
- reports/<前缀>/comparison_config.json：配置、清单指纹、运行状态。
- reports/<前缀>/steps_N.log：每组日志，失败时查看。

脚本拒绝复用已有报告/结果/图片目录，防止覆盖或混入旧结果。重跑用
--experiment_name设置新名称，不删除已有结果。失败时配置会记录已经
完成的组和当前组；本脚本没有自动断点续跑。

采样后的统计以公共保存/评价协议为准。total_wall_seconds包含模型加载、
编译、生成、保存和指标计算，不是纯采样耗时。不要把它直接写成模型每图
推理速度。现有evaluate.py要求300条val，脚本预先检查数量及ID唯一性。

可另用--method euler做独立对照，其目录也独立；不要把两种求解器与
步数同时变更的结果当作单变量对照。第一轮先只比较Heun标签下四种步数。
结果应同时看PSNR/SSIM/LPIPS和核形态，不根据单一指标自动选优。
