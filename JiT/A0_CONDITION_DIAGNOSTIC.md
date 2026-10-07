# A0 模糊图条件敏感性检查

这个脚本只诊断已有checkpoint，不训练、不更新模型或EMA，不改造网络。
只加载可信项目checkpoint。普通/EMA1/EMA2严格加载对应参数，配置优先取
checkpoint中的args。默认关闭torch.compile以避免诊断编译等待。

## 两种检查

1. **固定起始噪声，完整ODE恢复**：每个sample_id固定一份噪声，对正确
   模糊图、循环错配的另一张图、RGB灰色、RGB白色、关闭条件分支分别采样。
   各轨迹会自行分化，不是每一步都强行保持相同状态。
2. **固定中间状态，单次终点预测**：用清晰参考构造相同的z_t，在多个t
   下只改变条件，比较网络预测。这是离线teacher-forced诊断，清晰参考
   不是部署输入，误差低也不能等同于真实采样效果好。t=0状态只含噪声；
   晚期状态已含大量答案，预测更准是预期现象，不能据此判定条件充分。

灰图在JiT范围里是全0，但仍经过有偏置的编码器；它不等于移除条件。
branch_off使用临时forward hook把条件编码器输出置零，不改gate数值，
结束后移除hook。错配/常量/关闭条件都是诊断干预，不是正常训练分布。

## 服务器启动（本地未运行）

将diagnose_a0_condition.py放入服务器JiT目录，保留完整项目及依赖。

```bash
conda activate jit-a0
CUDA_VISIBLE_DEVICES=0 python -u /home/qht/大创/CC_defocus/JiT/diagnose_a0_condition.py \
  --checkpoint /home/qht/大创/CC_defocus/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth \
  --weights ema2 --device cuda --amp_bf16 --num_samples 8 \
  --output_dir /home/qht/大创/CC_defocus/reports/a0_full_e20_condition_check
```

用--sample_ids指定至少两个不同ID，可针对之前的失败样本。默认在验证
清单按等距位置取8张，错配条件为选中列表的下一张，最后一张回到第一张。
没有根据评分选择样本。默认采样步数沿用checkpoint；可用--steps覆盖，
但应先保留官方实验使用的50步。

CPU理论上可运行：--device cpu，不加--amp_bf16；仍必须安装PyTorch、
NumPy、Pillow、einops等项目依赖。CPU多次ODE会慢得多，优先服务器GPU。
本地无环境时不能直接运行。本次不安装环境、不启动脚本或训练。

## 输出与解释

- panels/：8格图，含blur、clear、donor及5种恢复；保存PNG时裁剪至[0,1]。
- images/：各输入/输出原尺寸PNG。
- generation_metrics.csv：完整采样条件变化对输出的影响，以及对清晰图的
  MAE/MSE和越界比例。统计在保存前未裁剪的FP32 RGB尺度上计算。
- fixed_state_metrics.csv：相同z_t不同条件在各时间点的预测差异和误差。
- summary.json：条件gate、权重来源、模型/采样配置、清单指纹及均值。

输出目录必须为空，防止混入旧结果。重跑使用新目录；脚本不覆盖checkpoint。
生成表correct的output_mae_vs_correct=0是定义，不是恢复误差为0。

正确/错配输出高度相似，只提示条件响应不足；先排除错配图本身太相近。
明显变化只证明敏感，不保证忠实恢复。正确条件对参考的误差应结合错配
和关闭条件比较，并查看图像。常量输入异常不能单独证明编码器有问题。
无通用“差异大于多少就算通过”的阈值，不自动给网络下合格结论。

如果真实采样误差大，而某些固定状态预测较准，可提示需要查采样/时间段；
但这不是确定诊断，因为两者状态分布不同。该脚本不算SSIM/LPIPS，不替代
公共evaluate.py，不将诊断子集结果作为正式模型排名。
