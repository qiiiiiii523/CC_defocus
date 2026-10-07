# JiT A1：退化条件整合说明

整合分支：jit-a1。来源：origin/a1-degradation 的 JiT/degradation，
选择性引入模块，没有覆盖 C 分支中旧的 A0 主干、数据入口及损失代码。
本地不启动训练/模型；语法检查不代表服务器运行测试已经通过。

## 三个实验模式

| degradation_mode | 条件 | 训练对象 |
|---|---|---|
| none | 只有原有 RGB 模糊空间条件 | 原 A0 主干及 RGB 分支 |
| fixed | 上述条件 + [1,0,0] 固定轻度概率 | 主干、RGB 分支、新适配器 |
| predicted | 上述条件 + C 估计器预测的轻/中/重概率 | 同 fixed，估计器完全冻结 |

这些是一个代码入口下的三组对照，不是三个 Git 分支，也不是 model/EMA1/EMA2。
固定条件组控制“新增可训练调制模块”的影响；A0-continued 控制额外训练预算。
不得仅将训练后的 A1 与未继续训练的 20 轮 A0 比较。

## 网络和数据流

保留 z_t、t、null 类别和模糊 RGB 空间分支。
新增 net.degradation_condition：
Linear(3,hidden) → SiLU → Linear(hidden,hidden)，乘可训练零初始化 gate。
其输出加到原有时间 + null 类别调制向量，各 block 和 final_layer 都读取该向量。
没有改变 patch 编码、图像输出头或 flow 轨迹，没有加入 PSF loss。

predicted 模式将本批实际 blur 从 [-1,1] 转为 [0,1]，
在 FP32/no_grad 下估计概率，适用于同步裁剪、翻转及恒等替换后的输入。
不能读取 clear、核掩膜或真实 PSF 标签。
每次训练 forward 估计一次；每次 generate 估计一次并供全部 ODE 调用复用。
估计器是全局三分类模型，不是局部模糊图/核掩膜模型。
Denoiser.train() 保持估计器 eval，冻结 BatchNorm 统计及全部参数。
checkpoint 包含估计器参数和 buffers，EMA 不平滑冻结参数。

## 从 A0-v1 开始三组新训练

使用 --a0_checkpoint 指定 A0 文件或含 checkpoint-last.pth 的目录。
--a0_weights ema2（默认）读取 A0 的 model_ema2。
三组必须选同一权重来源；新增层之外的全部 A0 键必须匹配，
加载不重新复制 RGB 分支、不覆盖它已训练的 gate。
旧优化器、epoch 和旧 EMA 不恢复：三组重建 optimizer/EMA，epoch 从 0 开始。
--epochs 是本轮额外训练预算。--resume 则是恢复同一实验，不是 A0→A1 转换。
两个参数互斥。不要把 --eval_weights ema2 误认为训练从 EMA2 开始。

推荐共同参数（只是配置示意，本文件未执行）：

    --a0_checkpoint /home/qht/大创/CC_defocus/checkpoints/jit_a0_scratch_full66976_charb1_e20/checkpoint-last.pth
    --a0_weights ema2
    --train_manifest /home/qht/大创/CC_defocus/prepared/manifests/3dh_full_train.jsonl
    --expected_train_pairs 66976
    --val_manifest /home/qht/大创/CC_defocus/prepared/manifests/debug_val.jsonl
    --expected_val_pairs 300
    --lambda_pix 1 --charbonnier_eps 1e-3 --identity_ratio 0.10
    --eval_weights ema2 --sampling_method euler --num_sampling_steps 1

训练学习率/额外 epochs、batch、warmup、seed 等须三组统一显式指定。
新增模式不是重新从头训练，init_mode 继承 A0 的 scratch 来源。
默认清单仍是 2000 对，因此完整训练必须指定以上 full manifest 和数量。
每组使用不同 model_name/output_dir；严禁写入 A0-v1 归档目录。

模式分别指定：

    --degradation_mode none
    --degradation_mode fixed
    --degradation_mode predicted --degradation_checkpoint /绝对路径/estimator_a1_3dh_pseudo.pt

估计器权重需从交接包单独放到服务器；未将二进制权重写入 Git。
推荐的是 pseudo 权重，不是 CNSeg 合成权重；其真实近似标签准确率未超过固定轻度，
因此预测条件的收益尚未证明。A1 本轮就是验证该收益，不应宣称是真实光学 PSF。

## 恢复及评价

同一实验中断：--resume 新实验/checkpoint-last.pth，并重复原训练/损失参数。
恢复自动继承 degradation_mode；显式给错模式会拒绝。
估计器从 A1 checkpoint 恢复，不再传 degradation_checkpoint。
评价使用 --eval_only --resume 新实验/checkpoint-last.pth
及 --eval_weights ema2 --sampling_method euler --num_sampling_steps 1。
新条件适配器随主干一起维护两套 EMA。
输出 PNG、固定 per-sample seed、公共 evaluate.py 路径不变。

run_config.json 记录模式、估计器路径/哈希、A0 起点/来源/哈希、
训练验证清单哈希；每轮日志/TensorBoard 记录退化 gate。
本轮保留 Flow+Charbonnier，暂不加入 PSF、核结构或特征损失。

## 文件职责及验证边界

- model_jit.py：退化适配器及全层调制。
- denoiser.py：冻结估计器、输入转换、三模式、采样条件缓存、兼容 A0 加载。
- main_restoration.py：参数、A0 起点、新实验/断点区分、配置保存。
- engine_restoration.py：原训练/评价循环，新增 gate 记录。
- degradation/：C 原始代码和交接文档；PSF 工具保留但 A1 未调用其 loss。
- test_a1_integration.py：轻量 CPU 接线检查（替代小主干，无完整 JiT 运行）。

服务器环境可先执行 python -m unittest test_a1_integration；
它检查零 gate 等价、一步终点、严格加载、冻结 BN、概率归一化及采样只估计一次。
它不能代替真实 JiT 的 GPU 检查。正式训练前应在同一噪声下比较 A0 与新零 gate A1，
再小批 forward/backward 和一张图生成；此处没有启动这些操作。
