# JiT A1-local 接入说明

日期：2026-10-11。接入分支：`jit-a1`；C 模块来源：`a1-degradation` 的 `8ee33c940052507cabaf72e37f73ae63550a3b4c`。

2026-10-11 更新：新实验采用 C 的 3.0 权重 `local-estimator-v3.pt`。
网络结构、局部输出和 token 注入方式不变，仅增加权重格式兼容并更新实验配置。

## 本轮范围

本轮仅做局部 A1 条件接入，保留原 Flow + Charbonnier 目标。引入 C 的
`degradation/local_model.py`、`degradation/local_condition.py`，新增严格加载器
`degradation/local_inference.py`；不引入 `spatial_psf.py`、局部重模糊损失或其训练脚本。
原分支已有的全局 PSF 工具仍为旧实验辅助代码，A1 训练不调用它们。

C 已训练好退化估计器；本轮训练 JiT 主干、原 RGB 模糊分支和局部适配器。
估计器保持 FP32、eval、no_grad、requires_grad=False。A0 固定指保存基线，
不表示后续恢复训练冻结 JiT 主干。

## 接线

```text
实际模糊输入 [-1,1]
  → [0,1] → 冻结 LocalBlurEstimator → softmax
  → [B,3,ceil(H/4),ceil(W/4)] 轻/中/重概率图
  → 双线性缩放到图像 patch 网格
  → 逐位置 MLP → 零初始化可学习门控
  → 加到原状态 token + 位置编码 + RGB 模糊条件
  → Transformer → 清晰图预测
```

256×256 的 JiT-B/16 对应 16×16 网格、256 个图像 token。概率图按
`(y,x)` 行优先展开，与 `BottleneckPatchEmbed.flatten(2)` 一致。注入发生在
Transformer 入口、类别 context token 插入前；保留局部差异，不做空间平均。
新模块参数前缀为 `net.local_degradation_condition.`，零门控使加载 A0 后的
初始输出与 A0 一致，原 RGB 分支的已训练门控不会重新初始化。

条件只依赖本批同步裁剪、翻转、恒等替换后的实际 blur，不读取目标图或退化标签。
每个训练 forward 估计一次；每个 generate 估计一次、供所有 ODE 调用复用。

## 模式及兼容性

| 模式 | 条件路径 | 权重 |
|---|---|---|
| `none` | 原 A0 RGB 条件 | 不需要估计器 |
| `fixed_local` | 恒定轻度概率图，经新的局部 token 适配器 | 不需要估计器 |
| `predicted_local` | C 局部预测图，经局部 token 适配器 | 新实验用 `local-estimator-v3.pt`；兼容旧 `local-estimator.pt` |
| `fixed` / `predicted` | 旧全局向量、时间/类别调制 | 保留原全局权重及断点行为 |

新局部模式需要显式选择；默认仍为 `none`。不能把局部权重传给旧 `predicted`。
首次接入用 `--a0_checkpoint ... --a0_weights ema2`，新建 optimizer/EMA，
额外 epoch 从 0 开始。恢复同一 A1-local 实验用 `--resume ...`，自动继承局部模式，
估计器权重随 checkpoint 保存，不再传 `--degradation_checkpoint`。
指定不一致模式、错误通道顺序、错误估计器结构或缺失 A0 主干键都会拒绝。
局部适配器跟随主干维护 EMA；冻结估计器不做 EMA 平滑。

旧局部权重提供 `levels` 元数据，继续严格核对。3.0 交付权重未提供该字段，
加载器仅接受下方已核实 SHA-256 对应的这份 3.0 文件，再严格加载网络参数。
不会因为字段缺失就默认所有未知权重都使用相同通道含义。新包中的 `model.py`
是旧全局估计器，未用它替换本分支的 `local_model.py`。

若已用旧估计器训练 A1，3.0 应从同一 A0 起点另开新实验。`--resume` 会恢复
断点内的估计器，不能通过换外部文件把旧实验变成 3.0，也不支持同时传新的估计器路径。

## 交付权重

新 3.0 本地：`D:\大创\JiT\degradation\weights\local-estimator-v3.pt`。

3.0 SHA-256：`04cb85d6d59016e4a7e3cf2f83ce607bb771d0556c8bdb96c7fe6682178adc0b`。

来源：用户提供的 `A1~A3局部_估计器3.0` 目录。文件哈希、50 个参数名称及尺寸
已与交付和旧权重静态核对；未执行模型加载或推理。

旧版仍保留：`D:\大创\JiT\degradation\weights\local-estimator.pt`。

SHA-256：`6e52146ee49939ddcf32d7f745bb5803db403eecc6278fa4cd77c27f6269f62b`。

权重来自用户提供的 `A1~A3局部改进.zip`，与 C 交接记录一致。该文件被 Git 忽略，
旧、新权重均被 Git 忽略；代码推送或拉取不会带上它们，服务器需单独上传
`local-estimator-v3.pt` 到项目 `JiT/degradation/weights/`，保留旧文件。

C 报告的合成验证强度 MAE 为局部 `0.1729`、旧全局 `0.9231`；真实配对图
离线重模糊 MSE 为局部 `0.00073517`、旧全局 `0.00073662`。这些是 C 的模块结果，
没有在本轮重跑；尚未证明 JiT 恢复 PSNR/SSIM 改善。它是三种预设高斯核的局部概率，
不能称为真实显微镜 PSF 测量。

C 的 3.0 README 报告合成强度 MAE `0.1583`、合成重模糊 MSE `0.0001262`、
真实配对重模糊 MSE `0.00066446`。用户截图的合成指标为 MAE `0.1655`、
MSE `0.0001353`，两份材料合成数值不同，不能混作同一评价口径；真实配对值一致。
这些均为估计器离线结果，本次未重跑，也不是 JiT 恢复的 PSNR/SSIM/LPIPS。

## 接入检查与服务器训练

按用户要求，本机仅做语法和代码逻辑检查，不运行模型、不检查 GPU、不启动训练。
以下运行测试留给服务器，尚未执行，不能视为已通过数值或梯度检查。

在 JiT 目录中运行：

```bash
python -m unittest test_a1_integration test_a1_local_integration
```

局部测试使用小尺寸真实 JiT，并仅在测试中替换 CPU attention、关闭 compile；
检查零门控等价、图像 token 空间顺序、梯度、冻结估计器、已交付权重、
采样只估计一次、严格加载和 EMA/断点恢复。不能替代完整 B/16 的 GPU 检查。

以下为服务器训练示例，本次没有启动训练。A0 路径以实际归档文件为准，
训练清单以实际 66,976 对清单为准（不是估计器训练用的 2,000 对 debug 清单）。
三组共同使用同一起点、数据、学习率、额外 10 轮、batch 和评价设置。

```bash
cd "/home/qht/大创/CC_defocus/JiT"
python -u main_restoration.py \
  --model "JiT-B/16" --img_size 256 \
  --a0_checkpoint "/home/qht/大创/CC_defocus/checkpoints/a0_v1_frozen/checkpoint-last.pth" \
  --a0_weights ema2 \
  --degradation_mode predicted_local \
  --degradation_checkpoint "/home/qht/大创/CC_defocus/JiT/degradation/weights/local-estimator-v3.pt" \
  --train_manifest "/home/qht/大创/CC_defocus/prepared/manifests/3dh_full_train.jsonl" \
  --expected_train_pairs 66976 \
  --val_manifest "/home/qht/大创/CC_defocus/prepared/manifests/debug_val.jsonl" \
  --expected_val_pairs 300 \
  --epochs 10 --batch_size 8 --lr 1e-4 --lr_schedule cosine \
  --warmup_epochs 2 --min_lr 1e-6 --weight_decay 0 --clip_grad 1 \
  --lambda_pix 1 --charbonnier_eps 1e-3 --identity_ratio 0.10 \
  --eval_weights ema2 --sampling_method euler --num_sampling_steps 1 \
  --noise_scale 1 --t_eps 0.05 --seed 0 --eval_seed 0 \
  --eval_batch_size 4 --amp_bf16 --num_workers 4 --run_metrics \
  --model_name jit_a1_local_v3_predicted_e10 \
  --output_dir "/home/qht/大创/CC_defocus/checkpoints/jit_a1_local_v3_predicted_e10"
```

`fixed_local` 和 `none` 对照使用相同共同参数，移除估计器参数并更换模式、
model_name 和 output_dir。旧全局 10 轮结果可供参考；若训练配置不能逐项核对，
应重新运行匹配的 A0-continued 对照。所有新 checkpoint 和输出均使用新实验目录。

`run_config.json` 记录局部/全局表示、注入位置、估计器哈希、A0 和清单哈希及
`psf_loss_enabled=false`；训练继续记录退化门控。服务器完整模型先做小批次
forward/backward 和一张图生成，再启动正式训练。
