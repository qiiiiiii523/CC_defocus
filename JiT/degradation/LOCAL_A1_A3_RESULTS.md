# 成员 C：10.08 局部 A1–A3 实验与交接

## 已交付的内容

局部 A1 估计器只读取模糊 RGB 图，输出三种高斯 PSF 的空间概率图 `[B,3,ceil(H/4),ceil(W/4)]`，通道依次为轻、中、重。A1/A3 的 `local_condition.py` 将它转换为 JiT 图像 token 网格上的残差条件；A2/A3 的 `spatial_psf.py` 用同一概率图计算可微分的局部重模糊及一致性损失。`verify_spatial_psf.py` 和 `verify_local_condition.py` 已通过数值及梯度检查。实际接入 JiT 及恢复网络训练由成员 A 负责。

代码：`/root/autodl-tmp/CC_defocus/JiT/degradation/`。远程实验目录：`/root/autodl-tmp/CC_defocus/runs/local_degradation_v2/`。交付权重为该目录下的 `local-estimator.pt`（约 3.4 MB；SHA-256：`6e52146ee49939ddcf32d7f745bb5803db403eecc6278fa4cd77c27f6269f62b`）。权重是实验文件，不纳入 Git。旧全局对照权重为 `runs/degradation_estimator_continuous_v1/estimator-continuous.pt`。

## 训练协议

固定清单为 `prepared/manifests/debug_train.jsonl` 的 2,000 张和 `debug_val.jsonl` 的 300 张；训练清单 SHA-256：`c4ea702e559a97027c7476154f6565ae6848ccf3d39c155270bb7624636d640b`，验证清单 SHA-256：`6ce1474e51c1814191ea01734a98375a6b50261d0b6fedc74a58693745bb821c`。仅用训练清单中的清晰图合成已知空间变化的三核混合失焦。估计器的输入始终只有合成后的模糊图；合成真值用于监督。训练集内部按原图组留一验证选轮数，固定验证集不参与选择。

环境：RTX 6000D，PyTorch `2.8.0+cu128`，独立虚拟环境 `/root/autodl-tmp/a1_local_env`；旧 `dgno` 环境的 PyTorch 2.4.1 不支持此卡的 `sm_120`。随机种子 `20261010`，AdamW 学习率 `3e-4`，batch size 8。第二版训练目标为局部概率交叉熵、局部强度 Smooth L1、边缘加权（权重 2）与预测参数的重模糊 MSE（权重 200），训练 8 轮。第一版没有后两项，会退化成近似恒定图，因此未作为交接权重。

## 结果与适用范围

训练集内部三组留出验证：第一版平均局部强度 MAE `0.6442`；第二版 `0.2442`。第二版第 8 轮在三个留出组上的 MAE 为 `0.3069 / 0.1868 / 0.2390`；预测强度图空间标准差为 `0.623 / 0.640 / 0.661`，不再是第一版约 `0.01–0.07` 的近似恒定图。按平均 MAE 选第 8 轮，再使用全部 2,000 张训练。

冻结模型后，在 300 张固定验证清晰图生成的**合成局部失焦**上，强度 MAE：局部模型 `0.1729`、旧全局模型 `0.9231`、固定轻度 `1.1470`；重模糊 MSE 分别为 `0.0001403 / 0.0025062 / 0.0050781`。这证明模型对本次合成规则有效，**不能当作真实局部 PSF 的准确率**。

另在同一 300 张**真实模糊—清晰配对图**上，冻结模型并仅用模糊图预测，清晰图只在离线评价中重模糊。平均内部像素 MSE：局部 `0.00073517`、旧全局 `0.00073662`、固定轻度 `0.00076155`。局部分别在 `175/300` 和 `184/300` 张上优于旧全局和固定轻度。优势较小，且 300 张来自同一原图组，不能据此推断跨原图泛化显著改善，也不能推断 JiT 恢复的 PSNR/SSIM/LPIPS 改善。真实图没有局部 PSF 真值。

内部选模记录：`runs/local_degradation_v2/cv_summary.json`；最终训练记录：`final_summary.json`；合成固定验证：`synthetic_evaluation.json` 与 `examples/`；真实配对：`real_paired/real_paired_summary.json`、`real_paired_per_image.csv` 和 `real_paired/examples/`（包含表现较好和失败案例）。

## 成员 A 接入要点

1. 按 `LOCAL_A1_A3_INTERFACE.md` 的尺寸与顺序，从模糊输入预测概率图，冻结 `LocalBlurEstimator`，并使图像 token 网格与空间坐标对齐。不要在推理时读取清晰目标；A1/A3 的条件和 A2/A3 的一致性损失应使用同一次预测。
2. A1 仅注入局部条件；A2 仅加局部 PSF 一致性损失；A3 两者都加。对照中保持同一 A0 检查点、训练/验证清单与评估脚本，并报告 PSNR、SSIM、LPIPS。
3. 一致性损失的权重、启动轮数和 JiT 接入点仍需在恢复训练中确定。本交接不宣称 A1/A2/A3 的恢复实验已经完成。

