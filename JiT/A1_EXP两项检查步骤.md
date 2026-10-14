# S0/S1/S2：真值配对检查＋非EMA权重评价

适用本次已完成实验：`20261014_114501`。服务器根目录为 `/home/qht/大创/CC_defocus`，分支 jit-a1-exp。

两项都不训练、不重新生成整套数据、不覆盖原EMA2结果。一次命令会依次完成：

1. 读取已保存真值场和清晰图，抽查训练16张、验证16张，重建合成模糊并比较原PNG。
2. 从三组现有checkpoint读取model权重（最后训练参数），GPU 0/1/2分别重做S0/S1/S2同一合成300张的一步评价。
3. 把原EMA2分数、三个新model分数及配对差值汇总。

## 1. 更新并启动

在原训练环境执行：

```bash
cd "/home/qht/大创/CC_defocus"
git pull --ff-only
bash JiT/run_a1_exp_checks.sh 20261014_114501
```

脚本自动后台运行，打印输出目录和日志路径。它直接从本次三组run_config读取数据、采样、精度等设置，不依赖之前终端的ROOT、A1_EXP_TAG或A1_EXP_DATA_DIR变量。无需重训或重新生成data_2000。

新结果放到独立目录，例如 `reports/a1_exp_checks_20261014_114501_<检查时间>/`。这是自动生成的位置说明，不是终端命令。

## 2. 查看进度

```bash
CHECK_DIR="$(cat reports/a1_exp_checks_latest.txt)"
tail -f "$CHECK_DIR/launcher.log"
```

Ctrl+C只退出日志查看。启动日志先显示复现检查，结束后显示三组model评价的进度。详细日志分别为reblur.log、s0_model.log、s1_model.log、s2_model.log。

两个检查独立汇报：若复现检查发现不一致，仍会执行三组评价，同时在最终汇总标记失败，不把它当作已通过。

## 3. 检查完成后看什么

```bash
cat "$CHECK_DIR/reblur/summary.json"
cat "$CHECK_DIR/comparison.md"
```

最终check_summary.json应为complete。若不是，先看对应日志并把整个检查包发回。重复启动会使用新目录，不覆盖上一次检查。

### 检查一：真值配对

reblur/summary.json中的all_passed应为true。per_sample.csv逐张记录内容哈希检查是否通过、重建PNG与原PNG的最大差和平均差。

- exact：重建PNG逐像素相同。
- rounding_tolerance：最大差不超过1个uint8灰度级，平均绝对差不超过0.05个灰度级。这个容差只用于处理浮点/量化差异，阈值明确写入报告。
- mismatch/error：明显不一致、文件改变或输入不符合协议，需要先查配对/生成流程。

复现使用已经保存的NPZ真值场，不重新随机采样。源PSF代码哈希必须与生成时一致。抽样通过只支持这些样本的原生真值与模糊图对应，不能单独证明所有样本及全部下游变换都正确。

还保存少量四联图：清晰图、已有模糊图、重建模糊图、差异放大16倍。文件名明确包含四列顺序。

### 检查二：model与EMA2

comparison.md列出六行：S0/S1/S2各自的原EMA2和新model。固定同一数据、Euler一步、eval_seed0和原BF16设置。

重点比较同一权重口径下：

| 对照 | 用途 |
|---|---|
| S2-model－S0-model，与S2-EMA2－S0-EMA2 | 真值条件的相对收益是否受EMA影响 |
| S1-model－S0-model | 预测条件在最后训练参数下是否有收益 |
| 各组model－EMA2 | 每组的参数平均对最终分数影响多大 |

不能只看到S2-model变高就判定融合有效；S0-model也可能同时提高。若model下S2仍明显差于S0，则当前负结果不只由EMA造成，下一步再查真值使用、融合/幅度和优化预算。

程序使用eval_only，不创建训练优化器。resume只读取对应组checkpoint；S1使用其中嵌入的冻结估计器，S2仍读取相同合成真值。评价配置和所有输出均写入新reports子目录，不修改原checkpoint和final_eval_config.json。

## 4. 打包发回

```bash
cd "/home/qht/大创/CC_defocus"
CHECK_DIR="$(cat reports/a1_exp_checks_latest.txt)"
tar --exclude='outputs' --exclude='*.pth' --exclude='*.pt' \
  -czf a1_exp_checks.tar.gz -C "$(dirname "$CHECK_DIR")" "$(basename "$CHECK_DIR")"
```

下载项目根目录的 `a1_exp_checks.tar.gz`。包内包含抽样报告/少量四联图、三组新指标及原EMA2指标、评价命令和日志；不打包900张新预测图或模型权重。

## 5. 本机已验证与服务器待验证

本机只做Python语法、命令参数和纯数据测试，没有导入Torch、运行模型或检查GPU。需要时可在服务器额外执行纯检查脚本测试：

```bash
cd "/home/qht/大创/CC_defocus/JiT"
python -m unittest test_a1_exp_checks
```

这验证像素容差、样本配对和eval_only路由，不等同于两项实际检查已通过；实际结论以运行输出为准。
