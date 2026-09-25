# CVA depth dynamics 执行记录（2026-09-26）

状态：独立 P0 验收及 D0/D1 的 50-update 配对运行完成，已从完整状态续跑至 500 updates；本记录不宣称塌缩机制已确定。

## 环境与初始化

- 分支：`codex/cva-depth-gradient-dynamics`，从 main `52d09f925059bec3643610ecf1f1722894627ee5` 建立。
- 训练服务器：gpu02 / `10.30.7.117`，隔离目录 `/home/robotarm/EconomicGrasp-depth-dynamics`。原目录的用户修改保留。
- 解释器：`/home/robotarm/miniconda3/envs/grasp/bin/python`；PyTorch 2.5.0+cu118、CUDA 11.8、RTX 3090 24 GiB。
- 用户提供的 gpu04 Stage-1 文件已复制并核验两端 SHA256：`0cb8cd54bd5eef44a81c8d8de6969003377224d41a84b2002ba662bb8ce0c69c`。
- 文件记录 epoch 16、Stage-1、predicted geometry、image-FPS、global_film、use_fuse_depth=True。该文件无 optimizer；两组均从相同权重重置 AdamW。
- 预定配对设置：batch 1、FP32、LR 1e-4 constant、weight decay 0、global clip 1，KD 关闭、DINO 冻结。与原训练优化状态不同，不能称为原训练的精确 resume。
- 固定探针：16 个 train 帧及用户明确指定的 32 个 test_seen 验证帧；保留原 split 名和逐帧清单。

## 已取得的证据

1. 路径控制的小张量测试、状态保存与恢复测试、方向有限差分测试已通过。默认 detach 行为保留。
2. 第一次真实前向验收发现 CDF/width 标签映射含重复 CUDA 写入。固定输入重复十次仍可变化；当前检查 batch 中变化位于有效监督 mask 外。它是基础重复性错误，不能直接解释深度塌缩。
3. `73a1b4d` 将多对一 inverse-view 映射的代表选择显式固定为最大 scene index；CPU/CUDA 重复性测试通过。所有正式配对须从修复后的同一基础版本开始。
4. `20260926_stage1_seed0` 的 8 个真实 train/validation batch 已通过 detach 前向一致性和路径连通性检查：D0 task→depth 不连通，depth supervision 连通，E/Q/C 单路均有非零 task 梯度。
5. 相同运行的审计前后单步更新检查在 6 个权重张量上超过 atol=1e-6、rtol=1e-5；其余状态比较通过。随后增加 plain/plain/audit 三分支对照，没有直接将它解释为审计状态泄漏。
6. `fd810a4` / `20260926_replay_control_seed0` 三分支实测：audit 前后完整状态与已有 `.grad` 的 SHA256 完全一致；三次更新前状态、实际 batch、前向 tensor、各项 loss 完全一致。梯度范数均为 2.236729621887207，clip 系数均为 0.44708110588492117。

| 比较 | 权重最大绝对差 | 权重差 L2 | 超过原权重容差的元素 | 梯度相对 L2 差 |
|---|---:|---:|---:|---:|
| 无审计 A / 无审计 B | 4.083e-6 | 2.336e-5 | 44 | 3.617e-7 |
| 无审计 A / 有审计 | 3.576e-6 | 2.405e-5 | 43 | 3.568e-7 |
| 无审计 B / 有审计 | 3.703e-6 | 2.360e-5 | 41 | 3.716e-7 |

所有梯度均通过原 atol=1e-6、rtol=1e-5；最大梯度差不超过 2.19e-8。
审计分支与无审计重复分支的误差同量级，支持 CUDA backward 数值波动的解释。
原 strict gate 的失败报告保留。下一次独立验证采用显式 calibrated policy：状态仍逐位核验，
gradient/optimizer 保留原容差，参数最大差的固定绝对上限为 1e-5，并以 plain-repeat
包络作额外检查。上限在新实验前锁定，不按新结果继续放宽，不更改模型、detach 或 Adam eps。

本机安装版 PyTorch 2.5 的算子检查表明，CUDA grid-sample backward 在 deterministic
warn-only 模式下发出非确定性警告；bilinear interpolate 会切换到其 decomposition 实现。
正式运行保持原后端设置，没有因此切换训练算子。单步数值一致性不等于逐位确定性。

## 独立验收与配对启动

`aa4ffd20ee52f0a60f776c1c3b90d048ef7fa216` 的
`20260926_stage1_calibrated_seed0` 完成独立完整验收：8 个真实 batch 的开关前向/梯度检查、
三分支非干扰验证、16 train + 32 test_seen 帧的 train/eval 探针均通过。
审计状态和前向逐位一致；plain/plain 参数最大差为 7.484e-6，audit/plain 为
5.977e-6 / 5.465e-6，均在运行前锁定的 1e-5 绝对上限内。
`diagnostic_noninterference_strict_passed=False` 如实保留，calibrated gate 为 true。

| 初始 eval 探针 | 帧数 | GT-valid MAE (mm) | 平均前景 std ratio | 平均局部斜率 |
|---|---:|---:|---:|---:|
| train | 16 | 1.8564 | 0.99279 | 0.68271 |
| test_seen（指定验证集） | 32 | 3.9312 | 0.96739 | 0.63700 |

16 个 train 帧均满足事件判读的 GT std 条件；初始 flat fraction=0，initial_flat=false。
随后在 GPU 0/1 以相同实验 commit 启动 D0=none、D1=all 的 50 次更新。
汇总工具的后续修正单独提交，不改变正在运行的模型代码或实验 commit。

## 50-update 配对结果

两组均完成 50 次实际更新、50 个训练图像。逐张比较初始 checkpoint 的 model state 和
optimizer state 完全相同；初始与 step 50 的 loader、Python/NumPy/CPU/CUDA RNG 也均相同。
训练日志中的全部配对记录使用相同样本索引，首步总 loss 均为 0.6337293386459351。

| Arm / eval 探针 | GT-valid MAE (mm) | 平均前景 std ratio | 平均局部斜率 | 平均 depth bias (mm) |
|---|---:|---:|---:|---:|
| D0 / train | 2.5247 | 0.98354 | 0.67467 | -1.1059 |
| D1 / train | 6.7550 | 1.01977 | 0.74622 | +1.7370 |
| D0 / test_seen | 4.0257 | 0.97034 | 0.64095 | -0.1059 |
| D1 / test_seen | 6.5838 | 1.04205 | 0.68141 | +1.8055 |

两组 train/eval 事件探针的 flat fraction 都为 0。D1 的 MAE 上升，但图内对比度并未整体
变平；局部斜率反而较初始值增大。不能把这 50 步结果称为 constant-depth collapse，也不能
由此证明长期稳定。固定图、共享米制色标的误差图及逐图指标已保存。

50 次更新的观测用时（不含初始/结束探针）为 D0 168.65 秒、D1 130.33 秒；共享机器上的
I/O 等条件会影响此数值，不能据此比较路径的固有计算成本。观测到的进程 GPU 显存为
6516 / 6570 MiB；未发生 OOM 或非有限数值。完成检查后，两组从各自 step 50 完整状态
继续到 500，仍使用同一实验 commit 与优化设置。

逐 split/mode 的原始汇总见 [summary50.csv](depth_dynamics_results_20260926/summary50.csv)，
P0 gate 见 [p0_gate.json](depth_dynamics_results_20260926/p0_gate.json)。
P1 的定量复核与有限差分适用范围见 [P1_REVIEW_ZH.md](depth_dynamics_results_20260926/P1_REVIEW_ZH.md)。
该有限差分仅验证单图、all route、foreground 固定连续子函数；没有验证所有 batch/路由。
Q/C 在两个局部方向上有明显抵消，不能用 Q-only 梯度大小代替 all 的实际网络更新。

## 产物与判读范围

完整产物保存在训练服务器：

- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_labelcheck_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_stage1_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_replay_control_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_stage1_calibrated_seed0/diagnostics/`
- `/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics/20260926_stage1_seed0/P0/`

产物包括 contract、逐项梯度、route connectivity、前向相等性、方向探针、梯度图及单步非干扰报告。前两次 strict 运行的 `p0_gate.json` 保持 false；calibrated 新运行独立验收通过。局部导数不等于实际优化器更新效果；配对轨迹正在生成，尚无必要性/充分性或修正方案结论。
