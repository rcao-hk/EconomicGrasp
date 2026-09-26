# CVA depth dynamics 执行记录（2026-09-26）

状态：D0/D1 的 1000-update 配对及 E-only/Q-only 的 500-update 对照完成；主线继续分段推进到 2000，当前预算为 1500。正在运行 QC（all-minus-E）对照及真实状态单步重放。本记录不宣称常量深度塌缩或机制已确定。

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
原有限差分仅验证单图、all route、foreground 固定连续子函数；后续八批补充见下文。
Q/C 在两个局部方向上有明显抵消，不能用 Q-only 梯度大小代替 all 的实际网络更新。

## 500-update 配对结果与路径对照

两组各完成 500 次实际更新、500 张训练图像。完整状态的 Python/NumPy/CPU/CUDA RNG、
loader、module mode、is_training 逐项完全一致；全部已记录步骤的样本索引也一致。
校验及完整 checkpoint SHA256 见 [paired_state500.json](depth_dynamics_results_20260926/paired_state500.json)。

| Arm / test_seen eval（32 帧） | GT-valid MAE (mm) | GT-valid bias (mm) | 平均前景 std ratio | 平均局部斜率 | 平均局部 contrast ratio |
|---|---:|---:|---:|---:|---:|
| D0 | 4.0010 | -0.0509 | 0.98264 | 0.62174 | 0.98580 |
| D1 | 34.8060 | +31.6324 | 1.37767 | -0.71352 | 2.37939 |

D1 在 step 100/200/300/400 的验证 MAE 分别为 13.992/10.679/8.503/9.333 mm，
step 500 明显恶化。固定图中多个物体呈现正深度偏移和局部深度关系反转。
其全 GT-valid 图内 std ratio 仍约 1.00556，不能仅用这一指标判断几何健康；前景 ratio
与全 GT-valid ratio 也不可混称。预设 constant-depth flat fraction 两组均为 0，无该类
confirmed event。这里测得的是几何偏移/失真，尚未证明其持续性、必要路径或任务标签逃逸。

完整曲线汇总见 [summary500.csv](depth_dynamics_results_20260926/summary500.csv)。
step 100/200/300/400/500 的 D0/D1 完整状态额外以 hardlink 保留在各 arm 的
`geometry_review_at_500/`，避免后续正常滚动保留删除早期证据；训练状态未改写。
主线继续使用 `aa4ffd2`，从 500 续到 1000；分段检查使后续事件观察可控制在约 200 步。

基于 P1 的 GSE→view 冲突线索及 Q/C 局部抵消，增加 `E_only=gse`、`Q_only=seed_xyz`
两组 500-update 对照。二者使用同一初始权重、重置优化器、数据流、超参数和已通过的
P0 gate，分别在 GPU 2/3 运行；终点是检查同类几何退化，不能将其自动称为常量塌缩复现。
必要性仍需要 all-minus-suspect，实际有害更新还需要完整状态反事实，当前均未完成。

## 1000-update 主线与 500-update 路径结果

后台运行均正常完成指定预算。D1 在 step 600 的验证 MAE 降至 15.267 mm、局部斜率
恢复为 0.57058，说明 step 500 的反转并非永久状态；其后误差仍高于 D0。

| Arm | updates | test_seen MAE (mm) | 前景 MAE (mm) | 前景 bias (mm) | 局部斜率 |
|---|---:|---:|---:|---:|---:|
| D0 | 1000 | 4.9596 | 7.8430 | +2.8884 | 0.64945 |
| D1 | 1000 | 12.9877 | 28.8184 | +27.4503 | 0.40304 |
| D0 | 500 | 4.0010 | 7.3332 | +0.4571 | 0.62174 |
| D1 | 500 | 34.8060 | 91.9722 | +91.0672 | -0.71352 |
| E-only | 500 | 16.6857 | 34.7381 | +23.2904 | 0.25541 |
| Q-only | 500 | 5.8970 | 12.6338 | +7.0370 | 0.56033 |

E-only 已产生明显几何退化，但没有重现 D1@500 的平均负斜率；Q-only 的退化较小。
这支持继续检验 GSE 相关影响，不足以推出它是唯一原因。已启动 `QC_no_E=seed_xyz,support`
的 500-update 对照，检验当前配置下移除 E 的结果。所有轨迹仍使用 `aa4ffd2`。
四组初始完整 model、optimizer、RNG、loader 与 D0 逐项相同，见
[p3_initial_pairing.json](depth_dynamics_results_20260926/p3_initial_pairing.json)。
逐步结果见 [summary_main1000_paths500.csv](depth_dynamics_results_20260926/summary_main1000_paths500.csv)，
其中 QC 当时仅有初始化测量；新增 `local_geometry.png` 单独显示有符号局部斜率、相关性与前景误差。

P4 工具使用同一次前向/反向捕获的梯度，比较正常重复、移除 view→DPT/FiLM 梯度但固定
正常裁剪系数、移除后重算裁剪三种条件。保留历史 Adam 状态，不把独立更新相加。
三个源状态 step 200/300/400 的真实重放已完成，11 项控制/状态恢复检查均通过。
固定正常 clip 系数移除 view→DPT/FiLM 后，下一步验证 MAE 相对正常分别降低
0.218426/0.044659/0.135386 mm，31/27/25 张（共 32 张）验证帧降低。
局部斜率未同步改善，step 200/400 移除后 MAE 仍有增加；不能推广为阻止所有退化。
完整反事实、实际 Adam 更新与逐图结果见 [P4_REVIEW_ZH.md](depth_dynamics_results_20260926/P4_REVIEW_ZH.md)。
step 400 首次启动在 CUDA 初始化阶段失败，独立 retry 成功，失败日志保留。

基于上述一致的局部 MAE 方向，从同一 D1@400 完整状态启动正常/移除 view 两组 100 步短段，
代码 `c15fecb`，GPU 2/3，run_id=`20260926_stage1_rescue_view_seed0`。
两组保留历史 Adam、原数据流、全部 forward 路由及 native 标签；移除组每步保留该步移除前
完整梯度对应的 clip 系数。7 项 CPU 测试已通过，含 50 次连续移除更新与显式保留 loss 的一致性。
短段已完成：正常/移除组的终点验证 MAE 为 13.546466/4.859918 mm，局部斜率
0.241393/0.609464。两组完整起点及全部 100 步输入/loader 配对检查均通过。
正常续训未精确复现原主线 D1@500 的 34.806 mm，因此正在补做同配置正常重复，
不把当前结果直接称为对原主线全帧反转的精确救援。原始对照和此限制详见 P4 报告。

## 1500-update 与 all-minus-E 终点

主线 D0/D1 均完成 1500 次更新，验证 MAE 分别为 4.205752/18.018973 mm，
局部斜率 0.636066/0.623897；均无预定义 constant-depth 事件，连续阈值计数为 0。
已使用原 GPU 0/1、相同 `aa4ffd2` 代码和完整状态续至 2000 的预算，尚未完成。

QC_no_E 完成 500 步，验证 MAE 4.190952 mm、前景 MAE 7.894006 mm、局部斜率
0.625634，32 张验证帧没有负斜率。相同步数 D0/D1 为 4.0010/34.8060 mm。
在本 seed/初始化/预算下，关闭 E、保留 Q/C 没有重现 all 的严重失真；结合 E-only
的退化，支持 E 相关影响，但不证明 E 普遍必要或唯一。

seed 1/2 的八批 P0 均通过预先锁定的 calibrated gate，strict replay 仍为 false。
两 seed 都保留了 raw route/数值检查记录。seed 1 的 D0/D1 500 步配对已启动，
seed 2 的 D0 先在空闲 GPU 5 启动；其 D1 待 GPU 4 的正常短段重复结束后启动，
两组预算/初始状态/数据配置会一致，启动时刻无需相同。全部复制训练使用 `aa4ffd2`。

## 八批有限差分与标签读取核查

`20260926_stage1_fd8_retry_seed0` 完成 4 train + 4 test_seen 的 all-route 前景方向检查，
共 1248 个唯一 FD 行。mu 最小相对步长的 task AD/FD 相对误差均小于 1%；alpha 的
直接 FP32 总 loss 差分对步长和相消较敏感，其中 batch3 的中间步长误差仍为 29.506%。
使用已记录 raw 分项在 FP64 中重组小步长 secant 可改善总和精度，但这不是网络 float64
前向，也未替换原测量。全部数值、范围与例外见
[FD8_REVIEW_ZH.md](depth_dynamics_results_20260926/FD8_REVIEW_ZH.md) 和
[FD8_AGGREGATE.csv](depth_dynamics_results_20260926/FD8_AGGREGATE.csv)。
首次辅助启动遇到 CUDA 初始化错误，原失败日志保留；retry 独立目录完成，主线未受影响。

旧 NumPy 的 NPZ membership 会实际读取解压数组。`aca81f7` 将两个存在性检查改用
archive key 元数据；真实四帧 ABBA 对照的 16 次样本和 collate 全值/hash/RNG 完全一致，
每样本 NPZ 读取从 22 次降到 11 次，两个合成回归测试也通过。
共享机器上的平均 getitem 时间旧/新为 3.922/5.213 秒，中位数为 2.954/2.403 秒；
受 I/O/并发影响，本次没有证实平均速度提升。主线及路径对照继续使用原 `aa4ffd2`。
原始核查见 [npz_abba_aca81f7.json](depth_dynamics_results_20260926/npz_abba_aca81f7.json)。

## 产物与判读范围

为满足重复 seed 的空间预算，已把完成的 E-only、Q-only 六个时刻的完整状态各复制一份到
gpu04（10.30.7.119）`/data2/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics_archive_20260926/`。
12 个文件均逐项核对 SHA256 与大小；训练机保留初始/终点，step 100/200/300/400 的八份
中间状态转为在 gpu04 归档，并在原位置写入 `.archive.json` 指针，共释放 5,485,569,088 bytes。
没有移动正在使用的 D0/D1、QC 或救援状态。归档映射与校验见
[completed_path_checkpoints_verified_archive.json](depth_dynamics_results_20260926/completed_path_checkpoints_verified_archive.json)。

重复 seed 先执行独立 P0；seed 1 的 `20260926_stage1_replica_seed1` 已在 GPU 4 启动。
GPU 4 首次单张量预热遇到 CUDA driver initialization failed，随后审计进程成功构造真实模型；
训练仍须等待该 seed 的正式 gate。尚未把重复 seed 的任何结果列为已通过。

完整产物保存在训练服务器：

- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_labelcheck_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_stage1_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_replay_control_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_stage1_calibrated_seed0/diagnostics/`
- `/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics/20260926_stage1_seed0/P0/`

产物包括 contract、逐项梯度、route connectivity、前向相等性、方向探针、梯度图及单步非干扰报告。前两次 strict 运行的 `p0_gate.json` 保持 false；calibrated 新运行独立验收通过。局部导数不等于实际优化器更新效果；已取得 500-update 配对轨迹，尚无必要性/充分性或修正方案结论。
