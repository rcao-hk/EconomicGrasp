# P4：三个完整状态的一步 view→depth 反事实

来源为 D1（E/Q/C 全开放）的 step 200、300、400；原主轨迹代码 `aa4ffd2`，
重放代码 `1c397a1`。对应其后 step 500 的暂时性几何失真，而非预设的 constant-depth 事件。
全部使用同一 Stage-1 初始化、FP32 AdamW、LR 1e-4、global clip 1。
原始完整 JSON 保留在服务器 experiment 根目录的
`20260926_stage1_counterfactual_view_seed0/step_000200/`、`step_000300/`、
`step_000400_retry1/`；文件名均为 `counterfactual_report.json`。
本目录 [P4_three_states/sources.json](P4_three_states/sources.json) 记录输入报告、源 checkpoint、
执行源码的 SHA256 和逐项控制结果。第 400 步首次启动在 CUDA 初始化阶段失败，独立重试成功。

## 干预及有效性

每个状态只执行一次真实 native forward/backward，先取得同单位的加权 view loss 对
depth 专属 DPT/FiLM 的导数，再捕获完整 `.grad`。四个分支恢复同一模型、AdamW 历史、
RNG、loader、buffers、运行模式及计数器，并共用这一次前向后的 buffers 和梯度：

- normal_A、normal_B：完整梯度正常 AdamW 更新；这是优化器层的精确重复，**不是两次独立 CUDA backward**。
- remove_fixed_clip：仅在 DPT/FiLM 参数中减去当前 view 导数，保留正常分支裁剪系数。
- remove_recomputed_clip：相同移除，再重新计算 global clip。

三份报告均 completed，11 项控制/恢复检查全部为 true；包括正常重复状态完全一致、
各分支更新前状态一致、固定裁剪移除分支的非 depth 参数实际更新与正常分支完全相同、
源状态不被修改以及模型/优化器/RNG/loader/runtime/routes 恢复一致。
保留旧 Adam moments，没有把独立 loss 的 Adam 更新相加，也没有重置动量。

每次在同一 16 train + 32 test_seen（用户指定验证集）上做 eval-mode 深度测量。
先记录源状态，再记录前向后但优化前的状态，以分开 buffer 变化。以下误差为每图均值再平均。

## 验证结果

| 源 step | 优化前 MAE mm | 正常一步 | 移除 view、固定 clip | 移除 view、重算 clip | 固定 clip 相对正常降低 mm | 降低的帧数 |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 10.679005 | 11.087297 | 10.868870 | 10.869054 | 0.218426 | 31/32 |
| 300 | 8.502569 | 8.527053 | 8.482394 | 8.482077 | 0.044659 | 27/32 |
| 400 | 9.332820 | 9.472513 | 9.337126 | 9.333472 | 0.135386 | 25/32 |

固定 clip 已可观察到这个条件效应，因而这些单步 MAE 差异不必由重新裁剪非 depth
参数来解释。重新裁剪会小幅改变结果，不能将这两个分支混称。
step 200、400 移除后仍有 MAE 增长，不能写成三次均“阻止所有退化”。

| 源 step | 正常一步局部斜率 | 移除、固定 clip 局部斜率 |
|---|---:|---:|
| 200 | 0.544402 | 0.540401 |
| 300 | 0.472234 | 0.469777 |
| 400 | 0.465822 | 0.458522 |

局部斜率的变化没有与 MAE 一致改善，三次正常分支斜率反而更高。不能将 MAE 的局部
条件收益推广成所有几何量都更好，更不能仅凭这个表证明 view 是唯一有害 loss。
view 对 depth 的连通性在 P1 中只经过 E，但此处的效应仍以当前全路径模型/其他梯度/
历史 Adam 状态为条件，不提供 E 对任意训练动态普遍必要的证明。

完整 24 行聚合及 576 行逐图结果见
[interventions.csv](P4_three_states/interventions.csv) 和
[interventions_per_image.csv](P4_three_states/interventions_per_image.csv)。
包含 train/test_seen、MAE/bias、局部相关性/对比度、sigmoid 状态、裁剪系数和实际参数更新范数。

## 短段验证

三个状态的 MAE 方向一致，支持从共同 D1@400 完整状态启动 100-update 的正常/移除
配对短段。入口 `rescue_cva_depth_counterfactual.py`，代码 `c15fecb`；每步保留原生
forward、匹配、其他 task 梯度和 Adam 历史。移除分支先算**自己这一步完整梯度**的 clip
系数，然后移除 view→DPT/FiLM，再使用该系数；不是借用另一条已分叉轨迹的 clip。
两条轨迹后续 forward/其他参数梯度可自然不同，因此不能声称 100 步的非 depth 更新固定。
两组源状态 hash、逐步 batch hash、loader、实际更新和固定 probe 都会保存。

7 项 CPU AdamW/状态恢复检查通过，包括 50 次连续梯度移除与显式保留 loss 的更新一致。
两组已完成 step 400→500 的 100 次更新。完整起点哈希、resolved config（除 log_dir）、
全部 100 个 batch 的内容哈希与 loader 状态一致，step 401 的总 loss/完整梯度 norm/clip
均相同。见 [rescue_pairing.json](P4_rescue_seed0/rescue_pairing.json)。

| 更新终点 | 组别 | 验证 MAE mm | 前景 MAE mm | 前景 bias mm | 局部斜率 | 局部相关性 |
|---|---|---:|---:|---:|---:|---:|
| 400 | 共同起点 | 9.332820 | 20.232781 | +9.262178 | 0.446480 | 0.501736 |
| 450 | 正常 | 11.570203 | 19.997038 | +0.360322 | 0.680413 | 0.483284 |
| 450 | 移除 view | 6.367903 | 12.582007 | +7.189352 | 0.589098 | 0.579569 |
| 500 | 正常 | 13.546466 | 26.639700 | +16.207364 | 0.241393 | 0.298234 |
| 500 | 移除 view | 4.859918 | 9.155402 | +1.074370 | 0.609464 | 0.596095 |

在这次配对续训中，移除组终点的 MAE、前景误差及局部几何优于正常组，且保留了其他
任务梯度。此处比三个单步结果多了短轨迹证据；仍不是所有初始化和 seed 的一般结论。
两组均无预定义 constant-depth 事件，sigmoid 导数均约 0.244，未出现边界饱和。
完整 24 行 train/eval、train/test_seen 表见 [summary.csv](P4_rescue_seed0/summary.csv)，
[局部几何图](P4_rescue_seed0/local_geometry.png) 已检查。

**正常续训没有精确复现原主线的终点。** 原 D1@500 为 34.806 mm、斜率 -0.71352；
这次正常组为 13.546 mm、斜率 +0.24139。不能将两者混用，不能把当前干预直接写成
“阻止了原主线那次全帧反转”。两者还存在已记录的执行入口/诊断节奏与 NPZ membership
实现差别；已证明单样本加载等价和 CUDA backward 存在小数值波动，但尚未将 100 步
差异唯一归因于 CUDA、数据实现或诊断节奏。额外的同配置正常续训重复已启动，
run_id=`20260926_stage1_rescue_repeat_seed0`，用于衡量当前短段的复现程度。
seed 1/2 的完整 P0 均通过 calibrated gate（strict 仍为 false），重复训练将另行报告。
