# Shared Visible Geometry P4-0 / P4-1 — Codex 工作单

日期：2026-11-17。保持分支 `exp/shared-visible-geometry-multibatch`，直接父提交为 P3 的 `a0927a19776753f71badb40c2d904b633751e76c`。本轮只新增文件，不覆盖 P1/P2/P3 缓存、权重或服务器已有 FP16/B4 和 `cli_limit_fix.patch` 适配。

## 0. 目标和边界

来源：用户上传 `P3_PROTOCOL_RESULTS_CN_20261117.md`。报告在冻结 P2 reader 下得到 Predicted Q×100=73.2227、Clean=75.8431、native=74.1048；Clean−Pred=+2.6204 Q-pp。区域替换使用15/24/36个点，开口核心的影响最大，但点数、空间位置和非线性混合输入均可能影响结论。精确结果以服务器签名 JSON 为准，不用这些四舍五入数值代替来源校验。

本轮回答两件事：

**P4-0：** 几何变化是否扰乱关键的 native-relative 评分差？等点数和空间分布匹配以后，区域差异是否仍存在？

**P4-1：** 普通二维深度修正能否恢复 Clean 的部分动作判别优势，而不引入新的3D隐式场、抓取关系图或动作相关损失？

不实现 P4-2 action-relative supervision；不重新生成动作，不调用新的 CAD 标签，不改 frozen backbone、GVAR、native selector 或 P2 reader。所有评价为训练域留出验证的 Q，不是官方 AP 或真机成功率。正式 GraspNet test 不在本工作单中。

## 1. 代码入口

| 文件 | 用途 |
|---|---|
| `shared_geometry/p4_common.py` | 完整帧集合、冻结 reader、可微物理token评分、配对统计、原P2结果一致性 |
| `shared_geometry/p4_score_diagnostics.py` | P4-0：评分差/几何差、输入梯度、积分路径、等点数双向替换 |
| `shared_geometry/p4_depth_data.py` | 严格复用旧RGB特征，仅新增配准可见深度target索引 |
| `shared_geometry/p4_depth_baseline.py` | P4-1：真实多图训练、深度预测、解析几何重建、固定reader动作验证 |
| `scripts/shared_geometry/run_p4.sh` | 显式分阶段运行，不自动启动全部训练或测试 |
| `tests/test_shared_geometry_p4.py` | 20项新增CPU单元/合成集成测试 |

## 2. P4-0：冻结评分函数的机制诊断

固定同一 P2 `shared_reader/best.pt`，使用同一 P1 验证缓存。

### 2.1 相对评分与几何差

对所有5120族×6替代动作，记录：

- `pred_margin = z_pred(k) - z_pred(0)`，`clean_margin` 同理；两者的符号变化和差值。
- exact `ΔQ=(count_k-count_0)/6`，按绝对count差0…6与位移分别统计，真实ties不冒充错误pair。
- P3同定义的UDF/projective差分误差、保守联合可观测mask和各区域测量。
- 按scene/domain的描述性相关性：几何误差、敏感度加权变化与相对评分变化；不报告把所有pairs视为IID的p值。

同时计算 frozen reader 对输入6通道的梯度。对margin的贡献必须同时包含替代动作与native项：

`∇z_k(P_k)·(C_k−P_k) − ∇z_0(P_0)·(C_0−P_0)`。

默认另使用8步中点积分，报告总和与真实margin变化之间的completeness residual。端点线性近似误差、积分残差均保留，不把不精确的近似当作证明。距离通道梯度相对于归一化输入计算，归因值是logit单位；不能误写成毫米。

**这些是模型函数的路径分析。** 物理token直线路径会使one-hot状态变成soft状态，并不对应真实可见表面；梯度/积分不是物理因果重要性。

### 2.2 等点数与空间匹配替换

三块区域最小只有15点，因此共同预算为 **15点**，不是每块24点。每个模板高度层固定抽5点；seed0、16组固定掩码，在看数据和Q之前生成，并对同族全部7个动作使用相同索引。

对照：

1. 掌部/侧指/开口核心各15点。
2. 全75点中高度分层随机15点。
3. 对每个区域子集，保持其 **approach × height** 单元点数不变，仅随机化opening位置。

掌部15点已包含整个approach=0平面；其xz匹配随机对照必然等于自身。输出明确标记，不把这个零差异称为独立证据。

每个掩码做 Pred→Clean、Clean→Pred 双向替换，重新评分并argmax。**先逐掩码做决策，再平均每族结果**；不能先平均分数再选动作。bootstrap只重采样scene，域在scene内配对，掩码重复不是额外独立样本。报告区域−全局随机、区域−xz匹配对照；不使用最佳掩码，不据此训练区域权重。

### 2.3 输出

`p40/summary.json`、`RESULTS_CN.md`、`pairs.jsonl.gz`、两来源逐族记录及各掩码策略逐族平均结果。全部子文件哈希进入签名summary。正式160帧必须复现原P2两来源gain至2e-6；全token替换logits至1e-5。partial smoke不声称已复现完整均值。

## 3. P4-1：普通二维metric-depth修正

### 3.1 模型与监督

输入仅为冻结的完整 `feature_map` 和 RGB `pred_depth`；不输入quality label、action bank、native评分、rendered depth或局部region ID。模型在整张模型裁剪图上预测共享深度：

`D_corrected = D_pred + 0.05 m * tanh(CNN(feature_map, D_pred, valid_pred))`。

feature 1×1/3×3投影至32通道，双线性对齐depth分辨率，再经小型3×3卷积、逐图GroupNorm和零初始化输出层。128通道来源时新增 **32,609参数**。每次forward是真正B张图；不同场景不共享归一化统计。

50mm是预先固定的像素修正边界，不根据Q选取。统计teacher所需修正超过此边界的比例，以及模型接近边界的比例。基线pred_depth缺失的位置仍为0；只在原预测和teacher都有效的位置监督、评价，并报告覆盖率。该实现不是孔洞补全模型。

损失：全图有效像素 `L1 + 0.5*二维一阶梯度L1`，以0.01m统一归一化。先逐图求均值，再按真实batch平均。不加grasp-region权重、不加Q或相对几何loss；梯度损失只是普通图像深度空间平滑/边缘学习控制。

### 3.2 固定预算与选模

| 项目 | 默认 |
|---|---|
| 训练 | scene0–79、每scene16帧、两域，共2560帧 |
| 验证 | scene80–99、每scene4帧、两域，共160帧 |
| Batch / accumulate | 真实4 / 1；支持其他组合且协议记录 |
| 训练 | 20轮，12800更新，seed0，无早停 |
| Optimizer | AdamW，lr/wd均1e-4，gradient clip1 |
| 选模 | scene/domain-balanced **depth MAE最小**，ties优先早期 |
| epoch0 | 零残差精确复现原深度，允许入选 |

不能按验证Q另选轮次、残差系数或移动阈值。即使depth MAE改善，动作Q也允许为负并如实报告。`best.pt`选定epoch可以是0；`last.pt`仍完成20轮。每轮原子保存、严格resume；完整正式动作比较要求selection完成且weight hash与depth-only选模一致。

### 3.3 缓存和teacher隔离

复用原FP16 feature map文件，不重复保存约40GiB图像特征。新缓存只保存配准FP32 rendered-visible target及签名来源。原帧SHA、原index、target config、raw depth、intrinsics及原sidecar均核验。目标深度不进入模型forward白名单。

在448×448下，2720帧未压缩target约2.2GB，实际以smoke测量为准。每轮仍需读取原NPZ中的RGB feature_map；磁盘吞吐可能成为瓶颈，不自动改变历史存储精度。

必须使用原始真实 `visible_target.json` 配置，不猜测路径，不用sensor/predicted depth代替teacher。teacher在train/validation可读；predict子命令只读input NPZ中的feature_map/pred_depth，不实例化TargetLoader。

### 3.4 固定reader验证

在原有同一7动作bank上，从修正深度重新生成 KDTree UDF、projective和四状态one-hot，使用完全相同的75点模板。三条输入都使用 **同一个冻结P2 reader**：Predicted / Corrected / Clean。

原动作bank、尺寸、physical IDs、exact labels不改变；不将修正深度传回生成器重新生成grasp center。因此不用新增CAD标注，不混入proposal变化。原P1 Predicted几何重新构造并逐元素校验；原P2 Pred/Clean gain需再次复现。

报告：
- 全图depth MAE/RMSE/gradient error和有效覆盖。
- 同一组共同可观测支撑点上的绝对及动作差分UDF/projective误差，避免mask变化制造虚假改善。
- native/selected/oracle Q、rescue/harm/move、native-relative pair、偏差，Corrected−Pred与Corrected−native的scene区间。
- 校正深度与紧凑corrected geometry的逐帧输出，签名来源和两种基线结果。

这不是完整GraspNet inference/eval。即使验证Q为正，也不能自动报告AP提升或部署成功。固定reader已经适配Pred/Clean两来源，Corrected仍可能产生分布移位，阴性结果不自动证明必须增加3D场。

## 4. 117运行步骤

先读取 `AGENTS.md`，检查 `git status --short`。保留服务器已有FP16/B4和P2 CLI修复；如pull存在冲突，停止并人工核对，禁止reset/force/stash掉未提交工作。

```bash
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01
export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
export P2_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p2_run01
export P4_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p4_run01
export TARGET_CONFIG=/ABS/EXACT/original/visible_target.json
export DEVICE=cuda:0
export BATCH_SIZE=4
export VAL_BATCH_SIZE=4
export ACCUMULATE=1
```

`TARGET_CONFIG`是占位，必须换成旧cache原始配置。所有新输出必须与旧结果目录无包含/相等关系，不覆盖旧结果。

```bash
bash scripts/shared_geometry/run_p4.sh check

# P4-0：5帧（跨两域，含B4+尾批B1）烟测，再正式160帧
bash scripts/shared_geometry/run_p4.sh p40-smoke
bash scripts/shared_geometry/run_p4.sh p40

# P4-1：每partition 5帧跨两域烟测、1轮训练和动作验证
bash scripts/shared_geometry/run_p4.sh depth-smoke

# 全量target索引、20轮训练、固定reader验证
bash scripts/shared_geometry/run_p4.sh depth-cache
bash scripts/shared_geometry/run_p4.sh depth-train
bash scripts/shared_geometry/run_p4.sh depth-evaluate
```

真实CUDA的显存、吞吐、单帧源一致性先在smoke测量。禁止把本地CPU测试当成117验收。默认NUM_WORKERS=2，可按CPU/磁盘调整；不改变科学数据协议。

缓存或整轮训练中断后可显式 `RESUME=1` 重跑同一阶段；代码/签名/训练参数发生变化则拒绝续跑，使用新目录。诊断与评价目录不覆盖，不提供静默覆盖模式。

单独深度推断（不读teacher）：

```bash
export INPUT_NPZ=/ABS/native_feature_map_and_pred_depth.npz
export OUTPUT_NPZ=/ABS/new_corrected_depth.npz
bash scripts/shared_geometry/run_p4.sh depth-predict
```

## 5. 预注册判读

P4-0：如果等点数/xz匹配后区域差异减弱，之前的“核心区域更关键”需要收缩；如果保留差异，也只支持当前reader的模型敏感度，不等同于物理接触因果。按真实质量差分层后再解读MAE与排序关系，不能以高误差直接定义hard pair。

P4-1：若普通深度修正已经改善动作质量，优先保留简单一致的二维深度表示。若MAE降低而动作判别不变/变差，再分析局部几何响应、reader适配或信息不足；不能只凭一次小型CNN阴性结果证明隐式场必要。

P4-2未实现。新action-relative监督设计应在完成本轮后另立协议，不根据现有验证结果直接把开口区loss加权为最大。

## 6. 验收清单与本地交付状态

- [ ] 真实两域输入/teacher/parity smoke通过。
- [ ] P4-0完整160帧、5120族、30720相对pairs，复现P2来源结果。
- [ ] 区域每次都15点；mask重复先族内平均，再scene bootstrap；保存seed和索引。
- [ ] 梯度/IG误差、等质量动作、无效观测与相关性空分组正确处理。
- [ ] P4-1 teacher只用于损失与验证；完整B4训练及尾批验收。
- [ ] 20轮/12800更新；epoch选择只基于depth MAE，epoch0不能被隐藏。
- [ ] 固定bank/reader/labels，原P1解析几何和P2 Q复现，Corrected结果不替换负数。
- [ ] 汇总`p40/summary.json`、`depth_model/history.json`、`selection.json`、`depth_evaluate/summary.json`，保留逐族压缩记录。

本地已经运行新增20项CPU测试、61项既有非CLI回归，以及旧CLI专项复核；详情见 LOCAL_VALIDATION JSON和日志。没有117上的新CUDA、真实depth训练或P4数值。P4的科学结果栏保持未运行。
