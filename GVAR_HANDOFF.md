# GVAR_HANDOFF.md — GVAR / EconomicGrasp 研究与工程交接

> **截至 2026-10-13（香港时间）。面向完全没有历史聊天上下文的新会话。**
>
> **一句话状态**：七组 GVAR 变体（至 `volume_hybrid`）已在服务器完成 10%+10% mixed 训练、Original GraspNet e19 官方 AP 和配对分析；P1 的非 AP 预选 120 帧 probe 可靠性诊断也已完成。**下一批 P0-A / P0-B / P0-C 的代码已提交至 GitHub，但尚无经核验的真实服务器执行结果**；必须先做服务器已验证修复与 GitHub 源码的合并及 parity 审计，随后执行 smoke → 正式实验 → 分析。不要把“代码提交／合成单测通过”写成“实验已经完成”。
>
> **当前 GitHub 代码锚点（本交接创建前核验）**：`rcao-hk/EconomicGrasp`，分支 `exp/gripper-volume-action-reader`，HEAD `ded3a32d97bac94da1600a456f88a6a7732cfa66`，提交信息 `Add GVAR P0-A frozen-reader counterfactual, P0-B inverse hybrid, and P0-C matched RGB-domain evaluation`。主分支 `main` 仍为 `52d09f925059bec3643610ecf1f1722894627ee5`，禁止直接改动。提交本交接文档后，该实验分支 HEAD 会再前进一个 commit；以上 SHA 仍是 **P0-A/B/C 实现的可复现锚点**。
>
> **优先阅读执行合同**：[`experiments/GVAR_P0ABC_CODEX_20261012.md`](experiments/GVAR_P0ABC_CODEX_20261012.md)。所有具体调用以该文件与服务器真实配置为准。本文件是全局交接，不是该合同的替代品。

## 1. 我们究竟在研究什么？（不要偏离科学问题）

目标：以 EconomicGrasp 的 RGB-only *network* 为基础，在 cluttered scenes 中预测 6-DoF parallel-jaw grasps。以 DINO/DAV2 特征、DPT metric-depth、CVA（center–view–angle）＋ CDF score / depth-wise width 为主干。我们要研究 **具体动作的视觉／几何证据如何被读取、何时可靠、能否提高跨物体及材质泛化**，而不是继续堆 global score gate、center correction 或任意深度 trick。

GVAR（Gripper-Volume Action-conditioned Reader）对每个候选动作插入深度 `d` 构造 **36 probes/action**：left/right contact、closing、finger body、palm、approach，6 roles × 6 probes。局部坐标：`x=approach`、`y=opening`、`z=height`；finger length=0.06m，support envelope width 固定 **0.06m**，而最终执行 width 仍由模型预测，**当前 reader 不是完整的 exact-(t,R,d,w) 几何评估器**。

- `volume_fixed`：固定 insertion 参考位置 `d_ref=0.025m` 的视觉采样，无 relative geometry。
- `volume`：RGB probes 随实际离散 insertion depth 移动，无 relative geometry。
- `volume_fixed_rel`：固定 RGB/geometry support，附加 action-relative predicted geometry。
- `volume_rel`：RGB 和 geometry 都随实际 insertion depth 移动，在同一动态空间位置融合。
- `volume_hybrid`：RGB fixed，geometry dynamic；相同 role 的数值对应不同三维位置。
- **新 `volume_inverse_hybrid`（仅实现、尚无正式 AP）**：RGB dynamic，geometry fixed；补齐 RGB support × Geometry support 的第四格。
- `baseline` / `slot`：原 CVA-CDF 对照和仅显式插入深度 embedding 的控制组。

相对几何包括 gripper-frame surface xyz、ray depth residual、known/valid flags，均是 **soft cues**；已知射线深度不等于真实 free-space／occupancy，投影有效不等于真实可见或成功接触。数值几何对 grasp loss 采用完整 **E/Q/C detach**，但 metric depth head 自身仍直接受监督，不能把所有 RGB／grasp 特征 detach。

## 2. 固定实验合同（任何比较都必须遵守）

| 项目 | 必须保持 |
|---|---|
| 训练 | 10% Original GraspNet RealSense＋10% GN-Trans，各 100 scenes×26 frames=2600，总 **5200/epoch**；两域各自抽样后混合 |
| 训练计划 | 3 张 RTX 3090（每张 24 GiB）× **per-GPU batch=3**，global batch=9；**21 epochs，e0–e20**；seed=0；fresh DINO/DPT task initialization（不从别的 task checkpoint warm start） |
| 验证 | **仅 Original GraspNet test_seen 780 frames**；不要用 GN-Trans/Novel AP 选择 checkpoint |
| 正式模型 | 固定 **e19 / checkpoint_epoch_019.tar** 做主 AP；e20 是最后轮存档 |
| 网络 | DPT predicted metric depth，pose `global_film`，image-FPS，A1 Top-1，CDF＋depth-wise width，原监督及 E/Q/C detach |
| 评测 | Original GraspNet Seen/Similar/Novel 各 **30 scenes × frames 0,10,…,250（780帧）**；官方 AP NPY 必须是 finite float `[30,26,50,6]` |
| 推理/碰撞 | inference batch 3、workers 4；Original sensor cloud collision，threshold 0.01，voxel size 0.01m；官方 CPU evaluator **workers 16、各 split 串行** |
| 运行实现 | 大型 volume reader 曾用 `ACTION_CHUNK=2048`、activation checkpoint 关闭、train workers 3、val workers 4、prefetch 1；与旧 baseline/slot 的 chunk/AC 有执行成本差异，不能混作架构 FLOPs 证明 |
| 统计 | 30 scenes 作为 resampling clusters，26 frames 不能按 IID 抽；50,000 次 paired scene-bootstrap，三个 split 分别抽样后等权组成 Mean；bootstrap seed 20261009 |

**术语底线**：网络输入可以严格是 RGB，但数据预处理的 workspace/crop、GraspNet 原协议 collision filter 依赖 Original sensor depth。所以只能称 **RGB-only network with depth-assisted preprocessing/filtering**，不能写成严格 raw-RGB-only whole pipeline。P0-C 会进一步固定这个协议。

历史 checkpoint 有训练随机性；**scene bootstrap 95% CI 不覆盖 training seed、checkpoint 选择、多重比较**。不同独立训练模型生成不同 grasp candidates，AP/prefix 的变化不能直接断言为纯 reranking。`prefix_precision@K` 不是 candidate recall。

## 3. 已完成并经报告核验的正式结果（与新 P0-A/B/C 分开）

### 3.1 七组 e19 Official AP

单位 AP (%)，Original GraspNet 10% testing，seed 0：

| Variant | Seen | Similar | Novel | Mean |
|---|---:|---:|---:|---:|
| baseline | 64.4888 | 45.9789 | 18.3026 | 42.9234 |
| slot | 64.5002 | 46.1659 | 17.7229 | 42.7963 |
| volume_fixed | 66.1720 | 50.2445 | 18.0469 | 44.8212 |
| volume | 65.3287 | 49.9963 | 18.4697 | 44.5982 |
| **volume_rel** | **66.3975** | **50.6917** | **19.4638** | **45.5176** |
| volume_fixed_rel | 64.8909 | 49.3070 | 18.3212 | 44.1730 |
| volume_hybrid | 64.8502 | 48.7168 | 18.7832 | 44.1167 |
| **volume_inverse_hybrid** | **尚无正式评测** | — | — | — |

### 3.2 机制证据（不能过度解释）

1. `volume_rel−baseline`：Mean **+2.5942 pp**，scene CI **[+1.996,+3.194]**，74/90 scenes 正向；Similar 增益最大（约 +4.713）。
2. `volume−volume_fixed`：Mean **−0.2230 pp**，CI **[−0.754,+0.298]**。**“随 d 移动 probe 必然更好”没有得到独立支持**。
3. `volume_rel−volume`：Mean **+0.9194 pp**，CI **[+0.484,+1.369]**；但是 `volume_fixed_rel−volume_fixed` 为 **−0.6482 pp**，CI **[−1.173,−0.125]**。原固定/动态 × relative-geometry 交互为 **+1.5676 pp**，CI **[+0.831,+2.319]**。
4. 三点路径（fixed RGB/fixed geom → fixed RGB/dynamic geom → dynamic RGB/dynamic geom）：`hybrid−fixed_rel` Mean **−0.0563**，CI **[−0.4308,+0.3268]**；`volume_rel−hybrid` Mean **+1.4009**，CI **[+0.9288,+1.8746]**。在 Novel，后者 +0.6806、CI **[+0.0021,+1.3532]**，下界脆弱。**不能说纯视觉内容独立带来该收益**，因为 RGB support、position metadata、validity mask 和空间共址一起变化。
5. Novel 严格摩擦系数 `μ=0.4`：`volume_rel−baseline` 仅 **+0.0096 pp**，CI **[−0.943,+0.902]**；`μ=0.8` 为 +1.5888，CI [+0.318,+2.873]。当前并未证明改善更严格接触质量。

机制假设：**action-aligned co-registration of RGB/geometry evidence** 可能是关键，但 *原因还没有从 feature、metadata、mask、训练权重与候选集合变化中完全隔离*。这正是 P0-A/B 的任务。

### 3.3 已完成 P1：Novel probe 几何可靠性诊断

初始按 AP 选择 5 scenes×4 frames=20 帧，存在 outcome-selection bias；后续改为 **AP-blind 固定 120 帧**：Novel scenes `0160–0189`，每 scene 预先固定 frame `0000,0080,0160,0240`，先冻结 selection hash 再读取 AP。

120 帧 × post-collision Top-20 grasps × 36 probes = **86,400 probe records**；rendered depth reference 有效 **31,734 (36.73%)**，rendered/sensor 双参考同 probe 有效 **28,328 (32.79%)**。共同有效的 rendered MAE **18.178mm**，sensor MAE **18.435mm**（注意这不是 `abs(rendered−sensor)`）。Role 有效覆盖严重不均：closing≈71.86%，approach≈22.74%，左右 contact≈45.4%／27.47%；**有效性与物理可见性或 contact success 不等价**。

旧 outcome-selected vs 新 AP-blind 120 帧，场景中心化的 “feature vs Frame ΔAP” 相关：

| 变量 | 旧 20 帧 | 新 120 帧 |
|---|---:|---:|
| local depth STD | −0.247 | **−0.033** |
| rendered depth MAE | −0.294 | −0.180 |
| rendered evidence coverage | +0.099 | −0.044 |

因此不应马上给 GVAR 加 local depth STD／gradient confidence gate。二者只是启发式 **proxy**，无外部校准 `sigma`，不能声称 uncertainty calibration。以后若做 uncertainty-aware evidence，先用未按 AP 选取的样本验证同 scene/role 的误差预测信息；不要把 UA-Depth 的 sensor-conditioned uncertainty 当作 RGB-only network inference 输入。

历史失败较集中的 Novel scenes 包括 **0165、0167、0174**；正向举例 **0160、0189**，但不能凭这些场景的展示选择证明全 Novel 成立。

### 3.4 最后一次真实服务器实验与代码来源

2026-10-12 08:28（香港）提交的 `GVAR_NEXT_P0_P1_FINAL_ZH.md` 明确报告 Hybrid 完成训练／三 split 官方 AP／P1 120 帧诊断，实际服务器执行代码 SHA **`59336c461ad40f5e6d8d759c08e1347b877435d3`**。旧部署另有修复 `625cc26`、`d32b7df` 等。该服务器执行版本不应与当前 GitHub 分支视为自动等同。

上述七组结果与统计有服务器报告及审计文件；**新 P0-A/B/C 尚无同等验收文件或最新 AP**，不要据此推测性能。

## 4. 工程资产、服务器路径和两套版本

- 仓库：[EconomicGrasp](https://github.com/rcao-hk/EconomicGrasp)。
- 当前实验分支：[exp/gripper-volume-action-reader](https://github.com/rcao-hk/EconomicGrasp/tree/exp/gripper-volume-action-reader)；最新 P0-A/B/C 提交锚点 [`ded3a32d`](https://github.com/rcao-hk/EconomicGrasp/commit/ded3a32d97bac94da1600a456f88a6a7732cfa66)。
- 服务器：`robotarm@10.30.7.117`，RTX 3090 24 GiB；**现在哪些 GPU 空闲、磁盘容量、后台任务是否正在运行均未知，必须重新查询**，不得沿用历史空闲状态。
- 原仓库：`/home/robotarm/EconomicGrasp`；先前受保护 GVAR worktrees：`/home/robotarm/EconomicGrasp-gvar`、`/home/robotarm/EconomicGrasp-gvar-p0p1`，以及 follow-up 隔离目录；新 P0ABC 建议 `/home/robotarm/EconomicGrasp-gvar-p0abc`（必须先 `git worktree list` 确认是否存在）。
- 数据：`DATASET_ROOT=/data/robotarm/dataset/graspnet`；`GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans`。
- **历史正式结果根目录**：`ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007`；不能从较早工作单里的 `/data2/…` 示例推断现有结果真实位置。
- 训练协议和 checkpoint：`$ROOT/train/{variant}/gvar_protocol.json`、`gntrans_mix_protocol.json`、`gvar_epochs.jsonl`、`checkpoint_epoch_019.tar`；正式 Original AP：`$ROOT/eval/{variant}_e19/test_seen|test_similar|test_novel/ap_test_*_realsense.npy`；每个 split 有 inference protocol/summary。
- P1 旧诊断：`$ROOT/analysis/novel_cases_20261010`、`probe_reliability_20261010`；120 帧分析：`novel_uniform_20261011`、`novel_uniform_depth_sidecars_20261011`、`novel_uniform_conditional_20261011`、`novel_uniform_vs_selected_20261011`；实际存在与 SHA 以服务器审计为准。
- 先前报告：本聊天附件 `/mnt/data/GVAR_result_20261009.md`、`GVAR_SCENE_PAIRED_RESULTS_ZH.md`、`GVAR_P0_P1_FOLLOWUP_RESULTS_ZH.md`、`GVAR_NEXT_P0_P1_FINAL_ZH.md`。附件在新会话未必自动挂载；上述统计已在本文件保留核心内容。

**重要的源代码衔接**：GitHub 上的模型接口／训练器可能不包含服务器已验证的所有 bugfix。不得直接把新的 Github 变体与服务器旧实验比较而不验证源码、checkpoint strict-load、原 `volume_rel` forward parity、3帧 decoder/dump parity、sampler hashes、参数量及 gradients。

## 5. 当前真正待完成：P0-A / P0-B / P0-C（代码已入 GitHub）

### P0-A — 同一个 `volume_rel` e19 checkpoint 的 Reader 内部反事实

**已提交代码**：`inference_gvar_p0.py`、`models/gripper_volume_reader.py` 的 `set_counterfactual`、`scripts/run_gvar_p0_eval.sh`、`analyze_gvar_p0a.py`、`audit_gvar_p0_support.py`、`utils/gvar_p0_analysis.py` 等。

| Mode | RGB feature 位置 | 位置 metadata | Attention validity mask | Relative geometry |
|---|---|---|---|---|
| R0 | Dynamic | Dynamic | Dynamic | Dynamic |
| R1 | **Fixed** | Dynamic | Dynamic | Dynamic |
| R2 | Dynamic | **Fixed** | Dynamic | Dynamic |
| R3 | Dynamic | Dynamic | **Fixed** | Dynamic |
| R4 | **Fixed** | **Fixed** | **Fixed** | Dynamic |

目的：判断已训练 reader 对 RGB content、位置 metadata、mask 的敏感性；并结合 AP-blind 120 帧的几何投影 `audit_gvar_p0_support.py` 测量 fixed/dynamic probes 的像素位移、`geometry_known but fixed-RGB-masked` 比例，按 gripper role 和 insertion bin 分层。

**当前状态**：代码/合成 CPU tests 已提交；**没有已验证的正式 R0–R4 AP**。必须先保证 R0 与未干预 reader 同固定输入时数值一致、相同 checkpoint SHA、depth 输入与 DPT depth 相符、R1–R4 finite。R4 的 *同权重* token 路径可以与 Hybrid 近似对应，但历史 Hybrid 使用另一个训练 checkpoint；绝不可把 R4 AP 差值等同于 Hybrid 重新训练的因果效应。

### P0-B — `volume_inverse_hybrid`（第四格）正式训练及 2×2 分析

**已提交代码**：`models/gripper_volume_reader.py` 新 variant `volume_inverse_hybrid`、`scripts/run_gvar_inverse_hybrid_train.sh`、`analyze_gvar_p0b.py`、测试 `tests/test_gvar_p0b_support.py`。

| | Geometry Fixed | Geometry Dynamic |
|---|---|---|
| RGB Fixed | `volume_fixed_rel` (44.173) | `volume_hybrid` (44.117) |
| RGB Dynamic | **`volume_inverse_hybrid`（待训练）** | `volume_rel` (45.518) |

目标：补齐 **RGB sampling strategy × Geometry sampling strategy** 2×2（改变位置时也改变相关 metadata/有效性条件）。主要交互定义：

`I = (AP_volume_rel − AP_volume_inverse_hybrid) − (AP_volume_hybrid − AP_volume_fixed_rel)`。

**当前状态**：支持实现、启动脚本、分析 pending guard 已提交，**未证实服务器正式训练或 e19 AP**。只能在 3 GPU、21 epochs、seed0 和旧修复对齐的条件下启动 fresh training；checkpoint e19，Original 三 split 官方 AP；分析脚本在缺结果时必须输出 `pending`，不能填估计数字。

### P0-C — Same preprocessing, RGB-only source swap：Original vs GN-Trans

**已提交代码**：`inference_gvar_p0.py` 中的 `PairedRGBDataset`，`utils/gvar_p0_sampling.py`（per-frame stateless seed），`check_gvar_p0_rgb_registration.py`，`scripts/run_gvar_p0_eval.sh`，`analyze_gvar_p0c.py`。

三个 **冻结 e19** 模型：`baseline`、`volume_fixed`、`volume_rel`。对每个 Original vs GN-Trans、Seen/Similar/Novel 运行，共 **3×2×3=18 split runs，每 run 780 frames**。用 **Original sensor depth 创建相同 workspace/crop、K、img_idxs、sample seed 和 collision cloud**，模型只替换 RGB；新协议要求两域各自都重跑，**不可拿历史 Original AP 对比新 GN-Trans AP**。除了图像 `img` 外的网络固定输入须逐帧相同；两域 checkpoint SHA 一致。

**最重要前置阻塞：人工审核原始 RGB 与 GN-Trans RGB 是否在同像素几何视场配准**。`check_gvar_p0_rgb_registration.py` 会画 Original 与 GNTrans 叠加原始 segmentation boundary 的图，机器可检查路径／尺寸，却不能自动证明物体轮廓对齐。如果不配准，**停止称“仅材质变化”**；要么调整设计并重新明确定义实验，要么如实记录 confound。旧 `inference_cva_gntrans.py` 使用 virtual depth 创建 crop/workspace，**不能直接与 Original pipeline 当作纯 RGB swap 比较**。

主要汇总：`MaterialGap(v)=AP_GNTrans(v)−AP_Original(v)`；`GapChange(v)=MaterialGap(v)−MaterialGap(baseline)`；每 split、各 μ、prefix 及 scene-paired CI。GapChange 改善不等于 Novel strict friction quality 绝对上升。

## 6. 当前卡点／真实执行状态

### 确认的事实

1. **GitHub 已提交 P0-A/B/C 源码**：`ded3a32d97bac94da1600a456f88a6a7732cfa66`，含 `experiments/GVAR_P0ABC_CODEX_20261012.md`、`tests/test_gvar_p0abc.py`、`tests/test_gvar_p0b_support.py`。上一会话有 CPU 合成测试通过的进度汇报，但**本次交接未在服务器重跑真实 CUDA／dataset／official AP**；应由新会话独立验收。
2. **缺 P0-A/C 完整正式结果；缺 P0-B inverse-hybrid 的训练与 AP**。不能把先前七组结果误认为新 P0-A/B/C 完成。
3. 服务器上 GitHub/本地已验证修复版本的差异尚需重新比对，**这是第一项硬门槛**。
4. GN-Trans RGB 与 Original 的物体轮廓是否严格对齐，**没有可核验的人工验收结果**，是 P0-C 科学解释的第二项硬门槛。
5. 当前 GPU/磁盘状态、是否有异步作业、是否已有新实验产物均未在本交接时实际探测；需要服务器检查，不能推断。

### 不是真实 blocker、但会诱发错误的状态表述

- `volume_fixed_rel` 已完成正式 AP **44.1730**（早期 10/10 报告中的 pending 已过期）。
- `volume_hybrid` 已完成正式 AP **44.1167**（10/12 早上已验收）。
- P1 120 帧分析已完成（不要无理由重复生成）。
- **未完成的是新 `volume_inverse_hybrid`，不是历史 `volume_hybrid`。**

## 7. 给新会话的执行步骤（按此顺序，不要跳）

**Step 0 — 不破坏已有工作区。** 在服务器检查用户/父目录 `AGENTS.md`、`git status --short`、`git worktree list`、分支 SHA、GPU 进程、`df -h /data /data2`，确认磁盘余量。曾经因 `/data` 仅 12 GiB 可用而暂停训练；后来释放空间，不代表现在仍充足。**不要自动删除历史 checkpoints/dumps、kill 其他进程、强制 reset/stash/checkout 用户工作树。**

~~~bash
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
git fetch origin exp/gripper-volume-action-reader
git rev-parse origin/exp/gripper-volume-action-reader
nvidia-smi
df -h /data /data2
~~~

**Step 1 — 新建或复用独立 worktree，移植已验证修复。** 以本文件上方的实际服务器版本 `59336c461ad40f5e6d8d759c08e1347b877435d3` 对比新远端源码；严格核对 E/Q/C detach、mixed sampler、dataset crop、width单位、训练 prefetch/DDP、官方 GraspGroup decode。旧 Volume-rel checkpoint strict-load；同帧新旧 forward 的 depth、center、view、CDF、width 及 decoded dumps parity 必须通过，否则暂停 P0。

~~~bash
# 仅示例：若该 worktree 目录不存在
git worktree add --detach ../EconomicGrasp-gvar-p0abc origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar-p0abc
python -m pytest -q tests/test_gvar.py tests/test_gvar_scene_paired.py \
  tests/test_gvar_p0_p1.py tests/test_gvar_p0_p1_followup.py \
  tests/test_gvar_next_p0_p1.py tests/test_gvar_p0abc.py tests/test_gvar_p0b_support.py
bash -n scripts/run_gvar_train.sh scripts/run_gvar_p0_eval.sh scripts/run_gvar_inverse_hybrid_train.sh
~~~

**Step 2 — P0-A 先 smoke 后正式，保存新结果目录**（下方是执行合同的入口，GPU ID 仅示例，先查占用）：

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
DATASET_ROOT=/data/robotarm/dataset/graspnet
GPUS=0 KIND=p0a SPLITS=test_seen MAX_BATCHES=2 RUN_EVAL=0 \
  OUT_ROOT="$ROOT/analysis/p0abc_smoke_<unique>" bash scripts/run_gvar_p0_eval.sh

GPUS=0,1,2 KIND=p0a BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
  OUT_ROOT="$ROOT/analysis/p0abc_<unique>" bash scripts/run_gvar_p0_eval.sh
~~~

同一套 official infer 与 AP；检查 `R0=r0` parity、同 checkpoint 5 modes、所有 AP 文件 finite shape、输入指纹、mask 冲突和 insertion-bin/probe-role 分层。**每个 mode 分配独立输出目录，禁止部分 smoke 进入官方 AP**。R0 新协议重新采样，不要无理由要求其与历史 Original AP 数值严格一致；forward parity 应在完全一致输入上比较。

**Step 3 — P0-C 先人工配准审查，再跑 matched domain 对照。**

~~~bash
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
python check_gvar_p0_rgb_registration.py \
  --dataset-root "$DATASET_ROOT" \
  --gntrans-rgb-root "$GNTRANS_RGB_ROOT" \
  --output-dir "$ROOT/analysis/p0c_registration_<unique>" \
  --scenes 100 130 160 167 174 --frames 0 80 160 240
~~~

人工审核通过、原始与 GNTrans 除 RGB 外的 model inputs parity、sample fingerprints 均通过后：

~~~bash
KIND=p0c GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
  OUT_ROOT="$ROOT/analysis/p0abc_<unique>" bash scripts/run_gvar_p0_eval.sh
~~~

启动前补齐环境变量 `DATASET_ROOT`、`GNTRANS_RGB_ROOT`、`ROOT`。18 jobs、每个 780 帧，CPU eval16 串行，最后 `p0c_analysis` 出 paired gap/change。如果检查失败，不要直接称已测材质鲁棒性。

**Step 4 — P0-B（会耗费训练 GPU，后排执行）**。先单 GPU batch3 smoke，再双卡 DDP smoke；确认三个 GPU 真空闲、磁盘足够、checkpoint parity 与旧样本选择 hashes 一致，才 fresh 训练：

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
GPUS=3,4,5 bash scripts/run_gvar_inverse_hybrid_train.sh

CKPT="$ROOT/train/volume_inverse_hybrid/checkpoint_epoch_019.tar" \
 GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
 DATASET_ROOT="$DATASET_ROOT" \
 OUTPUT_ROOT="$ROOT/eval/volume_inverse_hybrid_e19" \
 bash scripts/run_gvar_eval.sh

python analyze_gvar_p0b.py --root "$ROOT" \
  --output-dir "$ROOT/analysis/p0b_inverse_factorial_<unique>" \
  --bootstrap 50000 --seed 20261009
~~~

若训练无结果，分析脚本应该标识 `pending` 而不是输出补全 AP。预计同量级 Hybrid 一次正式训练约 8–9 小时（3 GPU），不能据此承诺准确耗时；曾见 rank0 peak allocated≈19.7GiB。

**Step 5 — 统一汇报，不发散到其他课题。** 交付 `GVAR_P0ABC_RESULTS_ZH.md`：P0-A 全部 R0–R4 AP/CI＋role/depth-bin mask/displacement，P0-B 完整四格+interaction（如完成），P0-C 两域 Original/GNTrans AP、`MaterialGap`／`GapChange`，全部关键 Novel μ=.4/.8、prefix metrics，训练/推理成本，输入/源 SHA、checkpoint/hash、失败/修复和配置。保留负结果。若任一部分尚未完成，明确列 pending，不可把三项合并宣称全部完成。

## 8. 已踩过的坑：绝对不要再踩

1. **不要混用两个代码来源。** 服务器真正训练七组的修复分支 != 当前 GitHub 实验分支；未经 checkpoint strict-load、forward+decoder parity 和 sampling hash 比较，不能将新 checkpoint 纳入严格实验矩阵。
2. **不要复用旧 Original AP 去和新的 GNTrans 推理比较。** 旧推理的随机点／像素抽样和虚拟深度裁剪协议不同。P0-C 必须在匹配的新协议下 **Original、GNTrans 双方重跑**；把 Original 的 sensor crop/K/img_idxs/collision 固定，只替换 RGB。
3. **不要把分辨率相等当作渲染图对齐。** GN-Trans 可能有物体轮廓或背景视场差异；先用 segmentation outline overlay 人工确认。若不配准，material-only 的科学解释不成立。
4. **不要让 GT、observed 或 virtual depth 进入 GVAR 网络。** 参考深度只能在离线评价/配准核验；sensor depth 用于已披露的原始 preprocessing/filtering。运行时检查 `depth_map_used_for_geometry==depth_net_pred`。
5. **不要让 reader 的 mask 错误丢弃几何证据或误判未知空间。** Hybrid 中固定 RGB validity mask 会影响动态 geometry tokens；R3/R4 用于敏感性检查。投影 valid ≠ 可见，深度未知 ≠ free，probe ≠ contact 真实标签。
6. **不要把动作支持区域与完整执行动作混为一谈。** 固定 support width=0.06m、reader 根据 insertion bins 构造；最终 width 独立预测。移动 center/insertion/width 后原标签可能不再是 exact-action label。不能复用错误动作标签或声明已解决 contact-quality mismatch。
7. **不要把 test-time R1–R4 OOD counterfactual 当作重新训练消融。** 即使同 checkpoint，解码后的候选动作和 ranking 可变化；不能声称纯固定候选 rerank。历史 Hybrid 和 R4 不是同权重训练模型。
8. **不要把数据／统计口径混淆。** 训练 10% Original＋10% GNTrans =5200/epoch，不是 20% Original；Original/GNTrans Seen/Similar/Novel 每 split 780；推理 `sample_interval=0.1`（fraction）对应 evaluator `--sample_interval 10`（stride）；AP NPY `[30,26,50,6]`。评测 worker16 **串行**，别启动 3×16 pool。bootstrap 以 scene 为单位，不是 26帧或每个 probe 独立；CI 不覆盖 seed 和多次选择。
9. **不要从 mean AP 推断严格抓取成功。** Novel μ=.4 净提升几乎为零。若要解决这个瓶颈，先做冻结物理动作的 candidate-oracle/width/contact/CDF 分解，而不是直接加评分 gate 或更多 36-probe tokens。
10. **不要把局部 STD／gradient 当作校准 uncertainty。** 在 AP-blind 120帧上与 ΔAP 的场景中心化相关约 −0.033，无 external sigma。不要未经 calibration 直接加入 uncertainty-aware loss/gate，更不要使用依赖 observed depth 的 UA-Depth sigma 在严格 RGB-only 网络推理中作弊。
11. **不要重复旧 Bug。** 检查 Width 是否在 mm→m 中重复换算；训练中检查 DPT 是否因 grasp gradient 非 detach 出现 checkerboard/near-constant depth；核对 DDP variable-length label placement、missing workers/prefetch、output heads CDF/width shape、CPU/CUDA smoke。
12. **不要搞坏用户工作。** 曾因 `/data` 空间不足导致 queue 等待；不能自行清理旧训练／失败证据、kill 非本任务进程、覆盖已有输出目录、自动 force-push/reset/stash 或合并 main。训练结果必须用独立目录、完整 protocol/manifest/checkpoint SHA 保存；失败日志保留。

## 9. 研究方向决策门槛（做完 P0-A/B/C 再决定）

- 若 P0-A 显示 **metadata 或 mask 改动**已可解释大部分 R0–R4 的下降，则下一步先做 metadata/mask 控制训练，而非把 Dynamic RGB 当成唯一来源。
- 若 P0-B 完整 crossed factorial 支持 **RGB/geometry 需要空间共址**，考虑带显式 pose/validity 的 **co-registered role-aware evidence fusion**；若 inverse-hybrid 与 volume_rel 接近，则固定 geometry support 的简化版可能足够。需要看 scene CI、Novel、摩擦 μ=.4 和运行成本，不能只看 Mean。
- 若 P0-C 在 material-only matched 输入下证实 `GapChange>0`，可讨论材质鲁棒性；若不成立，就不应只凭 Original Similar +4.713 AP 声称 material-agnostic generalization。
- 若 Novel μ=.4 仍无改善，先在相同物理 `(t,R,d,w)` 候选上做 exact-action evaluator/oracle gap、width/contact vs ranking 分解。GVAR 的 60 mm support envelope 与预测 width 不完全一致，可能是科学问题，但目前**只是待验证假设**。
- **暂不优先：**把 RayRoPE/MoGe/UA-Depth 直接混入 GVAR；继续加 probes；纯 depth STD confidence gate；扩大至 100% mixture；只围绕 checkerboard 重训。先完成受控信息归因，再批准新的大训练。

## 10. 新会话第一条指令建议

> 请先阅读仓库 `exp/gripper-volume-action-reader/GVAR_HANDOFF.md` 以及 `experiments/GVAR_P0ABC_CODEX_20261012.md`。先检查 GitHub 最新 HEAD 与服务器 `/home/robotarm/EconomicGrasp-gvar-p0abc`／历史训练代码的差异、GPU及磁盘状态、当前是否已有 P0-A/B/C 产物。**不要立即训练**。给出已执行／待执行／blocker 的三栏审计和前向 parity、GNTrans RGB 注册预审结果。之后按 P0-A smoke→AP、P0-C 先配准再 paired inference→AP、P0-B 单卡/DDP smoke→fresh 21epoch→e19 AP 的顺序完成，产出 `GVAR_P0ABC_RESULTS_ZH.md`。请严格保留原始评测合同和所有 checkpoint/manifest/hash，不碰 main 和用户其他工作区。

---

**来源优先级**：服务器最新 `FINAL_ACCEPTANCE.json`／实际训练和评测 manifests／真实 AP NPY & checkpoint SHA > 已交付 `GVAR_NEXT_P0_P1_FINAL_ZH.md` 等完整结果报告 > GitHub Codex 工作单 > 历史聊天里的进度语句。任何前后矛盾都先审计产物，不要凭印象补全结果。
