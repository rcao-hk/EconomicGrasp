# Codex 实验工作单：Mixed v2 Point RayRoPE，50% RealSense + 50% GN-Trans

**Experiment ID:** `rayrope_point_mixed_v2_rs50_gn50_s0_e20`  
**Status:** `REGISTERED / NOT STARTED`（本文件仅注册待执行实验，不代表 GPU 训练或 AP 测试完成）。  
**Repository:** `rcao-hk/EconomicGrasp`  
**Branch:** `exp/moge-rayrope-grasp`  
**登记时 HEAD:** `9552af6883651a8a8eab1d3efcc2c02e000d0175`（执行前 fetch 并审计更新，不自动回退/覆盖）。

## 1. 科学目标和范围

在 **Mixed Dataset training v2 + Point RayRoPE** 下，将 GraspNet-1B RealSense 与 GN-Trans **各自的 train fraction 从 0.1 提升至 0.5**，保持其余训练和测试协议与已完成的 10%+10% Mixed-v2 Point 一致，测量 Seen / Similar / Novel / Mean GraspNet AP 的变化。

**仅执行一个实验：Point，即 `P1_VARIANT=U0`。** 不训练 U3/U1/U2/U4、MoGe 或多 seed；不加 KD、score calibration、collision head 等新模块；不改 depth GT v2 定义。不占用或终止其他正在运行的训练。

**50%+50% 的精确定义：** 不是总量 5200 帧的 50:50 分配，而是 **每域各抽取原训练域的 50%**，并使用完全配对的 scene/frame。当前 `paired_frame_indices()` 采用 per-scene stride：

| 训练项目 | Mixed v2 10%+10% | 本次 Mixed v2 50%+50% |
|---|---:|---:|
| RealSense train | 2600（26/scene，帧 0,10,…,250） | **12800**（128/scene，帧 0,2,…,254） |
| GN-Trans train | 2600 | **12800** |
| Total unique train samples | 5200 | **25600** |
| Seen validation | 780 | **780（不变）** |
| Official test | 每 split 780 | **每 split 780（不变）** |

训练为 100 个 train scenes、每个 scene 有 256 帧，所以每域 100×128=12800，而**不是 13000**。10% 的采样 indices 是 50% indices 的子集；训练 scene 集合不变。三张 GPU × 每卡 batch 3、PyTorch DDP sampler 按默认方式填充至 **25602 sample-visits/epoch**，理论上约 **2845 steps/epoch**，20 轮共 **56900 optimizer steps**。验收需从真实日志核查，不要把 DDP padding 误报为多出训练数据。

**因果边界：** 20 epochs 不变但 optimizer update 数量比旧 5200 样本训练约大 4.92 倍。因此这是数据量与训练步数一起扩大、固定 epoch 的 scaling 研究，不能称为严格保持相同步数的 data-only 消融。不得额外增加“equal step”实验，除非用户另行要求。

## 2. Mixed Dataset v2 的深度监督合同（不改变）

| 数据源 | RGB | Foreground GT (`seg>0`) | Background GT (`seg==0`) |
|---|---|---|---|
| GraspNet RealSense | 原始 RGB | GraspNet `virtual_scenes` rendered depth | GraspNet `tsdf_depth` TSDF depth |
| GN-Trans | rendered RGB | rendered depth | **完整 rendered depth** |

使用已有 `moge_rayrope/mixed.py` 中的 `RealSenseRenderedForegroundTSDFBackground` 和 `GNTransFullRendered`；`DEPTH_CONTRACT.contract_version==2`。无效像素不跨源补值。两域样本对应相同场景与帧，GN-Trans 不能使用 RealSense TSDF 背景。GT / sensor depth **只用于监督及外部碰撞后过滤，不得作为 RGB-only 网络输入**；原 crop/K/pose metadata、CDF compact labels 和 workspace preprocessing 保持一致。

训练 `TRAIN_FRACTION=0.5`；验证固定 `EVAL_FRACTION=0.1`。不要把测试帧加入训练，不用 Similar/Novel AP 调参。

## 3. 模型、训练和测试：全部沿用之前 Point 基线

| 配置 | 固定值 |
|---|---|
| 模型 | `P1_VARIANT=U0` -> `VARIANT=rayrope_point`；`USE_MOGE=0`、`USE_RAYROPE=1`、`RAY_ENCODING=point`、`UNCERTAINTY=fixed` |
| 深度 | Frozen DAV2 ViT-B，pose-aware metric depth `global_film`，原 metric-depth L1；无 learned sigma head |
| Grasp | image-FPS 1024 seeds；7×7 RayRoPE grid，radius 40 px；`RAY_APPLY_VO=1`、`GROUP_CHUNK=32`；CVA-CDF、A1 Top-1 view、原 width head |
| Gradient contract | Grasp loss 对 numeric depth detach，保留现有 checkerboard-safe 梯度边界 |
| Train | `EPOCHS=20`、`SEED=0`、fresh task/depth heads；3 GPUs × batch/GPU=3，effective batch 9；无 gradient accumulation |
| Optimization | AdamW，LR=3e-4、WD=1e-3、epoch cosine LR schedule、clip=1.0 |
| Checkpoint | 末轮 `checkpoint_latest.pt`，zero-based epoch 19；不选择最佳测试 AP、不 early stop |
| Official inference | RealSense `test_seen,test_similar,test_novel`，各 30 scenes×26 frames=780；`INFER_BATCH_SIZE=3`；无 depth bias |
| Collision | `both`：on 阈值 0.01，voxel 0.01m，approach 0.05m；off 不过滤；两种输出来自同一次 forward |

严格保持上轮 Mixed v2 Point 的 sampling/label/evaluator 逻辑。若显存紧张，仅允许在验证数学等价的条件下降低 `GROUP_CHUNK`；不要改变 batch、LR、epochs 或 proposal 数量。

## 4. Codex 必须先处理两个硬编码障碍

**当前代码不能直接通过设置环境变量完成本实验**：

- `scripts/run_rayrope_mixed_p1.sh` 中写死了 `export TRAIN_FRACTION=0.1`，会覆盖用户的 0.5；
- `train_moge_rayrope.py` 拒绝非 0.1 的 mixed train fraction，同时要求 eval fraction=0.1。

在**新的独立工作树**做最小改动，保留原来 10%+10% 的默认行为：

`scripts/run_rayrope_mixed_p1.sh`：

```bash
# old:
export TRAIN_FRACTION=0.1
# new:
export TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
# EVAL_FRACTION 始终固定为 0.1
export EVAL_FRACTION=0.1
```

`train_moge_rayrope.py` 将 mixed fraction guard 改为只接受 `train_fraction∈{0.1,0.5}`、`eval_fraction=0.1`：

```python
if a.gntrans_rgb_root and (
    not any(abs(a.train_fraction - f) < 1e-8 for f in (0.1, 0.5))
    or abs(a.eval_fraction - 0.1) > 1e-8
):
    raise ValueError("Mixed train fraction must be 0.1 or 0.5; eval fraction must be 0.1")
```

对 0.1 的原 protocol 做回归测试，对 0.5 做数据计数、signature 和 smoke 测试。**不修改** `make_mixed_dataset` 的 frame-sampling 算法，不改 `dataset/graspnet_dataset.py`、`model.py`、`objective.py` 或 evaluator。旧 checkpoint 因 trainer fingerprint 变化可能无法在新代码下推断，**禁止关闭 source hash 校验**；历史 AP 从旧结果文件读取。正式新 run 启动前应提交最小兼容 patch、记录冻结的 commit/SHA，并使用独立输出目录。

`scripts/run_rayrope_mixed_p1.sh` 的诊断 phase 也硬编码 `--fraction .1`；为节省工作量，50% preflight 请**直接调用** `scripts/diagnose_rayrope_paired_depth.py --fraction 0.5`，不要误以为旧 `PHASES=diagnose` 已检查 50%。

## 5. 工作树与实验注册目录

由 Codex 先确认真实 local repo（可参考服务器历史路径 `/home/robotarm/EconomicGrasp`，但以实际 `git rev-parse --show-toplevel` 为准）。禁止在已有活跃 worktree 中强制 checkout/stash/reset，不打断 100% RealSense 或其他任务；GPU 4 禁用。

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
git fetch origin exp/moge-rayrope-grasp
git rev-parse origin/exp/moge-rayrope-grasp
nvidia-smi
df -h /data2

# 仅当目标目录不存在时执行；存在则先检查，不覆盖
git worktree add --detach /home/robotarm/EconomicGrasp-Mixed50-Point \
  origin/exp/moge-rayrope-grasp
cd /home/robotarm/EconomicGrasp-Mixed50-Point
git status --short

export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export GPUS=0,1,2  # 仅在三张卡空闲时使用；禁止 GPU4
export WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_v2_point_rs50_gn50
```

`WORK_ROOT` 是**新实验唯一目录**，必须与旧 10% Mixed v2 Point、U3、此前 MoGe、U3 Halfwidth diagnostic 和 smoke 目录隔离。训练 `$WORK_ROOT/train`；推断 `$WORK_ROOT/test_native`；日志 `$WORK_ROOT/logs`。

记录实际 git commit、Python/PyTorch/CUDA、CUDA devices、DAV2 权重 SHA256、CPU/RAM/GPU、磁盘可用容量。原 Mixed v2 Point 10%+10% 20-epoch 约 13 小时；本次总训练帧数约是 4.92 倍，若吞吐量类似，**60–70 小时量级只作粗略排期估计，不保证实际耗时**。

## 6. P0 Preflight（先做，不得直接启动长训练）

```bash
bash -n scripts/run_rayrope_mixed_p1.sh scripts/run_moge_rayrope20.sh
"$PYTHON_BIN" -m pytest -q \
  tests/test_rayrope_mixed_p1.py tests/test_moge_rayrope.py

# 快速检查 GT composition / crop / K (64 pairs 仅是 smoke)
"$PYTHON_BIN" scripts/diagnose_rayrope_paired_depth.py \
  --dataset-root "$DATASET_ROOT" --gntrans-rgb-root "$GNTRANS_RGB_ROOT" \
  --split train --fraction 0.5 \
  --max-pairs 64 --loader-audit-pairs 8 \
  --output-prefix "$WORK_ROOT/preflight/paired_depth_64"
```

此外通过 `make_mixed_dataset(...fraction=0.5...)` 对完整 12800 个 paired indices 执行资产路径存在性核查，报告每域 `12800`、总 `25600`、每 scene 的偶数帧索引、与 10% 原 indices 的包含关系；跨多个 scene/时间位置抽查完整 `get_data_label` 输出与 CDF/width labels。不能只检查前 64 帧便宣称整个数据集没有缺失。

**DDP 两步 smoke**（必须独立于正式目录）：

```bash
TRAIN_FRACTION=0.5 P1_VARIANT=U0 PHASES=train \
  EPOCHS=1 BATCH_SIZE=3 MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 \
  WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_v2_point_rs50_gn50_smoke \
  bash scripts/run_rayrope_mixed_p1.sh
```

检查 CUDA loss/grad finite、grasp→geometry 梯度为 0、DAV2 冻结、数据源 v2、DDP val、checkpoint 签名；smoke checkpoint 严禁作为正式官方 AP。若源码必须修复，先完成、重新测试并冻结 commit；不要在正式训练中修改 fingerprint 范围内的文件。

## 7. 正式训练与官方测试

```bash
cd /home/robotarm/EconomicGrasp-Mixed50-Point
export P1_VARIANT=U0
export TRAIN_FRACTION=0.5
export EVAL_FRACTION=0.1
export ENCODER=vitb
export POSE_MODE=global_film
export USE_MOGE=0
export USE_RAYROPE=1
export RAY_ENCODING=point
export SEEDS=1024
export RAY_GRID=7
export RAY_RADIUS_PX=40
export RAY_APPLY_VO=1
export GROUP_CHUNK=32
export EPOCHS=20
export SEED=0
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3
export LR=0.0003
export WEIGHT_DECAY=0.001
export GRAD_CLIP=1.0
export WORKERS=2
export EVAL_WORKERS=1
export OFFICIAL_WORKERS=2
export SPLITS=test_seen,test_similar,test_novel
export COLLISION=both
export COLLISION_THRESH=0.01
export COLLISION_VOXEL_SIZE=0.01
export COLLISION_APPROACH_DIST=0.05
export DEPTH_BIAS_MM=0
export RESUME=0

unset TRAIN_ROOT TEST_ROOT MAX_STEPS MAX_TRAIN_FRAMES MAX_VAL_FRAMES \
  INFER_MAX_FRAMES INIT_CHECKPOINT GRAD_ACCUM

PHASES=train,infer,eval bash scripts/run_rayrope_mixed_p1.sh
```

恢复只能在相同代码、采样与源路径、world-size、模型初始化/optimizer protocol 下使用原生 `RESUME=1`，不能绕过 hash 或把 10% checkpoint 当作初始权重。正式选最终 `epoch=19`；禁止根据 Seen/Similar/Novel AP 选 checkpoint。

## 8. 验收、历史对照与交付

验收 `train/protocol.json`：`train_fraction=.5`、`eval_fraction=.1`、`train_frames=25600`、`val_frames=780`、`world_size=3`、`effective_batch=9`、`epochs=20`、`seed=0`、`model.ray_encoding=point`、`model.uncertainty=fixed`、`mixed_depth_supervision.contract_version=2`；最后 checkpoint `epoch=19`、`partial_run=false`，二十轮训练日志非有限值为 0，记录真实 `sampling_sha256`、`code_sha256`、DAV2 hash。

官方测试：Seen/Similar/Novel 各 **780** 张，collision **on/off**，原始 GraspNet RGB；每 split `accuracy.npy` shape **(30,26,50,6)**，逐帧 dump `N×17` 合法且 marker/manifest hash 可追溯。AP 从官方 accuracy 重新计算，以三个 split 算术均值为 Mean，collision-on 为主表，collision-off 为诊断。

**唯一直接历史性能参考**：Mixed v2 Point 10% RS + 10% GN，20 epoch，seed 0，collision-on Seen `71.022973`、Similar `56.258518`、Novel `20.416294`、Mean `49.232595%`；collision-off Mean `46.956956%`。本次需报告各 split 和 Mean 的 `ΔAP (pp)`、深度 MAE、CDF ranking、训练时长/显存及推断开销。旧 U3 Mean `49.367215%` 仅供 context，不能作为新 Point scaling 的主对照。

单 seed 不宣称统计显著；固定 epoch 与更大训练样本会同时增加 optimizer step count，不做 data-only 的因果归因。

Codex 输出真实状态 `REGISTERED / RUNNING / COMPLETE / BLOCKED`；在 `$WORK_ROOT/reports/` 写 `DATA_AUDIT.md`、`EXECUTION.md`、`POINT_RS50_GN50_RESULTS_CN.md`、`comparison.csv`、`acceptance.json`。轻量报告可审阅后提交至 `experiments/rayrope_mixed_v2_point_rs50_gn50/`。不提交大型 checkpoint、grasp dumps、RGB 文件或巨量日志；未跑的结果统一标为 `NOT AVAILABLE`，不虚构 AP。
