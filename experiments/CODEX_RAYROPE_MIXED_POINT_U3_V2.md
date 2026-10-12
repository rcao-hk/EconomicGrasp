# Codex 工作单：修正 Mixed Depth GT 后重训 RayRoPE Point 与 U3

> **任务**：在 `exp/moge-rayrope-grasp` 新版本中使用修正后的 RealSense 深度监督，仅重新训练 **Point（P1_VARIANT=U0）** 与 **U3（P1_VARIANT=U3）**，并完成配对深度诊断、训练、Seen/Similar/Novel 的 collision on/off 官方 GraspNet AP、定量比较。其余训练和测试配置与上一轮双源完整深度 U0–U4 实验一致。
>
> **范围**：不要启动 U1/U2/U4、MoGe、额外 random seeds、RS-only data scaling、KD 或新模块。不要覆盖旧 U0–U4 的结果或影响仍在运行的其他训练。

## 0. 本次唯一的数据政策修正

使用 **2600 RealSense + 2600 GN-Trans**，场景/帧配对：train scenes 0000–0099，每 scene 帧 `0000,0010,...,0250`。Depth 仅为 **训练监督**，仍然维持 RGB-only 前向网络和原有 crop/K/camera-pose 协议。

| 数据源 | RGB | Foreground GT（GraspNet seg>0） | Background GT（seg==0） |
|---|---|---|---|
| **RealSense** | GraspNet `scenes/scene_xxxx/realsense/rgb/xxxx.png` | `virtual_scenes/scene_xxxx/realsense/xxxx_depth.png`（rendered） | `tsdf_depth/scene_xxxx/realsense/xxxx_depth.png`（TSDF） |
| **GN-Trans** | `GN-Trans/scenes/xxxxx/xxxx_color.png` | `virtual_scenes/scene_xxxx/realsense/xxxx_depth.png`（rendered） | 同一 **完整 rendered depth** |

物体 rendered depth 无效时，RealSense **不能**使用 TSDF 补该前景像素；背景 TSDF 无效时不能回填 rendered background。GN-Trans 前/背景始终使用 rendered depth，不可替换为 GraspNet TSDF。

代码：`moge_rayrope/mixed.py` 的 `RealSenseRenderedForegroundTSDFBackground` 复用原 `GraspNetMultiDataset.build_fused_gt_depth_m`；`GNTransFullRendered` 保持原样。`DEPTH_CONTRACT.contract_version` 必须为 **2**。`train/protocol.json` 内的 `mixed_depth_supervision` 必须记录这一数据合同。

**历史版本必须保留**：

- 旧 Mixed Point：Mean AP **49.3792%**；RealSense rendered FG/TSDF BG、GN-Trans rendered FG/TSDF BG，且历史执行/推断实现可能不同。
- 上一轮新 U0（full TSDF RS / full rendered GN）：**48.417327%**。
- 上一轮新 U3（full TSDF RS / full rendered GN）：**48.986667%**。

这些数据只能作 **historical reference**，不能替代本轮新 Point、也不能把 AP 差异全部归因于一个 GT 改动。本轮仍需完整重新训练两组，不能复用旧 checkpoint。

## 1. 源码和服务器预检

代码仓库：`https://github.com/rcao-hk/EconomicGrasp`，目标分支：`exp/moge-rayrope-grasp`。执行前记录并冻结实际 commit；检查其包含修正的数据加载器、诊断及单元测试。

服务器参考：`/home/robotarm/EconomicGrasp`；`/home/robotarm/miniconda3/envs/grasp/bin/python`。远端原仓库/其他工作树有可能存在未提交修改和其他正在运行的训练，因此：

- 不在原仓库 `reset`、`stash`、`checkout` 覆盖改动；采用新的独立 worktree。
- 禁止占用 GPU4。`GPUS=0,1,2` 只是默认，必须先核查空闲；若已有任务占用，等候或使用三张其他允许的空闲 GPU，保持相同 world size=3、每卡 batch=3。
- 记录 Git SHA、DAV2 checkpoint SHA256、torch/CUDA、显卡、可用磁盘，确保训练固定到同一源码 fingerprint。
- 旧 `/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U0`、`U3` 及历史 `COMPARISON_CN.md` 保持不变。

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
git fetch origin exp/moge-rayrope-grasp
git rev-parse origin/exp/moge-rayrope-grasp
nvidia-smi
df -h /data2

# 如果目录已有内容，先检查，不覆盖。
git worktree add --detach /home/robotarm/EconomicGrasp-Mixed-PointU3-v2 \
  origin/exp/moge-rayrope-grasp
cd /home/robotarm/EconomicGrasp-Mixed-PointU3-v2
git status --short

export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export RESULT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_renderfg_tsdfbg_v2
```

先验证 `checkpoints/depth_anything_v2_vitb.pth` 可用且与前次实验权重相同（可用可信权重软链接）。不得因为独立 worktree 缺少权重而训练新的随机 backbone。

## 2. 运行数据诊断和代码测试（必须先通过）

要求先运行 `scripts/diagnose_rayrope_paired_depth.py` 的全部 **2600 pairs**。新诊断中 `all_/fg_/bg_` 前缀统计的是**真实合成后的 RS target 与 GN-Trans full-rendered target**；`raw_tsdf_vs_rendered_` 才是两个原始 depth source 的差异，不可混淆这两类指标。

```bash
cd /home/robotarm/EconomicGrasp-Mixed-PointU3-v2

P1_VARIANT=U0 PHASES=diagnose \
  WORK_ROOT="$RESULT_ROOT/depth_audit" \
  DIAG_MAX_PAIRS=0 DIAG_LOADER_AUDIT_PAIRS=8 \
  bash scripts/run_rayrope_mixed_p1.sh

bash -n scripts/run_rayrope_mixed_p1.sh scripts/run_moge_rayrope20.sh
"$PYTHON_BIN" -m pytest -q \
  tests/test_rayrope_mixed_p1.py tests/test_moge_rayrope.py
```

**验收条件：**

1. RS 文件必须同时存在 `gtdepthpath`（rendered）和 `fusedepthpath`（TSDF），GN-Trans rendered depth 与 RGB 路径存在；未发现错误配对或数据泄漏。
2. 抽查 8 对实际 `get_data_label`：通过每个样本的原生 `crop_box`（`x0,y0,x1,y1`）和最近邻 resize 重建 `gt_depth_m`；逐像素最大误差应在诊断阈值内。若两域 crop/K 不一致，报告原因，但不要强行逐像素比较不同 crop 坐标。
3. RS rendered foreground 与 GN-Trans rendered foreground 在同一 raw camera frame 下的 GT 来源应一致；背景保留 TSDF/rendered 差异。检查有效比例、MAE、P90/P99，并区分真实训练 target 与 raw reconstructions。
4. 新混合样本数应为 5200（每域 2600），Seen validation 为 780（RS rendered FG + TSDF BG）。如诊断失败，修复并重新审计后再启动 GPU。
5. 所有关键单元测试通过；尤其是 rendered/TSDF 像素选择、invalid 不回填、paired-source 必需文件和 U3 decoupled gradient。

## 3. 模型差异与统一的训练 / 测试配置

只运行：

| 名称 | 脚本 ID | RayRoPE | Depth head loss | Uncertainty |
|---|---|---|---|---|
| **Point** | `U0` | `point`（确定性位置编码） | 原 metric L1 | 无 learned sigma |
| **U3** | `U3` | `expected`（区间位置编码） | 同一 metric L1 | Learned Laplace halfwidth，**decoupled**，uncertainty loss 不反传进 depth |

除以上定义外，**全部保持相同**：

| 项目 | 统一配置 |
|---|---|
| Epoch / seed | 20 epochs，seed 0；fresh task/depth heads，不初始化于旧任务 checkpoint |
| DDP | 3 GPUs × batch/GPU 3 = effective batch 9；不使用 gradient accumulation |
| Optimizer | AdamW，LR=`3e-4`、WD=`1e-3`，cosine epoch learning rate，gradient clip=`1` |
| Backbone | Frozen DAV2 ViT-B；pose-aware depth `global_film`；MoGe OFF |
| Grasp | image-FPS 1024 seeds，A1 Top-1 view，CVA Transformer，CDF score + depth-wise width head |
| RayRoPE | 7×7 token grid；pixel radius 40；`RAY_APPLY_VO=1`，group chunk 32 |
| Depth / uncertainty | 原 depth 范围 0.2–1.0m；U3 halfwidth 1–80mm，固定参考 halfwidth 20mm，`INTERVAL_WEIGHT=0.1` |
| Labels | 原 compact `economic_grasp_label_300views_extend_angle_cdf_depth`；objectness/graspness/view/CDF/width loss 保持 |
| 梯度边界 | 所有 variant 保持 **grasp → numeric geometry** detach；U3 uncertainty loss 只更新 sigma head |
| Checkpoint | **最终 epoch 19**（第20轮）`checkpoint_latest.pt`；不用 test AP 或 validation AP 选择 epoch |
| Test | 原始 GraspNet RealSense RGB Seen/Similar/Novel，各 780 frames（30 scenes×26） |
| Inference | **batch size=3**；Top-1 view；不做额外 best-of-k 或 score calibration |
| Collision | on 与 off 都做；threshold=0.01，voxel=0.01m，approach=0.05m；用原传感器点云仅进行后过滤 |
| 其他 | `DEPTH_BIAS_MM=0`；不改变图像 crop / K / input / score decoding |

注意 DDP `DistributedSampler` 可能将 5200 帧填充到 **5202** 次 epoch sample-visits，以确保三张卡样本数相同；区分 `train_frames=5200` 和日志 `train.images=5202`，不把 padding 当成数据泄漏或额外数据量实验。

## 4. 小规模 DDP smoke（Point 和 U3 分开验证）

```bash
export GPUS=0,1,2                   # 用三张确认空闲的允许 GPU
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3
export GROUP_CHUNK=32
export SEEDS=1024
export SEED=0
export EPOCHS=20

for U in U0 U3; do
  P1_VARIANT="$U" PHASES=train \
    EPOCHS=1 MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 \
    WORK_ROOT="$RESULT_ROOT/smoke/$U" \
    bash scripts/run_rayrope_mixed_p1.sh || exit 1
done
```

逐项确认 dataset/labels/dataloader → GPU forward → DDP backward → val → checkpoint；记录非有限 loss、CUDA peak memory、task→depth/sigma gradient leakage（应为 0），确认 U3 sigma gradient 非零、DAV2 frozen。

如需修 bug，先在 smoke 阶段完成、固定同一新 commit，并记录与原 20-epoch 实验协议的差别；**开始正式训练后禁止修改任何被 `code_fingerprint()` 记录的源码**。smoke checkpoint 不可用于官方 AP。

## 5. 正式 20-epoch 训练、推断与 AP

**只有数据诊断 + smoke 通过后才能运行。** 五组 U0–U4 的历史目录不复用；本轮仅串行训练 **Point、U3**。

```bash
cd /home/robotarm/EconomicGrasp-Mixed-PointU3-v2
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python

export GPUS=0,1,2                   # 如被占用，统一换为三张其他允许的空闲 GPU
export ENCODER=vitb
export POSE_MODE=global_film
export USE_MOGE=0
export USE_RAYROPE=1
export SEEDS=1024
export RAY_GRID=7
export RAY_RADIUS_PX=40
export RAY_APPLY_VO=1
export GROUP_CHUNK=32
export FIXED_HALFWIDTH=0.02
export INTERVAL_WEIGHT=0.1

export EPOCHS=20
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3
export SEED=0
export LR=0.0003
export WEIGHT_DECAY=0.001
export GRAD_CLIP=1.0
export TRAIN_FRACTION=0.1
export EVAL_FRACTION=0.1
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

unset MAX_STEPS MAX_TRAIN_FRAMES MAX_VAL_FRAMES INFER_MAX_FRAMES INIT_CHECKPOINT GRAD_ACCUM
unset TRAIN_ROOT TEST_ROOT

for U in U0 U3; do
  echo "[FORMAL] $U started $(date -Is)"
  P1_VARIANT="$U" PHASES=train,infer,eval \
    WORK_ROOT="$RESULT_ROOT/formal/$U" \
    bash scripts/run_rayrope_mixed_p1.sh || {
      echo "[FAILED] $U: preserve all logs/checkpoints; do not start next run" >&2
      exit 1
    }
  echo "[FORMAL] $U finished $(date -Is)"
done
```

如果训练中断：只在同一 source/config/weights/data-root signature 下，针对对应 `WORK_ROOT` 使用 `RESUME=1 PHASES=train` 恢复；如果训练完整而 inference/eval 部分失败，只运行相应 phase。不要删 checkpoint、改模型后强制 resume 或绕过哈希。推断 batch=3 与之前全五组的协议一致；若 batch=3 异常，先定位原因，不单独给 U3 换推断批量。

## 6. 最终验收

两个运行目录（`formal/U0`、`formal/U3`）分别检查：

- `train/protocol.json`：`model.use_moge=false`、`model.use_rayrope=true`；train_frames=5200、val_frames=780、seed=0、epochs=20、3 GPUs×batch3、`mixed_depth_supervision.contract_version=2`；Point 的 encoding=point、U3 的 expected/learned/laplace_decoupled。
- `train/gradient_contract.json`：grasp→numeric depth/sigma 梯度为 0；U3 sigma loss 独立监督有效。
- `train/metrics.json`：20 个 epoch、finite loss、depth MAE、CDF metrics；末轮 checkpoint epoch=19、`partial_run=false`，不能按测试 AP 选 epoch。
- `test_native`：Seen/Similar/Novel 各 780 帧；collision-on/off 均完整；每帧 GraspNet `N×17`，无 NaN/Inf，on/off 从同一 forward 得到。
- `official/accuracy.npy` 每 split 形状 `(30,26,50,6)`，检查 completed manifests 与 SHA，AP 由 NPY 重算；官方 evaluator 参数一致。Mean AP 是三 split 平均（单位 %）。
- 推断 `INFER_BATCH_SIZE=3`；`depth_bias_mm=0`；collision 0.01/0.01m/0.05m。
- 输出所有标量和资源信息时区分训练指标、RS Seen validation 与官方测试，不将 RS validation interval coverage 宣称为 GN-Trans/Novel 的 calibration。

## 7. 分析问题与交付

以**新 Point（U0）**为直接基线，仅比较一个改动：引入 U3 的 learned uncertainty + expected interval encoding 与 decoupled Laplace NLL。U3−Point 是联合干预；因为 Point vs U3 不只有 uncertainty supervision 不同，不能将差值全部归因于单独的 sigma head。

报告结果：

| 新 mixed v2 | Seen AP% | Similar AP% | Novel AP% | Mean on AP% | Mean off AP% | ΔMean vs v2 Point |
|---|---:|---:|---:|---:|---:|---:|
| Point | 测量 | 测量 | 测量 | 测量 | 测量 | 0 |
| U3 | 测量 | 测量 | 测量 | 测量 | 测量 | 测量 |

同时报告 all-valid/foreground depth MAE，U3 training/RS validation coverage 与 mean halfwidth、CDF ranking/AUROC/AUPRC、训练时间、峰值显存、inference ms/frame；从官方 NPY 计算 scene-paired AP difference（可选 bootstrap），不要据此声称多 seed 显著性。

**历史参照（不参与因果归因）**：旧 Mixed Point 49.3792%、Full-TSDF RS 版本 U0 48.4173%、U3 48.9867%；新协议的两组结果尚未知晓，不能在文档里预填。

Codex 应在 `$RESULT_ROOT/reports/` 输出：

1. `DATA_AUDIT.md` — 实际两域 GT、native/crop 像素一致性、前背景 source discrepancy、26-frame×100-scene 检查、错误处理。
2. `EXECUTION.md` — 环境、commit、GPU、每组启动/恢复/停止时间、训练轮数、step 数、路径、资源。
3. `POINT_U3_COMPARISON_CN.md` — 正式 on/off AP、depth/CDF/uncertainty 指标、机制解读及限制。
4. `POINT_U3_RESULTS.csv`、`POINT_U3_ACCEPTANCE.json` — 从权威 raw NPY / manifests 核算的表格及逐项通过状态。

轻量报告可经审阅后 commit；禁止上传 checkpoint、NPY dumps、RGB 数据和大型日志。最终回复标明 Point/U3 是 DONE、PARTIAL 还是 BLOCKED，完整官方 AP 与结果路径；若 GPU/数据不可用，不得虚构成绩或称作完成。

## 8. 明确不执行

本轮不运行 U1/U2/U4、MoGe、B1/B2、100% RealSense 重训、多 seed、additional geometry/ranking head、超参 sweep 或 uncertainty shuffle。也不重新评价之前已经完成的五组 checkpoint：旧代码/数据合同与本次新协议不同，结果保持归档状态。
