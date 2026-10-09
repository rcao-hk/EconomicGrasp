# Codex 工作单：RayRoPE + 新 Mixed-Depth 数据，运行 U0–U4

> **任务**：在现有 `exp/moge-rayrope-grasp` 分支上，先核查新双源 depth GT，随后按完全统一的训练和测试协议，**真正完成** U0–U4 五组 20-epoch 训练、推断、GraspNet 官方 AP 评价和中文结果报告。不要只做 smoke。
>
> 仓库：`https://github.com/rcao-hk/EconomicGrasp`  
> 代码基线（截至工作单建立时）：`7c2fa413c666be09502edb00d6380e1617ff0293`  
> 文档之外的 U0–U4 实现已在 `moge_rayrope/mixed.py`、`moge_rayrope/objective.py`、`scripts/run_rayrope_mixed_p1.sh` 等文件中。执行前重新核验 Git HEAD。

## 0. 不可改变的实验合同

**只允许变化**：(a) 与上一轮相比，训练 GT depth 来源改为下表；(b) U0–U4 指定的 RayRoPE 编码 / uncertainty loss。除此之外不得擅自增加训练 tricks、改变模型、训练预算、数据采样或评价方式。

| 项目 | 固定设置 |
|---|---|
| RealSense 训练域 | 2600 帧原 GraspNet RGB + **完整 TSDF depth**，来自 `tsdf_depth/scene_xxxx/realsense/xxxx_depth.png`；包括物体和背景 |
| GN-Trans 训练域 | 2600 帧 rendered RGB + **完整 rendered depth**；代码目前指向 `virtual_scenes/scene_xxxx/realsense/xxxx_depth.png`，需先验证确为对应的 rendered GT |
| 配对关系 | 100 个 train scenes，每域每 scene 26 帧（0,10,...,250），对应 scene/frame 一致，总样本 5200 |
| 验证集 | RealSense `test_seen` 780 帧，GT 仍为完整 TSDF，仅监控，不按验证指标挑 epoch |
| 模型 | DAV2/DINOv2 ViT-B，frozen pretrained backbone；RGB-only pose-aware metric depth `global_film`；`USE_MOGE=0` |
| Grasp | image-space FPS、`SEEDS=1024`、CVA + CDF（A1 / top-1 view）、原 width decoder / label matcher |
| RayRoPE | 7×7 token grid，`RAY_RADIUS_PX=40`，`RAY_APPLY_VO=1`，`GROUP_CHUNK=32`；其他参数沿用代码默认 |
| 训练 | seed=0；**20 epochs**；DDP 3 GPU × 每卡 batch=3，effective batch=9；无 gradient accumulation |
| 优化 | AdamW；LR=`3e-4`；weight decay=`1e-3`；epoch cosine schedule；grad clipping=`1.0` |
| 初始化 | 与之前 RayRoPE suite 相同：DAV2 pretrained，task/depth/uncertainty heads 正常重新初始化；**不能复用旧训练 checkpoint** |
| Gradient boundary | grasp loss **不能**更新 numeric depth/pointmap/sigma；U4 只改变 depth-supervision 梯度 |
| 正式 checkpoint | 固定 20 epochs，使用最终 `checkpoint_latest.pt`（zero-based epoch=19）；不早停，不根据测试 AP 挑 epoch |
| 测试 | **原 GraspNet RealSense RGB**，Seen / Similar / Novel 各 780 帧，每组固定同一 frame schedule |
| 推断 batch | **`INFER_BATCH_SIZE=3`**，对齐此前最佳 mixed 组，而不是通用启动器的默认 1；五组统一 |
| Collision | 同一次 forward 导出 collision-on 和 off；on: threshold=0.01、voxel=0.01m、approach=0.05m；与先前一样使用原始 sensor cloud 做**后过滤**，不能作为网络输入 |
| 其他 | `DEPTH_BIAS_MM=0`；无 Top-4 infer；`INFER_MAX_FRAMES=0`；no KD / MoGe / center-correction / extra action mining |

**重要解释**：旧 mixed Point AP=49.3792% 是历史参考，不能替代新 U0。因为新 U0 的 depth GT 改变，必须同样重训 20 epochs，才能公平比较 U1–U4。

## 1. U0–U4 配置（五组全部执行）

| ID | RoPE | h（沿 optical depth 的 halfwidth） | Uncertainty training | Depth branch |
|---|---|---|---|---|
| U0 | point | 不使用 | 无 | 原 metric L1 |
| U1 | expected | fixed 20 mm | 无 | 原 metric L1 |
| U2 | expected | learned 1–80 mm | 90% central interval score | 原 metric L1；interval loss 不更新 depth |
| U3 | expected | learned 1–80 mm | decoupled Laplace NLL | 原 metric L1；Laplace uncertainty loss 不更新 depth |
| U4 | expected | learned 1–80 mm | coupled Laplace NLL | coupled uncertainty-weighted loss 替换原 depth L1 |

对应现有脚本中的 `P1_VARIANT=U0...U4`。每个 ID 独立初始化、独立输出目录、相同优化预算。

**不要把 U4−U3 写成“仅去掉 detach 的作用”**：现有 `objective.py` 中，U3 保留原 L1 并增加 uncertainty loss，而 U4 替换 L1，且 loss 权重/有效像素归一化并非完全相同。U4 是完整 confidence-aware depth-regression intervention。保留该事实并在最终报告中如实解释。

## 2. P0：隔离服务器工作树，审计数据与代码

### 2.1 先看现有作业，不能影响正在跑的 100% RealSense

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
git fetch origin exp/moge-rayrope-grasp
git rev-parse origin/exp/moge-rayrope-grasp
nvidia-smi
df -h /data2
```

若 `/home/robotarm/EconomicGrasp-RayRoPE-P1` 已存在，先确认内容，不覆盖/删除；否则创建 detached worktree：

```bash
git worktree add --detach /home/robotarm/EconomicGrasp-RayRoPE-P1 \
  origin/exp/moge-rayrope-grasp
cd /home/robotarm/EconomicGrasp-RayRoPE-P1
git rev-parse HEAD
git status --short
```

使用已有 GraspNet 的 `grasp` Conda 环境和可信 DAV2 权重。检查并记录 Python/PyTorch/CUDA、commit、预训练权重 hash、GPU/内存和磁盘。前次服务器方案明确 GPU 4 不可用；默认 GPU 0/1/2 **仅在空闲时**使用，若被 100% RS 占用则排队或统一改用三张允许的空闲 GPU，不抢占现有训练。

### 2.2 验证 **实际 depth GT 来源**，不是只看文件名

当前源码预期路径：

```text
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans

RS RGB          GraspNet/scenes/scene_0000/realsense/rgb/0000.png
RS full TSDF    GraspNet/tsdf_depth/scene_0000/realsense/0000_depth.png
GN-Trans RGB    GN-Trans/scenes/00000/0000_color.png
GN rendered GT  GraspNet/virtual_scenes/scene_0000/realsense/0000_depth.png
```

**必须确认 `virtual_scenes` 深度与 GN-Trans rendered RGB 在前景、背景、相机参数、frame ID 上真实对应**；不能仅因 loader 命名为 rendered 就假定正确。如果 GN-Trans 有另一套实际配对 depth，核对后决定是否修正路径。若无法确定对应关系，暂停正式训练、报告阻断，不得静默 fallback 至 TSDF 背景。

检查这几项：深度单位/scale、TSDF 前景/背景有效区、rendered 深度背景、raw RGB/Depth 分辨率、`crop_box`、crop-adjusted `K`、有效 GT mask。保持旧 dataset 的 crop/workspace、grasp label matcher 等其余行为不变，避免新引入混杂因素。

### 2.3 运行 paired-depth diagnostic

```bash
cd /home/robotarm/EconomicGrasp-RayRoPE-P1

export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export GPUS=0,1,2

P1_VARIANT=U0 PHASES=diagnose \
  WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U0 \
  bash scripts/run_rayrope_mixed_p1.sh
```

验收：生成 `paired_depth.csv` 与 `paired_depth.json`，覆盖 2600 对；统计 all/foreground/background 的 GT valid fraction、valid intersection/IoU、overlap MAE/P90/P99、crop/K 差异。差异不必为 0，重点排除错误配对、单位错误和区域错位。

额外抽查 4–8 对 **实际 `get_data_label` 输出**：RS 的 `gt_depth_m` 是否来自完整 TSDF；GN 的 `gt_depth_m` 是否来自完整 rendered depth；明确 network forward 不读取 sensor/TSDF/rendered depth。必要时对比 crop 后像素，不可仅比较原图 paths。

## 3. P1：测试和所有 U-ID 的最小 DDP smoke（先于正式训练）

```bash
cd /home/robotarm/EconomicGrasp-RayRoPE-P1
"$PYTHON_BIN" -m pytest -q tests/test_rayrope_mixed_p1.py
"$PYTHON_BIN" -m pytest -q tests/test_moge_rayrope.py

# 五组分别在隔离目录完成 2-step training + 小规模 validation
for U in U0 U1 U2 U3 U4; do
  P1_VARIANT="$U" PHASES=train EPOCHS=1 \
    MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 \
    WORK_ROOT="/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_smoke/$U" \
    bash scripts/run_rayrope_mixed_p1.sh || exit 1
done
```

必须证明：
- 五组均按 9+9 balanced train samples 正确加载新两个 GT 源（总 18），完成训练、DDP validation、checkpoint，loss/gradients finite。
- `gradient_contract.json` 的 grasp→geometry 最大梯度为 0（容许数值误差），DAV2 frozen；U2/U3 sigma head 确有监督梯度；U4 的 sigma 与 metric-depth decoder 都有有限非零监督梯度。
- U0 配置 point；U1 固定 h=20mm；U2/U3/U4 learned h∈[1,80]mm。
- 所有 U-ID 统一模型/CDF label contract；没有改变 top-1 / crop / support / loss 除 U-ID 定义外的项目。
- 现有原 depth L1 的 valid mask 使用 [0.2,1.0] 闭区间，而 U2–U4 confidence/interval 路径使用 (0.2,1.0) 开区间。正式运行前统计边界像素数量并审计影响；如必须统一，先加测试、同一提交一次性修正五组，不能在部分组训练后再改。
- 用相同模型、少量固定帧单独核查 inference batch=1 与 batch=3 的 decoded grasps/score 一致性；如果不一致，先定位 batch-dependent RNG / sampling，不能默认 batch size 不影响 AP。正式五组继续锁定 batch=3。
- 若 smoke 失败，只允许修正确认过的中性实现 bug，记录问题、影响、diff、修复 commit，并重跑相关测试。**代码修复完成后冻结 source fingerprint，再开始正式训练**。不要关掉 asserts 绕过失败。
- smoke checkpoint 不能用于正式 AP。

## 4. P2：统一参数启动完整训练 + 推断 + 评价

在 GPU 空闲、数据诊断和五组 smoke 通过之后，固定正式环境（避免任何临时 smoke 限制遗留）：

```bash
cd /home/robotarm/EconomicGrasp-RayRoPE-P1

export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export GPUS=0,1,2

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
export SEED=0
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3
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

# 逐个执行，不并行争用 3 GPU
for U in U0 U1 U2 U3 U4; do
  echo "Starting $U at $(date -Is)"
  P1_VARIANT="$U" \
    WORK_ROOT="/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/$U" \
    PHASES=train,infer,eval \
    bash scripts/run_rayrope_mixed_p1.sh || {
      echo "FAILED $U: preserve logs and checkpoints; stop queue" >&2
      exit 1
    }
  echo "Completed $U at $(date -Is)"
done
```

**正式要求**：所有 U-ID 用同一 `GPUS` 集合、DDP world_size=3、effective batch=9、20 epochs、相同 eval batch=3。新 run 绝不能与 smoke 目录或历史 run 共用输出。若正式训练中途出现错误，必须保留日志、源 hash、checkpoint、错误栈；不要调 LR/损失权重等掩盖。

**恢复与分阶段执行**：`RESUME=1` 仅允许在原 commit、protocol、数据路径、world size 等完全一致时使用。

```bash
# 训练中断：恢复 U2 训练（不要更改配置 / 源代码）
P1_VARIANT=U2 PHASES=train RESUME=1 \
  bash scripts/run_rayrope_mixed_p1.sh

# 若训练已完成，仅推断+评价：
P1_VARIANT=U2 PHASES=infer,eval RESUME=1 \
  bash scripts/run_rayrope_mixed_p1.sh
```

注意：通用 `run_moge_rayrope20.sh` 默认 `INFER_BATCH_SIZE=1`，本轮必须显式锁定为 **3**。若 batch 3 OOM，应先排除资源竞争和调整**只影响分块的** `GROUP_CHUNK`；若最终必须统一降低 inference batch，需五组全部统一重评并另记结果，不能单独对某组变更，也不能宣称与历史 batch-3 结果完全同协议。

## 5. P3：严格验收、统计与结果交付

### 5.1 每个 U-ID 需核对的证据

目录基准：`/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U*/`。

1. `train/protocol.json`：`train_frames=5200`，`val_frames=780`，`epochs=20`，`seed=0`，`effective_batch=9`，`mixed_depth_supervision` 正确，`use_moge=false`，`use_rayrope=true`，U-ID 对应 encoding / uncertainty / loss。
2. `train/sampling.json`：RS 和 GN-Trans 各 2600 帧、配对 scene/frame，Seen validation RS 780，训练不含测试 scene。
3. `train/gradient_contract.json`：task→geometry/sigma 无泄漏；U2–U4 confidence 训练有效，U4 depth loss coupling 有效。
4. `train/metrics.json`：20 个 epoch、loss/MAE 等 finite，无异常早停；`checkpoint_latest.pt` 的 `epoch=19` 且 `partial_run=false`。
5. GraspNet official evaluation：每个 split **780** 帧，`accuracy.npy` 形状 **(30,26,50,6)**；collision-on/off 的三 split 均完整。复核每帧 dump、completed marker、SHA、evaluation summary；不能把 smoke/partial 作为正式 AP。
6. Inference manifest：`batch_size=3`、`depth_bias_mm=0`、Top-1 view、collision threshold 0.01 / voxel 0.01 / approach 0.05；两种 collision 模式来自同一次 forward。
7. 训练和推断时间、GPU 峰值显存、checkpoint/source/model fingerprint 真实可追踪。

### 5.2 指标与比较

**主结果**：collision-on 的 Seen / Similar / Novel / Mean AP。  
**次结果**：collision-off 同样四项。  
**主要差值**：U1−U0、U2−U1、U3−U2、U4−U3；最后一项必须描述为 *loss replacement + gradient coupling* 的联合干预。

再整理 depth MAE（all-valid / foreground）、U1–U4 的 empirical coverage/mean interval width、U2–U4 的上限饱和率、CDF ranking regret / AUROC / AUPRC、可用时的 risk–coverage/AUSE，以及 20-epoch 时间、推断 ms/frame 和显存。

注意现有 `compare_moge_rayrope.py` 的默认比较名称可能仍采用早期 20% RS suite，并且可能把 `baseline` 而不是新 U0 作为配对 reference。请修改**仅报告/统计脚本**（不得在正式训练中途修改模型/训练源码），明确以新 U0 为对照，按 checkpoint metadata 而不是目录名校验五组协议，并从 official accuracy.npy 重新计算 AP。

**域分开评估**：现有 validation 为 RS Seen；不要把 RS 上的 interval coverage 宣称成 GN-Trans coverage。必要时增加**只读** GN-Trans paired validation / residual 脚本，不改变训练或 checkpoint 选择。TSDF vs rendered 的监督误差是重要混杂因素，在讨论 U4 结果时单独说明。

建议按 scene 对同一测试图像做 paired AP differences，可做 scene-paired bootstrap；**单 seed 不代表训练随机性统计显著**。旧 49.3792% mixed Point 的 AP 只能作为 GT 监督不同的历史参照。

### 5.3 必须交付

在仓库 `experiments/rayrope_mixed_p1_runs/` 生成并提交**轻量报告**：

- `EXECUTION.md`：主机、git SHA、Python/PyTorch/CUDA、固定参数、每组命令、起止时间、状态、恢复与失败。
- `DATA_AUDIT.md`：TSDF/rendered 实际文件、full-depth target、background discrepancy、crop/K、路径确认和 paired-depth diagnostic 结果。
- `COMPARISON_CN.md`：完整 AP 主表/次表、每组 uncertainty 和 depth 统计、机制分析、资源成本、显著限制。
- `comparison.csv`：可复核数值及 split、collision mode、epoch、batch 的明细。
- `acceptance.json`：每个 U-ID 的数据/代码/训练/评测合同验收，包含 checkpoint hash、frame counts 和结果文件状态。

**禁止提交** checkpoint、完整 NPY grasp dumps、RGB 原图、大型日志或其他实验的文件。代码如需修复，在有审查的前提下提交并写明 commit SHA；报告不能伪造结果，未完成处标为 NOT RUN / PARTIAL / FAILED。

## 6. 给用户的最终回复必须明确回答

1. 是否确认 RealSense 原 RGB→完整 TSDF、GN-Trans rendered RGB→对应 rendered depth；两种背景 GT 实际相差多少。
2. U0–U4 是否各训练满 20 epochs，并用相同 batch 3 / 3 splits ×780 / collision on-off 完整评测。
3. 每组 Seen / Similar / Novel / Mean AP，以及相对**新 U0** 的差值。
4. Learned interval 相对 fixed 的作用、uncertainty quality、U4 对 depth/AP/训练稳定性的影响（不过度归因）。
5. 关键失败/限制、日志路径、轻量结果文档和最终 commit。

## 7. 不要遗漏的前置问题

- 是否真实使用 `GN-Trans rendered depth`：当前 loader 具体使用 `GraspNet virtual_scenes`，须由服务器数据证据支持，而非仅依据代码注释。
- `INFER_BATCH_SIZE=3` 必须显式覆盖通用脚本的默认 1。
- `U4` 不等于只关掉一个 detach；结果报告必须标记 loss objective 的其他变化。
- 不启动 RS-only B1/B2、MoGe 或新超参数搜索，不影响正在执行的 100% RS scaling。
- 先跑数据检查和 DDP smoke，再冻结代码并执行全量实验；遇到故障先记录证据，必要时修复后从独立实验目录重跑。
