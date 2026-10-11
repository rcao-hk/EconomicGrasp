# GVAR 下一轮 P0/P1：Hybrid Reader 与 AP-blind Novel Probe Audit

日期：2026-10-11。执行仓库：`rcao-hk/EconomicGrasp`，目标分支：`exp/gripper-volume-action-reader`。

**这是服务器 Codex 可执行工作单，不代表 Hybrid 已完成训练或 P1 的 120 帧诊断已经运行。** 工作范围只包含下列 P0/P1，不要自行扩展到 MGF、RayRoPE、MoGe、UA-Depth 或大型 uncertainty-aware reader。

## 1. 科学问题、已知结果与动作定义

来自 `GVAR_P0_P1_FOLLOWUP_RESULTS_ZH.md` 的 e19 六组官方 AP：`volume_fixed_rel` Mean=44.1730，`volume_rel` Mean=45.5176；相同 Original GraspNet 10% test。两因素交互效应 Mean +1.5676 AP pp、scene-bootstrap CI [+0.8305,+2.3190]；**Novel 交互区间跨零**。Novel μ=0.4 尚未明确改善。

现有两端：

- `volume_fixed_rel`：appearance/probe 和 relative geometry 都在 `d_ref=25 mm` 对应位置读取，CDF 仍预测实际 insertion bin `d`。
- `volume_rel`：appearance/probe 和 relative geometry 都在实际 `d` 对应位置读取。

**P0 新增唯一主变体 `volume_hybrid`：**

```
query = (center, view, angle, actual insertion d)
RGB support: 36 role-tagged probes at d_ref=25 mm
geometry support: 36 role-corresponding probes at actual d
Geometry = action-relative predicted surface xyz / ray residual / valid flags
CDF and width: unchanged (B,T,Q,A,D), (B,D,Q,A)
```

代码分别计算 RGB probe 和 geometry probe 的 3D 坐标、相机投影、有效掩码。它们只按 **gripper region role + 采样序号**对应，**不是同一个 3D 点**；RGB 元数据仍对应固定 probe，geometry embedding 对应动态 probe。采用原 reader 参数、36 probes/stream、attention + depth embedding 与 CDF、width 监督，不引入 GT depth、GT width、oracle labels、center correction 或新的 loss。保留 E/Q/C detach，task loss 不得回传 numeric depth head。

新实验是一个**三点条件对照路径**：

```
Fixed RGB / Fixed geometry = volume_fixed_rel (44.1730)
Fixed RGB / Dynamic geometry = volume_hybrid (new)
Dynamic RGB / Dynamic geometry = volume_rel (45.5176)
```

它不是完整的两因素 RGB-support × geometry-support 2×2 factorial（缺 Dynamic RGB + Fixed geometry），不能把路径分解解释成无条件主效应。

需要检验的两个增量：

- `volume_hybrid − volume_fixed_rel`：固定视觉支持时，geometry 是否必须指向实际动作？
- `volume_rel − volume_hybrid`：实际 geometry 固定为动态条件后，RGB support 的动态变化是否还提供独立增量？

报告每个 split、Mean、六个摩擦阈值、Top-K、按 scene 聚类的 paired bootstrap CI；**特别报告 Novel μ=0.4 与 μ=0.8**。三个模型独立训练，candidate 集合可能不同，不要把 AP 差值宣称为固定候选评分效果。

## 2. 首要部署前提：服务器代码合并与公平性

最后一次已报告的服务器分析代码为 `codex/gvar-followup-20261011`（执行 commit `e06573d`），早期 GVAR 训练包含服务器本地修复，不能假定远端 `exp/gripper-volume-action-reader` 已经合入所有修复。**先审查，再在新 worktree 上 cherry-pick 或选择性移植必要的服务器修复；禁止在运行中的用户工作树直接 reset/checkout。**

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin exp/gripper-volume-action-reader
# 如果目标路径已存在，先检查再选择另一个工作树路径，不要删除旧目录
git worktree add --detach ../EconomicGrasp-gvar-hybrid origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar-hybrid
# 查日志中旧修复 SHA，确认所需的 train/loader/decoder 修复已应用
```

正式训练前：strict-load 旧 `volume_rel` e19 checkpoint，至少在 3 个固定输入帧上比对旧服务器版本与新工作树的预测 depth、CDF、decoded grasps 和 post-collision dump；旧模型路径不可因移植或新增 `volume_hybrid` 改变。检查 full detach、初始化、数据采样/optimizer 与旧六组实验一致。如有差异，停止并报告具体代码/指标差异。

通用结果根目录：

```bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
```

## 3. 静态测试与 P0 Hybrid GPU smoke

```bash
python -m py_compile \
  models/gripper_volume_reader.py select_gvar_uniform_novel.py \
  analyze_gvar_hybrid_p0.py compare_gvar_p1_cohorts.py
python -m pytest -q \
  tests/test_gvar.py tests/test_gvar_scene_paired.py \
  tests/test_gvar_p0_p1.py tests/test_gvar_p0_p1_followup.py \
  tests/test_gvar_next_p0_p1.py
bash -n scripts/run_gvar_hybrid_train.sh scripts/run_gvar_p1_uniform.sh
```

先 dry-run；再用**新 smoke 目录**分别单卡 per-GPU batch3、两 batch和双 GPU DDP 做有限 loss、所有 reader 参数反向梯度、numeric depth detach、shape `[B,6,Q,A,D]`、DDP all-reduce、checkpoint save/load、e19 等效输出结构及有效 width。smoke 可降 `ACTION_CHUNK`，但不能改变正式训练的候选集合、角度或插入深度 bins。

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
  DRY_RUN=1 bash scripts/run_gvar_hybrid_train.sh

DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GPUS=0 VARIANT=volume_hybrid BATCH_SIZE=3 MAX_EPOCH=1 MAX_BATCHES=2 \
ACTION_CHUNK=128 ACTIVATION_CHECKPOINT=1 \
OUTPUT_ROOT="$ROOT/smoke/volume_hybrid_20261011" \
bash scripts/run_gvar_train.sh
```

`VARIANT=volume_hybrid` 必须被新的 GVARConfig 与 train/eval 脚本识别；输入 RGB-only，depth supervision 仍存在，GT/reference depth 不进入 reader。烟雾实验目录不要用正式 output。

## 4. P0 正式训练、官方 AP 与直接增量分析

前置条件：旧版本 forward parity、单卡/双卡 smoke、3GPU 空闲、数据目录容量及协议检查通过。

**匹配此前六组的实际训练**：每卡 batch3，3 GPU/global9，Original 2600 + GN-Trans 2600，21 epochs（e0..e20），seed0，A1、global_film、top1 test、CDF/width，fused depth supervision，E/Q/C full detach；正式对比固定 e19。此前 server real runs 使用 `ACTION_CHUNK=2048`、no activation checkpoint，训练 workers=3、验证 workers=4/prefetch=1，优先保持一致并记录超出历史协议的差异。

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GPUS=3,4,5 ROOT="$ROOT" ACTION_CHUNK=2048 \
bash scripts/run_gvar_hybrid_train.sh
```

如果 GPU 3/4/5 不空闲，改用三张确实空闲 GPU，不要停止其他任务；GPU 数必须3。脚本不支持 resume/fine-tune 伪装为 fresh。OOM 优先调整 action chunk + activation checkpoint，并在报告中记录与旧配置的执行差异，不得直接改 batch/query/action 或使用旧 task checkpoint。

```bash
DATASET_ROOT="$DATASET_ROOT" \
CKPT="$ROOT/train/volume_hybrid/checkpoint_epoch_019.tar" \
GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT="$ROOT/eval/volume_hybrid_e19" \
bash scripts/run_gvar_eval.sh
```

严格核对三个 splits 各 780 帧，`[30,26,50,6]`，finite，checkpoint SHA256 和 canonical scene/frame fingerprints，sensor-collision on threshold=0.01；不可只报告 val loss。

**CPU 主分析（若 Hybrid 尚未完成，只输出 pending，不得伪造 AP）：**

```bash
python analyze_gvar_hybrid_p0.py \
  --root "$ROOT" \
  --output-dir "$ROOT/analysis/hybrid_p0_e19_20261011" \
  --bootstrap 50000 --seed 20261009 --ci 0.95
```

输出 `REPORT.md`、`STATUS.json`、`hybrid_path_effects.csv`、`paired/paired_metrics.csv`、`paired/scene_level.csv` 和完整来源 SHA256。分析器会核对 `volume_fixed_rel=44.1730`、`volume_rel=45.5176` 的历史均值（±0.02 AP pp）和**三份实际训练 checkpoint 的 SHA256**；错误或缺样本时停止，不把历史报告作为数据替代。

## 5. P1：非 AP 预选 Novel scenes 的 120-frame reliability

目标不是新的不确定性抓取训练，而是验证上一轮在 5个 AP-outcome-selected scenes、20帧得到的关联，在不以 AP 结果选帧时是否仍成立。**预注册固定帧 IDs：0、80、160、240；Novel scenes 0160–0189 共30 scenes×4帧=120帧。** 这些 frame IDs 都属于 10% 原始 GraspNet subset。选择时完全不读取 AP；之后才使用已存官方 e19 NPY 标注每帧 AP／μ=0.4／μ=0.8 delta。它是系统性抽样，不是随机 IID 样本，仍需按 scene 解读。

预期路径：

```bash
CASES="$ROOT/analysis/novel_uniform_20261011"
SIDECARS="$ROOT/analysis/novel_uniform_depth_sidecars_20261011"
PROBE="$ROOT/analysis/novel_uniform_probe_20261011"
COND="$ROOT/analysis/novel_uniform_conditional_20261011"
COMPARE="$ROOT/analysis/novel_uniform_vs_selected_20261011"
```

### P1-A：CPU 冻结 frame selector

```bash
ROOT="$ROOT" STAGE=select bash scripts/run_gvar_p1_uniform.sh
```

应输出固定 120 个 `selected_frames.json` 帧 identity、`novel_frames.csv`、`REPORT.md` 和完整 AP/manifest SHA256。`selection=uniform`，不是 best/worst。检查每个 scene 恰有 4 帧，且场景跨 30 个 Novel scene；若结果目录已非空，使用新的目录，不覆盖旧 20 帧诊断。

### P1-B：先独立单帧 smoke，再真实 GPU sidecar export

先用新的 `SIDECARS` 路径 `..._smoke` 和 `MAX_FRAMES=1` 做功能、crop/K/GT/sensor 与网络输入无泄漏检查。Smoke 和 full export manifest 不可混目录，否则 resume identity 不符。

```bash
# Smoke：输出到单独目录
ROOT="$ROOT" DATASET_ROOT="$DATASET_ROOT" STAGE=export \
  SIDECARS="$ROOT/analysis/novel_uniform_sidecars_smoke_20261011" MAX_FRAMES=1 \
  DEVICE=cuda:0 bash scripts/run_gvar_p1_uniform.sh

# 正式：120 frames，e19 的 volume_rel checkpoint
ROOT="$ROOT" DATASET_ROOT="$DATASET_ROOT" STAGE=export \
  MAX_FRAMES=0 DEVICE=cuda:0 bash scripts/run_gvar_p1_uniform.sh
```

必须核对 `checkpoint_epoch_019.tar` 的 SHA256 等于原官方 inference manifest；导出 `crop_rgb`, `K`, `pred_depth_m`, `gt_depth_m`, `sensor_depth_m`, `foreground_mask` 的同 448×448 裁剪坐标。进入模型的输入仍然只有 RGB/K/pose/indices；GT 与 sensor reference 仅用来计算诊断误差。不得把 observed-depth conditioned UA-Depth uncertainty 作为 strict RGB inference。

### P1-C：CPU probes/roles/reliability 与 frame AP

```bash
ROOT="$ROOT" STAGE=analyze bash scripts/run_gvar_p1_uniform.sh
ROOT="$ROOT" STAGE=conditional bash scripts/run_gvar_p1_uniform.sh
```

默认读取 archived Top-20 **post-collision** grasps，每个 36 个 probes：若所有帧都有20个 grasps，预计 120×20×36=86,400 probe rows；不足20时依据实际保存的抓取计数报告，不做造假补齐。要求 same pixel projection、同参考 validity mask、缺失参考记为 unknown。旧脚本会输出相同 probe 上的 sensor/rendered absolute error，role/scene 相关，及 scene-mean-centered frame/AP association。`local depth STD/gradient` 仍然只是 predicted-depth proxy，不是 calibrated sigma。

### P1-D：与既有 AP-selected 20-frame 诊断的描述性对照

```bash
ROOT="$ROOT" STAGE=compare bash scripts/run_gvar_p1_uniform.sh
```

对比上轮 `analysis/p1_conditional_20261011` 的 `P1_AUDIT.json`、`per_role.csv`、`ap_associations.csv`。输出两组 cohorts 的 role coverage、MAE、within-frame STD–error 相关、AP 关联变化及交集帧数量。**不可给独立样本 p-value**，两组 cohort 可能重叠；AP-selected 的 20 帧仍具有结果选择偏差。

## 6. 验收交付

Codex 返回并保存到新分析根目录：

1. Git HEAD、模型改动的 diff、是否包含服务器实际修复、旧 checkpoint full-forward parity 证据；旧五组独立检查不回归。
2. P0：Hybrid e19 AP／原始 NPY hash、3端直接效应及其配对 scene bootstrap CI、Novel μ=.4/.8 和 train/infer 时长及显存。若任务尚未完成只报告 pending。
3. P1：120 个预注册 frame list+hash、每 scene probe 覆盖率／rendered vs sensor matched MAE／STD/gradient Spearman、scene-centered AP correlations、与上一轮 selected cohort 对照；并注明实际有效 probes 数、诊断失败原因及缺失的 external sigma。
4. 记录实际运行命令、环境、CPU/GPU成本、单元测试与 smoke 日志、input checksums。保留 negative results、status manifests 和完整 AP tensors，不覆盖用户训练产物，也不擅自 push 修复主分支。

禁止根据结果挑选帧、改 checkpoint、增加新训练目标、把 scene-bootstrap CI 当多 seed CI，或者把关联解释成因果效果。
