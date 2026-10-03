# Codex 工作单：MoGe + RayRoPE 在线 GraspNet20 实验

## 任务目标与交付

在 `rcao-hk/EconomicGrasp` 的 main 基线上，验证 affine-shape/metric-grounding 解耦和 grasp-frame RayRoPE，目标为预测 metric depth 存在误差时更好的 RGB 抓取表征。所有训练从新 task/geometry heads 开始，复用 frozen DAV2 encoder；没有 teacher/action feature cache，没有 gradient accumulation。

本次会话交付了直接源码，但没有可用的 GitHub 写动作，**远端分支尚未创建或 push**。不要将本文件或本地测试结果解读成 GPU 实验已经执行。Codex 负责在有仓库写权限及数据/GPU 的环境内完成发布、集成验证、运行、整理结果。

最终需要：

- 新分支、可复现 commit 与 source protocol；
- baseline parity 和各设计的 CUDA gradient-contract/smoke 结果；
- 四个正式模型的 training logs、metrics、epoch-20 checkpoint；
- Seen/Similar/Novel 的 collision-on 主 AP 和 off 诊断 AP；
- 由 NPY 核算的比较表、逐 scene 配对差异、运行成本；
- `RESULTS_CN.md`：实际结果、问题定位、是否有继续研究的证据，不虚构结果或统计结论。

## 1. 仓库与文件导入

目标基线：`52d09f925059bec3643610ecf1f1722894627ee5`。
新分支：`exp/moge-rayrope-grasp20`。

先检查工作树、现有任务、磁盘空间。不要 stash、覆盖或删除用户未提交改动。不要停掉正在运行的 P1 或其他实验。建议使用新的 git worktree。

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin main
git cat-file -t 52d09f925059bec3643610ecf1f1722894627ee5
# 分支不存在时，在 pinned main 基线上创建；不是从 MGF/P0/P1 分支派生。
git worktree add -b exp/moge-rayrope-grasp20 \
  /home/robotarm/EconomicGrasp-MoGeRayRoPE \
  52d09f925059bec3643610ecf1f1722894627ee5
```

将实现包中的实际 `.py`、`scripts/`、`moge_rayrope/`、`tests/`、`experiments/` 及此工作文档复制到新工作树的对应目录。实现包不是完整原仓库：它依赖 pinned main 原有 `models/`、`dataset/` 和 `utils/`。

**不导入 patch、apply 脚本、旧分支模型或历史 P0/P1 trainer。** 不覆盖 main 现有文件。完成后检查 `git diff`，只应有新实现文件。

运行目录必须是新 worktree 根目录。使用现有 `grasp` Python 环境。官方 DAV2 权重位于 `checkpoints/depth_anything_v2_vitb.pth`；可链接用户已有的可信权重，禁止把 checkpoint 权重提交进 Git。

CPU 检查通过后可提交源码：

```bash
git add moge_rayrope train_moge_rayrope.py inference_moge_rayrope.py \
  eval_moge_rayrope.py compare_moge_rayrope.py verify_moge_rayrope_baseline.py \
  scripts/run_moge_rayrope20.sh scripts/run_moge_rayrope_ablation20.sh \
  tests/test_moge_rayrope.py tests/ddp_mr_validation_smoke.py \
  experiments/MOGE_RAYROPE_DESIGN.md CODEX_MOGE_RAYROPE_WORKPLAN.md
git diff --cached --stat
git commit -m "Add online MoGe-style geometry and grasp-frame RayRoPE controls"
git push -u origin exp/moge-rayrope-grasp20
```

若写权限不可用，记录真实失败信息；不要声称 push 成功。若 main 在基线之后变化，不自动更新代码哈希去绕过接口检查；先单独 review/rebase 并重做 parity。

## 2. 必须核对的数据和环境

```
DATASET_ROOT=/data/robotarm/dataset/graspnet
LABEL_FOLDER=economic_grasp_label_300views_extend_angle_cdf_depth
SUITE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20
GPUS=0,1,2       # 仅在这些 GPU 当前空闲时使用
```

必要依赖沿用原仓库：PyTorch/CUDA、MinkowskiEngine、PyTorch3D、Open3D、GraspNet API 及原有 CUDA extensions。新增核心算法只使用 PyTorch/NumPy；测试使用 pytest/scipy。

已有 compact CDF/width 文件属于数据集标注的存储方式，本任务不重新生成它们，也不需要预计算网络 feature、predicted-action labels 或 teacher outputs。Fused-depth target 文件必须存在。核对 width payload 是既定 uint16 millimetres，matcher 转为 metres；decode 仍为 `clamp(1.2*width_pred/10,0,.1)`。

记录 `nvidia-smi`、Python/PyTorch/CUDA、Git SHA、权重 SHA、dataset/label folder 路径。训练 protocol 自动记录关键参数和源码 hash。

## 3. P0：先做正确性验证，不先跑 20 epochs

### 3.1 CPU tests（可在无数据机执行）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_moge_rayrope.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/ddp_mr_validation_smoke.py
```

交付时本地已有 **33 个 CPU tests passed** 和 **2-rank Gloo uneven-validation smoke passed**。这些不代表 CUDA 主模型已通过集成。

测试涉及：sampled weighted-L1 solver 与线性规划一致、gauge invariance、RoPE 区间期望 Monte Carlo 核验、zero-width→point、rigid-frame invariance、geometry detach、checkpoint chunk 一致性、空标注/无效 token、CLI/Bash。

### 3.2 flags 全关必须复现原模型

```bash
CUDA_VISIBLE_DEVICES=0 python verify_moge_rayrope_baseline.py \
  --dataset-root /data/robotarm/dataset/graspnet \
  --output /data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20/baseline_parity.json
```

检查 `MR_BASELINE_PARITY_PASSED`。对同样权重/输入，depth、center、view、width、CDF、loss、decoded grasps 必须只存在浮点级误差。

这是 reduced-seed correctness test，不报告 AP。若不通过，先修接口/坐标/tuple顺序，不能继续正式实验或把误差解释成“新网络更好”。

### 3.3 四种模型的 DDP epoch-boundary smoke

```bash
SUITE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20_smoke \
VARIANTS=baseline,moge,rayrope,moge_rayrope \
PHASES=train EPOCHS=1 BATCH_SIZE=3 GPUS=0,1,2 \
SEEDS=32 GROUP_CHUNK=32 \
MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 LOG_EVERY=1 \
bash scripts/run_moge_rayrope_ablation20.sh
```

每个 variant 必须完成训练→distributed validation→checkpoint 保存。确认：

- `grasp_to_geometry_max_abs` 为 0（允许预设数值容差，但不放宽成有效回传）；
- geometry/task gradient 非零有限；MoGe shape decoder 和 metric anchor 均非零；
- frozen DAV2 没有梯度；
- CPU object payload 不被 DDP 自动搬到 GPU；
- validation 各 rank 覆盖互不重叠的样本，无 padding duplicates；
- 没有 rank0-only 长验证造成 NCCL barrier timeout；
- 没有 NaN、inf、width 单位异常、near-constant 非预期 geometry collapse。

Cold-start 两步不能用来判断最终 AP。Smoke 目录与正式目录隔离；formal inference/eval 拒绝 smoke checkpoint。

任何修复都要写出原因、影响哪些实验与 commit；重跑受影响 smoke，不要静默改变 loss、detach 或采样后继续旧结果。

## 4. 正式主实验：只做四个独立组合

| Variant | MoGe | Ray grouping | Encoding | Interval |
|---|---:|---:|---|---|
| baseline | 0 | 0 | 原 CVA | 无 |
| moge | 1 | 0 | 原 CVA | 无 |
| rayrope | 0 | 1 | expected | fixed ±20mm |
| moge_rayrope | 1 | 1 | expected | fixed ±20mm |

共同设置：

```
20% train frames: 5200 (100 scenes, stride 5)
10% Seen validation: 780 (30 scenes, stride 10)
20 epochs, fresh task/geometry heads, seed 0
3 GPU x batch/GPU 3 = effective batch 9
AdamW LR=3e-4, WD=1e-3, cosine epoch LR, clip=1
Fused-depth supervision ON
DAV2 encoder frozen; numeric geometry -> grasp detached
no accumulation, no feature/action cache
latest completed epoch-20 checkpoint only
```

```bash
SUITE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20 \
VARIANTS=baseline,moge,rayrope,moge_rayrope \
GPUS=0,1,2 BATCH_SIZE=3 EPOCHS=20 \
PHASES=train,infer,eval COLLISION=both \
bash scripts/run_moge_rayrope_ablation20.sh
```

四个模型依次使用同一个 GPU pool。不要并行抢占相同 GPU，也不要把 baseline 的 batch/seed/fuse-depth 配置单独改掉。

若 OOM，优先降低 GROUP_CHUNK；它只应改变内存分块，不改变数学任务。确实需要降低 BATCH_SIZE 时，对整套主对照统一更改并记录新的 effective batch；不要引入 GRAD_ACCUM。

训练途中读取 `logs/train.log` 和 `train/metrics.json`。必须记录 geometry MAE（foreground 可用时单独汇报）、CDF AUPRC/AUROC、query regret/hit、shape loss、interval coverage（启用 learned 时）。不要据 Similar/Novel AP 选择 checkpoint 或修改超参数。

## 5. 机制消融：主实验通过后再运行

优先最小集合：

```bash
VARIANTS=attention_none,rayrope_point,moge_rayrope_point \
GPUS=0,1,2 PHASES=train,infer,eval \
bash scripts/run_moge_rayrope_ablation20.sh
```

- `attention_none`：与 ray grouping 相同容量/采样，但无位置编码；
- `rayrope_point`：相同 grouping、确定位置 RoPE；
- `rayrope`：加入区间期望编码。

不能只比较 baseline 与 expected 并宣称全部 gain 来自 uncertainty，因为 grouping 和支持区域也改变。

后续可选（不要默认全部展开）：

```bash
VARIANT=moge_rayrope_learned GPUS=0,1,2 \
bash scripts/run_moge_rayrope20.sh

VARIANT=moge_rayrope SHAPE_TOKENS=1 \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20/shape_tokens \
bash scripts/run_moge_rayrope20.sh

VARIANT=moge SHAPE_LOCAL_WEIGHT=0 \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20/moge_no_local \
bash scripts/run_moge_rayrope20.sh
```

这些训练参数必须在独立输出目录中。不能将 learned interval 宽度自动解释成已校准的不确定性；报告 coverage/width。不能因为新 feature 增加而把参数量变化归因成纯 RoPE 效果。

## 6. 推断、评价和恢复

模型配置从 checkpoint 读取，推断不能手动切换一个不匹配的 architecture。

```bash
VARIANT=moge_rayrope GPUS=0,1,2 \
PHASES=infer,eval INFER_BATCH_SIZE=1 COLLISION=both \
bash scripts/run_moge_rayrope20.sh
```

Collision-on 主协议：threshold .01，voxel .01m，approach .05m，原始 sensor cloud；它不作为 RGB 网络输入。Off/on 从同一次 forward 派生。

恢复：

```bash
VARIANT=moge_rayrope RESUME=1 PHASES=train,infer,eval \
bash scripts/run_moge_rayrope20.sh
```

恢复严格校验 source/config/code signature。若没有 `checkpoint_latest.pt`，不能假装继续第一个未保存 epoch。若代码变化导致拒绝恢复，先评估影响并新开 run 或制定显式迁移，禁止删掉校验继续。

### 可选 depth stress test

预先固定 ±5/±10mm bias，而不是在测试集选择最优 bias。先做 Seen，再按预算扩展。

```bash
VARIANT=moge_rayrope PHASES=infer,eval SPLITS=test_seen \
DEPTH_BIAS_MM=10 \
TEST_ROOT=/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20/moge_rayrope/test_zplus10mm \
bash scripts/run_moge_rayrope20.sh
```

该干预在 seed/view/group 前改变 predicted numeric depth，会改变执行动作。因此它是 system-level geometry-error stress，不是“固定动作、只改变观测”的严格反事实实验。保持 collision 配置不变，比较 AP degradation；不能把非零 bias 的最高 AP 替换 native 主结果。

## 7. 整理结果

```bash
python compare_moge_rayrope.py \
  --root /data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20 \
  --variants baseline,moge,rayrope,moge_rayrope --collision both
```

自动生成：

```
comparison/comparison.md
comparison/comparison.csv
comparison/comparison.json
```

Comparator 从每个 `accuracy.npy` 重新计算 AP，检查 summary 和训练预算，输出相对 baseline 的 scene-paired bootstrap。配对 scene CI 不是多训练 seed 的不确定性。

Codex 另外整理 `RESULTS_CN.md`，至少包含：

1. 实际 commit / environment / 成功与失败运行清单；
2. 主表（collision-on），次表（off），Seen/Similar/Novel/Mean；
3. MoGe 与 RayRoPE 各自的增益和 combined interaction，不能将效应简单相加；
4. metric/foreground depth MAE、aligned shape error、ranking diagnostics；
5. deterministic RoPE vs expected RoPE vs no-PE attention 的区别；
6. 训练时间/epoch、实测峰值显存、推断 ms/frame（batch、warmup、collision是否计入说明清楚）；
7. depth-bias degradation（若执行），并解释改变动作的干预边界；
8. 失败分析：metric anchor、shape gauge、标注匹配、width 单位、query output contract；
9. 下一步建议，基于实际证据，不将新模块名称当成贡献证明。

大 checkpoint 和 grasp dumps 保存在 `/data2`，不要提交仓库。提交轻量结果报告与经过 review 的源码修复；不要提交凭空填写的 AP 表。

## 8. 必须停止并先诊断的情况

- main file contract 不符；
- baseline parity 不符；
- grasp gradient 进入 geometry；
- 同一模型所有 decoded widths 几乎为零或明显单位错误；
- 数值错误或从未完成 epoch-boundary validation；
- formal AP 文件缺帧、NPY 和 summary 不一致；
- 发现一项对照使用了不同采样、额外训练预算或测试数据调参。

不要通过调大 NCCL timeout、删 mask/contract、增加数据泄漏、取消 detach 等方法掩盖问题。
