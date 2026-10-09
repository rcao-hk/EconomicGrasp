# GVAR：Scene-paired AP Analysis — Codex 执行工作文档

状态：**工具代码已实现、通过合成数据回归测试；服务器真实结果尚待本任务执行。**
日期：2026-10-09。代码归属 `rcao-hk/EconomicGrasp` 的 `exp/gripper-volume-action-reader`，不修改 `main`。

## 1. 目标（不再训练或重做 inference）

利用 GVAR 四组**已完成**的官方 Original GraspNet e19 结果，做同 scene、同 frame 的配对比较，回答：

1. `slot` / `volume` / `volume_rel` 相对同一训练协议的 `baseline`，提升集中在哪些 Seen／Similar／Novel scenes？
2. 在以 **scene 为聚类单元**的 bootstrap 下，主要 ΔAP 和各 split 的 95% CI 是多少？
3. Novel 的 AP@μ=0.4、μ=0.8 是否方向一致？是否主要集中在少数 scenes？
4. Top-1/5/10/20/50 **prefix precision** 如何变化？这些结果不能直接归因于纯 ranking，因为各模型候选集合不同。
5. 对 `volume_rel − volume` 和 `volume − slot` 做**直接 scene-paired** 对照，而不只是报告它们各自减 baseline 的点估计。

**本任务只读**已有训练日志、protocol JSON 和 evaluator 的 AP NPY，不调用 CUDA、GraspNet evaluator、不触碰 checkpoint 内容、不创建新数据，不运行 GN-Trans 测试。CPU + numpy 即可。

## 2. 已知真实实验事实与路径

参考用户提交的 `GVAR_result_20261009.md`（截止 2026-10-09 14:06 香港时间）。

```text
服务器：robotarm@10.30.7.117
工作区：/home/robotarm/EconomicGrasp-gvar
部署代码分支（可能有未合并修复）：codex/gvar-deploy-20261007
GVAR 实验根目录：
/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
```

四个完整实验：`baseline`, `slot`, `volume`, `volume_rel`；`volume_fixed` **尚未训练，不可伪造／补零／跳过用户指定但不存在的 variant**。

训练：GraspNet Original 10%（2600）+ GN-Trans 10%（2600）；seed=0；3×RTX3090，每卡 batch3/global9；e0–e20，官方比较**固定 e19**；E/Q/C detach、`global_film`、A1、CDF/width。

最终评测：Original GraspNet test_seen/test_similar/test_novel，各 **30 scenes×26 frames**，frames `0,10,...,250`，Top-1 view，collision threshold=0.01，sensor collision cloud，evaluator 16 workers。

### 2.1 预期文件布局

在根目录下面：

```text
train/{variant}/gvar_protocol.json
train/{variant}/gntrans_mix_protocol.json

eval/{variant}_e19/test_seen/gvar_inference_protocol.json
eval/{variant}_e19/test_seen/gvar_inference_summary.json
eval/{variant}_e19/test_seen/ap_test_seen_realsense.npy

eval/{variant}_e19/test_similar/...
eval/{variant}_e19/test_novel/...
summary/gvar_ap.csv                     # 历史汇总，交叉核对而非输入依赖
```

每个正式 AP NPY 严格要求 `[30,26,50,6]`，所有数值 finite 且位于 `[0,1]`。
第 0 维对应按 scene 编号递增排列的 30 scenes；第 1 维按帧编号 `0,10,...,250` 排列。脚本以 manifest `frame_fingerprint` 和该 split 的 canonical scene-frame SHA256 双向审计对齐。**单纯数组 shape 一致不算对齐。**

## 3. 安全获取新分析脚本

首先检查原工作区，**不可覆盖或 reset 其他训练正在使用的目录**：

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin exp/gripper-volume-action-reader
git rev-parse origin/exp/gripper-volume-action-reader
```

强烈建议使用独立 worktree，仅消费分析脚本：

```bash
git worktree add --detach ../EconomicGrasp-gvar-scene-analysis origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar-scene-analysis
```

如果 worktree 目录已存在，先检查其 git status 和 commit；不要直接删除、重建或强制覆盖用户文件。仓库原始 `codex/gvar-deploy-20261007` 代码及所有训练结果保持原样。新分析脚本只依赖 Python/numpy，不依赖训练器当前版本的导入，因此可以从新的分析 worktree 直接读取旧输出。

新增脚本：

```text
analyze_gvar_scene_paired.py
experiments/GVAR_SCENE_PAIRED_CODEX_20261009.md
tests/test_gvar_scene_paired.py
```

## 4. 单元测试和预审

使用服务器上可用的 Python（numpy + pytest）；不改 CUDA 环境，不装新的 PyTorch：

```bash
python -m py_compile analyze_gvar_scene_paired.py
python -m pytest -q tests/test_gvar_scene_paired.py
python analyze_gvar_scene_paired.py --help
```

单测使用合成、但 shape 真实的 `[30,26,50,6]` AP，包括：正确 AP/μ/prefix 聚合、30-scene paired bootstrap、三 split 分层 Mean、`volume_rel − volume`、frame hash 错误、checkpoint 不一致、训练 sampler 不一致、missing split、NaN／shape 错误、拒绝误覆盖非分析结果等。

然后**只读**检查四组 `train/{variant}` 的 manifest、三 split 的 manifest 和 NPY 是否存在。若有缺失，先确认 `GVAR_RESULTS.md` 指向的文件，而不是自动把 incomplete variant 从表里删除。

## 5. 正式分析命令（默认四组，50,000 次 bootstrap）

```bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
OUT="$ROOT/analysis/scene_paired_e19"

python analyze_gvar_scene_paired.py \
  --root "$ROOT" \
  --output-dir "$OUT" \
  --variants baseline slot volume volume_rel \
  --baseline baseline \
  --epoch 19 \
  --bootstrap 50000 \
  --seed 20261009 \
  --ci 0.95 \
  2>&1 | tee "$ROOT/scene_paired_20261009.console.log"
```

若 `OUT` 已经存在，不覆盖未知文件。检查其内容属于该工具的输出后才可添加 `--overwrite`；更稳妥的方法是使用新的、有日期后缀的 `OUT`。

默认额外直接对照（自动生成，仍为配对 bootstrap）：

```text
slot → volume
volume → volume_rel
```

若后续 `volume_fixed` 的 e19 三 split、两种训练 manifest 全部到位，再使用全五组分析：

```bash
python analyze_gvar_scene_paired.py \
  --root "$ROOT" \
  --output-dir "$ROOT/analysis/scene_paired_e19_five_variants" \
  --variants baseline slot volume_fixed volume volume_rel \
  --bootstrap 50000 --seed 20261009
```

此时自动增加：`volume_fixed → volume`。缺任何文件会 fail-fast，不自动忽略。

如需其它直接比较，可加 `--contrast baseline:volume_rel` 或 `--contrast slot:volume_rel`（参数允许重复）。

## 6. 统计定义与数值单位

输入 `AP[scene,frame,rank,friction] ∈ [0,1]`。与仓库原 `summarize_gvar.py` 一致：

- Overall AP = `AP.mean() × 100`，输出为 **AP %**。
- AP@μ = 按该摩擦阈值所在的最后一维索引，平均所有 scene/frame/rank，再乘 100。
- Prefix precision@K = `AP[:,:,K-1,:].mean() × 100`，**不是 candidate recall**。
- Scene-level AP = `AP[scene].mean() × 100`；每 scene 的 26 frames 是一个 cluster，不作为 26 个独立 bootstrap 样本。
- Paired ΔAP = 对同 scene 同 frames 计算 `(variant - reference)`，再在 scene 维度求均值；单位 **百分点 pp**。
- 95% percentile bootstrap：每 split 从其 30 scenes 有放回抽取 30 个；对三个 split 的 Mean **分别**抽样、再取等权平均，不把 90 scenes 随意混作一个未分层总体。
- 同一批 bootstrap scene indices 在各 variants/指标中共用；seed 和 repetitions 写入输出审计文件。

这些 CI 只反映被评测 scenes 的抽样变化；**不包含**训练随机种子、checkpoint 选择、多组比较引入的误差，也不能证明改变的只是 ranking。不要把 CI 未跨 0 称为跨训练 seed 的显著提升。

## 7. 输出与验收

脚本输出：

```text
$OUT/REPORT.md                         # 表格、CI、Novel best/worst scenes、限制
$OUT/paired_summary.csv                # 对 baseline 的 AP + CI
$OUT/paired_metrics.csv                # 对 baseline 的全部 friction / prefix + CI
$OUT/scene_level.csv                   # 逐 scene AP 与各摩擦/prefix差值
$OUT/frame_level.csv                   # 逐 scene×frame mean AP (非独立样本)
$OUT/cross_variant_contrasts.csv       # volume_rel-volume等直接成对对照含CI
$OUT/cross_variant_scenes.csv          # 非baseline对照的逐scene AP变化
$OUT/input_audit.csv                   # manifest + AP files完整SHA256来源
$OUT/gvar_scene_paired_audit.json      # 参数、采样、checkpoint hash和来源
```

验收预期（默认四组）：`paired_summary.csv = 4×4=16` rows；`paired_metrics.csv = 4×12×4=192` rows；`scene_level.csv = 4×3×30=360` rows；`frame_level.csv = 4×3×30×26=9360` rows；`cross_variant_contrasts.csv = 2×12×4=96` rows；`cross_variant_scenes.csv = 2×3×30=180` rows；`input_audit.csv = 4×(2+3×3)=44` rows。

**严格核对**历史报告中的 e19 Mean AP（允许因浮点格式误差 0.001 AP pp）：

| Variant | 预期 Mean AP (%) |
|---|---:|
| baseline | 42.923436 |
| slot | 42.796338 |
| volume | 44.598192 |
| volume_rel | 45.517636 |

此外核对 `volume_rel - baseline ≈ +2.5942 pp`、`volume - baseline ≈ +1.6748 pp`、`volume_rel - volume ≈ +0.9194 pp`（仅作为复算验收目标，**不是预设 CI 的方向或统计结论**）。

若与上表不匹配，停止输出“分析已完成”的结论，定位具体 evaluator NPY、checkpoint hash、epoch/frame ID 或报告四舍五入错误；不得偷偷改动采样掩码、丢弃 frames 或对结果乘不一致的系数。

## 8. Codex 最终交付

工作结束后返回：

1. 服务器真实执行的 shell 命令、Python/numpy 版本、分析脚本 Git SHA，完整输入路径与 SHA256。
2. 四组 AP/ΔAP/各 split 95% scene-cluster CI，`volume_rel - volume` 的直接 CI 及 scene-win counts。
3. 各 friction μ 的变化，特别是 Novel μ=0.4 vs μ=0.8；prefix@1/5/10/20/50 变化。
4. Novel 最明显改善/退化各 5 个 scene ID，若需可从 `frame_level.csv` 找代表帧，但不能凭 AP 区分究竟是 ranking 还是物理 candidate 质量。
5. 各表实际 row counts、单元测试结果、所有文件路径，以及与 `GVAR_result_20261009.md` 四组 Mean AP 的核对结果。
6. 如有 blocker，明确缺失/冲突的文件和有效观察，不掩盖负结果，不擅自申请训练 GPU、重跑推理、修改 repo 主模型或变更指标定义。

可把最终整理后的中文总结保存为 `$ROOT/analysis/GVAR_SCENE_PAIRED_RESULTS_ZH.md`，附上产物链接／文件目录供后续读入。不要提交训练数据、checkpoints 或大批 AP dump 到 GitHub；代码及本文已在实验分支中维护。
