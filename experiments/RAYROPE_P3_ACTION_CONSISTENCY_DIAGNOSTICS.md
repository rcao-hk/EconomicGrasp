# RayRoPE P3 — Action-Consistent Insertion Selection Diagnostics

> **独立的新一轮工作文档；不覆盖 P1/P2 工作单。**
>
> Repository: rcao-hk/EconomicGrasp  
> Branch: exp/rayrope-p1-p2-diagnostics  
> Previous: experiments/RAYROPE_P1_P2_DIAGNOSTICS.md  
> Input evidence: RESULTS_CN_analysis_v3.md (2026-11-07, server 117)  
> Status: **代码已提交；新一轮117服务器结果尚未运行/核验。**

## 0. 科学问题与第三轮依据

第三轮汇总报告：288个 action families 的 fixed-width insertion sweep，mean oracle=0.3466、mean regret=**0.1140**、46.53%的 family 在改变 insertion 后质量发生变化；width sweep 的 mean regret=0.0075。随机 nonwinner 的平均质量低于 per-query winner 0.1528。Rank-1 mean Q=0.4352，Top-2 mean Q=0.4954（从而推得 rank-2 mean Q≈0.5556）。

这些属于**上一轮报告数值**，不是本工作单的新增测量。

当前 insertion 质量评估固定执行宽度 `w_native`，但每个深度 bin 的模型预测宽度可能是 `w_pred(d)`。因此 `score(d)` 与被评估的完整动作并不总一致。

本次 P3 优先回答：

> 同一 query、同一 angle、同一个 depth-bin CDF score，下，用逐深度预测宽度而非固定原生宽度执行时，原来的 0.114 insertion regret 还能剩下多少？

### 两个严格匹配的物理动作协议

```text
FIXED:     g_d = (center, rotation, insertion_d, native_selected_width)
PREDICTED: g_d = (center, rotation, insertion_d, depth_bin_predicted_width_d)
```

二者共享完全相同的四个 depth bins、CDF score、query、angle、RGB、预测深度、checkpoint、evaluator。只有非 selected depth 的物理宽度允许不同。

**特别注意：** CDF score 和 `w_pred(d)` 只是同一 depth bin 的两个输出；现有 Point RayRoPE score head **并未以最终执行宽度为显式输入**。因此本轮是 score/decoded-width protocol 的一致性诊断，不能仅凭它宣称实现了 width-conditioned scoring。

定义：
\[
R_\text{fixed}=\max_d Q(g_d^\text{fixed})-Q(g_{d^*}^\text{fixed}),\qquad
R_\text{pred}=\max_d Q(g_d^\text{pred})-Q(g_{d^*}^\text{pred}).
\]

必须逐 family 核验：

- 两种协议的原生 selected action ID、执行 width、质量完全相同。
- 对任何同一 depth，center、rotation、insertion、score 完全一致。
- 因 selected 质量不变，应有 `R_pred - R_fixed = oracle_pred - oracle_fixed`（在数值容差内）。

**统计解释红线：regret 降低并不表示选择策略更好。** 两种宽度协议的 selected action 本来就相同，因此 `R_pred` 降低可能仅意味着其他候选在新宽度下变差、oracle 降低。必须并列报告 selected Q、oracle Q、regret，以及相同四个 depth bins 下 CDF score vs exact quality 的 pairwise agreement（排除质量 ties，score ties 记半分）和 Spearman。

## 1. 诊断任务

### P3-A：fixed-width vs depth-predicted-width（最高优先级）

输出配对 `R_pred−R_fixed` 与 scene-bootstrap 95% CI、原生 selected Q 与 oracle Q、最优 depth 是否改变、每族执行 width 差值。Scene 为 bootstrap 单位，Seen/Similar/Novel 内分别重采样，再对 split 等权平均。禁止按 candidate 行 bootstrap。

### P3-B：Depth failure attribution 与 same-angle hard negatives

在相同 (center, view, angle) 下：
- 按 1/2/3/4cm bin 统计 score、exact quality、collision、empty、force-closure failure、六个摩擦阈值的成功率。
- 找出 score 最高的 depth `d*`、评价质量最优的 depth `d_oracle`、以及除 `d*` 之外**分数最高**的 hard-negative depth。
- 记录四个 depth bins 的质量可比较 pair 数、score-quality pairwise agreement、质量 ties 与同 angle Spearman；不能将四个全零质量动作算作四次正确排序。
- 输出 score margin、hard-negative 与 selected 的质量差、被错选动作的失败类型、selected→oracle 的类别转移、按类别分配的 regret mass。
- **不要**将不同角度的随机 nonwinner 混作同角度的 depth hard negative。

### P3-C：Across-query Rank-1 vs Rank-2

来自同一帧同一批 `query_winner`：
\[
\Delta Q_{2-1}=Q(\mathrm{rank2})-Q(\mathrm{rank1}).
\]

输出 95% scene-paired CI、rank-2 更好/更差/tie 比例、Top-1/Top-2 失败原因与六个 μ 的成功率差。不要把 Top-2 的平均质量直接当作逐帧 paired 检验。

### P3-D：Width prediction vs 常数宽度

从同一物理 (center,R,d) 的 `width_sweep` 严格匹配：
- 原生 predicted width；
- 固定 0.08m；
- 固定 0.10m；
- within-grid oracle。

按 family 输出 `Q(pred)-Q(0.08)`、`Q(pred)-Q(0.10)`、各自 regret、scene CI。若模型大部分输出被 clamp 到0.10m，须区分“预测器更好”与“常数宽度已足够好”。

### P3-E：Point RayRoPE vs Point+GVAR（有条件后续）

只有 P3-A/B 仍显示未解决的 insertion ranking gap，才进行 GVAR 对照。两个模型必须使用**相同物理 action IDs、RGB、深度/evaluator**，否则只是系统级比较。已有 GVAR 是 insertion-conditioned 但 width 使用固定 support envelope，不得称为精确 executed-width-conditioned reader。**本轮不声称已实现跨模型固定候选打分。**

## 2. 新增/更新代码

| Path | Purpose |
|---|---|
| `diagnostics/rayrope_action_lattice.py` | `insertion_width="both"`：同一模型 forward 同时构造 fixed 和 predicted width 插入动作；增加 selected/grasp/score parity 校验 |
| `diagnostics/run_rayrope_p2_lattice.py` | `--diag_insertion_width both`：用同一 exact CAD/DexNet evaluator 重新评价所有物理动作 |
| `diagnostics/replay_rayrope_p3_predicted_width.py` | **无需再次 forward 的快速路径**：若旧 CSV 含每个 depth 的 `width_predicted_decoded_m`，直接重放 predicted-width physical actions 并精确重评 |
| `diagnostics/analyze_rayrope_p3_action_consistency.py` | P3-A/B/C/D 指标、hard negatives、失败归因、scene CI、跨 split CSV 组合 |
| `tests/test_rayrope_p3_action_consistency.py` | CPU 合成协议/统计单元测试 |
| `experiments/RAYROPE_P3_ACTION_CONSISTENCY_DIAGNOSTICS.md` | 本文；新的独立工作单 |

不修改正常训练/默认推理路径，不生成新 checkpoint，不把 evaluator 标签输入网络。

## 3. 117服务器执行顺序

### Step 0 — 环境与证据

第三轮历史 checkpoint SHA256：

```text
0c189e1c70dce12d82098216318290ebd6a855ec0c85ee88263bbb721d894ca9
```

报告说明第三轮使用**原历史 strict loader + 进程内诊断适配**，不是 main checkpoint 封装。禁止在未验证模型一致性的情况下用 main strict loader 替代该历史运行时。

```bash
git fetch origin
git checkout exp/rayrope-p1-p2-diagnostics
git pull --ff-only

python -m unittest tests.test_rayrope_p3_action_consistency -v
python -m unittest tests.test_rayrope_action_lattice -v

export DATASET_ROOT=/path/to/graspnet
export FIXED_P2_CSV=/path/to/test_seen/candidate_lattice_labeled.csv
export P3_DIR=/path/to/p3_action_consistency
mkdir -p "$P3_DIR"
head -n 1 "$FIXED_P2_CSV"
```

核对原 P2 CSV 必须有 `sample_id,scene_id,anno_id,split,condition,parent_id,query_id,angle_id,depth_id,score,width_m,quality,friction,r00..r22` 等物理动作与 evaluator 字段。尤其检查 `width_predicted_decoded_m` 是否存在。

### Step 1A — 推荐：旧 CSV 直接生成 predicted-width 并独立评价

如果上一轮的 `insertion_sweep_fixed_width` 记录已经保存逐 depth 预测宽度 `width_predicted_decoded_m`，**不需要重新做 GPU forward**：

```bash
python diagnostics/replay_rayrope_p3_predicted_width.py \
  --candidate_csv "$FIXED_P2_CSV" \
  --dataset_root "$DATASET_ROOT" --camera realsense \
  --output_csv "$P3_DIR/test_seen_p3_both.csv" \
  --grasp_max_width_m 0.10 \
  --fc_mode reuse_contacts --verify_n 10
```

程序应在以下情况 fail-fast：缺列、逐 depth width 非法、原生 selected depth 的 predicted width 不等于 fixed native width、同一 family 的 fixed width 随 depth 变化、缺少合法物理姿态、无法加载 evaluator。

**每个新 predicted-width grasp 必须重新进行 CAD/桌面碰撞、空抓与 Dex-Net 评价，不能复用 fixed-width grasp 的标签**。输出会保留原始 fixed-width 行，并追加 predicted-width 行到一个**新 CSV**，不覆盖历史证据；同时输出包含 source CSV SHA256 的 manifest。

如果存在源 `lattice_manifest.json`，应与 checkpoint 和 width cap 核对。参数 `--expected_checkpoint_sha256` 仅在 manifest 真实包含该字段时使用；缺失 manifest 时另核验历史 `checkpoint_protocol.json`，不能声称源 checkpoint 已由程序自动核验。

分别对 Seen/Similar/Novel 运行这一步，使用各自的原始 P2 CSV。

### Step 1B — 旧 CSV 缺少 per-depth width 时，冻结历史 forward 生成 both

**禁止**把原生 native width 复制为所有 depth 的 predicted width。

需要冻结第16轮历史 checkpoint，在**原严格历史 runtime** 导出全 depth CDF scores + widths，使用同一个 query-anchor sample IDs。

若 checkpoint 与 main 的 strict grasp_design metadata 确实兼容，可用：

```bash
python diagnostics/run_rayrope_p2_lattice.py \
  --checkpoint_path "$CKPT" --dataset_root "$DATASET_ROOT" \
  --test_mode test_seen --camera realsense \
  --batch_size 1 --num_workers 2 \
  --multi_modal --use_cdf --enable_point_rayrope --kview_mode A1 \
  --diag_frame_manifest_csv /path/to/frames_seen.csv \
  --diag_output_dir "$P3_DIR/test_seen" \
  --diag_query_anchors 16 --diag_nonwinners_per_query 1 \
  --diag_insertion_width both \
  --diag_width_grid 0.04,0.06,0.08,0.10 \
  --diag_seed 20261106
```

**这是 main-compatible 的接口说明，不保证历史 checkpoint 能直接加载。** 如需历史适配，复用历史 `run.py` 的 strict checkpoint loader，不改原模型图，仅接入无参数候选生成函数 `build_p2_actions(..., insertion_width="both")` 和配对校验。必须确认 native winner 的17D pose/score/width 与上一轮逐元素一致，否则中止。

### Step 2 — P3-A/B/C/D 联合统计

既可以提供一份合并 CSV，也可以分别提供三个 split 的 CSV（输入不能重复任何 sample/condition/candidate 组合）：

```bash
python diagnostics/analyze_rayrope_p3_action_consistency.py \
  --candidate_csv /path/to/test_seen_p3_both.csv \
  --candidate_csv /path/to/test_similar_p3_both.csv \
  --candidate_csv /path/to/test_novel_p3_both.csv \
  --output_json "$P3_DIR/p3_action_consistency.json" \
  --require_both_width_modes \
  --expected_frames 18 --expected_scenes 9 --expected_families 288 \
  --n_boot 5000 --seed 20261107
```

工具会验证相同 family/depth 下的 scores、center、rotation、insertion、原生 selected physical action 和质量完全一致。任何 mismatch 均不能通过统计平均掩盖。18 frames×16 query anchors = 288 families，9 scenes（各 split 3 个 scene，每 scene 2 frames）是与第三轮一一对应的验收条件。

### Step 3 — 只分析已有 fixed-width 数据，不伪称 paired P3-A

若 Step 1A 不能运行、Step 1B 尚未完成，可直接从旧 P2 数据完成深度失败原因、hard negatives、Rank-1/Rank-2 与常数宽度基线：

```bash
python diagnostics/analyze_rayrope_p3_action_consistency.py \
  --candidate_csv "$FIXED_P2_CSV" \
  --output_json "$P3_DIR/p3_fixed_only_precheck.json" \
  --n_boot 5000 --seed 20261107
```

此时 `fixed_vs_predicted_width.available` 必须显示 `false`，**不能**称为完成了 P3-A，也不能推导 `R_pred`。

## 4. 结果表模板（仅 P2 历史数值为已知）

| 指标 | Fixed width | Predicted width(d) | 配对差 / 95% CI |
|---|---:|---:|---|
| Mean oracle Q | 0.3466（上一轮） | TBD | TBD |
| Selected Q | ≈0.2326（上一轮推导） | TBD | 理论应为0差值 |
| Mean insertion regret | 0.1140（上一轮） | TBD | TBD |
| Quality-varying families | 46.53%（上一轮） | TBD | TBD |
| Oracle depth changed | — | TBD | TBD |
| Selected failure: collision / empty / FC | TBD | TBD | — |
| Hard negative better than selected | TBD | TBD | — |
| Rank-2−Rank-1 quality | 上一轮推导≈+0.1204，待逐帧验证 | — | CI: TBD |
| Width predicted−0.08m | TBD | — | CI: TBD |
| Width predicted−0.10m | TBD | — | CI: TBD |

同时分 Seen/Similar/Novel 提供逐 depth (1/2/3/4cm) 的 score、质量、失败类型和六个 μ 的成功率。

## 5. 决策门槛

1. 若 `ΔR=R_pred−R_fixed` 的场景配对 CI 明确为负，先检查这是**oracle 变差**还是 depth score-quality agreement 真正提高。由于 selected action 固定，不能把 regret 缩小称为 selected grasp 性能提升；只可归因于 counterfactual 动作集合不同。
2. 若两种宽度协议下 insertion regret 均明显为正，且错误集中于同角度 hard-negative depth，则优先用已有 GVAR 验证 insertion-conditioned evidence 的价值。
3. 若固定0.10m 的表现与 learned width 相同，不要把 width 预测能力作为主要贡献；应分析 clamp 与物体尺度混杂。
4. 若 Rank-2−Rank-1 CI 包含0，不能声称高分前两名错排在场景级稳定；若 CI 稳定为正，进一步研究 top-rank calibration。
5. 仅9 scenes/单 checkpoint 的结果必须称为 pilot；在方法设计确定后扩展到独立测试场景、多训练 seed，最后再做官方 AP 与机器人执行。

## 6. 审计与交付

- [ ] 保持 P2 历史 18 frames / 9 scenes / 288 query families 不变。
- [ ] 使用历史 checkpoint SHA 与原始 strict runtime 的日志/`checkpoint_protocol.json` 核验模型身份。
- [ ] 所有新物理动作独立运行同一 CAD/table/empty/Dex-Net 评估；不启用推断前碰撞过滤。
- [ ] selected grasp 的物理 action ID、width 和 quality 在两种模式一致。
- [ ] 逐 depth score / center / rotation / insertion 逐元素对齐。
- [ ] `R_pred−R_fixed = oracle_pred−oracle_fixed` 对每个 family 成立。
- [ ] 保存三 split CSV、input/output SHA256、evaluator 设置、run.py 适配证据、P3 分析 JSON。
- [ ] CPU tests + 单帧 CUDA/CAD smoke + 三 split 完整任务退出0。
- [ ] 生成 `RESULTS_CN_P3.md`：明确区分 P2 旧数值、本轮新测量与仍不可验证的结论。

最终科学问题不是“depth 头错了多少”，而是：**完整执行动作的物理宽度和插入量是否与模型读取的视觉证据及其打分严格对应，以及这一缺口是否确实影响 action selection。**
