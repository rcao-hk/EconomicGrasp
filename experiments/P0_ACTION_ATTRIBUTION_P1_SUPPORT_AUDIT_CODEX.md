# P0 动作级失败归因 + P1 训练监督支持域审计（Codex 工作单）

**日期：2026-11-14；沿用原分支 `exp/quality-strata-conflict-supervision`，不新建分支。**

**研究来源：**用户上传的 `DETAILED_ANALYSIS_CN.md`（质量分层×竞争监督四臂三seed COMPLETE_ACCEPTED）。其主要现象：

- 主策略 `rank_executed` 中 C−A Novel +0.028916 pp，条件场景区间跨零；scene188 +2.915335 pp、其余29场景平均−0.070616 pp。B质量分层未显著提升Novel，D没有泛化协同。
- B/D 每帧期望16个高分零质量family，平均实际只有3.8766、640帧中95帧满足；现有U∪D标签缓存没有形成足够的新困难负例支持。
- C实际消费12.653125条异质量竞争边/帧，约49% pair质量；不能继续用“竞争pair根本没被训练”解释Novel不进步。
- 当前 P0需定位由评分变化造成的 **fixed-final reranking / object-global budget / NMS survivor** 三段变化；当前 P1需要确定现有U∪D缓存以外，实际有多少 sensor-valid / high-score / competition-relevant 训练动作，以及其中抽样可发现的 q=0 动作。

**不要做：**重训12个头、切换backbone、改GVAR、新测试集真值选模、重写官方评价器、把测试CAD标签用于新模型监督。

## 1. 保留和验证现有实验身份

沿用上一轮 `quality_next_20261114` 的 accepted head/推断/官方AP结果。冻结基座为历史 MIX50 Point+GVAR36 e15，SHA256：

`2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。

四臂 A/B/C/D 表示：

| Arm | 采样 | 竞争pair |
|---|---|---|
| A | 原 Decision | 随机 |
| B | 质量分层 | 随机 |
| C | 原 Decision | 定向竞争 |
| D | 质量分层 | 定向竞争 |

P0 从现成 `rank_executed` inference，P1 从已验收 `plan.json` 及 U∪D cache 进入；**分别使用两个新的结果目录**。绝不覆盖现有实验的 `heads/`、`test_i10/`、`reports/`。

服务器已有 `runtime`、`graspnetAPI`、`grasp_nms` 和 `ExactGraspNetActionEvaluator`。新的诊断通过同一 factory/源码 SHA 绑定；遇到旧版 MinkowskiEngine/编译NMS/GPU索引冲突，应使用已验收 runtime adapter，且保持 strict load/物理身份逐数组 parity，不允许放松容差。

## 2. P0 — 动作级归因：全Novel，不重训

### 2.1 输入和对照

主对照 **A vs C，三个固定head seed**，Original Novel完整30场景 × 26帧=780帧。次级机制比较可以独立运行 A vs B、B vs D、C vs D。Seen/Similar同接口，但不必重复训练。

每个已验收推断目录必须包含：
- `inference.json`，字段 `schema,kind,probe_arm,probe_seed,plan_sha256,collision_protocol,frozen_bank_sha256`；
- 已通过官方评估的 `evaluation.json` 与 `ap_test_novel_realsense.npy`；
- `rank_executed/scene_XXXX/realsense/NNNN.npy`：传感器过滤后、官方NMS前的全量N×17动作。

`RUNS_JSON` 沿用上一轮 report.py，结构：`seed -> A/B/C/D -> test_seen/test_similar/test_novel -> inference root`。

**硬约束：**比较的同一scene/frame不同arm和seed的动作几何和输出顺序必须严格一致，只有最终score不同；每帧AP模拟必须与两边官方`[50,6]` AP 数组一致，容差1e−6；否则该帧停止、整次分析不报告oracle增益。

### 2.2 标签范围与分解

为减少重新CAD评价的成本，仅将各seed/arm的**compiled NMS survivor并集**标注一次，缓存 `assigned_obj`、`friction`、`collision_or_empty`、`empty`、`pure_collision`。这个并集包含路径分解所需要的所有候选，但**不包含被所有方法一起删除的未知质量候选**，因此不能称为全候选动作质量审计。

在共同N×17候选上，按照固定顺序构造：

1. `AP0`：A模型的NMS存活集合＋A分数、正式每物体Top10/全局Top50。
2. `AP1`：固定A最终入选动作**身份不变**，只按C分数重新排序（不重选成员）。
3. `AP2`：固定A的**NMS存活集合**，按C分数重新选择每物体和全局预算成员。
4. `AP3`：使用C的NMS存活集合＋C分数，等于正式C的AP。

分解为：

\[
\Delta_{rank}=AP1-AP0,\quad
\Delta_{quota}=AP2-AP1,\quad
\Delta_{NMS}=AP3-AP2.
\]

严格声明：这是**顺序固定的反事实路径分解**，不是唯一的因果贡献或Shapley值。它依赖选择路径、候选集合、并列分数规则。

额外保存：
- 最终动作身份 lost/gained 和对应已知Q；
- 两种NMS存活集合差异；
- lost/gained之间符合几何NMS冲突的pair；
- 双方得分确实反转且 `Q_lost>0, Q_gained=0` 的高风险事件。
- 冲突边来自几何关系，抽样用**编译版NMS**核查；不是历史actual suppressor方向。

不要仅展示scene188正例。完整报告应覆盖Novel30场景，包括scene164/174/177等负例。按scene配对bootstrap，先在scene内聚合三个固定seed，不把90个scene-seed视为独立样本；scene188排除分析仅做敏感性，不删除主报告。

### 2.3 P0执行

先在服务器创建新的独立worktree，确保同一分支最新提交：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add /home/robotarm/EconomicGrasp-quality-diagnostics \
  -b codex/quality-diagnostics-exec origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-quality-diagnostics
git status --short
git worktree list
find .. -name AGENTS.md -print
```

根据真实117路径，设置：

```bash
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/quality_followup_20261114
export RUNS_JSON=/ABS/quality_next_20261114/runs.json
export ACCEPTED_PLAN=/ABS/quality_next_20261114/plan.json
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export BOUND_CONFIG=/ABS/accepted_mix50_gvar/bound.json
export DEVICE=cuda:0
export FACTORY=rare_grasp.runtime:Runtime

bash scripts/decision_supervision/run.sh check
```

先独立单帧 smoke，不能把其结果混入正式780帧的目录：

```bash
P0_SPLIT=test_novel P0_LEFT=A P0_RIGHT=C P0_LIMIT=2 \
 P0_OUTPUT="$RUN_ROOT/p0_smoke_A_C" \
 bash scripts/decision_supervision/run.sh p0-attribution
```

验收单帧所有arm/seed几何一致、编译NMS identities、两种原生AP逐元素一致以及标签缓存hash。测量一帧需要重评的NMS survivor并集动作数、CPU内存和耗时；再决定是否有资源完成780帧。若不足，保留完整 smoke、提出固定场景小样本版本，不能将其称为全Novel。

```bash
P0_SPLIT=test_novel P0_LEFT=A P0_RIGHT=C \
 bash scripts/decision_supervision/run.sh p0-attribution

# 中断后严格同合同resume：
RESUME=1 P0_SPLIT=test_novel P0_LEFT=A P0_RIGHT=C \
 bash scripts/decision_supervision/run.sh p0-attribution
```

产物：
- `p0_attribution/test_novel/A_vs_C/labels/scene_XXXX/NNNN.npz` — 对全seed共同NMS survivor动作并集的精确评价；
- `actions/SEED/scene_XXXX/NNNN.npz` — A/C分数、身份、各级入选集合、已校验Q和冲突反转；
- `per_frame.csv`、`per_scene_seed.csv`、`summary.json`、`RESULTS_CN.md`。

若有资源可运行主对照以外的B/D及Seen/Similar；均需另设目录与单独完整AP parity。

## 3. P1 — 监督支持域审计：只在训练场景获取有限新标签

### 3.1 问题与输入

旧U∪D标签池每帧最多约96个完整query families，不能因此推断1024-query完整native候选中不存在hard-zero。P1评估：

**现有U∪D漏掉多少sensor-valid/high-score/NMS competition-relevant动作？从预定新样本中能发现多少原评分很高但物理失败的动作？**

只使用已经验收的`plan.json` 800个训练域帧/四depth旧标签、冻结 GVAR checkpoint、旧U∪D `read_pack`、原生 `snapshot` 和 `sensor_collision_filter`。官方 test_seen/similar/novel scene>=100 绝不能用于选择、标注或后续模型训练。

### 3.2 预注册的采样预算

默认两训练域各100 scene×2个固定帧（原四帧的首/末帧），合计**400帧**。每帧最多新标注**64个native实际执行depth动作**，预算上限25,600个；不标注未选择的另外三个depth，也不重标已有U∪D四depth。

在得到CAD q之前，先基于native模型完整1024候选、传感器过滤、compiled NMS及几何冲突关系构造候选分层：

| Stratum | 计划每帧上限 | 来源 |
|---|---:|---|
| Native Top50 survivors | 16 | 不在U∪D中的native NMS前50 survivor |
| Top50 conflict neighbors | 24 | 不在U∪D的几何冲突邻居，优先与已有已知端点成对 |
| 高原生分数其他候选 | 16 | sensor-valid且native分数≥本帧sensor-valid的75%分位 |
| 随机sensor-valid对照 | 8 | 前三层之外的随机动作 |
| 无放回fallback | 补足64 | 剩余未标注native动作 |

某分层不足就明确记录欠额，**不重复样本凑配额，也不将随机补齐当作原分层已满足**。如果整帧缺少足够未标注sensor-valid动作，标注数小于64并显式记录；默认至少存在一个新候选，否则停止并检查。

所有选择在评估器调用前只使用**native分数、候选ID、sensor-valid和几何冲突边**，不能使用待标注动作真值。已知U∪D候选Q仅用于标注后比较和可计算的竞争边Q差。

每个新动作保留真实17D物理参数，不重新估计depth/width或手动调整接触位置。runtime精确调用官方CAD/Dex-Net（float64）；收集六摩擦阈值、collision、empty、friction和hash。传感器过滤预处理依旧依赖既有Original传感器数据，不得把整套pipeline称为纯RGB无额外深度输入。

### 3.3 分析哪些指标

**Exact support coverage（无需新Q）：**
- 完整sensor-valid native候选数；
- 其中有多少query已在U∪D（可计算准确覆盖率）；
- Native Top50 survivor：已缓存／缺失数量；
- Native Top50 conflict neighbor：已缓存／缺失数量。

**New label findings（只对抽样动作成立）：**
- 新样本真实q=0／q>0数量；
- 本帧原生分数前25%（sensor-valid分布第75百分位以上）新q=0数量；
- 分层q=0发现率（sampled-stratum hit rate，不是总体无偏失效率）；
- 新标注端与已有端／新增端的真实异质量竞争pair数量；
- 数据域 Original vs GN-Trans，训练/验证/旧审计场景组0:80/80:90/90:100分别报告。

**不可以从有限抽样直接推论：**“完整1024-query候选的hard-negative真实发生率为X%”；或“这些新q=0动作会使新评分器Novel AP提高”。

### 3.4 P1执行

同一worktree、新结果目录，先 smoke：

```bash
P1_LIMIT=2 P1_SUPPORT_ROOT="$RUN_ROOT/p1_support_smoke" \
 bash scripts/decision_supervision/run.sh p1-support-mine
```

正式分片示例（先测资源，再决定并发；不得与其他工作抢占GPU/CPU）：

```bash
P1_SUPPORT_ROOT="$RUN_ROOT/p1_support/cache" SHARDS=3 SHARD_INDEX=0 DEVICE=cuda:0 \
 bash scripts/decision_supervision/run.sh p1-support-mine
P1_SUPPORT_ROOT="$RUN_ROOT/p1_support/cache" SHARDS=3 SHARD_INDEX=1 DEVICE=cuda:1 \
 bash scripts/decision_supervision/run.sh p1-support-mine
P1_SUPPORT_ROOT="$RUN_ROOT/p1_support/cache" SHARDS=3 SHARD_INDEX=2 DEVICE=cuda:2 \
 bash scripts/decision_supervision/run.sh p1-support-mine

bash scripts/decision_supervision/run.sh p1-support-summarize
```

中断后对每个shard使用相同`SHARDS`、`SHARD_INDEX`和`RESUME=1`。resume必须校验对应帧物理bank、sensor-valid、编译NMS与缓存SHA，**不能再次执行昂贵的已完成CAD标签评价**。

产物：
- `p1_support/cache/frames/original_scene...npz`：新采样实际动作、q及竞争边；
- `support_shard_XX_of_XX.json`：各分片来源合同和标签SHA；
- `p1_support/summary/summary.json`、`per_frame.csv`、`RESULTS_CN.md`。

报告必须区分 complete/partial，验证400个唯一训练帧及全部shard。正文仅可引用完整并通过验收的真实结果；任何负例按支持域/score分层列出且不用于先验调参。

## 4. 接下来的研究决策（本工作单不训练）

根据P0与P1做可证伪的三路决策：

1. 如果P0错误主要来自 **固定最终动作集合的评分次序**，且P1发现大量新的高分零质量训练动作：优先做预注册的补标签/质量读出实验，不增加Graph。
2. 如果P0错误主要来自 **NMS survivor替换**，并且新增标签可覆盖真实高低Q竞争端点：做有限预算的结构化决策目标对照，但必须仍用独立训练域验证。
3. 如果P1训练域高分错误候选很少、P0 Novel真实质量错误却广泛：检查跨域几何证据的可泛化性；仅有训练集Q的下限不能证明RGB证据不足。

**场景188、174、177、164已被后验观察，只能作机制案例。** 未来模型选择不能再根据这些场景或全部反复看过的Novel测试做调参。

## 5. 验收清单

- [ ] 既有CPU回归与`test_p0_p1_decision_diagnostics.py`合成单元测试通过；
- [ ] 新工作单所在分支保持`exp/quality-strata-conflict-supervision`，`main`与前一轮代码和结果不改；
- [ ] P0所有参与对照的动作仍为同一固定bank，快照/后处理来源SHA、官方AP与推断元数据匹配；
- [ ] P0每帧用编译版NMS，完整AP两侧逐元素误差≤1e−6；标签只覆盖NMS-survivor union；
- [ ] P0阶段增益逐帧相加等于完整差异；scene CI/seed处理正确；lost/gained身份及局部Q inversion均有物理动作证据；
- [ ] P1严格排除测试scene>=100；每帧新动作采样在CAD Q读取之前完成、旧U∪D接受的缓存数据通过严格parity；
- [ ] P1编译NMS冲突边进行固定预算抽查；新标签仅为native executed depth，统计与旧四depth区别；
- [ ] P1已有frame hash合法时resume不重新CAD evaluator；旧缓存保留；
- [ ] 完整400帧／全部shard标签哈希和来源配置验收后输出结果；不把局部sampled q0 rate写成完整候选总体比例；
- [ ] 记录P0每帧新标签数量与P1每帧新增标签数／时间／内存；资源不足时允许部分独立smoke但不得称全量；
- [ ] 最终另写中文结论，区分事实、后验归因与未验证的训练假说。

**完成状态说明：**截至建立工作单，GitHub上新增脚本与CPU合成测试；未连接117/119服务器执行真实CAD/AP重算，因此没有新的P0/P1实验结果。
