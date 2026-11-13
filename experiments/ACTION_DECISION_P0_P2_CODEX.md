# Action Quality → Decision：P0–P2 实验工作单（Codex）

**建立时间：2026-11-13。**
分支：`exp/action-decision-p0-p2`，从 `exp/candidate-conflict-quality-oracle` 的最新提交建立。严格保留主分支和 RARE/GVAR 历史实验。**新脚本没有在117 GPU/CAD 环境正式运行过；本文是可执行协议和停机规则，不是已得到的 AP 数据。**

## 0. 为什么不是立即训练 Graph Network？

2026-11-12 的全量 candidate-conflict 报告显示，MM 原生 Mean AP=51.902%，仅使用测试场景真实质量 `Q` 的 NMS＋后排序可达83.029%。但这是离线 **known-quality greedy selection**，不是可部署模型，也不是严格AP全局上界；与此同时现有固定连通分量人数的 set oracle 只在约20.6%的节点上执行精确求解。

当前需要区分三个竞争假设：

- **P0／信息假设：** GVAR 的 reader latent 已经包含判断真实 Q 的信息，只是最终CDF/64维特征没有正确解码；或视觉证据本身不足。
- **P1／学习与决策假设：** 执行动作标签正确，但均匀 MSE／全图 RARE 的监督分布没有重视会进入候选竞争的高影响动作。
- **P2／搜索假设：** 即使给每个候选真值Q，固定规则的 greedy NMS 仍显著劣于相同约束下的集合选择。

**绝不使用 Seen/Similar/Novel 测试 CAD 质量训练 P0/P1。** P2 只把测试Q用于离线反事实诊断。不同职责必须分离。

## 1. 来源、标签与公共合同

- 冻结 `MIX50历史 Point RayRoPE＋GVAR36 e15` 权重（不是fresh e19），沿用经过严格绑定的 `bound.json`。
- 原训练数据：GraspNet Original 12800帧＋GN-Trans 12800帧，现有RARE MIX50 source exact-action cache `train_index.json` 内含25600帧×32 query×4 depth＝**3,276,800条标签**；不重新CAD评价这些标签。
- P0／P1 scorer 训练场景：0–89（Original、GN-Trans各11520帧，共23040）；scene-held-out scorer validation：90–99（两域各1280帧，共2560）。
- **这10个验证场景被历史共同GVAR基座训练见过，但新评分器没有见过**。因此它是 scorer 的 scene-held-out 验证，不是整个网络的严格未见场景验证。
- P0新缓存通过 hook 获取 reader输入与输出：原 `score_dropout` 64D，GVAR reader-after 256D，以及 reader residual 256D。所选择动作仍为原生 query＋angle＋四depth，对每个样本严格验证17D executed geometry ID、width、CDF base logits及64D feature parity。每帧只存储已标注的128个动作。
- Base score与真实动作参数来自同一原始模型。所有 scorer 仅改变预测分数，不改变中心／姿态／执行宽度／四depth，也不更新GVAR/backbone/CDF head。
- 新 probe 输入模式参数形状保持一致，epoch0的输出严格等于native原score；比较新模式时采用相同seed、训练图像顺序、optimizer预算和loss。
- P0/P1新 head 与此前4673参数RARE head容量不同，**只将其与本轮同架构控制实验直接归因**。旧MSE/RARE数值可以列为描述性参照，不可将绝对差全归因于新特征。

### 验证接口

原始 source index：`rare_grasp.pipeline.load_index`（SHA校验）。
新 feature index：`diagnostics.action_decision.feature_cache.index`（带原source SHA的逐帧reader id校验）。
原生前向：`rare_grasp.runtime.Runtime.snapshot`，只使用RGB预测深度；与原 `pred_decode_center_view_angle` 的17D动作parity要求≤1e−6。
评价：原 `rare_grasp.pipeline eval` 完整780帧/split，保持前处理、sensor model-free collision、官方NMS与官方摩擦评价。

## 2. P0：Frozen GVAR feature quality decodability

### 要比较的特征条件

| Mode | 模型看到的证据 | 问题 |
|---|---|---|
| `base_geometry` | 原CDF scalar logit＋执行动作的几何参数 | 已有score与动作参数能校准到什么程度？ |
| `score64` | 上述＋原64D CDF score-branch feature | 现有最终评分特征是否还有剩余信息？ |
| `reader256` | 上述＋GVAR reader输出256D | reader是否保留了score head遗失的物理信息？ |
| `reader_delta` | 上述＋GVAR reader residual 256D | evidence update本身能否提供额外判别信息？ |
| `score_reader` | 上述＋64D评分特征＋256D reader输出 | 联合证据是否有互补增益？ |

所有模型共用同形状网络与参数数量，最后residual输出层零初始化、原有CDF score作为base logit；训练20轮MSE，不改变现有Q定义。

**主要报告：**
- 训练域scene90–99分Original/GN-Trans的Q-MSE、Selected Q、Regret、Oracle@2、depth pair agreement、top1 sampled-family Q；
- 每个模式相对`base_geometry`与`score64`的场景配对差异；训练seed至少3个，在任何官方AP检查前预注册；
- 完整Seen/Similar/Novel测试原生1024-query official AP，严格区分“特征可解码性”和“已经成功泛化”。
- 若仅训练集Q-MSE下降、场景验证和测试Novel不改善，不能宣称GVAR reader已包含泛化的接触证据。

### 执行命令

先在117读取任何`AGENTS.md`，检查`git status`、GPU资源、已有缓存/权重SHA、硬盘剩余空间。不要杀已有训练作业或删除历史结果。

```bash
git fetch origin exp/action-decision-p0-p2
git worktree add /home/robotarm/EconomicGrasp-action-decision \
  -b codex/action-decision-exec origin/exp/action-decision-p0-p2
cd /home/robotarm/EconomicGrasp-action-decision

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/action_decision_p0p2_20261113
export BASE_CONFIG=/ABS/rare_mix50/point_gvar/bound.json
export TRAIN_INDEX=/ABS/rare_mix50/point_gvar/cache/train_index.json
export PROTOCOL=/ABS/rare_mix50/frames_mix50_test10.json
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export DEVICE=cuda:0

bash scripts/action_decision/run.sh check
bash scripts/action_decision/run.sh p0-smoke
```

两个数据域分别先跑一个训练场景和一个验证场景的真实reader hook smoke。**出现raw scorer feature/17D identity/native logits失配则立刻停，不得放大容差。**

正式缓存，支持按frame索引分片：

```bash
NUM_SHARDS=3 SHARD_INDEX=0 DEVICE=cuda:0 bash scripts/action_decision/run.sh p0-capture
NUM_SHARDS=3 SHARD_INDEX=1 DEVICE=cuda:1 bash scripts/action_decision/run.sh p0-capture
NUM_SHARDS=3 SHARD_INDEX=2 DEVICE=cuda:2 bash scripts/action_decision/run.sh p0-capture

# 全部25600帧完成后：
bash scripts/action_decision/run.sh p0-index
EPOCHS=20 SEED=0 bash scripts/action_decision/run.sh p0-fit
```

以上三路任务必须使用不同的实际空闲GPU，并可在各自shell运行；共享cache根目录时按不同帧输出，碰撞即报错而不是覆盖。float16缓存包含256D reader与256D residual，定量评估数值精度对结果影响。

不要默认整个五模式20轮同时启动。建议先测吞吐/磁盘：25600×128×(256＋256)×2 bytes≈3.36 GB未压缩，不计文件系统元数据和旧标签缓存。

### P0 全量AP例子

```bash
export HEAD=$RUN_ROOT/p0/score_reader/last.pt
export HEAD_TAG=p0_score_reader_seed0
DOMAIN=original bash scripts/action_decision/run.sh infer
DOMAIN=original bash scripts/action_decision/run.sh eval
```

这仅调用原`rare_grasp.pipeline eval`，保持每split780帧正式协议，不复用用于生成P2测试oracle的测试标签。

## 3. P1：Decision-aware quality supervision（同容量）

在P0相同的25600帧缓存上，固定从验证选出的一个`FEATURE_MODE`，比较：
1. `mse`：全部真实执行动作的10× quality MSE。
2. `rare`：MSE＋同图随机非并列pair，pair系数0.5。
3. `decision`：MSE＋从**原模型**高分query winner挖掘的NMS-geometry conflict pair、score-quality inversion pair，再用同图非并列pair补足预算；不得用评价test Q选pairs。

每次使用相同原训练动作标签、20轮、batch4、pair budget≤2048和相同随机初始化。pair仅使用可比较的不同质量动作；等质量pair不赋予虚假顺序。

**严格解释边界：** 当前`decision`是在每幅图**已标注的32个均匀query**内建立冲突代理，并非1024完整查询或传感器过滤后的真实NMS竞争图。它是零新标注的低成本**P1a feasibility pilot**，不能直接认定为完整 deployment-aware selection learning。

先检查`train_mean_conflict_pairs`及含至少一条有效conflict pair的帧比例。如果几乎为零，或者验证／AP没有独立收益，**应停在P1a，下一步从训练场景额外采集经过sensor filter的高排名和NMS竞争query并重新标注，不能假装均匀池代表部署竞争分布**。禁止以测试Novel的Q作为训练数据获取准则。

```bash
FEATURE_MODE=score_reader EPOCHS=20 SEED=0 bash scripts/action_decision/run.sh p1-fit
# 重复SEED=1,2时使用不同头目录；P0模式选定后不要按测试AP反复修改
```

对MSE/RARE/decision分别执行同一完整1024-query推断与官方评估；输出本轮的Native/MSE/RARE/decision paired scene CI。必须同时报告：MSE、local within-family regret、Top1 Q、各μ AP、NMS survivor改变、原生候选在pre/post-NMS的排名。

如果`decision`得到更低训练pair loss却不改善高排名动作的AP，不能证明集合交互机制有效。

## 4. P2：Global-budget CP-SAT set oracle

### 前提

依赖2026-11-12已经验收的**完整100% Original测试** graph/labels：
- 每split30场景×256帧，四方法MM/MR/RM/RR；
- graph stage必须是sensor collision prefilter后、官方NMS前的完整N×17候选；
- 每帧每物理动作（`grasp[1:16]`）有精确摩擦、empty/collision、assigned_obj和six-μ labels；
- graph/label manifests SHA一致；
- P2逐帧重新核验Native AP与官方`[30,256,50,6]`数组误差≤1e−5，不一致则停止。

### P2 solver的目标和约束

使用OR-Tools CP-SAT（`pip install ortools`，根据117的依赖规范安装），最大化：

\[
\max_{x_i\in\{0,1\}}\ \sum_i Q_i x_i,
\]

约束：
- 对每个NMS conflict edge $(i,j)$，`x_i + x_j≤1`；
- 全局`Σx_i≤50`，不再固定每分量旧survivor数；
- assigned object上的`Σ_{i:o_i=o}x_i≤10`；
- 模型预测score仅作真实Q相同情况下的次级tie-break；
- 所有动作仍来自传感器预筛后的固定候选池；
- 最终AP用**原官方后处理和六摩擦阈值评价**重算。

比较三个受控条件：

| Policy | 使用的信息 | 备注 |
|---|---|---|
| Q-greedy NMS | 真实Q，原NMS，再做per-object/top50 | 上一轮known-quality reference |
| Budgeted Q-greedy | 真实Q，Greedy且提前满足全局50与每物体10约束 | 直接可行基线 |
| CP-SAT set selection | 真实Q，同预算／同冲突边／同per-object cap | 若OPTIMAL，认证的是sum-Q，不是AP |

**任何FEASIBLE/UNKNOWN/time-limited不等于最优；max-candidates>0意味着只在截断候选池上优化，绝不宣称完整动作池最优。** 记录solver status、restricted比例、真正full-pool sum-Q认证帧占比、objective/count和official AP。Sum-Q增加不保证AP单调提高，也不能给未生成的grasp提供上界。

### P2运行（先按10%帧间隔做pilot）

```bash
export P2_GRAPH_ROOT=/ABS/candidate_conflict_oracle_20261112
export P2_LABELS_ROOT=/ABS/candidate_conflict_oracle_20261112

# 默认每split30×26=780帧，逐帧仍读取完整N×17候选
SAMPLE_INTERVAL=10 P2_TIME_LIMIT=5 P2_CANDIDATE_CAP=0 \
  bash scripts/action_decision/run.sh p2
```

验收oracle可行解、native parity、solver resource后再考虑：

```bash
# 完整30×256每split，CPU开销可能非常高：
SAMPLE_INTERVAL=1 P2_TIME_LIMIT=10 P2_CANDIDATE_CAP=0 \
  bash scripts/action_decision/run.sh p2
```

这两个命令使用不同的输出目录，不覆盖。三split结果应保留原scene stratification。可在资源受限时使用`P2_CANDIDATE_CAP=256`作明确标记的 restricted pilot，并始终附上最优性认证比例及与full-Q greedy的差别。

### 需要防止的逻辑错误

- 固定每分量survivor数量的是**上一轮component oracle**，不是本次P2；新的全局预算可以将名额从一个连通分量转移到另一个。
- conflict edge仅表示GraspNet NMS几何关系，**不是**抓取物理碰撞边。
- known-Q解决的是测试时上界/诊断，不是“从RGB预测出这个Q”的能力。
- 如果Q-greedy已经接近CP，并不排除learned ranker可以帮助；如果CP优于Q-greedy，则仍必须有可学习的集合关系才能成为新方法。
- P2不优化严格官方AP目标，不能将求解器`OPTIMAL`误读成AP upper bound。

## 5. 验收、产物和研究决策

### 工程验收

```bash
python -m unittest discover -s tests -p 'test_action_decision.py' -v
python -m unittest discover -s tests -p 'test_candidate_conflict.py' -v
python -m compileall -q diagnostics/action_decision
```

P0需检查：source_index SHA与GVAR bound config；capture一次＋全维reader索引；真实action pid、score64/logit、angle、width一致；全部25600帧cache完整；验证集不含任何scorer训练scene。

P1需检查：对比条件使用同容量／同数据／同预算；每帧pair均非并列；pair conflict coverage、训练场景与实际candidate coverage明确；Novel测试标签从不参与训练／选模。

P2需检查：不改变原官方eval；所有输入graph/labels/per-frame parity通过；solver quota、冲突可行性与certification逐帧记录；区分full与capped、OPTIMAL与FEASIBLE；scene cluster bootstrap不是训练seed方差。

### 实际产物

- `features/frames/*.npz`、`feature_index.json`：P0真实reader feature cache。
- `p0/{mode}/last.pt, protocol.json, history.json`：同预算特征可解码性对照。
- `p1/{mode}/{objective}/seed*/last.pt`：MSE／RARE／decision pilot。
- `inference/{tag}/{domain}/{split}/`：真实1024-query grasp dumps，`ap_test_*_realsense.npy`与官方评价验收。
- `p2/{split}/interval{10,1}/per_frame.csv,summary.json`：约束求解状态、质量和AP结果。

### 科学决策（在结果完成后才可以写）

1. Reader256＞Score64 且跨场景／Novel AP改善：现有GVAR证据可能未被原CDF head充分利用，优先设计物理质量读出／监督。
2. Reader256训练误差改善、跨域不改善：不能认为证据足以泛化；考虑几何与接触证据可靠性。
3. Decision pair＞同预算RARE且真实conflict覆盖足够：有依据研究更完整的selection-aware监督。
4. P2认证full-Q集合解明显优于Q-greedy：值得做candidate interaction／set selection；否则优先改善质量预测。
5. P2 Novel即使已知Q仍低：仍有原候选池和视觉几何泛化问题；不能只解决ranking。

## 6. 任务真实状态

GitHub新分支保存了P0特征捕获、P0/P1 scorer训练与1024-query推断、P2全局预算CP-SAT脚本、CPU合成测试和本工作单。**没有连接117服务器执行真实训练、额外CAD、完整AP或CP-SAT大规模求解，当前没有新增P0/P1/P2数值结果。** 真实端到端验收必须在117执行上述分阶段命令，并保存执行记录与SHA身份。
