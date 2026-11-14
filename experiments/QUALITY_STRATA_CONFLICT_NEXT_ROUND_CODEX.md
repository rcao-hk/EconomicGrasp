# 下一轮：共同候选池交叉审计 + 质量分层 × 竞争关系监督

日期：2026-11-14。新分支：`exp/quality-strata-conflict-supervision`。
基线：上一轮实际执行锁定的 `exp/quality-decoding-decision-p0-p2@c1e6e3eda78d3e9690cd05e554fbb7f155778890`。
**只新增实验包、测试和工作单；不覆盖上一轮结果、不改main、不改生产网络/官方评价器。**

## 0. 为什么做这一轮，以及不做什么

依据用户上传的 `QUALITY_P0_P2_AND_FOLLOWUPS_SUMMARY_20261114_CN.md`：

- Decision-only动作 q 均值0.6081、q=0占24.43%；Uniform-only为0.3379、51.51%。这是独有样本分布，不是两臂全样本比例。
- Decision的native竞争两端覆盖率4.59%，异质量完整边8.46条/帧，但随机2048对loss预计只抽到0.782条native竞争边/帧。
- Novel退化主要随rank-only出现；固定depth再取对应评分，并没有稳定修复。
- 旧确认集候选不同，不能用Decision较高selected q直接证明头更好。
- 这些观察没有证明正例富集或竞争监督不足是唯一根因。

本轮仅检验两个训练因素：**数据组成**与**pair监督分配**。保留原GVAR，不加新reader、GNN或guard，不重跑P2求解器，不解冻backbone。

**本文件描述实施协议，不是新增实验结果。** 本地测试使用合成数据；117真实训练、原始缓存重放与正式AP需要按下述门禁验收。

## 1. 两阶段、四臂

### Stage X：先做零新增标注的交叉审计

四个已有P1评分头 Uniform-MSE / Uniform-RARE / Decision-MSE / Decision-RARE，全部评价：
1. 同一Uniform确认候选池；2. 同一Decision池；3. 去重的U∪D池。
另列native score参考。不重训、不改标签、不中途筛选对自己有利的帧。

输出MSE、selected q、同family oracle q，以及固定原depth/选择新depth两种**样本池内**Top10/50的q和零成功比例。这里没有做全量传感器过滤/NMS，不能叫官方Top10假阳性率或AP。原90–99场景已经反复查看，这一步是重新配对的诊断，不是新的独立确认集。

### Stage T：同预算2×2，从相同初始权重重新训练

| Arm | 训练候选 | Pair目标 |
|---|---|---|
| A | 上轮Decision的原64 families | 图像内随机非并列pair |
| B | 质量分层64 families | 同A随机pair |
| C | **与A完全相同** | 定向竞争pair＋随机pair |
| D | **与B完全相同** | 与C相同目标 |

必须比较B−A、C−A、D−B、D−C、D−A及交互项D−C−B＋A。A是本轮重跑基准，不声称与历史Decision-RARE逐权重相同：pair RNG实现和固定epoch政策已明确改变，只有**本轮A–D间**作控制实验。

## 2. 冻结的共同合同

- 基座：历史MIX50 Point+GVAR36 e15，SHA256 `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。
- scorer架构和state_dict与上一轮匹配：64D post-GVAR feature＋15D执行动作几何＋4D depth one-hot＋原utility logit，5,633参数。
- 训练场景0–79、验证80–89、重复审计90–99；原backbone见过所有训练场景，所以不是完整系统未见场景验证。
- 两域每场景4个既有帧；640训练/80验证/80重复审计。验证/审计所有头都用**相同的U、D、union池**，不按新采样臂各自挑选。
- 固定10epochs、batch4 frames、每帧64×4=256 actions，160 optimizer steps/epoch、1,600 steps/头；AdamW lr1e−4、wd1e−4、clip1；10×MSE＋0.5×pair loss。
- 预先固定3个训练seed：20261113/20261114/20261115，共12个头；相同seed在A–D中初始化、帧顺序与更新数一致，记录initial_state_sha256。
- **固定last epoch10为主权重**，不根据测试AP选模；共同验证MSE/selected q完整记录，不再当作唯一决策质量代理。epoch0只检查原分数恢复。
- 正式主要推断：`rank_executed`，固定native depth、使用该实际depth对应的新分数；次要`both`。`rank_max`和`depth_only`保留为可选诊断，不事后挑最优policy。
- 先沿用10% Original：每split780帧，collision on和官方NMS/每物体10/global50不变。100%扩展用新目录和`INTERVAL=1`，不能混用数组。

## 3. 质量分层：第一版不新增CAD标签

**仅复用上一轮已验收的P1 Uniform∪Decision动作池。** 两臂共享至少32 families，因此每帧最多96 families，标签、64D特征、完整17D执行动作已经存在。两种训练采样都付出同一个U∪D标签池的成本，不能把总标签预算说成每臂只标注64。

A/C仍使用原Decision64。B/D从U∪D选择64：
- 从U∩D固定抽32个shared anchors；这是本轮确定的shared anchors，**不能恢复或声称正是历史最初32个**，因为旧缓存只保留排序后query IDs。
- 其余32：目标16个高原生分数且native执行q=0的family，8个q-count1–3，8个count4–6。
- 高分固定定义为：源池中通过原传感器过滤的native动作分数不低于该池中位数。只在训练scene0–79用质量标签分层；不筛选验证/审计池。
- 一个family被选中，四个depth都保留；不把其他depth的标签改为native动作标签。
- 缺某一层时随机无放回补齐，报告实际hard-zero、中/高质量数量、fallback和与原Decision不同的family数。**不重复少量负例凑配额，也不自动扩大到未标注池。**

这只检验已有标签支持域内的采样干预。若硬负例持续不足，应先报告可用覆盖；不能把未达到配额的结果称为充分hard-negative训练，也不能因此证明Novel信息不可学。

## 4. 定向竞争关系：准确说明本轮与旧统计的差别

旧报告统计的是“某次native greedy中survivor实际压制哪个动作”。现有GitHub代码不包含117的完整方向trace修复；重新在小候选子集上跑NMS不能恢复那些历史方向。

**本轮统一使用：全量native帧中，NMS前50个survivors与其NMS几何冲突邻居的无向关系。**

构造步骤：
1. 从已验收runtime/已保存snapshot取完整1024 query的native四depth bank，严格核对缓存动作ID、center/R/d/width、native score、64D feature和logit；禁止用不同forward长度重算旧score branch。
2. native每query选一个depth，做原传感器碰撞过滤。
3. 调用当前环境**编译版GraspGroup.nms**获取完整帧的native survivors和前50，保留其并列分数顺序。
4. 只对U∪D中实际存在的两端，用官方两候选NMS判定几何冲突；不自己复刻旋转/阈值/tie规则。
5. 监督端点是两query的**native实际执行depth**，不是随意拉取另一个depth；排序方向由真实q决定，q相等不进入pair loss。

这不是“实际directed suppressor trace”，不能把其边数直接与旧0.782/4.59%的方向边统计对比；也不是物理碰撞图。它保留完整帧上下文和精确官方二元冲突关系，用于检验**两端确实被loss消费**的效果。主要策略固定native depth，与边上的实际动作一致；次要both可能改变动作，须单独解释。

### Pair loss预算与权重

- Random臂：图像内所有q不同的无序对，最多2048，uniform无放回，逐帧求均值。
- Targeted臂：最多256条**唯一**已覆盖异质量竞争边，分配pair loss质量的50%；其余budget从未选过的异质量对中随机补足，分配50%。
- 缺少竞争边时逐帧退回与Random完全一致的抽样，不创建假边；少量边不重复采样。稀少边获得较高权重是**显式目标干预**，不是声称增加了独立信息量。
- 每epoch逐帧保存eligible数、unique竞争边数、实际选中竞争边数、target组大小、最大单pair权重、实际竞争边总loss权重、fallback和抽样hash。随机背景仍可能含其他竞争边，因此实际竞争loss权重不一定恰好50%。

## 5. 工程入口与兼容边界

新增`diagnostics/decision_supervision/{core,data,runtime,train,report}.py`；只依赖现有项目推断/eval接口，不修改它们。

**关键：上一轮117的float64 evaluator、MinkowskiEngine、GPU可见编号、编译NMS tie等修复不一定已推送GitHub。** 不得只checkout旧基线就假定继承全部修复。保留原已验收worktree；通过`--factory accepted_runtime:create_runtime`接入其严格runtime，接口为：

```python
create_runtime(bound_config_path, dataset_root, gntrans_root, device)
# -> object with config, config_digest,
# snapshot(row, seed) -> full grasps[1024,4,17], features[1024,4,64], base_logit[1024,4]
# sensor_collision_filter(row, grasps) -> unchanged subset rows
```

默认factory为`rare_grasp.runtime:Runtime`，仅在实际通过旧缓存native parity时使用。原始float64几何不转float32做评价；legacy physical ID使用float32编码仅为兼容旧键，shared动作的真实数组另做严格一致性检查。只读取并保留旧标签，不调用新的CAD evaluator。

## 6. Codex运行顺序

先读AGENTS.md、git status/worktree、磁盘和GPU状态；禁止覆盖旧产物、删缓存、改权重meta或strict=False。使用新worktree：

```bash
git fetch origin exp/quality-strata-conflict-supervision
git worktree add /home/robotarm/EconomicGrasp-quality-next \
  -b codex/quality-next-exec origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-quality-next
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/quality_next_20261114
export SOURCE_INDEX=/ABS/accepted_p1/cache/index_merged.json
export ACCEPTANCE=/ABS/accepted_run/reports/SCORER_ACCEPTANCE.json
export BOUND_CONFIG=/ABS/accepted_mix50_gvar/bound.json
export DATASET_ROOT=/ABS/graspnet
export GNTRANS_RGB_ROOT=/ABS/GN-Trans
export DEVICE=cuda:0
# 需要时接入已验收runtime（只提供入口，不能编造原生输出）
# export FACTORY=accepted_runtime:create_runtime
bash scripts/decision_supervision/run.sh check
```

### 6.1 交叉审计，先看已有模型

准备`OLD_HEADS` JSON，四个键命名为`uniform_mse/uniform_rare/decision_mse/decision_rare`，每个值：

```json
{"file": "/ABS/accepted_head/best.pt", "sha256": "ACTUAL_FULL_FILE_SHA256"}
```

整份JSON应含四个这样的键，不是把上面的单值直接当根对象。脚本严格检查旧checkpoint schema、5633参数、base SHA、cache index和source config，不允许仅因矩阵shape兼容就加载别的模型。

```bash
export OLD_HEADS=/ABS/old_heads.json
bash scripts/decision_supervision/run.sh cross-old
```

输出`cross_old/per_frame.csv,summary.json,REPORT_CN.md`。先对照上轮同池确认数值，确认兼容状态和权重完全一致；不一致先停。80审计帧已有过观察，不重新包装成未见验证。

### 6.2 原生关系重放，不重新标注

可复用722帧已保存snapshot，但须由accepted factory保证与缓存完全相同；缺失帧原生重跑。先在独立smoke索引/目录下执行，完成后再三GPU分片：

```bash
CUDA_VISIBLE_DEVICES=3 SHARDS=3 SHARD_INDEX=0 bash scripts/decision_supervision/run.sh trace
CUDA_VISIBLE_DEVICES=4 SHARDS=3 SHARD_INDEX=1 bash scripts/decision_supervision/run.sh trace
CUDA_VISIBLE_DEVICES=5 SHARDS=3 SHARD_INDEX=2 bash scripts/decision_supervision/run.sh trace
# 中断后同合同可 RESUME=1；不要混入不同SHARDS的重复产物。
bash scripts/decision_supervision/run.sh plan
```

帧、checkpoint、source文件、接受报告、编译NMS二进制和脚本均有hash。trace必须覆盖全部800帧，不能只取722帧继续训练。`plan`固定全部训练分层索引；输出`plan.json`，后续不再按测试反馈更改。

### 6.3 训练、共同池重复审计

```bash
bash scripts/decision_supervision/run.sh fit
bash scripts/decision_supervision/run.sh audit
# 如分配不同GPU任务，只限定ARMS/SEEDS；最终全部12个头都必须报告。
# CUDA_VISIBLE_DEVICES=3 ARMS='A B' SEEDS=20261113 bash scripts/decision_supervision/run.sh fit
```

正式模型位于`heads/SEED/ARM/last.pt`。训练日志含逐epoch全部共同验证池指标、实际pair计数。audit把每个seed的四头同时评价相同U/D/union重复审计池，不允许跨池用selected q排名。

### 6.4 正式10%官方评测

```bash
SEED=20261113 ARM=A INTERVAL=10 bash scripts/decision_supervision/run.sh infer
SEED=20261113 ARM=A INTERVAL=10 bash scripts/decision_supervision/run.sh eval
# 依次完成 A/B/C/D × 三seed；默认rank_executed和both两个策略。
SEED=20261113 ARM=native INTERVAL=10 bash scripts/decision_supervision/run.sh infer
SEED=20261113 ARM=native INTERVAL=10 bash scripts/decision_supervision/run.sh eval
```

官方评价输出AP及K=1/10/50的prefix质量/μ1.2成功率。`padded_non_success_rate`把不足K的空槽也算非成功，**不是**真实已输出动作的假阳性率。完整q=0/NMS替换事件需要再用已验收官方动作trace核查，不从AP均值倒推出病例数量。

所有head使用相同snapshot seed20261113，保留完整基座bank几何hash。报告器拒绝不同输入bank、碰撞规则、source config、帧集或plan的配对。推断拒绝smoke权重和非epoch10权重。旧`rank_max/depth_only`通过`POLICIES`指定到新RUN_ROOT测试目录，不覆盖主策略结果。

### 6.5 统计与报告

准备`RUNS_JSON`，结构为`seed -> A/B/C/D -> test_seen/test_similar/test_novel -> inference目录`。每个目录必须已经执行eval。

```bash
export RUNS_JSON=/ABS/matched_runs.json
POLICY=rank_executed bash scripts/decision_supervision/run.sh report
POLICY=both bash scripts/decision_supervision/run.sh report
```

30场景为每split配对单位；Mean区间在三个split内分别抽30场景、等权平均。每seed结果单列，跨seed均值的scene CI注明条件于这些已训练head；seed标准差另列，不能把270 scene-seed组合当独立样本。

## 7. 验收、停止与结论模板

- [ ] 旧四头在同池重复原确认结果；state keys/参数量/source hashes匹配。
- [ ] 800帧关系trace完整；真实编译NMS保留动作身份与tie顺序；没有局部子集NMS冒充全局关系。
- [ ] A/C及B/D分别共享完全相同候选；固定64families/256动作；记录hard-negative不足和fallback，不将期望当实际命中。
- [ ] 三seed各四臂初始化一致、固定10轮1600次更新；loss无NaN，不以MSE自动选最有利checkpoint。
- [ ] 主要rank_executed不改变native动作几何；评分对应实际depth；both另列。
- [ ] official AP数组完整`[30,26,50,6]`；与Native、A基准同运行协议，不与历史100%结果混比。
- [ ] 明确报告Novel，不能用Mean掩盖Novel退化；对阴性结果不临时改变配额、边定义或权重。
- [ ] 跨seed效果与场景CI分别报告；重复查看的测试集仍是探索性证据。

结论应按干预回答：B−A改善说明此标签池内的采样组成有价值；C−A改善说明显式竞争监督有价值；D存在交互需与单因素对照共同解释。即使两项失败，也不能直接断言RGB物理信息不存在——本轮限定了已标注池、native参考关系、固定reader和小型头。

**暂不执行：**新增GNN、扩大GVAR、P2重新求解、全量基座重训、按Novel测试AP搜索超参。若hard-negative不足，先形成缺额审计，再提出固定预算训练域新增标签方案，不自动扩张实验。

## 8. 产物与当前验证边界

保留：交叉审计矩阵、trace shards/每帧hash、固定plan及sampling audits、12头protocol/history/pairs_epoch/last.pt、完整官方AP与prefix指标、每seed/scene差值、因素主效应和交互报告。

本实现已经在本地用**合成CPU数据**测试四臂训练、旧head兼容、交叉池评价、关系追踪接口、无重复pair与ties排除、阶段隔离和shell入口。真实117权重、扩展库、编译NMS及CAD标签验收仍需在上述运行环境中完成。不存在本轮新增真实AP数值。
