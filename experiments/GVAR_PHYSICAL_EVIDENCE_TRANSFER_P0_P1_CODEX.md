# GVAR Physical Evidence：P0 物理残差归因与 P1 独立验证协议

**日期：2026-11-16。继续同一分支：`exp/quality-strata-conflict-supervision`。不得新建实验分支、修改main、覆盖任何已验收数据或重选旧checkpoint。**

本工作单是**代码与运行协议**，不表示已经在117上执行了新的CAD、GVAR forward或独立场景测试。

## 0. 为什么需要这一轮

源依据：用户上传的 `physical_evidence_p0p1_to_novel_summary_20261116.md` 与已验收的40帧验证／780帧Novel分析。

- 冻结 MIX50 Point+GVAR36 e15 基座，SHA256 `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。
- 单seed20261113；P1 quality probe在基座训练域内的40帧验证中，all对score64的同动作/前排质量回归显著改善；NMS后Top10真实Q 0.773750 vs 0.688333。
- Novel测试780帧上 all 相对native AP −0.687474pp、相对score64 +0.070498pp；固定最终集合重新排序相对native损失−1.062575pp，NMS候选替换+0.468191pp起补偿作用。
- Novel Top10中零质量动作数量 native4856 vs all4859 几乎一致，但all的CAD碰撞多453、empty少434，强质量（≥5/6）动作少155。只看二元成功/零质量不足以解释AP。
- 在每帧三个方法Top10并集的相同动作上，all/score64 MSE验证0.066955/0.126298，而Novel0.244727/0.202301，发生质量回归方向反转。
- **目前没有证据证明特定视觉/姿态/几何线索是迁移失败根因；训练域验证的基座曾见过所有训练场景。**

**P0目标：** 在不训练条件下，沿相同物理动作，分析 all−score64 的预测残差、失败类型、object/pose、score分位、选入Top10及（仅有配准参考深度时）几何误差，解释哪些信息被过度依赖。

**P1目标：** 提供与冻结基座训练场景隔离的验证合同，冻结native/score64/all权重，逐场景复核前排真实Q、错误类型和（若有外部原生官方评估）正式指标。**不能将反复分析过的 GraspNet test_seen/similar/novel 重命名为独立holdout。**

不进行：新ranker训练、按Novel调参、GNN、阈值扫描、默认增加多seed、大规模基座重训。

## 1. 新增代码位置

| 路径 | 用途 |
|---|---|
| `diagnostics/evidence_transfer/contract.py` | 严格匹配的17D grasp身份、三模式分数、六阈值质量、CAD对象与失败类别、可选预测/参考几何；帧/manifest SHA |
| `validation_replay.py` | 320帧拟合之外的原定训练场景80–89，用同一冻结GVAR forward生成原生1024候选下native/score64/all的完整对照（每域20帧） |
| `collect.py` | 从已有正式推断dump或验证replay读取同动作候选；编译NMS并集，复用已有CAD标签或显式重新标注；测试帧必须官方50×6 AP逐元素parity≤1e−6 |
| `residual.py` | 不重训的同动作delta分数、MSE、强质量、碰撞/empty/other-zero、场景/对象/姿态条件统计与scene-bootstrap |
| `intervene.py` | 冻结all质量头下零化/置乱预定义GVAR证据组及六role的敏感性实验；仅标签已存在的验证/审计场景 |
| `index.py` | 建立真正有SHA的CAD标签索引、验证/Novel多manifest严格合并、基座训练／历史开发场景清单与外部数据索引 |
| `holdout.py` | 验证基础训练、开发历史、外部真实来源和无重叠；同样的native/score64/all质量对照；无资格不能声称independent |
| `scripts/evidence_transfer/run.sh` | 全部独立执行入口 |
| `tests/test_evidence_transfer.py` | 无CAD和CUDA的合成单元测试 |

**关键边界：** 共同动作NPZ需要经过真实模型或者实际acceptance数据产生；“名字相同的文件”不是证据。脚本不会通过缺失字段生成替代质量标签、归属对象、GVAR几何或数据集来源。

## 2. 原服务器与 worktree

根据2026-11-16报告，用户已有：

- 源码原worktree `/home/robotarm/EconomicGrasp-evidence-p0p1`；
- 旧结果根 `/data/robotarm/result/grasp/rgbgrasp/evidence_p0p1_20261115`；
- 后续诊断子目录 `selection_audit_20261116`；
- 旧输出包含权重、原始17D推断dump、诊断用CAD标签。**具体NPZ/JSON文件列表必须先在117实地检查，不按报告标题猜文件结构。**

推荐继续同一分支的新worktree/独立目录（不新建GitHub远端分支）：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-evidence-transfer \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-evidence-transfer
git status --short
git worktree list
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/evidence_transfer_20261116
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/ABS/confirmed_GN-Trans
export BOUND_CONFIG=/ABS/confirmed_mix50_gvar_bound.json
export DEVICE=cuda:0

bash scripts/evidence_transfer/run.sh check
```

真实GPU扩展、旧MinkowskiEngine、float64 CAD evaluator、NMS tie handling用之前已验收的117 runtime；未通过state_dict或hash不能用strict=False/伪造metadata来继续。

## 3. P0-A：在验证场景重新生成全1024的同动作三模式dump

`EVIDENCE_CACHE_MANIFEST` 是已验收的400帧 evidence cache的 `manifest.json`；score64/all为相同缓存来源、固定last-epoch10的质量头。不能用失败分类checkpoint替代。

```bash
export EVIDENCE_CACHE_MANIFEST=/ABS/evidence_p0p1/evidence_cache/manifest.json
export SCORE64_WEIGHTS=/ABS/evidence_p0p1/p1_heads/quality/20261113/score64/last.pt
export ALL_QUALITY_HEAD=/ABS/evidence_p0p1/p1_heads/quality/20261113/all/last.pt

# 单帧smoke：必须独立目录
LIMIT=1 VALIDATION_REPLAY_ROOT="$RUN_ROOT/val_smoke" \
    bash scripts/evidence_transfer/run.sh validation-replay

# 原定训练域留出，Original20帧（基座已见过这些场景）
DOMAIN=original bash scripts/evidence_transfer/run.sh validation-replay
# 如果需要GNTrans20帧，另设 DOMAIN=gntrans，输出不得合并为同一Original协议
```

每帧强制检查：冻结基座SHA、配置/代码来源、原生角度与depth、完整1024×4 bank物理身份、旧缓存 score64/base_logit一致、三评分方案 center/R/width/depth相同、预测未读CAD标签。

结果目录含三个 `native/`、`score64/`、`all/`，其下对应的 `inference.json` 和post-sensor-pre-NMS 17D npy。验证场景的AP不是官方test AP；不能称其为基座unseen验证。

## 4. P0-B：从已经验收的Novel／验证dump重建可引用的共同动作质量集合

准备实际 `ACCEPTED_RUNS_JSON`，格式为：

```json
{
  "native": "/ABS/accepted/native/inference_root",
  "score64": "/ABS/accepted/score64/inference_root",
  "all": "/ABS/accepted/all/inference_root"
}
```

Novel目录需要含对应的 `inference.json`、`evaluation.json`、`ap_test_novel_realsense.npy` 与每帧 `N×17 .npy`；collector逐帧验证三者已验收的 source SHA、完整候选几何、编译NMS survivor，以及各自官方 `50×6` AP 完全相符。验证场景则使用上一节创建的相同目录，不要求不存在的官方test AP。

**优先复用旧精确标签，不要无条件重做126,974个CAD评估。**

若已有精确动作label NPZ可以导出包含以下字段：
`physical_id,assigned_obj,pure_collision,empty,friction`，用现成NPZ路径和SHA组成一个 `EXACT_LABEL_FILES_JSON`，每条另包括 `dataset,split,scene,anno`。建立签名索引：

```bash
export EXACT_LABEL_FILES_JSON=/ABS/verified_existing_label_files.json
bash scripts/evidence_transfer/run.sh index-labels
export EXACT_LABEL_MANIFEST="$RUN_ROOT/exact_label_index.json"
```

与当前三模式NMS并集的物理动作ID不完全匹配时直接停止；如确需补标，明确单独启用 `ALLOW_CAD_RELABEL=1` 并计算时间/内存。不得将旧all/score64标签直接按行号贴在不同几何上。

```bash
export ACCEPTED_RUNS_JSON=/ABS/accepted_novel_run_roots.json

# 小规模调试必须新路径
LIMIT=1 COLLECT_ROOT="$RUN_ROOT/novel_collect_smoke" \
    bash scripts/evidence_transfer/run.sh collect

# 已验收标签reuse条件满足后全780帧
SPLIT=test_novel COLLECT_ROOT="$RUN_ROOT/collected/test_novel" \
    bash scripts/evidence_transfer/run.sh collect
```

如果要分析**真实预测几何误差**，附加 `CAPTURE_GEOMETRY=1` 让冻结GVAR在该帧重新读取predicted-depth gripper volume，并严格验证对应物理动作。只有提供**原像素对齐、单位显式声明**的 `REFERENCE_DEPTH_ROOT`，才计算参考深度下的相对几何差异；它是离线参考，绝不可进入score推断。没提供reference时只报告抓取姿态、位置和reader几何证据，不声称depth error。

同样创建一次Original验证专用 `ACCEPTED_RUNS_JSON` 指向 `validation_replay_original/{native,score64,all}`，使用 `SPLIT=probe_validation`；需复用验证标签或显式CAD补标。GNTrans独立用 `--dataset GNTrans`（或直接CLI），不能与Original混为一份label协议。

将两份**完整且签名已核验**的Original validation与Novel manifest合并，避免用不同动作池定义的指标直接作“迁移反转”结论：

```bash
export MERGE_MANIFESTS="$RUN_ROOT/collected/probe_validation/manifest.json $RUN_ROOT/collected/test_novel/manifest.json"
bash scripts/evidence_transfer/run.sh merge-manifests
export ACTION_MANIFEST="$RUN_ROOT/merged_matched_manifest.json"
```

合并会重验每个NPZ的物理动作、精确标签和SHA，重复scene/frame被拒绝。两部分条件分布相同指的是**相同筛选机制**（三模式compiled NMS survivor并集），而不是验证与Novel的动作在物理上完全相同，后者不可能成立。

## 5. P0-C：精细残差归因，限制事后故事

```bash
# 若已构建merged manifest，ACTION_MANIFEST指向其路径；
# 否则对单一split的manifest分别输出独立P0报告。
export ACTION_MANIFEST="${ACTION_MANIFEST:-$RUN_ROOT/collected/test_novel/manifest.json}"
bash scripts/evidence_transfer/run.sh p0
```

输出：
- `per_action.csv`：native/score64/all同动作预测Q、真实Q、δ=all−score64、各自误差、collision/empty/other、Top10与Top50身份、物体分配、score百分位与姿态；
- `per_frame.csv`、`per_scene.csv`、`per_object.csv`：场景和物体条件统计（物体ID按scene局部，不虚构类别标签）；
- `strata_failure.csv`、`strata_native_high.csv`、`strata_q_strong.csv`、`strata_polar_bin.csv` 等；
- `summary.json`、`REPORT_CN.md`：以scene配对的条件bootstrap；不把动作/对象当独立场景。

尤其关注：
1. q=0 collision 的平均δ和高估率，与验证的空抓取改善是否发生权衡；
2. 强质量 \(Q≥5/6\) 动作是否被系统性降分；
3. 本轮训练域验证和Novel是否在**同一种选择条件**（三个方法NMS survivor并集及Top10身份）下出现评分残差反转；
4. 结果是否被少数场景/物体/姿态箱主导；
5. 预测几何误差只能在已知参考几何且mask有效的样本上报告，不把projected-depth residual等价于CAD无碰撞真值。

不进行新的依据Novel AP选择阈值/特征/epoch；这些统计只定位机制，不能证明appearance shortcut或可迁移性因果机制。

## 6. P0-D：冻结证据干预（先验证，不重训）

在已验收400帧 `evidence_cache/manifest.json` 和固定 `all` head 上，对场景80–89实施预先定义的掩码/置乱：

- mask predicted-depth relative geometry；
- mask pre-reader / post-reader / reader-delta；
- mask全部role evidence或只mask六角色之一；
- 在同一帧中置乱role tokens或pred geometry作action alignment负控制。

```bash
export EVIDENCE_CACHE_MANIFEST=/ABS/accepted_evidence_cache/manifest.json
export ALL_QUALITY_HEAD=/ABS/accepted_all_last.pt
bash scripts/evidence_transfer/run.sh stress
```

输出相同动作上的score shift、按success/collision/empty/other分类的MSE变化、Top10质量。这些都是**测试时OOD敏感性操作**：零化role并不能严谨地“删除真实接触”，role tokens本身混合appearance、metadata与预测几何；需要受控学习实验或独立证据才能进一步称为因果机制。

## 7. P1：真正与基座训练场景隔离的验证

### 7.1 必须先建立真实的来源清单

基座历史训练场景包括GraspNet train 0–99；当前项目已反复查看 GraspNet test_seen/similar/novel 100–189。**这些后者虽然在基座训练外，已经不满足“未参与研究开发”的untouched条件。**

因此P1要有两层不同的独立性：
- `base_unseen`：不在冻结基座训练场景；
- `untouched_for_model_development`：不在本项目历次定性/定量开发使用场景。

对象/CAD模型独立性是第三层，需要单独的CAD、object或mesh overlap核查；单纯不同scene ID不证明没有对象交叉。

准备文件：
- `BASE_TRAIN_SCENES_JSON`：真实全量base训练场景的 `["GraspNet::0",...,"GraspNet::99"]` 形式JSON数组；按真实数据补充其他域/场景，不能漏项；
- `DEVELOPMENT_SCENES_JSON`：真实历次使用场景的数组，**必须包含GraspNet 100–189**，以及曾用于实验的其他数据集场景；
- `OBJECT_OVERLAP_REVIEW_JSON`：对不同object/CAD的真实review证据（没有做过就明确写 documented=false）。
创建带真实源文件SHA的可重复ledger：

```bash
export BASE_TRAIN_SCENES_JSON=/ABS/base_train_scene_keys.json
export DEVELOPMENT_SCENES_JSON=/ABS/all_prior_development_scene_keys.json
export OBJECT_OVERLAP_REVIEW_JSON=/ABS/object_overlap_review.json
bash scripts/evidence_transfer/run.sh make-ledger
export BASE_LEDGER="$RUN_ROOT/base_development_ledger.json"
```

文件的“签名”是 SHA256 完整性校验，不是外部第三方签章；历史记录的**完整性仍需人工独立审查**。

### 7.2 为新数据集准备同动作源

建议先审查用户已有 GraspClutter6D / HouseCat6D / GNTrans 等资源，但**不能假定任一数据集未曾参与本项目、对象不重叠或已经拥有兼容的执行动作CAD evaluator**。

必须首先由该数据集自己的、经过复核的 RGB-only frozen GVAR inference adapter，在一组同一物理动作上提供：
- native、score64、all 三种不变center/R/width/depth的分数；
- 真实该数据集 evaluator 的六阈值quality bits、collision/empty和按其定义的assigned_obj；
- 三方法的top10/top50标记及native score percentile；
- 每帧N×17 grasp、physical SHA、真实dataset version、来源与evaluator记录；
- **外部指标不能直接称为GraspNet官方AP**，除非实现了相同的评价合同并独立核对。

当前 repo的 `rare_grasp.runtime.Runtime` 是GraspNet读入与evaluator路径，不能因换目录就宣称已支持GraspClutter6D。请Codex先读数据集接口及协议，明确实现相应dataset-native adapter之后再转换成 `contract.py` 的帧NPZ格式。缺源不能凭空补齐或跳过CAD标签。

创建外部帧索引 `EXTERNAL_FRAME_INDEX`（JSON数组，含scene/anno/file/sha256），同时必须记录 `EXTERNAL_DATASET_PROVENANCE` 与真实 `EXTERNAL_EVALUATOR`：

```bash
export EXTERNAL_FRAME_INDEX=/ABS/external_frame_files_and_SHA.json
export EXTERNAL_DATASET_NAME=ActualDatasetName
export EXTERNAL_DATASET_PROVENANCE=/ABS/dataset_version_protocol.txt
export EXTERNAL_EVALUATOR=ExactExternalEvaluatorName
export PRIOR_USE=never  # 仅在有真实历史证据时；否则 used

bash scripts/evidence_transfer/run.sh index-external
export ACTION_MANIFEST="$RUN_ROOT/external_manifest.json"
REQUIRE_UNTOUCHED=1 bash scripts/evidence_transfer/run.sh holdout
```

遇到训练场景交叉、此前已检查场景、缺数据集来源SHA、缺evaluator或无质量标签，一律fail closed；允许 `REQUIRE_UNTOUCHED=0` 输出**探索性**报告，但不得换个名称称为独立确认。

输出 `holdout/summary.json, per_frame.csv, RESULTS_CN.md`；scene-bootstrap以场景为重采样单位，报告同动作quality MSE、Top10Q、Top10 strong-Q、collision/empty组成。若外部具有独立官方AP，按其原始任务协议另测并说明，不能把本脚本Top10Q称为官方AP。

**重要：**这套验证工具不会自己生成新数据集或实验结果；只有上述数据源和eval adapter确实存在，并通过完整性和独立性门禁时，才有资格做“独立场景验证”结论。

## 8. 验收和终止条件

- [ ] 原分支和main身份确认；新文件在相同远端分支，禁止覆盖旧 `selection_audit_20261116`。
- [ ] CPU合成测试、Python `compileall`、Shell `bash -n` 均实际PASS；如失败先修复，不能继续真实结果。
- [ ] 用真正已验收的native/score64/all N×17动作对照，物理ID、来源config、snapshot seed和并列NMS顺序逐帧一致。
- [ ] GraspNet正式测试collect逐帧端点官方50×6 AP parity≤1e−6；接受的6阈值标签与真实CAD对象关联正确。
- [ ] Label cache复用必须有原始物理ID、真实 assigned_obj/pure_collision/empty/friction和SHA；不符合就显式重评或停止。
- [ ] P0 residual所报scene/object/pose/geometry分层可追溯，未提供reference时几何误差字段缺失而不是被臆造。
- [ ] Evidence intervention不改变任何 grasp、label或checkpoint；干预只能称OOD sensitivity，不能作为物理因果证明。
- [ ] P1 ledger包含已知base GN train0–99、项目已看过的GN test100–189，实际来源文件hash匹配；物体重叠另审。
- [ ] P1严格拒绝把现有GraspNet Novel命名为untouched；新external adapter若尚未实现则停止，而不是声称P1完成。
- [ ] 新AP权重/阈值/训练不根据已经多次查看的Novel调节；所有报告明确单seed/基础模型冻结及scene-bootstrap限制。
- [ ] 结果必须单列：物理质量可解码、错误类型、评分迁移与最终动作选择，不能把某一项改善替代其他项。

**本工作单状态：**代码和静态执行协议已写入GitHub；截至本次答复，还没有117上的新验证/新增GT测试结果。CPU合成测试是否完成需以实际日志为准，不能凭代码存在声称通过。
