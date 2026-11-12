# Interactive Grasp Stage 1：冻结动作、跨 query 交互与再次取证

日期：2026-11-13。Repo：`rcao-hk/EconomicGrasp`。
分支：`exp/interactive-grasp-stage1`。
**直接父提交：main@7a956d37aaa427a32e598f21fdc6b910ed3c2704。**
不合并 RARE/graph 诊断分支；不更改正常模型、训练器、官方评测器。

## 0. 第一阶段的研究问题

> 在固定的真实抓取动作集合上，跨抓取 query 交流之后重新读取动作证据，是否比独立更新、单独再次读取、单独特征交流更有效？

这是 **EASE/MonoDETR 启发的交互式质量判断**，不是 EASE-DETR 复现。
不使用 leader/trailer 标签、不规定旧高分者应该获胜、不使用显式 graph、NMS edges、半径 attention mask、物体唯一赢家或相对评分 softmax。
不声称 NMS-free 或 set-level optimum。正在运行的 graph oracle 独立继续，不能把 test oracle 或 test graph 标签用于本阶段梯度。

固定每张图全部 1024 个实际执行动作 `g=(t,R,d,w)`。新模块只改变 scalar score；不能改变任意物理参数、候选数、query source、碰撞条件或官方 NMS。

主 reference 使用 **历史 MIX50 GVAR + MSE scorer**（旧目录 `point` 指 MSE head，不是 Point-only），以 `native` 命名的是该冻结 MSE reference，不是最初的原生 CDF。原 CDF 17D decoder 仍单独作底层 parity 检查。以后使用 RARE reference 必须独立绑定和导出，不能与当前 cache 混合。

## 1. 核心 2×2 对照

|组|跨中心 query interaction|新增 GVAR-memory 读取|作用|
|---|---|---|---|
|A|无，用近似等参数的逐动作 FFN 替代|无，用逐动作 FFN 替代|独立 refinement 容量控制|
|B|无，FFN|有|Read-again only|
|C|有，全候选 self-attention|无，FFN|Interaction only|
|D|有，全候选 self-attention|有|Interact → Read|

每个 block：

```
current action state
  → LN → [full-candidate self-attention OR pointwise FFN] → residual
  → LN → [per-action evidence cross-attention OR pointwise FFN] → residual
  → LN → FFN → residual
```

默认 2 blocks、hidden128、4 attention heads、dropout0、read chunk256。
初始 query = pre-CDF/width-branch action feature + 连续完整动作几何编码。
几何包含 width/height/insertion、R9、camera-frame t3；不包含对象ID、query索引、NMS邻接或GT几何。
完整动作编码只描述物理参数，不人为指定谁与谁竞争。无几何更新，无中间 score feedback（第二阶段）。
最终 `z_new = z_frozen_MSE + residual`；新 scalar residual 最后一层零初始化。
所有组 epoch0 与原 MSE utility 一致（数值容差 ≤1e-6）。输出独立 sigmoid 质量，不对候选做总和为1的 quality normalization。

**不能只拿 D 与旧4673参数头比较然后声称交互有效。** 主要对比：D−B、D−C，辅以B−A、C−A和D−C−B+A。后者只是2×2实验交互项，不是物理因果贡献比例。

参数预算近似匹配，不用未参加计算的 dummy weights 凑数。F=128/E=64时：A399873、B400257、C400129、D400513；F=256/E=64时各增加16640。差异小于0.2%，报告实际 `model_description`，另外报告实测时延/显存；不称计算量完全相同。输入投影、相同类型模块及公共FFN按独立固定RNG初始化，训练图像顺序与pair RNG严格相同。

## 2. 冻结、证据与监督合同

### 2.1 所有实际动作冻结

- Source GVAR、DPT/backbone、RayRoPE、CDF/width heads、旧MSE head：全部 `eval()` + `requires_grad_(False)`。
- 原生最佳 angle 固定；MSE在该angle的4depth中选一个，使用对应的原生预测width（含1.2扩张/0.1m截断）。每个原始中心仅保留这一个实际动作，但保留全部1024中心。
- A/B/C/D 对同一1024动作只更新score。不能加深、变浅、移动center、改width或重新选择angle。
- 所有变体在同一帧使用相同sensor-collision keep mask，随后仍使用未修改的官方NMS与AP。
- 导出每个动作 `SHA256(float32 grasp[1:16])`；新评分必须保持所有physical IDs逐元素不变。

### 2.2 Evidence memory 来自原生 GVAR，不重写支持几何

`capture.py` 通过临时 hooks 捕获：
1. `decoder.width_proj` 的输入，即原生 action-adapter finish 后、CDF/width分支分离前的 action feature；
2. 原 score-branch feature（用于严格重现冻结 MSE）；
3. reader 的 queries、context 与原始 output。

完成完整 Q×A×D native forward 后，**按原action_chunk大小重放整个冻结reader**，只从其真实 MHA K/V 输入中采集最终1024个动作的36个tokens。
同时要求 reader重放output == 原reader output，误差≤1e-6；仍按原完整native decoder检查17D grasp误差≤1e-6。

不提前裁剪原Conv1d的长度，不用短输入重算旧score logits，避免历史TF32形状敏感性。
原生 all-invalid attention 为数值安全临时解禁第0个token，导出时必须根据真实projected-valid mask恢复“全部无证据”，不能将该token视为有效。
所有临时hook在异常时也移除，不替换生产forward，不改变state_dict。

GVAR现有0.06m support envelope和六role probes原样保留；它们仍是soft evidence，**不宣称这些tokens完整建模实际执行width对应的夹爪体积**。
第一阶段只检验交互与重新加权是否有效，不同时更改probes或采样策略。

### 2.3 完整上下文、稀疏标签

训练输入始终是完整1024个query，不先按旧分数top-k，不仅输入32个有标签query。
每训练帧默认32个均匀query有exact-action标签；validation默认128个。
其余候选 `quality_count=-1, label_mask=False`：作为上下文参与attention，**绝不当作失败动作**。

复用旧RARE per-frame NPZ时必须核对：source base SHA、domain/scene/frame、旧索引/文件hash、executed physical ID、六μ成功位与count一致、official evaluator来源。
**不能只按query_id复用标签**。不匹配的实际动作独立运行现有 `ExactGraspNetActionEvaluator(fc_mode='official', strict=True)`。
网络输入没有GT；CAD只用于离线label。Graph/test labels不进训练。

四组固定同一损失：`10*MSE(sigmoid(z), count/6) + 0.5*pair`。
Pair是同一图像有标签且质量不同的动作对，最多2048对/图；quality ties不排序。无margin、无方向偏好、无原动作必胜。图像内独立采pair并按图平均，不跨域跨图生成pair。

## 3. 默认原型数据与预算（在看新test结果前锁定）

这一轮不是再做全部MIX50 scorer训练，而是在其冻结基座上验证新架构。

|用途|场景|帧|两域总量|
|---|---|---|---:|
|新增模块训练|0–79|0,16,…,240，共16帧/scene|2560|
|新增模块验证|80–99|8,72,136,200，共4帧/scene|160|
|正式Original测试|100–189|全部256帧|23040|

两RGB训练域均用相同scene/frame；这些帧均属于50%偶数帧池。
Train/validation在**新增模块阶段**按scene留出；历史GVAR和MSE可能见过所有0–99场景的其他帧，**不能称整个系统未见过validation场景**。
Novel已经影响研究动机，是探索性测试，不要继续根据Novel调层数、epoch、loss权重或挑seed。

默认每组20轮、image batch1+4帧梯度累积（等效4帧）、AdamW lr1e-4 wd1e-4、clip1、seed0；固定末轮，不用test选模。每组12800 optimizer updates。先1seed原型，进入确认阶段需重复预先指定seed。

相同frame plan/labels/cache/source checkpoints，四组独立训练。先按scene覆盖控制原型规模，不重复先前仅192图/16val的窄覆盖配置。

## 4. 代码与入口

- `interactive_grasp/model.py`：A/B/C/D可训练模块；无graph/leader prior；per-action memory读取；eval-only pre-interaction-read-query干预。
- `capture.py`：原生完整长度捕获、reader重放、mask恢复；所有临时hooks异常清理。
- `native.py`：严格main模型/冻结MSE加载；预测深度检查；两域dataset、sensor过滤、exact标签。
- `contracts.py` / `data.py`：signed manifests、physical IDs、完整context+缺标签检查、旧标签按动作身份复用。
- `export.py`：两个域完整context缓存、分片/续跑、标签补全、索引。
- `train.py`：四组同监督预算训练、scene验证、末轮权重、epoch-boundary严格续跑。
- `infer.py`：流式完整native forward，一次提取供四模型打分；输出不变物理动作和相同sensor过滤；调用原eval.py。
- `analyze.py`：repair/damage、排列等变性、重复候选敏感性、D的交互前读取query干预；三split场景分层bootstrap。
- `scripts/interactive_stage1/run.sh`：check/prepare/bind/smoke/export/index/train/infer/eval/summarize。
- `tests/test_interactive_stage1*.py`：CPU单元与合成集成测试。

不导入旧RARE branch的模块；冻结source scorer的小型state-compatible读取器单独包含在capture.py。普通main推断不会导入该包。

## 5. Codex步骤

### 5.0 工作区和单元测试

先读取所有适用的AGENTS.md，检查git/worktree/GPU/磁盘。不能覆盖正在运行的graph oracle，也不能删除旧checkpoint或改变其分支。

```bash
git fetch origin main exp/interactive-grasp-stage1
git worktree add /home/robotarm/EconomicGrasp-interactive-stage1 \
  -b codex/interactive-stage1-run origin/exp/interactive-grasp-stage1
cd /home/robotarm/EconomicGrasp-interactive-stage1
bash scripts/interactive_stage1/run.sh check
```

### 5.1 绑定冻结MIX50 GVAR和MSE

```bash
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/interactive_stage1_run01
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export BASE_CHECKPOINT=/ABS/path/to/verified_mix50_gvar_e15.pt
export MSE_CHECKPOINT=/ABS/path/to/rare_mix50/heads/point/last.pt
export DEVICE=cuda:0

bash scripts/interactive_stage1/run.sh prepare
bash scripts/interactive_stage1/run.sh bind
```

路径仅示例，必须从既有执行记录核实。模板默认历史GVAR epoch字段16、每域12800训练图、selector objective=`point`；chunk/checkpoint flags从源checkpoint元数据同步。不能因路径含50%就认为符合合同，不能补造元数据。

模板中的matmul/cudnn TF32分别记录。真实源运行若不同，应在**首帧前**按原协议设置模板并另建run。严禁遇到parity失败后随意放宽容差。

**历史runtime边界：** 默认provider只支持当前main图、可信grasp_design、完整base和RARE sidecar metadata、严格state_dict。历史.pt缺元数据或模型图不一致时会停止；不自动`strict=False`、删权重key或伪造设计。
可以显式用 `--provider local_module:HistoricalRuntime` 复用117原严格loader，但必须独立保存adapter代码SHA，并提供接口：

```
__init__(config, dataset_root, gntrans_root, device)
.provenance: base_sha256, selector_sha256, config_sha256, camera, preprocessing...
.snapshot(frame, seed) -> (完整arrays, native_decode_max_abs/native_reader_replay_max_abs)
.sensor_keep(frame, grasps) -> bool[1024]
.label(frame, grasps) -> int count[0..6]
```

优先复用`capture.snapshot`，必须保留全部1024query、原feature位置与native reader token语义。自定义provider不是绕开核验的通道。**这个历史adapter的真实117接入未在交付时完成/验证。**

### 5.2 两域smoke与存储预算

```bash
# 旧索引可选；同一组reuse索引必须用于train和validation export，以保留一致来源。
export REUSE_TRAIN_INDEX=/ABS/path/to/rare_mix50/cache/train_index.json
export REUSE_VAL_INDEX=/ABS/path/to/rare_mix50/cache/validation_index.json
bash scripts/interactive_stage1/run.sh smoke
```

每域1帧、仅评价2个uniform query，但上下文仍完整1024。核验两种native parity、帧索引、预测深度、真实mask、physical IDs及hash。

**缓存空间必须先量测。** FP32 evidence在 E=64 时，每帧 `1024*36*64*4≈9MiB`（未压缩），还包括action features和metadata；2720个训练+验证帧未压缩至少约24GiB evidence。脚本输出首帧压缩空间外推并保留4GiB空余；不得因空间不足偷偷截断候选。
本版本是offline full-token cache训练；test采用streaming forward，不缓存整套test tokens。尚未实现“共享低分辨率feature-map压缩缓存”或在线训练provider。资源不足时，先登记更小的每scene视角协议而保留scene覆盖；可显式使用`--memory-dtype float16`且记录quantization error，A/B/C/D必须共用同一份量化cache，正式推断同步量化，不能混用。

### 5.3 导出全context＋稀疏标签

```bash
# 单进程；不能与graph标签任务无节制争用CPU/Dex-Net。
bash scripts/interactive_stage1/run.sh export
# 或分片：各进程分别设置不同空闲DEVICE与互斥SHARD。
# SHARDS=3 SHARD=0 DEVICE=cuda:0 bash scripts/interactive_stage1/run.sh export
# SHARDS=3 SHARD=1 DEVICE=cuda:1 bash scripts/interactive_stage1/run.sh export
# SHARDS=3 SHARD=2 DEVICE=cuda:2 bash scripts/interactive_stage1/run.sh export
bash scripts/interactive_stage1/run.sh index
```

train 2560×32=81920个稀疏标签，val160×128=20480；旧标签能复用多少由物理ID实际匹配决定，不预填。缓存仍有每帧全部1024动作/evidence。
匹配文件按hash安全续跑；孤立.tmp或数据/sidecar缺失不擅自删除，先调查。正式index不能接受missing frames。

### 5.4 A/B/C/D训练

```bash
bash scripts/interactive_stage1/run.sh train
# 也可显式分配不同GPU，每组独立输出，但全部超参数/数据/seed相同。
# VARIANTS=D DEVICE=cuda:2 bash scripts/interactive_stage1/run.sh train
```

`models/{A,B,C,D}/protocol.json` 包含实际维度、参数量、固定loss、来源SHA、数据索引SHA、训练预算。epoch0必须恢复原MSE。

训练中断使用CLI同参 `--resume`，只在epoch边界恢复；优化器、RNG、源代码/数据协议必须完全一致。没有“换了loss或数据继续训练并称同一次对照”的路径。

Validation的`labelled_subset_top1_q`仅是已标注子集的Top1，不是1024动作全集的Top1，也不是插入选择质量（depth已冻结）。不得误写成官方AP或insertion-regret improvement。

### 5.5 机制检查（先validation）

```bash
for V in A B C D; do
  python -m interactive_grasp.cli diagnose \
    --index "$RUN_ROOT/validation_index.json" \
    --weight "$RUN_ROOT/models/$V/last.pt" --partition validation \
    --output "$RUN_ROOT/diagnose_$V.json" --device cuda:0
 done
```

包含：
- 标签明确的candidate pairs上的repair/damage，排除Q ties；
- 打乱query顺序，恢复原序后logits必须一致；
- 追加64个重复candidate，量测原候选评分漂移，不强行宣称重复不变；
- 对D保留interaction residual，却让read用**交互前query**，检查“交流影响取证”的路径；仅是推理干预，不代替训练对照。

未实现第二阶段中间score反馈，也未实现score扰动反馈测试；第一阶段原score只用于最终residual锚点，不进入交互关系。

### 5.6 一帧真实推断 → 正式全量推断与AP

先手动CLI `infer --limit 1`，四组同时加载，检查输出物理IDs和sensor存活数全相同。正式要求已有MSE source pre-NMS dump逐帧parity，以避免历史runtime漂移被误称新架构收益。

```bash
export REFERENCE_ROOT=/ABS/path/to/frozen_MSE_full_dumps
bash scripts/interactive_stage1/run.sh infer
bash scripts/interactive_stage1/run.sh eval
bash scripts/interactive_stage1/run.sh summarize
```

`REFERENCE_ROOT`直接包含 `scene_0100/realsense/0000.npy` 等；若每split分别存放，按 `SPLITS=test_seen` 分别运行并设置正确reference root。
正式同一图像只提取一次frozen上下文供A/B/C/D使用，所有版本和native冻结MSE参考共享1024动作、sensor keep mask。没有新proposal生成、没有换depth/width、没有NMS修改。

每split **7680 = 30×256**。官方AP数组 `[30,256,50,6]`；三个split合计23040帧，每帧native+A/B/C/D共5份dump。当前默认plan只含Original正式测试，GN-Trans需要在看测试结果前另行登记plan/test rows，不能假装已完成。

训练未见test标签，推断不读exact labels。`eval.py`沿用main，结果来自服务器真实官方API，代码不自造AP。

## 6. 主要判别与结果表

|方法|Seen|Similar|Novel|Mean|每帧额外延迟|新增参数|
|---|---:|---:|---:|---:|---:|---:|
|冻结MSE reference|TBD|TBD|TBD|TBD|0|0|
|A independent|TBD|TBD|TBD|TBD|TBD|实测|
|B read-only|TBD|TBD|TBD|TBD|TBD|实测|
|C interact-only|TBD|TBD|TBD|TBD|TBD|实测|
|D interact→read|TBD|TBD|TBD|TBD|TBD|实测|

总结器输出D−B、D−C、B−A、C−A、D−A和D−C−B+A；按scene配对并在Seen/Similar/Novel内分层bootstrap，三个split等权，不把candidate/frame当独立scene。

- D仅优于旧MSE、却不优于B/C：不能宣称“交互后取证”有独立价值。
- D与C相近：可能现有feature交互足够，不支持再读的必要性。
- D与B相近：可能独立再次读取足够，不支持跨候选交互。
- Repair增加但damage同样增加/AP无收益：不称成功；应检查候选保留与噪声放大。
- 仅某个Novel开发case提高：不扩大到泛化结论；不根据Novel改层数再只报告最优。
- 第一阶段没有改变动作集合，所以不存在proposal oracle变化作为增益解释；graph oracle仍可作为独立上界诊断。

## 7. 验收边界

- [ ] 直接从指定main新建分支，无生产源码diff。
- [ ] CPU单元/合成训练/续跑测试通过。
- [ ] 原始/材质域真实native decoder与reader replay各通过；当前数据与source配置签名完整。
- [ ] 旧标签按frame+physical ID严格复用；未标注context不当负例。
- [ ] 全候选上下文和动作参数不变；source MSE dump逐帧复现；sensor mask一致。
- [ ] 四组源权重、data indices、loss、训练顺序/预算一致；实际参数量/时延/显存明确。
- [ ] 新方法的正式AP使用原官方后处理；没有用test graph或oracle训练。
- [ ] 最终结果有单seed限制说明，stage-only验证不夸大为全系统未见scene。

本交付只完成代码、工作单和本地CPU合成验收；没有117真实checkpoint、GPU、CAD数据，因此**未完成真实CUDA/CAD smoke和正式A/B/C/D训练/评测**。阶段2（预测反馈）、阶段3（物理动作更新/set-output）不在本分支范围。
