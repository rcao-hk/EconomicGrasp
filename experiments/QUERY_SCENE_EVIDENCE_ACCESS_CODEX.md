# Query–Scene Evidence Access：Local / Candidate / Scene 验证工作单

日期：2026-11-14。仓库：`rcao-hk/EconomicGrasp`。新分支：`exp/query-scene-evidence-access`。
直接父提交：`exp/interactive-grasp-stage1@a1caddd2de2637648730b4bdf2973b6b1607d56c`。
本轮不合并 graph 或 follow-up 分支，不修改生产模型、原训练器、官方 evaluator。

## 0. 问题与范围

Locked-alpha报告：native Mean AP=51.9024；A/B/C/D分别51.8533/51.7971/51.9421/51.8827；C相对native仅+0.0397pp，尚不能证明交互后再次取证的价值。本轮不继续寻找各组最佳alpha，而验证：

> 同一个实际抓取的质量判断，能否从共享场景特征中获得局部动作特征和其他候选特征没有提供的有效信息？

本实验只更新评分。原RGB/depth/GVAR/CDF/MSE全冻结；1024个实际动作、view/angle/center/insertion/width和sensor filter保持不变；不改变NMS，不做动作refinement、不称NMS-free。

### 主比较

|组名|每个动作的初始输入|新增上下文|用途|
|---|---|---|---|
|`local`|同一个pre-branch GVAR action feature + 完整动作参数|无；逐动作MLP|活跃参数容量控制|
|`candidate`|相同|1024个冻结动作的feature+geometry tokens|候选提供上下文|
|`scene`|相同|32×32共享图像网格tokens + 连续相机几何|场景直接提供上下文|

Candidate/Scene使用相同的cross-attention读取结构，固定source tokens、迭代更新query；两层之间**不更新memory**。Candidate不是旧C的逐层动态self-attention复现，而是为了隔离“memory来源”的新对照。Local不使用单token attention（Q/K会退化成无效参数），采用真实参与计算的MLP匹配预算。

没有graph、NMS edge、距离邻域mask、leader/trailer标签、对象ID或query序号embedding。相机几何和动作参数是物理描述，不规定候选竞争关系。

## 1. Scene memory的精确定义

从同一次原生GVAR forward的`reader.context`捕获：
- `pregeom`：进入GVAR probe采样前的完整图像feature map，不使用已经聚合的36个probe tokens构造场景。
- `depth`：与原GVAR完全相同的RGB预测深度。
- `K`、`image_hw`：模型输入裁剪图的相机内参与尺寸。

在固定32×32网格cell centers均匀采样，不依赖query分布、旧score、foreground GT或NMS关系。每个scene token包含原feature向量与8维连续描述：normalized UV、camera ray XY、预测camera XYZ、depth-known flag。

**“Scene”仅指模型输入crop内的完整特征图**，不声称恢复被workspace crop裁掉的区域，也不引入新观测。原数据预处理可能使用sensor/virtual depth做workspace crop；网络场景几何仍使用预测深度。官方sensor碰撞后处理保持原协议，因此不能宣称整个系统严格无任何传感器深度。

无效预测深度只令XYZ及depth-known为零，RGB token保留。完整feature map不可有NaN；若相机/shape不匹配，直接停止。

Hook只观察原reader，不改变state_dict或生产forward；异常也清理。保留原生17D decoder、reader replay和已有MSE dump的parity门槛1e-6。历史provider必须暴露`.model.kview_grasp_module.decoder.reader`及上述context；没有这些接口就需要显式接入adapter，不能用旧36-probe缓存冒充scene memory。

## 2. 训练和数据合同

### 冻结来源

绑定同一已核验的MIX50 GVAR和MSE sidecar；`native`指冻结MSE参考，而不是原CDF。基座与sidecar SHA、历史epoch、`grasp_design`、provider源码均保留。复用原Stage1 strict loader，不允许`strict=False`或伪造checkpoint metadata。

每个query仍由冻结MSE在原angle的四depth中选一个动作，再读取该depth对应的执行width。全1024query在sensor过滤之前进入三个新模型，sensor mask不输入评分器。

### 数据

|用途|场景与帧|两域图像数|每图质量监督|
|---|---|---:|---:|
|训练|scenes0–79，0,16,…,240|2560|32个均匀动作|
|验证|scenes80–99，8/72/136/200|160|**全部1024动作**|
|正式Original测试|scenes100–189，所有256帧|23040|推断不读标签|

新增模块阶段scene留出；冻结基座/MSE可能见过验证场景。Novel已用于前期研究探索，不继续用它选择架构或超参。

训练81,920个质量标签、验证163,840个；相同物理ID可以从已签名Stage1 cache索引复用，剩余由official CAD/Dex-Net重新标注。实际复用率由输出记录决定。

`--reuse-index`目前接受已签名Stage1或本实验索引；**不直接接受旧RARE index或follow-up label sidecar**。需要复用其他格式时必须先写显式、保留源SHA/scene/domain/action-ID的转换器，不靠query_id匹配。

未标注动作`quality_count=-1, label_mask=False`，参与context但不当作负例。validation完整标注用于区分完整候选前列质量与“标注子集Top1”。

### 模型和目标

- 两层，hidden128，4 heads，无dropout；query为原action feature＋连续完整动作geometry。
- `z_new = z_native_MSE + 0.20 * delta`。**0.20为三组共同、训练/验证/测试一致的预注册系数**，不是锁定alpha实验中各组单独挑出的值，不作新的alpha搜索。网络可能通过delta幅度补偿该系数，因此不声称它保证不退化。
- 三组最后scalar层零初始化；epoch0必须复现native logits。
- 损失统一`10*MSE(sigmoid(z), Q) + 0.5*pair`；同图非并列质量对，最多2048，按图平均。
- 20轮、effective batch4（单图＋梯度累积4）、AdamW1e-4、wd1e-4、clip1、seed0、12,800更新/组，固定末轮。
- 配对RNG、图像顺序相同，公共projection按独立component seed初始化。
- 根据实际feature/scene维度，通过真实活跃FFN宽度匹配参数；测试的128/256维组合差异均<0.3%。实际`model.describe()`为准，**不称计算量相同**。
- 新head固定FP32/TF32-off；推断临时切换后恢复，原冻结provider使用自身已验收precision，不破坏其parity。

训练、缓存和checkpoint均记录code/data/source hash；不同版本不能直接resume。仅支持epoch-boundary续跑；不支持任意中途batch精确恢复。

## 3. 输出和入口

- `query_scene/core.py`：uniform scene sampling、backprojection、data contracts。
- `runtime.py`：包裹原Stage1 provider，取完整pregeom scene map，移除无需保存的旧36-probe memory。
- `model.py`：Local/Candidate/Scene；活跃容量控制、evidence干预。
- `pipeline.py`：export/index/train/infer/eval，模型协议、完整固定动作与source一致性。
- `diagnose.py`：validation repair/damage、质量、排列/重复/scene错配与场景配对CI；额外head benchmark。
- `report.py`：三split官方AP和六μ/prefix，Scene−Local、Scene−Candidate、Candidate−Local、各组−native。
- `cli.py`、`scripts/query_scene/run.sh`：显式分阶段任务。

既有`interactive_grasp`文件不修改；graph任务可继续运行，本实验不导入graph标签/边/teacher。

## 4. Codex执行步骤

先读取AGENTS.md，检查git状态/worktrees、GPU/CPU和磁盘。不能清除旧实验、改动正在运行的graph进程或覆盖其目录。

```bash
git fetch origin exp/query-scene-evidence-access
git worktree add /home/robotarm/EconomicGrasp-query-scene \
  -b codex/query-scene-exec origin/exp/query-scene-evidence-access
cd /home/robotarm/EconomicGrasp-query-scene
bash scripts/query_scene/run.sh check

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/query_scene_run01
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export BASE_CHECKPOINT=/ABS/verified_mix50_gvar.pt
export MSE_CHECKPOINT=/ABS/verified_mix50_mse/last.pt
export DEVICE=cuda:0
bash scripts/query_scene/run.sh prepare
bash scripts/query_scene/run.sh bind
```

路径是示例；用户实际历史runtime若不同，显式设置`PROVIDER=verified_adapter:HistoricalRuntime`，保持其原生`.snapshot/.sensor_keep/.label/.model`合同和源码SHA。默认loader未验证用户服务器历史.pt；不承诺所有历史checkpoint直接兼容。

### 4.1 两域smoke，先验收提取再正式标注

```bash
# 可选：复用已签名Stage1索引；不能提供旧RARE索引。
export REUSE_TRAIN_INDEX=/ABS/stage1/train_index.json
export REUSE_VAL_INDEX=/ABS/stage1/validation_index.json
bash scripts/query_scene/run.sh smoke
```

检查每域原生decoder/reader parity、scene map完整性、相机坐标、完整1024动作、32×32scene tokens、无GT输入。验证scene sampling与proposal顺序无关。

只保存action features和scene features，不保存1024×36 probes；空间通常低于Stage1缓存，但以真实首帧测量为准。默认保留4GiB空闲。不能为节省空间偷偷删低分queries或只留foreground。

```bash
bash scripts/query_scene/run.sh export
# 可按帧分片：SHARDS=3 SHARD=0 DEVICE=cuda:0 ...；分别0/1/2，禁止重叠写。
bash scripts/query_scene/run.sh index
```

两组reuse索引应在train/validation export中统一给出，便于来源审计。缓存支持严格hash续跑，index默认拒绝缺帧。正式validation默认为全部1024质量，不以稀疏标签代替full-pool指标。

### 4.2 三组同预算训练

```bash
bash scripts/query_scene/run.sh train
# 或VARIANTS=scene DEVICE=cuda:2 ... 分别分配空闲GPU。
bash scripts/query_scene/run.sh diagnose
bash scripts/query_scene/run.sh benchmark
```

`history.json`包含epoch0和逐轮质量指标；`validation_last.json`保留逐帧输出。不能只看最终AP忽略Local是否已经退化。benchmark只测新增head前向，不包括冻结GVAR、CAD、传输或官方后处理；GPU峰值字段包含所有已加载heads及输入的驻留空间，不当作单head净增显存。

### 4.3 机制诊断的正确解释

`evidence_diagnostics.json`在完整验证集输出：
- 同一物理动作的MSE、tie-aware pair accuracy、repair/damage；
- 完整sensor-surviving pool的pre-NMS Top1/5/10/50 Q与标签覆盖；空候选集写null；
- 候选排列等变性；复制64个候选后的原候选评分/质量变化；
- `scene_joint_permute`：feature/geometry/mask一起重排，应保持结果，这是invariance test；
- `scene_appearance_shuffle`：只重排appearance，固定UV/ray/XYZ，破坏对应关系；
- `scene_swap`：换成**同RGB域另一验证场景**的完整scene memory；donor按固定规则选取，不依赖Q或分数；
- `scene_mean`：将appearance变成场景均值，保留空间位置，检查是否只使用全局外观统计。

复制候选后Local/Scene应基本不变，这是结构性质，不是性能证明。Candidate的密度敏感性需结合quality/repair-damage解释，不能仅凭logit变化称作物理推理。

所有错配均是测试时干预，可能分布外，**不能单独证明新信息贡献**。若Scene优于Local/Candidate，追加一个同预算训练控制：

```bash
python -m query_scene.cli train \
  --train-index "$RUN_ROOT/train_index.json" --val-index "$RUN_ROOT/validation_index.json" \
  --variant scene --train-scene-shuffle --output "$RUN_ROOT/control_scene_shuffle" \
  --epochs 20 --accumulate 4 --hidden 128 --heads 4 --layers 2 --residual-scale 0.2 --device cuda:0
```

这组只打乱训练时appearance/position对应，参数与标签不变；验证仍输入真实memory。它是条件性补充，不纳入三组主表自动汇总。pregeom本身可能已有全局语义/位置线索，不把shuffle视为彻底删除所有几何信息。

### 4.4 一帧正式链路smoke，再全量测试

手动`python -m query_scene.cli infer --help`查看参数；用`--limit 1`、三份`--weight`和实际MSE reference先验收。smoke推断不允许作为正式AP。

```bash
export REFERENCE_ROOT=/ABS/frozen_MSE_full_pre_NMS_dumps
bash scripts/query_scene/run.sh infer
bash scripts/query_scene/run.sh eval
bash scripts/query_scene/run.sh summarize
```

若reference按split分目录，逐个设置`SPLITS=test_seen`和对应REFERENCE_ROOT。每split7680帧、四份dump(native+3variants)、官方数组`[30,256,50,6]`。保存哈希、所有物理ID不变、同一sensor keep mask；官方eval.py未改。推断按帧流式取scene map，不保存全测试memory。

本版推断输出要求空目录，**未实现长推断的自动逐帧续跑**；失败后保留原目录并独立重跑相应split。不得把不完整manifest当成complete。官方eval可单独运行。

## 5. 主要判别和停止条件

1. **Scene仅胜native、却不胜Local/Candidate**：不足以证明场景证据有独立价值。
2. **Scene有优势且正确对应显著更好**：支持图像map证据可访问性路线，再考虑query-adaptive sampling；不马上宣称动作refinement已成立。
3. **Scene和均值/错配memory相近**：可能主要使用全局统计或忽略memory；先做匹配训练控制，不用attention可视化替代证据。
4. **候选上下文随重复数量改变而Scene稳健**：只有当质量选择也更好，才支持“scene优于proposal distribution”的研究结论。
5. **三组均退化**：如实报告，不测试集搜索alpha；先查已有history/质量分布/训练标签覆盖，不立即扩大网络。
6. **源权重、动作、参考dump、scene坐标、代码哈希不一致**：停止，不放宽parity或绕过strict loading。

主表报告Seen/Similar/Novel/Mean、六μ和prefix、场景CI、实际新增参数/延迟。CI是固定权重下场景变化，不包括训练seed；新模块验证图像对冻结基座不一定未见。

本轮没有实现物理动作更新、grasp-family counterfactual扩展、多视角、geometry belief、共享scene的GT重建监督、NMS-free或新global selector。Original/GNTrans目前分别生成自身冻结动作；**没有把相同query_id跨域强行当成同一物理动作**，也不宣称完成固定动作跨材质不变性验证。

## 6. 验收交付

- [ ] 本地CPU测试和shell/compileall通过；实际源码hash可追踪。
- [ ] 用户117历史runtime两域scene捕获和native parity通过。
- [ ] 2560/160全帧缓存、实际标签/复用计数正确，scene不依赖候选抽样。
- [ ] 三组模型初始native parity、同预算、训练/验证曲线、实际capacity和计算成本完整。
- [ ] validation真实质量对照、证据错配/不变性诊断与场景配对统计完整。
- [ ] Original三个split各7680帧，四组官方AP、源dump和物理身份逐帧校验。
- [ ] `RESULTS_CN_QUERY_SCENE.md`区分已支持机制、负结果和未验证解释。

实现交付仅包含本地CPU合成验收；没有访问117的真实GPU/CAD/weights，不能预填任何新AP或宣称正式训练已运行。
