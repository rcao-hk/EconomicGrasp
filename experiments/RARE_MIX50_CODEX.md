# RARE-style Grasp Ranking — 50% Original + 50% GN-Trans（Codex 执行工作单）

**日期：2026-11-10。分支：`exp/rare-ranking-mix50`。**
本分支从 `exp/rare-ranking-mix10@ed1f0881aa9e9a89c5396a1c77a3326a70f1b4db` 分出；最初 main 基线为 `7a956d37aaa427a32e598f21fdc6b910ed3c2704`。
不要在 main 上覆盖实验文件，也不要改写、重用 10% 的 cache、head、结果目录。旧文档 `experiments/RARE_MIX10_CODEX.md` 仍是 10% 协议记录。

## 0. 研究目标与已知结论

用户现有 10%+10% RARE 实验在历史 Point RayRoPE+GVAR36 e19 上**已完成**：

|方案|Seen AP|Similar AP|Novel AP|Mean AP|
|---|---:|---:|---:|---:|
|Native|71.674149|55.006352|21.665119|49.448540|
|MSE-only|71.898279|55.153868|21.867721|49.639956|
|RARE|71.930260|55.308792|21.894000|49.711017|

上表均为 **Original GraspNet、每split780帧、collision-on、both**；RARE − MSE Mean = +0.071061 pp，scene-bootstrap 95%CI [+0.006365,+0.131863] pp。只有单训练seed；留出帧上的 Selected Q/MSE 没有改善，因此尚不能宣称 insertion selection gap 已解决。

**本轮目标**：把同一 RARE-style loss 和匹配 MSE 控制迁移到 **50% Original + 50% GN-Trans** 的共同基座上，完成训练、真实动作标注、相同预算评分器微调、三split推断及官方 AP。主要比较仍是 **同基座 RARE−MSE**；其次比较 Native、depth-only / rank-only / both。检查增加训练数据是否增强 RARE 在 Novel 的效益，而不是默认 10% 的正信号会放大。

RARE 仅迁移其 learning-to-rank 思想：
- 真实 Q = 6 个 `mu={.2,.4,.6,.8,1.,1.2}` 下对完整执行抓取的 exact CAD/Dex-Net 成功位均值。
- Absolute quality loss = MSE(sigmoid(z),Q)，relative loss = `softplus(-sign(Q_i-Q_j)*(z_i-z_j))`。
- 只使用**同图内非并列质量**的pair，最多2048对/帧；不对等质量动作施加强制margin、无加深方向偏好。
- MSE和RARE使用**完全相同的4673参数scalar ranker**、相同exact labels、初始化与优化预算；两组区别只是pairwise loss。不是完整 RARE 3D detection 复现。

**训练阶段**：S0 可全新训练共同Point RayRoPE＋GVAR36基座；S1/S2冻结全部基座，仅拟合新的scalar scorer；S3推断冻结view、angle、center和每depth执行width，仅对score/depth/query priority重排序。**不是用RARE损失端到端更新整套网络**。

## 1. 50%训练与10%测试：固定、不混用

|项目|新协议|
|---|---|
|原始训练域|GraspNet `scene_0000–0099`，每场景 frame `0,2,...,254` = 128帧，共**12,800**|
|材质变化训练域|GN-Trans 同一scene/frame，**12,800**帧|
|训练总量|**25,600**帧，两域1:1；每域50%确实为128/256|
|验证|训练域内scene0–9、frames5/135，两域40帧；仅帧独立，**不是独立场景**|
|测试|延续10% RARE：Original Seen/Similar/Novel，每split 30场景×26帧=**780**；GN-Trans可选单独评测|
|S0全新共同基座|20 epochs e0–e19、训练seed0，其他模型损失与main保持一致|
|S1 actions|每帧均匀32个native query ×4depth =128执行动作，train **3,276,800**，val **5,120**|
|S2 ranker|20 epochs、每batch4 frames、6,400 updates/epoch、共**128,000 updates/组**，两组同预算|
|S3|每帧原生完整1024 query，包含所有angle选择与native逐depth width；ranker不修改原动作生成器|
|后处理|Model-free collision detector：Original sensor cloud、voxel0.01、threshold0.01、approach0.05；两域使用相同来源|
|AP evaluator|main原 `eval.py` 与GraspNet官方API，六摩擦阈值，`[30,26,50,6]` 数组；与10%实验测试帧完全一致|

**数据放大约4.923倍**（25,600/5,200），所以必须在真实117服务器检查显存、存储和标签速度，避免在现有少量剩余磁盘上同时开过多缓存和官方评测池。新的shell默认基座要求数据盘至少12GiB空余、标注至少6GiB空余；这些只是最低防护，不保证完整运行一定有足够空间。禁止通过删除其他人训练结果或历史证据来腾空间。

**GN-Trans 的预处理可能仍使用 virtual/rendered depth 做 workspace/crop**；网络的 RayRoPE/GVAR 几何仍必须来自 RGB 预测深度。原传感器点云碰撞预筛是网络外后处理；结果不能描述为“全系统无任何额外深度信息”的严格单图 RGB-only。

## 2. 基座准备：选一种路径，不要混称

### 路线 A：从main架构全新训练50%+50% e19（标准主协议）

新 `rare_grasp/train_base.py` 继承 `train_cva_gntrans_mix_ddp.py`，以 `--mix_train_fraction 0.5` 使用逐场景 stride2 的拼接数据。原始网络任务loss和优化器不变。验证被替换为上述40张训练域额外视角，**不使用test_seen做选模**。生成 `checkpoint.tar` （每轮覆盖）和少量区间快照；`--ckpt_save_interval 20`，所以绑定用 `base/checkpoint.tar`，**不是 `checkpoint_19.tar`**。

脚本默认：3GPU、per-GPU batch3、lr3e-4、weight decay/depth decay1e-3、20 epochs、seed0；使用 `rayrope_group_chunk=2048`、`gvar_action_chunk=2048`、关闭两处activation checkpoint。这些是本实验**规定的配置**，不是对所有硬件都保证无OOM。若显存不足，需先调整方案、同步记录源签名与配置，并用实际checkpoint设计校验；不得悄悄改变训练协议还沿用同一实验名。

### 路线 B：复用已训练的50%+50% checkpoint（降低重复训练成本）

用户此前记录的历史 `point_gvar36_50pct_20261016/train/checkpoint_latest.pt` 实际只完成 e0–e15。**只有在真实文件包含完整、可信的GN-Trans mixed协议（每域12,800/总25,600、paired frame hashes、真实数据分布、严格架构元数据），并能数值严格加载时**，才可以设置 `HISTORICAL_E15=1` 显式绑定。

该模式要求 checkpoint `epoch=16`，并记录为 `historical_e15_seen_monitored`。不能把它称作重新训练的e19，也不能把历史用test_seen验证与路线A的训练域留出帧验证混同。脚本会从checkpoint设计元数据同步 RayRoPE/GVAR chunk和checkpoint flags，保留严格 `validate_checkpoint_design` 与 `load_state_dict(strict=True)`；其他模型图不兼容仍必须报错，**禁止人为补造mixed协议元数据、直接设strict=False或改写权重**。

如果历史checkpoint没有可信的完整mixed协议，正确处理是停止该复用路径并运行路线A，而不是按路径名字猜测它符合50%条件。

## 3. 新文件与扩展

- `rare_grasp/core.py`：`make_protocol(train_fraction=0.5)` 产生新50协议；`make_protocol()` 保留10协议及原SHA行为；`verify_protocol` 精确重算。
- `rare_grasp/runtime.py`：`check_mix_protocol` 按fraction核对checkpoint的真实训练数量/paired hashes/epoch；`bind_config` 支持 `--historical-e15`；继续使用严格native decoder 17D parity。
- `rare_grasp/train_base.py`：S0支持0.5与0.1；对真实训练子集和验证子集逐帧核验，不混用test scene。
- `rare_grasp/pipeline.py`：manifest CLI `--train-fraction 0.5`；cache、rank训练、inference均校验绑定checkpoint与frame协议一致；正式fit必需25,600train/40val。保留原3×780 test协议。
- `configs/rare_mix50/{point_gvar,point}.json`：新50模板。bind后保存真实checkpoint文件SHA和设计信息；不要用模板直接运行模型。
- `scripts/rare_mix50/run.sh`：显式stage，磁盘空余检查、S0、strict bind、两域smoke、S1分片、index、S2、S3、compare。可以分别 `infer` 与 `eval`，方便失败恢复。
- `tests/test_rare_mix50.py`：采样/验证不交集、反串数据集/错误epoch、历史模型显式授权、内存chunk同步、正式fit拒绝不完整cache以及shell语法。
- 本工作文档与10%工作文档并存；不修改任何生产网络、训练主脚本、官方evaluator或已有10%结果。

## 4. Codex 执行指令（先读本文件，严格分阶段）

先在117读取任何 `AGENTS.md`，`git status --short` 与 `git worktree list`，确认GPU、目录权限、磁盘空间、GNTrans帧数量。不要杀其他任务、删缓存、自动挪动历史checkpoint或强推任何已有分支。

```bash
git fetch origin main exp/rare-ranking-mix10 exp/rare-ranking-mix50
git worktree add /home/robotarm/EconomicGrasp-rare-mix50 \
  -b codex/rare-mix50-exec origin/exp/rare-ranking-mix50
cd /home/robotarm/EconomicGrasp-rare-mix50

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/rare_mix50_run01
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export ARCH=point_gvar
export DEVICE=cuda:0

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash scripts/rare_mix50/run.sh check
bash scripts/rare_mix50/run.sh prepare
```

路径为示例。现有10%路径与旧50%GVAR训练目录严禁用作 `RUN_ROOT`。若系统显存/磁盘无法完成主配置，先形成资源报告与可控的缩小协议，不把未完成结果填成完成。

### 4.1 S0 新训练（或经严格检查跳过）

```bash
# 路线A：原始网络新训练20轮
GPUS=0,1,2 BATCH_SIZE=3 bash scripts/rare_mix50/run.sh base
bash scripts/rare_mix50/run.sh bind

# 路线B：历史50%+50% e15，必须有完整可信元数据且严格兼容；否则停止
# BASE_CHECKPOINT=/ABS/history/checkpoint_latest.pt HISTORICAL_E15=1 \
#   bash scripts/rare_mix50/run.sh bind
```

绑定后检查：模型实际e19/e15、所记录frame hash、`grasp_design`、fusion监督配置、所有参数严格加载。两个路线必须在输出报告中使用不同的身份标签。全新20轮才可叫fresh e19。

### 4.2 S1 两域单帧smoke与大规模精确标签

```bash
bash scripts/rare_mix50/run.sh smoke
```

这一步必须通过：原生1024×12×4 CDF、原生17D抓取与本地decoder逐元素parity、prediction depth == used geometry、64维score-branch features、每depth native实际width、CAD评价有限性以及无旧有效mask。

正式S1：

```bash
# 单GPU顺序执行（默认）：
bash scripts/rare_mix50/run.sh mine

# 也可以三GPU分片，各命令在不同shell/分配实际空闲GPU后启动：
# NUM_SHARDS=3 SHARD_INDEX=0 DEVICE=cuda:0 bash scripts/rare_mix50/run.sh mine
# NUM_SHARDS=3 SHARD_INDEX=1 DEVICE=cuda:1 bash scripts/rare_mix50/run.sh mine
# NUM_SHARDS=3 SHARD_INDEX=2 DEVICE=cuda:2 bash scripts/rare_mix50/run.sh mine

# 仅在全部分片完成后执行严格index：
bash scripts/rare_mix50/run.sh index
```

每个训练frame32 family，每族4个depth，准确标注包括预测中心、旋转、执行宽度1.2扩张/0.1m截断之后的完整17D动作。Collision/empty须成为实际的有效负例；GT 仅用于离线 evaluator，不传入RGB网络。验证40帧同样标注。

所有缓存按frame保存动作参数、labels、源权重SHA、代码SHA、协议hash；运行中断后可按哈希安全复用已完成的同协议frame。**不得将mixed10的665,600动作cache直接用于mixed50模型，因为基座动作分布及标签身份不同。** 不能只复制ranker权重当作50%训练结果。

### 4.3 S2 严格预算匹配的MSE vs RARE

```bash
bash scripts/rare_mix50/run.sh fit
```

产生 `heads/point/last.pt`（MSE-only）与 `heads/rare/last.pt`（MSE+pair），分别保存20轮、128,000次optimizer updates、每轮validation MSE/Q，固定epoch20为主报告。原始CDF/width、GVAR、RayRoPE全部冻结；不会将新的scalar ranker解释成六μ的概率校准模型。

必须输出与10%报告可比的 pair 对数、ties比例、训练/验证质量直方图，并记录数据源 Original/GNTrans 分域差异。注意: 50%更大不保证validation MSE或AP更好。

### 4.4 S3 原生1024-query推断与官方AP

```bash
DOMAIN=original bash scripts/rare_mix50/run.sh infer
DOMAIN=original bash scripts/rare_mix50/run.sh eval
DOMAIN=original POLICY=both bash scripts/rare_mix50/run.sh compare

# 若需材质域泛化，独立在GNTrans上做同协议推断/评价：
# DOMAIN=gntrans bash scripts/rare_mix50/run.sh infer
# DOMAIN=gntrans bash scripts/rare_mix50/run.sh eval
# DOMAIN=gntrans POLICY=both bash scripts/rare_mix50/run.sh compare
```

每split运行Native、MSE、RARE，共7种政策组合：native ×1，MSE/RARE各 `both,depth_only,rank_only`。正式AP拒绝smoke结果，只接受每split780帧全量dump、`[30,26,50,6]`、全部finite，必须哈希一致。

可以只重跑指定部分：
`SPLITS='test_novel' METHODS='rare' bash scripts/rare_mix50/run.sh infer`；
对应eval单独运行，不覆盖已有 `infer` 结果。对外AP计算仍须涵盖全部预注册条件。

### 4.5 配对结果与分析

主比较：同一个50%基座的 **RARE/both − MSE/both**，按scene配对bootstrap。应另列三域完整六μ、`depth_only`、`rank_only`，并与Native参照比较；明确指出如果depth-only坏而both好，其收益可能来自跨query排序/NMS而非局部depth改进。

必须执行：
- S0基座真实来源与epoch全程保留；若历史e15，和10% e19比较仅是描述性数据规模对照，不是纯训练规模因果效果。
- 40验证图中native/MSE/RARE：selected Q、oracle Q、quality MSE、有效depth pairs与top2 oracle。
- 逐scene/逐摩擦阈值AP变化；旧query保留情况、depth transition、collision prefilter/NMS/top50存活变化。
- 同一冻结基座的MSE/RARE同初始化，均有固定20轮/相同真实标签；不使用测试AP选模。
- 如报告仍只有一个训练seed，保留其限制，不把scene-bootstrap区间说成训练seed重复。

按下表生成 `RESULTS_CN_RARE_MIX50.md`，未知值保持 `TBD`，不得使用10%结果替代：

|50%+50%方法/策略|Seen|Similar|Novel|Mean|RARE−MSE CI|
|---|---:|---:|---:|---:|---|
|Native GVAR|TBD|TBD|TBD|TBD|—|
|MSE/both|TBD|TBD|TBD|TBD|—|
|RARE/both|TBD|TBD|TBD|TBD|TBD|
|MSE/depth-only|TBD|TBD|TBD|TBD|—|
|RARE/depth-only|TBD|TBD|TBD|TBD|TBD|
|MSE/rank-only|TBD|TBD|TBD|TBD|—|
|RARE/rank-only|TBD|TBD|TBD|TBD|TBD|

## 5. 停止条件与验收标准

1. **S0失败：** 缺少真实训练协议、模型设计严格不一致、数据scene/frame pair错位、出现Observed/GT depth作network geometry，停止；不得修改meta绕过。
2. **S1失败：** Native full-forward logits/17D parity>1e-6、width/center/angle/4depth不一致、CAD质量或物理身份有非有限/哈希冲突，停止。
3. **S2失败：** 任一组缺失训练/验证帧、cache mix10/mix50混用、loss出现NaN、数据对比预算不同，停止。
4. **S3失败：** 缺少任意应有frame、错误使用10%训练checkpoint、官方AP数组异常或不匹配，停止；不得只报告有利split。
5. **科研停止：** RARE−MSE接近零或负数，不能为了获得正值临时调整pair权重、使用测试AP调epoch、删除Novel负例。优先分析pair构成/同族排序，并保留原始阴性结果。

验收材料：`acceptance.json`、`protocol.json`、`frames_mix50_test10.json`、基座checkpoint SHA与严格设计、train/val cache索引、S2两head SHA、正式21份AP arrays、最终 `results.csv`、per-scene／per-mu／depth转移／碰撞存活日志、场景bootstrap JSON及完整中文结论。

**当前状态（工作单建立时）**：GitHub脚本与参数合同已实现；未连接117执行真实CUDA、CAD评价、50%基座新训练、评分器训练或官方AP。因此没有50%+50%的新数值结果。
