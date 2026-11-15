# 共享、可查询、可验证的可见几何场 → 一步实际抓取修正

日期：2026-11-15。仓库 `rcao-hk/EconomicGrasp`。
新分支 `exp/shared-visible-geometry-refinement`，直接继承
`exp/query-scene-evidence-access@7c41b7551a2209c228b1e2669cf524942c0f9361`。
**不是新建 graph，不改生产模型，不改官方 evaluator，不启动远程训练。**

## 0. 研究合同：本轮到底改变什么

现有 Stage1 / Query–Scene 冻结实际动作，只更新 scalar score。报告确认后置交互没有独立收益、末轮存在共同过拟合，而 bounded Local+pair仅获得验证收益。本轮不以增加attention为主线，也不把有界残差当核心贡献。

本轮可证伪假设：
> 同一图像形成一个action-independent的空间函数；不同抓取只查询这个函数。对局部几何施加可核验监督，是否能比相同结构的quality-only学习更准确判断实际动作？实际动作移动后，访问新位置是否比沿用原位置的证据更好？

实现是 **Visible-Surface Query Field（VSQF，实验代号，不作首创声明）**，不是完整场景重建，也不是接触动力学求解器。第一版只做单步有限候选搜索，不宣称已完成连续SE(3)优化、闭环机器人控制或NMS-free抓取。

## 1. 表征定义：共享场景函数，不允许动作身份改变场景

冻结RGB骨干和原DPT/GVAR/MSE候选源。由原reader输入捕获完整`pregeom`特征图、RGB预测深度和crop内参K。新训练的空间adapter使用1×1/3×3卷积。对任意相机坐标点x，投影到图像取特征，结合全局平均特征、连续相机坐标、UV和预测深度提示，得到：

\[
\mathcal F_I(x)=\{\hat d_{vis}(x),\hat r_z(x),\hat p(s\mid x,I)\}.
\]

**其输入不含抓取ID、候选数量、初始评分、query索引或其他动作的特征。** 同一I、同一x，不论由哪个动作发起查询，都必须得到相同结果（数值容差内）。

### 1.1 两个距离及四种状态

令干净可见depth的点集为V，截断距离τ=0.03m，表面带ε=0.002m：

1. `udf = min(min_{v in V} ||x-v|| / τ, 1)`：到**已渲染可见表面采样点**的无符号距离。
2. `projective = clip((D_visible(pi_K(x))-x_z)/τ, -1, 1)`：沿相机z的投影残差。
3. 状态：`free / surface_band / behind_visible_surface / unobserved`。

**负projective不表示物体内部。behind_visible_surface不是occupied；深度缺失不是free。** UDF不是到完整CAD表面的距离，不推断被遮挡背面。状态概率没有被证明校准，不能当安全保证。渲染表面分辨率和像素深度不连续会影响目标；必须保留有效性mask和原深度来源。

图外或相机后方：状态unobserved；不监督UDF或projective。图内、源depth无效：projective不监督，状态unobserved；UDF仍可描述与已存在可见采样点的距离，但不声称它是完整物理表面距离。

### 1.2 严格几何瓶颈

每个实际动作g=(t,R,w,h,d)生成5×5×3=75个测量位置：局部x沿approach为[-0.03,d]，局部y跨实际开口宽度和两侧0.01m手指，z跨实际height。坐标约定对照GraspNet官方：
`https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/utils/eval_utils.py`。

这是覆盖夹爪包络的数值测量网格，不是手工collision classifier，也不保证75点足以证明连续空间无碰撞。网格与机器人夹爪尺寸属于明确的物理/计算假设；本轮不更改它来追逐AP。

将这些点变换到相机坐标并查询**同一个F_I**。quality reader只能读取：
- 2个距离值；
- 4个状态概率；
- 对应测量点在当前动作坐标系中的连续坐标；
- 当前w/h/d。

quality reader不能绕过这些通道读取hidden RGB feature、global latent、原CDF/MSE score或camera translation token。它是PointNet式逐点MLP + mean/max pooling + scalar quality head。原图像adapter和field decoder可学习，骨干与源候选生成器冻结。

H1的这些6个通道未接受几何监督，只能称**无约束的同形瓶颈**，不能称已经具有物理意义。H2必须用独立几何误差证明这些含义。

## 2. 最小比较矩阵

|mode|来源与学习|用途|
|---|---|---|
|`local`|父动作冻结feature + 新动作参数；`z=z0+.2*tanh(delta)`|安全的旧证据参考，不声称参数匹配|
|`predicted`|直接从RGB预测depth计算同定义几何，训练相同quality reader|原hard predicted depth可提供多少判断力|
|`h1`|可训练共享field + reader，只用exact-Q监督|**同结构、同初始化、同预算**的核心控制|
|`h2`|与h1完全同构，加几何监督|核心表示假设|
|`h2_stale`|同样几何监督，但对动作替代项仍只查询原动作的75个位置；将这些点转到新动作坐标|**独立训练的旧证据控制**，检验新位置取证是否必要|

`h2`另有同权重stale推理干预，仅作机制补充，不能代替独立训练的`h2_stale`。stale不伪造坐标：旧点在相机中不动，它们相对于新动作的位置真实变化，所以不是所有替代项必然同分的无效控制。

**最关键比较是h2−h1，以及h2−h2_stale。** local/analytic的参数量不同，单独记录，不把它们当严格容量控制。

不在这轮同时引入scene graph、leader规则、时序记忆、随机diffusion或新的NMS。

## 3. 动作更新合同

第一版是一个有预算的一步search：
- 每个源动作包含native index0，以及沿相机**单位射线**平移±5/±10/±20mm，共7项。
- 平移量是Euclidean米，不是camera-z偏差，也不是insertion bin。
- R、width、height和insertion保持不变；实际执行中心改变。
- 网格包含原动作，不强迫移动；score并列优先保留native。
- 每个变体用官方CAD/Dex-Net exact evaluator重新评价，禁止复用旧动作标签。

此有限邻域是首轮受控入口，不是最终连续动作表征。它不能证明完整6DoF修正能力。训练与正式推断使用同样7项，未来方向/幅度泛化须另行预注册，本版不声称已经实现。

## 4. 几何教师必须显式提供

复制 `configs/shared_geometry/visible_target.example.json`，填写：
- clean rendered visible depth路径模板；
- 同相机raw intrinsics路径（npy/npz/mat）；
- raw depth到metre的转换；
- renderer、相机extrinsics与native camera frame一致性的说明。

**不猜测GN-Trans虚拟depth目录，不自动用sensor或DPT depth代替clean target。** 示例的REPLACE说明未填写会拒绝；文件不存在、负值/NaN、尺度异常均停止。真正的文件路径必须在117核对。

映射用 `K_raw @ inv(K_crop)`，最近邻读取depth，不用简单resize，也不在遮挡边界混合深度。此公式只适用于同一相机中心/坐标系、crop/resize已编码到K的情况；有外参变化必须由renderer先完成正确配准，不能靠此脚本猜。

每帧缓存保存raw depth/K SHA、目标配置SHA及数值几何身份。paired诊断只有在Original/GN-Trans同scene/frame的raw visible depth和K数值身份一致时才允许复用物理Q；它只适用于光学变化、固定摩擦和固定真实几何。

GT只在export/label和离线指标中使用。`tensor_scene`白名单只有feature_map、pred_depth、K；test infer函数不构造TargetLoader，不调用label/evaluator。

## 5. 数据与预算

沿用当前新增模块阶段的scene划分，**历史基座见过验证场景，不是全系统未见数据**。

|分区|场景/帧|数量|每帧动作族|实际动作数|
|---|---|---:|---:|---:|
|train|0–79，frames0,16,…,240，两个RGB域|2560|16|286,720|
|validation|80–99，frames8,72,136,200，两个RGB域|160|32|35,840|
|official test|100–189，全部256帧，Original|23,040|全部1024|不预先标注|

动作族一半从全部query均匀抽样，一半从原模型top128中无放回抽样；先去重。选择不读GT、不使用sensor-mask特征；所有mode完全同一bank。它不是此前uniform32的纯架构比较；新的h1/h2内部才是受控对照。

每帧4096几何query：约一半来自动作测量点，其他来自表面扰动、自由/后表面射线。目标位置允许用于训练采样；推断位置由预测动作产生，不能读取GT surface采样点。

训练默认20轮、AdamW lr/wd=1e-4、clip1、4帧累积、seed0；field hidden64，quality loss=10MSE+0.5同族非并列pair，geometry loss=UDF L1+projective L1+state CE，权重1。H1/H2相同更新数与初始化；几何额外监督及其标注成本明确记录。

validation最大scene/domain-balanced selected-Q gain选best，ties保留早期权重。`selection.json`的admitted只有gain>0才为true；没有正收益不建议部署。**infer仍输出所有raw学习策略用于诊断，不自动隐藏负结果或把未admitted模型包装成安全策略。** 此validation已反复用于开发，选模结果不是确认性统计证据。

局部family gain/regret/repair-harm不等于official AP；首先判断物理表示是否学会，再考虑全量AP。不要凭小型family oracle宣布全场景抓取上界。

## 6. 服务器执行

### 6.1 独立worktree；不影响已有graph/RARE实验

```bash
git fetch origin exp/shared-visible-geometry-refinement
git worktree add /home/robotarm/EconomicGrasp-shared-geometry \
  -b codex/shared-geometry-exec origin/exp/shared-visible-geometry-refinement
cd /home/robotarm/EconomicGrasp-shared-geometry
bash scripts/shared_geometry/run.sh check
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_run01
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export BASE_CHECKPOINT=/ABS/verified_mix50_gvar.pt
export MSE_CHECKPOINT=/ABS/verified_mse/last.pt
export TARGET_CONFIG=/ABS/verified_clean_visible_config.json
export DEVICE=cuda:0
bash scripts/shared_geometry/run.sh prepare
bash scripts/shared_geometry/run.sh bind
bash scripts/shared_geometry/run.sh smoke
```

复用历史strict adapter可设置`PROVIDER=module:HistoricalRuntime`和`SOURCE_CONFIG`。接口仍为snapshot/label/sensor_keep，须暴露model与provenance，并验证原decoder/reader parity≤1e-6。不能通过strict=False、造metadata或放宽阈值接入。继承的默认native loader只支持main-compatible来源；117历史适配不是已远程完成的事项。

Smoke对两个RGB域各1帧2族，检查真实输出、7项physical ID、teacher投影、单位及所有finite值。必须可视化/人工核对至少一帧clean depth投影与RGB物体边界一致；脚本数值检查不等价于证明外参正确。

### 6.2 导出标注、建索引

```bash
# 先检查磁盘、GPU和CPU；单进程开始测实际成本。
bash scripts/shared_geometry/run.sh export
bash scripts/shared_geometry/run.sh index
# 可按scene分片，保持原seed/协议：
# SHARDS=3 SHARD=0 DEVICE=cuda:0 bash scripts/shared_geometry/run.sh export
# SHARDS=3 SHARD=1 DEVICE=cuda:1 bash scripts/shared_geometry/run.sh export
# SHARDS=3 SHARD=2 DEVICE=cuda:2 bash scripts/shared_geometry/run.sh export
```

每frame独立NPZ/签名sidecar；同协议resume核验cache和原target文件。孤立半成品停止。完整index要求所有train/validation帧，没有“忽略失败样本”。目标标签涉及322,560个实际动作及约1114万个空间query；后者是可见点云KDTree距离，不调用每点Dex-Net。文件大小/吞吐由smoke实测，不预填远程运行时间。

已有旧标签对应动作不相同，默认不复用。native变体也独立重评，便于校验；新目录绝不覆盖历史cache。

### 6.3 核心训练和诊断

```bash
MODES='h1 h2 h2_stale' bash scripts/shared_geometry/run.sh train
# 辅助基线可独立运行：
MODES='local predicted' bash scripts/shared_geometry/run.sh train
MODE=h2 bash scripts/shared_geometry/run.sh diagnose
MODE=h2 bash scripts/shared_geometry/run.sh paired
```

CLI支持epoch-boundary `--resume`，需要完全相同数据/源码/参数合同；脚本默认拒绝覆盖。先pilot seed0；成功后预先指定其他seed，不按test最优seed挑结果。

诊断输出：UDF毫米误差、近表面误差、projective毫米误差、观测状态准确率、同族selected Q/regret/repair/harm/排序精度；appearance_mean、appearance_shuffle、同权重stale干预；场景配对CI。H1的6通道无几何语义保证，仍可测误差，但不得把它们直接画成“正确几何”。

配对诊断固定Original物理动作，分别在Original与GN-Trans图像证据中评价，GT只作固定离线Q比较。不会把两个域各自生成的不同动作当同一个动作。

### 6.4 实际动作更新推断和官方测试

只有几何/动作验证有独立价值且来源通过验收后才启动昂贵全量测试：

```bash
export REFERENCE_ROOT=/ABS/native_MSE_full_pre_NMS_dumps
MODE=h2 bash scripts/shared_geometry/run.sh infer
MODE=h2 bash scripts/shared_geometry/run.sh eval
MODE=h2 bash scripts/shared_geometry/run.sh summarize
```

每帧完整1024源query，实例化7种动作（7168 hypotheses）。同一个scene只编码一次/评分调用，field查询分块。

输出五组：
1. `native`：原动作、原评分；必须和原dump逐帧parity。
2. `field_score_native`：原动作，只改score。
3. `move_keep_priority`：根据field选动作，保留原query priority。
4. `move_rescore`：更新动作与评分。
5. `stale_move_keep_priority`：同权重stale推理控制；独立训练h2_stale仍须另跑。

**动作一旦平移，sensor collision mask必须重新计算。** 只有native和field_score_native可用原mask。官方CAD评价、NMS、每物体top10/top50路径不改。保留原priority不代表最终NMS集合固定，也不代表原score是新动作校准概率。

源动作columns1:13必须相同，平移可变；每个策略保存新的physical IDs、选择index和独立sensor mask。Test不读取rendered teacher或新动作质量标签。正式评估逐split要求30×256帧、AP形状[30,256,50,6]，scene-paired/split-stratified bootstrap不替代训练seed。

## 7. 成功/停止门槛

**G0工程门槛：** strict源模型parity、相机单位/配准、finite值、测试不读GT、native/变更动作身份、changed-action重新过滤均通过。

**G1表征门槛：** H2对独立空间query的可见几何误差优于raw-predicted geometry，且same-x返回不依赖动作身份。仅质量拟合改善而几何无效，不能宣称共享物理表示成立。

**G2决策门槛：** 同族h2优于h1，同时报告自然错误修复与原好动作损坏；h2_stale判断是否新位置证据重要。没有差异不扩大网络。

**G3系统门槛：** move_keep_priority与score_native分开确认“改动作”价值；官方AP、单一物体可执行性与真机成功率不混用。只有离线物理标签/实际输出更新，没有执行任何机器人动作。

若只有clean teacher oracle有效、RGB学不到，不能假定加层能解决；再考虑多视图。若H2几何准确却质量差，可能缺少法线/接触关系或隐藏信息，不应直接归罪于scorer大小。本版没有二视图融合、完整CAD field、friction dynamics或连续optimizer。

## 8. 验收与输出

- source/plan/cache/teacher/模型SHA、实际参数量、训练历史、best/last/selection；
- 几何毫米误差与状态统计、same-action光学变化、同族选择质量及CI；
- 真实native参考parity；每策略过滤与动作变更记录；
- 三split完整官方AP及统计；未跑的项目写NOT_RUN，不预填正结果；
- `RESULTS_CN_SHARED_VISIBLE_GEOMETRY.md`必须区分：合成CPU测试、真实CUDA smoke、实际训练、official test；
- Local/predicted/h1/h2/h2_stale是实验条件而非已经成立的新贡献。

当前交付包括设计、源码和本地CPU合成/接口测试。没有访问117的执行通道，未运行真实渲染depth接入、CUDA、CAD重标注、正式训练或AP。用户给定目标是方法与脚本，不是承诺模型会提升。
