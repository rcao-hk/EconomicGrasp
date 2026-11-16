# Shared Visible Geometry P0–P1：原动作相对决策与特权可见几何上界

日期：2026-11-16。Repo：`rcao-hk/EconomicGrasp`；**同一分支** `exp/shared-visible-geometry-multibatch`，以实际更新前的分支 HEAD 为唯一父提交。保留原有真实 B=4 / accumulated=1、H1/H2/H2-stale 与官方 evaluator。

## 0. 前置事实和不能改变的研究问题

原报告 `EXPERIMENT_PROTOCOL_RESULTS_CN_20261116.md`（2026-11-16）：Shared Visible Geometry 20-epoch 验证结果，Native selected Q×100=74.1048，Local=72.4805、Predicted=73.0957、H1=72.2982、H2=70.9408、H2-stale=73.7174，而不可部署的 oracle=84.6810。H2 within-family pair accuracy=82.2353%，但 move=70.5273%、rescue=10.8008%、harm=19.5898%；H2 的 UDF MAE=4.4353mm，Predicted 解析几何=2.6180mm。**本轮没有官方AP，不能将 Q×100 写为 AP。** H2 B1×acc4训练已完成但报告未比较完整轨迹；另行保留工程核查。

P0：现有方法在 **native 与其6个替代动作** 的取舍中具体犯了哪些错误？误判集中于高分原动作、某平移幅度、边界／未知几何、或 max-selection？

P1：当共享的几何摘要与75点reader保持不变时，**给准确的 Clean-Visible 解析几何能否优于 Predicted 解析几何并改善原始动作？** 如果这都无法恢复动作价值，就不能继续假定只把神经场预测得更准就有用。

P1是privileged geometry diagnostic，**不是部署可用的RGB-only模型，也不进行官方 GraspNet test**。

## 1. 不新建分支、不覆盖既有训练

代码文件：

- `shared_geometry/p0_relative_decision.py`：P0冻结既有 `best.pt`，验证缓存全量重算真实动作与估计分数，native-relative pair、false-improvement、argmax决策、移动幅度、source native校准、几何可观测状态、分scene与domain分解、scene配对bootstrap。
- `shared_geometry/p1_clean_visible.py`：P1从原NPZ和原签名sidecar读取同一动作bank及exact-action质量；按原 render-visible config和哈希加载teacher，批量一次性缓存 **Predicted / Clean-Visible** 两种解析几何的75点输入；两臂分别训练相同的 `PredictedGeometryControl.point_reader`＋`quality`，再做场景配对比较。
- `scripts/shared_geometry/run_p0_p1.sh`：独立的check、p0、p1-cache、p1-train、p1-compare入口。
- `tests/test_shared_geometry_p0_p1.py`：单位、组批、几何同构、签名、训练和比较的CPU测试。

新的输出路径与源 `RUN_ROOT` 严格不同。旧 source checkpoint、原缓存、测试脚本、CAD evaluator、模型和原正式训练结果只读。已有 P0/P1 输出文件拒绝覆盖；`p1-cache`与 `p1-train`支持显式RESUME；签名与SHA不同则停止。

## 2. P0：不训练，先定位高pair准确率为何不能做好native-relative决策

严格复用 `RUN_ROOT/validation_index.json`（scene80–99、两RGB域、共160帧，每帧32族×7个exact labels）。从每个方法真实 `best.pt` 推断，不使用test。P0允许真实多图B=4验证头，输入只从原模型白名单；标签仅用于后处理比较。

对一个动作族 \\(g_0,g_1,...,g_6\\)：

\\[
\Delta Q_k=Q(g_k)-Q(g_0),\qquad
\Delta\hat Q_k=\sigma(z_k)-\sigma(z_0).
\\]

主要输出：

1. **原动作相对 pair correctness** 与所有可比较 pair accuracy 分开统计。真实ties不当错误。对 `Q(g_k)<Q(g_0)` 但预测更高的动作标记 `wrong_improvement`；有真实改善机会却未预测为更好则为 `missed_improvement`。
2. **实际argmax** 的 move、rescue、harm、equal、selected gain 与 oracle gain、headroom recovery，不对Q相同的动作强迫移动或宣称提升。
3. Native vs shifted 的预测概率、真实质量、平方误差和有符号偏差；基线原始source native分数单独记录，不能把动作无关的base prior 当成移动动作的校准值。
4. 按 ±5/±10/±20mm（欧氏相机射线平移）、实际选中偏移、原Q count、`best alternative minus native` logit margin、以及预测深度下**75个支持点的unknown/behind-visible fraction**分解harm/rescue。
5. 按 scene/domain 等权聚合；同一个scene的两RGB域作为**配对**聚类bootstrap。固定模型/种子区间**不包含训练随机性**，不做 test 选模。

分析优先区别：绝对Q拟合失败、相对原动作比较失败、以及六个替代动作取最大导致的误差放大。**不要直接用平均pair accuracy支持或否定是否应移动。**

P0可先跑全部160帧。抽样报告只是 smoke；正式运行不打开 `--smoke`。

## 3. P1：完全匹配的解析几何来源对照

### 3.1 两臂唯一不同之处

| 训练模式 | 75点几何来源 | 训练模块 | 监督 | 是否部署RGB-only |
|---|---|---|---|---|
| `predicted` | 当前冻结DPT预测深度构建可见点云 KDTree＋projective residual＋四状态 | 与clean同构reader | exact-action Q | 可获得输入，需单独部署验收 |
| `clean_visible` | **渲染干净可见深度**构建同样的 KDTree＋projective residual＋四状态 | 与pred相同reader、相同随机初始化 | exact-action Q | **否，特权诊断** |

二者均由同一个已缓存bank的75点构造几何，包含 UDF / projective / four-state one-hot（6维）＋真实动作相对坐标（3维），`point_reader`＋mean/max pooling＋`quality`。不另加新的更大MLP、foundation输入、重新采样动作或CAD完整隐藏表面。

生成缓存时，直接调用原 `VisibleSurfaceTeacher.label()`。先使用当前 `predicted` 的同一个解析 reader 实现进行**前向 parity 测试**；两个arm形状 `[B,F,7,75,6]`，质量标签 `[B,F,7]`。两组学习率、weight decay、seed、20epoch、真实B=4、accum1、目标 `10*MSE+0.5*same-family pair`、梯度裁剪和验证模型选择完全一致。两臂的质量标签、bank、帧顺序、物理ID、训练总量全部相同，**不能只复用旧Predicted结果与新Clean进行不匹配比较**。

### 3.2 证明 Clean 是真实特权信息（不泄漏）

- 原训练与验证缓存每帧拥有独立 `NPZ` 和签名 sidecar，其中包含 clean渲染深度与相机K来源及 `SHA256`。
- `p1-cache` 从显式 `TARGET_CONFIG`（必须与 `contract.target_config` 完全相同）读取 clean depth，核对**原 frame SHA、render-visible config SHA、raw geometry identity、depth SHA和K SHA**；一旦文件变化停止。不得猜测 `virtual_scenes`路径或拿 sensor/pred 替代。
- 原 clean 仅在 **train/validation** 构建缓存与训练、验证时使用。新模块只有 P1 `build/train/compare` 子命令；**没有 clean_visible official inference/eval 命令**。不修改原 `shared_geometry.inference` 和现有官方 AP 流程。
- 源动作与新的物理动作仍依 `physical_id` 绑定旧 exact标签；P1无 GT model-input shortcut。新的NPZ只缓存两种可比较的物理几何输入、动作尺寸／相对位置和质量标签。label从不进入reader张量。

### 3.3 报告和停止条件

验证（scene80–99、两域）报告 native/selected/oracle Q、gain、regret、move、rescue/harm、pair accuracy、Q-MSE；按真实scene配对bootstrap比较 `clean_visible - predicted`。两个模型的 `best.pt` **只依据验证selected-Q gain选取**，没有正收益则`admitted=false`，不隐藏负结果。不同最佳epoch并不是相同权重；两臂训练预算相同并且有逐轮历史，可补充相同epoch对照。

**A. Clean仍不优于native：** 当前75点压缩表示／reader／可见表面可能不够，不能将全部失败归因于H2的几何预测；再区分几何输入能否成功表征接触法线等信息。

**B. Clean显著优于Pred且原动作：** 说明当前约束下准确visible geometry有可判别价值；下一步才研究保留解析Pred几何并学习互补误差。

**C. Pred和Clean都失败：** 不能声称RGB不可能，首先考虑6通道＋75点是否表达了需要的接触拓扑；隐藏接触信息也是待检验解释。

原报告的 `clean-rendered teacher` 是**可见表面**不是完整对象CAD占据，不是机器人传感器输入。这里的oracle依然是根据GT Q挑动作的**候选集上界**，不是RGB可学习上界。

## 4. 117服务器操作

**保持当前分支名不变：**

```bash
cd /home/robotarm/EconomicGrasp-shared-geometry-batch
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01
export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
export TARGET_CONFIG=/ABS/EXACT/original/rendered_visible_config.json
export DEVICE=cuda:0

bash scripts/shared_geometry/run_p0_p1.sh check
bash scripts/shared_geometry/run_p0_p1.sh p0
bash scripts/shared_geometry/run_p0_p1.sh p1-cache
bash scripts/shared_geometry/run_p0_p1.sh p1-train
bash scripts/shared_geometry/run_p0_p1.sh p1-compare
```

不要直接复制示例TARGET_CONFIG路径。**必须重用原导出运行的实际配置文件**；由原 sidecar `target_config`/provenance 找到，找不到则停止，不靠猜测或写假的路径。cache校验时读取target源一次，训练以后只读取新的pair geometry NPZ，不再每epoch构建KDTree或render。

缓存估算：`16*7*75*6*4*2 ≈ 403,200 bytes/train frame`，乘2560约0.96GiB；验证每帧32族约806KiB，160帧约0.12GiB，额外local坐标和metadata另计。正式内存/磁盘需求由首次smoke测量，实际保存是float32、NPZ可压缩差异和file metadata未计。

为了节省读大特征图的CPU成本，`p1-cache`读取历史原NPZ，但只导出两种紧凑解析几何和身份；P0依然需要从原NPZ读取RGB/动作特征。

### Smoke和续跑

在原 `RUN_ROOT` 上先验证真实teacher哈希、两域一个原frame解析几何；执行可对 `p1_analytic build/train/compare` 使用 `--smoke`，自定义新的smoke输出目录，避免占用正式输出路径。完成小规模校验再跑正式任务。

若构建中断：`RESUME=1 bash scripts/shared_geometry/run_p0_p1.sh p1-cache`；若训练在完整epoch边界中断：`RESUME=1 ... p1-train`。不要在已完成的不同协议目录里继续训练。

## 5. 验收与局限

- `p0_relative_decisions.json`：冻结模型SHA、验证来源、每族动作与估计、标签、calibration、margin/offset/visibility breakdown、各方法场景区间；**只有验证诊断，没有训练或正式AP**。
- `p1_cache/{train,validation}/index.json`：保持原框架bank/labels/physical IDs，两个几何source同文件绑定，rendered teacher provenance安全。
- `p1_models/{predicted,clean_visible}/...`：source、seed、完整20epoch历史、best/last、严格resume协议、独立两组权重。
- `p1_matched_comparison.json`：同验证动作、两臂选择质量与scene-paired CI、已有 H2 的描述性比较；privileged/source范围明确；**不是GraspNet官方AP**。可选读入现有已训练 `H2 best.pt` 作描述性比较；它的训练监督与两臂不同，不能当作第三条严格同预算解析几何arm。
- `tests/test_shared_geometry_p0_p1.py`：解析Pred parity；B4 vs逐图forward/grad；source篡改拒绝；同预算双arm训练/compare；P0分类与bootstrap。

尚未实际访问117，**本交付只实现脚本／本地CPU验收**，没有新的P0/P1真实数值或P1官方AP。不要根据之前已经探索过的Novel test集重选超参数。