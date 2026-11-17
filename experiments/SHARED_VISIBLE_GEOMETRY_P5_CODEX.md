# P5-0 / P5-1：深度→动作几何传递与reader分布适配诊断

日期：2026-11-17。目标分支：`exp/shared-visible-geometry-multibatch`。P5以正式P4报告 `P4_PROTOCOL_RESULTS_CN_20261117.md` 为依据，使用已有P1几何、P2共享reader和P4深度修正选模结果，不修改任何历史权重、物理动作、CAD标签或官方AP流程。

## 0. 已验证事实与研究目标

P4报告：同一冻结P2 reader，Native Q×100=74.1048，Predicted=73.2227，Corrected=73.5189，Clean=75.8431；Corrected−Pred均值仅+0.2962 Q-pp，场景配对95%区间[−0.0911,+0.6348]。普通二维深度修正使全图验证MAE从12.5213降到7.3849 mm（约41%），但夹爪支撑点UDF MAE 2.7521→2.7081 mm，动作相对UDF 1.8814→1.8592 mm。P4-0等点数/空间匹配对照不支持将 opening-core 区域直接视为特殊物理监督区域。

两个独立问题：

- **P5-0（不训练）**：深度改善是否落在夹爪查询投影附近、可见深度边缘或其它区域？经过解析几何 UDF/KDTree 后在哪里丢失？原动作vs移动动作的相对几何变化与真Q选择如何对应？
- **P5-1（训练reader，不训练深度）**：若固定P4选定的普通Depth Corrector，仅用已训练过的真实物理动作和exact quality重新适配reader，能否比冻结P2 reader更好？这种改善是否只是重新训练任意readout，而不是Corrected特异性适配？

**所有方法仍是训练域scene0–79／开发验证scene80–99；没有外部Novel测试和官方AP。Clean-Visible只能做离线参照。** P4深度校正参数、50mm bounded residual、checkpoint选择均冻结；不得用P5抓取结果回头调整它。

## 1. 具体代码

| 文件 | 职责 |
|---|---|
| `shared_geometry/p5_geometry_diagnostics.py` | P5-0：固定P4几何/深度与P2 reader，pixel-projection bins、可见深度边缘分层、支撑点解析几何、KDTree最近像素对应、逐族相对误差与动作选择；signed JSON与行式记录 |
| `shared_geometry/p5_reader_adaptation.py` | P5-1：构建Corrected解析几何缓存；两套同初始化/预算readout训练；冻结P2参考、匹配重训练、交叉来源、差分中的差分和场景配对区间 |
| `scripts/shared_geometry/run_p5.sh` | check / p50-smoke / p50 / p51-cache / p51-train / p51-compare |
| `tests/test_shared_geometry_p5.py` | 纯CPU及合成源的真实P1/P2/P4→P5 end-to-end测试 |

### 输入身份及隔离

- 源目录 `RUN_ROOT`、`P0P1_ROOT`、`P2_ROOT`、`P4_ROOT` 只读；新的 `P5_ROOT` 必须是分离的、非嵌套目录。
- P5-0读取 `P4_ROOT/depth_evaluate/summary.json` 的每帧 `corrected_depth` 和 `corrected_physics`，核对SHA、同一 `physical_id`、P1帧指纹、原深度与P2冻结权重。正式160帧必须严格重现P4的Pred/Corrected/Clean gain与全图depth MAE；不重新调用P4训练。
- P5-1重新计算**train与val**的Corrected几何，保持P1的原Pred/Clean几何、75个support位置、动作尺寸、完整physical_id与quality_count；**仅新增Corrected物理tokens**。Corrector的forward只接收白名单 `feature_map`/`pred_depth`，不接收teacher/Q/grasp动作。动作几何解析在冻结corrector forward以后执行。
- P5-1建库使用**P4已完成且按深度MAE选定**的 `best.pt`，不允许偷偷选择任意epoch、扩大修正上界或用质量改动Depth Corrector。
- 只生成train/validation缓存，禁止正式test/官方AP路径、禁止重新DexNet标签、禁止按GT选动作。

## 2. P5-0 协议（不训练）

1. 对每张原始深度图，以clean-visible深度为离线目标，在共同有效像素上计算 `|Dpred−Dclean|`、`|Dcorr−Dclean|` 及 `improvement`。
2. 将全部7动作×75点按相机K投影到原深度图，将投影点扩展预定半径2像素作为局部区域；并用clean-visible一像素相邻深度差>20mm（或有效/无效边界）形成边缘区域。得到**互斥**的近支持×边缘、近支持×平滑、远支持×边缘、远支持×平滑四区；另报告全图／投影支持区。所有连续误差只在三来源共同有效像素上计算，不把缺失深度当成0误差。
3. 在**同一个夹爪相对75点模板**上比较Pred/Corrected与Clean的UDF、projective、四状态一致性、各区域及动作相对几何误差。保留缺失有效点null和共同可观测mask；这不等价于真实接触或力闭合区域。
4. 对同一3D查询点分别在Pred/Corrected可见点云KDTree中查找最近采样点所处的图像像素，记录最近像素变化率及像素坐标漂移。**这本身不表示正确/错误，仅表示非线性最近邻路径的敏感性。**
5. 在冻结P2 reader下重新计算各动作评分与selected gain；正式全160帧应与P4所有来源误差低于2e-6 Q、深度MAE低于0.001mm。逐族记录相对UDF/投影差分误差的变化与 `Qcorr(selected)-Qpred(selected)`。关联统计仅为描述性的（单seed、scene/domain聚类、不可把5120动作族视为独立）。

输出：`$P5_ROOT/p50_geometry_diagnostics/{summary.json,pixel_geometry_frames.jsonl.gz,region_geometry_frames.jsonl.gz,action_geometry_families.jsonl.gz}`。

## 3. P5-1 协议（匹配reader训练）

### 3.1 为什么必须同时重训Predicted

如果只对Corrected训练一个reader，而拿旧P2双来源reader当唯一对照，训练架构与输入分布同时改变，无法判断收益来自Corrected-specific适配，还是重训练任何reader都会改善。

因此增加两套**相同初始化、同一训练样本、同一exact-action count、同一目标与预算**的reader：

- `predicted_retrained`：只在P1原Pred解析几何上训练。
- `corrected_adapted`：只在P4深度Corrected解析几何上训练。

二者结构均为原 `PredictedGeometryControl.point_reader + quality`，不使用场景feature隐藏通道、native分数shortcut或privileged clean作为输入。`hidden=64`（与P1/P2同容量），AdamW lr/wd=1e−4，seed=0，真实batch=4、accum=1、20轮，每epoch640更新，共12800次。损失仍为 `10*MSE(sigmoid(z),Q)+0.5*同族非同分排序`。每臂按scene/domain-balanced验证 selected-Q gain挑选最佳epoch，同分取最早，无早停；**这是reader实验，允许Q选模型，但不能按Q再选Depth Corrector**。

### 3.2 冻结reader / 重训reader 矩阵

| Reader | Predicted geometry | Corrected geometry | Clean geometry |
|---|---|---|---|
| P2已冻结共享reader | 基线 | P4已验证弱改进 | 特权参考 |
| 新Pred reader | in-domain重训 | cross-source OOD探针 | 非必要 |
| 新Corrected reader | cross-source OOD探针 | in-domain适配 | 非必要 |

报告`predicted_retrained`、`corrected_adapted`、`predicted_p2_fixed`、`corrected_p2_fixed`、`clean_p2_fixed`，并且给出native-relative pair、rescue/harm、selected Q、20场景配对bootstrap区间。另输出同一Q bank的**Difference-in-Differences**：

\[
\mathrm{DiD}=
[Q_{corr,adapted}-Q_{pred,retrained}]
-[Q_{corr,P2}-Q_{pred,P2}].
\]

如果`corrected_adapted−corrected_fixed`很大，但DiD接近0，则可能仅仅是普通reader重训练带来收益，并不能支持Corrected-specific分布适配。Cross-source得分标注为OOD，不能用来宣称可部署的性能提升。

### 3.3 拒绝隐含泄漏

- Clean-derived几何仅离线报告，不进入新两套reader训练输入。
- 每次优化的图像数、更新次数与动作标签一致；不混用以前只有Q模型选模的P2共享reader训练预算来宣称完全相同条件。
- 验证集已在P0–P4多次分析；场景bootstrap只覆盖场景变化、不覆盖不同训练seed；后续部署需要未见过的场景和最终AP验证。

## 4. 117服务器脚本

```bash
cd /home/robotarm/EconomicGrasp-shared-geometry-batch
git status --short
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01
export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
export P2_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p2_run01
export P4_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p4_run01
export P5_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p5_run01
export DEVICE=cuda:0
export BATCH_SIZE=4
export VAL_BATCH_SIZE=4
export ACCUMULATE=1

bash scripts/shared_geometry/run_p5.sh check
bash scripts/shared_geometry/run_p5.sh p50-smoke
bash scripts/shared_geometry/run_p5.sh p50

# 建议先跑独立smoke输出，使用SMOKE=1，分四步运行：
SMOKE=1 LIMIT=4 EPOCHS=1 bash scripts/shared_geometry/run_p5.sh p51-cache
SMOKE=1 LIMIT=4 EPOCHS=1 bash scripts/shared_geometry/run_p5.sh p51-train
SMOKE=1 LIMIT=4 EPOCHS=1 bash scripts/shared_geometry/run_p5.sh p51-compare

# 正式完整缓存与20轮对照
bash scripts/shared_geometry/run_p5.sh p51-cache
bash scripts/shared_geometry/run_p5.sh p51-train
bash scripts/shared_geometry/run_p5.sh p51-compare
```

`RESUME=1`只用于被打断的同协议缓存／训练，不得复用已经完成但不同设置的目录。`p50`不要求新缓存和训练，可以先单独运行。P5输出目录应当不等于也不嵌套任何先前结果目录。

## 5. 验收与决定性结果解释

本地CPU合成源：包括P1双源缓存、P2共享reader、P4深度修正及冻结评价，再跑P5-0与P5-1。必须检查完整动作identity、边缘／平滑掩码、无效点不计0、动作相对几何误差、冻结P4权重签名、multi-B、resume、两臂预算、OOD标识、重复输出保护。

核心判定：

- 若P5-0发现全图改进主要来自远离support区域/平滑表面，P4负结果更像监督空间位置错位。
- 若支持邻域深度改善明显而UDF最近邻不动，检验解析几何构造（尤其可见点云KDTree投影位置、nearest-sample对应），而非贸然添加learned field。
- 若P5-1适配的Corrected reader相对匹配重训Pred reader**得到场景配对正差**且DiD成立，才有较强依据说Reader adaptation限制了P4。若二者都未超native，则P4当前深度源未提供足以稳定改善动作选择的证据。
- **无论如何不能仅凭P5证明隐式3D Field必要、不能根据现有验证集重新定义supervision区域或选择分数阈值。**
