# P3-0 — 选择翻转与动作相对几何证据诊断

日期：2026-11-16；仓库 `rcao-hk/EconomicGrasp`；**同一分支** `exp/shared-visible-geometry-multibatch`。在已完成的P1/P2上追加一个**完全只读**的诊断。P3-1（普通深度修正对照）和P3-2（action-relative geometry监督训练）留待本轮结论后再决定；本次不训练新模型、不启动官方test。

## 1. 研究假设与来源

P2同一共享reader权重下，Clean-Visible的验证Selected-Q×100为75.8431，Predicted为73.2227，差值+2.6204质量百分点。这是**固定权重、开发验证集、privileged visible geometry**上的诊断；不是GraspNet官方AP、不是RGB-only deployment、不能把scene bootstrap当作跨训练seed确认。

本轮要回答：这2.62 pp的收益由哪些**实际动作身份改变**形成，且这些改变是否与**动作相对的局部几何差分错误**相关？

只读前置：

```text
P0P1_ROOT/p1_cache/validation/index.json
P2_ROOT/shared_reader/best.pt
P2_ROOT/shared_fixed_weight_comparison.json
```

原 P1 signed validation cache具有 `quality_count[F,7]`、`physical_id[F,7]`、`family_id[F]`、`predicted_physics[F,7,75,6]`、`clean_visible_physics[F,7,75,6]`、`local_xyz[F,7,75,3]`、`bank_dims[F,7,3]`。必须为160帧、20个验证scene、Original与GN-Trans两域（测试可用 `--smoke`、`--limit` 明确小子集）。原动作index0，六个移动动作沿相机单位射线±5/±10/±20mm，保持其他执行参数。所有GT质量只用于离线后处理；网络评分始终只读几何、动作相对位置和尺寸。

新文件：

- `shared_geometry/p3_action_geometry_diagnostics.py`：同一P2共享reader按两个来源评分；选择翻转、动作相对几何差分、区域互换、场景配对bootstrap；输出签名JSON。
- `scripts/shared_geometry/run_p3.sh`：`check / smoke / diagnose`；输出到独立 `P3_ROOT`，严格拒绝覆盖。
- `tests/test_shared_geometry_p3.py`：数值定义、可见性、区域互斥、完全来源互换校验、读取已有P2权重的合成端到端路径。

## 2. P3-0a：同一动作族的源间选择翻转

固定共享reader的两套输入，只改变 geometry tokens：

```text
k_pred  = argmax_k score(shared_reader, predicted_physics[k])
k_clean = argmax_k score(shared_reader, clean_visible_physics[k])
```

对两者都计算 `Q(k)-Q(0)`，并输出：selected physical IDs、平移量、原动作与选中动作的离散Q、`Q(k_clean)-Q(k_pred)`、是否真实改善/损害/保持、是否存在GT改善机会。不同物理动作但Q相同不是成功。所有互斥翻转分为：same_action、harm_to_keep、keep_to_rescue、harm_to_rescue、harm_to_less_harmful_move、rescue_to_harm、keep_to_harm、rescue_to_better_rescue以及其他改变。另有`keep/rescue/harm/equal_quality_move`完整选择矩阵（不以类别名推断因果）。

按native Q=1与其他、有无GT改善机会、每个offset、真实pair对错、flip类别分层；所有source difference使用**scene-paired** bootstrap而不是把5120族视为IID。P3在正式完整160帧上检查复现P2同权重已发表的两个源gain（容差2e-6），不同直接停止。smoke因子集不同，只检查权重、索引和teacher身份，**不**比较完整均值。

## 3. P3-0b：动作相对几何误差

对动作 `g_k` 的75个**gripper-relative**模板索引 `j`，几何输入Phi包含可见UDF / τ、投影z残差 / τ、free / surface_band / behind_visible_surface / unobserved四状态。截断尺度τ来自旧P1签名（默认30mm）。

\[
D^s_{k,j}=\Phi^s(g_k,j)-\Phi^s(g_0,j),\quad
E_{\Delta\Phi,k,j}=\big|D^{pred}_{k,j}-D^{clean}_{k,j}\big|.
\]

**注意：模板索引j的夹爪相对坐标相同，但移动后的相机3D位置不同。** 这不是同一点的世界空间误差，而是不同执行动作引起的几何响应差分。距离差分乘τ换算为mm；状态通道只比较类别及其转移，不将不可见部分当作物体占据。

连续距离误差的保守mask：native与替代动作、Pred/Clean两路，四个对应状态均非unobserved；无有效点返回null，不记为零误差。另报告joint-known覆盖率、近Clean可见表面（UDF≤0.25τ）的连续误差、状态变化一致性。对于UDF近表面阈值，特权Clean只用于**离线分层**，不能作为部署时的输入。

固定、互不重叠的三个粗区域：

| 区域 | 5×5×3 模板规则 | 点数 |
|---|---|---:|
| palm_proxy | approach index 0 | 15 |
| lateral_finger_proxy | approach index 1..4，opening index 0或4 | 24 |
| opening_core_proxy | approach index 1..4，opening index 1..3 | 36 |

此处*proxy*只定义测量位置，不是GT接触、碰撞或力闭合区域。统计每区域差分MAE、native/alternative普通UDF MAE、near-surface MAE、状态变化、可观测覆盖。对6个替代动作先求族内均值，再按scene/domain聚合。额外报告原动作对替代动作的pair被Clean纠正/破坏时的差分误差、±5/10/20mm分组、预设[0,2)、[2,5)、[5,10)、≥10mm几何误差段。关联不构成因果判断，也不能对验证结果再定阈值。

## 4. P3-0c：固定reader的分区域几何输入替换

在固定**同一P2 reader**下，评估三组单区域混合输入：

- `Pred→Clean region`：只替换该区域全部6个物理通道，其余仍为Pred；重新对同一族7个动作评分。
- `Clean→Pred region`：反向替换。
- 全75点Pred→Clean必须逐logit复现完整Clean（误差≤1e-5），不通过立即停止。

输出每区域源间gain、move、clean/pred基线相比的gain差、scene-paired bootstrap CI；禁止把三块贡献相加，或仅选有利区域作为新推断策略。混合token可能在空间上不连续，属于分布偏移干预，**不**是独立因果效应或可部署输入。

## 5. 服务器执行

保持当前工作树和同一branch：

```bash
cd /home/robotarm/EconomicGrasp-shared-geometry-batch
git status --short
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_fp16_run01
export P0P1_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p0p1_run01
export P2_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p2_run01
export P3_ROOT=/data/robotarm/result/grasp/rgbgrasp/shared_visible_geometry_p3_run01
export DEVICE=cuda:0
export VAL_BATCH_SIZE=4
export BOOTSTRAP=5000

bash scripts/shared_geometry/run_p3.sh check
bash scripts/shared_geometry/run_p3.sh smoke
bash scripts/shared_geometry/run_p3.sh diagnose
```

前两者分别输出 `p3_smoke.json` 与正式 `p3_action_delta_diagnostics.json`，不能覆盖。若只需观察性结果可 `SKIP_REGION_SWAPS=1 bash scripts/shared_geometry/run_p3.sh diagnose`，但正式机制分析建议保留区域输入替换。**没有模型训练、CAD重标、原始rendered-depth加载、RGB推断或test访问**。P1/P2的模型和cache只读。先对CUDA/数据检查，不应假定本地CPU测试等于服务器真实验收。

## 6. 解释与停止条件

- 若Clean收益主要来自`harm_to_keep`，优先研究已有好动作的可靠保留；若主要来自`keep_to_rescue`或`harm_to_rescue`，优先研究新动作上空间差分的判别能力。
- 若action-relative误差集中在特定测量区域，先用源位置几何读出验证，再考虑P3-2监督；若整体普通UDF误差下降但差分误差不变，不能宣称动作相关几何已改善。
- 若三个区域分别替换都不产生可重现的收益，可能是跨区域组合、状态交互、候选选择噪声或reader压缩问题；不能推定该区域的几何没有用。
- 所有结论都是多次使用的单seed验证集探索性证据。只在独立测试和多seed后讨论generalization；不能把Q×100写为AP。

本次仅实现P3-0，不提前训练P3-1/P3-2。任何新的数据自适应区域划分、阈值学习或超参数选择都应作为独立实验预注册。