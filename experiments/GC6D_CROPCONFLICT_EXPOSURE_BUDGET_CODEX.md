# GC6D：P0同物理点Crop梯度冲突与P1监督曝光预算（200/600-step）诊断

**日期：2026-11-17。必须继续原分支 `exp/quality-strata-conflict-supervision`，不得创建新远端分支、修改main、覆盖历史GC6D/A/B/GraspNet AP或已经验收的固定ROI/Latin-square Multi3 200-step last.pt。**

**当前状态：仅代码/协议已提交。** 117上真实CPU/CUDA/CAD数据新验收未执行，新数值不得出现在本次报告中。若本工作单与117实际源码补丁不一致，必须先追溯差异，不允许降低SHA或checkpoint门禁。

## 0. 来自用户本轮报告的实证依据

上传文件 `gc6d_crosscrop_REPORT_CN.md`，其验收时间2026-11-17T07:58:20Z；原HEAD `37e3078` + 117私有本地兼容补丁构成本轮实际执行 `a4a2f969...`；新GitHub远端已有此前对source/Blender修复的对应提交，**但不自动保证所有117本地差异已合并**。

当前CAD2/5训练、CAD8/9测试（对象均已经被研究开发观察），同MIX50 Point+GVAR36 e15基座，仅更新10,890,305参数DPT depth_head，200步×256有效像素=51,200监督像素、seed20261117、AdamW LR2e−5/WD1e−4、SmoothL1 beta0.03、gradient clip1。每组固定最后一步权重，不依据测试指标选模型。

关键数值（**只用于选诊断问题，不用于为新训练挑checkpoint/学习率**）：

| 0.85 m测试crop | ROI-wide 200步 MAE / β | Multi-wide 200步 MAE / β |
|---|---|---|
| Fixed ROI | 14.491 cm / 0.7160 | 22.332 cm / 0.4065 |
| Adaptive | 48.590 cm / 0.0308 | 20.948 cm / 0.4272 |
| Full | 6.414 cm / 0.1652 | 1.322 cm / 0.7319 |

同一原始世界物理点的三crop远距预测range：ROI-wide47.992cm、Multi-wide27.207cm；但是Multi-near range仅3.164cm、远距预测仍约46–48cm错误。**跨crop一致≠物理深度正确。**

在200步总监督预算下，Multi3每个crop仅66–67步曝光而ROI200固定ROI200步。ROI优势丢失可能来自：
1. 多crop曝光稀释（per-crop data/optimizer budget）；
2. 同一真实物理点在不同crop下造成梯度方向冲突；
3. 训练分布、局部形状或网络表示无法在统一权重中同时使用相互变化的视觉尺度；
4. 三者交互。现有两CAD/单seed无法唯一确定。

**下一轮只验证1和2，不先设计Camera-K embedding或新模型。**

## 1. P0：同物理点crop目标梯度冲突（只读、无训练）

代码：
- `diagnostics/depth_factorial/gradient_conflict_core.py`：从相同原始RGB/渲染传感器z/seg样本在三crop交集抽固定128个物理射线，梯度方向余弦/点积/按DPT decoder-block分组；
- `diagnostics/depth_factorial/gradient_conflict_p0.py`：在基座、历史ROI-wide LAST、历史Multi-wide LAST三个状态运行训练对象CAD2/5 × 原物理距离index0/2/4 × 三crop的SmoothL1，**只执行autograd.grad，不执行optimizer.step**。

参考z来源是同一个raw渲染depth pixel。采用原Blender物距与crop源，不允许按预测误差挑像素；每一case的相同128个世界射线交集通过crop坐标反投影到448RGB DPT输出。前景像素不足128则失败，不以其他crop特有像素补足。

计算：

\[
c_{ij} = \frac{g_i^\top g_j}{\|g_i\|\|g_j\|},
\qquad
g_i = \nabla_{\theta_{\mathrm{DPT}}}L_i
\]

并按group报告。`c<0` 只表示当前权重下从crop i的负梯度方向做足够小更新，会**一阶增加**crop j的损失。它不是某一输入成因的证明，也不保证训练600步时会冲突。须同时报告不同物距、对象、基座与已训练权重状态，不能挑个别负例。

GPU memory上只保存每case三个crop的DPT梯度，不保存DINO、GVAR、grasp的梯度；模型不更新，参数HASH在同一个状态前后必须一致。历史权重必须提供真实文件SHA。

## 2. P1：两种不同预算匹配口径的最小2×2训练

本阶段只补新两组600步 Wide（0.35/0.55/0.85m）；旧200步 Wide不重新训练：

| 训练crop | 200步、总51,200监督像素 | 600步、总153,600监督像素 |
|---|---|---|
| 只用Fixed ROI | **旧ROI200 LAST + Eval** | **新ROI600** |
| 3种crop Latin-square | **旧Multi200 LAST + Eval** | **新Multi600** |

四组初始完整MIX50 e15相同；训练CAD固定2/5、测试CAD旧8/9可用于探索性对照。600两组从基座**重新初始化**，不在200步头上继续训练，优化器也不延续；seed/AdamW LR/WD/梯度/冻结参数、训练对象和物距采样顺序与原200步协议一致。

600步正交日程：
- 距离0/2/4（0.35/0.55/0.85m）每种200步；
- ROI600只看ROI，共600步；
- Multi600在同样600步中，Full/ROI/Adaptive各**精确200步**，三crop×三距离组合各66或67步。
- 每步256个真实有效前景depth监督像素，总153,600像素。
- 绝对不能改变物距grid/seed/优化器/模型可训练层/epoch根据CAD测试调参。

将结果按三种不同contrast解释，避免混淆：

**匹配总计算量：**ROI200 vs Multi200；ROI600 vs Multi600。观察crop广度的损益。

**固定crop的训练预算：**ROI600−ROI200；Multi600−Multi200。观察追加优化是否帮助缓解原定性能缺口。

**匹配单ROI曝光：**ROI200 vs Multi600，都见过ROI200次，但Multi600多了400次其他crop且总算力3倍；这是机制辅助证据，**不是等计算量对照**。

所有指标必须在同一物理CAD测试对象×全部5距离×三测试crop上比较：dense MAE、AbsRel、同世界表面β、Δz误差和距离×crop DiD，不允许只按0.85m单一最低MAE做论文主结果。固定last.pt，不执行best-epoch选择。

## 3. 新CAD对象与外观来源（与旧的两枚探索性eval CAD分开）

代码：
- `distance_budget_holdout_register.py`：新source manifest实际文件SHA、mesh单位与CAD mesh SHA、不同semantic_group、显式材质/粗糙度、真实−6EV RGB饱和检查、Blender camera-K/raycast深度几何，拒绝旧注册训练CAD2/5和旧评估CAD8/9。
- `render_blender.py`与`factorial_contract.py`现在可在计划 `appearance.base_color_linear` 与 `appearance.roughness` 显式定义每个CAD的材质，**同一CAD的5物理距离必须固定这些属性**。不能换一个材质的距离0.85与另一个材质的0.35配对。
- `distance_budget_holdout_eval.py`：训练CAD registration仍是原始不可变文件，但新测试对象来自**独立注册**；四个训练头（以及可选native）共用同一新测试物体与全部5×3条件。
- `distance_budget_compare.py`：支持在原2已观察CAD上的exploratory比较以及新CAD上的验证；新数据要求四组eval绑定同一个签名来源。

新CAD评估需真实提供`NEW_CAD_SOURCES_INDEX`（JSON [{manifest, sha256, semantic_group, prior_development_use}]）及`NEW_CAD_LINEAGE_REVIEW`（实际CAD/语义重复/曾用历史审查文件）。**不生成不存在的CAD，也不凭两个旧对象宣称新的unseen-object验证。**

正式登记门槛：至少4枚新的CAD mesh、至少3个不同事先声明的语义组、至少2种实际材质，显式声明此前未用于项目模型选择/开发。程序仅能验证文件SHA和声明一致性，不能从技术上证明用户的历史完整；状态仍标记`PROVISIONAL_NEW_CAD...`。少量材料可显式`--exploratory`，但不得将其当作正式确认。

每CAD源须先通过Blender修正曝光（−6EV）、RGB质量、K投影和深度raycast校验，使用不同于历史的**真正新原始mesh文件**，不能把旧mesh换文件名或新材质重新算独立对象。

## 4. 在117执行

使用**同一远端分支新detached worktree**，先核对117实际执行source patch与GitHub实现的数值差异：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-depth-budget \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-depth-budget
git status --short
git rev-parse HEAD
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/gc6d_cropexposure_20261117
export P1_REGISTRATION=/ABS/accepted_original_p1b/p1_cad_split.json
export ROI_NEAR_EVAL=/ABS/accepted/p1_eval/near_only
export ROI_WIDE_EVAL=/ABS/accepted/p1_eval/near_mid_far
export MULTI_NEAR_EVAL=/ABS/accepted/crosscrop_eval/near_only
export MULTI_WIDE_EVAL=/ABS/accepted/crosscrop_eval/near_mid_far
export ROI_WIDE_LAST=/ABS/accepted/p1_heads/near_mid_far/last.pt
export MULTI_WIDE_LAST=/ABS/accepted/crosscrop_heads/near_mid_far/last.pt
export BOUND_CONFIG=/ABS/accepted_MIX50_GVAR_e15_bound.json
export GRASPNET_ROOT=/ABS/GraspNet
export DEVICE=cuda:0
```

注意上述`/ABS/`仅描述输入变量，**不允许按工作单示例猜117上真实路径**，必须按 `ACCEPTANCE.json`、`SOURCE_HASHES.json`、`EXECUTION.md`和当前服务器实际文件核查。

### P0/P1主流程

```bash
# CPU数学合同与代码语法，必须实际PASS
bash scripts/depth_factorial/budget_next_round.sh check

# 必须先验收旧四模型/源和实际200-step LAST
bash scripts/depth_factorial/budget_next_round.sh preflight

# P0：训练CAD2/5上的同世界射线梯度；不更新任何权重
bash scripts/depth_factorial/budget_next_round.sh gradient

# P1：只新增ROI600 / Multi600，固定模型、步数、监督像素
bash scripts/depth_factorial/budget_next_round.sh train-roi600
bash scripts/depth_factorial/budget_next_round.sh train-multi600

# 旧已观察CAD8/9上的相同评估（探索性）
bash scripts/depth_factorial/budget_next_round.sh eval-roi600-old
bash scripts/depth_factorial/budget_next_round.sh eval-multi600-old
bash scripts/depth_factorial/budget_next_round.sh compare-old
```

产物：
- `gradient_p0/{gradient_pair_by_case.csv,summary.json,REPORT_CN.md}`；
- `budget_preflight.json`，内含旧200 LAST和eval真实SHA；
- `heads600/{roi,multi3}/{last.pt,complete.json,loss_history.json,sample_schedule.json}`；
- `eval_old/{roi600,multi600}/`的15个测试条件、同世界表面分析；
- `budget_old_CAD_comparison/{summary.json,per_cell_budget_contrasts.csv,REPORT_CN.md}`。

### 新CAD受控评估（仅真实数据存在且已预注册时）

```bash
# 首先对新的CAD mesh按预注册5物距×相机固定协议运行现有
# scripts/depth_factorial/run.sh b-render (逐CAD独立新目录)，
# 确保源manifest有物理CAD/RGB/相机/Depth Pass真实SHA。

export NEW_CAD_SOURCES_INDEX=/ABS/new_cad_source_manifest_index.json
export NEW_CAD_LINEAGE_REVIEW=/ABS/cad_object_material_lineage_review.md
bash scripts/depth_factorial/budget_next_round.sh register-new

# 用原注册训练好的四组fixed LAST，不用新CAD训练
bash scripts/depth_factorial/budget_next_round.sh eval-new-roi200
bash scripts/depth_factorial/budget_next_round.sh eval-new-multi200
bash scripts/depth_factorial/budget_next_round.sh eval-new-roi600
bash scripts/depth_factorial/budget_next_round.sh eval-new-multi600
bash scripts/depth_factorial/budget_next_round.sh compare-new
```

没有新CAD真实render/source时，**停止在旧CAD exploratory 结果，不要声称新CAD已验收**。

## 5. 如何做结论，哪些仍然不能证明

P0：如多状态下 same-ray gradient cosine频繁<0，则支持“存在局部监督方向竞争”的机制，但不能仅由负cosine证明长期优化失效。若cosine多为正，仍无法排除容量/分布冲突，因为局部梯度与多步优化不同。

P1：
- Multi600若在固定ROI上恢复ROI200的优点，且adaptive改善延续，**支持曝光稀释是多crop200退化的重要因素**；但不能因为增加400steps把它称为“纯crop因果效应”。
- Multi600若在ROI与adaptive上仍存在显著trade-off，且有负gradient cosine，**支持训练目标干扰的可能性**，仍需多seed/不同物体核查。
- ROI600也必须评估；如果ROI600的近/远误差自己发生变化，则不能简单把Multi600改善全归因于crop变化。
- Multi600即使同点range趋近0，也须同时检查物理β≈1、绝对MAE与近距离性能；“共同预测错误”仍然是低一致性损失陷阱。
- 新登记CAD的同shape/material类别覆盖必须真实记录；材料变化必须固定在各物距序列内部；非穷尽CAD历史审查不构成身份绝对证明。

**不在本轮训练/修改GVAR、RayRoPE或任何6D grasp head；无新GC6D或GraspNet官方AP、无真实机器人成功率。**

## 6. 验收清单

- [ ] 与原远端分支HEAD一致；main、旧200step训练/CAD/GC6D AP结果不动。
- [ ] CPU unittest/compileall/bash -n真实PASS；数值复核gradient/distance/crop纯函数。
- [ ] 历史4模型接受的实际last.pt、test推断结果和train registration SHA一致；不能新训练旧头替代。
- [ ] P0训练对象2/5，不读取test对象8/9进行梯度/模型选择；同一128真实射线3crop抽样，参数从头到尾无更新。
- [ ] P1两组600step相同物距/对象抽样、监督像素/初始DPT SHA/AdamW；ROI600 600步，Multi600每crop200步，九格每格66–67步。
- [ ] 冻结DINO/pose-FiLM/GVAR/grasp在新训练前后SHA相同，且只有DPT head参数更新。
- [ ] 每个模型测试固定LAST在全部5距离×3crop×同CAD真实3D表面点，按对象聚合，不能用best-epoch/最远单点选择。
- [ ] 如有新CAD，mesh SHA、类别、材料、渲染源、相机K、RGB质量与来源历史都通过独立注册；排除原CAD2/5/8/9；不假装文件SHA等于类别独立。
- [ ] 全部结果划分exploratory旧CAD与provisional新CAD，报告计数和限制；无新抓取AP或机器人部署结论。
