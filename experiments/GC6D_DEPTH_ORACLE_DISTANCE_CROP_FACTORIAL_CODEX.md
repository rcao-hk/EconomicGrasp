# GC6D Follow-up：A Spatial Held-out Depth Oracle 与 B Physical Distance × Crop 因子实验

**日期：2026-11-17。继续同一远端分支 `exp/quality-strata-conflict-supervision`。** 不新建远端branch、不修改 `main`、不覆盖之前的78场景GC6D数据、实验报告或官方AP。

**执行状态：本工作单及代码写入GitHub；117真实CUDA/Blender运行与验收未执行前，不得把模拟结果或代码合同当成真实结果。**

## 0. 研究问题与上轮证据

来源：用户上传的 `gc6d_depth_followup_REPORT_CN.md` 与上轮同域支持/同表面跟踪：

- 原MIX50 Point+GVAR36 e15冻结DepthNet在Original train400帧与GNTrans400帧，参考/预测约`0.41/0.41m`，前景MAE约`9.65/8.20mm`；GC6D D43578scene×13=1014帧参考/预测`0.699/0.411m`，前景MAE约`0.291m`。
- 100% GC6D帧的reference z落在Original训练抽样第5–95百分位区间外；有效fx/fy与crop尺寸明显偏移。支持域的共同变化不能直接确定唯一因果来源。
- GC6D同一3D表面经过sensor/seg/已验证C2W门禁后，共990,192次配对，78scene的距离响应斜率`−0.022706`，说明不只是可见前景组成变化。
- 当前没有证据说明局部几何形状是否可读，也没有分离物距本身与native adaptive crop的交互。
- 不应在已经多次检查的GC6D78scene上学习1/0.60等全局倍率再称为泛化修复。

目标：
- **A** 用空间留出的oracle `scale / shift / affine` 对齐，判断冻结预测中剩余局部几何可恢复程度；
- **B** 在物理意义一致的固定CAD场景里，**只改变真实透视相机的沿光轴平移距离**（不是对一张RGB resize），并在每张相同渲染图上比较 `fixed_full`、`fixed_roi`、`adaptive_workspace` 3种预处理，得到距离×crop交互。

这轮依然**完全冻结**模型、不拟合全局部署尺度、不增加GraspAP/真实机器人成功率，不借视觉分割或depth作为depth网络输入；但native自适应workspace crop使用有特权的渲染深度／seg，需如实标注。

## 1. 文件清单

| 新增文件 | 功能 |
|---|---|
| `diagnostics/depth_factorial/oracle_core.py` | A空间四折，fit/test不重叠、训练侧8px间隔，scale/shift/affine和近常数affine不适定拒绝；独立测试区MAE/AbsRel/Gradient/Spearman |
| `oracle_run.py` | A读上轮经过验收的GC6D1014预测float32缓存与原始sensor depth/seg，按帧计算4fold×4model，并scene bootstrap；不产生部署校准器 |
| `factorial_contract.py` | B受控物理相机规划、CAD SHA/单位与透视K合同；拒绝非中心主点/不等fx-fy而静默近似 |
| `render_blender.py` | B内置Blender Cycles生成真实相机沿光轴移动的同CAD RGB/深度/instance mask；固定mesh位姿、材质、光照、镜头K与相机朝向；深度render pass与物理raycast比对后转相机z(m) |
| `factorial_run.py` | B严格加载历史冻结global_film头，在每个距离的同一RGB源上以full-frame、fixed ROI、native workspace depth adaptive crop三模式做RGB-only depth预测，保存真实源图与输出SHA |
| `factorial_analyze.py` | B通过真实渲染depth+mask+C2W跟踪同一世界表面点，要求三crop共同匹配，报告距离响应β与crop相对full的同点MAE与DiD |
| `factorial_aggregate.py` | 多独立CAD mesh结果按对象bootstrap，而非把像素/重复深度/帧当IID独立对象 |
| `experiments/physical_distance_crop_plan_smoke.json` | 可渲染的固定box/五物距纯合成smoke计划；**不是跨未知物体泛化结果** |
| `scripts/depth_factorial/run.sh` | check、a-smoke、a-run、b-render、b-infer、b-analyze、b-aggregate |
| `tests/test_depth_factorial.py` | CPU合成数学合同：空间留出、常数深度affine退化、物理C2W位移、固定K、相同世界表面3crop对照 |

## 2. 共同checkpoint /现有GC6D缓存

- 冻结模型为`MIX50 Point+GVAR36 e15`，完整checkpoint SHA256：
  `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。
- 使用 `rare_grasp.runtime.Runtime` 已验收的 `bound.json`，内部要求`global_film`，`geometry_depth_source=pred`，`use_obs_depth=False`，`epoch=16`；本轮只有`model.depth_net`参与前向、参数全部冻结。
- 上轮P1的GC6D缓存：`gc6d_depth_followup_20261117/p1_cache/manifest.json`（以117真实路径为准），包含每帧`prediction_file` float32[448,448]、`source_paths.depth/seg`、原始`raw_K`、`K`、`crop_box`、`depth_factor_divisor`，以及来源SHA。
- A直接复用这个1014帧缓存，**不需重复CUDA前向**。没有rendered GT的1014帧仍可用原始sensor测深。
- B用全新物理控制数据，不在这78个GC6D场景上手调crop/距离倍率后报告“提升”。

## 3. A：空间不泄漏的Oracle上限

### 对齐方式

对每张448预测和同crop同射线传感器参考`r`，将有效前景mask划分为四个固定象限。每折拟合仅用其他3个象限（拟合侧距离留出象限边缘8px的保护带），评价只用本折留出象限。

对齐函数与限制：

| 方式 | 数学式 | 解释 |
|---|---|---|
| identity | `z'=z` | 原始冻结RGB深度 |
| scale | `z'=az`，`a≥0` | 仅乘法尺度的privileged oracle |
| shift | `z'=z+b` | 仅平移米制距离的privileged oracle |
| affine | `z'=az+b`，`a∈R` | 有条件的统计上限，**负a属于非物理映射** |

- Scale/Shift/Affine的`a/b`全部由**该帧另外三个象限的真实sensor深度**拟合，永不用于部署或跨数据集推断。
- `affine` 若训练区域预测深度标准差`<0.003m`，禁止用近常数值计算不稳定斜率，标为 invalid；不能偷偷回退到shift/identity然后冒充affine成功。
- 每fold最少512拟合有效像素、128留出有效像素；不足时该fold明确missing。
- 留出区同时计算 MAE、RMSE、AbsRel、`gradient MAE`（相邻前景像素的局部深度差）、同区域Spearman。**只比较相同fold的原始与aligned**，跨frame汇总先到scene再bootstrap。
- 一帧四个fold并非四个独立场景，不用16000 fold数量作为置信区间有效样本量。
- 不能单独把affine MSE低解释成可部署K修复。正affine不会改变相对深度排序；负affine会倒转排序，但非物理。

### A执行

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-depth-factorial \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-depth-factorial
git status --short
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/depth_oracle_factorial_20261117
export GC6D_CACHE_MANIFEST=/ABS/gc6d_depth_followup_20261117/p1_cache/manifest.json

bash scripts/depth_factorial/run.sh check
bash scripts/depth_factorial/run.sh a-smoke
bash scripts/depth_factorial/run.sh a-run
```

A输出：`a_oracle/per_fold.csv`、`summary.json`、`REPORT_CN.md`。对比paired ΔMAE，并记录affine斜率符号/失败数量。

## 4. B：真正的物理Distance×Crop控制

### B实验来源：Blender CAD摄像机平移（不复用静态RGB数码缩放）

示例计划`experiments/physical_distance_crop_plan_smoke.json`：

- 示例为明确声明的procedural box（非未知物体语义验证），固定世界位置、材质、面积光、背景板；
- 固定透视相机分辨率640×480、fx=fy=550、主点居中、无skew；
- 物体保持不动，相机C2W旋转不动，只令光心沿世界中同一光学轴移动至预注册距离`[0.35,0.45,0.55,0.70,0.85]m`；
- 生成每距离真实RGB PNG、对齐的相机光轴z(m)数组、同源object mask PNG以及原相机C2W/K、物体/mesh SHA。
- Blender compositor Depth/Z pass需要通过对样本点执行Blender真实raycast核查，明确区分相机z与euclidean ray length。偏差/投影>容差应直接拒绝生成正式manifest，而非按数字resize补一个所谓距离响应。
- 计划支持`object.kind=mesh`引用真实OBJ/PLY/STL/GLB文件，需明确文件SHA及 `metres_per_mesh_unit`。选CAD时核查是否与模型训练object重叠；没有核查不能写“unseen object”。

**当前生成器对于principal point严格居中、fx=fy、无skew才支持；其他相机K故意拒绝而不是偷偷改变镜头模型。** 这保证先能做干净的Distance×Crop交互，不把未经校验的非对称K Blender投影混进去。

要换成真实物体CAD：复制示例json到`/ABS/controlled_cad_plan.json`，只把 `object` 改为例如：
```json
{
  "kind": "mesh",
  "path": "/ABS/CAD/model.obj",
  "sha256": "ACTUAL_FILE_SHA256",
  "metres_per_mesh_unit": 0.001
}
```
并确认对象mesh世界轴、单位、对象位姿、真实尺寸与镜头视场；**不能以不存在的路径或假SHA启动正式实验**。可对多个不同CAD计划分别运行，最后按object bootstrap。

### 三种crop策略

对同一份渲染像素/同一真实物距：

1. `fixed_full`：永远使用整幅640×480原图resize至448；
2. `fixed_roi`：在最近距离用真实对象mask确定一次ROI（预先固定margin20%），**之后所有距离沿用完全相同的原图ROI坐标**；
3. `adaptive_workspace`：使用真实渲染depth、object mask与C2W，沿用GC6D `build_mask(workspace_depth,json,0.02)`，按每个距离重新计算workspace crop。

所有策略都做原`ToTensor→Resize448→ImageNet Normalize`，有对应的`K'=A_cropK`。网络只接收crop RGB+pose元数据，渲染depth/seg**只作为特权crop计算与离线真值**。因此不是完全无深度特权的端到端RGB-only pipeline。

### B对照、指标及报告

使用全部固定物距与三种crop的**同一个冻结模型**，不对任何条件改变checkpoint。
- 密集前景MAE、AbsRel、输入crop尺度和K'；
- 物理真实距离变化`Δz_ref`与冻结预测`Δz_pred`，使用同一真实世界点在多个相机位置的投影；
- 通过真实depth、实例seg、C2W、z-buffer与3D回投门禁后，**三种crop的matched pixel取交集**，不能因为一种crop看不到某点就只给另一种crop计分；
- 逐距离和逐crop报告过原点距离响应斜率 `β`（理想约1）及`Δz MAE`；
- 在完全相同源表面点上计算crop相对full的错误差`gap(z)`，报告物距交互差分：
  \[
  \mathrm{DiD}(z)
    =[\mathrm{MAE}_{\text{crop},z}-\mathrm{MAE}_{\text{full},z}]
     -[\mathrm{MAE}_{\text{crop},z_0}-\mathrm{MAE}_{\text{full},z_0}].
  \]
  单对象仅描述结果，不应在每像素上bootstrap当成独立CAD；多mesh通过对象级bootstrap。
- 无官方GC6D/graspnet AP，这个实验目标只检验深度距离响应和crop×distance交互。

### B运行

需要Blender包含Cycles及深度pass，当前服务器上是否安装Blender必须现场确认：

```bash
# 先用程序化box做渲染与相机/深度mask全部验收
export RENDER_PLAN="$PWD/experiments/physical_distance_crop_plan_smoke.json"
export BOUND_CONFIG=/ABS/accepted_MIX50_GVAR_e15_bound.json
export GRASPNET_ROOT=/ABS/GraspNet

bash scripts/depth_factorial/run.sh b-render
bash scripts/depth_factorial/run.sh b-infer
bash scripts/depth_factorial/run.sh b-analyze

# 使用真实、经签名检查的CAD；必须分开全新输出根和plan
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/physical_distance_cad_01
export RENDER_PLAN=/ABS/controlled_cad_plan.json
export REQUIRE_REAL_CAD=1
bash scripts/depth_factorial/run.sh b-render
bash scripts/depth_factorial/run.sh b-infer
bash scripts/depth_factorial/run.sh b-analyze
```

每个renderer计划产生独立`b_render/manifest.json`，预测`b_frozen_preds/manifest.json`和分析`b_analysis/summary.json`。需要多个独立对象时，建立`B_ANALYSES_INDEX`为结果目录JSON列表，并执行：

```bash
export B_ANALYSES_INDEX=/ABS/different_mesh_analysis_directories.json
export RUN_ROOT=/ABS/new_object_aggregate_output
bash scripts/depth_factorial/run.sh b-aggregate
```

**重要：** 这个Blender synthetic domain与真实GC6D差异较大，单独看到β变化不能证明真实GC6D的crop策略是因果根因；要结合A的局部几何可恢复性与真实观测数据再判断是否值得训练camera-aware模块。

## 5. Stop conditions /验收

- [ ] 仍为原分支HEAD，main不变；原GC6D78与Novel/AP全部保留，不覆盖。
- [ ] CPU unittest、Python compileall、bash -n由117**实际PASS**；Blender脚本本地无bpy时不能假冒测试。
- [ ] A用同一P1缓存、原始sensor/depth/seg SHA，严格四折空间train/test不重叠；不适定affine记录invalid。
- [ ] A诊断oracle参数不会应用于任何正式推断、新候选或AP；仅空间外区域评估。
- [ ] B Blender对透视K进行World→pixel数值核对（默认≤0.25px），renderer Depth Pass vs真实raycast有mm级验证，整个物距序列mesh/灯光/相机K和旋转恒定。
- [ ] B模型冻结，所有输入为同一render经过不同crop；相机真正沿光轴平移，不用单图zoom代替。
- [ ] B三个crop保留同一世界表面点的可见性/掩码交集及源目标深度，不能各挑最有利的不同物体像素。
- [ ] B数据生成/评测不得使用训练标签、GC6D已观察AP或后验选取有利渲染距离/样本；选择真实CAD要核查训练对象重叠。
- [ ] 结果需区分：A的oracle形状上限、B的物理物距响应、crop交互与真实GC6D上的最终grasp AP。后三者不可互相替代。

**本工作单当前状态：实现已写入GitHub；实际的CPU/Blender/CUDA测试及新结果仍需在可访问相应软件和数据的执行环境验收。**
