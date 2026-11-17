# GC6D P0/P1 下一轮：Same-Ray Crop Audit 与 2×2 距离×Crop 监督

**日期：2026-11-17。仍在原远端分支 `exp/quality-strata-conflict-supervision`。** 不新建远端分支、不改main、不覆盖已验收的GC6D/GraspNet/Novel AP或修正曝光的 B/CAD 结果。本工作单记录新实现和真实117验收要求；不是新的实验结果。

## 0. 为什么现在要做这两个诊断

依据用户上传的 `depth_followup_p0p1_REPORT_CN.md`。P0已完成78×13帧Constant Oracle对照：Constant Mean留出MAE5.3954cm，非负Affine5.4031cm，几乎相同；但非负Affine相对常数的中心化形状误差降低约1.195mm、edge梯度MAE降低0.371mm/px。故模型**有有限正向形状信息，但不足以成为可靠的精细几何重建**。不应重复训练或冒充新的P0形状改善。

P1a四个修正曝光CAD，远/近投影宽度比=0.397，fixed_full/ROI输入物体面积比=0.159，而adaptive=1.011；有效fx变化比例adaptive=2.529、ROI/full=1。说明native adaptive crop近似消除输入物体尺寸变化，但K′也随之变化。

P1b历史两组固定ROI的200步×256像素实验，仅更新DPT Depth Head：
- ROI近距离监督：0.85m下MAE42.014cm，同表面响应β=0.0975；
- ROI多距离监督：0.85m下MAE14.491cm，β=0.7160；
- 但同一个wide头在adaptive_workspace下0.85m MAE48.590cm，β=0.0308。
- 所有数值来自**已多次研究的两枚测试CAD 8/9**，且训练CAD 2/5也参与过项目开发。不能当作确认性或跨类别统计结果。

这提示：**距离监督在匹配训练crop时能工作，但对其他crop的响应迁移失败。** 当前还没有将物距监督与crop多样性独立交叉验证。下一轮必须把两种因素分离，而不是直接改变网络结构/K conditioning。

## 1. 目标及最小2×2设计

| 训练物距 | 只用fixed_roi | 同时使用三种crop |
|---|---|---|
| Near only（0.35m） | **历史ROI-near：直接复用已验收last.pt及eval** | 新训练Multi3-near |
| Near + Far（0.35/0.55/0.85m） | **历史ROI-wide：直接复用已验收last.pt及eval** | 新训练Multi3-wide |

**训练参数、模型和CAD全部不变。** 仅新跑两个Multi3深度头；在同一2个测试CAD的每一物距×三crop上统一比较：

- Fixed ROI下增加物距监督的增益：`wide_roi − near_roi`；
- Multi3下增加物距监督的增益：`wide_multi − near_multi`；
- 同样距离监督下增加crop支持的增益：`multi − roi`；
- 因子交互（先每CAD/同一测试cell配对，再按CAD等权）：
  \[
    I=(\mathrm{wide_{multi}}-\mathrm{near_{multi}})
        -(\mathrm{wide_{roi}}-\mathrm{near_{roi}}).
  \]

对于MAE，越负越有益；对于同表面β更接近1，不能仅看增减，需要结合绝对误差与近距离退化一起判断。不能只选最远距离MAE最好的一组。

原始数据和checkpoint只在117；**旧worktree的本地执行修复commit（报告记录 `842ecda7...`）尚未推送到GitHub远端。** 新worktree必须先核对当前远端HEAD与旧worktree实际执行源码哈希、127退出和Blender改动。不准以“同一文件名”推定配置兼容。若签名/运行合同不符，停下核查而不是绕过。

## 2. 新增源码

| 文件 | 用途 |
|---|---|
| `distance_crosscrop_core.py` | 预注册的训练物距×crop九格正交日程，四组训练合同验收（无CUDA） |
| `distance_crosscrop_preflight.py` | 在训练前确认历史ROI-near/wide真实CAD注册、源SHA、200×256预算、模型初始化与对象顺序，拒绝不一致的117旧结果 |
| `distance_crosscrop_train.py` | 新Multi3-near与Multi3-wide训练，仅DPT depth_head，所有其他参数冻结且训练前后SHA逐字一致；输出固定last.pt |
| `crop_consistency_core.py` | 同一物理原始CAD像素在三crop上同射线采样448 depth，计算跨crop分歧与同点MAE |
| `crop_consistency_audit.py` | 从真实已验收P1b eval数据中只读重算native/ROI/Multi所有物距×crop的一致性；无训练，无AP |
| `distance_crosscrop_compare.py` | 同CAD物理点和五距离×三测试crop下的四臂因子主效应、交互、对象级bootstrap |
| `scripts/depth_factorial/crosscrop_next_round.sh` | check→preflight→P0-before→两个Multi训练→两个eval→P0-after→2×2compare |
| `tests/test_distance_crosscrop.py` | 无CUDA 9-cell日程、固定预算/身份、同一raw-ray三crop一致性、拒绝遗漏/混淆的CPU测试 |

新增代码不应改动历史`distance_train_p1.py`的训练口径；仅在旧`distance_eval_p1.py`的测试参数中允许 `--train-crop-policy multi3`，调用原相同的depth/CAD世界表面比较，不单独重写测试评估器。

## 3. 确保训练真正解耦crop和距离

历史ROI两组：
- 相同MIX50 Point+GVAR36 e15 SHA=`2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`；
- 训练CAD 2/5（经实际物体SHA验证），测试CAD 8/9；
- seed`20261117`，200步×256有效前景像素；AdamW LR 2e−5、WD1e−4、SmoothL1 beta0.03、grad clip1.0；
- 所有层冻结，**仅10,890,305参数DPT depth_head可更新**，不训练pose-FiLM/DINO/GVAR/抓取头；
- 固定输出last.pt，不通过Novel、GC6D或合成test MAE选checkpoint。

新Multi3两组遵循**完全相同预算**，且：
\[
d_t=
\begin{cases}
0,&\text{Near-only}\\
(0,2,4)[t\bmod3],&\text{Near+far}
\end{cases}
\qquad
c_t=(\mathrm{full},\mathrm{ROI},\mathrm{adaptive})
[(\lfloor t/3\rfloor + t\bmod3)\bmod3].
\]

注意**不能**让物距与crop均按`t%3`轮换，否则只会看到(近,full)、(中,ROI)、(远,adaptive)三个组合，完全混淆距离和裁剪。采用上述9-step Latin-square正交日程后，wide的三距离×三crop九格约各22–23个step，200步总监督像素51200；near-only三crop曝光精确为67/67/66步。历史ROI仅固定crop，四组随机CAD对象index仍由`seed+701`生成同一序列；像素抽样由`seed+1701`随机流按每步同样个数执行。

**额外source约束：**所有训练/测试CAD渲染使用相同物理K、相机旋转、曝光、Cycles设置、深度正向与crop代码；旧几乎全白RGB不得复用；actual source mesh路径、版本、SHA及历史对象重叠审核都必须真实存在。

## 4. P0：在相同原始物理表面点上审计crop差异

不重新训练。现有`native`、`roi_near`、`roi_wide`已有的评估输出可直接开始：

- 每CAD×每物距从**同一张原渲染**的真实depth/object mask抽定量物理前景点；
- 将完全相同的raw pixel通过原crop box重参数化到448，查询三种crop的冻结预测；
- 对每个模型，三个crop必须保留**同一批原raw物理点**。如果adaptive裁掉一部分点，则其他crop也只在三者交集上评估；
- 比较三crop同点的预测深度标准差、range、与训练ROI预测的偏差，以及相同像素参考深度下的MAE/bias；
- 若后续新Multi3已训练，再重复同一套，不按对模型有利的物理像素池改变选择。
- P0的同视角同点crop差异不等于P1的真实相机运动distance response，不能混淆两种因果变量。

受控输入`EVAL_RUN_INDEX_JSON`格式：
```json
{
  "native": "/ABS/accepted_eval/native",
  "roi_near": "/ABS/accepted_eval/near_only",
  "roi_wide": "/ABS/accepted_eval/near_mid_far"
}
```

完成Multi3后增加`multi_near`与`multi_wide`根路径，由launcher自动生成并SHA绑定。报告同物理点MAE、跨crop不一致和相同物距不同训练策略的差异；本轮测试CAD仅2个，按对象等权，**不把像素当IID作CI**。

## 5. 117 执行（一个新结果目录）

先以当前源仓库创建同远端分支新worktree：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-depth-crosscrop \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-depth-crosscrop
git status --short
git rev-parse HEAD
find .. -name AGENTS.md -print
```

必须确认历史ROI eval根和预注册`p1_cad_split.json`来自已验收P1b，同一注册文件字节不可在本轮重新创建替代。用真实路径设置：

```bash
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/gc6d_crosscrop_20261117
export P1_REGISTRATION=/ABS/accepted_original_P1b/p1_cad_split.json
export NATIVE_EVAL=/ABS/accepted_original_P1b/p1_eval/native
export ROI_EVAL_NEAR=/ABS/accepted_original_P1b/p1_eval/near_only
export ROI_EVAL_WIDE=/ABS/accepted_original_P1b/p1_eval/near_mid_far
export BOUND_CONFIG=/ABS/verified_MIX50_GVAR_e15_bound.json
export GRASPNET_ROOT=/ABS/GraspNet
export DEVICE=cuda:0

# Step 0: CPU contract + actual old ROI input/source preflight
bash scripts/depth_factorial/crosscrop_next_round.sh check
bash scripts/depth_factorial/crosscrop_next_round.sh preflight

# Step 1: no-training baseline P0; valid on existing native/ROI eval outputs
bash scripts/depth_factorial/crosscrop_next_round.sh p0-before

# Step 2: only the two missing 2×2 cells, exact200×256 budget
bash scripts/depth_factorial/crosscrop_next_round.sh train-near-multi
bash scripts/depth_factorial/crosscrop_next_round.sh train-wide-multi

# Step 3: same fixed last.pt on same CAD evaluation objects
bash scripts/depth_factorial/crosscrop_next_round.sh eval-near-multi
bash scripts/depth_factorial/crosscrop_next_round.sh eval-wide-multi

# Step 4: recompute same-raw-point P0 with all five models and full 2×2 factor
bash scripts/depth_factorial/crosscrop_next_round.sh p0-after
bash scripts/depth_factorial/crosscrop_next_round.sh compare
```

**Stop conditions：**preflight失败、source SHA不匹配、旧ROI训练序列/epoch/裁剪参数变了、冻结非DPT参数发生更改、九格距×crop不均衡或CUDA loss非有限，都必须停。不要改strict=False、制造签名、训练新的ROI权重再当原来的。若旧117本地代码有必要兼容补丁，应记录来源并独立验收，且不得改变模型数值/采样合同。

## 6. 产物、判别与解释

- `roi_preflight.json`：历史ROI源注册、训练序列和初始head/物理源已验收。
- `p0_existing_roi/{summary.json,per_object_distance_crop_arm.csv,REPORT_CN.md}`：已有模型相同原始ray的3crop预测分歧（新训练之前就可执行）。
- `crosscrop_heads/{near_only,near_mid_far}/{last.pt,complete.json,sample_schedule.json,history.json}`：新两组、DPT头的固定last权重和实际supervised pixels。
- `crosscrop_eval/{near_only,near_mid_far}/cad_*/`：15测试物距×crop的同源冻结深度与3D同表面响应。
- `p0_post_multi/`：五模型的同ray三crop预测一致性与相同GT误差分解。
- `p1_factorial_comparison/{summary.json,per_cell_factor_effects.csv,REPORT_CN.md}`：四臂主效应、交互、场景内不同物距和测试crop下β/MAE/Δz误差。

**主要决策指标预先固定为：**
1. Test adaptive crop的`β`能否从约0.031明显接近1，同时MAE、Δz误差下降；
2. Fixed ROI远距优点（β≈0.716）是否保留；近0.35m是否明显退化；
3. 所有三个测试crop在同一物理表面点上的深度预测不一致性是否减小；
4. 因子交互`(wide−near)_multi-(wide−near)_roi`是否支持crop多样性改变了距离监督的可迁移性。

注意: **当前仅两枚历史已检视的eval CAD，任何CI只用于展示对象差异，不可用于确认跨未知类别/材质泛化。** 要做后续确认必须先收集/登记新的CAD对象和独立RGB/材质，再采用预注册集测试。合成深度改善还必须经过新的物理grasp动作生成/CAD eval才能涉及抓取AP，本轮没有这部分。

## 7. 验收清单

- [ ] 原Git分支worktree及main身份已实际核查；
- [ ] CPU tests/compileall/bash -n在117实际PASS；
- [ ] 先验收historical ROI heads和 source SHA、不可用本轮重跑旧头替代；
- [ ] 每步physical distance schedule与crop schedule正确交叉9格，不出现crop与距离绑定；
- [ ] 两新头 only original DPT depth_head可训练，其他模型参数及权重SHA不变；
- [ ] 两新头与ROI旧头的`registration_identity_sha`、初始Depth Head SHA、train CAD抽样序列、steps/pixels/optimizer/seed一致；
- [ ] 每种测试方法5距离×3crop×2CAD同源，3D同表面β/MAE全量；
- [ ] P0训练前后的同点物理像素选择一致、目标CAD真实签名一致；
- [ ] 2×2报告不根据GC6D已观察AP选择或修改权重，按对象等权；
- [ ] 记录旧117代码`842ecda7...`本地执行patch与远端最新HEAD差异，不吞掉数值变更；
- [ ] 不覆盖任何历史结果，不声称新真实GC6D指标或机器人成功率；
- [ ] 本轮输出只有服务器真正运行后才可写结果数值；本工作单不包含猜测结果。
