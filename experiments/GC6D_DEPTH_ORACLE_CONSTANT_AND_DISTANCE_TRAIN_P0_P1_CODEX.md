# GC6D Depth Follow-up P0/P1：Constant Oracle 与物理 Distance Coverage

**日期：2026-11-17 · 同一分支 `exp/quality-strata-conflict-supervision`，不修改 main、不新建远端分支、不覆盖此前的 GC6D 78场景、A Oracle、修正曝光 B 或旧 AP。**

本工作单交付**脚本／数据合同／测试**，并非新的实验数值。真实117上的CPU、Blender、CUDA、深度训练或CAD物距结果需在执行时验收；不要宣称后台执行或在本聊天完成了117真实测试。

## 0. 实施依据和判别问题

上传的 `depth_oracle_factorial_REPORT_CN.md`（2026-11-17），**仅使用修正曝光后的B结果**：

A：78scene×13=1014 GC6D帧，四象限空间留出，原始/scale/shift/affine的留出MAE为30.168/9.413/7.289/5.256cm；affine有效3999折中的2005折（50.14%）出现负系数，a中位约−0.00391。Affine的MAE与梯度改善可能只是向常数压缩或非物理排序翻转，**不能解释为局部几何已恢复**。

B：修正曝光后仅CAD编号2/5/8/9四个对象，物距[0.35,0.45,0.55,0.70,0.85]m，originalK固定fx=fy=550，5距离×3crop共60冻深度前向。固定ROI β略优于full，adaptive最差，但三者离理想β=1很远。旧RGB几乎全白的 `cad_*`、`b_aggregate` 被明确撤销，**禁止进入新审计与训练**。

当前问题：
- **P0**：原Affine低误差到底是局部形状证据，还是等价Constant oracle？限制a≥0后是否明显失去收益？在真实深度边缘与光滑区域表现如何？
- **P1a**：真实物体随物距的投影尺寸信息在三个crop下到底保留多少？有效K′、448像素object占比与Frozen预测的距离响应有什么关系？
- **P1b（受控训练）**：如果人为增加**物理距离训练覆盖**，而不改变架构、初始权重、对象、训练步数、监督像素预算及crop，能否改善在不同CAD对象上的米制深度和同一物理表面距离响应？

不将P1a描述性相关性写成K/crop因果证明；不将P1b的少CAD合成测试当作GraspNet/GC6D真实域泛化或新抓取AP。

## 1. 代码与模块位置

在当前分支新增：

| 文件 | 核心功能 |
|---|---|
| `diagnostics/depth_factorial/oracle_followup_p0.py` | 常数均值/中位数oracle、a≥0的受限Affine、沿用原四折的配对MAE/梯度与真实深度edge/smooth分层 |
| `scale_cues_p1.py` | 四个已接受的 corrected CAD B 渲染：读取签名source+frozen+analysis，RGB QC重新拒绝近白图，测object投影bbox与448占比、crop area、有效fx′、distance响应 |
| `distance_split_p1.py` | 显式注册物理渲染CAD对象train/eval集合，CAD mesh SHA身份互斥、历史审计记录与RGB QC；不根据test结果自动拆分 |
| `distance_train_p1.py` | 在同一个完整MIX50冻结模型上**只解冻 DPT depth_head**，Near-only与Near+far等参数、等步骤、等每步256像素监督，固定last权重 |
| `distance_eval_p1.py` | 三模式native/near/wide在相同eval CAD原5物距×3crop上预测，复用原B同世界表面depth分析 |
| `distance_compare_p1.py` | 校验两arm的初始化、CAD序列、训练步数、pixels/step、优化器和crop；按CAD对象统计paired ΔMAE/β/Δz误差/DiD |
| `scripts/depth_factorial/followup_p0_p1.sh` | check、P0、P1a、P1b训练／评测／对照stage |
| `tests/test_depth_factorial_followup.py` | 无CUDA/Blender的数学合同、空间留出、近白render拒绝、物理CAD来源身份审计与分组测试 |

所有真实估计器新参数都留在实验输出目录的 `last.pt` 中，不改变生产分支已有的DepthNet、Grasp Head或CUDA推断入口。

## 2. P0：Constant Oracle + 非负Affine +边缘分层

严格复用已经验收的 `gc6d_depth_followup_20261117/p1_cache/manifest.json`（真实路径必须读取117）。

原有4模式另增：

1. `constant_mean`：`z'=mean(z_ref,fit)`，是给定拟合区域的L2最优常数，**不使用冻结模型的局部深度变化**。
2. `constant_median`：`z'=median(z_ref,fit)`，对应L1/MAE最优常数，防止用较弱的mean oracle高估模型相对形状优势。
3. `affine_nonnegative`：对 `z'=a z_pred+b` 做a≥0的有截距约束OLS。若无约束a<0，严格落在a=0的边界、b取拟合区域参考均值，相当于constant_mean。若输入深度std<3mm，拒绝Affine识别，不自动回退常数再冒充受限affine。

保持原样：
- 每帧[448×448]四折，三象限fit/一象限test；
- fit侧8像素保护带、最少512/128真实前景像素；
- 原有 `identity/scale/shift/affine` 的同fold MAE必须逐值复现（可用CPU测试先比较）；
- Ground truth仅参与fit其余象限，不进入网络输入或部署；
- 报告四级聚合：fold→frame→78 scene→scene bootstrap，而非按3999fold当3999独立场景。

新增**局部几何指标**：
- 留出区域的mean-centered depth MAE；
- 相邻有效前景像素的局部gradient MAE；
- 以原传感器参考的真实相邻深度差预定义：
  `smooth≤2mm/pixel`、`edge≥6mm/pixel`、其余中间层；
- edge-gradient MAE、edge梯度方向一致率、同区域Spearman；
- 在**相同fold与相同heldout像素**上，比较constant vs constrained affine、shift、unconstrained affine的pairedΔ，并统计受限Affine落在a=0的频率。

**解释门槛：**若受限affine无明显优于constant_median/mean，且真实depth edge梯度未改善，则不应以“局部形状已经足够好，只差米制锚点”写TRO主故事。

运行：

```bash
# /home/robotarm/EconomicGrasp 的已存在原分支，可用独立detached worktree。
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-depth-followup2 \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-depth-followup2
git status --short

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/depth_followup_p0p1_20261117
export GC6D_CACHE_MANIFEST=/ABS/gc6d_depth_followup_20261117/p1_cache/manifest.json

bash scripts/depth_factorial/followup_p0_p1.sh check
bash scripts/depth_factorial/followup_p0_p1.sh p0-smoke
bash scripts/depth_factorial/followup_p0_p1.sh p0-run
```

产物：`p0_oracle/per_fold.csv, summary.json, REPORT_CN.md`，含各mode同场景CI与negative/clamped比例。严禁回写旧a_oracle。

## 3. P1a：修正曝光的物理尺度证据审计（无训练）

先**读取真正验收的四个 corrected 物理渲染目录**，而不是根据目录名猜数据正确。每CAD提供三份文件及实际SHA：

- `cad_fixed_*/b_render/manifest.json`（source）
- `cad_fixed_*/b_frozen_preds/manifest.json`（frozen）
- `cad_fixed_*/b_analysis/summary.json`（same-world surface）
- 原 `reports/B_ACCEPTANCE.json` 或对应只读独立验收证据与实际SHA。

已同步上轮117实际接受的Blender修复到原分支：固定−6EV曝光、Blender 4.2的RGB EXR标量通道、Composite输出链接、对实际保存的RGB做饱和/低对比度拒绝、脚本异常强制非零退出。**新CAD训练源必须使用更新后的渲染器并独立验收相机投影和光轴深度；旧全白render不能复用。**

为这些文件创建 `CORRECTED_B_INDEX`，格式JSON数组：

```json
[
  {
    "source_manifest": "/ABS/cad_fixed_02/b_render/manifest.json",
    "source_sha256": "ACTUAL_SHA256",
    "frozen_manifest": "/ABS/cad_fixed_02/b_frozen_preds/manifest.json",
    "frozen_sha256": "ACTUAL_SHA256",
    "analysis_manifest": "/ABS/cad_fixed_02/b_analysis/summary.json",
    "analysis_sha256": "ACTUAL_SHA256",
    "acceptance_evidence": "/ABS/reports/B_ACCEPTANCE.json",
    "acceptance_sha256": "ACTUAL_SHA256"
  }
]
```

实际列表需要4项（2/5/8/9），本工作单只示范结构，不创造实际SHA/路径。

审计对每个对象5距离×3crop测量：
- 原RGB饱和白比例、全图std、对象vs背景颜色差；**近乎全白旧渲染直接拒绝**；
- 原图真实投影物体bbox宽、高及占全图比例；
- crop大小、area、对象bbox在448内的归一化宽高、mask像素占比；
- crop的有效`f'_x,f'_y`及对应冻结预测深度中位数与参考中位数；
- 物距、object pixels、有效fx和pred depth的Spearman与远近比例；
- 复用原B的同world surface β/DiD，不对像素重新拟合物距修复。

**这些指标只是说明adaptive是否在视觉输入中压低物体尺度变化。** 同时失效的full/ROI/adaptive说明不能自动宣称“crop就是全部问题”。

运行：

```bash
export CORRECTED_B_INDEX=/ABS/corrected_4cad_B_manifest_index.json
bash scripts/depth_factorial/followup_p0_p1.sh p1-cues
```

产物：`p1_scale_cues/per_object_distance_crop.csv, paired_surface_beta.csv, summary.json, REPORT_CN.md`。

## 4. P1b：Near-only vs Near+far 等预算depth head训练

这部分**只在实际具备来源的物理渲染CAD数据上执行**，不伪造CAD或训练/测试独立性。

### 4.1 分组先登记，不能根据已看过的远距表现挑对象

准备两个JSON数组 `TRAIN_CAD_INDEX` 和 `EVAL_CAD_INDEX`，每个条目：

```json
[
  {"manifest": "/ABS/cad_fixed_X/b_render/manifest.json",
   "sha256": "ACTUAL_SHA256_OF_MANIFEST"}
]
```

与 `CAD_OVERLAP_REVIEW`（真实Object CAD同类变体/文件重复核查记录）。

运行：

```bash
export TRAIN_CAD_INDEX=/ABS/train_cad_physical_manifest_index.json
export EVAL_CAD_INDEX=/ABS/eval_cad_physical_manifest_index.json
export CAD_OVERLAP_REVIEW=/ABS/cad_overlap_review.json

bash scripts/depth_factorial/followup_p0_p1.sh p1-register
export P1_REGISTRATION="$RUN_ROOT/p1_cad_split.json"
```

注册器验收：train≥2 CAD，eval≥1 CAD；物体mesh SHA不同，固定0.35/0.45/0.55/0.70/0.85m、固定渲染K和物理C2W、RGB非近白且有效；身份重叠拒绝。**不同mesh哈希不保证语义独立**，原本四个CAD已在B开发中查看过，即使做对象拆分也只能称`CAD_object_holdout_from_finetuning`，不能称之前未见过的完整外部测试。

如果仅有当前四个已看过CAD，建议选2train/2eval并固定；统计对象数2的CI不强。更好的确认性研究应先收集**新的**、此前未用于本项目设计的CAD，然后预先注册对象train/eval集合。

### 4.2 严格匹配两arm

- 共同初始化：历史MIX50 GVAR36 e15全模型SHA `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。
- 固定冻结：`depth_net.depthnet.pretrained` DINO、pose-FiLM、GVAR、全部grasp模块、其余net参数。**仅 `depth_net.depthnet.depth_head` 可训练**；BN仍用Eval行为（无running-stat更新）。
- 共同source：相同训练CAD对象序列（固定seed）、相同crop策略（默认fixed_roi）、相同500? **默认200** optimizer steps、每步256有效前景像素、相同AdamW LR 2e-5、weight-decay1e-4、SmoothL1 β0.03、梯度clip=1。
- 唯一干预：
  * `near_only`：每step只取物理距离index0=0.35m（重复近距离）；
  * `near_mid_far`：按step index%3循环真实距离0.35/0.55/0.85m。
- **总监督像素预算**为steps×256，在两个arm完全相同；不因远距对象pixel变少就自动降低样本数。远距有效像素不足256直接停止并登记。
- 训练禁止用eval CAD任何depth标签或AP调学习率、epoch。每arm固定`last.pt`，不使用test指标选best。
- 训练从相同base初始权重加载，sample顺序、超参数、Head参数量用SHA/协议在compare时强校验；保存训练loss与每步具体物距/对象来源。
- 已在之前出现depth-gradient影响grasp head不稳定：此处完全独立深度head supervision，抓取分支不参与训练和更新，也不产生Checkerboard相关对照结论。

执行：

```bash
export BOUND_CONFIG=/ABS/verified_mix50_GVAR_e15_bound.json
export GRASPNET_ROOT=/ABS/GraspNet
export DEVICE=cuda:0
export TRAIN_CROP_POLICY=fixed_roi
export STEPS=200
export PIXELS_PER_STEP=256
export SEED=20261117

bash scripts/depth_factorial/followup_p0_p1.sh p1-train-near
bash scripts/depth_factorial/followup_p0_p1.sh p1-train-wide
```

### 4.3 在与训练CAD不重叠的对象上统一评测

为避免“只测密集MAE”的误判，评测三模式native/near/wide，**三者在相同eval物体及原生5距离×3crop上执行**：

```bash
bash scripts/depth_factorial/followup_p0_p1.sh p1-eval-native
bash scripts/depth_factorial/followup_p0_p1.sh p1-eval-near
bash scripts/depth_factorial/followup_p0_p1.sh p1-eval-wide
bash scripts/depth_factorial/followup_p0_p1.sh p1-compare
```

输出：
- `p1_heads/{near_only,near_mid_far}/last.pt`、`complete.json`、`history.json`、`sample_schedule.json`；
- `p1_eval/{native,near_only,near_mid_far}/cad_*/manifest.json`及预测npy；
- `cad_*/same_surface_analysis/summary.json`：原B完整同3D点物距响应斜率和crop DiD；
- `p1_comparison/paired_object_contrasts.csv, summary.json, REPORT_CN.md`：按heldout CAD对象pairedΔ，bootstrap对象数而非重复像素。

**判别：**near+far在额外距离下MAE及同表面β更接近1，且近距离显著回退受控，支持覆盖训练数据范围可缓解部分失败。但即使合成CAD改善，也不能直接外推真实GC6D物体或最终grasp AP。若wide不改善，需继续分析网络输出表示、DINO尺度线索、材质/渲染偏移；不能仅由合成失败断言model架构不可能泛化。

## 5. 严格验收门禁

- [ ] 原分支HEAD一致、main不动；Source code SHA记录、git status干净。
- [ ] CPU unittest／Python compileall／shell bash-n **实际运行并通过**；旧有A/B CAD修复后的Renderer RGB/Camera raycast验收不应跳过。
- [ ] P0原4模式同fold指标与上轮3999/4056折、MAE/Gradient一致；新增Constant与受限Affine用相同留出区。
- [ ] P0负Affine clamp=0的系数/预测退化Constant-mean，且负系数不被当作可部署物理深度。
- [ ] P1a原白色被拒绝的CAD源不得混入，四个修正CAD源和B独立验收SHA完整。
- [ ] P1b独立CAD注册、无训练/测试文件哈希重叠、原B物理K/pose/object材质固定。旧四CAD本身已被看过，不能拿它做未触碰的Novel泛化。
- [ ] 两arm同样的初始Depth Head SHA、相同对象抽样顺序、训练步数、监督有效像素数和优化器；只改变物理距离。
- [ ] 每一测试对象都计算5距离×3crop前景误差和同world surface β，决策不能仅基于单个MAE。
- [ ] 训练权重只保存在新结果目录，绝不覆盖已验收的MIX50 e15基座、GC6D78场景或Novel AP数据。
- [ ] 无真实GC6D改动/新grasp AP时，不声称系统抓取性能提升。

## 6. 2026-11-17：已验收P0/P1结果后的最小2×2物距×Crop后续

上一轮在117完成Constant/非负Affine（GC6D 1014帧）及两个DPT深度头的200步固定ROI near-only/near+far对照。已验收结论：Constant MAE5.395cm、非负Affine5.403cm，局部几何仅有小幅正向证据；原物理B测试集两对象中，near+far在fixed ROI 0.85m下β≈0.716，但在adaptive crop下β≈0.031，说明受控距离监督存在很强的裁剪依赖。旧117执行源码本地commit `842ecda7...`未推送，当前远端需在实际运行前核对其修复。

**后续独立Codex工作单：** [GC6D_DISTANCE_CROSSCROP_NEXT_ROUND_CODEX.md](GC6D_DISTANCE_CROSSCROP_NEXT_ROUND_CODEX.md)。

新代码只补充两组训练：near-only×multi3和near+far×multi3，历史两组fixed ROI头及其评估作为固定对照，构成2×2实验。采用9-step Latin-square的距离×crop正交日程，200步下wide九格各22–23步，near-only三crop分别67/67/66步。另有不训练的P0 same-raw-ray跨crop质量审计，必须沿同一原始物理像素和真实CAD来源比较，而不是逐crop前景平均值。强制跑历史ROI源/模型preflight，拒绝篡改旧权重、签名或隐性重训原头。

执行脚本：`scripts/depth_factorial/crosscrop_next_round.sh`。新物理模型训练、CPU/117验收、统计结果仍待实际运行；原GC6D/GraspNet AP不变。
