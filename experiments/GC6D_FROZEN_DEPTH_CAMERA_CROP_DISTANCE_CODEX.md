# GC6D: Frozen Metric-Depth Camera / Crop / Distance Response

**日期：2026-11-16 · 使用原分支** `exp/quality-strata-conflict-supervision`。不新建远端分支、不修改 `main`、不覆盖前一轮任何AP或CAD验收结果。

**状态：代码及工作合同已写入仓库；117真实CUDA与78场景结果尚需执行。本文件不报告新的观测结果。**

## 0. 实验动机：先诊断距离机制，再考虑修复

来源：上传的 `WORKPLAN_TO_GC6D_AUDIT_SUMMARY_CN.md`，其中的GC6D 78个D435场景、1014帧已经过探索性评估。

该报告的主要事实：
- Native/score64/all AP 仅0.211850/0.220530/0.242985%；523/1014帧经过CAD背景邻域过滤后没有动作。
- GC6D预测动作深度中位约0.397m，参考深度约0.656m；预测/参考比值中位约0.595；跨帧深度Spearman约0.1005。
- 390/523空帧全部候选的z都在真实CAD几何前方至少5cm。参考深度移动同射线动作中心能够大幅恢复5cm邻域覆盖，但不是新可部署AP。
- GC6D原生相机/内参/crop/单位/C2W变换此前已排查，无足以解释20–50cm偏差的明显实现错误。
- 该历史冻结深度 `global_film` 使用 `camera_pose_vec` 对DINO特征做FiLM；虽然 `DINOv2DepthRegressionNet.forward` 接收 `camera_K`，**这一模式不直接用K调制depth features**；K由后续反投影和RayRoPE使用。

这一轮要区分：
1. **Distance prior / response compression**：在同一个GC6D场景13视角中，真实距离变化时预测米制深度是否有效跟踪？
2. **Digital crop/zoom sensitivity**：相同物理场景的图像重新裁剪/缩放、同时严格更新K，模型的预测深度在相同相机射线上是否稳定？
3. **Pose-conditioned response**：固定RGB和K、替换为同场景其他实测朝向，FiLM是否显著改变深度？
4. **K-only negative control**：固定图像和pose，仅改变K，global_film深度输出应与baseline数值完全一致（这是代码路径检查，不是K有用/无用的性能试验）。

**严禁据此直接得出“焦距导致depth错误”或“pose FiLM是唯一原因”。** GC6D中的camera、工作距离、crop及物体/背景分布共变。数字zoom与换真实相机镜头也不是同一种干预。

## 1. 精确复用的模型和数据

模型：历史 **MIX50 Point RayRoPE + GVAR36 e15**：

- 固定checkpoint SHA256 `2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`；
- `rare_grasp.runtime.Runtime` 复用已有 `bound.json` 严格加载完整checkpoint，明确 epoch=16、mix_training_fraction=.5、point_gvar、global_film、geometry_depth_source='pred'、use_obs_depth=False；
- 初始化后 `model.eval().requires_grad_(False)`；只调用原本完整模型的 `depth_net`，无梯度、无训练、无fine-tune；
- 深度映射本来就是 `sigmoid(raw_depth)*max_depth`，不是自动强制min_depth=0.2；不可在诊断里修改输出范围或夹爪几何。

数据：**同一份**前次GC6D来源审计选出的78个 D435 test scenes，13张视图/scene，共1014帧。文件 `GC6D_SCENES` 必须是**真实已验收的78个整型scene ID**构成的JSON数组，不能因为报告没有列ID就随意选78个代替。

预处理严格复用 `dataset/gc6d_dataset.py:GraspClutter6DMultiDataset` 的 `get_data(load_label=False)`：
- 原1920×1080传感器depth与label参与native `workspace_depth` mask、crop选取；
- 原 `img_transforms`：ToTensor → Resize(448,448) → ImageNet Normalize；
- 使用dataset返回的 `img`、crop-resized `K`、`sensor_depth_m` 和 `gt_depth_m`；
- 由于unlabeled loader不返回pose，按源 `scene_camera.json`、`build_camera_orientation_metadata` 独立获得 `camera_pose_vec`，保留原数据 C2W 语义；不把JSON的BOP键名反解释；
- 由原K/已resize K反推出整数crop box，并用源码 `resize_intrinsics_with_crop` 等价的 `K'=A_crop Kraw` 严格复核；否则停止；
- foreground仅取crop后的原分割 `seg>0` 且参考depth>0的有效像素；不把背景/空白padding或无效传感器depth计入测距误差。

**重要：**模型预测仍只输入标准化RGB及pose/K元数据，真实sensor/rendered depth仅用于loader crop/workspace和**离线评价**；因此整个输入流水线并非完全无传感器深度特权。不要以此实验宣称整条pipeline没有传感器深度。

## 2. 实验A：同一场景多视角距离响应（全1014帧）

每帧独立预测冻结深度，保存：
- scene、GC6D原始img_id和anno_id；真实crop box、K' fx/fy/cx/cy、pose、tilt、源图像/参考深度哈希；
- 前景同射线的预测/参考深度 **中位数**、中位预测/参考比值、中位误差，MAE、AbsRel、有效像素数；另外记录rendered GT深度下的相同统计，严格分开两种reference来源。
- 逐场景13视角的 `z_pred` 对 `z_reference` OLS斜率、Spearman，以及真实距离span；最小span默认3cm，不足不作可解释斜率；
- 汇总**按scene重采样的bootstrap 95%CI**、跨场景与场景去中心化后的距离响应斜率。不要把1014个视角当作互不相关的1014个独立场景。

核心解释：
- 预测值集中、真实深度变化但斜率接近0：支持距离响应不足；
- 同场景中能随真实距离变化但全局仍偏浅：支持存在可再分析的相机/场景层偏差；
- 两者都不足以直接识别该误差源于FiLM、K、DINO或训练距离分布哪个模块。

**绝不在这78个已观察场景上拟合线性/全局深度倍率，然后把同一批场景的AP当泛化收益。**

## 3. 实验B：数字zoom/crop，与K同步的物理射线对齐

为了避免重新跑整个1024-query抓取网络，这一实验**只跑冻结metric depth head**。

默认每场景预先取第1和最后1视角，各78×2=156张，固定四组数字图像坐标变换：

| 名称 | A_img定义 | 用途 |
|---|---|---|
| `zoom_in_115` | 以中心缩放s=1.15 | 同射线接近／裁剪响应 |
| `zoom_out_085` | 以中心缩放s=.85，外圈padding | 图像有效区域/视角扩大响应 |
| `shift_right_08` | 横向平移0.08×W | crop中心偏移敏感性 |
| `shift_down_08` | 纵向平移0.08×H | crop中心偏移敏感性 |

对所有变换，构造明确的二维单应 `A_img`：

\[
u' = A_{\rm img}u,\qquad K'=A_{\rm img}K.
\]

在新图像坐标u'做采样 `I'(u')=I(A^{-1}u')`，在同一个source ray上比较 `D'(u')` 与 `D(A^{-1}u')`。参考depth与前景mask以对应方式warp（nearest），无效/新padding区域不计入评估。

**必须测试：**独立3D点投影 `A·project(K,X)` 与 `project(K',X)` 误差≤3e−4像素；反推原GC6D loader crop内参≤2e−3；任何不一致停止，不调整容差。

记录每组：
- 同射线 `median(D_new - warped D_original)`；
- 同射线 median absolute difference 及 median ratio；
- 同样前景参考depth下的median error、MAE、AbsRel；
- 原图有效重叠比例。
- Zoom-out的合成padding像素必须排除；不能把padding 0 depth误当背景或空抓。

**局限：**数字resize/crop是图像重参数化，同时更改有效K，而不是换真实相机焦距、真实距离或光学畸变。K不直接进入global_film depth，图像中的物体像素尺寸仍变化，因此此实验体现的是**image-scale/crop sensitivity**，不是独立的physical-focal-length causal effect。

## 4. 实验C：K-only negative control 与 pose FiLM stress

在同一156视角：
- **K-only**：图像和pose完全不变，令K的fx/fy乘1.2。冻结 `global_film` 代码不使用K参与depth head，因此预测应严格与baseline一致；容差默认1e−7。若失败，说明运行分支/模型代码/随机状态不符合冻结合同，停止。
- **Pose swap**：同一个GC6D scene13视角中选择与当前pose角差最大的另一视角的真实 `camera_pose_vec`，保持RGB和K不变。记录角度差、depth shift、ratio与原参考误差。若角差很小，不能解释为明显的pose敏感性检验。
- 保持gravity元数据、完整RGB crop及模型权重不变。**这种固定图像只替换pose的组合一般不是物理一致观测**；只能量化FiLM的输入敏感性，不能证明真实相机姿态变化导致某种几何偏差。

K-only仅用于代码路径负控制，不是“补K有无价值”的学习实验；global_film没有读K，预期不响应本就是设计合同。

## 5. 命令（服务器117）

在现有 branch 的独立worktree继续，不动 main：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-frozen-depth-response \
  origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-frozen-depth-response
git status --short
git worktree list
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/frozen_depth_response_20261116
export BOUND_CONFIG=/ABS/verified_mix50_GVAR_e15_bound.json
export GC6D_ROOT=/data2/robotarm/dataset/GraspClutter6D
export GC6D_SCENES=/ABS/accepted_78_gc6d_D435_scene_ids.json
export DEVICE=cuda:0
```

**不能从上一轮总结猜78个scene ID。** Codex应读取 `selection_audit_20261116` 或 `gc6d_low_ap_audit_20261116` 的真实provenance/scene列表并比对exact source。格式为 `[2, ...]` 整型数组；列表需另行对实际文件做hash记录。实际数据根以117检查结果为准，不以示例路径覆盖已有结果。

CPU纯函数测试（不调用CUDA或CAD）：

```bash
bash scripts/frozen_depth_response/run.sh check
```

单独的2帧smoke（**不能混入正式1014帧分析**）：

```bash
SMOKE_FRAMES=2 SMOKE_OUTPUT="$RUN_ROOT/smoke" \
  bash scripts/frozen_depth_response/run.sh smoke
```

Smoke验收：
1. 通过checkpoint SHA/epoch16 MIX50 Point+GVAR，原 `depth_net` 的K-only输出差≤1e−7；
2. GC6D crop K重建与原K匹配≤2e−3；
3. RGB baseline与深度值有限、同射线有效前景数≥128；
4. 所有zoom/crop的K'=AK投影最大误差≤3e−4像素、valid mask有效；
5. 同源图像和参考深度哈希、pose metadata、各干预输出数值均可复现；无模型梯度、无新权重；
6. 运行进程没有依赖额外 CAD evaluator；必要环境扩展与runtime来源与上轮一致。

正式全78scene/1014帧：

```bash
SAVE_MAPS=1 PERTURB_VIEWS_PER_SCENE=2 \
  bash scripts/frozen_depth_response/run.sh run

bash scripts/frozen_depth_response/run.sh analyze
```

产物：
- `full/protocol.json`：模型、场景清单、K和crop规则、干预预注册参数及代码SHA；
- `full/per_frame_response.csv`：每帧baseline和固定干预的配对几何/深度响应；
- `full/manifest.json`：确认1014帧与每个输出SHA、冻结模型和运行协议；
- `full/maps/scene_XXXX/img_XXXX.npz`：干预场景的基准预测、参考深度、前景mask与K，可用于人工可视化；不是GT训练数据；
- `analysis/per_scene_distance.csv`、`analysis/summary.json`、`analysis/REPORT_CN.md`：场景内关联/斜率、数字缩放与pose敏感性、按场景bootstrap。

**不可把下一轮想要的GC6D AP数字编造进本结果。** 这轮不做模型更新、倍数拟合、候选中心修正、CAD新动作标签或GraspNet/GC6D官方AP，必须保留之前AP单独不变。

## 6. 后续研究决策门槛

| 实验观察 | 可支持的判断 | 尚不能支持 |
|---|---|---|
| 同场景真实距离变化、预测斜率很小 | 距离响应压缩或训练几何范围限制 | 唯一归因于camera pose/焦距 |
| zoom/crop后同射线预测显著变化 | 冻结图像特征受尺度/裁剪重参数化影响 | 单独证明真实光学焦距是根因 |
| K-only近乎零变化 | 当前 global_film 深度分支不直接消费K的代码路径被复验 | 添加K embedding一定改善 |
| pose swap影响明显 | 模型对pose FiLM元数据有敏感性 | 真实多视角pose变化是bias根因 |
| 所有干预弱，但跨域距离仍偏浅 | 当前干预不能解释完整偏差 | 排除训练距离、对象尺度/appearance先验 |

完成后，若需要实际学习相机/距离校准，应在**不曾用于诊断或选模**的新数据上预先注册方案与校准集，不能用已多次看过的GC6D78场景/GraspNet Novel选择全局尺度或修复方法。若深度改善，还须重新用改变后的物理动作执行真实CAD质量与官方AP，不能继承旧动作评分和标签。

## 7. Stop conditions /验收

- [ ] 当前远端HEAD为相同 `exp/quality-strata-conflict-supervision`、main无改动；
- [ ] 使用真实GC6D78scene_ID与1014帧、D435相机、native workspace crop/pose/factor配置；
- [ ] Python语法检查、`test_frozen_depth_response.py` CPU合成测试及shell `bash -n` 均已**实际通过**；静态检查不能替代CPU测试。
- [ ] 真实模型SHA与已验收e15一致、所有权重冻结、深度仅从RGB+pose预测；
- [ ] 真实同射线/seg foreground有效、rendered和sensor参考来源分开；空/不充分前景显式标missing，不伪造零误差；
- [ ] K'=AK的forward/backprojection一致；zoom-out padding不被计入；
- [ ] 每场景13-view距离响应、场景去中心化slope与scene cluster CI；bootstrap不能把帧当IID；
- [ ] 真实pose swap选本场景另一已录制相机朝向，记录角差，不称真实物理干预；
- [ ] 原AP/CAD标签/geometry-correction权重与脚本不被修改；
- [ ] 新结果按冻结模型响应作探索性诊断，不声称新GC6D泛化AP；
- [ ] 交付报告明确回答distance响应、crop/zoom稳定性、K代码路径、pose敏感性四项，各列原始证据和限制。

**本工作单撰写时未连接117执行真实GPU结果或CPU tests。需要Codex在117严格按步骤验收，未通过就停。**
