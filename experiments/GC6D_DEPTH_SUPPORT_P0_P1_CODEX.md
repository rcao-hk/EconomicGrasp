# GC6D Depth Follow-up：P0训练域几何支持域审计 + P1同一3D表面多视角距离响应

**日期：2026-11-17；继续同一远端分支 `exp/quality-strata-conflict-supervision`。不得创建新远端分支、覆盖原GC6D/Novel报告、修改main或运行校准训练。**

**这是实验实现与Codex执行协议。** 至本工作单写入时，117上的真实 P0/P1 新结果尚未运行；不能把上轮冻结响应的负斜率重新包装成新结果。

## 0. 直接动机与来源

用户上传 `frozen_depth_response_DETAILED_REPORT_CN.md` 是已验收的**冻结 MIX50 Point+GVAR36 e15** GC6D78场景／1014帧响应报告：

- 场景去中心化米制距离回归斜率 `-0.031293`，场景CI `[-0.054923,-0.006347]`；说明仅靠不同scene的平均距离偏差不能解释预测响应问题，仍受每视角前景构成变化影响。
- 预测／参考场景聚合比值约 `0.599364`；zoom-in1.15产生约−5.34mm同射线变化，zoom-out0.85约+6.28mm，pose-swap平均绝对变化约13.61mm；不足以解释几十厘米缺口。K-only为0只是global_film代码路径负控制。
- **1014个rendered depth文件缺失**，因此这轮只能使用已核验的sensor reference；不能以临时0、sensor别名或不存在的虚拟GT填充。
- 上轮只统计每视角有效前景深度中位数。前景像素集合会随相机位置、遮挡、crop与物体可见性变化，因此不能确定场景内负斜率是否代表**同一个物理表面点**的距离响应。
- 上轮采样点是GC6D D435 camera，对应78个场景已被项目反复查看；新实验仍为探索性，不能按这批场景选Depth Head修复倍率。

**本轮两个目标：**
- P0：核查实际训练输入（GraspNet Original、GNTrans）与GC6D的**同原生预处理下**传感器/虚拟距离、workspace crop、有效焦距K′和物体前景像素尺度支持域，检查 \(\rho(z_{\rm ref}, f'_x)\) 与 \(\rho(z_{\rm pred}, f'_x)\)，分析模型是否进入训练几何范围之外。
- P1：用GC6D历史验证过的C2W，将源视角有效深度像素恢复成**世界3D点**，重投影到同scene其他视角，以目标观测深度、分割与三维闭环保证同一个可见表面。然后比较真值 \(\Delta z_{\rm sensor}\) 和预测 \(\Delta \hat z\)，不再比较不同前景区域的中位数。

**不做：**深度乘倍率、修改crop policy训练、新Depth Head、用sensor depth当RGB模型输入、改变grasp center、重标CAD或报告“修复AP”。

## 1. 文件和接口

新增模块：`diagnostics/depth_followup/`：

| 文件 | 作用 |
|---|---|
| `common.py` | 对GraspNet Original／GNTrans／GC6D复用原workspace crop、K′计算、原始深度和分割、C2W；只读冻结global_film DPT。 |
| `p0_collect.py` | Original与GNTrans各100训练场景×4帧（默认0/64/128/192，均属MIX50的偶数采样帧）；GC6D使用以前接受的78场景×13视角。输出单帧distance/crop/K来源和米制Depth响应。 |
| `p0_analyze.py` | 三域描述性分布、每场景bootstrap、距离与fx′ Spearman、按原始Original的5–95%分位统计GC6D支持域外比例；GNTrans-Original按scene/frame成对分析。 |
| `p1_cache.py` | 全78×13帧冻结模型深度[448,448] FLOAT32缓存，记录原始RGB/depth/label/相机JSON/scene_gt来源SHA及raw K、crop K、C2W；支持相同合同断点续跑。**不要求任何rendered depth。** |
| `multiview.py` | C2W正逆与scene_gt物体世界位姿一致性核查、源3D点重投影、目标sensor depth遮挡检查、原seg标签一致性、目标3D回投误差门禁；网络预测仅用于最后评估。 |
| `p1_track.py` | 固定每场景有序第1/7/13视角为source，按seg标签均衡抽最多512个真实表面像素，向其余12个view寻找共同可见点，保存每scene原始对应、真实／预测米制距离变化与逐层门禁通过数。 |
| `p1_analyze.py` | 逐scene的through-origin \(\Delta\hat z\) 对 \(\Delta z_{\rm ref}\) 斜率、准确度/可见性、分场景bootstrap CI；未满足位移/样本阈值的scene显式标缺失。 |
| `scripts/depth_followup/run.sh` | 分开smoke/正式P0与P1的执行入口，不覆盖历史结果。 |
| `tests/test_depth_followup.py` | CPU合成刚体多视角、已知距离位移、错误遮挡、标签变化、K裁剪对应及场景bootstrap测试。 |

## 2. 严格匹配哪些数据与模型

冻结 `MIX50 Point+GVAR36 e15` checkpoint SHA：
`2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。

必须使用上轮真实已验收 `bound.json`，模型内部 `pose_depth_mode='global_film'`、`geometry_depth_source='pred'`、`use_obs_depth=False`，整个模型`eval().requires_grad_(False)`。后续只调用 `model.depth_net` 的RGB+pose/K参数接口；**当前global_film虽然传K，但depth仍不直接按K做调制**。

Original：
- `dataset/graspnet_dataset.py` 的GraspNetMultiDataset + Realsense + Original原始传感器depth／label／meta；
- 用native `get_workspace_mask(remove_outlier=True,outlier=.02)`、原`_crop_box_from_mask`、`resize_intrinsics_with_crop` 与 `img_transforms`。不需要虚拟rendered GT；
- 100场景×4预注册偶数帧，属于原MIX50每个domain的训练支持范围，不是假想base-unseen数据。

GNTrans：
- 与Original相同scene/frame和camera元数据，用GraspNetTransDataset的材质增强RGB；
- depth是真正读取 `virtual_scenes` 的**虚拟/渲染深度**，不是现实传感器depth；
- P0报告GNTrans作为辅助域，**不要直接把其Depth MAE与Original/GC6D同类传感器误差相比。**

GC6D：
- 使用上轮真正接受的78个D435 scene IDs；不得任意找78场景代替；
- 不依赖缺失 `virtual_scenes`，直接用原 `GraspClutter6DMultiDataset` 的 `depth_to_cloud/build_mask(workspace_depth,json,0.02)`、原crop及ImageNet归一化；
- `scene_camera.json` 中虽然键名是w2c，但本项目从多视角实例世界位置确认**实质为C2W**；`cam_t_w2c`从mm转m；
- 参考为真实传感器深度（并非无误差），仅供crop及离线P0/P1评价；RGB模型只接收crop后的RGB与姿态元数据。

**注意：三个域的完整流水线都由sensor/virtual depth或seg参与workspace crop，因此这里不是“纯RGB、无深度特权的整个测试系统”。** P0是已接受预处理下的几何分布审计，而非替代实际无特权部署。

## 3. P0：训练域crop/距离/焦距支持域

默认每个train scene的annotations `0,64,128,192`，合计每域400帧。每个GC6D scene13帧，共1014帧。每帧保存以下独立信息：

- 原始RGB尺寸和workspace crop宽高、crop占原图的比例；
- segmented foreground原图像素占比／有效crop区域比例；
- 原K.fx/fy、crop后K′.fx/fy以及其缩放比例；
- 对应前景**同像素**的传感器或虚拟参考z、预测z、MAE、AbsRel、有效像素数；
- 前景有效深度P10/P90、相机朝向与最终源文件SHA；
- P0汇总分别计算 \(\rho(z_{\rm ref}, f'_x)\)、\(\rho(\hat z, f'_x)\)、同scene距离响应斜率，按scene bootstrap而不是逐帧IID；
- 固定Original训练域P05/P95阈值，统计GC6D参考距离、K′、crop宽/高等几何量的out-of-support比例，同时保留GNTrans的虚拟深度性质。

这一分析可能支持“训练几何支持范围狭窄／有效焦距分布漂移”的假说，但**不能唯一证明缺少K输入就是根因**。真实物体尺寸未被直接测量，foreground bbox/crop比例只能作像素尺度代理，不能冒充CAD 3D object size。

## 4. P1：同一个3D表面点的多视角响应

### Step 1. 全视角冻结预测缓存

先全量生成 `1014×[448,448]` FLOAT32预测，保持原`K′/crop/C2W`和源图哈希，不把用于可视化的FP16 PNG当数值真值。严禁从已经缺失的rendered depth中填充几何。可由P0 GC6D收集器使用 `--reuse-gc6d-cache` 直接读取该缓存，省去重复1014次冻结网络前向。

### Step 2. 场景C2W正确性（强门禁）

从`scene_gt.json`逐视角object-to-camera位姿 \(T_{c\leftarrow o}\)，结合对应GC6D相机位姿计算：

\[
T_{w\leftarrow o}^{(v)}=T_{w\leftarrow c}^{(v)}T_{c\leftarrow o}^{(v)}.
\]

同一个obj_id在不同视角的world pose应一致。对至少两个视角可确认的唯一obj_id，报告world translation/rotation差异；若scene median平移>2.5cm或角差>5°，停止该场景/整次正式验收，不擅自反向解释camera外参。

原seg标签值跨view相等仅作为保守匹配门禁，不足以单独认证CAD instance identity。重复obj_id若在单视角出现则跳过该obj_id而不随意选一个对应。

### Step 3. 决定来源点：不使用模型预测质量或AP

对每scene预先固定有序第1、7、13张GC6D帧作为anchors。每张source里按seg>0、sensor depth>0，逐numeric segmentation label抽最多64点、全帧最多512点，SHA稳定种子；不通过任何模型输出筛点。

源像素 \((u_s,v_s,z_s)\) 用raw K和已验证C2W反投影：

\[
X_w=T_{w\leftarrow c_s}K_s^{-1}[u_s,v_s,1]^T z_s.
\]

向其他12个view投影：
\[
X_t=T_{c_t\leftarrow w}X_w,\quad (u_t,v_t)=\pi(K_t X_t).
\]

### Step 4. 三层遮挡/对应过滤，防止错看成同一表面

- 目标raw像素确实在相机视场内；
- source/target的segmentation numeric label相同、有效sensor depth非零；
- z-buffer条件 \(|z_{\rm proj}-z_{\rm sensor,target}|\le\max(8{\rm mm},0.012z_{\rm proj})\)；
- 用目标raw K/深度backproject回world后，与原世界点3D距离≤18mm；
- 两视角预测depth应在当前crop内能真实双线性插值，否则拒绝；**预测depth不会参与对应点选择或可见性门禁**。
- 保存每层数量，使低重叠场景不能被静默扔掉。

### Step 5. 距离响应只基于同一物理点

每个有效match有：
\[
\Delta z_{\rm true}=z_{\rm sensor,target}-z_{\rm sensor,source},
\quad
\Delta z_{\rm pred}=\hat z_{\rm target}-\hat z_{\rm source}.
\]

当 \(|\Delta z_{\rm true}|\ge3\)cm，场景≥40成功配对且至少10个达到3cm时，计算最小二乘**过原点斜率**（不拟合新的深度校准系数）：
\[
\beta_s=\frac{\sum_i \Delta z_{{\rm true},i}\Delta z_{{\rm pred},i}}
               {\sum_i(\Delta z_{{\rm true},i})^2}.
\]

理想同一点距离变化响应应近似1（忽略原传感器与匹配噪声）；近0说明即使同一3D表面移动到不同距离，冻结depth几乎不跟随。也报告 \(\Delta z\) 的符号一致率、变化量绝对误差、同一点轨迹的重复观测数及分scene覆盖情况。

**不能将不满足共同可见性、运动跨度、或scene配对数要求的场景斜率填成0。** 报告78scene中能计算斜率的数量，CI先在每scene聚合后对scene bootstrap，不能用百万像素计数冒充独立统计单位。

### 如何解释结果

- 如果P1 same-surface斜率仍接近0，而旧P0前景中位斜率−0.031：更明确指向米制相机运动响应不足，而非仅可见物体组成变化。
- 如果same-surface斜率明显正向且显著高于旧前景中位斜率：视角下可见区域变化是旧中位数负斜率的重要混杂；但整体绝对距离偏浅仍存在。
- 如果场景大部分没有符合门禁的同一表面点：本轮**不能判断**米制多视角响应，应报告遮挡/分割/标定一致性和视角覆盖，检查数据条件再决定下一轮。

## 5. 117服务器命令

使用独立worktree，不创建新的远端分支，不在已有 `selection_audit_20261116` 或 `frozen_depth_response_20261116` 写结果：

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/quality-strata-conflict-supervision
git worktree add --detach /home/robotarm/EconomicGrasp-depth-followup \
    origin/exp/quality-strata-conflict-supervision
cd /home/robotarm/EconomicGrasp-depth-followup
git status --short
find .. -name AGENTS.md -print

export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/gc6d_depth_followup_20261117
export BOUND_CONFIG=/ABS/accepted_mix50_GVAR_e15_bound.json
export GRASPNET_ROOT=/ABS/GraspNet
export GNTRANS_ROOT=/ABS/GN-Trans_RGB
export GC6D_ROOT=/ABS/GraspClutter6D
export GC6D_SCENES=/ABS/accepted_exact_78_D435_scene_ids.json
export DEVICE=cuda:0

bash scripts/depth_followup/run.sh check
```

**两个单独smoke目录：**
```bash
bash scripts/depth_followup/run.sh p1-cache-smoke
bash scripts/depth_followup/run.sh p1-track-smoke

SMOKE_FRAMES=2 bash scripts/depth_followup/run.sh p0-original
SMOKE_FRAMES=2 bash scripts/depth_followup/run.sh p0-gntrans
SMOKE_FRAMES=2 bash scripts/depth_followup/run.sh p0-gc6d
```

**优先缓存GC6D，再复用以节省GPU时间：**
```bash
bash scripts/depth_followup/run.sh p1-cache
# 任一中断后，校验同checkpoint/source/code SHA并复用已完成的帧
RESUME=1 bash scripts/depth_followup/run.sh p1-cache

# P0每域独立数据，原生GraspNet/GNTrans样本必须与GC6D协议完全记录
bash scripts/depth_followup/run.sh p0-original
bash scripts/depth_followup/run.sh p0-gntrans
REUSE_P1_CACHE=1 bash scripts/depth_followup/run.sh p0-gc6d
bash scripts/depth_followup/run.sh p0-analyze

# P1，同scene3个固定锚视角→其他view
bash scripts/depth_followup/run.sh p1-track
RESUME=1 bash scripts/depth_followup/run.sh p1-track
bash scripts/depth_followup/run.sh p1-analyze
```

## 6. 产物与报告

- `p0_original/per_frame.csv`、`p0_gntrans/per_frame.csv`、`p0_gc6d/per_frame.csv`：包含每帧原始cropped K/源frame SHA/参考距离；
- `p0_report/summary.json`、`domain_summary.csv`、`paired_original_gntrans.csv`、`REPORT_CN.md`：模型冻结，训练距离/crop条件描述和domain support gap；
- `p1_cache/frames/scene_XXXXXX/img_XXXXXX.npz` 和来源json：FLOAT32原生预测深度、RGB/depth/seg/camera/scene_gt实际SHA；
- `p1_tracks/scene_XXXXXX/matches.csv`、`audit.json`：同一世界表面点的source→target配对和visibility gate；
- `p1_report/per_scene.csv`、`visibility_counts.csv`、`summary.json`、`REPORT_CN.md`：每场景真实/预测 \(\Delta z\)、through-origin斜率和CI，不制造缺失结果。

注意两个缓存规模均有成本：P0 原生800训练域+GC6D1014帧的depth forward（GC6D可复用P1）；P1保存1014个float32 448图。不要向用户承诺后台运行／固定完成时间。预先确认显存、存储和已有任务进程；不得为本工作擅自kill其他训练。

## 7. 真正的验收门槛

- [ ] HEAD严格同原分支，main不变；
- [ ] CPU unittest、Python compileall、bash -n在117实际PASS；源码静态检查不等于测试PASS；
- [ ] 使用用户已经接受的78个scene ID，不根据过去AP/可见性挑场景；
- [ ] 缓存完整78×13、Original/GNTrans各100×4、scene/anno身份无重复；所有源文件SHA、base模型SHA、Kcrop/C2W满足合同；
- [ ] 不因rendered GT缺失造假或依赖不存在文件；GNTrans虚拟深度与真实传感器区别清楚；
- [ ] scene_gt反投影世界位置一致性通过，否则不得声称C2W正确；
- [ ] P1遮挡门禁、seg numeric相等、3D反投影误差通过，回归不读GT选择样本以优化结果；
- [ ] CI以场景为单位，报告场景数量/有效track数和对应数量；
- [ ] 不用P0 Original/GC6D直接对齐假装同相机/同物体；K/crop统计是描述性证据，不是因果实验；
- [ ] 本轮**不训练、没有新AP、没有depth scale fitting或camera embedding**；
- [ ] 原开发GC6D78场景已被反复观察，报告探索性，不宣称独立泛化。

## 8. 2026-11-17：A深度局部形状Oracle与B物距×crop受控实验

上一轮P0训练域与GC6D支持域审计、P1同一3D表面追踪已完成：Original训练域米制深度约0.41m且误差低；GC6D深度严重偏浅，78场景同点响应斜率近0。

继续相同分支，新的两个实验已实现：

- **A**：复用已验收GC6D P1 depth cache，在每帧独立的空间象限用传感器参考拟合scale/shift/affine，严格留出其他象限评价局部几何，不把oracle参数用于部署。
- **B**：Blender Cycles按预注册真实世界相机平移距离渲染相同CAD，固定相机K/姿态、物体/材质/光源；在同源RGB上交叉比较fixed_full/fixed_roi/adaptive_workspace crop。使用相同CAD世界表面点比较Δz响应和distance×crop差分交互，禁止用数字zoom代替物理相机运动。

详细工作单：**[GC6D_DEPTH_ORACLE_DISTANCE_CROP_FACTORIAL_CODEX.md](GC6D_DEPTH_ORACLE_DISTANCE_CROP_FACTORIAL_CODEX.md)**。

统一入口：`scripts/depth_factorial/run.sh check|a-smoke|a-run|b-render|b-infer|b-analyze|b-aggregate`。

状态：代码与单元测试文件已写入GitHub，真实服务器CPU/Blender/CUDA验收及新数值结果未在此聊天执行；原有AP、P0/P1验收产物不动。
