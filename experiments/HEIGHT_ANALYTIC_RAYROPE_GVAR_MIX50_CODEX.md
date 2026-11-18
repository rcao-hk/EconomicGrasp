# Codex 工作单：从 main 改为无 FiLM Height-to-Depth 的 RayRoPE+GVAR MIX50

## 0. 目标、出处和完成边界

- **Branch**: `exp/rayrope-gvar-height-analytic-mix50`。
- **唯一父分支**: `main`，基座 SHA `7a956d37aaa427a32e598f21fdc6b910ed3c2704`。
- **训练**: GraspNet-1B (RealSense) + GN-Trans RGB，每个域各取训练帧的50%；1024 image-FPS seeds + Point RayRoPE + GVAR36 + 原有CDF/width heads。使用现有 paired mixed trainer，非 GC6D triple-mix。
- **测试**: 100% **原始 GraspNet** test_seen/test_similar/test_novel；30 scene ×256 frame=7680帧/组，总计23040帧。不得把 `sample_interval=0.1` 的780帧/组称为全量推断。
- **替换部分仅为 metric-depth head**：无 global Pose-Aware FiLM、无 Ray/Gravity FiLM、无 PlaneFiLM；DPT直接预测参考桌面法向的高度，利用外部表面标定平面与crop后的K解析恢复camera-z。
- **强限制**：网络实际输入为 RGB + K + 外部标定相机到桌面位姿；**不是无几何先验的单目RGB**。预处理仍依赖历史sensor depth、seg、虚拟深度标签等；`COLLISION_THRESH>0`还会调用sensor cloud。
- 此工作单准备代码、严格协议、测试和操作命令；**并未在117/119上训练或完成官方AP**。后续Codex需在原机按阶段执行、审计、保存报告。不能把此前受控CAD上T2的79% MAE下降当作GraspNet的预期或已实现提升。

## 1. 科学假设与实现

### 假设

历史 `global_film` 只输入相机观察方向，不显式使用米制桌面高度。已标定桌面场景的物理工作距离可由 `K` 和 `T_table_from_cam` 解析给定；网络应主要估计与物体表面相关的未知高度 `h`，而非重新推断整段绝对相机距离。该假设限于有可靠桌面几何、视线与平面不近似平行的场景。

### 确切数据流

```text
RGB 448² ── frozen DINOv2 ViT-B ── DPT raw logits ── sigmoid ── h(u,v)
                                                        │
K_crop ────────────────────────────── ray r=K^-1[u,v,1]│
T_table_from_cam ──────────────────── plane (n,b)       │
                                                        ▼
                                  z(u,v) = (h(u,v)-b) / (n · r)
                                                        │
                        existing metric depth / proposal / RayRoPE / GVAR / CDF
```

**不实例化任何FiLM层**：`DINOv2DepthRegressionNet.pose_aware_adapter is None`，不引入`PlaneFiLM`、`RayGravityDenseDPTFiLM`或替代性的MLP条件调制。旧pose vectors仍可由数据集返回供历史兼容，但height模式不读取/使用它们。不得把height解码又偷偷移入FiLM。

**平面参数来源**：

```python
T_table_from_cam = cam0_wrt_table @ camera_poses[anno]
n = T_table_from_cam[2, :3]   # unit table-up normal in camera axes
b = T_table_from_cam[2, 3]    # table origin z in metres
```

数据集在每个原始GraspNet与GNTrans配对帧返回 `camera_plane` [4]。两域共享GraspNet相机外参，GNTrans仅改变RGB外观及对应深度监督。不能使用该帧的GT深度拟合平面；需要在真实数据确认table z=0确实是物理支撑面。

**高度和深度**：

```text
h_min=-0.05 m; h_max=+0.35 m; h = h_min+(h_max-h_min)*sigmoid(raw)
r_z=1, z=(h-b)/(n^T r)
```

DPT最后输出卷积置零初始化，使初始h=0.15m，与先前受控T2的初始条件相同。没有继承之前Pose-FiLM权重；DINO仍按main从 `checkpoints/depth_anything_v2_vitb.pth` 加载预训练backbone，DPT head及全部抓取模块按新模型正常训练。模型若加载旧depth checkpoint，其signature和协议必须失败，禁止非严格降级。

**无效条件**：只在 `n^T r < -0.1` 时解析z；其他视线z=0并记录 `height_ray_valid_mask`，使深度损失剔除该射线，已有seed选择的depth-valid规则剔除这些像素。对于满足角度但h过大导致负z的情况，不clamp到有效深度：保留回归梯度，seed选择自动滤除。应记录全帧eligible比例、非正/越界z比例和裁剪差异。此方案在俯视桌面外可能不适用。

## 2. 相关文件

| 文件 | 变化 |
|---|---|
| `height_analytic_geometry.py` | 无参数的ray/plane解析、米制单位和姿态有效性门禁 |
| `dataset/graspnet_dataset.py` | `camera_plane`返回，原创与GNTrans同源的相机到桌面位姿 |
| `models/economicgrasp_depth.py` | `height_analytic`模式；无FiLM；height logits → camera-z；persistent signature |
| `models/economicgrasp_bip3d.py` | 在RayRoPE+GVAR模型forward接入解析深度，记录高度/可解析mask |
| `models/loss_economicgrasp_depth_kview_transformer.py` | 深度监督与解析射线有效域交集，不让无定义ray进入回归loss |
| `utils/arguments.py` | height配置flag及模式选项 |
| `train_cva_ddp.py`, `train_cva_gntrans_mix_ddp.py` | height参数、严格checkpoint resume与协议元数据 |
| `inference_cva.py` | strict protocol校验；matching height参数；保持原有抓取解码 |
| `run_gntrans_mix_cva_cdf_{train,eval}.sh` | 仅height模式透传参数，其他mode保持原行为 |
| `tools/preflight_height_rayrope_gvar.py` | 只读位姿、米制、frame覆盖和train/test场景检查 |
| `scripts/run_height_rayrope_gvar_mix50_train.sh` | 一键启动前置检查＋MIX50训练 |
| `scripts/run_height_rayrope_gvar_full_graspnet_infer.sh` | 检查ckpt后执行GraspNet三组全量推断，核对23040文件 |

## 3. 源码/环境准备

```bash
git fetch origin main exp/rayrope-gvar-height-analytic-mix50
git worktree add /home/robotarm/EconomicGrasp-height-analytic \
  -b codex/execute-height-mix50 origin/exp/rayrope-gvar-height-analytic-mix50
cd /home/robotarm/EconomicGrasp-height-analytic

# 在既有grasp环境中执行；需MinkowskiEngine、PointNet2、CUDA、GraspNetAPI、DINO。
python -m unittest discover -s tests -p 'test_height_analytic_geometry.py' -v
python -m compileall -q height_analytic_geometry.py models/economicgrasp_depth.py \
    models/economicgrasp_bip3d.py dataset/graspnet_dataset.py train_cva_ddp.py \
    train_cva_gntrans_mix_ddp.py inference_cva.py
bash -n scripts/run_height_rayrope_gvar_mix50_train.sh
bash -n scripts/run_height_rayrope_gvar_full_graspnet_infer.sh
```

必须准备：`DATASET_ROOT`中的GraspNet realsense、`tsdf_depth`、`virtual_scenes`与CDF标签缓存；`GNTRANS_RGB_ROOT`的对应材质扰动RGB；历史DINO backbone文件（不是历史pose-aware full grasp checkpoint）。若任何文件缺失应停止，不得自动切到GT-depth/CDF老版本。

## 4. 数据／相机几何 preflight（训练前强制）

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/height_rayrope_gvar_mix50_seed0
python tools/preflight_height_rayrope_gvar.py \
 --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
 --height_min_downward_dot 0.1 --check_rgb_files \
 --output "$OUTPUT_ROOT/preflight_height.json"
```

检查100训练scene与90测试scene的cam0/table变换、256视角、刚体旋转、b是否在米制合理范围、主光线方向；training有12800配对frame，每个test split共7680帧。历史 `check_gntrans_mix_data.py --fraction .5 --load_one` 再检查pair RGB/depth/label路径和数据结构。**这只是几何必要条件**；Codex还应选择每个域至少10个真实样本可视化`n,b`与传感器/渲染表面，验证reference table z=0的方向和原点。不能单凭n/b数值范围宣称真实桌面高度已认证。

建议对单个B=1训练和B=1推断真实GPU smoke强制验证：
- `model.depth_net.pose_aware_adapter is None`；state_dict没有任何 `pose_aware_adapter.*`。
- height解析与外部直接numpy/torch double precision结果一致；该帧crop后 `K_crop` 与原图共同射线一致。
- 模型参数在depth supervision下确实有梯度；不应从RayRoPE/GVAR把numeric depth错误地反传（沿用main的detach策略）。
- `camera_plane`严格来自本帧相机外参，混batch/graspnet GNTrans不交叉；特殊倾角或min-depth范围的候选数量可监控。
- 尝试注入错误单位(×1000)、错误变换方向、n dot r≈0，必须明确失败或标记invalid，不允许得到貌似合理的正深度。
- 高度输出范围在训练GT上的越界率；若物体过高需仅基于train/dev确定新的固定范围，再正式训练，不能根据test AP调整。

## 5. MIX50 正式训练

MIX50严格定义为每个训练scene固定取frame `0,2,...,254`。100个train scene，每源100×128=12800；总计25600帧/epoch。不同RGB域属于配对场景，不是200个独立物理scene。GNTrans使用rendered GT物体深度/virtual graspness，以及main既有fused-background监督策略；不把GNTrans virtual depth输入预测网络。

```bash
cd /home/robotarm/EconomicGrasp-height-analytic
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/height_rayrope_gvar_mix50_seed0
export GPUS=3,4,5
export BATCH_SIZE=4 MAX_EPOCH=20 LR=0.0001 SEED=0
export NUM_WORKERS=4 EVAL_NUM_WORKERS=3 EVAL_START_EPOCH=10
export HEIGHT_MIN_M=-0.05 HEIGHT_MAX_M=0.35 HEIGHT_MIN_DOWNWARD_DOT=0.1
bash scripts/run_height_rayrope_gvar_mix50_train.sh
```

注意：一键脚本会创建`preflight_height.json`，不能先手工在相同路径创建此文件然后再次启动脚本；如手动完成preflight，另设新的训练输出目录。脚本遇现有checkpoint和preflight文件停止，防止覆盖；需要resume时在审计旧结果后另行明确实现，不要临时删除检查。

核对：log显示train_fraction_each_domain=0.5、original_train_count=12800、gntrans_train_count=12800、mixed_train_count=25600；模型设计`enable_point_rayrope=true`和`enable_gvar=true`；`use_cdf=true`、`kview_mode=A1`、`use_obs_depth=false`、`pose_depth_mode=height_analytic`。训练各项损失、depth MAE、height rays eligible比例、负z与候选数量、GNTrans与原域dev分项结果均保存；不擅自调整训练预算以迎合AP。

## 6. 100% GraspNet 推断

训练得到 `$OUTPUT_ROOT/checkpoint.tar`；不要用e15 T2受控CAD权重或旧global-FiLM full grasp checkpoint替代。`HEIGHT_*`值必须和训练时完全相同。新推断输出目录必须为空。

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export CKPT=/data/robotarm/result/grasp/rgbgrasp/height_rayrope_gvar_mix50_seed0/checkpoint.tar
export OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/height_rayrope_gvar_fullgn_seed0
export GPUS=3,4,5
export HEIGHT_MIN_M=-0.05 HEIGHT_MAX_M=0.35 HEIGHT_MIN_DOWNWARD_DOT=0.1

# 严格不调用sensor点云碰撞后过滤，先判断RGB+标定的纯模型候选覆盖。
export COLLISION_THRESH=0 ENABLE_EVAL=0
bash scripts/run_height_rayrope_gvar_full_graspnet_infer.sh
```

固定 `DATASETS=original`，三个split `test_seen,test_similar,test_novel`，`SAMPLE_INTERVAL=1.0`、每组`7680`张预测npy，合计23040张。原有`run_gntrans_mix_cva_cdf_eval.sh`可设置`ENABLE_INFERENCE=0 ENABLE_EVAL=1`对已有预测跑官方AP，或者让新推断脚本`ENABLE_EVAL=1`一次运行。要模拟旧GraspNet standard collision后处理，可在**新的独立输出目录**设置`COLLISION_THRESH=0.01`，明确该配置额外使用sensor point cloud；不得将这两种AP混为同条件结果。

**推断不向网络输入深度标签**，但现有`GraspNetMultiDataset.get_data`会读取sensor depth、seg/virtual GT，以确定workspace/crop和检查标签等；模型也显式使用相机—桌面标定，因此不要称为端到端`RGB-only / no geometry input`。需要完整传感器无关部署，后续还须替换crop/workspace逻辑与sensor collision。

## 7. 验收、正式对比与结果报告（Codex需执行）

1. 先只读检查 `git status`、main父提交；验证所有diff仅在新分支。保存源码SHA、数据路径签名、CUDA/torch版本、batch/seed/worker、checkpoint SHA，确保无旧实验覆盖。
2. CPU单测+GPU真实batch=1、batch=4训练前向反向、GVAR数值detach、严格模式check；DDP三卡2步smoke后才正式训练。请先审计`height_ray_valid_mask`的覆盖和loss是否可能偏向少量可解射线。
3. 正式MIX50 20epoch，不允许偷偷使用PoseFiLM或额外PlaneFiLM；记录每轮train域配比、损失、优化step及depth/height统计；从头训练和旧checkpoint warm start不得混比。
4. 正式GraspNet三split 100%推断，检查`scene_xxxx/realsense/0000...0255.npy`数量、finite值、空候选比例，保存sensor collision 关闭的先验候选结果；再视需求单独计算official AP。禁止使用test标签选checkpoint。
5. 后续与原global_film RayRoPE+GVAR做**同数据、同训练更新数、同预处理、同推断率与碰撞配置**的对照。除AP外，报告全量center-z bias/MAE分距离、candidate coverage、frame-empty、Top1/Top10 collision、GraspNet AP（seen/similar/novel）与内存/吞吐。
6. 明确“平面先验是否被真实提供”及单位/坐标来源；T2受控CAD结果不代表此次实验已改进，真实GraspNet上的最终结论依赖本分支后续训练和官方评测。出现table几何与depth参考不一致、h范围严重越界、无效射线过多时，先停止诊断，不使用oracle depth伪造标定。

## 8. 进一步限制

- 高度相对一个已标定水平桌面，不支持水平视线、非平面支撑或没有参考标定的任意书柜抓取（这些情况当前显式标记不可解析或需要不同表示）。
- 相机与桌面位置提供了额外米制信息；与未提供此先验的Pose-Aware DPT相比，数字提升不能被解释为纯模型结构的公平等信息比较。
- 本次脚本**未在用户服务器上执行**；本地CPU测试也不能代替实际MinkowskiEngine、GraspNet数据、DINO权重与CUDA运行。Codex需要按上面验收流程生成正式运行报告。
