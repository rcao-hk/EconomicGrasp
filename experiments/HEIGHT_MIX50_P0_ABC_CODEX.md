# Height Analytic MIX50：P0-A/B/C 深度泛化诊断 Codex 工作单

## 0. 目标与分支

**分支**：`exp/height-mix50-p0-generalization`，从 GitHub 上 `exp/rayrope-gvar-height-analytic-mix50` 的 `1d6c90ccfe528fd0dedac3748db2d802c6df58b5` 派生。

**重要版本差异**：2026-11-22汇总报告中，正式训练使用的117本地源码为 `0a3d2accf6ad5154362c7ce347d2f5ae9a754f94`，该提交未在GitHub现有Height远端分支上出现。本工具完全不修改模型/数据集历史生产路径；读取本地checkpoint并严格加载Depth子模型。如果本地修正版的 `depth_net` 权重结构与已推送的远端Height代码不一致，必须停止、核查并同步实际117代码，**禁止使用strict=False补丁蒙混过关**。

本轮只读复用两个模型：

- Height: `/data/robotarm/result/grasp/rgbgrasp/height_analytic_mix50_rendergn_lr3e4_wd1e3_seed0_20261119/train/checkpoint_15.tar` (e15 / 完成16轮，报告SHA256 `9679619c...`)。
- 原Pose-FiLM Point RayRoPE+GVAR36 MIX50: `/data/robotarm/result/grasp/rgbgrasp/log/point_gvar36_50pct_20261016/train/checkpoint_latest.pt` (e15，报告SHA256 `2c7a4097...`)。

文件路径必须以实际117/119挂载为准。只注册真实存在的checkpoint，`epoch`是**完成轮数**，因此index15写16。报告中的历史checkpoint原训练协议不可猜测：工具读取`pose_depth_mode`、`grasp_design`及`gntrans_mix_protocol`，并对DINO/DPT/FiLM深度子树 strict=True。两种模型都要求每域12800帧MIX50及Point RayRoPE+GVAR。不做恢复训练，不执行抓取头，不能产生新Grasp AP。

## 1. 三个实验与需要成立的推论

### P0-A 跨epoch、跨split深度泛化曲线

固定每scene的场景和相机帧索引，所有模型、所有checkpoint使用**同一RGB裁剪、内参、桌面平面、GT及共同前景mask**：

| 评估划分 | 物理scene | 默认每scene 8帧 | 含义 |
|---|---:|---:|---|
| `train_used` | scene0000–0099 | 偶数帧采样 | 实际MIX50训练时见过的frame |
| `train_heldout_view` | scene0000–0099 | 奇数帧采样 | 相同训练物体的未用观察视角 |
| `test_seen` | scene0100–0129 | 固定等间隔采样 | 官方Seen物体 |
| `test_similar` | scene0130–0159 | 固定等间隔采样 | 官方Similar物体 |
| `test_novel` | scene0160–0189 | 固定等间隔采样 | 官方Novel物体 |

默认 `frames_per_scene=8` 总计 1840 个物理帧/model；开启配对外观会计算另外1840张GNTrans RGB。场景按时间帧固定等间隔采样； train used恰好在偶数128帧中取8帧，odd holdout同索引范围。可调整`FRAMES_PER_SCENE`，但同一次正式实验必须锁定，不可因结果调整。测试split-only允许256帧；含train的组合最多128帧。

记录每checkpoint×split×RGB域的Depth MAE/P95、Height MAE、高度方差与泛化差距，并输出同完成epoch的模型**逐帧配对差值**及**按物理scene聚类的95% bootstrap区间**。不可把几千帧当成几千个独立物体。训练/Similar误差随epoch反向变化才支持“随训练出现过拟合”；只有e16两个模型时只是一个cross-section，不是训练曲线。旧报告e16之外有保留哪些checkpoint必须到原训练目录核查；绝不伪造或插值缺失epoch。

### P0-B Height空间分解与常数高度基线

对两个模型统一转到表面相对桌面高度：

```text
h_GT = (n · K_crop^-1[u,v,1]) * z_GT + b
h_Pose = (n · ray) * z_Pose + b
h_Height = Height model原生h
```

模型共同前景mask：GT rendered depth在[0.2,1]m、seg>0，且由标定独立确定 `n·ray < -0.1`。**不得用模型预测valid与否决定评估像素**。每个frame计算Depth/Height MAE、signed bias、abs-P95、高度GT/预测方差、GT Height范围越界率、Height模型输出近端点2%比例；对旧Pose模型不报告“sigmoid高度饱和”。

物体分组为 `(split, scene_id, seg instance label)`，输出真实/预测h的平均、方差、signed/abs误差、基于同一实例多个观察帧聚合的Height-R²。不要直接对每帧R²进行未加权平均，薄平面真实方差极小可能使R²不稳定；记录此时`null`。label为GraspNet原标识，不冒充已核验的物体类别或材质。

**常数高度h0**必须通过单独`fit-prior`阶段，用`train_used`原GraspNet已选偶数帧的虚拟前景GT计算：默认100 scenes×16帧，每帧最多2048条确定性采样有效射线，汇总为h中位数。两个架构共用此h0；不可使用test_seen/similar/novel/odd holdout的深度计算h0。然后`z_const=(h0-b)/(n·ray)`，比较model与const的同像素MAE，区分几何锚点本身的收益与可迁移视觉Height信息。

必查scene149/seg13、scene151/seg43（高度偏差方向相反），以及提升的scene134、142。若Height在Similar上的局部高度预测没有超过常数基线，不足以支持学习到跨物体表面形状。

### P0-C 配对RGB外观稳定性

每个RS原始RGB与对应GNTrans RGB来自完全相同`(scene,frame)`；**只使用RS原始传感器点云/seg生成一次workspace/crop/K/plane**。用同一crop box和同一`torchvision` Resize+Normalize变换处理两张RGB，对应共同mask和GT也只生成一次。GNTrans RGB不能自行根据rendered depth裁剪，否则“外观变化”与crop/geometry同时变动。

两张图都用同一冻结checkpoint前向，逐frame报告：

- `mean |z_RS-z_GNTrans|`、`mean |h_RS-h_GNTrans|`；
- GNTrans-RS参考Depth MAE差，及GNTrans更差的像素比例；
- 输入RGB预处理后的平均L1差异；Scene-balanced 95% CI；
- 可按Seen/Similar/Novel分析外观扰动对几何泛化的影响。

GN-Trans不仅可能改变表面材质，还可能有光照、背景等渲染差异，因此不能直接将结果称作纹理的严格因果效应。配对预测一致但两者都错，也不是良好泛化。

## 2. 代码文件

| 路径 | 职责 |
|---|---|
| `diagnostics/height_p0/metrics.py` | 固定共同mask、深度/高度误差、实例池化、场景聚类CI与epoch匹配 |
| `diagnostics/height_p0/data.py` | 按GraspNet主干复刻mask→workspace→crop→Resize→K；配对RGB强制共享几何 |
| `diagnostics/height_p0/checkpoints.py` | full checkpoint协议检查；仅load DPT深度子树，strict=True |
| `tools/height_p0_abc.py` | fit-prior / evaluate / summarize；JSONL部分续跑、CSV/JSON报告 |
| `tests/test_height_p0_abc.py` | CPU协议测试，不依赖真实抓取模型 |
| `scripts/run_height_p0_abc.sh` | 无训练的分阶段入口 |

为什么复刻数据前处理而不是直接调用`dataset.get_data`？原数据集会构造随机采样20k点及D×448²概率监督张量，本次只需要RGB、crop后K、相机平面、rendered前景GT。新的诊断数据路径仍调用原仓库的`CameraInfo`、`create_point_cloud_from_depth_image`、`get_workspace_mask`、`_crop_box_from_mask`、`resize_intrinsics_with_crop`及`img_transforms`，必须通过真实batch比较原`get_data`在相同frame得到的`img/K/crop_box/gt_depth_m`，最大容差明确记录。仅当前景mask共同有效，才允许比较两个模型。Preprocessing仍依赖sensor depth和seg；这不是原生无传感器的RGB部署实验。

## 3. 使用方法

1. 在117或有同样数据路径的服务器中检出此分支，安装原EconomicGrasp的`grasp`环境；checkpoint须为**可信本地文件**（`torch.load(weights_only=False)`）。

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/height-mix50-p0-generalization
git worktree add /home/robotarm/EconomicGrasp-height-p0 \
  -b codex/run-height-p0 origin/exp/height-mix50-p0-generalization
cd /home/robotarm/EconomicGrasp-height-p0
```

使用独立worktree避免影响此前Height修正版的未提交文件；不要把`0a3d2acc`未知改动默默合并。
2. 将`configs/height_p0_checkpoints.example.json`复制到结果目录，填入真实存在的Height和旧Pose GVAR36 checkpoint，补齐所有存在的epoch。没有保存的epoch从JSON中省略但保留`expected_epochs`，让工具显式报告缺失。
3. CPU单测：

```bash
cd /home/robotarm/EconomicGrasp-height-p0
bash scripts/run_height_p0_abc.sh check
```

4. P0-B先拟合仅训练帧上的常数高度：

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/height_p0_abc_seed0
export CHECKPOINTS=$RUN_ROOT/checkpoints.json
export DEVICE=cuda:0
export BATCH_SIZE=2 WORKERS=2 FRAMES_PER_SCENE=8
bash scripts/run_height_p0_abc.sh fit
```

5. 正式执行A/B/C，运行期间可安全中断；`RESUME=1`会核对checkpoint SHA、数据路径、代码源码SHA、prior SHA与采样协议后才续跑。不要手工删除一半JSONL记录以制造“完整”。

```bash
APPEARANCE=1 bash scripts/run_height_p0_abc.sh eval
# 只有明确已存在严格匹配的同一eval run时：
RESUME=1 APPEARANCE=1 bash scripts/run_height_p0_abc.sh eval
bash scripts/run_height_p0_abc.sh summarize
```

诊断单独选择Similar和快速smoke示例：

```bash
# 使用**新的**RUN_ROOT以免把smoke与正式数据合并
MAX_SCENES=2 SPLITS="test_similar" FRAMES_PER_SCENE=4 \
  PRIOR=/ABS/formal_train_only/prior_train_only.json \
  APPEARANCE=1 bash scripts/run_height_p0_abc.sh eval
```

`MAX_SCENES`仅用于smoke，并在manifest中标记，**不能当作正式泛化证据**。

## 4. 验收与判读

报告位置：`$RUN_ROOT/evaluation/REPORT_ZH.md`，各数表为：

- `p0a_epoch_curves.csv`：每epoch×split×原/GN域的Depth/Height误差曲线及CI；
- `p0a_matched_epoch_differences.csv`：只在相同epoch、scene、frame、domain计算Height-Pose配对差；
- `p0a_generalization_gaps.csv`：Train到各测试划分的scene-balanced误差gap及独立场景bootstrap CI；
- `p0a_paired_gap_differences.csv`：同epoch逐场景Height-Pose配对差的Test-Train差异；
- `p0b_per_object.csv`：逐物体高度偏差、GT/pred高度标准差和R²；
- `p0c_per_pair.csv`：同RGB物理场景不同外观下的预测变化；
- `p0_abc_summary.json`：完整统计、缺失epoch、工作进度；
- `evaluations.jsonl`：可审计的逐帧结果；`manifest.json`锁定参数与源码。

严格验收：默认每个split先在3个不同位置与原生产`GraspNetMultiDataset.get_data`对照RGB、K、rendered GT、camera_plane、camera_pose_vec，最大差<=1e-5（结果在`preprocess_parity.json`）；所有已注册模型所有指定帧必须成功、不同模型共同mask像素数相同、每个已完成epoch包含所有声明split的帧、无任何NaN或加载失败、相同epoch才求跨模型delta、每个物理scene为bootstrap单元。**不要仅从总MAE得出过拟合结论**：还应看Train–Similar gap随epoch变化、旧/新对照、Height范围饱和、常数基线表现以及paired RGB变化。只有e16时应明确`P0-A learning-trajectory NOT_AVAILABLE`。

最后，P0-A/B/C不能直接量化Similar AP下降2.77pp是否由Depth造成；仍需要之前独立规划的**固定动作中心深度反事实AP**，本分支没有触碰正式抓取模型/官方evaluator。

## 5. 当前执行状态

本次GitHub变更只提供实验脚本、文档及CPU单元测试。用户117的实际checkpoint及GraspNet数据未挂载在执行环境，**没有运行GPU深度推断或正式P0-A/B/C实验**。在原机运行后，应附带保存脚本版本、Python/torch/CUDA版本、checkpoint SHA及完整manifest、表格与失败现场。
