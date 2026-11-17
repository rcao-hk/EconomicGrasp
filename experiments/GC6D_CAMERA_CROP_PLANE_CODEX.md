# GC6D：相机／裁剪米制一致性与标定桌面参数化

新分支：`exp/gc6d-camera-crop-plane`。
父提交：`beaf89d33a9a9d56857d1f8466cf4522ceb388d2`（原 `exp/shared-visible-geometry-multibatch`）。

**范围：前两个方法优先级的实验实现，不是重复旧ROI/Multi200–600，也不是P6-2A动作bank实验。**
代码只新增独立 `camera_geometry` 包，不修改生产抓取网络、旧数据缓存、P0–P6结果或官方evaluator。

## 1. 两个问题与七个实验臂

### C：相机条件 × 跨裁剪米制一致性

| Arm | 条件路径 | 相同两crop的米制监督 | 额外同射线XYZ一致性 |
|---|---|---|---|
| C0 | 历史global FiLM（冻结） | 是 | 无 |
| C1 | 相同global + 可训练RayGravityDenseDPTFiLM | 是 | 无 |
| C2 | 同C0 | 是 | 有 |
| C3 | 同C1 | 是 | 有 |

**初始化细节很重要：这不是直接把已经训练好的global FiLM替换成随机ray FiLM。**
四组严格复用同一historical depthnet（DINO+DPT）及global FiLM权重；global FiLM冻结。
C1/C3把仓库已有RayGravityDenseDPTFiLM作为恒等初始化的增量接在global FiLM之后。
因此C0/C1初始预测必须在同一个真实输入上满足最大差≤1e-6 m；`preflight`检查这一点。
C1/C3新增参数量会单独记录，不能把结果写成严格参数量匹配的比较，也不能叫“pure ray_gravity_film替换”。

所有组更新DPT；只在对应组更新新相机adapter。DINO始终冻结且eval；不建立grasp graph，
不会把抓取loss梯度送入depth。两组都看同样的两张crop、同样256条原图射线、同样监督标签，
所以consistency-on不额外获得更多训练图像或监督点。

对同一原图射线r（r_z=1）上的两个独立预测z_A/z_B：

    L = 0.5*(|z_A-z_gt| + |z_B-z_gt|)/0.01m
        + lambda*|| (z_A-z_B)*r ||_2/0.01m
        + negative-depth penalty

逐图求均值，再按实际图像batch平均。lambda在C0/C1为0，在C2/C3为配置值（默认1）。
两个crop均保留梯度，原有米制监督不可删除，防止“一致但一起错”的塌缩。
主对照：C1−C0、C2−C0、C3−C1、C3−C2；另输出2×2 interaction。

### T：相同平面信息下，camera-z还是surface-height

| Arm | 平面adapter输入 | 标量输出解码 |
|---|---|---|
| T0 | 全零（相同容量控制） | z = sigmoid(raw)*max_depth |
| T1 | 已标定n,b | 同T0 |
| T2 | 完全相同n,b | h → z=(h-b)/(n·r) |

三组都使用相同global+ray结构，并含完全相同的PlaneFiLM模块；T0只屏蔽其平面输入。
所有T组共同重置DPT最后标量卷积为零权重/零bias；不能将历史z logit直接解释成h。
三组共享其他初始化和训练预算，使用同一个解码后z监督，不为T2额外增加height loss。
T0/T1的零输出层对应初始z=max_depth/2；T2对应初始h=(h_min+h_max)/2。
**米制初始预测并不相同，因此保存initial_dev.json和final_dev.json：不能隐瞒已知平面带来的epoch0优势。**
T1−T0测额外米制平面信息；T2−T1测相同信息下的参数化。C组与T组不是同初始化的跨组对照。

T_table_from_cam将相机米制坐标映射到真实桌面坐标，桌面z=0：

    n = T[2,:3]; b = T[2,3]; h=n·X+b; X=z*r.

仅接收外部标定或renderer已知平面，不允许从当前GT depth拟合平面后冒充RGB-only。
原GC6D相机JSON的world frame不自动等于table frame。手眼标定也不自动给出桌面位置。
对n·r >= -0.1的掠射/背向射线，三个T组共同使用相同的输入决定的mask；报告eligible coverage。
若可用射线上的预测z为负，保留其误差并施加惩罚，不按预测好坏丢弃样本。
同时报告GT z/height超出预先固定输出范围的比例；不得看holdout后选择范围。

## 2. 文件与依赖

- `camera_geometry/geometry.py`：半像素crop/resize、完整K、平面公式、米制loss。
- `data.py`：签名manifest、分组划分、真实GC6D导入、双crop共同射线。
- `model.py`：复用仓库DINO/DPT和FiLM；不加载MinkowskiEngine抓取图。
- `render.py`：CPU CAD z-buffer、真实相机位移与焦距干预、已知物理表面轨迹。
- `evaluate.py`：同射线误差、同点距离响应、按scene/CAD聚类的统计。
- `run.py`：prepare/render/import-gc6d/preflight/train/evaluate/predict/summarize。
- `scripts/camera_geometry/run.sh`：分阶段入口，不默认运行训练。
- `tests/test_camera_geometry.py`：CPU几何／协议／tiny端到端测试。

基础依赖：Python 3、numpy、Pillow、torch；生产DPT另沿用仓库已有torchvision与DINO依赖。
PLY/OBJ/STL导入需要trimesh；自带的vertices/faces NPZ网格不需要trimesh。
CPU renderer无需Open3D、显示服务器、EGL、CAD/Dex-Net evaluator或CUDA。
它使用固定颜色和确定性光照，不支持纹理/阴影，不能当作逼真的GC6D重建或实景性能证明。

## 3. 数据入口A：控制真实距离和焦距的CAD探针

复制并修改 `configs/camera_geometry/render.example.json`：登记CAD路径、明确的`scale_to_m`、
train/dev/holdout身份、是否已被开发观察。模板名字不是已经注册的新CAD！
同一个CAD文件不能改名后跨partition；不同序列化的重复CAD仍需人工核查。
网格只平移到桌面，绝不自动归一化尺寸。过大的网格显式拒绝，不自动简化。

默认训练距离0.35/0.45/0.55m，评价增加0.70/0.85m；距离定义为camera到table原点的欧氏距离，
像素监督仍是实际optical-z，不把二者混用。两种真实相机条件：

- `fixed`：K固定，真实移动相机。
- `matched_table_projection`：f随相机距离同比变化；table平面的f/d固定，但3D物体parallax仍存在。

后者不是数字zoom，也不声称不同距离的3D RGB图像完全相同。
每次真实渲染后再生成full/固定ROI/foreground-bbox workspace三种crop。
workspace使用renderer前景，因此是明确标注的特权诊断预处理，不是可部署无深度crop方案。
可见三角面中心提供跨距离相同物理点ID；评价仍检查重投影及可见depth一致性。

```bash
cd /home/robotarm/EconomicGrasp-camera-crop-plane
export SPEC=/ABS/registered_render.json
export OUTPUT=/data/robotarm/result/grasp/rgbgrasp/camera_geometry_data_v1
bash scripts/camera_geometry/run.sh render
# 输出：records.json、manifest.json、逐frame RGB/深度/前景mask。
```

## 4. 数据入口B：真实GC6D（不自动猜桌面）

复制 `configs/camera_geometry/gc6d.example.json`，填写scenes与实际来源说明。例如：

```json
"scenes": [
  {"scene": 123, "partition": "exploratory", "previously_used": true, "image_ids": [2, 26, 50]}
]
```

这里123只是格式示例，不是推荐数据划分。D435读取image_id=2,6,...,50。
`camera_transform_semantics`必须根据已有审计显式选择c2w；不要因为JSON键名w2c就静默求逆。
导入器沿用BOP传感器单位raw*depth_scale/1000；rendered reference需另外提供明确的
`rendered_depth_template`（可含{root}/{scene:06d}/{image:06d}/{camera}）和`rendered_depth_scale_m`。
缺失rendered文件直接失败，绝不回退为sensor标签。

提供已标定`T_table_from_world`和`plane_provenance`后才生成table参数（可按scene覆盖）。
未提供时只生成朝向信息，可用于C组；T组应拒绝缺失平面输入。
原生全图、固定ROI和label-bbox workspace只是本实验的显式crop policy；不是偷偷复用历史3D workspace。
若有历史精确crop，给每scene提供`crop_boxes_by_image`和全局`crop_provenance`，保留完整full/roi/workspace三项。
传感器reference、segmentation crop及碰撞后处理的使用范围不可混写为端到端RGB-only。

```bash
SPEC=/ABS/gc6d_registered.json OUTPUT=/ABS/new_native_gc6d \
  bash scripts/camera_geometry/run.sh import-gc6d
```

导入器不会用帧中位深度伪造同表面点轨迹。没有已验证的tracks时，真实GC6D距离响应标记NOT_AVAILABLE。
使用旧78场景必须标为exploratory；官方test场景不可被导入成train/dev。

## 5. 通用raw manifest（接入已有渲染/训练数据）

`prepare --spec records.json --output manifest.json`可登记其他数据。每entry需要：

- `id`唯一；`group`是共享物理scene/CAD身份；`domain`、`partition`、`previously_used`、`object_ids`。
- `rgb_path`；`depth_path`（PNG或NPY optical-z）；`depth_scale_m`；可选`mask_path`。
- `K`原图3×3；`pose_vec`和`gravity_vec`，或可靠的`T_table_from_cam`以解析生成。
- `crop_boxes`含full/roi/workspace，格式[x0,y0,x1,y1)，像素边缘坐标。
- `reference_source`、`crop_source`；有桌面时还需`plane_source=calibrated|renderer`及来源说明。
- 可选`tracks`：`[{"id":"physical-point-id","uv":[u,v],"xyz_table":[x,y,z]}]`。
- 可选`condition`与`distance_m`只用于分组报告，不进入forward。

深度无效=0，不接受NaN/负值；crop不修改目标米制深度。RGB采用与grid_sample align_corners=False
一致的半像素变换：u'=sx*(u-x0+0.5)-0.5，K'=A*K。
这不是旧脚本所有resize路径逐位一致的声明；本实验所有臂使用同一个明确的新像素约定。
所有匹配先回到原图共同射线，监督直接取原图参考值，不能拿相同crop像素索引当同一物理点。
每frame共同有效射线不足声明数量时停止，不静默drop/pad。
同一group不能跨分区；声明holdout的object IDs不能与开发/训练交叉，且previously_used必须false。
这些程序检查不能替代跨数据集CAD身份与历史使用记录的人工审计。

## 6. checkpoint预检与运行

在独立worktree执行，保留原117所有未提交修复和旧结果：

```bash
git fetch origin exp/gc6d-camera-crop-plane
git worktree add /home/robotarm/EconomicGrasp-camera-crop-plane \
  -b codex/camera-crop-plane-exec origin/exp/gc6d-camera-crop-plane
cd /home/robotarm/EconomicGrasp-camera-crop-plane
bash scripts/camera_geometry/run.sh check

python -m camera_geometry.run inspect-checkpoint --checkpoint /ABS/actual_checkpoint.pt
```

复制 `experiment.example.json`，填入实际manifest/checkpoint/encoder、顶层权重key和depth前缀。
默认key=model_state_dict、prefix=depth_net.只是示例，不能根据文件名猜实际版本。
支持顶层权重dict（checkpoint_key为空）或明确的model/state_dict等key；DDP可显式使用module.depth_net.。
DA2与global-FiLM子树逐项strict=True；缺key、额外不明key、shape不一致全部停止，不改权重或宽松加载。
C组max_depth_m必须匹配历史深度解码值；改这个值会改变初始预测尺度，不能冒充纯相机条件消融。
checkpoint使用torch.load，限于用户自己的可信文件。

```bash
export CONFIG=/ABS/locked_experiment.json
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/camera_crop_plane_seed0_v1
export DEVICE=cuda:0
bash scripts/camera_geometry/run.sh preflight
bash scripts/camera_geometry/run.sh train-c
bash scripts/camera_geometry/run.sh train-t
bash scripts/camera_geometry/run.sh summarize
```

默认600次更新、真实B=4、每图两crop×256共同射线；AdamW lr2e-5/wd1e-4、clip1。
600是先行实验预算，不承诺已经收敛。所有臂固定last，不按开发Q/AP或者holdout挑checkpoint。
配置seed1/2时使用独立config与RUN_ROOT；汇总不同seed前先各自做配对对照，不挑最优seed。
支持独立启动：`ARM=C3 ... run.sh train-one`；可由用户在不同GPU独立进程运行，不支持本轮DDP。
`RESUME=1`仅在相同数据指纹、源码、配置及初始化权重下恢复；保存模型/optimizer/采样RNG/torch/CUDA RNG。
CPU tiny断点恢复已经测试逐张量一致；CUDA grid_sample可能非确定，不承诺GPU逐位重放。
不得为了复用旧结果而放宽容差，本轮不继承旧ROI/Multi重放协议调整。

## 7. 评价、无标签推断及输出

```bash
CHECKPOINT=$RUN_ROOT/C3/last.pt MANIFEST=/ABS/manifest.json PARTITION=holdout \
  OUTPUT=$RUN_ROOT/C3/holdout_once.json bash scripts/camera_geometry/run.sh evaluate

CHECKPOINT=$RUN_ROOT/T2/last.pt MANIFEST=/ABS/manifest.json PARTITION=exploratory \
  OUTPUT=/ABS/new_depth_predictions bash scripts/camera_geometry/run.sh predict
```

predict仅打开RGB，不打开depth/mask标签文件；plane和crop作为外部元数据，其来源仍需披露。
输出z/eligible/positive/K和crop名称；负预测不静默抹成正确值。

训练产物：protocol.json、initial_dev.json、history.json、last.pt、final_dev.json、status.json。
评价包含逐frame/crop MAE/bias/XYZ误差（mm）、跨crop极差、coverage、invalid fraction、
GT输出范围越界率、height误差；按condition和真实距离标签分层。
距离响应只在经重投影+参考depth验证的同表面点上、真实z跨度≥2cm时计算；缺少可见轨迹时明确缺失。
聚类单位为scene或CAD，先域内平均；两个CAD的bootstrap不代表强统计证据。
`COMPARISON.json/.md`列出七组结果及预声明配对差值；配置、seed、训练预算或数据不同则拒绝比较。

**本轮不生成任何新抓取动作，不计算候选覆盖、固定bank Q、官方AP或机器人成功率。**
那些属于后续系统验证，原P6实现仍原样保留在父分支。

## 8. 判读顺序与限制

先看C0→C1及C0→C2是否分别改善同射线一致性和绝对距离响应，再看C3是否有互补收益。
一致性降低而绝对误差不降，不能宣称尺度恢复；只有depth均值更准而同点beta仍失效，也不能宣称几何一致性成立。
T组先检查T1−T0；T2的优势应相对于同样看见平面的T1解释，并同时报告epoch0与height误差。
平面帮助不等于任意无标定单张RGB获得可观测的米制尺度。

全部本地测试是CPU合成或tiny backend；生产DINO/DPT、真实historical checkpoint、117 CUDA、
实际CAD/GC6D完整训练未在交付环境运行。CPU通过不能替代真实preflight与人工投影核验。
