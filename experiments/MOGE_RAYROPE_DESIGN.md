# MoGe + RayRoPE：EconomicGrasp 的单视图在线适配

## 0. 代码来源与边界

目标仓库：`rcao-hk/EconomicGrasp`。
基线：`main@52d09f925059bec3643610ecf1f1722894627ee5`。
建议新分支：`exp/moge-rayrope-grasp20`。

本实现是**算法机制的单视图抓取适配**，不是把 MoGe/RayRoPE 两个完整预训练系统串接，也不复现它们原论文的训练数据或 benchmark。
原有 DAV2 frozen encoder 继续作为 foundation feature prior。没有加载额外 MoGe checkpoint，因此不能把本实验的 shape head 称为“已经拥有 MoGe 开放域重建性能的预训练几何模型”。

所有改动均为新增文件。原有 `models/economicgrasp_bip3d.py`、CVA decoder、label matcher、dataset、loss 文件不直接修改。入口 wrapper 根据 flags 替换相应子模块；flags 全关时保留原 metric-depth 与 CVA grouping。

实现包不包含 patch/apply 程序。发布状态是“可导入 main 工作树的直接源码”；当前会话没有远端写操作，远端 branch/push 尚未完成。Codex 工作文档包含在有写权限环境中创建分支、提交和验证的步骤。

## 1. 问题定义

目标不是使错误的执行 translation 无害，而是减少“唯一预测深度同时决定几何形状、证据位置以及动作执行尺度”的过强耦合。

本实现分开：

1. 学习局部 3D 形状，但允许全局 scale 与 camera-Z shift 不确定；
2. 用已知 crop-adjusted K 固定 projective gauge，用 learned metric anchor 赋予米制尺度；
3. 用预测深度的区间在 ray–query positional encoding 中保留不确定性，而非将分布仅作为额外 scalar feature；
4. 保留原有 objectness/graspness/view/CDF/width 任务链与评价约定。

本版本仍从预测 metric depth 产生 grasp center。它没有引入多中心搜索、真实深度网络输入、multi-view memory 或完整隐式 occupancy。它不能保证修复候选集合中不存在的好动作。

## 2. MoGe-inspired shape–metric factorization

开启 `--use-moge 1` 后，替换 `base.depth_net`。

```
Frozen DAV2/DINO features
    ├── three-channel DPT → affine pointmap P
    │                         ↓ known-K Z-shift + scale gauge
    │                      canonical shape
    │                         ↓ detach to metric calibration
    └── optional pose FiLM → pooled metric-anchor head
                              ↓
                    median optical depth in metres
                              ↓
               metric depth = anchor × normalized shape Z
```

### 2.1 已知 K 的 gauge

相机光轴深度为 Z，不是 Euclidean range。像素射线为 `(rx, ry, 1)`。
根据 `Pxy ≈ ray_xy (Pz+t)`，用预测 pointmap 与 K 解 camera-Z shift；再用 transverse RMS 固定正 scale gauge。这个拟合**没有读取 GT**。

真实 metric scale 不能由单张图像和 K 的投影约束唯一确定。它由 pose-conditioned pooled features 预测的 metric anchor 学习，接受原有 metric-depth L1 监督。

### 2.2 损失

- **Global aligned shape**：在每图最多 128 个有效点上，求正 scale + camera-Z shift 的 weighted L1 最优对齐。权重与 GT depth 的倒数有关。
- **Local shape**：三个 patch 尺度，分别做 median-centering 和 positive-scale alignment；关注相对局部形状。
- **Ray consistency/positivity**：抑制 canonical pointmap 偏离已知 camera ray 或落到相机后方。
- **Metric L1**：保留 main 原有 metric-depth loss；只训练 metric anchor/pose path，不把 metric regression 压力回传给 shape decoder。

**与原 MoGe 的区别：**使用已知 K、预测 robot-task metric anchor；实现的是有界采样上的 **untruncated weighted-L1** 对齐问题，不是原论文完整 truncated robust alignment pipeline；local patch sampling/centering 也是此抓取任务的适配。默认未加载 MoGe 的 pretrained pointmap decoder。

### 2.3 原有损失与新增损失

```
L = main_objectness + main_graspness + main_view
  + main_CDF_BCE + main_width + main_metric_depth
  + λglobal Lshape_global + λlocal Lshape_local + λray Lray_consistency
  + λinterval Linterval              # only for learned uncertainty
```

默认新增权重：1.0 / 0.5 / 0.05 / 0.1。它们是初始实验设定，不是已有调优结果。
不加入 P0/P1 的 residual scorer、global ranking、KD 或 material consistency，避免同时改变过多因素。

## 3. RayRoPE：替换 CVA grouping 的几何关联

开启 `--use-rayrope 1` 后替换 `base.kview_grasp_module.group`，保留原 CDF/width decoder。

输入仍是 center–view–angle queries；**此时 width 和 insertion depth 尚未由最终 decoder 决定**。因此不能把它描述成旧 MGF 的“27 个完整夹爪 support points”。

### 3.1 固定 image-space 支持区域

每个 image seed 周围采样固定 7×7 像素网格，半径 40 px。支持区域不再由预测深度决定半径。所有 finite、image-valid、query-camera-front 的 tokens 都可参与 attention；不额外进行依赖 depth-distance 的 hard candidate pruning。

由于 grouping 架构和采样方式也改变，必须有 `attention_none` 与 `rayrope_point` 两个容量/关联对照，不能将 baseline→RayRoPE 的全部差值直接归因于 uncertainty。

### 3.2 虚拟 grasp query frame

原 RayRoPE 在 query-camera frame 表示 rays。单视图抓取没有不同的输入相机，因此这里使用由 center/view/angle 定义的虚拟 query camera：

- z 轴：夹爪 approach；
- x/y 轴：closing/vertical；
- 原点：center 后退 0.15 m；
- 源相机 ray-origin 与 ray endpoint 转到这个坐标系。

位置有六个坐标：`[origin_XYZ / unit, x/z, y/z, unit/z]`，unit=0.1m。此设计使单视图输入中的 ray–grasp 相对关系能够进入 RoPE。对虚拟相机近裁剪平面使用数值保护；不将其解释为 occupancy。

### 3.3 区间期望，而非 Gaussian feature

对于投影坐标区间 `[lo,hi]`，中点 m、半宽 h：

```
E[cos(ωx)] = cos(ωm) sinc(ωh)
E[sin(ωx)] = sin(ωm) sinc(ωh)
```

`torch.sinc` 的归一化参数为 `ωh/π`。零宽度精确退化为 point RoPE。

先投影 ray 的两个 optical-depth endpoints，再在投影坐标间构造 uniform marginal interval。它不是“uniform metric depth 的精确投影概率分布”，也不是对整个 attention softmax 求期望。解析期望作用于 RoPE 基函数/旋转矩阵。

默认对 Q/K 及 V/output 应用相应旋转。输出使用 deterministic query 的反向旋转；**不会反除 uncertainty damping**。

### 3.4 不确定性

- `fixed`：固定 optical-depth halfwidth=20mm，作为首先验证的简单参照。
- `learned`：预测有界 1–80mm halfwidth；使用 detached mean 的 central interval score 监督，目标 nominal coverage=90%。

Learned interval score 是本实现的几何监督适配，不是照搬原 RayRoPE 的训练 loss。低 entropy/窄区间不自动等于正确或校准，必须报告实际 validation coverage 与区间宽度。

## 4. 梯度边界和 end-to-end 含义

- DAV2 encoder：frozen/eval，读取官方 pretrained 权重。
- Proposal / spatial enhancer / view / CVA / new grouping：在线训练。
- 原 depth decoder（MoGe OFF）：metric loss 在线训练。
- Pointmap DPT（MoGe ON）：shape/ray losses 在线训练。
- Metric anchor/pose（MoGe ON）：metric L1 在线训练。
- Learned interval head：interval supervision 在线训练。
- 所有 numeric depth、pointmap、interval 进入 grasp head 时 detach。

“在线 end-to-end”不表示允许 grasp loss 把几何变成 task latent；所有模块在同一 minibatch 内计算并优化，没有离线 feature/action cache。

训练器启动时验证 grasp→geometry gradient 为零、geometry supervision 梯度非零，并记录 shape/metric/interval 分支的梯度范数。

## 5. Flags

| 参数 | 默认 | 意义 |
|---|---:|---|
| `--use-moge` | 0 | 替换 scalar depth DPT 为 affine pointmap + metric anchor |
| `--use-rayrope` | 0 | 替换 CVA grouping 为新的 ray-attention grouping |
| `--ray-encoding` | expected | none / point / expected；同 grouping 下拆分 PE 和 uncertainty |
| `--uncertainty` | fixed | fixed / learned；learned 仅允许 expected 模式 |
| `--ray-apply-vo` | 1 | Q/K 之外，是否对 V/output 应用编码 |
| `--shape-tokens` | 0 | 在 combined 模型中额外加入局部 canonical shape offsets |
| `--shape-local-weight` | 0.5 | 设为 0 单独检验 local shape loss |
| `--fixed-halfwidth` | 0.02 | optical-depth halfwidth，单位 m |
| `--group-chunk` | 64 | 只改变内存分块；不应改变候选数/数学目标 |
| `--seeds` | 1024 | 正式实验保持固定；smoke 可降低 |

默认四个正式模型只改变 use_moge/use_rayrope。shape_tokens 是后续独立设计，不默认和两个主开关捆绑。

## 6. 数据与评价

- 20% GraspNet：100 train scenes 内 stride=5，共 5200 帧，不是随机挑 20 个 scene。
- 10% Seen validation：780 帧。只监控，不选最低 BCE 的 checkpoint。
- Latest epoch-20 为正式 checkpoint。
- 无梯度累积：3 GPUs × batch 3 = effective batch 9。
- 原始 compact CDF/width **数据集标注**仍需存在。该实现不创建 feature cache、teacher cache 或 predicted-action label cache。
- Fused depth 仅作为几何监督。RGB 网络输入不含 sensor/GT depth；原始数据集 crop/workspace 的依赖须如实披露。
- 主 AP：collision-on；threshold .01、voxel .01m、approach .05m。
- Off/on 从同一次 forward 派生；标准 evaluator 读取实际 `.npy` grasps。

`--depth-bias-mm` 只用于 inference stress test，对预测 numeric geometry 在 seed/view/group 之前加偏差。它会同时改变输出动作，**不是 fixed-action observation invariance 测试**。必须独立目录保存，不能混入 native 主表或用测试集选择最优偏差。

## 7. 参考文献/官方实现

- MoGe: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.html
- MoGe paper: https://arxiv.org/abs/2410.19115
- Official MoGe: https://github.com/microsoft/MoGe
- RayRoPE paper: https://arxiv.org/abs/2601.15275
- Official RayRoPE: https://github.com/Lucas-707/RayRoPE

代码为针对上述机制的独立实现，不附带第三方 pretrained 模型权重。实验结果尚待 GPU 运行验证。
