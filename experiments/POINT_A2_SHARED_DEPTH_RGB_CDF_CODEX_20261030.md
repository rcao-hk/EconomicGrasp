# Codex 工作单：A2 — Shared Frozen Pose-aware Geometry + Point-centric RGB Fusion + CDF

日期：2026-10-30。Repo: rcao-hk/EconomicGrasp；工作分支：exp/point-depth-pose-cdf-graspnet100。

## 0. 科学问题和实施边界

这次不再重新做已完成的 P0/P1 归因，仅验证：

- A1：同一个**冻结的** pose-aware DPT 输出 metric depth；image-space backproject → Minkowski point backbone → CDF。
- **A2（本次新实现）**：完全相同冻结 DPT、RGB/K/pose、point backbone、center/view/grouping、CDF/width 监督；额外将同一个 frozen DINOv2 ViT-B 的四层 raw patch features (A3 proposal head 的上游特征源) 通过 crop 像素索引双线性采样并融合到每个点。
- A3：image-centric action-conditioned grasp；训练与推断必须确实采用与 A1/A2 同源、**固定**的 depth 权重，才可严格声称共同几何。A3 当前 DINO→trainable proposal/enhancer 得到的 grasp-ready feature 并**不等同**于 A2 的 raw DINO RGB features；因此本组比较控制的是 RGB backbone/source，而不是确保 A2/A3 所有下游 RGB latent 一致。

**关键严格条件**：A1 和 A2 必须使用同一个 `SHARED_DEPTH_CKPT` 绝对路径，model.depth_net 包括 DINO backbone、DPT decoder、global_film adapter strict-load 并冻结；**禁止**A2不传共享checkpoint或让 grasp loss 更新 DPT。对于 A3，只能在其冻结DPT权重经核验与A1/A2完全一致、RGB/K/pose/crop和候选/评价协议对齐后，将它放进严格 A1/A2/A3 比较。否则明确写为“source-aligned but not identical geometry training”或仅报告 A1 vs A2。

### 相同与不同

| 维度 | A1 | A2 |
|---|---|---|
| Frozen pose-aware DPT (DAV2 ViT-B + global_film) | 同一严格加载的checkpoint | 完全相同 |
| 预测 XYZ | `img_idxs` gathering $\hat D$、crop K backproject、detach | 完全相同 |
| TDUnet backbone | 原始 Minkowski 3-channel ones输入；512-channel output | 完全相同 |
| RGB features | 不给 point grasp 分支 | 四层 DINO raw patch tokens（各768） |
| Image-space sampling | 无 | 每层先投影768→32，再按 `img_idxs` 在32×32 patch-grid bilinear sampling，拼接128 channels |
| Point fusion | 无 | 在 TDUnet output、graspness/view之前，512+128→残差512；最后层zero-init，初始与A1相同 |
| Grasp head | 原 Point-CDF、angle×depth、width | 相同 |
| CDF targets / width units | 扩展角度/插入深度 cache、六档friction CDF | 相同 |
| Candidate selection | point FPS + view + cylinder grouping | 相同 |
| Action-condition region sampling | 无 | **无**（这正是A2不是A3的核心区别） |

四层DINO patch feature来源与A3的冻结ViT一致，但A3另有trainable proposal head；不要把A2描述为复用完整A3 proposal map。A2不新增与Dex-Net标签有关的工作。

## 1. GitHub 实现和脚本

- `models/point_dino_rgb_fusion.py`：`MultiLevelDinoPointSampler` 与 `ResidualPointRGBFusion`；不创建448×448×768的稠密图。
- `models/economicgrasp_depth.py`：新参数 `point_rgb_fusion=none|dino_multiscale`，复用原depth forward的raw四层DINO；A1默认 `none` 与历史模型兼容。
- `models/economicgrasp.py`：只在 `use_point_rgb=True` 时对backbone output融合；Minkowski、FPS、view、cylinder grouping和CDF head都不变。
- `train_point_depth_pose_graspnet100.py`：新参数 `--point_rgb_fusion`，A2强制 `point_head=cdf` 且必须有 `--shared_depth_checkpoint`；模型checkpoint+manifest记录fusion flag、原冻结深度source SHA。
- `inference_point_depth_pose_graspnet100.py`：用保存的fusion flag与传入flag互相核验，strict-load Point-CDF weights。
- `scripts/run_point_depth_pose_graspnet100_train.sh`：环境变量 `POINT_RGB_FUSION`、`SHARED_DEPTH_CKPT`。
- `scripts/run_point_depth_pose_graspnet100_eval.sh`：`POINT_RGB_FUSION`，检查输出 manifest。
- `tools/verify_point_a2_shared_geometry.py`：A1/A2模型冻结DPT权重逐key一致性及同一训练合同；可选比对来源checkpoint与A3。
- `tests/test_point_a2_dino_rgb_fusion.py`：特征布局、像素到patch bilinear采样、梯度隔离、A1初始化一致性等。
- `tests/test_point_a2_shared_geometry.py`：仅比较depth state的同一性，避免把grasp head权重差异误作几何差异。
- `scripts/run_point_a2_shared_depth_pair.sh`：强制两组共享 `SHARED_DEPTH_CKPT`、GPU/批大小/epoch/seed，顺序启动 A1、A2 独立 CDF 训练且拒绝覆盖结果。

**特别注意**：如果A3采用的并非global_film的DPT架构、其深度权重也没冻结，则不能把当前 global_film A2 当成和A3严格匹配；先统一几何合同。训练方法不存在“只用名称宣称相同模型权重”的替代手段。

## 2. 数据、训练预算和模型选择

建议先沿用当前已部署的 RS50%＋GN-Trans50% (各12800帧/epoch，共25600)、Original Seen 10% validation，而不是新增数据集；两组A1/A2使用相同scene/frame哈希、GT监督、种子、GPU数、local/global batch、LR、优化器和21 epochs/e19固定checkpoint。不对 test Novel 选epoch；同一真实数据增强/随机采样合同与同样CPU-resident CDF labels。

**优化注意**：A2包含额外小型RGB projection/fusion参数，所以参数量并非完全相同，但没有额外DINO forward，也不引入action-conditioned head；需要报告新增params和峰值显存/耗时。A2和A1在相同seed下的共享point backbone/head构造顺序保持一致，A2的zero-init fusion保证初始未融合的 point features相同。

## 3. 执行前最小静态检查（不做单→双→三卡 smoke 阶梯）

按用户当前已批准的节奏，不再组织完整的单/双/三GPU smoke；先确认数据路径、CUDA/磁盘工作区，并做快速单元/静态验收，然后按既定GPUs直接启动训练。

~~~bash
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
nvidia-smi
git fetch origin exp/point-depth-pose-cdf-graspnet100
# 在独立 worktree 中使用该branch最新commit，不要reset当前工作区
python -m py_compile models/point_dino_rgb_fusion.py models/economicgrasp_depth.py \
  models/economicgrasp.py train_point_depth_pose_graspnet100.py \
  inference_point_depth_pose_graspnet100.py tools/verify_point_a2_shared_geometry.py
bash -n scripts/run_point_depth_pose_graspnet100_train.sh scripts/run_point_depth_pose_graspnet100_eval.sh
python -m pytest -q tests/test_point_a2_dino_rgb_fusion.py tests/test_point_depth_cdf.py \
  tests/test_point_depth_training_protocol.py tests/test_point_a2_shared_geometry.py
~~~

除代码单测外，确认source checkpoint确实存在、具备 global_film 和 `depth_net.*`，且训练选中场景/标签不缺失。禁止静默换成另一深度来源。

## 4. A1 与 A2 训练（同一冻结depth source）

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export SHARED_DEPTH_CKPT=/ABSOLUTE/PATH/TO/VERIFIED_GLOBAL_FILM_DEPTH_SOURCE.tar

# A1: geometry-only point + CDF，fresh point weights, freeze exact DPT
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 HEADS=cdf POINT_RGB_FUSION=none MAX_EPOCH=21 \
BATCH_SIZE=3 SEED=0 MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_a1_shareddepth_cdf_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

# A2: frozen same DPT + same point/CDF + raw four-level DINO sampled to points
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 HEADS=cdf POINT_RGB_FUSION=dino_multiscale MAX_EPOCH=21 \
BATCH_SIZE=3 SEED=0 MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_a2_shareddepth_dinofusion_cdf_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

也可以用**单一配对启动器**避免分别输入两套命令导致source或训练合同不一致：

~~~bash
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 BATCH_SIZE=3 MAX_EPOCH=21 SEED=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_a1_a2_pair_UNIQUE \
bash scripts/run_point_a2_shared_depth_pair.sh
~~~

它会使用同一个 SHARED_DEPTH_CKPT，写入 OUTPUT_ROOT/a1/cdf 和 OUTPUT_ROOT/a2/cdf。**这是上面两条单独训练命令的替代方式，不要同时重复执行。**

两个run要有独立结果目录，默认串行执行即可。A2 训练脚本会按稀疏间隔输出 `A2-RGB-GRAD`（point RGB sampler/fuser gradient absolute sum、冻结DPT可训练参数数）；zero-init使第一个step的sampler梯度为零是预期，但后续应激活，否则须视为A2未充分利用RGB、不能直接解释负结果。每个run的主checkpoint：`<OUTPUT_ROOT>/cdf/checkpoint_epoch_019.tar`；e20仅存档。必须保证A2训练时 frozen depth_net 不更新，point RGB projection 和 fusion的训练梯度非零（zero-initialized final conv导致第一个step上游RGB projection梯度为零是预期；到后续优化步需要出现非零梯度）。如GPU资源不足必须把两个模型统一回退卡数/批大小，披露global batch变化；不要单方面换协议。

## 5. 共同几何严格审计（不重复P0/P1）

~~~bash
python tools/verify_point_a2_shared_geometry.py \
  --a1 /path/to/point_a1_shareddepth_cdf_UNIQUE/cdf/checkpoint_epoch_019.tar \
  --a2 /path/to/point_a2_shareddepth_dinofusion_cdf_UNIQUE/cdf/checkpoint_epoch_019.tar \
  --source "$SHARED_DEPTH_CKPT" \
  --expected-domain mixed \
  --output /path/to/NEW_a1_a2_shared_geometry_audit.json

# A3已具备完全相同且冻结的DPT时，额外添加：
# --a3 /path/to/verified_a3_frozen_depth_checkpoint.tar
~~~

检查通过代表 **DPT tensor-by-tensor完全一致**，不能自动代表A3训练过程中冻结，也不能自动代表三种方法的RGB crop/seed集合完全对齐。后者在报告里按真实协议查实，不加一轮冗余的逐probe因果拆解。

## 6. RealSense 10% inference / official AP

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet

# A1
TRAIN_ROOT=/path/to/point_a1_shareddepth_cdf_UNIQUE \
HEADS=cdf POINT_RGB_FUSION=none EXPECTED_TRAIN_DOMAIN=mixed \
GPUS=0,1,2 TEST_FRACTION=0.1 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_a1_shareddepth_ap10_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

# A2
TRAIN_ROOT=/path/to/point_a2_shareddepth_dinofusion_cdf_UNIQUE \
HEADS=cdf POINT_RGB_FUSION=dino_multiscale EXPECTED_TRAIN_DOMAIN=mixed \
GPUS=0,1,2 TEST_FRACTION=0.1 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_a2_shareddepth_ap10_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

# A2-A1 paired 30-scene bootstrap:
python tools/summarize_point_depth_pose_ap.py \
 --root /data/robotarm/result/grasp/rgbgrasp/eval/point_a2_shareddepth_ap10_UNIQUE/cdf \
 --reference-root /data/robotarm/result/grasp/rgbgrasp/eval/point_a1_shareddepth_ap10_UNIQUE/cdf \
 --bootstrap 50000 \
 --output /path/to/NEW_a2_minus_a1_summary.json
~~~

**相同算法不保证完全相同的候选集合**：A2的RGB融合可能改变 graspness、point FPS和view selection，结果是系统级表征对照而非固定同一候选的纯score消融。\n\n同一RGB/K/pose、同一 RealSense 测试split/frame sampling与 per-frame seed，collision source/threshold、evaluator Top50、三split [30,26,50,6]，官方AP以及µ0.4/µ0.8、prefix@1/5/10/20/50和场景聚类CI。仅报告实际结果，不能杜撰AP。

## 7. 成果与结论规则

Codex生成 `experiments/POINT_A2_SHARED_DEPTH_RGB_CDF_RESULTS_ZH.md`，包含代码SHA、两checkpoint和source SHA、DPT逐权重相等的审计、两组训练日志、depth MAE或误差条件、valid CDF比例、AP/CI、显存及推理成本。谨慎解释：

- A2>A1：原 point pipeline确实从保留RGB信息获益；
- A3>A2且严格共享冻结深度：可支持action-conditioned evidence organization优于本实现的point-wise fusion，但仍需注明backbone/head/candidate/feature加工的结构差异，不能宣称仅隔离了“组织方式”；
- A2≈A3：不能再把A3优势简单归因于 action-conditioning；
- DPT shared方案使各方法差距缩小：应重估几何学习作为性能来源的贡献。

**当前状态**：本GitHub实现与工作单已提交；没有在GPU服务器跑正式A1/A2训练、三split官方AP或共同DPT审计。任何新数值均PENDING。
