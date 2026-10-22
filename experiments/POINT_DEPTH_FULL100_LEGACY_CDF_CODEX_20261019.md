# Codex 工作合同 — Full GraspNet Point-centric Pose-aware Depth：Legacy vs CDF

2026-10-19。**这是实验执行与验收工作单，不是已完成训练报告。**

- 仓库：rcao-hk/EconomicGrasp
- 新分支：**exp/point-depth-pose-cdf-graspnet100**
- 父分支：exp/point-depth-pose-aware-mixed@396ac38aa5b7700dc50b54d90942edc2ea9799c2（最初来自 main@52d09f925059bec3643610ecf1f1722894627ee5）
- 默认实验：**100% Original GraspNet-1Billion RealSense TRAIN**；首先执行10% matching TEST，随后需要时可以执行100% TEST。**不使用 GN-Trans**。
- 不准改 main、覆盖/删除任何 RayRoPE/GVAR/mixed experiments，也不得把 smoke 当正式 AP。

**2026-10-22 更新：** 当前同一分支也支持 RS50%＋GN-Trans50% mixed training（Legacy 和 CDF 两个head）。该模式有独立工作单 [POINT_DEPTH_RS50_GNTRANS50_LEGACY_CDF_CODEX_20261022.md](./POINT_DEPTH_RS50_GNTRANS50_LEGACY_CDF_CODEX_20261022.md)，并使用 TRAIN_DOMAIN=mixed、MIX_TRAIN_FRACTION=0.5。**本文件余下的 Original-only 命令仍然有效；两种设置不得混用输出目录或将25,600帧的名义数量误认为同一来源合同。** 新统一 trainer 的协议清单命名为 point_depth_training_protocol.json，checkpoint仍保留 full100_protocol 兼容字段并通过 train_domain 判定训练来源。

## 1. 研究问题和两组方法的严格边界

目的：给此前的 RGB→pose-aware DPT depth→predicted points→EconomicGrasp 添加与 CVA-CDF 尽可能接近的 **joint angle×depth evaluator-aligned CDF + depth-wise width**，评估原点云 backbone 的性能上限，减少 score/width 监督不同带来的不公平。

两个变体固定：
- RGB448, camera_pose_vec, crop-aware K, indexed pixels; DAV2 ViT-B frozen, trainable DPT + global_film pose conditioning; metric depth [0.2,1.0]m；20k predicted XYZ（depth_map_pred.detach()）；
- 完全相同的原 EconomicGrasp3D Minkowski TDUnet、objectness/graspness、1024 seed FPS、300 view、cylinder group；
- full-map GT L1 depth（rendered 前景＋TSDF 背景，weight10）；相同optimizer LR、batch、epoch、seed、collision；
- inference仅给网络 RGB、K、pose、img_idxs；RealSense sensor depth仍用于历史 workspace/crop 与 model-free collision，严格术语是 **RGB-only network with depth-assisted preprocessing/filtering**。

差异：

| 项目 | Legacy Point | Point CDF |
|---|---|---|
| Head | 原四路 attention：angle(13)、depth(5)、score(6)、width(1) | CDF [B,6,Q,12,4]＋depth-wise width [B,4,Q,12] |
| 标签缓存 | economic_grasp_label_300views | economic_grasp_label_300views_extend_angle_cdf_depth |
| 匹配 | 原 process_grasp_labels（5 mm） | 已实现 process_grasp_labels_cdf_width（同5mm、view+angle旋转对齐） |
| Loss | CE angle/depth/score＋width SmoothL1 | 与 CVA-CDF 相同 unbalanced BCE(T)＋depth-wise width SmoothL1 |
| Decode | 原 expected-score/argmax angle/depth | mean sigmoid success over six thresholds，joint (angle,depth) argmax，selected width |

CDF bins 0 表示六个摩擦阈值全失败，1..6 表示首次成功的0.2/0.4/0.6/0.8/1.0/1.2阈值；width 缓存为整数mm，matcher转换到m一次，loss监督乘10，decode宽度为1.2×head/10m。必须用实际cache值验证，防止双重转换。

**科学限制**：legacy best-angle/depth labels 与 CDF full-grid labels 不同；该比较是 head＋supervision package，不是严格单变量因果消融。虽然 DPT 的结构相同，独立训练得到的 predicted depth 可能不同，不能称两个模型的 metric geometry 完全相同。可选环境 SHARED_DEPTH_CKPT 将两模型加载并冻结同一 global_film depth_net 权重用于独立 shared-geometry 诊断，默认关闭。此前 RayRoPE/GVAR 10% Original＋10% GN-Trans 的 AP 不能与这里100% Original直接视为训练预算公平比较。

## 2. 核验文件

- models/point_grasp_cdf_head.py：新的 CDF head / point CDF joint decoder。
- models/economicgrasp.py：原 point grasp net 的 use_cdf switch；原 grouping/view/center 不变。
- models/economicgrasp_depth.py：pose-aware depth 外层增加 point_use_cdf 参数，默认False向后兼容。
- models/loss_point_depth_cdf.py：复用原 point objectness/graspness/view/depth loss 和同一 CVA-CDF BCE＋width loss。
- train_point_depth_pose_graspnet100.py：100% train 两头配置，CPU保留 CDF label，21 epochs/e19。
- inference_point_depth_pose_graspnet100.py：strict epoch/head/full-data checkpoint，10%/100% Original test。
- scripts/run_point_depth_pose_graspnet100_train.sh：HEADS=legacy,cdf；默认顺序训练不并发。
- scripts/run_point_depth_pose_graspnet100_eval.sh：可配置测试比例，三个split、serial official AP。
- tools/summarize_point_depth_pose_ap.py：shape [30,26,50,6] 或 [30,256,50,6]，相同shape的scene bootstrap。
- tests/test_point_depth_cdf.py、tests/test_point_depth_mixed_contract.py：单元与协议测试。

## 3. 固定训练、验证、推断合同

| 参数 | 值 |
|---|---|
| Training scenes | GraspNet Original 0000..0099 |
| Full training frames | 100 scenes ×256 = **25,600 每 epoch**；单一 Original 域 |
| Val | Original test_seen scenes 0100..0129；每scene 0,10,..250＝**780 frames**；不选Novel checkpoint |
| GPU/seed | 目标为单节点3×RTX3090 24GB、per-rank BS3（global BS9）、seed0；允许经 smoke 验收按 **3→2→1 GPU** 降级，两个 head 必须同一最终配置 |
| Steps/epochs | 21 epochs e0..e20，**e19 predeclared**为主评测，e20存档 |
| Loss/optimizer | AdamW, LR1e−4 cosine, weight_decay0, depth_weight_decay0, grad clipping1.0 |
| DDP samples | 25,600 nominal；3 rank 时 DistributedSampler padding 到25,602；2 rank/1 rank 均为25,600。日志必须记录实际 world size、global batch 与 optimizer steps/epoch |
| 10% test | Seen/Similar/Novel 每split 30scene×26frame=780 |
| Optional full test | 每split30scene×256frame=7,680 |
| Collision | Original sensor 0.01 threshold，0.01m voxel |
| Official AP | GraspNet eval.py；CPU 16 workers，每个split和模型串行；finite [30,26|256,50,6] |

## 4. Step 0：服务器体检与隔离 worktree

以前的GPU服务器为 robotarm@10.30.7.117，**当前连通性、CUDA环境、磁盘和 GPU 负载本次均未验证**。请先检查：

~~~bash
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
find /home/robotarm -name AGENTS.md -print
nvidia-smi
df -h /data /data2
git fetch origin exp/point-depth-pose-cdf-graspnet100
git rev-parse origin/exp/point-depth-pose-cdf-graspnet100
# 只在此路径确认不存在后创建独立 worktree
git worktree add --detach ../EconomicGrasp-point-cdf-full100 origin/exp/point-depth-pose-cdf-graspnet100
cd ../EconomicGrasp-point-cdf-full100
~~~

不得删除旧checkpoint、kill其他作业、reset/stash非本工作区。对缺少GPU和空间明确记录BLOCKED或PENDING。

## 5. Step 1：静态、CPU、数据和真实模型smoke

~~~bash
python -m py_compile models/point_grasp_cdf_head.py models/economicgrasp.py \
 models/economicgrasp_depth.py models/loss_point_depth_cdf.py \
 train_point_depth_pose_graspnet100.py inference_point_depth_pose_graspnet100.py \
 tools/summarize_point_depth_pose_ap.py
bash -n scripts/run_point_depth_pose_graspnet100_train.sh scripts/run_point_depth_pose_graspnet100_eval.sh
python -m pytest -q tests/test_point_depth_cdf.py tests/test_point_depth_mixed_contract.py
~~~

随后检查全100个train scenes及30个seen-val scenes的标准/扩展缓存、rendered GT和TSDF是否均存在，尤其扩展cache的 [P,K,12,4]、CDF thresholds、mm width、object id、angle alignment。检查实际训练图像和GT有效像素/比例、pose vec方向、K/img_idxs backprojection，逐例证明 predicted depth 是唯一点云输入（传给网络的GT只作为loss监督）。

Single GPU与DDP smoke须验证：
- 每个网络分支点云 [B,20000,3]，grasp seed数1024；
- CDF [B,6,1024,12,4]、width [B,4,1024,12]，matcher对应labels [B,1024,12,4]；
- Legacy angle [B,13,1024]、depth [B,5,1024]、score [B,6,1024]、width [B,1,1024]；
- CDF object list留CPU，仅matched rows transfer到GPU；DDP device_ids=None 以防自动搬运CPU labels；
- DPT depth loss 传递梯度至 DPT/FiLM 而不更新 frozen ViT；grasp loss无法经detached预测XYZ反传depth；
- loss finite、valid ratio非零或解释为何为空、CDF正例比例/width有效比例合理、预测深度mean/std无塌缩；多rank执行不卡死。
- 检查 CDF 特殊单元样本能把唯一高utility动作解码为正确angle与1-4cm insertion及宽度单位。

## 6. Step 2：单卡→双卡→三卡验收、可回退的 fresh 100% 训练

**执行准则：默认目标是3 GPU；只有 Legacy 与 CDF 两个 head 在相同 GPU 数、相同 per-rank batch、相同训练协议下都通过 forward、backward、optimizer、validation、checkpoint 的 smoke，才允许以该配置正式训练。多卡失败时按 3→2→1 降级，而不是无限调 NCCL。** 所有目录使用真实不存在的 UNIQUE 路径；以下 GPU 编号只作示例，先检查实际空闲卡。

首先检查单卡、双卡，再执行三卡 smoke。短程 smoke 限制 batch，不是完整 epoch、不是正式 checkpoint，不能送入官方 AP。

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet

# (A) 单卡：两种 head 顺序跑；torchrun nproc=1，不创建 DDP wrapper
GPUS=0 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=2 \
MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=2 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_full100_1gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

# (B) 双卡：至少两次 backward/step 与 validation，排查 rank/ME/NCCL
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
GPUS=0,1 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=2 \
MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=2 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_full100_2gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

# (C) 三卡：更长的 10-batch DDP smoke，同时使用目标 per-rank BS3 做显存验收
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
GPUS=0,1,2 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=3 \
MAX_TRAIN_BATCHES=10 MAX_VAL_BATCHES=2 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_full100_3gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

**smoke 验收清单：**
- 每个模式、每个 rank 使用其独立的 local CUDA device；MinkowskiEngine sparse coordinates/features、backbone outputs、loss 均在正确设备，不发生跨 GPU 混用。
- 每 rank 完成所请求的 train steps（包括 backward、optimizer.step）、Seen validation、checkpoint 保存，loss 与梯度有限且两种 head 具有合理的有效标签；没有 collective mismatch、NCCL timeout、CUDA illegal memory access 或 ME coordinate-manager 错误。
- CDF 的变长 cache 保持 CPU resident，匹配时只向对应 GPU 搬运选中行；DDP(device_ids=None) 的语义不可为消除报错而未经论证随意修改。
- 下一轮正式训练的 **实际 GPU 数和 BATCH_SIZE** 都必须在 smoke 中测试过；单/双卡 smoke 的 BS2 通过并不保证正式 BS3 不会 OOM。如果拟以双卡或单卡 BS3 正式训练，先在相同卡数、BS3 下补跑 5–10 batch smoke（使用独立输出根目录）。

### DDP 失败后的 fallback 决策（强制，适用于两个 head）

| 情况 | 后续决策 |
|---|---|
| 两个 head 均通过 3 GPU、目标 batch smoke | 使用同一 3 GPU 设置正式训练两组 |
| 3 GPU 任一 head 失败，但两组均通过 2 GPU | 在 2 GPU 上用**相同** batch/seed/21 epochs 训练两组；若目标 batch 尚未验收，先补双卡相应 batch smoke |
| 2 GPU 也有 DDP 失败，而两组单卡均正常 | 回退 **1 GPU**；两个 head 使用完全相同的 batch、seed、21 epochs 和数据集合 |
| 只是一种 head OOM 或报错 | 先分类修复或统一降低 per-rank batch，并重跑两种 head 的 smoke；不要一组3卡、一组单卡直接比较 |
| 任一 head 单 GPU 也无法完成 | 停止该实验；修正数据、CUDA extension 或模型问题，不得以一组不完整结果替代正式对照 |

排错按 **单卡本体错误 → local device / ME kernel → GPU OOM → DDP/NCCL collective** 顺序记录失败 rank、首个 Python/CUDA 异常、nvidia-smi 与完整日志，明确区分 root cause 与后续 NCCL timeout。允许有限度修复并回归，**不应无限等待超时、kill 非本作业进程、清空其他实验的显存或重建正在使用的环境**。

**公平性要求：两个 head 必须最终在共同最高可用配置中完成 fresh training。** 如果 Legacy 已在 3 卡完成而 CDF 只能 2 或 1 卡，不可直接把二者当作训练协议一致的主对照；应当在共同回退配置下重新运行 Legacy，保留旧结果做辅助分析。失败/中断后重新选择全新的训练输出根目录，不覆盖或混用旧 checkpoints；两组均固定训练21 epochs并取 e19。

**优化步数变化必须如实披露：** 本训练器目前不支持 gradient accumulation，改变 GPU 数、per-rank batch 会改变 effective global batch 和每 epoch 的 optimizer.step 数。例如相同 per-rank BS3 时，3卡 global9、2卡 global6、1卡 global3。保持两个 head **采用相同** GPU数及 BS 是内部公平比较的最低要求，但不能把回退运行声称为与预设3卡/global9完全等价的优化协议；尤其不可静默更改学习率、loss weights 或 epoch 数。若以后确实需要严格保持 global batch，应单独实现、测试梯度累积，而非在此处假定已支持。

### 正式训练命令（三选一；仅运行已完成对应 smoke 验收的配置）

~~~bash
# 首选：两组各完整25,600帧/epoch，共21 epochs，顺序执行
GPUS=0,1,2 HEADS=legacy,cdf MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_full100_3gpu_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

# 三卡失败时：优先使用两卡共同通过的配置
GPUS=0,1 HEADS=legacy,cdf MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_full100_2gpu_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

# 两卡也失败时：单卡 fallback（如 BS3 OOM，则两者均降至已验收的 BS2/BS1）
GPUS=0 HEADS=legacy,cdf MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_full100_1gpu_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

**注意：上面 3 段命令仅列备选配置，不应全部运行。** 单卡/双卡的 BS3 须先额外 smoke 验收；如果使用 BS2 或 BS1，需在正式命令中对两个 head 一起修改 BATCH_SIZE，并重新检查实际每 epoch optimizer steps。

真实 full100 数据量远大于之前 mixed5200/epoch，必须根据通过的 smoke 测量吞吐、估算训练时间。结果报告记录：3/2/1 GPU 各方案是否通过、失败错误和修复情况、最终选择的 GPU 数/卡号、per-rank/global batch、每 epoch 样本填充/steps、学习率策略、实际 source SHA、checkpoint SHA256、peak GPU memory、wall time、Seen val depth MAE、grasp valid ratio。严格核验 e19 checkpoint；不得基于 test Novel AP 调整配置。


## 7. Step 3：配对推理＋官方AP

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
# 使用上一步真正完成并验收的 3GPU/2GPU/1GPU 正式输出根目录
export TRAIN_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_full100_CHOSEN_UNIQUE

# inference smoke，不能运行官方AP
GPUS=0 HEADS=legacy,cdf SPLITS=test_seen TEST_FRACTION=0.1 \
MAX_BATCHES=2 EVAL=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point100_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

# 首先使用既有10% test protocol，可与历史研究比较时明确训练域差异
GPUS=0,1,2 HEADS=legacy,cdf TEST_FRACTION=0.1 \
SPLITS=test_seen,test_similar,test_novel \
BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point100_10pct_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

# 若要正式全test paper table，在全新目录执行：
GPUS=0,1,2 HEADS=legacy,cdf TEST_FRACTION=1.0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point100_fulltest_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh
~~~

注意 inference SAMPLE_INTERVAL=0.1是数据比例，而 eval.py sample_interval=10是帧步长。完整测试为1.0比例／stride1。所有 AP finite 且 shape严格一致；两个变体必须使用相同per-frame sampled-image indices、camera，Original sensor collision，NMS与Top50，不能把无碰撞过滤结果掺入正式AP。

## 8. Step 4：统计、论文解释与交付报告

~~~bash
export EVAL_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point100_10pct_UNIQUE
python tools/summarize_point_depth_pose_ap.py \
 --root "$EVAL_ROOT/cdf" --reference-root "$EVAL_ROOT/legacy" \
 --bootstrap 50000 --output "$EVAL_ROOT/point_cdf_vs_legacy.json"
~~~

生成 POINT_DEPTH_FULL100_LEGACY_CDF_RESULTS_ZH.md，必须包含：
- 主/父分支和实际训练源码 SHA，数据/label-cache和checkpoint SHA，pass/fail日志；
- 100% Original train数量、Original Seen Val数量和e19声明；单→双→三卡 smoke 通过情况、DDP/NCCL/ME故障及 fallback 原因，最终 GPU 数、batch/global batch、optimizer steps/epoch 和两组 head 的协议一致性；
- 两版本 Seen/Similar/Novel/Mean AP、AP0.4/0.8、prefix Top1/5/10/20/50、分scene bootstrap CI；
- DPT foreground MAE/valid像素/variance、grasp matched ratio、CDF positive/width valid；
- 模型params、GPU peak、training walltime、inference latency；
- 两种cache监督与独立预测几何可能不同造成的公平性限制；100% Original vs mixed（10%+10%）不能作为单因子归因。

**当前状态：GitHub代码与工作单已提交；本会话没有运行服务器 CUDA smoke、正式100%训练、官方推断与AP。所有新实验性能均 PENDING。**
