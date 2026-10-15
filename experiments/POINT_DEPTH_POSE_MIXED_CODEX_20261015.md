# Codex 工作文档：Pose-aware Point-Depth Baseline / Mixed 10%+10%

日期：2026-10-15。**这是代码实现与后续 GPU 执行合同，并非训练已完成报告。**

- 仓库：rcao-hk/EconomicGrasp
- Base: main@52d09f925059bec3643610ecf1f1722894627ee5
- Branch: exp/point-depth-pose-aware-mixed
- 目标：复用 pose-aware metric DPT 和 mixed dataset，让 RGB→预测深度→point-centric EconomicGrasp 构成具有说服力的对照；不得更改 main 或 RayRoPE/GVAR worktree。

## A. Scientific contract

对照问题：若保持相同的 RGB+已知相机姿态、DAV2/DPT (global_film) metric-depth 预测器、混合数据分布与独立点云反馈，那么 image-centric grasp representation 相比将预测深度 backproject 后交给 point-centric grasp encoder 的优势有多大？

必须保留以下架构分工：

1. Metric depth：既有 DINOv2DepthRegressionNet (ViT-B, frozen backbone, pose_depth_mode=global_film)，共享 PoseAwareDPTFiLM。输入448×448 RGB、camera_pose_vec；输出米制 depth [0.2,1.0] 和独立 RGB features。
2. Point-cloud baseline：对同一 crop-resized img_idxs 从预测 depth gather 并按 crop-aware K backproject；传入原 EconomicGrasp3D (Minkowski TDUnet、original objectness/graspness/view/angle/depth/score/width heads、standard grasp-label cache)；**depth_map_pred.detach()** 不允许 grasp loss 经 3D 坐标返回 DPT。
3. Supervision：full-map rendered/TSDF GT L1 × depth_prob_loss_weight=10；原 models/loss_economicgrasp_depth.py 的 point-specific grasp supervision 保持不变。不要引入 CVA/CDF、RayRoPE、GVAR、obs-depth refinement、额外 train-time DexNet evaluator 或 KD。
4. Network inference 输入是 RGB、K、img_idxs、camera_pose_vec，不允许 observed/rendered/GT/virtual depth 进入网络几何输入。原 GraspNet workspace/crop 与 model-free collision 可使用 sensor depth，但必须写明**RGB-only network with depth-assisted preprocessing/filtering**。

比较的科学边界：两个抓取 backbone/head 的标签监督、候选策略和计算不同。该实验是共享几何架构与训练数据约束下的 **pipeline-level comparison**，不是完全消除其他因素的严格 representation-only 因果消融。相同 DPT 结构也不保证独立训练后预测深度逐像素一致；补充 MAE 和误差图、检查模型权重与对应数据合同。

## B. Fixed data/training/test contract

| Item | Value |
|---|---|
| Original train | RealSense scenes 0000..0099, frames 0,10,..250 (26), 2600 |
| GN-Trans train | Paired scene/frame IDs，独立 domain 2600，合计 5200 nominal/epoch |
| Original depth target | rendered foreground + TSDF background, USE_FUSE_DEPTH=1 |
| GN-Trans depth target | full rendered, GNTRANS_FUSED_BACKGROUND=0 (Mixed-v2 protocol) |
| Point sampling / graspness | Existing Original sensor depth workspace; GN-Trans virtual rendered depth workspace + virtual graspness. These are existing loader differences; log them honestly |
| Label cache | standard economic_grasp_label_300views; no CDF/extended-angle |
| Validation | Original test_seen 10% only, 780 frames; no GN-Trans/Novel checkpoint selection |
| Training | 3 RTX3090, per GPU batch3/global9, AdamW LR 1e-4 cosine, WD 0, seed0, 21 epochs e0..e20 |
| Official checkpoint | e19 = checkpoint_epoch_019.tar, regardless of validation |
| Official testing | Original RealSense test_seen / test_similar / test_novel, each 30 scenes × 26 frames = 780 |
| Collision | Original sensor cloud, threshold 0.01, voxel 0.01m |
| Evaluation | Official eval.py, 16 CPU workers serialized, 10% frame stride10; finite AP tensor [30,26,50,6] |
| Statistical comparison | Paired 30-scene bootstrap, do not count 26 correlated frames or different seeds as independent |

IMPORTANT: Mixed-v1 (Original full TSDF) and historical GVAR background supervision may differ from Mixed-v2. If comparing to a particular existing checkpoint, inspect its actual protocol first. GNTRANS_FUSED_BACKGROUND=1 is supported only as a separately named contract, not interchangeable with this default. DDP DistributedSampler may pad nominal 5200 to 5202 examples on 3 ranks; record it.

## C. Implemented files to review

- models/economicgrasp_depth.py — constructor/forward pose_depth_mode applied to the original point-depth baseline (reuse existing pose adapter); original point backbone and pred_decode untouched.
- utils/point_depth_mixed_contract.py — deterministic 10% scene/frame selection and RGB-only network input allowlist.
- train_point_depth_pose_mixed_ddp.py — fresh 10%+10% training with Original Seen validation, e19 checkpoint, manifest, strict resume.
- scripts/run_point_depth_pose_mix_train.sh — safe shell training entrypoint.
- inference_point_depth_pose.py — e19 strict load; stateless frame-local pixel sampling; original sensor collision; no external depth network inputs.
- scripts/run_point_depth_pose_mix_eval.sh — 10% Original three splits inference, serialized official AP and no-overwrite.
- tools/summarize_point_depth_pose_ap.py — official AP summary and optional paired scene-bootstrap CI.
- tests/test_point_depth_mixed_contract.py — CPU contract tests.

## D. Step 0: audit remote GPU/disk/workspaces, then isolate

Previous server: robotarm@10.30.7.117. Current network access, free GPU/disk, job status, and dataset files are **unknown**. Do not promise or invent completed server execution.

~~~bash
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
find .. -name AGENTS.md -print
git fetch origin main exp/point-depth-pose-aware-mixed
git rev-parse origin/main
git rev-parse origin/exp/point-depth-pose-aware-mixed
nvidia-smi
df -h /data /data2
# ONLY if absent, without changing existing worktrees:
git worktree add --detach ../EconomicGrasp-point-depth-pose-mixed origin/exp/point-depth-pose-aware-mixed
cd ../EconomicGrasp-point-depth-pose-mixed
~~~

Never auto kill workloads, remove checkpoints, force reset/checkout, or reuse old result roots.

## E. Step 1: CPU/static and real-data smoke acceptance

~~~bash
python -m py_compile models/economicgrasp_depth.py \
  utils/point_depth_mixed_contract.py \
  train_point_depth_pose_mixed_ddp.py \
  inference_point_depth_pose.py \
  tools/summarize_point_depth_pose_ap.py
bash -n scripts/run_point_depth_pose_mix_train.sh scripts/run_point_depth_pose_mix_eval.sh
python -m pytest -q tests/test_point_depth_mixed_contract.py
~~~

Further checks with the real dataset and GPU:
- Confirm standard grasp caches exist for Original/GNTrans train AND test_seen validation, plus all virtual depth/virtual_graspness and TSDF directories. Confirm each domain 2600 and held-out Seen780; both training scene/frame signatures identical.
- Check two frames/domain: RGB img [3,448,448], K crop-aware [3,3], img_idxs [20000] in [0,448²), pose_vec [3] finite; depth GT in meters and source correct; point-level 20k labels correspond to selected pixels.
- Verify depth output [B,1,448,448] finite, point output [B,20k,3] finite, coordinates backproject *only* from depth_map_pred.detach(); supervised DPT + pose FiLM receives nonzero depth-loss gradient, frozen DINO backbone no updates, grasp loss does not update depth head, point grasp model receives nonzero gradients.
- Verify single GPU and multi-GPU DDP are free of CPU-label placement, width m/mm conversion, MinkowskiEngine CUDA/ME sparse-quantization and optimizer-state errors.
- Distinguish measured depth MAE from AP; do not assume same depth accuracy just because DPT architecture matches.

## F. Step 2: training (run only after Step 1 passes)

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans

# Single-GPU 1-epoch smoke; no official test AP from this
GPUS=0 MAX_EPOCH=1 BATCH_SIZE=3 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_depth_pose_smoke_UNIQUE \
bash scripts/run_point_depth_pose_mix_train.sh

# Optional 2-GPU DDP smoke, separate root
GPUS=0,1 MAX_EPOCH=1 BATCH_SIZE=3 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_depth_pose_ddp_smoke_UNIQUE \
bash scripts/run_point_depth_pose_mix_train.sh

# Formal 3-GPU 21-epoch training, new output root
GPUS=0,1,2 MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
USE_FUSE_DEPTH=1 GNTRANS_FUSED_BACKGROUND=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_depth_pose_mixed_v2_UNIQUE \
bash scripts/run_point_depth_pose_mix_train.sh
~~~

Record exact source SHA, git dirty status, model/checkpoint SHA256, domain sampling SHA, optimizer, val history, max GPU memory, wall time and failed attempts. If full run cannot start, report pending instead of extrapolating AP.

## G. Step 3: inference and official 10% evaluation

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export CKPT=/data/robotarm/result/grasp/rgbgrasp/log/point_depth_pose_mixed_v2_UNIQUE/checkpoint_epoch_019.tar

# GPU smoke, no evaluator
GPUS=0 SPLITS=test_seen MAX_BATCHES=2 ENABLE_EVAL=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_depth_pose_smoke_UNIQUE \
bash scripts/run_point_depth_pose_mix_eval.sh

# Official 3 splits × 780 = 2340 frames; new root
GPUS=0,1,2 SPLITS=test_seen,test_similar,test_novel \
BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_depth_pose_v2_UNIQUE \
bash scripts/run_point_depth_pose_mix_eval.sh

python tools/summarize_point_depth_pose_ap.py \
  --root /data/robotarm/result/grasp/rgbgrasp/eval/point_depth_pose_v2_UNIQUE \
  --output /data/robotarm/result/grasp/rgbgrasp/eval/point_depth_pose_v2_UNIQUE/summary.json
~~~

Inference uses sample_interval=0.1 **fraction**; official evaluator uses sample_interval=10 **stride**. No smoke AP; no overwriting logs or existing frame files; preserve dumps and manifests.

For optional paired scene CI compare with verified official RayRoPE Mixed-v2 arrays, after verifying collision/sampling/target/heldout protocols:

~~~bash
python tools/summarize_point_depth_pose_ap.py \
  --root /path/to/point_depth_pose_v2 \
  --reference-root /path/to/verified_rayrope_v2_evaluation \
  --output /path/to/NEW_point_vs_image_comparison.json
~~~

## H. Deliverable and scientific decision

Write POINT_DEPTH_POSE_MIXED_RESULTS_ZH.md with source commit/actual server patches, checkpoint/hash, train/eval manifests, depth quality, official Seen/Similar/Novel AP/AP0.4/AP0.8/Mean, prefix@1/5/10/20/50, paired CIs, compute/cost and failure cases.

Conclude whether under closely matched predicted metric geometry and 10%+10% training, converting depth to a point cloud and running classic EconomicGrasp is competitive with image-centric action-conditioned representation. Describe supervision/backbone/point-sampling differences; **do not call this a representation-only isolation**.

**Status at document creation: implementation committed to GitHub; no server CUDA, 21-epoch training or official AP has been executed from this chat.** All numerical AP results for this new branch remain pending.
