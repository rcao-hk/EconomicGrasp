# Codex — Point-centric Legacy / CDF: RS50% + GN-Trans50%

日期：2026-10-22。仓库：rcao-hk/EconomicGrasp；工作分支：exp/point-depth-pose-cdf-graspnet100。

**本工作单只针对 mixed RS50% + GN-Trans50% 的两种 point-centric head。** 现有100% Original 工作单仍然有效。仓库代码已适配；服务器训练、CUDA smoke、官方AP尚未运行，必须如实标注 PENDING。

## 1. 研究目标与固定合同

本实验回答：当 pose-aware DAV2/DPT metric depth、预测点云、Minkowski TDUnet、center FPS、300-view、cylinder grouping 和 mixed training exposure 相同时，Legacy score/angle/depth/width 与 evaluator-aligned Point-CDF + depth-wise width 的性能差异是什么？

训练域定义（每个域100个 scene，且每scene选择相同的帧ID）：

| Dataset | Scenes | Frames per scene | Total |
|---|---|---|---|
| Original GraspNet RealSense | 0000..0099 | 0,2,...,254 (128) | 12,800 |
| GN-Trans RGB | same 100 scenes | same 128 frame IDs | 12,800 |
| Combined | ConcatDataset, 1:1 count | **25,600 images/epoch** | 25,600 |
| Validation | Original test_seen 0100..0129 | 0,10,...,250 | 780 |

不是抽取50个scene；也不能因为25,600总帧数与100% Original相同，就把两个训练合同混为一谈。混合域的配对ID及两个SHA256需完全一致。

监督协议：
- Original RGB由RealSense提供，sample/workspace辅助使用RealSense depth；metric depth训练目标 rendered foreground + TSDF background。
- GN-Trans RGB为材质渲染域，sample/workspace使用virtual/rendered depth，graspness为virtual_graspness；metric depth训练目标默认full rendered（Mixed-v2）。只有显式GNTRANS_FUSED_BACKGROUND=1才替换background为TSDF。
- 两个域均使用global_film pose-aware RGB-to-depth，并将depth prediction detach后backproject成点云。训练期GT/observed/virtual depth不可直接进入网络预测的XYZ；预处理仍是depth-assisted，不能声称整个pipeline raw-RGB-only。
- Legacy使用standard economic_grasp_label_300views；CDF通过唯一CVAExtendedLabelAdapter读取economic_grasp_label_300views_extend_angle_cdf_depth（六阈值CDF、angle×depth width）。大object cache保留CPU，仅匹配行传GPU。
- 相同数据分布并不保证两种grasp head采用完全相同标签监督；CDF/Legacy是head+label package对照。和RayRoPE/GVAR比较时还需对齐depth target、crop、label和训练预算。

## 2. 文件与参数

主要统一入口：
- train_point_depth_pose_graspnet100.py: --point_head legacy|cdf; --train_domain original|mixed; --gntrans_rgb_root; --mix_train_fraction 0.5; --gntrans_fused_background（可选）。
- scripts/run_point_depth_pose_graspnet100_train.sh: TRAIN_DOMAIN=mixed, MIX_TRAIN_FRACTION=0.5, GNTRANS_RGB_ROOT, GNTRANS_FUSED_BACKGROUND=0, HEADS=legacy,cdf。
- inference_point_depth_pose_graspnet100.py 和 scripts/run_point_depth_pose_graspnet100_eval.sh: EXPECTED_TRAIN_DOMAIN=mixed；官方10%或full RealSense test。
- utils/point_depth_training_protocol.py: 严格校验 Original/mixed counts、frame hashes、head、e19；不再硬编码5,200或仅通过25,600来判断域。
- tests/test_point_depth_training_protocol.py: 测试原始100%、混合10%、混合50%、不同head、错误比例、错误hash和smoke checkpoint。
- 历史Legacy-only mixed脚本也可用 TRAIN_FRACTION=0.5；推荐双head统一脚本。100% Original默认训练路径仍保留。

Checkpoint依然使用兼容字段full100_protocol，但其中train_domain是来源的唯一有效判断依据；文件名不能代表训练源。两个版本必须分别strict-load e19 checkpoint。

## 3. Step 0：服务器、代码与数据安全审计

历史GPU服务器：robotarm@10.30.7.117。Codex须先检查当前连接/GPU/磁盘/工作树，不得操作其他正在运行的任务。

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
# 在确认目录不存在后，创建隔离 worktree（不要reset现有仓库）
git worktree add --detach ../EconomicGrasp-point-mixed50 origin/exp/point-depth-pose-cdf-graspnet100
cd ../EconomicGrasp-point-mixed50
~~~

静态/CPU测试：

~~~bash
python -m py_compile train_point_depth_pose_graspnet100.py inference_point_depth_pose_graspnet100.py inference_point_depth_pose.py utils/point_depth_training_protocol.py
bash -n scripts/run_point_depth_pose_graspnet100_train.sh scripts/run_point_depth_pose_graspnet100_eval.sh scripts/run_point_depth_pose_mix_train.sh
python -m pytest -q tests/test_point_depth_training_protocol.py tests/test_point_depth_mixed_contract.py tests/test_point_depth_cdf.py
~~~

人工审计：两个域相同100scene ×128 frames，Original与GN-Trans RGB、virtual_graspness、virtual depth、rendered/TSDF targets及标准/扩展grasp cache完整；CDF labels有效且object payload只由adapter加载，width mm→m不重复换算。报告每个域foreground depth target、sample count和hash。

## 4. Step 1：1→2→3 GPU smoke 与fallback

**先分别对两个域取真实batch验证forward/backward和监督标签。** ConcatDataset shuffle后仅训练4个batch不能保证两个域都进入这4个batch，因此data-only预检必须按域直接抽样；必要时补手动构造各域一个batch做梯度smoke。

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans

TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=2 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
MAX_TRAIN_BATCHES=4 MAX_VAL_BATCHES=2 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_mixed50_1gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=2 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
MAX_TRAIN_BATCHES=4 MAX_VAL_BATCHES=2 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_mixed50_2gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh

TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 HEADS=legacy,cdf MAX_EPOCH=1 BATCH_SIZE=3 NUM_WORKERS=0 EVAL_NUM_WORKERS=0 \
MAX_TRAIN_BATCHES=10 MAX_VAL_BATCHES=2 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_mixed50_3gpu_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

必须检查各rank的CUDA/ME设备、有限loss、grasp与FiLM gradients、CPU CDF labels、DDP optimizer和validation、保存checkpoint、rank同步。若3GPU失败则在两个head均通过的2GPU配置运行；2GPU失败回退1GPU。减小batch时必须两个head一致、重新smoke并报告global batch/steps变化。若单GPU也失败则停止正式训练，不伪造结果。

## 5. Step 2：正式Mixed50训练与10% Original官方AP

仅在smoke验收后执行，GPU IDs及输出路径必须按实际空闲资源改成未使用目录。两组head顺序运行，各训练21 epochs，固定e19作为评价checkpoint（e20归档）。

~~~bash
TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 HEADS=legacy,cdf MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

正式预测/评估仅Original RealSense：test_seen、test_similar、test_novel每个30scene×26frame=780，固定同样样本选择与sensor collision，CPU evaluator串行。两版本各自e19 checkpoint且训练manifest必须标train_domain=mixed。

~~~bash
export TRAIN_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_UNIQUE

# smoke，不运行AP
EXPECTED_TRAIN_DOMAIN=mixed GPUS=0 HEADS=legacy,cdf SPLITS=test_seen \
TEST_FRACTION=0.1 MAX_BATCHES=2 EVAL=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_mixed50_smoke_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

# 正式10%测试及官方AP
EXPECTED_TRAIN_DOMAIN=mixed GPUS=0,1,2 HEADS=legacy,cdf \
TEST_FRACTION=0.1 SPLITS=test_seen,test_similar,test_novel \
BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/eval/point_mixed50_ap10_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_eval.sh

python tools/summarize_point_depth_pose_ap.py \
 --root /data/robotarm/result/grasp/rgbgrasp/eval/point_mixed50_ap10_UNIQUE/cdf \
 --reference-root /data/robotarm/result/grasp/rgbgrasp/eval/point_mixed50_ap10_UNIQUE/legacy \
 --bootstrap 50000 \
 --output /data/robotarm/result/grasp/rgbgrasp/eval/point_mixed50_ap10_UNIQUE/comparison.json
~~~

输出官方finite shape [30,26,50,6]、Seen/Similar/Novel/Mean AP、AP0.4/0.8、Top1/5/10/20/50、scene-paired CI。报告两种head监督、GN-Trans point sampling与Original不完全相同、checkpoint depth质量可能不同等限制。不可拿训练数据不同的旧100%Original/GVAR模型计算直接单因子差值。

## 6. 交付验收

创建独立中文结果报告 POINT_DEPTH_RS50_GNTRANS50_LEGACY_CDF_RESULTS_ZH.md：每域样本+SHA、训练/val loss与depth MAE、两head checkpoint SHA、GPU smoke/fallback、模型推理耗时、full official AP、scene CIs和失败场景。仅使用真实结果；未完成部分标PENDING。

**状态：本回合仅在GitHub更新代码/工作单，未在远程GPU执行上述实验。**
