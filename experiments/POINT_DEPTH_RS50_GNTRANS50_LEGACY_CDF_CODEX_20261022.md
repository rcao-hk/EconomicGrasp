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

## 4. Step 1：直接启动三卡正式训练（不做分阶段 GPU smoke）

**本轮不进行单卡→双卡→三卡的 smoke 阶梯，也不需要先运行有限 batch 的 GPU smoke。** Step 0 的 Python 语法/CPU 单测、两域配对数量与标签缓存检查仍然保留，确认数据路径、GPU 空闲与独立输出目录后，直接以正式3卡协议启动 Legacy 和 Point-CDF 两组训练。

默认训练协议：RS50%（12,800帧）＋GN-Trans50%（12,800帧）、每epoch名义25,600帧、Original Seen 780帧 validation；3 GPUs × BS3，AdamW/cosine、21 epochs，固定 e19 评测。两个 head **顺序运行**且各自独立训练，不能共用或覆盖 checkpoint。

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans

TRAIN_DOMAIN=mixed MIX_TRAIN_FRACTION=0.5 GNTRANS_FUSED_BACKGROUND=0 \
GPUS=0,1,2 HEADS=legacy,cdf MAX_EPOCH=21 BATCH_SIZE=3 SEED=0 \
MAX_TRAIN_BATCHES=0 MAX_VAL_BATCHES=0 \
OUTPUT_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_UNIQUE \
bash scripts/run_point_depth_pose_graspnet100_train.sh
~~~

**不另做 smoke，但正式训练的早期日志必须正常：** 两个 head 的首批及后续训练 loss finite、DPT/FiLM 与 grasp loss 的输出合理、CDF 有效标签与 width 单位无异常、各 rank 没有 device mismatch、MinkowskiEngine 异常或 NCCL collective mismatch，checkpoint/validation 按既有脚本记录。这是正式训练期间的常规错误监测，**不作为额外预训练阶段或人为中断条件**。

**仅在实际出现故障时 fallback：**

- 若3卡 DDP/MinkowskiEngine/NCCL 运行失败，保留日志和已有输出并排查首个 root-cause 异常；确认环境后可尝试2卡训练；若仍失败再改1卡。**不要求每一种卡数事先跑 smoke。**
- 若需要降低 GPU 数或 per-rank batch，两个 head 的最终正式训练必须采用**同一个** GPU 数、batch、seed、数据与epoch合同。如果 Legacy 在3卡完成、CDF 只能2卡，则应在共同的2卡配置下重新训练 Legacy；不能将两个不同全局batch的结果作为严格匹配的主对照。
- 当前没有 gradient accumulation，改变3→2→1 GPU 会改变 global batch（BS3时为9→6→3）与 optimizer steps/epoch；记录最终训练设置，不得声称与默认3卡训练的优化过程等价。
- 单卡仍存在基础模型/数据错误时停止正式实验并明确标注 BLOCKED；不得以不完整训练的checkpoint作官方AP。失败或回退应选择新的输出根目录，不 reset、覆盖或清理原结果。

## 5. Step 2：正式10% Original RealSense inference＋官方评测

两组21 epochs正式训练完成后，使用各自固定 e19 checkpoint。**本轮不执行单独的两批 inference smoke**，直接在全新输出目录运行10% Original RealSense Seen/Similar/Novel 官方推断与AP；已有脚本会核查 checkpoint 的 head/domain/e19，以及每split完整780帧后才启动 evaluator。

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export TRAIN_ROOT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_UNIQUE

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

结果要求：Official finite AP tensor [30,26,50,6]，Seen/Similar/Novel/Mean AP、AP0.4/AP0.8、Top1/5/10/20/50与scene-paired CI。评测只对Original RealSense，同样帧选择、同样sensor-depth辅助workspace/collision与NMS。若训练中曾回退GPU配置，必须在报告中披露；不可拿训练域不同的旧100% Original或GVAR结果做单因子因果对比。
## 6. 交付验收

创建独立中文结果报告 POINT_DEPTH_RS50_GNTRANS50_LEGACY_CDF_RESULTS_ZH.md：每域样本+SHA、训练/val loss与depth MAE、两head checkpoint SHA、实际 GPU 数与global batch、实际故障/fallback（如有）、模型推理耗时、10% RealSense官方AP、scene CIs和失败场景。不额外提交1/2/3卡 smoke 报告；仅使用真实结果，未完成部分标PENDING。

**状态：仅更新本 Codex 工作文档的执行顺序；未在远程GPU启动训练、推理或评测。**
