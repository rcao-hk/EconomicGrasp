# Codex 工作文档：RayRoPE 新 Mixed Dataset — U0–U4 全量实验

> **目标：** 使用新的 source-specific mixed-depth supervision，在现有 RayRoPE / CVA-CDF 框架下，从头训练 U0–U4 五组模型。每组固定 20 epochs、seed=0 和完全相同的训练/测试预算；完成 10% GraspNet RealSense Seen/Similar/Novel 的 collision on/off 官方评价，生成经 NPY 复核的结果表和机制诊断。
>
> **禁止范围：** 不重训 B1/B2，不再运行 MoGe，不更换抓取表示/CDF decoder，不加 KD、center correction 或 score calibration，不占用/终止现有 100% RealSense 训练，不把旧 checkpoint 视为新 mixed U0。

## 0. 代码依据与工作区

- Repo: https://github.com/rcao-hk/EconomicGrasp
- 基础分支：**exp/moge-rayrope-grasp**
- 预期初始 commit：**7c2fa413c666be09502edb00d6380e1617ff0293**。执行前先 fetch + 检查当前 HEAD；如果远端已经有新提交，记录 diff 和是否影响实验合同，不悄悄覆盖。
- 现有入口：scripts/run_rayrope_mixed_p1.sh；训练：train_moge_rayrope.py；推断/评估：inference_moge_rayrope.py、eval_moge_rayrope.py。
- 数据来源实现：moge_rayrope/mixed.py；uncertainty：moge_rayrope/objective.py、config.py；数据诊断：scripts/diagnose_rayrope_paired_depth.py；现有说明：experiments/RAYROPE_MIXED_P1.md。
- 既有 7 组的详细执行记录据用户描述在 Windows 路径 D:/Research/Paper/RGB-Only Grasp/results/rayrope_datascale_20261007/EXECUTION.md，**不能假设该路径在服务器可访问**。如果有同步副本或历史 protocol.json，必须核实 batch、种子、采样、网络与 evaluation 的实际参数；缺失的项目标注“未独立核实”，不得猜测。

服务器已有仓库可能存在未提交修改或正在进行的 100% RS 训练。**使用新的 git worktree**，不要 stash/reset 原仓库，不要修改现有正在运行的工作树和 checkpoint，也不要在非空结果目录上重新开始。

~~~bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin exp/moge-rayrope-grasp

# 如果该目录已存在，先审计，禁止覆盖。
git worktree add --detach /home/robotarm/EconomicGrasp-RayRoPE-Mixed-P1 \
  origin/exp/moge-rayrope-grasp
cd /home/robotarm/EconomicGrasp-RayRoPE-Mixed-P1
git rev-parse HEAD
git status --short

export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
nvidia-smi
df -h /data2/robotarm/result
~~~

先核对 DAV2 权重软链接及 SHA、依赖环境、config 和 runtime 的 pinned-main 校验；如独立 worktree 缺权重，只链接已验证的**同一**预训练权重。用户曾限制 GPU4，不使用 GPU4；优先 GPU 0/1/2，如果仍被 100% RS 等任务占用，只能在确认空闲并记录设备配置后改用允许的 3/5/6，始终保持 **3 张卡 × 每卡 batch3**，不得抢占正在进行的任务。

## 1. 数据合同：这次唯一允许变化的数据源

| 训练域 | RGB | GT depth（完整图像） | 帧数 |
|---|---|---|---:|
| GraspNet RealSense | 原始 RealSense RGB | GraspNet **full TSDF depth**：tsdf_depth/scene_xxxx/realsense/xxxx_depth.png | 2600 |
| GN-Trans | GN-Trans rendered RGB | **full rendered depth**：virtual_scenes/scene_xxxx/realsense/xxxx_depth.png | 2600 |

- 100 个 train scenes；每个 scene 对应 frame 0、10、…、250，共 26 张，两个域的 (scene, frame) 必须一一对应；训练总数 **5200**，不是混合 10% 后变成 2600。
- **严禁**让 GN-Trans 背景深度 fallback 到 GraspNet TSDF；更不能沿用旧的“rendered object + TSDF background”融合。
- **注意**新的 RealSense 监督是全图 TSDF，不仅是背景 TSDF；因此新 U0 与历史的 mixed-point 49.3792 AP 具有不同 depth target，**必须从头训练**，旧 AP 只作为历史参考。
- 网络保持 RGB-only：训练 GT、sensor depth、TSDF depth 不得作为神经网络观察输入；K / camera pose / crop 等元数据仍按照既有实现。真实图像与 rendered 图像不一定采用同样 crop，禁止在不同坐标系的 cropped depth 上直接逐像素求差。
- validation 仍然 **原始 RealSense Seen 780 帧**，使用新 RS/full-TSDF depth target；只做训练诊断，不看 Similar/Novel AP 来调参数或选择 epoch。
- 现有上层 USE_FUSE_DEPTH=1 只是共享 runtime 选项；是否符合上述合同，以 mixed.py **实际读取的 source-specific 文件路径**为准。

## 2. 不得改变的训练/测试配置

| 项目 | 固定协议 |
|---|---|
| training fraction | RS 10% + GN-Trans 10%，100 train scenes 各 frame stride=10 |
| optimizer / schedule | AdamW、LR 3e-4、weight_decay 1e-3、cosine-by-epoch，grad clip 1.0 |
| epochs / seed | **20 epochs，seed=0**，固定最后的 zero-based epoch 19 checkpoint；不基于 AP early stop |
| distributed | 3 GPUs，batch/GPU=3，effective batch=9；无梯度累积 |
| encoder / depth | Frozen DAV2 ViT-B、pose_mode=global_film；原 metric depth head（MoGe OFF），预测范围 0.2–1.0m |
| grasp representation | image-FPS seeds=1024；A1 / top-1 view；CVA Transformer + evaluator-aligned CDF score + depth-wise width；不启用新 collision head / top4 view |
| RayRoPE grouping | 同一 7×7 fixed image-space neighborhood，radius 40px，virtual grasp query frame；RAY_APPLY_VO=1；GROUP_CHUNK=32（允许只为 OOM 调 chunk，不改变数学目标） |
| label / loss | 同一 economic_grasp_label_300views_extend_angle_cdf_depth；原 objectness/graspness/view/CDF/width loss 系数全部保留 |
| testing | test_seen / test_similar / test_novel，**每 split 780 帧**，即 30 scenes × 26 frames；评估全为原始 GraspNet RealSense RGB |
| inference | **INFER_BATCH_SIZE=3 必须显式设置**。通用脚本默认 1，与此前最佳 mixed-point 实验使用的 3 不同；先做同模型 batch=1/3 的一致性 smoke |
| collision | COLLISION=both：on 主结果 / off 诊断；COLLISION_THRESH=0.01，COLLISION_VOXEL_SIZE=0.01m，COLLISION_APPROACH_DIST=0.05m |
|其它 | SPLITS=test_seen,test_similar,test_novel；DEPTH_BIAS_MM=0；INFER_MAX_FRAMES=0；训练/推断采样、K/crop、输出格式和 GraspNet evaluator 不变 |

**执行前需核实的混杂项：** 当前原 depth L1 使用 GT 范围 [0.2, 1.0]（含边界），而 U2–U4 代码中用于 confidence/interval 的 valid mask 使用开区间 (0.2, 1.0)。检查这两个边界的有效像素数量；若非零且会导致 U0–U4 有效监督样本不同，正式运行前统一 mask 定义、加回归测试并冻结同一个新 commit。绝不在某一 variant 的训练中途修复。

## 3. U0–U4 唯一允许的模型差异

| ID | Ray encoding | depth interval | Uncertainty/depth objective |
|---|---|---|---|
| **U0** | point | 不使用 uncertainty（零 halfwidth） | 原始 depth L1 |
| **U1** | expected | fixed ±20mm | 原始 depth L1 |
| **U2** | expected | learned 1–80mm | 90% central interval score + 原 depth L1；depth residual detach |
| **U3** | expected | learned 1–80mm | decoupled Laplace NLL + 原 depth L1；depth residual detach |
| **U4** | expected | learned 1–80mm | coupled Laplace confidence-weighted depth regression，**替代**原 depth L1 |

- U2–U4 使用相同 sigma head 和相同 RayRoPE interval operator。U3/U4 中 learned 90% halfwidth 与 Laplace scale 关系为 h90 = ln(10)·b；具体 normalizer / loss 权重以冻结代码实现为准，记录在协议。
- **仅 U4** 允许 uncertainty-aware depth loss 回传到 depth head；U0–U4 的 **grasp loss → numeric depth / sigma 必须始终 detach**。
- U4 改变的是 depth regression loss 的梯度，而不是通过 grasp loss 重新训练 depth；遇到 checkerboard、depth collapse、非有限 loss 时先诊断，不从已有失败 checkpoint 报正式 AP。
- 确保由 scripts/run_rayrope_mixed_p1.sh 锁定 variant flags；不允许外部环境覆盖 USE_MOGE、RAY_ENCODING、UNCERTAINTY_LOSS 等核心开关。

## 4. P0-A：数据预检（不可跳过）

使用已有的 paired-depth diagnostic，先检查全部 2600 pairs，至少 8 对实际 loader 的 crop/K；同时针对 RS full TSDF 和 GN full rendered 分别验证 loader 输出的 gt_depth_m。无法访问或路径不同就停止，不静默回退。

~~~bash
export GPUS=0,1,2               # 仅当这些 GPU 空闲
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3
export GROUP_CHUNK=32
export SEED=0
export EPOCHS=20

P1_VARIANT=U0 PHASES=diagnose \
  DIAG_MAX_PAIRS=0 DIAG_LOADER_AUDIT_PAIRS=8 \
  bash scripts/run_rayrope_mixed_p1.sh
~~~

检查 paired_depth.csv / paired_depth.json 中的 scene/frame 匹配、RGB/depth 形状、相机单位、crop/K、一致性/有效像素比例、前景/背景差异、TSDF/rendered 文件的绝对路径。**不同渲染深度本来可能不一致**；关注其差异分布和 RGB 与对应 GT 的几何一致性，不要求两域深度逐像素相同。对不同 crop，在 native-resolution 坐标比较，不拿各自 crop 后的相同数组下标求差。

**停止条件：** 输入资源缺失、标签/CDF 不可用、有效像素严重不足、RGB/GT 坐标错位、单位错配、GT 来源与新 policy 不符、paired schedule 不是 2600+2600。数据通过前禁止开始 GPU 正式训练。

## 5. P0-B：代码测试、梯度合同与 DDP smoke

~~~bash
bash -n scripts/run_rayrope_mixed_p1.sh scripts/run_moge_rayrope20.sh
"$PYTHON_BIN" -m pytest -q \
  tests/test_rayrope_mixed_p1.py tests/test_moge_rayrope.py
~~~

随后对五组分别跑隔离的 **1 epoch / 2 optimizer-step smoke**，每组单独输出目录，避免覆盖和混用 checkpoint：

~~~bash
for v in U0 U1 U2 U3 U4; do
  echo "[SMOKE] $v"
  P1_VARIANT="$v" PHASES=train EPOCHS=1 \
    MAX_TRAIN_FRAMES=18 MAX_VAL_FRAMES=9 MAX_STEPS=2 \
    WORK_ROOT="/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1_smoke/$v" \
    bash scripts/run_rayrope_mixed_p1.sh
done
~~~

逐项确认：

1. 所有 U-ID 至少两步训练、一次 distributed validation、一次 checkpoint 写入；train、val 的 CDF/width/depth loss 和 gradient norms 有限。
2. 原有验证函数输出 grasp_to_geometry_max_abs 为零（容差沿用已有断言），DAV2 frozen 不累计梯度。
3. U2/U3：sigma head 有 uncertainty-supervision 梯度；其 uncertainty loss **不更新** depth head。U4：coupled NLL 对 sigma/depth head 都有梯度；grasp loss 仍无法更新 depth。
4. DDP sampler 不在同一验证集重复样本；正式 val 采用 780 个 RealSense Seen 帧。debug 检查不允许使用测试 AP 选 checkpoint。
5. 另跑相同 checkpoint、相同三组中极少量样本的 batch=1 与 batch=3 forward+decode 比较，确认结果等价或给出具体差异。正式五组 **全部用 batch=3**；如果存在不可解决的显存问题，必须**正式运行前**统一调整并明确与旧 mixed AP 的比较限制。
6. 检查 learned sigma 是否大量饱和到 1mm 或 80mm、是否存在 NaN/Inf、U4 是否 depth collapse / checkerboard。出现不可解释异常先修复，不得以 smoke 成功代替正式训练。
7. 任何必要代码修复都先完成、测试、提交为同一冻结 commit；**冻结后所有五组使用完全相同 source hash**，不得在正式运行过程中修改 geometry/model/loss/trainer。

smoke checkpoint 是 partial run；官方 infer/evaluator 必须拒绝其作为完整结果。

## 6. P1：正式执行五组 train → infer → eval

先确认原 100% RealSense 训练没有占用目标 GPU；同一 GPU pool 内五组**串行**运行，保证互不抢资源。全部结果使用新的 root，不初始化于旧 mixed-point checkpoint。

~~~bash
cd /home/robotarm/EconomicGrasp-RayRoPE-Mixed-P1

export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python

export GPUS=0,1,2                 # 仅确认空闲后使用，GPU4 禁止
export BATCH_SIZE=3
export INFER_BATCH_SIZE=3        # 覆盖原脚本默认为 1
export GROUP_CHUNK=32
export EPOCHS=20
export SEED=0
export COLLISION=both
export SPLITS=test_seen,test_similar,test_novel
export DEPTH_BIAS_MM=0
export INFER_MAX_FRAMES=0
export RESUME=0
unset WORK_ROOT TRAIN_ROOT TEST_ROOT || true

for v in U0 U1 U2 U3 U4; do
  echo "===== FORMAL $v ====="
  P1_VARIANT="$v" PHASES=train,infer,eval \
    bash scripts/run_rayrope_mixed_p1.sh
done
~~~

预计各组独立结果 root：

- /data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U0
- /data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U1
- /data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U2
- /data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U3
- /data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_p1/U4

每组以第 20 轮 checkpoint_latest.pt 进行推断和评价，不使用测试集或 Seen validation AP 选择最佳 epoch。中断后仅在 signature、checkpoint、source hash 完全相同的情况下使用 RESUME=1；不要删除输出目录、绕过 signature check 或从 smoke checkpoint 假装恢复。若 U4 non-finite，保存诊断信息，标记 FAILED/PARTIAL，不伪造完整 AP。

## 7. 正式结果验收条件

### 7.1 训练

逐组审计 train/protocol.json、train/metrics.json、train/checkpoint_latest.pt：

- train_frames=5200 (2600/2600)，val_frames=780，train/eval schedule & hashes 相同。
- 20 个完整 epoch，最后 epoch==19；seed0、3GPUs×batch3、same label folder、optimizer、schedule、frozen DAV2 权重 hash。
- RS 监督只能来自 full TSDF，GN 监督只能来自 full rendered，且两域 depth 不作为 model input。
- U-ID 对应的 ray_encoding / uncertainty / uncertainty_loss 精确一致；U4 不重复包含原始 depth L1；模型无 MoGe。
- 不出现 depth collapse、NaN/Inf 或未审查的 checkpoint 复用。

### 7.2 推断 + GraspNet 官方评价

官方 eval 由 eval_moge_rayrope.py 调用 graspnetAPI，按以下要求核对：

- test_seen/test_similar/test_novel **各 780 张**，每个 split 固定相同 scene/frame。
- 对每组 collision on/off 均有完整 dump、completed markers 与 summary；on/off 从同一次 forward 产生；on 是 model-free collision filtering 后的结果。
- 每帧输出合法且有限的 GraspNet N×17 grasp array；各 split 的 official accuracy.npy **形状为 (30,26,50,6)**，并与 summary 重算一致。
- collision on/off 采用相同 voxel size 0.01m / threshold 0.01 / approach distance 0.05m；正常 depth bias=0，无测试时额外评分/筛选。
- **Mean AP 是三个 split AP 的算术平均**（单位 %）；最后一列报告 collision-off 的同样均值。不得混用前一轮不同 depth policy 的数值作为 U0。

## 8. 比较与额外诊断（无需新训练）

现有 compare_moge_rayrope.py 默认标题仍写“20% GraspNet”，并且仅以名为 baseline 的组生成 scene-paired 对照。Codex 应单独修改**报告脚本**（不动已冻结的 model/trainer），把新 U0 设为 reference，并增加数据合同检查：gntrans_rgb_root、mixed_depth_supervision、paired sampling_sha256、train/val counts、model encoder/pose_mode、checkpoint epoch、inference batch、collision 参数、source/dav2 hash。只允许 U0–U4 的 encoding/uncertainty objective 不同。

结果从官方 NPY 计算，不依赖抄录日志，至少包括：

| U-ID | Seen(AP%) | Similar(AP%) | Novel(AP%) | Mean collision-on | Mean collision-off | ΔMean vs **new U0** |
|---|---:|---:|---:|---:|---:|---:|
| U0 | measured | measured | measured | measured | measured | 0 |
| U1 | measured | measured | measured | measured | measured | measured |
| U2 | measured | measured | measured | measured | measured | measured |
| U3 | measured | measured | measured | measured | measured | measured |
| U4 | measured | measured | measured | measured | measured | measured |

重点分组差值：

- U1−U0：fixed interval 与 point encoding。
- U2−U1：学习空间变化的 interval 是否有额外作用。
- U3−U2：interval-score vs decoupled Laplace supervision。
- U4−U3：confidence 与 depth regression 梯度耦合的影响。

对 U2–U4，整理 interval empirical 90% coverage、mean halfwidth、上下界 saturation、all-valid/foreground depth MAE、CDF-ranking metrics；若现有 val 只有 RS Seen，就只报告 RS Seen 的 uncertainty calibration，不声称已完成 GN-Trans 或 Novel confidence 验证。输出 paired-scene ΔAP 和 bootstrap CI 作为测试样本的变化范围，**不能把其当作多训练 seed 显著性证明**。

历史 best point+mixed AP：Seen 71.3253 / Similar 55.9849 / Novel 20.8274 / Mean 49.3792。它来自不同 depth supervision 和先前优化后的代码，仅在报告末尾标成 **historical protocol reference**，禁止直接替代新 U0，也不能用于归因新 U1–U4 的效果。

## 9. 最终交付物与汇报规范

在统一结果 root 下输出：

- **P0_DATA_AUDIT.md**：新 mixed depth path、paired-frame/crop/K、前景/背景差异、有效像素和任何阻断项。
- **P0_SMOKE_REPORT.md**：CPU/CUDA/DDP 测试、每组 gradient contract、batch=1/3 一致性、有限性、修复记录。
- **U0_U4_RESULTS_CN.md**：所有完整/失败训练及实际 AP、uncertainty quality、相邻对照结论、计算与效率、局限。
- **U0_U4_COMPARISON.csv**、**U0_U4_COMPARISON.json**、必要的 **U0_U4_SCENE_PAIRED.csv**。
- 每组的 train/checkpoint、training protocol & metrics、三 split×collision on/off 的原始 grasp dumps、official/accuracy.npy、official/summary.json。
- 如有 bugfix，提供冻结 commit、diff、受影响的测试重跑与输出是否作废的声明；轻量报告可提交 GitHub，**不上传大 checkpoint 与 grasp dumps**。

必须在最终响应标出五组各自 **DONE / PARTIAL / BLOCKED**、最后有效 epoch、官方测试是否完整、结果根目录、GPU/环境/commit 与下一步。不可虚构训练、AP、校准质量或宣称未跑过的测试通过。

## 10. 明确排除

- 不执行 B1（5200 RS-only 改版）、B2（repeat original RGB）或数据量 sweep。
- 不重做 MoGe、MoGe+RayRoPE。
- 不增加其它 random seeds、KD/teacher、shape tokens、center correction、SpatialEnhancer gating 或 grasp-ranking head。
- 不干预用户正在进行的 100% RS 全量训练。
- 不基于测试集选择 depth uncertainty 超参、checkpoint 或修改碰撞过滤协议。
