# GVAR P0/P1 — Codex 实验部署与诊断工作单（2026-10-10）

目标仓库 rcao-hk/EconomicGrasp；分支 exp/gripper-volume-action-reader；绝不更改 main。
**这是已实现的源码及待执行计划，不是服务器实验已完成的声明。**

## 科学目标及现有证据

- 现有 e19 五组基线：baseline 42.9234、slot 42.7963、volume_fixed 44.821、volume 44.5982、volume_rel 45.5176 Mean AP（%）。
- volume_rel - baseline：+2.5942 pp，[+1.996,+3.194] 场景级 bootstrap CI。
- volume - volume_fixed：−0.223 pp，[−0.754,+0.298]，未支持动态 insertion sampling 优势。
- volume_rel - volume：+0.9194 pp，[+0.484,+1.369]。
- Novel μ=0.4 对 baseline 仅 +0.0096 pp；负向场景 0165／0167／0174，正向对照 0160／0189。
- Scene bootstrap 只能衡量测试场景抽样差异，不覆盖训练随机性、多重比较或新模型独立生成动作的变化。

此轮 P0 是 **完整2×2关系机制消融**；P1 是 **Novel 失败帧＋gripper probe可靠性诊断**，不启动新的不确定性引导抓取训练。

## 0. 首要阻塞：核对服务器已经验证的修复

历史五组训练在服务器独立工作树上运行，存在最后记录的本地修复提交 625cc26：

~~~bash
# 服务器线索（实际路径需核实）
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp-gvar
git status --short
git branch --show-current
git rev-parse HEAD
git show --stat --oneline 625cc26
git fetch origin exp/gripper-volume-action-reader
git log --left-right --oneline origin/exp/gripper-volume-action-reader...HEAD
~~~

**不要假定远端实验分支包含服务器全部修复。** 不可强制 checkout/reset/stash 正在训练的工作区。建议新建 worktree，将正式训练已验证的代码修复与本次远端新增脚本合并／选择性移植。在该 worktree 用旧 volume_rel e19 checkpoint strict-load，并对已保存的若干固定帧进行 forward/decoder parity，确认旧变体不因移植意外变化。若不兼容，停止新训练并说明原因，不将不相同代码来源的 fixed_rel 与历史五组解释为单因素实验。

既有结果根目录：

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
~~~

## P0-A：不需 GPU，补齐 volume_rel − volume_fixed 的直接配对 CI

~~~bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007 \
OUT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007/analysis/p0_fixed_vs_rel_20261010 \
BOOTSTRAP=50000 SEED=20261009 \
bash scripts/run_gvar_p0_contrasts.sh
~~~

复用 analyze_gvar_scene_paired.py 的 scene-cluster bootstrap，在相同 30scene×26frames、相同 e19 官方 AP NPY 上，计算固定参照 Volume-fixed → Volume-rel 的 Seen／Similar／Novel／Mean ΔAP 及 95% CI，不可用两个独立 CI 相减替代。输出 CSV、场景统计和来源校验。**全五组已存在，因此无需重跑推理**。

## P0-B：训练第六组 volume_fixed_rel

本次修改：
- models/gripper_volume_reader.py 的 VARIANTS 增加 volume_fixed_rel。
- 与 volume_fixed 相同，所有插入深度从固定 d_ref=25mm 的采样位置取证据；与 volume_rel 相同，增加 action-relative predicted geometry。Insertion embedding 仍使用各个实际 d，geometry 在**固定支持位置**计算，不伪造与动态位置一一对应。
- 与其他 volume 系列保持相同 reader 参数量、36 probes/action、CDF/width、标签与监督；E/Q/C grasp→numeric depth 全 detach，depth head 保留直接监督。
- scripts/run_gvar_train.sh 支持新 variant，默认 per-GPU batch3、validation workers16；正式运行按已完成五组实际设置显式覆盖。
- 未添加 GT width、oracle center、KD、额外物理动作标签，也不改历史五组。

先进行单卡 batch3、一/两 batch、双卡 DDP smoke；确认 finite、梯度、unreferenced params、CDF维度、depth detach，GPU 显存。正式训练与真实五组协议一致：

~~~bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GPUS=0,1,2 VARIANT=volume_fixed_rel \
BATCH_SIZE=3 MAX_EPOCH=21 NUM_WORKERS=3 EVAL_NUM_WORKERS=4 \
ACTION_CHUNK=2048 ACTIVATION_CHECKPOINT=0 SEED=0 \
OUTPUT_ROOT="$ROOT/train/volume_fixed_rel" \
bash scripts/run_gvar_train.sh
~~~

既有五组实际训练 e0–e20 共21epochs，主比较固定 e19；每卡3，3GPU global batch9；Original 2600+GN-Trans 2600。需要确认 train index hashes、初始化、最初 lr、A1、global_film、fused background、collision protocol 与五组一致。只在输出目录为空且修复核验完成后启动。

正式推理：

~~~bash
DATASET_ROOT="$DATASET_ROOT" \
CKPT="$ROOT/train/volume_fixed_rel/checkpoint_epoch_019.tar" \
GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT="$ROOT/eval/volume_fixed_rel_e19" \
bash scripts/run_gvar_eval.sh
~~~

官方 Original GraspNet Seen/Similar/Novel 各 780 帧，shape=[30,26,50,6]，同一 checkpoint SHA256，collision-on threshold 0.01，GT depth **不作为网络几何**。

完整 2×2 消融之后：

~~~bash
python analyze_gvar_factorial.py \
  --root "$ROOT" \
  --output-dir "$ROOT/analysis/factorial_e19_20261010" \
  --epoch 19 --bootstrap 50000 --seed 20261009
~~~

对照矩阵：

| Support | Without relative geometry | With relative geometry |
|---|---|---|
| Fixed | volume_fixed | volume_fixed_rel |
| Dynamic | volume | volume_rel |

关键交互效应：
Interaction = (volume_rel − volume) − (volume_fixed_rel − volume_fixed)。

脚本输出所有 split、Mean 的 AP、摩擦阈值、prefix precision 主效应／交互效应以及场景级 CI。**需要直接评价交互效应，而非凭均值直觉判断两个模块能否协同。**

## P1-A：已完成 AP 结果中的 Novel 失败帧筛选（CPU）

~~~bash
python diagnose_gvar_novel_failures.py \
  --root "$ROOT" \
  --output-dir "$ROOT/analysis/novel_cases_20261010" \
  --variants baseline volume_fixed volume volume_rel \
  --reference baseline --comparison volume_rel \
  --scenes 165 167 174 160 189 \
  --worst-frames 3 --best-frames 1 --topk 10
~~~

对每个 scene，输出全部 26 帧的整体 AP、μ=.4、μ=.8、配对 delta，以及仅对选择的帧读现有 post-collision GraspGroup [N,17] dumps 的 score/width/insertion/camera-z 描述性统计。缺少 dump 时明确 unavailable；可用 --require-grasps 强制 fail-fast。

产物：novel_frames.csv、novel_scenes.csv、grasp_stats.csv、selected_frames.json、REPORT.md。**selection 在任何 uncertainty/probe 分析之前冻结**，不依据发现结果再挑选样本。

## P1-B：只对选定帧重跑诊断用 RGB forward，导出对齐的深度 sidecars

脚本 export_gvar_probe_sidecars.py，使用与正式 AP 完全相同的 checkpoint（自动做 SHA256 核验），GraspNet 官方 dataloader 裁剪后的 K、RGB、rendered/fused GT、sensor depth。模型 forward **只接收 RGB/K/pose/indices**，GT与sensor不会作为网络输入；runtime 再断言 used_geometry=predicted_depth。

先在**独立 smoke 目录**运行 --max-frames 1，然后正式输出完整选帧：

~~~bash
python export_gvar_probe_sidecars.py \
  --root "$ROOT" --dataset-root "$DATASET_ROOT" \
  --variant volume_rel --epoch 19 \
  --selection-json "$ROOT/analysis/novel_cases_20261010/selected_frames.json" \
  --checkpoint "$ROOT/train/volume_rel/checkpoint_epoch_019.tar" \
  --output-dir "$ROOT/analysis/novel_depth_sidecars_20261010" \
  --device cuda:0
~~~

输出数值单位 metre 的 pred_depth_m, gt_depth_m, sensor_depth_m、foreground_mask、K、crop_rgb、export_manifest.json。不得把 raw 1280×720 的深度与 448×448 的预测直接比较，也不能拿未经一致 crop 的外部 sigma 冒充网络 uncertainty。

如确有另一个模型输出的同 crop 对齐 predictive depth sigma，显式声明来源：

~~~bash
--uncertainty-map-root /path/to/aligned_sigma_maps \
--uncertainty-units m --uncertainty-source rgb_only
~~~

若使用 UA-Depth 的 observed-depth-conditioned sigma，则声明 --uncertainty-source depth_assisted_privileged，**仅作特权诊断**。没有外部 sigma 时只计算 predicted-depth local std/gradient 两个**启发式 proxy**，不声称 learned/calibrated uncertainty。

## P1-C：Probe 级物理位置投影与误差统计（CPU）

~~~bash
python analyze_gvar_probe_uncertainty.py \
  --root "$ROOT" \
  --selection-json "$ROOT/analysis/novel_cases_20261010/selected_frames.json" \
  --sidecar-root "$ROOT/analysis/novel_depth_sidecars_20261010" \
  --output-dir "$ROOT/analysis/probe_reliability_20261010" \
  --variants volume_rel --topk 20 --bins 10 --render-max 20
~~~

从保存的实际 **post-collision grasp poses** 读取 camera-coordinate center/R/insertion depth，按照该 variant 真正的 fixed/dynamic 36-probe reader 投影到同 crop 图像。仅在前景有效且 pred、rendered参考都有效的 probe 像素上计算 depth误差；另报 sensor参考误差以了解 CAD/rendered vs observed bias。输出 proxy vs error Spearman、risk–coverage、分位 bin、scene/frame/role 统计。若有外部 sigma，计算 Gaussian 1σ/1.645σ/1.96σ nominal coverage 的描述性检验；没有 sigma 不输出伪 coverage。可选保存 RGB/深度/error/probe overlay 的诊断图。

**采样点投影并非真实接触点**；可见深度不等于 GraspNet CAD/DexNet outcome。选取的 5 scenes 有 AP outcome-selection bias，不能凭这个子集断言全 Novel uncertainty 可靠或原因被证明。

如果需要比较 volume_fixed 与 volume_rel，先用同一 frame selector 分别导出两个 checkpoint 的 sidecars，并在 CPU 脚本 --variants volume_fixed volume_rel 中联合分析；脚本按变体分别匹配 checkpoint hash，不交叉混用不同模型预测深度。

## 执行顺序／验收

1. 先审计原五组训练的实际源码修复；保护用户工作区。独立 worktree 合入必要修复，确认 strict-checkpoint parity。
2. CPU 测试和 P0-A direct paired CI；整理 volume_rel vs volume_fixed 的 CI。
3. P1-A 原有 AP 的负向帧选择；P1-B 单帧 CUDA smoke；P1-C CPU probe calibration/可视化。
4. 若两 GPU smoke 和协议对齐成立，再启动 P0-B fixed_rel 21-epoch 训练，执行官方 e19 inference/eval 和 factorial。
5. 保存完整命令、Git SHA、GPU型号、运行耗时、输入文件 SHA256、输出路径、测试日志与 negative results；不要自动 push 训练产物或覆盖旧5组。

测试命令：

~~~bash
python -m pytest -q tests/test_gvar.py tests/test_gvar_scene_paired.py tests/test_gvar_p0_p1.py
python -m py_compile analyze_gvar_factorial.py diagnose_gvar_novel_failures.py export_gvar_probe_sidecars.py analyze_gvar_probe_uncertainty.py
bash -n scripts/run_gvar_train.sh scripts/run_gvar_p0_contrasts.sh
~~~

作者本地使用 CPU 合成 GraspNet 结构的 NPY/sidecars 完成 regression；真实 GPU 和原有服务器修复合并 **未执行**。Codex 不得在这些验收前宣称 complete 或 silently 修改数值定义。
