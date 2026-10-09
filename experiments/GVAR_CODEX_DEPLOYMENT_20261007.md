# GVAR：Gripper-Volume Action-Conditioned Reader 实验部署

日期：2026-10-07。本文是 **Codex 执行工作文档**，不是已完成训练结果。

## 1. 任务与边界

仓库：`rcao-hk/EconomicGrasp`。
分支：`exp/gripper-volume-action-reader`。
起点：`main@52d09f925059bec3643610ecf1f1722894627ee5`。

从该分支部署、验证和运行已实现的实验，不要只返回建议。不得改动用户
未提交文件、停止无关训练、清理其他结果或把本分支合并回 main。
先读取服务器祖先目录和仓库中的 AGENTS.md。所有命令、结果必须留档。

固定实验合同：

- 训练：GraspNet 原始域每 scene 取 frame `0,10,...,250`（2600 images）
  + 同样 10% GN-Trans（2600 images），共 5200 images/epoch。
  两域独立抽样，再 ConcatDataset；不做配对 consistency。
- validation：**只有 Original GraspNet test_seen 的相同 780 frames**。
  不用 GN-Trans/Similar/Novel 选择 checkpoint。
- 最终测试：Original GraspNet Seen/Similar/Novel 各 780 frames。**本轮不跑 GN-Trans AP。**
- RGB predicted metric depth；pose-aware `global_film`；image-FPS；A1 随机 view
  训练 / Top-1 view 推理；CDF + depth-wise width；完整原监督。
- 三路全 detach：E/GSE 几何、Q/seed backprojection、C/support depth values。
  depth head 的直接 metric-depth loss 仍保留；原有可训练视觉/抓取模块继续训练。
- 不加 KD、repair、center offset、confidence gate、新 collision head 或在线 Dex-Net。
- 正式训练固定 **per-GPU batch size=3**；3GPU 时 global batch=9。训练期 validation workers=16。\n- 正式 inference 固定 **batch size=3 / GPU**；官方 evaluator workers=16。
- 主 AP 使用原始 sensor cloud、collision threshold=0.01、voxel=0.01。
  这叫 RGB-only network + 原协议 depth-assisted preprocessing/filtering；
  不能写成 raw-RGB-only 整体系统。

## 2. 已实现的 variations

| VARIANT | 表示改动 | 首轮作用 |
|---|---|---|
| baseline | 原 CVA-CDF，显式统一 E/Q/C detach | 同协议基线 |
| slot | angle feature 增加 insertion embedding + FFN；每个 d 有自己的特征 | 动作索引控制 |
| volume_fixed | slot + 独立 pre-GSE RGB gripper reader，所有 d 的采样均固定在 d_ref=25 mm | 参数/采样量完全匹配的关键控制 |
| volume | 同上，但采样位置随实际 insertion bin d 改变 | 主实验 |
| volume_rel | volume + action-relative predicted geometry | 几何兼容性增量 |

`volume_fixed / volume / volume_rel` 的参数、36 probes/action、attention 结构完全相同。
非 rel variants 把 6D geometry input 置零，但执行同一 geometry projection，
避免 DDP 未使用参数、也保留相同参数容量。slot 的参数/计算量小于 volume；
**因此 volume vs slot 不是精确的容量匹配对照，volume vs volume_fixed 才是。**

所有新 variants 保留原来的 metric angle-grouping 和 angle context。
新 reader 增加独立 pre-spatial-enhancer proposal DPT feature 读取，作为内部
representation residual；不是在 frozen Stage-1 score 后乘 scalar gate。
各自重新训练后候选当然可能不同，不可宣称这些 AP 比较是 fixed-candidate ranking。

### 2.1 Gripper 坐标和信息语义

沿用 `utils/collision_detector.py`：R 的列为局部轴，x=approach，y=opening，z=height。
每个 insertion bin `d_idx` 对应 `(d_idx+1)*0.01 m`，不是 camera-z center offset。
默认 finger length=0.06 m、thickness=0.01 m、height=0.02 m、approach=0.05 m。

36 个 probes，6 roles × 6 points：left contact、right contact、closing、finger body、palm、approach。
finger 沿 x 的区间是 `[d-0.06,d]`，palm 在其后方，approach 更靠后。
中心和旋转不被 reader 修正。

**本版 envelope width 固定为 0.06 m**，不是预测 width，也不是 GT width。
width 仍由原 depth-wise output weights 预测。
因此这是 insertion-conditioned finite-gripper-envelope reader，尚不是对任意
完整 `(t,R,d,w)` 提供 exact utility 的 evaluator。不能声称已解决 width/action-label mismatch。

volume_rel 输入：预测表面在动作坐标系中的 xyz、signed ray-depth residual、
可用深度标志、投影有效标志。它们是 soft cues：
投影有效不等于可见，缺少深度不等于 free space，遮挡后方不被赋予确定 occupancy 标签。

训练标签仍是 main 的 CVA label adapter/matcher，不新建 cache，不平移旧 grasp labels。
读出 envelope 的改变不授权把 nearby arbitrary action 视为已有 exact-action label。

## 3. 代码文件

- `models/gripper_volume_reader.py`：纯 PyTorch geometry/probes/reader/shared output ops。
- `models/economicgrasp_gvar.py`：原模型接口适配，pre-GSE hook，Q detach，新 CDF decoder。
- `utils/gvar_runtime.py`：独立 CLI 预解析、checkpoint/manifest 合同。
- `train_gvar.py`：继承已有 mixed trainer、DDP、原 loss；原始 Seen-only validation。
- `inference_gvar.py`：checkpoint 自动重建 variant，原始 GraspNet 10% 推理。
- `scripts/run_gvar_train.sh` / `scripts/run_gvar_eval.sh`：启动、日志、并行 inference / 串行 eval。
- `summarize_gvar.py`：只接收完整 `[30,26,50,6]` AP tensors，自动汇总。
- `tests/test_gvar.py`：CPU geometry/shape/gradient/CLI tests。

当前作者环境完成了纯 CPU 单测和语法检查；**尚未运行真实 CUDA extensions、
数据集 forward/backward、DDP 训练或官方 AP**。以下 smoke 必须先完成。

## 4. 部署前检查（不得跳过）

### 4.1 保留工作区

```bash
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin
git rev-parse origin/exp/gripper-volume-action-reader
```

现有目录若有未提交修改或正在跑其他分支，优先创建独立 worktree：

```bash
git worktree add --detach ../EconomicGrasp-gvar origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar
```

不强行 reset / stash / checkout 用户工作目录。
若需要修复 bug，在 worktree 新建本地修复分支，记录 diff 和 commit，未经要求不要 push main。

### 4.2 环境和数据

复用服务器当前能运行 CVA-CDF 的 conda/Python 环境；不要新装一套 PyTorch
替换 CUDA extensions。检查 GPU 空闲情况、GPU型号、CUDA/PyTorch/extension版本。
记录 torch、numpy、graspnetAPI 版本与 git SHA。

默认路径（必须核实存在，不存在自行检查已配置路径）：

```text
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
results=/data2/robotarm/result/grasp/rgbgrasp/gvar_10pct
```

需要原 GraspNet train/test 原图、meta、seg、graspness、virtual/TSDF depth；
GN-Trans 0–99 scenes RGB；现有 extend_angle_cdf_depth label cache。
主训练用 fused background。若文件缺失，报告/补齐正确路径，不静默关闭 fused target。

运行已有检查脚本（它会检查所有域/splits 的文件，但不会训练或跑 GN-Trans AP）：

```bash
python check_gntrans_mix_data.py \
  --dataset_root "$DATASET_ROOT" --gntrans_rgb_root "$GNTRANS_RGB_ROOT" \
  --fraction 0.1 --check_all_selected --load_one --output /tmp/gvar_data_check.json
```

重点检查 train 两域各2600、seen各780；新 trainer 最终只保留原始 seen validation。
不要把 `use_gt_depth=True` 的 GN-Trans dataset preprocessing 误解为网络使用GT；
模型内网络 geometry 强制 pred，并且 inference 运行时验证 used_depth=depth_net_pred。

## 5. 单测与 smoke

```bash
python -m pytest -q tests/test_gvar.py
bash -n scripts/run_gvar_train.sh
bash -n scripts/run_gvar_eval.sh
```

先 dry-run，不启动训练：

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
VARIANT=volume DRY_RUN=1 bash scripts/run_gvar_train.sh
```

对每个 variant 先单 GPU、两 batch smoke；使用全新的输出目录：

```bash
for v in baseline slot volume_fixed volume volume_rel; do
  DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
  VARIANT="$v" GPUS=0 BATCH_SIZE=1 MAX_EPOCH=1 MAX_BATCHES=2 \
  ACTION_CHUNK=128 NUM_WORKERS=1 EVAL_NUM_WORKERS=0 \
  OUTPUT_ROOT="/data2/robotarm/result/grasp/rgbgrasp/gvar_smoke/$v" \
  bash scripts/run_gvar_train.sh
done
```

接受条件：loss有限、backward/optimizer完成、所有variants保存完整checkpoint、
CDF shape为 `[B,6,Q,12,4]`、width `[B,4,Q,12]`；无 forbidden legacy aliases。
`D: GVAR E/Q/C detach` 都是1；depth loss有梯度；不把全部depth_net冻结。

对 volume 和 volume_rel 再做**双GPU smoke**，确认 no unused-parameter/collective hang。
随后至少对 `volume` 做一次 **BATCH_SIZE=3 的单GPU一-batch显存 smoke**，确认正式 per-GPU batch3
可以运行；若 OOM，先降低 `ACTION_CHUNK`，不要直接把正式 batch 改回1。
DDP 的 variable-length label lists 必须保持 CPU；不要改 base 的 device_ids=None。

使用 smoke checkpoint 做推理 smoke，不运行 evaluator：

```bash
DATASET_ROOT="$DATASET_ROOT" CKPT=/path/to/smoke/volume/checkpoint_latest.tar \
GPUS=0 SPLITS=test_seen MAX_BATCHES=2 RUN_EVAL=0 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/gvar_smoke_infer/volume \
bash scripts/run_gvar_eval.sh
```

检查 `.npy` 17 columns、有限数值、正确 scene/frame 命名；inference 默认 batch size=3；summary complete=false
是 smoke 的预期结果，不能将部分样本当作正式AP。

## 6. 正式训练

默认每 variant **3GPU × per-GPU batch3，global batch9**，20epoch，AdamW lr1e-4、
cosine schedule、weight decay0、grad clip1，沿用原始监督权重。训练期 Original Seen validation 的
`EVAL_NUM_WORKERS=16`。由于 batch size 相比历史 mixed baseline 发生变化，**baseline 也必须在这一
batch3/global9 协议下重新训练**；历史 batch1/global3 checkpoint 只能作背景参考，不能作为严格的
GVAR 架构因果对照。
DINO使用原模型冻结设定；其余原本可训练模块继续训练。
从 pretrained DINO/DPT 的正常 task initialization 起跑，**不加载 Stage-1/P5/MGF checkpoint**。

推荐先 baseline/slot/volume，随后 volume_fixed 和 volume_rel 补齐因果比较。
若资源允许也可先全部运行，但每个配置必须使用相同GPU数/global batch和训练计划。
6GPU条件下可两个训练并行（0,1,2 和 3,4,5），每个训练仍保持 per-GPU batch3 / global batch9；
不要把某个配置擅自改成6GPU，否则 global batch 会变为18并破坏受控比较。

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GPUS=0,1,2 VARIANT=volume BATCH_SIZE=3 MAX_EPOCH=20 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/gvar_10pct/train/volume \
bash scripts/run_gvar_train.sh
```

依次将VARIANT换为 baseline/slot/volume_fixed/volume_rel，结果目录单独命名。
参数与命令写入执行日志，脚本同时保存父目录下 `${OUTPUT_ROOT}.console.log`。
不能把 smoke 输出目录用于正式训练。

输出：

```text
log_train.txt
gvar_protocol.json
gntrans_mix_protocol.json
gvar_epochs.jsonl
checkpoint_latest.tar
checkpoint_best_val_loss.tar
checkpoint_epoch_004.tar
checkpoint_epoch_009.tar
checkpoint_epoch_014.tar
checkpoint_epoch_019.tar
```

`epoch` field=下一个epoch，`completed_epoch`=刚完成的0-based epoch。
checkpoint内包含variant、reader参数、完整模型结构参数、detach policy、optimizer和RNG。
主模型比较**预先固定 e19**。e9/e14只用于学习轨迹，不从Novel挑最好checkpoint。
`best_val_loss`不是“best AP”，不要用不同选择策略美化某一个variant。

断点续训（同variant/结构、per-GPU batch3 / global batch9，启动脚本其他设置保持不变）：

```bash
VARIANT=volume GPUS=0,1,2 BATCH_SIZE=3 EVAL_NUM_WORKERS=16 \
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
RESUME_CKPT=/path/to/volume/checkpoint_latest.tar \
OUTPUT_ROOT=/path/to/volume bash scripts/run_gvar_train.sh
```

保存了main进程RNG，CUDA算子与persistent DataLoader worker状态仍不保证逐bit复现。
不能把此记录解释为多seed统计保证。

## 7. 正式推理与 AP

对每个配置的 e19：

```bash
DATASET_ROOT="$DATASET_ROOT" \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/gvar_10pct/train/volume/checkpoint_epoch_019.tar \
GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=2 EVAL_NUM_WORKERS=16 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/gvar_10pct/eval/volume/e19 \
bash scripts/run_gvar_eval.sh
```

- 一个split一个GPU；GPU不足时分wave；每split780 frames；**每个 inference 进程 batch size=3**。
- 自动读取checkpoint配置，不手写不同的reader/pose/head flags。
- 推理是 `sample_interval=0.1`；官方 `eval.py` 是 `sample_interval=10`。
- 三split推理完成后串行CPU evaluation；**每个官方 evaluator 使用 16 workers**，避免同时启动3个 16-worker CPU pool。
- `RUN_INFERENCE=0 RUN_EVAL=1` 可只评测已完整dump。
- `RUN_INFERENCE=1 RUN_EVAL=0` 仅推理。
- `RESUME_INFERENCE=1` 只允许同manifest重跑，跳过已验证的同帧dump。
- `REMOVE_DUMP=1` 仅在AP保存且shape/finite验证通过后，删除对应scene/*.npy。
  AP tensors、manifest、日志、summary全部保留；默认不删除。
- `MAX_BATCHES>0` 必须配 `RUN_EVAL=0`，不允许partial AP。
- inference loader只去掉未使用的depth-prob等training payload，保留原 crop/index
  和sensor-cloud filtering语义，不能临时改成GT collision cloud。

## 8. 必须返回的结果表与分析

运行：

```bash
python summarize_gvar.py \
  --root /data2/robotarm/result/grasp/rgbgrasp/gvar_10pct/eval \
  --output /data2/robotarm/result/grasp/rgbgrasp/gvar_10pct/summary
```

表1：variant × Seen/Similar/Novel/Mean AP，AP_mu0.4、AP_mu0.8。
表2：各split prefix precision@1/5/10/20/50；不是candidate recall。
表3：参数量、训练step时间、inference sec/frame、max GPU memory（在服务器真实测）。
表4：e9/e14/e19轨迹（可选，不能用于事后挑Novel最佳）。

关键增量：

1. slot-baseline：显式insertion特征是否有益；包括新增容量，不能称纯readout因果。
2. volume-volume_fixed：**同参数/同token/同query embedding，仅d是否改变实际读取位置**。
3. volume_rel-volume：同容量，加入动作坐标系内预测几何证据。

不要因为attention某role质量高就宣称获得了接触/避碰证明。
不要因为Top1变好就断言仅ranking改善，重新训练后候选动作可能不同。
不要从普通label-matched训练结果声称支持任意动作位置的精确质量预测。

最终交付到实验根目录：`GVAR_RESULTS.md`、CSV、protocol副本、命令记录、测试日志、
所有完整AP NPY和失败/修复记录。负结果也完整保留。

## 9. 遇到问题时的处理

- argparse：新flags必须在legacy `utils.arguments`导入前消耗；不要加到全局parser后随意覆盖cfgs。
- OOM：正式训练首先保持 **per-GPU batch3 / global batch9** 不变，将ACTION_CHUNK从512降到128/64；所有volume配置使用一致chunk。
  activation checkpoint默认开启。不要用减少Q/angle/view候选来“解决”正式实验OOM。
- CPU/RAM压力：减少workers，检查cached CDF label RAM；不能把variable-length labels移到CUDA。
- 新reader context缺失：检查proposal_head输出(path1, logits)、group keyword signature；不要静默fallback。
- CDF shape/monotonic失败：检查D维顺序、原depth-wise head权重索引；不要关闭错误检查。
- GRAD：depth数据detach不能变成整条RGB/evidence branch的detach；不要冻结整个Stage-1。
- width：第一轮固定support envelope，不在输入里取GT width；不得声称full-action exactness。
- AP完整性：每split必须30scene×26frame×50rank×6threshold；失败split不能被均值自动忽略。
- 某个variant原始域不改善：如实汇报，不自行开始KD/gate/repair/consistency或G20→GN-Trans矩阵。

## 10. e19 scene-paired analysis（2026-10-09）

已有四组 GVAR 正式 AP（baseline、slot、volume、volume_rel）完成后，
以 **scene 为 bootstrap cluster** 进行 paired AP/μ/Top-K 分析，
参见 [GVAR_SCENE_PAIRED_CODEX_20261009.md](GVAR_SCENE_PAIRED_CODEX_20261009.md)。

分析代码为仓库根目录的 `analyze_gvar_scene_paired.py`，
测试为 `tests/test_gvar_scene_paired.py`。它只读服务器已有
`gvar_deploy_20261007` 的三 split e19 AP NPY、training 和 inference
manifests；不重跑训练、推理或官方 evaluator。默认跳过尚未训练的
volume_fixed，直接补充 volume_rel-vs-volume 的成对置信区间。
