# GVAR P0-A/B/C — controlled action-evidence experiments

日期：2026-10-12。仓库：`rcao-hk/EconomicGrasp`，代码分支：`exp/gripper-volume-action-reader`。

**本文件是服务器 Codex 执行工作单，不是已完成的 P0-A/B/C 结果。** 已完成的是代码与 CPU 合成测试；尚未在真实数据上运行新增实验。不得覆盖此前 7 个 GVAR 变体、旧 e19/e20 checkpoint、历史 AP tensors 或用户未提交的代码。不得修改或合并 `main`。

## 0. 研究目标和已知事实

- 训练合同保持 10% Original GraspNet (2600) + 10% GN-Trans (2600)、3 GPU × per-GPU batch3、global9、21 epochs (e0–e20)、seed 0、A1 Top-1、CDF／depth-wise width、global_film、完整 E/Q/C detach，正式 checkpoint 固定 **e19**。
- 目前最佳 Volume-rel Original AP：Seen 66.3975、Similar 50.6917、Novel 19.4638、Mean 45.5176 (%), seed0。Hybrid fixed-RGB/dynamic-geometry Mean 44.1167；fixed-rel Mean 44.1730。
- Hybrid − Fixed-rel Mean −0.0563 pp，Volume-rel − Hybrid Mean +1.4009 pp，后者 Novel +0.6806 pp，CI 仅 [+0.0021,+1.3532]。这些结果来自独立训练候选，不能解释为固定候选排序收益。
- P1 的 AP-blind Novel 120 帧中，STD–AP gain 的场景中心化相关约 −0.033；不应基于此直接加 uncertainty gate。

本轮只回答三个不同的问题：

**P0-A：** 在**同一个已训练 Volume-rel checkpoint** 中，动态 RGB 内容、空间 metadata、attention validity mask 各有多大推理时敏感性？它们会影响哪些 gripper roles/insertion bins？

**P0-B：** 添加 Dynamic RGB / Fixed geometry 的 Inverse-Hybrid，补齐上一轮 Fixed/Fixed、Fixed/Dynamic、Dynamic/Dynamic 的第四格。重训后检验 RGB × geometry support 交互，而不是凭 Hybrid 单点结果宣称视觉信息的独立因果作用。

**P0-C：** 在相同 Original sensor-derived crop、K、sampled image indices、original sensor collision cloud 下，**只改变 RGB 来源**，比较 Baseline／Volume-fixed／Volume-rel 的 Original vs GN-Trans AP，计算 material shift gap 和 difference-in-differences。

## 1. 服务器代码安全核对（不可跳过）

原服务器已验证训练版本可能包含未上传的修复：上一轮记录实际服务器 SHA `59336c461ad40f5e6d8d759c08e1347b877435d3`，以及此前 `625cc26`、`d32b7df` 等修复。GitHub 当前分支不应被误认为自动包含这些 commit。

```bash
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp
git status --short
git fetch origin exp/gripper-volume-action-reader
git rev-parse origin/exp/gripper-volume-action-reader
# 查看已有本地 worktree 和运行中 GPU/训练作业；不要覆盖。
git worktree list
nvidia-smi
```

新建隔离 worktree（如已存在先检查，不要删除）：

```bash
git worktree add --detach ../EconomicGrasp-gvar-p0abc origin/exp/gripper-volume-action-reader
cd ../EconomicGrasp-gvar-p0abc
```

对比服务器上 7 个历史变体实际加载的代码来源、修复 diff、DPT/DINO 权重和 train/eval adapter。**必要修复应移植到独立 worktree 的本地修复分支，不要覆盖用户未提交修改、强制 reset 或直接更新 main。** 用旧 Volume-rel e19 checkpoint strict-load，检查 3 个已知帧的 R0 forward/decoder/CDF/width 与服务器已验证版本一致；若不一致，停止正式 P0 实验并记录 blocker。

所有新文件：

- `models/gripper_volume_reader.py`：runtime-only R0–R4 counterfactual、Inverse-Hybrid variant；**不修改原 reader 的 state_dict 形状／参数量**。
- `utils/gvar_p0_sampling.py`：跨 Original/GN-Trans 域一致的 per-frame pixel sampling seed，以及 GNTrans RGB 路径。
- `inference_gvar_p0.py`：完整 GVAR checkpoint；Original/GNTrans RGB-only **网络**；严格统一 Original sensor crop / K / sampled indices / collision cloud。
- `scripts/run_gvar_p0_eval.sh`：P0-A/P0-C 自动分 GPU wave 推理、串行 16-worker 官方 AP。
- `analyze_gvar_p0a.py`, `analyze_gvar_p0c.py`：配对场景 bootstrap AP/CDF prefix＋摩擦阈值及域差异。
- `audit_gvar_p0_support.py`：AP-blind 120 帧、真实 grasp projection 的固定／动态 probe displacement 和 mask conflict。
- `scripts/run_gvar_inverse_hybrid_train.sh`, `analyze_gvar_p0b.py`：第四格训练／2×2 factorial。
- `check_gvar_p0_rgb_registration.py`：Original/GN-Trans 图片、GT segmentation mask 共坐标叠图，需要人工配准验收。
- `utils/gvar_p0_analysis.py`：标准 AP `[30,26,50,6]`／来源 SHA／协议相容性检查。
- `tests/test_gvar_p0abc.py`, `tests/test_gvar_p0b_support.py`：CPU contract／geometry／bootstrap tests。

先执行：

```bash
python -m py_compile models/gripper_volume_reader.py inference_gvar_p0.py check_gvar_p0_rgb_registration.py \
  analyze_gvar_p0a.py analyze_gvar_p0b.py analyze_gvar_p0c.py audit_gvar_p0_support.py
python -m pytest -q tests/test_gvar.py tests/test_gvar_scene_paired.py \
  tests/test_gvar_p0_p1.py tests/test_gvar_p0_p1_followup.py \
  tests/test_gvar_next_p0_p1.py tests/test_gvar_p0abc.py tests/test_gvar_p0b_support.py
bash -n scripts/run_gvar_train.sh scripts/run_gvar_p0_eval.sh scripts/run_gvar_inverse_hybrid_train.sh
```

## 2. P0-A：R0–R4，同一 checkpoint 推理反事实

### 2.1 严格定义

| 模式 | RGB feature 位置 | Metadata 位置 | Attention validity mask | Relative geometry 位置 |
|---|---|---|---|---|
| R0 | Dynamic | Dynamic | Dynamic | Dynamic |
| R1 | **Fixed** | Dynamic | Dynamic | Dynamic |
| R2 | Dynamic | **Fixed** | Dynamic | Dynamic |
| R3 | Dynamic | Dynamic | **Fixed** | Dynamic |
| R4 | **Fixed** | **Fixed** | **Fixed** | Dynamic |

R0 使用未经干预的 Volume-rel reader。R4 在*转移同一权重*时应与 Hybrid reader 的 token 路径一致，但历史 Hybrid 是独立训练得到的权重，绝不直接比较两者当作同一模型的结果。

固定参考 insertion `d_ref=25mm`，真实 bin 对应 10/20/30/40mm。前向不修改 grasp center/view/angle 上游网络与 weights，但 decoder 输出排序、宽度和最终物理抓取集合仍可能变化。**R1–R4 是测试时 OOD 扰动，只能作为已训练模型敏感性证据，不是重新训练后的结构因果效应。**

先单 split／2 batches smoke，分别检查 R0 输出与正常 GVAR 一致、预测 depth 与用于几何的 depth 恒等、R0–R4 numerical finite、GVAR checkpoint strict-load：

```bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
DATASET_ROOT=/data/robotarm/dataset/graspnet
GPUS=0 KIND=p0a SPLITS=test_seen MAX_BATCHES=2 RUN_EVAL=0 \
 OUT_ROOT="$ROOT/analysis/p0abc_smoke_20261012" \
 bash scripts/run_gvar_p0_eval.sh
```

正式：

```bash
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
GPUS=0,1,2 KIND=p0a BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
  OUT_ROOT="$ROOT/analysis/p0abc_controlled_20261012" \
  bash scripts/run_gvar_p0_eval.sh
```

- 3 splits × 5 modes，split 各 780 帧，e19 Volume-rel 的**同一 checkpoint SHA256**；每一 mode 独立新目录。
- 训练时不启用任何干预；没有更改 reader 数量、label 或 GT depth。
- 各模式正常 inference 后官方 `eval.py --sample_interval 10` 由脚本**串行 CPU16**评测。禁止用 partial smoke AP。
- 自动输出 `p0a_analysis/reader_cf_effects.csv`（5模式×12指标×4 split＝240行）、scene.csv、REPORT.md、AUDIT.json（源文件 SHA256）。

P0-A 的 additional geometry-only audit 使用之前 AP-blind P1 120 帧的 sidecars，与当前模型已保存的 *post-collision* grasps：

```bash
python audit_gvar_p0_support.py \
  --root "$ROOT" \
  --selection-json "$ROOT/analysis/novel_uniform_20261011/selected_frames.json" \
  --sidecar-root "$ROOT/analysis/novel_uniform_depth_sidecars_20261011" \
  --output-dir "$ROOT/analysis/p0a_probe_shift_20261012" --topk 20
```

输出各 role × insertion bin 的动态／固定投影像素距离（仅两者投影都有效时）和 `geometry_known_and_fixed_rgb_masked` 比例。预测射线上有 depth 数值不是 3D 可见性，不是 free space，也不代表 contact utility。

## 3. P0-B：Inverse-Hybrid 补齐第四格

4 组形成真正的 crossed support 策略矩阵：

| | Geometry Fixed | Geometry Dynamic |
|---|---|---|
| RGB Fixed | `volume_fixed_rel` | `volume_hybrid` |
| RGB Dynamic | **`volume_inverse_hybrid`** | `volume_rel` |

Inverse-Hybrid：RGB、位置 metadata、attention mask 使用实际 insertion d；relative geometry 值从固定 `d_ref=25mm` 对应的投影支持区域读取。它保持 36 roles/probes、相同参数量、CNN/attention/CDF head、相同 GT 匹配与 supervised target。**两种 evidence 仍可能来自不同 3D 位置，实验只检验当前 fusion 下支持策略的交互，不是模型全部物理接触的证明。**

先检查新变体 gradient detach、单GPU batch3、双卡两batch DDP finite smoke。在GPU空闲、存储空间充足且工作区修复核对完成后启动 fresh 训练：

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
GPUS=3,4,5 bash scripts/run_gvar_inverse_hybrid_train.sh
```

与上轮 Hybrid 对齐：21 epochs、seed0、global9、chunk2048、no AC、train workers3／val4、prefetch1、训练 5200、Original Seen validation780。GPU分配先核实无关作业，输出必须不存在或为空；不加载旧 task checkpoint。注意服务器已修复 `prefetch_factor`／CUDA／width unit／RNG／部署代码必须合并。

固定 e19 评测（**仅 Original**）：

```bash
CKPT="$ROOT/train/volume_inverse_hybrid/checkpoint_epoch_019.tar" \
GPUS=0,1,2 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
DATASET_ROOT="$DATASET_ROOT" \
OUTPUT_ROOT="$ROOT/eval/volume_inverse_hybrid_e19" \
bash scripts/run_gvar_eval.sh
```

再运行：

```bash
python analyze_gvar_p0b.py --root "$ROOT" \
  --output-dir "$ROOT/analysis/p0b_inverse_factorial_20261012" \
  --bootstrap 50000 --seed 20261009
```

缺 e19/e20 任何checkpoint／3split AP／完整 train log 会输出 `pending`，绝不虚构结果。完整后核验 checkpoint 文件实际 SHA、历史三组 Mean AP、frame fingerprint 和混合训练 sampling hashes；输出 4×12×7 条效应／interaction 及 scene deltas。主要 contrast：

`I_RGB×Geom = (Rel − InverseHybrid) − (Hybrid − FixedRel)`。

必须分 Mean、Seen、Similar、Novel 和 μ=.4/.8，输出直接 scene-paired 95% CI。不能将 scene CI 写成多 seed CI 或等价性证明。

## 4. P0-C：Original–GN-Trans 只改变 RGB 的 matched evaluation

### 4.1 为什么不直接调用旧 GN-Trans inference

旧 `inference_cva_gntrans.py` 的 `GraspNetTransDataset` 以 virtual GT depth 构造 workspace/crop，而 Original 默认使用原始 sensor depth。二者直接对比 AP 会混入不同 crop/image-FPS selection、点云对齐等潜在因素。新 `inference_gvar_p0.py` **不调用**旧 GN-Trans 前向脚本：使用 Original GraspNet 的 sensor depth/labels/meta 得到相同 crop、K、pixel-selection和碰撞点云，把唯一的 RGB source 换成 `GNTRANS_RGB_ROOT/scenes/00165/0080_color.png` 这样的材质变化图片。

正式推理前运行纯 CPU 配准蒙版预览：

```bash
python check_gvar_p0_rgb_registration.py \
  --dataset-root "$DATASET_ROOT" --gntrans-rgb-root "$GNTRANS_RGB_ROOT" \
  --output-dir "$ROOT/analysis/p0c_rgb_registration_20261012" \
  --scenes 100 130 160 167 174 --frames 0 80 160 240
```

必须人工确认叠图上的物体轮廓、桌面遮挡及视角是否对应。如果不对应，记录 blocker 而不是继续声称材料干预具有因果解释。

正式推理前逐一核查 GN-Trans 780 张／split RGB 文件存在且与 Original plane 分辨率一致；GN-Trans 抽查三帧 Original/GNTrans 的 `K`,`img_idxs`,`camera_pose_vec` 等除 `img` 外固定输入必须逐元素相同。采用**按 frame index 确定的随机种子**，与 worker 调度无关。预测几何恒为 DPT/DINO depth head 的输出；GT、virtual、sensor 均不进入模型。预处理与碰撞仍使用 Original sensor cloud，因此这是 **RGB-only network under depth-assisted matched preprocessing**，不是 raw-RGB-only 系统或旧 virtual-crop 协议。

### 4.2 正式推理评测

**务必为 Original 与 GN-Trans 两域重新运行同一个新采样协议**，不可把旧 GVAR inference NPY 与新 GN-Trans 单独配对（历史未采用 per-frame deterministic sampling）。三个 frozen checkpoints：`baseline`,`volume_fixed`,`volume_rel`，均 e19；不训练新权重。

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet
GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
ROOT=/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007
GPUS=0,1,2 KIND=p0c BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
OUT_ROOT="$ROOT/analysis/p0abc_controlled_20261012" \
bash scripts/run_gvar_p0_eval.sh
```

18 个 test split runs（3 variants×2 domains×3 splits），每个 780 帧。默认 collision original_sensor threshold 0.01，CPU official eval16 串行，`sample_interval=10`。每一 split 保留 checkpoint SHA、数据根目录、GNTrans RGB 根目录、原始 sensor preprocessor 名称、同帧 fingerprint、sample seed 和 17列 grasp dump。

汇总 `p0c_analysis/domain_contrasts.csv`、`domain_scenes.csv`、REPORT.md、AUDIT.json；除原始域与 GNTrans 的绝对 AP，还计算：

`MaterialGap(v) = AP_GNTrans(v) − AP_Original(v)`

`GapChange(v) = MaterialGap(v) − MaterialGap(baseline)`

GapChange>0 意味相比 Baseline 的 material penalty 减轻，但它不证明 Novel@μ=0.4 的绝对 AP 上升。报告六个μ和 prefix@1/5/10/20/50，与 scene-cluster 95% CI（50k bootstrap）。本协议是否完全符合 GNTrans render/原始相机共视场须结合 GNTrans RGB/Original 的物体像素位置抽查确认；仅 image size 相等并不足以保证物体几何配准。若显著错位，应停止并报告 blocker，不能静默称 material-only intervention。

## 5. 终验与解释边界

1. 验证 P0-A R0 旧模型数值 parity，`r4` 在同一权重移植到 Hybrid reader 时一致。确认 R1–R4 在实际 forward 确实各只改变约定的 feature／metadata／mask，而 geometry 恒动态；记录异常 mask 比例。
2. 验证 P0-B 各变体结构、full checkpoint strict-load、训练日志21 epochs/e19固定、3 rank RNG、有限loss、同采样hash和 evaluator setting；不从 Novel 选最优 checkpoint。
3. 验证 P0-C Original/GNTrans 图像尺寸、**内容是否配准到同一原始几何视场**、除RGB外输入 parity、checkpoint identical、split/frame fingerprint identical、sensor collision identical。
4. 官方 AP NPY `shape=[30,26,50,6]`，finite，canonical order，每 split 780 dumps，完整一套 provenance；脚本 fail-fast 处理 missing/NaN/协议不相容，不重写历史结果。
5. 明确区分 test-time stress sensitivity vs retrained architecture effect vs controlled RGB appearance gap。scene-bootstrap 不包括训练 seed variance／多重比较。不能仅由 prefix precision 推断纯 ranking，也不能把 RGB-only 网络称为完全无深度系统。

最终报告：`GVAR_P0ABC_RESULTS_ZH.md`，包括各 split／μ／prefix／scene CI、R0 parity、3个重点 Novel failure scenes、material-gap change、代价与VRAM以及所有 logs/source SHA/checkpoint SHA。如结果为负也保留。不要自动训练其他分支（RayRoPE、UA-Depth）、扩大混合数据或调整实验协议以获得更好 AP。
