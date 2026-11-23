# Codex 工作文档：EconomicGrasp LAN Point RayRoPE Top-10 Demo 运行与验收

> **执行方式**：在装有 EconomicGrasp 依赖和目标 GPU 的机器上，将本文件全文交给 Codex。请直接执行可执行阶段、保留可核验日志；不要只输出运行建议。
>
> **任务性质**：已有代码的部署和验收，不是模型重训或方法重新设计。
>
> **任务日期**：2026-11-23。本文是执行计划；所有待测结果均标注为 `PENDING`，不能预填 `PASS`。

## 0. 一句话任务

在 `rcao-hk/EconomicGrasp` 的 `demo/lan-rayrope-top10` 分支运行 `demo/rayrope_lan/server.py` 和 `client.py`，使用 **pose-aware DAV2/DPT global-FiLM + Point RayRoPE + CVA-CDF**、**50% GraspNet-1Billion + 50% GN-Trans 的双源训练 checkpoint**，通过内网传输真实的 cropped RGB、aligned cropped depth、crop 参数、相机内参及相机姿态，并返回单帧整个 crop 场景的、按分数排序且经过过滤的最多 10 个抓取，含 `R(3×3)`、`t(3)`、`width_m`、`score`；完成真实加载、通信、几何和后处理验收，并形成报告。

**不可把 GraspNet + GN-Trans + GC6D 三源实验 checkpoint 当成 50/50 双源 checkpoint。** 旧实验报告中的三源 `triple_mix_20261024/point_rayrope/train/checkpoint.tar` 不属于本任务。

## 1. 工作目录及权威代码

- Repo：`https://github.com/rcao-hk/EconomicGrasp`
- 目标分支：`demo/lan-rayrope-top10`（已创建；**不要重新建分支**）
- 上游基线：`exp/shared-visible-geometry-multibatch`；避免覆盖其他研究实验的改动
- 主要入口：`demo/rayrope_lan/server.py`、`demo/rayrope_lan/client.py`
- 通信层：`demo/rayrope_lan/protocol.py`
- 接口说明：`demo/rayrope_lan/README.md`
- 单元测试：`tests/test_rayrope_lan_protocol.py`
- 模型与解码：`models/economicgrasp_bip3d.py:economicgrasp_dpt,pred_decode_center_view_angle`
- 设计元数据校验：`models/grasp_designs.py:validate_checkpoint_design`
- 后处理参照：`https://github.com/rcao-hk/graspnet-baseline/blob/main/demo_pick_place_aubo.py`。仅参考 grasp collision / NMS / sorting；**不执行 AUBO 机器人动作**。

仓库代码可能被更新，执行前必须从目标分支重新读取上述文件及 `--help`，不以本任务文档取代实际代码。

### 1.1 Git 检查（无破坏性）

```bash
pwd
git status --short --branch
git branch --show-current
git rev-parse HEAD
git fetch origin demo/lan-rayrope-top10
# 如果本地未在该分支且工作区干净，切换到：
git switch demo/lan-rayrope-top10
# 如分支仅存在于远端：git switch --track origin/demo/lan-rayrope-top10
git pull --ff-only origin demo/lan-rayrope-top10
git status --short --branch
```

**若本地存在未提交改动，不执行 `reset --hard`、`clean -fd`、强制 checkout 或覆盖文件。** 先记录并使用独立 worktree/安全分支隔离。运行产物写在仓库之外，checkpoint 只读。

## 2. 严格执行边界

1. **不训练、不恢复训练、不修改训练标签、不重写模型数值几何公式**。优先用当前代码运行；若发现部署 bug，只做有测试支撑的最小修复并记录 diff。
2. **不臆测 checkpoint 路径或混合比例**。先查运行档案和 manifest，确认双源训练 provenance；如果证据不足，标记 `BLOCKED_CHECKPOINT_PROVENANCE`，不能靠文件名推断“50/50”。
3. 只能使用自己有权限的本机实验文件和内网主机；不请求未知服务器凭据、不扫描无关主机/路径。
4. 不发送抓取结果给机器人控制器执行；不触发运动、夹爪、电机。机器人执行需后续单独安全审批。
5. 网络服务只监听 loopback 或可信内网 IP；若使用 `0.0.0.0`，同时确认防火墙、ACL 和无公网暴露。共享 token 不是 TLS，不视为加密。
6. 不能将真实相机图像、depth、token、内部 IP、敏感配置、完整 checkpoint 提交到公开 GitHub 仓库。
7. **不可虚构结果**。任何没有执行的 GPU/LAN test 都必须明确为 `PENDING` / `BLOCKED`。

## 3. 执行顺序和验收 Gate

请按下列阶段顺序执行；前一阶段不通过时要给出明确 blocker 和尽量独立完成后续可验证部分。

### P0 — 环境与依赖确认

记录 `hostname`、用户、repo commit、Python/PyTorch/CUDA/显卡、GPU 空闲显存、主要依赖版本：

```bash
hostname
python --version
python -m pip show torch torchvision numpy Pillow open3d graspnetAPI || true
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
python -m compileall -q demo/rayrope_lan
python -m unittest discover -s tests -p 'test_rayrope_lan_protocol.py' -v
```

本次部署涉及 `torch`、`torchvision`、`graspnetAPI`、`open3d`，以及仓库模型实际依赖的 CUDA/PointNet 组件和 `MinkowskiEngine`。缺包时先核查原环境，不盲目全量升级或重装 PyTorch/CUDA。

**P0 PASS**：当前 commit 已记录，目标模块通过 Python 编译检查，通信协议的三个测试通过。此状态**不等于**模型/GPU 推理通过。

### P1 — 找到且验证正确的 50/50 双源 checkpoint

优先读取已有训练完成标志、`log_train.txt`、`train_complete.json`、`manifest.json`、`run_config.json`、启动脚本、checkpoint 目录。可参考 `docs/RAYROPE_GVAR.md` 对 *Point RayRoPE RS50/GN50* 历史实验的描述，但不要把那个表中的 AP 当成本次部署验收数据。

在已知实验目录做**只读**查找，例如：

```bash
# 按实际 mount 路径调整根目录；不要在整个文件系统无限递归扫描。
find /data/robotarm/result/grasp/rgbgrasp /data2/robotarm/result/grasp/rgbgrasp \
  -type f \( -name 'checkpoint*.tar' -o -name 'checkpoint*.pt' -o -name 'epoch_*.tar' \) \
  -print 2>/dev/null | sort
```

逐一核验候选：
- 训练数据：GraspNet Original 50% / GN-Trans 50%，不是包含 GC6D 的 3-source；
- 结构：`enable_point_rayrope=True`、`enable_gvar=False`、`use_cdf=True`、`kview_mode=A1`；pose depth 为 `global_film`；
- 权重：完整 `model_state_dict` 和 `grasp_design`；记录 `grasp_design['rayrope']`，不手动猜 chunk/grid；
- 若原 checkpoint 来自旧 MoGe/RayRoPE 专用 trainer，可能是另一个命名空间；不可直接将 `checkpoint_best.pt` 当作当前 main 模型 checkpoint，必须证明兼容或显式转换并做等价性测试；
- 记录 `checkpoint_path`、`sha256sum`、文件大小、checkpoint epoch、对应源码 commit、训练域证据。必要时可由 Codex 只读地 `torch.load` **可信本地文件**检查键；注意完整 checkpoint 加载需要较多 RAM。

**P1 PASS**：有可读 checkpoint，具有双源训练来源证据，且与当前模型结构匹配。若候选不唯一，先比较证据；不能自动拿文件名最像的或 AP 最好的那一个。

### P2 — 单机模型启动和 strict loading

**第一个成功启动必须以 `strict=True` 完成 checkpoint 加载。** 不能为“通过测试”而改成 `strict=False`、忽略 `missing/unexpected keys` 或静默修改模型开关。

```bash
export CKPT='/ABS/PATH/TO/VERIFIED_RS50_GNTRANS50_POINT_RAYROPE_CHECKPOINT.tar'
export RUN_ROOT='/ABS/PATH/TO/PRIVATE_OUTPUT_ROOT/lan_rayrope_demo'
mkdir -p "$RUN_ROOT"/logs
sha256sum "$CKPT" | tee "$RUN_ROOT/checkpoint_sha256.txt"
CUDA_VISIBLE_DEVICES=0 python -m demo.rayrope_lan.server \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  --top-k 10 \
  --collision-thresh 0.05 \
  --collision-voxel-size 0.01 \
  --approach-dist 0.05 \
  2>&1 | tee "$RUN_ROOT/logs/server_loopback.log"
```

如果 `CKPT_SHA256_EXPECTED` 有经核对的来源，再传 `--expected-checkpoint-sha256 "$CKPT_SHA256_EXPECTED"`。上面启动为前台服务，Codex 需使用独立会话/受控子进程运行客户端；不要误以为服务进程占用终端就是卡死。部署时 `CUDA_VISIBLE_DEVICES=物理GPU`，服务仍使用逻辑 `cuda:0`，因为现有 `ModelFreeCollisionDetectorTorch` 在 `cuda:0` 分配。

**P2 PASS**：模型加载日志打印 epoch 和 SHA256，`grasp_design` 校验通过，严格 state dict 加载通过，服务端成功监听 8765；无 CUDA/OOM/依赖错误。

### P3 — 构造一份可重放的真实 RGB-D 输入

首选机器人实际采集的 aligned RGB-D 帧；若机器人当前不可用，可用本机合法持有的 **GraspNet RealSense 真正图像与其 metadata** 做首轮离线 smoke，不允许把随机图像得到的 grasps 当成真实效果验证。

需要：
- `rgb`：`uint8`，`[Hcrop,Wcrop,3]`，**RGB**（如果 OpenCV BGR，应显式转换）。
- `depth`：与 RGB 完全对齐的 `uint16` 或 `float32` 单通道深度 crop，尺寸相同。
- `crop_box_xyxy=[x0,y0,x1,y1]`：在**原始 full-resolution** 图中的裁剪区域，右/下边界 exclusive。
- `original_hw=[Hfull,Wfull]`，`camera_intrinsics`：原图 3×3 的 K，未重复应用 crop/resize。
- `depth_scale_to_m`：例如 uint16 mm 为 `0.001`、float meters 为 `1.0`，必须从真实传感器/元数据确认。
- `camera_pose_vec` 或 `T_table_from_camera`：**必需**。训练数据约定 `camera_pose_vec = -R_table_from_camera[:,2]`，表达在 table frame；不要固定写死 `[0,0,1]` 冒充真实值。
- 可选 `T_base_from_camera`：从 camera optical frame 到机器人 base 的 4×4 变换，用于返回 base-frame grasp。眼在手上时须使用实际当前 EEF pose 和已校准的手眼关系进行组合，核对变换方向。
- 内参对应**已去畸变/rectified 的对齐图像**；不要混用不同相机流的 intrinsics。

**特别核对 crop 几何**：服务端将 RGB resize 到 448×448，并按 dataset 约定计算
`cx'=(cx-x0)×448/(x1-x0)`、`cy'=(cy-y0)×448/(y1-y0)`、`fx'=fx×448/(x1-x0)`、`fy'=fy×448/(y1-y0)`。不得在 client 端已经 resize 后又传原始 `crop_box_xyxy` 造成二次缩放。

建议生成：
`$RUN_ROOT/inputs/crop_rgb.png`、`crop_depth.png` / `crop_depth.npy`、
`crop_meta.json`（存储在本机非公开目录），以及输入像素尺寸、有效深度比例、K/crop/pose 的审计 JSON。场景级 Top-10 仅覆盖该 crop 所见的 workspace；如果只裁剪单个物体，不得宣称整个原场景的 Top-10。

**P3 PASS**：RGB/depth/crop 一致；相机姿态为真实值；合法深度点足够；后投影与原图坐标一致。

### P4 — 本机 loopback TCP 端到端 smoke

服务端监听 `127.0.0.1:8765`，在**同机第二个终端**发送：

```bash
python -m demo.rayrope_lan.client \
  --server-ip 127.0.0.1 --port 8765 \
  --rgb "$RUN_ROOT/inputs/crop_rgb.png" \
  --depth "$RUN_ROOT/inputs/crop_depth.png" \
  --metadata "$RUN_ROOT/inputs/crop_meta.json" \
  --output "$RUN_ROOT/outputs/loopback_top10.npz"
```

客户端也支持 `.npy` depth；结果保存前先 `mkdir -p "$RUN_ROOT/outputs"`。服务端与客户端均有 request timeout；冷启动及耗时需要实测，不预先写性能结论。

**P4 PASS**：TCP 收发成功，`metadata_json.status == 'ok'`、`request_id` 正确回传、`grasps_camera` 形状为 `[N,17]` 且 `0<=N<=10`，无服务端异常。如果 N=0，按下节分析原因，不将空集合直接解释为成功抓取。

### P5 — 核验输出的几何与后处理

运行真实输出解析脚本，并生成逐项可复查报告：

1. 必备字段：`scores[N]`、`widths_m[N]`、`rotation_matrices_camera[N,3,3]`、`translations_camera_m[N,3]`、`grasps_camera[N,17]`。若发送 `T_base_from_camera`，还应有 `grasps_base[N,17]`、`rotation_matrices_base`、`translations_base_m`。
2. 逐项均 finite；`R^T R≈I`，`det(R)≈+1`，平移单位 m、夹爪宽度单位 m；`0<width<=grasp_max_width`；scores 应按降序。
3. `grasps_camera` 每行：[score, width, height, depth, R row-major 的 9 个值, t 的 3 个值, object_id]。**这里的 t 是 grasp representation 的 translation/center，不是经过 TCP 标定补偿的机器人执行位姿。**
4. 对照 `metadata_json` 中的 `raw_candidates`、`after_validity_filter`、`collision_removed`、`after_collision_filter`、`returned_top_k`，确认 pipeline 的顺序是 validity → collision → score sorting → NMS → top-K；NMS 后不能存在明显重复抓取。
5. 将 grasp `t` 投影到输入图像，判断物理合理性，记录 RGB 预测深度与传入传感器深度在抓取中心附近的偏差（只作诊断，不修改模型）；超出 crop、z 非法、明显漂移等标为 `GEOMETRY_WARNING`，不得送机械臂执行。
6. 如果 `T_base_from_camera` 已提供，用 `R_base=R_base_from_camera·R_camera`、`t_base=R_base_from_camera·t_camera+t_base_from_camera` 独立复算并对比服务端返回值。
7. 在同一真实场景分别测试 `--collision-thresh 0` 和 `0.05`（需分别启动服务），统计 raw/post-NMS/final 数量，确认碰撞筛选确实起作用；不要把 `N<10` 隐式补到 10，也不能以关碰撞为代价宣称安全。

**P5 PASS**：返回的 6D+width 数值、单位、坐标系和排序合理；碰撞/NMS 行为可解释。若模型返回 0 个结果且通信正常，单独记录 `EMPTY_VALID_GRASPS`，继续排查，不虚构抓取候选。

### P6 — 两台内网机器通信

在 GPU 服务器通过已验证的脚本监听受信任 LAN IP（可按需要 `--host 0.0.0.0`），先检查 `ss -lntp | grep 8765`、防火墙/ACL，然后在机器人机上运行同一 client，用服务端真实内网 IP：

```bash
python -m demo.rayrope_lan.client \
  --server-ip <GPU_SERVER_PRIVATE_IP> --port 8765 \
  --rgb <REAL_CROPPED_RGB_PATH> \
  --depth <REAL_CROPPED_DEPTH_PATH> \
  --metadata <REAL_CROP_METADATA_JSON> \
  --output <PRIVATE_OUTPUT_DIR>/lan_top10.npz
```

机器人机也可不写临时图片，直接在程序中调用：

```python
from demo.rayrope_lan.client import send_rgbd_request, decode_response_metadata

response = send_rgbd_request(
    server_ip=GPU_SERVER_PRIVATE_IP,
    port=8765,
    rgb=rgb_crop,       # uint8 RGB
    depth=depth_crop,   # aligned uint16/float, raw crop
    metadata=metadata,  # original_hw, crop_box_xyxy, K, pose, depth_scale
    timeout=60.0,
)
meta = decode_response_metadata(response)
R = response["rotation_matrices_camera"]
t = response["translations_camera_m"]
width_m = response["widths_m"]
score = response["scores"]
```

如果需要 token，服务端使用私下环境变量 `ECONOMICGRASP_LAN_TOKEN`，客户端使用 `--auth-token` 或调用函数的 `auth_token` 参数；不得把 token 写进 GitHub 或日志。测试重复请求、超时、错误格式、缺少 pose、depth 单位错误、断线重试的行为，确保不会把旧 grasps 当成新结果。

**P6 PASS**：两台机器使用真实私网 IP 成功请求/响应，结果字段、request_id 与 loopback 一致；记录传输大小、耗时、CPU/GPU 显存。没有第二台主机时，明确写 `P6=PENDING_REMOTE_CLIENT`。

### P7 — 最小修复、回归和交付

如果出现 bug，优先检查：
- checkpoint 不兼容（旧实验 vs 新主线、无 `grasp_design`、pose adapter 参数不匹配）；不能通过 `strict=False` 掩盖。
- `utils.arguments` 在 import 时解析 `sys.argv` 的全局状态；server 内的 argv 桥接须在新进程真实启动下验证。
- 抓取网络依赖的 `MinkowskiEngine`、pointnet2 CUDA 扩展、`graspnetAPI` 的 NMS 安装。
- `ModelFreeCollisionDetectorTorch` 固定使用 `cuda:0`；确认和服务端逻辑设备一致。
- RGB/BGR、uint16/mm 或 float/m、crop 坐标、K、pose transform 方向。
- `GraspGroup` NMS 返回值与 `[N,17]` 解码格式；输出排序。
- 现有服务并没有机器人工作空间边界过滤、角度可达性过滤、自动抓取执行；这些不是 P0–P6 的必备功能，且不能误宣称已实现。

修改只能限于本 demo 的部署问题，必要时加小型单元测试/真实场景 smoke；保持已有训练/测试入口行为不变。若原 checkpoint 无法加载，先提交定位报告与最小复现，待选择兼容 checkpoint/权重转换路线，不重新训练。

**必须交付的报告和证据**（建议位于 `$RUN_ROOT`，不得上传私有传感器数据）：

```text
$RUN_ROOT/
  report_zh.md                 # 总结、P0-P7 状态、阻塞、下一步
  env.json                     # 主机和软件版本、commit、设备信息
  checkpoint_audit.json        # path、sha256、provenance、epoch、grasp_design
  commands.sh                  # 实际执行过的命令，去掉 tokens
  logs/
    protocol_tests.log
    server_loopback.log
    client_loopback.log
    server_lan.log             # 如果已执行 P6
    client_lan.log             # 如果已执行 P6
  inputs/                      # 私有，不 git add
  outputs/
    loopback_top10.npz
    lan_top10.npz              # 如果已执行 P6
  validation/
    response_checks.json
    geometry_checks.json
    collision_on_off.json
```

`report_zh.md` 至少包含：各阶段 `PASS/FAIL/BLOCKED/PENDING`、checkpoint 训练来源证据、真实请求元数据（脱敏）、N、score 范围、width 范围、旋转正交误差、camera/base 帧一致性、collision ON/OFF 过滤数、耗时、发生的报错、修改文件及 commit SHA。实际未生成的文件必须说明原因，不能建立伪造的验收 JSON。

## 4. 最终通过标准

- [ ] 目标分支、干净可审计的执行环境与代码 commit 已核验。
- [ ] 已证明加载的是 **GraspNet50%+GN-Trans50%** 双源 Point RayRoPE checkpoint，而不是三源/GVAR/其他模型。
- [ ] 结构 metadata 核对 + `load_state_dict(strict=True)` 真正通过。
- [ ] 使用真实 crop/K/pose 与 aligned depth，完成单机 loopback 端到端推理。
- [ ] 获得有限、按分数排序且至多 10 个 grasp：`R,t,width_m,score`；记录 N=0 的合理解释。
- [ ] NMS/碰撞过滤（或显式标明关闭）的结果、输出单位及参考坐标系已核验。
- [ ] 已完成真实两机私网通信，或如实记录 `PENDING_REMOTE_CLIENT`。
- [ ] 中文执行报告和最小测试证据已落盘；无机器人自动运动；未经授权不发布任何真实图像或密钥。

## 5. Codex 完成时的回复格式

请在最终回复中按下面顺序回答，不要只说“运行成功”：

1. **运行状态**：能否实际通过 GPU server 和 LAN client；若否，最早失败于 P 几、报错原文与原因。
2. **Checkpoint**：来源和双源验证依据、sha256、epoch、`grasp_design`、严格加载状态。
3. **输入输出**：真实 crop 尺寸、K/pose、depth_scale、Top-N 返回数量及 `R,t,width,score` 校验统计；不展示敏感图像。
4. **性能**：实际模型推理时间、后处理时间、LAN 往返时间和显存，未测明确写未测。
5. **代码**：修改过的文件、测试结果和 commits；如无修改，明确说明。
6. **下一步**：最多 3 项需要用户提供的真实 blocker（如 checkpoint/内网机器/相机 pose），优先给可执行的最短路径。

**禁止**：把 protocol-only 测试当作真实 GPU 推理；把 Top-10 grasp candidates 当作 10 次成功抓取；把预测的 grasp pose 直接当作机器人末端 TCP 目标；仅为了通过而伪造 camera pose、改变 checkpoint 或绕过严格权重检查。
