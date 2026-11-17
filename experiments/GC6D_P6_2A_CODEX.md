# GC6D → P6-2A：冻结 RGB 七动作几何/精确质量 bank 生成器

日期：2026-11-17。目标分支：`exp/shared-visible-geometry-multibatch`。
基于 `5f740fb271912e1b96326ca690558da2d985d7dd`，保留已同步的 P6 bootstrap 修复；不覆盖 117 上的 P5 TF32/B4 修复。

## 1. 范围与当前状态

本实现补全 **GC6D RGB → native grasps → 同一七动作 bank → Predicted/Corrected 解析几何 → 每个 exact action 的质量标签 → P6-2A 冻结 reader 评价**。不训练、不重新选模、不运行官方 AP。它是 P6-2A 的数据生成与验收链，不是把 P4/P5 直接集成到端到端 AP。

本地 27 项 CPU 合成测试、编译与 shell 语法检查通过，包含几何重算验收与 DexNet 命名空间状态保持测试。真实 RGB checkpoint、CUDA 自定义算子、GC6D CAD/DexNet 标签及 117 大数据集需要在服务器 smoke/正式阶段验证；本文件不声明任何 P6-2A 数值。

## 2. 已核对的接口与一个重要更正

来源接口：
- 用户分支 `interactive_grasp/native.py`、`shared_geometry/runtime.py`：冻结 Point-RayRoPE+GVAR+CDF/global_film 及既有 MSE selector；原始 1024-query snapshot 与 decoder/reader replay。
- `shared_geometry/geometry.py`：七动作的确切顺序 `[0,-.005,+.005,-.010,+.010,-.020,+.020]` m；沿**单位相机射线**，不是 z-depth 增量，也不是夹爪插入深度。
- `dataset/gc6d_dataset.py`：GC6D 图像编号 `ann*4+camera_offset`；D435 为 `ann*4+2`，13 个 ann（0..12）。沿用仓库 pose-FiLM 的 JSON 旋转解释；不根据字段名称擅自取逆。
- 官方 `SeungBack/graspclutter6dAPI/graspclutter6dAPI/graspclutter6d_eval.py` 与 `utils/eval_utils.py`：模型点云 `models_m` 已为米，`cam_t_m2c` 是毫米，需 `/1000`。

**GC6D 官方 eval_scene 没有传入 GraspNet 的合成桌面。** 本生成器不虚构桌面/货架/箱壁碰撞几何；用官方 collision_detection + get_grasp_score，摩擦系数 0.2..1.2 共六档。把家具点云加入 scene 是另一个评价协议，不在此实验中偷偷修改。

官方 AP 包含 NMS、每物体 Top-10、全局截断与可选 GT 距离背景筛选。固定 bank 中不能删动作，因此本生成器**跳过这些候选筛选**，保留所有 exact actions，包含失败动作；这不是官方 AP。每帧额外用官方 `eval_grasp` 对预先选定的 singleton actions 复核，singleton 不会被 NMS/Top-K 删除。

两套 API 分别 vendored DexNet，存在严格 `isinstance` 检查。原 GraspNet 对象进入 GC6D 物理核前，会在**可信缓存对象**的序列化状态不变条件下映射包名，逐项核验 SDF、mesh、变换及缓存值；不重采样或重算几何。正式标签仍必须通过跨 API parity，不能只凭类型转换成功通过验收。不要加载来源不可信的 pickle。

## 3. 输入：无需先生成 graspness / economic labels / rendered depth

必需原始数据：

```
GC6D_ROOT/
  scenes/000xxx/rgb/00000x.png
  scenes/000xxx/scene_camera.json
  scenes/000xxx/scene_gt.json
  models_m/obj_000xxx.ply
  dex_models/obj_000xxx.pkl                 # 可信官方缓存存在时优先只读
  # 或没有 pkl 时：
  models_obj_m/obj_000xxx.obj
  models_obj_m/obj_000xxx.sdf
  split_info/grasp_test_scene_ids.json      # 也可传入自定义未见 scene pool
```

不需要 sensor depth、GT segmentation、virtual depth、graspness 或原离散 grasp labels。
RGB 分支只读取 RGB、K 与已知相机姿态元数据；默认全幅 resize 为 448×448，K 同步缩放。可以用 Python CLI 的 `--crop-box x0 y0 x1 y1` 在 plan 阶段预先固定裁剪；**不通过 GT mask 或 depth 求裁剪框**。这与以往 sensor-workspace crop 可能存在预处理差异，记录为跨数据集条件，不把它隐藏为完全同分布比较。

必需既有模型/缓存：
- 原 P1/P5 已绑定的 NativeRuntime source config（保持原文件与 hash），对应 base + selector checkpoint。
- P1 validation index：读取 tau、band、source identity、offset order。
- P4 depth validation index 与其原 source NPZ：自动确认训练时 `feature_map` 存储为 FP16 还是 FP32。
- P4 depth-only 选出的 `depth_model/best.pt` 及 selection.json。
- P5 **run02** 两套 reader best.pt 及 selection.json。
- 原 GraspNet 的 `validation_index.json` 和数据集：用于已知 exact-action 标签回放。

## 4. 数据独立性：不能仅凭 GC6D 名称声明 unseen

先生成 lineage 模板，再人工核查五个冻结组件：base、selector、depth_corrector、predicted_reader、corrected_reader。

模板中 `attested_complete=false`。需填写 reviewer/reviewed_at、每个组件的 GC6D training/development scene IDs，以及 namespace-qualified 已使用数据清单。空 GC6D exposure 列表表示**确认未使用**，不是未知。组件 hash 已自动绑定。

`training_scene_keys`、`development_scene_keys`、`official_test_scene_keys` 记录此次冻结前已使用/分析的数据，例如 `["graspnet",80]` 或 `["gc6d",301]`。本次要作冻结 external holdout 的 GC6D test 场景，不应被伪装成此前用于调参的场景；反之，之前已经看过结果并据此作设计选择的 GC6D 场景不能再标为独立。

代码拒绝 scene pool 与任一组件的 exposure 重叠，不静默剔除后重新抽样。它只能验证所提供清单的逻辑一致性，无法从一个 hash 证明训练历史。这里检验外部**场景**泛化，不自动构成 unseen-object 结论。

## 5. 阶段与产物

| 阶段 | 行为 |
|---|---|
| `lineage` | 自动填 checkpoint SHA256，创建未批准的 lineage 模板。 |
| `plan` | 绑定模型/P1/P4来源、选模记录、tau/band、feature 存储精度、源代码；从给定未见 scene pool 按 seed 抽样，冻结视角/动作预算。 |
| `parity` | 在原 GraspNet bank 重算 exact Q；比较历史标签、原 ExactGraspNet evaluator、新 GC6D physics kernels（此阶段两者均显式使用原 GN 桌面）。全部相等且至少一个正例才通过。 |
| `smoke` | 完整真实路径，小规模输出到隔离目录；只产出 `smoke_index.json`，不能当正式 holdout。 |
| `build` | 原 source 每帧 snapshot，corrector 真 B4（尾批按实际数量），逐帧 exact 标签；输出七字段 bank 与独立审计产物。 |
| `audit` | SHA、完整物理姿态、采样、标签原始 trace、几何重算、全量覆盖验收。 |
| `evaluate` | 审计后使用原 P5 run02 两套 reader，在 TF32=False 下调用现有 P6-2A evaluator。 |

默认 20 个场景 × 8 个 ann（0,2,3,5,7,9,10,12）× 32 族 × 7 动作 = **35,840 个 exact actions**。默认相机 D435。正式预算在 plan 固定，不根据标签更改场景/族。32 族沿用 half source-top128 + half remainder 的原采样策略。

每帧部署评价 NPZ 只有七项：

| 字段 | 形状 |
|---|---|
| predicted_physics / corrected_physics | `[F,7,75,6]`，float32：归一化 UDF、projective residual、四状态 one-hot |
| local_xyz | `[F,7,75,3]`，米 |
| bank_dims | `[F,7,3]`，width / height / insertion |
| physical_id | `[F,7]`，完整物理动作（排除分数）的 SHA256 |
| family_id | `[F]`，原 source query 索引 |
| quality_count | `[F,7]`，整数0..6；Q=count/6 |

审计 NPZ 另存 17D bank、全部 native grasps、K、pred/corrected depth、friction、collision/empty、物体实例分配。GT CAD/pose 只用于离线质量评价；不进入 RGB 或 reader 输入。重复类别的不同物体实例按该帧 scene_gt 顺序分别处理，不能用 obj_id 字典覆盖实例。

```
P62_ROOT/
  lineage.json
  plan.json
  graspnet_parity.json
  smoke/{smoke_index.json,frames/,audit/,run_contract.json,plan.json}
  bank/{index.json,frames/,audit/,run_contract.json,plan.json}
  reader_evaluation/{summary.json,predicted_families.jsonl.gz,corrected_families.jsonl.gz}
```

## 6. 117 执行示例

先保留服务器上的 P5 修复，再 fast-forward 更新当前分支。不要 reset 工作树。

```bash
cd /home/robotarm/EconomicGrasp-shared-geometry-batch
git status --short
git fetch origin exp/shared-visible-geometry-multibatch
git pull --ff-only origin exp/shared-visible-geometry-multibatch

export SOURCE_CONFIG=/absolute/path/to/the_existing_bound_native_source_config.json
export GC6D_ROOT=/data2/robotarm/dataset/GraspClutter6D
export GRASPNET_ROOT=/data/robotarm/dataset/graspnet
export P62_ROOT=/data/robotarm/result/grasp/rgbgrasp/gc6d_p62_run01
export DEVICE=cuda:0

bash scripts/shared_geometry/run_p62_gc6d.sh check
bash scripts/shared_geometry/run_p62_gc6d.sh lineage
# 核查/填写 $P62_ROOT/lineage.json；确认后设置 attested_complete=true。
# SCENES_JSON 默认取 GC6D 官方 test split；历史用过这些场景则改为另一个真实未见 pool。
bash scripts/shared_geometry/run_p62_gc6d.sh plan
bash scripts/shared_geometry/run_p62_gc6d.sh parity
bash scripts/shared_geometry/run_p62_gc6d.sh smoke
bash scripts/shared_geometry/run_p62_gc6d.sh audit-smoke
bash scripts/shared_geometry/run_p62_gc6d.sh build
bash scripts/shared_geometry/run_p62_gc6d.sh audit
bash scripts/shared_geometry/run_p62_gc6d.sh evaluate
```

P0P1_ROOT、P4_ROOT、P5_ROOT、ORIGINAL_ROOT 使用当前对话中已有目录默认值；服务器位置不同时覆盖环境变量。不要把 `P5_ROOT` 指回失效的 run01 reader cache。

只跑一个阶段可以单独调用。正常中断可用 `RESUME=1 ... build`：已完成帧逐项核验后复用；中断的 B4 批次重新执行完整原批，不改成仅剩缺失帧的小 batch。若进程被 SIGKILL 留下 `.build.lock`，确认旧进程已经退出再手工移除锁。存在没有完成 receipt 的部分文件时，先隔离该帧的部分文件再 resume；代码不会静默覆盖它们。

## 7. 数值与验收约束

- corrector 和最终 reader 评价显式关闭 CUDA matmul/cuDNN TF32；native provider 保留其已绑定的计算设置，不偷偷改变基础模型精度。
- 先按 P4 原 source 的 FP16/FP32 **存储**规则量化 feature_map，再用 FP32 corrector 计算；这不同于把几何计算改成 FP16。
- 原动作位置不根据修正深度重建；Predicted 和 Corrected 都查询同一个已冻结17D动作 bank。
- label parity 不通过时不生成正式 index；GC6D API/config/source dependency hash 改变后不能复用 parity。
- 不静默 clamp 宽度、不丢弃失败动作、不按 GT 分数筛族、不把缺失标签填0；真正碰撞/无接触失败由 exact evaluator 得到 count=0。
- 质量直方图用于验收日志，不能据此反复选择更有利的场景。
- 旧 P6-2A evaluator 的输入验证较弱，建议统一用本生成器 `evaluate` 入口，先完成增强审计；也可以把通过 audit 的 `bank/index.json` 交给原入口，但不要跳过验收。

## 8. 明确未完成的验证

没有在本次代码编写环境访问117、GC6D真实CAD、CUDA模型权重。CPU合成验证不等价于真实 force-closure parity 或独立泛化结论。第一次真实运行必须保留 smoke、parity、audit 的日志与签名文件。P6-2A仍报告固定动作bank的Q；P6-2B官方AP和端到端refinement不在本次改动范围。
