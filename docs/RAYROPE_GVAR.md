# main 的 DPT CVA-CDF：Point RayRoPE + GVAR

## 实现来源与接口

基于 main `5f7d348`；核心 `moge_rayrope/{attention,geometry,config}.py`、
`rayrope_gvar/{reader,decoder}.py` 与实验分支 `origin/exp/moge-rayrope-grasp`
的 `bb2cf57` 一致。main 已有核心移植，本次补齐训练脚本、checkpoint 开关与最佳权重元数据。

主模型是 `models/economicgrasp_bip3d.py:economicgrasp_dpt`。
`models/grasp_designs.py` 在模型构造时替换 grouping 和 CDF decoder，支持 baseline、
Point-only、GVAR-only、Point+GVAR 四种组合。两种设计默认关闭，不改变 baseline 模块或初始化 RNG。
要求 `--use_cdf --kview_mode A1`，几何来源为 RGB 预测深度，不允许 observed/GT geometry。
GVAR 使用几何调制前的 appearance feature；中心、旋转、内参、深度传入 reader 前 detach，
appearance 梯度保留。Point RoPE 不使用 learned uncertainty 或 MoGe shape tokens。

训练入口 `train_cva_ddp.py` 及其双域/三域派生 trainer 使用 `design_options`；
测试入口 `inference_cva.py`、`inference_cva_gntrans.py` 使用同一配置并检查 `grasp_design`。
完整 checkpoint 和启用设计时的最佳验证权重均携带设计元数据。旧 baseline 裸 state_dict 保持可用。
实验分支的 checkpoint 命名空间/模型封装不同，不能把历史 `checkpoint_best.pt` 直接当作 main checkpoint；
本同步不包含未经验证的权重转换，也不把历史 AP 作为 main 模型复现结果。

## 模型超参数

| 参数 | main 默认 | 本会话 Point+GVAR 配置 |
|---|---:|---:|
| ENABLE_POINT_RAYROPE / ENABLE_GVAR | 0 / 0 | 1 / 1 |
| RAYROPE_GROUP_CHUNK | 32 | 2048 |
| GVAR_ACTION_CHUNK | 256 | **2024**（不是2048） |
| RAYROPE_CHECKPOINT_CHUNKS | 1 | 0 |
| GVAR_ACTIVATION_CHECKPOINT | 1 | 0 |
| RoPE grid / radius | 7×7 / 40px | 同左 |
| RoPE dimension / heads | 192 / 4 | 同左 |
| frequency base / length unit / virtual standoff | 2 / 0.1m / 0.15m | 同左 |
| GVAR variant / probes | volume_rel / 36 | 同左 |
| reader dimension / heads / dropout | 64 / 4 / 0.05 | 同左 |
| envelope width / height | 0.06m / 0.02m | 同左 |
| finger length / thickness / approach | 0.06m / 0.01m / 0.05m | 同左 |
| geometry scale / fixed insertion | 0.06m / 0.025m | 同左 |

CLI 对应 `--enable_point_rayrope --enable_gvar --rayrope_group_chunk 2048
--gvar_action_chunk 2024 --rayrope_checkpoint_chunks 0 --gvar_activation_checkpoint 0`。
训练和推断必须传相同设计配置；元数据不匹配直接报错，避免静默改变参数无关的计算行为。

## main 双域训练示例

在仓库根运行；路径由调用者填写。此命令是 main 训练配方，不等价于历史实验 trainer。

```bash
export DATASET_ROOT=/path/to/graspnet
export GNTRANS_RGB_ROOT=/path/to/GN-Trans
export ENABLE_POINT_RAYROPE=1 ENABLE_GVAR=1
export RAYROPE_GROUP_CHUNK=2048 GVAR_ACTION_CHUNK=2024
export RAYROPE_CHECKPOINT_CHUNKS=0 GVAR_ACTIVATION_CHECKPOINT=0
GPUS=0,1,2 OUTPUT_ROOT=/path/to/new_run \
TRAIN_FRACTION=0.1 EVAL_FRACTION=0.1 BATCH_SIZE=3 MAX_EPOCH=20 \
LR=0.0003 WEIGHT_DECAY=0.001 DEPTH_WEIGHT_DECAY=0.001 \
NUM_WORKERS=4 EVAL_NUM_WORKERS=4 EVAL_START_EPOCH=9 SEED=0 \
bash run_gntrans_mix_cva_cdf_train.sh
```

main 的 `EVAL_START_EPOCH` 为 **zero-based**，9 表示第10轮。3卡每卡batch3，有效batch9。
main 双域脚本默认 LR=1e-4、weight decay=0、batch1、validation workers2、首轮验证；上例显式覆盖。
main optimizer 是 AdamW，学习率按 epoch cosine 调度；其初始化、depth loss、混合深度数据处理、
梯度路径由 main trainer 决定，不能仅修改上述数值就声称与历史 v2 实验协议完全一致。
三域入口 `run_gntrans_gc6d_mix_cva_cdf_train.sh` 也接受相同设计/chunk/checkpoint环境变量。

main 验证每次改善保存 `epoch_<index>_train_<loss>_val_<loss>.tar`；选择最低验证loss的该文件做推断。
`checkpoint.tar` 是滚动进度权重，会被后续 epoch 覆盖，不是固定 best。

## main 推断与官方 AP

保持上面的六个设计环境变量，设置 main 训练产出的 CKPT：

```bash
CKPT=/path/to/new_run/epoch_15_train_X_val_Y.tar \
GPUS=0,1,2 OUTPUT_ROOT=/path/to/new_eval DATASETS=both \
SAMPLE_INTERVAL=0.1 BATCH_SIZE=3 NUM_WORKERS=4 EVAL_NUM_WORKERS=16 \
COLLISION_THRESH=0.01 COLLISION_VOXEL_SIZE=0.01 \
GNTRANS_COLLISION_SOURCE=original_sensor USE_TOP4_VIEW_INFER=0 \
bash run_gntrans_mix_cva_cdf_eval.sh
```

`DATASETS=original|gntrans|both`；`SAMPLE_INTERVAL=0.1` 每split780帧，1.0 每split7680帧。
此脚本按GPU数分批调度 domain/split，每个job包含推断及AP；不是历史实验每split三shard的调度器。
AP每任务16workers。Collision on：threshold0.01、voxel0.01m；off：`COLLISION_THRESH=0`，须另设输出目录。
GN-Trans默认用原RS sensor点云做网络外碰撞过滤。GN-Trans的virtual-depth/workspace预处理
不代表无需深度辅助的预处理，但数值GT/sensor geometry不应进入RGB模型。

## 本会话历史实验（用于核对，非main复现实验）

| 配置 | Point+GVAR RS10/GN10 | Point RS50/GN50 |
|---|---|---|
| 每epoch unique / DDP visits / steps | 5200 / 5202 / 578 | 25600 / 25602 / 2845 |
| 训练 | 20epochs、seed0、fresh heads、Frozen DAV2 ViT-B、无MoGe | 同左 |
| optimizer / LR / decay / clip | AdamW / 3e-4 / 1e-3 / 1.0 | 同左 |
| batch / workers | 3每卡×3卡；训练和验证workers4 | 同左 |
| group chunk / activation checkpoint | 2048 / off；GVAR action2024 | 2048 / off |
| 验证 | Seen780帧，第10轮起 | Seen3840帧，第10轮起 |
| 最终权重 | epoch index19 latest | 验证总loss最小，index15 |
| 最终测试每split | RS780、GN-Trans780 | RS7680、GN-Trans7680 |
| inference / AP | batch3；3split并行，各3shards；每AP16workers | 同左 |
| collision | on主结果、off诊断 | on only |

历史GT合同v2：RS rendered前景(seg>0)+TSDF背景(seg==0)，invalid不互补；GN-Trans full rendered。
历史depth损失权重：global shape1、local shape0.5、reprojection0.05、interval0.1；
global/local points128/48、patch sizes8/16/32、每尺度2patches。

| 模型 | 域 / 测试比例 | Seen AP% | Similar | Novel | Mean |
|---|---|---:|---:|---:|---:|
| Point+GVAR | RS / 10% | 71.791953 | 55.182879 | 21.661118 | 49.545317 |
| Point+GVAR | GN-Trans / 10% | 63.867688 | 51.146942 | 18.223560 | 44.412730 |
| Point | RS / 100% | 75.643532 | 57.167585 | 19.746272 | 50.852463 |
| Point | GN-Trans / 100% | 69.935989 | 53.810547 | 18.845314 | 47.530617 |

上表collision on，单seed；数据量、测试比例、checkpoint选择不同，不能直接归因GVAR效果。

## 验证

`python -m unittest discover -s tests -p test_main_grasp_designs.py` 检查四种开关的真实
attention/decoder forward/backward、有限值、几何梯度隔离、appearance正梯度、默认baseline不变、
checkpoint开关一致性及最佳权重保存/加载。CPU测试不替代完整数据/CUDA训练和历史AP复现。
