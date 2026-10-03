# MoGe / RayRoPE 服务器执行记录

本次用户指令优先于包内建议：分支使用 `exp/moge-rayrope-grasp`，服务器 `robotarm@10.30.7.119`，允许 GPU 0/1/2/3/5/6，禁止 GPU 4。

## 来源与位置

- 原仓库 `/home/robotarm/EconomicGrasp` 有其他实验的未提交脚本改动，未修改。
- 独立工作树 `/home/robotarm/EconomicGrasp-MoGeRayRoPE`，固定基线 `52d09f925059bec3643610ecf1f1722894627ee5`。同名远端分支原本指向此基线。
- 用户源码包 `EconomicGrasp_MoGe_RayRoPE_source.zip` 为新增适配模块；原 `models/`、`utils/`、`dataset/` 未修改。
- Python `/home/robotarm/miniconda3/envs/grasp/bin/python`；Python 3.10.16、PyTorch 2.5.0+cu118、CUDA 11.8、RTX 3090。
- DAV2 权重软链指向原仓库已有权重，hash 随 protocol 保存，权重不提交。
- 数据 `/data/robotarm/dataset/graspnet`，compact 标注 `economic_grasp_label_300views_extend_angle_cdf_depth`。
- 正式结果 `/data2/robotarm/result/grasp/rgbgrasp/moge_rayrope20`；检查证据在 `preflight/`。

## 启动前发现及修复

1. 原包通过 33 项 CPU 测试及两进程 Gloo 不等长验证，但 CUDA RayRoPE grouping 输出 256 通道，原 decoder 需要 128 通道，导致实际 forward 失败。适配器现从原 decoder 的 `input_proj.in_channels` 获取输出通道，不修改 decoder。新增回归测试；修复后 CPU 共 34 项通过。所有 RayRoPE 及其机制对照均受此修复影响，未使用失败运行产生正式结果。
2. 增加每 epoch 各 rank 的 CUDA allocated/reserved 峰值、训练进度耗时、逐图深度空间标准差和零宽度比例；推断记录排除首个 warmup batch 的 forward+decode 时间、原始点云读取+collision 时间及含 IO 总耗时。不改变训练损失、采样或梯度路径。
3. 必需数据路径审计覆盖 train 5200 帧、Seen/Similar/Novel 各 780 帧。初次泛化检查发现 Similar/Novel 的 1560 个 `virtual_graspness` 文件缺失；核对原 loader 后确认只有 `use_gt_depth=True` 的标注训练分支使用它们，本协议为 False，推断不读取 graspness。该项保留为非必需缺失，未生成或伪造标注。真实必需路径无缺失。width payload 为 uint16 毫米，adapter 保留整数，matcher 单次乘 1e-3 转米。

## 已通过检查

- 八个固定 main 文件的 Git blob hash 全部吻合。
- CUDA flag-off parity：两帧的 depth、center、view、width、CDF、loss 最大差均为 0，decoded grasps 断言通过；不是 AP 测试。
- 三卡缩小规模 smoke：四个主 variant 均完成两步训练、九帧分布式验证及 checkpoint 保存。
- 四组 `grasp_to_geometry_max_abs=0`，geometry/task 梯度非零有限，DAV2 无梯度；MoGe shape/anchor 梯度均非零。
- 1024-seed 正式形状 smoke、正式运行和后续验收以服务器 JSON/日志为准，不将缩小规模 smoke 宣称为正式实验。

## 执行与恢复

正式四组顺序运行，GPU 0/1/2，每卡 batch 3，有效 batch 9；seed 0，20 epochs，5200/780 训练/验证帧，group chunk 32。chunk 仅调整内存分块。GPU 3/5/6 用于隔离的预检查，GPU 4 未使用。

正式阶段依次 train、infer、eval，collision on/off；仅用固定预算末尾 checkpoint。全部主结果验收后再按工作单运行 `attention_none,rayrope_point,moge_rayrope_point` 最小机制对照。可选 learned/shape/bias sweep 不默认展开。

禁止修改运行中代码或绕过 source/config signature。失败先检查退出状态、真实进程命令、日志和 checkpoint；只有已有有效 latest 时才使用 RESUME=1。不要删除其他实验数据。运行前 `/data2` 约 31 GiB 可用；后续必须持续检查空间。

正式验收需确认 20 epoch、有限指标、checkpoint hash、三 split 各 780 帧、on/off manifests 与 dumps、NPY 形状与重算 AP、scene paired contrasts、资源成本和中文结果报告。尚无正式 AP 结论。
