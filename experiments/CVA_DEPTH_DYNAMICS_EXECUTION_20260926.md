# CVA depth dynamics 执行记录（2026-09-26）

状态：P0/P1 工程验收中，尚未开始正式 D0/D1 更新；本记录不宣称塌缩机制已确定。

## 环境与初始化

- 分支：`codex/cva-depth-gradient-dynamics`，从 main `52d09f925059bec3643610ecf1f1722894627ee5` 建立。
- 训练服务器：gpu02 / `10.30.7.117`，隔离目录 `/home/robotarm/EconomicGrasp-depth-dynamics`。原目录的用户修改保留。
- 解释器：`/home/robotarm/miniconda3/envs/grasp/bin/python`；PyTorch 2.5.0+cu118、CUDA 11.8、RTX 3090 24 GiB。
- 用户提供的 gpu04 Stage-1 文件已复制并核验两端 SHA256：`0cb8cd54bd5eef44a81c8d8de6969003377224d41a84b2002ba662bb8ce0c69c`。
- 文件记录 epoch 16、Stage-1、predicted geometry、image-FPS、global_film、use_fuse_depth=True。该文件无 optimizer；两组均从相同权重重置 AdamW。
- 预定配对设置：batch 1、FP32、LR 1e-4 constant、weight decay 0、global clip 1，KD 关闭、DINO 冻结。与原训练优化状态不同，不能称为原训练的精确 resume。
- 固定探针：16 个 train 帧及用户明确指定的 32 个 test_seen 验证帧；保留原 split 名和逐帧清单。

## 已取得的证据

1. 路径控制的小张量测试、状态保存与恢复测试、方向有限差分测试已通过。默认 detach 行为保留。
2. 第一次真实前向验收发现 CDF/width 标签映射含重复 CUDA 写入。固定输入重复十次仍可变化；当前检查 batch 中变化位于有效监督 mask 外。它是基础重复性错误，不能直接解释深度塌缩。
3. `73a1b4d` 将多对一 inverse-view 映射的代表选择显式固定为最大 scene index；CPU/CUDA 重复性测试通过。所有正式配对须从修复后的同一基础版本开始。
4. `20260926_stage1_seed0` 的 8 个真实 train/validation batch 已通过 detach 前向一致性和路径连通性检查：D0 task→depth 不连通，depth supervision 连通，E/Q/C 单路均有非零 task 梯度。
5. 相同运行的审计前后单步更新检查在 6 个权重张量上超过 atol=1e-6、rtol=1e-5；其余状态比较通过。当前正在用 plain/plain/audit 三分支量化数值波动，并逐位核对审计本身是否改动状态。未绕过此门槛启动正式训练。

## 产物与判读范围

完整产物保存在训练服务器：

- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_labelcheck_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics/20260926_stage1_seed0/diagnostics/P0/`
- `/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics/20260926_stage1_seed0/P0/`

产物包括 contract、逐项梯度、route connectivity、前向相等性、方向探针、梯度图及单步非干扰报告。`p0_gate.json` 当前为 false，原因是更新一致性尚未通过。局部导数不等于实际优化器更新效果；尚无配对训练轨迹，也没有必要性/充分性或修正方案结论。
