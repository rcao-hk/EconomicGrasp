# 历史 no-detach 失败状态追溯（2026-09-26）

**本地证据中未找到可重建的“当前连续回归 CVA/CDF、放开 grasp→depth 后失败”的历史配置或 failure 前 checkpoint。** 因此目前无法确定该架构最早失败的 commit、精确 E/Q/C 开关、更新步数窗口或可恢复训练状态。这是本地检索的缺失证据，不能解释为远端没有此状态，也不能证明 no-detach 普遍稳定。

## 核查范围

- 当前工作区的相关 Markdown、文本和日志摘要；没有读取凭据、用户应用或访问远端。
- 本地 `EconomicGrasp/.git` 已有的所有 `origin/*` 分支：36 个不同的非第三方文本 blob；另核查全部可达历史提交中变更过的研究文档版本，共 61 个不同 blob（排除 README、依赖清单和第三方文档）。检索 no-detach、collapse、DDLA、深度失败与塌缩，并阅读相关命中的上下文。
- `origin/codex/depth-collapse-diagnostics` 本地指向 `6d13486b8aa960d7542e85bfc9598ffa02673de3`。其标题虽含 depth-collapse，文档明确保留 detach；不能把分支名称当作失败复现。
- 工作文档引用的 `CVA_depth_controls_analysis_20260907.md`、`Rep实验_HANDOFF.md` 未在本地工作树或研究文档历史中找到原文件。本次只引用工作文档对它们的明确摘要，未假定已重读原始附件。

## 证据与边界

| 时间 / 材料 | 可核实内容 | 对当前失败状态的意义 |
|---|---|---|
| 2026-01-12 会话线索 | 工作文档记载近乎平面 / 单一 bin，以及 depth-head 梯度约 3.57e-3→2.8e-6。[S1:73] | 最早提及的现象线索；未提供原日志、原 commit、checkpoint 或足以识别当前连续架构的配置。不能称为当前模型最早已验证失败。 |
| 2026-04-26 旧 DDLA | 工作文档记载 epoch 5→6 概率熵和 token-valid ratio 急降，seed depth 近 0.2 m。[S1:74] | 旧离散分布模型；工作文档明确标为历史会话解读、缺原日志/commit/成对 checkpoint。[S1:77] 不能当作当前连续回归失败状态。 |
| 2026-09-05～07 depth controls | 历史参考 `d0a2374745bcee866e2595e0561f6483b04d3ad6`；现存分支 HEAD `6d13486…`。原始 Git 文档写明 healthy Stage-1/2 初始化、所有 geometry detach 保留；joint training 也保留。[S2:3–5,116–118] | none / foreground / anchor 是相对深度监督对照，不是 detach 开关对照。AP 和 std ratio 摘要不能定位 no-detach 失败窗口。[S1:81–90] |
| 本次 2026-09-26 实验 | 当前基线为 main `52d09f925059bec3643610ecf1f1722894627ee5`；真实 batch 审计记录 D0 task→depth 不连通、E/Q/C 单路非零。[S4:7,20] 初始 flat fraction=0；已提交的 50-update 报告 D0/D1 flat fraction 均为 0。[S4:49–73] | 是当前架构的正式配对起点与有限预算观测，不是找回的旧失败复现。后续 500/2000-step 状态由主实验另行核查，本次历史追溯不替代其结果。 |

## 已有 checkpoint 线索

1. 用户指定、已有执行记录核验的 Stage-1 权重文件：
   `/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar`。
   SHA256 `0cb8cd54bd5eef44a81c8d8de6969003377224d41a84b2002ba662bb8ce0c69c`；元数据 epoch 16、Stage-1、predicted geometry、image-FPS、global_film、use_fuse_depth=True，**没有 optimizer state**。[S3:38–49]
2. 训练主机副本：`/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics/init/stage1_epoch15.tar`。[S3:96] 这是健康 warm-start 的 weights-only restart，不是 failure 前完整恢复点。
3. 执行文档记载同目录 sibling `checkpoint_15.tar` 有 optimizer state，LR `5.6476529721189974e-5`，但无 RNG/loader state；原训练基础 LR `3e-4`、21-epoch cosine、每 DDP 进程 batch 3。[S3:107–115] 此项仅为已记录的远端文件线索，本次未访问远端重新验真，也没有证据表明它是 no-detach failure 前状态。
4. 主任务随后提供的远端核查补充（本子任务没有直接连接）：Stage-1 原目录的 `checkpoint_0.tar` 已复制至训练主机 `/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics/init/stage1_checkpoint_epoch1.tar`，SHA256 `2d94fca4443e9a05082fc35f8b9c5fcb09030046241649c70e2dc394492811b6`。元数据 epoch 1、Stage-1、image-FPS、predicted geometry、global_film、use_fuse_depth=True；含 AdamW state（LR 0.0003、weight decay 0），缺 RNG/loader。此文件是可用的早期初始化候选；**没有对应历史 no-detach failure 的证据**。第二对仍明确按 weights-only restart 处理，不恢复其 optimizer。

主任务另在 gpu04 四个相关 `log_train.txt`（`economicgrasp_dpt_cva_cdf_width`、`economicgrasp_dpt_cva_cdf_width_new`、`economicgrasp_dpt_global_film_cva_cdf_width`、`economicgrasp_dpt_cva_cdf_distill_stage1`）检索 `detach_depth|depth_grad_routes|NaN|nan loss`，均无匹配。这只是指定文件和关键词的有界阴性结果；常量深度塌缩未必产生 NaN 或记录路由标志，因此不能据此证明没有失败。

## 对下一步选择的约束

若 warm 2000 未复现、D1 路径/权重/clip/seed mode 已核实，且主任务远端检索也找不到当前架构 failure 前状态，则本地历史核查支持进入计划的 **第二对当前架构 early/cold 共同初始化**，而不支持声称恢复旧失败。所选早期权重必须另记源文件、hash、元数据及 optimizer/RNG/loader 是否重置；不能把它称为精确历史 resume。旧 DDLA 的 epoch 5→6 不应换算成当前实验的失败预算。[S1:209–218]

## 来源定位

- **S1**：`D:/Research/Paper/RGB-Only Grasp/ECONOMICGRASP_CVA_CDF_DEPTH_DETACH_DYNAMICS_CODEX_PLAN_20260925.md`，行 69–90、209–218、372–375。
- **S2**：本地 Git 对象 `6d13486b8aa960d7542e85bfc9598ffa02673de3:DEPTH_GEOMETRY_EXPERIMENTS.md`；[该 commit 的原文](https://github.com/rcao-hk/EconomicGrasp/blob/6d13486b8aa960d7542e85bfc9598ffa02673de3/DEPTH_GEOMETRY_EXPERIMENTS.md)，行 3–5、116–118。本次从本地 `git show` 读取，未依赖网页缓存。
- **S3**：`D:/Research/Paper/RGB-Only Grasp/EconomicGrasp/experiments/CVA_DEPTH_DYNAMICS.md`，行 38–51、96、107–115。
- **S4**：`D:/Research/Paper/RGB-Only Grasp/EconomicGrasp/experiments/CVA_DEPTH_DYNAMICS_EXECUTION_20260926.md`，读取版本 `d6e9c735f9ec2ae151a09e3d6bf247ca965ff7ed`，行 7–20、49–73。
