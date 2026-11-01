# Codex 执行合同：A3 — Frozen A1/A2 Pose-aware Depth + Image-centric Point RayRoPE

**日期：2026-11-01**。本文件为待执行的代码实施、训练、推理、审计和比较工作单，**不是训练已完成报告**。

- 目标仓库：rcao-hk/EconomicGrasp。
- 工作分支：**exp/a3-rayrope-frozen-a1-depth-20261101**。
- 基于历史 RayRoPE 实现：exp/moge-rayrope-grasp，冻结父提交 **bb2cf573f0f78764e8c6923e22506a1a0cd6a448**。
- A1/A2 实现参考：exp/point-depth-pose-cdf-graspnet100；原训练脚本及历史 RayRoPE checkpoints 必须保持原状。
- 只做 **Point RayRoPE A3**：joint-variant=rayrope_point、use_moge=0、use_rayrope=1、ray_encoding=point、uncertainty=fixed、ray_apply_vo=1、kview=A1/top1、CDF+depth-wise width。
- **不做 GVAR36，不做 U3/learned uncertainty，不做 P0/P1 重新归因，也不新设计另一个 grasp head**。

## 1. 科学目标与解释边界

A1（已完成）是 Point-CDF，在 RS50%＋GN-Trans50% mixed training 中更新 pose-aware DPT，e19全量 RealSense Mean AP=42.924；A2（已完成）将其 e19 DPT checkpoint 加载并冻结，向点特征融合四层DINOv2视觉信息，Mean AP=46.490；历史 Point RayRoPE 有约50.853的全量RealSense Mean AP，但独立训练 depth head、训练时间和超参数不同，不能直接把该数值差解释为action-conditioned的纯收益。

**本次唯一主要目标：训练新的 A3，让它在全过程使用与 A1 e19、A2 完全相同的 frozen DPT predicted-depth field。** A3 仍使用历史 image-centric action-conditioned Point RayRoPE，而 A2 使用 point-wise RGB fusion；得到的差异是受控深度来源下两类 representation system 的差异，**仍不是固定同一候选集合的纯scorer消融**。

| 维度 | A1（历史） | A2（已完成） | **A3（本任务）** |
|---|---|---|---|
| Geometry source at inference | A1 e19 pose-aware DPT | A1 e19 frozen DPT | **同一 A1 e19 frozen DPT** |
| Geometry over training | jointly updated | frozen from initial step | **frozen from initial step** |
| RGB source | DINO只服务depth | four raw frozen DINO feature levels projected at point pixels | **同源 frozen DINO, trainable image proposal + RayRoPE image features** |
| Grasp representation | Minkowski XYZ point + CDF | Minkowski XYZ point + RGB + CDF | **image-FPS + action-conditioned Point RayRoPE + CDF** |
| Train | 50% RS + 50% GN-Trans | same | **same 12,800+12,800** |
| Nominal schedule | 21 epochs / e19 / global batch18 | 21 epochs / e19 / global batch18 | **21 epochs / fixed e19 / effective global batch18** |
| Test | all 23,040 RealSense frames, six μ | same | **same** |

A3使用与A2同一上游ViT-B/DINO source，但视觉 proposal/enhancer 是A3可训练的新模块，**不能声称A2/A3所有RGB latent逐张量相同**。A3的image-space FPS和action-conditioned grouping也可能选出不同grasp centers/views。

## 2. 已核对源码与必须修复的历史限制

RayRoPE 应从 exp/moge-rayrope-grasp 作为代码源而不是在Point-CDF分支强行重写：参考

- moge_rayrope/model.py：EconomicGraspMoGeRayRoPE包装了economicgrasp_dpt；A3要从 model.base.depth_net 访问 DINOv2DepthRegressionNet；非MoGe条件下没有FactorizedMetricDepth。
- rayrope_gvar/model.py：EconomicGraspPointGVAR(mc, enable_gvar=False) 是已有的 RayRoPE Point-only路径；无需加载或训练 GVAR reader。
- moge_rayrope/mixed.py：已有严格 source-separated RS rendered-foreground＋TSDF-background / GNTrans full-rendered；paired_frame_indices 支持0.5，扩展CDF cache只由CVAExtendedLabelAdapter加载。
- models/loss_economicgrasp_depth_kview_transformer.py：get_loss_cdf明确以 **balanced=False** 计算6阈值CDF BCE，并使用同一depth-wise width SmoothL1；保留该任务监督，禁止加入新的辅助质量标签或强行复制Point objectness head。
- train_rayrope_gvar.py：**旧训练入口写死 10%+10%、20 epochs、batch3、从头训练depth，并在gradient_contract要求depth梯度为正**，不能仅改shell的fraction或直接调用它。
- inference_rayrope_gvar.py：旧正式推理要求batch3、所有780帧、checkpoint_latest、最终epoch；不能直接用于新21 epochs的e19和每split7680帧。
- eval_rayrope_gvar.py：官方API可以接受 fraction=1.0，但须用新的推理manifest、全量数量与新checkpoint source验证。
- moge_rayrope/runtime.py 和 rayrope_gvar/runtime.py：有pinned MAIN_FILES/source fingerprint；**不更新旧主文件、不rebase改变pinned主线、不覆盖旧checkpoint schema**。新A3应使用单独版本签名与自己的file fingerprint。

**代码实现方式：从现有 RayRoPE 代码抽出最短新路径，建立独立 A3 trainer/infer/eval，不直接放宽/重写历史脚本限制，不使已完成 U0/GVAR 推断失效。**

## 3. 冻结深度来源（最重要，禁止绕过）

历史 A1 Mixed Point-CDF e19 checkpoint SHA256（来自已验收报告）：

**a1b8b6bed7f777ae8b14677c57ba59cba3c73d4b7057c78317c952f5d181af6b**

其已报告训练根：

/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_bs6_20261022_v3

候选checkpoint路径如下，**未经本工作单实际验证文件存在，Codex须读取目录核实，不能靠路径猜测加载**：

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export A1_SOURCE_CKPT=/data/robotarm/result/grasp/rgbgrasp/log/point_pose_rs50_gntrans50_bs6_20261022_v3/cdf/checkpoint_epoch_019.tar
test -f "$A1_SOURCE_CKPT" || { echo 'Check actual A1 e19 checkpoint location'; exit 1; }
sha256sum "$A1_SOURCE_CKPT"
# 必须与上面的 a1b8... 完整 SHA 匹配；否则停下查明源文件，不自动换e20/best/其他DPT
~~~

**A1 checkpoint 结构**：model_state_dict 字典有 depth_net.* 前缀；训练checkpoint的 pose_depth_mode 必须global_film、completed_epoch必须19、Point-CDF、train_domain=mixed、original_train_count=12800、gntrans_train_count=12800、seed0；实际检查后记录完整source fingerprint。

**A3模型结构**：EconomicGraspPointGVAR(enable_gvar=False).base.depth_net（即非MoGe DINOv2DepthRegressionNet）。实现 source→A3 完整前缀映射，剥离A1的 depth_net. 后，将 tensor keys 用 strict=True 加载到A3的 base.depth_net。预期 A2审计为251个DPT state tensors；**以实际 key-set 与 tensor equality 为准**，如果未匹配必须输出 missing/unexpected keys并停止，不可 strict=False、跳过FiLM、加载单独预训练ViT来替代。

参考实现伪代码（Codex完成并写真实模块，不是直接可运行的现有命令）：

~~~python
source = torch.load(A1_SOURCE_CKPT, map_location="cpu", weights_only=False)
assert source["completed_epoch"] == 19
assert source["pose_depth_mode"] == "global_film"
source_depth = {
    k[len("depth_net."):]: v for k, v in source["model_state_dict"].items()
    if k.startswith("depth_net.")
}
model.base.depth_net.load_state_dict(source_depth, strict=True)
model.base.depth_net.requires_grad_(False)
model.base.depth_net.eval()
# Check all named parameters and buffers before training, after each epoch,
# and in e19 checkpoint, exactly equal to source_depth (torch.equal).
~~~

**注意stride**：A1 wrapper 的depth_stride默认2，历史economicgrasp_dpt 的self.stride=1。二者可能产生不同depth_tok大小，但在DPT forward得到的448×448 metric depth应一致。Codex必须用**相同的预处理RGB tensor、crop-adjusted K、camera_pose_vec**输入两个DPT，对比448×448 output：全帧 finite、逐元素误差接近浮点精度。若像素级结果不同，优先排查state keys、pose conditioning、resize/crop和train/eval mode，不能只凭251个tensor一致宣布同一预测几何。

**冻结是全过程约束**：每次 model.train() 可能重新使depth decoder进入train模式。新A3模型必须确保 entire base.depth_net 在训练中始终eval，参数.requires_grad=False、optimizer不含其参数、DPT所有buffer不变。旧RayRoPE模型的train override只显式对 frozen pretrained ViT调用eval，不足以构成整个DPT冻结保证。任何depth loss可以计算作为报告指标，但其不能对DPT进行参数更新；模型loss构成必须显式记录constant depth loss与task loss。

## 4. 数据与训练合同（必须对齐真实 A2，而不是历史RayRoPE）

**Training：25,600 images / epoch**

- GraspNet Original RealSense：100 train scenes，frame 0,2,...254，每scene128，共12,800。
- GN-Trans：完全对应的100 scenes和frameIDs，RGB来自GN-Trans，计12,800。
- Original GT depth：rendered foreground＋TSDF background；GNTrans GT depth：full rendered；训练姿态、crop、K、物体标签、valid masks须复用现有 mixed.py 数据模块而不另造cache。
- CDF labels: economic_grasp_label_300views_extend_angle_cdf_depth；300view×12angle×4depth×6friction，width mm→m在label matcher一次转换，A3仍用balanced=False BCE。
- Validation：Original test_seen：30scene×26frames=780，frame 0,10,...250。不能将GNTrans val与test Novel引入checkpoint选型。

**优化预算：严格匹配 A2 的名义训练节奏**：

| 超参 | A3目标 |
|---|---|
| Epochs | **21（e0-e20），official checkpoint 固定e19** |
| GPU | 3 GPUs，seed0；无权占用其他人的卡 |
| Effective global batch | **18，A2实际为3卡×per-GPU batch6** |
| Optimizer | AdamW、LR=1e-4、weight decay=0、cosine over 21 epochs、grad clip1 |
| Nominal sample exposure | 25,600 / epoch；3-rank DistributedSampler padding到25,602 |
| Optimizer updates | 约 **1,423/epoch**（需由新脚本按真实DataLoader/accumulation复核） |
| DPT | strict loaded、frozen/eval、optimizer excluded |
| Proposal/RayRoPE/CDF/width | fresh seeded initialization，全部参与训练 |
| Model selection | 固定e19，不允许test/Novel AP选epoch |

RayRoPE比A2可能占更多显存。**先尝试 per-GPU physical batch6**；如实际CUDA OOM，可在独立A3入口实现 per-GPU microbatch3 × gradient accumulation2，使effective global18、optimizer update≈1423/epoch，并在manifest明确actual microbatch、accumulation、effective batch及最后不足额组的归一化规则。正确处理DDP no_sync及每个epoch末尾partial accumulation；检查梯度缩放和总样本权重。**累积梯度不等同于物理batch6的所有训练统计行为（例如batchnorm），必须披露**。不得直接改成3GPU×batch3、20epochs、LR3e-4来冒充A2对齐实验。

不要求重复之前已经验证过的单卡→双卡→三卡GPU smoke阶梯。先做最少的数据/静态合同检查，直接走正式训练；遇到真实故障才局部修复。所有出错checkpoint和日志保留，不静默中途切换超参。

## 5. Codex必须创建的独立文件与工作接口

建议输出（可按实际代码结构微调文件名，但所有语义与合同不能省）：

1. **train_rayrope_a3_frozen_depth.py**：复用 RayRoPE Point模型和mixed数据，但支持 0.5 train、0.1 Seen val、21epochs、冻结A1 depth、batch6或micro3×accum2、DDP、grad检查与e19持久化。不可复制原train_rayrope_gvar.py后仅删除硬编码检查；原gradient_contract会错误要求depth grad>0。
2. **rayrope_a3/frozen_source.py**：A1 checkpoint strict map/validation、SHA256、DPT frozen/eval lifecycle、buffer/weight exact equality audit；包含浮点predicted-depth对照和正确state prefix归一。
3. **inference_rayrope_a3_frozen_depth.py**：加载新A3 e19 checkpoint，strict source/frozen flags；full Original Seen/Similar/Novel each7680 frames；deterministic input sampling与出错防覆盖；无需训练用GT；depth_bias_mm=0，Top-1 view。
4. **eval_rayrope_a3_frozen_depth.py**：官方GraspNetEval full-frame，6个μ、AP数组finite [30,256,50,6]，验证23,040完整dump、frame schedule、collision元数据与e19；不能使用旧780固定限制。
5. **scripts/run_rayrope_a3_frozen_depth.sh**：分阶段 train/infer/eval 独立执行；CUDA_VISIBLE_DEVICES配置；train root、eval root不覆盖；失败日志与resume按同source signature检查。
6. **tools/verify_a3_a1_a2_depth.py**：A1和A2 ckpt为 depth_net.*，A3 ckpt为 base.depth_net.*；逐key tensor完全一致；输出JSON记录三个模型的完整checkpoint SHA与source SHA、251个tensor数量或实际数量。
7. **tools/compare_a1_a2_a3_full_ap.py**：三个split全量30×256×50×6，六系数Mean、每μ、Top1/5/10/20/50，A3−A2 / A3−A1 30-scene paired bootstrap (50k)，检查scene/frameID、postprocess/collision是否对齐。**严禁用历史 point summarizer 的 [:5] 旧口径**。
8. **tests/test_rayrope_a3_frozen_depth.py**：CPU state-key映射、missing/extra/shape mismatch、source SHA错误、模型freeze/.train lifecycle、optimizer exclude depth、checkpoint e19/e20、0.5 paired counts、loss_grad与评测manifest约束。
9. **experiments/A3_RAYROPE_FROZEN_A1_DEPTH_RESULTS_ZH.md**：正式结果、对齐/不对齐清单、AP和限制。

**历史文件不可被重写**：train_rayrope_gvar.py、inference_rayrope_gvar.py、eval_rayrope_gvar.py、原scripts/run_rayrope_point_gvar_mixed_v2.sh以及原历史checkpoint。新A3只能新增隔离代码或最小通用helper，并通过git diff明确无旧路径行为回归。

## 6. 服务器准备与Codex执行顺序

先在服务器确认可用的 GPU、硬盘、源码与AGENTS.md；此处历史服务器和文件路径来自此前实验记录，**本文件创建时没有SSH或实际GPU访问**。

~~~bash
ssh robotarm@10.30.7.117
cd /home/robotarm/EconomicGrasp
git status --short
git worktree list
find /home/robotarm -name AGENTS.md -print
nvidia-smi
df -h /data /data2
git fetch origin exp/a3-rayrope-frozen-a1-depth-20261101
git rev-parse origin/exp/a3-rayrope-frozen-a1-depth-20261101
# 在不存在同名路径且不影响他人作业的前提下：
git worktree add --detach ../EconomicGrasp-A3-RayRoPE-FrozenDepth origin/exp/a3-rayrope-frozen-a1-depth-20261101
cd ../EconomicGrasp-A3-RayRoPE-FrozenDepth
~~~

**步骤顺序**：

1. 确认A1 e19 source SHA256及depth_net key set；定位实际 A2 e19 checkpoint并核查同source SHA；若A2真实source与A1 e19不一致，必须停止共同几何声明。
2. 审查上述源码入口，新增独立A3 train/infer/eval；固定 source/full training manifest；CPU/静态检查，包括python -m py_compile、bash -n和本任务tests。新增代码不得破坏旧pinned-main校验。
3. 用同一实际的RGB/K/pose输入A1、A2、A3 DPT，逐像素核查448×448预测depth数值一致；forward的图像与深度label转换单位，CDF有效标签语义也要核查。
4. 以正式训练参数启动21 epochs。**不预设单卡/双卡/三卡smoke阶梯**。打印每epoch DPT fully frozen/state-equal、grasp gradients finite、proposal/RayRoPE/CDF有非零梯度、depth loss作为metric观测量以及训练steps/epoch。
5. e19必须**独立保存**为 checkpoint_epoch_019.pt；e20可存checkpoint_latest.pt归档，但绝不能让e20覆盖主评测e19。resume需验证optimizer/RNG/source/code/data签名，禁止混用不同batch或DPT source。
6. full Original RealSense测试23,040帧，collision ON threshold0.01、voxel0.01m、approach0.05m；原始sensor depth可用于crop/workspace和model-free collision，**不可以作为网络深度输入**。
7. 运行三组AP/深度等同性/计算开销比较，生成报告，commit轻量代码与工作文档，不上传checkpoint、GB级NPY、原始数据、敏感路径内容到GitHub。

### 启动接口（Codex需要实现；目前只是约定命令，不可在未新增脚本前直接运行）

~~~bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export A1_SOURCE_CKPT=/ABSOLUTE/PATH/TO/VERIFIED_A1_e19.tar

GPUS=0,1,2 \
TRAIN_FRACTION=0.5 VAL_FRACTION=0.1 \
EPOCHS=21 PER_GPU_BATCH=6 GRAD_ACCUM=1 \
SEED=0 LR=0.0001 WEIGHT_DECAY=0 \
VARIANT=rayrope_point USE_GVAR=0 USE_MOGE=0 RAY_ENCODING=point \
PHASES=train \
WORK_ROOT=/data/robotarm/result/grasp/rgbgrasp/a3_rayrope_frozen_a1_UNIQUE \
bash scripts/run_rayrope_a3_frozen_depth.sh

# 若physical BS6 OOM，仅新建或审计后resume的独立协议：
# PER_GPU_BATCH=3 GRAD_ACCUM=2 (effective global batch=18)
# 确保不会把它静默当作完全等价于physical BS6训练。

# 训练完成后正式 full-frame evaluation：
PHASES=infer,eval TEST_FRACTION=1.0 COLLISION=on \
GPUS=0,1,2 \
WORK_ROOT=/data/robotarm/result/grasp/rgbgrasp/a3_rayrope_frozen_a1_UNIQUE \
bash scripts/run_rayrope_a3_frozen_depth.sh
~~~

如果CUDA资源/缓存不足，明确报告BLOCKED；**不要承诺后台完成、不要推断训练耗时、不要合成AP值**。

## 7. 与A1/A2对比时的核心验收清单

| 项目 | 必须验收 |
|---|---|
| Frozen DPT | A1/A2/A3相同完整DPT的 state keys、weights、buffers；每epoch前后无变化；strict，无单独FiLM缺失 |
| Predicted depth | 同一crop RGB/K/pose → 448×448 depth tensor按浮点容差一致；记录A3的stride差异 |
| Dataset | 所有100scene奇偶协议一致，RS12800+GNTrans12800；val RS780；训练paired IDs一致 |
| Grasp labels | six μ CDF、depth-wise width、angle×depth matching；A3 token objectness/view与A2 point分支监督不同，必须披露 |
| Budget | 21epochs/e19、effective global18、cosine/AdamW/LR1e-4/WD0、nominal25600、约1423 optimizer updates/epoch；gradient accumulation另记录 |
| Inference | Original RealSense 30×256帧/split、同camera、RGB-only network input、同sensor collision参数、Top50、六个μ |
| Result statistics | Seen/Similar/Novel/Mean six-μ AP、APμ0.4/μ0.8、Top-k、A3−A2 paired 95% scene CI；非独立多seed CI |
| Cost | Trainable params、peak VRAM、训练walltime与推理ms/frame，与A1/A2逐项报告 |

**解释原则**：若A3在上述冻结DPT、数据和测试协议下仍明显优于A2，支持 action-conditioned image evidence organization 相比当前point-wise RGB融合更有价值；但A3仍同时改变image-FPS、proposal/view和结构容量，不应宣称严格隔离了唯一机制。尤其需要看Novel、μ=0.4和几何误差较大的场景；不要单看Mean AP。

**完成定义**：实现代码 + 静态测试 + 数据/共享深度合同 + 21epochs保存e19 + 全量3split官方AP + A1/A2/A3共同几何审计 + 中文结果报告。任何一项未完成，注明PENDING或BLOCKED，绝不能将历史独立训练的RayRoPE成绩冒充新A3成绩。
