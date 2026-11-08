# RARE-style Grasp Ranking：10% Original + 10% GN-Trans

日期：2026-11-09。独立分支 `exp/rare-ranking-mix10`，从
`main@7a956d37aaa427a32e598f21fdc6b910ed3c2704` 创建。不是旧 GVAR/P3 分支的合并。

## 0. 任务边界：这是哪一部分 RARE？

原论文：Park et al., **RARE: Learn to RAnk and REtrieve for Monocular 3D Object Detection**, CVPR 2026。
原文：[论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.pdf)，§3.1、式(2)–(5)、§4.1。

本轮迁移 **Learning to Rank**：

\[
Q_i=\tfrac16\sum_{\mu\in\{.2,.4,.6,.8,1.,1.2\}}1[\text{实际动作在该}\mu\text{成功}],\quad c_i=\sigma(z_i),
\]
\[
\ell_{point}=\operatorname{mean}(c_i-Q_i)^2,
\quad \ell_{pair}=\operatorname{mean}_{Q_i\ne Q_j}\operatorname{softplus}[-\operatorname{sign}(Q_i-Q_j)(z_i-z_j)].
\]

- 原文目标为 class-consistent 3D IoU，这里替换为完整执行抓取的 exact CAD/Dex-Net quality。
- point/pair 权重默认 **10/0.5**，来自原文实现设置。Pair 在**同一幅图像**内均匀无放回采样，不跨图像、不跨 RGB 域；每图最多2048对是本实验的预算设置。
- 质量用整数通过数0..6识别 ties；等质量对**不**排序，零质量动作仍参与 point-wise 监督。
- Logistic 使用 **raw scalar logits**，不是把 sigmoid 概率差再当 logit；没有固定margin、加深方向限制或“原动作更优”的预设。
- **不复现**原文§3.2的多3D框query、retrieval-aware box regression或完整MonoDETR检测器。不称为完整RARE复现。

### 两阶段训练为何必要

用户的 G0–depthguard 总结已证明：旧5mm近邻有效mask丢失大量实际动作；执行宽度放大/截断会改变标签语义；对等质量动作强制margin可能制造不必要偏好。

因此不把新排序loss直接叠到失配的旧标签上，也不在更新center/width之后继续使用失效的缓存标签：

1. **S0 common base**：完整模型以10%+10%训练，固定 e19（epoch字段20）。以 RayRoPE+GVAR36 为主，Point RayRoPE为可选平行对照。
2. **S1 exact cache**：冻结S0全部参数，对同一批10%+10%图像生成实际动作，精确标注。
3. **S2 quality scorer**：训练相同结构的 scalar residual scorer，分别使用MSE和MSE+pair。几何、GVAR、CDF、view/angle与逐depth宽度全部冻结。
4. **S3 test**：全1024query推断，评估new depth / new priority及二者分离对照。

**这是“共同基座全模型训练＋冻结动作生成器的评分阶段训练”，不是RARE loss端到端更新整张网络。** 不隐瞒该边界。若后续要解冻动作生成器，必须在线评价或定期重新采样/重标注；本轮不实现这种扩展。

## 1. 严格实验设置

|项目|本轮设置|
|---|---|
|训练图像|Original scenes0000–0099 × frames0,10,…,250 =2600；GN-Trans相同scene/frame =2600；总5200|
|域比例|两域相同图像数，每轮各遍历一次；不是50%训练权重，不从mixed50权重启动|
|验证图像|训练域scenes0000–0009，frames5/135，两域共40图；与训练帧不重叠，但**不是独立场景**|
|正式选模|S0固定e19；S2固定第20轮，验证指标仅监控，不按Seen/Novel AP或验证曲线挑正式模型|
|测试|Original优先；Seen/Similar/Novel各30scenes×26frames=780；GN-Trans可选分开测试|
|主模型|Point RayRoPE + GVAR36；同一基座用于native/MSE/RARE，不重训三份不同proposal模型|
|S0预算|20轮、seed0、3 GPU×per-GPU batch3；lr3e-4、weight decay/depth decay1e-3；main的既有优化器和任务loss|
|S1抽样|每图均匀32 native queries；冻结各query原最高分angle；每query全4 depth|
|S1标签|665,600个训练动作、5,120个验证动作；无旧有效mask，collision/empty是有效负例；official evaluator|
|S2评分头|原生64维score-branch feature + depth one-hot + frozen base utility logit；LN/64隐藏层/GELU/零初始化scalar residual|
|S2预算|20轮、AdamW lr1e-4 wd1e-4、4 frames/batch、grad clip1、seed0；MSE/RARE相同batch顺序和参数量|
|推断|原view/angle固定；每query四depth由scalar scorer评分；width随选中的原depth读取，width head不更新|
|碰撞预筛|Original sensor cloud，voxel=.01m、threshold=.01、approach=.05；无深度输入network|
|AP|复用main的eval.py / 官方GraspNet evaluator；TOP_K50、全部六μ；数组[30,26,50,6]|

10%是每场景stride=10，26/256≈10.16%，延续项目的“10%”口径。2600×2是**训练图像数**；40张额外验证图像单列，不冒充5200中的独立测试。

S0沿用main的标签匹配和任务loss。本分支不偷偷移植服务器的view修复或其他历史补丁；S1/S2实际动作质量不经过旧matcher。若Codex确需修复S0公共路径，必须先报告、同样作用于所有比较、重新记录源码身份，不能只改RARE组。

## 2. 预注册比较矩阵

|ID|动作生成器|新增评分头|监督|用途|
|---|---|---|---|---|
|B0/native|同一个S0|无|原训练|参照|
|B1/point|冻结S0|ResidualRanker|10×exact-quality MSE|容量/新增标签/训练预算匹配控制|
|B2/rare|冻结S0|与B1完全相同|10×MSE +0.5×pair，image scope|**主要比较B2−B1**|
|可选B3|冻结S0|同B1|MSE+pair，仅同family|问题定位ablation，不属于完整原文image-scope配对|

不要只比较B2 vs B0而把新增标签/容量全部归因于ranking loss。新head epoch0通过零residual复现原CDF utility；其输出是scalar confidence，**不再是六个CDF概率**。不能用它宣称逐μ概率校准改善。

每个B1/B2同时测试：
- `both`：新depth＋新最大评分优先级（主部署口径）。
- `depth_only`：新depth＋原query最大评分。
- `rank_only`：原depth＋新最大评分（诊断性混合，不是准确的实际动作概率）。

固定优先级也不能固定NMS/碰撞后的存活集合。所有版本走同一后处理，不能把候选Q当AP。

## 3. 代码入口

- `rare_grasp/core.py`：loss、scalar head、manifest、物理动作身份、四格decode。
- `rare_grasp/runtime.py`：main严格加载、完整长度feature捕获、原decoder parity、两域dataset和exact evaluator。
- `rare_grasp/train_base.py`：继承main mixed trainer，仅改成训练域帧留出验证并核对5200训练帧。
- `python -m rare_grasp.pipeline`：manifest/bind/mine/index/fit/infer/eval/compare/diagnose-cache。
- `scripts/rare_mix10/run.sh`：显式分阶段入口，不自动启动整条流水线。
- `tests/test_rare_mix10.py`：CPU公式、ties、数据/动作合同、合成训练、CLI和模拟native runtime测试。

普通main训练/推断不导入上述包；没有改动生产模型、evaluator或旧工作单。

## 4. Codex执行顺序（不要跳过smoke）

先读取AGENTS.md、检查git状态/worktrees、GPU和磁盘。不得清理他人任务、删除checkpoint、自动覆盖旧结果。

```bash
git fetch origin main exp/rare-ranking-mix10
git worktree add /home/robotarm/EconomicGrasp-rare-mix10 \
  -b codex/rare-mix10-run origin/exp/rare-ranking-mix10
cd /home/robotarm/EconomicGrasp-rare-mix10
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_rare_mix10.py' -v
python -m compileall -q rare_grasp tests/test_rare_mix10.py
bash -n scripts/rare_mix10/run.sh

export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export RUN_ROOT=/data/robotarm/result/grasp/rgbgrasp/rare_mix10_run01
export ARCH=point_gvar
bash scripts/rare_mix10/run.sh prepare
```

### S0：完整模型共同基座

```bash
GPUS=0,1,2 BATCH_SIZE=3 bash scripts/rare_mix10/run.sh base
bash scripts/rare_mix10/run.sh bind
```

运行前先确认资源，命令中的GPU编号不是实时空闲声明。继承main的DDP sampler会为整除world-size做padding；5200是唯一训练图像数，日志需单列实际更新曝光量。

可严格复用**main-compatible、同协议mixed10、固定e19**权重：设置 `BASE_CHECKPOINT` 后仅run bind。bind/runtime核验：checkpoint epoch=20、两域2600、总5200、训练比例0.1、architecture/design metadata、pose mode、文件SHA。**mixed50历史e15会被拒绝**。不能修改metadata/strict=False来绕过。

配置模板仅为本轮设置；已有权重chunk/design参数不同须如实调整template并验证，不能改权重metadata。保留训练来源和val使用方式；历史测试域选模权重不等价于新协议。

### S1：先各域单帧cache smoke，再正式标注

```bash
python -m rare_grasp.pipeline mine \
  --config "$RUN_ROOT/$ARCH/bound.json" --protocol "$RUN_ROOT/frames_mix10.json" \
  --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
  --partition train --domain original --families 2 --limit 1 \
  --output "$RUN_ROOT/$ARCH/smoke_original" --device cuda:0
# 同样运行 --domain gntrans，输出到 smoke_gntrans。
```

两个域都必须通过：完整1024×12 native forward、score-feature维度、predicted/used depth一致、17D production decode parity ≤1e-6、exact evaluator动作有限性。

**禁止把少量queries提前裁剪再重放原Conv1d以绕过计算。** 当前实现先运行完整native图，再提取features；避免历史TF32输入长度敏感问题。

正式标注：

```bash
bash scripts/rare_mix10/run.sh mine
bash scripts/rare_mix10/run.sh index
```

缓存按帧保存 `.npz/.json`，已有完成帧只在合同/hash完全相同情况下复用；孤立或损坏文件停止并要求调查。支持按frame列表取模分片：`NUM_SHARDS=3 SHARD_INDEX=0/1/2`，不同进程显式绑定GPU；每个任务自己有official evaluator，CPU占用需预算，禁止盲目开多个16-worker评测池。

index只有5200train和40val完整时通过。`--allow-partial`和fit的`--smoke`只用于合成/烟测；正式test拒绝smoke-trained head。

本轮exact标签预算为665,600训练动作/模型，不是仅几百个旧case。先检查CAD/Dex-Net和存储资源；若需缩小queries预算，须在读新测试结果前登记新的协议并同时作用于B1/B2，不得把缩小运行称为默认32-family实验。

### S2：相同预算训练MSE与RARE scorer

```bash
bash scripts/rare_mix10/run.sh fit
```

两组读取同一exact cache，初始化seed和image batch顺序相同。Pair RNG独立且由image key/epoch派生；MSE组不因少调用pair RNG而改变数据或模型随机性。每图无有效非并列pair时pair loss=可微0，但仍训练MSE。日志输出eligible/used pairs，检查pair监督覆盖。

正式模型为`heads/point/last.pt`和`heads/rare/last.pt`第20轮；`history.json`含epoch0及逐轮验证Q/MSE。不得依据Novel或Seen AP重选轮次。

### S3：单帧网络推断 smoke → 正式三split

```bash
python -m rare_grasp.pipeline infer \
  --config "$RUN_ROOT/$ARCH/bound.json" --protocol "$RUN_ROOT/frames_mix10.json" \
  --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
  --ranker "$RUN_ROOT/$ARCH/heads/rare/last.pt" \
  --domain original --split test_seen --limit 1 \
  --policies both,depth_only,rank_only --output "$RUN_ROOT/$ARCH/infer_smoke"

DOMAIN=original bash scripts/rare_mix10/run.sh test
# 需要GN-Trans时独立启动：
# DOMAIN=gntrans bash scripts/rare_mix10/run.sh test
```

每split/版本验证所有780帧dump哈希、finite N×17，再调用main eval.py `--sample_interval 10 --num_workers 16`，验收AP数组[30,26,50,6]、范围[0,1]。正式eval拒绝不足780帧的smoke结果。各split/条件顺序运行，避免CPU池叠加。

两域后处理统一使用**原Original sensor cloud**；GN-Trans自身crop仍可能依赖virtual depth，不宣称纯材质因果，也不宣称全系统strict raw-RGB-only。

### S4：主要比较及统计

按Seen/Similar/Novel顺序给出B1、B2的三个数组：

```bash
python -m rare_grasp.pipeline compare \
  --left "$RUN_ROOT/$ARCH/test/original/test_seen/point/both/ap_test_seen_realsense.npy" \
  --left "$RUN_ROOT/$ARCH/test/original/test_similar/point/both/ap_test_similar_realsense.npy" \
  --left "$RUN_ROOT/$ARCH/test/original/test_novel/point/both/ap_test_novel_realsense.npy" \
  --right "$RUN_ROOT/$ARCH/test/original/test_seen/rare/both/ap_test_seen_realsense.npy" \
  --right "$RUN_ROOT/$ARCH/test/original/test_similar/rare/both/ap_test_similar_realsense.npy" \
  --right "$RUN_ROOT/$ARCH/test/original/test_novel/rare/both/ap_test_novel_realsense.npy" \
  --output "$RUN_ROOT/$ARCH/rare_minus_mse_ap.json" --bootstrap 5000 --seed 0
```

检查inference manifests的同一基座、域、frame协议、碰撞、seed、angle/width合同；对每split按scene配对bootstrap，再等权汇总Mean。不是candidate/frame IID bootstrap，不覆盖training seed方差。

可用同模型冻结候选的validation/test exact cache补充`diagnose-cache`，核验selected-Q gain = regret reduction、oracle不变、Oracle@2/4与tie-aware depth agreement。**若已有GT来自不同动作/模型，不复用旧标签。** 本轮测试cache不进入fit；无需为了主AP强制重评全测试cache。

## 5. 结果表与停止条件

|Method|Seen|Similar|Novel|Mean|RARE−MSE scene CI|状态|
|---|---:|---:|---:|---:|---|---|
|Native S0|TBD|TBD|TBD|TBD|—|未运行|
|MSE / both|TBD|TBD|TBD|TBD|—|未运行|
|RARE / both|TBD|TBD|TBD|TBD|TBD|未运行|
|MSE / depth-only|TBD|TBD|TBD|TBD|—|未运行|
|RARE / depth-only|TBD|TBD|TBD|TBD|TBD|未运行|
|MSE / rank-only|TBD|TBD|TBD|TBD|—|未运行|
|RARE / rank-only|TBD|TBD|TBD|TBD|TBD|未运行|

必须记录：S0/RARE训练图像、family/action/pair数量、标签耗时与来源、epoch/updates、参数量、训练/推断时间、每split全部六μ、trace depth变化、碰撞后候选数；新head零初始化parity，negative/tie统计。

判别：
1. B2优于B1才支持pair objective的独立价值；仅优于B0不能分离新标签/容量影响。
2. depth-only改善而both不改善：排序耦合仍是问题，不按测试结果临时选择“最佳policy”作主表。
3. 验证/固定候选Q改善而AP不改善：如实保留，不称已解决NMS/top-k。
4. 若旧matcher与网络图出现实质错误，先停；不得靠更换checkpoint、删失败场景、放宽parity阈值通过。
5. 单seed正信号仍需重复seed及更广测试，才讨论稳定性；本轮无需新reader或全部网络解冻。

## 6. 验收与交付

- [ ] 新分支基于指定main；生产代码无diff；所有参数/源码SHA记录。
- [ ] CPU测试与compileall通过；真实Original/GN-Trans smoke分别通过。
- [ ] 基座固定e19，mixed10+10身份、原训练CLI、checkpoint SHA完整；不是mixed50微调。
- [ ] cache索引5200/40完整，GT只用于label，标签实际width包含1.2放大/0.1m截断。
- [ ] MSE/RARE相同capacity/data/order/update预算；排除equal-Q pairs且无方向margin。
- [ ] 所有推断1024queries，固定source angle/view/width；sidecar评分来自正确base SHA。
- [ ] Original三split全部AP验收；可选GN-Trans另表且保留预处理边界。
- [ ] 保存`RESULTS_CN_RARE_MIX10.md`、train protocols、cache indices、logs、完整AP数组与bootstrap JSON。
- [ ] 区分本地CPU测试、真实CUDA/CAD smoke、正式training/eval状态，不预填TBD结果。

当前交付：实现与本地CPU合成测试；没有在117上启动S0/S1/S2/S3，不存在新的实际AP。
