# GVAR 固定物理动作诊断 — Codex 独立工作单

日期：2026-11-07。新分支 `exp/gvar-fixed-action-diagnostics`，从 **main@7a956d37aaa427a32e598f21fdc6b910ed3c2704** 创建。**不是旧 RayRoPE P1/P2/P3 分支的续写或合并。** 本次生产模型、训练器与 evaluator 源码不改动；只增加 opt-in 诊断包、测试、配置示例和本文。

## 0. 任务与已有证据

问题不是再提出一次 action-conditioned reader，而是验证已有 GVAR 是否用动作对齐的 RGB/geometry evidence 改善实际 insertion selection。

用户 `RESULTS_CN_P3.md` 中 Point RayRoPE 的报告值（不是本轮新结果）：18帧、9场景、288族；predicted-width selected Q=0.232639、oracle Q=0.351852、regret=0.119213；136/288族具有质量不同的 depth pairs。Width predictor 优于固定8/10cm，fixed/predicted-width insertion 对照未明显解释原缺口。历史 GVAR 的 AP/R0–R4/支持策略四格实验已有结果，不要无理由重训或重复完整官方 AP。

**本工作单的唯一主效应：同一个冻结候选库、同一个 oracle 下，GVAR 是否选择更高质量的动作？**

\[
\Delta U=Q(g_{\arg\max s_G})-Q(g_{\arg\max s_P})=R_P-R_G.
\]

这里与 P3 宽度协议实验不同：本轮固定的是 oracle；此前是 selected action 固定、oracle 变化。

### 状态

- 新增脚本与 CPU 合成测试可在本地运行；未进行真实 checkpoint/CUDA/CAD 验收。
- `main` 未修改；没有训练、未启动服务器任务。
- 历史 `.pt` checkpoint 可能不能用 main 的 strict metadata loader 直接加载。**禁止 strict=False、修改 checkpoint metadata 或换 checkpoint 让脚本勉强通过。**

## 1. 代码清单

| 入口 | 功能 |
|---|---|
| `diagnostics/gvar_fixed/runtime.py` | native capture + 公共几何重算 + 固定 view/query 的 score replay + native parity |
| `diagnostics/gvar_fixed/hooks.py` | 临时 reader 干预；退出/异常均恢复；无新增权重与持久状态 |
| `diagnostics/gvar_fixed/contracts.py` | 完整17D物理动作、SHA、六μ标签、候选库与CSV合同 |
| `python -m diagnostics.gvar_fixed.run` | G0/G1/G2及可选 GVAR-only geometry stress；官方模式独立评价动作 |
| `python -m diagnostics.gvar_fixed.analysis` | 固定候选选择收益、pairwise、Oracle@2/4、repair/damage、六μ、深度剖面、scene CI及中文摘要 |
| `python -m diagnostics.gvar_fixed.calibration` | 独立场景拟合四个 depth offsets；测试时强制防止 scene 泄漏 |
| `tests/test_gvar_fixed.py` | 合成 CPU 配对、干预恢复、跨模型路由、统计和CLI回归测试 |

本包不依赖旧分支的 diagnostics。普通训练/推理不导入它，正常路径不变。

## 2. G0：先核验模型和执行合同

### 2.1 已知的历史身份线索（必须在117核对）

- P3 Point checkpoint SHA256：`0c189e1c70dce12d82098216318290ebd6a855ec0c85ee88263bbb721d894ca9`。
- 50% Original + 50% GN-Trans 的 Point+GVAR36 e15 报告 SHA256：`2c7a4097b6350935640d249aa1c0ef2f2871d454417a6a83b95fb37d7a6d1675`。
- GVAR报告实际完成e0–e15；不能写为完整e19训练。不要换成10%+10%的独立GVAR、或三源GC6D模型，却沿用本轮受控比较的名称。
- 上述为历史报告提供的身份，不代表当前服务器文件已核验。查找实际训练protocol、runtime、dataset sampler与严格加载器，保存源码/权重SHA和原始CLI。

### 2.2 不破坏用户工作区

```bash
git status --short
git worktree list
git fetch origin main exp/gvar-fixed-action-diagnostics
# 优先创建新worktree；已有分支不强制重置，不stash用户修改。
git worktree add /home/robotarm/EconomicGrasp-gvar-fixed \
  -b codex/gvar-fixed-execution origin/exp/gvar-fixed-action-diagnostics
cd /home/robotarm/EconomicGrasp-gvar-fixed
python -m unittest discover -s tests -p 'test_gvar_fixed.py' -v
python -m compileall -q diagnostics/gvar_fixed tests/test_gvar_fixed.py
```

先读取父目录及仓库 AGENTS.md。核实 GPU/磁盘现状；不要删除 checkpoint/dump、杀其他任务、自动启动训练或全量AP。

### 2.3 Runtime adapter

默认 `build_main_runtime(config, device)` 完整实现 main-compatible 模型加载：对checkpoint先检查必填SHA，`validate_checkpoint_design`，`load_state_dict(strict=True)`。参数来自config的 `model_args`；示例只是接口示例，不能替代真实训练参数。

历史runtime不兼容时，Codex应在隔离目录复用已验收的历史strict loader和模型图，提供显式 `module:function` factory，返回具有以下接口的对象：

- `checkpoint_sha`：实际核验的模型SHA；
- `settings`：至少 `max_width=0.1`、`num_points`，并保留原模型参数；
- `reader`：GVAR时为真实volume_rel reader，Point时为None；
- `activation()`：调用期间恢复历史全局配置的context manager；
- `capture(batch)`、`validate_native(snapshot, atol)`、`replay(own, reference, condition)`、`make_bank(snapshot,n,seed)`、`selected_outputs(...)`。

若历史模型结构与 main 相同，可复用 `MainRuntime(model,sha,settings,activation,decode_fn)`；`decode_fn` 应是历史生产 `pred_decode_center_view_angle`。若接口不同，须明确适配并加parity测试，**不能仅添加假属性绕过检查**。`--dataset-factory module:function` 可接历史 dataset 构造，返回 `(dataset, collate_fn)`，但两模型必须读取同一批输入张量。

当前adapter实际重算以下路径：
1. 每个模型自身 backbone/DPT/proposal 生成原始图像特征；不交换 hidden features。
2. 使用同一reference预测depth和K，重算 target spatial enhancer。
3. 在reference pixel indices上重新gather target seed features，使用reference物理centers。
4. 重算 target ViewNet；临时替换selector的选择函数以锁定reference view IDs（原selector的eval分支不会直接尊重training-only forced-view参数）。
5. 重新执行所有angle的RayRoPE group、GVAR reader和head；外部固定angle仅决定读取哪个score。
6. runtime分别检查spatial/ViewNet/RayRoPE/GVAR的公共几何路由；允许的例外仅为明确标注的GVAR-only geometry bias。

**native replay必须恢复原CDF logits、width输出；production decoder选中动作的17D bank必须匹配。identity residual permutation必须逐元素不变。** 默认tol=1e-5用于重复前向，bank 17D parity=1e-6。出现偏差先诊断，不扩大容差掩盖问题。

不直接拼接不同模型的query index或最终score；不将GT点云、GT depth、CAD标签输入网络。dataset workspace/crop可能仍依赖sensor depth，报告应保留此边界。

## 3. G1：同一GVAR权重、同一物理动作的证据干预

每帧均匀选16个native query anchors，不按AP或失败筛选；每个query冻结native最佳angle，保留全部4个depth以及reference逐depth执行width。三组共享相同中心、旋转、插入、宽度、物理ID和exact标签。

| Condition | 干预 |
|---|---|
| `native` | 完整dynamic reader |
| `rgb_fixed` | 复用R1：只固定RGB feature sampling位置；geometry/metadata/mask保持dynamic |
| `perm_cycle` | 固定置换 `[1,2,3,0]`，只交换reader residual `delta_h_d`，保留原query embedding、depth head、物理动作、labels |
| `perm_identity` | 自动parity检查，非性能条件；不执行减加以避免identity数值误差 |

\[
h'_d=h_d+\Delta h_{\pi(d)}.
\]

输出始终使用reference动作的实际width；intervention下width head的新输出只做日志，不能执行。所有干预仅在eval/inference-mode启用，结束与异常都会恢复hook/method/counterfactual状态。交换的是reader update，不是最终分数或标签；该试验检验动作更新对应关系，而非孤立RGB内容效应。

### 最小入口（只加载GVAR）

先把配置示例复制到服务器运行目录并填写真实路径、SHA、原始model_args；不在git内提交机器路径/权重：

```bash
python -m diagnostics.gvar_fixed.run \
  --gvar-config /ABS/run/gvar.json \
  --references gvar \
  --frames /ABS/run/one_verified_frame.csv \
  --dataset-root /ABS/graspnet --camera realsense \
  --output /ABS/run/g1_smoke --families 2 --device cuda:0
```

帧manifest为 `scene_id,anno_id,split`；split仅seen/similar/novel。配置中的smoke CSV是格式示例，不是历史18帧。正式探索沿用真实P3 frame manifest，不能替换成自动挑选的方便场景。

## 4. G2：双向reference的共同候选比较

在 **Point-reference** 和 **GVAR-reference** 两个库内分别评价：

- 同一reference depth、K、中心、view、angle、四个depth、逐depth width；
- Point自己的图像特征 + 公共几何；
- GVAR自己的图像特征 + 公共几何；
- 相同动作只运行一次official CAD/table/empty/Dex-Net，不做sensor collision预过滤。

两个库分开统计，不把它们当作增加一倍独立样本。参考来自模型生成分布，跨参考评分可能OOD；双向检查降低单一候选来源偏好，但不等于严格训练架构消融。

```bash
python -m diagnostics.gvar_fixed.run \
  --point-config /ABS/run/point.json --gvar-config /ABS/run/gvar.json \
  --references point,gvar \
  --frames /ABS/run/p3_original_18frames.csv \
  --dataset-root /ABS/graspnet --camera realsense \
  --output /ABS/run/gvar_fixed_pilot --families 16 --device cuda:0
```

首次只用一帧、两族做smoke。两个模型在同一device串行运行，不支持把mutable global cfg共享给并发线程。运行目录必须新建/为空；中途失败输出`status=failed`与已完成frame文件，不覆盖历史证据，不假装完整完成。

**限制：** 当前score并未以任意外部width为显式输入；GVAR envelope固定0.06m。公共width只是统一评价动作的合同，不能称exact-width-conditioned scorer。

### 正式统计

```bash
python -m diagnostics.gvar_fixed.analysis \
  --csv /ABS/run/gvar_fixed_pilot/scores.csv \
  --output /ABS/run/gvar_fixed_pilot/analysis.json \
  --markdown /ABS/run/gvar_fixed_pilot/RESULTS_CN_GVAR_FIXED.md \
  --expected-frames 18 --expected-scenes 9 \
  --expected-families-per-reference 288 \
  --require-cross-model --require-controls --bootstrap 5000
```

默认两reference、Point/native与GVAR三条件，共9216 score rows（18×16×4×2×4）。物理评价动作最多2304行；重复物理action按帧去重评价。G1单reference有3456 score rows。不能用这些行数替代独立scene数量。

主指标：paired selected-Q gain，且必须逐family等于paired regret reduction。辅助：same-angle tie-aware agreement、可比较pair/family数、Oracle@2/4、hard-negative质量差、repair/damage及selected failure分类、六个μ、逐depth profile。输出包含scene→frame→family等权、split内scene resampling；不足两个scene的split不产生看似确定的CI。Across-query Rank1/2是次要指标。

CSV物理参数、标签、geometry hash、source/scorer SHA、reference bank ID一项不一致即中止。缺失family不能取交集偷偷删掉。数据为冻结候选离线quality，不是官方AP或机器人成功率。

## 5. G3：竞争解释——独立场景四偏置baseline

先在**未进入当前18帧测试的独立scene**上运行相同协议形成validation CSV。为Point/native和GVAR/native分别拟合：
\[
\tilde s(d)=s(d)+b_d.
\]

四偏置由场景平衡的MSE解析拟合，固定ridge=0.1，不在测试上搜索；不clip，校正后值不再解释为概率。并列报告raw和calibrated，不只报告有利者。

```bash
python -m diagnostics.gvar_fixed.calibration \
  --validation-csv /ABS/run/gvar_fixed_validation/scores.csv \
  --output /ABS/run/frozen_depth_offsets.json --ridge 0.1

python -m diagnostics.gvar_fixed.analysis \
  --csv /ABS/run/gvar_fixed_pilot/scores.csv \
  --calibration /ABS/run/frozen_depth_offsets.json \
  --output /ABS/run/gvar_fixed_pilot/calibrated_analysis.json \
  --markdown /ABS/run/gvar_fixed_pilot/RESULTS_CN_GVAR_CALIBRATED.md \
  --require-cross-model --require-controls
```

分析器检查scene IDs完全不重叠（不同frame/reference也不能绕过）、source/scorer checkpoints一致。一个全局严格单调score变换不改变argmax，不能用它替代depth-dependent baseline。旧18帧已多轮后验分析，仍只可作为探索pilot；关键效应需要独立scene确认。

## 6. G4：有条件后续，不在本轮默认全部启动

### 已实现可选几何敏感性入口

若G1/G2显示selection收益，再添加 `--geometry-bias-mm=-5,5`。该设置仅修改GVAR relative geometry的depth，保持upstream spatial/ViewNet/RayRoPE、RGB支持位置、动作与标签不变。名称明确为reader-stream stress，不是全系统鲁棒性或部署校准。

### 当前未实现/需要下一阶段另行批准

- contact/closing vs finger/palm 的role干预及等有效token数随机控制；先看G2修正/破坏类型决定，不能按attention mass直接推断物理功能。
- 新的GVAR结构、训练、role auxiliary losses、完整官方AP、多seed、机器人执行。
- 独立新场景确认：可直接使用本脚本的新manifest，但由用户/研究计划预先冻结，不能按结果筛选。

## 7. Codex验收与停止条件

1. 记录main来源、新分支HEAD、源码改动、服务器未提交diff、两权重hash和真实训练protocol。只使用原严格loader；不修改旧worktree。
2. CPU合成测试与compileall通过；首次实际CUDA capture/replay需确认所有native logits/width、17D native winner parity通过。
3. `perm_identity` logits与width逐元素相同；干预结束state_dict和正常输出保持不变；无遗留hook。
4. 三条动作冻结不变：center/R/d/width、candidate ID、exact quality。两个scorer每条condition的geometry routes有审计。
5. 官方exact evaluator源码与配置留档，全部动作未经碰撞预过滤；friction与six-mu bits一致。
6. 同reference、同family的oracle必须一致；`gain=regret_reduction`逐条成立。
7. 总行数、scene/frame/family数完整；`run_manifest.status=complete`，分析器`--expected-*`通过。
8. 交付scores.csv、banks/*.npz、每帧audit.json、run_manifest.json、analysis.json、中文报告、test logs；分清mock测试与真实模型smoke。
9. 如果历史adapter无法达到native parity，先提交阻塞说明与差异，不允许用另一模型或只比较最终AP伪称固定候选评分已完成。

### 研究决策

- GVAR共同候选selected Q提高，R1/错配降低收益：已有设计兑现动作对齐读取目标，优先巩固论文，不先增加同类reader。
- 分数改变但选择质量不改善：只证明模型依赖该支路；查监督和校准，不宣称证据具有决策价值。
- 收益被四个独立验证depth offsets解释：不能主要归因于场景特定证据。
- GVAR确有选择收益但geometry stress下消失：下一步优先证据可靠性，而非再提一次action conditioning。

本工作单是实现/执行合同，不含新增实测GVAR结论。
