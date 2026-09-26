# FD8 补充独立复核（不替代原 P0/P1 证据）

本报告补充 `P0_evidence/P1_INDEPENDENT_REVIEW_ZH.md` 的单图 FD 复核。新的实测范围为 8 张图的初始状态方向检查；未运行模型更新，不是塌缩动态或机制实验。原文件保持不变。

## 来源与实际覆盖

- 运行：`20260926_stage1_fd8_retry_seed0`，GPU 上已执行的 audit；源码 HEAD `aa4ffd20ee52f0a60f776c1c3b90d048ef7fa216`，tracked diff 为空。
- 初始化 SHA256：`0cb8cd54bd5eef44a81c8d8de6969003377224d41a84b2002ba662bb8ce0c69c`。
- FD 文件：`20260926_stage1_fd8_retry_seed0/diagnostics/P1_FD8/local_directional_probe.csv`，SHA256 `db21b2440ea9cd4bc7bad90dbf522349fa8eb32f69e950516dc0444f8f66ae2e`。
- 1248 个唯一 FD 行：8 批 × 1 图 × 2 方向 × 3 步长 × 2 协议 × 13 loss；无重复键。全为 all 路由、GT-valid foreground、train 模式。每图只固定一个随机前向实现，不是 8 个独立训练 seed。
- native-autograd 表 112 行，仅 none/all 两路，8 批。contract 中的 replay audit 记录这两路的前向一致和 baseline 连通性检查通过；此补充运行不是新的完整五路 P0 gate，`p0_passed=false` 不应被当成 FD 失败。
- `directional_probe.rows=156,batch=7` 是最后一批记录，非全文件总数；`heldout_test_used=false` 对应 validation probe 帧数为 0，不代表这次 audit 没有使用 test_seen。

帧身份依据相同数据 manifest 与实际 audit 的 linspace 选样规则核对：

| batch | split | dataset index | scene | frame |
|---:|---|---:|---|---:|
| 0 | train | 0 | scene_0000 | 0 |
| 1 | validation_test_seen | 1097 | scene_0104 | 73 |
| 2 | train | 7314 | scene_0028 | 146 |
| 3 | validation_test_seen | 3291 | scene_0112 | 219 |
| 4 | train | 14628 | scene_0057 | 36 |
| 5 | validation_test_seen | 5485 | scene_0121 | 109 |
| 6 | train | 21942 | scene_0085 | 182 |
| 7 | validation_test_seen | 7679 | scene_0129 | 255 |

## frozen 连续子函数：多数局部检查吻合，不能统一宣布所有步长通过

下表仅统计日志中**直接 FP32 weighted-task 标量**的中心差分。相对误差分母为 `max(abs(FD),abs(AD),1e-10)`；1%/5%仅为透明的描述性分档，不是事后新增的自动验收标准。

| 方向 | 相对步长 | 误差中位数 | 最大误差 | <1% | <5% |
|---|---:|---:|---:|---:|---:|
| mu | 0.0001 | 0.079% | 0.647% | 8/8 | 8/8 |
| mu | 0.001 | 0.174% | 1.208% | 6/8 | 8/8 |
| mu | 0.01 | 2.618% | 19.388% | 2/8 | 6/8 |
| alpha | 0.0001 | 5.451% | 14.886% | 1/8 | 4/8 |
| alpha | 0.001 | 0.542% | 29.506% | 6/8 | 7/8 |
| alpha | 0.01 | 1.239% | 116.918% | 4/8 | 6/8 |

mu 最小步长的物理扰动为 0.0380–0.0437 mm；view/CDF/width 的直接加权 FD 均 8/8 在 1% 内。weighted-depth 为 7/8 在 1% 内，最大 1.184%。alpha 最小步长的最大像素扰动为 0.00663–0.02522 mm，较容易出现小差值相消；中间步长为 0.0663–0.2522 mm。较大步长不是更可靠的导数估计，尤其可能跨越分段边界。

objectness/graspness 在所有 8 图、两方向、全部步长均 unused/None，frozen 与 native 的 FD 均为 0。frozen 的 seed/query/NN/CDF-mask 身份切换率全为 0；attention-valid mask 由 replay 代码固定并逐次校验。表里的 `attention_native_mask_switch_rate` 是未施加固定覆盖前的候选 mask，不是实际 frozen mask 的变化。

## batch3 alpha：必须保留原始差异，并分开解释数值相消与非线性区间

batch3 是 test_seen 的 scene_0112/frame219。weighted task AD 为 `+0.001268297364`；weighted view AD 为 `+0.001227826990`。

| alpha 步长 | 直接 task FD | task 绝对误差 | task 相对误差 | 直接 view FD | view 相对误差 |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | +0.001490116119 | 0.000221818755 | 14.886% | +0.001192092896 | 2.910% |
| 1e-3 | +0.000894069672 | 0.000374227693 | 29.506% | +0.000894069672 | 27.183% |
| 1e-2 | -0.000214576721 | 0.001482874086 | 116.918% | -0.000107288361 | 108.738% |

无需新增 GPU 运算即可做一个后处理对照：将**已记录的各 raw-loss secant**按原权重在 Python FP64 中相加，不再对已舍入的 FP32 总 loss 相减。这保留同一网络输出和已有原始测量，并不等于网络做了 float64 前向，也不替换原 CSV。

- batch3 在 1e-4 的重组 task FD 为 `+0.001271255314`，相对 AD 误差 0.233%，明显小于直接总标量的 14.886%。8 图 alpha 最小步长重组后，误差中位数 0.686%，最大 1.382%，全部同号且 8/8 在 5% 内。末端求和/相减的量化能解释相当一部分小步长总 loss 误差。
- batch3 在 1e-3 的重组 FD 仍只有 `+0.000908505172`，误差 28.368%；1e-2 仍反号。因此不能把较大区间差异全部归因于最终总标量舍入。
- 同一 batch、1e-3 的 raw-view AD 为 `1.22782693e-5`，正向单边 secant 为 `1.21071935e-5`，负向为 `5.58793545e-6`。明显的不对称与区间跨过 ReLU/clamp/双线性网格单元等分段变化相容；尚不能确定具体是哪一处，也不是 backward 错误证据。冻结离散身份不等于整个网络在该区间光滑。

所以可以写“小扰动下重组总方向具有较好数值支持”，不能写“所有单项、所有步长已通过数值验证”。如果以后必须依赖 batch3/view 的精确斜率再决定是否加针对性检查；当前不建议仅盲目缩小步长，更不能将最终 scalar `.double()` 冒称 float64 loss-reduction 验证。

## 方向筛查：范围依赖，仍不是网络更新

8 图前景区域的 weighted-task mu 导数为 5 正/3 负；alpha 为 7 正/1 负。alpha 的 7 个正值表示这些特定局部输出方向的负梯度倾向降低前景对比度参数，但不等于实际 Adam 更新会变平，也不等于 GT 误差必然变差。

- mu：task 与 depth 的方向在 6/8 图相反；合计目标相对 depth 单项方向翻转仅 batch4、5、7。
- alpha：task 与 depth 在 batch1、2、6、7 相反，其余 4 图同向。合计目标相对 depth 单项方向翻转仅 batch1、7；其中 batch1、7 的 task alpha 分别 +0.035588/+0.089351，depth 分别 -0.020550/-0.028916。可列为后续观察点，不能认定有害更新。
- 补充 native-autograd 表在**全 GT-valid**区域的 all grasp alpha 仍为 4 正/4 负。该表逐项 raw、只有 grasp_total 加权，且在 audit 专用 RNG 中计算；FD 使用前景区域及另一种固定 RNG 上下文。两张表不能逐行互当校验，7/1 与 4/4 不构成矛盾。

本次仍没有逐路 GSE/Q/C FD，也没有参数空间 FD 或真实更新检验；不能借此更新必要性/充分性结论。

## native 重算：小扰动也会改变离散目标，secant 不是同一函数的导数

下表 NN/CDF 切换率先对每张图取正负扰动两侧较大值，再跨 8 图报告中位数/最大值。反号是 native secant 与该步长 frozen FD 的符号不同。

| 方向 | 相对步长 | task native/frozen 反号 | NN 切换：中位/最大 | CDF mask 切换：中位/最大 |
|---|---:|---:|---:|---:|
| mu | 0.0001 | 4/8 | 0.781% / 1.172% | 0.195% / 0.684% |
| mu | 0.001 | 4/8 | 7.227% / 7.812% | 1.709% / 3.809% |
| mu | 0.01 | 6/8 | 59.668% / 65.527% | 25.732% / 37.305% |
| alpha | 0.0001 | 3/8 | 0.098% / 0.195% | 0.000% / 0.098% |
| alpha | 0.001 | 6/8 | 0.830% / 1.270% | 0.244% / 0.586% |
| alpha | 0.01 | 5/8 | 7.324% / 10.156% | 1.758% / 5.859% |

所有图、所有扰动的 image-FPS seed 身份切换均为 0；然而 view/query 身份、NN assignment、valid masks 可以变化。mu 最小步长 8/8 已有 NN 切换，alpha 最小步长 6/8 有 NN 切换。这表明“固定 image seed”不足以固定训练目标。native 相比 frozen 的跳变可用于说明局部重匹配敏感性，不可用于判定 autograd 错误，也没有证明训练在逃避监督或该反馈会放大到塌缩。

## 交付与结论边界

`FD8_AGGREGATE.csv` 保留 80 个 batch×方向×加权 loss 汇总行，包括三步长的原始 AD/FD、误差、native secant/切换率，以及仅对 weighted-task 填写的 raw-components FP64 后处理对照。它保留原 CSV 的直接测量，不覆盖或伪装通过失败项。字段后缀 `1e4/1e3/1e2` 分别对应相对步长 `1e-4/1e-3/1e-2`。

补充证据扩大了局部方向检查覆盖，并揭示前景区域的对比度方向偏向及重匹配敏感性。它没有完成训练因果定位；继续依据 D0/D1 配对轨迹决定是否有必要进入路径干预。
