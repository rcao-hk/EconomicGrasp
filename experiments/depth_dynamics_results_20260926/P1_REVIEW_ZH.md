# P1 独立只读复核

来源：`P0/gradient_audit.jsonl`、`P0/local_directional_probe.csv`、`P0/native_autograd_directional.csv`、`P0/contract.json`；代码 HEAD aa4ffd20ee52f0a60f776c1c3b90d048ef7fa216。以下是初始状态的局部梯度证据，不是塌缩复现或机制结论。

## 数据范围和去重

梯度文件原有 20,172 行；按 `(step,batch,route,scale,loss,group)` 保留首次出现，去掉 replay 重复的 none/batch0 后为 19,680 行：8 批、5 路由、raw/weighted 两种尺度。native directional 从 287 行去重为 280 行。8 批是 4 train、4 用户指定 test_seen validation，batch_size=1。

实际 loss 权重：depth=10、objectness=1、graspness=10、view=100、CDF=1、width=10。下表使用实际加权梯度，不能与 raw 梯度混读。

## 连通性与梯度量级

- none：8/8 批 depth loss 连通且非零；五项 grasp loss 到 metric-depth 输出均为 unused/None。
- 所有路由：objectness/graspness 到 metric-depth 输出均不连通。
- view：只通过 GSE 连通；all 与 GSE 的梯度一致到数值误差。
- CDF/width：GSE、seed XYZ、support 各在 8/8 批有非零梯度。

将互不重叠的 `depth_*` 参数组拼接，按组 norm 平方求和、按组 cosine 恢复点积，得到下表；DINO 不计入 depth 专属参数。比值是该项梯度范数 / 加权 depth-loss 梯度范数；统计为跨 8 批中位数。

| 路由 / loss | 参数梯度范数比 | 与 depth 梯度 cosine | 负 cosine 批数 |
|---|---:|---:|---:|
| GSE 或 all / view | 0.414 | -0.332 | 6/8 |
| seed XYZ / CDF | 0.289 | -0.215 | 5/8 |
| seed XYZ / width | 0.340 | -0.433 | 6/8 |
| support / CDF | 0.161 | +0.582 | 2/8 |
| support / width | 0.257 | +0.506 | 2/8 |
| all / CDF | 0.177 | -0.113 | 5/8 |
| all / width | 0.170 | -0.164 | 8/8 |

metric-depth 输出空间与参数空间结论不可互换。例如 seed XYZ 的 CDF/width 输出梯度范数比分别为中位数 7.682/9.131，但参数梯度比仅 0.289/0.340；不能由前者宣称网络参数更新由该项支配。

全 GT-valid 区域的 native 局部方向中，all 的 grasp-total `dL/dalpha` 为 4 正、4 负；没有跨 8 批一致的“降低对比度”方向。Q 与 C 的 grasp-total 在 mu、alpha 两个方向均为 8/8 批符号相反；`|Q+C|/(|Q|+|C|)` 中位数分别 1.85%、2.79%。这是局部方向的明显抵消，不能将大 Q-only 梯度当作 all 更新同样大的证据。

## 有限差分实际验证了什么

FD 文件共 156 行，仅 batch0/image0、all 路由、98,499 个 GT-valid foreground 像素、一个固定随机前向 realization。它冻结 seed/view 身份、两次 label 匹配结果及 attention-valid masks；物理 query XYZ、采样网格/半径和 support depth 随扰动变化。

frozen 连续分支的加权 task 导数：

| 方向 | autograd | FD 采用相对步长 | 中心差分 | 相对误差 |
|---|---:|---:|---:|---:|
| mu | 3.18360885 | 1e-4 | 3.18244124 | 0.0367% |
| alpha | 0.00583066 | 1e-3 | 0.00584126 | 0.181% |

mu 的最小扰动约 0.0437 mm；最小步长时 view/CDF/width 导数相对误差为 0.0681%/0.0265%/0.0245%。alpha 在最小步长 1e-4 时 task 相对误差 6.84%，但绝对误差仅 0.000428，且 1e-3、1e-2 均降为约 0.181%；不能宣称所有步长都同样精确，也不能因最小步长更差即判 backward 错误。objectness/graspness 为 unused，所有 frozen FD 均为零。

该图前景区域中：加权 depth 的 mu 导数为 -4.79437，task 为 +3.18361，局部偏移方向相反；但两者合计仍为 -1.61076。alpha 的 view/CDF/width 导数分别为 -0.0177546/+0.0151063/+0.0084790，task 合计 +0.00583066，而 depth 也是 +0.00102044。因此该例 task 的正 alpha 导数不能单独证明它在损害 GT 几何。

**FD 没有验证全部报告的导数。** `native_autograd_directional.csv` 使用全 GT-valid 区域；逐项值为 raw，仅 grasp_total 加权；它在 audit 专用 RNG seed 中求导。FD 使用 foreground 区域，并在 audit context 外捕获另一份固定 RNG。两张表不是同一函数/同一随机前向的逐行校验。FD 只验证其自身 `autograd_fixed_branch` 所指的单图 all-route 连续子函数；未逐路验证 none/GSE/Q/C，也未覆盖其余 7 批、全部输出维度或参数空间导数。

## native 重算的跳变

- mu 相对步长 1e-3（约 0.437 mm）时，frozen task FD=+3.17842，native secant=-12.21288；两侧 NN 身份切换 3.42%/4.00%，CDF mask 切换均 0.293%，query 身份切换均 0.0977%，image seed 身份不变。
- mu 相对步长 1e-2（约 4.37 mm）时，NN 切换达 28.81%/34.86%，CDF mask 切换 2.34%/2.93%。这里明显不是固定平滑函数的导数检查。
- alpha 相对步长 1e-3 时，frozen task FD=0.0058413，native secant=0.0669062；NN 切换仅 0.0977%/0.1953%，CDF mask 不变。差异主要出现在 view loss，说明只看 CDF mask 会漏掉目标变化。

这些结果支持“当前局部前向对离散重匹配敏感”，不证明训练在逃避监督，也不证明该反馈会造成塌缩。

## 有条件的后续观察重点

在 P2 出现可比退化后，优先检查 GSE→view，以及 Q/C 在 CDF/width 中的联合平衡和实际更新；support 当前多数参数 cosine 为正，不应先验称为有害路径。保持对局部梯度、网络 Adam 更新和训练动态的区分；当前证据没有给出必要性、充分性或塌缩因果结论。
