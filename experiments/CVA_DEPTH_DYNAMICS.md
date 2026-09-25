# CVA metric-depth gradient dynamics

This experiment tests whether supervised grasp gradients through predicted
metric scene depth damage geometry. CDF insertion depth is a different variable.
The reference implementation starts at main
`52d09f925059bec3643610ecf1f1722894627ee5` on branch
`codex/cva-depth-gradient-dynamics`.

## Experimental boundaries

| Route | Controlled boundary |
| --- | --- |
| `gse` (E) | GSE mean depth before positional geometry encoding |
| `seed_xyz` (Q) | Predicted seed depth before backprojection |
| `support` (C) | CVA support depth map before spatial sampling |

`none` retains all three original stop-gradient boundaries. `all` opens all
three. Comma-separated combinations are supported. Changing these boundaries
must preserve forward values. The auxiliary-map detach and discrete selections
are unchanged. Closing C does not block gradients through a Q-dependent grid or
residual center. The diagnostic entry point uses the existing Stage-1 model,
dataset, label adapter and supervised CDF losses with KD disabled.

## Verified setup, 2026-09-26 Hong Kong

- Training host: `10.30.7.117`, `robotarm`; hostname `hkclrgpuf02`.
- Six RTX 3090 GPUs, 24 GiB each, were idle at preflight. Check again before launch.
- Interpreter: `/home/robotarm/miniconda3/envs/grasp/bin/python`;
  PyTorch `2.5.0+cu118`, CUDA runtime `11.8`; actual model import succeeded.
- The original `/home/robotarm/EconomicGrasp` has unrelated user modifications.
  Work is isolated in `/home/robotarm/EconomicGrasp-depth-dynamics`.
- Dataset: `/data/robotarm/dataset/graspnet`; RGB and extended CDF cache readable.
- Logs: `/data/robotarm/result/grasp/rgbgrasp/log/cva_depth_dynamics`.
- Diagnostics: `/data/robotarm/result/grasp/rgbgrasp/experiment/cva_depth_dynamics`.
- At preflight `/data` had about 47 GiB available. Retain six rolling complete
  snapshots plus initial/event/final evidence; monitor remaining space.

The user-supplied Stage-1 checkpoint was verified on `gpu04` (`10.30.7.119`):

```text
/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar
SHA256: 0cb8cd54bd5eef44a81c8d8de6969003377224d41a84b2002ba662bb8ce0c69c
```

Its metadata records epoch 16, distill stage 1, contract version 2, predicted
geometry, image-FPS, executed depth head, `global_film`, `use_fuse_depth=True`,
pose hidden dimension 64, ray gravity dimensions 64/32, camera keys
`camera_pose_vec` / `camera_gravity_vec`. It contains model weights but no
optimizer state. Both arms therefore use an explicit **weights-only restart**.
The copy on the training host is
`.../log/cva_depth_dynamics/init/stage1_epoch15.tar`.

## Evidence gates

1. P0: identical real-batch forward outputs across routes, depth supervision
   connected, D0 grasp-to-depth disconnected, each open route connected.
2. P1: per-loss raw/weighted gradients, None-versus-zero distinctions, state
   preservation, output-space direction probes. Finite differences with frozen
   assignments are a separate requirement; native graph derivatives do not
   substitute for it.
3. P2: only paired D0 (`none`) / D1 (`all`), first 50 successful updates, then
   review before extending to 500 and 2000. Match initialization, data order,
   per-example preprocessing RNG, optimizer configuration and update count.
4. P3/P4: path localization and intervention require observed reproducible
   failure. A negative gradient cosine alone is not a causal mechanism.

Do not rename `test_seen` to validation. The supplied checkpoint's original
training scenes must be checked before claiming any held-out validation split.
Fixed train-frame depth probes are training diagnostics; native losses on
fixed frames are not fixed physical-query/assignment probes. No official AP
evaluation is part of the initial mechanism screen.

Every executed run must store the source commit, full resolved configuration,
checkpoint hash, data manifest, route flags, actual paths, RNG/loader cursor,
parameter/optimizer contract and unresolved acceptance items. Report a missing
measurement as missing; never create placeholder result files implying success.

## Interpretation limits

Separate image flattening, loss of local contrast, mean drift and saturation.
Do not count a flat initialization as training-induced collapse. A warm-start
run without collapse only supports non-reproduction for its specific budget.
The historical preprocessing uses captured depth for the crop/workspace mask,
and fused synthetic depth for supervision where configured. Record these uses
separately from the RGB-only model geometry input.

Runtime commands and measured results are added after verification.
