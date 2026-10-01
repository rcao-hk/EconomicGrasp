# Local Action Selection × Global Scoring 2×2

This is a **no-training, no-cache** intervention on the completed P0-1 Full
control. It asks where the Full scorer's official-AP gain enters the decoding
pipeline.

For every frozen query, a CDF scorer has two logically distinct jobs:

1. **Local action selection:** choose one joint `(angle, insertion-depth)`
   candidate inside the query's `A x D` action family.
2. **Global scoring:** assign a scalar utility to that already-selected physical
   action so grasps from different queries can be ranked globally.

The original decoder uses the same scorer for both jobs. This experiment
decouples them.

| Condition | Local action selector | Global scorer |
|---|---|---|
| `lbase_gbase` | Base | Base |
| `lfull_gbase` | Full | Base |
| `lbase_gfull` | Base | Full |
| `lfull_gfull` | Full | Full |

For a cross condition such as `lfull_gbase`, Full chooses the exact
`(angle,depth)`; Base is then evaluated **at that same selected candidate**.
The global scorer is not allowed to re-select an action.

All four conditions share:

- the same frozen source forward;
- the same image-FPS centres and selected approach views;
- the same source width tensor;
- the same physical width associated with the locally selected action;
- the same collision protocol.

Conditions with the same local selector are asserted to have byte-identical
physical action columns `[:,1:]`; only the first score column may differ.
For the first live batch on each shard, `Base/Base` and `Full/Full` are also
checked against the repository's native `pred_decode_center_view_angle`
implementation.

## Why this is diagnostic, not an additive causal decomposition

Official GraspNet AP includes ranking, Top-K truncation, and optionally
collision filtering. These operations are nonlinear. Therefore quantities such
as

```text
(Full,Full) - (Full,Base) - (Base,Full) + (Base,Base)
```

are reported only as a descriptive 2×2 interaction. They should not be
presented as exact additive causal contributions.

## Run

The defaults assume the P0-1 Full control produced by
`scripts/run_mgf_p0_1_online.sh`.

```bash
cd /home/robotarm/EconomicGrasp

GPUS=0,1,2 \
bash scripts/run_mgf_p0_local_global_2x2.sh
```

If the P0-1 root differs:

```bash
P0_ROOT=/path/to/mgf_p0_1_frozen \
GPUS=0,1,2 \
bash scripts/run_mgf_p0_local_global_2x2.sh
```

Primary reporting is **collision-on**, matching the current EconomicGrasp
benchmark protocol. Collision-off is generated from the same forward and kept
as a secondary scoring/mechanism diagnostic.

Outputs:

```text
<WORK_ROOT>/
  lbase_gbase/test_collision_{on,off}/
  lfull_gbase/test_collision_{on,off}/
  lbase_gfull/test_collision_{on,off}/
  lfull_gfull/test_collision_{on,off}/
  comparison/
    comparison.md
    comparison.json
```

The comparator also reads every `accuracy.npy`, checks it against
`summary.json`, and reports scene-paired bootstrap intervals for the main
contrasts.

## Smoke

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/mgf_local_global_2x2_smoke \
P0_ROOT=/path/to/mgf_p0_1_frozen \
PHASES=infer \
SPLITS=test_seen \
INFER_MAX_FRAMES=6 \
GPUS=0,1,2 \
bash scripts/run_mgf_p0_local_global_2x2.sh
```

A smoke run intentionally skips official evaluation/comparison.
