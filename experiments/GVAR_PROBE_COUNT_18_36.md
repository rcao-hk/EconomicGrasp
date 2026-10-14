# GVAR configurable probe count (18 / 36)

**Branch:** `exp/gripper-volume-action-reader`. The flag `--gvar_num_probes`
selects **18** or **36** physical support probes per insertion-conditioned
action query. Default is **36**, preserving the existing published experimental
contract, default checkpoint serialization, original 36-point geometry order
and six-role attention-mass accumulation. This commit only changes the
sampling budget, not the network's parameter shapes, CDF targets, width
heads, numeric-depth detach, or physical grasp decoding.

## What is sampled?

Each point is a projected evidence-query location in the gripper frame
(`x` approach, `y` opening, `z` gripper height). These points are
**not** true contact points, collision labels or known-free-space evidence.
The reader projects them into RGB features and predicted-depth geometry.

The default 36 points are the historical template. The 18-point mode uses
the following **exact subset** (zero-based 36-point indices):

```python
PROBE_INDICES_18 = (
    0, 1, 4, 5,        # left contact: 4
    6, 7, 10, 11,      # right contact: 4
    12, 13, 16, 17,    # closing: 4
    20, 21,            # finger body: 2
    26, 27,            # palm: 2
    32, 33,            # approach: 2
)
```

The order, original coordinates, insertion offset, side symmetries and role
IDs are maintained. For 18 points, the reader computes each role's
attention mass using its actual IDs (`scatter_add`); the historical
36-point code retains the exact old `reshape(-1,6,6).sum(-1)` path.

## Training

Choose the count by the training CLI option:

```bash
python train_gvar.py --gvar_variant volume_rel --gvar_num_probes 18 \
  ... # retain all normal GraspNet/GT/CDF/DDP arguments
```

Or using the existing launcher:

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans

NUM_PROBES=18 VARIANT=volume_rel GPUS=0,1,2 BATCH_SIZE=3 \
    OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/gvar_volume_rel_p18_new \
    bash scripts/run_gvar_train.sh
```

`NUM_PROBES=36`, or omitting the option, preserves the historical default.
When no explicit `OUTPUT_ROOT` is specified, the launcher adds a `_p18`
suffix **only for the 18-probe run**, avoiding the old default 36-probe
training directory.

For a strict apples-to-apples experiment, train **separate checkpoints**
from fresh initialization under the same 10% Original + 10% GN-Trans schedule,
seed, epochs and checkpoint-selection rule. Do not treat switching the
probe count on a 36-point checkpoint as a trained 18-point variant.

## Inference and checkpoint compatibility

Inference **infers the count from the trained checkpoint**. The explicit
option checks the count, never silently changes the trained reader:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/path/to/fresh_18_probe_checkpoint_epoch_019.tar \
NUM_PROBES=18 OUTPUT_ROOT=/path/to/independent_18_probe_predictions \
bash scripts/run_gvar_eval.sh
```

If `NUM_PROBES` is not set for evaluation, the checkpoint supplies it.
`inference_gvar.py --gvar_num_probes 18` works identically as an assertion.
Mismatching a 36-point checkpoint with an 18-point inference flag fails
before model inference. Legacy 36-point checkpoints (without the new
`num_probes` config field) continue to load as 36. Their original
serialized `GVARConfig.to_dict()` representation is deliberately unchanged.

P0-A's frozen-model R0–R4 code supports both new 18-point checkpoints
and historical 36-point checkpoints. Existing archives remain untouched.

## Checks and interpretation

```bash
python -m pytest -q tests/test_gvar.py tests/test_gvar_p0abc.py
bash -n scripts/run_gvar_train.sh scripts/run_gvar_eval.sh
```

Tests check probe templates (18 is a true 36-point subset), correct role
histograms (4/4/4/2/2/2), depth insertion translation, output shapes,
visual-feature gradients with detached numeric geometry, all-invalid
behavior, checkpoint validation, training/evaluation flag forwarding and
R4-to-Hybrid consistency for both counts.

**Execution status:** Code and tests have been committed, but no CUDA/GraspNet
server training, 18-point formal AP or full-checkpoint parity was executed
in this chat. Compare AP (including Novel μ=0.4/0.8) alongside training
memory, inference latency and object/scene generalization. 18 points are
an efficiency hypothesis, not a proven optimal design.
