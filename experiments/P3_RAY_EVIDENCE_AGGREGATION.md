# P3: Selection-free multi-depth ray evidence aggregation

Base branch: `main` at `3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2`.

Experiment branch: `exp/p3-ray-evidence-aggregation`.

## Motivation

The preceding diagnostics separate three facts:

1. same-query GT-center intervention gives large headroom (~+12.21 mean AP on the controlled 10% protocol);
2. exposing fixed ray-depth hypotheses raises held-out point-support coverage from ~59.8% at the zero center to ~92.7% for any ray hypothesis;
3. hard cross-depth selection does not generalize: P2-v2 relational+raw is essentially tied with the zero-center baseline (~+0.071 mean AP).

P3 therefore removes the *pre-grasp hard depth selector*.  The representation keeps all K physical-center hypotheses alive while local grasp evidence is formed and exchanged.  A metric center is selected only at final task decode because a 6-DoF grasp must ultimately contain a translation.

## Controlled scope

Fresh P3 training starts directly from the controlled Stage-1 RGB checkpoint on `main`, not from P1/P2.  This avoids importing the learned biases of the negative selector experiments.

Frozen throughout P3 training:

- DINO/DPT RGB and metric-depth path;
- image-FPS seed selection;
- approach-view prediction and Top-1 view selection;
- metric local grouping weights;
- the existing CVA-CDF/width decoder.

Trainable:

- residual cross-depth evidence aggregator;
- contextual per-depth viability head inside that aggregator.

Thus an improvement cannot be attributed to another round of Stage-1 grasp-head fine-tuning.

## Representation

For each image-FPS ray `q`, use the fixed camera-z hypotheses

`[-40, -20, -10, 0, +10, +20, +40] mm`

around the Stage-1 predicted metric center.  The Stage-1 approach view is predicted once and held fixed across K so P3 isolates center-depth representation.

The frozen metric grouping produces angle-conditioned local features

`F[q,k,a]`.

For every `(q,a)`, a Transformer attends only across the K depth hypotheses:

`H[q,1:K,a] = Transformer({F[q,k,a], ray_descriptor[q,k]}_k)`.

A zero-initialized residual projection gives

`F_context[q,k,a] = F[q,k,a] + DeltaF[q,k,a]`.

The complete contextualized set is then passed through the frozen CDF/width decoder in one shared batch.  The model predicts a field over

`ray depth K x in-plane angle A x gripper insertion depth D`.

There is no `k*` before this grasp head.

## Final utility

The original CDF gives

`U_raw[q,k,a,d] = mean_t sigmoid(CDF[q,t,k,a,d])`.

The cross-depth hidden state also predicts a contextual label-domain viability

`p_valid[q,k]`.

The primary P3 joint utility is

`U_joint = p_valid * U_raw`.

Unlike P2-v1, `p_valid` is not an independent per-depth MLP: its hidden state has already attended to every K hypothesis on the same ray.  Unlike P2-v2, it does not choose a depth before grasp scoring.  The final decoder performs one joint argmax over `K x A x D` and still emits exactly one grasp per native image-FPS ray.

## Training target and loss

CDF and width keep the original masked label semantics from `process_grasp_labels_cdf_width`.

Viability uses the same 5-mm nearest grasp-point support contract but uses balanced positive/negative BCE, avoiding the strong majority-prior failure observed in P2-v2.

The final `U_joint` receives an additional balanced utility-calibration objective:

- evaluator-labelled `(k,a,d)` candidates use compact CDF target utility;
- centers known to be outside the 5-mm label domain are zero targets;
- geometrically supported centers with a missing selected-view cache label stay unknown rather than being forced negative.

Default objective:

`L = 1*L_CDF + 10*L_width + 1*L_viability_balanced + 1*L_joint_balanced`.

The analytic GraspNet/Dex-Net evaluator is never called during training.

## Fresh training

Use the original controlled Stage-1 checkpoint (the canonical e15 checkpoint in the current experiment series), not P1/P2:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p3_ray_10pct \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
bash run_p3_ray_train.sh
```

Before the full run:

```bash
pytest -q tests/test_p3_argparse.py tests/test_p3_ray_ops.py
```

A real CUDA/data smoke run should use a separate directory:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/path/to/stage1_e15.tar \
GPUS=0 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p3_ray_smoke \
BATCH_SIZE=1 MAX_EPOCH=1 P3_MAX_BATCHES=2 \
bash run_p3_ray_train.sh
```

Smoke checkpoints are marked and rejected by official inference.

## Training outputs

- `log_train_p3.txt`
- `p3_epochs.jsonl`
- `p3_training_protocol.json`
- `checkpoint_latest.tar`
- `checkpoint_best_utility.tar`
- `checkpoint_best_regret.tar`
- periodic `checkpoint_epoch_*.tar`

Primary diagnostics:

- `p3_base_point_support`
- `p3_any_point_support`
- `p3_joint_selected_point_support`
- `p3_joint_selected_target_utility`
- `p3_joint_selection_regret`
- `p3_joint_selected_nonzero`
- `p3_joint_selected_k0...k6`
- corresponding `p3_raw_*` metrics
- `aggregation_delta_norm`

The utility metric uses the **actual finally selected (k,a,d)** target, not the oracle-best operation at the selected depth.

## Primary inference

First evaluate the utility the model was explicitly trained to calibrate:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/p3_ray_10pct/checkpoint_best_utility.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/p3_ray_eval/joint_same \
SAMPLE_INTERVAL=0.1 \
SELECTION_SCORE=joint \
FINAL_SCORE=same \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_p3_ray_inference.sh
```

Useful controls require no retraining:

- `SELECTION_SCORE=joint FINAL_SCORE=raw`: keep contextual depth/operation selection but use raw CDF for cross-ray ranking;
- `SELECTION_SCORE=raw FINAL_SCORE=raw`: test contextualized CDF without viability gating;
- `FORCE_ZERO=1`: force the final center to the zero-offset slice.  This is an internal P3 slice, not a substitute for the canonical Stage-1 baseline because cross-depth evidence can still alter the zero-depth representation.

Collision threshold `0.01` reads the captured point cloud in post-processing.  Set `COLLISION_THRESH=0` for a strict network-only RGB evaluation and keep that setting identical across controls.

## Interpretation gate

The P3 mechanism succeeds only if the validation/AP results show more than a conservative return to zero depth.  Specifically inspect whether:

1. `p3_joint_selected_point_support` moves materially above the ~59.8% zero-center support toward the ~92.7% candidate ceiling;
2. non-zero selections rise without the shallow-depth bias seen in P2-v1;
3. final AP exceeds the paired Stage-1 zero-center baseline, particularly Similar/Novel;
4. `joint+raw` vs `joint+same` distinguishes depth/operation selection from cross-ray ranking calibration.

If candidate support remains high but P3 still collapses to zero, increasing Transformer size is not the next step; the evidence presented by the fixed local grouping is then insufficient to identify metric depth under domain shift.