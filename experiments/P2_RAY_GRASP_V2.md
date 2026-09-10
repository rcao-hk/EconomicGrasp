# P2-v2: Ray-wise cross-depth relational selection

Base branch: `exp/p2-ray-conditioned-grasp`.

## Motivation from P2-v1

P2-v1 expanded one deterministic physical center into seven fixed camera-z
hypotheses. On held-out Seen validation, zero-offset label-point support was
~59.8% while any-depth support was ~92.7%, so candidate coverage was largely
recovered. However raw CDF cross-depth selection reduced AP, and the independent
support gate mainly recovered performance by becoming conservative. Therefore
P2-v2 changes **selection**, not the ray grid.

## Controlled scope

P2-v2 requires a completed P2-v1 checkpoint (recommended: the e19/latest model
used in the four-way AP analysis). The entire P2-v1 network is frozen and held
in eval mode:

- RGB / DINO-DPT metric geometry: frozen
- image-FPS query pixels: frozen
- K=7 physical-center grid: unchanged
- ViewNet / CVA grouping: frozen
- P2-v1 ray conditioning, CDF, width, support heads: frozen
- **new trainable module:** a small Transformer over the K hypotheses of the
  same image ray

Thus the experiment asks a single question: can relational cross-depth reasoning
convert the already high candidate coverage into better depth selection?

## Selector token

For each `(image ray q, physical depth k)`, the token concatenates:

1. frozen local CVA grouping feature averaged over in-plane angle;
2. five existing ray descriptors (ray x/y, normalized base z, candidate z,
   normalized offset);
3. raw CDF best utility;
4. raw CDF mean utility;
5. raw CDF top1-top2 operation margin;
6. frozen P2-v1 support logit.

A two-layer Transformer (default hidden 128, four heads) attends only across K
hypotheses belonging to the same ray. There is no interaction across different
image-FPS rays.

## Training target

For each depth hypothesis, build an evaluator-aligned target from the existing
compact CDF cache:

`GT_depth_utility = geometric_5mm_support * max_{angle,insertion-depth}(GT CDF utility)`.

The frozen per-depth CDF head keeps P2-v1 semantics: off-support candidates are
not trained as CDF negatives. The zero target above is used only for the
**cross-depth selector**.

Training uses:

- listwise cross-depth CE: soft target distribution over K, default target
  temperature 0.1;
- balanced absolute contextual-utility BCE: positive and zero-utility depths
  contribute equally when both occur.

Default total selector loss:

`L = 1.0 * L_listwise + 0.5 * L_calibration`.

No analytic GraspNet/Dex-Net evaluator is called during training. The best
checkpoint is selected by minimum held-out Seen `v2_selection_regret`, not the
scalar training loss.

## Fresh training

Use the completed **P2-v1 e19/latest checkpoint**, not Stage-1 and not the P2-v1
loss-best e3 checkpoint:

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
INIT_CKPT=/path/to/p2_v1/checkpoint_latest.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/p2_ray_v2_10pct \
TRAIN_SAMPLE_INTERVAL=0.1 \
EVAL_SAMPLE_INTERVAL=0.1 \
BATCH_SIZE=1 \
MAX_EPOCH=20 \
bash run_ray_grasp_v2_train.sh
```

A smoke run can set `V2_MAX_BATCHES=2 MAX_EPOCH=1` and must use a separate
output directory. Smoke checkpoints are rejected by official v2 inference.

Training outputs:

- `log_train_v2.txt`
- `ray_v2_epochs.jsonl`
- `ray_v2_training_protocol.json`
- `checkpoint_latest.tar`
- `checkpoint_best.tar`
- `checkpoint_best_selector.tar`

Key validation metrics:

- `v2_selection_regret` (primary checkpoint metric, lower is better)
- `v2_selected_target_utility`
- `v2_oracle_hit`
- `v2_selected_label_point_support`
- `v2_selected_cdf_label_support`
- `v2_selected_nonzero`
- `v2_selected_k0...k6`
- `v2_selector_entropy`
- `v2_zero_probability`

## Inference

Primary test: relational depth selection but preserve the selected candidate's
raw CDF score for ranking across different image rays. This isolates the new
within-ray selector from the ranking distortion observed with P2-v1 support
multiplication.

```bash
DATASET_ROOT=/data/robotarm/dataset/graspnet \
CKPT=/path/to/p2_ray_v2_10pct/checkpoint_best_selector.tar \
GPUS=0,1,2 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/p2_ray_v2_eval/relational_raw \
SAMPLE_INTERVAL=0.1 \
V2_SELECTION=relational \
V2_FINAL_SCORE=raw \
COLLISION_THRESH=0.01 \
RUN_EVAL=1 \
bash run_ray_grasp_v2_inference.sh
```

`V2_SELECTION` controls only which physical depth is chosen:

- `relational`: new v2 Transformer;
- `zero`: fixed native zero-offset control;
- `raw`: P2-v1 raw-CDF best-depth control;
- `supported`: P2-v1 support*CDF best-depth control.

`V2_FINAL_SCORE` controls ranking **after** depth selection:

- `raw` (recommended primary): original CDF score of the selected grasp;
- `contextual`: sigmoid of v2 contextual utility logit;
- `product`: raw CDF multiplied by contextual utility.

The first result to inspect is `relational + raw`. Only if depth selection
improves but Top-K ranking remains poor should `contextual` or `product` become a
main design question.

## Interpretation gate

- If relational+raw > zero and > P2-v1 supported, the candidate-expansion
  hypothesis and relational selection are both supported.
- If selector regret/support improves strongly but AP does not, remaining error
  is cross-ray score/ranking or downstream grasp quality rather than depth
  selection itself.
- If selector regret does not improve, the frozen per-depth features do not
  contain enough information for reliable relative depth discrimination; the
  next change should enrich ray-depth features rather than expand K.
