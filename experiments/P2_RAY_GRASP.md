# P2: Discrete ray-conditioned grasp representation

Base: `main` at `3f3c08dcddf14f08f060c91b2b20a76ab6afc0b2`.
Branch: `exp/p2-ray-conditioned-grasp`.

## Scope

This is a first finite ray-field implementation, not a continuous implicit field
or a calibrated posterior over monocular depth. The frozen Stage-1 RGB frontend
selects M image-space FPS rays and predicts dense metric depth. For each ray the
model evaluates fixed camera-z offsets `[-40,-20,-10,0,10,20,40]` millimetres.
K=7 ray-depth hypotheses and D=4 gripper insertion-depth anchors are separate axes.
The image frontend runs once. The frozen CVA grouping is recomputed sequentially
at each center, followed by a shared trainable angle/CDF/width decoder.

The five-component conditioning descriptor is `(ray_x, ray_y, base_z_norm,
candidate_z_norm, offset_norm)`. A zero-initialized final projection adds this
descriptor to the grouped features. A small support head predicts proximity to
the existing grasp-label domain. The first experiment freezes the depth network,
RGB features, spatial enhancer, ViewNet, and CVA grouping, including their dropout
and normalization state. Only the shared CVA decoder, ray embedding, and support
head are trained. This keeps the center proposal grid fixed during training.

## Supervision and limitations

The existing `process_grasp_labels_cdf_width` is called separately at every
candidate's actual physical center AFTER the neural predictions. Cached CDF
labels are used only where the original nearest-point (<5 mm) and view-matching
masks hold. Unknown/out-of-support CDF labels are NOT replaced by negative
force-closure labels. Width retains the original cache mask and factor-10 scale.

The separate support BCE target is whether the candidate is within 5 mm of its
nearest available cached grasp point. It is independent of missing view labels.
This target measures label-domain support, not occupancy, collision freedom,
force closure, or complete physical grasp validity. At test time the optional
product `sigmoid(support) * mean(sigmoid(CDF))` is a ranking heuristic; no
calibration claim is made. `raw` scoring is included to isolate this factor.
There is no depth residual regression objective and no teacher/KD objective.
No online GraspNet/Dex-Net evaluator runs during training.

Offsets outside the configured metric range are excluded from loss and decoding.
Hypotheses outside the grid or unsupported by the cache remain limitations.
Top-1 approach view is fixed by the existing frozen view predictor. There is no
joint view search or cross-ray attention in v1. Default decoding maximizes over
ray depth, in-plane angle, and insertion depth, but emits ONE grasp per native
image ray, not M*K candidates. The zero-offset slice retains native geometry.

## Files

- `models/ray_grasp_ops.py`: camera-ray geometry, utility, selection, losses.
- `models/economicgrasp_ray.py`: RayConditionedGrasp, shared conditioned decoder.
- `utils/ray_grasp_runtime.py`: isolated CLI, datasets, DDP, checkpoint contracts.
- `train_ray_grasp.py`: frozen-frontend decoder-only training and validation.
- `inference_ray_grasp.py`: RGB-network inference and optional offline AP.

## Quick start

Use the ORIGINAL controlled Stage-1 checkpoint, not a P1 recenter checkpoint.
The checkpoint's pose-depth mode and fuse-depth construction are inherited when
not explicitly supplied. Conflicting explicit settings raise an error.
`graspness_mode` defaults to `scene`. The existing CDF label cache is required.

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export INIT_CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar

# Single-GPU training; batch size is per GPU. Use a fresh output directory.
CUDA_VISIBLE_DEVICES=0 python train_ray_grasp.py \
  --dataset_root "$DATASET_ROOT" --checkpoint_path "$INIT_CKPT" \
  --log_dir log/p2_ray_10pct --batch_size 1 --learning_rate 0.0001 \
  --max_epoch 20 --ray_train_sample_interval 0.1 \
  --ray_eval_sample_interval 0.1 --graspness_mode scene \
  --ray_offsets_mm=-40,-20,-10,0,10,20,40

# Multi-GPU alternative: same flags after the script name.
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  train_ray_grasp.py --dataset_root "$DATASET_ROOT" \
  --checkpoint_path "$INIT_CKPT" --log_dir log/p2_ray_ddp_10pct \
  --batch_size 1 --learning_rate 0.0001 --max_epoch 20 \
  --ray_train_sample_interval 0.1 --ray_eval_sample_interval 0.1
```

Training defaults to 10% deterministic frames per scene, all training scenes,
and 10% `test_seen` validation. Validation is not AP and does not call an analytic
evaluator. The `test_seen` use follows the existing diagnostic protocol; it
should not be represented as an untouched benchmark test set if used repeatedly
for checkpoint selection. DDP validation uses unique shards without padding.
CDF/width/support losses are normalized by global valid-element counts.

A smoke run can add `--ray_max_batches 2 --max_epoch 1`. Use a separate log_dir;
smoke checkpoints are marked in metadata. Decoder activation checkpointing is
on by default. K sequential grouping calls still increase compute and retain
some grouped features; batch size 1 is the recommended first runtime check.

## Testing without training and after training

```bash
# Original Stage-1 outputs: zero slice, raw score, native M-candidate budget.
CUDA_VISIBLE_DEVICES=0 python inference_ray_grasp.py \
  --dataset_root "$DATASET_ROOT" --checkpoint_path "$INIT_CKPT" \
  --test_mode test_seen --save_dir result/ray_stage1_zero/test_seen \
  --batch_size 1 --sample_interval 0.1 --ray_selection zero \
  --ray_score_mode raw --ray_run_eval

# No-training multi-depth search. The random support head is NOT used.
CUDA_VISIBLE_DEVICES=0 python inference_ray_grasp.py \
  --dataset_root "$DATASET_ROOT" --checkpoint_path "$INIT_CKPT" \
  --test_mode test_seen --save_dir result/ray_stage1_raw/test_seen \
  --batch_size 1 --sample_interval 0.1 --ray_selection best \
  --ray_score_mode raw --ray_run_eval

# Trained ray field; auto scoring uses support only when it was trained.
CUDA_VISIBLE_DEVICES=0 python inference_ray_grasp.py \
  --dataset_root "$DATASET_ROOT" \
  --checkpoint_path log/p2_ray_10pct/checkpoint_best.tar \
  --test_mode test_seen --save_dir result/ray_trained/test_seen \
  --batch_size 1 --sample_interval 0.1 --ray_run_eval
```

Repeat for `test_similar` and `test_novel` with distinct output directories.
Inference also supports torchrun with unique frame shards. Offline evaluation
runs only on rank 0 after distributed inference has finished. `--ray_eval_only`
reuses existing predictions and requires no checkpoint/model/CUDA construction.
Evaluation reads the saved sampling protocol: inference fraction 0.1 corresponds
to evaluator frame stride 10. Sampled evaluation requires the same GraspNetAPI
fork supporting `anno_sample_ratio` as the repository's existing `eval.py`.
Partial smoke inference cannot be evaluated as a complete AP result.

Default `collision_thresh=0` is point-cloud-free network plus decode. To match
historical experiments, explicitly use `--collision_thresh 0.01` for EVERY
comparison. That optional post-processing reads the captured point cloud and
must be reported as such; it is not part of an end-to-end RGB-only system.
`--save_nocollision` additionally saves predictions before that filter.

## Output and decisions

Training writes `ray_training_protocol.json`, `ray_epochs.jsonl`,
`checkpoint_latest.tar`, `checkpoint_best.tar`, and periodic epoch checkpoints.
The best checkpoint is chosen by the weighted CDF/width/support validation loss,
not by center MAE or AP. Resumption requires `--resume`, a P2 checkpoint, and the
same sampling/grid/loss/epoch schedule; it is not a silent continuation with a
changed experiment protocol.

Inference writes GraspNet-compatible `scene_XXXX/<camera>/YYYY.npy` files plus
`ray_inference_summary.json`. Optional evaluation adds `evaluation.log`, the raw
`ap_<split>_<camera>.npy` tensor and `ray_eval_summary.json`.

Compare the untrained zero slice, untrained raw ray search, and trained ray field
on identical frame subsets and collision settings. Monitor base/any/selected
label-point support, CDF support, and selected-offset histograms. Increased
coverage is not automatically successful ranking or increased grasp AP.

CPU tensor and stub-integration checks do not replace a real CUDA/GraspNet run.
No AP or real-data end-to-end performance is claimed by this implementation.
