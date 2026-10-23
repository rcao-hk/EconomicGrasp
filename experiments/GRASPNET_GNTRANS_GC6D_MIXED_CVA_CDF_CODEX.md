# Codex work order — GraspNet 10% + GN-Trans 10% + filtered GC6D 100% (RGB-only CVA-CDF)

**Repository:** `/home/robotarm/EconomicGrasp` (server) / `rcao-hk/EconomicGrasp`.
**Base:** `main`. **Intent:** training-data intervention only; preserve existing `economicgrasp_dpt`, CVA Transformer (`A1`), CDF score + depth-wise width, and depth/grasp gradient detachment protocol. **Do not switch to `use_obs_depth` or add KD/repair/collision heads.**

## Requested contract

| Source | Scenes | Frames per camera before sampling | Selected train frames |
|---|---:|---:|---:|
| GraspNet-1Billion, RealSense | 100 | 25,600 | **2,600** |
| GN-Trans RGB, same GraspNet scenes | 100 matched | 25,600 | **2,600** |
| GraspClutter6D, audited no-GN-similar/novel-overlap subset | **232** | **3,016** | **3,016** |
| **Total** | distinct scene families: 100 GN + 232 GC6D | | **8,216** |

The existing `--mix_train_fraction=0.1` implements a **per-scene stride of 10** over 256 views: `0000,0010,...,0250` (=26 frames per scene, **10.15625%**), not a mathematically exact 2560 frames. Keep this rule for comparability with the pre-existing two-source trainer; assert GraspNet and GN-Trans scene/frame keys are identical. GC6D uses 13 D435 frames per scene, with no 10% subsampling.

Train uses `train_cva_gntrans_gc6d_mix_ddp.py`; validation within the training loop, if enabled, stays paired GraspNet/GN-Trans `test_seen` subset. **External** inference/evaluation must be performed separately for `test_seen`, `test_similar`, `test_novel` in **both** GraspNet and GN-Trans, each sampled at the same stride 10. No GC6D test evaluation is requested in this run.

Expected external dump counts per domain: 30 scenes × 26 frames = **780 predictions per split**, **2,340 per domain**, **4,680** total across both domains and three splits (before any handling of skipped/error frames). Report missing frames explicitly instead of silently treating absent predictions as failures or successes.

## Implemented entrypoints / changes

- `train_cva_gntrans_gc6d_mix_ddp.py`: extends `GNTransMixedTrainer`, leaves inherited GraspNet and GN-Trans subset creation untouched, adds `CVAExtendedLabelAdapter(GraspClutter6DMultiDataset(...))` as third training source. DDP sampler is reconstructed after `ConcatDataset`.
- `dataset/gc6d_dataset.py`: adds `load_grasp_payload=False` support; object poses follow `obj_ids` from **the same GC6D CDF scene cache** used by the label adapter. Adds GraspNet-compatible camera orientation fields for `global_film` / `ray_gravity_film`, removes GC6D-only `img_id` from labeled samples so shared batches have consistent keys.
- `dataset/generate_economic_gc6d.py`: can generate `_cdf_depth` cache from GC6D *raw object friction + scene collision*, not from collapsed best-depth labels. Its bins match `dataset/generate_economic_cdf_depth.py`: 0 means fail at all thresholds; bins 1–6 mean first passing threshold; width is uint16 millimeters, with explicit validity.
- `check_gc6d_cdf_mix_data.py`: verifies filtered scene list, 232×13 frames, file existence, cache schema, and optionally a complete CVA-CDF sample+collate.
- `run_gntrans_gc6d_mix_cva_cdf_train.sh`: preflight and DDP training.
- `run_gntrans_gc6d_mix_cva_cdf_eval.sh`: delegates to the existing two-domain `run_gntrans_mix_cva_cdf_eval.sh` without modifying its behavior. Defaults to `COLLISION_THRESH=0`, i.e. **strict RGB-only prediction**. Legacy depth-based post-filtering may be selected explicitly, but report it as separate protocol.

## Before training — data audit (mandatory)

1. Ensure the filtered scene file is exactly:
   `$GC6D_ROOT/split_info/no_gn_test_overlap/grasp_train_scene_ids_no_graspnet_test_overlap.json`.
   This file must contain 232 **unique GC6D training** scene IDs, from the completed GraspNet test-object overlap analysis. Never use `split_info/grasp_train_scene_ids.json` directly in this experiment.
2. Confirm per-scene `scene_camera.json`, RGB/depth/label PNG, `virtual_scenes` rendered depth, and `graspness/<scene>/<camera>/<imgid>.npy` for all 3016 frames. Ensure the graspness array is aligned to `workspace_depth` masked points.
3. Confirm the GC6D CDF folder exists. The old `economic_grasp_label_300views_extend_angle` is *not* equivalent: it retains only best depth per (point, view, angle), and **cannot reconstruct** per-depth friction success bins.
4. Verify the BOP world-frame +Z/up convention before enabling global pose FiLM. GC6D `cam_R_w2c` maps world→camera; the adapter computes `camera_pose_vec=-R_c2w[:,2]`, `camera_gravity_vec=R_w2c*[0,0,1]`. Inspect camera vectors and camera-tilt histograms from representative GC6D scenes against expected physical D435 orientations. A syntactically valid unit vector is not evidence that world-up is correct.
5. Validate rendered depth units in meters and depth-mask/crop/K synchronization. Object-level grasp labels must have the same object order as CDF cache `obj_ids` and frame `scene_gt.json`. Check for missing or repeated object IDs.

### Generate GC6D CDF labels (only if absent)

```bash
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export GNTRANS_RGB_ROOT=/data/robotarm/dataset/GN-Trans
export GC6D_ROOT=/data2/robotarm/dataset/GraspClutter6D
export GC6D_SCENE_IDS_FILE="$GC6D_ROOT/split_info/no_gn_test_overlap/grasp_train_scene_ids_no_graspnet_test_overlap.json"

python dataset/generate_economic_gc6d.py \
  --dataset_root "$GC6D_ROOT" \
  --scene_ids_file "$GC6D_SCENE_IDS_FILE" \
  --camera realsense-d435 --keeping_views_numbers 300 \
  --extend_angle --cache_cdf_depth_labels \
  --cdf_thresholds 0.2,0.4,0.6,0.8,1.0,1.2 \
  --max_grasp_width 0.1 \
  --obj_order_source auto --collision_align auto \
  --device cpu --proc 1 --skip_done --write_stats
```

Review generator statistics and cache object IDs. This generator catches per-scene errors; a nonzero count of failed scenes must abort a CDF run.

### Preflight without model training

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GC6D_ROOT="$GC6D_ROOT" GC6D_SCENE_IDS_FILE="$GC6D_SCENE_IDS_FILE" \
CHECK_DATA=1 CHECK_ALL_CACHES=1 DRY_RUN=1 \
bash run_gntrans_gc6d_mix_cva_cdf_train.sh
```

This must pass before spending GPU training time. Also **run two-source `check_gntrans_mix_data.py`** and compare the first 2,600 matched scene/frame indices with its prior baseline.

## Training (unchanged RGB-only architecture)

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
GC6D_ROOT="$GC6D_ROOT" GC6D_SCENE_IDS_FILE="$GC6D_SCENE_IDS_FILE" \
GPUS=0,1,2 OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/log/cva_cdf_triple_10pct \
TRAIN_FRACTION=0.1 EVAL_FRACTION=0.1 \
BATCH_SIZE=1 MAX_EPOCH=20 LR=0.0001 POSE_DEPTH_MODE=global_film \
USE_FUSE_DEPTH=1 CHECK_DATA=1 TRAIN_VALIDATE=0 \
bash run_gntrans_gc6d_mix_cva_cdf_train.sh
```

Use a fresh model run (no `--checkpoint_path`), retaining the architecture's pretrained DINO/DPT initialization. GraspNet and GN-Trans use their existing fused-background supervision when `USE_FUSE_DEPTH=1`; GC6D uses its existing virtual depth, not a nonexistent `tsdf_depth` substitute.

If testing camera pose interpretation, do **not** silently change only GC6D to `pose_depth_mode=none`: that changes model inputs across sources. First verify physical metadata, and use a whole-experiment `POSE_DEPTH_MODE=none` control if necessary.

Record `gntrans_gc6d_mix_protocol.json`, GC6D filtered-list SHA256, training log, checkpoint SHA, dataloader batch counts, per-domain sample counts, and any skipped/cache-corrupt scenes. Expected source weights by natural frame count: GraspNet ≈31.65%, GN-Trans ≈31.65%, GC6D ≈36.71% (DDP pads when needed).

## Inference and official evaluation (10% × both domains)

```bash
DATASET_ROOT="$DATASET_ROOT" GNTRANS_RGB_ROOT="$GNTRANS_RGB_ROOT" \
CKPT=/data2/robotarm/result/grasp/rgbgrasp/log/cva_cdf_triple_10pct/checkpoint_19.tar \
GPUS=0,1,2,3,4,5 \
OUTPUT_ROOT=/data2/robotarm/result/grasp/rgbgrasp/cva_cdf_triple_10pct_eval/e19 \
SAMPLE_INTERVAL=0.1 DATASETS=both \
SPLITS=test_seen,test_similar,test_novel \
COLLISION_THRESH=0 GNTRANS_COLLISION_SOURCE=none \
ENABLE_INFERENCE=1 ENABLE_EVAL=1 \
bash run_gntrans_gc6d_mix_cva_cdf_eval.sh
```

Note that `checkpoint_19.tar` is available only if the corresponding epoch checkpoint is actually saved (`CKPT_SAVE_INTERVAL` governs interval saves; default 5 saves epoch indices 0,5,10,15, while `checkpoint.tar` is updated every epoch). For default 20 epochs use `checkpoint.tar` or an existing `checkpoint_15.tar`; **never invent a checkpoint path**.

Evaluation uses the *same original GraspNet 3D geometry* for both RGB domains and the existing official GraspNet evaluator, with `eval.py --sample_interval 10`. Keep clean/trans domains and seen/similar/novel splits separate. Running `ENABLE_INFERENCE=0 ENABLE_EVAL=1` must reuse complete prediction dumps without deleting them.

**Strict RGB-only distinction:** `COLLISION_THRESH=0` means no depth-derived model-free post-filter. To reproduce historical sensor-cloud filtered AP instead use `COLLISION_THRESH=0.01 GNTRANS_COLLISION_SOURCE=original_sensor`; these numbers must be presented under a separately labeled protocol.

## Codex acceptance checklist

- [ ] `python -m compileall -q train_cva_gntrans_gc6d_mix_ddp.py check_gc6d_cdf_mix_data.py dataset/gc6d_dataset.py dataset/generate_economic_gc6d.py`; `bash -n run_gntrans_gc6d_mix_cva_cdf_train.sh run_gntrans_gc6d_mix_cva_cdf_eval.sh`.
- [ ] Unit test CDF encoding exactly at friction thresholds; collision and out-of-width candidates have bin 0.
- [ ] Verify split JSON has 232 unique training scenes and indexes exactly 3016 frames.
- [ ] Verify all CDF cache files, no `obj_ids` vs scene GT pose mismatch, no label/point ordering corruption.
- [ ] Validate dataset contract with mixed 3-source batch: same keys/shapes, `object_poses_list` length matches CDF payload lists; CPU-resident lists preserved under DDP.
- [ ] One-batch forward + loss + backward + optimizer step with at least one GC6D sample on a single GPU, and then a 2-GPU DDP smoke with mixed-source batch. No unexpected CDF key, shape, NaN, or NCCL hanging.
- [ ] `global_film` camera metadata physically correct for GC6D; if not, run a controlled whole-experiment `none` pose mode.
- [ ] Training protocol JSON shows 2600/2600/3016/8216; training epoch finishes and checkpoint saves.
- [ ] For inference/eval, verify output count 780 per split/domain, log per-split AP/AP0.8/AP0.4 and mean AP, report errors and selection protocol.
- [ ] Baseline: compare with existing GraspNet + GN-Trans 10% model under matched training update count, same RGB-only collision protocol. Do not infer data benefit solely from extra steps.
- [ ] No leakage from GC6D into the defined GraspNet similar/novel evaluation objects; maintain filter list/checksum in experiment report.
- [ ] Keep existing `run_gntrans_mix_cva_cdf_train.sh`, `run_gntrans_mix_cva_cdf_eval.sh`, and old GC6D generator behavior backwards compatible.

## Required Codex report

Create `experiments/results/GRASPNET_GNTRANS_GC6D_MIX_10PCT_REPORT.md` after executing on the server. Include source commits, scene-list SHA, label-cache schema validation, all commands and logs, hardware and training time, selected checkpoint, AP table by dataset×split, comparison against two-source baseline, missing dump counts, reproducibility caveats, and failure diagnostics. **Do not claim actual training or AP results before the server jobs complete.**
