#!/usr/bin/env bash
# Mixed-setting-matched Grasp-Field control:
#   20% GraspNet train exposure (5200 frames/epoch)
#   10% Seen validation (780 frames)
#   3 GPUs x batch/GPU 3 = effective batch 9
#   AdamW, lr=3e-4 for both task/geometry, wd=1e-3
#   cosine epoch LR, grad clip=1.0
#   fused depth supervision ON
#   ranking loss ON (0.1)
#   fresh initialization, seed=0, 20 epochs
#
# This matches the current mixed CVA training regime as closely as possible
# while keeping the Metric Grasp Field-specific architecture/losses unchanged.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched}

GPUS=${GPUS:-0,1,2}
INFER_GPUS=${INFER_GPUS:-$GPUS}
PHASES=${PHASES:-train}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-latest}
RESUME=${RESUME:-0}

EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-3}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.2}
EVAL_FRACTION=${EVAL_FRACTION:-0.1}
LR=${LR:-0.0003}
GEOMETRY_LR=${GEOMETRY_LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
GRAD_CLIP=${GRAD_CLIP:-1.0}
USE_FUSE_DEPTH=${USE_FUSE_DEPTH:-1}
SEED=${SEED:-0}

WORKERS=${WORKERS:-4}
EVAL_WORKERS=${EVAL_WORKERS:-2}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-10}

RANKING_WEIGHT=${RANKING_WEIGHT:-0.1}
RANKING_TEMPERATURE=${RANKING_TEMPERATURE:-0.1}
BASE_CDF_WEIGHT=${BASE_CDF_WEIGHT:-0.25}
PROFILE_WEIGHT=${PROFILE_WEIGHT:-1}
PROFILE_MEAN_WEIGHT=${PROFILE_MEAN_WEIGHT:-10}

ENCODER=${ENCODER:-vitb}
POSE_MODE=${POSE_MODE:-global_film}
SEED_MODE=${SEED_MODE:-image_fps}
M_POINT=${M_POINT:-1024}
GROUP_CHUNK=${GROUP_CHUNK:-256}
ACTION_CHUNK=${ACTION_CHUNK:-256}
FIELD_BINS=${FIELD_BINS:-160}
FIELD_HIDDEN=${FIELD_HIDDEN:-64}
FIELD_STRIDE=${FIELD_STRIDE:-4}
EVIDENCE_MODE=${EVIDENCE_MODE:-learned}
SURFACE_EPSILON=${SURFACE_EPSILON:-0.005}
PRIOR_SIGMA=${PRIOR_SIGMA:-0.03}
FIXED_SIGMA=${FIXED_SIGMA:-0.02}

AMP=${AMP:-0}
TOP4=${TOP4:-0}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
SCORE_SOURCE=${SCORE_SOURCE:-field}

# If infer/eval are requested, default to the same historical system-level
# collision protocol used by the mixed/G20 controls.
COLLISION_THRESH=${COLLISION_THRESH:-0.01}
COLLISION_VOXEL_SIZE=${COLLISION_VOXEL_SIZE:-0.01}
COLLISION_APPROACH_DIST=${COLLISION_APPROACH_DIST:-0.05}

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
  echo "[ERROR] This matched control expects exactly 3 GPUs; got GPUS=$GPUS" >&2
  exit 2
fi
if [[ "$BATCH_SIZE" != "3" ]]; then
  echo "[WARN] BATCH_SIZE=$BATCH_SIZE differs from mixed setting 3." >&2
fi

echo "============================================================"
echo "[MGF MIXED-MATCHED20]"
echo "  train data          : 20% GraspNet"
echo "  validation          : 10% GraspNet Seen"
echo "  expected train/val  : 5200 / 780 frames"
echo "  GPUs                : $GPUS"
echo "  batch/GPU           : $BATCH_SIZE"
echo "  effective batch     : $(( ${#GPU_IDS[@]} * BATCH_SIZE ))"
echo "  epochs              : $EPOCHS"
echo "  task LR             : $LR"
echo "  geometry LR         : $GEOMETRY_LR"
echo "  weight decay        : $WEIGHT_DECAY"
echo "  LR schedule         : $LR_SCHEDULE"
echo "  grad clip           : $GRAD_CLIP"
echo "  fused depth target  : $USE_FUSE_DEPTH"
echo "  ranking weight      : $RANKING_WEIGHT"
echo "  seed                : $SEED"
echo "  init checkpoint     : EMPTY"
echo "  phases              : $PHASES"
echo "  output              : $WORK_ROOT"
echo "============================================================"

# Explicitly empty INIT_CHECKPOINT prevents the shared launcher from falling
# back to its historical Stage-1 initialization.
env \
  DATASET_ROOT="$DATASET_ROOT" \
  WORK_ROOT="$WORK_ROOT" \
  INIT_CHECKPOINT="" \
  GPUS="$GPUS" \
  INFER_GPUS="$INFER_GPUS" \
  PHASES="$PHASES" \
  SPLITS="$SPLITS" \
  CHECKPOINT_KIND="$CHECKPOINT_KIND" \
  RESUME="$RESUME" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  SAMPLE_FRACTION="$TRAIN_FRACTION" \
  TRAIN_FRACTION="$TRAIN_FRACTION" \
  EVAL_FRACTION="$EVAL_FRACTION" \
  LR="$LR" \
  GEOMETRY_LR="$GEOMETRY_LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  LR_SCHEDULE="$LR_SCHEDULE" \
  GRAD_CLIP="$GRAD_CLIP" \
  USE_FUSE_DEPTH="$USE_FUSE_DEPTH" \
  SEED="$SEED" \
  WORKERS="$WORKERS" \
  EVAL_WORKERS="$EVAL_WORKERS" \
  OFFICIAL_WORKERS="$OFFICIAL_WORKERS" \
  RANKING_WEIGHT="$RANKING_WEIGHT" \
  RANKING_TEMPERATURE="$RANKING_TEMPERATURE" \
  BASE_CDF_WEIGHT="$BASE_CDF_WEIGHT" \
  PROFILE_WEIGHT="$PROFILE_WEIGHT" \
  PROFILE_MEAN_WEIGHT="$PROFILE_MEAN_WEIGHT" \
  ENCODER="$ENCODER" \
  POSE_MODE="$POSE_MODE" \
  SEED_MODE="$SEED_MODE" \
  M_POINT="$M_POINT" \
  GROUP_CHUNK="$GROUP_CHUNK" \
  ACTION_CHUNK="$ACTION_CHUNK" \
  FIELD_BINS="$FIELD_BINS" \
  FIELD_HIDDEN="$FIELD_HIDDEN" \
  FIELD_STRIDE="$FIELD_STRIDE" \
  EVIDENCE_MODE="$EVIDENCE_MODE" \
  SURFACE_EPSILON="$SURFACE_EPSILON" \
  PRIOR_SIGMA="$PRIOR_SIGMA" \
  FIXED_SIGMA="$FIXED_SIGMA" \
  AMP="$AMP" \
  TOP4="$TOP4" \
  INFER_BATCH_SIZE="$INFER_BATCH_SIZE" \
  SCORE_SOURCE="$SCORE_SOURCE" \
  COLLISION_THRESH="$COLLISION_THRESH" \
  COLLISION_VOXEL_SIZE="$COLLISION_VOXEL_SIZE" \
  COLLISION_APPROACH_DIST="$COLLISION_APPROACH_DIST" \
  MAX_TRAIN_FRAMES=0 \
  MAX_VAL_FRAMES=0 \
  MAX_STEPS=0 \
  INFER_MAX_FRAMES=0 \
  bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh"

# Validate the protocol whenever a training protocol is available.
PROTOCOL="$WORK_ROOT/train/protocol.json"
if [[ -f "$PROTOCOL" ]]; then
  python - "$PROTOCOL" <<'PY'
import json
import math
import sys

p = json.load(open(sys.argv[1], "r", encoding="utf-8"))
errors = []

expected = {
    "train_fraction": 0.2,
    "eval_fraction": 0.1,
    "train_frames": 5200,
    "val_frames": 780,
    "use_fuse_depth": True,
    "seed": 0,
}
for key, value in expected.items():
    if p.get(key) != value:
        errors.append(f"{key}: expected {value!r}, got {p.get(key)!r}")

opt = p.get("optimizer", {})
opt_expected = {
    "task_lr": 3e-4,
    "geometry_lr": 3e-4,
    "weight_decay": 1e-3,
    "batch_per_gpu": 3,
    "world_size": 3,
    "effective_batch": 9,
    "lr_schedule": "cosine",
    "grad_clip": 1.0,
}
for key, value in opt_expected.items():
    got = opt.get(key)
    if isinstance(value, float):
        if got is None or not math.isclose(float(got), value, rel_tol=0, abs_tol=1e-12):
            errors.append(f"optimizer.{key}: expected {value!r}, got {got!r}")
    elif got != value:
        errors.append(f"optimizer.{key}: expected {value!r}, got {got!r}")

loss = p.get("loss_weights", {})
if not math.isclose(float(loss.get("ranking_weight", -1)), 0.1, rel_tol=0, abs_tol=1e-12):
    errors.append(
        f"loss_weights.ranking_weight: expected 0.1, got {loss.get('ranking_weight')!r}"
    )

if p.get("init_checkpoint") != "":
    errors.append(
        f"init_checkpoint: expected empty, got {p.get('init_checkpoint')!r}"
    )

if errors:
    raise SystemExit(
        "Mixed-matched20 protocol validation failed:\n  - "
        + "\n  - ".join(errors)
    )

print("[MGF MIXED-MATCHED20] protocol validation: OK")
PY
fi
