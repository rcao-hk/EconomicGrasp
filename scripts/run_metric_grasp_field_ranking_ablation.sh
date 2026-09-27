#!/usr/bin/env bash
# Controlled 10%-GraspNet / 20-epoch ablation:
#   A) no ranking loss       RANKING_WEIGHT=0
#   B) query-listwise rank   RANKING_WEIGHT=0.1 (configurable)
#
# Both runs are sequential on the SAME GPUs and differ only in ranking weight.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
ABLATION_ROOT=${ABLATION_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_ranking_ablation_10pct}

# Fair-comparison defaults: match the user's current 4-GPU setup.
GPUS=${GPUS:-0,1,2,3}
INFER_GPUS=${INFER_GPUS:-$GPUS}
BATCH_SIZE=${BATCH_SIZE:-3}
EPOCHS=${EPOCHS:-20}
SEED=${SEED:-42}
SAMPLE_FRACTION=${SAMPLE_FRACTION:-0.1}

# Keep all non-ranking settings identical.
LR=${LR:-0.0001}
GEOMETRY_LR=${GEOMETRY_LR:-0.00001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
RANKING_TEMPERATURE=${RANKING_TEMPERATURE:-0.1}
RANKING_WEIGHT_ON=${RANKING_WEIGHT_ON:-0.1}
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
WORKERS=${WORKERS:-2}
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
AMP=${AMP:-0}
TOP4=${TOP4:-0}
RESUME=${RESUME:-0}

# Default: training only. You can set PHASES=train,infer,eval later.
# For a ranking ablation, if inference is requested, compare epoch-20/latest
# rather than the BCE-selected "best" checkpoint.
PHASES=${PHASES:-train}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-latest}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}

NO_RANK_ROOT=${NO_RANK_ROOT:-$ABLATION_ROOT/no_ranking}
RANK_ROOT=${RANK_ROOT:-$ABLATION_ROOT/ranking_w0p1}
SUMMARY_DIR=${SUMMARY_DIR:-$ABLATION_ROOT/comparison}

if [[ "$SAMPLE_FRACTION" != "0.1" ]]; then
  echo "[WARN] SAMPLE_FRACTION=$SAMPLE_FRACTION; intended formal ablation is 0.1" >&2
fi
if [[ "$EPOCHS" != "20" ]]; then
  echo "[WARN] EPOCHS=$EPOCHS; intended formal ablation is 20" >&2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -ne 4 ]]; then
  echo "[WARN] GPUS=$GPUS gives ${#GPU_IDS[@]} GPUs; intended comparison is 4 GPUs." >&2
fi

echo "============================================================"
echo "[MGF RANK ABLATION] common protocol"
echo "  dataset       : $DATASET_ROOT"
echo "  GPUs          : $GPUS"
echo "  batch/GPU     : $BATCH_SIZE"
echo "  effective B   : $(( BATCH_SIZE * ${#GPU_IDS[@]} ))"
echo "  epochs        : $EPOCHS"
echo "  fraction      : $SAMPLE_FRACTION"
echo "  seed          : $SEED"
echo "  init checkpoint: EMPTY (from-scratch task heads)"
echo "  phases        : $PHASES"
echo "  checkpoint    : $CHECKPOINT_KIND"
echo "  no-rank root  : $NO_RANK_ROOT"
echo "  rank root     : $RANK_ROOT"
echo "============================================================"

run_one() {
  local name="$1"
  local root="$2"
  local rank_weight="$3"

  echo
  echo "============================================================"
  echo "[MGF RANK ABLATION] START $name"
  echo "  RANKING_WEIGHT=$rank_weight"
  echo "============================================================"

  env \
    DATASET_ROOT="$DATASET_ROOT" \
    WORK_ROOT="$root" \
    INIT_CHECKPOINT="" \
    GPUS="$GPUS" \
    INFER_GPUS="$INFER_GPUS" \
    PHASES="$PHASES" \
    SPLITS="$SPLITS" \
    CHECKPOINT_KIND="$CHECKPOINT_KIND" \
    RESUME="$RESUME" \
    ENCODER="$ENCODER" \
    POSE_MODE="$POSE_MODE" \
    SEED_MODE="$SEED_MODE" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    LR="$LR" \
    GEOMETRY_LR="$GEOMETRY_LR" \
    WEIGHT_DECAY="$WEIGHT_DECAY" \
    SAMPLE_FRACTION="$SAMPLE_FRACTION" \
    WORKERS="$WORKERS" \
    EVAL_WORKERS="$EVAL_WORKERS" \
    OFFICIAL_WORKERS="$OFFICIAL_WORKERS" \
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
    PROFILE_WEIGHT="$PROFILE_WEIGHT" \
    PROFILE_MEAN_WEIGHT="$PROFILE_MEAN_WEIGHT" \
    BASE_CDF_WEIGHT="$BASE_CDF_WEIGHT" \
    RANKING_WEIGHT="$rank_weight" \
    RANKING_TEMPERATURE="$RANKING_TEMPERATURE" \
    SEED="$SEED" \
    AMP="$AMP" \
    TOP4="$TOP4" \
    MAX_TRAIN_FRAMES=0 \
    MAX_VAL_FRAMES=0 \
    MAX_STEPS=0 \
    INFER_MAX_FRAMES=0 \
    bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh"

  echo "[MGF RANK ABLATION] DONE $name"
}

# Sequential by design: same GPUs, same resource conditions, no contention.
run_one "no_ranking" "$NO_RANK_ROOT" "0"
run_one "ranking" "$RANK_ROOT" "$RANKING_WEIGHT_ON"

# Training-only comparison is always available after both runs.
mkdir -p "$SUMMARY_DIR"
"$PYTHON_BIN" "$ROOT_DIR/compare_metric_grasp_field_ranking.py" \
  --no-ranking "$NO_RANK_ROOT/train" \
  --ranking "$RANK_ROOT/train" \
  --output-dir "$SUMMARY_DIR" \
  --expected-epochs "$EPOCHS"

echo
echo "============================================================"
echo "[MGF RANK ABLATION] complete"
echo "Summary:"
echo "  $SUMMARY_DIR/comparison.md"
echo "  $SUMMARY_DIR/comparison.json"
echo "  $SUMMARY_DIR/per_epoch.tsv"
echo "============================================================"
