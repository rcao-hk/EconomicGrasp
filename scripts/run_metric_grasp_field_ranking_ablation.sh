#!/usr/bin/env bash
# Controlled 10%-GraspNet / 20-epoch ablation:
#   A) no ranking loss       RANKING_WEIGHT=0
#   B) query-listwise rank   RANKING_WEIGHT=0.1 (configurable)
#
# Runs execute concurrently on disjoint 3-GPU sets:
#   no-ranking -> GPUs 0,1,2
#   ranking    -> GPUs 3,5,6
# They otherwise differ only in ranking weight.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
ABLATION_ROOT=${ABLATION_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_ranking_ablation_10pct}

# Fair-comparison defaults: two disjoint 3-GPU jobs in parallel.
NO_RANK_GPUS=${NO_RANK_GPUS:-0,1,2}
RANK_GPUS=${RANK_GPUS:-3,5,6}
NO_RANK_INFER_GPUS=${NO_RANK_INFER_GPUS:-$NO_RANK_GPUS}
RANK_INFER_GPUS=${RANK_INFER_GPUS:-$RANK_GPUS}
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

IFS=',' read -r -a NO_RANK_GPU_IDS <<< "$NO_RANK_GPUS"
IFS=',' read -r -a RANK_GPU_IDS <<< "$RANK_GPUS"

if [[ "${#NO_RANK_GPU_IDS[@]}" -ne 3 ]]; then
  echo "[ERROR] NO_RANK_GPUS must contain exactly 3 GPU ids; got $NO_RANK_GPUS" >&2
  exit 2
fi
if [[ "${#RANK_GPU_IDS[@]}" -ne 3 ]]; then
  echo "[ERROR] RANK_GPUS must contain exactly 3 GPU ids; got $RANK_GPUS" >&2
  exit 2
fi

for id in "${NO_RANK_GPU_IDS[@]}" "${RANK_GPU_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Invalid GPU id: $id" >&2
    exit 2
  }
done
for no_id in "${NO_RANK_GPU_IDS[@]}"; do
  for rank_id in "${RANK_GPU_IDS[@]}"; do
    if [[ "$no_id" == "$rank_id" ]]; then
      echo "[ERROR] GPU $no_id appears in both variants; GPU sets must be disjoint." >&2
      exit 2
    fi
  done
done

echo "============================================================"
echo "[MGF RANK ABLATION] common protocol"
echo "  dataset       : $DATASET_ROOT"
echo "  no-rank GPUs  : $NO_RANK_GPUS"
echo "  ranking GPUs  : $RANK_GPUS"
echo "  batch/GPU     : $BATCH_SIZE"
echo "  effective B   : $(( BATCH_SIZE * 3 )) per variant"
echo "  epochs        : $EPOCHS"
echo "  fraction      : $SAMPLE_FRACTION"
echo "  seed          : $SEED"
echo "  init checkpoint: EMPTY (from-scratch task heads)"
echo "  phases        : $PHASES"
echo "  checkpoint    : $CHECKPOINT_KIND"
echo "  no-rank root  : $NO_RANK_ROOT"
echo "  rank root     : $RANK_ROOT"
echo "============================================================"

NO_RANK_LAUNCHER_LOG="$NO_RANK_ROOT/launcher.log"
RANK_LAUNCHER_LOG="$RANK_ROOT/launcher.log"
NO_RANK_PID=""
RANK_PID=""

cleanup_parallel() {
  local pid
  for pid in "$NO_RANK_PID" "$RANK_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      # Each variant is launched as its own session. Sending TERM to the
      # session leader triggers run_metric_grasp_field.sh's cleanup trap,
      # which then terminates its torchrun process group.
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_parallel EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch_one() {
  local name="$1"
  local root="$2"
  local rank_weight="$3"
  local gpu_set="$4"
  local infer_gpu_set="$5"
  local launcher_log="$6"

  mkdir -p "$root"
  echo "[MGF RANK ABLATION] launching $name"
  echo "  GPUs=$gpu_set"
  echo "  RANKING_WEIGHT=$rank_weight"
  echo "  launcher log=$launcher_log"

  setsid env \
    DATASET_ROOT="$DATASET_ROOT" \
    WORK_ROOT="$root" \
    INIT_CHECKPOINT="" \
    GPUS="$gpu_set" \
    INFER_GPUS="$infer_gpu_set" \
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
    bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh" \
    >"$launcher_log" 2>&1 &

  LAST_PID=$!
}

launch_one \
  "no_ranking" "$NO_RANK_ROOT" "0" \
  "$NO_RANK_GPUS" "$NO_RANK_INFER_GPUS" "$NO_RANK_LAUNCHER_LOG"
NO_RANK_PID=$LAST_PID

launch_one \
  "ranking" "$RANK_ROOT" "$RANKING_WEIGHT_ON" \
  "$RANK_GPUS" "$RANK_INFER_GPUS" "$RANK_LAUNCHER_LOG"
RANK_PID=$LAST_PID

echo "[MGF RANK ABLATION] both variants are running in parallel"
echo "  no-ranking pid=$NO_RANK_PID"
echo "  ranking    pid=$RANK_PID"

# Reap whichever variant finishes first. If it fails, terminate the other
# variant immediately rather than wasting the remaining GPUs.
if wait -n "$NO_RANK_PID" "$RANK_PID"; then
  FIRST_STATUS=0
else
  FIRST_STATUS=$?
fi

if [[ "$FIRST_STATUS" -ne 0 ]]; then
  echo "[MGF RANK ABLATION ERROR] first completed variant failed (status=$FIRST_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  cleanup_parallel
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
  exit "$FIRST_STATUS"
fi

# One child has been reaped by wait -n. The still-live PID is the remaining
# variant; if both happened to finish together, wait on both is harmless.
SECOND_STATUS=0
if kill -0 "$NO_RANK_PID" 2>/dev/null; then
  wait "$NO_RANK_PID" || SECOND_STATUS=$?
elif kill -0 "$RANK_PID" 2>/dev/null; then
  wait "$RANK_PID" || SECOND_STATUS=$?
else
  # Both may have exited between wait -n and the checks. Reap any unreaped
  # child; "not a child" is ignored because wait -n already reaped one.
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
fi

if [[ "$SECOND_STATUS" -ne 0 ]]; then
  echo "[MGF RANK ABLATION ERROR] second variant failed (status=$SECOND_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  exit "$SECOND_STATUS"
fi

# Disable the cleanup trap after both variants have completed successfully.
NO_RANK_PID=""
RANK_PID=""

echo "[MGF RANK ABLATION] both variants completed successfully"

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
