#!/usr/bin/env bash
# Parallel inference + official evaluation for the matched MGF ranking ablation.
#
# Expected trained runs:
#   no-ranking: <ABLATION_ROOT>/no_ranking/train/checkpoint_latest.pt
#   ranking:    <ABLATION_ROOT>/ranking_w0p1/train/checkpoint_latest.pt
#
# Default GPU allocation:
#   no-ranking -> 0,1,2
#   ranking    -> 3,5,6
#
# Both variants evaluate checkpoint_latest.pt on Seen/Similar/Novel.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
ABLATION_ROOT=${ABLATION_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_ranking_ablation_10pct}

NO_RANK_GPUS=${NO_RANK_GPUS:-0,1,2}
RANK_GPUS=${RANK_GPUS:-3,5,6}
NO_RANK_INFER_GPUS=${NO_RANK_INFER_GPUS:-$NO_RANK_GPUS}
RANK_INFER_GPUS=${RANK_INFER_GPUS:-$RANK_GPUS}

NO_RANK_ROOT=${NO_RANK_ROOT:-$ABLATION_ROOT/no_ranking}
RANK_ROOT=${RANK_ROOT:-$ABLATION_ROOT/ranking_w0p1}
SUMMARY_DIR=${SUMMARY_DIR:-$ABLATION_ROOT/comparison}

SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
TOP4=${TOP4:-0}
RESUME=${RESUME:-0}

# Must compare the same epoch-20/latest checkpoints used by the training ablation.
CHECKPOINT_KIND=latest
PHASES=infer,eval

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

NO_RANK_CKPT="$NO_RANK_ROOT/train/checkpoint_latest.pt"
RANK_CKPT="$RANK_ROOT/train/checkpoint_latest.pt"

[[ -f "$NO_RANK_CKPT" ]] || {
  echo "[ERROR] Missing no-ranking checkpoint: $NO_RANK_CKPT" >&2
  exit 2
}
[[ -f "$RANK_CKPT" ]] || {
  echo "[ERROR] Missing ranking checkpoint: $RANK_CKPT" >&2
  exit 2
}

# Validate that the two completed training runs really form the intended
# controlled ablation before spending time on inference/evaluation.
mkdir -p "$SUMMARY_DIR"
"$PYTHON_BIN" "$ROOT_DIR/compare_metric_grasp_field_ranking.py"   --no-ranking "$NO_RANK_ROOT/train"   --ranking "$RANK_ROOT/train"   --output-dir "$SUMMARY_DIR"   --expected-epochs 20   >"$SUMMARY_DIR/pre_eval_comparison.log" 2>&1

echo "============================================================"
echo "[MGF RANK ABLATION EVAL]"
echo "  dataset          : $DATASET_ROOT"
echo "  splits           : $SPLITS"
echo "  checkpoint kind  : latest"
echo "  no-ranking GPUs  : $NO_RANK_GPUS"
echo "  ranking GPUs     : $RANK_GPUS"
echo "  no-ranking ckpt  : $NO_RANK_CKPT"
echo "  ranking ckpt     : $RANK_CKPT"
echo "  resume           : $RESUME"
echo "============================================================"

NO_RANK_LAUNCHER_LOG="$NO_RANK_ROOT/infer_eval_launcher.log"
RANK_LAUNCHER_LOG="$RANK_ROOT/infer_eval_launcher.log"
NO_RANK_PID=""
RANK_PID=""

cleanup_parallel() {
  local pid
  for pid in "$NO_RANK_PID" "$RANK_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_parallel EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch_eval_variant() {
  local name="$1"
  local root="$2"
  local gpu_set="$3"
  local infer_gpu_set="$4"
  local launcher_log="$5"

  echo "[MGF RANK ABLATION EVAL] launching $name on GPUs $gpu_set"
  mkdir -p "$root"

  setsid env     DATASET_ROOT="$DATASET_ROOT"     WORK_ROOT="$root"     GPUS="$gpu_set"     INFER_GPUS="$infer_gpu_set"     PHASES="$PHASES"     SPLITS="$SPLITS"     CHECKPOINT_KIND="$CHECKPOINT_KIND"     RESUME="$RESUME"     EVAL_WORKERS="$EVAL_WORKERS"     OFFICIAL_WORKERS="$OFFICIAL_WORKERS"     TOP4="$TOP4"     INFER_MAX_FRAMES=0     bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh"     >"$launcher_log" 2>&1 &

  LAST_PID=$!
}

launch_eval_variant   "no_ranking" "$NO_RANK_ROOT"   "$NO_RANK_GPUS" "$NO_RANK_INFER_GPUS" "$NO_RANK_LAUNCHER_LOG"
NO_RANK_PID=$LAST_PID

launch_eval_variant   "ranking" "$RANK_ROOT"   "$RANK_GPUS" "$RANK_INFER_GPUS" "$RANK_LAUNCHER_LOG"
RANK_PID=$LAST_PID

echo "[MGF RANK ABLATION EVAL] both variants running"
echo "  no-ranking pid=$NO_RANK_PID"
echo "  ranking    pid=$RANK_PID"

# Wait for the first job. On failure, stop the other immediately.
if wait -n "$NO_RANK_PID" "$RANK_PID"; then
  FIRST_STATUS=0
else
  FIRST_STATUS=$?
fi

if [[ "$FIRST_STATUS" -ne 0 ]]; then
  echo "[MGF RANK ABLATION EVAL ERROR] first completed variant failed (status=$FIRST_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  cleanup_parallel
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
  exit "$FIRST_STATUS"
fi

SECOND_STATUS=0
if kill -0 "$NO_RANK_PID" 2>/dev/null; then
  wait "$NO_RANK_PID" || SECOND_STATUS=$?
elif kill -0 "$RANK_PID" 2>/dev/null; then
  wait "$RANK_PID" || SECOND_STATUS=$?
else
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
fi

if [[ "$SECOND_STATUS" -ne 0 ]]; then
  echo "[MGF RANK ABLATION EVAL ERROR] second variant failed (status=$SECOND_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  exit "$SECOND_STATUS"
fi

NO_RANK_PID=""
RANK_PID=""

# Re-run the comparator after official evaluation. It will now pick up
# test_latest/official/<split>/summary.json and append official AP.
"$PYTHON_BIN" "$ROOT_DIR/compare_metric_grasp_field_ranking.py"   --no-ranking "$NO_RANK_ROOT/train"   --ranking "$RANK_ROOT/train"   --output-dir "$SUMMARY_DIR"   --expected-epochs 20   >"$SUMMARY_DIR/post_eval_comparison.log" 2>&1

echo
echo "============================================================"
echo "[MGF RANK ABLATION EVAL] complete"
echo "Official outputs:"
echo "  $NO_RANK_ROOT/test_latest/official/"
echo "  $RANK_ROOT/test_latest/official/"
echo "Updated comparison:"
echo "  $SUMMARY_DIR/comparison.md"
echo "  $SUMMARY_DIR/comparison.json"
echo "============================================================"
