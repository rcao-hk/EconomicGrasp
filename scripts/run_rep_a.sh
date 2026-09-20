#!/usr/bin/env bash
# Rep-A: one process/GPU; each job owns its process group. No DDP required.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
P0_CACHE_ROOT=${P0_CACHE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_p0_geometry_sources_5mm/cache}
CACHE_ROOT=${CACHE_ROOT:-${WORK_ROOT}/cache}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
PHASES=${PHASES:-prepare,train,test,summary}
GPUS=${GPUS:-0}
VARIANTS=${VARIANTS:-A0,A1,A2,A3}
PREP_SPLITS=${PREP_SPLITS:-train,test_seen,test_similar,test_novel}
TEST_SPLITS=${TEST_SPLITS:-test_similar,test_novel}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FEATURE_DTYPE=${FEATURE_DTYPE:-float16}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
SEED=${SEED:-0}
TEST_SEED=${TEST_SEED:-2026}
MAX_FRAMES=${MAX_FRAMES:-0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
MAX_BIAS_MM=${MAX_BIAS_MM:-20}
MAX_SCALE=${MAX_SCALE:-0.03}
SMOOTH_MM=${SMOOTH_MM:-10}
NOMINAL_PROB=${NOMINAL_PROB:-0.25}
RESUME=${RESUME:-1}
REPAIR_CORRUPT=${REPAIR_CORRUPT:-0}
SAVE_PER_QUERY=${SAVE_PER_QUERY:-0}
MIN_HOST_FREE_GIB=${MIN_HOST_FREE_GIB:-4}
TEST_CASES=${TEST_CASES:-nominal,bias:-5,bias:5,bias:-10,bias:10,bias:-20,bias:20,bias:-40,bias:40,scale:-0.02,scale:0.02,scale:-0.05,scale:0.05,smooth:5,smooth:10,smooth:20,edge:2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo 'setsid is required for safe process-group cleanup' >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a MODES <<< "$VARIANTS"
IFS=',' read -r -a PREP <<< "$PREP_SPLITS"
IFS=',' read -r -a TESTS <<< "$TEST_SPLITS"
IFS=',' read -r -a PHASE_LIST <<< "$PHASES"
for g in "${GPU_IDS[@]}"; do [[ -n "$g" ]] || { echo 'Empty GPU ID' >&2; exit 2; }; done
for m in "${MODES[@]}"; do case "$m" in A0|A1|A2|A3) ;; *) echo "Unknown variant $m" >&2; exit 2;; esac; done
PIDS=(); NAMES=()
cleanup() {
  local p
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  if (( ${#PIDS[@]} )); then
    sleep 2
    for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_wave() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-A ERROR] ${NAMES[$i]} failed; see its log" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}
launch() {
  local name="$1" gpu="$2" logfile="$3"
  shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[REP-A LAUNCH] $name GPU=$gpu LOG=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
for phase in "${PHASE_LIST[@]}"; do
  case "$phase" in
    prepare)
      [[ -f "$STAGE1_CKPT" && -d "$P0_CACHE_ROOT" ]] || { echo 'Missing Stage-1 or P0 cache root' >&2; exit 2; }
      for split in "${PREP[@]}"; do
        for shard in "${!GPU_IDS[@]}"; do
          args=("$ROOT_DIR/prepare_rep_a_cache.py" --dataset-root "$DATASET_ROOT"
            --p0-cache-root "$P0_CACHE_ROOT" --checkpoint "$STAGE1_CKPT" --output-root "$CACHE_ROOT"
            --split "$split" --pose-depth-mode "$POSE_DEPTH_MODE" --feature-dtype "$FEATURE_DTYPE"
            --shard-id "$shard" --num-shards "${#GPU_IDS[@]}" --max-frames "$MAX_FRAMES"
            --min-host-free-gib "$MIN_HOST_FREE_GIB")
          [[ "$REPAIR_CORRUPT" == 1 ]] && args+=(--repair-corrupt)
          launch "prepare/$split/$shard" "${GPU_IDS[$shard]}" "$WORK_ROOT/logs/prepare/${split}_${shard}.log" "${args[@]}"
        done
        wait_wave
      done
      ;;
    train)
      slot=0
      for mode in "${MODES[@]}"; do
        out="$WORK_ROOT/train/$mode"
        args=("$ROOT_DIR/train_rep_a.py" --cache-root "$CACHE_ROOT" --output-dir "$out" --variant "$mode"
          --epochs "$EPOCHS" --lr "$LR" --grad-accum-steps "$GRAD_ACCUM_STEPS" --seed "$SEED"
          --max-train-frames "$MAX_TRAIN_FRAMES" --max-val-frames "$MAX_VAL_FRAMES"
          --max-bias-mm "$MAX_BIAS_MM" --max-scale "$MAX_SCALE" --smooth-mm "$SMOOTH_MM" --nominal-prob "$NOMINAL_PROB")
        [[ "$RESUME" == 1 ]] && args+=(--resume)
        launch "train/$mode" "${GPU_IDS[$slot]}" "$out/train.log" "${args[@]}"
        slot=$((slot+1))
        if ((slot == ${#GPU_IDS[@]})); then wait_wave; slot=0; fi
      done
      wait_wave
      ;;
    test)
      # Check every requested checkpoint/cache BEFORE launching background jobs.
      for mode in "${MODES[@]}"; do
        [[ -f "$WORK_ROOT/train/$mode/checkpoint_best.pt" ]] || { echo "Missing checkpoint $mode" >&2; exit 2; }
      done
      for split in "${TESTS[@]}"; do [[ -d "$CACHE_ROOT/$split" ]] || { echo "Missing cache $split" >&2; exit 2; }; done
      slot=0
      for mode in "${MODES[@]}"; do
        for split in "${TESTS[@]}"; do
          out="$WORK_ROOT/test/$mode/$split"
          args=("$ROOT_DIR/test_rep_a.py" --cache-root "$CACHE_ROOT" --output-dir "$out"
            --checkpoint "$WORK_ROOT/train/$mode/checkpoint_best.pt" --split "$split"
            "--cases=$TEST_CASES" --seed "$TEST_SEED" --max-frames "$MAX_FRAMES")
          [[ "$RESUME" == 1 ]] && args+=(--resume)
          [[ "$SAVE_PER_QUERY" == 1 ]] && args+=(--save-per-query)
          launch "test/$mode/$split" "${GPU_IDS[$slot]}" "$out/test.log" "${args[@]}"
          slot=$((slot+1))
          if ((slot == ${#GPU_IDS[@]})); then wait_wave; slot=0; fi
        done
      done
      wait_wave
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_a.py" --test-root "$WORK_ROOT/test" --splits "$TEST_SPLITS"
      ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
  esac
done
