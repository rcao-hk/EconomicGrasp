#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_b_hypothesis_image}
REP_A_WORK_ROOT=${REP_A_WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/rep_a_depth_robustness}
CACHE_ROOT=${CACHE_ROOT:-${REP_A_WORK_ROOT}/cache}
PHASES=${PHASES:-train,test,summary}
GPUS=${GPUS:-0}
VARIANTS=${VARIANTS:-B0,B1,B2}
TEST_SPLITS=${TEST_SPLITS:-test_similar,test_novel}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
SEED=${SEED:-0}
TEST_SEED=${TEST_SEED:-2026}
DIM=${DIM:-0}
HEADS=${HEADS:-4}
LAYERS=${LAYERS:-2}
DROPOUT=${DROPOUT:-0.1}
PRIOR_SIGMA_MM=${PRIOR_SIGMA_MM:-30}
PAIRWISE_WEIGHT=${PAIRWISE_WEIGHT:-0}
PAIRWISE_TEMPERATURE=${PAIRWISE_TEMPERATURE:-0.1}
PAIRWISE_MIN_GAP=${PAIRWISE_MIN_GAP:-0.0001}
MAX_BIAS_MM=${MAX_BIAS_MM:-20}
MAX_SCALE=${MAX_SCALE:-0.03}
SMOOTH_MM=${SMOOTH_MM:-10}
NOMINAL_PROB=${NOMINAL_PROB:-0.25}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-0}
MAX_VAL_FRAMES=${MAX_VAL_FRAMES:-0}
MAX_TEST_FRAMES=${MAX_TEST_FRAMES:-0}
RESUME=${RESUME:-1}
PROGRESS_EVERY=${PROGRESS_EVERY:-100}
TEST_CASES=${TEST_CASES:-nominal,bias:-5,bias:5,bias:-10,bias:10,bias:-20,bias:20,bias:-40,bias:40,scale:-0.02,scale:0.02,scale:-0.05,scale:0.05,smooth:5,smooth:10,smooth:20,edge:2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1
command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 2; }
[[ -f "$CACHE_ROOT/reader_init.pt" ]] || { echo "Missing Rep-A cache at $CACHE_ROOT" >&2; exit 2; }
IFS="," read -r -a GPU_IDS <<< "$GPUS"
IFS="," read -r -a MODES <<< "$VARIANTS"
IFS="," read -r -a SPLITS <<< "$TEST_SPLITS"
IFS="," read -r -a PHASE_LIST <<< "$PHASES"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "No GPUs configured" >&2; exit 2; }
for m in "${MODES[@]}"; do
  case "$m" in B0|B1|B2) ;; *) echo "Unknown Rep-B variant: $m" >&2; exit 2;; esac
done
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
  local i failed=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-B ERROR] ${NAMES[$i]} failed; see log" >&2
      failed=1
    fi
  done
  PIDS=(); NAMES=()
  [[ $failed -eq 0 ]]
}
launch() {
  local name="$1" gpu="$2" logfile="$3"
  shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[REP-B LAUNCH] $name GPU=$gpu LOG=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}
for phase in "${PHASE_LIST[@]}"; do
  case "$phase" in
    train)
      slot=0
      for mode in "${MODES[@]}"; do
        out="$WORK_ROOT/train/$mode"
        args=(
          "$ROOT_DIR/train_rep_b.py"
          --cache-root "$CACHE_ROOT"
          --output-dir "$out"
          --variant "$mode"
          --epochs "$EPOCHS"
          --lr "$LR"
          --weight-decay "$WEIGHT_DECAY"
          --grad-accum-steps "$GRAD_ACCUM_STEPS"
          --seed "$SEED"
          --dim "$DIM"
          --heads "$HEADS"
          --layers "$LAYERS"
          --dropout "$DROPOUT"
          --prior-sigma-mm "$PRIOR_SIGMA_MM"
          --pairwise-weight "$PAIRWISE_WEIGHT"
          --pairwise-temperature "$PAIRWISE_TEMPERATURE"
          --pairwise-min-gap "$PAIRWISE_MIN_GAP"
          --max-bias-mm "$MAX_BIAS_MM"
          --max-scale "$MAX_SCALE"
          --smooth-mm "$SMOOTH_MM"
          --nominal-prob "$NOMINAL_PROB"
          --max-train-frames "$MAX_TRAIN_FRAMES"
          --max-val-frames "$MAX_VAL_FRAMES"
          --progress-every "$PROGRESS_EVERY"
        )
        [[ "$RESUME" == 1 ]] && args+=(--resume)
        launch "train/$mode" "${GPU_IDS[$slot]}" "$out/train.log" "${args[@]}"
        slot=$((slot+1))
        if (( slot == ${#GPU_IDS[@]} )); then wait_wave; slot=0; fi
      done
      (( ${#PIDS[@]} == 0 )) || wait_wave
      ;;
    test)
      for mode in "${MODES[@]}"; do
        [[ -f "$WORK_ROOT/train/$mode/checkpoint_best.pt" ]] || { echo "Missing Rep-B checkpoint: $mode" >&2; exit 2; }
      done
      slot=0
      for mode in "${MODES[@]}"; do
        for raw_split in "${SPLITS[@]}"; do
          split="$(echo "$raw_split" | xargs)"
          out="$WORK_ROOT/test/$mode/$split"
          args=(
            "$ROOT_DIR/test_rep_b.py"
            --cache-root "$CACHE_ROOT"
            --checkpoint "$WORK_ROOT/train/$mode/checkpoint_best.pt"
            --output-dir "$out"
            --split "$split"
            "--cases=$TEST_CASES"
            --seed "$TEST_SEED"
            --max-frames "$MAX_TEST_FRAMES"
            --progress-every "$PROGRESS_EVERY"
          )
          [[ "$RESUME" == 1 ]] && args+=(--resume)
          launch "test/$mode/$split" "${GPU_IDS[$slot]}" "$out/test.log" "${args[@]}"
          slot=$((slot+1))
          if (( slot == ${#GPU_IDS[@]} )); then wait_wave; slot=0; fi
        done
      done
      (( ${#PIDS[@]} == 0 )) || wait_wave
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_rep_b.py" --test-root "$WORK_ROOT/test" --splits "$TEST_SPLITS" --variants "$VARIANTS"
      ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
  esac
done
trap - EXIT INT TERM
echo "[REP-B] complete: $WORK_ROOT"
