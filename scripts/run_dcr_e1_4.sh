#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

# DCR-E1-4: corruption-suite evaluation for the decoupled center corrector.
# Two corrected-action models are run under the SAME frozen Stage-1 ranking:
#   dcr    = current DCR corrector checkpoint
#   e1_ref = original E1 corrector checkpoint loaded as a zero-residual DCR model
# Native is evaluated once under dcr. This isolates whether DCR fine-tuning
# improves physical center correction, including smooth/local depth errors.

E1_BASE_ROOT=${E1_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/e1e2_cva_10pct}
DCR_BASE_ROOT=${DCR_BASE_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_cva_10pct}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dcr_e1_4_10pct}

E1_CHECKPOINT=${E1_CHECKPOINT:-$E1_BASE_ROOT/train/E1/checkpoint_best.pt}
DCR_CHECKPOINT=${DCR_CHECKPOINT:-$DCR_BASE_ROOT/train/cdf/checkpoint_best.pt}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}

GPUS=${GPUS:-0}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
TEST_CASES=${TEST_CASES:-nominal,bias:-15,bias:15,bias:-25,bias:25,scale:-0.03,scale:0.03,smooth:5,smooth:10}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
INFER_QUERIES=${INFER_QUERIES:-0}
INFER_MAX_FRAMES=${INFER_MAX_FRAMES:-0}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
PHASES=${PHASES:-infer,eval,summary}
RESUME=${RESUME:-1}

# By default evaluate only the scores needed for the causal comparison.
# DCR native is the shared Stage-1 baseline; e1_ref native would be identical.
DCR_METHODS=${DCR_METHODS:-native,stage1}
E1_METHODS=${E1_METHODS:-stage1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1

command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 2; }
[[ -f "$DCR_CHECKPOINT" ]] || { echo "Missing DCR checkpoint: $DCR_CHECKPOINT" >&2; exit 2; }
[[ -f "$E1_CHECKPOINT" ]] || { echo "Missing E1 checkpoint: $E1_CHECKPOINT" >&2; exit 2; }
[[ -f "$STAGE1_CKPT" ]] || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT" >&2; exit 2; }

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a TESTS <<< "$SPLITS"
IFS=',' read -r -a STEPS <<< "$PHASES"
[[ ${#GPU_IDS[@]} -gt 0 && "$OFFICIAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
for split in "${TESTS[@]}"; do
  [[ "$split" == test_seen || "$split" == test_similar || "$split" == test_novel ]] || {
    echo "Bad split: $split" >&2; exit 2; }
done

PIDS=(); NAMES=()
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  if (( ${#PIDS[@]} )); then
    sleep 2
    for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  mkdir -p "$(dirname "$logfile")"
  echo "[DCR-E1-4] $name GPU=$gpu log=$logfile"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$@" >>"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}

wait_wave() {
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[DCR-E1-4 ERROR] ${NAMES[$i]} failed; stopping remaining jobs" >&2
      exit 1
    fi
  done
  PIDS=(); NAMES=()
}

resume=(); [[ "$RESUME" == 1 ]] && resume+=(--resume)

run_inference_source() {
  local label="$1" checkpoint="$2"
  for split in "${TESTS[@]}"; do
    for s in "${!GPU_IDS[@]}"; do
      launch "infer/$label/$split/$s" "${GPU_IDS[$s]}"         "$WORK_ROOT/logs/infer_${label}_${split}_${s}.log"         "$ROOT_DIR/inference_dcr_cva.py"         --dataset-root "$DATASET_ROOT"         --stage1-checkpoint "$STAGE1_CKPT"         --checkpoint "$checkpoint"         --output-root "$WORK_ROOT/test/$label"         --split "$split"         --cases "$TEST_CASES"         --sample-interval "$SAMPLE_INTERVAL"         --rank-strength 0         --query-limit "$INFER_QUERIES"         --max-frames "$INFER_MAX_FRAMES"         --shard-id "$s"         --num-shards "${#GPU_IDS[@]}"         "${resume[@]}"
    done
    wait_wave
  done
}

run_eval_source() {
  local label="$1" methods="$2"
  local slot=0
  for split in "${TESTS[@]}"; do
    launch "official/$label/$split" "${GPU_IDS[$slot]}"       "$WORK_ROOT/logs/official_${label}_${split}.log"       "$ROOT_DIR/eval_dcr_cva.py"       --dataset-root "$DATASET_ROOT"       --inference-root "$WORK_ROOT/test/$label"       --split "$split"       --methods "$methods"       --workers "$OFFICIAL_WORKERS"       "${resume[@]}"
    slot=$((slot+1))
    if (( slot == ${#GPU_IDS[@]} )); then
      wait_wave
      slot=0
    fi
  done
  wait_wave
}

for phase in "${STEPS[@]}"; do
  case "$phase" in
    infer)
      run_inference_source dcr "$DCR_CHECKPOINT"
      run_inference_source e1_ref "$E1_CHECKPOINT"
      ;;
    eval|official)
      run_eval_source dcr "$DCR_METHODS"
      run_eval_source e1_ref "$E1_METHODS"
      ;;
    summary)
      "$PYTHON_BIN" "$ROOT_DIR/summarize_dcr_e1_4.py"         --work-root "$WORK_ROOT"         --dcr-label dcr         --e1-label e1_ref
      ;;
    audit)
      "$PYTHON_BIN" "$ROOT_DIR/audit_dcr_selection.py" \
        --dcr-root "$WORK_ROOT/test/dcr" \
        --e1-root "$WORK_ROOT/test/e1_ref" \
        --output-root "$WORK_ROOT/selection_audit" \
        --splits "$SPLITS" \
        --topks 10,50
      ;;
    *)
      echo "Unknown phase: $phase" >&2
      exit 2
      ;;
  esac
done
