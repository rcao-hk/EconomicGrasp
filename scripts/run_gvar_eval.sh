#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CKPT:?Set CKPT to a GVAR full checkpoint}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/gvar_eval}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
REMOVE_DUMP="${REMOVE_DUMP:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
IFS=',' read -r -a GPUs <<< "${GPUS}"
IFS=',' read -r -a split_array <<< "${SPLITS}"
[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPUS" >&2; exit 2; }
for split in "${split_array[@]}"; do
  case "$split" in test_seen|test_similar|test_novel) ;; *) echo "Invalid split ${split}" >&2; exit 2;; esac
done
for flag in "$RUN_INFERENCE" "$RUN_EVAL" "$REMOVE_DUMP"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || { echo "Flags must be 0 or 1" >&2; exit 2; }
done
if [[ "$MAX_BATCHES" != 0 && "$RUN_EVAL" == 1 ]]; then echo "Smoke inference cannot be evaluated. Set RUN_EVAL=0." >&2; exit 2; fi
if [[ "${DRY_RUN:-0}" != 1 && ! -f "$CKPT" ]]; then echo "Missing CKPT: $CKPT" >&2; exit 2; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run_inference() {
  local split="$1" gpu="$2" out="${OUTPUT_ROOT}/$1"
  local extra=()
  if [[ "${RESUME_INFERENCE:-0}" == 1 ]]; then extra+=(--gvar_resume_inference); fi
  local cmd=("$PYTHON" inference_gvar.py --dataset_root "$DATASET_ROOT" --checkpoint_path "$CKPT"
    --save_dir "$out" --test_mode "$split" --camera realsense --sample_interval 0.1
    --batch_size "${BATCH_SIZE:-3}" --num_workers "${NUM_WORKERS:-2}" --seed "${SEED:-0}"
    --collision_thresh "${COLLISION_THRESH:-0.01}" --collision_voxel_size "${COLLISION_VOXEL_SIZE:-0.01}"
    --multi_modal --use_cdf --gvar_variant auto --gvar_max_batches "$MAX_BATCHES" "${extra[@]}")
  echo "[GVAR] $split GPU=$gpu -> $out"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; return; fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$out/inference.log" 2>&1
}
if [[ "$RUN_INFERENCE" == 1 ]]; then
  pids=()
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
  for ((start=0; start<${#split_array[@]}; start+=${#GPUs[@]})); do
    pids=()
    for ((j=0; j<${#GPUs[@]} && start+j<${#split_array[@]}; j++)); do
      run_inference "${split_array[start+j]}" "${GPUs[j]}" & pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    [[ "$failed" == 0 ]] || { echo "Inference failed; inspect split inference.log" >&2; exit 1; }
  done
fi
if [[ "$RUN_EVAL" == 1 ]]; then
  for split in "${split_array[@]}"; do
    out="${OUTPUT_ROOT}/${split}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "$PYTHON eval.py --split $split --dump_dir $out --sample_interval 10"; continue; fi
    "$PYTHON" - "$out" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
s=json.loads((p/'gvar_inference_summary.json').read_text())
m=json.loads((p/'gvar_inference_protocol.json').read_text())
assert s['complete'] and s['valid_dump_count']==780 and m['frame_stride']==10, 'Incomplete/mismatched dumps'
assert len(list(p.glob('scene_*/*/*.npy')))==780, 'Dumps missing or polluted; re-run inference in an isolated directory'
PY
    echo "[GVAR] sequential CPU evaluation $split"
    "$PYTHON" eval.py --dataset_root "$DATASET_ROOT" --camera realsense --split "$split" \
      --dump_dir "$out" --sample_interval 10 --num_workers "${EVAL_NUM_WORKERS:-16}" > "$out/evaluation.log" 2>&1
    # Delete only raw scene prediction files AFTER a valid AP tensor is saved.
    "$PYTHON" - "$out" "$split" "$REMOVE_DUMP" <<'PY'
import numpy as np, pathlib, sys
p=pathlib.Path(sys.argv[1]); a=np.load(p/f'ap_{sys.argv[2]}_realsense.npy',allow_pickle=False)
assert a.shape==(30,26,50,6) and np.isfinite(a).all(), f'Invalid AP tensor {a.shape}'
if sys.argv[3]=='1':
    for f in p.glob('scene_*/*/*.npy'): f.unlink()
PY
  done
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$PYTHON" summarize_gvar.py --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary"; fi
fi
