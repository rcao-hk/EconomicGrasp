#!/usr/bin/env bash
# Frozen-reader P0-A counterfactuals, and P0-C paired RGB-domain evaluation.
# DOES NOT TRAIN. Keeps all original 17-column grasp dumps unless user removes
# them manually after archiving AP tensors and provenance.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATASET_ROOT:?Set original GraspNet dataset root}"
KIND="${KIND:-p0a}"                # p0a or p0c
ROOT="${ROOT:-/data/robotarm/result/grasp/rgbgrasp/log/gvar_deploy_20261007}"
OUT_ROOT="${OUT_ROOT:-$ROOT/analysis/gvar_p0_controlled_20261012}"
GPUS="${GPUS:-0,1,2}"
SPLITS="${SPLITS:-test_seen,test_similar,test_novel}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-16}"
SAMPLE_SEED="${SAMPLE_SEED:-20261012}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python}"
[[ "$KIND" == p0a || "$KIND" == p0c ]] || { echo 'KIND must be p0a|p0c' >&2; exit 2; }
for x in "$RUN_INFERENCE" "$RUN_EVAL" "$RESUME" "$DRY_RUN"; do
  [[ "$x" =~ ^[01]$ ]] || { echo 'Run flags must be 0/1' >&2; exit 2; }
done
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Bad GPU list $GPUS" >&2; exit 2; }
[[ "$MAX_BATCHES" =~ ^[0-9]+$ ]] || { echo 'MAX_BATCHES must be nonnegative integer' >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[0-9]+$ && "$BATCH_SIZE" -gt 0 ]] || { echo 'BATCH_SIZE must be positive' >&2; exit 2; }
if [[ "$MAX_BATCHES" -gt 0 && "$RUN_EVAL" == 1 ]]; then
  echo 'Partial smoke dumps cannot enter official evaluation; set RUN_EVAL=0' >&2; exit 2
fi
IFS=',' read -r -a gpu_array <<< "$GPUS"
IFS=',' read -r -a split_array <<< "$SPLITS"
for split in "${split_array[@]}"; do
  case "$split" in test_seen|test_similar|test_novel) ;; *) echo "Invalid split $split" >&2; exit 2;; esac
done
if [[ "$KIND" == p0a ]]; then
  variants=(volume_rel); domains=(original); modes=(r0 r1 r2 r3 r4)
else
  variants=(baseline volume_fixed volume_rel); domains=(original gntrans); modes=(r0)
  : "${GNTRANS_RGB_ROOT:?Set GNTRANS_RGB_ROOT for paired RGB test}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" PYTHONUNBUFFERED=1

jobs_variant=();jobs_domain=();jobs_mode=();jobs_split=()
for variant in "${variants[@]}"; do
  for domain in "${domains[@]}"; do
    for mode in "${modes[@]}"; do
      for split in "${split_array[@]}"; do
        jobs_variant+=("$variant"); jobs_domain+=("$domain")
        jobs_mode+=("$mode");jobs_split+=("$split")
      done
    done
  done
done
run_job() {
  local i="$1" gpu="$2" variant="${jobs_variant[$i]}" domain="${jobs_domain[$i]}" \
        mode="${jobs_mode[$i]}" split="${jobs_split[$i]}"
  local ckpt="$ROOT/train/$variant/checkpoint_epoch_019.tar"
  local out="$OUT_ROOT/$KIND/$variant/$domain/$mode/$split"
  local extra=()
  [[ "$domain" == original ]] || extra+=(--p0_gntrans_rgb_root "$GNTRANS_RGB_ROOT")
  [[ "$RESUME" == 0 ]] || extra+=(--p0_resume)
  local cmd=("$PYTHON" inference_gvar_p0.py --dataset_root "$DATASET_ROOT"
    --checkpoint_path "$ckpt" --camera realsense --test_mode "$split"
    --save_dir "$out" --sample_interval .1 --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
    --seed 0 --collision_thresh .01 --collision_voxel_size .01 --multi_modal --use_cdf
    --gvar_variant auto --p0_domain "$domain" --p0_cf_mode "$mode"
    --p0_sampling_seed "$SAMPLE_SEED" --p0_max_batches "$MAX_BATCHES" "${extra[@]}")
  echo "[P0-$KIND] infer GPU=$gpu variant=$variant domain=$domain cf=$mode split=$split"
  if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; return; fi
  [[ -f "$ckpt" ]] || { echo "Missing full e19 checkpoint: $ckpt" >&2; return 1; }
  # Avoid truncating previous inference/failure evidence on an accidental rerun.
  if [[ "$RESUME" == 0 && -e "$out/inference.log" ]]; then
    echo "Existing inference log: $out/inference.log; choose new OUT_ROOT or RESUME=1" >&2
    return 1
  fi
  mkdir -p "$out"
  if [[ "$RESUME" == 1 ]]; then
    printf '\n[RESUME] %s mode=%s split=%s\n' "$(date -Is)" "$mode" "$split" >>"$out/inference.log"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >>"$out/inference.log" 2>&1
  else
    (set -C; CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$out/inference.log" 2>&1)
  fi
}

if [[ "$RUN_INFERENCE" == 1 ]]; then
  pids=()
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
  for ((start=0;start<${#jobs_variant[@]};start+=${#gpu_array[@]})); do
    pids=();labels=()
    for ((j=0;j<${#gpu_array[@]} && start+j<${#jobs_variant[@]};j++)); do
      i=$((start+j))
      run_job "$i" "${gpu_array[$j]}" & pids+=("$!");labels+=("${jobs_variant[$i]}/${jobs_domain[$i]}/${jobs_mode[$i]}/${jobs_split[$i]}")
    done
    failed=0
    for ((k=0;k<${#pids[@]};k++)); do
      if ! wait "${pids[$k]}"; then echo "FAILED ${labels[$k]}" >&2;failed=1; fi
    done
    [[ "$failed" == 0 ]] || exit 1
  done
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  for ((i=0;i<${#jobs_variant[@]};i++)); do
    variant="${jobs_variant[$i]}";domain="${jobs_domain[$i]}";mode="${jobs_mode[$i]}"
    split="${jobs_split[$i]}";out="$OUT_ROOT/$KIND/$variant/$domain/$mode/$split"
    [[ "$DRY_RUN" == 1 ]] && { echo "[DRY RUN] evaluate $out"; continue; }
    "$PYTHON" - "$out" "$variant" "$domain" "$mode" "$split" <<'PY'
import json, pathlib,sys
p=pathlib.Path(sys.argv[1]);v,dom,mode,split=sys.argv[2:]
mp=json.loads((p/'gvar_p0_protocol.json').read_text())
s=json.loads((p/'gvar_p0_summary.json').read_text())
assert (mp['gvar_config']['variant'],mp['source_domain'],mp['reader_cf_mode'],mp['split']) == (v,dom,mode,split)
assert s['complete'] and s['valid_dump_count']==780 and s['expected_count']==780
assert mp['frame_stride']==10 and mp['collision_source']=='original_sensor'
assert len(list(p.glob('scene_*/*/*.npy')))==780, 'missing or polluted dumps'
PY
    # Official AP and evaluation logs are evidence; never silently replace them.
    if [[ -e "$out/evaluation.log" || -e "$out/ap_${split}_realsense.npy" ]]; then
      echo "Existing official evaluation: $out; choose a fresh OUT_ROOT" >&2
      exit 1
    fi
    echo "[P0-$KIND] serial official evaluator $variant/$domain/$mode/$split"
    (set -C; "$PYTHON" eval.py --dataset_root "$DATASET_ROOT" --camera realsense --split "$split" \
      --dump_dir "$out" --sample_interval 10 --num_workers "$EVAL_NUM_WORKERS" \
      >"$out/evaluation.log" 2>&1)
    "$PYTHON" - "$out" "$split" <<'PY'
import pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1])/f'ap_{sys.argv[2]}_realsense.npy'
a=np.load(p,allow_pickle=False)
assert (a.shape==(30,26,50,6) and np.isfinite(a).all()
        and 0 <= a.min() and a.max()<=1), f'invalid official AP {p}: {a.shape}'
PY
  done
fi
if [[ "$DRY_RUN" != 1 && "$RUN_EVAL" == 1 ]]; then
  if [[ "$KIND" == p0a ]]; then
    "$PYTHON" analyze_gvar_p0a.py --root "$OUT_ROOT/p0a" --output-dir "$OUT_ROOT/p0a_analysis" \
      --bootstrap "${BOOTSTRAP:-50000}" --seed 20261009
  else
    "$PYTHON" analyze_gvar_p0c.py --root "$OUT_ROOT/p0c" --output-dir "$OUT_ROOT/p0c_analysis" \
      --bootstrap "${BOOTSTRAP:-50000}" --seed 20261009
  fi
fi
