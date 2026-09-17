#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
ABLATION_ROOT=${ABLATION_ROOT:-${WORK_ROOT}/feature_ablation_3layer}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/feature_ablation_3layer_heldout}
ADAPTED_ROOT=${ADAPTED_ROOT:-${OUTPUT_ROOT}/_adapted_checkpoints}

# Default to the two scientifically informative follow-ups.  Any of
# raw_offset,selected_residual,mean_residual,selected_mean,full can be supplied.
FEATURE_MODES=${FEATURE_MODES:-selected_mean,mean_residual}
SPLITS=${SPLITS:-test_similar,test_novel}
GPUS=${GPUS:-0,3,5,6}

CAMERA=${CAMERA:-realsense}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
OFFSETS_MM=${OFFSETS_MM:--40,-20,-10,0,10,20,40}
QUERY_EVAL_NUM=${QUERY_EVAL_NUM:-128}
QUERY_EVAL_MODE=${QUERY_EVAL_MODE:-topk_uniform}
POSE_DEPTH_MODE=${POSE_DEPTH_MODE:-global_film}
FC_MODE=${FC_MODE:-reuse_contacts}
VERIFY_N=${VERIFY_N:-0}
NOOP_CHECK_SAMPLES=${NOOP_CHECK_SAMPLES:-2}
NOOP_ATOL=${NOOP_ATOL:-5e-5}
PROFILE_TIMING=${PROFILE_TIMING:-1}
SAVE_CANDIDATE_ROWS=${SAVE_CANDIDATE_ROWS:-0}
SELECTOR_THRESHOLD=${SELECTOR_THRESHOLD:-}

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "Stage-1 checkpoint not found: ${STAGE1_CKPT}" >&2; exit 2; }
[[ -d "${ABLATION_ROOT}" ]] || { echo "Ablation root not found: ${ABLATION_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${ADAPTED_ROOT}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
IFS=',' read -r -a MODE_ARRAY <<< "${FEATURE_MODES}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUS specified" >&2; exit 2; }
[[ ${#MODE_ARRAY[@]} -gt 0 ]] || { echo "No FEATURE_MODES specified" >&2; exit 2; }
[[ ${#SPLIT_ARRAY[@]} -gt 0 ]] || { echo "No SPLITS specified" >&2; exit 2; }

# Convert each feature-ablation checkpoint exactly once.  The adapter verifies
# that the expanded full-input model reproduces the original subset model before
# writing the checkpoint.
declare -A ADAPTED_CKPTS
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  case "${mode}" in
    raw_offset|selected_residual|mean_residual|selected_mean|full) ;;
    *) echo "Unknown feature mode: ${mode}" >&2; exit 2 ;;
  esac
  src="${ABLATION_ROOT}/${mode}/checkpoint_best.tar"
  dst="${ADAPTED_ROOT}/${mode}_checkpoint_best_fullcontract.tar"
  [[ -f "${src}" ]] || { echo "Missing ablation checkpoint: ${src}" >&2; exit 2; }
  echo "[ABLATE-HELDOUT][ADAPT] mode=${mode}"
  "${PYTHON_BIN}" "${ROOT_DIR}/adapt_ray_pairwise_ablation_checkpoint.py" \
    --input_checkpoint "${src}" \
    --output_checkpoint "${dst}"
  ADAPTED_CKPTS["${mode}"]="${dst}"
done

PIDS=()
ACTIVE=()

wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[ABLATE-HELDOUT][ERROR] ${ACTIVE[$i]} failed; inspect its diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=()
  ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_job() {
  local mode="$1" split="$2" gpu="$3"
  local selector_ckpt="${ADAPTED_CKPTS[$mode]}"
  local out="${OUTPUT_ROOT}/${mode}/${split}"
  mkdir -p "${out}"
  local args=(
    "${ROOT_DIR}/test_ray_pairwise_selector.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${STAGE1_CKPT}"
    --selector_checkpoint "${selector_ckpt}"
    --output_dir "${out}"
    --split "${split}"
    --camera "${CAMERA}"
    --sample_interval "${SAMPLE_INTERVAL}"
    --max_samples "${MAX_SAMPLES}"
    --num_workers "${NUM_WORKERS}"
    --pose_depth_mode "${POSE_DEPTH_MODE}"
    "--offsets_mm=${OFFSETS_MM}"
    --query_eval_num "${QUERY_EVAL_NUM}"
    --query_eval_mode "${QUERY_EVAL_MODE}"
    --fc_mode "${FC_MODE}"
    --verify_n "${VERIFY_N}"
    --noop_check_samples "${NOOP_CHECK_SAMPLES}"
    --noop_atol "${NOOP_ATOL}"
  )
  [[ "${PROFILE_TIMING}" == "1" ]] && args+=(--profile_timing)
  [[ "${SAVE_CANDIDATE_ROWS}" == "1" ]] && args+=(--save_candidate_rows)
  [[ -n "${SELECTOR_THRESHOLD}" ]] && args+=(--selector_threshold "${SELECTOR_THRESHOLD}")

  echo "[ABLATE-HELDOUT][LAUNCH] mode=${mode} split=${split} gpu=${gpu} checkpoint=${selector_ckpt}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" \
    >"${out}/diagnostic.log" 2>&1 &
  PIDS+=("$!")
  ACTIVE+=("${mode}/${split}")
}

slot=0
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  for raw_split in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw_split}" | xargs)"
    [[ -n "${split}" ]] || continue
    launch_job "${mode}" "${split}" "${GPU_ARRAY[$slot]}"
    slot=$((slot + 1))
    if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
      wait_wave
      slot=0
    fi
  done
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

# Compact comparison table for immediate inspection.  This reads only final
# summary.json files and does not alter any evaluation result.
OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" FEATURE_MODES_ENV="${FEATURE_MODES}" SPLITS_ENV="${SPLITS}" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT_ENV"])
modes = [x.strip() for x in os.environ["FEATURE_MODES_ENV"].split(",") if x.strip()]
splits = [x.strip() for x in os.environ["SPLITS_ENV"].split(",") if x.strip()]
rows = []
for mode in modes:
    for split in splits:
        path = root / mode / split / "summary.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        a = d["aggregate"]
        native_u = float(a["native"]["utility"])
        learned_u = float(a["learned"]["utility"])
        oracle_u = float(a["oracle"]["utility"])
        headroom = oracle_u - native_u
        gain = learned_u - native_u
        native_s = float(a["native"]["success08"])
        learned_s = float(a["learned"]["success08"])
        rows.append({
            "feature_mode": mode,
            "split": split,
            "threshold": d.get("threshold"),
            "native_utility": native_u,
            "learned_utility": learned_u,
            "utility_gain": gain,
            "oracle_utility": oracle_u,
            "utility_headroom_recovery": gain / headroom if abs(headroom) > 1e-12 else float("nan"),
            "native_success08": native_s,
            "learned_success08": learned_s,
            "success08_gain": learned_s - native_s,
            "learned_rescue08": d.get("learned_rescue08"),
            "learned_harm08": d.get("learned_harm08"),
            "learned_matches_oracle": d.get("learned_matches_oracle"),
        })
if rows:
    with (root / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (root / "comparison.json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    print("[ABLATE-HELDOUT] comparison:")
    for r in rows:
        print(
            f"  {r['feature_mode']:18s} {r['split']:12s} "
            f"dU={r['utility_gain']:+.5f} "
            f"dS08={100*r['success08_gain']:+.2f}pp "
            f"rescue={100*float(r['learned_rescue08']):.2f}% "
            f"harm={100*float(r['learned_harm08']):.2f}% "
            f"headroom={100*r['utility_headroom_recovery']:.1f}%"
        )
else:
    raise SystemExit("No summary.json files found after held-out diagnostics.")
PY

echo "[ABLATE-HELDOUT] completed: ${OUTPUT_ROOT}"
echo "  comparison: ${OUTPUT_ROOT}/comparison.csv"
