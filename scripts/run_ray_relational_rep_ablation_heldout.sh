#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
STAGE1_CKPT=${STAGE1_CKPT:-/data2/robotarm/result/grasp/rgbgrasp/log/economicgrasp_dpt_cva_cdf_distill_stage1/epoch_15_train_0.6009606198008898_val_1.1028128399874995.tar}
WORK_ROOT=${WORK_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/ray_pairwise_selector}
ABLATION_ROOT=${ABLATION_ROOT:-${WORK_ROOT}/relational_rep_ablation}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/relational_rep_ablation_heldout}

REPRESENTATION_MODES=${REPRESENTATION_MODES:-G0_current_full,G1_no_abs_raw,G2_residual_only,G3_residual_profile,G4_mean_profile}
SPLITS=${SPLITS:-test_similar,test_novel}
GPUS=${GPUS:-0,1,2,3}

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
MOVE_THRESHOLD=${MOVE_THRESHOLD:-}

export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "Stage-1 checkpoint not found: ${STAGE1_CKPT}" >&2; exit 2; }
[[ -d "${ABLATION_ROOT}" ]] || { echo "Ablation root not found: ${ABLATION_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a MODE_ARRAY <<< "${REPRESENTATION_MODES}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }

PIDS=()
ACTIVE=()
wait_wave() {
  local failed=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "[REP-ABL-TEST][ERROR] ${ACTIVE[$i]} failed; inspect diagnostic.log" >&2
      failed=1
    fi
  done
  PIDS=(); ACTIVE=()
  [[ ${failed} -eq 0 ]] || exit 1
}

launch_job() {
  local mode="$1" split="$2" gpu="$3"
  local ckpt="${ABLATION_ROOT}/${mode}/checkpoint_best.tar"
  local out="${OUTPUT_ROOT}/${mode}/${split}"
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint: ${ckpt}" >&2; exit 2; }
  mkdir -p "${out}"
  local args=(
    "${ROOT_DIR}/test_ray_relational_rep_ablation.py"
    --dataset_root "${DATASET_ROOT}"
    --checkpoint_path "${STAGE1_CKPT}"
    --selector_checkpoint "${ckpt}"
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
  [[ -n "${MOVE_THRESHOLD}" ]] && args+=(--move_threshold "${MOVE_THRESHOLD}")

  echo "[REP-ABL-TEST][LAUNCH] mode=${mode} split=${split} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" >"${out}/diagnostic.log" 2>&1 &
  PIDS+=("$!"); ACTIVE+=("${mode}/${split}")
}

slot=0
for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  [[ -n "${mode}" ]] || continue
  for raw_split in "${SPLIT_ARRAY[@]}"; do
    split="$(echo "${raw_split}" | xargs)"
    [[ -n "${split}" ]] || continue
    launch_job "${mode}" "${split}" "${GPU_ARRAY[$slot]}"
    slot=$((slot+1))
    if [[ ${slot} -ge ${#GPU_ARRAY[@]} ]]; then
      wait_wave; slot=0
    fi
  done
done
[[ ${#PIDS[@]} -eq 0 ]] || wait_wave

OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" REPRESENTATION_MODES_ENV="${REPRESENTATION_MODES}" SPLITS_ENV="${SPLITS}" "${PYTHON_BIN}" - <<'PY'
import csv, json, os
from pathlib import Path
root=Path(os.environ["OUTPUT_ROOT_ENV"])
modes=[x.strip() for x in os.environ["REPRESENTATION_MODES_ENV"].split(",") if x.strip()]
splits=[x.strip() for x in os.environ["SPLITS_ENV"].split(",") if x.strip()]
rows=[]
for mode in modes:
    for split in splits:
        p=root/mode/split/"summary.json"
        if not p.is_file():
            continue
        d=json.loads(p.read_text())
        a=d["aggregate"]
        nu=float(a["native"]["utility"]); lu=float(a["learned"]["utility"]); ou=float(a["oracle"]["utility"])
        ns=float(a["native"]["success08"]); ls=float(a["learned"]["success08"])
        head=ou-nu; gain=lu-nu
        rows.append({
            "representation_mode":mode,
            "split":split,
            "threshold":d.get("move_threshold"),
            "utility_gain":gain,
            "success08_gain":ls-ns,
            "rescue08":d.get("learned_rescue08"),
            "harm08":d.get("learned_harm08"),
            "move_rate":d.get("move_rate"),
            "headroom_recovery":gain/head if abs(head)>1e-12 else float("nan"),
        })
if not rows:
    raise SystemExit("No summary.json files found.")
with (root/"comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with (root/"comparison.json").open("w") as f:
    json.dump(rows,f,indent=2,sort_keys=True)
print("[REP-ABL-TEST] comparison:")
for r in rows:
    print(f"  {r['representation_mode']:20s} {r['split']:12s} "
          f"dU={r['utility_gain']:+.5f} dS08={100*r['success08_gain']:+.2f}pp "
          f"rescue={100*float(r['rescue08']):.2f}% harm={100*float(r['harm08']):.2f}% "
          f"move={100*float(r['move_rate']):.2f}% headroom={100*r['headroom_recovery']:.2f}%")
PY

echo "[REP-ABL-TEST] completed: ${OUTPUT_ROOT}"
echo "  comparison: ${OUTPUT_ROOT}/comparison.csv"
