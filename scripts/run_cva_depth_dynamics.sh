#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: run_cva_depth_dynamics.sh --mode audit|pair --run-id ID --checkpoint FILE
       --gpus 0[,1] [--max-steps 50] [--seed 0] [--batch-size 1]
       [--python /path/python] [--dataset-root DIR] [--log-root DIR]
       [--experiment-root DIR] [--resume] [--audit-batches 8]
       [--probe-interval 100] [--audit-interval 200]

audit: one GPU, real-model P0/P1 and diagnostic noninterference check.
pair:  two distinct GPUs, D0=none and D1=all, requires this run's P0 gate.
Resume extends both arms from their recorded latest complete snapshots to the
same TOTAL update budget. Review 50 -> 500 -> 2000; extensions are explicit.
The Stage-1 pose/fused-depth settings are inherited from checkpoint metadata.
test_seen is the user-designated validation source; the contract records this.
HELP
}

mode= run_id= checkpoint= gpus= resume=0
max_steps=50 seed=0 batch=1 audit_batches=8 probe_interval=100 audit_interval=200
python_bin="${PYTHON:-python}"
dataset_root="${GRASPNET_ROOT:-/data/robotarm/dataset/graspnet}"
log_root="${DEPTH_DYNAMICS_LOG_ROOT:-/data/robotarm/result/grasp/rgbgrasp/log}"
experiment_root="${DEPTH_DYNAMICS_EXPERIMENT_ROOT:-/data/robotarm/result/grasp/rgbgrasp/experiment}"
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0;;
    --mode) mode="$2"; shift 2;;
    --run-id) run_id="$2"; shift 2;;
    --checkpoint) checkpoint="$2"; shift 2;;
    --gpus) gpus="$2"; shift 2;;
    --max-steps) max_steps="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --batch-size) batch="$2"; shift 2;;
    --python) python_bin="$2"; shift 2;;
    --dataset-root) dataset_root="$2"; shift 2;;
    --log-root) log_root="$2"; shift 2;;
    --experiment-root) experiment_root="$2"; shift 2;;
    --audit-batches) audit_batches="$2"; shift 2;;
    --probe-interval) probe_interval="$2"; shift 2;;
    --audit-interval) audit_interval="$2"; shift 2;;
    --resume) resume=1; shift;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$mode" == audit || "$mode" == pair ]] || { usage >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Explicit numeric GPU IDs required' >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<< "$gpus"
if [[ "$mode" == pair ]]; then
  [[ ${#gpu_ids[@]} == 2 && "${gpu_ids[0]}" != "${gpu_ids[1]}" ]] || {
    echo 'Pair mode needs two distinct GPU IDs' >&2; exit 2;
  }
else
  [[ ${#gpu_ids[@]} == 1 && "$resume" == 0 ]] || { echo 'Audit needs one GPU and a new run' >&2; exit 2; }
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
log_dir="${log_root}/cva_depth_dynamics/${run_id}"
exp_dir="${experiment_root}/cva_depth_dynamics/${run_id}"
mkdir -p "$log_dir" "$exp_dir/diagnostics"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
common=(--init_checkpoint "$checkpoint" --dataset_root "$dataset_root" --seed "$seed"
        --batch_size "$batch" --learning_rate 0.0001 --lr_schedule constant
        --weight_decay 0 --depth_weight_decay 0 --camera realsense --graspness_mode scene
        --kview_mode A1 --audit_batches "$audit_batches" --probe_interval "$probe_interval"
        --audit_interval "$audit_interval" --train_probe_frames 16 --heldout_test_probe_frames 32)
if [[ "$mode" == audit ]]; then
  [[ ! -e "$log_dir/P0/contract.json" && ! -e "$exp_dir/diagnostics/P0/contract.json" ]] || {
    echo 'Audit run already exists; select a new run ID' >&2; exit 2;
  }
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" train_cva_depth_dynamics.py \
    --mode audit --arm P0 --routes none --output "$log_dir/P0" \
    --diagnostics_dir "$exp_dir/diagnostics/P0" --verify_diagnostic_step --save_gradient_maps "${common[@]}"
  exit
fi
gate="$exp_dir/diagnostics/P0/p0_gate.json"
[[ -f "$gate" ]] || { echo "P0 gate missing: $gate" >&2; exit 2; }
arms=(D0 D1)
routes=(none all)
# Resolve BOTH arms before launching either, so a bad resume cannot leave an
# accidental unpaired job running.
resume_paths=()
for arm in "${arms[@]}"; do
  if [[ "$resume" == 1 ]]; then
    resume_paths+=("$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$log_dir/$arm/latest_checkpoint.json")")
    [[ -f "${resume_paths[-1]}" ]] || { echo "Resume snapshot missing for $arm" >&2; exit 2; }
  else
    [[ ! -e "$log_dir/$arm" ]] || { echo "Arm already exists: $arm; use --resume" >&2; exit 2; }
    resume_paths+=("")
  fi
done
pids=()
for index in 0 1; do
  extra=()
  [[ -z "${resume_paths[$index]}" ]] || extra+=(--resume_checkpoint "${resume_paths[$index]}")
  CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "$python_bin" train_cva_depth_dynamics.py \
    --mode train --arm "${arms[$index]}" --routes "${routes[$index]}" \
    --output "$log_dir/${arms[$index]}" --diagnostics_dir "$exp_dir/diagnostics/${arms[$index]}" \
    --max_steps "$max_steps" --skip_initial_audit --audit_gate "$gate" \
    "${common[@]}" "${extra[@]}" > "$exp_dir/${arms[$index]}_to_${max_steps}.console.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
"$python_bin" summarize_cva_depth_dynamics.py --input "$exp_dir/diagnostics" --output "$exp_dir/diagnostics/summary"
exit "$status"
