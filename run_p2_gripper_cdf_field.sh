#!/usr/bin/env bash
# End-to-end P2 scratch representation study:
# exact-action cache -> field enrichment -> 4 capacity-matched 3-layer MLPs -> AP.
set -Eeuo pipefail

readonly COLLISION_THRESH="0.01"
readonly COLLISION_VOXEL_SIZE="0.01"
readonly P2_VARIANTS="p2_0,p2_a,p2_b,p2_c"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
EXACT_ACTION_CACHE_DIR="${EXACT_ACTION_CACHE_DIR:-${P1_CACHE_DIR:-}}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/p2_gripper_cdf_runs/scratch_mlp}"
CAMERA="${CAMERA:-realsense}"
PHASES="${PHASES:-enrich,validate,train,infer,eval}"
RESUME_PIPELINE="${RESUME_PIPELINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

P2_CACHE_DIR="${P2_CACHE_DIR:-${WORK_ROOT}/cache}"
TRAIN_ROOT="${TRAIN_ROOT:-${WORK_ROOT}/train}"
PREDICTION_ROOT="${PREDICTION_ROOT:-${WORK_ROOT}/predictions}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${WORK_ROOT}/evaluation}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

EXPECTED_POSE_DEPTH_MODE="${EXPECTED_POSE_DEPTH_MODE:-global_film}"
EXPECTED_USE_FUSE_DEPTH="${EXPECTED_USE_FUSE_DEPTH:-1}"
GRASPNESS_MODE="${GRASPNESS_MODE:-scene}"
NUM_POINT="${NUM_POINT:-20000}"
M_POINT="${M_POINT:-1024}"
NUM_VIEW="${NUM_VIEW:-300}"
NUM_ANGLE="${NUM_ANGLE:-12}"
NUM_DEPTH="${NUM_DEPTH:-4}"
GRASP_MAX_WIDTH="${GRASP_MAX_WIDTH:-0.10}"
MIN_DEPTH="${MIN_DEPTH:-0.2}"
MAX_DEPTH="${MAX_DEPTH:-1.0}"
BIN_NUM="${BIN_NUM:-256}"

ENRICH_GPUS="${ENRICH_GPUS:-0,1,2,3,4,5}"
ENRICH_SCENE_IDS="${ENRICH_SCENE_IDS:-}"
ENRICH_DATA_WORKERS="${ENRICH_DATA_WORKERS:-2}"
ENRICH_ACTION_CHUNK="${ENRICH_ACTION_CHUNK:-2048}"
ENRICH_STORE_DTYPE="${ENRICH_STORE_DTYPE:-float32}"
ENRICH_OVERWRITE="${ENRICH_OVERWRITE:-0}"
ENRICH_STRICT="${ENRICH_STRICT:-1}"
ENRICH_SEED="${ENRICH_SEED:-0}"
RESIDUAL_TAU_M="${RESIDUAL_TAU_M:-0.02}"
SURFACE_TAU_M="${SURFACE_TAU_M:-0.01}"
REQUIRE_ALL_SCENES="${REQUIRE_ALL_SCENES:-1}"
MIN_FRAMES_PER_SCENE="${MIN_FRAMES_PER_SCENE:-26}"

VAL_SCENE_START="${VAL_SCENE_START:-90}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_MIN_LR="${TRAIN_MIN_LR:-1e-6}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.0}"
TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-256}"
TRAIN_AMP="${TRAIN_AMP:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-5}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

INFER_GPUS="${INFER_GPUS:-0,1,2,3,4,5}"
INFER_DATA_WORKERS="${INFER_DATA_WORKERS:-2}"
INFER_ROW_CHUNK="${INFER_ROW_CHUNK:-512}"
TEST_SAMPLE_INTERVAL="${TEST_SAMPLE_INTERVAL:-10}"
INFER_SEED="${INFER_SEED:-0}"
FORCE_INFER="${FORCE_INFER:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

mkdir -p "$P2_CACHE_DIR" "$TRAIN_ROOT" "$PREDICTION_ROOT" "$EVAL_OUTPUT_DIR" "$LOG_DIR"

log() { printf '[P2-SCRATCH][%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }
phase_enabled() { case ",${PHASES}," in *",$1,"*) return 0;; *) return 1;; esac; }
normalize_csv() {
    local value="$1"; value="${value// /,}"
    while [[ "$value" == *",,"* ]]; do value="${value//,,/,}"; done
    value="${value#,}"; value="${value%,}"; printf '%s' "$value"
}
split_csv_to_array() {
    local value; value="$(normalize_csv "$1")"; local -n dst="$2"; dst=()
    [[ -z "$value" ]] || IFS=',' read -r -a dst <<< "$value"
}
run_logged() {
    local logfile="$1"; shift; mkdir -p "$(dirname "$logfile")"; log "RUN: $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@" | tee "$logfile"; printf '\n' | tee -a "$logfile"; return 0
    fi
    "$@" 2>&1 | tee "$logfile"
}

[[ -d "$REPO_ROOT" ]] || die "REPO_ROOT missing: $REPO_ROOT"
[[ -n "$DATASET_ROOT" ]] || die "Set DATASET_ROOT"
[[ -n "$BASE_CHECKPOINT" ]] || die "Set BASE_CHECKPOINT"
[[ -n "$EXACT_ACTION_CACHE_DIR" ]] || die "Set EXACT_ACTION_CACHE_DIR (P1_CACHE_DIR is accepted as a deprecated alias)"
if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$DATASET_ROOT" ]] || die "DATASET_ROOT missing: $DATASET_ROOT"
    [[ -f "$BASE_CHECKPOINT" ]] || die "BASE_CHECKPOINT missing: $BASE_CHECKPOINT"
    [[ -d "$EXACT_ACTION_CACHE_DIR" ]] || die "EXACT_ACTION_CACHE_DIR missing: $EXACT_ACTION_CACHE_DIR"
fi
[[ "$EXPECTED_USE_FUSE_DEPTH" == "0" || "$EXPECTED_USE_FUSE_DEPTH" == "1" ]] || die "EXPECTED_USE_FUSE_DEPTH must be 0/1"
[[ "$RESUME_PIPELINE" == "0" || "$RESUME_PIPELINE" == "1" ]] || die "RESUME_PIPELINE must be 0/1"
if ! [[ "$TEST_SAMPLE_INTERVAL" =~ ^[0-9]+$ ]] || (( TEST_SAMPLE_INTERVAL <= 0 )); then
    die "TEST_SAMPLE_INTERVAL must be a positive integer"
fi
TEST_SAMPLE_FRACTION="$($PYTHON_BIN - "$TEST_SAMPLE_INTERVAL" <<'PY'
import sys
k=int(sys.argv[1]); print("1.0" if k==1 else repr(1.0/k))
PY
)"

COMMON_MODEL_ARGS=(
    --dataset_root "$DATASET_ROOT" --camera "$CAMERA"
    --num_point "$NUM_POINT" --m_point "$M_POINT"
    --num_view "$NUM_VIEW" --num_angle "$NUM_ANGLE" --num_depth "$NUM_DEPTH"
    --grasp_max_width "$GRASP_MAX_WIDTH"
    --min_depth "$MIN_DEPTH" --max_depth "$MAX_DEPTH" --bin_num "$BIN_NUM"
    --graspness_mode "$GRASPNESS_MODE"
    --pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE"
    --multi_modal --use_cdf
)
[[ "$EXPECTED_USE_FUSE_DEPTH" == "1" ]] && COMMON_MODEL_ARGS+=(--use_fuse_depth)

log "Protocol: Base + {P2-0,P2-A,P2-B,P2-C}; common 3-layer MLP from scratch"
log "No P1 checkpoint, no residual; CDF BCE only; Top-1; collision=0.01"
log "Phases=$PHASES WORK_ROOT=$WORK_ROOT"

if phase_enabled enrich; then
    split_csv_to_array "$ENRICH_GPUS" GPU_ARRAY
    ((${#GPU_ARRAY[@]} > 0)) || die "ENRICH_GPUS empty"
    SCENES=()
    if [[ -n "$(normalize_csv "$ENRICH_SCENE_IDS")" ]]; then
        split_csv_to_array "$ENRICH_SCENE_IDS" SCENES
    else
        mapfile -t SCENES < <("$PYTHON_BIN" - "$EXACT_ACTION_CACHE_DIR" <<'PY'
import glob, os, re, sys
root=sys.argv[1]
ids=[]
for path in glob.glob(os.path.join(root,'scene_*')):
    m=re.search(r'scene_(\d+)$', path)
    if m: ids.append(int(m.group(1)))
for value in sorted(set(ids)): print(value)
PY
)
    fi
    ((${#SCENES[@]} > 0)) || die "No exact-action cache scenes found"

    declare -a SHARDS
    for ((i=0;i<${#GPU_ARRAY[@]};i++)); do SHARDS[$i]=""; done
    for ((i=0;i<${#SCENES[@]};i++)); do
        slot=$((i % ${#GPU_ARRAY[@]}))
        [[ -z "${SHARDS[$slot]}" ]] && SHARDS[$slot]="${SCENES[$i]}" || SHARDS[$slot]+=",${SCENES[$i]}"
    done
    PIDS=()
    for ((slot=0;slot<${#GPU_ARRAY[@]};slot++)); do
        gpu="${GPU_ARRAY[$slot]}"; scene_ids="${SHARDS[$slot]}"; [[ -n "$scene_ids" ]] || continue
        tag="gpu${slot}_cuda${gpu}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            run_logged "${LOG_DIR}/enrich_${tag}.log" \
                "$PYTHON_BIN" "$REPO_ROOT/mine_cva_p2_gripper_field_cache.py" \
                --p2_source_cache_dir "$EXACT_ACTION_CACHE_DIR" \
                --p2_cache_dir "$P2_CACHE_DIR" \
                --p2_reference_base_checkpoint "$BASE_CHECKPOINT" \
                --p2_scene_ids "$scene_ids" \
                --p2_action_chunk "$ENRICH_ACTION_CHUNK" \
                --p2_store_dtype "$ENRICH_STORE_DTYPE" \
                --p2_overwrite "$ENRICH_OVERWRITE" \
                --p2_strict "$ENRICH_STRICT" \
                --p2_seed "$ENRICH_SEED" \
                --p2_worker_tag "$tag" \
                --p2_expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                --p2_expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
                --p2_residual_tau_m "$RESIDUAL_TAU_M" \
                --p2_surface_tau_m "$SURFACE_TAU_M" \
                --checkpoint_path "$BASE_CHECKPOINT" \
                --batch_size 1 --num_workers "$ENRICH_DATA_WORKERS" \
                "${COMMON_MODEL_ARGS[@]}"
        ) &
        pid="$!"; PIDS+=("$pid")
        log "Started enrichment pid=$pid gpu=$gpu scenes=$scene_ids"
    done
    failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || die "At least one P2 enrichment worker failed"
fi

if phase_enabled validate; then
    run_logged "${LOG_DIR}/validate_cache.log" \
        "$PYTHON_BIN" "$REPO_ROOT/validate_cva_p2_gripper_field_cache.py" \
        --cache_dir "$P2_CACHE_DIR" \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
        --expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
        --residual_tau_m "$RESIDUAL_TAU_M" \
        --surface_tau_m "$SURFACE_TAU_M" \
        --max_grasp_width_m "$GRASP_MAX_WIDTH" \
        --min_metric_depth_m "$MIN_DEPTH" \
        --max_metric_depth_m "$MAX_DEPTH" \
        --strict 1 --require_all_scenes "$REQUIRE_ALL_SCENES" \
        --min_frames_per_scene "$MIN_FRAMES_PER_SCENE" \
        --output_json "${P2_CACHE_DIR}/cache_inventory.json"
fi

if phase_enabled train; then
    split_csv_to_array "$TRAIN_GPUS" TRAIN_GPU_ARRAY
    split_csv_to_array "$P2_VARIANTS" VARIANT_ARRAY
    ((${#TRAIN_GPU_ARRAY[@]} > 0)) || die "TRAIN_GPUS empty"
    PIDS=()
    for ((slot=0;slot<${#TRAIN_GPU_ARRAY[@]};slot++)); do
        gpu="${TRAIN_GPU_ARRAY[$slot]}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            for ((i=slot;i<${#VARIANT_ARRAY[@]};i+=${#TRAIN_GPU_ARRAY[@]})); do
                variant="${VARIANT_ARRAY[$i]}"
                output="${TRAIN_ROOT}/${variant}"
                best="${output}/checkpoint_best_p2_scratch.tar"
                if [[ "$RESUME_PIPELINE" == "1" && "$FORCE_TRAIN" != "1" && -f "$best" && -f "${output}/best.json" ]]; then
                    log "Skip completed training: $variant"
                    continue
                fi
                [[ "$FORCE_TRAIN" != "1" ]] || rm -rf "$output"
                mkdir -p "$output"
                run_logged "${LOG_DIR}/train_${variant}.log" \
                    "$PYTHON_BIN" "$REPO_ROOT/train_cva_p2_cdf_field.py" \
                    --variant "$variant" \
                    --cache_dir "$P2_CACHE_DIR" \
                    --output_dir "$output" \
                    --base_checkpoint "$BASE_CHECKPOINT" \
                    --expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                    --expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
                    --val_scene_start "$VAL_SCENE_START" \
                    --require_all_scenes "$REQUIRE_ALL_SCENES" \
                    --min_frames_per_scene "$MIN_FRAMES_PER_SCENE" \
                    --epochs "$TRAIN_EPOCHS" \
                    --batch_size "$TRAIN_BATCH_SIZE" \
                    --num_workers "$TRAIN_NUM_WORKERS" \
                    --learning_rate "$TRAIN_LR" \
                    --min_learning_rate "$TRAIN_MIN_LR" \
                    --weight_decay "$TRAIN_WEIGHT_DECAY" \
                    --hidden_dim "$TRAIN_HIDDEN_DIM" \
                    --amp "$TRAIN_AMP" \
                    --seed "$TRAIN_SEED" \
                    --save_interval "$TRAIN_SAVE_INTERVAL" \
                    --device cuda:0
            done
        ) &
        pid="$!"; PIDS+=("$pid")
        log "Started training worker pid=$pid gpu=$gpu"
    done
    failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || die "At least one P2 training job failed"
fi

if phase_enabled infer; then
    split_csv_to_array "$INFER_GPUS" INFER_GPU_ARRAY
    ((${#INFER_GPU_ARRAY[@]} > 0)) || die "INFER_GPUS empty"
    MODES=(base p2_0 p2_a p2_b p2_c)
    TASK_MODES=(); TASK_SPLITS=()
    for mode in "${MODES[@]}"; do
        for split in test_seen test_similar test_novel; do
            TASK_MODES+=("$mode"); TASK_SPLITS+=("$split")
        done
    done
    PIDS=()
    for ((slot=0;slot<${#INFER_GPU_ARRAY[@]};slot++)); do
        gpu="${INFER_GPU_ARRAY[$slot]}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            for ((task=slot;task<${#TASK_MODES[@]};task+=${#INFER_GPU_ARRAY[@]})); do
                mode="${TASK_MODES[$task]}"; split="${TASK_SPLITS[$task]}"
                save_dir="${PREDICTION_ROOT}/${mode}/${split}"
                manifest="${save_dir}/_inference_complete.json"
                if [[ "$FORCE_INFER" == "1" ]]; then
                    rm -rf "$save_dir"
                elif [[ "$RESUME_PIPELINE" == "1" && -f "$manifest" ]]; then
                    log "Skip completed inference $mode/$split"
                    continue
                fi
                mkdir -p "$save_dir"
                predictor_args=()
                if [[ "$mode" != "base" ]]; then
                    predictor="${TRAIN_ROOT}/${mode}/checkpoint_best_p2_scratch.tar"
                    [[ "$DRY_RUN" == "1" || -f "$predictor" ]] || die "Missing P2 checkpoint: $predictor"
                    predictor_args=(--p2_predictor_checkpoint "$predictor")
                fi
                run_logged "${LOG_DIR}/infer_${mode}_${split}.log" \
                    "$PYTHON_BIN" "$REPO_ROOT/inference_cva_p2_cdf_field.py" \
                    --p2_mode "$mode" \
                    --p2_reference_base_checkpoint "$BASE_CHECKPOINT" \
                    "${predictor_args[@]}" \
                    --p2_expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                    --p2_expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
                    --p2_row_chunk "$INFER_ROW_CHUNK" \
                    --checkpoint_path "$BASE_CHECKPOINT" \
                    --save_dir "$save_dir" --test_mode "$split" \
                    --batch_size 1 --num_workers "$INFER_DATA_WORKERS" \
                    --sample_interval "$TEST_SAMPLE_FRACTION" \
                    --collision_thresh "$COLLISION_THRESH" \
                    --collision_voxel_size "$COLLISION_VOXEL_SIZE" \
                    --seed "$INFER_SEED" \
                    "${COMMON_MODEL_ARGS[@]}"
            done
        ) &
        pid="$!"; PIDS+=("$pid")
        log "Started inference worker pid=$pid gpu=$gpu"
    done
    failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || die "At least one P2 inference worker failed"
fi

if phase_enabled eval; then
    run_logged "${LOG_DIR}/official_eval.log" \
        "$PYTHON_BIN" "$REPO_ROOT/eval_p2_gripper_cdf_field.py" \
        --dataset_root "$DATASET_ROOT" \
        --prediction_root "$PREDICTION_ROOT" \
        --train_root "$TRAIN_ROOT" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --camera "$CAMERA" \
        --sample_interval "$TEST_SAMPLE_INTERVAL" \
        --num_workers "$EVAL_NUM_WORKERS" \
        --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
        --bootstrap_seed "$TRAIN_SEED"
fi

log "P2 scratch-MLP pipeline complete"
