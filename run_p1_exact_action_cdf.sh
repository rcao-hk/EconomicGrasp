#!/usr/bin/env bash
# End-to-end P1 exact-action CDF-only learnability experiment.
#
# Phases:
#   mine -> validate -> train -> infer -> eval
#
# The protocol is intentionally strict:
#   * corrected Stage-1 RGB student only;
#   * deterministic image-FPS and Top-1 view;
#   * exact physical actions labeled by clean CAD/DexNet evaluator;
#   * only existing decoder.cdf_head weight/bias are trainable;
#   * ordinary unbalanced threshold-wise CDF BCE is the only loss;
#   * inference collision_thresh and collision_voxel_size are fixed to 0.01.
set -Eeuo pipefail

readonly COLLISION_THRESH="0.01"
readonly COLLISION_VOXEL_SIZE="0.01"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/p1_exact_action_cdf_runs/default}"
CAMERA="${CAMERA:-realsense}"
PHASES="${PHASES:-mine,validate,train,infer,eval}"
RESUME_PIPELINE="${RESUME_PIPELINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

CACHE_DIR="${CACHE_DIR:-${WORK_ROOT}/cache}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TRAIN_DIR="${TRAIN_DIR:-${WORK_ROOT}/train_seed_${TRAIN_SEED}}"
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
MIN_DEPTH="${MIN_DEPTH:-0.2}"
MAX_DEPTH="${MAX_DEPTH:-1.0}"
BIN_NUM="${BIN_NUM:-256}"

MINE_GPUS="${MINE_GPUS:-0,1,2,3,4,5}"
MINE_SCENE_IDS="${MINE_SCENE_IDS:-}"
MINE_ANNO_IDS="${MINE_ANNO_IDS:-}"
MINE_SAMPLE_FRACTION="${MINE_SAMPLE_FRACTION:-0.1}"
MINE_TOP_CENTERS="${MINE_TOP_CENTERS:-16}"
MINE_RANDOM_CENTERS="${MINE_RANDOM_CENTERS:-4}"
MINE_DATA_WORKERS="${MINE_DATA_WORKERS:-2}"
MINE_EVAL_WORKERS_PER_GPU="${MINE_EVAL_WORKERS_PER_GPU:-2}"
MINE_MAX_PENDING_PER_GPU="${MINE_MAX_PENDING_PER_GPU:-4}"
MINE_COLLISION_CHUNK="${MINE_COLLISION_CHUNK:-512}"
MINE_EVAL_THREADS="${MINE_EVAL_THREADS:-1}"
MINE_FC_MODE="${MINE_FC_MODE:-reuse_contacts}"
MINE_FC_VERIFY_N="${MINE_FC_VERIFY_N:-0}"
MINE_CDF_INCREMENT_BIAS="${MINE_CDF_INCREMENT_BIAS:--4.0}"
MINE_COMPRESS="${MINE_COMPRESS:-0}"
MINE_OVERWRITE="${MINE_OVERWRITE:-0}"
MINE_STRICT="${MINE_STRICT:-1}"
MINE_SEED="${MINE_SEED:-0}"
REQUIRE_ALL_SCENES="${REQUIRE_ALL_SCENES:-1}"
MINE_MIN_FRAMES_PER_SCENE="${MINE_MIN_FRAMES_PER_SCENE:-26}"

VAL_SCENE_START="${VAL_SCENE_START:-90}"
TRAIN_GPU="${TRAIN_GPU:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_MIN_LR="${TRAIN_MIN_LR:-1e-6}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.0}"
TRAIN_AMP="${TRAIN_AMP:-0}"
TRAIN_SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-5}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

INFER_GPUS="${INFER_GPUS:-0,1,2,3,4,5}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-2}"
TEST_SAMPLE_INTERVAL="${TEST_SAMPLE_INTERVAL:-10}"
INFER_SEED="${INFER_SEED:-0}"
FORCE_INFER="${FORCE_INFER:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-10}"

mkdir -p "$CACHE_DIR" "$TRAIN_DIR" "$PREDICTION_ROOT" "$EVAL_OUTPUT_DIR" "$LOG_DIR"

log() {
    printf '[P1][%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

phase_enabled() {
    local needle="$1"
    case ",${PHASES}," in
        *",${needle},"*) return 0 ;;
        *) return 1 ;;
    esac
}

run_logged() {
    local logfile="$1"
    shift
    mkdir -p "$(dirname "$logfile")"
    log "RUN: $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@" | tee "$logfile"
        printf '\n' | tee -a "$logfile"
        return 0
    fi
    "$@" 2>&1 | tee "$logfile"
}

normalize_csv() {
    local value="$1"
    value="${value// /,}"
    while [[ "$value" == *",,"* ]]; do value="${value//,,/,}"; done
    value="${value#,}"
    value="${value%,}"
    printf '%s' "$value"
}

split_csv_to_array() {
    local value
    value="$(normalize_csv "$1")"
    local -n destination="$2"
    destination=()
    if [[ -n "$value" ]]; then
        IFS=',' read -r -a destination <<< "$value"
    fi
}

if [[ "$EXPECTED_USE_FUSE_DEPTH" != "0" && "$EXPECTED_USE_FUSE_DEPTH" != "1" ]]; then
    die "EXPECTED_USE_FUSE_DEPTH must be 0 or 1"
fi
if [[ "$RESUME_PIPELINE" != "0" && "$RESUME_PIPELINE" != "1" ]]; then
    die "RESUME_PIPELINE must be 0 or 1"
fi
[[ -d "$REPO_ROOT" ]] || die "REPO_ROOT does not exist: $REPO_ROOT"
[[ -n "$DATASET_ROOT" ]] || die "Set DATASET_ROOT"
[[ -n "$BASE_CHECKPOINT" ]] || die "Set BASE_CHECKPOINT"
[[ -d "$DATASET_ROOT" ]] || die "DATASET_ROOT does not exist: $DATASET_ROOT"
[[ -f "$BASE_CHECKPOINT" ]] || die "BASE_CHECKPOINT does not exist: $BASE_CHECKPOINT"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN not found: $PYTHON_BIN"

if ! [[ "$TEST_SAMPLE_INTERVAL" =~ ^[0-9]+$ ]] || (( TEST_SAMPLE_INTERVAL <= 0 )); then
    die "TEST_SAMPLE_INTERVAL must be a positive integer"
fi
TEST_SAMPLE_FRACTION="$($PYTHON_BIN - "$TEST_SAMPLE_INTERVAL" <<'PY'
import sys
k = int(sys.argv[1])
print("1.0" if k == 1 else repr(1.0 / k))
PY
)"

COMMON_MODEL_ARGS=(
    --dataset_root "$DATASET_ROOT"
    --camera "$CAMERA"
    --num_point "$NUM_POINT"
    --m_point "$M_POINT"
    --num_view "$NUM_VIEW"
    --num_angle "$NUM_ANGLE"
    --num_depth "$NUM_DEPTH"
    --min_depth "$MIN_DEPTH"
    --max_depth "$MAX_DEPTH"
    --bin_num "$BIN_NUM"
    --graspness_mode "$GRASPNESS_MODE"
    --pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE"
    --multi_modal
    --use_cdf
)
if [[ "$EXPECTED_USE_FUSE_DEPTH" == "1" ]]; then
    COMMON_MODEL_ARGS+=(--use_fuse_depth)
fi

log "Protocol: CDF-BCE-only, Top-1, collision_thresh=${COLLISION_THRESH}, collision_voxel=${COLLISION_VOXEL_SIZE}"
log "Phases: $PHASES"
log "Work root: $WORK_ROOT"

if phase_enabled mine; then
    log "Phase mine: exact Student-action cache"
    split_csv_to_array "$MINE_GPUS" MINE_GPU_ARRAY
    ((${#MINE_GPU_ARRAY[@]} > 0)) || die "MINE_GPUS is empty"

    SCENE_ARRAY=()
    if [[ -n "$(normalize_csv "$MINE_SCENE_IDS")" ]]; then
        split_csv_to_array "$MINE_SCENE_IDS" SCENE_ARRAY
    else
        for ((scene=0; scene<100; scene++)); do SCENE_ARRAY+=("$scene"); done
    fi
    ((${#SCENE_ARRAY[@]} > 0)) || die "No mining scenes selected"

    declare -a SHARD_IDS
    for ((slot=0; slot<${#MINE_GPU_ARRAY[@]}; slot++)); do SHARD_IDS[$slot]=""; done
    for ((index=0; index<${#SCENE_ARRAY[@]}; index++)); do
        slot=$((index % ${#MINE_GPU_ARRAY[@]}))
        if [[ -n "${SHARD_IDS[$slot]}" ]]; then
            SHARD_IDS[$slot]+=",${SCENE_ARRAY[$index]}"
        else
            SHARD_IDS[$slot]="${SCENE_ARRAY[$index]}"
        fi
    done

    MINE_PIDS=()
    for ((slot=0; slot<${#MINE_GPU_ARRAY[@]}; slot++)); do
        gpu="${MINE_GPU_ARRAY[$slot]}"
        scene_ids="${SHARD_IDS[$slot]}"
        [[ -n "$scene_ids" ]] || continue
        worker_tag="gpu${slot}_cuda${gpu}"
        logfile="${LOG_DIR}/mine_${worker_tag}.log"
        anno_args=()
        if [[ -n "$(normalize_csv "$MINE_ANNO_IDS")" ]]; then
            anno_args=(--ea_anno_ids "$(normalize_csv "$MINE_ANNO_IDS")")
        fi
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            run_logged "$logfile" "$PYTHON_BIN" "$REPO_ROOT/mine_cva_exact_action_cdf_cache.py" \
                --ea_cache_dir "$CACHE_DIR" \
                --ea_dataset_split train \
                --ea_scene_ids "$scene_ids" \
                "${anno_args[@]}" \
                --ea_sample_interval "$MINE_SAMPLE_FRACTION" \
                --ea_top_centers "$MINE_TOP_CENTERS" \
                --ea_random_centers "$MINE_RANDOM_CENTERS" \
                --ea_eval_workers "$MINE_EVAL_WORKERS_PER_GPU" \
                --ea_max_pending "$MINE_MAX_PENDING_PER_GPU" \
                --ea_collision_chunk "$MINE_COLLISION_CHUNK" \
                --ea_eval_threads "$MINE_EVAL_THREADS" \
                --ea_fc_mode "$MINE_FC_MODE" \
                --ea_fc_verify_n "$MINE_FC_VERIFY_N" \
                --ea_cdf_increment_bias "$MINE_CDF_INCREMENT_BIAS" \
                --ea_compress "$MINE_COMPRESS" \
                --ea_overwrite "$MINE_OVERWRITE" \
                --ea_strict "$MINE_STRICT" \
                --ea_seed "$MINE_SEED" \
                --ea_worker_tag "$worker_tag" \
                --ea_expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                --checkpoint_path "$BASE_CHECKPOINT" \
                --batch_size 1 \
                --num_workers "$MINE_DATA_WORKERS" \
                "${COMMON_MODEL_ARGS[@]}"
        ) &
        miner_pid="$!"
        MINE_PIDS+=("$miner_pid")
        log "Started miner pid=${miner_pid} gpu=${gpu} scenes=${scene_ids}"
    done

    mine_failed=0
    for pid in "${MINE_PIDS[@]}"; do
        if ! wait "$pid"; then
            mine_failed=1
            log "Miner failed: pid=$pid"
        fi
    done
    (( mine_failed == 0 )) || die "At least one cache miner failed"
fi

if phase_enabled validate; then
    log "Phase validate: strict cache contract"
    run_logged "${LOG_DIR}/validate_cache.log" \
        "$PYTHON_BIN" "$REPO_ROOT/validate_cva_exact_action_cdf_cache.py" \
        --cache_dir "$CACHE_DIR" \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
        --expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
        --strict 1 \
        --require_all_scenes "$REQUIRE_ALL_SCENES" \
        --min_frames_per_scene "$MINE_MIN_FRAMES_PER_SCENE" \
        --output_json "${CACHE_DIR}/cache_inventory.json"
fi

BEST_EXACT_CHECKPOINT="${TRAIN_DIR}/checkpoint_best_exact_action.tar"
if phase_enabled train; then
    log "Phase train: existing CDF head only"
    if [[ "$RESUME_PIPELINE" == "1" && "$FORCE_TRAIN" != "1" && -f "$BEST_EXACT_CHECKPOINT" && -f "${TRAIN_DIR}/best.json" ]]; then
        log "Skip completed training: $BEST_EXACT_CHECKPOINT"
    else
        if [[ "$FORCE_TRAIN" == "1" ]]; then
            rm -f "${TRAIN_DIR}/metrics.jsonl" "${TRAIN_DIR}/best.json" \
                "${TRAIN_DIR}/head_checkpoint_best.tar" \
                "${TRAIN_DIR}/checkpoint_best_exact_action.tar"
        fi
        (
            export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            run_logged "${LOG_DIR}/train_seed_${TRAIN_SEED}.log" \
                "$PYTHON_BIN" "$REPO_ROOT/train_cva_exact_action_cdf_head.py" \
                --cache_dir "$CACHE_DIR" \
                --output_dir "$TRAIN_DIR" \
                --base_checkpoint "$BASE_CHECKPOINT" \
                --expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                --expected_use_fuse_depth "$EXPECTED_USE_FUSE_DEPTH" \
                --val_scene_start "$VAL_SCENE_START" \
                --require_all_scenes "$REQUIRE_ALL_SCENES" \
                --min_frames_per_scene "$MINE_MIN_FRAMES_PER_SCENE" \
                --epochs "$TRAIN_EPOCHS" \
                --batch_size "$TRAIN_BATCH_SIZE" \
                --num_workers "$TRAIN_NUM_WORKERS" \
                --learning_rate "$TRAIN_LR" \
                --min_learning_rate "$TRAIN_MIN_LR" \
                --weight_decay "$TRAIN_WEIGHT_DECAY" \
                --amp "$TRAIN_AMP" \
                --seed "$TRAIN_SEED" \
                --save_interval "$TRAIN_SAVE_INTERVAL" \
                --device "$TRAIN_DEVICE"
        )
    fi
fi

if phase_enabled infer; then
    if [[ "$DRY_RUN" != "1" ]]; then
        [[ -f "$BEST_EXACT_CHECKPOINT" ]] || die "Missing exact checkpoint: $BEST_EXACT_CHECKPOINT"
    else
        log "DRY_RUN: assuming exact checkpoint will be produced at $BEST_EXACT_CHECKPOINT"
    fi
    log "Phase infer: paired Base/Exact, sample fraction=${TEST_SAMPLE_FRACTION}"
    split_csv_to_array "$INFER_GPUS" INFER_GPU_ARRAY
    ((${#INFER_GPU_ARRAY[@]} > 0)) || die "INFER_GPUS is empty"

    TASK_MODES=(base base base exact exact exact)
    TASK_SPLITS=(test_seen test_similar test_novel test_seen test_similar test_novel)
    INFER_PIDS=()
    for ((slot=0; slot<${#INFER_GPU_ARRAY[@]}; slot++)); do
        gpu="${INFER_GPU_ARRAY[$slot]}"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            for ((task_index=slot; task_index<${#TASK_MODES[@]}; task_index+=${#INFER_GPU_ARRAY[@]})); do
                mode="${TASK_MODES[$task_index]}"
                split="${TASK_SPLITS[$task_index]}"
                if [[ "$mode" == "base" ]]; then
                    checkpoint="$BASE_CHECKPOINT"
                else
                    checkpoint="$BEST_EXACT_CHECKPOINT"
                fi
                save_dir="${PREDICTION_ROOT}/${mode}/${split}"
                manifest="${save_dir}/_inference_complete.json"
                if [[ "$FORCE_INFER" == "1" ]]; then
                    rm -rf "$save_dir"
                elif [[ "$RESUME_PIPELINE" == "1" && -f "$manifest" ]]; then
                    "$PYTHON_BIN" - "$manifest" "$mode" "$split" "$CAMERA" "$TEST_SAMPLE_FRACTION" "$COLLISION_THRESH" "$COLLISION_VOXEL_SIZE" <<'PY'
import json, math, sys
path, mode, split, camera, fraction, collision, voxel = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
checks = {
    "status": (str(payload.get("status")), "complete"),
    "mode": (str(payload.get("mode")), mode),
    "test_mode": (str(payload.get("test_mode")), split),
    "camera": (str(payload.get("camera")), camera),
}
for key, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{path}: {key}={actual!r}, expected={expected!r}")
for key, expected in (("sample_interval", float(fraction)), ("collision_thresh", float(collision)), ("collision_voxel_size", float(voxel))):
    actual = float(payload.get(key, float("nan")))
    if not math.isfinite(actual) or abs(actual - expected) > 1e-9:
        raise SystemExit(f"{path}: {key}={actual!r}, expected={expected!r}")
PY
                    log "Skip completed inference: ${mode}/${split}"
                    continue
                fi
                mkdir -p "$save_dir"
                run_logged "${LOG_DIR}/infer_${mode}_${split}.log" \
                    "$PYTHON_BIN" "$REPO_ROOT/inference_cva_exact_action_cdf.py" \
                    --exact_action_mode "$mode" \
                    --exact_action_expected_pose_depth_mode "$EXPECTED_POSE_DEPTH_MODE" \
                    --exact_action_reference_base_checkpoint "$BASE_CHECKPOINT" \
                    --checkpoint_path "$checkpoint" \
                    --save_dir "$save_dir" \
                    --test_mode "$split" \
                    --batch_size "$INFER_BATCH_SIZE" \
                    --num_workers "$INFER_NUM_WORKERS" \
                    --sample_interval "$TEST_SAMPLE_FRACTION" \
                    --collision_thresh "$COLLISION_THRESH" \
                    --collision_voxel_size "$COLLISION_VOXEL_SIZE" \
                    --seed "$INFER_SEED" \
                    "${COMMON_MODEL_ARGS[@]}"
            done
        ) &
        infer_pid="$!"
        INFER_PIDS+=("$infer_pid")
        log "Started inference worker pid=${infer_pid} gpu=${gpu}"
    done

    infer_failed=0
    for pid in "${INFER_PIDS[@]}"; do
        if ! wait "$pid"; then
            infer_failed=1
            log "Inference worker failed: pid=$pid"
        fi
    done
    (( infer_failed == 0 )) || die "At least one inference worker failed"
fi

if phase_enabled eval; then
    log "Phase eval: strict paired official AP"
    run_logged "${LOG_DIR}/official_eval.log" \
        "$PYTHON_BIN" "$REPO_ROOT/eval_p1_exact_action_cdf.py" \
        --dataset_root "$DATASET_ROOT" \
        --prediction_root "$PREDICTION_ROOT" \
        --train_dir "$TRAIN_DIR" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --camera "$CAMERA" \
        --sample_interval "$TEST_SAMPLE_INTERVAL" \
        --num_workers "$EVAL_NUM_WORKERS" \
        --expected_collision_thresh "$COLLISION_THRESH" \
        --expected_collision_voxel_size "$COLLISION_VOXEL_SIZE"
fi

log "Complete. Report: ${EVAL_OUTPUT_DIR}/P1_REPORT.md"
