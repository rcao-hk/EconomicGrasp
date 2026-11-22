#!/usr/bin/env bash
# Height MIX50 P0-A / P0-B / P0-C read-only checkpoint diagnostics.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-check}"
PYTHON="${PYTHON:-python}"
if [[ "$MODE" == "check" ]]; then
  "$PYTHON" -m unittest discover -s tests -p 'test_height_p0_abc.py' -v
  "$PYTHON" -m compileall -q diagnostics/height_p0 tools/height_p0_abc.py
  exit 0
fi

: "${RUN_ROOT:?Set a unique P0 diagnostic root}"
case "$MODE" in fit|eval|summarize|all) ;; *) echo "MODE: check|fit|eval|summarize|all" >&2; exit 2 ;; esac
if [[ "$MODE" != "summarize" ]]; then
  : "${DATASET_ROOT:?GraspNet dataset root required}"
  [[ -d "$DATASET_ROOT" ]] || { echo "Missing GraspNet root: $DATASET_ROOT" >&2; exit 2; }
fi
mkdir -p "$RUN_ROOT"
PRIOR="${PRIOR:-$RUN_ROOT/prior_train_only.json}"
OUT="${OUT:-$RUN_ROOT/evaluation}"
DEVICE="${DEVICE:-cuda:0}"
FRAMES_PER_SCENE="${FRAMES_PER_SCENE:-8}"
PRIOR_FRAMES_PER_SCENE="${PRIOR_FRAMES_PER_SCENE:-16}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
APPEARANCE="${APPEARANCE:-1}"
SPLITS="${SPLITS:-train_used train_heldout_view test_seen test_similar test_novel}"
read -r -a SELECTED_SPLITS <<< "$SPLITS"

if [[ "$MODE" == "fit" || "$MODE" == "all" ]]; then
  [[ ! -e "$PRIOR" ]] || { echo "Prior already exists; refuse to refit: $PRIOR" >&2; exit 2; }
  "$PYTHON" tools/height_p0_abc.py fit-prior \
    --dataset-root "$DATASET_ROOT" \
    --frames-per-scene "$PRIOR_FRAMES_PER_SCENE" \
    --pixels-per-frame "${PRIOR_PIXELS_PER_FRAME:-2048}" \
    --seed "${SEED:-0}" --output "$PRIOR"
fi

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  : "${CHECKPOINTS:?Set checkpoint registry JSON path}"
  [[ -f "$CHECKPOINTS" ]] || { echo "Missing checkpoint registry" >&2; exit 2; }
  [[ -f "$PRIOR" ]] || { echo "Run fit stage first; expected train-only prior: $PRIOR" >&2; exit 2; }
  args=()
  if [[ "$APPEARANCE" == "1" ]]; then
    : "${GNTRANS_RGB_ROOT:?Paired appearance needs GNTrans RGB root}"
    args+=(--gntrans-rgb-root "$GNTRANS_RGB_ROOT" --appearance)
  elif [[ "$APPEARANCE" != "0" ]]; then
    echo "APPEARANCE must be 0 or 1" >&2; exit 2
  fi
  if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
  if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max-scenes "$MAX_SCENES"); fi
  "$PYTHON" tools/height_p0_abc.py evaluate \
    --dataset-root "$DATASET_ROOT" --checkpoints "$CHECKPOINTS" \
    --prior "$PRIOR" --output-dir "$OUT" --device "$DEVICE" \
    --frames-per-scene "$FRAMES_PER_SCENE" \
    --batch-size "$BATCH_SIZE" --workers "$WORKERS" --bootstrap "$BOOTSTRAP" \
    --splits "${SELECTED_SPLITS[@]}" "${args[@]}"
fi
if [[ "$MODE" == "summarize" ]]; then
  "$PYTHON" tools/height_p0_abc.py summarize --output-dir "$OUT" --bootstrap "$BOOTSTRAP"
fi
echo "[P0 A/B/C] $MODE complete. Report: $OUT/REPORT_ZH.md"
