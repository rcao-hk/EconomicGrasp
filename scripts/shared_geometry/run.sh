#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|prepare|bind|smoke|export|index|train|diagnose|infer|eval|summarize}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_shared_geometry*.py' -v
  python -m compileall -q shared_geometry tests/test_shared_geometry.py
  bash -n scripts/shared_geometry/run.sh
  exit
fi
: "${RUN_ROOT:?Use a NEW isolated output directory}"
mkdir -p "$RUN_ROOT"
PLAN="$RUN_ROOT/plan.json"
CONFIG="${SOURCE_CONFIG:-$RUN_ROOT/source.json}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-h2}"
PROVIDER="${PROVIDER:-interactive_grasp.native:NativeRuntime}"
# Inherit source TF32/config, but new learned geometry/readout modules use FP32.
args=(--config "$CONFIG" --plan "$PLAN" --dataset-root "${DATASET_ROOT:-}" \
      --gntrans-root "${GNTRANS_RGB_ROOT:-}" --provider "$PROVIDER" --device "$DEVICE" --seed 0)
case "$STAGE" in
prepare)
  python -m shared_geometry.cli prepare --output "$PLAN"
  ;;
bind)
  : "${BASE_CHECKPOINT:?}" "${MSE_CHECKPOINT:?}"
  python -m shared_geometry.cli bind --template "${SOURCE_TEMPLATE:-configs/interactive_stage1/source_template.json}" \
    --base-checkpoint "$BASE_CHECKPOINT" --selector-checkpoint "$MSE_CHECKPOINT" --output "$CONFIG"
  ;;
smoke)
  : "${TARGET_CONFIG:?Explicit clean rendered visible depth mapping}" "${DATASET_ROOT:?}"
  for domain in original gntrans; do
    python -m shared_geometry.cli export "${args[@]}" --targets "$TARGET_CONFIG" \
      --partition train --domain "$domain" --families 2 --field-points 256 --limit 1 \
      --output "$RUN_ROOT/smoke/$domain"
  done
  ;;
export)
  : "${TARGET_CONFIG:?}" "${DATASET_ROOT:?}"
  for part in train validation; do
    families=16; [[ "$part" != validation ]] || families=32
    python -m shared_geometry.cli export "${args[@]}" --targets "$TARGET_CONFIG" \
      --partition "$part" --families "$families" --field-points 4096 \
      --shards "${SHARDS:-1}" --shard "${SHARD:-0}" --resume --output "$RUN_ROOT/cache/$part"
  done
  ;;
index)
  for part in train validation; do
    python -m shared_geometry.cli index --plan "$PLAN" --cache "$RUN_ROOT/cache/$part" \
      --partition "$part" --output "$RUN_ROOT/${part}_index.json"
  done
  ;;
train)
  # Explicit MODE, or MODES='h1 h2 h2_stale local predicted'. No implicit grid search.
  for mode in ${MODES:-$MODE}; do
    python -m shared_geometry.cli train --train-index "$RUN_ROOT/train_index.json" \
      --val-index "$RUN_ROOT/validation_index.json" --mode "$mode" --hidden 64 \
      --epochs 20 --accumulate 4 --seed 0 --geometry-weight 1 --device "$DEVICE" \
      --output "$RUN_ROOT/models/$mode"
  done
  ;;
diagnose)
  python -m shared_geometry.cli diagnose --index "$RUN_ROOT/validation_index.json" \
    --weight "$RUN_ROOT/models/$MODE/best.pt" --device "$DEVICE" --output "$RUN_ROOT/diagnose_$MODE.json"
  ;;
paired)
  python -m shared_geometry.cli paired --index "$RUN_ROOT/validation_index.json" \
    --weight "$RUN_ROOT/models/$MODE/best.pt" --device "$DEVICE" --output "$RUN_ROOT/paired_$MODE.json"
  ;;
infer)
  : "${REFERENCE_ROOT:?Original frozen-MSE full pre-NMS dumps required}" "${DATASET_ROOT:?}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m shared_geometry.cli infer "${args[@]}" --weight "$RUN_ROOT/models/$MODE/best.pt" \
      --reference-root "$REFERENCE_ROOT" --split "$split" --output "$RUN_ROOT/test/$MODE/$split"
  done
  ;;
eval)
  : "${DATASET_ROOT:?}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m shared_geometry.cli eval --inference "$RUN_ROOT/test/$MODE/$split" \
      --dataset-root "$DATASET_ROOT" --workers "${EVAL_WORKERS:-8}"
  done
  ;;
summarize)
  python -m shared_geometry.cli summarize \
    --split-dir "test_seen=$RUN_ROOT/test/$MODE/test_seen" \
    --split-dir "test_similar=$RUN_ROOT/test/$MODE/test_similar" \
    --split-dir "test_novel=$RUN_ROOT/test/$MODE/test_novel" \
    --output "$RUN_ROOT/results_$MODE.json"
  ;;
*) echo "Unknown stage: $STAGE" >&2; exit 2;;
esac
