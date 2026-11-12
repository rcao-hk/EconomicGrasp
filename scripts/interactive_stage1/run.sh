#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Use check|prepare|bind|smoke|export|index|train|infer|eval|summarize}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_interactive_stage1*.py' -v
  python -m compileall -q interactive_grasp
  bash -n scripts/interactive_stage1/run.sh
  exit 0
fi
: "${RUN_ROOT:?Use a new isolated run directory}"
mkdir -p "$RUN_ROOT"
PLAN="$RUN_ROOT/plan.json";CFG="$RUN_ROOT/source.json"
DEVICE="${DEVICE:-cuda:0}"
case "$STAGE" in
prepare)
  python -m interactive_grasp.cli plan --output "$PLAN" ;;
bind)
  : "${BASE_CHECKPOINT:?}" "${MSE_CHECKPOINT:?}"
  python -m interactive_grasp.cli bind --template configs/interactive_stage1/source_template.json \
    --base-checkpoint "$BASE_CHECKPOINT" --selector-checkpoint "$MSE_CHECKPOINT" --output "$CFG" ;;
smoke|export)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  extra=(); [[ -z "${REUSE_TRAIN_INDEX:-}" ]] || extra+=(--reuse-index "$REUSE_TRAIN_INDEX")
  [[ -z "${REUSE_VAL_INDEX:-}" ]] || extra+=(--reuse-index "$REUSE_VAL_INDEX")
  if [[ "$STAGE" == smoke ]]; then
    for domain in original gntrans; do
      python -m interactive_grasp.cli export --config "$CFG" --plan "$PLAN" \
        --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
        --partition train --domain "$domain" --limit 1 --labels 2 \
        --output "$RUN_ROOT/smoke/$domain" --device "$DEVICE" "${extra[@]}"
    done
  else
    for part in train validation; do
      labels=32; [[ "$part" != validation ]] || labels=128
      python -m interactive_grasp.cli export --config "$CFG" --plan "$PLAN" \
        --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
        --partition "$part" --labels "$labels" --output "$RUN_ROOT/cache/$part" \
        --device "$DEVICE" --shard "${SHARD:-0}" --shards "${SHARDS:-1}" --resume "${extra[@]}"
    done
  fi ;;
index)
  for part in train validation; do
    python -m interactive_grasp.cli index --plan "$PLAN" --partition "$part" \
      --cache "$RUN_ROOT/cache/$part" --output "$RUN_ROOT/${part}_index.json"
  done ;;
train)
  for v in ${VARIANTS:-A B C D}; do
    python -m interactive_grasp.cli train --variant "$v" --train-index "$RUN_ROOT/train_index.json" \
      --val-index "$RUN_ROOT/validation_index.json" --output "$RUN_ROOT/models/$v" \
      --epochs 20 --accumulate 4 --seed 0 --device "$DEVICE"
  done ;;
infer)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}" "${REFERENCE_ROOT:?Original frozen MSE pre-NMS dump root}"
  weights=();for v in A B C D; do weights+=(--weight "$v=$RUN_ROOT/models/$v/last.pt");done
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m interactive_grasp.cli infer --config "$CFG" --plan "$PLAN" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" --device "$DEVICE" \
      --split "$split" --domain original --reference-root "$REFERENCE_ROOT" \
      --output "$RUN_ROOT/test/$split" "${weights[@]}"
  done ;;
eval)
  : "${DATASET_ROOT:?}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m interactive_grasp.cli eval --dataset-root "$DATASET_ROOT" \
      --inference "$RUN_ROOT/test/$split" --workers "${EVAL_WORKERS:-16}"
  done ;;
summarize)
  args=();for s in test_seen test_similar test_novel; do args+=(--split-dir "$s=$RUN_ROOT/test/$s");done
  python -m interactive_grasp.cli summarize "${args[@]}" --output "$RUN_ROOT/results.json" \
    --markdown "$RUN_ROOT/RESULTS_CN_STAGE1.md" ;;
*) echo "Unknown stage: $STAGE" >&2;exit 2;;
esac
