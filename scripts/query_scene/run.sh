#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?check|prepare|bind|smoke|export|index|train|diagnose|benchmark|infer|eval|summarize}"
if [[ "$STAGE" == check ]]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_query_scene*.py' -v
  python -m compileall -q query_scene
  bash -n scripts/query_scene/run.sh
  exit 0
fi
: "${RUN_ROOT:?Use a new isolated experiment root}"
PLAN="$RUN_ROOT/plan.json"; CONFIG="$RUN_ROOT/source.json"; DEVICE="${DEVICE:-cuda:0}"
PROVIDER="${PROVIDER:-interactive_grasp.native:NativeRuntime}"
mkdir -p "$RUN_ROOT"
weights=()
for v in local candidate scene; do weights+=(--weight "$v=$RUN_ROOT/models/$v/last.pt"); done
case "$STAGE" in
prepare) python -m query_scene.cli prepare --output "$PLAN" ;;
bind)
  : "${BASE_CHECKPOINT:?}" "${MSE_CHECKPOINT:?}"
  python -m query_scene.cli bind --template configs/interactive_stage1/source_template.json \
    --base-checkpoint "$BASE_CHECKPOINT" --selector-checkpoint "$MSE_CHECKPOINT" --output "$CONFIG" ;;
smoke|export)
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  reuse=(); [[ -z "${REUSE_TRAIN_INDEX:-}" ]] || reuse+=(--reuse-index "$REUSE_TRAIN_INDEX")
  [[ -z "${REUSE_VAL_INDEX:-}" ]] || reuse+=(--reuse-index "$REUSE_VAL_INDEX")
  if [[ "$STAGE" == smoke ]]; then
    for domain in original gntrans; do
      python -m query_scene.cli export --config "$CONFIG" --plan "$PLAN" \
        --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" --provider "$PROVIDER" \
        --partition train --domain "$domain" --labels 2 --limit 1 --device "$DEVICE" \
        --output "$RUN_ROOT/smoke/$domain" "${reuse[@]}"
    done
  else
    for part in train validation; do
      labels=32; [[ "$part" != validation ]] || labels=1024
      python -m query_scene.cli export --config "$CONFIG" --plan "$PLAN" \
        --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" --provider "$PROVIDER" \
        --partition "$part" --labels "$labels" --device "$DEVICE" --output "$RUN_ROOT/cache/$part" \
        --shards "${SHARDS:-1}" --shard "${SHARD:-0}" --resume "${reuse[@]}"
    done
  fi ;;
index)
  for part in train validation; do
    python -m query_scene.cli index --plan "$PLAN" --partition "$part" --cache "$RUN_ROOT/cache/$part" --output "$RUN_ROOT/${part}_index.json"
  done ;;
train)
  for v in ${VARIANTS:-local candidate scene}; do
    python -m query_scene.cli train --train-index "$RUN_ROOT/train_index.json" --val-index "$RUN_ROOT/validation_index.json" \
      --variant "$v" --output "$RUN_ROOT/models/$v" --device "$DEVICE" --epochs 20 --accumulate 4 \
      --hidden 128 --heads 4 --layers 2 --residual-scale 0.2 --seed 0
  done ;;
diagnose)
  python -m query_scene.cli diagnose --index "$RUN_ROOT/validation_index.json" "${weights[@]}" \
    --device "$DEVICE" --output "$RUN_ROOT/evidence_diagnostics.json" ;;
benchmark)
  python -m query_scene.cli benchmark --index "$RUN_ROOT/validation_index.json" "${weights[@]}" \
    --device "$DEVICE" --output "$RUN_ROOT/benchmark.json" ;;
infer)
  : "${DATASET_ROOT:?}" "${REFERENCE_ROOT:?Verified frozen MSE pre-NMS dumps}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m query_scene.cli infer --config "$CONFIG" --plan "$PLAN" --dataset-root "$DATASET_ROOT" \
      --gntrans-root "${GNTRANS_RGB_ROOT:-}" --provider "$PROVIDER" --device "$DEVICE" \
      --reference-root "$REFERENCE_ROOT" --split "$split" "${weights[@]}" --output "$RUN_ROOT/test/$split"
  done ;;
eval)
  : "${DATASET_ROOT:?}"
  for split in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m query_scene.cli eval --dataset-root "$DATASET_ROOT" --inference "$RUN_ROOT/test/$split" --workers "${WORKERS:-16}"
  done ;;
summarize)
  python -m query_scene.cli summarize --split-dir "test_seen=$RUN_ROOT/test/test_seen" \
    --split-dir "test_similar=$RUN_ROOT/test/test_similar" --split-dir "test_novel=$RUN_ROOT/test/test_novel" \
    --output "$RUN_ROOT/results.json" --markdown "$RUN_ROOT/RESULTS_CN_QUERY_SCENE.md" ;;
*) echo "Unknown stage $STAGE" >&2;exit 2 ;;
esac
