#!/usr/bin/env bash
# Isolated P0/P1/P2 experiments; never change main or production inference.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|p0-smoke|p0-capture|p0-index|p0-fit|p1-fit|infer|eval|p2}"
: "${RUN_ROOT:?Set new isolated RUN_ROOT (not MIX50 base/cache output)}"
mkdir -p "$RUN_ROOT"
DEVICE="${DEVICE:-cuda:0}"
SPLITS="${SPLITS:-test_seen test_similar test_novel}"

need_config() {
  : "${BASE_CONFIG:?Set bound MIX50 GVAR configuration}"
  : "${TRAIN_INDEX:?Set existing RARE MIX50 train_index.json}"
}
need_data() {
  : "${DATASET_ROOT:?Set Original GraspNet path}"
  : "${GNTRANS_RGB_ROOT:?Set GNTrans RGB path}"
}
need_features() {
  need_config
  [[ -f "$RUN_ROOT/feature_index.json" ]] || { echo "Run p0-index first" >&2; exit 2; }
}
need_new_file() {
  [[ ! -e "$1" ]] || { echo "Refusing overwrite: $1" >&2; exit 2; }
}

case "$STAGE" in
check)
  python -m unittest discover -s tests -p 'test_action_decision.py' -v
  python -m unittest discover -s tests -p 'test_candidate_conflict.py' -v
  python -m compileall -q diagnostics/action_decision tests/test_action_decision.py
  bash -n scripts/action_decision/run.sh
  ;;
p0-smoke)
  need_config;need_data
  # Two genuine train-domain scenes: 0 for scorer fit and 90 for held-out validation.
  # Each domain contributes one frame from each scene, no CAD relabeling.
  python -m diagnostics.action_decision.feature_cache capture \
    --config "$BASE_CONFIG" --index "$TRAIN_INDEX" \
    --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
    --scene-ids 0,90 --max-per-scene 1 --smoke \
    --output "$RUN_ROOT/smoke/features" --device "$DEVICE"
  python -m diagnostics.action_decision.feature_cache index \
    --index "$TRAIN_INDEX" --cache "$RUN_ROOT/smoke/features" \
    --output "$RUN_ROOT/smoke/feature_index.json" --smoke
  python -m diagnostics.action_decision.run_quality fit \
    --index "$TRAIN_INDEX" --feature-index "$RUN_ROOT/smoke/feature_index.json" \
    --feature-mode reader256 --objective decision \
    --output "$RUN_ROOT/smoke/head" --epochs 1 --batch-frames 2 \
    --device "$DEVICE" --smoke
  ;;
p0-capture)
  need_config;need_data
  python -m diagnostics.action_decision.feature_cache capture \
    --config "$BASE_CONFIG" --index "$TRAIN_INDEX" \
    --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
    --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}" \
    --dtype "${CACHE_DTYPE:-float16}" \
    --output "$RUN_ROOT/features" --device "$DEVICE"
  ;;
p0-index)
  need_config
  need_new_file "$RUN_ROOT/feature_index.json"
  python -m diagnostics.action_decision.feature_cache index \
    --index "$TRAIN_INDEX" --cache "$RUN_ROOT/features" \
    --output "$RUN_ROOT/feature_index.json"
  ;;
p0-fit)
  need_features
  for mode in base_geometry score64 reader256 reader_delta score_reader; do
    out="$RUN_ROOT/p0/$mode"
    [[ ! -e "$out" ]] || { echo "Refusing overwrite: $out" >&2; exit 2; }
    python -m diagnostics.action_decision.run_quality fit \
      --index "$TRAIN_INDEX" --feature-index "$RUN_ROOT/feature_index.json" \
      --feature-mode "$mode" --objective mse --output "$out" \
      --epochs "${EPOCHS:-20}" --batch-frames 4 --hidden 64 \
      --seed "${SEED:-0}" --device "$DEVICE"
  done
  ;;
p1-fit)
  need_features
  # Feature mode must be selected on held-out TRAIN-domain scenes, never test AP.
  FEATURE_MODE="${FEATURE_MODE:-score_reader}"
  for objective in mse rare decision; do
    out="$RUN_ROOT/p1/$FEATURE_MODE/$objective/seed${SEED:-0}"
    [[ ! -e "$out" ]] || { echo "Refusing overwrite: $out" >&2; exit 2; }
    python -m diagnostics.action_decision.run_quality fit \
      --index "$TRAIN_INDEX" --feature-index "$RUN_ROOT/feature_index.json" \
      --feature-mode "$FEATURE_MODE" --objective "$objective" --output "$out" \
      --epochs "${EPOCHS:-20}" --batch-frames 4 --hidden 64 \
      --pair-budget 2048 --top-k 16 --pair-weight .5 \
      --seed "${SEED:-0}" --device "$DEVICE"
  done
  ;;
infer)
  need_config;need_data
  : "${HEAD:?Point to P0/P1 selected last.pt}"
  : "${HEAD_TAG:?Set unique HEAD_TAG matching HEAD identity}"
  : "${PROTOCOL:?Set existing RARE MIX50 10%-test manifest}"
  DOMAIN="${DOMAIN:-original}"
  for split in $SPLITS; do
    out="$RUN_ROOT/inference/$HEAD_TAG/$DOMAIN/$split"
    [[ ! -e "$out" ]] || { echo "Refusing overwrite: $out" >&2; exit 2; }
    python -m diagnostics.action_decision.run_quality infer \
      --config "$BASE_CONFIG" --protocol "$PROTOCOL" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "$GNTRANS_RGB_ROOT" \
      --head "$HEAD" --domain "$DOMAIN" --split "$split" \
      --policies "${POLICIES:-both,depth_only,rank_only}" \
      --seed 0 --device "$DEVICE" --output "$out"
  done
  ;;
eval)
  need_data
  : "${HEAD_TAG:?}" 
  DOMAIN="${DOMAIN:-original}"
  for split in $SPLITS; do
    python -m rare_grasp.pipeline eval --dataset-root "$DATASET_ROOT" \
      --inference "$RUN_ROOT/inference/$HEAD_TAG/$DOMAIN/$split" \
      --workers "${EVAL_WORKERS:-16}"
  done
  ;;
p2)
  : "${P2_GRAPH_ROOT:?Set accepted graph / split /graph dirs}"
  : "${P2_LABELS_ROOT:?Set accepted graph / split /labels dirs}"
  for split in $SPLITS; do
    out="$RUN_ROOT/p2/$split/interval${SAMPLE_INTERVAL:-10}"
    [[ ! -e "$out" ]] || { echo "Refusing overwrite: $out" >&2; exit 2; }
    python -m diagnostics.action_decision.global_oracle \
      --graph-root "$P2_GRAPH_ROOT/$split/graph" \
      --labels-root "$P2_LABELS_ROOT/$split/labels" \
      --output "$out" --sample-interval "${SAMPLE_INTERVAL:-10}" \
      --max-candidates "${P2_CANDIDATE_CAP:-0}" \
      --time-limit "${P2_TIME_LIMIT:-5}" --workers "${P2_WORKERS:-1}" \
      --budget 50 --per-object 10 --bootstrap 3000
  done
  ;;
*)
  echo "Unknown stage $STAGE" >&2;exit 2
  ;;
esac
