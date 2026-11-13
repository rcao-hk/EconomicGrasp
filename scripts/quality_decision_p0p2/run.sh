#!/usr/bin/env bash
# P0-P2 independent research stages. Run one stage per invocation.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?Usage: run.sh check|p0-fit|p1-mine|p1-index|p1-fit|infer|eval|p2}"
: "${RUN_ROOT:?Set a NEW run directory under the data volume}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$RUN_ROOT"
case "$STAGE" in
check)
  python -m unittest discover -s tests -p 'test_quality_decision_p0p2.py' -v
  python -m compileall -q diagnostics/quality_decision tests/test_quality_decision_p0p2.py
  bash -n scripts/quality_decision_p0p2/run.sh
  ;;
p0-fit)
  : "${RARE_TRAIN_INDEX:?Point to existing full MIX50 RARE train_index.json}"
  for MODE in score score_geometry score_feature score_feature_geometry score_feature_permuted; do
    python -m diagnostics.quality_decision.fit \
      --index "$RARE_TRAIN_INDEX" --strategy uniform \
      --mode "$MODE" --output "$RUN_ROOT/p0/$MODE" \
      --device "$DEVICE" --seed 20261113 --epochs 10 --frames-per-scene 4 \
      --train-scene-start 0 --train-scene-end 80 \
      --val-scene-start 80 --val-scene-end 90
  done
  ;;
p1-mine)
  : "${BOUND_CONFIG:?Strict MIX50 GVAR bound.json required}"
  : "${DATASET_ROOT:?}" "${GNTRANS_RGB_ROOT:?}"
  python -m diagnostics.quality_decision.mine \
    --config "$BOUND_CONFIG" --dataset-root "$DATASET_ROOT" \
    --gntrans-root "$GNTRANS_RGB_ROOT" --output "$RUN_ROOT/p1/cache" \
    --device "$DEVICE" --seed 20261113 --frames-per-scene 4 \
    --scene-start 0 --scene-end 100 \
    --shard-count "${SHARD_COUNT:-1}" --shard-index "${SHARD_INDEX:-0}" \
    ${RESUME:+--resume}
  ;;
p1-index)
  args=()
  for f in "$RUN_ROOT"/p1/cache/index_shard*_of*.json; do
    [[ -f "$f" ]] || { echo "No P1 cache shards: $f" >&2; exit 2; }
    args+=(--shard-index "$f")
  done
  python -m diagnostics.quality_decision.index "${args[@]}" \
    --min-scenes 100 --output "$RUN_ROOT/p1/cache/index_merged.json"
  ;;
p1-fit)
  index="$RUN_ROOT/p1/cache/index_merged.json"
  [[ -f "$index" ]] || { echo "Run p1-index first" >&2; exit 2; }
  for STRATEGY in uniform decision; do
    for LOSS in mse rare; do
      pair=0
      [[ "$LOSS" != rare ]] || pair=0.5
      python -m diagnostics.quality_decision.fit \
        --index "$index" --strategy "$STRATEGY" \
        --mode score_feature_geometry --pair-weight "$pair" \
        --output "$RUN_ROOT/p1/heads/${STRATEGY}_${LOSS}" \
        --device "$DEVICE" --seed 20261113 --epochs 10 --frames-per-scene 4 \
        --train-scene-start 0 --train-scene-end 80 \
        --val-scene-start 80 --val-scene-end 90
    done
  done
  ;;
infer)
  : "${BOUND_CONFIG:?}" "${DATASET_ROOT:?}" "${PROBE_WEIGHTS:?}"
  LABEL="${LABEL:-probe}"
  INTERVAL="${INTERVAL:-10}"
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m diagnostics.quality_decision.infer infer \
      --config "$BOUND_CONFIG" --weights "$PROBE_WEIGHTS" \
      --dataset-root "$DATASET_ROOT" --gntrans-root "${GNTRANS_RGB_ROOT:-}" \
      --split "$SPLIT" --sample-interval "$INTERVAL" \
      --policies both,depth_only,rank_only --device "$DEVICE" \
      --output "$RUN_ROOT/eval/$LABEL/$SPLIT"
  done
  ;;
eval)
  : "${DATASET_ROOT:?}"
  LABEL="${LABEL:-probe}"
  for SPLIT in ${SPLITS:-test_seen test_similar test_novel}; do
    python -m diagnostics.quality_decision.infer eval \
      --inference "$RUN_ROOT/eval/$LABEL/$SPLIT" \
      --dataset-root "$DATASET_ROOT" --workers "${EVAL_WORKERS:-16}"
  done
  ;;
p2)
  : "${GRAPH_ROOT:?Graph root for one full GraspNet split}"
  : "${LABELS_ROOT:?Exact-action label root for the same split}"
  : "${P2_REFERENCE_FRAME_AP:?Accepted prior analysis frame_ap.npz required}"
  METHOD="${METHOD:-MM}"
  MODE="${P2_SOLVER:-off}"
  python -m diagnostics.quality_decision.global_budget \
    --graph-root "$GRAPH_ROOT" --labels-root "$LABELS_ROOT" \
    --method "$METHOD" --solver "$MODE" \
    --reference-frame-ap "$P2_REFERENCE_FRAME_AP" \
    --max-solver-nodes "${MAX_SOLVER_NODES:-96}" \
    --solver-seconds "${SOLVER_SECONDS:-10}" \
    --frames-per-scene "${FRAMES_PER_SCENE:-2}" \
    --output "$RUN_ROOT/p2/${SPLIT_NAME:-split}/$METHOD/$MODE"
  ;;
*)
  echo "Unknown stage: $STAGE" >&2
  exit 2
  ;;
esac
