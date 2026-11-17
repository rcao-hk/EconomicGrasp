#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${1:-help}"
case "$STAGE" in
  check)
    "$PYTHON" -m unittest discover -s tests -p test_camera_geometry.py -v
    "$PYTHON" -m compileall -q camera_geometry tests/test_camera_geometry.py
    ;;
  render|import-gc6d|prepare)
    : "${SPEC:?Set SPEC to an explicit data/render configuration}"
    : "${OUTPUT:?Set OUTPUT to a NEW output directory (prepare: manifest JSON path)}"
    "$PYTHON" -m camera_geometry.run "$STAGE" --spec "$SPEC" --output "$OUTPUT"
    ;;
  preflight)
    : "${CONFIG:?Set CONFIG to the experiment JSON}"
    : "${RUN_ROOT:?Set RUN_ROOT to a NEW experiment root}"
    mkdir -p "$RUN_ROOT"
    "$PYTHON" -m camera_geometry.run preflight --config "$CONFIG" --output "$RUN_ROOT/preflight.json" --device "$DEVICE"
    ;;
  train-c|train-t|train-one)
    : "${CONFIG:?Set CONFIG}"; : "${RUN_ROOT:?Set RUN_ROOT}"
    if [[ "$STAGE" == train-c ]]; then ARMS="C0 C1 C2 C3";
    elif [[ "$STAGE" == train-t ]]; then ARMS="T0 T1 T2";
    else : "${ARM:?Set ARM to C0-C3 or T0-T2}"; ARMS="$ARM"; fi
    extra=(); [[ "${RESUME:-0}" == 1 ]] && extra+=(--resume)
    [[ "${SMOKE:-0}" == 1 ]] && extra+=(--smoke)
    for arm in $ARMS; do
      "$PYTHON" -m camera_geometry.run train --config "$CONFIG" --arm "$arm" \
        --output "$RUN_ROOT/$arm" --device "$DEVICE" "${extra[@]}"
    done
    ;;
  evaluate|predict)
    : "${CHECKPOINT:?Set fixed final CHECKPOINT}"; : "${MANIFEST:?Set MANIFEST}"
    : "${PARTITION:?Set dev, holdout or exploratory}"; : "${OUTPUT:?Set NEW OUTPUT}"
    "$PYTHON" -m camera_geometry.run "$STAGE" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" \
      --partition "$PARTITION" --output "$OUTPUT" --device "$DEVICE"
    ;;
  summarize)
    : "${RUN_ROOT:?Set RUN_ROOT}"
    files=(); for arm in C0 C1 C2 C3 T0 T1 T2; do
      [[ -f "$RUN_ROOT/$arm/final_dev.json" ]] && files+=("$RUN_ROOT/$arm/final_dev.json")
    done
    [[ ${#files[@]} -gt 0 ]] || { echo 'No completed evaluations'; exit 1; }
    "$PYTHON" -m camera_geometry.run summarize --inputs "${files[@]}" --output "$RUN_ROOT/COMPARISON.json"
    ;;
  *) echo 'Stages: check | render | import-gc6d | prepare | preflight | train-c | train-t | train-one | evaluate | predict | summarize';;
esac
