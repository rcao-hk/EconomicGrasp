#!/usr/bin/env bash
# Base-CVA decoder diagnostic for the two trained MGF ranking-ablation checkpoints.
#
# For each checkpoint, evaluate the SAME forward pass with:
#   1) Base CVA CDF scorer + collision OFF
#   2) Base CVA CDF scorer + collision ON
#
# Existing Field-scorer outputs (test_latest and test_latest_collision0p01) are
# never overwritten. The final summary forms:
#
#   Field/Base scorer x Collision OFF/ON
#
# for both no-ranking and ranking checkpoints.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
ABLATION_ROOT=${ABLATION_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_ranking_ablation_10pct}

NO_RANK_GPUS=${NO_RANK_GPUS:-0,1,2}
RANK_GPUS=${RANK_GPUS:-3,5,6}
NO_RANK_INFER_GPUS=${NO_RANK_INFER_GPUS:-$NO_RANK_GPUS}
RANK_INFER_GPUS=${RANK_INFER_GPUS:-$RANK_GPUS}

NO_RANK_ROOT=${NO_RANK_ROOT:-$ABLATION_ROOT/no_ranking}
RANK_ROOT=${RANK_ROOT:-$ABLATION_ROOT/ranking_w0p1}

SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
TOP4=${TOP4:-0}
RESUME=${RESUME:-0}

COLLISION_THRESH=${COLLISION_THRESH:-0.01}
COLLISION_VOXEL_SIZE=${COLLISION_VOXEL_SIZE:-0.01}
COLLISION_APPROACH_DIST=${COLLISION_APPROACH_DIST:-0.05}

CHECKPOINT_KIND=latest
PHASES=infer,eval
SCORE_SOURCE=base

NO_RANK_BASE_OFF_WORK=${NO_RANK_BASE_OFF_WORK:-$NO_RANK_ROOT/base_cva_eval_nocollision}
RANK_BASE_OFF_WORK=${RANK_BASE_OFF_WORK:-$RANK_ROOT/base_cva_eval_nocollision}
NO_RANK_BASE_ON_WORK=${NO_RANK_BASE_ON_WORK:-$NO_RANK_ROOT/base_cva_eval_collision0p01}
RANK_BASE_ON_WORK=${RANK_BASE_ON_WORK:-$RANK_ROOT/base_cva_eval_collision0p01}

NO_RANK_BASE_OFF_TEST=${NO_RANK_BASE_OFF_TEST:-$NO_RANK_ROOT/test_latest_base_cva_nocollision}
RANK_BASE_OFF_TEST=${RANK_BASE_OFF_TEST:-$RANK_ROOT/test_latest_base_cva_nocollision}
NO_RANK_BASE_ON_TEST=${NO_RANK_BASE_ON_TEST:-$NO_RANK_ROOT/test_latest_base_cva_collision0p01}
RANK_BASE_ON_TEST=${RANK_BASE_ON_TEST:-$RANK_ROOT/test_latest_base_cva_collision0p01}

SUMMARY_DIR=${SUMMARY_DIR:-$ABLATION_ROOT/comparison_base_vs_field_2x2}

IFS=',' read -r -a NO_RANK_GPU_IDS <<< "$NO_RANK_GPUS"
IFS=',' read -r -a RANK_GPU_IDS <<< "$RANK_GPUS"

if [[ "${#NO_RANK_GPU_IDS[@]}" -ne 3 ]]; then
  echo "[ERROR] NO_RANK_GPUS must contain exactly 3 GPU ids; got $NO_RANK_GPUS" >&2
  exit 2
fi
if [[ "${#RANK_GPU_IDS[@]}" -ne 3 ]]; then
  echo "[ERROR] RANK_GPUS must contain exactly 3 GPU ids; got $RANK_GPUS" >&2
  exit 2
fi
for id in "${NO_RANK_GPU_IDS[@]}" "${RANK_GPU_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Invalid GPU id: $id" >&2
    exit 2
  }
done
for no_id in "${NO_RANK_GPU_IDS[@]}"; do
  for rank_id in "${RANK_GPU_IDS[@]}"; do
    if [[ "$no_id" == "$rank_id" ]]; then
      echo "[ERROR] GPU $no_id appears in both variants; GPU sets must be disjoint." >&2
      exit 2
    fi
  done
done

[[ "$INFER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ERROR] INFER_BATCH_SIZE must be a positive integer; got $INFER_BATCH_SIZE" >&2
  exit 2
}

NO_RANK_CKPT="$NO_RANK_ROOT/train/checkpoint_latest.pt"
RANK_CKPT="$RANK_ROOT/train/checkpoint_latest.pt"
[[ -f "$NO_RANK_CKPT" ]] || {
  echo "[ERROR] Missing no-ranking checkpoint: $NO_RANK_CKPT" >&2
  exit 2
}
[[ -f "$RANK_CKPT" ]] || {
  echo "[ERROR] Missing ranking checkpoint: $RANK_CKPT" >&2
  exit 2
}

mkdir -p "$SUMMARY_DIR"

echo "============================================================"
echo "[BASE-CVA 2x2 EVAL]"
echo "  scorer                : base CVA CDF"
echo "  checkpoint            : latest / epoch 20"
echo "  splits                : $SPLITS"
echo "  inference batch/GPU   : $INFER_BATCH_SIZE"
echo "  no-ranking GPUs       : $NO_RANK_GPUS"
echo "  ranking GPUs          : $RANK_GPUS"
echo "  collision threshold   : $COLLISION_THRESH"
echo "  collision voxel (m)   : $COLLISION_VOXEL_SIZE"
echo "  collision approach (m): $COLLISION_APPROACH_DIST"
echo "============================================================"

PIDS=()
NAMES=()
LOGS=()

cleanup_wave() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_wave EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch_condition() {
  local name="$1"
  local variant_root="$2"
  local work_root="$3"
  local test_root="$4"
  local gpu_set="$5"
  local infer_gpu_set="$6"
  local collision_thresh="$7"

  local logfile="$work_root/launcher.log"
  mkdir -p "$work_root"

  echo "[BASE-CVA] launch $name"
  echo "  GPUs=$gpu_set collision_thresh=$collision_thresh"
  echo "  output=$test_root"

  setsid env \
    DATASET_ROOT="$DATASET_ROOT" \
    WORK_ROOT="$work_root" \
    TRAIN_ROOT="$variant_root/train" \
    TEST_ROOT="$test_root" \
    GPUS="$gpu_set" \
    INFER_GPUS="$infer_gpu_set" \
    PHASES="$PHASES" \
    SPLITS="$SPLITS" \
    CHECKPOINT_KIND="$CHECKPOINT_KIND" \
    SCORE_SOURCE="$SCORE_SOURCE" \
    INFER_BATCH_SIZE="$INFER_BATCH_SIZE" \
    RESUME="$RESUME" \
    EVAL_WORKERS="$EVAL_WORKERS" \
    OFFICIAL_WORKERS="$OFFICIAL_WORKERS" \
    TOP4="$TOP4" \
    INFER_MAX_FRAMES=0 \
    COLLISION_THRESH="$collision_thresh" \
    COLLISION_VOXEL_SIZE="$COLLISION_VOXEL_SIZE" \
    COLLISION_APPROACH_DIST="$COLLISION_APPROACH_DIST" \
    bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh" \
    >"$logfile" 2>&1 &

  PIDS+=("$!")
  NAMES+=("$name")
  LOGS+=("$logfile")
}

wait_wave() {
  local status=0
  local i rc

  # Both children have already been launched, so these waits do not serialize
  # the actual GPU work.
  for i in "${!PIDS[@]}"; do
    rc=0
    wait "${PIDS[$i]}" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "[BASE-CVA ERROR] ${NAMES[$i]} failed (status=$rc)" >&2
      echo "  inspect: ${LOGS[$i]}" >&2
      status=1
    fi
  done

  PIDS=()
  NAMES=()
  LOGS=()

  if [[ "$status" -ne 0 ]]; then
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Wave 1: Base CVA, collision OFF. Both ranking variants run concurrently.
# ---------------------------------------------------------------------------
launch_condition \
  "no_ranking/base/off" "$NO_RANK_ROOT" \
  "$NO_RANK_BASE_OFF_WORK" "$NO_RANK_BASE_OFF_TEST" \
  "$NO_RANK_GPUS" "$NO_RANK_INFER_GPUS" "0"

launch_condition \
  "ranking/base/off" "$RANK_ROOT" \
  "$RANK_BASE_OFF_WORK" "$RANK_BASE_OFF_TEST" \
  "$RANK_GPUS" "$RANK_INFER_GPUS" "0"

wait_wave

# ---------------------------------------------------------------------------
# Wave 2: Base CVA, collision ON. Again run the two checkpoints concurrently.
# ---------------------------------------------------------------------------
launch_condition \
  "no_ranking/base/on" "$NO_RANK_ROOT" \
  "$NO_RANK_BASE_ON_WORK" "$NO_RANK_BASE_ON_TEST" \
  "$NO_RANK_GPUS" "$NO_RANK_INFER_GPUS" "$COLLISION_THRESH"

launch_condition \
  "ranking/base/on" "$RANK_ROOT" \
  "$RANK_BASE_ON_WORK" "$RANK_BASE_ON_TEST" \
  "$RANK_GPUS" "$RANK_INFER_GPUS" "$COLLISION_THRESH"

wait_wave

# Disable cleanup after all jobs complete.
PIDS=()

# ---------------------------------------------------------------------------
# Build a compact Field/Base x collision OFF/ON summary. Existing Field results
# are optional; Base results produced by this script are mandatory.
# ---------------------------------------------------------------------------
"$PYTHON_BIN" - \
  "$NO_RANK_ROOT" "$RANK_ROOT" \
  "$NO_RANK_BASE_OFF_TEST" "$RANK_BASE_OFF_TEST" \
  "$NO_RANK_BASE_ON_TEST" "$RANK_BASE_ON_TEST" \
  "$SUMMARY_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

no_root = Path(sys.argv[1])
rank_root = Path(sys.argv[2])
no_base_off = Path(sys.argv[3])
rank_base_off = Path(sys.argv[4])
no_base_on = Path(sys.argv[5])
rank_base_on = Path(sys.argv[6])
out = Path(sys.argv[7])
out.mkdir(parents=True, exist_ok=True)

splits = ("test_seen", "test_similar", "test_novel")


def load(root, split, required=False):
    path = root / "official" / split / "summary.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    row = json.loads(path.read_text())
    return row


def ap(row):
    return None if row is None else row.get("reported_ap")


def subtract(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [float(x) - float(y) for x, y in zip(a, b)]
    return None


def rejected_ratio(root, split):
    before = after = 0
    for p in (root / split).glob("shard_*.json"):
        row = json.loads(p.read_text())
        before += int(row.get("grasps_before_collision", 0))
        after += int(row.get("grasps_after_collision", 0))
    return None if before <= 0 else 1.0 - after / before


conditions = {
    "no_ranking": {
        "field_off": no_root / "test_latest",
        "field_on": no_root / "test_latest_collision0p01",
        "base_off": no_base_off,
        "base_on": no_base_on,
    },
    "ranking": {
        "field_off": rank_root / "test_latest",
        "field_on": rank_root / "test_latest_collision0p01",
        "base_off": rank_base_off,
        "base_on": rank_base_on,
    },
}

result = {}
for variant, roots in conditions.items():
    result[variant] = {}
    for split in splits:
        rows = {
            name: load(root, split, required=name.startswith("base_"))
            for name, root in roots.items()
        }

        # Guard the causal diagnostic: the new mandatory outputs must really be
        # Base-CVA decodes.
        for name in ("base_off", "base_on"):
            source = rows[name].get("score_source")
            if source != "base":
                raise RuntimeError(
                    f"{variant}/{split}/{name}: expected score_source='base', "
                    f"got {source!r}"
                )

        base_off_ap = ap(rows["base_off"])
        base_on_ap = ap(rows["base_on"])
        field_off_ap = ap(rows["field_off"])
        field_on_ap = ap(rows["field_on"])

        result[variant][split] = {
            "field_collision_off": field_off_ap,
            "field_collision_on": field_on_ap,
            "base_collision_off": base_off_ap,
            "base_collision_on": base_on_ap,
            "base_minus_field_collision_off": subtract(
                base_off_ap, field_off_ap
            ),
            "base_minus_field_collision_on": subtract(
                base_on_ap, field_on_ap
            ),
            "base_collision_gain": subtract(base_on_ap, base_off_ap),
            "base_collision_rejection_ratio": rejected_ratio(
                roots["base_on"], split
            ),
        }

(out / "comparison.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# Field vs Base-CVA 2x2 Evaluation",
    "",
    "Same trained checkpoint/forward path; only final CDF score source and "
    "post-hoc collision filtering are changed.",
    "",
]

for variant in ("no_ranking", "ranking"):
    lines += [f"## {variant}", ""]
    lines += [
        "| Split | Field off | Field on | Base off | Base on | "
        "Base-Field off | Base-Field on | Base collision gain | Reject |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for split in splits:
        r = result[variant][split]
        reject = r["base_collision_rejection_ratio"]
        reject_text = "N/A" if reject is None else f"{100*reject:.2f}%"
        lines.append(
            f"| {split} | "
            f"{r['field_collision_off']} | {r['field_collision_on']} | "
            f"{r['base_collision_off']} | {r['base_collision_on']} | "
            f"{r['base_minus_field_collision_off']} | "
            f"{r['base_minus_field_collision_on']} | "
            f"{r['base_collision_gain']} | {reject_text} |"
        )
    lines.append("")

(out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo
echo "============================================================"
echo "[BASE-CVA 2x2 EVAL] complete"
echo "Base no-ranking collision off: $NO_RANK_BASE_OFF_TEST/official/"
echo "Base no-ranking collision on : $NO_RANK_BASE_ON_TEST/official/"
echo "Base ranking collision off   : $RANK_BASE_OFF_TEST/official/"
echo "Base ranking collision on    : $RANK_BASE_ON_TEST/official/"
echo "2x2 summary                  : $SUMMARY_DIR/comparison.md"
echo "                               $SUMMARY_DIR/comparison.json"
echo "============================================================"
