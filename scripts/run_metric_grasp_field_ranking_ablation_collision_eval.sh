#!/usr/bin/env bash
# Parallel collision-on inference + official GraspNet evaluation for the
# matched MGF ranking ablation.
#
# Historical EconomicGrasp-compatible collision protocol:
#   source          = original GraspNet sensor point cloud
#   collision thresh= 0.01
#   voxel size      = 0.01 m
#   approach dist   = 0.05 m
#
# Network inputs remain RGB + camera metadata. Sensor depth is used ONLY by
# the post-hoc model-free collision detector.
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
EVAL_WORKERS=${EVAL_WORKERS:-1}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-2}
TOP4=${TOP4:-0}
RESUME=${RESUME:-0}

COLLISION_THRESH=${COLLISION_THRESH:-0.01}
COLLISION_VOXEL_SIZE=${COLLISION_VOXEL_SIZE:-0.01}
COLLISION_APPROACH_DIST=${COLLISION_APPROACH_DIST:-0.05}

CHECKPOINT_KIND=latest
PHASES=infer,eval

# Keep collision-on dumps/logs isolated from the already-computed no-collision
# test_latest outputs.
NO_RANK_COLLISION_WORK_ROOT=${NO_RANK_COLLISION_WORK_ROOT:-$NO_RANK_ROOT/collision_eval_c001}
RANK_COLLISION_WORK_ROOT=${RANK_COLLISION_WORK_ROOT:-$RANK_ROOT/collision_eval_c001}
NO_RANK_TEST_ROOT=${NO_RANK_TEST_ROOT:-$NO_RANK_ROOT/test_latest_collision0p01}
RANK_TEST_ROOT=${RANK_TEST_ROOT:-$RANK_ROOT/test_latest_collision0p01}
SUMMARY_DIR=${SUMMARY_DIR:-$ABLATION_ROOT/comparison_collision0p01}

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

# Reuse the existing protocol validator before spending GPU time. This does not
# change either checkpoint.
mkdir -p "$SUMMARY_DIR"
"$PYTHON_BIN" "$ROOT_DIR/compare_metric_grasp_field_ranking.py"   --no-ranking "$NO_RANK_ROOT/train"   --ranking "$RANK_ROOT/train"   --output-dir "$SUMMARY_DIR/training_protocol_check"   --expected-epochs 20   >"$SUMMARY_DIR/preflight.log" 2>&1

echo "============================================================"
echo "[MGF COLLISION-ON EVAL]"
echo "  source             : original GraspNet sensor point cloud"
echo "  collision threshold: $COLLISION_THRESH"
echo "  voxel size (m)     : $COLLISION_VOXEL_SIZE"
echo "  approach dist (m)  : $COLLISION_APPROACH_DIST"
echo "  checkpoint         : latest / epoch 20"
echo "  splits             : $SPLITS"
echo "  no-ranking GPUs    : $NO_RANK_GPUS"
echo "  ranking GPUs       : $RANK_GPUS"
echo "  no-ranking dump    : $NO_RANK_TEST_ROOT"
echo "  ranking dump       : $RANK_TEST_ROOT"
echo "============================================================"

NO_RANK_LAUNCHER_LOG="$NO_RANK_COLLISION_WORK_ROOT/launcher.log"
RANK_LAUNCHER_LOG="$RANK_COLLISION_WORK_ROOT/launcher.log"
NO_RANK_PID=""
RANK_PID=""

cleanup_parallel() {
  local pid
  for pid in "$NO_RANK_PID" "$RANK_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_parallel EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch_collision_variant() {
  local name="$1"
  local variant_root="$2"
  local collision_work_root="$3"
  local test_root="$4"
  local gpu_set="$5"
  local infer_gpu_set="$6"
  local launcher_log="$7"

  mkdir -p "$collision_work_root"
  echo "[MGF COLLISION-ON] launching $name on GPUs $gpu_set"

  setsid env     DATASET_ROOT="$DATASET_ROOT"     WORK_ROOT="$collision_work_root"     TRAIN_ROOT="$variant_root/train"     TEST_ROOT="$test_root"     GPUS="$gpu_set"     INFER_GPUS="$infer_gpu_set"     PHASES="$PHASES"     SPLITS="$SPLITS"     CHECKPOINT_KIND="$CHECKPOINT_KIND"     RESUME="$RESUME"     EVAL_WORKERS="$EVAL_WORKERS"     OFFICIAL_WORKERS="$OFFICIAL_WORKERS"     TOP4="$TOP4"     INFER_MAX_FRAMES=0     COLLISION_THRESH="$COLLISION_THRESH"     COLLISION_VOXEL_SIZE="$COLLISION_VOXEL_SIZE"     COLLISION_APPROACH_DIST="$COLLISION_APPROACH_DIST"     bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh"     >"$launcher_log" 2>&1 &

  LAST_PID=$!
}

launch_collision_variant   "no_ranking" "$NO_RANK_ROOT"   "$NO_RANK_COLLISION_WORK_ROOT" "$NO_RANK_TEST_ROOT"   "$NO_RANK_GPUS" "$NO_RANK_INFER_GPUS" "$NO_RANK_LAUNCHER_LOG"
NO_RANK_PID=$LAST_PID

launch_collision_variant   "ranking" "$RANK_ROOT"   "$RANK_COLLISION_WORK_ROOT" "$RANK_TEST_ROOT"   "$RANK_GPUS" "$RANK_INFER_GPUS" "$RANK_LAUNCHER_LOG"
RANK_PID=$LAST_PID

echo "[MGF COLLISION-ON] both variants running"
echo "  no-ranking pid=$NO_RANK_PID"
echo "  ranking    pid=$RANK_PID"

if wait -n "$NO_RANK_PID" "$RANK_PID"; then
  FIRST_STATUS=0
else
  FIRST_STATUS=$?
fi
if [[ "$FIRST_STATUS" -ne 0 ]]; then
  echo "[MGF COLLISION-ON ERROR] first completed variant failed (status=$FIRST_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  cleanup_parallel
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
  exit "$FIRST_STATUS"
fi

SECOND_STATUS=0
if kill -0 "$NO_RANK_PID" 2>/dev/null; then
  wait "$NO_RANK_PID" || SECOND_STATUS=$?
elif kill -0 "$RANK_PID" 2>/dev/null; then
  wait "$RANK_PID" || SECOND_STATUS=$?
else
  wait "$NO_RANK_PID" 2>/dev/null || true
  wait "$RANK_PID" 2>/dev/null || true
fi
if [[ "$SECOND_STATUS" -ne 0 ]]; then
  echo "[MGF COLLISION-ON ERROR] second variant failed (status=$SECOND_STATUS)." >&2
  echo "  inspect: $NO_RANK_LAUNCHER_LOG" >&2
  echo "  inspect: $RANK_LAUNCHER_LOG" >&2
  exit "$SECOND_STATUS"
fi

NO_RANK_PID=""
RANK_PID=""

# Build a compact paired summary. If the previous no-collision test_latest
# results exist, also report collision-on minus collision-off deltas.
"$PYTHON_BIN" -   "$NO_RANK_ROOT" "$RANK_ROOT"   "$NO_RANK_TEST_ROOT" "$RANK_TEST_ROOT"   "$SUMMARY_DIR" <<'PY'
import json
import sys
from pathlib import Path

no_root = Path(sys.argv[1])
rank_root = Path(sys.argv[2])
no_col_root = Path(sys.argv[3])
rank_col_root = Path(sys.argv[4])
out = Path(sys.argv[5])
collision_thresh = float(sys.argv[6])
collision_voxel = float(sys.argv[7])
collision_approach = float(sys.argv[8])
out.mkdir(parents=True, exist_ok=True)
splits = ("test_seen", "test_similar", "test_novel")


def load_summary(root, split):
    p = root / "official" / split / "summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def retention(root, split):
    before = after = 0
    for p in sorted((root / split).glob("shard_*.json")):
        row = json.loads(p.read_text())
        before += int(row.get("grasps_before_collision", 0))
        after += int(row.get("grasps_after_collision", 0))
    if before <= 0:
        return None
    return {
        "before": before,
        "after": after,
        "retention_ratio": after / before,
        "rejection_ratio": 1.0 - after / before,
    }


def vec_delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return None
    return [float(x) - float(y) for x, y in zip(a, b)]


result = {
    "collision_protocol": {
        "source": "original_graspnet_sensor_point_cloud",
        "threshold": collision_thresh,
        "voxel_size_m": collision_voxel,
        "approach_dist_m": collision_approach,
    },
    "no_ranking": {},
    "ranking": {},
}

for name, base_root, col_root in (
    ("no_ranking", no_root / "test_latest", no_col_root),
    ("ranking", rank_root / "test_latest", rank_col_root),
):
    for split in splits:
        col = load_summary(col_root, split)
        if col is None:
            raise FileNotFoundError(col_root / "official" / split / "summary.json")
        noc = load_summary(base_root, split)
        row = {
            "collision_on_ap": col.get("reported_ap"),
            "collision_filter": col.get("collision_filter"),
            "retention": retention(col_root, split),
        }
        if noc is not None:
            row["collision_off_ap"] = noc.get("reported_ap")
            row["delta_on_minus_off"] = vec_delta(
                col.get("reported_ap"),
                noc.get("reported_ap"),
            )
        result[name][split] = row

(out / "comparison.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)

lines = [
    "# MGF Collision-On Evaluation",
    "",
    "Protocol: original GraspNet sensor-cloud model-free collision filter; "
    f"threshold={collision_thresh}, voxel={collision_voxel} m, "
    f"approach={collision_approach} m.",
    "",
]
for name in ("no_ranking", "ranking"):
    lines += [f"## {name}", ""]
    lines += [
        "| Split | Collision-off AP | Collision-on AP | Delta | Rejected grasps |",
        "|---|---|---|---|---:|",
    ]
    for split in splits:
        row = result[name][split]
        off = row.get("collision_off_ap", "N/A")
        on = row["collision_on_ap"]
        delta = row.get("delta_on_minus_off", "N/A")
        ret = row.get("retention")
        rej = "N/A" if ret is None else f"{100*ret['rejection_ratio']:.2f}%"
        lines.append(f"| {split} | {off} | {on} | {delta} | {rej} |")
    lines.append("")

(out / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo
echo "============================================================"
echo "[MGF COLLISION-ON] complete"
echo "No-ranking official: $NO_RANK_TEST_ROOT/official/"
echo "Ranking official:    $RANK_TEST_ROOT/official/"
echo "Paired summary:      $SUMMARY_DIR/comparison.md"
echo "                     $SUMMARY_DIR/comparison.json"
echo "============================================================"
