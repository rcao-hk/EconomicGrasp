#!/usr/bin/env bash
# Zero-training Base/Field CDF-logit fusion sweep on the existing absolute
# matched20 checkpoint. Intermediate alphas are evaluated; alpha=0/1 reuse the
# already-computed Base/Field collision-on endpoints when available.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_ROOT=${DATASET_ROOT:-/data/robotarm/dataset/graspnet}
MATCHED_ROOT=${MATCHED_ROOT:-/data2/robotarm/result/grasp/rgbgrasp/dav2_mgf_graspnet20_mixed_matched}
GPUS=${GPUS:-0,1,2}
INFER_GPUS=${INFER_GPUS:-$GPUS}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
RESUME=${RESUME:-0}

# We already have alpha=0 (Base) and alpha=1 (Field) from previous diagnostics.
# Compute only the informative intermediate points by default.
ALPHAS=${ALPHAS:-0.25,0.5,0.75}
SPLITS=${SPLITS:-test_seen,test_similar,test_novel}
OFFICIAL_WORKERS=${OFFICIAL_WORKERS:-10}
EVAL_WORKERS=${EVAL_WORKERS:-2}

TRAIN_ROOT=${TRAIN_ROOT:-$MATCHED_ROOT/train}
FIELD_ENDPOINT_ROOT=${FIELD_ENDPOINT_ROOT:-$MATCHED_ROOT/test_latest}
BASE_ENDPOINT_ROOT=${BASE_ENDPOINT_ROOT:-$MATCHED_ROOT/test_latest_base_cva_collision0p01}
SWEEP_ROOT=${SWEEP_ROOT:-$MATCHED_ROOT/blend_logit_sweep}
SUMMARY_DIR=${SUMMARY_DIR:-$SWEEP_ROOT/summary}

CKPT="$TRAIN_ROOT/checkpoint_latest.pt"
[[ -f "$CKPT" ]] || {
  echo "[ERROR] Missing matched20 checkpoint: $CKPT" >&2
  exit 2
}

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "[ERROR] Need at least one GPU" >&2
  exit 2
fi

alpha_tag() {
  local x="$1"
  x="${x//./p}"
  x="${x//-/m}"
  echo "$x"
}

echo "============================================================"
echo "[MGF LOGIT BLEND SWEEP]"
echo "  checkpoint       : $CKPT"
echo "  GPUs             : $GPUS"
echo "  alphas to compute: $ALPHAS"
echo "  endpoints        : alpha=0 Base, alpha=1 Field"
echo "  collision        : on, threshold=0.01"
echo "============================================================"

IFS=',' read -r -a ALPHA_ARRAY <<< "$ALPHAS"
for alpha in "${ALPHA_ARRAY[@]}"; do
  tag="$(alpha_tag "$alpha")"
  work="$SWEEP_ROOT/a$tag"
  test="$MATCHED_ROOT/test_latest_blend_a${tag}_collision0p01"

  echo
  echo "[MGF LOGIT BLEND] alpha=$alpha -> $test"

  env \
    DATASET_ROOT="$DATASET_ROOT" \
    WORK_ROOT="$work" \
    TRAIN_ROOT="$TRAIN_ROOT" \
    TEST_ROOT="$test" \
    GPUS="$GPUS" \
    INFER_GPUS="$INFER_GPUS" \
    PHASES=infer,eval \
    SPLITS="$SPLITS" \
    CHECKPOINT_KIND=latest \
    RESUME="$RESUME" \
    SCORE_SOURCE=blend \
    BLEND_ALPHA="$alpha" \
    INFER_BATCH_SIZE="$INFER_BATCH_SIZE" \
    EVAL_WORKERS="$EVAL_WORKERS" \
    OFFICIAL_WORKERS="$OFFICIAL_WORKERS" \
    COLLISION_THRESH=0.01 \
    COLLISION_VOXEL_SIZE=0.01 \
    COLLISION_APPROACH_DIST=0.05 \
    bash "$ROOT_DIR/scripts/run_metric_grasp_field.sh"
done

mkdir -p "$SUMMARY_DIR"
"$PYTHON_BIN" - \
  "$FIELD_ENDPOINT_ROOT" "$BASE_ENDPOINT_ROOT" "$MATCHED_ROOT" \
  "$SWEEP_ROOT" "$SUMMARY_DIR" "$ALPHAS" <<'PY'
import json
import sys
from pathlib import Path

field_root = Path(sys.argv[1])
base_root = Path(sys.argv[2])
matched_root = Path(sys.argv[3])
sweep_root = Path(sys.argv[4])
out = Path(sys.argv[5])
alphas = [float(x) for x in sys.argv[6].split(",") if x.strip()]
splits = ("test_seen", "test_similar", "test_novel")
out.mkdir(parents=True, exist_ok=True)


def tag(alpha):
    return str(alpha).replace(".", "p").replace("-", "m")


def load(root, split, required=True):
    p = root / "official" / split / "summary.json"
    if not p.is_file():
        if required:
            raise FileNotFoundError(p)
        return None
    return json.loads(p.read_text())


def primary(ap):
    while isinstance(ap, list):
        if not ap:
            raise ValueError("Empty reported_ap list")
        ap = ap[0]
    return float(ap)


conditions = []
if all((base_root / "official" / s / "summary.json").is_file() for s in splits):
    conditions.append((0.0, base_root, "existing-base"))
else:
    print("[WARN] alpha=0 Base endpoint missing; summary will omit it")

for alpha in alphas:
    root = matched_root / f"test_latest_blend_a{tag(alpha)}_collision0p01"
    conditions.append((alpha, root, "blend"))

if all((field_root / "official" / s / "summary.json").is_file() for s in splits):
    conditions.append((1.0, field_root, "existing-field"))
else:
    print("[WARN] alpha=1 Field endpoint missing; summary will omit it")

conditions.sort(key=lambda x: x[0])
result = []
for alpha, root, source in conditions:
    row = {"alpha": alpha, "source": source, "root": str(root), "splits": {}}
    for split in splits:
        s = load(root, split)
        if source == "blend":
            if s.get("score_source") != "blend":
                raise RuntimeError(
                    f"{root}/{split}: expected score_source=blend, "
                    f"got {s.get('score_source')!r}"
                )
            got = float(s.get("blend_alpha"))
            if abs(got - alpha) > 1e-9:
                raise RuntimeError(
                    f"{root}/{split}: expected alpha={alpha}, got {got}"
                )
        row["splits"][split] = {
            "reported_ap": s["reported_ap"],
            "primary_ap": primary(s["reported_ap"]),
            "mean_accuracy": s.get("mean_accuracy"),
        }
    row["primary_mean_over_splits"] = sum(
        row["splits"][s]["primary_ap"] for s in splits
    ) / len(splits)
    result.append(row)

(out / "sweep.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "# Base–Field CDF Logit Fusion Sweep",
    "",
    "final_logits = (1-alpha) * Base + alpha * Field; "
    "all rows use collision threshold 0.01.",
    "",
    "| alpha | Seen AP | Similar AP | Novel AP | Mean | source |",
    "|---:|---:|---:|---:|---:|---|",
]
for row in result:
    lines.append(
        f"| {row['alpha']:.2f} | "
        f"{100*row['splits']['test_seen']['primary_ap']:.2f} | "
        f"{100*row['splits']['test_similar']['primary_ap']:.2f} | "
        f"{100*row['splits']['test_novel']['primary_ap']:.2f} | "
        f"{100*row['primary_mean_over_splits']:.2f} | {row['source']} |"
    )

(out / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo
echo "============================================================"
echo "[MGF LOGIT BLEND SWEEP] complete"
echo "Summary: $SUMMARY_DIR/sweep.md"
echo "         $SUMMARY_DIR/sweep.json"
echo "============================================================"
