# Frozen U3 RayRoPE halfwidth diagnostic

## Purpose and invariants

This is a **no-retraining, within-checkpoint** comparison using the completed Mixed-v2 U3 checkpoint. The Python wrapper injects a pre-hook into RayRoPE grouping to change **only** the optical-depth halfwidth fed to the positional encoding. The unchanged `inference_moge_rayrope.py` still handles image loading, model forward, CVA-CDF decoding and model-free collision filtering. The unchanged `eval_moge_rayrope.py` computes official GraspNet AP.

New files are under `scripts/`, `tests/` and `experiments/`, outside the original `code_fingerprint()`. Do not edit fingerprinted model/runtime/source files, bypass checkpoint hashes, or overwrite previous U3 results.

| Mode | Grouping halfwidth | Controlled question |
|---|---|---|
| `learned` | Original predicted sigma | Exact native U3 reference |
| `point` | Zero everywhere | Does any interval help within fixed U3 weights? |
| `constant_train` | Mean from **epoch-19 U3 train metrics** | Does spatial variation help beyond mean width? |
| `fixed20` | Fixed 20 mm | Comparison with original fixed-interval scale |
| `shuffle` | Spatially permuted sigma, same per-frame values | Does spatial alignment matter? |

`constant_train` uses only `train/metrics.json` (about 7.73 mm in the reported v2 run), never test AP. The shuffle uses an independent per-frame RNG indexed by scene and annotation, preserving the halfwidth histogram on finite, positive predicted-depth pixels without advancing the global inference RNG. All five modes use the **same U3 trained weights**, not independently trained U0 weights.

The wrapper records SHA256 of all pre-group `token_sel_idx`, `seed_xyz` and `top_view_rot` entries for every frame; the comparator rejects any mismatches across modes. With `REQUIRE_NATIVE_PARITY=1` the `learned` mode also compares each N×17 grasp dump to existing native U3 dumps at 1e-6 tolerance before writing it.

## Preflight and execution

Use a separate Git worktree from `exp/moge-rayrope-grasp`; do not disrupt active training or modify existing checkout. Use the same trustworthy DAV2 ViT-B checkpoint, verified by the original loader. Do not use GPU 4.

```bash
cd /home/robotarm/EconomicGrasp
git fetch origin exp/moge-rayrope-grasp
git worktree list
# Only if the target path does not already exist:
git worktree add --detach /home/robotarm/EconomicGrasp-U3-Diag origin/exp/moge-rayrope-grasp
cd /home/robotarm/EconomicGrasp-U3-Diag

export PYTHON_BIN=/home/robotarm/miniconda3/envs/grasp/bin/python
export DATASET_ROOT=/data/robotarm/dataset/graspnet
export U3_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_renderfg_tsdfbg_v2/formal/U3
export U3_CHECKPOINT="$U3_ROOT/train/checkpoint_latest.pt"
export NATIVE_ROOT="$U3_ROOT/test_native"
export DIAG_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rayrope_mixed_renderfg_tsdfbg_v2/diagnostics/u3_halfwidth
export GPUS=0,1,2   # Confirm GPUs are idle; GPU4 prohibited
```

Run CPU contract tests and a short CUDA smoke; **never use smoke AP as a result**:

```bash
bash -n scripts/run_rayrope_u3_diagnostic.sh
"$PYTHON_BIN" -m pytest -q tests/test_rayrope_u3_halfwidth.py

PHASE=infer MODES="learned point constant_train fixed20 shuffle" \
SPLITS="test_novel" MAX_FRAMES=9 REQUIRE_NATIVE_PARITY=0 \
DIAG_ROOT="${DIAG_ROOT}_smoke" bash scripts/run_rayrope_u3_diagnostic.sh
```

Full official protocol: final epoch-19 U3, inference batch=3, three GPU shards, seed control as native, original RealSense Seen/Similar/Novel each 780 frames, Top-1 view, no depth bias, collision both (threshold .01 / voxel .01 m / approach .05 m). Original sensor depth is never the RGB network input, only post-filter context.

```bash
PHASE=all MODES="learned point constant_train fixed20 shuffle" \
SPLITS="test_seen test_similar test_novel" REQUIRE_NATIVE_PARITY=1 \
MAX_FRAMES=0 RESUME=0 bash scripts/run_rayrope_u3_diagnostic.sh
```

On a verified interruption only, keep source/checkpoint/mode/sharding signatures fixed:

```bash
PHASE=all RESUME=1 bash scripts/run_rayrope_u3_diagnostic.sh
# Or compare previously completed evaluations:
PHASE=compare bash scripts/run_rayrope_u3_diagnostic.sh
```

### Outputs and acceptance

- `$DIAG_ROOT/<mode>/test_collision_on/` and `test_collision_off/`: dump files, per-frame completion markers with proposal SHA, manifest, original official AP arrays
- `$DIAG_ROOT/comparison.csv`, `comparison.json` and `comparison.md`: Seen/Similar/Novel/Mean AP, Δ against learned U3, scene-paired bootstrap CI and collision-on/off
- Learned-mode native N×17 parity must pass before scientific interpretation. Full proposal hashes must match, 780 frames per split and official accuracy shape `(30,26,50,6)`.
- `point` is zero-width **within the U3 checkpoint**, not independent U0. `learned` versus `constant_train`/`shuffle` tests the spatial uncertainty contribution. Scene CI is **not** training-seed variance.
- If learned does not outperform constant/shuffle, no independent spatial-uncertainty AP benefit is established. The predicted halfwidth may still be under-calibrated as a 90% interval.

No new training, no hyperparameter sweep, no AP-based constant selection, no collision-filter changes.
