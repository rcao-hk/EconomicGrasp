# Rep-C1 / C2 / C3 follow-up experiments

Branch: `exp/rep-a-depth-robust-reader`.

These experiments follow the diagnosis that A1 learns useful physical
translation correction under depth error, but sometimes makes unnecessary
corrections on clean inputs. They deliberately separate three questions:

1. **Rep-C1 — Oracle acceptance upper bound:** if A1's proposed correction is
   fixed, how much could a perfect stay/move decision recover?
2. **Rep-C2 — Evidence diagnosability:** can harmful vs beneficial A1 moves be
   predicted from score evidence, paired gripper-local geometry, and
   deployment-available reliability evidence?
3. **Rep-C3 — Realistic error generalization:** does A1 remain useful under
   predictor-shaped spatial errors and paired material-appearance changes?

No experiment silently retunes on Similar/Novel.

---

## Rep-C1: oracle stay/move upper bound

### What is fixed

For every query, the candidate proposed by the existing A1 policy is fixed.
C1 only chooses between:

[
\{g_0, g_{A1}\}.
]

The oracle accepts the move iff the fresh exact-action utility is strictly
higher than native; ties keep native. This uses test labels and is therefore an
**upper bound**, never a deployable method.

C1 emits two official-AP score controls:

- `rep_c1_oracle_accept_stage1score`: oracle action, original Stage-1 query
  score. This isolates the maximum action-acceptance benefit without rescoring.
- `rep_c1_oracle_accept_a1score`: oracle action, A1 score matched to the
  actually executed action (A1 selected score when accepted; A1 zero-offset
  score when native is retained).

### Cheap first run on an already exact-evaluated pilot

```bash
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_pilot \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c1_oracle_pilot \
SPLITS=test_seen,test_similar,test_novel \
CASES=nominal,bias:-20,bias:20 \
PHASES=analyze \
bash scripts/run_rep_c1.sh
```

Outputs:

```text
WORK_ROOT/per_frame.csv
WORK_ROOT/comparison.csv
WORK_ROOT/comparison.json
```

### Full-query official oracle AP

The formal AP run may contain only inference + official AP, not fresh
per-action exact labels. In that case C1 first needs its `label` phase.

```bash
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c1_oracle_formal \
SPLITS=test_seen,test_similar,test_novel \
CASES=nominal,bias:-20,bias:20 \
EVAL_SHARDS=1 \
PHASES=label,analyze \
bash scripts/run_rep_c1.sh
```

Then run official AP only on the C1 oracle dumps:

```bash
SOURCE_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_fullpath_formal_joint_ap \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c1_oracle_formal \
SPLITS=test_seen,test_similar,test_novel \
OFFICIAL_WORKERS=4 \
RESUME=1 \
PHASES=official \
bash scripts/run_rep_c1.sh
```

The full-query `label` phase is expensive. The 64-query pilot is the intended
go/no-go experiment. Only run formal oracle AP if the pilot shows substantial
headroom over the current A1 policy.

---

## Rep-C2: can harmful corrections be recognized?

C2 keeps A1's **fixed-0 proposed candidate** and trains only an accept/reject
verifier. It compares three nested evidence sets with the same 160-D MLP and
the same parameter count:

| Variant | Evidence available |
|---|---|
| `score` | A1 six-bin CDF profiles for native/candidate/difference, score margin, offset, original native score |
| `local` | score + paired gripper-local predicted-depth evidence in closing/finger/palm/approach regions |
| `reliable` | local + projection visibility, depth gradient, local roughness and discontinuity evidence |

The local boxes are the same five gripper regions used by Rep-A's
`IndependentImageReader`. No sensor depth, rendered depth, CAD or GT depth is
an input to C2.

Training target for a proposed move:

[
y=1[U(g_{A1}) > U(g_0)].
]

The classifier is trained only where A1 actually proposes a move. The
validation threshold is selected on Seen using nominal plus **in-support**
bias/scale/smooth perturbations. Similar/Novel and off-grid/edge/severe tests
are never retuned.

### Smoke

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2_smoke \
GPUS=0 \
EPOCHS=2 \
MAX_TRAIN_FRAMES=16 \
MAX_VAL_FRAMES=4 \
MAX_TEST_FRAMES=4 \
PHASES=train,test \
bash scripts/run_rep_c2.sh
```

### Formal diagnostic

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c2_verifier \
GPUS=0,1,2 \
EPOCHS=12 \
SPLITS=test_seen,test_similar,test_novel \
PHASES=train,test \
bash scripts/run_rep_c2.sh
```

Read the result in this order:

1. `utility_gain` vs A1 fixed-0 `a1_fixed0_gain`;
2. `harmful_rejection` at comparable `beneficial_retention`;
3. `gate_auc` only as a secondary diagnostic;
4. compare `score -> local -> reliable`.

A high classification accuracy is not a success criterion. Rejecting almost
all moves can look accurate but destroys useful correction.

---

## Rep-C3: predictor-shaped and material-appearance errors

C3 reruns the actual Stage-1 and regenerates new physical grasps. It reuses the
existing exact evaluator and official GraspNet AP scripts.

### A. Predictor-shaped residual stress

Let the original RGB-only prediction be (hat D) and the diagnostic GT depth
be (D^*). C3 constructs:

[
D_{stress} = hat D + alpha(hat D-D^*)
]

for `residual_full`. This amplifies the model's **own spatial error field**,
rather than injecting an artificial global bias.

For `residual_local`, the per-frame median residual is removed first, which
isolates local shape/edge distortion from shared placement bias.

GT is used only to construct this stress test and report error statistics. It
is never passed into Stage-1, A0 or A1.

### Residual-stress pilot

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c3_residual \
GPUS=0,1,2 \
SCORERS=A0,A1 \
SPLITS=test_seen,test_similar,test_novel \
QUERY_LIMIT=64 \
RESIDUAL_CASES=residual_full:0.5,residual_full:1.0,residual_local:0.5,residual_local:1.0 \
EVAL_SHARDS=1 \
PHASES=infer,evaluate,summary \
bash scripts/run_rep_c3.sh
```

Outputs include `c3_comparison.csv` with both grasp metrics and
`gt_mae_mm / gt_rmse_mm / gt_bias_mm`.

### B. Paired material RGB

Material testing must use a paired RGB for the same scene/frame. C3 replaces
only the RGB path before the dataset crop. The original sensor depth/segmentation
still define the same crop and intrinsics. Stage-1 is rerun from the augmented
RGB, so the image features and predicted depth are **both generated from that
same image**.

Pass one or more material roots:

```bash
MATERIALS=mat_aug=/data/robotarm/graspnet_material_aug \
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c3_material \
GPUS=0,1,2 \
SCORERS=A0,A1 \
QUERY_LIMIT=64 \
PHASES=infer,evaluate,summary \
bash scripts/run_rep_c3.sh
```

C3 searches these common paired layouts:

```text
ROOT/scenes/scene_XXXX/<camera>/rgb/0000.png
ROOT/scene_XXXX/<camera>/rgb/0000.png
ROOT/scenes/scene_XXXX/rgb/0000.png
ROOT/scene_XXXX/rgb/0000.png
ROOT/scene_XXXX/0000.png
```

Missing pairs fail loudly; C3 never silently falls back to the original RGB.

### Formal AP after a positive pilot

Use a **new WORK_ROOT** because `QUERY_LIMIT` is part of the protocol:

```bash
WORK_ROOT=/data2/robotarm/result/grasp/rgbgrasp/rep_c3_formal \
GPUS=0,1,2 \
SCORERS=A0,A1 \
QUERY_LIMIT=0 \
RESIDUAL_CASES=residual_full:1.0,residual_local:1.0 \
PHASES=infer,official \
bash scripts/run_rep_c3.sh
```

For material official AP, add the same `MATERIALS=...` setting during
inference. Official evaluation does not need the material root after all dumps
have been generated.

---

## Unit tests

```bash
python -m pytest -q tests/test_rep_c123.py
```

The tests cover C1 oracle accept/reject semantics, C2 equal-capacity evidence
masking and gripper-local feature extraction, and C3 full/local residual stress
construction. They are synthetic unit tests; the first server run should still
use the small smoke configurations above before launching full experiments.
