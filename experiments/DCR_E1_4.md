# DCR-E1-4: Corruption-suite closure

This experiment asks whether the **fine-tuned DCR center corrector itself**
improves the E1-4 corruption regime, especially smooth/local geometry errors.

A direct comparison against the old E1-4 numbers is confounded because old
E1-4 used the E1 CDF as the final global score while the current recommended
DCR configuration uses the frozen Stage-1 score. DCR-E1-4 therefore evaluates
two correctors under the **same Stage-1 query ranking**:

1. `dcr/stage1`: current DCR corrector + frozen Stage-1 ranking.
2. `e1_ref/stage1`: original E1 corrector + the same frozen Stage-1 ranking.
3. `dcr/native`: uncorrected Stage-1 action baseline, evaluated once.

The E1 checkpoint is loaded by `inference_dcr_cva.py` through its existing
zero-residual compatibility path. Both sources use the same deterministic
corruption seed for the same scene/annotation/case. The action grid remains
`[-40,-20,-10,0,10,20,40]` mm.

Default corruption suite:

```text
nominal
bias:-15, bias:+15
bias:-25, bias:+25
scale:-0.03, scale:+0.03
smooth:5, smooth:10
```

Run:

```bash
cd /home/robotarm/EconomicGrasp
git switch exp/e1-e2-center-hypothesis-cva
git pull --ff-only

GPUS=0,1,2 \
OFFICIAL_WORKERS=2 \
bash scripts/run_dcr_e1_4.sh
```

The wrapper does **not train**. It expects:

```text
E1_BASE_ROOT/train/E1/checkpoint_best.pt
DCR_BASE_ROOT/train/cdf/checkpoint_best.pt
```

Override `E1_CHECKPOINT` or `DCR_CHECKPOINT` when needed.

To run only inference first:

```bash
PHASES=infer GPUS=0,1,2 bash scripts/run_dcr_e1_4.sh
```

To resume only official evaluation and summary:

```bash
PHASES=eval,summary GPUS=0,1,2 OFFICIAL_WORKERS=2 \
bash scripts/run_dcr_e1_4.sh
```

Inference shards each split across all listed GPUs. Official evaluation uses the
GPU list as split-level concurrency slots, so three GPUs evaluate Seen/Similar/
Novel concurrently. GraspNetEval remains CPU-heavy; do not set
`OFFICIAL_WORKERS` too high.

Outputs:

```text
WORK_ROOT/test/dcr/
WORK_ROOT/test/e1_ref/
WORK_ROOT/comparison.csv
WORK_ROOT/correction_effects.csv
WORK_ROOT/macro_summary.json
```

The key quantity is:

```text
dcr_over_e1
= AP(DCR corrector + Stage1 ranking)
- AP(E1 corrector + Stage1 ranking)
```

For the current research question, inspect `smooth:5` and `smooth:10` on
Similar/Novel first. If DCR does not materially improve them, the result supports
moving to an independent evidence path rather than further center-corrector
optimization.
