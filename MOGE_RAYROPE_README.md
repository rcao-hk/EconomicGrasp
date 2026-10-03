# EconomicGrasp — MoGe + RayRoPE online implementation

Direct source additions for `rcao-hk/EconomicGrasp`, based on:

```
main:   52d09f925059bec3643610ecf1f1722894627ee5
branch: exp/moge-rayrope-grasp20 (intended; not published by this session)
```

This package is **not a complete clone**. Place its new files at the corresponding
paths in a clean worktree of the pinned main revision. There is no apply/patch
program, and existing main model/loss files are not overwritten.

See:

- `experiments/MOGE_RAYROPE_DESIGN.md`: architecture, loss, flags, paper fidelity and limitations.
- `CODEX_MOGE_RAYROPE_WORKPLAN.md`: repository publication, CUDA validation, complete experiment commands and reporting.
- `MOGE_RAYROPE_VERIFICATION.json`: checks actually executed in this session.

New modules replace the metric depth parameterization and/or CVA grouping through
independent flags. Main's proposal/view/CDF/width training and grasp decoding are
reused. The DAV2 encoder is frozen; geometry receives its own online supervision;
grasp losses cannot rewrite numeric geometry. Training is on 20% GraspNet frames,
20 epochs, no feature/action cache and no gradient accumulation.

This is a single-view adaptation of the **MoGe affine pointmap/alignment** and
**RayRoPE projective interval expectation** mechanisms, not a reproduction of the
original full models. It does not load MoGe pretrained weights. Learned interval
supervision and virtual grasp cameras are task-specific adaptations.

CPU tests and two-rank CPU validation passed. Real CUDA/main-model parity,
training, inference and official AP have **not been run in this environment**.
The current GitHub connector exposes reads but no write actions, so this session
has **not** created a remote branch or pushed these files.
