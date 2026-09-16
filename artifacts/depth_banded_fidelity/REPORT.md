# Depth-banded decay eligibility

A genuine forward-mode sensitivity experiment for decay leaves only, L=4, T=16, 3 seeds, memory disabled. K restricts retained destination layers to destination-source <= K. K=3 is the complete depth transport reference within this four-layer test.

- K=0: mean per-layer cosine=0.9697, mean relative error=0.2570
- K=1: mean per-layer cosine=0.9943, mean relative error=0.1709
- K=2: mean per-layer cosine=0.9977, mean relative error=0.1034
- K=3: mean per-layer cosine=1.0000, mean relative error=0.0000

This is an experiment, not yet the production online estimator. It establishes whether adding cross-layer eligibility transport is worth its persistent-buffer and runtime cost.
