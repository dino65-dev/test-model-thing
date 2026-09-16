# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.9774 +- 0.0000, relative=0.2112, norm=0.9786
- L=1, memory=0, T=256, trace: cosine=0.9993 +- 0.0000, relative=0.0373, norm=1.0004
- L=1, memory=1, T=256, instant: cosine=0.9343 +- 0.0000, relative=0.3572, norm=0.9131
- L=1, memory=1, T=256, trace: cosine=0.9980 +- 0.0000, relative=0.0641, norm=1.0085
- L=2, memory=0, T=256, instant: cosine=0.8093 +- 0.0000, relative=0.6531, norm=0.5239
- L=2, memory=0, T=256, trace: cosine=0.9617 +- 0.0000, relative=0.2799, norm=0.9042
- L=2, memory=1, T=256, instant: cosine=0.7942 +- 0.0000, relative=0.6082, norm=0.8202
- L=2, memory=1, T=256, trace: cosine=0.9011 +- 0.0000, relative=0.4413, norm=0.9830
- L=4, memory=0, T=256, instant: cosine=0.7423 +- 0.0000, relative=0.6770, norm=0.6461
- L=4, memory=0, T=256, trace: cosine=0.8831 +- 0.0000, relative=0.4702, norm=0.8515
- L=4, memory=1, T=256, instant: cosine=0.7057 +- 0.0000, relative=0.7146, norm=0.6126
- L=4, memory=1, T=256, trace: cosine=0.9354 +- 0.0000, relative=0.3539, norm=0.9519

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
