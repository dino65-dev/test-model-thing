# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.8876 +- 0.0000, relative=0.4630, norm=0.8408
- L=1, memory=0, T=256, trace: cosine=0.9995 +- 0.0000, relative=0.0310, norm=0.9995
- L=1, memory=1, T=256, instant: cosine=0.8610 +- 0.0000, relative=0.5142, norm=0.7860
- L=1, memory=1, T=256, trace: cosine=0.9730 +- 0.0000, relative=0.2322, norm=0.9484
- L=2, memory=0, T=256, instant: cosine=0.1567 +- 0.0000, relative=0.9899, norm=0.2239
- L=2, memory=0, T=256, trace: cosine=0.9075 +- 0.0000, relative=0.6035, norm=0.4742
- L=2, memory=1, T=256, instant: cosine=0.5781 +- 0.0000, relative=0.8167, norm=0.5441
- L=2, memory=1, T=256, trace: cosine=0.9440 +- 0.0000, relative=0.3622, norm=0.7943
- L=4, memory=0, T=256, instant: cosine=0.6808 +- 0.0000, relative=0.7340, norm=0.7294
- L=4, memory=0, T=256, trace: cosine=0.8718 +- 0.0000, relative=0.4918, norm=0.8276
- L=4, memory=1, T=256, instant: cosine=0.0642 +- 0.0000, relative=1.0183, norm=0.2666
- L=4, memory=1, T=256, trace: cosine=0.0684 +- 0.0000, relative=1.0514, norm=0.4001

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
