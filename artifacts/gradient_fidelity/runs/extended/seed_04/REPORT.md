# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.7999 +- 0.0000, relative=0.6022, norm=0.7492
- L=1, memory=0, T=256, trace: cosine=0.9999 +- 0.0000, relative=0.0106, norm=0.9999
- L=1, memory=1, T=256, instant: cosine=0.7721 +- 0.0000, relative=0.6669, norm=0.5697
- L=1, memory=1, T=256, trace: cosine=0.9932 +- 0.0000, relative=0.1266, norm=1.0425
- L=2, memory=0, T=256, instant: cosine=0.6411 +- 0.0000, relative=0.7799, norm=0.5022
- L=2, memory=0, T=256, trace: cosine=0.9748 +- 0.0000, relative=0.3083, norm=0.7621
- L=2, memory=1, T=256, instant: cosine=0.7125 +- 0.0000, relative=0.7071, norm=0.6257
- L=2, memory=1, T=256, trace: cosine=0.9551 +- 0.0000, relative=0.3655, norm=1.1691
- L=4, memory=0, T=256, instant: cosine=0.6126 +- 0.0000, relative=0.7959, norm=0.5188
- L=4, memory=0, T=256, trace: cosine=0.8896 +- 0.0000, relative=0.4591, norm=0.8437
- L=4, memory=1, T=256, instant: cosine=0.6595 +- 0.0000, relative=0.8099, norm=0.3581
- L=4, memory=1, T=256, trace: cosine=0.9370 +- 0.0000, relative=0.3516, norm=0.8968

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
