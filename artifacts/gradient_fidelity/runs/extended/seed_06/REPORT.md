# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.6058 +- 0.0000, relative=0.7958, norm=0.5886
- L=1, memory=0, T=256, trace: cosine=0.9993 +- 0.0000, relative=0.0371, norm=0.9992
- L=1, memory=1, T=256, instant: cosine=0.7461 +- 0.0000, relative=0.6864, norm=0.5795
- L=1, memory=1, T=256, trace: cosine=0.9977 +- 0.0000, relative=0.0875, norm=0.9425
- L=2, memory=0, T=256, instant: cosine=0.6866 +- 0.0000, relative=0.7281, norm=0.6459
- L=2, memory=0, T=256, trace: cosine=0.8810 +- 0.0000, relative=0.4800, norm=0.9616
- L=2, memory=1, T=256, instant: cosine=0.6645 +- 0.0000, relative=0.7478, norm=0.6906
- L=2, memory=1, T=256, trace: cosine=0.9746 +- 0.0000, relative=0.2306, norm=1.0299
- L=4, memory=0, T=256, instant: cosine=0.5266 +- 0.0000, relative=0.8713, norm=0.3358
- L=4, memory=0, T=256, trace: cosine=0.8625 +- 0.0000, relative=0.5204, norm=0.7412
- L=4, memory=1, T=256, instant: cosine=0.6911 +- 0.0000, relative=0.7561, norm=0.4692
- L=4, memory=1, T=256, trace: cosine=0.7649 +- 0.0000, relative=0.6445, norm=0.7883

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
