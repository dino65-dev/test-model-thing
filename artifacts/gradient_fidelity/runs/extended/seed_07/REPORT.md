# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.5498 +- 0.0000, relative=0.8378, norm=0.6147
- L=1, memory=0, T=256, trace: cosine=0.9997 +- 0.0000, relative=0.0244, norm=0.9997
- L=1, memory=1, T=256, instant: cosine=0.9417 +- 0.0000, relative=0.3378, norm=0.9122
- L=1, memory=1, T=256, trace: cosine=0.9878 +- 0.0000, relative=0.1621, norm=0.9428
- L=2, memory=0, T=256, instant: cosine=0.6817 +- 0.0000, relative=0.7317, norm=0.6738
- L=2, memory=0, T=256, trace: cosine=0.8680 +- 0.0000, relative=0.5043, norm=0.7800
- L=2, memory=1, T=256, instant: cosine=0.5368 +- 0.0000, relative=0.8745, norm=0.3067
- L=2, memory=1, T=256, trace: cosine=0.9084 +- 0.0000, relative=0.5151, norm=0.6074
- L=4, memory=0, T=256, instant: cosine=0.3332 +- 0.0000, relative=0.9431, norm=0.3129
- L=4, memory=0, T=256, trace: cosine=0.8531 +- 0.0000, relative=0.5555, norm=0.6623
- L=4, memory=1, T=256, instant: cosine=0.3713 +- 0.0000, relative=0.9323, norm=0.2878
- L=4, memory=1, T=256, trace: cosine=0.5317 +- 0.0000, relative=0.8604, norm=0.6832

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
