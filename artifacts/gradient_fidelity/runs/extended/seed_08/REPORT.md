# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.8372 +- 0.0000, relative=0.5620, norm=0.7082
- L=1, memory=0, T=256, trace: cosine=1.0000 +- 0.0000, relative=0.0079, norm=1.0000
- L=1, memory=1, T=256, instant: cosine=0.8088 +- 0.0000, relative=0.5910, norm=0.7498
- L=1, memory=1, T=256, trace: cosine=0.9916 +- 0.0000, relative=0.1542, norm=1.0761
- L=2, memory=0, T=256, instant: cosine=0.4986 +- 0.0000, relative=0.8695, norm=0.4314
- L=2, memory=0, T=256, trace: cosine=0.9178 +- 0.0000, relative=0.4310, norm=0.7502
- L=2, memory=1, T=256, instant: cosine=0.6544 +- 0.0000, relative=0.7755, norm=0.4823
- L=2, memory=1, T=256, trace: cosine=0.9520 +- 0.0000, relative=0.3119, norm=1.0124
- L=4, memory=0, T=256, instant: cosine=0.5339 +- 0.0000, relative=0.8859, norm=0.2696
- L=4, memory=0, T=256, trace: cosine=0.9006 +- 0.0000, relative=0.5168, norm=0.6211
- L=4, memory=1, T=256, instant: cosine=0.7330 +- 0.0000, relative=0.6908, norm=0.6127
- L=4, memory=1, T=256, trace: cosine=0.8268 +- 0.0000, relative=0.5632, norm=0.8545

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
