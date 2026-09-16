# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.9578 +- 0.0000, relative=0.2875, norm=0.9590
- L=1, memory=0, T=256, trace: cosine=0.9997 +- 0.0000, relative=0.0260, norm=1.0001
- L=1, memory=1, T=256, instant: cosine=0.8986 +- 0.0000, relative=0.4446, norm=0.8275
- L=1, memory=1, T=256, trace: cosine=0.9949 +- 0.0000, relative=0.1112, norm=1.0411
- L=2, memory=0, T=256, instant: cosine=0.4770 +- 0.0000, relative=0.8813, norm=0.4123
- L=2, memory=0, T=256, trace: cosine=0.9708 +- 0.0000, relative=0.2919, norm=0.8045
- L=2, memory=1, T=256, instant: cosine=0.8339 +- 0.0000, relative=0.6028, norm=0.5916
- L=2, memory=1, T=256, trace: cosine=0.9554 +- 0.0000, relative=0.3177, norm=1.0730
- L=4, memory=0, T=256, instant: cosine=0.5439 +- 0.0000, relative=0.8578, norm=0.3660
- L=4, memory=0, T=256, trace: cosine=0.7191 +- 0.0000, relative=0.7091, norm=0.5780
- L=4, memory=1, T=256, instant: cosine=0.6363 +- 0.0000, relative=0.7963, norm=0.4389
- L=4, memory=1, T=256, trace: cosine=0.8731 +- 0.0000, relative=0.4898, norm=0.8258

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
