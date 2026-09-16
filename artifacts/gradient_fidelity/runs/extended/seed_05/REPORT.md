# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.8145 +- 0.0000, relative=0.5801, norm=0.8124
- L=1, memory=0, T=256, trace: cosine=0.9997 +- 0.0000, relative=0.0244, norm=0.9999
- L=1, memory=1, T=256, instant: cosine=0.7219 +- 0.0000, relative=0.7681, norm=0.3886
- L=1, memory=1, T=256, trace: cosine=0.9993 +- 0.0000, relative=0.0480, norm=0.9705
- L=2, memory=0, T=256, instant: cosine=0.5457 +- 0.0000, relative=0.8404, norm=0.4816
- L=2, memory=0, T=256, trace: cosine=0.7426 +- 0.0000, relative=0.6757, norm=0.6534
- L=2, memory=1, T=256, instant: cosine=0.3777 +- 0.0000, relative=0.9268, norm=0.3382
- L=2, memory=1, T=256, trace: cosine=0.9230 +- 0.0000, relative=0.4510, norm=0.6877
- L=4, memory=0, T=256, instant: cosine=0.6786 +- 0.0000, relative=0.7449, norm=0.5547
- L=4, memory=0, T=256, trace: cosine=0.8018 +- 0.0000, relative=0.5977, norm=0.7895
- L=4, memory=1, T=256, instant: cosine=0.6584 +- 0.0000, relative=0.7529, norm=0.6789
- L=4, memory=1, T=256, trace: cosine=0.8981 +- 0.0000, relative=0.4513, norm=0.9990

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
