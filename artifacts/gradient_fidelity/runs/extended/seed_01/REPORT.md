# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.8924 +- 0.0000, relative=0.4541, norm=0.9436
- L=1, memory=0, T=256, trace: cosine=0.9998 +- 0.0000, relative=0.0181, norm=0.9998
- L=1, memory=1, T=256, instant: cosine=0.9001 +- 0.0000, relative=0.4493, norm=0.7900
- L=1, memory=1, T=256, trace: cosine=0.9902 +- 0.0000, relative=0.1416, norm=1.0128
- L=2, memory=0, T=256, instant: cosine=0.7798 +- 0.0000, relative=0.6291, norm=0.7183
- L=2, memory=0, T=256, trace: cosine=0.9469 +- 0.0000, relative=0.3226, norm=0.9227
- L=2, memory=1, T=256, instant: cosine=0.7111 +- 0.0000, relative=0.7148, norm=0.5825
- L=2, memory=1, T=256, trace: cosine=0.9794 +- 0.0000, relative=0.2276, norm=0.8748
- L=4, memory=0, T=256, instant: cosine=0.6061 +- 0.0000, relative=0.8115, norm=0.4454
- L=4, memory=0, T=256, trace: cosine=0.9465 +- 0.0000, relative=0.3635, norm=0.7789
- L=4, memory=1, T=256, instant: cosine=0.6580 +- 0.0000, relative=0.7684, norm=0.5050
- L=4, memory=1, T=256, trace: cosine=0.9157 +- 0.0000, relative=0.4106, norm=0.8309

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
