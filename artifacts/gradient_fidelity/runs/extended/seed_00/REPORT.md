# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=256, instant: cosine=0.7218 +- 0.0000, relative=0.6927, norm=0.6932
- L=1, memory=0, T=256, trace: cosine=1.0000 +- 0.0000, relative=0.0091, norm=1.0000
- L=1, memory=1, T=256, instant: cosine=0.7158 +- 0.0000, relative=0.7157, norm=0.5591
- L=1, memory=1, T=256, trace: cosine=0.9957 +- 0.0000, relative=0.1158, norm=0.9260
- L=2, memory=0, T=256, instant: cosine=0.7605 +- 0.0000, relative=0.6540, norm=0.6826
- L=2, memory=0, T=256, trace: cosine=0.9364 +- 0.0000, relative=0.3889, norm=0.7689
- L=2, memory=1, T=256, instant: cosine=0.6641 +- 0.0000, relative=0.7482, norm=0.6344
- L=2, memory=1, T=256, trace: cosine=0.8326 +- 0.0000, relative=0.5548, norm=0.8005
- L=4, memory=0, T=256, instant: cosine=0.3673 +- 0.0000, relative=0.9305, norm=0.3419
- L=4, memory=0, T=256, trace: cosine=0.5139 +- 0.0000, relative=0.8630, norm=0.6077
- L=4, memory=1, T=256, instant: cosine=0.5008 +- 0.0000, relative=0.8663, norm=0.5371
- L=4, memory=1, T=256, trace: cosine=0.8912 +- 0.0000, relative=0.4541, norm=0.9123

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
