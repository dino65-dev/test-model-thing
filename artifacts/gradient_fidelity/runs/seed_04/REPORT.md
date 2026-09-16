# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9910 +- 0.0000, relative=0.1340, norm=0.9924
- L=1, memory=0, T=2, trace: cosine=0.9997 +- 0.0000, relative=0.0248, norm=0.9999
- L=1, memory=0, T=4, instant: cosine=0.9958 +- 0.0000, relative=0.0916, norm=1.0029
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0053, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.9660 +- 0.0000, relative=0.2593, norm=0.9459
- L=1, memory=0, T=8, trace: cosine=0.9999 +- 0.0000, relative=0.0103, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.7802 +- 0.0000, relative=0.6318, norm=0.6914
- L=1, memory=0, T=16, trace: cosine=0.9999 +- 0.0000, relative=0.0102, norm=1.0000
- L=1, memory=0, T=32, instant: cosine=0.7588 +- 0.0000, relative=0.6536, norm=0.7032
- L=1, memory=0, T=32, trace: cosine=1.0000 +- 0.0000, relative=0.0057, norm=1.0000
- L=1, memory=0, T=64, instant: cosine=0.7954 +- 0.0000, relative=0.6082, norm=0.7451
- L=1, memory=0, T=64, trace: cosine=1.0000 +- 0.0000, relative=0.0083, norm=0.9999
- L=1, memory=0, T=128, instant: cosine=0.7945 +- 0.0000, relative=0.6097, norm=0.7409
- L=1, memory=0, T=128, trace: cosine=0.9999 +- 0.0000, relative=0.0109, norm=0.9999
- L=1, memory=1, T=2, instant: cosine=0.9965 +- 0.0000, relative=0.0841, norm=0.9870
- L=1, memory=1, T=2, trace: cosine=0.9998 +- 0.0000, relative=0.0196, norm=0.9997
- L=1, memory=1, T=4, instant: cosine=0.9780 +- 0.0000, relative=0.2088, norm=0.9793
- L=1, memory=1, T=4, trace: cosine=0.9972 +- 0.0000, relative=0.0750, norm=1.0011
- L=1, memory=1, T=8, instant: cosine=0.9395 +- 0.0000, relative=0.3427, norm=0.9445
- L=1, memory=1, T=8, trace: cosine=0.9991 +- 0.0000, relative=0.0427, norm=0.9988
- L=1, memory=1, T=16, instant: cosine=0.9818 +- 0.0000, relative=0.1900, norm=0.9858
- L=1, memory=1, T=16, trace: cosine=0.9992 +- 0.0000, relative=0.0412, norm=0.9997
- L=1, memory=1, T=32, instant: cosine=0.8688 +- 0.0000, relative=0.5072, norm=0.7590
- L=1, memory=1, T=32, trace: cosine=0.9984 +- 0.0000, relative=0.0608, norm=1.0211
- L=1, memory=1, T=64, instant: cosine=0.8104 +- 0.0000, relative=0.6116, norm=0.6351
- L=1, memory=1, T=64, trace: cosine=0.9975 +- 0.0000, relative=0.0794, norm=1.0340
- L=1, memory=1, T=128, instant: cosine=0.7945 +- 0.0000, relative=0.6325, norm=0.6174
- L=1, memory=1, T=128, trace: cosine=0.9949 +- 0.0000, relative=0.1119, norm=1.0434
- L=2, memory=0, T=2, instant: cosine=0.9723 +- 0.0000, relative=0.2364, norm=0.9376
- L=2, memory=0, T=2, trace: cosine=0.9851 +- 0.0000, relative=0.1761, norm=0.9470
- L=2, memory=0, T=4, instant: cosine=0.9659 +- 0.0000, relative=0.2589, norm=0.9621
- L=2, memory=0, T=4, trace: cosine=0.9925 +- 0.0000, relative=0.1226, norm=0.9806
- L=2, memory=0, T=8, instant: cosine=0.8913 +- 0.0000, relative=0.4541, norm=0.9156
- L=2, memory=0, T=8, trace: cosine=0.9916 +- 0.0000, relative=0.1338, norm=1.0255
- L=2, memory=0, T=16, instant: cosine=0.8311 +- 0.0000, relative=0.5962, norm=0.6162
- L=2, memory=0, T=16, trace: cosine=0.9553 +- 0.0000, relative=0.3100, norm=0.8618
- L=2, memory=0, T=32, instant: cosine=0.7993 +- 0.0000, relative=0.6197, norm=0.6477
- L=2, memory=0, T=32, trace: cosine=0.9594 +- 0.0000, relative=0.3203, norm=0.8079
- L=2, memory=0, T=64, instant: cosine=0.6722 +- 0.0000, relative=0.7510, norm=0.5459
- L=2, memory=0, T=64, trace: cosine=0.9748 +- 0.0000, relative=0.2901, norm=0.7894
- L=2, memory=0, T=128, instant: cosine=0.6657 +- 0.0000, relative=0.7639, norm=0.5022
- L=2, memory=0, T=128, trace: cosine=0.9726 +- 0.0000, relative=0.3172, norm=0.7566
- L=2, memory=1, T=2, instant: cosine=0.9646 +- 0.0000, relative=0.2637, norm=0.9578
- L=2, memory=1, T=2, trace: cosine=0.9958 +- 0.0000, relative=0.0933, norm=0.9757
- L=2, memory=1, T=4, instant: cosine=0.9699 +- 0.0000, relative=0.2434, norm=0.9628
- L=2, memory=1, T=4, trace: cosine=0.9932 +- 0.0000, relative=0.1177, norm=0.9788
- L=2, memory=1, T=8, instant: cosine=0.5903 +- 0.0000, relative=0.8121, norm=0.5013
- L=2, memory=1, T=8, trace: cosine=0.9931 +- 0.0000, relative=0.1324, norm=0.9311
- L=2, memory=1, T=16, instant: cosine=0.7203 +- 0.0000, relative=0.7111, norm=0.5636
- L=2, memory=1, T=16, trace: cosine=0.9852 +- 0.0000, relative=0.2169, norm=1.1184
- L=2, memory=1, T=32, instant: cosine=0.6796 +- 0.0000, relative=0.7365, norm=0.6139
- L=2, memory=1, T=32, trace: cosine=0.9769 +- 0.0000, relative=0.2740, norm=1.1481
- L=2, memory=1, T=64, instant: cosine=0.6943 +- 0.0000, relative=0.7243, norm=0.6127
- L=2, memory=1, T=64, trace: cosine=0.9723 +- 0.0000, relative=0.3035, norm=1.1660
- L=2, memory=1, T=128, instant: cosine=0.7048 +- 0.0000, relative=0.7154, norm=0.6129
- L=2, memory=1, T=128, trace: cosine=0.9603 +- 0.0000, relative=0.3435, norm=1.1605

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
