# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9997 +- 0.0000, relative=0.0229, norm=1.0035
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0002, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9963 +- 0.0000, relative=0.0870, norm=0.9852
- L=1, memory=0, T=4, trace: cosine=0.9997 +- 0.0000, relative=0.0228, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.8895 +- 0.0000, relative=0.4679, norm=0.7886
- L=1, memory=0, T=8, trace: cosine=0.9999 +- 0.0000, relative=0.0109, norm=1.0001
- L=1, memory=0, T=16, instant: cosine=0.8128 +- 0.0000, relative=0.5954, norm=0.6899
- L=1, memory=0, T=16, trace: cosine=1.0000 +- 0.0000, relative=0.0071, norm=1.0000
- L=1, memory=0, T=32, instant: cosine=0.9131 +- 0.0000, relative=0.4078, norm=0.9107
- L=1, memory=0, T=32, trace: cosine=1.0000 +- 0.0000, relative=0.0046, norm=1.0000
- L=1, memory=0, T=64, instant: cosine=0.7739 +- 0.0000, relative=0.6334, norm=0.7639
- L=1, memory=0, T=64, trace: cosine=0.9999 +- 0.0000, relative=0.0100, norm=1.0000
- L=1, memory=0, T=128, instant: cosine=0.7515 +- 0.0000, relative=0.6599, norm=0.7360
- L=1, memory=0, T=128, trace: cosine=1.0000 +- 0.0000, relative=0.0077, norm=1.0000
- L=1, memory=1, T=2, instant: cosine=0.9943 +- 0.0000, relative=0.1062, norm=0.9949
- L=1, memory=1, T=2, trace: cosine=0.9995 +- 0.0000, relative=0.0322, norm=0.9992
- L=1, memory=1, T=4, instant: cosine=0.9866 +- 0.0000, relative=0.1633, norm=0.9907
- L=1, memory=1, T=4, trace: cosine=0.9994 +- 0.0000, relative=0.0340, norm=1.0027
- L=1, memory=1, T=8, instant: cosine=0.9544 +- 0.0000, relative=0.3073, norm=0.8813
- L=1, memory=1, T=8, trace: cosine=0.9988 +- 0.0000, relative=0.0486, norm=1.0002
- L=1, memory=1, T=16, instant: cosine=0.9835 +- 0.0000, relative=0.1808, norm=0.9752
- L=1, memory=1, T=16, trace: cosine=0.9995 +- 0.0000, relative=0.0318, norm=0.9980
- L=1, memory=1, T=32, instant: cosine=0.8298 +- 0.0000, relative=0.5633, norm=0.7535
- L=1, memory=1, T=32, trace: cosine=0.9993 +- 0.0000, relative=0.0390, norm=1.0120
- L=1, memory=1, T=64, instant: cosine=0.7435 +- 0.0000, relative=0.6783, norm=0.6302
- L=1, memory=1, T=64, trace: cosine=0.9992 +- 0.0000, relative=0.0422, norm=0.9872
- L=1, memory=1, T=128, instant: cosine=0.7397 +- 0.0000, relative=0.6906, norm=0.5847
- L=1, memory=1, T=128, trace: cosine=0.9980 +- 0.0000, relative=0.0780, norm=0.9515
- L=2, memory=0, T=2, instant: cosine=0.9863 +- 0.0000, relative=0.1698, norm=0.9452
- L=2, memory=0, T=2, trace: cosine=0.9917 +- 0.0000, relative=0.1302, norm=0.9723
- L=2, memory=0, T=4, instant: cosine=0.9555 +- 0.0000, relative=0.2962, norm=0.9280
- L=2, memory=0, T=4, trace: cosine=0.9815 +- 0.0000, relative=0.1917, norm=0.9748
- L=2, memory=0, T=8, instant: cosine=0.9217 +- 0.0000, relative=0.4057, norm=0.8025
- L=2, memory=0, T=8, trace: cosine=0.9654 +- 0.0000, relative=0.2611, norm=0.9557
- L=2, memory=0, T=16, instant: cosine=0.7247 +- 0.0000, relative=0.6943, norm=0.6399
- L=2, memory=0, T=16, trace: cosine=0.9409 +- 0.0000, relative=0.3878, norm=0.7521
- L=2, memory=0, T=32, instant: cosine=0.6753 +- 0.0000, relative=0.7384, norm=0.6393
- L=2, memory=0, T=32, trace: cosine=0.9157 +- 0.0000, relative=0.4323, norm=0.7562
- L=2, memory=0, T=64, instant: cosine=0.7666 +- 0.0000, relative=0.6490, norm=0.6728
- L=2, memory=0, T=64, trace: cosine=0.9395 +- 0.0000, relative=0.3797, norm=0.7757
- L=2, memory=0, T=128, instant: cosine=0.7431 +- 0.0000, relative=0.6740, norm=0.6620
- L=2, memory=0, T=128, trace: cosine=0.9401 +- 0.0000, relative=0.3841, norm=0.7630
- L=2, memory=1, T=2, instant: cosine=0.9904 +- 0.0000, relative=0.1380, norm=0.9943
- L=2, memory=1, T=2, trace: cosine=0.9977 +- 0.0000, relative=0.0703, norm=1.0170
- L=2, memory=1, T=4, instant: cosine=0.9337 +- 0.0000, relative=0.3583, norm=0.9217
- L=2, memory=1, T=4, trace: cosine=0.9888 +- 0.0000, relative=0.1490, norm=0.9912
- L=2, memory=1, T=8, instant: cosine=0.9072 +- 0.0000, relative=0.4250, norm=0.8463
- L=2, memory=1, T=8, trace: cosine=0.9829 +- 0.0000, relative=0.1850, norm=0.9633
- L=2, memory=1, T=16, instant: cosine=0.6044 +- 0.0000, relative=0.7967, norm=0.6107
- L=2, memory=1, T=16, trace: cosine=0.9205 +- 0.0000, relative=0.3914, norm=0.9434
- L=2, memory=1, T=32, instant: cosine=0.6333 +- 0.0000, relative=0.7740, norm=0.6475
- L=2, memory=1, T=32, trace: cosine=0.8887 +- 0.0000, relative=0.4610, norm=0.8401
- L=2, memory=1, T=64, instant: cosine=0.6899 +- 0.0000, relative=0.7240, norm=0.6792
- L=2, memory=1, T=64, trace: cosine=0.8641 +- 0.0000, relative=0.5044, norm=0.8313
- L=2, memory=1, T=128, instant: cosine=0.6538 +- 0.0000, relative=0.7570, norm=0.6315
- L=2, memory=1, T=128, trace: cosine=0.8468 +- 0.0000, relative=0.5340, norm=0.7998

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
