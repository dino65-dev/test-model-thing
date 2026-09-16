# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9943 +- 0.0000, relative=0.1081, norm=0.9786
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0062, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9929 +- 0.0000, relative=0.1194, norm=0.9871
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0065, norm=1.0001
- L=1, memory=0, T=8, instant: cosine=0.9160 +- 0.0000, relative=0.4084, norm=0.8397
- L=1, memory=0, T=8, trace: cosine=0.9996 +- 0.0000, relative=0.0289, norm=0.9996
- L=1, memory=0, T=16, instant: cosine=0.8297 +- 0.0000, relative=0.5660, norm=0.7363
- L=1, memory=0, T=16, trace: cosine=0.9998 +- 0.0000, relative=0.0222, norm=0.9998
- L=1, memory=0, T=32, instant: cosine=0.8758 +- 0.0000, relative=0.4872, norm=0.8089
- L=1, memory=0, T=32, trace: cosine=0.9996 +- 0.0000, relative=0.0271, norm=0.9996
- L=1, memory=0, T=64, instant: cosine=0.8955 +- 0.0000, relative=0.4488, norm=0.8380
- L=1, memory=0, T=64, trace: cosine=0.9996 +- 0.0000, relative=0.0297, norm=0.9996
- L=1, memory=0, T=128, instant: cosine=0.8799 +- 0.0000, relative=0.4776, norm=0.8328
- L=1, memory=0, T=128, trace: cosine=0.9996 +- 0.0000, relative=0.0298, norm=0.9996
- L=1, memory=1, T=2, instant: cosine=0.9945 +- 0.0000, relative=0.1047, norm=0.9866
- L=1, memory=1, T=2, trace: cosine=0.9999 +- 0.0000, relative=0.0124, norm=1.0001
- L=1, memory=1, T=4, instant: cosine=0.9881 +- 0.0000, relative=0.1541, norm=0.9794
- L=1, memory=1, T=4, trace: cosine=0.9998 +- 0.0000, relative=0.0184, norm=0.9998
- L=1, memory=1, T=8, instant: cosine=0.9368 +- 0.0000, relative=0.3542, norm=0.8815
- L=1, memory=1, T=8, trace: cosine=0.9999 +- 0.0000, relative=0.0164, norm=0.9993
- L=1, memory=1, T=16, instant: cosine=0.9472 +- 0.0000, relative=0.3221, norm=0.9169
- L=1, memory=1, T=16, trace: cosine=0.9989 +- 0.0000, relative=0.0467, norm=0.9970
- L=1, memory=1, T=32, instant: cosine=0.8625 +- 0.0000, relative=0.5116, norm=0.7871
- L=1, memory=1, T=32, trace: cosine=0.9985 +- 0.0000, relative=0.0561, norm=0.9896
- L=1, memory=1, T=64, instant: cosine=0.8813 +- 0.0000, relative=0.4792, norm=0.8017
- L=1, memory=1, T=64, trace: cosine=0.9930 +- 0.0000, relative=0.1188, norm=0.9792
- L=1, memory=1, T=128, instant: cosine=0.8720 +- 0.0000, relative=0.4949, norm=0.7997
- L=1, memory=1, T=128, trace: cosine=0.9846 +- 0.0000, relative=0.1759, norm=0.9649
- L=2, memory=0, T=2, instant: cosine=0.9988 +- 0.0000, relative=0.0490, norm=0.9957
- L=2, memory=0, T=2, trace: cosine=0.9989 +- 0.0000, relative=0.0478, norm=0.9971
- L=2, memory=0, T=4, instant: cosine=0.9926 +- 0.0000, relative=0.1220, norm=0.9837
- L=2, memory=0, T=4, trace: cosine=0.9950 +- 0.0000, relative=0.0994, norm=0.9948
- L=2, memory=0, T=8, instant: cosine=0.5281 +- 0.0000, relative=0.8562, norm=0.4191
- L=2, memory=0, T=8, trace: cosine=0.9460 +- 0.0000, relative=0.3740, norm=0.7595
- L=2, memory=0, T=16, instant: cosine=0.4501 +- 0.0000, relative=0.8975, norm=0.5400
- L=2, memory=0, T=16, trace: cosine=0.8831 +- 0.0000, relative=0.5329, norm=0.6303
- L=2, memory=0, T=32, instant: cosine=0.2487 +- 0.0000, relative=0.9689, norm=0.2738
- L=2, memory=0, T=32, trace: cosine=0.9118 +- 0.0000, relative=0.5700, norm=0.5164
- L=2, memory=0, T=64, instant: cosine=0.2449 +- 0.0000, relative=0.9705, norm=0.2879
- L=2, memory=0, T=64, trace: cosine=0.9068 +- 0.0000, relative=0.5908, norm=0.4930
- L=2, memory=0, T=128, instant: cosine=0.1810 +- 0.0000, relative=0.9850, norm=0.2356
- L=2, memory=0, T=128, trace: cosine=0.9087 +- 0.0000, relative=0.5981, norm=0.4805
- L=2, memory=1, T=2, instant: cosine=0.9926 +- 0.0000, relative=0.1250, norm=0.9623
- L=2, memory=1, T=2, trace: cosine=0.9915 +- 0.0000, relative=0.1345, norm=0.9578
- L=2, memory=1, T=4, instant: cosine=0.9822 +- 0.0000, relative=0.1893, norm=0.9569
- L=2, memory=1, T=4, trace: cosine=0.9816 +- 0.0000, relative=0.1917, norm=0.9651
- L=2, memory=1, T=8, instant: cosine=0.8986 +- 0.0000, relative=0.4618, norm=0.7546
- L=2, memory=1, T=8, trace: cosine=0.9822 +- 0.0000, relative=0.1879, norm=0.9718
- L=2, memory=1, T=16, instant: cosine=0.9262 +- 0.0000, relative=0.3882, norm=0.8341
- L=2, memory=1, T=16, trace: cosine=0.9566 +- 0.0000, relative=0.2916, norm=0.9581
- L=2, memory=1, T=32, instant: cosine=0.7503 +- 0.0000, relative=0.6624, norm=0.7089
- L=2, memory=1, T=32, trace: cosine=0.9294 +- 0.0000, relative=0.3818, norm=0.8320
- L=2, memory=1, T=64, instant: cosine=0.6000 +- 0.0000, relative=0.8015, norm=0.5505
- L=2, memory=1, T=64, trace: cosine=0.9407 +- 0.0000, relative=0.3740, norm=0.7834
- L=2, memory=1, T=128, instant: cosine=0.5921 +- 0.0000, relative=0.8066, norm=0.5578
- L=2, memory=1, T=128, trace: cosine=0.9339 +- 0.0000, relative=0.3887, norm=0.7814

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
