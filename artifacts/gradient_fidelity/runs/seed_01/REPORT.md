# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9945 +- 0.0000, relative=0.1050, norm=0.9901
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0083, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9120 +- 0.0000, relative=0.4220, norm=0.8133
- L=1, memory=0, T=4, trace: cosine=0.9999 +- 0.0000, relative=0.0106, norm=0.9999
- L=1, memory=0, T=8, instant: cosine=0.9576 +- 0.0000, relative=0.2882, norm=0.9684
- L=1, memory=0, T=8, trace: cosine=0.9999 +- 0.0000, relative=0.0173, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.9385 +- 0.0000, relative=0.3454, norm=0.9278
- L=1, memory=0, T=16, trace: cosine=0.9999 +- 0.0000, relative=0.0172, norm=0.9999
- L=1, memory=0, T=32, instant: cosine=0.9382 +- 0.0000, relative=0.3467, norm=0.9601
- L=1, memory=0, T=32, trace: cosine=0.9998 +- 0.0000, relative=0.0203, norm=0.9998
- L=1, memory=0, T=64, instant: cosine=0.9059 +- 0.0000, relative=0.4268, norm=0.9585
- L=1, memory=0, T=64, trace: cosine=0.9998 +- 0.0000, relative=0.0199, norm=0.9998
- L=1, memory=0, T=128, instant: cosine=0.9033 +- 0.0000, relative=0.4312, norm=0.9475
- L=1, memory=0, T=128, trace: cosine=0.9998 +- 0.0000, relative=0.0182, norm=0.9998
- L=1, memory=1, T=2, instant: cosine=0.9930 +- 0.0000, relative=0.1204, norm=0.9702
- L=1, memory=1, T=2, trace: cosine=0.9998 +- 0.0000, relative=0.0208, norm=0.9957
- L=1, memory=1, T=4, instant: cosine=0.9811 +- 0.0000, relative=0.1941, norm=0.9939
- L=1, memory=1, T=4, trace: cosine=0.9997 +- 0.0000, relative=0.0264, norm=0.9967
- L=1, memory=1, T=8, instant: cosine=0.9350 +- 0.0000, relative=0.3612, norm=0.8666
- L=1, memory=1, T=8, trace: cosine=0.9997 +- 0.0000, relative=0.0246, norm=0.9943
- L=1, memory=1, T=16, instant: cosine=0.6619 +- 0.0000, relative=0.7763, norm=0.4601
- L=1, memory=1, T=16, trace: cosine=0.9999 +- 0.0000, relative=0.0212, norm=0.9862
- L=1, memory=1, T=32, instant: cosine=0.7381 +- 0.0000, relative=0.6989, norm=0.5557
- L=1, memory=1, T=32, trace: cosine=0.9991 +- 0.0000, relative=0.0592, norm=0.9585
- L=1, memory=1, T=64, instant: cosine=0.8025 +- 0.0000, relative=0.6149, norm=0.6536
- L=1, memory=1, T=64, trace: cosine=0.9973 +- 0.0000, relative=0.0803, norm=0.9659
- L=1, memory=1, T=128, instant: cosine=0.8712 +- 0.0000, relative=0.5071, norm=0.7444
- L=1, memory=1, T=128, trace: cosine=0.9947 +- 0.0000, relative=0.1026, norm=0.9904
- L=2, memory=0, T=2, instant: cosine=0.9987 +- 0.0000, relative=0.0511, norm=1.0003
- L=2, memory=0, T=2, trace: cosine=0.9982 +- 0.0000, relative=0.0610, norm=1.0050
- L=2, memory=0, T=4, instant: cosine=0.9504 +- 0.0000, relative=0.3370, norm=0.8209
- L=2, memory=0, T=4, trace: cosine=0.9919 +- 0.0000, relative=0.1317, norm=0.9572
- L=2, memory=0, T=8, instant: cosine=0.9143 +- 0.0000, relative=0.4052, norm=0.9256
- L=2, memory=0, T=8, trace: cosine=0.9623 +- 0.0000, relative=0.2767, norm=1.0130
- L=2, memory=0, T=16, instant: cosine=0.9508 +- 0.0000, relative=0.3100, norm=0.9429
- L=2, memory=0, T=16, trace: cosine=0.9786 +- 0.0000, relative=0.2063, norm=0.9606
- L=2, memory=0, T=32, instant: cosine=0.8278 +- 0.0000, relative=0.5620, norm=0.7949
- L=2, memory=0, T=32, trace: cosine=0.9495 +- 0.0000, relative=0.3141, norm=0.9323
- L=2, memory=0, T=64, instant: cosine=0.7750 +- 0.0000, relative=0.6349, norm=0.7144
- L=2, memory=0, T=64, trace: cosine=0.9418 +- 0.0000, relative=0.3373, norm=0.9149
- L=2, memory=0, T=128, instant: cosine=0.7887 +- 0.0000, relative=0.6171, norm=0.7358
- L=2, memory=0, T=128, trace: cosine=0.9472 +- 0.0000, relative=0.3213, norm=0.9277
- L=2, memory=1, T=2, instant: cosine=0.9875 +- 0.0000, relative=0.1584, norm=1.0025
- L=2, memory=1, T=2, trace: cosine=0.9877 +- 0.0000, relative=0.1577, norm=1.0077
- L=2, memory=1, T=4, instant: cosine=0.9605 +- 0.0000, relative=0.2786, norm=0.9774
- L=2, memory=1, T=4, trace: cosine=0.9781 +- 0.0000, relative=0.2111, norm=1.0134
- L=2, memory=1, T=8, instant: cosine=0.9263 +- 0.0000, relative=0.3806, norm=0.8731
- L=2, memory=1, T=8, trace: cosine=0.9257 +- 0.0000, relative=0.3785, norm=0.9149
- L=2, memory=1, T=16, instant: cosine=0.9327 +- 0.0000, relative=0.3734, norm=0.8361
- L=2, memory=1, T=16, trace: cosine=0.9468 +- 0.0000, relative=0.3411, norm=1.0602
- L=2, memory=1, T=32, instant: cosine=0.8691 +- 0.0000, relative=0.5025, norm=0.7804
- L=2, memory=1, T=32, trace: cosine=0.9542 +- 0.0000, relative=0.3160, norm=1.0563
- L=2, memory=1, T=64, instant: cosine=0.7850 +- 0.0000, relative=0.6337, norm=0.6515
- L=2, memory=1, T=64, trace: cosine=0.9752 +- 0.0000, relative=0.2228, norm=0.9502
- L=2, memory=1, T=128, instant: cosine=0.7224 +- 0.0000, relative=0.7010, norm=0.6074
- L=2, memory=1, T=128, trace: cosine=0.9790 +- 0.0000, relative=0.2207, norm=0.8943

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
