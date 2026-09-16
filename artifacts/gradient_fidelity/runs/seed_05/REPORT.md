# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9902 +- 0.0000, relative=0.1448, norm=1.0292
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0068, norm=1.0001
- L=1, memory=0, T=4, instant: cosine=0.9497 +- 0.0000, relative=0.3132, norm=0.9455
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0013, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.9627 +- 0.0000, relative=0.2705, norm=0.9549
- L=1, memory=0, T=8, trace: cosine=0.9996 +- 0.0000, relative=0.0265, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.8577 +- 0.0000, relative=0.5200, norm=0.7798
- L=1, memory=0, T=16, trace: cosine=0.9999 +- 0.0000, relative=0.0165, norm=0.9998
- L=1, memory=0, T=32, instant: cosine=0.8361 +- 0.0000, relative=0.5488, norm=0.8196
- L=1, memory=0, T=32, trace: cosine=0.9998 +- 0.0000, relative=0.0208, norm=0.9999
- L=1, memory=0, T=64, instant: cosine=0.8519 +- 0.0000, relative=0.5238, norm=0.8459
- L=1, memory=0, T=64, trace: cosine=0.9998 +- 0.0000, relative=0.0223, norm=0.9999
- L=1, memory=0, T=128, instant: cosine=0.8216 +- 0.0000, relative=0.5701, norm=0.8158
- L=1, memory=0, T=128, trace: cosine=0.9997 +- 0.0000, relative=0.0244, norm=0.9999
- L=1, memory=1, T=2, instant: cosine=0.9933 +- 0.0000, relative=0.1229, norm=1.0339
- L=1, memory=1, T=2, trace: cosine=0.9996 +- 0.0000, relative=0.0297, norm=1.0102
- L=1, memory=1, T=4, instant: cosine=0.9856 +- 0.0000, relative=0.1692, norm=0.9905
- L=1, memory=1, T=4, trace: cosine=0.9994 +- 0.0000, relative=0.0356, norm=1.0104
- L=1, memory=1, T=8, instant: cosine=0.9380 +- 0.0000, relative=0.3623, norm=0.8325
- L=1, memory=1, T=8, trace: cosine=0.9990 +- 0.0000, relative=0.0459, norm=1.0130
- L=1, memory=1, T=16, instant: cosine=0.7577 +- 0.0000, relative=0.6795, norm=0.5683
- L=1, memory=1, T=16, trace: cosine=0.9996 +- 0.0000, relative=0.0295, norm=1.0043
- L=1, memory=1, T=32, instant: cosine=0.7027 +- 0.0000, relative=0.7511, norm=0.4620
- L=1, memory=1, T=32, trace: cosine=0.9996 +- 0.0000, relative=0.0298, norm=0.9891
- L=1, memory=1, T=64, instant: cosine=0.6991 +- 0.0000, relative=0.7777, norm=0.3932
- L=1, memory=1, T=64, trace: cosine=0.9997 +- 0.0000, relative=0.0309, norm=0.9828
- L=1, memory=1, T=128, instant: cosine=0.7194 +- 0.0000, relative=0.7667, norm=0.3947
- L=1, memory=1, T=128, trace: cosine=0.9995 +- 0.0000, relative=0.0420, norm=0.9712
- L=2, memory=0, T=2, instant: cosine=0.9888 +- 0.0000, relative=0.1998, norm=1.1215
- L=2, memory=0, T=2, trace: cosine=0.9896 +- 0.0000, relative=0.1730, norm=1.0851
- L=2, memory=0, T=4, instant: cosine=0.9686 +- 0.0000, relative=0.3010, norm=1.1382
- L=2, memory=0, T=4, trace: cosine=0.9898 +- 0.0000, relative=0.1725, norm=1.0872
- L=2, memory=0, T=8, instant: cosine=0.8426 +- 0.0000, relative=0.5533, norm=0.9694
- L=2, memory=0, T=8, trace: cosine=0.8770 +- 0.0000, relative=0.5358, norm=1.1142
- L=2, memory=0, T=16, instant: cosine=0.7754 +- 0.0000, relative=0.6322, norm=0.7445
- L=2, memory=0, T=16, trace: cosine=0.9502 +- 0.0000, relative=0.3117, norm=0.9550
- L=2, memory=0, T=32, instant: cosine=0.7487 +- 0.0000, relative=0.6646, norm=0.7012
- L=2, memory=0, T=32, trace: cosine=0.8513 +- 0.0000, relative=0.5322, norm=0.7621
- L=2, memory=0, T=64, instant: cosine=0.6271 +- 0.0000, relative=0.7817, norm=0.5618
- L=2, memory=0, T=64, trace: cosine=0.7818 +- 0.0000, relative=0.6339, norm=0.6676
- L=2, memory=0, T=128, instant: cosine=0.5669 +- 0.0000, relative=0.8261, norm=0.5050
- L=2, memory=0, T=128, trace: cosine=0.7525 +- 0.0000, relative=0.6669, norm=0.6473
- L=2, memory=1, T=2, instant: cosine=0.9900 +- 0.0000, relative=0.1478, norm=1.0350
- L=2, memory=1, T=2, trace: cosine=0.9878 +- 0.0000, relative=0.1629, norm=1.0352
- L=2, memory=1, T=4, instant: cosine=0.9574 +- 0.0000, relative=0.2915, norm=0.9959
- L=2, memory=1, T=4, trace: cosine=0.9661 +- 0.0000, relative=0.2667, norm=1.0324
- L=2, memory=1, T=8, instant: cosine=0.9390 +- 0.0000, relative=0.3439, norm=0.9375
- L=2, memory=1, T=8, trace: cosine=0.9618 +- 0.0000, relative=0.2794, norm=1.0168
- L=2, memory=1, T=16, instant: cosine=0.6171 +- 0.0000, relative=0.7901, norm=0.5461
- L=2, memory=1, T=16, trace: cosine=0.9363 +- 0.0000, relative=0.3725, norm=0.8121
- L=2, memory=1, T=32, instant: cosine=0.4729 +- 0.0000, relative=0.8818, norm=0.4381
- L=2, memory=1, T=32, trace: cosine=0.9228 +- 0.0000, relative=0.4206, norm=0.7540
- L=2, memory=1, T=64, instant: cosine=0.4077 +- 0.0000, relative=0.9136, norm=0.3761
- L=2, memory=1, T=64, trace: cosine=0.9238 +- 0.0000, relative=0.4364, norm=0.7144
- L=2, memory=1, T=128, instant: cosine=0.4014 +- 0.0000, relative=0.9176, norm=0.3460
- L=2, memory=1, T=128, trace: cosine=0.9229 +- 0.0000, relative=0.4440, norm=0.7018

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
