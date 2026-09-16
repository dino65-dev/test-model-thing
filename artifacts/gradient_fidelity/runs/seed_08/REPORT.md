# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9994 +- 0.0000, relative=0.0336, norm=1.0028
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0030, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9916 +- 0.0000, relative=0.1291, norm=0.9944
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0070, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.9436 +- 0.0000, relative=0.3315, norm=0.9592
- L=1, memory=0, T=8, trace: cosine=1.0000 +- 0.0000, relative=0.0084, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.8579 +- 0.0000, relative=0.5335, norm=0.7146
- L=1, memory=0, T=16, trace: cosine=0.9999 +- 0.0000, relative=0.0136, norm=0.9999
- L=1, memory=0, T=32, instant: cosine=0.8845 +- 0.0000, relative=0.4731, norm=0.8056
- L=1, memory=0, T=32, trace: cosine=1.0000 +- 0.0000, relative=0.0056, norm=1.0000
- L=1, memory=0, T=64, instant: cosine=0.8449 +- 0.0000, relative=0.5473, norm=0.7289
- L=1, memory=0, T=64, trace: cosine=1.0000 +- 0.0000, relative=0.0079, norm=1.0000
- L=1, memory=0, T=128, instant: cosine=0.8389 +- 0.0000, relative=0.5591, norm=0.7112
- L=1, memory=0, T=128, trace: cosine=1.0000 +- 0.0000, relative=0.0062, norm=1.0000
- L=1, memory=1, T=2, instant: cosine=0.9928 +- 0.0000, relative=0.1203, norm=1.0067
- L=1, memory=1, T=2, trace: cosine=0.9995 +- 0.0000, relative=0.0315, norm=0.9997
- L=1, memory=1, T=4, instant: cosine=0.9874 +- 0.0000, relative=0.1592, norm=1.0032
- L=1, memory=1, T=4, trace: cosine=0.9998 +- 0.0000, relative=0.0186, norm=0.9998
- L=1, memory=1, T=8, instant: cosine=0.9150 +- 0.0000, relative=0.4037, norm=0.9001
- L=1, memory=1, T=8, trace: cosine=0.9987 +- 0.0000, relative=0.0506, norm=0.9997
- L=1, memory=1, T=16, instant: cosine=0.9605 +- 0.0000, relative=0.2801, norm=0.9274
- L=1, memory=1, T=16, trace: cosine=0.9982 +- 0.0000, relative=0.0610, norm=0.9901
- L=1, memory=1, T=32, instant: cosine=0.9019 +- 0.0000, relative=0.4322, norm=0.8882
- L=1, memory=1, T=32, trace: cosine=0.9965 +- 0.0000, relative=0.0846, norm=1.0058
- L=1, memory=1, T=64, instant: cosine=0.8282 +- 0.0000, relative=0.5619, norm=0.7869
- L=1, memory=1, T=64, trace: cosine=0.9933 +- 0.0000, relative=0.1228, norm=1.0345
- L=1, memory=1, T=128, instant: cosine=0.8134 +- 0.0000, relative=0.5840, norm=0.7616
- L=1, memory=1, T=128, trace: cosine=0.9925 +- 0.0000, relative=0.1407, norm=1.0629
- L=2, memory=0, T=2, instant: cosine=0.9975 +- 0.0000, relative=0.0711, norm=0.9955
- L=2, memory=0, T=2, trace: cosine=0.9995 +- 0.0000, relative=0.0325, norm=0.9979
- L=2, memory=0, T=4, instant: cosine=0.9435 +- 0.0000, relative=0.3337, norm=0.9048
- L=2, memory=0, T=4, trace: cosine=0.9550 +- 0.0000, relative=0.3002, norm=0.9083
- L=2, memory=0, T=8, instant: cosine=0.8295 +- 0.0000, relative=0.5729, norm=0.7022
- L=2, memory=0, T=8, trace: cosine=0.9433 +- 0.0000, relative=0.3709, norm=0.7781
- L=2, memory=0, T=16, instant: cosine=0.5991 +- 0.0000, relative=0.8142, norm=0.4512
- L=2, memory=0, T=16, trace: cosine=0.9138 +- 0.0000, relative=0.4201, norm=0.8066
- L=2, memory=0, T=32, instant: cosine=0.6110 +- 0.0000, relative=0.7956, norm=0.5317
- L=2, memory=0, T=32, trace: cosine=0.9387 +- 0.0000, relative=0.3608, norm=0.8322
- L=2, memory=0, T=64, instant: cosine=0.5433 +- 0.0000, relative=0.8423, norm=0.4749
- L=2, memory=0, T=64, trace: cosine=0.9220 +- 0.0000, relative=0.4118, norm=0.7816
- L=2, memory=0, T=128, instant: cosine=0.5110 +- 0.0000, relative=0.8636, norm=0.4276
- L=2, memory=0, T=128, trace: cosine=0.9232 +- 0.0000, relative=0.4183, norm=0.7582
- L=2, memory=1, T=2, instant: cosine=0.9900 +- 0.0000, relative=0.1430, norm=1.0125
- L=2, memory=1, T=2, trace: cosine=0.9829 +- 0.0000, relative=0.1860, norm=1.0079
- L=2, memory=1, T=4, instant: cosine=0.9571 +- 0.0000, relative=0.3060, norm=0.8592
- L=2, memory=1, T=4, trace: cosine=0.9821 +- 0.0000, relative=0.2080, norm=0.8937
- L=2, memory=1, T=8, instant: cosine=0.9515 +- 0.0000, relative=0.3227, norm=0.8543
- L=2, memory=1, T=8, trace: cosine=0.9687 +- 0.0000, relative=0.2540, norm=0.9154
- L=2, memory=1, T=16, instant: cosine=0.7500 +- 0.0000, relative=0.6805, norm=0.5898
- L=2, memory=1, T=16, trace: cosine=0.9647 +- 0.0000, relative=0.2818, norm=0.8643
- L=2, memory=1, T=32, instant: cosine=0.6994 +- 0.0000, relative=0.7361, norm=0.5231
- L=2, memory=1, T=32, trace: cosine=0.9710 +- 0.0000, relative=0.2397, norm=0.9548
- L=2, memory=1, T=64, instant: cosine=0.6607 +- 0.0000, relative=0.7677, norm=0.4998
- L=2, memory=1, T=64, trace: cosine=0.9716 +- 0.0000, relative=0.2380, norm=0.9982
- L=2, memory=1, T=128, instant: cosine=0.6583 +- 0.0000, relative=0.7731, norm=0.4821
- L=2, memory=1, T=128, trace: cosine=0.9650 +- 0.0000, relative=0.2649, norm=1.0032

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
