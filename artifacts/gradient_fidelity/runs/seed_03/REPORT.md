# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9930 +- 0.0000, relative=0.1199, norm=0.9732
- L=1, memory=0, T=2, trace: cosine=0.9998 +- 0.0000, relative=0.0173, norm=0.9998
- L=1, memory=0, T=4, instant: cosine=0.9920 +- 0.0000, relative=0.1265, norm=0.9979
- L=1, memory=0, T=4, trace: cosine=0.9999 +- 0.0000, relative=0.0127, norm=0.9999
- L=1, memory=0, T=8, instant: cosine=0.8738 +- 0.0000, relative=0.5002, norm=0.7564
- L=1, memory=0, T=8, trace: cosine=0.9998 +- 0.0000, relative=0.0204, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.9349 +- 0.0000, relative=0.3616, norm=0.8654
- L=1, memory=0, T=16, trace: cosine=0.9997 +- 0.0000, relative=0.0254, norm=1.0002
- L=1, memory=0, T=32, instant: cosine=0.9694 +- 0.0000, relative=0.2458, norm=0.9591
- L=1, memory=0, T=32, trace: cosine=0.9994 +- 0.0000, relative=0.0350, norm=1.0003
- L=1, memory=0, T=64, instant: cosine=0.9842 +- 0.0000, relative=0.1773, norm=0.9907
- L=1, memory=0, T=64, trace: cosine=0.9994 +- 0.0000, relative=0.0348, norm=1.0003
- L=1, memory=0, T=128, instant: cosine=0.9809 +- 0.0000, relative=0.1946, norm=0.9808
- L=1, memory=0, T=128, trace: cosine=0.9993 +- 0.0000, relative=0.0366, norm=1.0003
- L=1, memory=1, T=2, instant: cosine=0.9913 +- 0.0000, relative=0.1319, norm=1.0015
- L=1, memory=1, T=2, trace: cosine=0.9971 +- 0.0000, relative=0.0757, norm=1.0027
- L=1, memory=1, T=4, instant: cosine=0.9825 +- 0.0000, relative=0.1865, norm=0.9827
- L=1, memory=1, T=4, trace: cosine=0.9981 +- 0.0000, relative=0.0611, norm=1.0011
- L=1, memory=1, T=8, instant: cosine=0.9845 +- 0.0000, relative=0.1753, norm=0.9784
- L=1, memory=1, T=8, trace: cosine=0.9989 +- 0.0000, relative=0.0471, norm=1.0031
- L=1, memory=1, T=16, instant: cosine=0.8978 +- 0.0000, relative=0.4436, norm=0.8454
- L=1, memory=1, T=16, trace: cosine=0.9987 +- 0.0000, relative=0.0516, norm=1.0070
- L=1, memory=1, T=32, instant: cosine=0.8896 +- 0.0000, relative=0.4616, norm=0.8231
- L=1, memory=1, T=32, trace: cosine=0.9979 +- 0.0000, relative=0.0652, norm=0.9858
- L=1, memory=1, T=64, instant: cosine=0.9816 +- 0.0000, relative=0.1913, norm=0.9873
- L=1, memory=1, T=64, trace: cosine=0.9979 +- 0.0000, relative=0.0652, norm=1.0033
- L=1, memory=1, T=128, instant: cosine=0.9788 +- 0.0000, relative=0.2048, norm=0.9823
- L=1, memory=1, T=128, trace: cosine=0.9971 +- 0.0000, relative=0.0768, norm=1.0073
- L=2, memory=0, T=2, instant: cosine=0.9858 +- 0.0000, relative=0.1678, norm=0.9878
- L=2, memory=0, T=2, trace: cosine=0.9960 +- 0.0000, relative=0.0923, norm=1.0179
- L=2, memory=0, T=4, instant: cosine=0.9156 +- 0.0000, relative=0.4064, norm=0.8570
- L=2, memory=0, T=4, trace: cosine=0.9373 +- 0.0000, relative=0.3545, norm=0.8720
- L=2, memory=0, T=8, instant: cosine=0.8916 +- 0.0000, relative=0.4700, norm=0.7657
- L=2, memory=0, T=8, trace: cosine=0.9242 +- 0.0000, relative=0.3822, norm=0.9395
- L=2, memory=0, T=16, instant: cosine=0.8472 +- 0.0000, relative=0.5835, norm=0.6059
- L=2, memory=0, T=16, trace: cosine=0.9486 +- 0.0000, relative=0.3476, norm=0.8050
- L=2, memory=0, T=32, instant: cosine=0.8250 +- 0.0000, relative=0.6422, norm=0.5201
- L=2, memory=0, T=32, trace: cosine=0.9528 +- 0.0000, relative=0.3130, norm=0.8764
- L=2, memory=0, T=64, instant: cosine=0.8251 +- 0.0000, relative=0.6422, norm=0.5197
- L=2, memory=0, T=64, trace: cosine=0.9618 +- 0.0000, relative=0.2846, norm=0.8843
- L=2, memory=0, T=128, instant: cosine=0.8108 +- 0.0000, relative=0.6531, norm=0.5213
- L=2, memory=0, T=128, trace: cosine=0.9631 +- 0.0000, relative=0.2745, norm=0.9088
- L=2, memory=1, T=2, instant: cosine=0.9826 +- 0.0000, relative=0.1924, norm=0.9333
- L=2, memory=1, T=2, trace: cosine=0.9883 +- 0.0000, relative=0.1593, norm=0.9424
- L=2, memory=1, T=4, instant: cosine=0.9705 +- 0.0000, relative=0.2415, norm=0.9538
- L=2, memory=1, T=4, trace: cosine=0.9852 +- 0.0000, relative=0.1724, norm=0.9681
- L=2, memory=1, T=8, instant: cosine=0.9367 +- 0.0000, relative=0.3539, norm=0.8854
- L=2, memory=1, T=8, trace: cosine=0.9881 +- 0.0000, relative=0.1536, norm=0.9876
- L=2, memory=1, T=16, instant: cosine=0.6649 +- 0.0000, relative=0.7575, norm=0.5389
- L=2, memory=1, T=16, trace: cosine=0.9834 +- 0.0000, relative=0.1982, norm=0.9032
- L=2, memory=1, T=32, instant: cosine=0.8433 +- 0.0000, relative=0.5378, norm=0.8624
- L=2, memory=1, T=32, trace: cosine=0.9326 +- 0.0000, relative=0.3706, norm=1.0165
- L=2, memory=1, T=64, instant: cosine=0.8184 +- 0.0000, relative=0.5751, norm=0.8408
- L=2, memory=1, T=64, trace: cosine=0.8993 +- 0.0000, relative=0.4453, norm=0.9829
- L=2, memory=1, T=128, instant: cosine=0.7984 +- 0.0000, relative=0.6025, norm=0.8205
- L=2, memory=1, T=128, trace: cosine=0.9122 +- 0.0000, relative=0.4166, norm=0.9875

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
