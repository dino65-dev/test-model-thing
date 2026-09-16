# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9995 +- 0.0000, relative=0.0302, norm=0.9999
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0020, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9922 +- 0.0000, relative=0.1247, norm=0.9911
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0084, norm=0.9999
- L=1, memory=0, T=8, instant: cosine=0.9763 +- 0.0000, relative=0.2166, norm=0.9786
- L=1, memory=0, T=8, trace: cosine=0.9998 +- 0.0000, relative=0.0184, norm=1.0004
- L=1, memory=0, T=16, instant: cosine=0.7384 +- 0.0000, relative=0.6872, norm=0.6058
- L=1, memory=0, T=16, trace: cosine=1.0000 +- 0.0000, relative=0.0097, norm=0.9999
- L=1, memory=0, T=32, instant: cosine=0.6989 +- 0.0000, relative=0.7300, norm=0.5530
- L=1, memory=0, T=32, trace: cosine=0.9998 +- 0.0000, relative=0.0214, norm=0.9996
- L=1, memory=0, T=64, instant: cosine=0.6166 +- 0.0000, relative=0.7874, norm=0.6330
- L=1, memory=0, T=64, trace: cosine=0.9993 +- 0.0000, relative=0.0377, norm=0.9991
- L=1, memory=0, T=128, instant: cosine=0.6321 +- 0.0000, relative=0.7766, norm=0.5806
- L=1, memory=0, T=128, trace: cosine=0.9994 +- 0.0000, relative=0.0336, norm=0.9993
- L=1, memory=1, T=2, instant: cosine=0.9896 +- 0.0000, relative=0.1440, norm=0.9838
- L=1, memory=1, T=2, trace: cosine=0.9999 +- 0.0000, relative=0.0140, norm=0.9999
- L=1, memory=1, T=4, instant: cosine=0.9846 +- 0.0000, relative=0.1764, norm=0.9594
- L=1, memory=1, T=4, trace: cosine=0.9991 +- 0.0000, relative=0.0421, norm=0.9996
- L=1, memory=1, T=8, instant: cosine=0.9162 +- 0.0000, relative=0.4143, norm=0.8108
- L=1, memory=1, T=8, trace: cosine=0.9995 +- 0.0000, relative=0.0310, norm=1.0012
- L=1, memory=1, T=16, instant: cosine=0.7444 +- 0.0000, relative=0.7389, norm=0.4279
- L=1, memory=1, T=16, trace: cosine=0.9997 +- 0.0000, relative=0.0251, norm=0.9911
- L=1, memory=1, T=32, instant: cosine=0.8004 +- 0.0000, relative=0.6289, norm=0.6102
- L=1, memory=1, T=32, trace: cosine=0.9994 +- 0.0000, relative=0.0338, norm=1.0018
- L=1, memory=1, T=64, instant: cosine=0.7838 +- 0.0000, relative=0.6430, norm=0.6173
- L=1, memory=1, T=64, trace: cosine=0.9990 +- 0.0000, relative=0.0476, norm=0.9851
- L=1, memory=1, T=128, instant: cosine=0.7623 +- 0.0000, relative=0.6681, norm=0.5966
- L=1, memory=1, T=128, trace: cosine=0.9983 +- 0.0000, relative=0.0722, norm=0.9560
- L=2, memory=0, T=2, instant: cosine=0.9984 +- 0.0000, relative=0.0578, norm=0.9859
- L=2, memory=0, T=2, trace: cosine=0.9988 +- 0.0000, relative=0.0496, norm=0.9873
- L=2, memory=0, T=4, instant: cosine=0.9684 +- 0.0000, relative=0.2548, norm=0.9164
- L=2, memory=0, T=4, trace: cosine=0.9576 +- 0.0000, relative=0.2884, norm=0.9439
- L=2, memory=0, T=8, instant: cosine=0.8490 +- 0.0000, relative=0.5569, norm=0.6732
- L=2, memory=0, T=8, trace: cosine=0.9507 +- 0.0000, relative=0.3103, norm=0.9518
- L=2, memory=0, T=16, instant: cosine=0.7626 +- 0.0000, relative=0.6471, norm=0.7800
- L=2, memory=0, T=16, trace: cosine=0.8394 +- 0.0000, relative=0.5744, norm=1.0252
- L=2, memory=0, T=32, instant: cosine=0.6531 +- 0.0000, relative=0.7587, norm=0.7002
- L=2, memory=0, T=32, trace: cosine=0.8141 +- 0.0000, relative=0.6105, norm=1.0028
- L=2, memory=0, T=64, instant: cosine=0.7224 +- 0.0000, relative=0.6936, norm=0.6688
- L=2, memory=0, T=64, trace: cosine=0.9012 +- 0.0000, relative=0.4408, norm=0.9820
- L=2, memory=0, T=128, instant: cosine=0.6737 +- 0.0000, relative=0.7393, norm=0.6538
- L=2, memory=0, T=128, trace: cosine=0.8733 +- 0.0000, relative=0.4955, norm=0.9634
- L=2, memory=1, T=2, instant: cosine=0.9805 +- 0.0000, relative=0.1996, norm=1.0163
- L=2, memory=1, T=2, trace: cosine=0.9851 +- 0.0000, relative=0.1738, norm=1.0100
- L=2, memory=1, T=4, instant: cosine=0.9204 +- 0.0000, relative=0.3947, norm=0.9740
- L=2, memory=1, T=4, trace: cosine=0.9559 +- 0.0000, relative=0.2956, norm=0.9881
- L=2, memory=1, T=8, instant: cosine=0.8630 +- 0.0000, relative=0.5096, norm=0.7955
- L=2, memory=1, T=8, trace: cosine=0.9590 +- 0.0000, relative=0.2920, norm=0.8891
- L=2, memory=1, T=16, instant: cosine=0.8853 +- 0.0000, relative=0.4783, norm=0.7729
- L=2, memory=1, T=16, trace: cosine=0.9658 +- 0.0000, relative=0.2631, norm=0.9211
- L=2, memory=1, T=32, instant: cosine=0.7846 +- 0.0000, relative=0.6209, norm=0.8170
- L=2, memory=1, T=32, trace: cosine=0.9378 +- 0.0000, relative=0.3567, norm=1.0193
- L=2, memory=1, T=64, instant: cosine=0.7164 +- 0.0000, relative=0.6986, norm=0.7519
- L=2, memory=1, T=64, trace: cosine=0.9729 +- 0.0000, relative=0.2364, norm=1.0218
- L=2, memory=1, T=128, instant: cosine=0.6880 +- 0.0000, relative=0.7261, norm=0.7139
- L=2, memory=1, T=128, trace: cosine=0.9647 +- 0.0000, relative=0.2754, norm=1.0460

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
