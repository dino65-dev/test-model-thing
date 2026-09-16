# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9973 +- 0.0000, relative=0.0739, norm=0.9901
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0030, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9978 +- 0.0000, relative=0.0660, norm=0.9994
- L=1, memory=0, T=4, trace: cosine=1.0000 +- 0.0000, relative=0.0041, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.9442 +- 0.0000, relative=0.3347, norm=0.8841
- L=1, memory=0, T=8, trace: cosine=0.9998 +- 0.0000, relative=0.0207, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.9736 +- 0.0000, relative=0.2286, norm=0.9596
- L=1, memory=0, T=16, trace: cosine=0.9997 +- 0.0000, relative=0.0255, norm=1.0003
- L=1, memory=0, T=32, instant: cosine=0.9778 +- 0.0000, relative=0.2104, norm=0.9597
- L=1, memory=0, T=32, trace: cosine=0.9996 +- 0.0000, relative=0.0270, norm=1.0002
- L=1, memory=0, T=64, instant: cosine=0.9670 +- 0.0000, relative=0.2572, norm=0.9328
- L=1, memory=0, T=64, trace: cosine=0.9997 +- 0.0000, relative=0.0255, norm=1.0001
- L=1, memory=0, T=128, instant: cosine=0.9505 +- 0.0000, relative=0.3108, norm=0.9573
- L=1, memory=0, T=128, trace: cosine=0.9996 +- 0.0000, relative=0.0267, norm=1.0001
- L=1, memory=1, T=2, instant: cosine=0.9605 +- 0.0000, relative=0.2787, norm=0.9748
- L=1, memory=1, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0035, norm=1.0000
- L=1, memory=1, T=4, instant: cosine=0.9679 +- 0.0000, relative=0.2516, norm=0.9810
- L=1, memory=1, T=4, trace: cosine=0.9999 +- 0.0000, relative=0.0160, norm=0.9974
- L=1, memory=1, T=8, instant: cosine=0.9705 +- 0.0000, relative=0.2415, norm=0.9851
- L=1, memory=1, T=8, trace: cosine=0.9994 +- 0.0000, relative=0.0340, norm=0.9971
- L=1, memory=1, T=16, instant: cosine=0.9397 +- 0.0000, relative=0.3507, norm=0.8622
- L=1, memory=1, T=16, trace: cosine=0.9996 +- 0.0000, relative=0.0290, norm=1.0040
- L=1, memory=1, T=32, instant: cosine=0.9362 +- 0.0000, relative=0.3560, norm=0.8789
- L=1, memory=1, T=32, trace: cosine=0.9989 +- 0.0000, relative=0.0506, norm=1.0150
- L=1, memory=1, T=64, instant: cosine=0.9472 +- 0.0000, relative=0.3242, norm=0.8989
- L=1, memory=1, T=64, trace: cosine=0.9982 +- 0.0000, relative=0.0618, norm=1.0155
- L=1, memory=1, T=128, instant: cosine=0.9166 +- 0.0000, relative=0.4042, norm=0.8566
- L=1, memory=1, T=128, trace: cosine=0.9955 +- 0.0000, relative=0.1048, norm=1.0397
- L=2, memory=0, T=2, instant: cosine=0.9878 +- 0.0000, relative=0.1603, norm=0.9495
- L=2, memory=0, T=2, trace: cosine=0.9922 +- 0.0000, relative=0.1301, norm=0.9542
- L=2, memory=0, T=4, instant: cosine=0.9712 +- 0.0000, relative=0.2406, norm=0.9379
- L=2, memory=0, T=4, trace: cosine=0.9902 +- 0.0000, relative=0.1421, norm=0.9639
- L=2, memory=0, T=8, instant: cosine=0.7258 +- 0.0000, relative=0.7061, norm=0.5666
- L=2, memory=0, T=8, trace: cosine=0.9211 +- 0.0000, relative=0.3908, norm=0.8873
- L=2, memory=0, T=16, instant: cosine=0.7651 +- 0.0000, relative=0.6441, norm=0.7492
- L=2, memory=0, T=16, trace: cosine=0.8552 +- 0.0000, relative=0.5238, norm=0.9315
- L=2, memory=0, T=32, instant: cosine=0.6376 +- 0.0000, relative=0.7715, norm=0.5956
- L=2, memory=0, T=32, trace: cosine=0.9892 +- 0.0000, relative=0.1516, norm=0.9498
- L=2, memory=0, T=64, instant: cosine=0.5481 +- 0.0000, relative=0.8386, norm=0.4876
- L=2, memory=0, T=64, trace: cosine=0.9811 +- 0.0000, relative=0.2344, norm=0.8485
- L=2, memory=0, T=128, instant: cosine=0.4901 +- 0.0000, relative=0.8738, norm=0.4298
- L=2, memory=0, T=128, trace: cosine=0.9775 +- 0.0000, relative=0.2604, norm=0.8250
- L=2, memory=1, T=2, instant: cosine=0.9974 +- 0.0000, relative=0.0729, norm=0.9892
- L=2, memory=1, T=2, trace: cosine=0.9993 +- 0.0000, relative=0.0377, norm=0.9926
- L=2, memory=1, T=4, instant: cosine=0.9323 +- 0.0000, relative=0.3750, norm=0.8336
- L=2, memory=1, T=4, trace: cosine=0.9837 +- 0.0000, relative=0.1803, norm=0.9725
- L=2, memory=1, T=8, instant: cosine=0.8266 +- 0.0000, relative=0.6315, norm=0.5401
- L=2, memory=1, T=8, trace: cosine=0.9927 +- 0.0000, relative=0.1215, norm=0.9810
- L=2, memory=1, T=16, instant: cosine=0.7097 +- 0.0000, relative=0.7202, norm=0.5599
- L=2, memory=1, T=16, trace: cosine=0.9734 +- 0.0000, relative=0.2294, norm=0.9639
- L=2, memory=1, T=32, instant: cosine=0.7784 +- 0.0000, relative=0.6809, norm=0.5146
- L=2, memory=1, T=32, trace: cosine=0.9873 +- 0.0000, relative=0.1602, norm=0.9680
- L=2, memory=1, T=64, instant: cosine=0.7980 +- 0.0000, relative=0.6479, norm=0.5602
- L=2, memory=1, T=64, trace: cosine=0.9790 +- 0.0000, relative=0.2075, norm=1.0168
- L=2, memory=1, T=128, instant: cosine=0.8226 +- 0.0000, relative=0.6145, norm=0.5894
- L=2, memory=1, T=128, trace: cosine=0.9633 +- 0.0000, relative=0.2864, norm=1.0636

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
