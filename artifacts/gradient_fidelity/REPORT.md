# Structured online-gradient fidelity

Measured against exact BPTT across 10 independently initialized random byte sequences per configuration (L=1/2 through T=128; T=256 includes L=1/2/4).

Memory-on rows compare backbone gradients only; the memory controller is intentionally owned by separate temporal BPTT and therefore excluded from this online-gradient comparison.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9958 +- 0.0035, relative=0.0804, norm=0.9960
- L=1, memory=0, T=2, trace: cosine=0.9999 +- 0.0001, relative=0.0076, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9809 +- 0.0265, relative=0.1631, norm=0.9708
- L=1, memory=0, T=4, trace: cosine=0.9999 +- 0.0001, relative=0.0090, norm=1.0000
- L=1, memory=0, T=8, instant: cosine=0.9374 +- 0.0321, relative=0.3409, norm=0.8978
- L=1, memory=0, T=8, trace: cosine=0.9998 +- 0.0001, relative=0.0170, norm=1.0000
- L=1, memory=0, T=16, instant: cosine=0.8579 +- 0.0700, relative=0.4993, norm=0.7902
- L=1, memory=0, T=16, trace: cosine=0.9999 +- 0.0001, relative=0.0156, norm=1.0000
- L=1, memory=0, T=32, instant: cosine=0.8528 +- 0.1029, relative=0.4842, norm=0.8104
- L=1, memory=0, T=32, trace: cosine=0.9998 +- 0.0002, relative=0.0173, norm=0.9999
- L=1, memory=0, T=64, instant: cosine=0.8257 +- 0.1210, relative=0.5196, norm=0.8111
- L=1, memory=0, T=64, trace: cosine=0.9997 +- 0.0002, relative=0.0215, norm=0.9999
- L=1, memory=0, T=128, instant: cosine=0.8092 +- 0.1312, relative=0.5436, norm=0.7931
- L=1, memory=0, T=128, trace: cosine=0.9997 +- 0.0002, relative=0.0219, norm=0.9999
- L=1, memory=0, T=256, instant: cosine=0.8044 +- 0.1343, relative=0.5486, norm=0.7888
- L=1, memory=0, T=256, trace: cosine=0.9997 +- 0.0002, relative=0.0226, norm=0.9998
- L=1, memory=1, T=2, instant: cosine=0.9905 +- 0.0103, relative=0.1260, norm=0.9940
- L=1, memory=1, T=2, trace: cosine=0.9995 +- 0.0008, relative=0.0250, norm=1.0009
- L=1, memory=1, T=4, instant: cosine=0.9818 +- 0.0060, relative=0.1884, norm=0.9872
- L=1, memory=1, T=4, trace: cosine=0.9992 +- 0.0008, relative=0.0365, norm=1.0007
- L=1, memory=1, T=8, instant: cosine=0.9431 +- 0.0206, relative=0.3304, norm=0.8974
- L=1, memory=1, T=8, trace: cosine=0.9992 +- 0.0004, relative=0.0395, norm=1.0013
- L=1, memory=1, T=16, instant: cosine=0.8760 +- 0.1080, relative=0.4430, norm=0.7809
- L=1, memory=1, T=16, trace: cosine=0.9992 +- 0.0005, relative=0.0392, norm=0.9958
- L=1, memory=1, T=32, instant: cosine=0.8496 +- 0.0793, relative=0.5169, norm=0.7466
- L=1, memory=1, T=32, trace: cosine=0.9986 +- 0.0009, relative=0.0530, norm=0.9973
- L=1, memory=1, T=64, instant: cosine=0.8441 +- 0.0912, relative=0.5150, norm=0.7354
- L=1, memory=1, T=64, trace: cosine=0.9974 +- 0.0022, relative=0.0697, norm=0.9980
- L=1, memory=1, T=128, instant: cosine=0.8424 +- 0.0860, relative=0.5247, norm=0.7274
- L=1, memory=1, T=128, trace: cosine=0.9951 +- 0.0040, relative=0.1004, norm=0.9954
- L=1, memory=1, T=256, instant: cosine=0.8300 +- 0.0834, relative=0.5531, norm=0.7075
- L=1, memory=1, T=256, trace: cosine=0.9921 +- 0.0073, relative=0.1243, norm=0.9911
- L=2, memory=0, T=2, instant: cosine=0.9902 +- 0.0080, relative=0.1322, norm=0.9894
- L=2, memory=0, T=2, trace: cosine=0.9940 +- 0.0047, relative=0.1034, norm=0.9939
- L=2, memory=0, T=4, instant: cosine=0.9570 +- 0.0202, relative=0.2903, norm=0.9316
- L=2, memory=0, T=4, trace: cosine=0.9773 +- 0.0190, relative=0.1999, norm=0.9621
- L=2, memory=0, T=8, instant: cosine=0.8276 +- 0.1133, relative=0.5463, norm=0.7518
- L=2, memory=0, T=8, trace: cosine=0.9423 +- 0.0291, relative=0.3373, norm=0.9344
- L=2, memory=0, T=16, instant: cosine=0.7415 +- 0.1310, relative=0.6554, norm=0.6571
- L=2, memory=0, T=16, trace: cosine=0.9213 +- 0.0444, relative=0.3982, norm=0.8494
- L=2, memory=0, T=32, instant: cosine=0.6699 +- 0.1594, relative=0.7274, norm=0.5939
- L=2, memory=0, T=32, trace: cosine=0.9214 +- 0.0498, relative=0.4010, norm=0.8188
- L=2, memory=0, T=64, instant: cosine=0.6430 +- 0.1596, relative=0.7513, norm=0.5626
- L=2, memory=0, T=64, trace: cosine=0.9201 +- 0.0546, relative=0.4066, norm=0.7947
- L=2, memory=0, T=128, instant: cosine=0.6123 +- 0.1771, relative=0.7719, norm=0.5329
- L=2, memory=0, T=128, trace: cosine=0.9143 +- 0.0635, relative=0.4214, norm=0.7808
- L=2, memory=0, T=256, instant: cosine=0.6037 +- 0.1851, relative=0.7757, norm=0.5296
- L=2, memory=0, T=256, trace: cosine=0.9107 +- 0.0658, relative=0.4286, norm=0.7782
- L=2, memory=1, T=2, instant: cosine=0.9763 +- 0.0309, relative=0.1905, norm=0.9842
- L=2, memory=1, T=2, trace: cosine=0.9899 +- 0.0056, relative=0.1359, norm=0.9909
- L=2, memory=1, T=4, instant: cosine=0.9458 +- 0.0302, relative=0.3167, norm=0.9256
- L=2, memory=1, T=4, trace: cosine=0.9791 +- 0.0104, relative=0.2026, norm=0.9688
- L=2, memory=1, T=8, instant: cosine=0.8658 +- 0.1019, relative=0.4836, norm=0.7653
- L=2, memory=1, T=8, trace: cosine=0.9677 +- 0.0246, relative=0.2400, norm=0.9337
- L=2, memory=1, T=16, instant: cosine=0.7473 +- 0.1179, relative=0.6456, norm=0.6388
- L=2, memory=1, T=16, trace: cosine=0.9550 +- 0.0230, relative=0.2985, norm=0.9422
- L=2, memory=1, T=32, instant: cosine=0.7092 +- 0.1160, relative=0.6975, norm=0.6272
- L=2, memory=1, T=32, trace: cosine=0.9395 +- 0.0315, relative=0.3471, norm=0.9283
- L=2, memory=1, T=64, instant: cosine=0.6715 +- 0.1201, relative=0.7354, norm=0.5845
- L=2, memory=1, T=64, trace: cosine=0.9390 +- 0.0400, relative=0.3508, norm=0.9065
- L=2, memory=1, T=128, instant: cosine=0.6579 +- 0.1179, relative=0.7485, norm=0.5679
- L=2, memory=1, T=128, trace: cosine=0.9355 +- 0.0379, relative=0.3686, norm=0.9055
- L=2, memory=1, T=256, instant: cosine=0.6527 +- 0.1244, relative=0.7522, norm=0.5616
- L=2, memory=1, T=256, trace: cosine=0.9326 +- 0.0414, relative=0.3778, norm=0.9032
- L=4, memory=0, T=256, instant: cosine=0.5625 +- 0.1255, relative=0.8252, norm=0.4521
- L=4, memory=0, T=256, trace: cosine=0.8242 +- 0.1187, relative=0.5547, norm=0.7302
- L=4, memory=1, T=256, instant: cosine=0.5678 +- 0.1969, relative=0.8106, norm=0.4767
- L=4, memory=1, T=256, trace: cosine=0.7642 +- 0.2589, relative=0.5631, norm=0.8143

`families.csv` contains both structured-trace and instantaneous metrics plus each family’s exact-gradient energy share. This measures a structured approximation; it does not claim full RTRL.
