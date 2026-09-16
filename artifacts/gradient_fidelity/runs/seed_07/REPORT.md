# Structured online-gradient fidelity

Measured against exact BPTT across 1 random deterministic byte sequences per configuration.

The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.

## Aggregate whole-model metrics

- L=1, memory=0, T=2, instant: cosine=0.9995 +- 0.0000, relative=0.0321, norm=1.0004
- L=1, memory=0, T=2, trace: cosine=1.0000 +- 0.0000, relative=0.0041, norm=1.0000
- L=1, memory=0, T=4, instant: cosine=0.9884 +- 0.0000, relative=0.1521, norm=0.9908
- L=1, memory=0, T=4, trace: cosine=0.9999 +- 0.0000, relative=0.0115, norm=1.0002
- L=1, memory=0, T=8, instant: cosine=0.9444 +- 0.0000, relative=0.3313, norm=0.9025
- L=1, memory=0, T=8, trace: cosine=1.0000 +- 0.0000, relative=0.0082, norm=1.0001
- L=1, memory=0, T=16, instant: cosine=0.8553 +- 0.0000, relative=0.5237, norm=0.9312
- L=1, memory=0, T=16, trace: cosine=1.0000 +- 0.0000, relative=0.0090, norm=1.0001
- L=1, memory=0, T=32, instant: cosine=0.6757 +- 0.0000, relative=0.7390, norm=0.6241
- L=1, memory=0, T=32, trace: cosine=1.0000 +- 0.0000, relative=0.0060, norm=1.0000
- L=1, memory=0, T=64, instant: cosine=0.6214 +- 0.0000, relative=0.7853, norm=0.6744
- L=1, memory=0, T=64, trace: cosine=0.9998 +- 0.0000, relative=0.0193, norm=0.9998
- L=1, memory=0, T=128, instant: cosine=0.5394 +- 0.0000, relative=0.8467, norm=0.6282
- L=1, memory=0, T=128, trace: cosine=0.9997 +- 0.0000, relative=0.0248, norm=0.9997
- L=1, memory=1, T=2, instant: cosine=0.9989 +- 0.0000, relative=0.0473, norm=1.0004
- L=1, memory=1, T=2, trace: cosine=0.9999 +- 0.0000, relative=0.0103, norm=1.0013
- L=1, memory=1, T=4, instant: cosine=0.9759 +- 0.0000, relative=0.2210, norm=1.0116
- L=1, memory=1, T=4, trace: cosine=0.9993 +- 0.0000, relative=0.0377, norm=0.9981
- L=1, memory=1, T=8, instant: cosine=0.9409 +- 0.0000, relative=0.3420, norm=0.8929
- L=1, memory=1, T=8, trace: cosine=0.9986 +- 0.0000, relative=0.0538, norm=1.0060
- L=1, memory=1, T=16, instant: cosine=0.8852 +- 0.0000, relative=0.4675, norm=0.8395
- L=1, memory=1, T=16, trace: cosine=0.9986 +- 0.0000, relative=0.0550, norm=0.9811
- L=1, memory=1, T=32, instant: cosine=0.9663 +- 0.0000, relative=0.2581, norm=0.9483
- L=1, memory=1, T=32, trace: cosine=0.9987 +- 0.0000, relative=0.0511, norm=0.9943
- L=1, memory=1, T=64, instant: cosine=0.9635 +- 0.0000, relative=0.2679, norm=0.9498
- L=1, memory=1, T=64, trace: cosine=0.9989 +- 0.0000, relative=0.0479, norm=0.9928
- L=1, memory=1, T=128, instant: cosine=0.9559 +- 0.0000, relative=0.2943, norm=0.9360
- L=1, memory=1, T=128, trace: cosine=0.9955 +- 0.0000, relative=0.0988, norm=0.9672
- L=2, memory=0, T=2, instant: cosine=0.9874 +- 0.0000, relative=0.1585, norm=0.9754
- L=2, memory=0, T=2, trace: cosine=0.9901 +- 0.0000, relative=0.1413, norm=0.9757
- L=2, memory=0, T=4, instant: cosine=0.9385 +- 0.0000, relative=0.3525, norm=0.8671
- L=2, memory=0, T=4, trace: cosine=0.9817 +- 0.0000, relative=0.1953, norm=0.9381
- L=2, memory=0, T=8, instant: cosine=0.8822 +- 0.0000, relative=0.4824, norm=0.7777
- L=2, memory=0, T=8, trace: cosine=0.9417 +- 0.0000, relative=0.3372, norm=0.9193
- L=2, memory=0, T=16, instant: cosine=0.7095 +- 0.0000, relative=0.7349, norm=0.5009
- L=2, memory=0, T=16, trace: cosine=0.9477 +- 0.0000, relative=0.3671, norm=0.7663
- L=2, memory=0, T=32, instant: cosine=0.6725 +- 0.0000, relative=0.7529, norm=0.5345
- L=2, memory=0, T=32, trace: cosine=0.9318 +- 0.0000, relative=0.4048, norm=0.7524
- L=2, memory=0, T=64, instant: cosine=0.7050 +- 0.0000, relative=0.7093, norm=0.6925
- L=2, memory=0, T=64, trace: cosine=0.8901 +- 0.0000, relative=0.4628, norm=0.8104
- L=2, memory=0, T=128, instant: cosine=0.6917 +- 0.0000, relative=0.7231, norm=0.6558
- L=2, memory=0, T=128, trace: cosine=0.8849 +- 0.0000, relative=0.4780, norm=0.7776
- L=2, memory=1, T=2, instant: cosine=0.8871 +- 0.0000, relative=0.4645, norm=0.9384
- L=2, memory=1, T=2, trace: cosine=0.9833 +- 0.0000, relative=0.1833, norm=0.9627
- L=2, memory=1, T=4, instant: cosine=0.8741 +- 0.0000, relative=0.4887, norm=0.8204
- L=2, memory=1, T=4, trace: cosine=0.9767 +- 0.0000, relative=0.2335, norm=0.8845
- L=2, memory=1, T=8, instant: cosine=0.8187 +- 0.0000, relative=0.5945, norm=0.6647
- L=2, memory=1, T=8, trace: cosine=0.9228 +- 0.0000, relative=0.4159, norm=0.7661
- L=2, memory=1, T=16, instant: cosine=0.6620 +- 0.0000, relative=0.7599, norm=0.5364
- L=2, memory=1, T=16, trace: cosine=0.9177 +- 0.0000, relative=0.3993, norm=0.8777
- L=2, memory=1, T=32, instant: cosine=0.5813 +- 0.0000, relative=0.8416, norm=0.3662
- L=2, memory=1, T=32, trace: cosine=0.8942 +- 0.0000, relative=0.4905, norm=0.6936
- L=2, memory=1, T=64, instant: cosine=0.5444 +- 0.0000, relative=0.8676, norm=0.3227
- L=2, memory=1, T=64, trace: cosine=0.8909 +- 0.0000, relative=0.5392, norm=0.6003
- L=2, memory=1, T=128, instant: cosine=0.5369 +- 0.0000, relative=0.8716, norm=0.3179
- L=2, memory=1, T=128, trace: cosine=0.9072 +- 0.0000, relative=0.5113, norm=0.6166

`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL.
