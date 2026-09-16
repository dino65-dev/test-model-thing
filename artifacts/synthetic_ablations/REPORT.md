# Synthetic architectural ablations

3 seeds; train length 8; evaluation lengths [8, 16].

`no_complex` bypasses every complex rotation/controller path and freezes those leaves. `no_memory` bypasses both memory read and matrix update. `fixed_decay` freezes rho. All memory-enabled conditions receive the same bounded future-byte CE memory BPTT.

- copy | fixed_decay | train 8 -> eval 8: 0.7188 +- 0.0776
- copy | fixed_decay | train 8 -> eval 16: 0.6693 +- 0.0803
- copy | full | train 8 -> eval 8: 0.7188 +- 0.0776
- copy | full | train 8 -> eval 16: 0.6719 +- 0.0837
- copy | no_complex | train 8 -> eval 8: 0.6042 +- 0.0368
- copy | no_complex | train 8 -> eval 16: 0.5417 +- 0.0097
- copy | no_memory | train 8 -> eval 8: 0.7188 +- 0.0776
- copy | no_memory | train 8 -> eval 16: 0.6667 +- 0.0769
- mod3 | fixed_decay | train 8 -> eval 8: 0.3594 +- 0.0510
- mod3 | fixed_decay | train 8 -> eval 16: 0.3125 +- 0.0460
- mod3 | full | train 8 -> eval 8: 0.3594 +- 0.0510
- mod3 | full | train 8 -> eval 16: 0.3177 +- 0.0434
- mod3 | no_complex | train 8 -> eval 8: 0.3438 +- 0.0776
- mod3 | no_complex | train 8 -> eval 16: 0.3568 +- 0.0097
- mod3 | no_memory | train 8 -> eval 8: 0.3646 +- 0.0655
- mod3 | no_memory | train 8 -> eval 16: 0.3125 +- 0.0442
- mod5 | fixed_decay | train 8 -> eval 8: 0.1875 +- 0.0460
- mod5 | fixed_decay | train 8 -> eval 16: 0.1979 +- 0.0258
- mod5 | full | train 8 -> eval 8: 0.1875 +- 0.0460
- mod5 | full | train 8 -> eval 16: 0.2005 +- 0.0288
- mod5 | no_complex | train 8 -> eval 8: 0.2031 +- 0.0510
- mod5 | no_complex | train 8 -> eval 16: 0.2135 +- 0.0351
- mod5 | no_memory | train 8 -> eval 8: 0.1927 +- 0.0448
- mod5 | no_memory | train 8 -> eval 16: 0.1927 +- 0.0224
- parity | fixed_decay | train 8 -> eval 8: 0.4167 +- 0.0448
- parity | fixed_decay | train 8 -> eval 16: 0.4089 +- 0.0415
- parity | full | train 8 -> eval 8: 0.4167 +- 0.0448
- parity | full | train 8 -> eval 16: 0.4089 +- 0.0415
- parity | no_complex | train 8 -> eval 8: 0.4792 +- 0.0195
- parity | no_complex | train 8 -> eval 16: 0.4974 +- 0.0461
- parity | no_memory | train 8 -> eval 8: 0.4167 +- 0.0448
- parity | no_memory | train 8 -> eval 16: 0.4062 +- 0.0447
- xor | fixed_decay | train 8 -> eval 8: 0.5938 +- 0.0585
- xor | fixed_decay | train 8 -> eval 16: 0.5495 +- 0.0885
- xor | full | train 8 -> eval 8: 0.5990 +- 0.0642
- xor | full | train 8 -> eval 16: 0.5547 +- 0.0877
- xor | no_complex | train 8 -> eval 8: 0.6406 +- 0.0556
- xor | no_complex | train 8 -> eval 16: 0.5677 +- 0.0808
- xor | no_memory | train 8 -> eval 8: 0.5938 +- 0.0585
- xor | no_memory | train 8 -> eval 16: 0.5469 +- 0.0920
