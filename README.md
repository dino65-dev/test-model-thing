# Test-Model-Thing: TMT-v4

TMT-v4 is an experimental MLX streaming byte model. Its pairwise recurrent
state combines multi-timescale decay, raw complex rotations, elementwise gated
writes, structured online eligibility traces, and an associative delta memory.
It remains a bounded-memory model: state is constant in stream length, not a
claim of unlimited context.

## Mathematics and optimizer ownership

For each state pair, `s_t = rho R(theta_t)s_(t-1) + sqrt(1-rho^2) gate(x_t)x_t`,
where `rho = sigmoid(decay)`, the base phase is a raw angle, and the input phase
controller is `theta_t = phase + pi*tanh(phase_scale*pair_mean(x_t)+phase_bias)`.

The online optimizer is bias-corrected AdamW with **zero decay** for decay,
phase, gate/controller, LayerNorm, and bias parameters. A small decoupled decay
is applied only to selected dense matrices. Memory is frozen from online
optimization and has exactly one owner: a separate bounded temporal-BPTT
optimizer trained on future byte cross entropy. This prevents two Adam moment
histories from updating the same controller leaves.

`memory_enabled=False`, `complex_enabled=False`, and `learn_decay=False` are
architectural ablations: each bypasses/fixes its relevant dynamics rather than
setting a trainable output to zero.

## Checkpoints

TMT-v4 saves model parameters, both optimizer states, and every runtime buffer.
It requires exact `tmt-v4` metadata and exact parameter/buffer keys and shapes
on load. Earlier checkpoints intentionally do not load partially.

## Validation and experiments

Run from this directory:

```sh
../.venv/bin/python tmt_v3_validation.py
../.venv/bin/python gradient_fidelity.py --seeds 10 --max-t 128  # L=1/2; use --layers-list 1,2,4 --min-t 256 --max-t 256 for the T=256 extension
../.venv/bin/python synthetic_ablations.py --seeds 10
../.venv/bin/python language_monitor.py tmt-v4.safetensors HELDOUT.bin
```

The strict suite performs 50 randomized finite-difference checks for the local
pair Jacobian and phase eligibility, exact delayed-byte-CE memory-BPTT gradient
checks, optimizer ownership checks, checkpoint rejection checks, and a 512x16
online step. Gradient fidelity reports trace versus instantaneous gradients for
L=1/2 with memory on/off, including per-family energy shares. Synthetic results
are multi-seed extrapolation measurements, not capability claims.

During `Runtime.train`, every 100 bytes logs byte loss, bytes/sec, rho
half-life range, phase distribution, forget gate, and state/trace RMS. The
language monitor adds held-out BPB for an explicitly supplied corpus.
