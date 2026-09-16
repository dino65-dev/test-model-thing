# Test-Model-Thing: TMT-v3

TMT-v3 is an experimental streaming byte-level recurrent model built with MLX.
It combines multi-timescale complex state-space modes, structured local
eligibility traces, multi-horizon predictive latent learning, and delta-rule
associative fast memory. Runtime recurrent storage is constant with stream
length; this is compressed memory, not unlimited information capacity.

## Core recurrence

For each two-dimensional state pair:

    s_t = rho R(theta_t) s_(t-1) + sqrt(1-rho^2) gate(x_t) * x_t
    theta_t = theta + pi*tanh(phase_scale * pair_mean(x_t) + phase_bias)
    gate(x_t) = sigmoid(write_scale * x_t + write_bias)

Half-lives and periods use a cross product, so persistent period-2 and
period-3 modes coexist with short and long smoothing modes. The elementwise
gate is deliberate: its local injection derivative is cheap and explicit.

TMT uses a structured local eligibility approximation, not full RTRL. Recurrent
state, eligibility buffers, EMA target, and fast-memory matrix are private MLX
buffers, explicitly serialized, and never passed to AdamW.

## Training and inference

- train_step(current, target, future) uses byte cross entropy, weighted
  multi-horizon cosine prediction, streaming variance, and writer loss.
- frozen_step(token) updates only recurrent/fast memory.
- stateless_step(token) restores every mutable buffer afterward.
- Output is one 257-way byte/EOS categorical distribution.

Legacy checkpoints are intentionally incompatible with TMT-v3. Retrain rather
than partially loading v1/v2 weights.

## Validation

From test-model-thing:

    ../.venv/bin/python tmt_v3_validation.py
    ../.venv/bin/python math_diagnostics.py

The first script runs real MLX checks for buffer separation, gate and phase
finite differences, exact tiny-model BPTT, associative-controller gradients,
state cycles, and an online step. Reports live in artifacts/.

benchmark.py trains a linear CoLA head on train and reports held-out dev MCC,
resetting state per sentence and reading the post-memory representation.

## Scope limits

Entropy patches currently schedule/measures boundaries but do not yet skip
recurrent computation. Chat is a byte-LM interface, not a role-trained
assistant. Before long corpus training, use validation BPB, parity/modular
tracking, delayed-copy, and associative-recall experiments.
