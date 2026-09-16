# TMT-v3 MLX Validation

- Buffer invariant: PASS
- Elementwise injection diagonal finite difference max error: 9.549e-05
- Phase eligibility finite difference max error: 4.810e-12
- Tiny actual-model BPTT loss: 5.561574
- Writer BPTT gradient norms: {'memory.key.weight': 0.010768917389214039, 'memory.value.weight': 0.01297310832887888, 'memory.forget.weight': 4.245615912168432e-07, 'memory.forget.bias': 3.380355906301702e-07, 'memory.rate.weight': 0.00033534737303853035, 'memory.rate.bias': 0.00020487303845584393}
- Persistent period-2/period-3 cycle residuals: {'parity': 2.010764683385949e-87, 'mod3': 0.0}
- One real online train step: {'loss': 5.806110382080078, 'byte': 5.508256435394287, 'latent': 0.9125990867614746, 'variance': 0.0, 'writer': 0.697040319442749}

The tiny BPTT harness establishes exact differentiability through state and delta-memory updates. It is a baseline for future cosine/relative-error comparisons with the structured-local estimator; it does not mislabel that estimator as full RTRL.
