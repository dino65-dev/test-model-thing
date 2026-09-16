# TMT-v2 Mathematical Diagnostics
P0: private underscore buffers are used in main.py and should not be in MLX parameters.
P1: alpha=2^(-1/H), normalized injection, and scalar eligibility finite-difference check passed (max error 3.383e-11).
P2: rho R(omega) has eigenvalue magnitude rho; plot confirms rho=.995 decay.
P3: delta-rule associative-memory retrieval error is plotted.
P4: entropy threshold produces shorter patches at uncertain bytes.
The trace validation is scalar/local only; it is not a claim of full BPTT, UORO, or KF-RTRL equivalence.
