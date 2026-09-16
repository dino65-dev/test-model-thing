# TMT-v4 Strict MLX Validation
- Pair-block embedding Jacobian randomized FD: {'samples': 50, 'max_abs': 8.702278137207031e-05, 'max_relative_fro': 0.00026009501233758837}
- Base phase eligibility randomized FD: {'samples': 50, 'max_abs': 0.00014257431030273438, 'max_relative_fro': 0.0002139141959812728}
- Exact delayed-byte-CE memory BPTT loss: 5.902454
- Memory-controller gradient norms: {'key.weight': 0.026490863412618637, 'query.weight': 0.06172357127070427, 'value.weight': 0.030722659081220627, 'out.weight': 0.017590539529919624, 'forget.weight': 1.1230457857891452e-05, 'forget.bias': 9.462769412493799e-06, 'rate.weight': 0.01649368926882744, 'rate.bias': 0.009215010330080986}
- Optimizer ownership: online optimizer excludes memory; temporal optimizer is sole owner
- Raw-angle cycle closure: {'period2': 0.001749983257311556, 'period3': 0.001749983257311556}
- Checkpoints: round-trip, metadata rejection, and every-buffer rejection PASS
- Fixed-decay check: fixed rho parameters remain exactly fixed under online training
- Exact parameter count: 5952259
- 512x16 online metrics: {'loss': 7.030419826507568, 'byte': 6.785922050476074, 'latent': 0.9779909253120422, 'variance': 0.0}
All assertions passed.
