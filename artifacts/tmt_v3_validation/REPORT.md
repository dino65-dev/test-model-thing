# TMT-v3 Strict MLX Validation
- Pair-block embedding Jacobian FD errors over three seeds: [0.00022536516189575195, 0.00022536516189575195, 0.00022536516189575195]
- Base phase eligibility FD errors over three seeds: [0.00012180954217910767, 0.00012180954217910767, 0.00012180954217910767]
- Exact tiny BPTT loss: 5.770318
- BPTT memory-controller gradient norms: {'memory.key.weight': 0.02291223593056202, 'memory.value.weight': 0.01643671840429306, 'memory.forget.weight': 2.073945779557107e-06, 'memory.forget.bias': 3.6748579077539034e-06, 'memory.rate.weight': 0.002864386886358261, 'memory.rate.bias': 0.0034120851196348667}
- Raw-angle cycle closure: {'period2': 0.001749983257311556, 'period3': 0.001749983257311556}
- Checkpoints: roundtrip and legacy rejection PASS
- Exact parameter count: 5952259
- 512x16 online metrics: {'loss': 7.338406085968018, 'byte': 6.988320350646973, 'latent': 0.9971076846122742, 'variance': 0.0, 'writer': 1.0080839395523071}
All assertions passed.
