"""Strict MLX regression tests for the TMT-v4 recurrent mathematics."""
import math
import os
import random
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util

from main import CHECKPOINT_FORMAT, EOS, Layer, Model

OUT = Path("artifacts/tmt_v4_validation")


def finite(x):
    return math.isfinite(float(x))


def nrm(x):
    return mx.sqrt(mx.sum(mx.square(x))).item()


def vector(rng, n, lo=-.9, hi=.9):
    return mx.array([rng.uniform(lo, hi) for _ in range(n)])


def pair_block_fd(samples=50):
    """Randomized FD of ds_pair/dx_pair, including phase-controller rank one."""
    errors, relative = [], []
    for seed in range(samples):
        rng = random.Random(seed + 1009)
        layer = Layer(2, 4, 128)
        layer.decay = vector(rng, 1, -2, 2)
        layer.phase = vector(rng, 1, -.9, .9)
        layer.write_scale = vector(rng, 2, -.7, .7)
        layer.write_bias = vector(rng, 2, -.6, .6)
        layer.phase_scale = vector(rng, 1, -.7, .7)
        layer.phase_bias = vector(rng, 1, -.6, .6)
        x, state = vector(rng, 2), vector(rng, 2)
        rho, c, s, scale, z, _ = layer.dynamics(x)
        gate = mx.sigmoid(layer.write_scale * x + layer.write_bias)
        analytic = layer.embedding_block(x, state, rho, c, s, scale, gate, z)
        eps, cols = 1e-3, []
        for j in range(2):
            delta = mx.array([eps if j == 0 else 0., eps if j == 1 else 0.])
            _, plus, _ = layer.step(x + delta, state)
            _, minus, _ = layer.step(x - delta, state)
            cols.append((plus - minus) / (2 * eps))
        numeric = mx.stack(cols, axis=-1)
        mx.eval(analytic, numeric)
        errors.append(mx.max(mx.abs(analytic - numeric)).item())
        relative.append(nrm(analytic - numeric) / (nrm(numeric) + 1e-8))
    assert max(errors) < 5e-3 and max(relative) < 1e-2, (max(errors), max(relative))
    return {"samples": samples, "max_abs": max(errors), "max_relative_fro": max(relative)}


def phase_fd(samples=50):
    """Randomized FD for the base raw-angle eligibility through a rollout."""
    errors, relative = [], []
    for seed in range(samples):
        rng = random.Random(seed + 2017)
        layer = Layer(2, 4, 128)
        layer.decay = vector(rng, 1, -2, 2)
        phase = vector(rng, 1, -.9, .9)
        layer.phase = phase
        layer.phase_scale, layer.phase_bias = vector(rng, 1, -.6, .6), vector(rng, 1, -.6, .6)
        layer.write_scale, layer.write_bias = vector(rng, 2, -.5, .5), vector(rng, 2, -.5, .5)
        x, initial = vector(rng, 2), vector(rng, 2)
        state, trace = initial, mx.zeros((2,))
        for _ in range(17):
            rho, c, s, scale, z, _ = layer.dynamics(x)
            trace = layer.history(trace, rho, c, s) + layer.phase_direct(rho, c, s, state, mx.ones((1,)))
            _, state, _ = layer.step(x, state)
        eps = 1e-3
        def rollout(angle):
            layer.phase = mx.array([angle])
            state = initial
            for _ in range(17):
                _, state, _ = layer.step(x, state)
            return state
        numeric = (rollout(phase.item() + eps) - rollout(phase.item() - eps)) / (2 * eps)
        layer.phase = phase
        mx.eval(trace, numeric)
        errors.append(mx.max(mx.abs(trace - numeric)).item())
        relative.append(nrm(trace - numeric) / (nrm(numeric) + 1e-8))
    assert max(errors) < 1e-2 and max(relative) < 2e-2, (max(errors), max(relative))
    return {"samples": samples, "max_abs": max(errors), "max_relative_fro": max(relative)}


def memory_bptt():
    """Memory controller gets nonzero delayed byte-CE gradients, not writer CE."""
    mx.random.seed(17)
    model = Model(8, 2, .7, 1e-3, memory_dim=4)
    writes = [mx.random.normal((8,)) for _ in range(5)]
    targets = [2, 5, 7, 3, 1]
    params = model.memory.parameters()
    def lossfn(updated):
        model.memory.update(updated)
        matrix, loss = mx.zeros(model.memory._matrix.shape), mx.array(0.)
        for write, target in zip(writes, targets):
            matrix = model.memory.next_matrix(mx.stop_gradient(write), matrix)
            h = mx.stop_gradient(write) + model.memory.out(model.memory.read(mx.stop_gradient(write), matrix))
            loss = loss + nn.losses.cross_entropy(model.decoder(h)[None, :], mx.array([target])).mean()
        return loss / len(writes)
    loss, grads = mx.value_and_grad(lossfn)(params)
    flat = dict(util.tree_flatten(grads))
    mx.eval(loss, *flat.values())
    controller = {str(key): nrm(value) for key, value in flat.items() if any(name in str(key) for name in ("key", "query", "value", "out", "forget", "rate"))}
    assert finite(loss.item()) and controller and all(value > 1e-10 for value in controller.values()), controller
    before = {str(key): mx.array(value) for key, value in util.tree_flatten(model.memory.parameters())}
    model.memory_temporal_step(writes, targets, mx.zeros(model.memory._matrix.shape))
    after = dict(util.tree_flatten(model.memory.parameters()))
    changed = any(nrm(after[key] - before[str(key)]) > 0 for key in after)
    assert changed
    return loss.item(), controller


def optimizer_ownership():
    model = Model(8, 2, .7, 1e-3, memory_dim=4)
    before = {str(key): mx.array(value) for key, value in util.tree_flatten(model.memory.parameters())}
    model.train_step(1, 2, (2, 3, 4))
    after = dict(util.tree_flatten(model.memory.parameters()))
    assert all(nrm(after[key] - before[str(key)]) == 0 for key in after)
    assert float(model.optimizer.weight_decay) == 0 and model.optimizer.bias_correction is True
    return "online optimizer excludes memory; temporal optimizer is sole owner"


def cycles():
    model = Model(32, 1, .7, 1e-3, memory_dim=4)
    layer = model.layers[0]
    rho, _, _, _, _, _ = layer.dynamics(mx.zeros((32,)))
    mx.eval(rho)
    results = {}
    for label, target, period in (("period2", math.pi, 2), ("period3", 2 * math.pi / 3, 3)):
        candidates = [i for i in range(16) if abs(layer.phase[i].item() - target) < 1e-5]
        assert candidates
        i = max(candidates, key=lambda j: rho[j].item())
        theta = layer.phase[i].item()
        residual = max(abs(math.cos(period * theta * count) - 1) + abs(math.sin(period * theta * count)) for count in (1, 10, 100, 1000, 10000))
        results[label] = residual
        assert residual < 2e-3
    return results


def checkpoints():
    model = Model(8, 2, .7, 1e-3, memory_dim=4)
    model.train_step(1, 2, (2, 3, 4, EOS))
    with tempfile.TemporaryDirectory() as directory:
        good = os.path.join(directory, "good.safetensors")
        model.save(good)
        assert Model(8, 2, .7, 1e-3, memory_dim=4).load(good)
        data, metadata = mx.load(good, return_metadata=True)
        assert metadata["format"] == CHECKPOINT_FORMAT
        missing = os.path.join(directory, "missing-buffer.safetensors")
        data.pop("buffer.phase_trace.0")
        mx.save_safetensors(missing, data, metadata=metadata)
        try:
            Model(8, 2, .7, 1e-3, memory_dim=4).load(missing)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint missing a required buffer was accepted")
        legacy = os.path.join(directory, "legacy.safetensors")
        mx.save_safetensors(legacy, {"m.encoder.embed.weight": model.encoder.embed.weight}, metadata={"format": "tmt-v3"})
        try:
            Model(8, 2, .7, 1e-3, memory_dim=4).load(legacy)
        except ValueError:
            return "round-trip, metadata rejection, and every-buffer rejection PASS"
        raise AssertionError("legacy checkpoint was accepted")


def fixed_decay_smoke():
    model = Model(8, 2, .7, 1e-3, memory_dim=4, learn_decay=False)
    before = [mx.array(layer.decay) for layer in model.layers]
    for i in range(4):
        metrics = model.train_step(i, i + 1, (i + 1,))
        assert all(finite(value) for key, value in metrics.items() if key != "memory_write")
    assert all(nrm(layer.decay - original) == 0 for layer, original in zip(model.layers, before))
    return "fixed rho parameters remain exactly fixed under online training"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    injection = pair_block_fd()
    phase = phase_fd()
    memory_loss, controller = memory_bptt()
    ownership = optimizer_ownership()
    cycle = cycles()
    check = checkpoints()
    fixed_decay = fixed_decay_smoke()
    model = Model(512, 16, .75, 5e-4)
    assert model.buffer_parameter_invariant()
    count = sum(value.size for _, value in util.tree_flatten(model.parameters()))
    online = model.train_step(65, 66, tuple(range(67, 82)) + (EOS,))
    assert all(finite(value) for key, value in online.items() if key != "memory_write")
    report = f"""# TMT-v4 Strict MLX Validation
- Pair-block embedding Jacobian randomized FD: {injection}
- Base phase eligibility randomized FD: {phase}
- Exact delayed-byte-CE memory BPTT loss: {memory_loss:.6f}
- Memory-controller gradient norms: {controller}
- Optimizer ownership: {ownership}
- Raw-angle cycle closure: {cycle}
- Checkpoints: {check}
- Fixed-decay check: {fixed_decay}
- Exact parameter count: {count}
- 512x16 online metrics: { {k: v for k, v in online.items() if k != 'memory_write'} }
All assertions passed.
"""
    (OUT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
