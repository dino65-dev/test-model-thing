"""Genuine forward-mode depth-banded eligibility experiment for recurrent decay."""
import csv
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util

from main import Model

OUT = Path("artifacts/depth_banded_fidelity")


def zeros(tree):
    if isinstance(tree, dict): return {key: zeros(value) for key, value in tree.items()}
    if isinstance(tree, list): return [zeros(value) for value in tree]
    return mx.zeros(tree.shape)


def metric(exact, estimate):
    mx.eval(exact, estimate)
    a = mx.sqrt(mx.sum(exact * exact)).item()
    b = mx.sqrt(mx.sum(estimate * estimate)).item()
    return mx.sum(exact * estimate).item() / (a * b + 1e-12), mx.sqrt(mx.sum((exact-estimate)**2)).item() / (a + 1e-12)


def exact_decay(model, sequence):
    params = model.trainable_parameters()
    def lossfn(updated):
        model.update(updated)
        states, loss = [mx.zeros((model.dim,)) for _ in model.layers], mx.array(0.)
        for current, target in sequence:
            logits, _, states, _, _, _ = model.core(current, states_override=states)
            loss = loss + nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean()
        return loss / len(sequence)
    grads = mx.value_and_grad(lossfn)(params)[1]
    return [grads["layers"][i]["decay"] for i in range(model.layercount)]


def banded_decay(model, sequence, radius):
    """Forward-mode e_(source -> destination) with destination-source <= K."""
    parameter_tree = model.trainable_parameters()
    leaves = list(util.tree_flatten(parameter_tree))
    keys, parameter_values = [key for key, _ in leaves], [value for _, value in leaves]
    layer_decay_leaf = [next(i for i, key in enumerate(keys) if str(key) == f"layers.{layer}.decay") for layer in range(model.layercount)]
    states = [mx.zeros((model.dim,)) for _ in model.layers]
    out = [mx.zeros(layer.decay.shape) for layer in model.layers]
    eligibilities = [[[mx.zeros((model.dim,)) for _ in model.layers] for _ in range(layer.decay.shape[0])] for layer in model.layers]
    zero_parameters = [mx.zeros(value.shape) for value in parameter_values]
    for current, target in sequence:
        def transition(*arguments):
            updated = util.tree_unflatten(list(zip(keys, arguments[:len(keys)])))
            previous = list(arguments[len(keys):])
            model.update(updated)
            logits, _, next_states, _, _, _ = model.core(current, states_override=previous)
            loss = nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean()
            return (loss, *next_states)
        primals = parameter_values + states
        _, _ = mx.jvp(transition, primals, zero_parameters + [mx.zeros((model.dim,)) for _ in model.layers])
        # The primal state is independent of source parameter coordinate.
        primal, _ = mx.jvp(transition, primals, zero_parameters + [mx.zeros((model.dim,)) for _ in model.layers])
        next_states = list(primal[1:])
        for source, layer in enumerate(model.layers):
            for coordinate in range(layer.decay.shape[0]):
                parameter_tangent = list(zero_parameters)
                parameter_tangent[layer_decay_leaf[source]] = mx.eye(layer.decay.shape[0])[coordinate]
                state_tangent = [eligibilities[source][coordinate][destination] if destination - source <= radius else mx.zeros((model.dim,)) for destination in range(model.layercount)]
                _, tangent = mx.jvp(transition, primals, parameter_tangent + state_tangent)
                out[source] = out[source] + mx.eye(layer.decay.shape[0])[coordinate] * tangent[0] / len(sequence)
                for destination in range(model.layercount):
                    eligibilities[source][coordinate][destination] = tangent[destination + 1] if 0 <= destination - source <= radius else mx.zeros((model.dim,))
        states = next_states
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Memory is intentionally off: this isolates recurrent depth transport.
    for seed in range(3):
        mx.random.seed(600 + seed)
        model = Model(8, 4, .7, 1e-3, memory_enabled=False)
        sequence = [((seed + 3*i) % 17, (seed + 5*i + 1) % 17) for i in range(16)]
        exact = exact_decay(model, sequence)
        for radius in (0, 1, 2, 3):
            estimate = banded_decay(model, sequence, radius)
            for layer, (truth, value) in enumerate(zip(exact, estimate)):
                cosine, relative = metric(truth, value)
                rows.append({"seed": seed, "K": radius, "source_layer": layer, "cos": cosine, "rel": relative})
    assert all(math.isfinite(row["cos"]) and math.isfinite(row["rel"]) for row in rows)
    with (OUT / "raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for radius in (0, 1, 2, 3):
        values = [row for row in rows if row["K"] == radius]
        grouped[radius] = (sum(row["cos"] for row in values)/len(values), sum(row["rel"] for row in values)/len(values))
    report = ["# Depth-banded decay eligibility", "", "A genuine forward-mode sensitivity experiment for decay leaves only, L=4, T=16, 3 seeds, memory disabled. K restricts retained destination layers to destination-source <= K. K=3 is the complete depth transport reference within this four-layer test.", ""]
    report += [f"- K={k}: mean per-layer cosine={c:.4f}, mean relative error={r:.4f}" for k, (c, r) in grouped.items()]
    report += ["", "This is an experiment, not yet the production online estimator. It establishes whether adding cross-layer eligibility transport is worth its persistent-buffer and runtime cost."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(grouped)


if __name__ == "__main__":
    main()
