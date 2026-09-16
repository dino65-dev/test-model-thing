"""Measured structured-online-gradient fidelity against exact short BPTT."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util

from main import Model

OUT = Path("artifacts/gradient_fidelity")


class Capture:
    def __init__(self):
        self.state, self.grads = {}, None

    def update(self, module, grads):
        self.grads = grads


def flatvec(tree):
    values = [value.reshape((-1,)) for _, value in util.tree_flatten(tree)]
    return mx.concatenate(values) if values else mx.zeros((0,))


def norm(x):
    return mx.sqrt(mx.sum(x * x))


def metrics(exact_grad, estimated_grad):
    mx.eval(exact_grad, estimated_grad)
    a, b = norm(exact_grad).item(), norm(estimated_grad).item()
    return {
        "cos": mx.sum(exact_grad * estimated_grad).item() / (a * b + 1e-12),
        "rel": norm(exact_grad - estimated_grad).item() / (a + 1e-12),
        "norm": b / (a + 1e-12),
    }


def exact(model, sequence):
    model.reset_memory()
    parameters = model.trainable_parameters()
    def lossfn(updated):
        model.update(updated)
        states = [mx.zeros((model.dim,)) for _ in model.layers]
        matrix = mx.zeros(model.memory._matrix.shape)
        loss = mx.array(0.)
        for current, target in sequence:
            logits, _, states, _, _, base = model.core(current, states_override=states, matrix_override=matrix)
            loss = loss + nn.losses.cross_entropy(logits[None, :], mx.array([target])).mean()
            if model.memory_enabled:
                matrix = model.memory.next_matrix(base, matrix)
        return loss / len(sequence)
    return mx.value_and_grad(lossfn)(parameters)[1]


def online(model, sequence, use_trace):
    model.reset_memory()
    original = model.optimizer
    capture = Capture()
    model.optimizer = capture
    old_weights = (model.latent_weight, model.variance_weight)
    model.latent_weight = model.variance_weight = 0.
    total = {}
    try:
        for current, target in sequence:
            if not use_trace:
                for layer in model.layers:
                    layer._decay_trace = mx.zeros((model.dim,))
                    layer._phase_trace = mx.zeros((model.dim,))
                    layer._write_scale_trace = mx.zeros((model.dim,))
                    layer._write_bias_trace = mx.zeros((model.dim,))
                    layer._phase_scale_trace = mx.zeros((model.dim,))
                    layer._phase_bias_trace = mx.zeros((model.dim,))
                model.encoder._embed_trace = mx.zeros(model.encoder._embed_trace.shape)
            model.train_step(current, target, (target,))
            for key, value in util.tree_flatten(capture.grads):
                total[key] = total.get(key, mx.zeros(value.shape)) + value
    finally:
        model.optimizer = original
        model.latent_weight, model.variance_weight = old_weights
    return util.tree_unflatten([(key, value / len(sequence)) for key, value in total.items()])


def groups(tree):
    out = {}
    for key, value in util.tree_flatten(tree):
        name = str(key)
        family = (
            "embedding" if name.startswith("encoder") else
            "decay" if name.endswith(".decay") else
            "phase" if name.endswith(".phase") else
            "write" if "write_" in name else
            "phase_controller" if "phase_" in name else
            "decoder" if name.startswith("decoder") else
            "predictor" if name.startswith("predictors") else
            "other"
        )
        out.setdefault(family, []).append(value.reshape((-1,)))
    return {key: mx.concatenate(value) for key, value in out.items()}


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def write_svg(rows):
    # Compact plot: aggregated whole-model cosine as horizon grows.
    w, h, pad = 900, 470, 55
    series = defaultdict(list)
    for row in rows:
        series[(row["layers"], row["memory"], row["mode"])].append(row)
    xs = sorted({row["T"] for row in rows})
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5"]
    def point(x, y):
        return pad + (x - min(xs)) / max(max(xs) - min(xs), 1) * (w - 2 * pad), h - pad - y * (h - 2 * pad)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="55" y="28" font-family="sans-serif">Gradient fidelity: cosine to exact BPTT</text>']
    for index, (key, values) in enumerate(sorted(series.items())):
        values.sort(key=lambda row: row["T"])
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row["T"], row["cos_mean"]) for row in values))
        color = colors[index % len(colors)]
        label = f"L{key[0]} mem={key[1]} {key[2]}"
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/><text x="{590}" y="{55 + 18 * index}" fill="{color}" font-family="sans-serif" font-size="13">{label}</text>')
    for x in xs:
        px, py = point(x, 0)
        parts.append(f'<text x="{px - 8:.1f}" y="{h - 25}" font-family="sans-serif" font-size="12">{x}</text>')
    parts.append('</svg>')
    (OUT / "cosine.svg").write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--layers-list", default="1,2", help="comma-separated recurrent depths")
    parser.add_argument("--min-t", type=int, default=2)
    parser.add_argument("--max-t", type=int, default=128, choices=(2, 4, 8, 16, 32, 64, 128, 256))
    parser.add_argument("--output", default="artifacts/gradient_fidelity")
    args = parser.parse_args()
    global OUT
    OUT = Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)
    horizons = [value for value in (2, 4, 8, 16, 32, 64, 128, 256) if args.min_t <= value <= args.max_t]
    layer_values = [int(value) for value in args.layers_list.split(",")]
    if not horizons or any(value < 1 for value in layer_values):
        raise ValueError("provide valid layers and a nonempty horizon range")
    raw, family_rows = [], []
    for layers in layer_values:
        for memory in (False, True):
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for horizon in horizons:
                    mx.random.seed(1000 * layers + 100 * int(memory) + seed)
                    model = Model(8, layers, .7, 1e-3, memory_dim=4, memory_enabled=memory)
                    sequence = [((seed * 17 + i * 7) % 19, (seed * 29 + i * 11 + 1) % 19) for i in range(horizon)]
                    exact_grad = exact(model, sequence)
                    traced = online(model, sequence, True)
                    instantaneous = online(model, sequence, False)
                    exact_vector = flatvec(exact_grad)
                    exact_groups, estimates = groups(exact_grad), {"trace": groups(traced), "instant": groups(instantaneous)}
                    for mode, estimate in (("trace", traced), ("instant", instantaneous)):
                        result = metrics(exact_vector, flatvec(estimate))
                        raw.append({"layers": layers, "memory": int(memory), "seed": seed, "T": horizon, "mode": mode, **result})
                        for family, exact_family in exact_groups.items():
                            if family not in estimates[mode]:
                                continue
                            item = metrics(exact_family, estimates[mode][family])
                            item["energy_share"] = norm(exact_family).item() ** 2 / (norm(exact_vector).item() ** 2 + 1e-20)
                            family_rows.append({"layers": layers, "memory": int(memory), "seed": seed, "T": horizon, "mode": mode, "family": family, **item})
    for row in raw + family_rows:
        assert all(math.isfinite(float(value)) for key, value in row.items() if key not in ("mode", "family"))
    summary = []
    buckets = defaultdict(list)
    for row in raw:
        buckets[(row["layers"], row["memory"], row["T"], row["mode"])].append(row)
    for (layers, memory, horizon, mode), values in sorted(buckets.items()):
        summary.append({
            "layers": layers, "memory": memory, "T": horizon, "mode": mode, "seeds": len(values),
            **{f"{metric}_mean": mean_std([row[metric] for row in values])[0] for metric in ("cos", "rel", "norm")},
            **{f"{metric}_std": mean_std([row[metric] for row in values])[1] for metric in ("cos", "rel", "norm")},
        })
    fields = list(raw[0])
    with (OUT / "raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(raw)
    with (OUT / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(summary)
    with (OUT / "families.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(family_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(family_rows)
    write_svg(summary)
    report = ["# Structured online-gradient fidelity", "", f"Measured against exact BPTT across {args.seeds} random deterministic byte sequences per configuration.", "", "The memory-on rows compare backbone parameters only: memory controller is intentionally owned by its separate temporal-BPTT optimizer.", "", "## Aggregate whole-model metrics", ""]
    for row in summary:
        report.append(f"- L={row['layers']}, memory={row['memory']}, T={row['T']}, {row['mode']}: cosine={row['cos_mean']:.4f} +- {row['cos_std']:.4f}, relative={row['rel_mean']:.4f}, norm={row['norm_mean']:.4f}")
    report.extend(["", "`families.csv` includes both trace and instantaneous metrics plus exact-gradient energy shares. These quantify an approximation; they are not a claim of full RTRL."])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"wrote {len(raw)} whole-model and {len(family_rows)} per-family measurements")


if __name__ == "__main__":
    main()
