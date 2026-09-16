"""Architectural recurrent ablations with multi-seed length extrapolation."""
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from main import Model

OUT = Path("artifacts/synthetic_ablations")


def sequence(task, length, delay, rng):
    bits = [rng.randrange(2) for _ in range(length)]
    if task == "parity":
        state, targets = 0, []
        for bit in bits:
            state ^= bit
            targets.append(state)
    elif task.startswith("mod"):
        base, state, targets = int(task[3:]), 0, []
        for bit in bits:
            state = (state + bit) % base
            targets.append(state)
    elif task == "xor":
        targets = [0 if i < delay else bits[i] ^ bits[i - delay] for i in range(length)]
    elif task == "copy":
        targets = [0 if i < delay else bits[i - delay] for i in range(length)]
    else:
        raise ValueError(task)
    return bits, targets


def train_episode(model, task, length, delay, rng):
    model.reset_memory()
    bits, targets = sequence(task, length, delay, rng)
    writes, delayed_targets = [], []
    block_start = mx.stop_gradient(model.memory._matrix)
    for bit, target in zip(bits, targets):
        metrics = model.train_step(bit, target, (target,))
        if model.memory_enabled:
            writes.append(metrics["memory_write"])
            delayed_targets.append(target)
            if len(writes) == model.memory_bptt_block:
                model.memory_temporal_step(writes, delayed_targets, block_start)
                writes, delayed_targets = [], []
                block_start = mx.stop_gradient(model.memory._matrix)
    # Do not discard the final short sequence: it has valid bounded BPTT too.
    if writes:
        model.memory_temporal_step(writes, delayed_targets, block_start)


def score(model, task, length, delay, episodes, seed):
    rng, right, total = random.Random(seed), 0, 0
    for _ in range(episodes):
        model.reset_memory()
        bits, targets = sequence(task, length, delay, rng)
        for bit, target in zip(bits, targets):
            right += mx.argmax(model.frozen_step(bit)).item() == target
            total += 1
    return right / total


def build(condition, seed):
    complex_on = condition != "no_complex"
    memory_on = condition != "no_memory"
    learn_decay = condition != "fixed_decay"
    mx.random.seed(seed)
    return Model(16, 2, .7, 2e-3, memory_dim=8, future_block=4,
                 memory_enabled=memory_on, complex_enabled=complex_on,
                 learn_decay=learn_decay, memory_bptt_block=8)


def mean_std(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128, help="training episodes per seed")
    parser.add_argument("--length", type=int, default=16, help="training sequence length")
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=128, help="evaluation episodes per point")
    parser.add_argument("--eval-lengths", default="16,32,64")
    args = parser.parse_args()
    eval_lengths = [int(value) for value in args.eval_lengths.split(",")]
    tasks = ["parity", "mod3", "mod5", "xor", "copy"]
    conditions = ["full", "no_complex", "no_memory", "fixed_decay"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        for condition in conditions:
            for seed in range(args.seeds):
                rng = random.Random(10_000 * (tasks.index(task) + 1) + seed)
                model = build(condition, seed)
                for _ in range(args.steps):
                    train_episode(model, task, args.length, args.delay, rng)
                for length in eval_lengths:
                    rows.append({"task": task, "condition": condition, "seed": seed,
                                 "train_length": args.length, "eval_length": length,
                                 "accuracy": score(model, task, length, args.delay, args.episodes, 20_000 + seed)})
    summary = []
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["task"], row["condition"], row["train_length"], row["eval_length"])].append(row["accuracy"])
    for (task, condition, train_length, eval_length), values in sorted(buckets.items()):
        mean, std = mean_std(values)
        summary.append({"task": task, "condition": condition, "train_length": train_length,
                        "eval_length": eval_length, "seeds": len(values), "accuracy_mean": mean, "accuracy_std": std})
    with (OUT / "raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    with (OUT / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(summary)
    report = ["# Synthetic architectural ablations", "", f"{args.seeds} seeds; train length {args.length}; evaluation lengths {eval_lengths}.", "", "`no_complex` bypasses every complex rotation/controller path and freezes those leaves. `no_memory` bypasses both memory read and matrix update. `fixed_decay` freezes rho. All memory-enabled conditions receive the same bounded future-byte CE memory BPTT.", ""]
    for row in summary:
        report.append(f"- {row['task']} | {row['condition']} | train {row['train_length']} -> eval {row['eval_length']}: {row['accuracy_mean']:.4f} +- {row['accuracy_std']:.4f}")
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"wrote {len(rows)} seed-level and {len(summary)} aggregate ablation measurements")


if __name__ == "__main__":
    main()
