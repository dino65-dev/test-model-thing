"""Held-out BPB plus recurrent-dynamics health report for a v4 checkpoint."""
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx

from main import EOS, Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("validation")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--memory-dim", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    model = Model(args.dim, args.layers, .75, 5e-4, memory_dim=args.memory_dim)
    model.load(args.checkpoint)
    data = list(Path(args.validation).read_bytes()) + [EOS]
    if len(data) < 2:
        raise ValueError("validation corpus must contain at least one byte")
    model.reset_memory()
    loss, count, health = 0., 0, None
    for current, target in zip(data, data[1:]):
        logits, _, states, _, _, write = model.represent(current)
        loss += (mx.logsumexp(logits) - logits[target]).item()
        count += 1
        health = model.health_metrics(write)
        if args.limit and count >= args.limit:
            break
    print(json.dumps({"heldout_bpb": loss / (count * math.log(2)), "bytes": count, "health": health}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
