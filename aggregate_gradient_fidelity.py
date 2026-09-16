"""Aggregate independent gradient-fidelity seed shards without recomputation."""
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import gradient_fidelity as fidelity

OUT = Path("artifacts/gradient_fidelity")


def cast(row):
    for key in ("layers", "memory", "seed", "T"):
        row[key] = int(row[key])
    for key in ("cos", "rel", "norm", "energy_share"):
        if key in row:
            row[key] = float(row[key])
    return row


def mean_std(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def main():
    roots = sorted(glob.glob("artifacts/gradient_fidelity/runs/**/seed_*", recursive=True))
    if len(roots) != 20:
        raise ValueError(f"expected 20 complete shards, found {len(roots)}")
    raw, families = [], []
    for root in roots:
        with open(Path(root) / "raw.csv", newline="") as file:
            raw.extend(cast(row) for row in csv.DictReader(file))
        with open(Path(root) / "families.csv", newline="") as file:
            families.extend(cast(row) for row in csv.DictReader(file))
    keys = {(row["layers"], row["memory"], row["seed"], row["T"], row["mode"]) for row in raw}
    if len(keys) != len(raw):
        raise ValueError("duplicated whole-model measurements")
    if {row["seed"] for row in raw} != set(range(10)):
        raise ValueError("missing seed in whole-model measurements")
    buckets = defaultdict(list)
    for row in raw:
        buckets[(row["layers"], row["memory"], row["T"], row["mode"])].append(row)
    summary = []
    for (layers, memory, horizon, mode), values in sorted(buckets.items()):
        summary.append({"layers": layers, "memory": memory, "T": horizon, "mode": mode, "seeds": len(values),
                        **{f"{metric}_mean": mean_std([row[metric] for row in values])[0] for metric in ("cos", "rel", "norm")},
                        **{f"{metric}_std": mean_std([row[metric] for row in values])[1] for metric in ("cos", "rel", "norm")}})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("raw.csv", raw), ("families.csv", families), ("summary.csv", summary)):
        with (OUT / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    fidelity.OUT = OUT
    fidelity.write_svg(summary)
    report = ["# Structured online-gradient fidelity", "", "Measured against exact BPTT across 10 independently initialized random byte sequences per configuration (L=1/2 through T=128; T=256 includes L=1/2/4).", "", "Memory-on rows compare backbone gradients only; the memory controller is intentionally owned by separate temporal BPTT and therefore excluded from this online-gradient comparison.", "", "## Aggregate whole-model metrics", ""]
    for row in summary:
        report.append(f"- L={row['layers']}, memory={row['memory']}, T={row['T']}, {row['mode']}: cosine={row['cos_mean']:.4f} +- {row['cos_std']:.4f}, relative={row['rel_mean']:.4f}, norm={row['norm_mean']:.4f}")
    report.extend(["", "`families.csv` contains both structured-trace and instantaneous metrics plus each family’s exact-gradient energy share. This measures a structured approximation; it does not claim full RTRL."])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print({"raw": len(raw), "families": len(families), "summary": len(summary), "seeds": 10})


if __name__ == "__main__":
    main()
