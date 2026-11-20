#!/usr/bin/env python3
"""Recompute the tables in docs/BENCHMARKS.md from the raw result files.

Usage:
    python aggregate.py [--raw]

Reads docs/bench/results/{process,python,rust}/*.json and prints the summary
table, plus (with --raw) the per-repetition values behind each median.
Nothing here re-runs a measurement; it only aggregates what was recorded.
"""
import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(path):
    with open(path) as handle:
        text = handle.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pct(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def process_tables(raw):
    print("## Process floor (20 fresh processes per row)\n")
    print("| measurement | wall median ms | wall p95 ms | RSS median KB | RSS p95 KB |")
    print("| --- | --- | --- | --- | --- |")
    for path in sorted(glob.glob(os.path.join(RESULTS, "process", "*.json"))):
        rows = load(path)
        wall = [row["wall_s"] * 1000.0 for row in rows]
        rss = [row["maxrss_kb"] for row in rows]
        print(
            "| {} | {:.2f} | {:.2f} | {} | {} |".format(
                os.path.basename(path)[:-5],
                median(wall),
                pct(wall, 0.95),
                median(rss),
                pct(rss, 0.95),
            )
        )


def workload_tables(raw):
    print("\n## Workloads (30 reps, first 3 discarded)\n")
    print("| side | workload | iterations/rep | median ms | p95 ms | per iteration |")
    print("| --- | --- | --- | --- | --- | --- |")
    for side in ("rust", "python"):
        for workload in ("prompt", "graph"):
            for path in sorted(glob.glob(os.path.join(RESULTS, side, f"{workload}_run*.json"))):
                data = json.load(open(path))
                values = [row["ms"] for row in data["reps"]][data["reps_discarded"]:]
                iterations = data["iterations_per_rep"]
                unit = "us" if median(values) * 1000.0 / iterations < 1000 else "ms"
                per = median(values) * (1000.0 if unit == "us" else 1.0) / iterations
                print(
                    "| {} | {} | {} | {:.3f} | {:.3f} | {:.4f} {} |".format(
                        side,
                        workload,
                        iterations,
                        median(values),
                        pct(values, 0.95),
                        per,
                        unit,
                    )
                )
                if raw:
                    print("|  -> reps | | | | | {} |".format(
                        ", ".join("{:.3f}".format(v) for v in values)))


if __name__ == "__main__":
    process_tables("--raw" in sys.argv)
    workload_tables("--raw" in sys.argv)
