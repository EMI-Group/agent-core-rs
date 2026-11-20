#!/usr/bin/env python3
"""Measure one child process: wall time and peak RSS.

Usage:
    python measure_proc.py -- <command> [args...]

Prints one JSON line:
    {"wall_s": <float>, "maxrss_kb": <int>, "exit": <int>}

The command runs twice:

1. Timed directly with `subprocess.run` and `time.perf_counter`, which gives a
   millisecond-precision wall time for short-lived processes.
2. Re-run under GNU `time -v` to read "Maximum resident set size (kbytes)".

Two reasons for the split. `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`
cannot be used from this script: the child is forked from a full CPython
process, and Linux keeps the pre-exec peak, so every child would report this
interpreter's own ~16 MB regardless of its real footprint. GNU `time` forks
from a ~2 MB image, so its peak reflects the measured binary. And GNU time's
own "Elapsed" field only has 10 ms resolution, which is useless for a 2 ms
process.

Both measured workloads are deterministic and side-effect free, so the second
run measures the same work as the first.

The measured binary's output is discarded; it is not part of the metric.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def time_binary():
    """Absolute path of GNU time, which is not on PATH on NixOS by default."""
    for candidate in (
        os.environ.get("BENCH_TIME_BIN"),
        "/run/current-system/sw/bin/time",
        shutil.which("time"),
    ):
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit("GNU time not found; set BENCH_TIME_BIN to its absolute path")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no command given; usage: measure_proc.py -- <command> [args...]")

    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall_s = time.perf_counter() - start

    measured = subprocess.run(
        [time_binary(), "-v", *cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = RSS_RE.search(measured.stderr)
    maxrss_kb = int(match.group(1)) if match else -1

    print(json.dumps({"wall_s": wall_s, "maxrss_kb": maxrss_kb, "exit": proc.returncode}))
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
