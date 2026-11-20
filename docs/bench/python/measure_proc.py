#!/usr/bin/env python3
"""Measure one child process: wall time and peak RSS.

Usage:
    python measure_proc.py -- <command> [args...]

Runs the command once with subprocess.run, then prints one JSON line:

    {"wall_s": <float>, "maxrss_kb": <int>, "exit": <int>}

`maxrss_kb` comes from resource.getrusage(RUSAGE_CHILDREN).ru_maxrss, which is
KiB on Linux. The wrapper intentionally imports nothing from the SDK so that it
can measure any child process (Python or Rust).
"""

import argparse
import json
import resource
import subprocess
import sys
import time


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
    proc = subprocess.run(cmd)
    wall_s = time.perf_counter() - start

    maxrss_kb = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    print(json.dumps({"wall_s": wall_s, "maxrss_kb": maxrss_kb, "exit": proc.returncode}))
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
