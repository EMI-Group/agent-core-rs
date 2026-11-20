#!/usr/bin/env python3
"""Python-side micro-benchmarks for the Rust-vs-Python comparison.

Workloads (same shape as the Rust harness):
  prompt  -- 2000 PromptTemplate.format() calls per rep, 30 reps (3 discarded).
  graph   -- 200 PregelBuilder graph runs per rep, 30 reps (3 discarded).

Each workload writes raw per-repetition JSON to ./docs/bench/results/python/
<workload>.json and prints the same JSON to stdout.

Usage:
    python bench.py [prompt|graph|all]

No network, no LLM calls.
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import time

REPS = 30
REPS_DISCARDED = 3
PROMPT_ITERATIONS = 2000
GRAPH_ITERATIONS = 200

TEMPLATE_TEXT = (
    "You are {{role}}. Answer the question: {{question}} in at most {{limit}} words."
)
KEYWORDS = {"role": "a terse assistant", "question": "What is a monad?", "limit": "20"}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "python")
PY_VERSION = ".".join(str(p) for p in sys.version_info[:3])


def silence_logging():
    """Keep log formatting out of the timed sections."""
    logging.disable(logging.CRITICAL)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("openjiuwen"):
            logging.getLogger(name).setLevel(logging.CRITICAL)


def write_result(workload, iterations, reps):
    payload = {
        "workload": workload,
        "reps_discarded": REPS_DISCARDED,
        "iterations_per_rep": iterations,
        "reps": reps,
        "python": PY_VERSION,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, workload + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps(payload))
    return payload


def run_prompt():
    from openjiuwen.core.foundation.prompt import PromptTemplate

    template = PromptTemplate(name="bench", content=TEMPLATE_TEXT)

    # Sanity check: interpolation must work before any timing is trusted.
    probe = template.format(KEYWORDS)
    if "{{" in probe.content or "monad" not in probe.content:
        raise RuntimeError(f"template did not interpolate: {probe.content!r}")

    def batch():
        for _ in range(PROMPT_ITERATIONS):
            template.format(KEYWORDS)

    reps = []
    for i in range(REPS):
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            batch()
        reps.append({"rep": i, "ms": (time.perf_counter() - start) * 1000.0})
    return write_result("prompt", PROMPT_ITERATIONS, reps)


def graph_main():
    from openjiuwen.core.graph.pregel import END, START, PregelBuilder

    state = {"a": 0, "b": 0}

    def node_a():
        state["a"] += 1

    def node_b():
        state["b"] += 1

    pregel = (
        PregelBuilder()
        .add_node("a", node_a)
        .add_node("b", node_b)
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .build()
    )

    async def main():
        # One event loop for the whole benchmark (asyncio.run creates it once).
        await pregel.run()
        if state != {"a": 1, "b": 1}:
            raise RuntimeError(f"graph did not execute all nodes in warmup: {state!r}")

        reps = []
        for i in range(REPS):
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(GRAPH_ITERATIONS):
                    await pregel.run()
            reps.append({"rep": i, "ms": (time.perf_counter() - start) * 1000.0})
        return write_result("graph", GRAPH_ITERATIONS, reps)

    return asyncio.run(main())


def main(argv):
    workload = argv[1] if len(argv) > 1 else "all"
    if workload not in ("prompt", "graph", "all"):
        print("usage: bench.py [prompt|graph|all]", file=sys.stderr)
        return 2

    silence_logging()
    if workload in ("prompt", "all"):
        run_prompt()
    if workload in ("graph", "all"):
        graph_main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
