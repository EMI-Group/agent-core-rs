# Benchmarks

## Scope

Raw measurement appendix for the Rust rewrite (`agent-core-rs`) against the Python reference SDK (`openJiuwen agent-core` v0.1.20).
It records process-level cost (wall time, peak RSS) and two pure-local operations, with the exact commands, sample sizes, per-repetition values and the limits of what the numbers mean.
Headline figures are only comparable where this document says the two sides run the same semantics.

Everything here is reproduceable from `docs/bench/` and the raw JSON under `docs/bench/results/`.

## Environment

| Item | Value |
| --- | --- |
| CPU | Intel Core Ultra 5 325, 8 logical CPUs, 1 thread/core, 3.2 GHz base / 4.5 GHz boost |
| RAM | 24,725,495,808 bytes (23.0 GiB) total |
| OS | NixOS 26.11, kernel `7.3.5` (`#1-NixOS SMP PREEMPT_DYNAMIC`), x86_64 |
| Rust | `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `cargo 1.99.0`, LLVM 23.1.3 |
| Rust profile | `release`: `lto = "thin"`, `codegen-units = 1`, `opt-level = 3` (both the workspace and the bench package) |
| Python | CPython 3.11.15 (uv-managed standalone build), interpreter floor measured too |
| Python package | `openjiuwen` 0.1.20, installed from an out-of-tree copy of `/home/bill/Source/agent-core` (`HEAD 455e2a338903c173be34f5453de0290a989ba1e0`) |
| Python deps | resolved from PyPI at install time, not from the repo's `uv.lock`: numpy 2.4.6, pandas 3.1.1, transformers 5.23.0, sqlalchemy 2.1.9, pydantic 2.14.4, grpcio 1.85.3, tiktoken 0.14.0, fastmcp 3.4.11, aiohttp 3.14.5 (173 distributions) |
| Host state | shared machine, 1-minute load average 2.7–3.2 (5-minute 4.6) on 8 CPUs during the reported rounds |
| Page cache | warm for every measurement (see [Limitations](#limitations-and-what-this-does-not-measure)) |

The reference repo is never written to: it is rsync-copied to `$TMPDIR/src/agent-core` (282 MB), installed from that copy, and every measurement script runs with `PYTHONDONTWRITEBYTECODE=1`.

## What was measured

Four process-level rows, each with an explicit semantic label, and two pure-local operations that run the same code path on both sides.

| id | Rust side | Python side | Same semantics? |
| --- | --- | --- | --- |
| `rust_bench_floor` | 1.95 MB `openjiuwen-bench floor` binary: constructs a `PromptTemplate` and formats it, compiles a graph, creates a `WorkflowSession`, prints `openjiuwen::VERSION` | — | Rust only (Python has no static-link equivalent) |
| `rust_cli_version` | `openjiuwen --version` (34.0 MB fully linked CLI, clap arg parse, no SDK init) | — | no Python equivalent |
| `rust_hello_agent` | `01_hello_agent` example: card → agent → scripted-model turn → stream a second turn | — | Python side not measured (no built-in scripted model in the reference SDK) |
| `py_noop` | — | `python -c "pass"` | interpreter floor only |
| `py_import` | — | `python -c "import openjiuwen"` (94 modules) | process + package-import floor |
| `prompt` | `PromptTemplate::format` 20 000×/rep | `PromptTemplate.format` 2 000×/rep | **yes** — same template text, same keywords, same call |
| `graph` | 2-node linear graph, 2 000 runs/rep | 2-node linear pregel graph, 200 runs/rep | **yes** — same logical `entry → a → b → exit`; Python spells the endpoints as explicit `START`/`END` no-op nodes, Rust as builder entry/exit |

The prompt workload renders `"You are {{role}}. Answer the question: {{question}} in at most {{limit}} words."` with `role="a terse assistant"`, `question="What is a monad?"`, `limit="20"` — 85 bytes of output per call.
The graph workload runs two nodes that each increment a counter in state and checks the result, with no conditionals, barriers, sleeps or I/O.

## Method

### Rust

```
cargo build --release -j 4 -p openjiuwen-cli --bins
cargo build --release -j 4 -p openjiuwen-examples --example 01_hello_agent
CARGO_TARGET_DIR=$TMPDIR/bench-target cargo build --release --offline -j 4 \
    --manifest-path docs/bench/rust/Cargo.toml
```

`docs/bench/rust` is its own package with an empty `[workspace]` table and path dependencies on the shipped crates, so the root workspace is untouched.
The release profile matches the workspace (`lto = "thin"`, `codegen-units = 1`).

### Python

```
uv venv --python 3.11 "$TMPDIR/ojpy"
rsync -a --exclude=.git /home/bill/Source/agent-core/ "$TMPDIR/src/agent-core/"
uv pip install --python "$TMPDIR/ojpy/bin/python" "$TMPDIR/src/agent-core"
```

The install succeeds online; dependency resolution is not pinned to `uv.lock`, so the Python numbers include whatever the current PyPI versions cost to import.

### Timing protocol

- One timing sample per repetition: `Instant::now()` (Rust) / `time.perf_counter()` (Python) around a whole batch of iterations.
- 30 repetitions, the first 3 discarded as warmup; median and p95 (nearest-rank, `round((n-1)*0.95)`) over the 30 recorded values.
- Batch sizes differ by side because the per-operation cost differs by ~20×: Rust uses 20 000 renders and 2 000 graph runs per rep, Python uses 2 000 and 200, so every batch is ≥ 7 ms and clear of timer and scheduler noise. Per-operation figures are the batch median divided by the batch size.
- Both workloads were run three times per side, interleaved A/B/A/B, to expose drift on the shared host.
- Python timings are taken with `logging.disable(logging.CRITICAL)` and per-logger levels for `openjiuwen.*`, and stdout redirected, so log formatting is outside the timed section.
- The graph benchmark reuses one prebuilt graph and one asyncio event loop (Python) / one current-thread tokio runtime (Rust); graph construction is outside the timed section.

### Process protocol

Twenty fresh processes per row, driven by `docs/bench/python/measure_proc.py`:

- wall time from `time.perf_counter()` around `subprocess.run`;
- peak RSS from a second run under GNU `time -v` (`Maximum resident set size`).

RSS cannot be measured with `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` from a Python driver: the child is forked from a ~13.7 MiB CPython image and Linux keeps that pre-`exec` peak, so every child (including a 2.1 MiB Rust hello-world) reported ~16.4 MiB.
GNU `time` forks from a ~2 MiB image, so its peak reflects the measured binary.

## Results

### Process level (20 fresh processes per row, warm page cache)

| measurement | wall median | wall p95 | peak RSS median | peak RSS p95 |
| --- | --- | --- | --- | --- |
| `rust_bench_floor` (umbrella-linked SDK binary, 1.95 MB) | **1.20 ms** | 1.29 ms | **3.60 MiB** | 3.68 MiB |
| `rust_cli_version` (`openjiuwen --version`, 34.0 MB binary) | 2.45 ms | 2.95 ms | 8.82 MiB | 9.69 MiB |
| `rust_hello_agent` (scripted agent turn end-to-end, 19.7 MB binary) | 2.85 ms | 3.28 ms | 8.55 MiB | 8.76 MiB |
| `py_noop` (`python -c "pass"`) | 16.58 ms | 25.36 ms | 13.66 MiB | 13.83 MiB |
| `py_import` (`python -c "import openjiuwen"`) | **51.60 ms** | 58.27 ms | **19.96 MiB** | 20.17 MiB |
| `py_first_graph_run` (import + build/run a 2-node pregel graph) | 1683.85 ms | 1695.77 ms | 235.7 MiB | 235.9 MiB |

RSS is what GNU `time` reports in KB (`1 MiB = 1024 KB`); the raw KB samples are listed below.

Reading the floor rows:

- `py_import` minus `py_noop` ≈ 35 ms / 6.3 MiB is what importing the umbrella package costs above starting the interpreter.
- `rust_bench_floor` is a lower bound for "a Rust process that has SDK code reachable": the linker keeps only what `floor` references (a 1.95 MB binary), so this is not "the cost of loading the whole SDK" — Rust has no import step and links at build time.
- `rust_cli_version` is the fully linked binary but initialises nothing from the SDK, so it is a process-start floor, not an "SDK ready" number.
- `rust_hello_agent` is the closest Rust analogue of a real workload: process start plus an agent turn through the runtime with a scripted model. The Python side has no equivalent scripted-model path in this SDK, so no ratio is claimed.
- `py_first_graph_run` is Python-only: `import openjiuwen` is lazy, and the first graph execution pulls in the runner/logging/provider chain (3 466 modules loaded at that point). It is a one-time cost per process and is not comparable to any Rust row.

### Pure-local operations

Round 1 of the interleaved sequence; the three-round spread is in [Repeatability](#repeatability).

| operation | side | per operation | throughput | batch median | batch p95 |
| --- | --- | --- | --- | --- | --- |
| prompt render (85 B output) | Rust | **0.356 µs** | ≈ 2.81 M renders/s | 7.119 ms / 20 000 | 8.016 ms |
| prompt render (85 B output) | Python | **16.37 µs** | ≈ 61.1 k renders/s | 32.748 ms / 2 000 | 33.241 ms |
| graph run (2 nodes, no I/O) | Rust | **7.23 µs** | ≈ 138 k runs/s | 14.451 ms / 2 000 | 14.537 ms |
| graph run (2 nodes, no I/O) | Python | **112.3 µs** | ≈ 8.9 k runs/s | 22.468 ms / 200 | 22.692 ms |

Ratios for these two same-semantics workloads: prompt render **≈ 46×** (45.3–46.8× across the three rounds), graph run **≈ 15×** (15.0–15.6×).

### Repeatability

Batch medians per round, same host, interleaved A/B:

| workload | side | round 1 | round 2 | round 3 | per-operation spread |
| --- | --- | --- | --- | --- | --- |
| prompt | Rust | 7.119 ms | 7.109 ms | 7.198 ms | 0.3554–0.3599 µs |
| prompt | Python | 32.748 ms | 33.271 ms | 32.575 ms | 16.29–16.64 µs |
| graph | Rust | 14.451 ms | 14.577 ms | 14.494 ms | 7.225–7.289 µs |
| graph | Python | 22.468 ms | 21.903 ms | 22.342 ms | 109.5–112.3 µs |

Process-floor medians move more between runs than the in-process batches (earlier floor series during heavier host load gave `py_import` 57.6 ms and `py_noop` 15.5 ms); treat single-digit-millisecond differences in the floor table as noise and read the median, not a single rep.

### Raw values

Process floors — wall ms, 20 samples each, in run order:

```
rust_bench_floor : 1.22 1.16 1.20 1.11 1.27 0.94 0.95 1.15 1.28 1.26 0.93 1.42 1.08 1.25 1.29 1.26 1.24 0.99 1.11 1.14
rust_cli_version : 2.34 2.20 2.26 1.83 2.45 2.56 2.76 3.11 1.75 2.81 2.82 2.82 2.61 2.95 2.43 2.26 2.39 2.54 2.20 1.61
rust_hello_agent : 2.71 2.59 3.02 2.17 2.90 2.87 3.10 2.89 2.32 2.35 2.82 2.85 3.29 2.28 2.38 2.06 1.97 3.28 3.14 3.13
py_noop          : 22.92 18.62 19.96 22.85 23.16 31.15 25.36 10.54 10.48 10.67 16.77 16.58 11.09 10.22 10.50 16.08 10.34 16.75 10.41 10.85
py_import        : 46.13 54.40 34.67 56.65 56.51 34.40 34.98 54.49 45.66 44.17 35.03 52.44 58.27 51.92 34.15 51.60 35.75 35.37 56.91 59.55
py_first_graph   : 1683.85 1682.17 1690.12 1671.72 1695.77
```

Process floors — peak RSS KB, 20 samples each (5 for `py_first_graph`):

```
rust_bench_floor : 3672 3688 3716 3664 3716 3596 3656 3540 3652 3660 3716 3768 3544 3652 3776 3636 3768 3720 3756 3768
rust_cli_version : 8768 8592 9940 9584 9924 8916 8736 9428 9180 8996 9476 9032 9468 8908 8872 9004 8860 8956 9028 9116
rust_hello_agent : 8660 8736 8584 9068 8784 8884 8600 8672 8588 8584 8652 8868 8596 8756 8728 8808 8804 8768 8860 8972
py_noop          : 14164 13876 13992 13944 13908 14160 14004 13972 14028 14092 14032 13744 13920 13812 13816 14052 13960 13808 14132 14008
py_import        : 20684 20652 20404 20448 20460 20212 20468 20436 20612 20100 20380 20300 20276 20328 20612 20648 20376 20212 20368 20532
py_first_graph   : 241212 241384 241452 241592 241132
```

In-process workloads — per-rep batch ms, 30 reps, warmups 1–3 included in the table but dropped from the statistics:

Rust prompt (20 000 renders/rep):

```
1: 10.459  2: 10.507  3: 10.418  4: 8.016   5: 7.250   6: 8.611
7: 7.100   8: 7.072   9: 7.091  10: 7.161  11: 7.073  12: 7.119
13: 7.130  14: 7.109  15: 7.109  16: 7.111  17: 7.122  18: 7.103
19: 7.252  20: 7.143  21: 7.128  22: 7.135  23: 7.126  24: 7.107
25: 7.149  26: 7.144  27: 7.105  28: 7.108  29: 7.106  30: 7.098
```

Python prompt (2 000 renders/rep):

```
1: 42.138  2: 33.602  3: 33.277  4: 32.637  5: 33.241  6: 32.706
7: 32.666  8: 33.204  9: 32.495 10: 32.715 11: 32.416 12: 32.799
13: 32.901 14: 32.708 15: 32.813 16: 32.912 17: 32.495 18: 32.484
19: 32.558 20: 32.449 21: 32.533 22: 32.808 23: 33.541 24: 32.785
25: 32.672 26: 32.899 27: 32.952 28: 32.748 29: 32.930 30: 33.046
```

Rust graph (2 000 runs/rep):

```
1: 18.141  2: 14.629  3: 14.686  4: 14.519  5: 14.486  6: 14.454
7: 14.368  8: 14.496  9: 14.494 10: 14.432 11: 14.475 12: 14.384
13: 14.440 14: 14.537 15: 14.454 16: 14.425 17: 14.416 18: 14.412
19: 14.451 20: 14.502 21: 14.441 22: 14.424 23: 14.623 24: 14.479
25: 14.337 26: 14.408 27: 14.459 28: 14.466 29: 14.367 30: 14.414
```

Python graph (200 runs/rep):

```
1: 21.979  2: 22.020  3: 22.353  4: 22.332  5: 22.252  6: 22.464
7: 22.468  8: 22.422  9: 22.402 10: 22.385 11: 22.524 12: 22.611
13: 22.575 14: 22.692 15: 22.470 16: 22.409 17: 22.415 18: 22.267
19: 22.601 20: 22.406 21: 22.543 22: 22.814 23: 22.590 24: 22.634
25: 22.510 26: 22.377 27: 22.491 28: 22.550 29: 22.297 30: 22.252
```

## Limitations and what this does NOT measure

- **Warm page cache only.** The machine runs as uid 1000; dropping caches needs root and was not attempted. Every wall time and RSS figure is a warm-cache figure. "Cold start" is not measured; the closest proxy is the first repetition in a series, which is warm in the page cache but cold in the process's own caches.
- **No LLM, no network, no provider latency.** Every workload is pure-local on purpose. None of these numbers say anything about end-to-end agent latency, where the model dominates.
- **No equivalence claim for the process floors.** `py_import` measures "start the interpreter and import the umbrella package". `rust_bench_floor` measures "start a small binary with some SDK code linked". `rust_hello_agent` measures "start + one scripted-model turn". These are different amounts of work; the table labels them instead of dividing them.
- **The Python side was not measured for an agent turn.** The reference SDK has no built-in scripted model, so there is no Python row for `01_hello_agent`. Implementing a fake model to match the Rust scripted model would compare harness code, not framework overhead.
- **The graph workload compares logical shape, not engines.** Both sides run `entry → a → b → exit` with two state-updating nodes, no I/O. Internally they are different engines (Python pregel loop with channels/routers; Rust superstep executor), and Python also executes its `START`/`END` no-op nodes. Read the ~15× as "framework overhead per small graph run", not "the engines are 15× apart in all workloads".
- **Batch sizes differ.** Rust batches are 10× larger (20 000 / 2 000 vs 2 000 / 200) because its per-operation cost is ~20× smaller; per-operation figures are the batch median divided by the batch size. A per-operation difference this large is not sensitive to batch size in this range, but it is a difference in method.
- **Shared host.** Load average was 2.7–3.2 (1-min) on 8 CPUs during the reported rounds, with other builds running. Medians are stable across three interleaved rounds (see [Repeatability](#repeatability)); p95 values for the short Rust batches occasionally show a scheduler hiccup (8.0–10.4 ms against a 7.1 ms median), which is host noise, not workload variance.
- **Python dependency versions are not the lock file's.** `uv pip install` resolved current PyPI versions (transformers 5.23.0, pandas 3.1.1, numpy 2.4.6 …). Import cost and the 3 466-module first-graph chain depend on those versions; a locked install would likely move `py_import` and `py_first_graph_run`, and would not move the prompt/graph per-operation figures much.
- **Not measured:** throughput under concurrency (tokio multi-thread scaling, Python asyncio parallelism), streaming frame latency, checkpoint/SQLite persistence, retrieval index build/query cost, tokenizer accuracy or speed, memory growth over long sessions, build times, binary size differences beyond the three listed, and behaviour under memory pressure.
- **RSS is peak, not steady state.** `Maximum resident set size` is the high-water mark of a short process; it is not a memory-leak or long-run footprint measurement.
- **One machine, one architecture.** x86_64 Linux only; no ARM, macOS or Windows numbers.

## Reproduction

```
# Rust (uses the already-built release artifacts)
cargo build --release -j 4 -p openjiuwen-cli --bins
cargo build --release -j 4 -p openjiuwen-examples --example 01_hello_agent
CARGO_TARGET_DIR=$TMPDIR/bench-target cargo build --release --offline -j 4 \
    --manifest-path docs/bench/rust/Cargo.toml

B=$TMPDIR/bench-target/release/openjiuwen-bench
$B prompt --iterations 20000 --out docs/bench/results/rust/prompt_run1.json
$B graph  --iterations 2000  --out docs/bench/results/rust/graph_run1.json
$B floor

# Python (venv already installed as described above)
$VENV/bin/python docs/bench/python/bench.py prompt
$VENV/bin/python docs/bench/python/bench.py graph

# Process floors, 20 samples each
for i in $(seq 20); do
  $VENV/bin/python docs/bench/python/measure_proc.py -- target/release/openjiuwen --version
done

# Recompute every table above from the recorded JSON
$VENV/bin/python docs/bench/aggregate.py
```

`docs/bench/results/` holds the recorded samples: `process/*.json` (floor rows), `rust/{prompt,graph}_run{1,2,3}.json`, `python/{prompt,graph}_run{1,2,3}.json`.
