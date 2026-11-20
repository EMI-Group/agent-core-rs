# Bench

## Intent

Standalone measurement harnesses that produce the numbers in `../BENCHMARKS.md`.
Nothing here is part of the shipped workspace: the Rust harness is its own cargo package with an empty `[workspace]` table, and the Python harness runs against an out-of-tree copy of the reference SDK.
Results are raw JSON so the appendix can be recomputed rather than copied.

## API Surface

- `rust/` — cargo package `openjiuwen-bench` (own `[workspace]`, path dependencies on the shipped crates) with subcommands `floor` (link the umbrella, print `VERSION`), `prompt` (`PromptTemplate::format` batches), `graph` (2-node linear graph runs); `--iterations N` sets the batch size, `--out FILE` records the run.
- `python/bench.py` — CPython harness for the same prompt-render and pregel-graph workloads (`prompt`, `graph`, `all`).
- `python/measure_proc.py` — one-child process probe printing `{"wall_s","maxrss_kb","exit"}`; wall from `perf_counter`, RSS from a second run under GNU `time -v`.
- `aggregate.py` — recomputes every table in `../BENCHMARKS.md` from `results/` (`--raw` adds the per-rep values).
- `results/process/` — 20-sample floor rows; `results/{rust,python}/{prompt,graph}_run{1,2,3}.json` — three interleaved rounds per side.

## Constraints

- Harness workloads must be semantically identical on both sides: same template text, same keywords, same logical graph shape.
- Batch sizes differ by side (Rust 20 000 renders / 2 000 graph runs, Python 2 000 / 200) because per-operation cost differs ~20×; per-operation figures are the batch median divided by the batch size.
- Never write into the Python reference repo at `/home/bill/Source/agent-core`; measure an out-of-tree copy or installed wheel, with `PYTHONDONTWRITEBYTECODE=1`.
- No LLM/network calls: every workload is pure-local so the numbers measure framework overhead, not provider latency.
- Keep the Rust package out of the root workspace (empty `[workspace]` table) so `cargo test --workspace` at the root is unaffected.

## Notes for Agents

- Never take peak RSS from `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` inside a Python driver: Linux keeps the pre-`exec` peak, so every child reports the driver's own ~13.7 MiB (a 2.1 MiB Rust hello-world "measured" 16.4 MiB). Use the GNU `time -v` path in `python/measure_proc.py`.
- GNU `time` is not on `PATH` on NixOS; the wrapper resolves `/run/current-system/sw/bin/time`, overridable with `BENCH_TIME_BIN`.
- Python's first graph run in a fresh process loads ~3 466 modules (241 MB peak); the harness runs one untimed warmup so the per-run figures measure steady-state cost.
- The host is shared, so compare medians across the three recorded rounds, not single reps.

## Routing Table

- `../BENCHMARKS.md` → environment, commands, medians/p95, raw tables, limitations
