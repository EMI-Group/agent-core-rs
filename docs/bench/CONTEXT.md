# Bench

## Intent

Standalone measurement harnesses that produce the numbers in `../BENCHMARKS.md`.
Nothing here is part of the shipped workspace: the Rust harness is its own cargo package with an empty `[workspace]` table, and the Python harness runs against an out-of-tree copy of the reference SDK.
Results are raw JSON so the appendix can be recomputed rather than copied.

## API Surface

- `rust/` — cargo package `openjiuwen-bench` (own `[workspace]`, path dependencies on workspace crates) exposing the `floor`, `prompt`, and `graph` subcommands.
- `python/bench.py` — CPython harness for the same prompt-render and pregel-graph workloads, writing per-repetition JSON to `results/python/<workload>.json`.
- `python/measure_proc.py` — single-child process wrapper printing `{"wall_s", "maxrss_kb", "exit"}` as one JSON line; used for the import-floor measurement of either language.
- `results/` — raw per-repetition output (`*.json`) from each side; `results/python/floor.json` holds the 20 import-floor samples; the appendix tables are derived from these files.

## Constraints

- Harness workloads must be semantically identical on both sides: same template text, same keywords, same graph shape, same iteration counts.
- Never write into the Python reference repo at `/home/bill/Source/agent-core`; measure an out-of-tree copy or installed wheel.
- No LLM/network calls: every workload is pure-local so the numbers measure framework overhead, not provider latency.
- Keep the Rust package out of the root workspace (empty `[workspace]` table) so `cargo test --workspace` at the root is unaffected.

## Routing Table

- `../BENCHMARKS.md` → environment, commands, medians/p95, raw tables, limitations
