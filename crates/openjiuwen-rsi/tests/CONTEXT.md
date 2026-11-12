# openjiuwen-rsi tests

## Intent

Offline, deterministic integration suites for `openjiuwen-rsi`.
They drive the public API only: the orchestrator through a scripted fake `HarnessRunner`, the evolution protocol through synthetic candidates and reports, and the PUCT tree through hand-computed scores.
No test touches the network, a real model, git, subprocesses, or the clock-dependent parts of `Budget` beyond injected limits.

## API Surface

One file per concern:
- `config.rs` — `AutoHarnessConfig` defaults, YAML/JSON parsing, `validate()` rejections, derived paths, `load_or_default` bootstrap.
- `orchestrator.rs` — iteration lifecycle, improvement/acceptance, convergence/target/budget stops, `stop()`, event ordering, `run()` summary.
- `checkpoint.rs` — `SessionState` round-trip through `InMemoryCheckpointStore`, `KvCheckpointStore` (in-memory and file KV backends), and `resume()` continuing from persisted state.
- `evolution.rs` — `EvaluationReport` aggregation, `ElitistSelection` determinism, `DeterministicMutator` output, duplicate-candidate handling.
- `program_opt.rs` — PUCT scores, rank-score edges, ancestor visits, candidate limit, `best()` ties, config validation, serde round-trip.
- `deferred.rs` — every stub returns `ErrorKind::Unsupported`; `EvolutionPolicy` round-trips.

## Constraints

- Fakes live in the test crate (`ScriptedRunner`, helper constructors); the library gains no test-only public API.
- Assertions never depend on wall-clock durations; `duration_ms` is asserted only as "present".
- Cross-crate fixtures come from `openjiuwen-test-support` and `tempfile` (dev-dependencies only).
- Every test file maps behaviors to the Python reference where one exists (`tests/unit_tests/rsi/…`) in a comment.

## Routing Table

- `tests/config.rs` → configuration parsing and validation
- `tests/orchestrator.rs` → optimisation session lifecycle and stop conditions
- `tests/checkpoint.rs` → state persistence, KV/session stores, resume
- `tests/evolution.rs` → candidate/evaluation/selection protocol
- `tests/program_opt.rs` → PUCT tree search and program-opt configuration
- `tests/deferred.rs` → deferred stub error surfaces
