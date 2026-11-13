# openjiuwen-rsi tests
## Intent
Offline, deterministic integration suites for `openjiuwen-rsi`.
They drive the public API only: the orchestrator through a scripted fake `HarnessRunner`, the evolution protocol through synthetic candidates and reports, the PUCT tree through hand-computed scores, and the evaluator through scripted execution backends and `FakeCommandRunner`.
No test touches the network, a real model, git, Docker, subprocesses, or the clock-dependent parts of `Budget` beyond injected limits.
## API Surface
One file per concern:
- `config.rs` — `AutoHarnessConfig` defaults, YAML/JSON parsing, `validate()` rejections, derived paths, `load_or_default` bootstrap.
- `orchestrator.rs` — iteration lifecycle, improvement/acceptance, convergence/target/budget stops, `stop()`, event ordering, `run()` summary.
- `checkpoint.rs` — `SessionState` round-trip through `InMemoryCheckpointStore`, `KvCheckpointStore` (in-memory and file KV backends), and `resume()` continuing from persisted state.
- `evolution.rs` — `EvaluationReport` aggregation, `ElitistSelection` determinism, `DeterministicMutator` output, duplicate-candidate handling.
- `program_opt.rs` — PUCT scores, rank-score edges, ancestor visits, candidate limit, `best()` ties, config validation, serde round-trip.
- `evaluator_scoring.rs` — case/dataset parsing, judge verdict parsing, weighted/penalty score math, failure classification, requirement rows, transient markers, metrics, and the `CaseRunner`/`CandidateCaseRunner`/`CommandCaseBackend` bridge.
- `evaluator_benchmarks.rs` — SWE-bench and terminal-bench argv, report classification, failure literals, results parsing, `evaluate_case` over `FakeCommandRunner`, and both `Evaluator` adapters.
- `deferred.rs` — every stub returns `ErrorKind::Unsupported`; `EvolutionPolicy` round-trips.
## Constraints
- Fakes live in the test crate (`ScriptedRunner`, `ScriptedBackend`, `JsonEcho`, helper constructors); the library gains no test-only public API.
- Assertions never depend on wall-clock durations; `duration_ms` is asserted only as "present".
- Cross-crate fixtures come from `openjiuwen-test-support` and `tempfile` (dev-dependencies only).
- The benchmark suites never run a harness: `FakeCommandRunner` returns the scripted exit code and the test writes the report/results file the real harness would have written.
- Every test file maps behaviors to the Python reference where one exists (`tests/unit_tests/rsi/…`) in a comment.
## Routing Table
- `tests/config.rs` → configuration parsing and validation
- `tests/orchestrator.rs` → optimisation session lifecycle and stop conditions
- `tests/checkpoint.rs` → state persistence, KV/session stores, resume
- `tests/evolution.rs` → candidate/evaluation/selection protocol
- `tests/program_opt.rs` → PUCT tree search and program-opt configuration
- `tests/evaluator_scoring.rs` → evaluation records, judge scoring contract, case runner bridge, metrics
- `tests/evaluator_benchmarks.rs` → SWE-bench and terminal-bench runtimes and adapters
- `tests/deferred.rs` → deferred stub error surfaces
