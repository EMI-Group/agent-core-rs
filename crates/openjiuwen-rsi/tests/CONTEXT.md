# openjiuwen-rsi tests
## Intent
Offline, deterministic integration suites for `openjiuwen-rsi`.
They drive the public API only: the orchestrator through a scripted fake `HarnessRunner`, the evolution protocol through synthetic candidates and reports, the PUCT tree through hand-computed scores, the evaluator through scripted execution backends, and the runtime through `FakeCommandRunner`.
No test touches the network, a real model, git, Docker, subprocesses, or the clock-dependent parts of `Budget` beyond injected limits.
## API Surface
One file per concern:
- `config.rs` — `AutoHarnessConfig` defaults, YAML/JSON parsing, `validate()` rejections, derived paths, `load_or_default` bootstrap.
- `orchestrator.rs` — iteration lifecycle, improvement/acceptance, convergence/target/budget stops, `stop()`, event ordering, `run()` summary.
- `checkpoint.rs` — `SessionState` round-trip through `InMemoryCheckpointStore`, `KvCheckpointStore` (in-memory and file KV backends), and `resume()` continuing from persisted state.
- `evolution.rs` — `EvaluationReport` aggregation, `ElitistSelection` determinism, `DeterministicMutator` output, duplicate-candidate handling.
- `program_opt.rs` — PUCT scores, rank-score edges, ancestor visits, candidate limit, `best()` ties, config validation, serde round-trip.
- `single_harness_promotion.rs` — batch filter and target-improvement boundaries, every promotion reason literal, the full-evaluation outcomes, fingerprint and state YAML round-trip, idempotency keys, `resume_from` mismatches.
- `single_harness_loops.rs` — both promotion loops over a scripted evaluator: promotion, rejection, stops, resume without replay, byte-identical journals, legacy override, per-batch promotion, trial cap.
- `evaluator_scoring.rs` — case/dataset parsing, judge verdict parsing, weighted/penalty score math, failure classification, requirement rows, transient markers, metrics, and the `CaseRunner`/`CandidateCaseRunner`/`CommandCaseBackend` bridge.
- `evaluator_benchmarks.rs` — SWE-bench and terminal-bench argv, report classification, failure literals, results parsing, `evaluate_case` over `FakeCommandRunner`, and both `Evaluator` adapters.
- `paper_opt.rs` — rubric aggregation, dispersion, the greedy adoption gate, the deterministic proposer, and the offline orchestration loop.
- `member_optimizer.rs` — member selection scores, action typing and dependency waves, the planner/executor/verifier seams, and the end-to-end offline pass.
- `ledger.rs` — usage records, totals and per-model/per-stage metrics, `InMemoryLedgerStore`/`FileLedgerStore` append-and-replay, duplicate `call_id` suppression, pending-call delivery checks.
- `progress.rs` — `OptimizationProgressEvent` serde round-trip and `emit_progress` callback delivery.
- `runtime_command.rs` — `CommandSpec` rendering, `ProcessCommandRunner` exit/timeout/cancellation behaviour, and `FakeCommandRunner` matching.
- `runtime_git.rs` — branch, status, diff, worktree, push and cherry-pick parsing through `FakeCommandRunner`, plus credential scrubbing.
- `runtime_ci.rs` — gate config parsing, command normalisation, changed-line scoping, output sanitising, and gate outcomes.
- `runtime_python.rs` — virtualenv/pip/pytest command construction and the pytest JSON/JUnit parsers.
- `deferred.rs` — the three remaining stubs (`EvolutionEngine`, `LlmJudgeEvaluator`, `RuntimeExtensionLoader`) return `ErrorKind::Unsupported`; `EvolutionPolicy` round-trips.
- `crate_surface.rs` — every implemented flat re-export resolves and works offline: command/git/CI/Python/subprocess layers, ledger, evaluator and benchmark runtimes, optimisers, single-harness types, and the core protocol.
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
- `tests/single_harness_promotion.rs` → promotion math, reason literals, resumable state
- `tests/single_harness_loops.rs` → iterative and continuous promotion loops
- `tests/evaluator_scoring.rs` → evaluation records, judge scoring contract, case runner bridge, metrics
- `tests/evaluator_benchmarks.rs` → SWE-bench and terminal-bench runtimes and adapters
- `tests/paper_opt.rs` → paper rubric scoring, tree, proposal, orchestration
- `tests/member_optimizer.rs` → member selection, action planning, verification, optimiser pass
- `tests/ledger.rs` → run ledger records, metrics, stores, replay
- `tests/progress.rs` → optimisation progress events
- `tests/runtime_command.rs` → command specs, process and fake runners
- `tests/runtime_git.rs` → git operations over the command seam
- `tests/runtime_ci.rs` → CI gate configuration and outcomes
- `tests/runtime_python.rs` → Python toolchain and pytest report parsing
- `tests/deferred.rs` → the three remaining deferred stub surfaces
- `tests/crate_surface.rs` → implemented flat re-exports of the crate root
