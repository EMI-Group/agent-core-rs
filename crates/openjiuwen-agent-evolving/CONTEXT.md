# openjiuwen-agent-evolving

## Intent

Port the portable, non-ML core of Python `openjiuwen.agent_evolving` (313 files, ~70k LOC) to idiomatic Rust.
The Python area is marked deferred in `docs/PARITY.md` because the trainer/evaluator/optimizer packages and the online RL service depend on a Python ML stack.
This crate keeps the viable offline core — cases, evaluation/scoring, optimizer/checkpoint management, a bounded trainer loop, and evolution rail specs — and exposes API-shaped stubs for the ML/vendor/RL parts.
Everything except the stubs is offline and deterministic: no model calls, no environment reads, no wall-clock dependence in tests.

## API Surface

### `dataset` (`src/dataset/`)

- `Case { inputs: BTreeMap<String, Value>, label: BTreeMap<String, Value>, tools: Option<Vec<ToolInfo>>, case_id: String }` — Python field names; `new` validates non-empty inputs/label, `with_tools`, `with_case_id`, `validate`; default `case_id` is a fresh 32-lowercase-hex UUID.
- `EvaluatedCase { case, answer: Option<BTreeMap<String, Value>>, score: f64, reason: String, per_metric: Option<BTreeMap<String, f64>> }` — `new` rejects non-finite scores and clamps finite scores to `0.0..=1.0` (Python constructor clamp), accessors `inputs`/`label`/`tools`/`case_id`, `From<Case>`.
- `CaseLoader` — in-memory collection: `new`, `len`/`is_empty`/`size`, `cases`, `into_cases`, `get`, `iter`, `shuffle(seed)` (seeded xorshift64* Fisher-Yates), `split(ratio)` (shuffle-then-cut), serde through `{"cases": [...]}`.
- `shuffle_cases(cases, seed)` and `split_cases(cases, ratio)` — Python module-level helpers; `split_cases` cuts positionally at `floor(ratio * n)`.
- Loading: `CaseFormat { Json, JsonLines }`, `parse_cases_json`, `parse_cases_jsonl`, `parse_cases_str(content, format)`, `load_cases_from_path(path)` (async, `tokio::fs`, format inferred from `.jsonl`/`.ndjson`).
- JSON input accepts a bare array, `{"cases": [...]}`, or a single case object; JSONL skips blank lines and `#` comments; every parsed case is validated.

### `evaluator` (`src/evaluator/`)

- `Evaluator` (async trait): `name()`, `evaluate(case, prediction) -> EvaluatedCase`, default `batch_evaluate(cases, predictions, num_parallel)` with length-mismatch and `1..=20` parallel validation; parallel via `futures::stream::buffered`.
- `CaseRunner` (async trait): `run_case(case, cancel) -> Value` — the agent-execution seam; `name()`.
- `AgentCaseRunner` — wraps `Arc<dyn openjiuwen_agent::Agent>`; input is `Value::Object(case.inputs)`; prediction is the serialized `AgentOutput`.
- `MapCaseRunner` — deterministic `case_id -> Value` lookup for offline tests/baselines.
- `Metric` trait: `name()`, `higher_is_better()` (default `true`), `compute(prediction, label) -> MetricValue`; `MetricValue { Scalar(f64), Named(BTreeMap<String, f64>) }`.
- `ExactMatchMetric { normalize }` — compares the answer projection (label key `answer`, single-key maps, else the full JSON object); normalized comparison trims, lowercases, and collapses whitespace; non-strings fall back to JSON equality.
- `LlmAsJudgeMetric` — API-shaped stub; `compute` returns `Error::Unsupported` (needs a live model and is non-deterministic).
- `MetricEvaluator { metrics, aggregate: Aggregate }` — per-metric values merged into `per_metric`; `Aggregate { Mean, First }` over the ordered metric values; numeric coercion never raises (bad values become `0.0`).
- `DefaultEvaluator` — API-shaped stub for the Python model-judge evaluator; `evaluate` returns `Error::Unsupported`.
- `EvaluationReport { run_id, num_cases, mean_score, min_score, max_score, per_metric, cases: Vec<EvaluatedCase> }` with `from_evaluated`/`from_cases`; empty report has zero counts and scores.
- `evaluate_cases(run_id, runner, evaluator, cases, cancel, config) -> EvaluationReport` — runs cases with bounded parallelism, `EvaluationConfig { num_parallel, max_cases }`; the explicit `run_id` keeps reports deterministic (no uuid or clock).

### `optimizer` (`src/optimizer/`)

- `Candidate { id, parameters: BTreeMap<String, Value>, score: Option<f64>, generation }` — serde; `state()` projects `parameters` to a JSON object for state-restoring targets.
- `Optimizer` (async trait): `name()`, `propose(limit) -> Vec<Candidate>`, `observe(candidate, score)` (default no-op), `best() -> Option<&Candidate>` (default `None`).
- `ParameterSpace` — ordered `BTreeMap<String, Vec<Value>>`; `grid(limit)` enumerates the Cartesian product deterministically (last dimension fastest).
- `GridSearchOptimizer` — deterministic grid baseline implementing `Optimizer`; proposes successive chunks, tracks the best observed candidate, never calls a model.
- `BestSoFar` / `select_best` — best-so-far tracking (`score` maximisation) shared by the trainer and optimizer.
- `LlmInvokePolicy` — pure-data retry policy (`attempt_timeout_secs`, `total_budget_secs`, `max_attempts = 2`, `backoff_base_secs = 1.0`, `retry_empty_response`).
- Checkpointing: `EvolveCheckpoint { version, run_id, step{epoch,batch}, best{best_score}, seed, operators_state, updater_state, searcher_state, last_metrics }` with Python JSON keys.
- `CheckpointStore` (async trait): `save_checkpoint(checkpoint, filename) -> Option<String>`, `load_checkpoint(path) -> Option<EvolveCheckpoint>`, `load_state_dict(path) -> Option<Value>`.
- `FileCheckpointStore` — `base_dir: Option<PathBuf>`; `None` is a silent no-op; writes atomically (temp file + rename); default filename `latest.json`.
- `MemoryCheckpointStore` — deterministic in-process store for tests.
- `CheckpointManager` trait: `should_save(epoch, improved)`, `build_checkpoint(&CheckpointSnapshot)`, `restore(&EvolveCheckpoint) -> RestoredState`.
- `DefaultCheckpointManager { run_id, checkpoint_version = "v1", save_every_n_epochs = 1, save_on_improve = true }` — epoch `0` always saves; `RestoredState { start_epoch, best_score, run_id, operators_state }` skips unknown operators.

### `trainer` (`src/trainer.rs`)

- `TrainerConfig { num_iterations = 3, num_parallel = 1, early_stop_score = 1.0, candidates_per_epoch = 1, max_case_evaluations, time_budget_ms, checkpoint_filename = "latest.json", resume_from }`.
- `TrainableTarget` (async trait): `id()`, `run_case(case, cancel)`, `apply(candidate)`, `snapshot()` (default `Value::Null`), `restore(state)` (default no-op).
- `Trainer` — `new(config)`, `with_checkpoint(store, manager)`, `with_callbacks(vec)`, `with_resume_from(name)`; `evaluate(target, evaluator, cases, cancel) -> EvaluationReport`; `train(target, optimizer, evaluator, train_cases, val_cases, cancel) -> TrainerReport`.
- Loop semantics: optional resume, baseline validation, per-epoch `propose -> apply -> validate -> observe -> select best`, checkpoint save on improvement / every N epochs, early stop at `early_stop_score`, cancellation token checked at every await point, evaluation budget and wall-clock budget enforced before spending.
- Resume restores `best_score` and `operators_state` and re-runs the checkpointed epoch (`start_epoch .. num_iterations`); a missing checkpoint file is ignored.
- Best-so-far: the first candidate may tie the baseline, later candidates must strictly improve, so `best_score` never drops below the baseline and `best_candidate` is `None` when only the baseline scored best.
- `Progress` — Python fields; `run_epoch()` clamps at `max_epoch`.
- `Callbacks` trait with `on_train_begin`/`on_train_epoch_begin`/`on_train_epoch_end`/`on_train_end`; `NoopCallbacks`.
- `EpochReport { epoch, score, best_score, improved, candidates, evaluations, checkpoint_saved }`, `StopReason { Completed, EarlyStop, Cancelled, BudgetExhausted, NoCandidates }`, `TrainerReport { run_id, epochs_run, best_score, best_candidate, steps, evaluations, cancelled, stopped_reason }`.

### `rails` (`src/rails.rs`)

- `EvolutionEvent` — pure serde data: `InvokeStarted { agent_id, session_id, iteration }`, `TaskIterationCompleted { agent_id, session_id, round_id, iteration, is_follow_up }`.
- `EvolutionObserver` (sync trait): `observe(&EvolutionEvent) -> Result<()>`.
- `EvolutionRailSpec { name = "evolution", priority = 60, enabled = true }`.
- `HarnessEvolutionRail` — implements `openjiuwen_harness::rails::EvolutionRail` by translating harness context into pure events; `registerable_rail(...)` wraps it with `evolution_adapter` for registry mounting.
- `EVOLUTION_PRIORITY = 60` constant mirroring the harness rail priority.

### `service` (`src/service.rs`)

- `RlServiceConfig { model_id, listen_host = "127.0.0.1", listen_port = 18081, ready = true }`.
- `build_router(config) -> axum::Router`; `serve(config)` binds a `TcpListener` and serves until shutdown.
- Endpoints: `GET /health`, `POST /v1/evaluate` (offline `ExactMatchMetric` scoring; accepts `predictions` or an `answers` map keyed by `case_id`).
- RL/RLOps endpoints are registered but return HTTP `501` with the Python error envelope `{"error":{"code","message"}}` and an `Error::Unsupported` message naming the operation: `/v1/rl/tasks/*`, `/v1/rl/training/runs*`, `/v1/rl/trajectories*`, `/internal/v1/completions:*`, `/internal/v1/rl/tasks/*`.
- No Redis, no training stack, no model dependency; the router is pure and testable with `tower::ServiceExt::oneshot`.

### `rl` (`src/rl.rs`)

- API-shaped records: `RlConfig { training, rollout, runtime, persistence }`, `RlTrainingConfig`, `RlRolloutConfig`, `RlRuntimeConfig`, `RlPersistenceConfig`, `RlTask`, `Rollout`, `RolloutMessage`, `RolloutWithReward`, `RewardRegistry`.
- Stubs: `OfflineRlOptimizer`, `OnlineRlOptimizer` — constructors and JSON shapes work; training/rollout/export entry points return `Error::Unsupported` naming the missing ML/vendor stack.

### `deferred` (`src/deferred.rs`)

- `DeferredFeature { area, reason }`, `DEFERRED_FEATURES: &[DeferredFeature]`, `deferred_features()`, `unsupported(area) -> Error` — the machine-readable counterpart of the table below.

## Constraints

- Depends only on `openjiuwen-common`, `openjiuwen-core` (for `foundation::tool::ToolInfo` and `single_agent::{Agent, AgentOutput}`), and `openjiuwen-harness`; `openjiuwen-harness` must never depend on this crate.
- Errors: `openjiuwen_common::error::{Error, Result}` only; no panics on user input; `Error::Unsupported` for every deferred path.
- Async: tokio; traits use `async_trait`; cancellation is `tokio_util::sync::CancellationToken`; locks are never held across `.await`.
- Serde: every public data type is `Debug + Clone + PartialEq + Serialize + Deserialize` with `snake_case` Python field names and unknown-field tolerance; `Case`/`EvaluatedCase` also derive `schemars::JsonSchema`.
- No `unsafe`, no stdout logging; use `tracing` only where an operation genuinely needs diagnostics (service serving).
- Files stay under ~1000 lines; tests may use `unwrap`/`expect` via the crate-level `cfg(test)` allowance.
- `src/trainer.rs` carries its inline unit-test module and is the one accepted file above that guideline; its implementation proper stays under 1000 lines.
- Determinism: no `SystemTime`/`Instant` in report contents; a wall-clock budget is the only clock use, and defaults to `None`.

## Design Decisions

- `Case`/`EvaluatedCase` are defined here rather than imported from `openjiuwen_dev_tools::tune` because Python has two independent models and the `agent_evolving` one adds `per_metric` and an optional answer map; conversion is a JSON round-trip at the caller.
- Evaluation is split into `CaseRunner` (produce a prediction) and `Evaluator` (score it), matching Python's `predict` + `evaluate` seam and keeping scoring deterministic and unit-testable.
- `ExactMatchMetric` accepts the three label projections Python callers use in practice (keyed `answer`, single-key maps, full object) so dataset files and agent outputs need no adapter.
- The trainer applies candidates through `TrainableTarget::apply` instead of Python's duck-typed `operator.get_state()`/`set_parameter` chain; state snapshots are stored under the target id inside `EvolveCheckpoint.operators_state`.
- Checkpoint files are written atomically (temp + rename), an improvement over Python's direct `json.dump`, without changing the JSON shape.
- The service uses axum 0.8 and shared `AppState`; RL endpoints are declared so clients get a stable 501 contract instead of a 404 while the ML stack is deferred.
- `HarnessEvolutionRail` lives here (upper layer) and implements the harness trait, so the dependency arrow stays `agent-evolving -> harness` and the harness keeps its seam-only knowledge.
- `evaluate_cases` owns a clone of the selected cases (bounded by `max_cases`) so its future is `Send`; the axum service and `tokio::spawn` callers can use it directly.
- The service registers `POST /internal/v1/rl/tasks/{*operation}` as one catch-all because axum 0.8 cannot combine a path parameter and a literal `:finish`/`:abort` suffix in one segment.
- `AgentCaseRunner` serializes the full `AgentOutput` (`{"output", "result_type"}`); callers pairing it with `ExactMatchMetric` should project `prediction["output"]`, since the metric only auto-projects `answer`-keyed or single-key predictions.

## Deferred Scope

| Deferred feature | Reason | Behaviour |
|---|---|---|
| RL trainer (`agent_rl/rl_trainer`, PPO/verl/ray) | Python-only distributed RL stack (`torch`, `transformers`, `verl`, `ray`) | `OfflineRlOptimizer`/`OnlineRlOptimizer` return `Error::Unsupported` |
| Online RL service state machine (Redis tasks, capture pipeline, training runs) | Needs Redis, vLLM/verl executors, gateway integration | endpoints return 501 with the documented error envelope |
| LLM-as-judge / default model evaluator | Requires a live model and is non-deterministic | `LlmAsJudgeMetric`/`DefaultEvaluator` return `Error::Unsupported` |
| Optimizers: `InstructionOptimizer`, `ToolOptimizer`, `MemoryOptimizer`, `SkillExperienceOptimizer`, Metis/context-evolve, TTSE | Prompt/textual-gradient research code tied to Python LLM flows | not ported; `Optimizer` seam plus `GridSearchOptimizer` baseline provided |
| `evaluator_pipeline` (Docker + git harness) | External container protocol | documented in this table only |
| `reward/` (rewardpack, browser/PIL checks), `attribution/`, `experience/`, `sharing/`, `signal/`, `trajectory/`, `ttse/`, `prompts/`, `tools/` (evolution toolset) | Large, near-leaf subsystems; vendor/browser dependencies | not ported |
| `agent_rl` `proxy.py` (Flask), `judge_server.py`, `verification_gateway.py` | Vendor HTTP services and auth | not ported |
| RL config/training YAML ecosystem (`RLServiceConfig`, verl configs) | Python training stack configuration | reduced `RlServiceConfig` only |
| Full Python unit suite under `tests/unit_tests/agent_evolving` | Not ported wholesale | representative offline behavior tests in `tests/` |

## Dependencies

- `openjiuwen-common` — errors, ids (`uuid` helper), logging.
- `openjiuwen-core` — `foundation::tool::ToolInfo`, `single_agent::{Agent, AgentOutput}`.
- `openjiuwen-harness` — `EvolutionRail` seam and `evolution_adapter`.
- `axum`/`tower` — service router and endpoint smoke tests.
- Dev-only: `openjiuwen-test-support` (scripted model/fixtures), `tempfile`.

## Routing Table

- `./src/dataset/` → `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading (`./src/dataset/CONTEXT.md`)
- `./src/evaluator/` → evaluator/metric/runner seams, reports, offline scoring (`./src/evaluator/CONTEXT.md`)
- `./src/optimizer/` → candidates, optimizer baseline, checkpoints/stores/manager (`./src/optimizer/CONTEXT.md`)
- `./src/trainer.rs` → bounded train/evaluate loop, progress, callbacks, reports
- `./src/rails.rs` → pure evolution events, observer trait, harness rail adapter
- `./src/service.rs` → axum RL service skeleton (health/evaluate, 501 RL endpoints)
- `./src/rl.rs` → RL records and API-shaped ML stubs
- `./src/deferred.rs` → deferred-feature table and `Error::Unsupported` helper
- `./tests/` → offline integration suites (`./tests/CONTEXT.md`)
- `../openjiuwen-harness/src/rails/evolution.rs` → the consumed `EvolutionRail` seam (sibling — read-only, escalate writes to parent)
- `../openjiuwen-dev-tools/src/tune/` → the independent `dev_tools.tune` `Case` model (sibling — read-only, escalate writes to parent)

## Test Strategy

- Unit tests live beside each module (`#[cfg(test)]`); integration tests live in `tests/` and are fully offline and deterministic.
- `tests/dataset.rs` — JSON/JSONL parsing (array, `{"cases":...}`, single object, blank/comment lines), validation failures, seeded shuffle/split semantics, serde round-trip.
- `tests/evaluator.rs` — exact-match normalization, metric aggregation (`mean`/`first`, named metrics), batch mismatch/parallel bounds, report aggregation, fake `CaseRunner` evaluation, `AgentCaseRunner` with `ScriptedModel`/fake agent.
- `tests/checkpoint.rs` — file store round-trip/disabled/custom filename, memory store, `should_save` combinations, build/restore, resume from `latest.json`.
- `tests/trainer.rs` — baseline → epochs, early stop, improvement saves checkpoints, resume continues at the stored epoch, cancellation, budgets, no-candidate stop, callback order, fake `TrainableTarget`.
- `tests/service.rs` — `tower::ServiceExt::oneshot` smoke tests for `/health`, `/v1/evaluate` success/validation, and 501 error envelope on an RL endpoint.
- `tests/rails.rs` — event translation from `HarnessRailContext`, spec defaults, observer error propagation, `registerable_rail` wrapping.
- Definition of done: `cargo fmt -p openjiuwen-agent-evolving`, `cargo test -p openjiuwen-agent-evolving`, `cargo clippy -p openjiuwen-agent-evolving --all-targets -- -D warnings` all clean.
