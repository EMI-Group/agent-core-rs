# openjiuwen-agent-evolving

## Intent

Port the portable core of Python `openjiuwen.agent_evolving` (313 files, ~70k LOC) to idiomatic Rust.
The whole offline surface is ported: datasets, deterministic evaluation, the LLM-as-judge, the optimizer family, checkpointing and the trainer, evolution rails, the RL service contract, RL configuration/batching/rollout/policy/training, and the evolution subsystems (signals, rewards, attribution, trajectories, experience, sharing, TTSE) plus the Docker+git evaluator pipeline.
What stays deferred is only what genuinely needs a Python ML stack (`torch`, `transformers`, `verl`, `ray`), a distributed service (Redis, vLLM), a vendor HTTP service with authentication, a container browser/PIL runtime, or a remote hub.
Deferred paths keep their Python signatures and return `openjiuwen_common::error::Error::Unsupported` with a reason; nothing degrades silently.
Everything except those paths is offline and deterministic: no model calls, no environment reads, and no wall-clock dependence in tests.

## Implementation Status

| Area | Status |
| --- | --- |
| `dataset` — cases, validated records, JSON/JSONL loading, seeded shuffle/split | Implemented |
| `evaluator` — metric/runner/evaluator seams, exact match, rubric judge with verdict parsing, reports, `evaluate_cases` | Implemented |
| `optimizer` — candidate seam, grid baseline, textual-gradient research optimizers, evolve checkpoints | Implemented |
| `trainer` — bounded train/evaluate loop, resume, budgets, callbacks | Implemented |
| `rails` — pure evolution events and the harness rail adapter | Implemented |
| `service` — axum router: health, offline evaluate, 501 RL contract | Implemented |
| `rl` — config, packing, normalisation, policy/rollout seams, checkpoints, trainer loop | Implemented (GPU/online engines deferred) |
| `signal` / `reward` / `attribution` / `trajectory` / `experience` / `sharing` / `ttse` | Implemented (each module documents its remaining seams) |
| `evaluator_pipeline` — Docker + git harness over an injectable command runner | Implemented (agent adapters and evolution loop not ported) |
| `deferred` — machine-readable deferred table | Implemented |

## API Surface

### Datasets and evaluation
- `dataset` — `Case`, `EvaluatedCase`, `CaseLoader`, `CaseFormat`, `parse_cases_json`, `parse_cases_jsonl`, `parse_cases_str`, `load_cases_from_path`, `shuffle_cases`, `split_cases`.
- `evaluator` — `Evaluator`, `CaseRunner`, `AgentCaseRunner`, `MapCaseRunner`, `EvaluationConfig`, `EvaluationReport`, `evaluate_cases`, `Metric`, `MetricValue`, `ExactMatchMetric`, `LlmAsJudgeMetric`, `MetricEvaluator`, `Aggregate`, `DefaultEvaluator`.
- Judge — `JudgeCriterion`, `JudgeRubric`, `JudgeVerdict`, `build_judge_messages`, `LlmJudge`, `LlmJudgeEvaluator`.

### Optimizers and training
- `optimizer` — `Optimizer`, `Candidate`, `ParameterSpace`, `GridSearchOptimizer`, `BestSoFar`, `select_best`, `LlmInvokePolicy`.
- Research — `TextualParameter`, `OperatorState`, `OperatorKind`, `ProposalRecord`, `TextualStrategy`, `TextualGradientOptimizer`, `InstructionOptimizer`, `ToolOptimizer`, `MemoryOptimizer`, `SkillExperienceOptimizer`, `MetisOptimizer`, `TtseOptimizer`, `parse_tagged`.
- Checkpoints — `EvolveCheckpoint`, `CheckpointStore`, `FileCheckpointStore`, `MemoryCheckpointStore`, `CheckpointManager`, `DefaultCheckpointManager`, `CheckpointSnapshot`, `RestoredState`.
- `trainer` — `Trainer`, `TrainerConfig`, `TrainableTarget`, `Progress`, `Callbacks`, `NoopCallbacks`, `EpochReport`, `StopReason`, `TrainerReport`.

### Evolution subsystems
- `signal` — `EvolutionSignal`, `EvolutionCategory`, `EvolutionTarget`, `SignalTracker`, `TrackedSignal`, the `signals_from_*` extractors.
- `reward` — `RewardDimension`, `RewardSignal`, `RewardAggregator`, `RewardSummary`, `RewardPack`, `ScoreNormalisation`, `MissingDimensionPolicy`, `aggregate_scores`, `score_evaluated_cases`, `default_pack`.
- `attribution` — `AttributionGraph`, `AttributionNode`, `AttributionEdge`, `CreditAssigner`, `CreditReport`, `CreditRank`, `DiagnosticSignal`, `SignalSource`, `KeywordAttributor`, `KeywordRule`, `KeywordMatch`, `RewardStageAdapter`, `reward_tier`.
- `trajectory` — `Trajectory`, `TrajectoryStep`, `StepKind`, `StepStatus`, `TrajectoryOutcome`, `TrajectoryBuilder`, `from_messages`, `TrajectoryStore`, `TrajectorySpan`, `spans`, `windows`, `window_around`, `render_digest`, `first_error`, `last_n`, `TrajectoryDigest`.
- `experience` — `ExperienceRecord`, `ExperienceSection`, `ExperienceArchive`, `ExperienceQuery`, `ExperienceMatch`, `ExperienceExtractor`, `ExperienceLifecycle`, `LifecycleConfig`, `LifecycleReport`, `AddOutcome`.
- `sharing` — `SharedExperience`, `SharedSkillBundle`, `SharingMeta`, `SkillPackageMeta`, `SkillSearchResult`, `QueryKeywords`, `KeywordExtractor`, `ShareStager`, `StagedBundle`, `SharingBackend`, `LocalFileSharingBackend`, `MemorySharingBackend`.
- `ttse` — `TtseConfig`, `TipBank`, `TipBankStats`, `TipRecord`, `TipCategory`, `tip_id_for`, `TipClassifier`, `ClassificationRule`, `SuccessDetector`, `SuccessAssessment`, `TipParser`, `TipInducer`, `Dreamer`, `DreamReport`, `TtseEngine`, `consult`, `render_tips`, `ConsultResult`.

### Harness integration and service
- `rails` — `EvolutionEvent`, `EvolutionObserver`, `EvolutionRailSpec`, `HarnessEvolutionRail`, `registerable_rail`, `EVOLUTION_PRIORITY`.
- `service` — `RlServiceConfig`, `build_router`, `serve`, `ApiError`, `EvaluateRequest`, `EvaluateResponse`.
- `rl` — `RlConfig`, `RlTrainingConfig`, `RlRolloutConfig`, `RlRuntimeConfig`, `RlPersistenceConfig`, `RlTask`, `Rollout`, `RolloutMessage`, `RolloutWithReward`, `RewardRegistry`, `RewardFn`, `RewardNormalisation`, `RewardNormaliser`, `RolloutLog`, `BatchPacker`, `RlBatch`, `RlSample`, `token_estimate`, `Policy`, `RolloutEngine`, `DeterministicPolicy`, `FakeRolloutEngine`, `RlUpdate`, `RlTrainer`, `RlTrainerConfig`, `RlEpochReport`, `RlStopReason`, `RlTrainReport`, `RlCheckpoint`, `OfflineRlOptimizer`, `OnlineRlOptimizer`.
- `evaluator_pipeline` — `CommandRunner`, `ProcessCommandRunner`, `FakeCommandRunner`, `CommandSpec`, `ExecResult`, `DockerEnvironment`, `container_name`, `WorkspacePreparer`, `GitWorkspaceSpec`, `PreparedWorkspace`, `ScoreSpec`, `ScoredRun`, `TestCounts`, `parse_score`, `parse_last_json`, `parse_test_counts`, `EvaluationTask`, `TaskReport`, `EvaluationPipeline`, `PipelineConfig`, `PipelineSummary`.
- `deferred` — `DeferredFeature`, `DEFERRED_FEATURES`, `deferred_features`, `unsupported`.
- Crate root re-exports the primary type of every module plus `CRATE_NAME`/`CRATE_VERSION`.

## Constraints

- Depends only on `openjiuwen-common`, `openjiuwen-core` (for `foundation::llm::Model`, `foundation::tool::ToolInfo`, `foundation::prompt::BaseMessage`, `single_agent::{Agent, AgentOutput}`) and `openjiuwen-harness`; `openjiuwen-harness` must never depend on this crate.
- No new crates: no crypto, no `rand`, no embedding provider, no Docker/git client library; hashing is local FNV-1a, shuffling is a seeded xorshift64*, and external processes go through `tokio::process` behind a seam.
- Errors: `openjiuwen_common::error::{Error, Result}` only; no panics on user input; `Error::Unsupported` for every deferred path; `Error::validation` for malformed inputs.
- Async: tokio; traits use `async_trait`; cancellation is `tokio_util::sync::CancellationToken`; guards are never held across `.await`.
- Serde: every public data type is `Debug + Clone + PartialEq + Serialize + Deserialize` with `snake_case` Python field names and unknown-field tolerance; `Case`/`EvaluatedCase` also derive `schemars::JsonSchema`.
- Determinism: no `SystemTime`/`Instant` in report, checkpoint or record contents; the trainer's optional `time_budget_ms` is the only clock use and defaults to `None`; sequence counters and bank ticks replace wall-clock time everywhere else.
- No `unsafe`; no stdout logging (`tracing` only where an operation genuinely needs diagnostics, i.e. service serving).
- Files stay under ~1000 lines; `src/trainer.rs` is the accepted exception only because its inline unit-test module (from line 707) is larger than its 706-line implementation.
- Tests may use `unwrap`/`expect` via the crate-level `cfg(test)` allowance.

## Design Decisions

- `Case`/`EvaluatedCase` are defined here rather than imported from `openjiuwen_dev_tools::tune`, because Python has two independent models and the `agent_evolving` one adds `per_metric` and an optional answer map.
- Evaluation splits prediction production (`CaseRunner`) from scoring (`Evaluator`), matching Python's `predict` + `evaluate` seam and keeping scoring unit-testable.
- The judge is three separated layers — rubric data, a pure prompt/verdict-parse pair, and the model-calling `LlmJudge` — so prompts and parsing are tested offline and only the model call is non-deterministic.
- Judge fallbacks are explicit: `LlmJudgeEvaluator` and `DefaultEvaluator` name the failure in the reason and the fallback metric in `per_metric` instead of degrading silently.
- `evaluate_cases` owns a clone of the selected cases (bounded by `max_cases`) so its future is `Send`; the axum service and `tokio::spawn` callers use it directly.
- The trainer applies candidates through `TrainableTarget::apply` instead of Python's duck-typed `get_state()`/`set_parameter` chain; snapshots are stored under the target id inside `EvolveCheckpoint.operators_state`.
- Checkpoint files are written atomically (temp + rename) without changing Python's JSON shape.
- Every research optimizer shares one `TextualGradientOptimizer<S>` engine and per-domain strategies, so backward/propose/observe behave identically for instructions, tools, memory, skills, Metis sections and TTSE tips.
- Deterministic identifiers replace UUIDs/clock stamps: report `run_id` is caller-supplied, sharing `bundle_id` is `bundle-{16 hex}` of an FNV-1a hash over the skill id and sorted record ids, diagnostic signal ids are `diag-{16 hex}`, TTSE tip ids are the FNV-1a hash of `"{category}:{normalized rule}"`, and `experience`/`signal` recency uses monotonic sequence counters.
- The RL loop is seam-driven (`Policy` + `RolloutEngine`); `RolloutEngine::roll` deliberately takes only `(task, policy, cancel)` — the per-task sample count belongs to the engine implementation (`FakeRolloutEngine::with_sample_count`), not to the seam.
- The evaluator pipeline routes every side effect through `CommandRunner`, so it is testable offline; when `docker` is absent the failure is `Error::Unsupported` naming the binary instead of a silent no-op.
- The axum service declares the RL routes so clients get a stable `501` envelope instead of a `404`, and registers `POST /internal/v1/rl/tasks/{*operation}` as one catch-all because axum 0.8 cannot combine a path parameter with a literal `:finish`/`:abort` suffix in one segment.
- `HarnessEvolutionRail` lives here (upper layer) and implements the harness trait, so the dependency arrow stays `agent-evolving -> harness` and the harness keeps seam-only knowledge.
- Module-level (`//!`) docs resolve intra-doc links in the parent module's scope, so links written there use explicit `crate::…` paths; item-level docs may use plain relative links.
- `src/trainer.rs` keeps its inline unit tests in the same file rather than in `tests/`, which is why it is the crate's only file over the line guideline.

## Deferred Scope

| Deferred feature | Reason | Behaviour |
|---|---|---|
| RL trainer (PPO/verl/ray) | Python-only distributed ML stack (torch, transformers, verl, ray) | `OfflineRlOptimizer::train`/`start_training` return `Error::Unsupported`; `train_with` drives the portable `RlTrainer` |
| Online RL service state machine | Needs Redis task queues, vLLM/verl executors and gateway integration | `OnlineRlOptimizer::start_training`/`train_on_batch`/`export_lora` return `Error::Unsupported`; the HTTP routes answer `501` |
| agent_rl proxy/judge/verification gateway | Vendor HTTP services and authentication | not ported; no local seam |
| RL YAML configuration ecosystem | Python training-stack configuration (verl/ray configs) | reduced `RlConfig` records only |
| Vendor reward checks (browser/PIL/evidence agent) | Container browser sessions, image analysis and the evidence agent model flow | not ported; the deterministic reward records/aggregation are |
| Remote sharing hub client | Remote hub transport and authentication; local-file and memory backends are ported | not ported; `SharingBackend` seam plus two offline backends |
| LLM experience draft pipeline | Live model drafting and the online orchestrator; deterministic extraction is ported | not ported; `ExperienceExtractor` covers the offline half |
| TTSE embedding retrieval and LLM dreaming | Embedding provider, vector index and model merge passes; deterministic bank/consult are ported | not ported; Jaccard similarity and deterministic clustering used instead |
| Full Python agent_evolving unit suite | Representative offline tests only | not ported wholesale |

`src/deferred.rs` carries these nine entries as `DEFERRED_FEATURES`; `deferred::unsupported(area)` builds the corresponding error.

## Verification

- `cargo fmt -p openjiuwen-agent-evolving --check`
- `cargo test -p openjiuwen-agent-evolving --offline`
- `cargo clippy -p openjiuwen-agent-evolving --all-targets -- -D warnings`
- `cargo doc -p openjiuwen-agent-evolving --no-deps` stays warning-free.
- Surface at this revision: 309 inline unit tests, 17 integration binaries with 405 tests, and 23 doctests (737 total), all offline.

## Dependencies

- `openjiuwen-common` — errors, ids (`uuid_hex`), logging, utilities.
- `openjiuwen-core` — `foundation::{llm, prompt, tool}` types and `single_agent::{Agent, AgentOutput}`.
- `openjiuwen-harness` — `EvolutionRail` seam, `HarnessRail`, `HarnessRailContext` and `evolution_adapter`.
- `axum` (service router), `tower` (dev-only, `ServiceExt::oneshot` in tests).
- Dev-only: `openjiuwen-test-support`, `tempfile`.

## Routing Table

### Source modules
- `./src/dataset/` → `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading (`./src/dataset/CONTEXT.md`)
- `./src/evaluator/` → evaluator/metric/runner seams, judge rubric+verdict, reports (`./src/evaluator/CONTEXT.md`)
- `./src/optimizer/` → candidates, grid baseline, checkpoints and stores (`./src/optimizer/CONTEXT.md`); `./src/optimizer/research/` → textual-gradient engine and domain optimizers (`./src/optimizer/research/CONTEXT.md`)
- `./src/signal/` → signal records, extractors, tracker (`./src/signal/CONTEXT.md`)
- `./src/reward/` → reward records, aggregation, packs (`./src/reward/CONTEXT.md`)
- `./src/attribution/` → attribution graph, credit assignment, keyword rules, reward stage (`./src/attribution/CONTEXT.md`)
- `./src/trajectory/` → trajectory model, builder, spans, windows, store (`./src/trajectory/CONTEXT.md`)
- `./src/experience/` → experience records, archive, extractor, lifecycle (`./src/experience/CONTEXT.md`)
- `./src/sharing/` → sharing wire types, keyword extraction, stager, backends (`./src/sharing/CONTEXT.md`)
- `./src/ttse/` → tip bank, induction, dreaming, consultation, engine (`./src/ttse/CONTEXT.md`)
- `./src/evaluator_pipeline/` → Docker + git evaluation harness (`./src/evaluator_pipeline/CONTEXT.md`)
- `./src/trainer.rs` → bounded train/evaluate loop, progress, callbacks, reports
- `./src/rl.rs` → RL config, packing, normalisation, policy/rollout seams, checkpoints, trainer loop, optimizer stubs
- `./src/rails.rs` → pure evolution events, observer trait, harness rail adapter
- `./src/service.rs` → axum RL service (health/evaluate, 501 RL endpoints, error envelope)
- `./src/deferred.rs` → deferred-feature table and `Error::Unsupported` helper
- `./src/lib.rs` → crate documentation, module map and the curated re-export surface

### Tests
- `./tests/` → 17 offline integration binaries (`./tests/CONTEXT.md`)

### Siblings (read-only — escalate writes to the parent agent)
- `../openjiuwen-harness/src/rails/evolution/` → the consumed `EvolutionRail` seam and `evolution_adapter`
- `../openjiuwen-dev-tools/src/tune/` → the independent `dev_tools.tune` `Case` model
