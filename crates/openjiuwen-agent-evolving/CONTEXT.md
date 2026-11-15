# openjiuwen-agent-evolving

## Intent

Port the portable core of Python `openjiuwen.agent_evolving` (313 files, ~70k LOC) to idiomatic Rust.
The offline surface is ported end to end: datasets, deterministic evaluation, the LLM-as-judge, the optimizer family, checkpointing and the trainer, evolution rails, the online RL service (task/capture/turn/training-run state machines, HTTP routes, memory and Redis stores, local training executor, LoRA artifact repository), the RL service contract, RL configuration/batching/rollout/policy/training, the evolution subsystems (signals, rewards, attribution, trajectories, experience, sharing, TTSE) and the Docker+git evaluator pipeline.
What stays deferred is only what genuinely needs a Python ML stack (`torch`, `transformers`, `verl`, `ray`) or a GPU, a vendor HTTP gateway with authentication, a container browser/PIL runtime, or the Python training-stack config ecosystem.
Deferred paths keep their Python signatures and return `openjiuwen_common::error::Error::Unsupported` with a reason; nothing degrades silently.
Everything except those paths is offline and deterministic: no GPU, no model calls in tests, no environment reads, and no wall-clock dependence in report, checkpoint or record contents.

## Implementation Status

| Area | Status |
| --- | --- |
| `dataset` — cases, validated records, JSON/JSONL loading, seeded shuffle/split | Implemented |
| `evaluator` — metric/runner/evaluator seams, exact match, rubric judge with verdict parsing, reports, `evaluate_cases` | Implemented |
| `optimizer` — candidate seam, grid baseline, textual-gradient research optimizers, evolve checkpoints | Implemented |
| `trainer` — bounded train/evaluate loop, resume, budgets, callbacks | Implemented |
| `rails` — pure evolution events and the harness rail adapter | Implemented |
| `service` — axum router: health, offline evaluate, the full online RL route table | Implemented (a GPU-backed training run reports `Unsupported`) |
| `online` — task/capture/turn/training-run state machines, sample ledger, executor seam, LoRA repository, HTTP routes, memory + Redis stores | Implemented (executors are local/deterministic; the GPU PPO executor is deferred) |
| `rl` — config, packing, normalisation, policy/rollout seams, checkpoints, trainer loop | Implemented (GPU/online engines deferred) |
| `signal` / `reward` / `attribution` / `trajectory` / `experience` / `sharing` / `ttse` | Implemented (each module documents its remaining seams) |
| `experience` drafting — draft schema/normalizers, model-driven drafter with deterministic fallback, online evolution orchestrator | Implemented |
| `sharing` transport — local-file, in-memory and remote HTTP hub backends behind one seam | Implemented (the hub transport is new design) |
| `ttse` retrieval/dreaming — cosine embedding index with persistence, RRF-fused retrieval, model-backed merge/purge passes | Implemented |
| `reward` checks — read/python checks, evidence agent, model-judged scoring | Implemented (browser/PIL checks deferred) |
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

### Online RL service (`online`, `service`)
- Tasks — `TaskRegistry`, `TaskSpec`, `TaskRecord`, `TaskStatus`, `RewardMode`, `FinishReason`, `Clock`, `FixedClock`.
- Captures and turns — `CaptureMeta`, `CaptureStatus`, `PublishState`, `TurnTransition`.
- Samples — `SampleLedger`, `SampleRecord`, `SampleStatus`.
- Training runs — `TrainingRunner`, `TrainingRunnerConfig`, `TrainingRunRecord`, `RunStatus`, `RunStage`, `TrainingCheckpoint`.
- Executors — `TrainingExecutor`, `TrainingRequest`, `TrainingArtifact`, `LocalTrainingExecutor`, `PpoTrainingExecutor`.
- Artifacts — `LoraRepository`, `LoraVersion`, `LoraAvailability`, `LoraActivator`, `LoraActivation`, `NoopLoraActivator`, `UnsupportedLoraActivator`.
- Stores — `OnlineStore` (get/set/set_if_absent/del/set_add/set_remove/set_members/set_contains), `MemoryOnlineStore`, `RedisOnlineStore`, `RedisOnlineStoreOptions`.
- Errors and state — `OnlineError` (`code()`, `http_status()`), `RlServiceState`, `build_router`, `build_router_with_state`, `serve`, `RlServiceConfig`, `EvaluateRequest`, `EvaluateResponse`, `ApiError`.

### Evolution subsystems
- `signal` — `EvolutionSignal`, `EvolutionCategory`, `EvolutionTarget`, `SignalTracker`, `TrackedSignal`, the `signals_from_*` extractors.
- `reward` — `RewardDimension`, `RewardSignal`, `RewardAggregator`, `RewardSummary`, `RewardPack`, `ScoreNormalisation`, `MissingDimensionPolicy`, `aggregate_scores`, `score_evaluated_cases`, `default_pack`.
- Reward checks — `CheckRunner`, `RewardCheck`, `CheckTool`, `CheckOutcome`, `EvidenceAgent`, `EvidenceTask`, `EvidenceCriterion`, `CriterionKind`, `EvidenceGathered`, `ModelRewardJudge`, `CheckRewardReport`, `CriterionReport`, `CriterionStatus`, `score_criterion`, `to_reward_signal`.
- `attribution` — `AttributionGraph`, `AttributionNode`, `AttributionEdge`, `CreditAssigner`, `CreditReport`, `CreditRank`, `DiagnosticSignal`, `SignalSource`, `KeywordAttributor`, `KeywordRule`, `KeywordMatch`, `RewardStageAdapter`, `reward_tier`.
- `trajectory` — `Trajectory`, `TrajectoryStep`, `StepKind`, `StepStatus`, `TrajectoryOutcome`, `TrajectoryBuilder`, `from_messages`, `TrajectoryStore`, `TrajectorySpan`, `spans`, `windows`, `window_around`, `render_digest`, `first_error`, `last_n`, `TrajectoryDigest`.
- `experience` — `ExperienceRecord`, `ExperienceSection`, `ExperienceArchive`, `ExperienceQuery`, `ExperienceMatch`, `ExperienceExtractor`, `ExperienceLifecycle`, `LifecycleConfig`, `LifecycleReport`, `AddOutcome`.
- Drafting — `EvolutionSubject`, `EvolutionSubjectKind`, `EvolutionTarget`, `DraftSection`, `DraftExperience`, `EvolveDraft`, `SimplifyAction`, `SimplifyActionKind`, `SimplifyDraft`, the `normalize_*` normalizers, `ExperienceDrafter`, `DraftInput`, `OnlineEvolutionOrchestrator`, `EvolutionRequest`, `EvolutionStatus`, `OnlineEvolutionResult`, `initial_confidence`, the draft/simplify prompt constants.
- `sharing` — `SharedExperience`, `SharedSkillBundle`, `SharingMeta`, `SkillPackageMeta`, `SkillSearchResult`, `QueryKeywords`, `KeywordExtractor`, `ShareStager`, `StagedBundle`, `SharingBackend`, `LocalFileSharingBackend`, `MemorySharingBackend`, `RemoteHubOptions`, `RemoteHubSharingBackend`, `HUB_URL_ENV`, `HUB_API_KEY_ENV`.
- `ttse` — `TtseConfig`, `TipBank`, `TipBankStats`, `TipRecord`, `TipCategory`, `tip_id_for`, `TipClassifier`, `ClassificationRule`, `SuccessDetector`, `SuccessAssessment`, `TipParser`, `TipInducer`, `Dreamer`, `DreamReport`, `TtseEngine`, `consult`, `consult_async`, `rank`, `render_tips`, `ConsultResult`, `RetrieveMode`, `RankedTip`, `EmbeddingIndex`, `IndexMatch`, `cosine`, `RRF_K`, the merge/purge verdict types and parsers.

### Harness integration and service
- `rails` — `EvolutionEvent`, `EvolutionObserver`, `EvolutionRailSpec`, `HarnessEvolutionRail`, `registerable_rail`, `EVOLUTION_PRIORITY`.
- `rl` — `RlConfig`, `RlTrainingConfig`, `RlRolloutConfig`, `RlRuntimeConfig`, `RlPersistenceConfig`, `RlTask`, `Rollout`, `RolloutMessage`, `RolloutWithReward`, `RewardRegistry`, `RewardFn`, `RewardNormalisation`, `RewardNormaliser`, `RolloutLog`, `BatchPacker`, `RlBatch`, `RlSample`, `token_estimate`, `Policy`, `RolloutEngine`, `DeterministicPolicy`, `FakeRolloutEngine`, `RlUpdate`, `RlTrainer`, `RlTrainerConfig`, `RlEpochReport`, `RlStopReason`, `RlTrainReport`, `RlCheckpoint`, `OfflineRlOptimizer`, `OnlineRlOptimizer`.
- `evaluator_pipeline` — `CommandRunner`, `ProcessCommandRunner`, `FakeCommandRunner`, `CommandSpec`, `ExecResult`, `DockerEnvironment`, `container_name`, `WorkspacePreparer`, `GitWorkspaceSpec`, `PreparedWorkspace`, `ScoreSpec`, `ScoredRun`, `TestCounts`, `parse_score`, `parse_last_json`, `parse_test_counts`, `EvaluationTask`, `TaskReport`, `EvaluationPipeline`, `PipelineConfig`, `PipelineSummary`.
- `deferred` — `DeferredFeature`, `DEFERRED_FEATURES`, `deferred_features`, `unsupported`.
- Crate root re-exports the primary type of most modules plus `CRATE_NAME`/`CRATE_VERSION`; `online` and `deferred` stay path-qualified, and `service::build_router_with_state` is not re-exported.

## Constraints

- Depends only on `openjiuwen-common`, `openjiuwen-core` (for `foundation::llm::Model`, `foundation::tool::ToolInfo`, `foundation::prompt::BaseMessage`, `retrieval::embedding::{Embedding, HashEmbedding}`, `retrieval::vector_store`, `single_agent::{Agent, AgentOutput}`), `openjiuwen-harness`, and the optional `openjiuwen-extensions` (`redis` feature only); `openjiuwen-harness` must never depend on this crate.
- No new crates.io dependencies: HTTP goes through `openjiuwen_common::http`, embeddings through `openjiuwen_core::retrieval::embedding`, shelling out through the `evaluator_pipeline::CommandRunner` seam, HTTP fakes through `openjiuwen-test-support`'s `wiremock` re-export.
- The `redis` feature forwards `openjiuwen-extensions/redis` (which pulls no extra crate) and gates `online::redis_store` plus its tests; the default build does not compile it.
- Errors: `openjiuwen_common::error::{Error, Result}` at module boundaries; `online::OnlineError` inside the state machine, carrying `code()`/`http_status()` for the HTTP layer and converting into `Error` (not_found / validation / workflow); `Error::Unsupported` for every deferred path; `Error::validation` for malformed inputs.
- Async: tokio; traits use `async_trait`; cancellation is `tokio_util::sync::CancellationToken`; guards are never held across `.await`.
- Serde: every public data type is `Debug + Clone + PartialEq + Serialize + Deserialize` with `snake_case` Python field names and unknown-field tolerance; model handles and the `Dreamer`'s/`EvidenceAgent`'s injected clients are the documented exceptions; `Case`/`EvaluatedCase` also derive `schemars::JsonSchema`.
- Determinism: no `SystemTime`/`Instant` in report, checkpoint or record contents; `online` takes an injected `Clock` (`FixedClock` by default) and uses monotonic sequence counters; the trainer's optional `time_budget_ms` is the only clock use and defaults to `None`.
- Online state layout is fixed and documented in `./src/online/CONTEXT.md`: task/capture/turn/training/sample keys live under `rl:v1:*`, and the Redis store prefixes them with `openjiuwen:rl` by default.
- The Redis store issues single commands (no Lua, no `WATCH`/`MULTI`); single-flight paths are each one `SET … NX`, so one writable service process per database is the documented operational model.
- No `unsafe`; no stdout logging (`tracing` only where an operation genuinely needs diagnostics, i.e. service serving).
- Files stay under ~1000 lines; `src/trainer.rs` (1419 lines, test module from line 707) and `src/reward/check.rs` (1142 lines, test module from line 716) both exceed it.
- `RlServiceState` always constructs a `MemoryOnlineStore`: `RlServiceConfig` has no store/Redis field, so the Redis backend (feature `redis`) is library-only — reachable by building `TaskRegistry`/`SampleLedger`/`TrainingRunner` directly, never through `build_router`/`serve`.
- No HTTP route answers `501`: `OnlineError::Unsupported` maps to `503` (`service::ApiError::from_online`), a `ppo` executor only fails `/health` with `503`, and `ApiError::from_error`'s `Error::Unsupported` → `501` mapping has no caller; `ApiError::unsupported` is the only way to obtain one, and `service.rs`'s error-contract list says so explicitly.
- The merged router registers 17 paths: `rl_router`'s 15 plus `/health` and `/v1/evaluate`; the catch-all `/internal/v1/rl/tasks/{*operation}` covers both `:finish` and `:abort`.
- Under `executor = "ppo"` only `/health` answers `503`; `POST /v1/rl/training/runs` still claims and answers `201` with a `failed` run whose `failure_reason` names `torch` (`tests/online_service.rs`, `health_reports_executor_readiness`).
- `POST /v1/evaluate` implements only `exact_match`; any other `metric` string (including a judge metric, although `evaluator::judge` exists) is `400 invalid_request`.
- `POST /v1/rl/training/runs` runs the local executor inline and answers with the terminal record (`201` new / `200` reused active run), not a queued record.

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
- The online service is a store-agnostic state machine (`OnlineStore` is string/set primitives only), so the in-memory and Redis backends share one implementation and the same tests run over both.
- Training is a seam (`TrainingExecutor`): `LocalTrainingExecutor` is the deterministic default so the whole claim→run→activate→checkpoint path runs offline, while `PpoTrainingExecutor` returns `Unsupported` naming exactly what is missing (`torch`, `transformers`, `verl`, `ray`, CUDA) and `/health` reports it as `503` rather than pretending to be ready.
- Training-run single-flight is `SET … NX` on `rl:v1:training_run:active` (the loser re-reads the winner's run) instead of Python's `WATCH`/`MULTI` transaction.
- LoRA activation is a seam too: `NoopLoraActivator` records the activation deterministically for local runs, and `UnsupportedLoraActivator` names the vendor AIGW gateway.
- The remote experience hub is an HTTP implementation of the existing `SharingBackend` seam with a documented protocol v1, bearer auth, and the same client-side normalisation (`get` 404 → `Ok(None)`, sorted `list`, `0.0`-score drop, `limit == 0` = unlimited) as the local backends; Python had no HTTP hub, so this transport is new design, not a port.
- Experience drafting and TTSE dreaming are fail-open by design: model errors, unparsable output and missing models fall back to deterministic derivation (extractor-based drafts; cosine/Jaccard clustering with the representative rule as canonical), so a model outage never yields an empty draft or a shed tip.
- Retrieval fuses lexical and embedding ranks with reciprocal rank fusion (`RRF_K = 60`, equal weights) and falls back to bank/dump order rather than returning an empty result while the bank holds tips.
- Reward checks route read/python side effects through `CommandRunner`, keep the rubric arithmetic pure (`0.5·floor + 0.5·challenge`, all-unknown → `None`, any floor `unmet` caps the criterion at 25), and treat the model as a seam; browser/PIL checks are typed `Unsupported` naming playwright/PIL.
- The axum service declares the whole RL route table so clients get stable envelopes instead of `404`s, and registers `POST /internal/v1/rl/tasks/{*operation}` as one catch-all because axum 0.8 cannot combine a path parameter with a literal `:finish`/`:abort` suffix in one segment.
- `HarnessEvolutionRail` lives here (upper layer) and implements the harness trait, so the dependency arrow stays `agent-evolving -> harness` and the harness keeps seam-only knowledge.
- Module-level (`//!`) docs resolve intra-doc links in the parent module's scope, so links written there use explicit `crate::…` paths; item-level docs use plain relative links and link private items as code spans.

## Deferred Scope

| Deferred feature | Reason | Behaviour |
|---|---|---|
| RL trainer (PPO/verl/ray) | Python-only distributed ML stack (torch, transformers, verl, ray) | `OfflineRlOptimizer::train`/`start_training` and `online::PpoTrainingExecutor` return `Error::Unsupported`; `train_with` drives the portable `RlTrainer`, `LocalTrainingExecutor` drives the online run path |
| agent_rl proxy/judge/verification gateway | Vendor HTTP gateway services and their authentication | not ported; `LoraActivator`/`TrainingExecutor` are the local seams, `UnsupportedLoraActivator` names the gateway |
| RL YAML configuration ecosystem | Python training-stack configuration files (verl/ray YAML) | reduced `RlConfig` records only |
| Vendor reward checks (browser/PIL) | Container browser runtime (playwright) and image decoding (PIL); the evidence-agent and model-judged checks are ported | `CheckTool::Browser` and picture reads return `Error::Unsupported` naming the runtime |
| Full Python agent_evolving unit suite | Representative offline tests only | not ported wholesale |

`src/deferred.rs` carries these five entries as `DEFERRED_FEATURES`; `deferred::unsupported(area)` builds the corresponding error.

## Verification

- `cargo fmt -p openjiuwen-agent-evolving --check`
- `cargo metadata --offline --no-deps`
- `cargo test -p openjiuwen-agent-evolving --offline`
- `cargo test -p openjiuwen-agent-evolving --offline --all-features`
- `cargo clippy -p openjiuwen-agent-evolving --all-targets --all-features --offline -- -D warnings`
- `cargo doc -p openjiuwen-agent-evolving --no-deps` and `--all-features` stay warning-free.
- Surface at this revision: 543 inline unit tests (560 with `--all-features`, which adds the Redis-store suite), 22 integration binaries holding 490 tests, and 23 doctests — 1056 tests by default and 1073 with all features, all offline.
- Not every unit test is an inline `mod tests`: `src/online/tests/` is a `#[cfg(test)]` subdirectory (its `redis` child is gated on `feature = "redis"` and drives the machines over `RedisOnlineStore` against the in-process fake `RESP2` server in `fake_redis.rs`), plus `src/reward/end_to_end.rs` and `src/experience/pipeline_tests.rs`.
- The 17 Redis-gated tests come from `src/online/tests/redis.rs` (10) and `src/online/redis_store.rs` (7).

## Dependencies

- `openjiuwen-common` — errors, ids (`uuid_hex`), logging, utilities, `http::HttpClient`.
- `openjiuwen-core` — `foundation::{llm, prompt, tool}` types, `retrieval::embedding`, `retrieval::vector_store`, `single_agent::{Agent, AgentOutput}`.
- `openjiuwen-harness` — `EvolutionRail` seam, `HarnessRail`, `HarnessRailContext` and `evolution_adapter`.
- `openjiuwen-extensions` (optional, `redis` feature) — the RESP client behind `online::RedisOnlineStore`.
- `axum` (service router), `tower` (dev-only, `ServiceExt::oneshot` in tests).
- Dev-only: `openjiuwen-test-support` (scripted models, `wiremock` re-export), `tempfile`.

## Routing Table

### Source modules
- `./src/dataset/` → `Case`, `EvaluatedCase`, `CaseLoader`, JSON/JSONL loading (`./src/dataset/CONTEXT.md`)
- `./src/evaluator/` → evaluator/metric/runner seams, judge rubric+verdict, reports (`./src/evaluator/CONTEXT.md`)
- `./src/optimizer/` → candidates, grid baseline, checkpoints and stores (`./src/optimizer/CONTEXT.md`); `./src/optimizer/research/` → textual-gradient engine and domain optimizers (`./src/optimizer/research/CONTEXT.md`)
- `./src/online/` → online RL service: task/capture/turn/training-run state machines, sample ledger, executor seam, LoRA repository, HTTP routes, memory and Redis stores (`./src/online/CONTEXT.md`)
- `./src/signal/` → signal records, extractors, tracker (`./src/signal/CONTEXT.md`)
- `./src/reward/` → reward records, aggregation, packs, checks/evidence agent/model-judged scoring (`./src/reward/CONTEXT.md`)
- `./src/attribution/` → attribution graph, credit assignment, keyword rules, reward stage (`./src/attribution/CONTEXT.md`)
- `./src/trajectory/` → trajectory model, builder, spans, windows, store (`./src/trajectory/CONTEXT.md`)
- `./src/experience/` → experience records, archive, extractor, lifecycle, draft schema/drafter/orchestrator (`./src/experience/CONTEXT.md`)
- `./src/sharing/` → sharing wire types, keyword extraction, stager, local/memory/remote-hub backends (`./src/sharing/CONTEXT.md`)
- `./src/ttse/` → tip bank, induction, dreaming (deterministic and model-backed), embedding index, retrieval, consultation, engine (`./src/ttse/CONTEXT.md`)
- `./src/evaluator_pipeline/` → Docker + git evaluation harness (`./src/evaluator_pipeline/CONTEXT.md`)
- `./src/trainer.rs` → bounded train/evaluate loop, progress, callbacks, reports
- `./src/rl.rs` → RL config, packing, normalisation, policy/rollout seams, checkpoints, trainer loop, optimizer stubs
- `./src/rails.rs` → pure evolution events, observer trait, harness rail adapter
- `./src/service.rs` → axum service: health, evaluate, and the online RL route table
- `./src/deferred.rs` → deferred-feature table and `Error::Unsupported` helper
- `./src/lib.rs` → crate documentation, module map and the curated re-export surface

### Tests
- `./tests/` → 22 offline integration binaries (`./tests/CONTEXT.md`)

### Siblings (read-only — escalate writes to the parent agent)
- `../openjiuwen-harness/src/rails/evolution/` → the consumed `EvolutionRail` seam and `evolution_adapter`
- `../openjiuwen-dev-tools/src/tune/` → the independent `dev_tools.tune` `Case` model
- `../openjiuwen-extensions/src/redis/` → the RESP client consumed by `online::RedisOnlineStore`
