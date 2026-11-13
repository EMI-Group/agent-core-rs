# openjiuwen-rsi

## Intent

Recursive self-improvement (RSI) for the openJiuwen runtime, ported from the Python package `openjiuwen.rsi` (v0.1.20, ~97k lines / 313 files) with the ML/vendor half explicitly deferred.
The crate owns exactly four things:
1. `harness_rsi::auto_harness` — a bounded, resumable self-optimisation session loop: propose candidate configurations, evaluate each one through a harness run, keep the elite, checkpoint every iteration, and stop on convergence, budget, or cancellation.
2. `harness_rsi::single_harness` — the deterministic evaluate→gate→promote core of Python's single-harness engine: exact promotion math, resumable epoch state with idempotency keys, and the iterative (epoch-checkpointed) and continuous (two-stage, attempt-based) loops.
3. `artifact_rsi::program_opt` — the portable PUCT tree search that ranks candidate programs by rank score plus visit-count exploration, as pure data structures and algorithms.
4. `evolution` — the crate-wide candidate/mutation/evaluation/selection protocol shared by all halves, so a fake harness or a different search strategy plugs in without touching the orchestrator.

Everything that needs Python ML runtime, a vendor harness SDK, subprocess/git execution, or distributed evolution is an API-shaped stub returning `Error::Unsupported` — never a silent stub and never a panic.

The crate is layer 6 (`docs/ARCHITECTURE.md` §1): it may depend on `openjiuwen-core`, `openjiuwen-harness`, and `openjiuwen-common`, never upward, and never on `openjiuwen-agent-evolving` (that would create a cycle).

## API Surface

Module paths mirror Python: `openjiuwen_rsi::harness_rsi::auto_harness::*`, `openjiuwen_rsi::artifact_rsi::program_opt::*`.
Everything below is also re-exported flat from the crate root (`openjiuwen_rsi::AutoHarnessOrchestrator`, `openjiuwen_rsi::PuctTree`, …), because `openjiuwen-auto-harness` is a re-export shim over `openjiuwen_rsi::auto_harness`.

### `evolution` — shared protocol (`src/evolution/`)

```
candidate.rs  CandidateId(String) — new/as_str/Display/From<&str>/serde
              Candidate { id, parent: Option<CandidateId>, generation: u32, body: String,
                          change_summary: String, status: CandidateStatus,
                          evaluation: Option<EvaluationReport>, metadata: BTreeMap<String, Value> }
              CandidateStatus { Pending, Evaluated, Invalid, Selected } (serde snake_case)
              Candidate::new(id, body) → pending, generation 0, empty metadata
              Candidate::score() -> Option<f64>  // evaluation.report score, None when invalid
mutation.rs   MutationRequest { parents: Vec<Candidate>, generation: u32, count: usize,
                                metadata: BTreeMap<String, Value> }
              #[async_trait] trait MutationStrategy { fn name(&self) -> &str;
                  async fn propose(&self, request: MutationRequest) -> Result<Vec<Candidate>>; }
              DeterministicMutator { comment_prefix: String /* "# " */, variants_per_parent: usize /* 1 */ }
                  — no ML: appends `<prefix>rsi variant g<generation>-<n>` to the parent body,
                    sets change_summary, deterministic ids `<parent>/g<generation>/v<n>`.
evaluation.rs Case { id, name: Option<String>, weight: f64 /* 1.0 */, input: Value,
                     expectations: Value, metadata: BTreeMap<String, Value> }
              CaseResult { case_id, passed: bool, score: Option<f64> /* None = ungraded */,
                           weight: f64, error: Option<String>, details: Value }
              EvaluationReport { results: Vec<CaseResult>, total, passed, graded: usize,
                                 average_score: Option<f64>, weighted_score: Option<f64>,
                                 cost_usd: f64 }
                  from_results(Vec<CaseResult>) — totals; average/weighted over graded cases only
                  score() -> Option<f64>      // weighted_score, else average_score
                  is_valid() -> bool          // every result is graded and has no error
              #[async_trait] trait Evaluator { fn name(&self) -> &str;
                  async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport>; }
selection.rs  #[async_trait]-free trait SelectionStrategy { fn name(&self) -> &str;
                  fn select(&self, candidates: &[Candidate], count: usize) -> Vec<CandidateId>; }
              ElitistSelection { elite_size: usize /* 1 */ }
                  — deterministic: score desc, unscored last, ties by candidate id asc.
              fn rank_score(rank: usize, total: usize) -> f64   // total<=1 → 0.5, else rank/(total-1)
              fn ranked(candidates: &[Candidate]) -> Vec<&Candidate>  // ascending stable by score, None = -inf
```

### `harness_rsi::auto_harness` (`src/harness_rsi/auto_harness/`)

```
config.rs   AutoHarnessConfig — serde data only (no live models/clients):
              data_dir, local_repo, repo_url ("https://gitcode.com/openJiuwen/agent-core.git"),
              config_path, experience_dir, skills_dirs, language ("cn"),
              optimization_goal, pipeline_preference: PipelinePreference { Auto, Meta, Extended },
              session_budget_secs (900000.0), cost_limit_usd (10.0), task_timeout_secs (300000.0),
              model_timeout_secs (300000.0), max_tasks_per_session (10),
              max_iterations (10), target_score: Option<f64>, min_improvement (0.05), patience (2),
              agent_iterations: BTreeMap<String, u32> (Python defaults: implement 30, assess 30,
                plan 15, select_pipeline 10, eval 10, pr_draft 5, learnings 5, explore_subagent 20,
                browser_subagent 20, merge_ext 8), immutable_files, high_impact_prefixes (["openjiuwen/core/"]),
              git_base_branch ("develop"), fix_phase1_max_retries (10), fix_phase2_max_retries (9),
              config_bootstrapped (false), suggested_local_repo, extra: Value
            Default (Python defaults), from_json_value, from_yaml_str, load(path) (async tokio::fs),
            load_or_default(path) (bootstrap semantics: unreadable/invalid YAML → defaults, never an error),
            validate(), resolve_agent_iterations(&self) -> BTreeMap<String, u32>,
            resolved_experience_dir/worktrees_dir/runs_dir/cache_repo_dir() -> PathBuf (under data_dir)
budget.rs   BudgetLimits { wall_clock: Duration, cost_limit_usd: f64, task_timeout: Duration }
            Budget — start(elapsed clock), elapsed(), remaining() -> Duration, remaining_cost() -> f64,
              spent_usd(), record_cost(f64), status() -> BudgetStatus, should_stop() -> bool,
              allows_task(Option<Duration>) -> bool, exhausted() -> Option<StopReason>
              Python parity: stop when elapsed >= wall_clock OR spent >= cost_limit; one warn log per
              dimension at 80% (target `openjiuwen_rsi::auto_harness::budget`).
            BudgetStatus { elapsed_secs, remaining_secs, spent_usd, remaining_usd,
                           wall_clock_exhausted, cost_exhausted }  (not serde; process-local)
task.rs     TaskStatus { Pending, Running, Success, Failed, Timeout, Reverted }
            OptimizationTask { topic, description, files, issue_ref: Option<String>, expected_effect,
                               pipeline_name, status }  (Python field names)
            StageStatus { Success, Failed }; StageResult { status, artifacts: BTreeMap<String, Value>,
                               messages, metrics: BTreeMap<String, f64>, error: String }
            Gap { id, description, impact: f64, feasibility: f64 } with priority() = impact * feasibility
            IterationRecord { iteration, parents: Vec<CandidateId>, proposed, evaluated, valid: usize,
                              best_candidate: Option<CandidateId>, best_score: Option<f64>,
                              previous_best_score: Option<f64>, improvement: f64, accepted: bool,
                              duration_ms: u64, stop: Option<StopReason> }
            SessionSummary { session_id, iterations, best_candidate, best_score, evaluated, accepted,
                             stop_reason: Option<StopReason>, spent_usd }
            StopReason { MaxIterations, TargetScore, Converged, BudgetExhausted, Cancelled, Failed }
            StopConditions { max_iterations, target_score, min_improvement, patience }
              from_config(&AutoHarnessConfig), evaluate(&self, history: &[IterationRecord],
              iteration: u32, budget: &Budget) -> StopDecision
              — order: cancellation (caller-supplied) > budget > max iterations > target score > converged;
                converged = the last `patience` records each improved best_score by < min_improvement.
            StopDecision { stop: bool, reason: Option<StopReason>, detail: String }
event.rs    AutoHarnessEvent — #[serde(tag = "type", rename_all = "snake_case")]:
              SessionStarted { session_id, max_iterations }
              IterationStarted { iteration, parents: Vec<String> }
              CandidatesEvaluated { iteration, proposed, evaluated, valid, best_score }
              IterationFinished { record }
              CheckpointSaved { session_id, iteration }
              StageResult { stage: String, result: StageResult }
              SessionFinished { summary }            (serde rename "harness_session_finished")
              SessionCancelled { iteration }
runner.rs   HarnessRunRequest { session_id, iteration, candidate: Candidate, cases: Vec<Case>,
                                metadata: BTreeMap<String, Value> }
            HarnessRunOutcome { report: EvaluationReport, artifacts: BTreeMap<String, Value>,
                                messages: Vec<String> }
            #[async_trait] trait HarnessRunner { fn name(&self) -> &str;
                async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome>; }
              — the only seam to `openjiuwen-harness`; callers wire `DeepAgent` runs, tests fake it.
            HarnessEvaluator { runner: Arc<dyn HarnessRunner> } impl Evaluator — one run per candidate.
checkpoint.rs SessionState { session_id, iteration, running: bool, best_candidate, best_score,
                              candidates: Vec<Candidate>, history: Vec<IterationRecord>,
                              spent_usd, stop_reason }   (full serde round-trip = resume)
            #[async_trait] trait CheckpointStore { save(&SessionState) -> Result<()>,
              load(&str) -> Result<Option<SessionState>>, delete(&str) -> Result<()> }
              — backend/deserialization failures are `Error::session`, never "no checkpoint"
            InMemoryCheckpointStore (RwLock<BTreeMap>), KvCheckpointStore { store: Arc<dyn KvStore>,
              key_prefix: String /* "rsi:auto_harness:" */ } over
              `openjiuwen_core::foundation::store::kv::KvStore` (state stored as JSON under
              `<prefix><session_id>`), SessionCheckpointStore { session: Arc<dyn Session>,
              component_id /* "rsi_auto_harness" */ } over `update_component_state`.
orchestrator.rs AutoHarnessOrchestrator — `builder(config)` (a required runner **or** evaluator) with
              with_runner/with_mutator/with_selector/with_evaluator/with_checkpoint_store/with_cases/
              with_seed_body/with_session_id/with_mutation_count/build;
              builder defaults: `DeterministicMutator` (one variant per parent), `ElitistSelection`
              (elite of one), `InMemoryCheckpointStore`, empty cases, `DEFAULT_SEED_BODY`,
              fresh `uuid_v4` session id, one proposal per iteration; a runner-only build is wrapped
              in `SessionIterationEvaluator`, which stamps the session id and current iteration onto
              each `HarnessEvaluator` run (the `Evaluator` trait carries neither).
              config(), session_id(), state() -> SessionState (snapshot), is_running(), is_cancelled(),
              start() -> Result<()>, resume() -> Result<Option<SessionState>>,
              run_iteration() -> Result<IterationRecord>, stop(), finish() -> Result<SessionSummary>,
              stream(self: &Arc<Self>) -> BoxStream<'static, Result<AutoHarnessEvent>>,
              run(self: &Arc<Self>) -> Result<SessionSummary> (start → iterations → finish).
              One iteration: seed if empty → select parents → mutate → evaluate each candidate through
              the Evaluator → mark statuses → compare best to incumbent (improvement, accepted) →
              append IterationRecord → checkpoint session state → evaluate stop conditions.
              `improvement` is `best - previous_best` clamped at 0 (0.0 for the first best, which is
              accepted); the winner becomes `CandidateStatus::Selected`.
              `run_iteration` on a finished/stopped session returns Error::Cancelled; evaluation errors
              mark the candidate Invalid and are recorded, they never abort the session; a checkpoint
              write failure is returned as Error::Session. `finish()` is idempotent, emits no event, and
              defaults a missing stop reason to MaxIterations (cap reached) → last record → Failed.
              `run` and `stream` are alternative single-call entries with a terminal-event latch, so a
              session reports at most one terminal event.
```

### `artifact_rsi::program_opt` (`src/artifact_rsi/program_opt/`)

```
config.rs   ReplyFormat { Files, Tagged }; MAX_WORKERS = 8; DEFAULT_WORKERS = 1
            ProgramOptConfig { c_puct (1.0), prior_exponent (0.0), candidate_limit: Option<usize>,
                               max_expansions (16), max_depth (8), workers (1),
                               entrypoint ("candidate.py"), candidate_timeout_secs (60.0),
                               reply_format: ReplyFormat } — Default, validate(), to_tree()
tree.rs     Program { program_id, iteration, parent_id: Option<String>, code, change_summary,
                      metrics: BTreeMap<String, f64>, valid: bool, error: Option<String> }
            Node { index, parent_index: Option<usize>, program: Program, score: Option<f64>,
                   num_visits: u64, promise: Option<f64> }
            PuctTree { c_puct, prior_exponent, candidate_limit, nodes: Vec<Node>, next_iteration }
              new(&ProgramOptConfig), with_root(Program), from_nodes(..), nodes(), len(), is_empty(),
              root(), next_iteration(), candidate_limit_reached(),
              add_node(parent_index: Option<usize>, program: Program) -> Result<usize>
                — assigns index and iteration, rejects unknown parent and a reached candidate limit
                  with Error::Validation,
              select(&mut self) -> Option<usize> — pure PUCT selection, then +1 visit on the chosen
                node and every ancestor (virtual loss),
              puct_scores() -> Vec<f64>, priors() -> Vec<f64>, rank_scores() -> Vec<f64>,
              record_visits(index), best() -> Option<&Node> (highest score, smallest index on ties)
            PUCT formula (Python `artifact_rsi/program_opt/tree.py`, no numeric unit test there):
              rank_score_i = rank_i / (N-1)  (N<=1 → 0.5), rank over ascending stable score sort,
                None score → -inf;
              P_i = 1/N when prior_exponent <= 0, else promise^prior_exponent normalised over rated
                nodes with unrated nodes taking the mean prior;
              puct_i = rank_score_i + c_puct * P_i * sqrt(sum(visits)) / (1 + visits_i);
              first maximum wins (stable ties).
```

### `harness_rsi::single_harness` (`src/harness_rsi/single_harness/`)

```
promotion.rs  EPSILON = 1e-9; type CaseScores = BTreeMap<String, f64>
              case_scores(&EvaluationReport) -> CaseScores      // graded, error-free results only
              is_batch_improvement(before, after) -> bool        // after > before + EPSILON
              batch_filter(&CaseScores, min_improvement) -> Vec<String>
                  // ids whose satisfaction gap 1.0 - score > min_improvement + EPSILON, ascending
              threshold_saturated(&CaseScores) -> bool           // non-empty and every case >= 1.0 - EPSILON
              targets_improved(before, after) -> Vec<String>
                  // strict after > before + EPSILON; missing before counts as improved, missing after not
              mean_improvement(before, after) -> f64             // mean delta over the improved targets, 0.0 if none
              StageOneDecision { passed, improved, reason } + stage_one_gate(before, after, content_changed, engineering_failed)
                  // precedence: candidate_content_unchanged > "engineering_failed: <detail>" >
                  //   target_threshold_saturated > candidate_failed_target_replay_checkpoint >
                  //   candidate_passed_batch_gate_pending_epoch_checkpoint
              FullEvaluationDecision { promoted, reason, failed_targets, regressed_others, mean_improvement } +
              full_evaluation_decision(before, after, improved_targets, protected_cases, min_improvement, max_regression)
                  // first match wins: protected_case_regressed (loss > max_regression + EPSILON, missing after = total
                  //   loss) > candidate_failed_target_replay_checkpoint (empty targets, a target that lost its margin
                  //   or vanished) > selected_score_regressed (retention mean not > -max_regression, or overall mean
                  //   delta not > EPSILON) > selected_global_non_regression (promoted)
                  // min_improvement is diagnostic only (the batch gate already enforced it); regressed_others is
                  //   diagnostic; mean_improvement is the mean delta over the union of before/after ids
              Reason literals are pub consts: REASON_CONTENT_UNCHANGED, REASON_ENGINEERING_FAILED,
              REASON_THRESHOLD_SATURATED, REASON_TARGET_REPLAY_FAILED, REASON_BATCH_GATE_PASSED,
              REASON_SELECTED, REASON_SELECTED_SCORE_REGRESSED, REASON_PROTECTED_CASE_REGRESSED.
config.rs     SingleHarnessConfig { mode: PromotionPolicyMode /*ContinuousTwoStageV2*/, max_epochs 5,
                max_iterations 10, batch_size 5, min_improvement 0.05, max_regression 0.05, patience 2,
                target_score: Option<f64>, max_attempts_per_batch 2, max_trials 8, promote_on_best_found true,
                protected_cases, evaluation_protocol_signature "v1", extra } — Default, validate()
                (Error::config for zero batch_size/max_iterations, non-finite or out-of-range margins),
                fingerprint(&[String]) -> String (FNV-1a 64-bit hex over dataset ids + signature + mode token),
                protected_case_set()
state.rs      PromotionPolicyMode { LegacyEpochV1, ContinuousTwoStageV2 } (serde snake_case, token())
              SingleHarnessStatus { Running, Completed, Failed, Cancelled } (is_terminal())
              UsageState { calls, cost_usd }; CandidateGateRecord { batch_key, candidate_id, passed, reason, round }
              JournalEntry { epoch, batch_index, reason, score, promoted }
              SingleHarnessState { version 7, status, mode, fingerprint, dataset, best_harness_refs_path,
                best_score, best_body, baseline_score, completed_batches, candidate_gates, optimization_journal,
                historical_peak_score, continuous_attempts, continuous_trials, max_iteration, task_id,
                evaluation_protocol_signature, stop_reason: Option<StopReason>, usage, metadata }
                new(task_id, fingerprint, dataset, max_iteration, mode), is_batch_completed, mark_batch_completed,
                iterations(), consecutive_non_promotions(), resume_from(previous, fingerprint, task_id,
                max_iteration, signature) (Error::validation naming the mismatched field; resets status/stop),
                save(path)/load(path) async YAML (Error::session on IO/parse failure)
              batch_key(epoch, batch_index) = "epoch_{epoch:03}:batch_{batch_index:03}";
              attempt_key(batch_key, attempt) = "<batch_key>:attempt_{attempt:03}";
              state_path(output_dir) = <dir>/single_harness_state.yaml
iterative.rs  SessionCore (crate-internal: config, Arc<dyn Evaluator>, Arc<dyn MutationStrategy>, state,
                best_body, best_scores, Budget, CancellationToken, cases, seed_body, counters)
                ensure_baseline (seed over ALL cases; only what the state lacks is recomputed),
                select_batch(epoch) (batch_filter over dataset scores, epoch-offset slice, wraps),
                batch_cases, scores_for, evaluate (cost-accounted), propose, record_gate, record_outcome,
                apply_promotion, stop_reason (cancellation > budget > max_iterations > target > converged),
                run_attempt (batch gate -> full evaluation -> optional legacy override)
              IterativeSingleHarness { new(config, evaluator, improver, task_id, dataset, cases, seed_body),
                with_state, with_budget, state, best_body, is_cancelled, run, run_epoch(epoch) -> bool, stop }
              IterativeResult { state, best_score, best_body, promotions, evaluations }
continuous.rs ContinuousSingleHarness (same surface; always the continuous policy) + type ContinuousResult
              run_epoch: up to max_attempts_per_batch attempts per batch, at most one promotion per batch,
              max_trials cap across the run (persisted as continuous_trials), no legacy override, batch marked
              completed after the attempts; trials cap reports StopReason::MaxIterations.
```

Both loops: one journal entry per attempted batch, one gate record per evaluated candidate, per-case incumbent
scores persisted in `state.metadata["best_case_scores"]`, `promotions`/`evaluations` counted per process.
Candidate ids come from the mutation strategy with generation `iterations + 1` (iterative) / `continuous_trials + 1`
(continuous), so ids, ordering and tie-breaks are deterministic. Evaluation errors on a candidate become
`"engineering_failed: …"` journal entries and never abort the loop; a baseline error marks the run failed and is
returned.

### `deferred` (`src/deferred/`)

API-shaped stubs, all documented and all returning
`Err(Error::Unsupported("<surface> is not implemented in the Rust port: <reason>"))`:
`agentdescent::EvolutionEngine` (+ implemented `EvolutionPolicy { Full, Guarded, Reflective }` data),
`runtime::{SubprocessEvaluator, RuntimeExtensionLoader, CiGate, GitBackend}`,
`evaluators::{SweBenchEvaluator, TerminalBenchEvaluator, LlmJudgeEvaluator}`,
`paper_opt::PaperOptimizer`, `member_optimizer::MemberOptimizer`.

## Constraints

- Errors: `openjiuwen_common::error::{Error, Result}` only; Python category names preserved
  (`Error::config` for bad config, `Error::validation` for bad input/limits, `Error::unsupported`
  for deferred surfaces, `Error::cancelled` for a cancelled session, `Error::session` for
  checkpoint failures). No `unwrap`/`expect`/`panic` outside `#[cfg(test)]`.
- Async-first: tokio; `async_trait` for object-safe seams; public streams are
  `BoxStream<'static, Result<..>>` built with `async_stream::try_stream!`; locks are
  `std::sync::Mutex`/`RwLock` and never held across `.await`.
- Cancellation: `tokio_util::sync::CancellationToken`; `stop()` cancels it, in-flight and subsequent
  iterations observe it at await points.
- Serde: every public data type round-trips (`snake_case`, `#[serde(default)]`, unknown fields
  tolerated); Python field names preserved for records (`StageStatus`/`TaskStatus` values, config keys).
- Config: YAML via `serde_yaml_ng`; env loading happens in binaries only, never here.
- Logging: `tracing` only (budget warnings, skipped duplicate candidates, evaluation failures); no
  stdout/stderr in library code.
- Determinism: ids, ordering, tie-breaks, and mutation output are deterministic given the same inputs —
  no wall-clock in scores or ids; `duration_ms` is the only timing value and is never asserted.
- Size: files ≤1000 lines, one responsibility per file; shared protocol lives in `evolution/`.
- Clippy (`-D warnings`) with workspace lints: every public item documented, `#[must_use]` on
  getters, `# Errors` sections on every `Result` return; `#![cfg_attr(test, allow(clippy::unwrap_used,
  clippy::expect_used, clippy::panic))]` in `lib.rs`.

## Deferred Scope

| Python surface | Status | Reason |
|---|---|---|
| `artifact_rsi.program_opt.{puct_engine,puct_provider,runtime,execution,script_domain,probe,provision}` | stub (`deferred::agentdescent`, `deferred::runtime`) | Needs `agentdescent==0.4.6` evolve/ledger/aggregator internals, subprocess/pip execution, `Model` bridge. |
| agentdescent itself (`selection`, `aggregator`, `ledger`, `staleness`, `filetree`, `evolution`) | deferred | Exact 0.4.6 semantics; distributed/deep-RL training has no maintained Rust equivalent. |
| `artifact_rsi.paper_opt.auto_research` (paper optimisation, LaTeX/bibliography modules) | stub (`deferred::paper_opt`) | Six DeepAgent stage agents + a Python host toolchain (LaTeX, pyright, Pillow). |
| `harness_rsi.evaluator.{case_backend,case_runner,team_evaluator,swebench_runtime,terminal_bench_runtime,runtime_adapters}` and `evaluator.judger.*` | stub (`deferred::evaluators`) | SWE-bench docker/WSL runtimes, guarded judge model plumbing. |
| `harness_rsi.{evaluation_result_analyzer,member_optimizer,improver_evolution}` | stub (`deferred::member_optimizer`) | Agent-driven diagnosis/attribution/action planning; LLM prompts, no portable core. |
| `harness_rsi.auto_harness.infra.{git_operations,worktree_manager,ci_gate_runner,runtime_extension_loader,runtime_extension_static_checks,github_cli,skill_source_manager}` | stub (`deferred::runtime`) | Subprocess git/pip/pytest/ruff, synthetic `sys.modules` hot-loading, GitCode HTTP. |
| `harness_rsi.auto_harness.{stages,agents.factory,rails,tools}` live LLM orchestration | not stubbed — caller's `HarnessRunner` | DeepAgent construction and stage LLM turns belong to the caller (`openjiuwen-harness`). |
| `rsi.schema/events/usage` run ledger, `harness_rsi.progress` | deferred | No in-repo consumer in Python either (auto_harness never emits them). |
| `single_harness.{source_evidence,behavior_check,candidate_feedback,events_translate}` and the LLM analyser/improver turns | seam only — injected `Evaluator` / `MutationStrategy` | Evidence materialisation reads harness snapshots from disk and the analyser/improver are agent-driven LLM prompts; the loops themselves are implemented in `single_harness` and never build a model call. |
| Process-global state (`program_opt/state._DIRECTORIES`, `_MODEL_ENV_LOCK`, logging/checkpointer singletons) | not ported by design | Replaced by explicit per-session `SessionState` + injected stores. |
| `live-tests` (network/LLM) | never enabled by default | `docs/ARCHITECTURE.md` §5. |

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, flat re-exports, lint header
- `./src/evolution/` → shared candidate, mutation, evaluation, selection protocol (used by both halves)
  - `./src/evolution/candidate.rs` → `Candidate`, `CandidateId`, `CandidateStatus`
  - `./src/evolution/mutation.rs` → `MutationStrategy`, `MutationRequest`, `DeterministicMutator`
  - `./src/evolution/evaluation.rs` → `Case`, `CaseResult`, `EvaluationReport`, `Evaluator`
  - `./src/evolution/selection.rs` → `SelectionStrategy`, `ElitistSelection`, `rank_score`, `ranked`
- `./src/harness_rsi/` → harness-side RSI
  - `./src/harness_rsi/auto_harness/` → bounded self-optimisation session loop
    - `config.rs`, `budget.rs`, `task.rs`, `event.rs`, `runner.rs`, `checkpoint.rs`, `builder.rs` (collaborator defaults/validation), `orchestrator.rs` (+ `orchestrator_tests.rs`, a `#[cfg(test)]` child module)
  - `./src/harness_rsi/single_harness/` → single-harness promotion loops
    - `promotion.rs` (pure promotion math + reason literals), `config.rs`, `state.rs` (resumable state + idempotency keys), `iterative.rs` (`IterativeSingleHarness` + the shared `SessionCore`), `continuous.rs` (`ContinuousSingleHarness`)
- `./src/artifact_rsi/` → artifact-side RSI
  - `./src/artifact_rsi/program_opt/` → `ProgramOptConfig` and the PUCT tree (`config.rs`, `tree.rs`)
- `./src/deferred/` → documented `Error::Unsupported` stubs for ML/vendor surfaces (`agentdescent.rs`, `runtime.rs`, `evaluators.rs`, `paper_opt.rs`, `member_optimizer.rs`)
- `./tests/` → offline integration suites (`tests/CONTEXT.md`)

## Test Strategy

- `tests/config.rs` — Python defaults, YAML/JSON parsing with unknown fields and missing keys, `validate()` rejections, derived path properties, `load_or_default` bootstrap fallback.
- `tests/orchestrator.rs` — a scripted fake `HarnessRunner` drives: one accepted improvement across iterations, elite retention, convergence stop after `patience` flat iterations, target-score stop, budget stop (zero cost limit), `stop()` → cancelled iteration, and event-stream ordering (`session_started` … `harness_session_finished`).
- `tests/checkpoint.rs` — state save/load round-trip through `InMemoryCheckpointStore` and `KvCheckpointStore` over `openjiuwen_foundation::store::kv::InMemoryKvStore`/`LocalFileKvStore` (tempdir), plus `resume()` continuing at the persisted iteration.
- `tests/evolution.rs` — `EvaluationReport` aggregation (total/passed, ungraded never averaged, weighted score), `ElitistSelection` determinism and tie-breaking, `DeterministicMutator` stable output/ids, duplicate-id skip.
- `tests/single_harness_promotion.rs` — batch-filter boundaries at `min_improvement + EPSILON`, strict target improvement with missing entries, `mean_improvement`, every `stage_one_gate` reason literal, `threshold_saturated`, the four `full_evaluation_decision` outcomes (promotion, protected regression, target replay, score regression) including the retention/EPSILON boundaries, `case_scores` omission of ungraded/errored cases, fingerprint stability, state YAML round-trip, `batch_key`/`attempt_key`, and `resume_from` mismatch errors.
- `tests/single_harness_loops.rs` — a scripted `Evaluator` plus `DeterministicMutator` drive both policies offline: promotion on an improvement, rejection on a regression, iteration/target/cancellation stops, engineering-failure recording, resume without replaying a completed batch, byte-identical journals across two runs, the legacy "beats the incumbent" override, continuous per-batch promotion, the `max_trials` cap, protected-case retention, the per-batch attempt retry, and constructor rejections.
- `tests/program_opt.rs` — hand-computed PUCT scores (`c_puct = 1.0`, uniform priors), rank-score edges (`N = 1 → 0.5`, unscored `-inf`), ancestor visit increments, candidate limit rejection, `best()` tie-break, config validation, serde round-trip.
- `tests/deferred.rs` — every stub returns `ErrorKind::Unsupported` with a non-empty message; `EvolutionPolicy` parsing round-trips.
- Unit tests beside modules for budget warning thresholds and stop-condition ordering.
- All tests offline and deterministic; no network, no `#[ignore]`.
- Definition of done: `cargo fmt -p openjiuwen-rsi`, `cargo test -p openjiuwen-rsi`, `cargo clippy -p openjiuwen-rsi --all-targets -- -D warnings` all clean.

## Design Decisions

- Python's `AutoHarnessOrchestrator` has no explicit lifecycle (`run_session_stream` + `cancel`); the Rust orchestrator exposes `start`/`run_iteration`/`finish`/`stream` so iterations are testable and resumable, while `run()`/`stream()` preserve the single-call session shape.
- Python's harness dependency (`create_deep_agent`, `DeepAgent`, rails) is inverted into the `HarnessRunner` trait: RSI never constructs a harness, so the crate stays testable offline and the harness crate stays free of RSI knowledge.
- Candidate bodies are text (config YAML or program source) rather than Python objects: the same `Candidate` type serves harness optimisation and program search.
- Evaluation scores are `Option<f64>` per case so "ungraded" is representable and never folded into an average (Python `test_evaluator.py` behavior).
- Deferred surfaces are one-module-per-family stubs with typed constructors and a single `Result<Value>` method, so call-site shape exists without inventing fake type graphs; the table above is the authoritative gap list.
- `SessionState` is the single checkpoint unit (candidates + history + counters), which makes resume a full state restore instead of a separate replay protocol.
- The only injected traits are `HarnessRunner`, `Evaluator`, `MutationStrategy`, `SelectionStrategy`, `CheckpointStore`; there is no task-source or scoring trait.
- `task.rs` is data-only: the loop consumes `IterationRecord`, `SessionSummary`, `StopConditions`, `StopDecision`; `OptimizationTask`, `Gap`, `StageResult`, `TaskStatus`, `StageStatus` are unused by the orchestrator (kept for Python interface parity).
- Python's single-harness engine is one 200k-line driver; the Rust port keeps only its deterministic decision core (promotion math, idempotency keys, resume validation) and inverts the model halves into the crate's existing `Evaluator`/`MutationStrategy` seams, so both loops run offline and a harness run or an LLM improver plugs in without changing the orchestration.
- Publication stays gated by the promotion rules, not by a model: the analyser/improver may propose anything, but `stage_one_gate` plus `full_evaluation_decision` decide, so a bad proposal can never replace the incumbent.
- `SingleHarnessState` is the checkpoint unit; per-case incumbent scores travel in `state.metadata["best_case_scores"]` because the Python checkpoint has no field for them, and a resumed run therefore keeps its regression context instead of comparing against the seed again.
