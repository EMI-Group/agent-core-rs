# openjiuwen-rsi

## Intent

Recursive self-improvement (RSI) for the openJiuwen runtime, ported from the Python package `openjiuwen.rsi` (v0.1.20, ~97k lines / 313 files).
The crate owns the deterministic, offline-portable core of that package:

1. `harness_rsi::auto_harness` — a bounded, resumable self-optimisation session loop over candidate configurations (evaluate through a harness run, keep the elite, checkpoint every iteration, stop on convergence/budget/cancellation).
2. `harness_rsi::single_harness` — the evaluate→gate→promote core of Python's single-harness engine: exact promotion math, resumable epoch state with idempotency keys, and the iterative (epoch-checkpointed) and continuous (two-stage, attempt-based) loops.
3. `harness_rsi::evaluator` — case/dataset records, the judge scoring contract, the case-runner bridge, and the SWE-bench / terminal-bench runtimes (argv, report classification, results parsing) plus metrics.
4. `harness_rsi::member_optimizer` — one plan → execute → verify → publish pass: target selection, action typing, dependency waves, verification, repair.
5. `harness_rsi::progress` — the progress event consumed by downstream UIs.
6. `artifact_rsi::program_opt` — the portable PUCT tree search ranking candidate programs.
7. `artifact_rsi::paper_opt` — rubric scoring, the greedy-frontier tree with its strict adoption gate, proposal, and the deterministic orchestration loop.
8. `runtime` — the real process layer: a typed command seam, git operations, CI gates, the Python toolchain (venv/pip/pytest) and the subprocess program evaluator.
9. `ledger` — the persisted model-call run ledger (JSONL), with replay-safe resume and per-model/per-stage aggregation.
10. `evolution` — the crate-wide candidate/mutation/evaluation/selection protocol shared by all optimisers.

The crate is layer 6 (`docs/ARCHITECTURE.md` §1): it may depend on `openjiuwen-core`, `openjiuwen-harness`, and `openjiuwen-common`, never upward, and never on `openjiuwen-agent-evolving` (that would create a cycle).

Everything that needs an ML runtime (`agentdescent`), guarded judge-model plumbing, synthetic `sys.modules` hot-loading, Docker/WSL containers, or a `DeepAgent` stage pipeline stays behind a trait seam or in `deferred` as an `Error::Unsupported` stub — never a silent stub, never a panic.

## API Surface

Module paths mirror Python: `openjiuwen_rsi::harness_rsi::auto_harness::*`, `openjiuwen_rsi::artifact_rsi::paper_opt::*`, `openjiuwen_rsi::runtime::*`, `openjiuwen_rsi::ledger::*`.
The major types are also re-exported flat from the crate root (`openjiuwen_rsi::AutoHarnessOrchestrator`, `openjiuwen_rsi::PaperOptimizer`, `openjiuwen_rsi::RunLedger`, `openjiuwen_rsi::ProcessCommandRunner`, …).
`openjiuwen-auto-harness` is a re-export shim over `openjiuwen_rsi::auto_harness`.

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
selection.rs  trait SelectionStrategy { fn name(&self) -> &str;
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
              SessionStarted, IterationStarted, CandidatesEvaluated, IterationFinished,
              CheckpointSaved, StageResult, SessionFinished ("harness_session_finished"), SessionCancelled
runner.rs   HarnessRunRequest { session_id, iteration, candidate, cases, metadata }
            HarnessRunOutcome { report: EvaluationReport, artifacts, messages }
            #[async_trait] trait HarnessRunner { fn name(&self) -> &str;
                async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome>; }
              — the only seam to `openjiuwen-harness`; callers wire `DeepAgent` runs, tests fake it.
            HarnessEvaluator { runner: Arc<dyn HarnessRunner> } impl Evaluator — one run per candidate.
checkpoint.rs SessionState { session_id, iteration, running, best_candidate, best_score,
                              candidates, history, spent_usd, stop_reason }  (full serde round-trip = resume)
            #[async_trait] trait CheckpointStore { save/load/delete }
              — backend/deserialization failures are `Error::session`, never "no checkpoint".
            InMemoryCheckpointStore, KvCheckpointStore (over `foundation::store::kv::KvStore`,
              one JSON-text record per session under key prefix `DEFAULT_KEY_PREFIX` = "rsi:auto_harness:",
              so a caller with the backing store lists every session via `KvStore::get_by_prefix`),
            SessionCheckpointStore (over `update_component_state`).
orchestrator.rs AutoHarnessOrchestrator — `builder(config)` (a required runner **or** evaluator) with
              with_runner/with_mutator/with_selector/with_evaluator/with_checkpoint_store/with_cases/
              with_seed_body/with_session_id/with_mutation_count/build;
              config(), session_id(), state(), is_running(), is_cancelled(), start(), resume(),
              run_iteration(), stop(), finish(), stream(), run().
              One iteration: seed if empty → select parents → mutate → evaluate → mark statuses →
              compare best to incumbent (improvement, accepted) → append IterationRecord → checkpoint →
              evaluate stop conditions.
              `improvement` = `best - previous_best` clamped at 0 (0.0 for the first best, accepted);
              the winner becomes `CandidateStatus::Selected`.
              `run_iteration` on a finished/stopped session returns Error::Cancelled; evaluation errors
              mark the candidate Invalid and never abort the session; a checkpoint write failure is
              Error::Session; `finish()` is idempotent and defaults a missing stop reason to MaxIterations.
              `run`/`stream` are alternative single-call entries with a terminal-event latch.
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
                  // first match wins: protected_case_regressed > candidate_failed_target_replay_checkpoint >
                  //   selected_score_regressed > selected_global_non_regression (promoted)
                  // min_improvement is diagnostic only (the batch gate already enforced it).
              Reason literals are pub consts: REASON_CONTENT_UNCHANGED, REASON_ENGINEERING_FAILED,
              REASON_THRESHOLD_SATURATED, REASON_TARGET_REPLAY_FAILED, REASON_BATCH_GATE_PASSED,
              REASON_SELECTED, REASON_SELECTED_SCORE_REGRESSED, REASON_PROTECTED_CASE_REGRESSED.
config.rs     SingleHarnessConfig { mode: PromotionPolicyMode /*ContinuousTwoStageV2*/, max_epochs 5,
                max_iterations 10, batch_size 5, min_improvement 0.05, max_regression 0.05, patience 2,
                target_score: Option<f64>, max_attempts_per_batch 2, max_trials 8, promote_on_best_found true,
                protected_cases, evaluation_protocol_signature "v1", extra } — Default, validate(),
                fingerprint(&[String]) -> String (FNV-1a 64-bit hex over dataset ids + signature + mode),
                protected_case_set()
state.rs      PromotionPolicyMode { LegacyEpochV1, ContinuousTwoStageV2 }
              SingleHarnessStatus { Running, Completed, Failed, Cancelled }
              UsageState { calls, cost_usd }; CandidateGateRecord { batch_key, candidate_id, passed, reason, round }
              JournalEntry { epoch, batch_index, reason, score, promoted }
              SingleHarnessState { version 7, status, mode, fingerprint, dataset, best_harness_refs_path,
                best_score, best_body, baseline_score, completed_batches, candidate_gates, optimization_journal,
                historical_peak_score, continuous_attempts, continuous_trials, max_iteration, task_id,
                evaluation_protocol_signature, stop_reason, usage, metadata }
                new(..), is_batch_completed, mark_batch_completed, iterations(), consecutive_non_promotions(),
                resume_from(previous, fingerprint, task_id, max_iteration, signature)
                  (Error::validation naming the mismatched field; resets status/stop),
                save(path)/load(path) async YAML (Error::session on IO/parse failure)
              batch_key(epoch, batch_index) = "epoch_{epoch:03}:batch_{batch_index:03}";
              attempt_key(batch_key, attempt) = "<batch_key>:attempt_{attempt:03}";
              state_path(output_dir) = <dir>/single_harness_state.yaml
iterative.rs  SessionCore (crate-internal: config, Arc<dyn Evaluator>, Arc<dyn MutationStrategy>, state,
                best_body, best_scores, Budget, CancellationToken, cases, seed_body, counters)
                ensure_baseline, select_batch(epoch), batch_cases, scores_for, evaluate (cost-accounted),
                propose, record_gate, record_outcome, apply_promotion, stop_reason, run_attempt
              IterativeSingleHarness { new(config, evaluator, improver, task_id, dataset, cases, seed_body),
                with_state, with_budget, state, best_body, is_cancelled, run, run_epoch(epoch) -> bool, stop }
              IterativeResult { state, best_score, best_body, promotions, evaluations }
continuous.rs ContinuousSingleHarness (same surface; always the continuous policy) + type ContinuousResult
              run_epoch: up to max_attempts_per_batch attempts per batch, at most one promotion per batch,
              max_trials cap across the run (persisted), no legacy override; trials cap reports MaxIterations.
```

Both loops: one journal entry per attempted batch, one gate record per evaluated candidate, per-case incumbent scores persisted in `state.metadata["best_case_scores"]`, `promotions`/`evaluations` counted per process.
Candidate ids come from the mutation strategy with generation `iterations + 1` (iterative) / `continuous_trials + 1` (continuous), so ids, ordering and tie-breaks are deterministic.
Evaluation errors on a candidate become `"engineering_failed: …"` journal entries and never abort the loop; a baseline error marks the run failed and is returned.

### `harness_rsi::evaluator` (`src/harness_rsi/evaluator/`)

```
case.rs    AnswerRole { Criterion /*default*/, Reference }; PenaltyMode { Ceiling /*default*/, Subtract }
           BehaviorSpec { id, weight 1.0 }; ForbiddenBehaviorSpec { id, penalty 0.3 }
           CaseReference { answer, expected_output, rubric, files, required_behaviors, forbidden_behaviors,
             answer_role, penalty_mode, solution, expected_artifacts, deliverables, artifact_contract }
           SweBenchCase { instance_id, repo, base_commit, patch, test_patch, fail_to_pass, pass_to_pass,
             split, extra } (id aliases instance/id)
           EvaluationCase { case_id, name, input, assets, reference, workspace_source_dir, swebench, metadata }
             from_json_value(index, value): case_id ← case_id|id|case_{index:03};
             input ← input|inputs|task_input|query|prompt|question; unknown keys → metadata
           Dataset { dataset_id, cases } — from_json_value (array or {dataset_id, cases}),
             from_jsonl_str, load(path) (jsonl/ndjson by extension), case_ids, len, is_empty;
             malformed entries → Error::validation naming the case index, duplicate ids rejected
scoring.rs DEFAULT_SUCCESS_THRESHOLD 0.8; DEFAULT_FORBIDDEN_PENALTY 0.3; VERDICT_COMPLETED; METHOD_JUDGE
           BehaviorVerdict { id, score 0..=1, reason, evidence }; ForbiddenHit { id, triggered, reason, evidence }
           JudgeVerdict { status, overall_reason, behaviors, forbidden_hits }
           JudgeResult { method, score, passed, reason, metadata }
           ScoringContract { required, forbidden, penalty_mode, success_threshold, binary }
           parse_judge_verdict(text) — bare/fenced/prose-wrapped, one payload only; rejects non-finite
             numbers, duplicate ids, a non-object root, a status other than completed
           score_verdict(contract, verdict): base = Σ(score·weight)/Σweight over required (missing → 0);
             penalty = Σ spec penalty over triggered forbidden; Ceiling → min(base, 1-penalty);
             Subtract → max(0, base-penalty); passed = score >= threshold; binary stores 1.0/0.0 with
             metadata["raw_score"]
           verdict_from_case(reference, threshold, binary) — expands required/forbidden behaviours,
             rubric_NNN and a criterion reference_answer into a contract
           classify_judge_failure(text) -> JudgeFailure { Truncated, RepairInvalid, RepairUnavailable,
             Unavailable, InvalidSchema }; requirement_results(contract, verdict) -> Vec<RequirementResult>
runner.rs  ExecutionStatus { Passed, Failed }
           CaseExecutionResult { response, execution_status, error, workspace_dir, metadata }
           #[async_trait] trait CaseExecutionBackend { name; execute(CaseExecutionRequest) -> Result<..>;
             cleanup(case_id) -> Result<()> (default Ok) }
           CaseExecutionRequest { case, output_dir, session_id }
           CaseTrace { case_id, status (passed|failed|error), result: CaseResult, execution, artifacts }
             is_infrastructure_error() — true when the error matches a transient marker
           CaseRunner { backend, threshold, binary } — run(request) -> Result<CaseTrace>; grades the
             response as a judge verdict or a {score, passed} object; a non-completed verdict is ungraded
           is_transient_error(message) + TRANSIENT_ERROR_MARKERS (Python's 23 lower-case substrings)
           CaseCommandBuilder trait + CommandCaseBackend { runner, builder, field } — runs the case
             command and reads a JSON response from stdout; non-zero exit → Failed with stderr
           CandidateCaseRunner impl Evaluator — maps evolution::Case.input into EvaluationCase and one
             CaseResult per case
swebench.rs SweBenchConfig { python "python3", dataset "princeton-nlp/SWE-bench_Verified", split "test",
             namespace "swebench", instance_image_tag "latest", run_id (uuid), model_name "openjiuwen",
             timeout_secs 1800, cache_level "env", extra_args, reports_dir }
           SweBenchClassification { Resolved, Unresolved, Error, Incomplete }
           SweBenchOutcome { instance_id, classification, score (1.0/0.0/None), report_path, error, metadata }
           SweBenchEvaluator::run_command(instance_id, predictions) — `<python> -m
             swebench.harness.run_evaluation -d … -s … -i … -p … --max_workers 1 -t … --cache_level …
             --clean false -id … --namespace … --instance_image_tag …` (+ extras), timeout +600 s
           classification_from_report (resolved > error > incomplete > unresolved)
           failure_reason / per_instance_failure_reason / infrastructure_failure_reason — Python literals
           SweBenchDatasetEvaluator impl Evaluator (resolved 1.0, unresolved 0.0, error/incomplete ungraded)
terminal_bench.rs TerminalBenchConfig { command "terminal-bench", dataset, timeout_secs 3600, task_filter,
             results_dir, extra_args }
           TerminalBenchCommandLog { command, cwd, exit_code, stdout_excerpt, stderr_excerpt, timeout_sec,
             background } (1000-char excerpts)
           TerminalBenchOutcome { task_id, passed, score, log, error, metadata }
           TerminalBenchEvaluator::run_command / evaluate_case (missing results file is an outcome error)
           parse_results — {"results": […]} | {"tasks": {id: …}} | {"resolved_ids", "unresolved_ids"}
           TerminalBenchDatasetEvaluator impl Evaluator
metrics.rs CaseResultRecord { case_id, status, score, error, metadata }
           MetricsSummary { total_cases, passed_cases, failed_cases, average_score, evaluation_method,
             case_results } (+ all_passed)
           collect_metrics(traces, method), summary_from_results(results, method),
           summarize_reports(reports, method), write_summary(path, summary) async
sanitize_id(value) — Python `_safe_id`: lowercase, alnum kept, other → '-', 48-char cap, "" → "case"
```

The benchmark evaluators never provision containers: they build the official argv, run it through `runtime::command`, and read the report/results file the harness wrote.
Tests inject `FakeCommandRunner` and write those files themselves.

### `harness_rsi::member_optimizer` (`src/harness_rsi/member_optimizer/`)

```
config.rs   MemberOptimizerConfig { max_roles_per_run 2, max_actions_per_plan 3 (0 = unlimited),
              max_repair_rounds 1, min_confidence 0.0, fail_on_unverifiable false, extra } — Default, validate()
selection.rs Severity { Critical, High, Medium, Low } (weights 4/3/2/1)
            MemberTarget { role, severity, confidence, eligible_actions, issue_ids, member_name, metadata }
            selection_score = severity.weight()*10 + confidence*5 + min(eligible, 5)/5
            select_targets(targets, beam_width, min_confidence) -> Vec<SelectedTarget>
              — filter confidence/eligible, sort score desc → severity desc → role asc, take beam width
            rank_targets (full ordered list); SelectedTarget { target, score }
            AttributionInput/AttributionResult/IssueObservation/UnassignedIssue + SynthesizedAttributor
              (deterministic offline stand-in reading issue metadata `role`)
plan.rs     RunIf { DependencySucceeded, DependencyFailed, Always }
            OptimizationAction { action_id, role, action_group, operation, action_type, target_path,
              description, rationale, attributed_issue_ids, depends_on, run_if, allowed_skills,
              allowed_tools, expected_effect, risk_notes, declared_write_paths, member_name, metadata }
            OptimizationPlan { plan_id, targets, actions, action_waves, metadata }
            ActionDefinition — the template library a planner picks from
            plan_waves(actions) — Kahn layering; duplicate id / unknown dependency / cycle → Error::validation
            ActionPlanner trait + DeterministicActionPlanner { definitions, max_actions }
              (ids `<opt>/r<round>/a<n>`, per-role chaining, allowlist + per-target cap)
verify.rs   CheckStatus { Passed, Failed }; VerificationCheck { name, status, detail }
            ActionOutcome { action_id, success, output, written_paths, error }
            MemberVerificationResult { status, checked_roles, checks, role_results, repairable, metadata }
            WorkspaceProbe trait + FsWorkspaceProbe + InMemoryWorkspaceProbe
            ActionVerifier trait + StaticActionVerifier — checks action_succeeded,
              declared_write_paths (absence, `..`/absolute escape, prefix membership), target_exists;
              checks sorted by action id then name; repairable = a fixable failure without a write violation
optimizer.rs MemberOptimizationRequest { optimization_id, output_dir, targets, workspace_root, allowlist, metadata }
            MemberOptimizationStatus { Success, PartialSuccess, Failed, Skipped, NoIssues, NoTargets,
              PlanningRejected, NoActions }; PromotionStatus { PendingGate, NotEvaluated, NotApplicable }
            MemberOptimizationArtifact { optimization_id, output_dir, status, plan, verification,
              promotion_status, targets, rounds, metadata }
            ActionExecutor trait + DeterministicActionExecutor (creates declared write paths + a JSON record)
            MemberOptimizer { new, with_planner, with_executor, with_verifier, with_probe, config,
              optimize(request) } — select → plan → execute wave by wave → verify → repair once → publish
```

Planning validation failure yields status `PlanningRejected` (an `Err` only for infrastructure failures); an empty selection is `NoTargets`, an empty plan `NoActions`.
The LLM attributors, planner, per-action executor and repair agent stay behind traits or caller-supplied data.

### `artifact_rsi::paper_opt` (`src/artifact_rsi/paper_opt/`)

```
config.rs   GeneralWeights / ExperimentWeights / PaperWeights (each set sums to 1 within 1e-6;
              composite 0.5 general + 0.5 experiment)
            PaperOptConfig { max_iterations 5, max_rounds 5, rubric_samples 5, beam_width 2, weights,
              min_score_range 3.0, mad_threshold 1.0, min_composite, require_artifact true, extra }
              Default, validate(), from_json_value, from_yaml_str
scoring.rs  GeneralScores { soundness, clarity, contribution, overall }
            ExperimentScores { question_alignment, design_and_controls, measurement_and_statistics,
              reporting_and_reproducibility, claim_evidence_alignment, overall_experimental_rigor }
            RubricSample (one general + one experiment set, 1..=10)
            aggregate/aggregate_with_weights(samples) -> ScoreBreakdown { general, experiment, composite,
              sample_count } — mean per dimension; empty input → Error::validation
            general_score/experiment_score/composite_score; dispersion + flags_dispersion(range 3.0, mad 1.0)
            round_half_even(value, decimals) + compact_scoresheet (4 dp, banker's rounding)
tree.rs     PaperNodeType { Root, Reporting, Pruned }; PaperChange { group, operation, function, target, summary }
            PaperNode { node_id, iteration, parent_id, node_type, adopted, score, summary, failure_class,
              changes, metadata }
            PaperTree { nodes, next_iteration } — new, with_root, add_candidate (unknown parent →
              Error::validation; ids n000…), record_score, set_outcome, node, nodes, len, is_empty, best,
              frontier, path_to, pruned
            adoption_gate(parent_score, candidate, required_artifact_present, scoring_error, pipeline_status)
              -> AdoptionOutcome { Adopted, Pruned { failure_class }, Rejected { failure_class } }
              order: pipeline_<status> → paper_generation → scoring_error → strict composite > parent
              (ties rejected_by_score)
proposal.rs PaperCandidate { candidate_id, parent_id, summary, body, changes, samples,
              required_artifact_present, pipeline_status, metadata }
            PaperProposalRequest { iteration, frontier, parent_body, count, existing_ids }
            PaperProposer trait + TemplatePaperProposer (deterministic section revisions; samples cycle)
orchestrator.rs PaperStatus { Running, Completed, Failed, Cancelled }; PaperTaskState { task_id, status,
              max_iterations, artifact_path, best_node_id, score, baseline, frontier_node_id, node_count,
              attempts_since_last_adoption, last_reason, error_code, error_message }
            PaperRunOutcome { state, tree, adopted, rejected }
            PaperScorer trait + RubricPaperScorer (aggregates candidate samples)
            PaperOptimizer { new, with_proposer, with_scorer, with_task_id, with_seed_body,
              with_artifact_path, config, state, tree, run, run_iteration, stop }
              — per iteration: propose beam candidates → score (scoring error prunes) → rank with the
                existing PuctTree → adoption gate → Reporting/Pruned node → stop on iterations,
                min_composite, or cancellation
```

### `runtime` (`src/runtime/`)

```
command.rs  CommandSpec { program, args, cwd, env, env_remove, timeout, merge_stderr, stdin }
              new + arg/args/cwd/env/env_remove/timeout/merge_stderr/stdin/with_default_timeout,
              command_line()
            CommandOutcome { code: Option<i32>, stdout, stderr, timed_out, cancelled, duration }
              success(), combined_output(), truncated_output(limit), success_with/failure_with/
              timed_out_after/cancelled_after, require_success(label)
            #[async_trait] trait CommandRunner { name; run(CommandSpec); run_cancellable(spec, token)
              (default: select over run and cancellation; the child is kill-on-drop) }
            ProcessCommandRunner { new, with_default_timeout, default_timeout } — tokio::process,
              kill_on_drop, stdin per spec, lossy UTF-8 with one trailing newline stripped,
              missing executable → Error::config, other spawn failure → Error::tool, timeout → data
            FakeCommandRunner { reply, fallback, calls, call_count, command_lines } + FakeReply
              { program ("*" wildcard), args_contain, outcome }; unmatched → Error::tool
            decode_output(bytes) -> String
git.rs      GitOperations { runner, workspace, program "git", base_branch "develop", remote "origin",
              env, timeout } — new/with_* + getters, raw(args)
            create_branch -> GitBranchResult; current_branch/current_head; collect_status -> GitStatus
              { dirty_files, tracked_modified_files, untracked_files, renamed_files } (porcelain
              M/A/D/R/??; renames take the new path; `\`→`/`, deduped, sorted)
            list_dirty_files, diff_stat, diff_stat_against_base, diff_name_only, diff_name_only_against_base,
            diff_against_base, status_porcelain, show_last_commit_stat, has_commits_against_base,
            add_paths (empty → success:false "no paths provided"), commit, discard_worktree_changes,
            worktree_add/remove/prune, push (3 escalating attempts),
            cherry_pick_ref_commits (abort on failure) -> GitCherryPick
            slugify_topic, build_auth_env(user, token), scrub_credentials(text, user, token)
ci.rs       GateKind { Lint, TypeCheck, Command } (unknown YAML kind → Command)
            CiGateSpec { name, kind, command, commits }; GateOutcome { name, passed, output }
            CiGateReport { passed, gates, errors }
            CiGateRunner { new, with_gates, with_python, with_timeout, gates, run(action) }
            parse_gates(yaml) / default_gates_yaml (fail-closed: malformed/missing → empty),
            normalize_command, extract_commits, shell_command, changed_line_ranges (git diff -U0 hunks),
            filter_findings, sanitize_failure_output, gate_outcome; FAIL_MARKER "__CI_CHECK_FAIL__",
            OUTPUT_LIMIT 4000
pytest.rs   TestFailure { name, message }; PytestReport { total, passed, failed, errors, skipped, failures }
            parse_json_report / parse_junit_xml / parse_report (hand-written tolerant XML scanner,
            entity unescaping, malformed root → Error::validation)
python.rs   PythonToolchain { runner, workspace, python, venv_dir, timeout, report_format }
              new, with_python/with_venv/with_timeout/with_report_format, python, venv_python,
              ensure_venv (idempotent via pyvenv.cfg) -> ToolchainSetup,
              pip_version, ensure_pip, pip_install (failure text prefixed "pip_install_failed: "),
              pip_install_requirements_file, run_pytest -> PytestOutcome
            ReportFormat { Json, JUnit }; pytest_args(format, report_path); discover_python()
subprocess.rs SubprocessConfig { python "python3", workdir_root (temp dir), timeout 60 s,
              entrypoint "candidate.py", reply_format Files, env }
            SubprocessResult { success, response, outcome, workdir, error }
            SubprocessEvaluator { new(runner), with_config, keep_workdir, config, evaluate(program) }
              — writes the program into a unique workdir, runs it, parses the reply
                (Files: output.json/response.json; Tagged: last ```json fence → <result> → whole stdout),
                always removes the workdir unless kept
            parse_tagged_reply, parse_files_reply
```

Exit codes are data, never errors; only a missing executable, an I/O failure, or cancellation is an `Error`.
Every layer builds on the [`CommandRunner`] seam rather than spawning processes itself.

### `ledger` (`src/ledger/`)

```
record.rs   ModelCallStatus { Succeeded, Failed, Incomplete }
            UsageTokens { input, output, cache_hit: Option<u64> } (None = unreported)
            UsageSnapshot { tokens, cost_estimate, call_count }
            ModelCall { model, call_count, tokens, status, duration_ms }
            UsageRecord { event_id, task_id, ts, call_id, model_call, node_ref, stage_ref, cost_usd,
              family "progress", kind "usage", event_type "progress.usage" }
            StageRecord { node_ref, stage, note, event_type "node.stage" }
            RunEvent { Usage(UsageRecord), Stage(StageRecord) } — flat JSONL line per record
            tokens_from_usage(raw) — input ← input_tokens|prompt_tokens|input;
              output ← output_tokens|completion_tokens|output;
              cache_hit ← cache_read_tokens|prompt_cache_hit_tokens|cache_hit|prompt_tokens_details.cached_tokens
            usage_snapshot(raw) -> Option<UsageSnapshot> (None when nothing is reported);
            now_rfc3339/format_rfc3339 (no chrono dependency)
usage.rs    UsageTotals { calls, input_tokens, output_tokens, cache_hit_tokens, cost_usd } (+ observe/merge)
            LedgerMetrics { totals, failed_calls, incomplete_calls, by_model, by_stage }
            aggregate(records) — stage bucket = stage_ref else "unattributed"; None tokens contribute 0
store.rs    #[async_trait] trait LedgerStore { append, load, remove }
            InMemoryLedgerStore (events(task_id) for assertions);
            FileLedgerStore::new(root) — <root>/<sanitised task id>.jsonl, create+append one JSON line,
            blank lines skipped, malformed line → Error::session
ledger.rs   RunLedger { task_id, store, records, seen_calls, pending, totals }
              new, with_store, task_id, bind (replay + task_id check → Error::validation; idempotent),
              record_usage (dedupe by call_id → bool), mark_pending, finish_pending,
              record_stage, totals, records, metrics, snapshot,
              check_delivery (pending → Error::session "RSI model usage delivery failed; persisted calls
              can be reconciled"), clear
```

The ledger is content-free (model, counters, status, references — never prompts, responses or credentials).
`harness_rsi::progress` provides `OptimizationProgressEvent { phase, epoch, batch_index, stage, score, improved, message, metrics, artifacts }`, `ProgressCallback = Arc<dyn Fn(&OptimizationProgressEvent) + Send + Sync>`, `emit_progress(Option<&ProgressCallback>, &event)` and `SeedOptimizationDecision { Accept, Reject }`.

### `artifact_rsi::program_opt` (`src/artifact_rsi/program_opt/`)

```
config.rs   ReplyFormat { Files, Tagged }; MAX_WORKERS = 8; DEFAULT_WORKERS = 1
            ProgramOptConfig { c_puct (1.0), prior_exponent (0.0), candidate_limit: Option<usize>,
                               max_expansions (16), max_depth (8), workers (1),
                               entrypoint ("candidate.py"), candidate_timeout_secs (60.0),
                               reply_format: ReplyFormat } — Default, validate(), to_tree()
tree.rs     Program { program_id, iteration, parent_id, code, change_summary, metrics, valid, error }
            Node { index, parent_index, program, score, num_visits, promise }
            PuctTree { c_puct, prior_exponent, candidate_limit, nodes, next_iteration }
              new(&ProgramOptConfig), with_root, from_nodes, nodes, len, is_empty, root,
              next_iteration, candidate_limit_reached,
              add_node(parent_index, program) -> Result<usize> (unknown parent / reached limit →
                Error::Validation), select(&mut self) (PUCT + ancestor visit increments),
              puct_scores/priors/rank_scores, record_visits, best() (highest score, smallest index on ties)
            PUCT (Python `artifact_rsi/program_opt/tree.py`):
              rank_score_i = rank_i / (N-1) (N<=1 → 0.5), ranks over ascending stable score sort, None = -inf;
              P_i = 1/N when prior_exponent <= 0, else promise^prior_exponent normalised (unrated = mean prior);
              puct_i = rank_score_i + c_puct * P_i * sqrt(sum(visits)) / (1 + visits_i); first maximum wins.
```

### `deferred` (`src/deferred/`)

API-shaped stubs, all documented and all returning `Err(Error::Unsupported("<surface> is not implemented in the Rust port: <reason>"))`:

- `agentdescent::EvolutionEngine` (plus the implemented `EvolutionPolicy { Full, Guarded, Reflective }` data) — needs the `agentdescent==0.4.6` evolve/ledger/aggregator internals.
- `evaluators::LlmJudgeEvaluator` — guarded judge-model plumbing and judge prompt construction live in the harness layer.
- `runtime::RuntimeExtensionLoader` — synthetic `sys.modules` hot-loading of Python runtime extensions has no Rust analogue.

## Constraints

- Errors: `openjiuwen_common::error::{Error, Result}` only; Python category names preserved (`Error::config` for bad config/spawn, `Error::validation` for bad input/limits, `Error::unsupported` for deferred surfaces, `Error::cancelled` for cancellation, `Error::session` for checkpoint/ledger/state failures, `Error::tool` for process failures). No `unwrap`/`expect`/`panic` outside `#[cfg(test)]`.
- Async-first: tokio; `async_trait` for object-safe seams; public streams are `BoxStream<'static, Result<..>>` built with `async_stream::try_stream!`; locks are `std::sync::Mutex`/`RwLock` and never held across `.await`.
- Process execution: only through `runtime::command::CommandRunner`; exit codes are data, timeouts kill and reap the child, cancellation is `Error::cancelled`, prompts are suppressed for git (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=""`), credentials are scrubbed from any reported output.
- Cancellation: `tokio_util::sync::CancellationToken`; `stop()` cancels it, in-flight and subsequent iterations observe it at await points.
- Serde: every public data type round-trips (`snake_case`, `#[serde(default)]`, unknown fields tolerated); Python field names preserved for records (`StageStatus`/`TaskStatus` values, plan/action/verification/benchmark field names, config keys).
- Config: YAML via `serde_yaml_ng`; env loading happens in binaries only, never here.
- Logging: `tracing` only; no stdout/stderr in library code.
- Determinism: ids, ordering, tie-breaks, mutation output, and state fingerprints are deterministic given the same inputs — no wall-clock in scores or ids; `duration_ms`/`duration` are the only timing values and are never asserted.
- Injected seams (never constructed internally): `HarnessRunner`, `Evaluator`, `MutationStrategy`, `SelectionStrategy`, `CheckpointStore`, `PaperProposer`, `PaperScorer`, `ActionPlanner`, `ActionExecutor`, `ActionVerifier`, `WorkspaceProbe`, `CaseExecutionBackend`, `CaseCommandBuilder`, `CommandRunner`, `LedgerStore`.
- Size: files ≤1000 lines, one responsibility per file; the shared protocol lives in `evolution/`, the process seam in `runtime/`.
- Clippy (`-D warnings`) with workspace lints: every public item documented, `#[must_use]` on getters, `# Errors` sections on every `Result` return; `#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]` in `lib.rs`.

## Deferred Scope

| Python surface | Status | Reason |
|---|---|---|
| `artifact_rsi.program_opt.{puct_engine,puct_provider,script_domain,probe,provision}` | stub (`deferred::agentdescent`, `deferred::runtime`) | Needs the `agentdescent==0.4.6` evolve/ledger internals and distributed search; the portable PUCT tree is implemented. |
| agentdescent itself (`selection`, `aggregator`, `ledger`, `staleness`, `filetree`, `evolution`) | deferred | Exact 0.4.6 semantics; distributed/deep-RL training has no maintained Rust equivalent. |
| `artifact_rsi.paper_opt.auto_research` (six DeepAgent stage agents, `ManagerRuntime`, paper scoring prompts, LaTeX/pyright/Pillow) | seam only — `PaperProposer` / `PaperScorer` | Model-driven stages and a Python host toolchain; the scoring math, tree, adoption gate and loop are implemented. |
| `harness_rsi.evaluator.{team_evaluator,runtime_adapters,trajectory_paths,trajectory_usage,baseline_context,optimization_signals}` and `evaluator.judger.*` | seam only — `CaseExecutionBackend` / judge text | DeepAgent solver construction, trajectory rails and guarded judge-model plumbing; the records, verdict math and grading bridge are implemented. |
| `harness_rsi.evaluator.{swebench,terminal_bench}_runtime` container internals (Docker/WSL provisioning, solver containers, per-instance reports) | out of scope — the official CLI runs through `runtime::command` | Container provisioning needs Docker/WSL hosts; argv construction, report classification and results parsing are implemented. |
| `harness_rsi.{evaluation_result_analyzer,improver_evolution}` and the member optimiser's LLM attributors/planner/executor/repair | seam only — `ActionPlanner` / `ActionExecutor` / `ActionVerifier` + caller attribution | Agent-driven diagnosis and action planning are LLM prompts; selection, waves, verification and the pass loop are implemented. |
| `harness_rsi.auto_harness.infra.{github_cli,skill_source_manager}` | not ported | Shells out to `gh` / downloads GitHub archives over the network; offline builds cannot exercise either. |
| `harness_rsi.auto_harness.infra.{runtime_extension_loader,runtime_extension_static_checks}` | stub (`deferred::runtime`) | Synthetic `sys.modules` hot-loading and Python import introspection have no Rust analogue. |
| `harness_rsi.auto_harness.infra.{parsers,commit_scope,edit_scope,fix_loop,pipeline_selector,runtime_manifest}` | not ported | Stage-pipeline helpers whose consumers are the live LLM stages; `session_budget` is covered by `auto_harness::Budget`. |
| `harness_rsi.auto_harness.{stages,agents.factory,rails,tools}` live LLM orchestration | not stubbed — caller's `HarnessRunner` | DeepAgent construction and stage LLM turns belong to the caller (`openjiuwen-harness`). |
| `single_harness.{source_evidence,behavior_check,candidate_feedback,events_translate}` and the LLM analyser/improver turns | seam only — `Evaluator` / `MutationStrategy` | Evidence materialisation reads harness snapshots from disk and the analyser/improver are agent-driven prompts; the loops themselves are implemented. |
| `usage.py` `Runner.callback_framework` registration and `@model_usage_stage` ContextVar | not ported by design | Replaced by an explicit `RunLedger` that the caller records into (per-run `Arc`, no global state). |
| Process-global state (`program_opt/state._DIRECTORIES`, `_MODEL_ENV_LOCK`, logging/checkpointer singletons) | not ported by design | Replaced by explicit per-session state + injected stores. |
| `live-tests` (network/LLM) | never enabled by default | `docs/ARCHITECTURE.md` §5. |

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, flat re-exports, lint header
- `./src/evolution/` → shared candidate, mutation, evaluation, selection protocol (used by every optimiser)
  - `./src/evolution/candidate.rs` → `Candidate`, `CandidateId`, `CandidateStatus`
  - `./src/evolution/mutation.rs` → `MutationStrategy`, `MutationRequest`, `DeterministicMutator`
  - `./src/evolution/evaluation.rs` → `Case`, `CaseResult`, `EvaluationReport`, `Evaluator`
  - `./src/evolution/selection.rs` → `SelectionStrategy`, `ElitistSelection`, `rank_score`, `ranked`
- `./src/runtime/` → process-level infrastructure
  - `./src/runtime/command.rs` → `CommandSpec`/`CommandOutcome`, `ProcessCommandRunner`, `FakeCommandRunner`
  - `./src/runtime/git.rs` → `GitOperations`, status/diff/worktree/push/cherry-pick parsing, credential scrubbing
  - `./src/runtime/ci.rs` → `CiGateRunner`, gate config, command normalisation, changed-line scoping
  - `./src/runtime/python.rs` → `PythonToolchain` (venv, pip, pytest invocation)
  - `./src/runtime/pytest.rs` → JSON and JUnit report parsers
  - `./src/runtime/subprocess.rs` → `SubprocessEvaluator` and reply parsing
- `./src/ledger/` → persisted model-call ledger
  - `./src/ledger/record.rs` → records, statuses, token normalisation
  - `./src/ledger/usage.rs` → totals and per-model/per-stage metrics
  - `./src/ledger/store.rs` → `LedgerStore`, in-memory and JSONL file stores
  - `./src/ledger/ledger.rs` → `RunLedger` (bind/replay, dedupe, pending calls)
- `./src/harness_rsi/` → harness-side RSI
  - `./src/harness_rsi/auto_harness/` → bounded self-optimisation session loop
    - `config.rs`, `budget.rs`, `task.rs`, `event.rs`, `runner.rs`, `checkpoint.rs`, `builder.rs`, `orchestrator.rs` (+ `orchestrator_tests.rs`, a `#[cfg(test)]` child module)
  - `./src/harness_rsi/single_harness/` → promotion loops
    - `promotion.rs`, `config.rs`, `state.rs`, `iterative.rs` (`SessionCore` + `IterativeSingleHarness`), `continuous.rs`
  - `./src/harness_rsi/evaluator/` → evaluation records, scoring, benchmark runtimes
    - `case.rs`, `scoring.rs`, `runner.rs`, `swebench.rs`, `terminal_bench.rs`, `metrics.rs`
  - `./src/harness_rsi/member_optimizer/` → member optimisation pass
    - `config.rs`, `selection.rs`, `plan.rs`, `verify.rs`, `optimizer.rs`
  - `./src/harness_rsi/progress.rs` → `OptimizationProgressEvent` and the progress callback
- `./src/artifact_rsi/` → artifact-side RSI
  - `./src/artifact_rsi/program_opt/` → `ProgramOptConfig` and the PUCT tree (`config.rs`, `tree.rs`)
  - `./src/artifact_rsi/paper_opt/` → `config.rs`, `scoring.rs`, `tree.rs`, `proposal.rs`, `orchestrator.rs`
- `./src/deferred/` → documented `Error::Unsupported` stubs (`agentdescent.rs`, `evaluators.rs`, `runtime.rs`)
- `./tests/` → offline integration suites (`tests/CONTEXT.md`)

## Test Strategy

- `tests/config.rs` — Python defaults, YAML/JSON parsing with unknown fields and missing keys, `validate()` rejections, derived path properties, `load_or_default` bootstrap fallback.
- `tests/orchestrator.rs` — a scripted fake `HarnessRunner` drives: one accepted improvement across iterations, elite retention, convergence stop after `patience` flat iterations, target-score stop, budget stop (zero cost limit), `stop()` → cancelled iteration, and event-stream ordering.
- `tests/checkpoint.rs` — state save/load round-trip through `InMemoryCheckpointStore` and `KvCheckpointStore` over `openjiuwen_foundation::store::kv::InMemoryKvStore`/`LocalFileKvStore` (tempdir), plus `resume()` continuing at the persisted iteration.
- `tests/evolution.rs` — `EvaluationReport` aggregation (total/passed, ungraded never averaged, weighted score), `ElitistSelection` determinism and tie-breaking, `DeterministicMutator` stable output/ids, duplicate-id skip.
- `tests/program_opt.rs` — hand-computed PUCT scores (`c_puct = 1.0`, uniform priors), rank-score edges (`N = 1 → 0.5`, unscored `-inf`), ancestor visit increments, candidate limit rejection, `best()` tie-break, config validation, serde round-trip.
- `tests/single_harness_promotion.rs` — batch-filter boundaries at `min_improvement + EPSILON`, strict target improvement with missing entries, `mean_improvement`, every `stage_one_gate` reason literal, `threshold_saturated`, the four `full_evaluation_decision` outcomes including retention/EPSILON boundaries, `case_scores` omission of ungraded/errored cases, fingerprint stability, state YAML round-trip, idempotency keys, and `resume_from` mismatch errors.
- `tests/single_harness_loops.rs` — a scripted `Evaluator` plus `DeterministicMutator` drive both policies offline: promotion/rejection, iteration/target/cancellation stops, engineering-failure recording, resume without replaying a completed batch, byte-identical journals, the legacy override, continuous per-batch promotion, the `max_trials` cap, protected-case retention, and per-batch retries.
- `tests/evaluator_scoring.rs` — case/dataset parsing (id and input aliases, nested `swebench`, unknown keys, JSONL, duplicate ids, malformed index), verdict parsing (fenced/prose, `NaN`/`Infinity`, duplicate ids, non-object root, unknown status), hand-computed score math (weighted mean, ceiling vs subtract, inclusive threshold, missing behaviour → 0, binary raw score, empty contract), failure classification, requirement rows, transient markers, metrics, and the `CaseRunner`/`CandidateCaseRunner`/`CommandCaseBackend` bridge.
- `tests/evaluator_benchmarks.rs` — SWE-bench argv (every flag and the `+600 s` deadline), all four report classifications, every failure literal plus the infrastructure-pattern matcher, `evaluate_case` over `FakeCommandRunner` for resolved/unresolved/error/missing-report, terminal-bench argv, all three `parse_results` shapes and excerpts, and both `Evaluator` adapters.
- `tests/paper_opt.rs` — hand-computed rubric aggregation, weight validation, half-even rounding, dispersion flags, every adoption-gate branch, tree ids/path/frontier/tie-break, orchestration with a scripted proposer/scorer (promotion, rejection, iteration and `min_composite` stops, cancellation, scoring-error pruning, determinism, state round-trip), and the template proposer.
- `tests/member_optimizer.rs` — selection formula/order/beam/confidence filter, wave layering and cycle/unknown/duplicate errors, verification checks (success, missing outcome, undeclared write path, escape, missing target, repairable), and the end-to-end pass (`NoTargets`, `NoActions`, `Success`, `PartialSuccess`, `PlanningRejected`, one repair round, artifact round-trip).
- `tests/ledger.rs` — token alias matrix, `usage_snapshot`, JSONL round trip and replay with no double counting, malformed line → `Error::session`, task-id mismatch, pending-call delivery checks, and per-model/per-stage aggregation.
- `tests/progress.rs` — progress-event serde round trip, defaults, `from_bool`, callback delivery and the `None` callback no-op.
- `tests/runtime_command.rs` — real `ProcessCommandRunner` exit/merge/timeout/cancellation/missing-program behaviour (Unix-gated where needed) and `FakeCommandRunner` matching/recording.
- `tests/runtime_git.rs` — porcelain status buckets, `add_paths` empty, branch/commit failures as data, `has_commits_against_base`, push escalation, cherry-pick abort, `slugify_topic`, credential scrubbing and auth env.
- `tests/runtime_ci.rs` — gate config parsing (fail-closed), command normalisation, changed-line ranges, finding filter, failure-output sanitising, gate outcomes, action selection and `__CI_CHECK_FAIL__`.
- `tests/runtime_python.rs` — JSON/JUnit report parsing (attribute order, self-closing, entities, malformed root), `pytest_args`, venv idempotency, pip failure text, `run_pytest` over a fake runner with a written report, and the subprocess evaluator's Tagged/Files success, failed exit and cleanup.
- `tests/deferred.rs` — the three remaining stubs (`EvolutionEngine`, `LlmJudgeEvaluator`, `RuntimeExtensionLoader`) return `ErrorKind::Unsupported` with a non-empty surface-specific reason; `EvolutionPolicy` round-trips.
- `tests/crate_surface.rs` — every implemented flat re-export resolves and works offline (command/git/CI/Python/subprocess, ledger, evaluator and benchmark runtimes, paper/member optimisers, single-harness types, core protocol).
- Unit tests beside modules for budget warning thresholds, stop-condition ordering, rounding, config parsing, and small helpers.
- All tests offline and deterministic; no network, no Docker, no real model, no `#[ignore]`.
- Definition of done: `cargo fmt -p openjiuwen-rsi`, `cargo test -p openjiuwen-rsi --offline`, `cargo clippy -p openjiuwen-rsi --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-rsi --no-deps` all clean.

## Design Decisions

- Python's `AutoHarnessOrchestrator` has no explicit lifecycle (`run_session_stream` + `cancel`); the Rust orchestrator exposes `start`/`run_iteration`/`finish`/`stream` so iterations are testable and resumable, while `run()`/`stream()` preserve the single-call session shape.
- Python's harness dependency (`create_deep_agent`, `DeepAgent`, rails) is inverted into the `HarnessRunner` trait: RSI never constructs a harness, so the crate stays testable offline and the harness crate stays free of RSI knowledge.
- Candidate bodies are text (config YAML or program source) rather than Python objects: the same `Candidate` type serves harness optimisation and program search.
- Evaluation scores are `Option<f64>` per case so "ungraded" is representable and never folded into an average (Python `test_evaluator.py` behaviour).
- `SessionState`/`SingleHarnessState`/`PaperTaskState` are the checkpoint units; resume is a full state restore, plus idempotency keys (`batch_key`/`attempt_key`) for the batch-level loops.
- Process execution is one seam (`CommandRunner`) with exit codes as data and timeouts as outcomes: every git/CI/pytest/subprocess layer composes it, so a scripted fake drives all of them and no library code spawns a process directly.
- Git prompts are suppressed (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=""`, `GCM_INTERACTIVE=never`) and credentials/tokens are scrubbed from any output that can reach logs or reports.
- The CI gate is fail-closed: a malformed gate config yields no gates (a failed report), findings are scoped to changed lines for display only, and `__CI_CHECK_FAIL__` in a successful command's output still fails the gate.
- The evaluator package is split at the model boundary: records, verdict parsing, score/penalty arithmetic, benchmark argv/report parsing and metrics are local; the solver (`CaseExecutionBackend`) and the judge model stay injected. An unavailable or unusable verdict is an ungraded result, never a pass.
- `score_verdict` sums triggered forbidden penalties in both modes (Python's legacy `ceiling` caps with the largest single penalty); with Python's single default penalty of `0.3` the two agree, and one rule keeps ceiling/subtract symmetric.
- `verdict_from_case` is the only place that expands a reference into a contract (explicit behaviours, `rubric_NNN`, a criterion `reference_answer`), so the judge is graded against trusted dataset weights rather than its own.
- The benchmark evaluators never provision containers: they build the official argv, run it through the command seam, and read the report/results file the harness wrote. A missing results file is a case-level error with `score: None`.
- The paper optimiser keeps Python's greedy-frontier discipline: one frontier, a strict composite improvement to adopt (ties rejected), and the gate order `pipeline_<status>` → `paper_generation` → `scoring_error` → score comparison. The existing `PuctTree` ranks candidates within a round rather than a new search structure.
- The member optimiser ports Python's exact selection formula (`severity*10 + confidence*5 + eligible/5`) and Kahn wave scheduling, and keeps verification static (action outcome, declared write paths, target existence) so a candidate can never be reported as verified without evidence.
- The ledger is explicit and per-run (`RunLedger` + injected `LedgerStore`) instead of Python's ContextVar/global observer: the same replay/dedupe guarantees hold, concurrent runs cannot charge each other, and `check_delivery` surfaces interrupted calls instead of hiding them.
- `task.rs` is data-only: the auto-harness loop consumes `IterationRecord`, `SessionSummary`, `StopConditions`, `StopDecision`; `OptimizationTask`, `Gap`, `StageResult`, `TaskStatus`, `StageStatus` are kept for Python interface parity.
- Deferred surfaces are one-module-per-family stubs with typed constructors and a single `Result<Value>` method, so call-site shape exists without inventing fake type graphs; the `## Deferred Scope` table is the authoritative gap list.
