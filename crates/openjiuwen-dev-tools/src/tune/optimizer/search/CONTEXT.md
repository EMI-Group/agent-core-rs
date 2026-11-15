# openjiuwen-dev-tools / tune / optimizer / search

## Intent

Ports Python's `openjiuwen.dev_tools.tune.optimizer.prompt_search.optimizer` loop: a bounded, multi-round prompt search that drives the domain vocabulary of `../../prompt_search/` (task, candidate, execution, reward, convergence, drift, memory, history).
One round proposes a candidate pool, executes every candidate against every case through the `CaseEvaluator` seam, scores it through `CompositeReward`, records the round, and feeds the compressed lessons into the next round's rewrite request.
The loop is deterministic and offline by default: elitism, the shared mutation bank, ordered execution, and hash-derived ids make repeated runs byte-identical.
The module is private to `crate::tune`; `PromptSearchOptimizer` and `PromptSearchConfig` are re-exported from the `tune` module root.

## API Surface

- `PromptSearchOptimizer` — `new`/`Default` (default `OptimizerConfig` and `PromptSearchConfig`, no model, deterministic evaluator) plus the builders `with_config`, `with_search_config`, `with_model`, `with_evaluator`, `with_reward`, `with_drift_judge`, `with_memory`, `with_run_log`, `with_timestamp`, `with_adopt_policy`, `with_run_id_seed`; readers `config()`, `search_config()`, `has_model()`, `has_run_log()`, `has_adopt_policy()`, `timestamp()`.
- `with_run_log(Arc<Mutex<dyn RunLog>>)` receives exactly one `RunLogRecord` per completed round; the caller keeps its own handle (an `InMemoryRunLog` offline) and reads back what the run wrote. Without a log nothing is built, and a poisoned mutex or a rejecting sink is logged with `tracing::warn!` and swallowed.
- `with_timestamp(f64)` is the caller-supplied timestamp stamped on every record (`0.0` by default, never a clock read), and `with_adopt_policy(AdoptPolicy)` adds that round's `AdoptDecision` to its record.
- The per-round record contract: `iteration` is the 1-based round number, `candidate` is the round's winner, `candidate_prompts` the other candidates in evaluation order, `scores` every candidate id with its reward, `reward`/`reward_components`/`reward_notes` the winner's `RewardBreakdown`, `best_score` the run's best after the round, `drift` the winner's deviation, `convergence` the `ConvergenceState` when the detector fired this round (else `None`), `converged`/`convergence_reason` the round's stopping verdict, `adopt` the policy verdict (`None` without a policy), `selected` whether the round moved the elite, `timestamp`/`run_id` the caller's values, and `observation` the winner's observations joined with `"; "` (`None` when empty).
- `PromptSearchOptimizer::run_id(seed, task)` — `{seed:016x}-{hash:08x}`, the suffix a 32-bit FNV-1a hash of the objective, the base prompt, and the case count, so equal tasks share an id and different tasks do not.
- `PromptSearchOptimizer::optimize(prompt, cases)` — the convenience entry point: the prompt is both the task objective and the base prompt, and every case is visible.
- `PromptSearchOptimizer::optimize_task(&PromptTaskSpec)` — keeps the objective, constraints, base prompt, hidden cases, and metadata; validates a non-blank objective, a non-empty case list, and that every case converts to `Case`.
- `OptimizationResult` — `best_prompt`, `score`, `iterations` (rounds actually run, i.e. `records.len()`), serde-defaulted `converged`, `convergence_reason` (`None` when the budget ended the run), empty `token_usage`, `run_id`, and the per-round `records` (`Vec<IterationRecord>`).
- The round budget is `min(PromptSearchConfig::max_iterations, OptimizerConfig::iterations)` — 3 rounds with the defaults (`min(6, 3)`).
- Round seeding is elitist: round 1 starts from the task base prompt (falling back to the objective when it is blank), later rounds from the best prompt seen so far.
- The round pool is the seed, the round-1 warm-start prompts from `PromptMemory`, every deterministic mutation of the round seed, and at most one model rewrite; missing `{{placeholder}}`s are repaired against the round seed, the first occurrence of a prompt wins, and the pool is truncated to `PromptSearchConfig::candidate_prompts`.
- Candidate ids come from `prompt_search::candidate_id` over a run-wide counter, so they are unique within a run and reproducible across runs.
- The model rewrite receives the round seed, the worst cases, `PromptSearchConfig::policy_temperature`, and the round's policy context (the compressed lessons and the warm-start prompts).
- `execute.rs` provides the round mechanics: `execute_candidate`/`execute_candidates` (batched by `OptimizerConfig::num_parallel`), `drift_scores`, `warm_start`, `policy_context`, `base_prompt`, `answer_text`, `best_index`, `observations`, `metrics_of`, `record_metadata`, and `CandidateOutcome`.
- `execute_candidate` writes one `CaseResult` per case keyed `case-{index}`; a failing evaluation becomes that result's `error` and the run continues, and `Execution::error` is set only when every case failed.
- `execute_candidate` evaluates each case through `CaseEvaluator::evaluate_run`, so the case's measured `RunTelemetry` is stored on its `CaseResult` and merged into `Execution::telemetry`: the execution's latency and token totals are the sum over its cases (merged in case order through `RunTelemetry::merge`).
- The legacy `Execution::latency_ms`/`total_tokens` fields stay for wire compatibility: `execute_candidate` fills them from the merged telemetry only when it was measured, so an evaluator that reports nothing leaves them at `0.0`/`0` rather than fabricating a measurement.
- An errored case contributes no telemetry, so a failed case leaves both its `CaseResult::telemetry` and the merge untouched.
- `CandidateOutcome.evaluated` keeps the successfully evaluated cases, which are the evidence a rewrite is shown (`worst_cases`, at most `WORST_CASE_LIMIT` = 5).
- `warm_start` takes the `WARM_START_LIMIT` = 3 most similar prior records, sorted by reward descending then prompt ascending.
- `best_index` picks the highest score and keeps the earlier candidate on a tie.
- `observations` is the candidate rationale plus the reward notes plus the strongest and weakest raw metric (`<name> (<value>)`); `metrics_of` renders raw components as record metrics; `record_metadata` writes `iterations` plus the task metadata.
- `prompt_search::PromptTaskCase` carries no id, so case identities in a round are positional (`case-{index}`).
- `run_log_record.rs` builds the record: `RoundRecord` (the round's scoring facts) and its `build()` renderer, `adopt_decision` (the winner's correctness is its visible-case mean score, and the history is the round-best rewards of the *preceding* rounds), and `emit` (a poisoned `Mutex` or a sink error warns and is swallowed).
- The record is written after the round's convergence check, so it always carries that round's verdict; the round's `selected` flag is `tracker.score > previous`, which makes round 1 (which establishes the elite's floor) always selected and every later round selected only on a strictly better score.
- Constants: `WARM_START_LIMIT`, `WORST_CASE_LIMIT`, `FNV32_OFFSET`, `FNV32_PRIME`.

## Constraints

- No `unsafe`, no panics on user input, no printing; failures are either returned as `Error` or recorded in `CaseResult.error`/`Execution.error`.
- Errors come from `openjiuwen_common::error` only: `Error::validation` for a blank objective or prompt, an empty case list, a non-convertible case, or an out-of-range `OptimizerConfig`/`PromptSearchConfig`; nothing in this module returns `Error::Unsupported`.
- The evaluator, reward, drift judge, and memory are injected seams with deterministic defaults (`default_evaluator`, `CompositeReward::default().with_min_correctness(..).with_drift_penalty(..)`, `LexicalDriftJudge`, `NullPromptMemory`); nothing reads the environment or constructs a model from configuration.
- Async-first: `optimize`, `optimize_task`, and the round mechanics are `async fn`; the deterministic path awaits only the evaluator seam.
- Determinism: execution results keep input order regardless of completion order, the loop reads no clock and no random number generator, and every id is derived from the run counter or the run id.
- Only the evaluator seam can supply latency or tokens: a telemetry-free evaluator leaves every `Execution` at `latency_ms = 0.0`/`total_tokens = 0` (its latency/token-usage/cost reward components stay constant), while a measuring evaluator feeds them per case; `OptimizationResult::token_usage` stays empty either way.
- Memory persistence is opt-in: without `with_memory` nothing is read or written, and a failing `store` is logged with `tracing::warn!` and never fails the run.
- Run logging is opt-in and inert: without `with_run_log` no record is built, and a logged run returns an `OptimizationResult` identical to an unlogged one (the run-log suite asserts it with `PartialEq`).
- The adoption policy is reporting only: it never gates a candidate or changes the elite, it is skipped entirely when no run log is attached, and its correctness input is the winner's visible-case mean (the same mean `CorrectnessReward` scores).
- Shared vocabulary only: the module composes types from `../../prompt_search/`, the tunables of `../search_config.rs`, and the shared helpers of `../suggestion.rs`, and duplicates none of them.
- Files stay small and single-purpose: the loop and its values in `mod.rs`, the round mechanics in `execute.rs`, the record builder in `run_log_record.rs`, the suite in `tests.rs`, and the run-log suite in `run_log_tests.rs`.

## Design Decisions

- The policy is deterministic by design: the shared mutation bank plus one optional model rewrite per round. Python's `LLMPromptPolicy` multi-candidate JSON generation is not ported, so a round's variety comes from the bank and elitism guarantees the result is never worse than the starting prompt.
- The environment is the caller's `CaseEvaluator`, not Python's `LLMEnvironment`: the same loop runs against `DeterministicEvaluator` offline or a model evaluator online, and per-case evaluator failures are data (an errored `CaseResult`) instead of aborting a round.
- Elitism and the budget coupling are explicit: later rounds seed from the elite, `iterations` reports rounds actually run, and an early stop is visible as a smaller number plus a `convergence_reason`.
- The seed runs first and is also the evidence source for the rewrite, so the round always has the seed's score to beat and the rewrite is grounded in its worst cases.
- Stopping is checked after every recorded round: the `ConvergenceDetector` first (target reward, then plateau, then population variance), then the shared `OptimizerConfig::early_stop_score` on the best score.
- Drift is skipped entirely when `PromptSearchConfig::drift_penalty == 0.0` (all deviations reported as `0.0`, Python's early return), and the judge is otherwise injected or `LexicalDriftJudge`.
- The composite reward is rebuilt from the search config (`min_correctness`, `drift_penalty`) unless one is injected, so the gate and penalty live in one place.
- The policy context is optional: `policy_context` returns `None` when there are neither lessons nor warm-start prompts, which keeps a first-round rewrite request byte-identical to the other optimizers' requests (`model_rewrite_with` with no context).
- Each round appends an `IterationRecord` and a compressed `HistoryEntry` (`HistoryCompressor::default()` over the whole history), and the elite is absorbed only on a strictly better score.
- Persistence stores one `PromptRecord` (`{run_id}-best`) carrying characteristics, reward metrics, observations, metadata (`iterations` plus the task metadata), and the prior `baseline_reward` when memory held one for the objective.
- The `run_id` hashes the objective, base prompt, and case count only, so a seeded run is reproducible for a given task shape, and `with_run_id_seed` changes only the prefix.
- A blank `PromptTaskSpec::base_prompt` falls back to the objective, which keeps `optimize` (objective == prompt) and `optimize_task` on one code path.
- Adaptation note: the drift judge may call a model, which is why it stays an async trait while the rest of the vocabulary is synchronous.

## Test Strategy

- `tests.rs` is the in-tree suite, entirely offline and deterministic: `ScriptedModel` for model paths, fixed cases and ids, no network, no environment reads, no wall-clock dependence.
- Round mechanics: `runs_multiple_rounds_and_records_each_one` (one `IterationRecord` per executed round with its candidates and round best), `the_run_is_deterministic` (repeat runs return the same prompt, score, and records), `keeps_placeholders_and_the_seed_contract`, and `the_candidate_budget_bounds_every_round`.
- Input validation: `rejects_invalid_inputs` (blank prompt/objective, empty cases, out-of-range configuration) and `optimize_task_validates_its_inputs`.
- Model path: `a_model_rewrite_can_win_a_round`, `tool_declarations_reach_the_rewrite_prompt`, `the_policy_temperature_and_lessons_reach_the_rewrite_prompt`, and `an_unusable_model_answer_keeps_the_bank`.
- Elitism and scoring: `the_seed_wins_when_nothing_improves` and `hidden_cases_punish_an_overfitted_candidate`.
- Stopping: `converges_on_a_target_reward`, `converges_on_a_flat_plateau`, `stops_on_the_early_stop_score`, and `a_budget_bound_run_reports_no_convergence`.
- Seams: `warm_starts_and_persists_through_the_memory` (warm start plus one persisted record) and `an_injected_drift_judge_penalises_a_marker_prompt`.
- Task shape: `optimize_task_keeps_constraints_and_metadata` and `a_blank_base_prompt_falls_back_to_the_objective`.
- Telemetry: `each_case_keeps_its_own_telemetry_and_the_execution_sums_them` (per-case `CaseResult::telemetry` plus the merged 15 ms/4+5-token execution, the legacy fields and `measured_*` readers) and `execute_candidates_carries_telemetry_per_candidate` (per-candidate telemetry in input order), both driven by a stub evaluator that implements only `evaluate_run`.
- Telemetry absence: `a_telemetry_less_evaluator_keeps_the_legacy_fields_zero` (empty `Execution::telemetry`, `latency_ms == 0.0`, `total_tokens == 0`, and unchanged results/scores/error for an `evaluate`-only stub).
- Telemetry and the decision: `telemetry_does_not_change_the_chosen_prompt_or_score` (a cost-carrying stub, whose component has zero reward weight, yields the same prompt, score, round count, and iterations, and a latency/token-carrying stub still picks the same prompt).
- Readers: `debug_reports_the_model_choice` (including the run-log, adoption, and timestamp fields) and `run_id_is_deterministic_and_task_sensitive`.
- Run log (`run_log_tests.rs`, over a shared `InMemoryRunLog` and the fixed-score evaluator): `one_record_per_round_with_ascending_indices` (record count == `result.iterations`, ascending `iteration`, non-blank candidates, the caller's timestamp on every record, the run id, and the last record's `best_score` == `result.score`), `every_record_mirrors_its_round` (scores, winner, `candidate_prompts` order, reward components, observation), `selected_marks_the_rounds_that_moved_the_elite`, `only_the_stopping_round_reports_convergence`, and `repeated_runs_produce_equal_records`.
- Run log and adoption: `the_adoption_verdict_is_absent_without_a_policy`, `a_policy_that_demands_a_gain_rejects_every_later_round` (`FirstCandidate` in round 1, `NoImprovement` afterwards, and a result identical to the unlogged run), and `the_builders_and_readers_report_their_configuration`.
- Run log resilience: `a_rejecting_sink_does_not_fail_the_run` (an `impl RunLog` returning `Error::validation`), `a_poisoned_run_log_does_not_fail_the_run` (a mutex poisoned under `catch_unwind`), and `a_logged_run_matches_a_logless_run`.

## Routing Table

- `./mod.rs` → `PromptSearchOptimizer` (builders, readers, `run_id`, `optimize`/`optimize_task`), the `run`/`prepare`/`run_round`/`round_candidates` loop, the `Setup`/`Tracker`/`RoundInput`/`Collaborators` values, `record_round`, `stopping_reason`, `Debug`, and the persistence step
- `./execute.rs` → per-candidate execution through `CaseEvaluator::evaluate_run` with the per-case telemetry merge, `num_parallel` batches, drift gathering, warm start, policy context, answer/observation/metric/metadata helpers
- `./tests.rs` → the in-tree prompt-search loop suite
- `./run_log_record.rs` → `RoundRecord`/`build` (the per-round record) and `adopt_decision`/`emit` (the adoption verdict and the never-fail sink write)
- `./run_log_tests.rs` → the run-log suite (record-per-round contract, determinism, adoption verdict, resilient sinks), included as `run_log_tests`
- `../../prompt_search/` → the domain vocabulary this loop drives (models, reward, convergence, drift, memory, history, and the `RunLog`/`RunLogRecord` sink); read-only here
- `../search_config.rs` → `PromptSearchConfig` and its `DEFAULT_*` tunables (rounds, candidates, convergence, drift, gate, temperature)
- `../suggestion.rs` → `Mutator`, `ensure_placeholders`, `worst_cases`, `validate_inputs`, `default_evaluator`, `describe_model` (shared with the other optimizers)
- `../config.rs` → `OptimizerConfig` (round cap, parallelism, early stop)
- `../../evaluator/` → `CaseEvaluator` and its traced seam (the environment of a round)
