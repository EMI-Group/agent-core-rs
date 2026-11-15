# openjiuwen-dev-tools / tune / optimizer / adopt

## Intent

Port the *decision* half of `openjiuwen.dev_tools.tune.optimizer.adopt` (`adopt_optimizer.py`, `utils.py`) — deciding whether a tuned prompt is worth installing — as a deterministic layer over the crate's existing prompt-search seams.
Python's `AdoptOptimizer` rewrites every `LLMCall` of a multi-node workflow in place (conclude each node's job → critique the whole-workflow output → derive a corrected node output → compute a per-node "textual gradient" through a chain of model prompts), and `PartialOptimizer` is the instruction optimizer it uses for one parameter.
None of the mutation machinery is portable: there is no `LLMCall`/`TextualParameter`, no `OptimizeHistory` of traced node input/output pairs, no prompt bank, and no callback protocol.
What is portable and useful is the judgement: a candidate only replaces an installed prompt when it is correct enough, faithful to the objective, and measurably better than what is recorded.
This node states that judgement once, purely, and reuses `DriftJudge`, `ConvergenceDetector`, `PromptMemory`, and `clamp01` instead of re-implementing them.
Everything is deterministic: no clock (`now` is a parameter), no randomness, no environment reads, no sleeps, no network.

## API Surface

- `AdoptConfig` — the thresholds: `min_gain` (`DEFAULT_ADOPT_MIN_GAIN` 0.01), `max_drift` (`DEFAULT_ADOPT_MAX_DRIFT` 0.25), `min_correctness` (`DEFAULT_ADOPT_MIN_CORRECTNESS` 0.5), `min_history` (`DEFAULT_ADOPT_MIN_HISTORY` 1), `require_improvement` (true); `new`/`Default`, five `with_*` builders, `validate`.
- `AdoptPolicy` — `new`, `with_config`, `config`, and the pure, synchronous `decide(candidate_reward, correctness, drift, history) -> AdoptDecision`; `Copy`, `Default`, `PartialEq`.
- `AdoptReason` — `FirstCandidate`, `Improved { gain }`, `NoImprovement { gain }`, `Drift { deviation }`, `InsufficientCorrectness { score }`, `InsufficientHistory { rounds }`; `Display`, serde, `Default` = `FirstCandidate`.
- `AdoptDecision` — `adopted` + `reason`, with `new`/`Default`, `adopted()`, `reason()`, `gain()` (`0.0` outside the two comparison reasons), `describe()` (`"adopted: …"` / `"rejected: …"`), serde.
- `AdoptMemory` — `new` (over `NullPromptMemory`), `with_memory(Arc<dyn PromptMemory>)`, `memory()`, `record_adopted(record, now) -> PromptRecord`, `record_rejected(record, now) -> PromptRecord`, `adopted()`, `rejected()`, `previous_best(objective) -> Option<f64>`, `pending(threshold)`; `Debug` reports the two decision counts.
- `AdoptService` — `new(AdoptPolicy)`/`Default`; `with_drift_judge(Arc<dyn DriftJudge>)`, `with_convergence(ConvergenceDetector)`, `with_memory(Arc<dyn PromptMemory>)`, `with_characteristics`; readers `policy()`, `config()`, `drift_judge()`, `convergence()`, `memory()`, `characteristics()`; the single `async fn decide(task, candidate, reward, case_scores, history, now) -> Result<AdoptOutcome>`.
- `AdoptOutcome` — `decision`, `drift`, `correctness`, `convergence: Option<ConvergenceState>`, `record: Option<PromptRecord>`, plus `adopted()`; serde with per-field defaults.
- `select_bad_cases(&[EvaluatedCase], limit)` / `select_good_cases(&[EvaluatedCase], limit)` — Python's exact-score split (`0.0` / `1.0`), ordered by case id then score and truncated; `DEFAULT_BAD_CASES_SAMPLE_NUM` = 5.
- `describe_gradient_context(&[EvaluatedCase]) -> String` — the deterministic textual-gradient artifact: one six-line block per case (id, score, input, answer, expected, feedback), `no failing cases` when empty.
- `render_gradient(task, candidate, bad) -> String` — the three-line header (`objective:` from `PromptTaskSpec::characteristics`, `candidate:` id or `(unnamed candidate)`, `gradient:`) plus `describe_gradient_context`.
- `GradientPolicy` — `async fn gradient(&self, task, candidate, bad) -> Result<String>`, with `DeterministicGradientPolicy` (`new`/`Default`) as the default implementation.
- `mod.rs` is module docs, five private `mod` declarations (`cases`, `gradient`, `memory`, `policy`, `service`), the two `#[cfg(test)]` modules (`support`, `tests`), and the `pub use` re-exports.

## Constraints

- Only `src/tune/optimizer/adopt/` is editable here; `../optimizer.rs` keeps `mod adopt;` + `pub use adopt::*;` untouched.
- The family is reachable inside the crate through that glob, but `src/tune/mod.rs` lists its re-exports explicitly, so the adopt surface is not publicly reachable until that list includes it.
- That gap is what the module-level `#![allow(dead_code, unreachable_pub)]` and the per-statement `#[allow(unused_imports)]` on each `pub use` cover; no other lint is allowed anywhere in the subtree, and all of them become inert once `src/tune/mod.rs` re-exports the family.
- The module example is a `text` sketch for the same reason: it compiles and passes as a compiled doctest once the tune root re-exports the family.
- `../optimizer.rs`'s `pub use adopt::*;` is reported unused for the same reason; that file is outside this node.
- No new dependencies: serde, serde_json, async-trait, tokio (tests), tracing, and the existing prompt-search seams only.
- Determinism is a hard rule: no clock reads, no `random`, no environment reads, no sleeps, no network; `now` is always a caller-supplied parameter.
- Errors come from `openjiuwen_common::error` only, and only `Error::validation` is constructed (in `AdoptConfig::validate`); nothing here returns `Error::Unsupported`.
- `AdoptMemory` never fails the caller: a failing `PromptMemory::store` is logged with `tracing::warn!` and the adopted record is still reported, matching the prompt-search memory convention.
- Serde: `rename_all = "snake_case"`, `default` on every struct so unknown/missing fields are tolerated; `AdoptReason` is internally tagged on `kind`.
- No `unsafe`, no printing, no panics on user input; mutexes recover from poisoning instead of panicking.
- Files stay well under 1000 lines; one responsibility per file; the shared test fakes live in `support.rs`, never copied per suite.

## Design Decisions

- The policy is a pure value with no I/O and no `async`, so a decision is reproducible from its four inputs alone; the `async` service is the only place a seam can be awaited.
- The evaluation order is correctness gate → drift gate → history → improvement, and the first gate that fires decides: an incorrect prompt cannot win on reward, a drifted prompt cannot win on score, and a thin history cannot justify an improvement claim.
- A history with no finite value is *no history*: the candidate is adopted as `FirstCandidate` rather than being rejected, which is what makes the first adoption of an objective possible.
- `min_history` is checked only for a non-empty comparable history, so its default of `1` means "any recorded reward", and `FirstCandidate` keeps priority over `InsufficientHistory`.
- The comparison baseline is the *largest* finite value in the history, not the last one, so a candidate cannot win by beating a regression.
- The gain threshold is strictly greater (`gain > min_gain`), so a gain exactly at `min_gain` is `NoImprovement`.
- `require_improvement: false` still reports `Improved`/`NoImprovement` truthfully; only `adopted` changes, so the caller can see that nothing improved even when adoption was forced.
- Non-finite inputs are normalised rather than rejected: a non-finite reward or correctness counts as `0.0` (it cannot pass), a non-finite drift counts as `1.0` (it fails the drift gate), and finite values are clamped into `0.0..=1.0` before comparison.
- `validate` names the field for every rejection and adds a dedicated message for a `min_gain`/`max_drift` above `1.0`, because such a threshold can never be met and would silently disable the check it configures.
- The constant names carry an `ADOPT_` infix (`DEFAULT_ADOPT_MIN_GAIN`, …) so the glob re-export cannot collide with `search_config`'s `DEFAULT_MIN_CORRECTNESS`.
- `AdoptMemory` persists adopted records marked applied at `now` and never persists rejected ones: a prompt that failed the policy must not become history a later run reasons from.
- `previous_best` considers the persistent store's best record for the objective *and* this run's own adoptions, so a run over the default `NullPromptMemory` still supplies a baseline for its second decision; the service reads it before archiving, so `baseline_reward` is genuinely the prior best.
- Convergence is reporting only: the detector runs over the caller's history but never changes the verdict, and no detector means `convergence: None`.
- The drift judge is skipped with `drift = 0.0` when `max_drift >= 1.0`, because no deviation can then fail the gate and no measurement is needed.
- Correctness is the mean of the finite case scores (`0.0` for none), clamped, so a caller-supplied out-of-range score cannot smuggle a candidate past the floor.
- Archived records carry `reward`/`correctness`/`drift`/`gain` as metrics, the decision sentence as the single observation, the task metadata plus `candidate_id`/`rationale`, and a deterministic `adopt-<id>` record id derived from the candidate.
- The gradient is behind a trait seam rather than a model call: the deterministic default renders the failing cases as the artifact, and a caller who wants Python's model-written gradient implements the trait outside this module.
- Python's `random.sample` selection becomes a case-id ordering, and its exact-score split is preserved: a partially scored case is evidence of neither failure nor success.
- Python's known-broken paths are not reproduced: no un-awaited model calls, no `ModelWithRetry` that returns an exception object, no in-place parameter mutation, and no retry/parallelism constants that nothing honours.

## Test Strategy

- `policy.rs` covers the defaults and builders, every validation rejection (non-finite, out of range, impossible `min_gain`/`max_drift`, each naming its field), config JSON round-trip with per-field defaulting, all six decision branches, the strict threshold, negative gains, the gate ordering (correctness beats drift beats improvement), `FirstCandidate` for empty/NaN/infinite history, `InsufficientHistory`, `require_improvement` off, determinism, non-finite normalisation, and the decision JSON contract.
- `memory.rs` covers the null default (records the run, no store), adopted records persisted and marked applied, `pending` over a pre-seeded store, rejected records kept out of the store, `created_at` filling, decision order, `previous_best` over store and run, a failing store being non-fatal, and the `Debug` counts.
- `cases.rs` covers the exact-score split, case-id ordering with input-order independence, the limit (zero, one, exact, over), and the gradient rendering (empty, every line, blank feedback, empty maps, determinism).
- `gradient.rs` pins the rendered header and context exactly, the unnamed candidate and empty case list, and the fact that the rendered case order follows the slice given.
- `service.rs` covers the correctness mean, the deterministic record ids, and the `AdoptOutcome` default.
- `tests.rs` covers the service end to end: default adopt and archive, a baseline from `previous_best`, a rejected candidate persisting nothing, an injected fixed drift judge, the `max_drift >= 1.0` skip, the correctness gate through case scores, convergence with and without a detector, the characteristics override and task metadata, a failing store, repeated-run determinism, the readers, the whole family composed over one run, and `AdoptOutcome` JSON round-trip/defaulting.
- `support.rs` holds the two shared fakes (`RecordingMemory`, `FixedDriftJudge`) plus the `evaluated` case builder so no suite duplicates them.
- Everything is offline: no model, no network, no temp directory, no clock; latencies and timestamps are constants.

## Deferred

- Python's node-level credit assignment (conclude each node's job, derive corrected node outputs, per-node prompt gradients) has no counterpart: there is no `LLMCall`, no `TextualParameter`, and no traced node input/output history to attribute to.
- The prompt bank of `utils.py` (`OUTPUT_CHANGE_*`, `DEEP_OUTPUT_ANALYSIS_*`, `EXPECTED_OUTPUT_*`, `GRADIENT_GENERATE_*`, `GRADIENT_REDUCE_*`, `CONCLUDE_NODE_*`, and the unused `CONCLUDE_AGENT_*`/`PROMPT_UPDATE_*`) is not ported; `render_gradient` is the deterministic artifact and `GradientPolicy` the seam a model-backed replacement would implement.
- `AdoptOptimizer`/`PartialOptimizer` themselves are not ported as optimizers: the crate's `InstructionOptimizer`/`JointOptimizer`/`PromptSearchOptimizer` remain the ways to rewrite a prompt, and this node only decides whether to keep the result.
- `ModelWithRetry` (five attempts, and returning an error object instead of raising) is not reproduced; a caller that wants retries wraps its own `Model`.
- `DEFAULT_PARALLEL_NUM`/`DEFAULT_MODEL_RETRY_NUM` are not exposed: nothing here runs in parallel or retries.
- The frozen-prompt flags and the callback restoration around `set_optimizer_callback` have no counterpart, because nothing here mutates a parameter.

## Routing Table

- `./mod.rs` → module docs, the private `mod` declarations, the `pub use` surface, and the usage sketch
- `./policy.rs` → `AdoptConfig` (thresholds, validation), `AdoptReason`, `AdoptDecision`, and `AdoptPolicy::decide`
- `./memory.rs` → `AdoptMemory`: adopted/rejected bookkeeping, `previous_best`, `pending`, store-failure handling
- `./service.rs` → `AdoptService` (measure → gate → decide → archive), `AdoptOutcome`, and the record/metrics/metadata helpers
- `./cases.rs` → `select_bad_cases`/`select_good_cases`, `describe_gradient_context`, `DEFAULT_BAD_CASES_SAMPLE_NUM`
- `./gradient.rs` → `GradientPolicy`, `DeterministicGradientPolicy`, `render_gradient`
- `./support.rs` → the shared test fakes (`RecordingMemory`, `FixedDriftJudge`, `evaluated`)
- `./tests.rs` → the cross-module service/outcome suite
- `..` → the optimizer facade that declares `mod adopt;` and re-exports it (`../optimizer.rs`, read-only here)
- `../prompt_search/` → `DriftJudge`, `ConvergenceDetector`, `PromptMemory`/`PromptRecord`, `PromptCandidate`/`PromptTaskSpec`, `clamp01`, `candidate_id` (sibling — read-only here)
- `../../base.rs` → `Case`/`EvaluatedCase` (outside this node — read-only)
