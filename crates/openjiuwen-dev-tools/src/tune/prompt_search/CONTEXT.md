# openjiuwen-dev-tools / tune / prompt_search

## Intent

Port the prompt-search (RLAF-P) domain layer of `openjiuwen/dev_tools/tune/optimizer/prompt_search/`: the task/execution vocabulary, the composite reward model with its anti-reward-hacking guards, convergence detection, the drift judge, the prompt memory, and the history compressor.
The RLAF-P optimization loop is ported above this node, in `../optimizer/search/` (`PromptSearchOptimizer`), and is built exclusively from these types; Python's `policy`, `environment`, `run_log`, and `service` modules are not ported.
Everything is offline and deterministic — no clocks, environment reads, randomness, or global state — and it is a layer-5 consumer of `openjiuwen_foundation` models plus the shared `crate::tune::base::Case`.

## API Surface

- Models (`models.rs`), every struct `Debug + Clone + PartialEq + Serialize + Deserialize`, `#[serde(rename_all = "snake_case", default)]`, unknown-field tolerant:
  - `PromptTaskCase { input_text, expected_text, hidden }` — `from_text(text, expected)` (delegates to `Case::from_text`, blank text is `Error::validation`), `with_hidden`, `from_case(&Case, hidden)`, `to_case() -> Result<Case>`.
  - `PromptTaskSpec { objective, cases, constraints, base_prompt, metadata }` — `new(objective)` and `with_cases`/`with_constraints`/`with_base_prompt`/`with_metadata`, `characteristics()` (`objective`, plus `constraints: a; b` when constraints exist), `visible_cases()`/`hidden_cases()` (borrowed, declaration order).
  - `PromptCandidate { prompt, candidate_id, rationale }` — `new(prompt, rationale)` ids via `candidate_id(prompt, 0)`, `with_id`, accessors `prompt()`/`candidate_id()`/`rationale()`.
  - `candidate_id(prompt, index)` — 8 lowercase hex chars of the FNV-1a 64 hash over `"{index}\x1f{prompt}"`; replaces Python's `uuid4().hex[:8]` so runs and tests are reproducible.
  - `CaseResult { case_id, output, score, error, hidden, telemetry }` — `new(case_id, output)`, `with_score`, `with_hidden`, `ok()` (no error), `blank()`; `telemetry` is the per-case `crate::tune::tool::RunTelemetry` the caller's evaluator measured and defaults to "nothing reported".
  - `Execution { candidate, latency_ms, total_tokens, error, results, telemetry }` — `new(candidate)`, `push_result`, `latency_s()`, `visible_results()`/`hidden_results()` (borrowed, run order), `combined_output()`/`combined_output_hidden()` (non-empty outputs joined by `\n\n---\n\n`).
    `latency_ms`/`total_tokens` are legacy caller-supplied fields kept for wire compatibility; they only win when no telemetry was measured for the run.
    `measured_latency_s()` is the telemetry latency in ms (else the legacy `latency_ms`) over 1000, `measured_total_tokens()` is `RunTelemetry::total_tokens()` (else the legacy `total_tokens`), and `measured_cost()` is `telemetry.cost` — there is no legacy cost field, so an absent measurement stays absent.
  - `RewardBreakdown { score, components, raw_components, correctness, drift, drift_penalty_applied, gated, notes }` with `Default`, `is_gated()`; `Evaluation { execution, reward }` with `candidate()`/`score()`.
  - `IterationRecord { index, best_score, candidates, scores }` — `index` is the 1-based round number (Python's `iteration`), `new(index)`, `push(candidate, score)`, `best_candidate()` (ties keep the first).
  - `PromptRecord { record_id, prompt, reward, objective, task_characteristics, metrics, observations, metadata, created_at, baseline_reward, applied, applied_at }` — `new(record_id, prompt, reward, objective)`, builders, `gain()` = `reward - baseline_reward.unwrap_or(0.0)`, `is_pending(threshold)` = `!applied && gain > threshold`.
- Reward (`reward.rs`):
  - `clamp01` (non-finite → `0.0`), `min_max_normalize` (empty → empty, constant/spread ≤ `1e-9` → all `1.0`, non-finite treated as `0.0`), `OVERFIT_MARGIN = 0.25`, `OVERFIT_PENALTY = 0.5`.
  - `RewardComponent { name(), is_correctness() -> bool = false, score(execution, task) }`, synchronous because nothing here does I/O.
  - Components: `CorrectnessReward` (`correctness_on(&[CaseResult])` = mean clamped `score` of error-free results), `CompletenessReward { min_chars }` (`new` = 1, `with_min_chars` clamped ≥ 1), `LatencyReward { half_life_s }` (5.0, ≥ `1e-3`), `TokenUsageReward { half_life_tokens }` (2000.0, ≥ 1.0), `CostReward { price_per_mtok, half_life_cost }` (0.0 / 0.01), `StructuredValidationReward::new(validator)`/`::json()`, `CustomReward::new(name, fn)` (clamped).
  - The three cost-of-run components score the measured values: `LatencyReward` uses `measured_latency_s()`, `TokenUsageReward` uses `measured_total_tokens()`, and `CostReward` uses `measured_cost()` when reported, else the token total priced at `price_per_mtok`.
    They stay constant only when the run reports no telemetry (0 ms → `1.0`, 0 tokens → `1.0`, 0 cost → `1.0`), so a measured-but-zero value also scores `1.0` and is indistinguishable from an absent measurement.
  - `default_weights()` = correctness 1.0, completeness 0.3, latency 0.1, token_usage 0.1, cost 0.0.
  - `CompositeReward { components, weights, min_correctness: 0.5, drift_penalty: 0.5, normalize: false, correctness_name: "correctness" }` — `new(Vec<Box<dyn RewardComponent>>, weights)` drops non-positive weights, `default()`/`default_components()`, `with_*` builders, accessors (`component_names`, `weights`, `min_correctness`, `drift_penalty`, `normalize`, `correctness_name`), and `evaluate(&[Execution], &PromptTaskSpec, &BTreeMap<String, f64>) -> Vec<RewardBreakdown>`.
- Convergence (`convergence.rs`): `population_variance` (`/n`, `0.0` below two values), `ConvergenceState { converged, reason, best, moving_average, variance }` (`Default`), `ConvergenceDetector { threshold: 0.01, window: 3, target: None }` — `new(threshold, window, target) -> Result<Self>` (validation), `with_target`, accessors, `check(&[f64]) -> Option<ConvergenceState>`.
- Drift (`drift.rs`): `#[async_trait] DriftJudge { async fn deviation(objective, candidate) -> f64 }`, `NullDriftJudge` (always `0.0`), `LexicalDriftJudge` (default: lowercased word-token sets, `1 - |objective ∩ prompt| / |objective|`, blank prompt `1.0`, blank objective `0.0`), `ModelDriftJudge::new(Arc<dyn Model>)` (one `invoke`, fenced or bare `{"deviation": …}`, clamped, failures → `0.0`, blank prompt → `1.0` without a call).
- Memory (`memory.rs`): `PromptMemory` trait (`&self` methods, interior mutability) with `store`, `search_similar(&PromptTaskSpec, top_k)`, `best_for_objective`, `pending(threshold)`, `mark_applied -> Result<bool>`, `records`; `NullPromptMemory`; `JsonlPromptMemory::new(dir) -> Result<Self>` with `path()` and `skipped_lines()`.
- History (`history.rs`): `HistoryEntry { index, reward, prompt, observations }` (`new(index, reward, prompt)`, `with_observations`, `render()`, `render_with_limit(limit)`), `OptimizationHistory { entries, compressed }` (`add`, `best_reward`, `best_prompt`, `render_recent(limit)`, `render`, `is_empty`), `HistoryCompressor { budget: 1200 }` (`new`, `with_budget`, `budget`, `compress`).
- The module root re-exports every public item; `models`, `reward`, `convergence`, `drift`, `memory`, `history` stay private, as do the `token_set`/`jaccard` helpers shared between memory and drift.
- How the search loop consumes this surface (`../optimizer/search/`): each round builds `PromptCandidate`s and `Execution`s, scores them into `Evaluation { execution, reward: RewardBreakdown }`, records an `IterationRecord`, compresses a `HistoryEntry` into `OptimizationHistory`, asks a `DriftJudge` for deviations (skipped at `drift_penalty == 0`), asks a `ConvergenceDetector` whether to stop, persists the elite as a `PromptRecord`, and returns `OptimizationResult::records` as `Vec<IterationRecord>`.

## Constraints

- Serialization keeps the Python field names exactly (see the struct lists above); deserialization defaults every field, so a partial record loads.
- `crate::tune::tool::RunTelemetry` is copied in from the evaluator seam: `CaseResult` and `Execution` both carry it serde-defaulted, so a payload written before telemetry existed still parses and the legacy `latency_ms`/`total_tokens` keys keep their shape.
- Ids and timestamps are parameters: `candidate_id` is a hash, `created_at`/`applied_at` come from the caller, and `mark_applied` never reads a clock (it sets `applied` only).
- Errors are `openjiuwen_common::error::{Error, Result}`: validation for blank text and invalid convergence config, the filesystem error for an unreadable memory file (via `From<io::Error>`), `Error::not_found` nowhere here.
- `RewardComponent::score` is synchronous (deviation from Python's `async def`) because components only read an `Execution`; only `DriftJudge` stays async because it may call a model.
- A panicking `StructuredValidationReward`/`CustomReward` closure is not caught (deviation from Python's broad `except`): a panic is caller code failure, not a case outcome.
- No `unsafe`, no printing, no panics on user input or a poisoned memory lock; tests may `unwrap`.
- Files stay at or below ~1000 lines; shared tokenization and Jaccard live only in `memory.rs` (`pub(crate)`).

## Design Decisions

- `PromptTaskCase` stores the extracted `input_text`/`expected_text` instead of Python's wrapped `Case`, so it is serde-friendly; `from_case`/`to_case` keep it composable with the shared tuning `Case`.
- `CorrectnessReward` reads the per-case `score` the caller's evaluator already produced instead of calling an evaluator (`BaseEvaluator`/LLM judge in Python), so no second judging stack lives inside `tune`.
- Run telemetry is additive: `Execution`'s `measured_*` accessors are the only place the preference order (reported telemetry, then the legacy caller-supplied field) is decided, so no reward component reads `telemetry` directly and a caller's legacy value is never silently dropped.
- `CompositeReward` keeps Python's exact guard order — weighted mean, min-correctness gate, overfitting penalty, drift penalty, execution-error override, final clamp — and notes every adjustment with the same message shape, so a report is reproducible and human-readable.
- `correctness` in a breakdown is the raw (pre-normalization) value of the component named by `correctness_name`, defaulting to `1.0` when absent (Python parity); a zero total weight is treated as `1.0`.
- Hidden-set correctness is computed for every execution whenever a correctness component exists and the task declares a hidden case, so a candidate with no hidden result is measured at `0.0` (Python parity).
- `ConvergenceDetector::check` is stateless over a reward slice instead of Python's accumulating `update`, so a caller can re-check arbitrary history without owning mutable state; the rule priority and messages match Python.
- `LexicalDriftJudge` is the default rather than an LLM judge: it is deterministic and needs no model, while `ModelDriftJudge` fails safe to `0.0` on an outage (never penalizing a candidate for a judge failure).
- `JsonlPromptMemory` rewrites the whole file atomically (`records.jsonl.tmp` then rename) for both `store` and `mark_applied`, so a concurrent reader never observes a partial line; malformed lines are skipped and counted, never fatal.
- `search_similar` scores `jaccard(task.characteristics(), record.task_characteristics or record.objective) * (0.5 + 0.5 * reward)`, keeps positive scores, stable-sorts by score descending (insertion order on ties), and falls back to all records by reward descending when the query has no tokens or nothing overlaps.
- `HistoryCompressor` is the deterministic `_heuristic` port (best/weakest buckets with up to three observations each), deduplicated by exact lesson line and truncated on line boundaries to the character budget; Python's LLM bucket compression is not ported.
- `HistoryEntry::render` keeps Python's byte shape, including the em dash and the `…` prompt truncation.

## Test Strategy

- Unit tests live beside each module and are fully offline: `openjiuwen_test_support::ScriptedModel` for the model drift judge, `TempWorkspace` for the JSONL memory, fixed ids and rewards, no network or wall-clock reads.
- The reward suite lives in the sibling `reward_tests.rs` (included as `reward::tests` via `#[path]`) because it is the largest; every other module keeps its `#[cfg(test)] mod tests` in-file.
- `models.rs` covers the from-text/from-case/to-case round trip and blank-text validation, `characteristics`, visible/hidden splits, candidate-id determinism and index sensitivity, `CaseResult` flags, combined outputs, best-candidate selection, gain/pending, reward-breakdown defaults, Python field names, and missing/unknown-field tolerance.
- `models.rs` also covers run telemetry: the `measured_*` accessors preferring reported telemetry and falling back to the legacy fields, `Some(0)` beating a non-zero legacy value, `CaseResult` telemetry defaults and round-trips, and an old payload without a `telemetry` key parsing with absent telemetry while the legacy keys survive a round trip.
- `reward.rs` covers normalization edges, every component at its boundaries (zero and half-life latency/tokens/cost, errored and empty results, custom validators), the weighted mean, the zero-weight filter, the normalize flag, gate plus overfitting in one case (exact score and both notes), drift clamping, the execution-error override, determinism, and the breakdown JSON round-trip.
- `reward.rs` also covers the run-cost components against telemetry: a measured half-life scoring exactly `0.5`, a prompt-only token measurement counting the missing completion side as zero, telemetry overriding the legacy fields, absent telemetry keeping the historical constants, and a measured zero scoring `1.0` like an absent measurement.
- `convergence.rs` covers each rule, its priority, the `window`/`window + 1` boundary, growth that must not converge, invalid configuration, and state serialization.
- `drift.rs` covers the null and lexical bands, the model judge's fenced/bare/clamped/blank-prompt paths, unusable answers, provider failures, and the recorded request.
- `memory.rs` covers tokenization/Jaccard, the no-op memory, store/reopen round trips, malformed-line skipping, ranking by overlap then reward, `top_k`, fallbacks, `best_for_objective`, `pending` gain ordering, and `mark_applied` persistence and unknown ids.
- `history.rs` covers the exact render bytes (em dash, truncation), empty-history defaults, best prompt/reward ties, recent-section limits, single- and multi-entry compression, lesson deduplication, budget truncation, and serialization.
- `mod.rs` composes the re-exports into one round (task → executions → reward → iteration → history → memory → convergence) and reaches all three judges.

## Routing Table

- `./models.rs` → task/case/candidate/execution/reward/evaluation/iteration/record models and `candidate_id`
- `./reward.rs` → reward components, normalization, weights, `CompositeReward` guards
- `./reward_tests.rs` → the reward component/`CompositeReward` test suite (included as `reward::tests`)
- `./convergence.rs` → `population_variance`, `ConvergenceState`, `ConvergenceDetector`
- `./drift.rs` → `DriftJudge`, `NullDriftJudge`, `LexicalDriftJudge`, `ModelDriftJudge`
- `./memory.rs` → `PromptMemory`, `NullPromptMemory`, `JsonlPromptMemory`, shared `token_set`/`jaccard`
- `./history.rs` → `HistoryEntry`, `OptimizationHistory`, `HistoryCompressor`
- `./mod.rs` → module docs and the public re-export surface
- `../../base.rs` → shared `Case` consumed by `PromptTaskCase` (outside this node — read-only, escalate writes to `src/tune`)
- `../../tool.rs` → `RunTelemetry` carried by `CaseResult`/`Execution` (outside this node — read-only, escalate writes to `src/tune`)
- `../optimizer/search/` → `PromptSearchOptimizer`, the RLAF-P loop that consumes every type here (sibling — read-only, other agents own it)
- `../optimizer.rs` → `OptimizationResult`, whose `records` field is `Vec<IterationRecord>` (sibling — read-only, other agents own it)
- `../../util.rs` → `extract_json_fence` used by the model drift judge (outside this node — read-only, escalate writes to the crate root)
