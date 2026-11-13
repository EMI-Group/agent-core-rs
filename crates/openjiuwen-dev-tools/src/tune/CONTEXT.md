# openjiuwen-dev-tools / tune

## Intent

Port `openjiuwen.dev_tools.tune` onto four cooperating layers: case data, an evaluator seam, the prompt-search (RLAF-P) domain layer, and the optimizers plus a trainer.
Python's tune package mutates an `LLMCall` parameter in place and reports through callbacks; here every optimizer is a value that returns an `OptimizationResult`, and every candidate is scored through one trait, so the same code runs offline or through a model.
The prompt-search layer is split in two: `prompt_search/` owns the domain vocabulary (task, execution, reward, convergence, drift, memory, history) and `optimizer/search/` owns the bounded multi-round loop that composes it.
Everything is offline and deterministic by default: no environment reads, no wall-clock dependence, no network.

## API Surface

- `Case` — `inputs`, `label`, `tools`, `case_id` (Python field names); `new` (validating), `with_tools`, `with_case_id`, `validate` (which also rejects a blank or duplicated declared tool name under `Case.tools`), plus the read-only accessors `tools`/`tool_names`/`has_tools` and the Python `Case::from_text(text, expected)` constructor building `{"query": …}` / `{"output": …}`.
- `EvaluatedCase` — `case`, `answer`, `score`, `reason`; `new` (score bounds), `validate`, accessors `inputs`/`label`/`tools`/`case_id`, `From<Case>` producing score `0.0` and empty answer/reason.
- `CaseLoader` — `new`, `size`, `is_empty`, `cases`, `into_cases`, `get`, `iter`, `shuffle` (seeded xorshift64* Fisher-Yates), `split` (floor of `ratio * size`), iteration, JSON round-trip through `{"cases":[...]}`.
- Evaluator seam (`evaluator/`, details in `evaluator/CONTEXT.md`):
  - `PromptRunner` — `async fn run(&self, prompt, case) -> Result<BTreeMap<String, Value>>` plus the provided `run_traced` returning a `RunOutcome` (default: `run` with an empty `ToolTrace`).
  - `ModelPromptRunner` — system message = candidate prompt, user message = `render_case_inputs(case)`; a fenced or whole-answer JSON object becomes the answer map, plain text becomes `{"output": text}`.
    `with_stubs(ToolStubs)` and `with_max_tool_rounds(usize)` (default `DEFAULT_MAX_TOOL_ROUNDS = 8`) make `run_traced` declare `Case::tools()`, execute the model's tool calls in order against the stubs, record every call, and feed each outcome back as a `Role::Tool` message.
  - `ScriptedPromptRunner` — canned answers keyed by case id, optional fallback, recorded `PromptCall`s, `Error::not_found` for an unscripted case without a fallback; runs no tools.
  - `CaseScorer` — `fn score(&self, case, answer) -> (f64, String)` plus the provided `score_with_trace` (default ignores the trace); `LabelScorer` is the deterministic default (see scoring bands below).
  - `ToolUsageScorer` — multiplies an inner `LabelScorer` (or any `Box<dyn CaseScorer>`) by the fraction of the required tools the trace used: `new(required)`, `from_case(&Case)`, `with_inner`, `required()`, `missing(&ToolTrace)`.
  - `CaseEvaluator` — `async fn evaluate(&self, prompt, case) -> Result<EvaluatedCase>`, the provided `evaluate_traced` yielding `(EvaluatedCase, ToolTrace)`, the provided `score_cases` (mean plus per-case results), and the provided `score_cases_traced` (mean plus per-case pairs); `tool_usage(&[(EvaluatedCase, ToolTrace)])` aggregates per-tool call counts.
  - `RunnerEvaluator` — any `PromptRunner` plus any `CaseScorer` (default `LabelScorer`); evaluates through `run_traced` + `score_with_trace`.
  - `ModelEvaluator` — the model preset: `ModelPromptRunner` + `LabelScorer`, `new(Arc<dyn Model>)`, `with_stubs(model, stubs)`, `with_scorer`.
  - `DeterministicEvaluator` — offline default, `new(seed_prompt)`, `seed()`, `seed_placeholders()`, `score_parts(candidate, case) -> DeterministicScore` (total plus the four dimensions) and the `CaseEvaluator` impl.
  - `render_case_inputs(case)` — `[key]: value` per input, strings bare and other JSON compact.
- Prompt-search domain layer (`prompt_search/`, details in `prompt_search/CONTEXT.md`):
  - `PromptTaskCase` — `input_text`/`expected_text`/`hidden`, `from_text`, `with_hidden`, `from_case`, `to_case`.
  - `PromptTaskSpec` — `objective`, `cases`, `constraints`, `base_prompt`, `metadata`, `characteristics()`, `visible_cases()`, `hidden_cases()`.
  - `PromptCandidate`/`candidate_id` — a prompt with a rationale and a deterministic 8-hex FNV-1a id (index-sensitive, replacing Python's `uuid4().hex[:8]`).
  - `CaseResult`/`Execution`/`Evaluation`/`IterationRecord` — one case outcome, one candidate run (results, latency, tokens, error), a scored run, and one round's `candidates`/`scores`/`best_score`; `OptimizationResult::records` is `Vec<IterationRecord>`.
  - `RewardBreakdown`/`RewardComponent`/`CompositeReward` — weighted components plus the gate → overfit → drift → error → clamp guards; `default_weights()` is correctness 1.0, completeness 0.3, latency 0.1, token_usage 0.1, cost 0.0; `clamp01`, `min_max_normalize`, `OVERFIT_MARGIN`, `OVERFIT_PENALTY`.
  - `ConvergenceDetector`/`ConvergenceState`/`population_variance` — `check(&[f64]) -> Option<ConvergenceState>` with target → plateau → variance priority.
  - `DriftJudge` — async trait with `NullDriftJudge`, the deterministic `LexicalDriftJudge`, and `ModelDriftJudge::new(Arc<dyn Model>)`.
  - `PromptMemory` — `store`, `search_similar`, `best_for_objective`, `pending`, `mark_applied`, `records` over `&self`; `NullPromptMemory` and the JSONL `JsonlPromptMemory`.
  - `HistoryEntry`/`OptimizationHistory`/`HistoryCompressor` — the per-round history and its budgeted textual "policy update".
  - `PromptRecord` — a persistable outcome (`reward`, `objective`, `task_characteristics`, `metrics`, `observations`, `metadata`, `baseline_reward`, `applied`) with `gain()` and `is_pending(threshold)`.
- Search (`optimizer.rs` facade and `optimizer/`):
  - `OptimizerConfig` — `iterations`, `beam_width`, `num_parallel`, `early_stop_score`, `num_examples`; `Default` = the tuning constants, `with_*` builders, `validate()`.
  - `OptimizationResult` — `best_prompt`, `score`, `iterations` (rounds actually run), plus serde-defaulted `converged`, `convergence_reason` (`None` when the run used its whole budget), `token_usage`, `run_id`, and `records` (`Vec<IterationRecord>`); `candidate_prompts()` lists the distinct non-empty candidates tried, best-first (ties keep first appearance).
  - `InstructionOptimizer::optimize(prompt, cases)` — beam search over instruction rewrites.
  - `ExampleOptimizer::optimize(prompt, cases, num_examples)` — greedy few-shot selection; public `render_examples`/`format_prompt` helpers.
  - `JointOptimizer::optimize(prompt, cases)` — alternating instruction and example ascent.
  - `PromptSearchOptimizer` — `new`/`Default` plus `with_config`, `with_search_config`, `with_model`, `with_evaluator`, `with_reward`, `with_drift_judge`, `with_memory`, `with_run_id_seed`; readers `config()`, `search_config()`, `has_model()`; `run_id(seed, task)` (deterministic `{seed:016x}-{hash:08x}`); `async optimize(prompt, cases)` (objective and base prompt = the seed, all cases visible) and `async optimize_task(&PromptTaskSpec)` (keeps objective, constraints, base prompt, hidden cases, metadata).
  - `PromptSearchConfig` — `max_iterations` 6, `candidate_prompts` 5, `convergence_threshold` 0.01, `convergence_window` 3, `drift_penalty` 0.5, `min_correctness` 0.5, `policy_temperature` 0.9, `target_score: Option<f64>`; `new`, `with_*`, `validate()` against the `TuneConstant` bounds.
  - `ParameterSearcher` — `new`, `with_sample_num`, `with_seed`, `sample_num()`, `seed()`, `async search_best(base_score, base_parameters, parameters, evaluate)` sampling the cross product of alternative parameter maps around a baseline.
  - `ParameterSearchOutcome` — `best_score`, `best_parameters`, `best_cases`, `last_score`, and the Rust-only `evaluated` count.
  - `generate_candidates`/`candidate_space`/`DEFAULT_CANDIDATES_SAMPLE_NUM = 6` — the deterministic cross-product helpers behind `ParameterSearcher`.
  - `Trainer::train(prompt, train, val)` — train/validation loop with early stop; `Trainer::evaluate(prompt, cases)` returns the mean and per-case results.
  - `Trainer::evaluate_traced(prompt, cases)` — `EvaluationSummary` (`mean_score`, `results`, `tool_usage`) with the same mean/results as `evaluate` plus per-tool call counts aggregated from the evaluator's traced seam; `is_empty()` reports no results and `tool_names()` lists the called tools sorted.
  - `optimize_prompt(prompt, cases)` — default `InstructionOptimizer`.
  - `placeholders(prompt)` — the `{{name}}` names of a prompt.
- Tool layer (`tool.rs`): `ToolHandler` (synchronous `Fn(&Value) -> Result<Value>` shared through an `Arc`), `ToolStub` (named in-process fixture with `new`/`constant`/`name`/`call`), `ToolStubs` (a `BTreeMap` lookup with `insert`/`with_stub`/`get`/`contains`/`names`/`len`/`is_empty` and `call`, unknown names reported as `Error::not_found`), `ToolCallRecord` (`name`/`arguments`/`result`/`error` plus `success`/`failure`/`is_error`), `ToolTrace` (ordered records with `names`/`used`/`used_all`/`count`/`coverage`/`missing`/`usage`, serialized as a bare JSON array), and `RunOutcome` (`answer` plus `trace`, `new`/`with_trace`/`into_answer`).
- Every optimizer is no-arg constructible and offers `with_config`, `with_model(Arc<dyn Model>)`, `with_evaluator(Arc<dyn CaseEvaluator>)`, plus `config()`/`has_model()` readers; `PromptSearchOptimizer` adds the prompt-search builders listed above.
- Constants (values unchanged from Python `TuneConstant`): `DEFAULT_EXAMPLE_NUM=1`, `DEFAULT_ITERATION_NUM=3`, `DEFAULT_PARALLEL_NUM=1`, `DEFAULT_EARLY_STOP_SCORE=1.0`, `MIN_ITERATION_NUM=1`, `MAX_ITERATION_NUM=20`, `MIN_PARALLEL_NUM=1`, `MAX_PARALLEL_NUM=20`, `MIN_EXAMPLE_NUM=0`, `MAX_EXAMPLE_NUM=20`, `DEFAULT_MAX_SAMPLED_EXAMPLE_NUM=10`, `DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES=10`.
- Prompt-search constants: `DEFAULT_SEARCH_MAX_ITERATIONS=6`, `DEFAULT_SEARCH_CANDIDATE_PROMPTS=5`, `DEFAULT_CONVERGENCE_THRESHOLD=0.01`, `DEFAULT_CONVERGENCE_WINDOW=3`, `DEFAULT_DRIFT_PENALTY=0.5`, `DEFAULT_MIN_CORRECTNESS=0.5`, `DEFAULT_POLICY_TEMPERATURE=0.9`, `DEFAULT_CANDIDATES_SAMPLE_NUM=6`.
- The module root (`mod.rs`) re-exports every public type, helper, and constant; `base`, `evaluator`, `loader`, `optimizer`, `prompt_search`, `tool`, and their submodules stay private.

## Scoring bands (`LabelScorer`)

The answer value is compared with the expected label value per label key, after trimming: exact match `1.0`, case-insensitive match `0.9`, either value containing the other `0.75`, token-set Jaccard in `0.5..=1.0` mapped linearly into `0.3..=0.6`, otherwise `0.0`.
The case score is the mean over the label keys; the reason names the first key that is not an exact match.
When the answer has exactly one entry (the `output` entry a plain-text model answer produces) that entry stands in for a missing label key, so prose answers are still scored.

## Deterministic evaluator weights

`total = 0.35·placeholder + 0.20·structure + 0.30·coverage + 0.15·hygiene`.
`placeholder` — fraction of the seed's `{{placeholder}}`s still present (`1.0` when the seed has none).
`structure` — fraction of four instruction cue categories present: imperative task statement, explicit steps, output format, constraint/refusal.
`coverage` — fraction of the case's label-value and input-value tokens present in the candidate, case-insensitively (`1.0` when the case yields no tokens).
`hygiene` — `×0.5` under 20 characters, `×0.7` over 8000, `×(1 − duplicate-line ratio)` above a 30% duplicate ratio, `×0.2` without any alphabetic character.
A blank candidate scores `0.0`; the score is a pure function of (seed, candidate, case) and varies with the case through coverage.

## Search algorithms

`InstructionOptimizer`: score the seed, then per round expand every beam member with the deterministic mutation bank and the best member with one model rewrite, repair missing placeholders, drop candidates already visited, score all new candidates over all cases, keep the best `beam_width`, and track the global best; stop before a round when the best score reaches `early_stop_score`, so an already-good seed runs zero rounds.
The mutation bank is an ordered, marker-identified list (step-by-step reasoning, output format, no-invention/refusal, self-check, constraints, context-only); a mutation whose marker already appears is not re-applied, which makes repeated rounds idempotent.
`ExampleOptimizer`: score the seed, optionally seed the selection from a validated model ` ```list ` of indices, then append greedily the example that maximises the evaluator score (ties by `case_id` ascending) until `min(num_examples, cases.len())` examples are in the prompt; return the best-scoring prompt seen, which is the seed when no addition helped.
`JointOptimizer`: measure three baselines first (seed, instruction-only search, example-only selection), then per round run one instruction round from the current best followed by example selection on the rewritten prompt, keeping the best prompt under the tie-break.
`PromptSearchOptimizer`: the round budget is `min(search.max_iterations, config.iterations)` — 3 with the defaults — and each round is propose → execute → reward → converge.
`PromptSearchOptimizer` proposes an elitist seed first (round 1 = the base prompt, later rounds = the best prompt so far), then the round-1 warm-start prompts from an injected memory, every deterministic mutation of the round seed, and one model rewrite when a model is attached; the rewrite request carries the worst cases, the `policy_temperature`, and a `<policy_context>` block with the compressed lessons and warm prompts.
`PromptSearchOptimizer` repairs missing placeholders against the round seed, deduplicates the pool (first occurrence wins), and truncates it to `search.candidate_prompts`.
`PromptSearchOptimizer` executes every candidate against every visible and hidden case through the `CaseEvaluator` (`num_parallel` candidates at a time, results in input order); a per-case evaluator error becomes that `CaseResult`'s `error` and the run continues, and the `Execution` is marked failed only when every case failed.
`PromptSearchOptimizer` scores drift through the injected judge against the task objective (skipped, all zeros, when `drift_penalty == 0`) and then the composite reward; each round appends an `IterationRecord` and a compressed `HistoryEntry` and carries the strictly better round best forward.
`PromptSearchOptimizer` stops on the convergence detector (target reward, then plateau, then variance) or on the shared `early_stop_score`, and a run that uses its whole budget reports `converged: false` with `convergence_reason: None`.
`PromptSearchOptimizer` persists the elite as a `PromptRecord` (with `baseline_reward` when memory held a prior reward) only into an injected memory; a `store` failure is logged and never fails the run.
`ParameterSearcher`: build the parameter cross product with the first key outermost, drop the all-baseline combination, shuffle the rest with the seeded xorshift64* Fisher-Yates, keep `min(sample_num, len)`, evaluate them in that order, and replace the winner only on a strictly greater score (the first evaluated candidate's cases are the fallback when nothing wins).
`Trainer`: baseline validation score, early stop when it reaches the threshold, then per epoch one instruction round on the train split scored on the validation split, keeping the best validation score (ties by lexicographically smaller prompt).

## Constraints

- Serialization: `Debug, Clone, PartialEq, Serialize, Deserialize` on public data types, `schemars::JsonSchema` on `Case`/`EvaluatedCase`, `rename_all = "snake_case"`, unknown fields ignored.
- Python JSON field names round-trip exactly: `inputs`, `label`, `tools`, `case_id`, `case`, `answer`, `score`, `reason`, `best_prompt`, `iterations`, `converged`, `convergence_reason`, `best_parameters`, `best_cases`, `last_score`, `evaluated`.
- Deserialization is permissive (missing `case_id` gets a fresh simple v4 UUID, missing `score` is `0.0`, missing `CaseLoader.cases` is empty, missing config fields default to the constants, missing `OptimizationResult` prompt-search fields default to `false`/`None`/empty); only `new`/`validate` enforce the Python `pydantic` constraints.
- Errors come from `openjiuwen_common::error` only: `Error::validation` for bad input (a blank tool or required-tool name, a duplicated declaration, a zero `max_tool_rounds`, a blank objective, an empty case list, an out-of-range `PromptSearchConfig`/`OptimizerConfig`, a non-finite `base_score`, a zero `sample_num`), `Error::not_found` for an unscripted case or an unregistered tool, `Error::llm` for provider failures; nothing in this module returns `Error::Unsupported`.
- `Case`/`EvaluatedCase` keep Python's JSON field names; `EvaluatedCase` gains no fields, so the tool trace travels only through the traced methods and `EvaluationSummary`.
- No `unsafe`, no panics on user input or on a poisoned call-recording or memory lock, no printing; tests may use `unwrap`/`expect`.
- Async-first: optimizer entry points are `async` because they may call a model; the deterministic path never awaits anything external.
- Ids are deterministic: `candidate_id` is an FNV-1a hash of `"{index}\x1f{prompt}"`, `PromptSearchOptimizer::run_id` hashes the objective, base prompt, and case count, and memory timestamps are caller-supplied; no clock or uuid is read.
- `OptimizationResult::token_usage` stays empty and every `Execution` carries `latency_ms = 0.0` / `total_tokens = 0`, because `CaseEvaluator` reports neither; the latency, token-usage, and cost reward components are therefore constant (a documented adaptation), and correctness, completeness, and structured validation carry the signal.
- Prompt-memory persistence and warm starts are opt-in: without `with_memory`, `NullPromptMemory` answers empty and nothing is written; a failing store is logged, never fatal.
- Files ≤1000 lines, one responsibility per file; shared candidate-generation helpers live in `optimizer/suggestion.rs`, never duplicated per optimizer.

## Design Decisions

- Python's optimizers are stateful mutators of an `LLMCall`; the port makes them values with builder methods and a returned result, so one process can run several optimizers concurrently and tests need no fixture agent.
- Every optimizer takes an `Arc<dyn Model>` rather than constructing one from configs, so configuration stays with the caller and no builder reads the environment.
- Candidate scoring goes through `CaseEvaluator`, never through inline model calls: the deterministic evaluator makes the search reproducible offline, and injecting a model evaluator or a custom scorer changes only the scorer, not the algorithm.
- The prompt-search layer is split by responsibility: `prompt_search/` holds the pure, model-free domain mechanics (no loop, no policy), `optimizer/search/` holds the bounded loop, and `PromptSearchConfig` holds the tunables, so the domain layer can be tested and composed without a search.
- `PromptSearchOptimizer`'s policy is deterministic — the shared mutation bank plus at most one model rewrite per round — because Python's `LLMPromptPolicy` (multi-candidate JSON generation) is not ported; elitism carries the best prompt forward, so the result is never worse than the starting prompt.
- `PromptSearchOptimizer`'s environment is the caller's `CaseEvaluator` rather than Python's `LLMEnvironment`: the same loop runs against the deterministic evaluator or a model, and per-case evaluator failures become `CaseResult` data instead of aborting a round.
- Elitism and the budget coupling are explicit: the RLAF budget is capped by `OptimizerConfig::iterations`, `iterations` reports rounds actually run (`records.len()`), and an early convergence stop is visible as a smaller number with a reason.
- The convergence detector is stateless over the round-best rewards; the loop re-checks the slice each round, and the reason strings distinguish target, plateau, and variance stops.
- Drift is optional in two ways: the judge is injectable, and a zero `drift_penalty` skips it entirely with all-zero deviations (Python's early return); `LexicalDriftJudge` is the model-free default and `ModelDriftJudge` fails safe to `0.0` on an outage.
- Prompt memory only ever helps: it supplies round-1 warm starts and receives the final elite when injected, and the default `NullPromptMemory` keeps a memoryless run byte-identical to a run without the feature.
- The deterministic tie-break is total and explicit — higher score, then the earlier candidate index, then the lexicographically smaller prompt — so repeated runs return identical results and no choice depends on map or completion order.
- Model output is optional everywhere: an unusable rewrite (no `<PROMPT_OPTIMIZED>` block, empty, identical to the base, provider error) or an unusable index list (unparsable, out of range, repeated, non-integer) logs with `tracing::warn!` and falls back to the deterministic bank or greedy selection instead of failing the run.
- `iterations` counts rounds actually run (beam rounds, greedy steps, epochs, or prompt-search rounds), so an early stop is visible as a smaller number; an already-sufficient seed reports `0`.
- `Trainer::train` is `async` although Python's is synchronous, because the instruction optimizer it drives may call a model; the reported score is always measured on the validation split so a train-overfitting candidate cannot win.
- `num_examples == 0` is a documented no-op returning the prompt with its measured score, matching Python's "skip do example optimization" branch.
- Python's `random.shuffle`/`random.sample` and `random.choice` strategy coin flip are replaced by seeded or deterministic selection so a run is reproducible; shuffling lives in `loader.rs` (xorshift64*) and `ParameterSearcher` repeats the same generator privately so the crate needs no random-number dependency.
- `ParameterSearcher` never reads a variable that only exists inside its loop: an empty candidate space (every alternative is the baseline) returns the baseline with `evaluated == 0` and empty `best_cases`, where Python raises `UnboundLocalError`.
- Scores are clamped into `0.0..=1.0` and float comparisons use a `1e-12` tolerance, so a custom scorer returning a wild value cannot panic `EvaluatedCase::new`.
- `placeholders` and the `ensure_placeholders` repair mirror Python's `PromptAssembler` input keys and its "append the missing `{{name}}` lines" repair, without pulling in the prompt assembler.
- Tool-aware evaluation is additive: the trace is a return value (`run_traced`, `evaluate_traced`, `score_cases_traced`, `EvaluationSummary`), so `EvaluatedCase` keeps the Python wire shape and an untraced `PromptRunner`/`CaseScorer`/`CaseEvaluator` implementor keeps compiling.
- Tool failures are data, not control flow: an unknown tool, a failing stub, and malformed call arguments are recorded in the `ToolTrace` and fed back to the model as the tool result, and the loop stays bounded by `max_tool_rounds`; only model/transport errors propagate.
- `ToolStubs` is injected by the caller and never built from configuration, so tests pass deterministic fixtures and nothing reads the environment.
- The rewrite prompt describes the case set's declared tools (`tools_description`/`tools_block` in `optimizer/suggestion.rs`) rather than offering them as a request `tools` array, matching Python's `str(tools)`; the `<tools_description>` block is omitted when no case declares tools, so tool-free prompts stay byte-identical.
- `model_rewrite_with` is the one rewrite entry point that takes a sampling temperature and a policy-context block; the instruction, example, and joint optimizers call the context-free wrapper, so their requests stay byte-identical.
- `ToolUsageScorer` scales the inner score instead of vetoing it, and the case's own declarations stay advisory: the required list is explicit (or `from_case`), so a case can declare a tool the run is not scored on.

## Test Strategy

- Unit tests live in each file's `#[cfg(test)]` module and are entirely offline: `ScriptedModel` for model paths, fixed case ids, no network, no environment reads, no wall-clock dependence.
- `base.rs` covers constructor validation, score bounds (`0.0`/`1.0` accepted; `-0.1`, `1.1`, `NaN`, infinities rejected), Python-field JSON round-trips, UUID defaulting, unknown-field tolerance, schema exposure, `From<Case>` accessors, and the tool accessors (`tools`/`tool_names` order and dedupe, `has_tools`, `from_text` shape and blank-text rejection, `validate` rejecting a blank or duplicated tool name).
- `tool.rs` covers stub construction and dispatch (blank-name validation, `constant` ignoring arguments, handler errors propagating, `Debug` showing the name), the table (`insert`/replacement/`get`/`contains`/sorted `names`/`with_stub` chaining, unknown-name `NotFound` listing the registered names), call records (`success`/`failure`/`is_error`, snake_case JSON round-trip, unknown-field and missing-`error` tolerance), the trace (order, duplicates, `used`/`used_all`/`count`, `coverage` bands, `missing` order and dedupe, `usage` counts, transparent serde), and `RunOutcome` (`new`/`with_trace`/`into_answer`, JSON defaults).
- `loader.rs` covers size/emptiness/get/iteration/split (including ratios `0.0`, `1.0`, `NaN`, out of range), shuffle determinism and permutation preservation, and JSON round-trip plus defaulting.
- `evaluator/tests.rs` covers every `LabelScorer` band and the Jaccard mapping, missing and single-entry answers, the `ModelPromptRunner` conversation and JSON parsing, provider-error propagation, scripted-runner replay and unknown cases, `RunnerEvaluator` mean scoring and custom scorers, `ModelEvaluator`, the deterministic dimensions (placeholder loss, coverage, structure, degenerate length/repetition, case dependence, repeatability), and the tool-aware paths (declarations reaching `ModelCall::request`, the multi-round loop and trace order, parallel calls, unknown tools and failing stubs fed back, the zero-budget rejection, the looping-model bound, `ToolUsageScorer` bands/reasons, and traced parity of both evaluator presets).
- `optimizer/instruction.rs` covers improvement over the seed, repeat-run equality, validation, early stop at the seed and after one round, iteration/beam caps, placeholder preservation, the model-rewrite path with placeholder repair, fallback on unparsable/failed/empty answers, an injected evaluator, beam ranking, the `<tools_description>` block reaching the rewrite prompt, and `Debug`.
- `optimizer/example.rs` covers the Python example format, `num_examples == 0`, the case-count cap, score monotonicity, determinism, validation (including `MAX_EXAMPLE_NUM + 1`), early stop, model index lists and the order they impose, fallback on bad lists/failures, greedy completion of a short list, and an injected evaluator.
- `optimizer/joint.rs` checks it is never worse than either coordinate alone, both coordinates contribute, `num_examples == 0`, determinism, early stop, iteration cap, validation, placeholder preservation, the model path, and a shared injected evaluator.
- `optimizer/search/tests.rs` covers the multi-round run and its per-round `IterationRecord`s, determinism, the placeholder/seed contract, input validation, a winning model rewrite, tool declarations and the `policy_temperature`/lessons context reaching the rewrite prompt, an unusable model answer, the seed winning when nothing improves, target/plateau/early-stop convergence, a budget-bound run reporting no convergence, the per-round candidate cap, hidden cases punishing an overfitted candidate, warm start plus persistence through a `JsonlPromptMemory`, an injected drift judge, `optimize_task` (constraints/metadata, validation, blank base-prompt fallback), `Debug`, and the deterministic task-sensitive `run_id`.
- `optimizer/search_config.rs` covers the Python defaults, every rejected field with its name, and the snake_case JSON round-trip with per-field defaulting.
- `optimizer/parameter_search.rs` covers cross-product order and dedupe, `candidate_space` validation, seeded sampling determinism, the never-exceeding-budget cap, strict-`>` replacement, the empty-space baseline fallback, evaluation-error propagation, and the outcome JSON round-trip plus defaulting.
- `optimizer/trainer.rs` covers baseline early stop, validation-score monotonicity, the epoch cap, the empty-validation fallback, empty-train validation, invalid config, `evaluate`, determinism, placeholder preservation, the model path, an injected evaluator, the deterministic default, and `evaluate_traced` (tool-usage counts, parity with `evaluate`, the empty case set, the tool-free path, an injected traced evaluator, and the `EvaluationSummary` JSON round-trip).
- `optimizer/suggestion.rs` covers placeholder parsing (including nested malformed braces), the repair, mutation ordering and idempotence, dedupe, input validation, the tie-break helpers, ordered batch scoring, model-rewrite extraction and degradation, worst-case selection, the `key:value | …` rendering, and the tools block (`tools_description` parity JSON and its `None` fallback, cross-case dedupe, absence for tool-free case sets, and the pinned no-tools rewrite message).
- `optimizer.rs` covers the constant values, prompt-search defaults and `PromptSearchConfig` validation, the `optimize_prompt` delegation, the serde-defaulted `OptimizationResult` fields (including a legacy JSON payload), and `candidate_prompts()` dedupe/ordering.
- `mod.rs` composes the re-exports into one loader workflow, one `optimize_prompt` run, one scripted-runner evaluation, and checks the evaluator, scorer, config, and placeholder entry points.

## Deferred

- Python's dataset samplers (`openjiuwen/dev_tools/tune/dataset/` beyond `CaseLoader`) are not ported: they schedule batches from external stores, while the port keeps the case list in memory and leaves splitting to `CaseLoader`.
- Python's `tune/chat_agent/` adapters (`ChatAgent`, `ChatConfig`) are not ported: they bind an optimizer to a Python agent runtime, whereas here an optimizer is driven by a `CaseEvaluator` the caller composes, so no agent adapter is needed.
- Python's prompt-search `policy.py` (`LLMPromptPolicy`'s multi-candidate JSON generation) is not ported: candidate generation is the deterministic mutation bank plus one model rewrite per round.
- Python's prompt-search `environment.py` (`LLMEnvironment`/`CallableEnvironment`) is not ported: candidates run through the caller's `CaseEvaluator`.
- Python's prompt-search `run_log.py` (JSONL run logger), `service.py`, and the `adopt/` optimizer family remain unported: there is no run-log sink, no service layer, and no adopt policy, memory, or drift service beyond the injected seams above.
- Python's LLM-backed reward/overfit/drift judges are replaced by the caller's per-case scores (`CorrectnessReward`), the deterministic `LexicalDriftJudge`, and the model-free `_heuristic` `HistoryCompressor`; no LLM judge lives inside `tune`.
- `tune` has no `LLMCall`/`TextualParameter` binding layer, so optimizers take prompt strings and `Case`s (and `ParameterSearcher` takes parameter maps) instead of agent parameters.
- Python's `evaluator` module's LLM-judge retry loop is replaced by `LabelScorer` plus an optional custom `CaseScorer`; rubric-driven model judging lives in the crate's `judge` module instead.

## Routing Table

- `./base.rs` → `Case`, `EvaluatedCase`, validation, JSON round-trips
- `./loader.rs` → `CaseLoader`, seeded PRNG, shuffle, split, iteration
- `./tool.rs` → `ToolStub`/`ToolStubs`, `ToolCallRecord`/`ToolTrace`, `RunOutcome`, tool handlers
- `./evaluator/` → the evaluator seam with its traced/untraced contract (`evaluator/CONTEXT.md`)
- `./evaluator/CONTEXT.md` → evaluator intent, traced API surface, tool-execution constraints, design decisions
- `./evaluator/runner.rs` → `PromptRunner`, `ModelPromptRunner` tool loop, `render_case_inputs`, `ScriptedPromptRunner`, `DEFAULT_MAX_TOOL_ROUNDS`
- `./evaluator/scorer.rs` → `CaseScorer`, `LabelScorer`, `ToolUsageScorer`, band comparison, tokenization
- `./evaluator/seam.rs` → `CaseEvaluator` (traced and untraced), `RunnerEvaluator`, `ModelEvaluator`, `tool_usage`, score clamping
- `./evaluator/deterministic.rs` → `DeterministicEvaluator`, `DeterministicScore`, weights and dimensions
- `./evaluator/tests.rs` → the evaluator test suite
- `./prompt_search/` → the prompt-search (RLAF-P) domain layer (`prompt_search/CONTEXT.md`)
- `./prompt_search/CONTEXT.md` → intent, model/reward/convergence/drift/memory/history surface, constraints, test strategy
- `./prompt_search/models.rs` → task/case/candidate/execution/reward/evaluation/iteration/record models and `candidate_id`
- `./prompt_search/reward.rs` → reward components, normalization, weights, `CompositeReward` guards
- `./prompt_search/reward_tests.rs` → the reward component/`CompositeReward` test suite
- `./prompt_search/convergence.rs` → `population_variance`, `ConvergenceState`, `ConvergenceDetector`
- `./prompt_search/drift.rs` → `DriftJudge`, `NullDriftJudge`, `LexicalDriftJudge`, `ModelDriftJudge`
- `./prompt_search/memory.rs` → `PromptMemory`, `NullPromptMemory`, `JsonlPromptMemory`, shared `token_set`/`jaccard`
- `./prompt_search/history.rs` → `HistoryEntry`, `OptimizationHistory`, `HistoryCompressor`
- `./prompt_search/mod.rs` → module docs and the domain-layer re-export surface
- `./optimizer.rs` → facade: tuning constants, `OptimizationResult`, `mod` declarations, re-exports, `optimize_prompt`
- `./optimizer/config.rs` → `OptimizerConfig`, bounds validation, JSON defaults
- `./optimizer/search_config.rs` → `PromptSearchConfig`, its `DEFAULT_*` constants, bounds validation
- `./optimizer/suggestion.rs` → placeholder parsing/repair, mutation bank, model-rewrite request (including the `<tools_description>` render and the `<policy_context>` block), tie-break, bounded scoring, worst-case selection
- `./optimizer/instruction.rs` → `InstructionOptimizer` beam search
- `./optimizer/example.rs` → `ExampleOptimizer`, `render_examples`, `format_prompt`
- `./optimizer/joint.rs` → `JointOptimizer` alternating ascent
- `./optimizer/search/` → `PromptSearchOptimizer`'s bounded multi-round loop (no directory `CONTEXT.md`; see the per-file entries below)
- `./optimizer/search/mod.rs` → the loop: builders, `run_id`, `optimize`/`optimize_task`, prepare/round/scoring/stopping/persistence
- `./optimizer/search/execute.rs` → per-candidate and batched execution, drift scores, warm start, policy context, observations/metrics/metadata
- `./optimizer/search/tests.rs` → the prompt-search loop test suite
- `./optimizer/parameter_search.rs` → `ParameterSearcher`, `ParameterSearchOutcome`, `candidate_space`, `generate_candidates`, seeded sampling
- `./optimizer/trainer.rs` → `Trainer` train/validation loop and `EvaluationSummary` traced reporting
- `./mod.rs` → module docs, private submodules, public re-exports
- `../judge/` → rubric-driven model judging (sibling — read-only here, not a dependency of `tune`)
- `../../util.rs` → `extract_tag`, `extract_list_fence`, `extract_json_fence` (outside this node — read-only, escalate writes to the crate root)
