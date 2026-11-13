# openjiuwen-dev-tools / tune

## Intent

Port `openjiuwen.dev_tools.tune` onto three cooperating layers: case data, an evaluator seam, and four prompt-search strategies plus a trainer.
Python's tune package mutates an `LLMCall` parameter in place and reports through callbacks; here every optimizer is a value that returns an `OptimizationResult`, and every candidate is scored through one trait, so the same code runs offline or through a model.
Everything is offline and deterministic by default: no environment reads, no wall-clock dependence, no network.

## API Surface

- `Case` — `inputs`, `label`, `tools`, `case_id` (Python field names); `new` (validating), `with_tools`, `with_case_id`, `validate` (which also rejects a blank or duplicated declared tool name under `Case.tools`), plus the read-only accessors `tools`/`tool_names`/`has_tools` and the Python `PromptTaskCase.from_text(text, expected)` constructor building `{"query": …}` / `{"output": …}`.
- `EvaluatedCase` — `case`, `answer`, `score`, `reason`; `new` (score bounds), `validate`, accessors `inputs`/`label`/`tools`/`case_id`, `From<Case>` producing score `0.0` and empty answer/reason.
- `CaseLoader` — `new`, `size`, `is_empty`, `cases`, `into_cases`, `get`, `iter`, `shuffle` (seeded xorshift64* Fisher-Yates), `split` (floor of `ratio * size`), iteration, JSON round-trip through `{"cases":[...]}`.
- Evaluator seam (`evaluator/`):
  - `PromptRunner` — `async fn run(&self, prompt, case) -> Result<BTreeMap<String, Value>>`.
  - `ModelPromptRunner` — system message = candidate prompt, user message = `render_case_inputs(case)`; a fenced or whole-answer JSON object becomes the answer map, plain text becomes `{"output": text}`.
  - `ScriptedPromptRunner` — canned answers keyed by case id, optional fallback, recorded `PromptCall`s, `Error::not_found` for an unscripted case without a fallback.
  - `CaseScorer` — `fn score(&self, case, answer) -> (f64, String)`; `LabelScorer` is the deterministic default (see scoring bands below).
  - `CaseEvaluator` — `async fn evaluate(&self, prompt, case) -> Result<EvaluatedCase>` plus a provided `score_cases` returning the mean and the per-case results (`0.0`/empty for no cases).
  - `RunnerEvaluator` — any `PromptRunner` plus any `CaseScorer` (default `LabelScorer`).
  - `ModelEvaluator` — the model preset: `ModelPromptRunner` + `LabelScorer`, `new(Arc<dyn Model>)`, `with_scorer`.
  - `DeterministicEvaluator` — offline default, `new(seed_prompt)`, `seed()`, `seed_placeholders()`, `score_parts(candidate, case) -> DeterministicScore` (total plus the four dimensions) and the `CaseEvaluator` impl.
  - `render_case_inputs(case)` — `[key]: value` per input, strings bare and other JSON compact.
- Search (`optimizer.rs` facade and `optimizer/`):
  - `OptimizerConfig` — `iterations`, `beam_width`, `num_parallel`, `early_stop_score`, `num_examples`; `Default` = the tuning constants, `with_*` builders, `validate()`.
  - `OptimizationResult` — `best_prompt`, `score`, `iterations` (rounds actually run).
  - `InstructionOptimizer::optimize(prompt, cases)` — beam search over instruction rewrites.
  - `ExampleOptimizer::optimize(prompt, cases, num_examples)` — greedy few-shot selection; public `render_examples`/`format_prompt` helpers.
  - `JointOptimizer::optimize(prompt, cases)` — alternating instruction and example ascent.
  - `PromptSearchOptimizer::optimize(prompt, cases)` — one breadth-first sweep over the whole pool.
  - `Trainer::train(prompt, train, val)` — train/validation loop with early stop; `Trainer::evaluate(prompt, cases)` returns the mean and per-case results.
  - `optimize_prompt(prompt, cases)` — default `InstructionOptimizer`.
  - `placeholders(prompt)` — the `{{name}}` names of a prompt.
- Tool layer (`tool.rs`): `ToolHandler` (synchronous `Fn(&Value) -> Result<Value>` shared through an `Arc`), `ToolStub` (named in-process fixture with `new`/`constant`/`name`/`call`), `ToolStubs` (a `BTreeMap` lookup with `insert`/`with_stub`/`get`/`contains`/`names`/`len`/`is_empty` and `call`, unknown names reported as `Error::not_found`), `ToolCallRecord` (`name`/`arguments`/`result`/`error` plus `success`/`failure`/`is_error`), `ToolTrace` (ordered records with `names`/`used`/`used_all`/`count`/`coverage`/`missing`/`usage`, serialized as a bare JSON array), and `RunOutcome` (`answer` plus `trace`, `new`/`with_trace`/`into_answer`).
- Every optimizer is no-arg constructible and offers `with_config`, `with_model(Arc<dyn Model>)`, `with_evaluator(Arc<dyn CaseEvaluator>)`, plus `config()`/`has_model()` readers.
- Constants (values unchanged from Python `TuneConstant`): `DEFAULT_EXAMPLE_NUM=1`, `DEFAULT_ITERATION_NUM=3`, `DEFAULT_PARALLEL_NUM=1`, `DEFAULT_EARLY_STOP_SCORE=1.0`, `MIN_ITERATION_NUM=1`, `MAX_ITERATION_NUM=20`, `MIN_PARALLEL_NUM=1`, `MAX_PARALLEL_NUM=20`, `MIN_EXAMPLE_NUM=0`, `MAX_EXAMPLE_NUM=20`.
- The module root (`mod.rs`) re-exports every public type, helper, and constant; `base`, `evaluator`, `loader`, `optimizer`, and the `evaluator/*` and `optimizer/*` submodules stay private.

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
`PromptSearchOptimizer`: pool = seed + every deterministic mutation + one model rewrite, deduplicated and placeholder-repaired, all scored, best returned, `iterations = 1`.
`Trainer`: baseline validation score, early stop when it reaches the threshold, then per epoch one instruction round on the train split scored on the validation split, keeping the best validation score (ties by lexicographically smaller prompt).

## Constraints

- Serialization: `Debug, Clone, PartialEq, Serialize, Deserialize` on public data types, `schemars::JsonSchema` on `Case`/`EvaluatedCase`, `rename_all = "snake_case"`, unknown fields ignored.
- Python JSON field names round-trip exactly: `inputs`, `label`, `tools`, `case_id`, `case`, `answer`, `score`, `reason`, `best_prompt`, `iterations`.
- Deserialization is permissive (missing `case_id` gets a fresh simple v4 UUID, missing `score` is `0.0`, missing `CaseLoader.cases` is empty, missing config fields default to the constants); only `new`/`validate` enforce the Python `pydantic` constraints.
- Errors come from `openjiuwen_common::error` only: `Error::validation` for bad input, `Error::not_found` for an unscripted case, `Error::llm` for provider failures; nothing in this module returns `Error::Unsupported`.
- No `unsafe`, no panics on user input or on a poisoned call-recording lock, no printing; tests may use `unwrap`/`expect`.
- Async-first: optimizer entry points are `async` because they may call a model; the deterministic path never awaits anything external.
- Files ≤1000 lines, one responsibility per file; shared candidate-generation helpers live in `optimizer/suggestion.rs`, never duplicated per optimizer.

## Design Decisions

- Python's optimizers are stateful mutators of an `LLMCall`; the port makes them values with builder methods and a returned result, so one process can run several optimizers concurrently and tests need no fixture agent.
- Every optimizer takes an `Arc<dyn Model>` rather than constructing one from configs, so configuration stays with the caller and no builder reads the environment.
- Candidate scoring goes through `CaseEvaluator`, never through inline model calls: the deterministic evaluator makes the search reproducible offline, and injecting a model evaluator or a custom scorer changes only the scorer, not the algorithm.
- The deterministic tie-break is total and explicit — higher score, then the earlier candidate index, then the lexicographically smaller prompt — so repeated runs return identical results and no choice depends on map or completion order.
- Model output is optional everywhere: an unusable rewrite (no `<PROMPT_OPTIMIZED>` block, empty, identical to the base, provider error) or an unusable index list (unparsable, out of range, repeated, non-integer) logs with `tracing::warn!` and falls back to the deterministic bank or greedy selection instead of failing the run.
- `iterations` counts rounds actually run (beam rounds, greedy steps, epochs, or `1` for the single search sweep), so an early stop is visible as a smaller number; an already-sufficient seed reports `0`.
- `Trainer::train` is `async` although Python's is synchronous, because the instruction optimizer it drives may call a model; the reported score is always measured on the validation split so a train-overfitting candidate cannot win.
- `num_examples == 0` is a documented no-op returning the prompt with its measured score, matching Python's "skip do example optimization" branch.
- Python's `random.shuffle`/`random.sample` and `random.choice` strategy coin flip are replaced by seeded or deterministic selection so a run is reproducible; shuffling lives in `loader.rs` (xorshift64*).
- Scores are clamped into `0.0..=1.0` and float comparisons use a `1e-12` tolerance, so a custom scorer returning a wild value cannot panic `EvaluatedCase::new`.
- `placeholders` and the `ensure_placeholders` repair mirror Python's `PromptAssembler` input keys and its "append the missing `{{name}}` lines" repair, without pulling in the prompt assembler.

## Test Strategy

- Unit tests live in each file's `#[cfg(test)]` module and are entirely offline: `ScriptedModel` for model paths, fixed case ids, no network, no environment reads, no wall-clock dependence.
- `base.rs` covers constructor validation, score bounds (`0.0`/`1.0` accepted; `-0.1`, `1.1`, `NaN`, infinities rejected), Python-field JSON round-trips, UUID defaulting, unknown-field tolerance, schema exposure, `From<Case>` accessors, and the tool accessors (`tools`/`tool_names` order and dedupe, `has_tools`, `from_text` shape and blank-text rejection, `validate` rejecting a blank or duplicated tool name).
- `tool.rs` covers stub construction and dispatch (blank-name validation, `constant` ignoring arguments, handler errors propagating, `Debug` showing the name), the table (`insert`/replacement/`get`/`contains`/sorted `names`/`with_stub` chaining, unknown-name `NotFound` listing the registered names), call records (`success`/`failure`/`is_error`, snake_case JSON round-trip, unknown-field and missing-`error` tolerance), the trace (order, duplicates, `used`/`used_all`/`count`, `coverage` bands, `missing` order and dedupe, `usage` counts, transparent serde), and `RunOutcome` (`new`/`with_trace`/`into_answer`, JSON defaults).
- `loader.rs` covers size/emptiness/get/iteration/split (including ratios `0.0`, `1.0`, `NaN`, out of range), shuffle determinism and permutation preservation, and JSON round-trip plus defaulting.
- `evaluator/tests.rs` covers every `LabelScorer` band and the Jaccard mapping, missing and single-entry answers, the `ModelPromptRunner` conversation and JSON parsing, provider-error propagation, scripted-runner replay and unknown cases, `RunnerEvaluator` mean scoring and custom scorers, `ModelEvaluator`, and the deterministic dimensions (placeholder loss, coverage, structure, degenerate length/repetition, case dependence, repeatability).
- `optimizer/instruction.rs` covers improvement over the seed, repeat-run equality, validation, early stop at the seed and after one round, iteration/beam caps, placeholder preservation, the model-rewrite path with placeholder repair, fallback on unparsable/failed/empty answers, an injected evaluator, beam ranking, and `Debug`.
- `optimizer/example.rs` covers the Python example format, `num_examples == 0`, the case-count cap, score monotonicity, determinism, validation (including `MAX_EXAMPLE_NUM + 1`), early stop, model index lists and the order they impose, fallback on bad lists/failures, greedy completion of a short list, and an injected evaluator.
- `optimizer/joint.rs` checks it is never worse than either coordinate alone, both coordinates contribute, `num_examples == 0`, determinism, early stop, iteration cap, validation, placeholder preservation, the model path, and a shared injected evaluator.
- `optimizer/search.rs` covers the single sweep, dedupe/determinism, placeholder/seed contract, validation, a model rewrite winning, an unusable model answer, the seed winning, and `Debug`.
- `optimizer/trainer.rs` covers baseline early stop, validation-score monotonicity, the epoch cap, the empty-validation fallback, empty-train validation, invalid config, `evaluate`, determinism, placeholder preservation, the model path, an injected evaluator, and the deterministic default.
- `optimizer/suggestion.rs` covers placeholder parsing (including nested malformed braces), the repair, mutation ordering and idempotence, dedupe, input validation, the tie-break helpers, ordered batch scoring, model-rewrite extraction and degradation, worst-case selection, and the `key:value | …` rendering.

## Deferred

- Python's dataset samplers (`openjiuwen/dev_tools/tune/dataset/` beyond `CaseLoader`) are not ported: they schedule batches from external stores, while the port keeps the case list in memory and leaves splitting to `CaseLoader`.
- Python's `tune/chat_agent/` adapters (`ChatAgent`, `ChatConfig`) are not ported: they bind an optimizer to a Python agent runtime, whereas here an optimizer is driven by a `CaseEvaluator` the caller composes, so no agent adapter is needed.
- Python's `optimizer/adopt/` and `optimizer/prompt_search/` service modules (policy, memory, reward, drift, convergence, run log) are not ported; their observable contract — propose, evaluate, return the best — is implemented as the stateless `PromptSearchOptimizer` sweep.
- Python's `evaluator` module's LLM-judge retry loop is replaced by `LabelScorer` plus an optional custom `CaseScorer`; rubric-driven model judging lives in the crate's `judge` module instead.

## Routing Table

- `./base.rs` → `Case`, `EvaluatedCase`, validation, JSON round-trips
- `./loader.rs` → `CaseLoader`, seeded PRNG, shuffle, split, iteration
- `./tool.rs` → `ToolStub`/`ToolStubs`, `ToolCallRecord`/`ToolTrace`, `RunOutcome`, tool handlers
- `./evaluator/` → the evaluator seam (`evaluator/mod.rs` re-exports the four submodules)
- `./evaluator/runner.rs` → `PromptRunner`, `ModelPromptRunner`, `ScriptedPromptRunner`, case rendering, answer-map parsing
- `./evaluator/scorer.rs` → `CaseScorer`, `LabelScorer`, band comparison, tokenization
- `./evaluator/seam.rs` → `CaseEvaluator`, `RunnerEvaluator`, `ModelEvaluator`, score clamping
- `./evaluator/deterministic.rs` → `DeterministicEvaluator`, `DeterministicScore`, weights and dimensions
- `./evaluator/tests.rs` → the evaluator test suite
- `./optimizer.rs` → facade: tuning constants, `OptimizationResult`, `mod` declarations, re-exports, `optimize_prompt`
- `./optimizer/config.rs` → `OptimizerConfig`, bounds validation, JSON defaults
- `./optimizer/suggestion.rs` → placeholder parsing/repair, mutation bank, model-rewrite request (including the `<tools_description>` render of the case set's declared tools), tie-break, bounded scoring, worst-case selection
- `./optimizer/instruction.rs` → `InstructionOptimizer` beam search
- `./optimizer/example.rs` → `ExampleOptimizer`, `render_examples`, `format_prompt`
- `./optimizer/joint.rs` → `JointOptimizer` alternating ascent
- `./optimizer/search.rs` → `PromptSearchOptimizer` single sweep
- `./optimizer/trainer.rs` → `Trainer` train/validation loop
- `./mod.rs` → module docs, private submodules, public re-exports
- `../judge/` → rubric-driven model judging (sibling — read-only here, not a dependency of `tune`)
- `../../util.rs` → `extract_tag`, `extract_list_fence`, `extract_json_fence` (outside this node — read-only, escalate writes to the crate root)
