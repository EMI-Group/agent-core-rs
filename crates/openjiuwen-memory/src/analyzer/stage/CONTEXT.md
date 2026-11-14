# analyzer::stage

## Intent

The staged, composable port of Python's `openjiuwen.core.memory.process.extract` pipeline (`long_term_memory_extractor.py`'s 6-step prompt, `generation.py`'s proactive `mem_instruct` path, `memory_analyzer.py`).
One `ExtractionStage` seam, five built-in stages (plan, extract, reasoning, proactive, conflict), deterministic rule implementations by default and model-backed variants that fall back to the rules.
`StagedAnalyzer` runs the stages in order over one mutable `StageInput`; `MemoryAnalyzer` keeps its existing two-seam API untouched, so both entry points coexist.
The directory sits inside `crate::analyzer` and re-exports nothing outside it — `analyzer::mod` is the public door.

## API Surface

`mod.rs`: `StageInput` (`new`, `with_candidates`, `fact_count`; serde `default`), `StageOutput` (`of`, `with_plan`, `with_facts`, `with_reasoning`, `with_extractor`, `with_decision`, `with_verdicts`, `degraded`, `is_degraded`, `apply`), `#[async_trait] trait ExtractionStage { fn name(&self) -> &str; async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> }`, `pub(crate) fn degraded_reasons` (joins non-empty reasons with `"; "`).
`plan.rs`: `PlanIntent` (correction/directive/variable/preference/identity/project/general/none, `as_str`, `is_actionable`, `parse` with aliases), `PlanOperation` (resolve_references/normalize_time/extract_facts/extract_variables/extract_summary/handle_directives), `ExtractionPlan` (`empty`, `extracts_facts`, `has`, `is_actionable`, `selected_turns`), `RulePlanStage`, `LlmPlanStage`, `PLAN_SYSTEM_PROMPT`, `PLAN_JSON_SCHEMA`, `parse_plan_response`, `SUMMARY_TOKEN_THRESHOLD`.
`reasoning.rs`: `ReasoningBasis` (explicit/identity/preference/project/variable/statement/planned/modeled/unknown), `FactReasoning { fact_index, rationale, basis }`, `RuleReasoningStage` (`new`, `with_metadata_key`, `key`, `config`, `reason`, `rationale`, `attach`), `LlmReasoningStage`, `REASONING_SYSTEM_PROMPT`, `REASONING_JSON_SCHEMA`, `parse_reasoning_response`, `DEFAULT_REASONING_KEY`.
`proactive.rs`: `ProactiveSignal` (explicit_directive/correction/decision/preference/repetition/high_importance, `promotes_batch`, `promotes_batch_write`), `ProactiveConfig` (`enabled`, `importance_threshold`, `min_transcript_chars`, `repetition_threshold`, `validate`), `ProactiveDecision` (`write`, `reason`, `signals`, `fact_indexes`, `importance`, `degraded`, `skip`, `is_degraded`), `RuleProactiveStage`, `LlmProactiveStage`, `PROACTIVE_SYSTEM_PROMPT`, `PROACTIVE_JSON_SCHEMA`, `parse_proactive_response`.
`pipeline.rs`: `StageKind { Rule (default), Llm }`, `StageConfig { plan, reasoning, proactive, proactive_config, reasoning_key }` (+ `validate`), `FactVerdict`, `ExtractStage`, `ConflictStage`, `StagedResult`, `StagedAnalyzer` (`new`, `from_config`, `from_engine`, `with_stage_config`, `with_overrides`, `with_dedupe`, `with_stages`, `config`, `overrides`, `stage_config`, `effective_config`, `analyze`, `analyze_against`, `consolidate`).
`overrides.rs`: `ScopeAnalyzerOverrides` (builders `with_model`/`with_embedding`/`with_extractor`/`with_classifier`/`with_min_importance`/`with_max_facts`/`with_default_importance`/`with_stages`, getters, `is_empty`, `validate`, `effective_config`); manual `Debug` prints handle names only.

## Stage Contract

Stages never mutate the shared state: they return a `StageOutput` and `StageOutput::apply` merges only the fields the stage set, so a stage can be reordered or omitted safely.
`StagedAnalyzer` always runs `plan` first (it fills `StageInput::plan`, defaulting to `ExtractionPlan::default()` for a custom sequence), and every stage receives the same `&CallOptions`, so a cancelled token short-circuits before any model call.
`extract` honours `plan.extracts_facts()` and narrows turns with `plan.selected_turns`; `reasoning` runs after `extract` (one `FactReasoning` per fact, in index order); `proactive` runs after `reasoning` and sees the fact importances; `conflict` runs last over `StageInput::candidates` capped by `max_candidates`, emitting one `FactVerdict` per (fact, candidate) pair.
`StagedResult::is_degraded()` is true if the extraction result, the plan, the decision, or any stage reason carries a degradation, and `StagedResult::staged()` reports the executed stage names in order.

## Degradation

Every model-backed stage checks `options.check()?` before its first call (zero invocations when already cancelled), retries `config.model_retries + 1` times, and treats unparseable/empty output as a failed attempt.
On failure it runs its deterministic counterpart and records `llm <stage> fell back to rules: <reason>` on its own output (`ExtractionPlan::degraded`, `ProactiveDecision::degraded`, the reasoning output's `degraded`) and logs via `tracing::warn!`.
A partial LLM reasoning answer is merged rather than discarded: the answered indices keep their model rationale and the rest are filled by rules, with `llm reasoning answered N of M facts; rule rationales filled the rest`.
`StageKind::Llm` without an injected model builds the rule stage and logs a warning — a documented degradation, never a panic or `Error::Unsupported`.

## Constraints

Same crate rules as the parent module: `openjiuwen_common::error::{Error, Result}`, no `unwrap`/`expect`/`panic!` outside tests, no `println!`, no `unsafe`, `tracing` target `openjiuwen_memory::analyzer`, serde `snake_case` with unknown-field tolerance, cancellation through `CallOptions`.
Regexes are compiled once in `std::sync::LazyLock<Option<Regex>>` and a failed compile disables the rule with a warning instead of panicking.
Text is bounded with `truncate_chars` at `max_input_chars`/`max_content_chars`, so oversized or multi-byte input never slices a char boundary.
No stage performs I/O besides the optional model call; nothing is stored, embedded, written, or scheduled here.
Files stay around ≤1000 lines (`plan.rs` is slightly above because its 8 unit tests live beside the stage); the split is seam+state, plan, reasoning, proactive, pipeline, overrides.
Stage prompts repeat their JSON schema inline because `concat!` accepts literals only.

## Known Issues

`RulePlanStage` detects reference and time signals (`PRONOUN`, `RELATIVE_TIME`) to select `ResolveReferences`/`NormalizeTime`, but no stage rewrites the fact text from those signals — the operations are advisory metadata, matching the parent module's documented gap.
`ScopeAnalyzerOverrides` carries an `Arc<dyn Embedding>` that nothing in this module calls; resolving per-scope overrides and feeding the embedding to the write path is facade/engine work.

## Routing Table

- `./mod.rs` → `StageInput`, `StageOutput`, `ExtractionStage` trait, module wiring
- `./plan.rs` → `PlanIntent`, `PlanOperation`, `ExtractionPlan`, `RulePlanStage`, `LlmPlanStage`, plan prompt/schema/parser
- `./reasoning.rs` → `ReasoningBasis`, `FactReasoning`, `RuleReasoningStage`, `LlmReasoningStage`, reasoning prompt/schema/parser
- `./proactive.rs` → `ProactiveSignal`, `ProactiveConfig`, `ProactiveDecision`, `RuleProactiveStage`, `LlmProactiveStage`, proactive prompt/schema/parser
- `./pipeline.rs` → `StageKind`, `StageConfig`, `ExtractStage`, `ConflictStage`, `StagedAnalyzer`, `StagedResult`
- `./overrides.rs` → `ScopeAnalyzerOverrides` and the facade/engine wiring notes

## Test Strategy

47 offline `#[cfg(test)] mod tests` across the six files, using `openjiuwen_test_support::ScriptedModel`; no network, no credentials, no wall-clock ordering.
Plan: intent/operation selection across representative transcripts, disabled analyzer, focus-turn selection and fallback, serde round trip plus intent aliases, prompt/schema embedding, LLM parse/fallback/cancellation.
Reasoning: ordered attachment and metadata, configurable and validated metadata key, empty input, partial-merge, unparseable fallback, cancellation, tolerant parsing.
Proactive: directive/correction/decision/preference/repetition writes, skip reasons for short/empty/disabled input, LLM parse/fallback/cancellation, config validation and decision serde.
Pipeline: parity with `MemoryAnalyzer`, plan-skipped extraction, conflict against existing records, override threshold and extractor switching, `StageKind::Llm` without a model, custom stage sequence, cancellation with zero model calls, `consolidate` parity, `StagedResult` JSON round trip, `StageConfig` validation.
Gates: `cargo test -p openjiuwen-memory --offline --lib analyzer::stage`, `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps`.
