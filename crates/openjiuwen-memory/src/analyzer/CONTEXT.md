# analyzer

## Intent

Turns conversation turns into memory facts and decides what to write, replacing Python `openjiuwen.core.memory.process.extract` and `manage/update/mem_update_checker.py`.
Two seams (`FactExtractor`, `ConflictClassifier`), each with a deterministic implementation and a model-backed implementation that falls back to it.
The rule paths are the default and work offline; the LLM paths are opt-in and are marked as degraded whenever they fall back.
The module never touches storage — callers apply a `ConsolidationPlan` through `crate::facade::LongTermMemory` / `crate::engine::MemoryEngine`.

## API Surface

`config`: `AnalyzerConfig` (own type, `snake_case`, `default`, tolerant of unknown fields), `ExtractorKind { Rule (default), Llm }`, `ClassifierKind { Rule (default), Llm }`, the importance-tier constants `DIRECTIVE_IMPORTANCE = 0.9`, `IDENTITY_IMPORTANCE = 0.85`, `PREFERENCE_IMPORTANCE = 0.75`, `PROJECT_IMPORTANCE = 0.6`, `STATEMENT_IMPORTANCE = 0.4`, plus `from_extraction(&ExtractionConfig, input_msg_max_len)`, `from_engine(&MemoryEngineConfig)` and `validate()`.
`fact`: `ConversationTurn { role, text }`, `ExtractionInput { turns, history }` (`from_messages`, `with_history_messages`, `push_turn`, `push_user`, `from_text`, `is_empty`, `len`, `validate`, `bounded`), `ExtractedFact` (`content`, `kind`, `importance`, `tags`, `source`, `metadata`; builders, `validate`, `to_new_memory`, `From<&NewMemory>`), `ExtractionResult { facts, degraded, extractor }` (`new`, `empty`, `is_degraded`), and the `#[async_trait] trait FactExtractor`.
`rule`: `RuleExtractor { config }` (`new`, `config`), name `"rule"`.
`llm`: `LlmExtractor { model, config, fallback }` (`new`, `with_fallback`, `config`), name `"llm"`.
`parse`: `parse_fact_response(raw, &AnalyzerConfig) -> Option<Vec<ExtractedFact>>` and `parse_conflict_response(raw, &[MemoryRecord]) -> Option<Vec<ConflictVerdict>>`.
`prompt`: `EXTRACTION_SYSTEM_PROMPT`, `EXTRACTION_JSON_SCHEMA`, `CONFLICT_SYSTEM_PROMPT`, `CONFLICT_JSON_SCHEMA` (`pub const`), `MEMORY_KINDS`, `render_conversation`, `render_conflict_input`.
`conflict`: `ConflictKind { Redundant, Conflicting, Unrelated (default) }`, `ConflictVerdict { kind, target_id, reason, merged_content, degraded }`, `#[async_trait] trait ConflictClassifier`, `RuleClassifier { config }`.
`llm_classifier`: `LlmClassifier { model, config, fallback }` (`new`, `with_fallback`, `config`), name `"llm"`.
`orchestrator`: `MemoryAnalyzer` (`rule`, `from_config`/`from_engine -> Result<Self>`, `with_extractor`, `config`, `extractor_name`, `analyze`, `classifier -> Result<Arc<dyn ConflictClassifier>>`, `consolidate`), `ConsolidationPlan { actions }` (+ `adds()`), `ConsolidationAction { Add { fact_index }, Merge { fact_index, target_id, merged_content }, Replace { fact_index, target_id }, Skip { fact_index, target_id } }` serialized with an `action` tag.
`stage` (staged pipeline): the composeable multi-stage port of Python's `long_term_memory_extractor` 6-step prompt, described in `stage/CONTEXT.md`.
- `#[async_trait] trait ExtractionStage { name(); async run(&self, &StageInput, &CallOptions) -> Result<StageOutput> }`, `StageInput` (conversation, config, plan, facts, reasoning, candidates, verdicts, decision), `StageOutput` (per-stage patches + `degraded`, applied by `StageOutput::apply`).
- Plan stage: `PlanIntent`, `PlanOperation`, `ExtractionPlan { intent, topic, operations, focus_turns, rationale, degraded }`, `RulePlanStage` (default), `LlmPlanStage`, `parse_plan_response`, `PLAN_SYSTEM_PROMPT`/`PLAN_JSON_SCHEMA`, `SUMMARY_TOKEN_THRESHOLD`.
- Reasoning stage: `FactReasoning { fact_index, rationale, basis }`, `ReasoningBasis`, `RuleReasoningStage` (default, configurable metadata key), `LlmReasoningStage`, `parse_reasoning_response`, `REASONING_SYSTEM_PROMPT`/`REASONING_JSON_SCHEMA`, `DEFAULT_REASONING_KEY = "rationale"`.
- Proactive stage: `ProactiveSignal`, `ProactiveDecision { write, reason, signals, fact_indexes, importance, degraded }`, `ProactiveConfig`, `RuleProactiveStage` (default), `LlmProactiveStage`, `parse_proactive_response`, `PROACTIVE_SYSTEM_PROMPT`/`PROACTIVE_JSON_SCHEMA`.
- Pipeline: `StagedAnalyzer` (`new`, `from_config`, `from_engine`, `with_stage_config`, `with_overrides`, `with_dedupe`, `with_stages`, `analyze`, `analyze_against`, `consolidate`, `effective_config`), `StageConfig { plan, reasoning, proactive, proactive_config, reasoning_key }`, `StageKind { Rule (default), Llm }`, `StagedResult { result, plan, decision, reasoning, verdicts, stages }` (+ `facts`, `writable_facts`, `should_write`, `rationale`, `is_degraded`), `ExtractStage`, `ConflictStage`, `FactVerdict`.
- Per-scope overrides: `ScopeAnalyzerOverrides` (`new`, `with_model`, `with_embedding`, `with_extractor`, `with_classifier`, `with_min_importance`, `with_max_facts`, `with_default_importance`, `with_stages`, getters, `is_empty`, `validate`, `effective_config`); `Debug` reports handle names only.
`mod.rs` re-exports every public item above (including the whole `stage` surface); nothing else in the module is public.

## Rule Inventory

Segmentation: each turn is split into lines, markdown list prefixes are stripped, then lines are split into sentences only after `.`, `!`, `?`, `;`, `。`, `！`, `？`, `；` that end the input or are followed by whitespace, so `src/main.rs` and `1.5` stay intact.
Sentences shorter than 3 characters are dropped, content is truncated to `max_content_chars`, and turns shorter than `min_message_len` are skipped entirely.
First match wins, in this order:
(a) explicit directive — `remember`, `note that`, `keep in mind`, `don't forget`, `memorize`, `record`; the fact is the remainder after the marker, tagged `explicit`, importance `max(directive_importance, tier of the remainder)`.
(b) identity — `my <attribute> is ...`, `I am a/an ...`, `I work at/for ...`, `I live in ...`, `my favourite/preferred ...`; `MemoryType::UserProfile`, tag `identity`.
(c) preference — `I/we prefer|like|love|enjoy|hate|dislike|avoid|want|need` (including negated and `would` forms); `UserProfile`, tag `preference`.
(d) project/technology — `we/I use`, `the project/repo/service uses|runs on`, `I'm building|working on|migrating`; `MemoryType::SemanticMemory`, tag `project`.
(e) variable assignment — `snake_case_key = value`, `key -> value`, `set key to value`; `SemanticMemory` with `variable`/`value` metadata and tag `variable` (the Python variable-manager write path is deferred, so `MemoryType::Variable` is never produced).
(f) generic declarative statement — at least four words plus a copula (`is/are/was/were/has/have/contains/supports/requires`) and no stronger match; `SemanticMemory`, tag `statement`.
Batch hygiene: facts below `min_importance` are dropped, facts whose `crate::record::normalize_text` repeats are merged (first content kept, tags unioned, maximum importance), and the batch is truncated to `max_facts`.
`enabled == false` returns an empty `ExtractionResult` with no warning; `options.check()?` runs before any work so cancellation is honoured.

## Conflict Rules

`incoming.validate()?` runs first, then every existing record is classified in input order.
Redundant — `crate::recall::find_duplicate` matched; the reason names the match kind and `merged_content` comes from `crate::recall::merge_records`.
Conflicting — one of three deterministic signals: (i) a correction marker (`actually`, `no longer`, `not ... anymore`, `instead`, `updated:`, `changed to`, `switched to`, `correction`, `rather than`) plus a shared salient token; (ii) same-subject different-value (`X is/are/was/were/am Y` on both sides with an equal normalized subject and a different value; `merged_content` is the incoming statement); (iii) negation flip (negator-stripped token sets with `jaccard >= dedupe.jaccard_threshold` and exactly one side negated).
Unrelated — none of the above; this is the verdict for every record a model-backed classifier did not mention, and a candidate the model never saw because it fell past `max_candidates` also keeps an unrelated verdict with a `degraded` reason naming the cap.
Verdicts describe a relation only; they never delete or mutate anything.

## Fallback and Degradation Contract

`LlmExtractor` and `LlmClassifier` check `options.check()?` before the first model call so a cancelled call returns `Error::Cancelled` with zero model invocations.
They retry `model_retries + 1` attempts in total, re-raising `ErrorKind::Cancelled` immediately and logging every other failure with `tracing::warn!`.
Unparseable or empty model output counts as a failed attempt, exactly like a provider error.
When the budget is spent the deterministic implementation runs and the degradation is recorded: `ExtractionResult::degraded` carries `llm extraction fell back to rules: <reason>` and `extractor` is retargeted to `"rule"`, while every `ConflictVerdict` carries `llm classification fell back to rules: <reason>`.
The first failure reason is the one reported, so a retry storm never hides the original cause.
`MemoryAnalyzer::from_config`/`from_engine` reject `ExtractorKind::Llm` without a model with a typed `Unsupported`, and `MemoryAnalyzer::classifier` does the same for `ClassifierKind::Llm`: silently running the rule implementation would misreport what produced the facts or verdicts, so a caller asks for rules on purpose (`MemoryAnalyzer::rule`, `ClassifierKind::Rule`).
No degraded path returns an empty result silently: a fallback always yields the rule facts/verdicts plus the recorded reason.
A candidate the model never saw because it fell past `AnalyzerConfig::max_candidates` keeps an unrelated verdict that also carries a `degraded` reason naming the cap, so "the model found no conflict" is distinguishable from "the model never looked".
The staged variants follow the same contract: `LlmPlanStage` records `llm plan fell back to rules: <reason>` on `ExtractionPlan::degraded`, `LlmReasoningStage` records `llm reasoning fell back to rules: <reason>` (or `llm reasoning answered N of M facts; rule rationales filled the rest` when it merged a partial answer), and `LlmProactiveStage` records `llm proactive decision fell back to rules: <reason>` on `ProactiveDecision::degraded`.
`StagedResult::is_degraded()` is true when the extraction result, the plan, the decision, or any stage output carries a reason, so a caller can surface one flag.

## Staged Pipeline

`StagedAnalyzer` runs an ordered `Vec<Arc<dyn ExtractionStage>>` over one mutable `StageInput`; each stage returns a `StageOutput` patch that `StageOutput::apply` merges into the state.
Default sequence: `plan` → `extract` → `reasoning` → `proactive` → `conflict`; `with_stages` replaces the whole sequence with caller-supplied stages (composable, not additive).
- plan (`RulePlanStage`, name `"plan"`): picks the `PlanIntent`, `PlanOperation` set, `focus_turns` and a rationale. Intent order: correction marker → directive marker → variable assignment → preference → identity → project → general; operations are `ResolveReferences` (history or pronoun), `NormalizeTime` (relative-time phrase), `ExtractFacts` (always), `ExtractVariables` (assignment present), `ExtractSummary` (`SUMMARY_TOKEN_THRESHOLD = 64` tokens), `HandleDirectives` (correction/update marker). `focus_turns` is the last `min(history depth, 2)` turns plus any correction/directive turn; `selected_turns` falls back to every turn when the plan has no focus.
- extract (`ExtractStage`): delegates to the wrapped `MemoryAnalyzer`, honours `plan.extracts_facts()` (no facts when false), narrows the conversation to `selected_turns`, then applies the `min_importance` and `max_facts` filters.
- reasoning (`RuleReasoningStage`, name `"reasoning"`): one `FactReasoning` per fact, in fact order, basis derived from the fact tags (`explicit`/`identity`/`preference`/`project`/`variable`/`statement`, `planned` when the plan is actionable, `unknown` otherwise). Attaches the rationale under `DEFAULT_REASONING_KEY = "rationale"` (configurable) plus `metadata["plan_intent"]`; a blank key is `Error::Validation`.
- proactive (`RuleProactiveStage`, name `"proactive"`): see `ProactiveConfig` — `importance_threshold = 0.7`, `min_transcript_chars = 12`, `repetition_threshold = 0.6` (token Jaccard against history), `enabled = true`. A write needs a promoting signal (`ExplicitDirective`, `Correction`, `Decision`, `Preference`), a repetition hit, or a fact at/above the threshold; the skip reason always names the strongest fact's importance and the threshold.
- conflict (`ConflictStage`): one `FactVerdict` per (fact, candidate) pair over the capped `max_candidates`, reusing the injected `ConflictClassifier` and `DedupeConfig`; a fact whose content is blank (only reachable with a caller-supplied stage sequence) is skipped and reported on the stage's `degraded` reason instead of only logged.

`ScopeAnalyzerOverrides` carries per-scope handles (`model`, `embedding`), `ExtractorKind`/`ClassifierKind`, `min_importance`/`max_facts`/`default_importance`, and a full `StageConfig`; `effective_config(&AnalyzerConfig)` merges them and `StagedAnalyzer::build_stages` rebuilds the stage sequence from the effective config, so an override of `extractor`/`classifier`/stage kinds really changes which implementation runs. The embedding handle is carried only — the analyzer never embeds.
Per-scope *resolution* lives outside this module: `crate::facade::handles::ScopeOverrideHandles` resolves a stored document against the registered named handles, `crate::facade::overrides` applies it per call, and `crate::engine`'s `*_with_embedding` entry points consume the embedding handle.

## Config Defaults

`enabled = true`, `extractor = Rule`, `classifier = Rule`, `max_facts = 8`, `min_message_len = 8`, `max_input_chars = 8192` (the engine's `input_msg_max_len`), `max_history_turns = 2`, `model_retries = 3`, `default_importance = 0.5`, `min_importance = 0.0`, `max_content_chars = 512`, `max_candidates = 8`.
Importance tiers: `directive_importance = 0.9`, `identity_importance = 0.85`, `preference_importance = 0.75`, `project_importance = 0.6`, `statement_importance = 0.4`.
`validate()` rejects zero `max_facts`/`max_input_chars`/`max_content_chars`/`max_candidates` and non-finite or out-of-`0.0..=1.0` importances with `Error::Validation` naming the field and value; `min_message_len = 0` is legal and disables the minimum.
`from_extraction`/`from_engine` copy the shared engine knobs (`enabled`, `max_facts`, `min_message_len`, input length, history depth, retries, `importance_default`) so analyzer and engine settings cannot drift.

## Constraints

Errors are `openjiuwen_common::error::{Error, Result}`: `Error::Validation` for bad input/options/config, `Error::Cancelled` for a fired cancellation token, and provider errors with their original kind when a model-backed call is not retried away.
Regexes are compiled once into `std::sync::LazyLock<Option<Regex>>` through `rule::compile`, which logs and disables the rule instead of panicking, so arbitrary input never panics.
No `unwrap`/`expect`/`panic!` outside `#[cfg(test)]`; `tracing` only (target `openjiuwen_memory::analyzer`); no `println!`; no `unsafe`.
Every public item is documented and every public data type is `serde`-serializable, `snake_case`, and tolerant of unknown fields.
Text handling is char-boundary safe (`truncate_chars`), so huge or multi-byte input is bounded rather than sliced blindly.
Async work stays on tokio and holds no lock across `.await`; the module is stateless apart from the compiled regexes and `Arc<dyn Model>`/`Arc<dyn FactExtractor>` fields.
Files stay around ≤1000 lines and split by responsibility: config, fact model, rule extraction, LLM extraction, parsing, prompts, conflicts, LLM classification, orchestration, and inside `stage/`: the seam and state, plan, reasoning, proactive, pipeline, overrides.
Stage prompts embed their JSON schema inline (a `const` cannot be interpolated into `concat!`), and stage parsers reuse `parse::payload` for fence/slice tolerance.

## Deferred

Vendor LLM extraction quality tuning (provider-specific prompts, few-shot exemplars, model-specific schemas) — the shared prompt keeps the Python intent, the staged prompts keep its shape.
Python's `process/extract` prompt fidelity beyond the shapes: subject locking and absolute-date rewriting are detected by the plan (`resolve_references`, `normalize_time`) but are not applied to the fact text, and the plan/reasoning stages do not re-run extraction with a second prompt.
Per-scope override *resolution* is not this module's job: the document store, the named handle registry and the per-call wiring live in `crate::facade` (`overrides.rs`, `handles.rs`), and the analyzer only consumes a resolved `ScopeAnalyzerOverrides`.
Python's variable-manager write path (`MemoryType::Variable` records) — variables are captured as `semantic_memory` facts carrying `variable`/`value` metadata.
`dreaming/` background consolidation, team/`agent_teams` memory analysis, and schema migrations for extracted facts.
Constraint/conflict *resolution* beyond the plan (`Replace`/`Merge`/`Skip` decisions are returned, not executed) — the engine applies them.
Proactive memory is a decision only: the stage never writes, schedules, or dedupes, and no Python `mem_instruct` UPDATE/DELETE/ADD operation is emitted.

## Routing Table

- `./config.rs` → `AnalyzerConfig`, `ExtractorKind`, `ClassifierKind`, importance tiers, `from_extraction`/`from_engine`
- `./fact.rs` → `ConversationTurn`, `ExtractionInput`, `ExtractedFact`, `ExtractionResult`, `FactExtractor`
- `./rule.rs` → regex inventory and `RuleExtractor`
- `./llm.rs` → `LlmExtractor` (retry, fallback, degradation)
- `./conflict.rs` → `ConflictKind`, `ConflictVerdict`, `ConflictClassifier`, `RuleClassifier`
- `./llm_classifier.rs` → `LlmClassifier`
- `./orchestrator.rs` → `MemoryAnalyzer`, `ConsolidationPlan`/`ConsolidationAction`
- `./parse.rs` → `parse_fact_response`, `parse_conflict_response`
- `./prompt.rs` → extraction/conflict prompts, schemas, conversation rendering
- `./stage/` → staged pipeline: `ExtractionStage` seam, plan/reasoning/proactive stages and their LLM variants, `StagedAnalyzer`, `ScopeAnalyzerOverrides` (`stage/CONTEXT.md`)
- `./mod.rs` → module declaration, public re-exports, lint policy

## Test Strategy

Unit tests live in `#[cfg(test)] mod tests` in every file and are fully offline: `openjiuwen_test_support::ScriptedModel` stands in for the provider.
Rule extraction covers preference/identity/project/directive/variable/generic facts from a short dialogue, sentence-vs-filename splitting, batch dedupe, the `max_facts` cap, the `min_importance` filter, and huge-input bounding.
LLM extraction covers fenced-JSON success, malformed output falling back with `degraded` recorded, provider error falling back, and cancellation returning `Error::Cancelled` with `call_count() == 0`.
Parsing covers array/object wrappers, alias fields, clamping, malformed entries and the Python conflict item shape.
Conflict classification covers all three conflicting signals, redundancy, unrelated, and verdict ordering; the LLM classifier covers parse success, malformed → fallback with `degraded` on every verdict, and candidates past `max_candidates` keeping a degraded unrelated verdict.
The constructors are covered too: `MemoryAnalyzer::from_config` (`ExtractorKind::Llm` with and without a model) and `MemoryAnalyzer::classifier` (`ClassifierKind::Llm` with and without a model) return the implementation or the typed `Unsupported`.
`consolidate` is covered for every `DedupePolicy` and for `dedupe.enabled == false`.
The staged pipeline is covered in `stage/` (47 unit tests): plan selection across representative transcripts (correction/directive/variable/preference/identity/project/general, operation sets, focus turns, disabled analyzer, `selected_turns` fallback), ordered reasoning with metadata attachment and configurable/validated key, proactive write-vs-skip decisions with their reason strings (directive, correction, decision, preference, repetition, short transcript, empty facts, disabled), model-backed stage success, unparseable output → deterministic fallback with `degraded`, cancellation → `Error::Cancelled` with `call_count() == 0`, prompt/schema embedding, serde round trips and unknown-field tolerance for every stage type.
`stage/pipeline.rs` additionally covers end-to-end parity with `MemoryAnalyzer` (same facts for the same input), plan-skipped extraction, conflict classification against existing records, override threshold dropping low-importance facts, an override switching the effective extractor to the model-backed stage, `StageKind::Llm` without a model running the rule stages with the reason recorded on `StagedResult::result.degraded`, an empty-content fact reported as skipped by `ConflictStage`, a custom stage sequence replacing the built-in one, cancelled pipelines making zero model calls, `consolidate` parity, `StagedResult` JSON round-trip, and `StageConfig` validation.
Gates: `cargo fmt -p openjiuwen-memory`, `cargo check -p openjiuwen-memory --offline`, `cargo test -p openjiuwen-memory --offline --lib analyzer`, `cargo clippy -p openjiuwen-memory --all-targets --all-features --offline -- -D warnings`, `cargo doc -p openjiuwen-memory --no-deps`.
