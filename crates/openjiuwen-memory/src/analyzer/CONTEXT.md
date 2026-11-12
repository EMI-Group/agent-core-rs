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
`orchestrator`: `MemoryAnalyzer` (`rule`, `from_config`, `with_extractor`, `config`, `extractor_name`, `analyze`, `classifier`, `consolidate`), `ConsolidationPlan { actions }` (+ `adds()`), `ConsolidationAction { Add { fact_index }, Merge { fact_index, target_id, merged_content }, Replace { fact_index, target_id }, Skip { fact_index, target_id } }` serialized with an `action` tag.
`mod.rs` re-exports every public item above; nothing else in the module is public.

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
Unrelated — none of the above; this is the verdict for every record a model-backed classifier did not mention.
Verdicts describe a relation only; they never delete or mutate anything.

## Fallback and Degradation Contract

`LlmExtractor` and `LlmClassifier` check `options.check()?` before the first model call so a cancelled call returns `Error::Cancelled` with zero model invocations.
They retry `model_retries + 1` attempts in total, re-raising `ErrorKind::Cancelled` immediately and logging every other failure with `tracing::warn!`.
Unparseable or empty model output counts as a failed attempt, exactly like a provider error.
When the budget is spent the deterministic implementation runs and the degradation is recorded: `ExtractionResult::degraded` carries `llm extraction fell back to rules: <reason>` and `extractor` is retargeted to `"rule"`, while every `ConflictVerdict` carries `llm classification fell back to rules: <reason>`.
The first failure reason is the one reported, so a retry storm never hides the original cause.
`MemoryAnalyzer::from_config` with `ExtractorKind::Llm` and no model, or `ClassifierKind::Llm` and no model, builds the rule implementation — a documented degradation, never a silent no-op and never an `Error::Unsupported`.
No degraded path returns an empty result silently: a fallback always yields the rule facts/verdicts plus the recorded reason.

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
Files stay around ≤1000 lines and split by responsibility: config, fact model, rule extraction, LLM extraction, parsing, prompts, conflicts, LLM classification, orchestration.

## Deferred

Vendor LLM extraction quality tuning (provider-specific prompts, few-shot exemplars, model-specific schemas) — the shared prompt keeps the Python intent.
Proactive memory (Python's proactive analyzer that decides *when* to remember) and the `process/extract` LLM analyzer's plan/reasoning stages.
Per-scope model and embedding override resolution; the analyzer takes one injected `Arc<dyn Model>`.
Python's variable-manager write path (`MemoryType::Variable` records) — variables are captured as `semantic_memory` facts carrying `variable`/`value` metadata.
`dreaming/` background consolidation, team/`agent_teams` memory analysis, and schema migrations for extracted facts.
Constraint/conflict *resolution* beyond the plan (`Replace`/`Merge`/`Skip` decisions are returned, not executed) — the engine applies them.

## Test Strategy

Unit tests live in `#[cfg(test)] mod tests` in every file and are fully offline: `openjiuwen_test_support::ScriptedModel` stands in for the provider.
Rule extraction covers preference/identity/project/directive/variable/generic facts from a short dialogue, sentence-vs-filename splitting, batch dedupe, the `max_facts` cap, the `min_importance` filter, and huge-input bounding.
LLM extraction covers fenced-JSON success, malformed output falling back with `degraded` recorded, provider error falling back, and cancellation returning `Error::Cancelled` with `call_count() == 0`.
Parsing covers array/object wrappers, alias fields, clamping, malformed entries and the Python conflict item shape.
Conflict classification covers all three conflicting signals, redundancy, unrelated, and verdict ordering; the LLM classifier covers parse success plus malformed → fallback with `degraded` on every verdict.
`consolidate` is covered for every `DedupePolicy` and for `dedupe.enabled == false`.
Every public data type has a serde round-trip and an unknown-field-tolerance test.
Gates: `cargo fmt -p openjiuwen-memory`, `cargo check -p openjiuwen-memory --offline`, `cargo test -p openjiuwen-memory --offline --lib analyzer`, `cargo clippy -p openjiuwen-memory --all-targets --offline -- -D warnings`.
