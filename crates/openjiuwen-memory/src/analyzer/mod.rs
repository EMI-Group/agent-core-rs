//! Memory analyzer: turn conversation input into memory facts, and decide what to write.
//!
//! This module is the Rust counterpart of the Python
//! `openjiuwen.core.memory.process.extract` package and
//! `manage/update/mem_update_checker.py`. It exposes two seams and a deterministic
//! implementation of each, plus an orchestrator that wires them together:
//!
//! - [`FactExtractor`] — [`RuleExtractor`] (deterministic rules) and [`LlmExtractor`]
//!   (model-backed, rules as fallback);
//! - [`ConflictClassifier`] — [`RuleClassifier`] (deterministic redundancy/conflict
//!   signals) and [`LlmClassifier`] (model-backed, rules as fallback);
//! - [`MemoryAnalyzer`] — config, extractor, classifier factory, and
//!   [`MemoryAnalyzer::consolidate`], which turns facts into a [`ConsolidationPlan`].
//!
//! # Rule inventory
//!
//! [`RuleExtractor`] splits each turn into lines and then sentences (only after `.`,
//! `!`, `?`, `;`, `。`, `！`, `？`, `；` that end the line or are followed by whitespace,
//! so `src/main.rs` stays intact) and applies, first match wins:
//!
//! 1. explicit directive (`remember`, `note that`, `keep in mind`, `don't forget`,
//!    `memorize`, `record`) — the fact is the remainder after the marker, tagged
//!    `explicit`, at the directive tier or the tier of the remainder, whichever is higher;
//! 2. identity (`my name/age/email/... is`, `I am a/an`, `I work at/for`, `I live in`,
//!    `my favourite/preferred ...`) — `user_profile`, tag `identity`;
//! 3. preference (`I/we prefer|like|love|enjoy|hate|dislike|avoid|want|need`, including
//!    the negated and `would` forms) — `user_profile`, tag `preference`;
//! 4. project/technology (`we/I use`, `the project/repo/service uses|runs on|...`,
//!    `I'm building|working on|migrating`) — `semantic_memory`, tag `project`;
//! 5. variable assignment (`snake_case_key = value`, `key -> value`, `set key to value`)
//!    — `semantic_memory` with `variable`/`value` metadata, tag `variable` (Python's
//!    variable store is deferred, so `MemoryType::Variable` is never written);
//! 6. generic declarative statement (four or more words plus a copula) —
//!    `semantic_memory`, tag `statement`.
//!
//! Sentences under three characters are dropped, content is truncated to
//! `max_content_chars`, turns under `min_message_len` are skipped, every turn is bounded
//! to `max_input_chars`, facts below `min_importance` are dropped, duplicates inside one
//! batch are merged (tags unioned, highest importance kept), and at most `max_facts`
//! facts are returned.
//!
//! # Fallback and degradation contract
//!
//! A model-backed component never fails silently and never pretends the model worked:
//! provider errors and unusable answers are retried up to `model_retries` times, and once
//! the budget is spent the deterministic implementation runs and the result records why
//! (`ExtractionResult::degraded` for extraction, `ConflictVerdict::degraded` on every
//! verdict for classification). A cancelled [`CallOptions`] returns
//! [`openjiuwen_common::error::Error::Cancelled`] before the first model call.
//!
//! # Deviations from Python
//!
//! - Extraction is rule-based by default instead of prompt-only, so memory works offline
//!   and deterministically; the LLM path keeps the Python prompt's intent (JSON-only,
//!   memory kinds, keep the input language).
//! - Conflict classification is deterministic by default: exact/cosine/Jaccard
//!   duplicates, correction markers, same-subject-different-value, and negation flips.
//!   New information never deletes anything; a verdict only describes the relation.
//! - The analyzer is an explicit value (config + extractor) rather than a global
//!   singleton, and it never touches storage: callers apply the plan.
//! - Planned-with-no-model behaviour: `ExtractorKind::Llm` without a supplied model
//!   degrades to the rule extractor instead of failing, mirroring the Python behaviour of
//!   running without a memory model.
mod config;
mod conflict;
mod fact;
mod llm;
mod llm_classifier;
mod orchestrator;
mod parse;
mod prompt;
mod rule;

pub use config::{
    AnalyzerConfig, ClassifierKind, ExtractorKind, DIRECTIVE_IMPORTANCE, IDENTITY_IMPORTANCE,
    PREFERENCE_IMPORTANCE, PROJECT_IMPORTANCE, STATEMENT_IMPORTANCE,
};
pub use conflict::{ConflictClassifier, ConflictKind, ConflictVerdict, RuleClassifier};
pub use fact::{ConversationTurn, ExtractedFact, ExtractionInput, ExtractionResult, FactExtractor};
pub use llm::LlmExtractor;
pub use llm_classifier::LlmClassifier;
pub use orchestrator::{ConsolidationAction, ConsolidationPlan, MemoryAnalyzer};
pub use parse::{parse_conflict_response, parse_fact_response};
pub use prompt::{
    render_conflict_input, render_conversation, CONFLICT_JSON_SCHEMA, CONFLICT_SYSTEM_PROMPT,
    EXTRACTION_JSON_SCHEMA, EXTRACTION_SYSTEM_PROMPT, MEMORY_KINDS,
};
pub use rule::RuleExtractor;
