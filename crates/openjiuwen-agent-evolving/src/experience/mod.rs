//! Experience records: extraction, dedupe, scoring, lifecycle, drafting and
//! online orchestration.
//!
//! Ports Python `openjiuwen.agent_evolving.experience`: the record types
//! (`experience/types.py`), the archive (`experience/archive.py`), query
//! (`experience/query.py`), scoring (`experience/scorer.py`), the
//! lifecycle/rebuild rules (`experience/lifecycle.py`, `experience/rebuild.py`),
//! the agent-facing draft schema (`experience/draft_schema.py`), the LLM
//! experience optimizer (`optimizer/skill_call/experience_optimizer.py` and
//! `experience_draft_parser.py`) and the online pipeline coordinator
//! (`experience/online_orchestrator.py`).
//!
//! - [`ExperienceRecord`] — one reusable lesson with provenance and counters.
//! - [`ExperienceArchive`] — dedupe, merge, score and query over records.
//! - [`ExperienceExtractor`] — derive candidate records from signals and
//!   trajectories.
//! - [`ExperienceLifecycle`] — promote proven records, decay and prune stale ones.
//! - [`ExperienceDrafter`] — the LLM draft pipeline with a deterministic
//!   fallback.
//! - [`OnlineEvolutionOrchestrator`] — precheck, draft, stage or persist.
//!
//! Drafting never depends on a model: [`ExperienceDrafter`] answers from
//! [`ExperienceExtractor`] evidence when no model is installed, and falls back to
//! it on every model failure.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_evolving::experience::{
//!     AddOutcome, ExperienceArchive, ExperienceQuery, ExperienceRecord, ExperienceSection,
//! };
//!
//! let record = ExperienceRecord::new(
//!     "exp-1",
//!     "pdf-report",
//!     ExperienceSection::Tool,
//!     "raise the read timeout before parsing large PDFs",
//! )
//! .map(|record| record.with_keywords(["timeout", "pdf"]).record_success())?;
//!
//! let mut archive = ExperienceArchive::new();
//! assert!(matches!(archive.add(record)?, AddOutcome::Inserted(_)));
//!
//! let hits = archive.query(&ExperienceQuery::new().with_keywords(["timeout"]));
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].matched_keywords, vec!["timeout".to_owned()]);
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

mod archive;
mod draft;
mod drafter;
mod extract;
mod lifecycle;
mod models;
mod orchestrator;
#[cfg(test)]
mod pipeline_tests;
mod prompts;

pub use archive::{
    jaccard, stopwords, tokens, AddOutcome, ExperienceArchive, ExperienceStats, MergeSummary,
    DEFAULT_CAPACITY, DEFAULT_DEDUPE_THRESHOLD,
};
pub use draft::{
    normalize_evolve_draft, normalize_simplify_draft, normalize_subject, normalize_subject_kind,
    validate_simplify_record_refs, DraftExperience, DraftSection, EvolutionSubject,
    EvolutionSubjectKind, EvolutionTarget, EvolveDraft, SimplifyAction, SimplifyActionKind,
    SimplifyDraft, SUMMARY_LIMIT, SUPPORTED_SUBJECT_KINDS,
};
pub use drafter::{
    draft_keywords, existing_summaries, extract_json_value, parse_draft_text,
    summarize_skill_content, DraftInput, DraftParse, ExperienceDrafter, DEFAULT_MAX_ITEMS,
    DEFAULT_MAX_SKILL_CHARS, DEFAULT_MAX_SUMMARY_CHARS, DRAFT_PROMPT, FORMAT_REPAIR_PROMPT,
    SIMPLIFY_KEEP_SCORE, SIMPLIFY_PROMPT,
};
pub use extract::{
    ExperienceExtractor, DEFAULT_KEYWORD_LIMIT, DEFAULT_MAX_RECORDS, DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SKILL_NAME,
};
pub use lifecycle::{ExperienceLifecycle, LifecycleConfig, LifecycleReport, STALE_DECAY_FACTOR};
pub use models::{ExperienceMatch, ExperienceQuery, ExperienceRecord, ExperienceSection};
pub use orchestrator::{
    initial_confidence, EvolutionRequest, EvolutionStatus, OnlineEvolutionOrchestrator,
    OnlineEvolutionResult, DEFAULT_INITIAL_SCORE, DEFAULT_SOURCE, INITIAL_SCORE_BY_SIGNAL,
};
