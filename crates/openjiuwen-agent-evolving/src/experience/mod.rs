//! Experience records: extraction, dedupe, scoring and lifecycle.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.experience`: the
//! record types (`experience/types.py`), the archive (`experience/archive.py`),
//! query (`experience/query.py`), scoring (`experience/scorer.py`) and the
//! lifecycle/rebuild rules (`experience/lifecycle.py`, `experience/rebuild.py`).
//!
//! - [`ExperienceRecord`] — one reusable lesson with provenance and counters.
//! - [`ExperienceArchive`] — dedupe, merge, score and query over records.
//! - [`ExperienceExtractor`] — derive candidate records from signals and
//!   trajectories.
//! - [`ExperienceLifecycle`] — promote proven records, decay and prune stale ones.
//!
//! The LLM draft pipeline and the online orchestrator are deferred; extraction and
//! ranking here are deterministic.
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
mod extract;
mod lifecycle;
mod models;

pub use archive::{
    jaccard, stopwords, tokens, AddOutcome, ExperienceArchive, ExperienceStats, MergeSummary,
    DEFAULT_CAPACITY, DEFAULT_DEDUPE_THRESHOLD,
};
pub use extract::{
    ExperienceExtractor, DEFAULT_KEYWORD_LIMIT, DEFAULT_MAX_RECORDS, DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SKILL_NAME,
};
pub use lifecycle::{ExperienceLifecycle, LifecycleConfig, LifecycleReport, STALE_DECAY_FACTOR};
pub use models::{ExperienceMatch, ExperienceQuery, ExperienceRecord, ExperienceSection};
