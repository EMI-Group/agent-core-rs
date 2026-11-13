//! Evolution signals: the feedback vocabulary every evolution subsystem shares.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.signal` and the
//! signal records in `optimizer/base.py`:
//!
//! - [`crate::signal::EvolutionSignal`] — one piece of feedback about one target section of one
//!   operator/skill, with a category, target, excerpt, provenance and weight.
//! - [`crate::signal::EvolutionCategory`] / [`crate::signal::EvolutionTarget`] — the closed vocabularies that
//!   route a signal to the parameter it should update.
//! - [`crate::signal::SignalTracker`] — the in-process accumulator: dedupe, weighting, decay and
//!   deterministic ordering so optimizers see a stable signal set.
//! - Extractors — signals derived from evaluated cases, conversations and review
//!   feedback, so the online and offline paths share one representation.
//! - [`crate::signal::normalize_excerpt`] / [`crate::signal::truncate_excerpt`] — the deterministic excerpt
//!   normalisation and bounding helpers the tracker and extractors share.
//!
//! Signals are pure serde data: no model calls, no clocks, no environment reads.

mod extract;
mod tracker;
mod types;

pub use extract::{
    signals_from_conversation, signals_from_evaluated_cases, signals_from_failure,
    signals_from_review_feedback, signals_from_team_feedback, MAX_SIGNALS_PER_INPUT,
    NEGATIVE_MARKERS,
};
pub use tracker::{SignalTracker, TrackedSignal};
pub use types::{
    normalize_excerpt, truncate_excerpt, EvolutionCategory, EvolutionSignal, EvolutionTarget,
    MAX_EXCERPT_CHARS,
};
