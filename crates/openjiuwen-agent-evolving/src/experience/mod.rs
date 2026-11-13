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
//!
//! The LLM draft pipeline and the online orchestrator are deferred; extraction and
//! ranking here are deterministic.
