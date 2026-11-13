//! Evolution signals: the feedback vocabulary every evolution subsystem shares.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.signal` and the
//! signal records in `optimizer/base.py`:
//!
//! - [`EvolutionSignal`] — one piece of feedback about one target section of one
//!   operator/skill, with a category, target, excerpt, provenance and weight.
//! - [`EvolutionCategory`] / [`EvolutionTarget`] — the closed vocabularies that
//!   route a signal to the parameter it should update.
//! - [`SignalTracker`] — the in-process accumulator: dedupe, weighting, decay and
//!   deterministic ordering so optimizers see a stable signal set.
//! - Extractors — signals derived from evaluated cases, conversations and review
//!   feedback, so the online and offline paths share one representation.
//!
//! Signals are pure serde data: no model calls, no clocks, no environment reads.
