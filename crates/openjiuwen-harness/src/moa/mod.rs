//! Mixture-of-agents (`MoA`) consultation: multi-proposal generation, verification and
//! integration.
//!
//! Ports `openjiuwen/harness/moa/` and `openjiuwen/harness/rails/moa_rail.py` from the
//! Python SDK. Optional reference consultation makes a `DeepAgent` ask several injected
//! reference models for independent proposals, rank them offline, and fold the result into
//! one answer.
//!
//! The module is self-contained and opens no connection of its own: every model handle is
//! injected and every HTTP path belongs to the caller. Python's `ContextVar` state is
//! replaced by explicit ownership, so a host holds [`MoaState`] for the lifetime of one
//! invocation and passes it around.
//!
//! Module layout:
//!
//! - `config` — [`MoaConfig`], [`VerifierConfig`], [`JevConfig`], [`MoaMode`],
//!   [`VerifierSelection`] and the Python-parity validation rules
//! - `state` — [`TurnKey`], [`ProposalStore`], [`MoaState`], [`MoaSnapshot`],
//!   [`ConsultationOutcome`] and [`ConsultationHistory`]
//! - `redaction` — [`redact`], the shared credential redaction of verification inputs
//! - `prompts` — the reference/aggregator instructions, proposal rendering, transcript
//!   flattening, request builders and [`moa_section`]
//! - `consultation` — bounded concurrent proposer fan-out over the configured models
//! - `verifier` — the progress-verifier seam, structural and model-driven verification,
//!   and proposal ranking
//! - `aggregate` — labelled or model-synthesised aggregation of the proposals
//! - `rail` — the harness rail that publishes [`moa_section`] and holds the last outcome
//! - `builtin_skills`/`skill_mount` — the embedded `consult-reference-models` skill
//!   descriptor and its mounting into a [`crate::skills::SkillLibrary`]
//!
//! Everything is offline by default: configuration validation only reads the optional
//! consultation skill directory, the prompts are pure text, and no function here logs
//! secrets or writes to stdout.
// The public names mirror the Python contract and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

pub mod aggregate;
pub mod builtin_skills;
pub mod config;
pub mod consultation;
pub mod prompts;
pub mod rail;
pub mod redaction;
pub mod skill_mount;
pub mod state;
pub mod verifier;

#[cfg(test)]
mod tests;

pub use config::{
    JevConfig, MoaConfig, MoaMode, VerifierConfig, VerifierSelection, DEFAULT_HISTORY_LIMIT,
    DEFAULT_MAX_ADVICE_CHARS, DEFAULT_MAX_CONCURRENCY, DEFAULT_MAX_REFERENCE_CHARS,
    DEFAULT_REFERENCE_TIMEOUT_SECS,
};
pub use prompts::{
    aggregation_messages, consultation_transcript, escape_xml, moa_section, reference_messages,
    render_reference_proposals, verifier_messages, AGGREGATOR_SYSTEM_PROMPT,
    MOA_PROMPT_SECTION_NAME, REFERENCE_SYSTEM_PROMPT,
};
pub use redaction::redact;
pub use state::{
    ConsultationHistory, ConsultationOutcome, ConsultationRecord, MoaSnapshot, MoaState,
    ProposalAttempt, ProposalBatch, ProposalSource, ProposalStore, ProposerFailure, RankedProposal,
    TurnKey,
};
