//! TTSE: test-time skill experience — tip induction, ranking and consultation.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.ttse`: the config
//! (`ttse/config.py`), the tip/inference schema, the bank (`ttse/stores.py`,
//! `ttse/index.py`), classification (`ttse/classify.py`), success detection
//! (`ttse/success.py`), tip parsing (`ttse/tip_parse.py`), consultation/rendering
//! (`ttse/consult.py`, `ttse/render.py`) and the deterministic parts of
//! induction and dreaming (`ttse/induction.py`, `ttse/dream.py`).
//!
//! - [`TtseConfig`] — Python-shaped, serde-persisted tunables.
//! - [`TipBank`] — record store with dedupe, clustering and pruning; [`TipBankStats`]
//!   is its aggregate summary.
//! - [`TipCategory`] / [`TipRecord`] — the tip schema, with [`TipClassifier`]
//!   assigning the category and [`crate::ttse::tip_id_for`] deriving the content-addressed id.
//! - [`SuccessDetector`] — decides whether a run counts as a successful experience.
//! - [`TipParser`] — turns a model answer into tip records ([`crate::ttse::DEFAULT_MAX_RULE_CHARS`]
//!   bounds each rule).
//! - [`TipInducer`] — proposes tips from failed trajectories and evolution signals,
//!   offline or through an injected model.
//! - [`Dreamer`] — merges near-duplicate tips and prunes stale ones.
//! - [`TtseEngine`] — the pipeline over signals/trajectories that induces and
//!   ranks tips, then renders them for prompt injection.
//! - [`consult`] / [`render_tips`] — selects and renders the tips injected into a
//!   prompt, producing a [`ConsultResult`]; [`DreamReport`] and
//!   [`SuccessAssessment`] are the pass results, and [`crate::ttse::TipEvidence`] the
//!   trajectory evidence a tip was induced from.
//!
//! LLM-driven merging and embedding-backed retrieval are optional seams; the
//! default paths are deterministic.
mod bank;
mod classify;
mod config;
mod consult;
mod dream;
mod engine;
mod induce;
mod models;
mod parse;
mod success;
pub use bank::{AddOutcome, TipBank, TipBankStats};
pub use classify::{ClassificationRule, TipClassifier};
pub use config::TtseConfig;
pub use consult::{consult, render_tips};
pub use dream::Dreamer;
pub use engine::TtseEngine;
pub use induce::TipInducer;
pub use models::{
    tip_id_for, ConsultResult, DreamReport, SuccessAssessment, TipCategory, TipEvidence, TipRecord,
};
pub use parse::{TipParser, DEFAULT_MAX_RULE_CHARS};
pub use success::SuccessDetector;
