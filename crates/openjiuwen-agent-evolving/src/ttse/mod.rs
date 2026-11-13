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
//! - [`TipBank`] — record store with dedupe, clustering and pruning.
//! - [`TtseEngine`] — offline pipeline over signals/trajectories that induces and
//!   ranks tips, then renders them for prompt injection.
//!
//! LLM-driven merging and embedding-backed retrieval are optional seams; the
//! default paths are deterministic.
mod bank;
mod classify;
mod config;
mod models;
pub use bank::{AddOutcome, TipBank, TipBankStats};
pub use classify::{ClassificationRule, TipClassifier};
pub use config::TtseConfig;
pub use models::{
    tip_id_for, ConsultResult, DreamReport, SuccessAssessment, TipCategory, TipEvidence, TipRecord,
};
