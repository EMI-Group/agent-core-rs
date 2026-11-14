//! TTSE: test-time skill experience — tip induction, retrieval and consultation.
//!
//! Ports the offline part of Python `openjiuwen.agent_evolving.ttse`: the config
//! (`ttse/config.py`), the tip/inference schema, the bank and retrieval index
//! (`ttse/stores.py`, `ttse/index.py`), classification (`ttse/classify.py`),
//! success detection (`ttse/success.py`), tip parsing (`ttse/tip_parse.py`),
//! consultation/rendering (`ttse/consult.py`, `ttse/render.py`), the deterministic
//! parts of induction and dreaming (`ttse/induction.py`, `ttse/dream.py`), the
//! LLM merge/purge prompts (`ttse/prompts.py`) and the model-backed dream pass.
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
//! - [`EmbeddingIndex`](crate::ttse::EmbeddingIndex) — cosine index over the tip rules, cold-filled from an
//!   `Embedding` provider and persisted as one JSON document.
//! - [`rank`](crate::ttse::rank) / [`RetrieveMode`](crate::ttse::RetrieveMode) — lexical, embedding and RRF-fused
//!   ranking; [`consult_async`](crate::ttse::consult_async) is the ranked counterpart of [`consult`].
//! - [`Dreamer`] — merges near-duplicate tips and prunes stale ones, offline or
//!   ([`Dreamer::dream_with_model`]) through a model.
//! - [`TtseEngine`] — the pipeline over signals/trajectories that induces and
//!   ranks tips, then renders them for prompt injection.
//! - [`consult`] / [`render_tips`] — selects and renders the tips injected into a
//!   prompt, producing a [`ConsultResult`]; [`DreamReport`] and
//!   [`SuccessAssessment`] are the pass results, and [`crate::ttse::TipEvidence`] the
//!   trajectory evidence a tip was induced from.
//!
//! Ranking and dreaming both have deterministic paths that need no model: lexical
//! retrieval, token-Jaccard clustering and the form/prune checks always work, and
//! the embedding index and model passes are additive seams.
mod bank;
mod classify;
mod config;
mod consult;
mod dream;
mod dream_llm;
mod dream_prompt;
mod engine;
mod index;
mod induce;
mod models;
mod parse;
mod retrieve;
mod success;
pub use bank::{AddOutcome, TipBank, TipBankStats};
pub use classify::{ClassificationRule, TipClassifier};
pub use config::TtseConfig;
pub use consult::{consult, render_tips};
pub use dream::Dreamer;
pub use dream_prompt::{
    normalize_merge_subset, parse_merge_verdict, parse_purge_verdicts, MergeDecision, MergeVerdict,
    PurgeDecision, PurgeVerdict, DEFAULT_MAX_LLM_MERGES, DREAM_MERGE_INSTRUCTION,
    DREAM_PURGE_INSTRUCTION,
};
pub use engine::TtseEngine;
pub use index::{cosine, EmbeddingIndex, IndexMatch};
pub use induce::TipInducer;
pub use models::{
    tip_id_for, ConsultResult, DreamReport, SuccessAssessment, TipCategory, TipEvidence, TipRecord,
};
pub use parse::{TipParser, DEFAULT_MAX_RULE_CHARS};
pub use retrieve::{consult_async, rank, RankedTip, RetrieveMode, RRF_K};
pub use success::SuccessDetector;
