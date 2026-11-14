//! The ML/GPU-backed Python areas that still have no offline implementation.
//!
//! Every other `openjiuwen.symphony` area now ships a real, deterministic implementation:
//!
//! - [`discovery`] — `SKILL.md` folder scanning plus lexical skill search.
//! - [`experience`] — the experience bank and the persisted
//!   [`ExperienceIndex`](crate::experience::ExperienceIndex): distilled capability
//!   combinations with deterministic, embedding-backed ranking and persistence.
//! - [`flow`] — flow distillation and reusable `SkillPack` artifacts.
//! - [`llm`] — LLM-guided stages (tree retrieval, ontology matching, semantic
//!   scoring, fingerprint extraction, judging) with deterministic offline fallbacks.
//! - [`object_artifact`] — S3/OBS graph artifact persistence over
//!   foundation's object stores.
//!
//! One backend cannot be expressed without a native ML/GPU stack, which an offline build
//! does not ship. Rather than dropping it silently, it keeps its public shape here so
//! callers can write against the eventual API and detect the gap programmatically with
//! [`crate::error::is_unsupported`].
//!
//! Every entry point returns [`openjiuwen_common::error::Error::Unsupported`] built by
//! [`crate::error::unsupported`], whose message names the feature and points at the crate
//! `CONTEXT.md` `## Deferred Scope` table. The documented constructor
//! ([`VllmLogitSelector::new`](crate::deferred::VllmLogitSelector::new)) always fails, so a
//! caller cannot build one of these types by accident and mistake it for a working
//! implementation. The operations are associated functions, so they report the same
//! explicit error without needing a value.

use crate::error::{unsupported, Result};

/// The feature name used by [`VllmLogitSelector`]'s errors.
const FEATURE_VLLM_LOGIT_SELECTION: &str = "vLLM logit selection";

/// `transformers`/`vLLM` logit-based branch selection — **deferred**.
///
/// Mirrors the logit-selection backend of Python `openjiuwen.symphony.retrieval.Retriever`
/// (`retrieval/`), which scores tree branches with a `transformers` or `vLLM` model and
/// keeps the top candidates. That needs a GPU inference stack, so it is part of the
/// crate's `## Deferred Scope`; the deterministic
/// [`crate::llm::TreeRetriever`] covers progressive tree disclosure offline.
///
/// The documented constructor always fails, so a caller cannot build one of these by
/// accident; [`Self::select`] is an associated function that reports the same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VllmLogitSelector;

impl VllmLogitSelector {
    /// Mirrors the model loading the logit backend performs at construction; always
    /// reports that logit selection is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`]
    /// (`vLLM logit selection`).
    pub fn new(_model_path: impl Into<String>) -> Result<Self> {
        Err(unsupported(FEATURE_VLLM_LOGIT_SELECTION))
    }

    /// Mirrors the logit-scored branch selection; always reports that it is deferred.
    ///
    /// On success the returned `Vec<String>` holds the selected candidates, best first.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`]
    /// (`vLLM logit selection`).
    pub fn select(_query: &str, _candidates: &[String]) -> Result<Vec<String>> {
        Err(unsupported(FEATURE_VLLM_LOGIT_SELECTION))
    }
}
