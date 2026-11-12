//! API-shaped stubs for the Python areas deferred to a future milestone.
//!
//! Python's `openjiuwen.symphony` package spans ~52k lines; a large part of it needs an
//! LLM/GPU stack or external services, which the offline, deterministic Rust port does
//! not ship. Rather than dropping those areas silently, every one of them keeps its
//! public shape here so callers can write against the eventual API and detect the gap
//! programmatically with [`crate::error::is_unsupported`].
//!
//! Every entry point returns [`openjiuwen_common::error::Error::Unsupported`] built by
//! [`crate::error::unsupported`], whose message names the feature and points at the
//! crate `CONTEXT.md` `## Deferred Scope` table. The documented constructors
//! ([`LlmTreeRetriever::new`](crate::deferred::LlmTreeRetriever::new),
//! [`FlowEngine::new`](crate::deferred::FlowEngine::new),
//! [`ExperienceBank::new`](crate::deferred::ExperienceBank::new),
//! ([`SkillFolderScanner::new`](crate::deferred::SkillFolderScanner::new)) always fail, so
//! a caller cannot build one of these types by accident and mistake it for a working
//! implementation. The operations are associated functions: `FlowEngine::distill()` reports
//! the same explicit error without needing a value.
//!
//! The data types in this module are ordinary `serde` types (`snake_case`,
//! default-tolerant) so a caller can build a request or decode a recorded result
//! without depending on any deferred machinery.
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{unsupported, Result};
use crate::model::CapabilityDescriptor;

/// The feature name used by [`LlmTreeRetriever`]'s errors.
const FEATURE_LLM_TREE_RETRIEVAL: &str = "LLM tree retrieval";
/// The feature name used by [`FlowEngine`]'s errors.
const FEATURE_FLOW_DISTILLATION: &str = "flow distillation";
/// The feature name used by [`ExperienceBank`]'s errors.
const FEATURE_EXPERIENCE_BANK: &str = "experience bank";
/// The feature name used by [`SkillFolderScanner`]'s errors.
const FEATURE_SKILL_SCANNING: &str = "skill folder scanning";

/// Configuration for the deferred [`LlmTreeRetriever`].
///
/// Mirrors Python `openjiuwen.symphony.retrieval.search.service.models.RetrieverConfig`
/// (its public `top_k` and model selection), reduced to the two knobs the deferred
/// retriever documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmTreeRetrieverConfig {
    /// The maximum number of payloads to return; Python's default is `10`.
    pub top_k: usize,
    /// The model that drives progressive tree disclosure, when one is configured.
    pub model: Option<String>,
}

impl Default for LlmTreeRetrieverConfig {
    /// Returns Python's `RetrieverConfig` defaults: `top_k = 10`, no model.
    fn default() -> Self {
        Self {
            top_k: 10,
            model: None,
        }
    }
}

/// One progressive-tree retrieval request.
///
/// Mirrors Python's per-request `RequestConfig`: `top_k: None` means "use the
/// retriever's configured `top_k`".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeSearchRequest {
    /// The per-request result limit; `None` defers to [`LlmTreeRetrieverConfig::top_k`].
    pub top_k: Option<usize>,
}

/// One progressive-tree retrieval result.
///
/// Mirrors Python's `SearchResult` (`method`, `payloads`, `elapsed_ms`): `method` names
/// the traversal that produced the payloads, and `payloads` are the selected skill
/// payloads in rank order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TreeSearchResult {
    /// The selected payloads, best first.
    pub payloads: Vec<String>,
    /// The traversal method that produced the result.
    pub method: String,
    /// Wall-clock duration of the search in milliseconds (metadata only).
    pub elapsed_ms: u64,
}

/// LLM-guided progressive skill-tree retriever — **deferred**.
///
/// Mirrors Python `openjiuwen.symphony.retrieval.Retriever` (the online retrieval
/// facade in `retrieval/search/service/retriever.py`), which walks a skill tree with an
/// LLM and, in some backends, selects branches with `transformers`/`vLLM` logits.
/// That needs an LLM/GPU stack, so it is part of the crate's `## Deferred Scope`; the
/// deterministic BM25 ([`crate::search::LexicalIndex`]) and embedding
/// ([`crate::search::SemanticIndex`]) paths cover the same selection job offline.
///
/// The documented constructor always fails, so a caller cannot build one of these by
/// accident; [`Self::search`] is an associated function that reports the same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmTreeRetriever;

impl LlmTreeRetriever {
    /// Mirrors `Retriever.__init__`; always reports that retrieval is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`LLM tree retrieval`).
    pub fn new(_config: LlmTreeRetrieverConfig) -> Result<Self> {
        Err(unsupported(FEATURE_LLM_TREE_RETRIEVAL))
    }

    /// Mirrors `Retriever.search_details`; always reports that retrieval is deferred.
    ///
    /// Synchronous — unlike Python's async call chain — because the stub fails before
    /// any I/O.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`LLM tree retrieval`).
    pub fn search(_query: &str, _request: &TreeSearchRequest) -> Result<TreeSearchResult> {
        Err(unsupported(FEATURE_LLM_TREE_RETRIEVAL))
    }
}

/// One published flow-distillation revision.
///
/// Mirrors the recipe/artifact revision Python's flow engine records when it distils a
/// verified capability combination into a reusable experience artifact.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FlowReceipt {
    /// The version of the published distillation artifact.
    pub revision: String,
}

/// Flow distillation and experience packaging engine — **deferred**.
///
/// Mirrors Python `openjiuwen.symphony.flow.SymphonyFlowEngine` (`flow/engine.py`),
/// which ingests execution traces, distils verified capability combinations
/// (`distill`), and packages them into installable skill artifacts (`pack`). It needs
/// an LLM narrative/review gate plus a large artifact pipeline, so it is part of the
/// crate's `## Deferred Scope`.
///
/// The documented constructor always fails, so a caller cannot build one of these by
/// accident; [`Self::distill`] and [`Self::pack`] are associated functions that report
/// the same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowEngine;

impl FlowEngine {
    /// Mirrors `SymphonyFlowEngine.__init__`; always reports that distillation is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`flow distillation`).
    pub fn new() -> Result<Self> {
        Err(unsupported(FEATURE_FLOW_DISTILLATION))
    }

    /// Mirrors `SymphonyFlowEngine.distill`; always reports that distillation is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`flow distillation`).
    pub fn distill() -> Result<FlowReceipt> {
        Err(unsupported(FEATURE_FLOW_DISTILLATION))
    }

    /// Mirrors `SymphonyFlowEngine.review_and_prepare_install` / the packager; always
    /// reports that packaging is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`flow distillation`).
    pub fn pack() -> Result<FlowReceipt> {
        Err(unsupported(FEATURE_FLOW_DISTILLATION))
    }
}

/// One experience-bank entry.
///
/// Mirrors Python `openjiuwen.symphony.experience.models.ExperienceItem`, reduced to
/// the `(text, score)` pair a caller reads back from a search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceRecord {
    /// The experience text.
    pub text: String,
    /// The similarity score that produced the hit.
    pub score: f64,
}

/// Vector-backed experience knowledge base — **deferred**.
///
/// Mirrors Python `openjiuwen.symphony.experience.ExperienceBank` (`experience/bank.py`),
/// which persists distilled experiences and searches them with `faiss` plus
/// `sentence-transformers` embeddings. That is an ML vector stack (and the bank is
/// orphaned in Python), so it is part of the crate's `## Deferred Scope`.
///
/// The documented constructor always fails, so a caller cannot build one of these by
/// accident; [`Self::add`] and [`Self::search`] are associated functions that report the
/// same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperienceBank;

impl ExperienceBank {
    /// Mirrors `ExperienceBank.__init__`; always reports that the bank is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`experience bank`).
    pub fn new() -> Result<Self> {
        Err(unsupported(FEATURE_EXPERIENCE_BANK))
    }

    /// Mirrors `ExperienceBank.add`; always reports that the bank is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`experience bank`).
    pub fn add(_record: ExperienceRecord) -> Result<()> {
        Err(unsupported(FEATURE_EXPERIENCE_BANK))
    }

    /// Mirrors `ExperienceBank.search_by_embedding`; always reports that the bank is
    /// deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`experience bank`).
    pub fn search(_query: &str, _top_k: usize) -> Result<Vec<ExperienceRecord>> {
        Err(unsupported(FEATURE_EXPERIENCE_BANK))
    }
}

/// SKILL.md folder scanner — **deferred**.
///
/// Mirrors Python `openjiuwen.symphony.shared.fingerprint.SkillFolderScanner`
/// (`shared/fingerprint/scanner.py`), which walks an installed-skill folder, parses
/// each `SKILL.md`, and fingerprints the result with an LLM. Filesystem scanning plus
/// LLM extraction is part of the crate's `## Deferred Scope`; the static
/// [`CapabilityDescriptor::fingerprint`] derivation is what ships.
///
/// The documented constructor always fails, so a caller cannot build one of these by
/// accident; [`Self::scan`] is an associated function that reports the same error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkillFolderScanner;

impl SkillFolderScanner {
    /// Mirrors `SkillFolderScanner.__init__`; always reports that scanning is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`skill folder scanning`).
    pub fn new(_root: impl AsRef<Path>) -> Result<Self> {
        Err(unsupported(FEATURE_SKILL_SCANNING))
    }

    /// Mirrors `SkillFolderScanner.scan` (Python's `ScanResult`, here just the
    /// descriptors); always reports that scanning is deferred.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::Error::Unsupported`] (`skill folder scanning`).
    pub fn scan(_root: impl AsRef<Path>) -> Result<Vec<CapabilityDescriptor>> {
        Err(unsupported(FEATURE_SKILL_SCANNING))
    }
}
