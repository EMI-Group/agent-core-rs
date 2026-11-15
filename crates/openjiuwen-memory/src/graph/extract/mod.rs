//! Entity/relation extraction for graph memory.
//!
//! [`HeuristicExtractor`] is the deterministic, dependency-free default: it scans for
//! quoted/backticked names, `@mentions`, path-like tokens, and non-sentence-initial
//! capitalized runs, then derives simple verb-pattern relations. [`LlmExtractor`] asks
//! a model for JSON and degrades to the heuristic extractor when the model output stays
//! unusable.
//!
//! A degraded pass is never silent: the fallback still succeeds (graph memory keeps
//! working without a healthy model) but reports why on [`Extraction::degraded`], which
//! [`Extraction::is_degraded`] exposes and which serializes only when set. Callers that
//! need a hard failure instead can match on it; extraction itself never turns an
//! unusable model answer into an error, because the retry budget plus the heuristic
//! fallback are the documented resilience contract.
//!
//! The module is self-contained: it owns the extraction data types ([`Extraction`],
//! [`ExtractedEntity`], [`ExtractedRelation`]) and never touches graph persistence.
//! The scanner lives in `heuristic.rs`, the model-backed extractor and its tolerant
//! JSON parsing in `llm.rs`; both are re-exported here.

mod heuristic;
mod llm;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::record::{CallOptions, MemoryScope};

pub use heuristic::HeuristicExtractor;
pub use llm::{LlmExtractor, EXTRACTION_SYSTEM_PROMPT};

/// One entity found in a memory text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedEntity {
    /// Entity name exactly as it should be persisted (`"Alice Smith"`, `"src/main.rs"`).
    pub name: String,
    /// Short kind label such as `Person`, `Project`, `Concept`, or `Organization`.
    pub entity_type: String,
    /// Optional salience in `0.0..=1.0`; heuristics leave it `None`.
    pub importance: Option<f32>,
}

/// One directed relation between two [`ExtractedEntity`] names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractedRelation {
    /// Name of the source entity.
    pub from: String,
    /// Name of the target entity.
    pub to: String,
    /// Short relation label such as `works_at` or `is_a`.
    pub relation_type: String,
}

/// The entities and relations extracted from one text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct Extraction {
    /// Entities in first-seen order.
    pub entities: Vec<ExtractedEntity>,
    /// Relations in first-seen order.
    pub relations: Vec<ExtractedRelation>,
    /// Why a degraded path produced this extraction, when one did.
    ///
    /// [`LlmExtractor`] sets it when the retry budget is spent and the heuristic
    /// extractor answers instead; a successful model pass and a plain heuristic pass
    /// both leave it `None`. The field is omitted from JSON when unset, so the wire
    /// shape of a non-degraded extraction is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded: Option<String>,
}

impl Extraction {
    /// Returns `true` when a degraded path (rather than the requested one) produced
    /// this extraction.
    ///
    /// The value is unchanged: the entities and relations are still usable, and
    /// [`Extraction::degraded`] carries the reason.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.is_some()
    }

    /// Returns the degradation reason, if one was recorded.
    #[must_use]
    pub fn degraded_reason(&self) -> Option<&str> {
        self.degraded.as_deref()
    }
}

/// Turns one text into an [`Extraction`].
///
/// Implementations must be usable from `GraphMemory` concurrently (`Send + Sync`),
/// must honour [`CallOptions`], and must never panic on arbitrary input.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Stable implementation name (`"heuristic"`, `"llm"`), used in diagnostics.
    fn name(&self) -> &str;

    /// Extracts entities and relations from `text` within `scope`.
    ///
    /// An implementation that answers through a cheaper fallback than the one the
    /// caller selected must report that on [`Extraction::degraded`] instead of
    /// returning a result that looks like a full-quality pass.
    ///
    /// # Errors
    ///
    /// Returns `Error::Cancelled` when `options` has been cancelled. Implementations
    /// that call a model may also surface the model's own error kind.
    async fn extract(
        &self,
        text: &str,
        scope: &MemoryScope,
        options: &CallOptions,
    ) -> Result<Extraction>;
}
