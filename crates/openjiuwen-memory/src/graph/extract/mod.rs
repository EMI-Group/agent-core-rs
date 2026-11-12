//! Entity/relation extraction for graph memory.
//!
//! [`HeuristicExtractor`] is the deterministic, dependency-free default: it scans for
//! quoted/backticked names, `@mentions`, path-like tokens, and non-sentence-initial
//! capitalized runs, then derives simple verb-pattern relations. [`LlmExtractor`] asks
//! a model for JSON and degrades to the heuristic extractor when the model output stays
//! unusable.
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
