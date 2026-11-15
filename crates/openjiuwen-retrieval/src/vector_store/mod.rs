//! Vector store abstractions: records, queries, the [`VectorStore`] trait, and
//! the provider registry.
//!
//! Port of the Python `openjiuwen/core/retrieval/vector_store/base.py` ABC plus
//! the `create_vector_store` factory from `store.py`. The factory becomes
//! [`VectorStoreRegistry`], and the offline in-memory implementation lives in
//! [`memory`]. Milvus, Chroma, `PGVector`, and Elasticsearch remain behind the
//! registry seam and report [`Error::Unsupported`] until they are ported.
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use crate::common::{Metadata, MetadataFilter, SearchResult, VectorStoreConfig, DEFAULT_TOP_K};

pub mod memory;

pub use memory::InMemoryVectorStore;

/// Metadata field holding the parent document id.
///
/// Vector stores expose this pseudo-field to metadata filters and inject it into
/// [`SearchResult::metadata`] when the record carries a `doc_id`.
pub const DOC_ID_FIELD: &str = "doc_id";

/// Metadata field holding the source chunk id.
///
/// Vector stores expose this pseudo-field to metadata filters and inject it into
/// [`SearchResult::metadata`] when the record carries a `chunk_id`.
pub const CHUNK_ID_FIELD: &str = "chunk_id";

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

/// A vector plus its payload: the unit the stores index.
///
/// Duplicate ids replace an existing record; `doc_id` and `chunk_id` are
/// optional because stores can be used without a knowledge base.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Record identifier (unique within a collection).
    pub id: String,
    /// Text payload returned verbatim in search results.
    #[serde(default)]
    pub text: String,
    /// Embedding vector.
    #[serde(default)]
    pub vector: Vec<f32>,
    /// Record metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
    /// Identifier of the parent document, when known.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Identifier of the source chunk, when known.
    #[serde(default)]
    pub chunk_id: Option<String>,
}

/// A similarity-search request.
///
/// `top_k` defaults to [`DEFAULT_TOP_K`]; `filter` and `score_threshold` are
/// optional. `top_k == 0` is valid and yields no results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorQuery {
    /// Query vector; records with a different dimension are skipped.
    #[serde(default)]
    pub vector: Vec<f32>,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional metadata predicate applied before scoring.
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
    /// Optional inclusive minimum score (`score >= threshold`).
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

impl VectorQuery {
    /// Creates a query for `vector` with [`DEFAULT_TOP_K`] results and no filter.
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            top_k: DEFAULT_TOP_K,
            filter: None,
            score_threshold: None,
        }
    }

    /// Sets the maximum number of results (`top_k`).
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the metadata predicate.
    #[must_use]
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the inclusive minimum score.
    #[must_use]
    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = Some(score_threshold);
        self
    }
}

/// Backend-agnostic vector store interface.
///
/// Every method is fallible: implementations must not panic on unknown ids,
/// empty inputs, or mismatched vector dimensions.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Store name (the collection name for the in-memory store).
    fn name(&self) -> &str;

    /// Inserts or replaces records and returns the number of records supplied.
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize>;

    /// Fetches a record by id; `None` when the id is unknown.
    async fn get(&self, id: &str) -> Result<Option<VectorRecord>>;

    /// Fetches several records by id, preserving the requested order.
    ///
    /// Ids that do not exist are skipped, so the result may be shorter than
    /// `ids`; the relative order of the returned records follows `ids`.
    /// The default implementation calls [`Self::get`] once per id, which keeps
    /// stores without a native batch read correct; stores that can serve a
    /// batch more cheaply should override it.
    async fn get_many(&self, ids: &[String]) -> Result<Vec<VectorRecord>> {
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(record) = self.get(id).await? {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Lists records in store order, optionally truncated to `limit`.
    async fn list(&self, limit: Option<usize>) -> Result<Vec<VectorRecord>>;

    /// Deletes records by id and returns how many of them existed.
    async fn delete(&self, ids: &[String]) -> Result<usize>;

    /// Deletes records matching `filter` and returns how many were removed.
    async fn delete_by_filter(&self, filter: &MetadataFilter) -> Result<usize>;

    /// Runs a similarity search and returns hits ordered by descending score.
    async fn search(&self, query: VectorQuery) -> Result<Vec<SearchResult>>;

    /// Number of stored records.
    async fn len(&self) -> Result<usize>;

    /// Whether the store holds no records.
    async fn is_empty(&self) -> Result<bool>;

    /// Removes every record.
    async fn clear(&self) -> Result<()>;
}

/// Constructor hook used by [`VectorStoreRegistry`].
pub type VectorStoreFactory =
    Arc<dyn Fn(&VectorStoreConfig) -> Result<Arc<dyn VectorStore>> + Send + Sync>;

/// Registry mapping a `store_provider` name to a factory.
///
/// This is the Rust counterpart of Python's `create_vector_store` factory:
/// unknown providers surface as [`Error::Unsupported`], which documents the
/// external-database seam without pulling those clients into the crate.
#[derive(Clone, Default)]
pub struct VectorStoreRegistry {
    factories: BTreeMap<String, VectorStoreFactory>,
}

impl std::fmt::Debug for VectorStoreRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStoreRegistry")
            .field("providers", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl VectorStoreRegistry {
    /// Creates an empty registry (no providers).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry with the offline in-memory providers registered.
    ///
    /// The names `"memory"` and `"in_memory"` both construct
    /// [`InMemoryVectorStore`] through [`InMemoryVectorStore::from_config`], so
    /// `collection_name` names the store while `database_name` and a
    /// non-`cosine` `distance_metric` surface as [`Error::Config`] instead of
    /// being ignored.
    pub fn with_builtin() -> Self {
        let mut registry = Self::new();
        let factory: VectorStoreFactory = Arc::new(|config| {
            Ok(Arc::new(InMemoryVectorStore::from_config(config)?) as Arc<dyn VectorStore>)
        });
        // The two names are distinct string keys; registering twice cannot fail.
        let _ = registry.register("memory", Arc::clone(&factory));
        let _ = registry.register("in_memory", factory);
        registry
    }

    /// Registers a provider factory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `provider` is blank or already
    /// registered.
    pub fn register(
        &mut self,
        provider: impl Into<String>,
        factory: VectorStoreFactory,
    ) -> Result<()> {
        let provider = provider.into();
        let provider = provider.trim();
        if provider.is_empty() {
            return Err(Error::validation(
                "vector store provider name must not be blank",
            ));
        }
        if self.factories.contains_key(provider) {
            return Err(Error::validation(format!(
                "vector store provider '{provider}' is already registered"
            )));
        }
        self.factories.insert(provider.to_owned(), factory);
        Ok(())
    }

    /// Whether `provider` is registered.
    pub fn contains(&self, provider: &str) -> bool {
        self.factories.contains_key(provider.trim())
    }

    /// Builds the store configured by `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when `config.store_provider` is unknown,
    /// and propagates factory failures.
    pub fn create(&self, config: &VectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
        let provider = config.store_provider.trim();
        if let Some(factory) = self.factories.get(provider) {
            return factory(config);
        }
        let available = if self.factories.is_empty() {
            "none".to_owned()
        } else {
            self.factories
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };
        Err(Error::unsupported(format!(
            "unavailable vector store provider: {provider}; available providers are: {available}"
        )))
    }
}

/// Cosine similarity in `[-1, 1]`; `0.0` for zero vectors or dimension mismatch.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Scales a cosine similarity into `[0, 1]` with `(cos + 1) / 2`.
///
/// This matches the Chroma and Milvus `cosine` distance conventions the Python
/// stores use, so scores are comparable across backends.
pub fn normalize_cosine(cos: f32) -> f32 {
    f32::midpoint(cos, 1.0)
}

#[cfg(test)]
mod tests;
