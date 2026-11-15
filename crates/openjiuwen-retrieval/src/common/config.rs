//! Configuration types and constants for the knowledge base, retrieval, index,
//! vector store, embedding, and reranker layers.
//!
//! Port of the Python `openjiuwen/core/retrieval/common/config.py` models plus the
//! foundation `EmbeddingConfig` and `RerankerConfig`. Every field is optional in
//! the JSON payload (unknown fields are ignored), so configurations can grow
//! without breaking existing callers.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;

use super::filter::MetadataFilter;

/// Default number of retrieval results returned.
pub const DEFAULT_TOP_K: usize = 5;
/// Default chunk size, in characters.
pub const DEFAULT_CHUNK_SIZE: usize = 512;
/// Default chunk overlap, in characters.
pub const DEFAULT_CHUNK_OVERLAP: usize = 50;
/// Vector-only index type.
pub const INDEX_TYPE_VECTOR: &str = "vector";
/// BM25 (keyword-only) index type.
pub const INDEX_TYPE_BM25: &str = "bm25";
/// Hybrid (vector + BM25) index type, the default.
pub const INDEX_TYPE_HYBRID: &str = "hybrid";
/// Default embedding request timeout.
pub const DEFAULT_EMBEDDING_TIMEOUT: Duration = Duration::from_secs(60);
/// Default embedding batch size.
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 8;
/// Default vector distance metric.
pub const DEFAULT_DISTANCE_METRIC: &str = "cosine";

/// Parses an index type name, ignoring surrounding whitespace and case.
///
/// The crate has exactly three retrieval channels, so any other name is a
/// configuration error rather than something to degrade to dense retrieval —
/// a typo must not silently change what a query searches.
///
/// # Errors
///
/// Returns [`Error::Config`] naming the offending value and the accepted names.
pub fn parse_index_type(value: &str) -> Result<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        INDEX_TYPE_VECTOR => Ok(INDEX_TYPE_VECTOR),
        INDEX_TYPE_BM25 => Ok(INDEX_TYPE_BM25),
        INDEX_TYPE_HYBRID => Ok(INDEX_TYPE_HYBRID),
        other => Err(Error::config(format!(
            "unsupported index type `{other}`: expected one of `{INDEX_TYPE_VECTOR}`, \
             `{INDEX_TYPE_BM25}`, `{INDEX_TYPE_HYBRID}`"
        ))),
    }
}

/// `serde` adapter that stores a [`Duration`] as a number of seconds.
mod duration_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.subsec_nanos() == 0 {
            serializer.serialize_u64(value.as_secs())
        } else {
            serializer.serialize_f64(value.as_secs_f64())
        }
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = f64::deserialize(deserializer)?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(serde::de::Error::custom(format!(
                "duration must be a non-negative number of seconds, got {seconds}"
            )));
        }
        Ok(Duration::from_secs_f64(seconds))
    }
}

/// Knowledge base configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Knowledge base identifier.
    #[serde(default)]
    pub kb_id: String,
    /// Index type: [`INDEX_TYPE_HYBRID`], [`INDEX_TYPE_BM25`], or [`INDEX_TYPE_VECTOR`].
    #[serde(default = "default_index_type")]
    pub index_type: String,
    /// Whether a graph index is built alongside the vector index.
    #[serde(default)]
    pub use_graph: bool,
    /// Chunk size in characters.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Chunk overlap in characters.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    /// Embed image chunks from their text/caption only.
    ///
    /// Image+text embedding is not implemented, so `true` (and any text-only
    /// embedder, which is all this crate ships) indexes an image document from
    /// its caption; `false` combined with an embedder that reports multimodal
    /// support is a typed `Error::Unsupported` rather than a silent caption.
    #[serde(default)]
    pub use_caption_for_images: bool,
}

impl KnowledgeBaseConfig {
    /// Creates a configuration with the Python defaults for `kb_id`.
    pub fn new(kb_id: impl Into<String>) -> Self {
        Self {
            kb_id: kb_id.into(),
            index_type: default_index_type(),
            use_graph: false,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            use_caption_for_images: false,
        }
    }

    /// Validates the fields that select behaviour.
    ///
    /// `index_type` must be one of [`INDEX_TYPE_VECTOR`], [`INDEX_TYPE_BM25`], or
    /// [`INDEX_TYPE_HYBRID`] (see [`parse_index_type`]); the chunk window is
    /// validated by the chunker, which owns that contract.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for an unknown index type.
    pub fn validate(&self) -> Result<()> {
        parse_index_type(&self.index_type)?;
        Ok(())
    }
}

/// Per-query retrieval configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum score a result must reach to be returned.
    #[serde(default)]
    pub score_threshold: Option<f32>,
    /// Whether graph retrieval runs; `None` keeps the knowledge base default.
    #[serde(default)]
    pub use_graph: Option<bool>,
    /// Whether agentic retrieval runs.
    #[serde(default)]
    pub agentic: bool,
    /// Whether graph expansion runs.
    #[serde(default)]
    pub graph_expansion: bool,
    /// Metadata filter passed to the vector store.
    #[serde(default)]
    pub filters: Option<MetadataFilter>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            score_threshold: None,
            use_graph: None,
            agentic: false,
            graph_expansion: false,
            filters: None,
        }
    }
}

impl RetrievalConfig {
    /// Sets the requested number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the minimum score.
    #[must_use]
    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = Some(score_threshold);
        self
    }

    /// Sets the metadata filter.
    #[must_use]
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filters = Some(filter);
        self
    }
}

/// Index configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Index name.
    #[serde(default)]
    pub index_name: String,
    /// Index type: [`INDEX_TYPE_HYBRID`], [`INDEX_TYPE_BM25`], or [`INDEX_TYPE_VECTOR`].
    #[serde(default = "default_index_type")]
    pub index_type: String,
    /// Embed image chunks from their text/caption only.
    ///
    /// Carried into [`KnowledgeBaseConfig::use_caption_for_images`] by
    /// [`IndexConfig::to_knowledge_base_config`].
    #[serde(default)]
    pub use_caption_for_images: bool,
}

impl IndexConfig {
    /// Validates the index name and type.
    ///
    /// Python requires `index_name`, so a blank name is a configuration error,
    /// as is an unknown `index_type` (see [`parse_index_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] for a blank `index_name` or an unknown
    /// `index_type`.
    pub fn validate(&self) -> Result<()> {
        if self.index_name.trim().is_empty() {
            return Err(Error::config("index_name must not be blank"));
        }
        parse_index_type(&self.index_type)?;
        Ok(())
    }

    /// Reinterprets this index configuration as a knowledge-base configuration.
    ///
    /// `index_name` becomes [`KnowledgeBaseConfig::kb_id`], `index_type` selects
    /// the retrieval channel, and `use_caption_for_images` carries the image
    /// embedding policy across; chunking keeps the Python defaults because an
    /// index does not configure a chunk window.
    ///
    /// No validation is performed here: call [`IndexConfig::validate`] first (as
    /// [`crate::knowledge_base::SimpleKnowledgeBase::from_index_config`] does).
    #[must_use]
    pub fn to_knowledge_base_config(&self) -> KnowledgeBaseConfig {
        KnowledgeBaseConfig {
            kb_id: self.index_name.clone(),
            index_type: self.index_type.clone(),
            use_graph: false,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            use_caption_for_images: self.use_caption_for_images,
        }
    }
}

/// Vector store configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Store provider identifier (`memory`, `milvus`, `chroma`, ...).
    #[serde(default)]
    pub store_provider: String,
    /// Database name; must stay empty for the in-memory store, which has no
    /// databases (see [`crate::vector_store::InMemoryVectorStore::from_config`]).
    #[serde(default)]
    pub database_name: String,
    /// Collection name inside the store.
    #[serde(default)]
    pub collection_name: String,
    /// Distance metric: `cosine`, `euclidean`, or `dot`.
    ///
    /// Only `cosine` is implemented; the in-memory store rejects the others with
    /// an `Error::Config` instead of silently scoring by cosine.
    #[serde(default = "default_distance_metric")]
    pub distance_metric: String,
}

/// Embedding model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name, sent as the `model` request field.
    #[serde(default)]
    pub model_name: String,
    /// OpenAI-compatible API base URL.
    #[serde(default)]
    pub base_url: String,
    /// API key; redacted in `Debug`/`Display` output.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Per-request timeout.
    #[serde(default = "default_embedding_timeout", with = "duration_secs")]
    pub timeout: Duration,
    /// Number of retries after a retryable failure.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Maximum number of inputs per embedding request.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// Extra HTTP headers sent with every request.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Expected embedding dimension, when the model fixes one.
    #[serde(default)]
    pub dimension: Option<usize>,
    /// Whether TLS certificates are verified.
    #[serde(default = "default_verify_ssl")]
    pub verify_ssl: bool,
}

impl EmbeddingConfig {
    /// Creates a configuration with the Python defaults.
    pub fn new(model_name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            base_url: base_url.into(),
            api_key: None,
            timeout: DEFAULT_EMBEDDING_TIMEOUT,
            max_retries: default_max_retries(),
            max_batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
            headers: BTreeMap::new(),
            dimension: None,
            verify_ssl: true,
        }
    }

    /// Sets the API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<SecretString>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the per-request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the retry budget.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the maximum batch size.
    #[must_use]
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Replaces the extra HTTP headers.
    #[must_use]
    pub fn with_headers(mut self, headers: BTreeMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    /// Sets the expected embedding dimension.
    #[must_use]
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Sets whether TLS certificates are verified.
    #[must_use]
    pub fn with_verify_ssl(mut self, verify_ssl: bool) -> Self {
        self.verify_ssl = verify_ssl;
        self
    }
}

/// Reranker model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// API key; redacted in `Debug`/`Display` output.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// API base URL.
    #[serde(default)]
    pub api_base: String,
    /// Model name; accepts Python's `model` alias on input.
    #[serde(default, alias = "model")]
    pub model_name: String,
    /// Per-request timeout.
    #[serde(default = "default_reranker_timeout", with = "duration_secs")]
    pub timeout: Duration,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Nucleus sampling probability.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

impl RerankerConfig {
    /// Validates the fields a hosted rerank client needs.
    ///
    /// The rules mirror Python's pydantic constraints (`api_base` non-empty,
    /// `timeout > 0`) and add finite ranges for the sampling parameters:
    /// `temperature` in `[0, 2]` and `top_p` in `(0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the first invalid field.
    pub fn validate(&self) -> Result<()> {
        if self.api_base.trim().is_empty() {
            return Err(Error::config("reranker api_base must not be blank"));
        }
        if self.model_name.trim().is_empty() {
            return Err(Error::config("reranker model_name must not be blank"));
        }
        if self.timeout.is_zero() {
            return Err(Error::config("reranker timeout must be greater than zero"));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(Error::config(format!(
                "reranker temperature must be a finite number between 0 and 2, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::config(format!(
                "reranker top_p must be a finite number in (0, 1], got {}",
                self.top_p
            )));
        }
        Ok(())
    }

    /// Returns the API key, rejecting a missing or empty credential.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `api_key` is `None` or empty.
    pub fn require_api_key(&self) -> Result<&str> {
        match self.api_key.as_ref() {
            Some(key) if !key.is_empty() => Ok(key.expose_secret()),
            _ => Err(Error::config(
                "reranker api_key is required and must not be empty",
            )),
        }
    }
}

fn default_index_type() -> String {
    INDEX_TYPE_HYBRID.to_owned()
}

fn default_chunk_size() -> usize {
    DEFAULT_CHUNK_SIZE
}

fn default_chunk_overlap() -> usize {
    DEFAULT_CHUNK_OVERLAP
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

fn default_distance_metric() -> String {
    DEFAULT_DISTANCE_METRIC.to_owned()
}

fn default_embedding_timeout() -> Duration {
    DEFAULT_EMBEDDING_TIMEOUT
}

fn default_max_retries() -> u32 {
    3
}

fn default_max_batch_size() -> usize {
    DEFAULT_EMBEDDING_BATCH_SIZE
}

fn default_verify_ssl() -> bool {
    true
}

fn default_reranker_timeout() -> Duration {
    Duration::from_secs(10)
}

fn default_temperature() -> f32 {
    0.95
}

fn default_top_p() -> f32 {
    0.95
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    #[test]
    fn knowledge_base_config_defaults() {
        let config = KnowledgeBaseConfig::new("kb-1");
        assert_eq!(config.kb_id, "kb-1");
        assert_eq!(config.index_type, INDEX_TYPE_HYBRID);
        assert!(!config.use_graph);
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.chunk_overlap, DEFAULT_CHUNK_OVERLAP);
        assert!(!config.use_caption_for_images);

        let parsed: KnowledgeBaseConfig = serde_json::from_value(json!({"kb_id": "kb-2"})).unwrap();
        assert_eq!(parsed, KnowledgeBaseConfig::new("kb-2"));
    }

    #[test]
    fn retrieval_config_defaults_and_builders() {
        let config = RetrievalConfig::default();
        assert_eq!(config.top_k, DEFAULT_TOP_K);
        assert!(config.score_threshold.is_none());
        assert!(config.use_graph.is_none());
        assert!(!config.agentic);
        assert!(!config.graph_expansion);
        assert!(config.filters.is_none());

        let config = config
            .with_top_k(3)
            .with_score_threshold(0.4)
            .with_filter(MetadataFilter::eq("author", "alice"));
        assert_eq!(config.top_k, 3);
        assert_eq!(config.score_threshold, Some(0.4));
        assert_eq!(config.filters, Some(MetadataFilter::eq("author", "alice")));

        // Filters survive JSON round-trip in the canonical Python shape.
        let text = serde_json::to_string(&config).unwrap();
        assert!(text.contains("\"filters\":{\"author\":\"alice\"}"));
        let reparsed: RetrievalConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(reparsed, config);
    }

    #[test]
    fn embedding_config_duration_and_secret_round_trip() {
        let mut headers = BTreeMap::new();
        headers.insert("X-Tenant".to_owned(), "acme".to_owned());
        let config = EmbeddingConfig::new("text-embedding-3-small", "https://api.example/v1")
            .with_api_key("sk-test")
            .with_timeout(Duration::from_millis(1500))
            .with_max_retries(5)
            .with_max_batch_size(16)
            .with_headers(headers.clone())
            .with_dimension(256)
            .with_verify_ssl(false);
        assert_eq!(config.timeout, Duration::from_millis(1500));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.headers, headers);
        assert_eq!(config.dimension, Some(256));
        assert!(!config.verify_ssl);

        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["timeout"], json!(1.5));
        assert_eq!(value["api_key"], json!("sk-test"));
        let restored: EmbeddingConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);

        let defaults: EmbeddingConfig =
            serde_json::from_value(json!({"model_name": "m", "base_url": "u"})).unwrap();
        assert_eq!(defaults.timeout, DEFAULT_EMBEDDING_TIMEOUT);
        assert_eq!(defaults.max_retries, 3);
        assert_eq!(defaults.max_batch_size, DEFAULT_EMBEDDING_BATCH_SIZE);
        assert!(defaults.headers.is_empty());
        assert!(defaults.dimension.is_none());
        assert!(defaults.verify_ssl);
        assert!(defaults.api_key.is_none());

        // Whole seconds serialize as integers.
        let whole = EmbeddingConfig::new("m", "u");
        assert_eq!(serde_json::to_value(&whole).unwrap()["timeout"], json!(60));
    }

    #[test]
    fn reranker_config_alias_and_defaults() {
        let parsed: RerankerConfig =
            serde_json::from_value(json!({"api_base": "https://rerank", "model": "bge-m3"}))
                .unwrap();
        assert_eq!(parsed.model_name, "bge-m3");
        assert_eq!(parsed.timeout, Duration::from_secs(10));
        assert_eq!(parsed.temperature, 0.95);
        assert_eq!(parsed.top_p, 0.95);
        assert!(parsed.api_key.is_none());

        let value = serde_json::to_value(&parsed).unwrap();
        assert_eq!(value["model_name"], json!("bge-m3"));
        assert_eq!(value["timeout"], json!(10));
        assert!(value.get("model").is_none());
    }

    #[test]
    fn configs_ignore_unknown_fields() {
        let config: KnowledgeBaseConfig = serde_json::from_value(json!({
            "kb_id": "kb",
            "chunk_size": 128,
            "future_flag": true,
            "nested": {"x": 1}
        }))
        .unwrap();
        assert_eq!(config.chunk_size, 128);

        let store: VectorStoreConfig = serde_json::from_value(json!({
            "store_provider": "milvus",
            "collection_name": "docs",
            "extra": 1
        }))
        .unwrap();
        assert_eq!(store.distance_metric, DEFAULT_DISTANCE_METRIC);
        assert_eq!(store.database_name, "");

        let index: IndexConfig = serde_json::from_value(json!({"index_name": "idx"})).unwrap();
        assert_eq!(index.index_type, INDEX_TYPE_HYBRID);
        assert!(!index.use_caption_for_images);
    }

    #[test]
    fn parse_index_type_normalizes_and_rejects_unknown_names() {
        assert_eq!(parse_index_type("vector").unwrap(), INDEX_TYPE_VECTOR);
        assert_eq!(parse_index_type(" BM25 ").unwrap(), INDEX_TYPE_BM25);
        assert_eq!(parse_index_type("Hybrid").unwrap(), INDEX_TYPE_HYBRID);
        let error = parse_index_type("faiss").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error.to_string().contains("unsupported index type `faiss`"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn knowledge_base_validate_checks_the_index_type() {
        assert!(KnowledgeBaseConfig::new("kb").validate().is_ok());
        let mut config = KnowledgeBaseConfig::new("kb");
        config.index_type = "graph".to_owned();
        let error = config.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }

    #[test]
    fn index_config_validates_and_converts_to_a_knowledge_base_config() {
        let blank = IndexConfig {
            index_name: String::new(),
            index_type: INDEX_TYPE_HYBRID.to_owned(),
            use_caption_for_images: false,
        };
        assert!(blank.validate().is_err());
        let index = IndexConfig {
            index_name: "docs".to_owned(),
            index_type: INDEX_TYPE_BM25.to_owned(),
            use_caption_for_images: true,
        };
        assert!(index.validate().is_ok());
        let kb = index.to_knowledge_base_config();
        assert_eq!(kb.kb_id, "docs");
        assert_eq!(kb.index_type, INDEX_TYPE_BM25);
        assert!(kb.use_caption_for_images);
        assert_eq!(kb.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(kb.chunk_overlap, DEFAULT_CHUNK_OVERLAP);
        assert!(!kb.use_graph);

        let unknown = IndexConfig {
            index_type: "sqlite".to_owned(),
            ..index
        };
        let error = unknown.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }

    #[test]
    fn reranker_validate_and_api_key_requirement() {
        let mut config = RerankerConfig {
            api_key: Some("sk-test".into()),
            api_base: "https://rerank.example".to_owned(),
            model_name: "bge-m3".to_owned(),
            timeout: Duration::from_secs(10),
            temperature: 0.95,
            top_p: 0.95,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.require_api_key().unwrap(), "sk-test");

        config.temperature = 2.5;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("temperature"));
        config.temperature = 0.5;
        config.top_p = 1.5;
        assert!(config.validate().unwrap_err().to_string().contains("top_p"));
        config.top_p = 0.5;
        config.timeout = Duration::ZERO;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("timeout"));
        config.timeout = Duration::from_secs(1);
        config.model_name.clear();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("model_name"));
        config.model_name = "bge-m3".to_owned();
        config.api_base = String::new();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("api_base"));

        config.api_base = "https://rerank.example".to_owned();
        config.api_key = None;
        let error = config.require_api_key().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        config.api_key = Some(String::new().into());
        assert!(config.require_api_key().is_err());
    }
}
