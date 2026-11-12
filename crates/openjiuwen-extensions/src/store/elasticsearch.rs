//! Deferred Elasticsearch vector backend.
//!
//! Ports `openjiuwen/extensions/store/vector/es_vector_store.py`
//! (`ElasticsearchVectorStore`), which stores dense vectors in Elasticsearch
//! indices and searches them with k-NN. This crate keeps the Python
//! configuration surface but adds no Elasticsearch client dependency: the
//! backend is deferred, so [`ElasticsearchVectorStore::new`] validates the
//! configuration and [`ElasticsearchVectorStore::connect`] answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported).
//!
//! # Documented behavior of the deferred backend
//!
//! - An index is named `"{index_prefix}__{collection}"`, as in Python.
//! - The default vector dimension is `768`.
//! - The similarity metric is cosine.
//!
//! Python counterpart tests:
//! `tests/unit_tests/extensions/store/test_es_vector_store.py`.
#![allow(clippy::module_name_repetitions)] // `Elasticsearch*` names are fixed by the contract.
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::VectorStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
/// Default index prefix, matching Python's `index_prefix="agent_vector"`.
const DEFAULT_INDEX_PREFIX: &str = "agent_vector";
/// Default vector dimension documented by the Python backend.
const DEFAULT_DIMENSION: usize = 768;
/// The message every deferred connection answers with.
const UNSUPPORTED: &str =
    "elasticsearch vector store backend is deferred; use the in-memory or file vector store";
/// Configuration for an [`ElasticsearchVectorStore`].
///
/// Mirrors the Python constructor's inputs: the endpoint, the index-name
/// prefix, and the vector dimension. The defaults are Python's: an empty URL,
/// the `"agent_vector"` prefix, and `768` dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ElasticsearchVectorStoreConfig {
    /// Elasticsearch endpoint; it must start with `http://` or `https://`.
    pub url: String,
    /// Prefix of every index this store owns.
    pub index_prefix: String,
    /// Dimension of the stored dense vectors; must be greater than zero.
    pub dimension: usize,
}
impl Default for ElasticsearchVectorStoreConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            index_prefix: DEFAULT_INDEX_PREFIX.to_string(),
            dimension: DEFAULT_DIMENSION,
        }
    }
}
/// A validated but deferred Elasticsearch vector store.
///
/// Construction validates the configuration and names indices like Python's
/// backend; connecting returns
/// [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported)
/// instead of a live [`VectorStore`].
#[derive(Debug, Clone)]
pub struct ElasticsearchVectorStore {
    config: ElasticsearchVectorStoreConfig,
}
impl ElasticsearchVectorStore {
    /// Validates `config` and wraps it in a store.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when the URL is empty or does not start with `http://`/`https://`, or
    /// when `dimension` is zero.
    pub fn new(mut config: ElasticsearchVectorStoreConfig) -> Result<Self> {
        config.url = config.url.trim().to_string();
        if config.url.is_empty() {
            return Err(Error::validation(
                "elasticsearch vector store requires a non-empty `url`",
            ));
        }
        if !(config.url.starts_with("http://") || config.url.starts_with("https://")) {
            return Err(Error::validation(format!(
                "elasticsearch url `{}` must start with http:// or https://",
                config.url
            )));
        }
        if config.dimension == 0 {
            return Err(Error::validation(
                "elasticsearch vector dimension must be greater than zero",
            ));
        }
        Ok(Self { config })
    }
    /// Validates `config` and then reports that the backend is deferred.
    ///
    /// The async signature is fixed by the contract so a future client-backed
    /// implementation keeps the call sites unchanged.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as
    /// [`ElasticsearchVectorStore::new`], followed by
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
    /// for a valid configuration.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn connect(config: ElasticsearchVectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
        Self::new(config)?;
        Err(Error::unsupported(UNSUPPORTED))
    }
    /// The validated configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &ElasticsearchVectorStoreConfig {
        &self.config
    }
    /// The index name for `collection`: `"{index_prefix}__{collection}"`.
    #[must_use]
    pub fn index_name(&self, collection: &str) -> String {
        format!("{}__{collection}", self.config.index_prefix)
    }
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    #[test]
    fn default_config_matches_python_defaults() {
        let config = ElasticsearchVectorStoreConfig::default();
        assert_eq!(config.url, "");
        assert_eq!(config.index_prefix, "agent_vector");
        assert_eq!(config.dimension, 768);
    }
    #[test]
    fn valid_http_url_builds_a_store() {
        let config = ElasticsearchVectorStoreConfig {
            url: "http://localhost:9200".to_string(),
            ..ElasticsearchVectorStoreConfig::default()
        };
        let store = ElasticsearchVectorStore::new(config).unwrap();
        assert_eq!(store.config().url, "http://localhost:9200");
        assert_eq!(store.index_name("docs"), "agent_vector__docs");
    }
    #[test]
    fn invalid_configs_are_rejected() {
        let empty =
            ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig::default()).unwrap_err();
        assert_eq!(empty.kind(), ErrorKind::Validation);
        assert!(empty.message().contains("url"));
        let wrong_scheme = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
            url: "redis://localhost:6379".to_string(),
            ..ElasticsearchVectorStoreConfig::default()
        })
        .unwrap_err();
        assert_eq!(wrong_scheme.kind(), ErrorKind::Validation);
        let zero_dimension = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
            url: "http://localhost:9200".to_string(),
            dimension: 0,
            ..ElasticsearchVectorStoreConfig::default()
        })
        .unwrap_err();
        assert_eq!(zero_dimension.kind(), ErrorKind::Validation);
        assert!(zero_dimension.message().contains("dimension"));
    }
    #[test]
    fn index_name_uses_the_prefix_separator() {
        let store = ElasticsearchVectorStore::new(ElasticsearchVectorStoreConfig {
            url: "https://localhost:9200".to_string(),
            index_prefix: "custom".to_string(),
            ..ElasticsearchVectorStoreConfig::default()
        })
        .unwrap();
        assert_eq!(store.index_name("docs"), "custom__docs");
    }
    #[tokio::test]
    async fn connect_reports_the_deferral() {
        let error = ElasticsearchVectorStore::connect(ElasticsearchVectorStoreConfig {
            url: "http://localhost:9200".to_string(),
            ..ElasticsearchVectorStoreConfig::default()
        })
        .await
        .err()
        .expect("a deferred backend must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error
            .message()
            .to_ascii_lowercase()
            .contains("elasticsearch"));
    }
}
