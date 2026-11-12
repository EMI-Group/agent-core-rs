//! Deprecated vendor-specific aliases: [`AliyunReranker`].
//!
//! Ports the Python `openjiuwen/extensions/vendor_specific/aliyun_reranker.py`
//! shim (v0.1.20), whose entire content is `AliyunReranker = DashscopeReranker`
//! plus an import-time deprecation warning. The reranking implementations
//! themselves live in the kernel (`openjiuwen_core::retrieval::reranker`), so
//! this module keeps only the backwards-compatible alias shape:
//! [`AliyunReranker`] adapts an already-configured [`Reranker`] and forwards
//! every call unchanged.
//!
//! The `DashScope` HTTP transport is deferred: [`AliyunReranker::from_config`]
//! always answers [`ErrorKind::Unsupported`]. Wrap a configured kernel reranker
//! with [`AliyunReranker::new`] instead.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_core::retrieval::ScoreReranker;
//! use openjiuwen_extensions::vendor_specific::AliyunReranker;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let reranker = AliyunReranker::new(Arc::new(ScoreReranker::default()));
//! assert!(reranker.config().is_none());
//! # Ok(())
//! # }
//! ```
//!
//! [`ErrorKind::Unsupported`]: openjiuwen_common::error::ErrorKind::Unsupported

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_core::retrieval::{Reranker, SearchResult};
use serde::{Deserialize, Serialize};

/// `DashScope`-compatible endpoint used when `api_base` is not configured.
const DEFAULT_API_BASE: &str = "https://dashscope.aliyuncs.com";
/// `DashScope` reranker model used when `model_name` is not configured.
const DEFAULT_MODEL_NAME: &str = "qwen3-reranker-0.6b";
/// Deferred-transport message, identical to the one documented in `CONTEXT.md`.
const HTTP_TRANSPORT_DEFERRED: &str =
    "AliyunReranker HTTP transport is deferred; wrap a configured reranker with AliyunReranker::new";

/// Configuration of a `DashScope`-compatible reranker.
///
/// Python's `DashscopeReranker` reads the same knobs from its constructor and
/// environment; the Rust surface keeps them as data. [`Debug`] redacts
/// `api_key` because it stores a [`SecretString`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AliyunRerankerConfig {
    /// Credential for the `DashScope` endpoint; redacted in `Debug` and logs.
    pub api_key: Option<SecretString>,
    /// `DashScope`-compatible endpoint, without the rerank service path.
    pub api_base: String,
    /// Reranker model name, for example `qwen3-reranker-0.6b`.
    pub model_name: String,
    /// Default number of results to keep; `None` keeps every result.
    pub top_n: Option<usize>,
}

impl Default for AliyunRerankerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_base: DEFAULT_API_BASE.to_owned(),
            model_name: DEFAULT_MODEL_NAME.to_owned(),
            top_n: None,
        }
    }
}

/// Deprecated backwards-compatibility alias for a `DashScope`-compatible
/// reranker.
///
/// Python aliases `AliyunReranker` directly to `DashscopeReranker` and logs a
/// deprecation warning at import time. Rust cannot alias a concrete type whose
/// transport is not ported, so this adapter wraps any injected [`Reranker`] and
/// delegates to it without changing arguments, ordering, `top_n` semantics, or
/// errors. Construction logs the deprecation warning once, matching Python's
/// import-time warning without logging on every call.
#[derive(Clone)]
pub struct AliyunReranker {
    inner: Arc<dyn Reranker>,
    config: Option<AliyunRerankerConfig>,
}

impl AliyunReranker {
    /// Wraps an already-configured reranker (the seam hosts use today).
    ///
    /// Emits the deprecation warning Python logs at import time, targeted at
    /// `openjiuwen_extensions::vendor_specific`.
    #[must_use]
    pub fn new(inner: Arc<dyn Reranker>) -> Self {
        tracing::warn!(
            target: "openjiuwen_extensions::vendor_specific",
            "AliyunReranker is deprecated, please use a configured DashScope-compatible \
             reranker from openjiuwen_core::retrieval::reranker instead"
        );
        Self {
            inner,
            config: None,
        }
    }

    /// Builds the `DashScope` HTTP reranker.
    ///
    /// The HTTP transport is not ported: this constructor exists for Python API
    /// parity and always fails. Wrap a configured kernel reranker with
    /// [`AliyunReranker::new`] instead.
    ///
    /// # Errors
    ///
    /// Always returns [`openjiuwen_common::error::ErrorKind::Unsupported`] with
    /// the deferred-transport message naming [`AliyunReranker::new`].
    #[allow(
        clippy::unused_async,
        clippy::unused_async_trait_impl,
        clippy::needless_pass_by_value
    )]
    // `async` and the by-value config are fixed by the Python factory signature
    // (`await AliyunReranker.from_config(config)`); a future transport will both
    // await the connection and consume the configuration.
    pub async fn from_config(config: AliyunRerankerConfig) -> Result<Self> {
        tracing::debug!(
            model = %config.model_name,
            api_base = %config.api_base,
            "DashScope reranker construction requested while the HTTP transport is deferred"
        );
        Err(Error::Unsupported(HTTP_TRANSPORT_DEFERRED.to_owned()))
    }

    /// The configuration this reranker was built from, if any.
    ///
    /// [`AliyunReranker::new`] takes an already-configured reranker, so it has no
    /// [`AliyunRerankerConfig`] to report.
    #[must_use]
    pub fn config(&self) -> Option<&AliyunRerankerConfig> {
        self.config.as_ref()
    }
}

#[async_trait]
impl Reranker for AliyunReranker {
    /// Delegates to the wrapped reranker unchanged.
    ///
    /// The returned order, the per-call [`top_n`](Reranker::rerank) semantics,
    /// and every error come from the inner implementation, exactly like the
    /// Python alias that inherits `DashscopeReranker`'s behavior.
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.inner.rerank(query, documents, top_n).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Mutex;

    use openjiuwen_common::constants::REDACTED;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_core::retrieval::ScoreReranker;

    use super::*;

    // Python counterpart: `tests/unit_tests/core/retrieval/` (the alias has no
    // tests of its own in Python; behavior comes from `DashscopeReranker`).
    fn hit(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: format!("text {id}"),
            score,
            metadata: serde_json::Map::new(),
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct RecordedCall {
        query: String,
        documents: Vec<SearchResult>,
        top_n: Option<usize>,
    }

    /// Records the arguments of each call and returns the documents reversed,
    /// so the test can tell delegation apart from kernel reranking.
    #[derive(Default)]
    struct RecordingReranker {
        calls: Mutex<Vec<RecordedCall>>,
    }

    impl RecordingReranker {
        fn calls(&self) -> Vec<RecordedCall> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Reranker for RecordingReranker {
        async fn rerank(
            &self,
            query: &str,
            documents: Vec<SearchResult>,
            top_n: Option<usize>,
        ) -> Result<Vec<SearchResult>> {
            self.calls.lock().unwrap().push(RecordedCall {
                query: query.to_owned(),
                documents: documents.clone(),
                top_n,
            });
            let mut documents = documents;
            documents.reverse();
            Ok(documents)
        }
    }

    /// Always fails, so the test can check the error passes through untouched.
    struct FailingReranker;

    #[async_trait]
    impl Reranker for FailingReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: Vec<SearchResult>,
            _top_n: Option<usize>,
        ) -> Result<Vec<SearchResult>> {
            Err(Error::validation("inner reranker failed"))
        }
    }

    #[tokio::test]
    async fn delegation_forwards_arguments_and_preserves_the_inner_order() {
        let fake = Arc::new(RecordingReranker::default());
        let reranker = AliyunReranker::new(fake.clone());
        let documents = vec![hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)];

        let results = reranker
            .rerank("what is retrieval?", documents.clone(), Some(2))
            .await
            .unwrap();

        // The inner implementation decides the order and the truncation; the
        // adapter neither re-sorts nor applies `top_n` itself.
        let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
        assert_eq!(ids, ["c", "b", "a"]);
        assert_eq!(
            fake.calls(),
            vec![RecordedCall {
                query: "what is retrieval?".to_owned(),
                documents,
                top_n: Some(2),
            }]
        );
    }

    #[tokio::test]
    async fn delegation_preserves_the_inner_reranker_semantics() {
        let reranker = AliyunReranker::new(Arc::new(ScoreReranker::default()));
        let documents = vec![hit("low", 0.1), hit("high", 0.9), hit("mid", 0.5)];

        let results = reranker.rerank("query", documents, Some(2)).await.unwrap();

        // Exactly `ScoreReranker`'s descending, truncated output.
        let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
        assert_eq!(ids, ["high", "mid"]);
    }

    #[tokio::test]
    async fn delegation_preserves_inner_errors() {
        let reranker = AliyunReranker::new(Arc::new(FailingReranker));
        let error = reranker
            .rerank("query", vec![hit("a", 1.0)], None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "inner reranker failed");
    }

    #[tokio::test]
    async fn from_config_defers_the_http_transport() {
        let error = AliyunReranker::from_config(AliyunRerankerConfig::default())
            .await
            .err()
            .expect("the HTTP transport is deferred");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "AliyunReranker HTTP transport is deferred; wrap a configured reranker with \
             AliyunReranker::new"
        );
    }

    #[test]
    fn config_defaults_match_the_dashscope_client() {
        let config = AliyunRerankerConfig::default();
        assert_eq!(config.api_key, None);
        assert_eq!(config.api_base, "https://dashscope.aliyuncs.com");
        assert_eq!(config.model_name, "qwen3-reranker-0.6b");
        assert_eq!(config.top_n, None);
    }

    #[test]
    fn config_debug_redacts_the_api_key() {
        let config = AliyunRerankerConfig {
            api_key: Some(SecretString::new("sk-super-secret")),
            top_n: Some(3),
            ..AliyunRerankerConfig::default()
        };

        let rendered = format!("{config:?}");

        assert!(!rendered.contains("sk-super-secret"), "leaked: {rendered}");
        assert!(rendered.contains(REDACTED), "not redacted: {rendered}");
        assert!(rendered.contains("qwen3-reranker-0.6b"), "{rendered}");
    }

    #[test]
    fn config_serde_uses_snake_case_defaults_and_tolerates_unknown_fields() {
        let config: AliyunRerankerConfig =
            serde_json::from_str(r#"{"api_key": "sk-123", "top_n": 3, "unknown_field": true}"#)
                .unwrap();
        assert_eq!(config.api_key, Some(SecretString::new("sk-123")));
        assert_eq!(config.api_base, DEFAULT_API_BASE);
        assert_eq!(config.model_name, DEFAULT_MODEL_NAME);
        assert_eq!(config.top_n, Some(3));

        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["api_base"], "https://dashscope.aliyuncs.com");
        assert_eq!(json["model_name"], "qwen3-reranker-0.6b");
        assert_eq!(json["top_n"], 3);
        // Serialization keeps the plaintext so configuration round-trips.
        assert_eq!(json["api_key"], "sk-123");
    }

    #[tokio::test]
    async fn new_has_no_reported_config() {
        let reranker = AliyunReranker::new(Arc::new(ScoreReranker::default()));
        assert!(reranker.config().is_none());
    }
}
