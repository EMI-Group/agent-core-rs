//! `DashScope` hosted reranker: validated configuration with a deferred transport.
//!
//! Port of the configuration surface of Python's
//! `openjiuwen/core/retrieval/reranker/dashscope_reranker.py`
//! `DashscopeReranker` (a `StandardReranker` over the `DashScope` text-rerank
//! endpoint). The HTTP transport is not ported — this crate has no vendor client
//! and no credentials story — so [`DashScopeReranker`] exists to consume
//! [`RerankerConfig`] honestly:
//!
//! - [`DashScopeReranker::new`] validates every configuration field, so an
//!   unusable config (blank endpoint or model, non-positive timeout, sampling
//!   parameters out of range, missing API key) fails with
//!   [`Error::Config`](openjiuwen_common::error::Error::Config);
//! - [`Reranker::rerank`] returns
//!   [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) naming
//!   the endpoint and model instead of silently returning the input order.
//!
//! A caller that wants model-backed reranking offline can use
//! [`crate::reranker::LlmReranker`] with its own [`crate::reranker::LlmScorer`].
use std::fmt;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use super::Reranker;
use crate::common::{RerankerConfig, SearchResult};

/// Path of the `DashScope` text-rerank operation, appended to `api_base`.
pub const DASHSCOPE_RERANK_PATH: &str = "/api/v1/services/rerank/text-rerank/text-rerank";

/// A `DashScope` rerank client whose transport is deferred.
///
/// Construction validates the [`RerankerConfig`]; every field is read there, so
/// no configured value is silently dropped.
#[derive(Clone, PartialEq)]
pub struct DashScopeReranker {
    /// Validated configuration supplied by the caller.
    config: RerankerConfig,
}

impl fmt::Debug for DashScopeReranker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DashScopeReranker")
            .field("config", &self.config)
            .finish()
    }
}

impl DashScopeReranker {
    /// Builds a client from `config` after validating every field.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when a field is invalid (see
    /// [`RerankerConfig::validate`], plus a required non-empty `api_key`),
    /// wrapped with the `"DashScope reranker configuration"` context.
    pub fn new(config: RerankerConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|error| error.with_context("DashScope reranker configuration"))?;
        config
            .require_api_key()
            .map_err(|error| error.with_context("DashScope reranker configuration"))?;
        Ok(Self { config })
    }

    /// The validated configuration.
    #[must_use]
    pub fn config(&self) -> &RerankerConfig {
        &self.config
    }

    /// The endpoint a real client would call: `api_base` + [`DASHSCOPE_RERANK_PATH`].
    #[must_use]
    pub fn endpoint(&self) -> String {
        format!(
            "{}{DASHSCOPE_RERANK_PATH}",
            self.config.api_base.trim().trim_end_matches('/')
        )
    }
}

#[async_trait]
impl Reranker for DashScopeReranker {
    /// Returns a typed [`Error::Unsupported`]; `DashScope` reranking is deferred.
    async fn rerank(
        &self,
        _query: &str,
        _documents: Vec<SearchResult>,
        _top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        Err(Error::unsupported(format!(
            "DashScope reranker transport is not implemented: cannot call {} with \
             model `{}` (timeout {:?}); use LlmReranker with an LlmScorer instead",
            self.endpoint(),
            self.config.model_name.trim(),
            self.config.timeout
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;

    use super::*;
    use crate::common::Metadata;

    fn config() -> RerankerConfig {
        RerankerConfig {
            api_key: Some("sk-test".into()),
            api_base: "https://dashscope.example/api".to_owned(),
            model_name: "gte-rerank".to_owned(),
            timeout: Duration::from_secs(5),
            temperature: 0.5,
            top_p: 0.9,
        }
    }

    fn hit(id: &str) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: format!("text {id}"),
            score: 0.5,
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn new_validates_every_configuration_field() {
        let reranker = DashScopeReranker::new(config()).unwrap();
        assert_eq!(reranker.config(), &config());
        assert_eq!(
            reranker.endpoint(),
            "https://dashscope.example/api/api/v1/services/rerank/text-rerank/text-rerank"
        );

        let mut missing_key = config();
        missing_key.api_key = None;
        let error = DashScopeReranker::new(missing_key).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("api_key is required"), "{error}");

        let mut blank_base = config();
        blank_base.api_base = "  ".to_owned();
        let error = DashScopeReranker::new(blank_base).unwrap_err();
        assert!(
            error.to_string().contains("api_base must not be blank"),
            "{error}"
        );

        let mut blank_model = config();
        blank_model.model_name.clear();
        let error = DashScopeReranker::new(blank_model).unwrap_err();
        assert!(
            error.to_string().contains("model_name must not be blank"),
            "{error}"
        );

        let mut zero_timeout = config();
        zero_timeout.timeout = Duration::ZERO;
        let error = DashScopeReranker::new(zero_timeout).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("timeout must be greater than zero"),
            "{error}"
        );

        let mut hot = config();
        hot.temperature = 3.0;
        let error = DashScopeReranker::new(hot).unwrap_err();
        assert!(error.to_string().contains("temperature"), "{error}");

        let mut top_p = config();
        top_p.top_p = 0.0;
        let error = DashScopeReranker::new(top_p).unwrap_err();
        assert!(error.to_string().contains("top_p"), "{error}");
    }

    #[tokio::test]
    async fn rerank_reports_the_deferred_transport() {
        let reranker = DashScopeReranker::new(config()).unwrap();
        let error = reranker
            .rerank("query", vec![hit("a")], Some(1))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("DashScope reranker transport"),
            "unexpected message: {error}"
        );
        assert!(error.to_string().contains("gte-rerank"), "{error}");
    }
}
