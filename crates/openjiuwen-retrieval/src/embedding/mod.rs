//! Vector representations of text.
//!
//! Port of the Python `openjiuwen/core/retrieval/embedding/` abstraction: the
//! [`Embedding`] trait, an OpenAI-compatible client ([`OpenAiEmbedding`]), and a
//! deterministic hash-based implementation ([`HashEmbedding`]) for offline tests.
//!
//! The trait is intentionally small: a model name, an optional dimension hint, a
//! single-query encode, and a batched document encode. Python's async `dimension`
//! property (implemented as a method there because pydantic cannot express async
//! properties) is preserved as a default method here.

pub mod fake;
pub mod openai;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

pub use fake::HashEmbedding;
pub use openai::OpenAiEmbedding;

/// Encodes text into dense vectors.
#[async_trait]
pub trait Embedding: Send + Sync {
    /// Returns the model name reported to callers and logs.
    fn model_name(&self) -> &str;

    /// Returns the vector length when the model fixes one up front.
    ///
    /// `None` means the dimension is only known after the first successful
    /// embedding call; [`Embedding::dimension`] falls back to probing in that case.
    fn dimension_hint(&self) -> Option<usize> {
        None
    }

    /// Encodes a single query string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query and provider-specific
    /// errors for transport or decoding failures.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;

    /// Encodes many documents, preserving input order in the output.
    ///
    /// `batch_size`, when set, overrides the implementation's default batch size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an empty list or a blank entry and
    /// provider-specific errors for transport or decoding failures.
    async fn embed_documents(
        &self,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>>;

    /// Returns the embedding dimension, probing the model when no hint is known.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Embedding::embed_query`], plus
    /// [`Error::Validation`] when the probe produces an empty vector.
    async fn dimension(&self) -> Result<usize> {
        if let Some(dimension) = self.dimension_hint().filter(|dimension| *dimension > 0) {
            return Ok(dimension);
        }
        let vector = self.embed_query("test").await?;
        if vector.is_empty() {
            return Err(Error::Validation(format!(
                "embedding model `{}` returned an empty vector, so its dimension is unknown",
                self.model_name()
            )));
        }
        Ok(vector.len())
    }
}

/// Validates a query string, rejecting empty and whitespace-only input.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `text` is blank; otherwise returns `text`.
pub fn validate_query(text: &str) -> Result<&str> {
    if text.trim().is_empty() {
        return Err(Error::Validation(
            "embedding query must not be empty".to_owned(),
        ));
    }
    Ok(text)
}

/// Validates a document batch, rejecting an empty list and blank entries.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the offending index when a text is blank.
pub fn validate_texts(texts: &[String]) -> Result<()> {
    if texts.is_empty() {
        return Err(Error::Validation(
            "embedding texts must not be empty".to_owned(),
        ));
    }
    if let Some((index, text)) = texts
        .iter()
        .enumerate()
        .find(|(_, text)| text.trim().is_empty())
    {
        return Err(Error::Validation(format!(
            "embedding text at index {index} is blank (got {text:?})"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Embedder whose dimension is only knowable by probing, with a configurable
    /// probe result.
    struct ProbeEmbedding {
        vector: Vec<f32>,
    }

    #[async_trait]
    impl Embedding for ProbeEmbedding {
        fn model_name(&self) -> &'static str {
            "probe"
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            validate_query(text)?;
            Ok(self.vector.clone())
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            validate_texts(texts)?;
            Ok(texts.iter().map(|_| self.vector.clone()).collect())
        }
    }

    #[test]
    fn validate_query_accepts_non_blank_text() {
        assert_eq!(validate_query("hello world").unwrap(), "hello world");
        assert_eq!(validate_query("  padded  ").unwrap(), "  padded  ");
    }

    #[test]
    fn validate_query_rejects_blank_text() {
        assert_eq!(
            validate_query("").unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            validate_query(" \t\n").unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn validate_texts_accepts_a_non_blank_batch() {
        let texts = vec!["one".to_owned(), "two".to_owned()];
        validate_texts(&texts).unwrap();
    }

    #[test]
    fn validate_texts_rejects_an_empty_batch() {
        let error = validate_texts(&[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("must not be empty"), "{error}");
    }

    #[test]
    fn validate_texts_names_the_blank_index() {
        let texts = vec!["one".to_owned(), "   ".to_owned(), "three".to_owned()];
        let error = validate_texts(&texts).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("index 1"), "{error}");
    }

    #[tokio::test]
    async fn dimension_uses_the_hint_without_probing() {
        let embedder = HashEmbedding::new(8);
        assert_eq!(embedder.dimension_hint(), Some(8));
        assert_eq!(embedder.dimension().await.unwrap(), 8);
    }

    #[tokio::test]
    async fn dimension_probes_when_no_hint_is_available() {
        let embedder = ProbeEmbedding {
            vector: vec![0.0; 5],
        };
        assert_eq!(embedder.dimension_hint(), None);
        assert_eq!(embedder.dimension().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn dimension_rejects_an_empty_probe_vector() {
        let embedder = ProbeEmbedding { vector: Vec::new() };
        let error = embedder.dimension().await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("empty vector"), "{error}");
    }

    #[tokio::test]
    async fn the_trait_is_object_safe_and_send_sync() {
        let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(4));
        assert_eq!(embedder.model_name(), "hash-embedding");
        let vector = embedder.embed_query("hello").await.unwrap();
        assert_eq!(vector.len(), 4);
    }
}
