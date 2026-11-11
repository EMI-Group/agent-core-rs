//! Embedding helpers: cached model dimension and cancellable embedding of text.
//!
//! The engine embeds one record at a time (content is short and dedupe needs the
//! incoming vector before the candidate scan), so [`MemoryEngine::embed_text`] is the
//! single embedding entry point of the crate.
//! The model dimension is resolved once per engine and cached in
//! [`MemoryEngine::dimensions`], which keeps every vector collection schema stable for
//! the lifetime of the instance.

use openjiuwen_common::error::{Error, Result};

use crate::record::CallOptions;

impl super::MemoryEngine {
    /// Returns the embedding dimension, resolving and caching it on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when no embedding model is attached and the
    /// model's own error (for example a failed dimension probe) otherwise.
    pub(crate) async fn embedding_dimension(&self) -> Result<usize> {
        let embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| Error::unsupported("no embedding model is configured"))?;
        let dimensions = self
            .dimensions
            .get_or_try_init(|| async { embedding.dimension().await })
            .await?;
        Ok(*dimensions)
    }

    /// Embeds `text`, honouring the call's cancellation and deadline.
    ///
    /// The vector length is checked against the cached dimension when it is known; a
    /// model whose dimension cannot be resolved is still usable (the foundation store
    /// creates the collection from the first vector it receives).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] without an embedding model,
    /// [`Error::Cancelled`]/[`Error::Timeout`] when the call is cancelled or exceeds its
    /// deadline, [`Error::Validation`] on a dimension mismatch, and the provider error
    /// when the model call fails.
    pub(crate) async fn embed_text(&self, text: &str, call: &CallOptions) -> Result<Vec<f32>> {
        call.check()?;
        let embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| Error::unsupported("no embedding model is configured"))?;
        let expected = self.embedding_dimension().await.ok();
        let vector = call.run(embedding.embed_query(text)).await??;
        if let Some(expected) = expected {
            if vector.len() != expected {
                return Err(Error::validation(format!(
                    "embedding model `{}` returned {} dimensions, expected {expected}",
                    embedding.model_name(),
                    vector.len()
                )));
            }
        }
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};

    use super::super::{MemoryEngine, MemoryEngineBuilder};
    use crate::config::MemoryEngineConfig;
    use crate::record::CallOptions;
    use crate::store::MemoryStores;

    fn engine() -> Arc<MemoryEngine> {
        MemoryEngineBuilder::new(MemoryEngineConfig::default())
            .embedding(Arc::new(HashEmbedding::new(8)))
            .build()
            .expect("engine")
    }

    /// An embedder that advertises four dimensions but returns three.
    struct MismatchedEmbedding;

    #[async_trait]
    impl Embedding for MismatchedEmbedding {
        fn model_name(&self) -> &'static str {
            "mismatched"
        }

        fn dimension_hint(&self) -> Option<usize> {
            Some(4)
        }

        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.1, 0.2, 0.3])
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1, 0.2, 0.3]).collect())
        }
    }

    #[tokio::test]
    async fn dimension_is_resolved_from_the_model() {
        let engine = engine();
        assert_eq!(engine.embedding_dimension().await.expect("dimension"), 8);
    }

    #[tokio::test]
    async fn embed_text_returns_a_vector_of_the_cached_dimension() {
        let engine = engine();
        let vector = engine
            .embed_text("calendar planning", &CallOptions::new())
            .await
            .expect("embedding");
        assert_eq!(vector.len(), 8);
    }

    #[tokio::test]
    async fn embed_text_without_a_model_is_unsupported() {
        let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())
            .expect("engine");
        let error = engine
            .embed_text("anything", &CallOptions::new())
            .await
            .expect_err("no embedding model");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_dimension_mismatch_is_a_validation_error() {
        let engine = MemoryEngineBuilder::new(MemoryEngineConfig::default())
            .embedding(Arc::new(MismatchedEmbedding))
            .build()
            .expect("engine");
        let error = engine
            .embed_text("anything", &CallOptions::new())
            .await
            .expect_err("dimension mismatch");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_cancelled_call_does_not_reach_the_model() {
        let engine = engine();
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let error = engine
            .embed_text("anything", &CallOptions::new().with_cancellation(token))
            .await
            .expect_err("cancelled");
        assert!(matches!(error, Error::Cancelled), "{error:?}");
    }
}
