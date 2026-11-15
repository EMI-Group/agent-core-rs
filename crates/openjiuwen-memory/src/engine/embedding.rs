//! Embedding helpers: cached per-model dimensions and cancellable embedding of text.
//!
//! The engine embeds one record at a time (content is short and dedupe needs the
//! incoming vector before the candidate scan), so
//! [`MemoryEngine::embed_text_with`] is the single embedding entry point of the crate.
//! Every method takes an optional *per-scope* embedding handle: the facade resolves a
//! scope's override document per call and passes the resolved handle here, and `None`
//! means "use the engine's own model".
//!
//! The dimension is resolved once per *model name* and cached in
//! [`MemoryEngine::dimensions`], so several embedders of different dimensions (one per
//! scope, for example) coexist without sharing a single cached value.

use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_retrieval::embedding::Embedding;

use crate::record::CallOptions;

impl super::MemoryEngine {
    /// Returns the handle a call embeds with: the per-scope one when supplied, the
    /// engine's own otherwise.
    ///
    /// `None` means "this call cannot embed": a per-scope handle is never replaced by the
    /// engine's, and an absent engine handle is reported by the callers as
    /// [`Error::Unsupported`].
    pub(crate) fn effective_embedding<'a>(
        &'a self,
        per_scope: Option<&'a Arc<dyn Embedding>>,
    ) -> Option<&'a Arc<dyn Embedding>> {
        per_scope.or(self.embedding.as_ref())
    }

    /// Returns `true` when a call with `per_scope` can embed at all.
    pub(crate) fn has_embedding_with(&self, per_scope: Option<&Arc<dyn Embedding>>) -> bool {
        self.effective_embedding(per_scope).is_some()
    }

    /// Returns the embedding dimension of `embedding`, caching it by model name.
    ///
    /// # Errors
    ///
    /// The model's own error (for example a failed dimension probe).
    async fn dimension_of(&self, embedding: &Arc<dyn Embedding>) -> Result<usize> {
        let name = embedding.model_name();
        if let Some(dimension) = self.dimensions.lock().get(name).copied() {
            return Ok(dimension);
        }
        let dimension = embedding.dimension().await?;
        self.dimensions.lock().insert(name.to_owned(), dimension);
        Ok(dimension)
    }

    /// Returns the engine model's embedding dimension, resolving and caching it on first
    /// use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] without an embedding model and the model's own error
    /// (for example a failed dimension probe) otherwise.
    #[cfg(test)]
    pub(crate) async fn embedding_dimension(&self) -> Result<usize> {
        let embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| Error::unsupported("no embedding model is configured"))?;
        self.dimension_of(embedding).await
    }

    /// Embeds `text` with `per_scope` (or the engine's model), honouring cancellation.
    ///
    /// The vector length is checked against the cached dimension of *that* model when it
    /// is known; a model whose dimension cannot be resolved is still usable (the
    /// foundation store creates the collection from the first vector it receives).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] without an embedding model,
    /// [`Error::Cancelled`]/[`Error::Timeout`] when the call is cancelled or exceeds its
    /// deadline, [`Error::Validation`] on a dimension mismatch, and the provider error
    /// when the model call fails.
    pub(crate) async fn embed_text_with(
        &self,
        per_scope: Option<&Arc<dyn Embedding>>,
        text: &str,
        call: &CallOptions,
    ) -> Result<Vec<f32>> {
        call.check()?;
        let embedding = self
            .effective_embedding(per_scope)
            .ok_or_else(|| Error::unsupported("no embedding model is configured"))?;
        let expected = self.dimension_of(embedding).await.ok();
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

    /// Embeds `text` with the engine's own embedding model.
    ///
    /// This is the convenience form of [`MemoryEngine::embed_text_with`] used by the
    /// engine's own tests; every production path passes the (optionally per-scope) handle
    /// explicitly.
    ///
    /// # Errors
    ///
    /// See [`MemoryEngine::embed_text_with`].
    #[cfg(test)]
    pub(crate) async fn embed_text(&self, text: &str, call: &CallOptions) -> Result<Vec<f32>> {
        self.embed_text_with(None, text, call).await
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
    async fn a_per_scope_embedding_is_used_and_cached_under_its_own_name() {
        let engine = engine();
        let wide = Arc::new(HashEmbedding::new(24).with_model_name("wide")) as Arc<dyn Embedding>;
        let vector = engine
            .embed_text_with(Some(&wide), "calendar planning", &CallOptions::new())
            .await
            .expect("embedding");
        assert_eq!(vector.len(), 24);
        // The engine's own model keeps its own cached dimension.
        assert_eq!(engine.embedding_dimension().await.expect("dimension"), 8);
        assert_eq!(
            engine
                .embed_text("calendar planning", &CallOptions::new())
                .await
                .expect("embedding")
                .len(),
            8
        );
        assert!(engine.has_embedding_with(Some(&wide)));
    }

    #[tokio::test]
    async fn embed_text_without_a_model_is_unsupported() {
        let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())
            .expect("engine");
        assert!(!engine.has_embedding_with(None));
        let error = engine
            .embed_text("anything", &CallOptions::new())
            .await
            .expect_err("no embedding model");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let error = engine
            .embed_text_with(None, "anything", &CallOptions::new())
            .await
            .expect_err("no embedding model");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let error = engine
            .embedding_dimension()
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
