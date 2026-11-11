//! A deterministic hash-based embedding for offline tests.
//!
//! Python has no fake embedder; this implementation exists so the retrieval tests
//! and examples run without network access. Vectors are derived purely from the
//! input text, so they are stable across runs, machines, and processes.

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use openjiuwen_common::error::Result;

use super::{validate_query, validate_texts, Embedding};

/// Fallback dimension when zero is requested.
pub const DEFAULT_HASH_DIMENSION: usize = 64;

/// Model name reported by a default [`HashEmbedding`].
const DEFAULT_HASH_MODEL_NAME: &str = "hash-embedding";

/// Length of a SHA-256 digest in bytes.
const DIGEST_LEN: usize = 32;

/// A deterministic embedding derived from SHA-256 hashes of whitespace tokens.
///
/// Each whitespace token is hashed, the 32 digest bytes are cycled to fill
/// `dimension` components mapped into `[-1, 1]`, the per-token vectors are summed
/// (mean pooled), and the result is L2-normalised. Empty or whitespace-only text
/// yields the zero vector, because there is no token to hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashEmbedding {
    dimension: usize,
    model_name: String,
}

impl HashEmbedding {
    /// Creates an embedder producing vectors of `dimension` components.
    ///
    /// A `dimension` of `0` falls back to [`DEFAULT_HASH_DIMENSION`].
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension: if dimension == 0 {
                DEFAULT_HASH_DIMENSION
            } else {
                dimension
            },
            model_name: DEFAULT_HASH_MODEL_NAME.to_owned(),
        }
    }

    /// Overrides the reported model name.
    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Returns the number of components in every produced vector.
    #[must_use]
    pub fn dimension_size(&self) -> usize {
        self.dimension
    }

    /// Hashes one text into a unit-length vector (or the zero vector when blank).
    fn hash_text(&self, text: &str) -> Vec<f32> {
        let mut accumulator = vec![0.0_f32; self.dimension];
        let mut token_count = 0_usize;
        for token in text.split_whitespace() {
            let digest = Sha256::digest(token.as_bytes());
            for (index, component) in accumulator.iter_mut().enumerate() {
                let byte = digest[index % DIGEST_LEN];
                // Map 0..=255 onto -1.0..=1.0 deterministically.
                *component += f32::from(byte) / 127.5 - 1.0;
            }
            token_count += 1;
        }
        if token_count == 0 {
            return vec![0.0; self.dimension];
        }
        let scale = 1.0 / token_count as f32;
        for component in &mut accumulator {
            *component *= scale;
        }
        let norm = accumulator
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for component in &mut accumulator {
                *component /= norm;
            }
        }
        accumulator
    }
}

impl Default for HashEmbedding {
    fn default() -> Self {
        Self::new(DEFAULT_HASH_DIMENSION)
    }
}

#[async_trait]
impl Embedding for HashEmbedding {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(self.dimension)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        validate_query(text)?;
        Ok(self.hash_text(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        validate_texts(texts)?;
        Ok(texts.iter().map(|text| self.hash_text(text)).collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn cosine(left: &[f32], right: &[f32]) -> f32 {
        let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
        let left_norm = left.iter().map(|a| a * a).sum::<f32>().sqrt();
        let right_norm = right.iter().map(|b| b * b).sum::<f32>().sqrt();
        dot / (left_norm * right_norm)
    }

    #[test]
    fn dimension_defaults_to_64() {
        assert_eq!(HashEmbedding::default().dimension_size(), 64);
        assert_eq!(HashEmbedding::new(0).dimension_size(), 64);
        assert_eq!(HashEmbedding::new(0).dimension_hint(), Some(64));
    }

    #[test]
    fn custom_dimensions_are_honoured() {
        for dimension in [1_usize, 8, 128] {
            let embedder = HashEmbedding::new(dimension);
            assert_eq!(embedder.dimension_hint(), Some(dimension));
            let vector = embedder.hash_text("hello deterministic world");
            assert_eq!(vector.len(), dimension);
        }
    }

    #[test]
    fn embedding_is_deterministic() {
        let embedder = HashEmbedding::new(16);
        assert_eq!(
            embedder.hash_text("the quick brown fox"),
            embedder.hash_text("the quick brown fox")
        );
        let other = HashEmbedding::new(16);
        assert_eq!(
            embedder.hash_text("the quick brown fox"),
            other.hash_text("the quick brown fox")
        );
    }

    #[test]
    fn identical_text_has_cosine_one() {
        let embedder = HashEmbedding::new(32);
        let left = embedder.hash_text("openjiuwen retrieval");
        let right = embedder.hash_text("openjiuwen retrieval");
        let similarity = cosine(&left, &right);
        assert!(
            (similarity - 1.0).abs() < 1e-5,
            "cosine similarity was {similarity}"
        );
    }

    #[test]
    fn non_empty_vectors_are_l2_normalised() {
        let embedder = HashEmbedding::new(24);
        let vector = embedder.hash_text("normalise me please");
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn distinct_texts_produce_distinct_vectors() {
        let embedder = HashEmbedding::new(32);
        let left = embedder.hash_text("alpha beta gamma");
        let right = embedder.hash_text("delta epsilon zeta");
        assert_ne!(left, right);
        assert!(cosine(&left, &right) < 0.99);
    }

    #[test]
    fn empty_text_produces_the_zero_vector() {
        let embedder = HashEmbedding::new(8);
        assert_eq!(embedder.hash_text(""), vec![0.0; 8]);
        assert_eq!(embedder.hash_text("   \t\n "), vec![0.0; 8]);
    }

    #[test]
    fn model_name_is_configurable() {
        let embedder = HashEmbedding::new(4).with_model_name("fake-embeddings");
        assert_eq!(embedder.model_name(), "fake-embeddings");
        assert_eq!(HashEmbedding::new(4).model_name(), "hash-embedding");
    }

    #[tokio::test]
    async fn embed_documents_preserves_order_and_counts() {
        let embedder = HashEmbedding::new(12);
        let texts = vec![
            "first text".to_owned(),
            "second text".to_owned(),
            "third text".to_owned(),
        ];
        let vectors = embedder.embed_documents(&texts, None).await.unwrap();
        assert_eq!(vectors.len(), texts.len());
        for (vector, text) in vectors.iter().zip(&texts) {
            assert_eq!(vector, &embedder.hash_text(text));
        }
    }

    #[tokio::test]
    async fn embed_query_matches_embed_documents() {
        let embedder = HashEmbedding::new(12);
        let query = embedder.embed_query("same text").await.unwrap();
        let documents = embedder
            .embed_documents(&["same text".to_owned()], None)
            .await
            .unwrap();
        assert_eq!(documents, vec![query]);
    }

    #[tokio::test]
    async fn blank_inputs_are_rejected() {
        let embedder = HashEmbedding::default();
        assert_eq!(
            embedder.embed_query("  ").await.unwrap_err().kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            embedder
                .embed_documents(&[], None)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            embedder
                .embed_documents(&["ok".to_owned(), String::new()], None)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
}
