//! Rust-only embedding-backed scoring for the lexical candidate set.
//!
//! Python's Symphony search runtime has no vector path: a live inventory is ranked by the
//! deterministic field-weighted BM25 in
//! [`lexical`](crate::retrieval::search::lexical). This module adds an *additive*, optional
//! path over `openjiuwen-retrieval`'s [`Embedding`](openjiuwen_retrieval::Embedding) trait,
//! so a caller that already runs an embedding model can blend cosine similarity into the
//! same [`LexicalHit`](crate::retrieval::search::lexical::LexicalHit) ranking — or ignore
//! the trait entirely, because a weight of `0.0` reproduces the lexical order exactly (see
//! [`blend_scores`](crate::retrieval::search::embedding::blend_scores)).
//!
//! # Determinism
//!
//! Only the embedder decides whether the numbers are reproducible: [`HashEmbedding`]
//! (openjiuwen-retrieval) derives vectors purely from the text, so tests are offline and
//! stable. Everything this module adds is deterministic — keys are [`BTreeMap`]-sorted,
//! ties break on the key, and no output depends on hash-map iteration order or the clock.
//!
//! A cached vector map is guarded by a [`std::sync::RwLock`]; the guard is never held across
//! an `.await` (vectors are snapshotted into a `Vec`, the guard is dropped, and only then is
//! the embedder — or nothing at all — awaited).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use openjiuwen_retrieval::Embedding;

use crate::error::{Error, Result};
use crate::retrieval::search::lexical::{LexicalDocument, LexicalHit};

/// Builds the text a document is embedded from.
///
/// The parts are the document's name, description, body and category, joined with `"\n"`
/// and with empty parts dropped. A document whose parts are all empty is embedded by its
/// key instead, because an embedder rejects a blank input and dropping the document
/// silently would make the index depend on which documents happened to be blank.
#[must_use]
pub fn embedding_text(document: &LexicalDocument) -> String {
    let parts = [
        document.name.as_str(),
        document.description.as_str(),
        document.body.as_str(),
        document.category.as_str(),
    ];
    let joined = parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.trim().is_empty() {
        document.key.clone()
    } else {
        joined
    }
}

/// An embedding index over [`LexicalDocument`]s.
///
/// One vector per document key, kept in memory and computed from
/// [`embedding_text`](crate::retrieval::search::embedding::embedding_text). The scorer owns
/// its vectors, so two scorers over one embedder never interfere.
pub struct EmbeddingScorer {
    /// The embedder used for documents and queries.
    embedding: Arc<dyn Embedding>,
    /// The cached document vectors, keyed by document key.
    vectors: RwLock<BTreeMap<String, Vec<f32>>>,
}

impl fmt::Debug for EmbeddingScorer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbeddingScorer")
            .field("model", &self.embedding.model_name())
            .field("vectors", &self.read_vectors().len())
            .finish()
    }
}

impl EmbeddingScorer {
    /// Creates an empty scorer over `embedding`.
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self {
            embedding,
            vectors: RwLock::new(BTreeMap::new()),
        }
    }

    /// Embeds and indexes every document, returning how many vectors were written.
    ///
    /// Equivalent to [`EmbeddingScorer::index_capped`] with no cap.
    ///
    /// # Errors
    ///
    /// Returns the embedder's error (including [`Error::Validation`] for a blank text).
    pub async fn index(&self, documents: &[LexicalDocument]) -> Result<usize> {
        self.index_capped(documents, 0).await
    }

    /// Embeds and indexes documents, at most `max_documents` of them (`0` = unlimited).
    ///
    /// Documents are embedded in key order and the cap keeps the first `max_documents` of
    /// that order, so a capped build is deterministic. Existing vectors for other keys are
    /// kept; a re-indexed key is replaced. Returns how many vectors were written.
    ///
    /// # Errors
    ///
    /// Returns the embedder's error, or [`Error::Retrieval`] when the embedder returns
    /// fewer vectors than texts.
    pub async fn index_capped(
        &self,
        documents: &[LexicalDocument],
        max_documents: usize,
    ) -> Result<usize> {
        let mut ordered: Vec<&LexicalDocument> = documents.iter().collect();
        ordered.sort_by(|left, right| left.key.cmp(&right.key));
        if max_documents > 0 {
            ordered.truncate(max_documents);
        }
        if ordered.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = ordered.iter().map(|document| embedding_text(document)).collect();
        let vectors = self.embedding.embed_documents(&texts, None).await?;
        if vectors.len() < ordered.len() {
            return Err(Error::retrieval(format!(
                "the embedder returned {} vectors for {} documents",
                vectors.len(),
                ordered.len()
            )));
        }
        let mut guard = self.write_vectors();
        let mut written = 0_usize;
        for (document, vector) in ordered.iter().zip(vectors) {
            guard.insert(document.key.clone(), vector);
            written += 1;
        }
        drop(guard);
        Ok(written)
    }

    /// Embeds and stores one document, replacing any vector for the same key.
    ///
    /// # Errors
    ///
    /// Returns the embedder's error, or [`Error::Retrieval`] when the embedder returns no
    /// vector at all.
    pub async fn upsert(&self, document: &LexicalDocument) -> Result<()> {
        let text = embedding_text(document);
        let mut vectors = self.embedding.embed_documents(&[text], None).await?;
        if vectors.is_empty() {
            return Err(Error::retrieval(format!(
                "the embedder returned no vector for document {}",
                document.key
            )));
        }
        let vector = vectors.remove(0);
        let mut guard = self.write_vectors();
        guard.insert(document.key.clone(), vector);
        drop(guard);
        Ok(())
    }

    /// Removes the vector of `key`, reporting whether it existed.
    ///
    /// Removing an unknown key is not an error.
    ///
    /// # Errors
    ///
    /// This implementation has no failure mode; the [`Result`] keeps the signature uniform
    /// with the other index backends in the crate.
    pub fn remove(&self, key: &str) -> Result<bool> {
        let mut guard = self.write_vectors();
        Ok(guard.remove(key).is_some())
    }

    /// Removes every cached vector.
    ///
    /// # Errors
    ///
    /// This implementation has no failure mode; the [`Result`] keeps the signature uniform
    /// with the other index backends in the crate.
    pub fn clear(&self) -> Result<()> {
        let mut guard = self.write_vectors();
        guard.clear();
        drop(guard);
        Ok(())
    }

    /// The cosine similarity of `query` against every indexed document.
    ///
    /// Keys appear even when their similarity is `0.0`, so a caller can tell "indexed but
    /// unrelated" from "not indexed".
    ///
    /// # Errors
    ///
    /// Returns the embedder's error for the query.
    pub async fn scores(&self, query: &str) -> Result<BTreeMap<String, f64>> {
        let query_vector = self.embedding.embed_query(query).await?;
        let snapshot: Vec<(String, Vec<f32>)> = {
            let guard = self.read_vectors();
            guard
                .iter()
                .map(|(key, vector)| (key.clone(), vector.clone()))
                .collect()
        };
        let mut scores = BTreeMap::new();
        for (key, vector) in snapshot {
            scores.insert(key, cosine_similarity(&query_vector, &vector));
        }
        Ok(scores)
    }

    /// Ranks the indexed documents by cosine similarity to `query`.
    ///
    /// `top_k == 0` means unlimited. Ties break on the key, ascending.
    ///
    /// # Errors
    ///
    /// Returns the embedder's error for the query.
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<LexicalHit>> {
        let scores = self.scores(query).await?;
        Ok(ranked_hits(scores, top_k))
    }

    /// How many documents are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read_vectors().len()
    }

    /// Whether nothing is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows the vector map for reading, recovering from a poisoned lock.
    fn read_vectors(&self) -> RwLockReadGuard<'_, BTreeMap<String, Vec<f32>>> {
        self.vectors
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Borrows the vector map for writing, recovering from a poisoned lock.
    fn write_vectors(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Vec<f32>>> {
        self.vectors
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Cosine similarity in `[-1, 1]`, as `0.0` for every degenerate case.
///
/// Empty vectors, vectors of different length, a zero-norm side and a non-finite result all
/// return `0.0`, so a caller never sees `NaN`. The sum runs in `f64` even though the inputs
/// are `f32`.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (left, right) in a.iter().zip(b.iter()) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    if similarity.is_finite() {
        similarity.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Blends embedding scores into a lexical ranking.
///
/// `weight` is clamped into `[0, 1]` (a non-finite weight becomes `0.0`) and decides how
/// much the embedding score counts:
///
/// ```text
/// lex_norm = lexical_score / max(lexical_score)        (0.0 when that maximum is 0.0)
/// emb      = clamp((cosine + 1) / 2, 0, 1)             (0.0 for a key with no cosine)
/// blended  = (1 - weight) * lex_norm + weight * emb
/// ```
///
/// Every key of either input is scored — a key present in only one of them contributes
/// `0.0` to the other term — so the result is the union of both key sets. The output is
/// sorted by descending score, ties broken by key ascending.
///
/// `weight == 0.0` therefore reproduces the lexical ranking (with lexical scores normalized
/// into `[0, 1]`) and `weight == 1.0` the embedding ranking, so a caller without an
/// embedding model can run the same code path.
#[must_use]
pub fn blend_scores(
    lexical: &[LexicalHit],
    embedding: &BTreeMap<String, f64>,
    weight: f64,
) -> Vec<LexicalHit> {
    let weight = if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let max_lexical = lexical
        .iter()
        .map(|hit| hit.score)
        .fold(0.0_f64, f64::max);
    let mut blended: BTreeMap<String, f64> = BTreeMap::new();
    for hit in lexical {
        let lexical_normalized = if max_lexical > 0.0 {
            (hit.score / max_lexical).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let entry = blended.entry(hit.key.clone()).or_insert(0.0);
        *entry += (1.0 - weight) * lexical_normalized;
    }
    for (key, cosine) in embedding {
        let embedding_normalized = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0);
        let entry = blended.entry(key.clone()).or_insert(0.0);
        *entry += weight * embedding_normalized;
    }
    ranked_hits(blended, 0)
}

/// Turns a score map into hits sorted by descending score, ties by key ascending.
fn ranked_hits(scores: BTreeMap<String, f64>, top_k: usize) -> Vec<LexicalHit> {
    let mut hits: Vec<LexicalHit> = scores
        .into_iter()
        .map(|(key, score)| LexicalHit { key, score })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.key.cmp(&right.key))
    });
    if top_k > 0 {
        hits.truncate(top_k);
    }
    hits
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use openjiuwen_retrieval::HashEmbedding;

    fn document(key: &str, name: &str, description: &str) -> LexicalDocument {
        LexicalDocument::new(key, name).with_description(description)
    }

    fn make_scorer() -> EmbeddingScorer {
        EmbeddingScorer::new(Arc::new(HashEmbedding::new(32)))
    }

    #[test]
    fn embedding_text_joins_the_parts_and_falls_back_to_the_key() {
        let document = LexicalDocument::new("k", "Name")
            .with_description("Desc")
            .with_body("Body")
            .with_category("Skills > Data");
        assert_eq!(embedding_text(&document), "Name\nDesc\nBody\nSkills > Data");
        let blank = LexicalDocument::new("only-key", "  ");
        assert_eq!(embedding_text(&blank), "only-key");
    }

    #[test]
    fn cosine_similarity_handles_the_degenerate_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        assert!(
            (cosine_similarity(&[1.0, 1.0], &[1.0, 0.0]) - std::f64::consts::FRAC_1_SQRT_2).abs()
                < 1e-6
        );
        assert!(cosine_similarity(&[f32::NAN, 1.0], &[1.0, 1.0]).is_finite());
    }

    #[tokio::test]
    async fn indexing_is_key_ordered_and_capped() {
        let scorer = make_scorer();
        let documents = vec![
            document("b", "Beta", "second"),
            document("a", "Alpha", "first"),
            document("c", "Gamma", "third"),
        ];
        assert_eq!(scorer.index(&documents).await.unwrap(), 3);
        assert_eq!(scorer.len(), 3);
        assert!(!scorer.is_empty());

        let capped = make_scorer();
        assert_eq!(capped.index_capped(&documents, 2).await.unwrap(), 2);
        assert_eq!(capped.len(), 2);
        // The first two keys in key order were embedded.
        let capped_scores = capped.scores("alpha").await.unwrap();
        assert_eq!(capped_scores.keys().cloned().collect::<Vec<_>>(), vec!["a", "b"]);

        let empty = make_scorer();
        assert_eq!(empty.index(&[]).await.unwrap(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.search("anything", 0).await.unwrap(), vec![]);
    }

    #[tokio::test]
    async fn scores_are_bounded_and_search_is_ordered() {
        let scorer = make_scorer();
        let documents = vec![
            document("skills.alpha", "Alpha skill", "handles alpha tasks"),
            document("skills.beta", "Beta skill", "handles beta tasks"),
        ];
        scorer.index(&documents).await.unwrap();
        let raw_scores = scorer.scores("alpha tasks").await.unwrap();
        assert_eq!(raw_scores.len(), 2);
        for score in raw_scores.values() {
            assert!((-1.0..=1.0).contains(score), "score out of range: {score}");
        }
        let hits = scorer.search("alpha tasks", 0).await.unwrap();
        assert_eq!(hits.len(), 2);
        let scores_are_descending = hits
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score);
        assert!(scores_are_descending);
        // Deterministic: the same query ranks the same way twice.
        let again = scorer.search("alpha tasks", 0).await.unwrap();
        assert_eq!(again, hits);
        // top_k truncates from the head of that deterministic order.
        assert_eq!(scorer.search("alpha tasks", 1).await.unwrap(), vec![hits[0].clone()]);
    }

    #[tokio::test]
    async fn upsert_and_remove_manage_one_key_at_a_time() {
        let scorer = make_scorer();
        let first = document("a", "Alpha", "first");
        scorer.upsert(&first).await.unwrap();
        assert_eq!(scorer.len(), 1);
        scorer
            .upsert(&document("b", "Beta", "second"))
            .await
            .unwrap();
        assert_eq!(scorer.len(), 2);
        // A re-upsert replaces rather than adds.
        scorer.upsert(&first).await.unwrap();
        assert_eq!(scorer.len(), 2);
        assert!(scorer.remove("a").unwrap());
        assert!(!scorer.remove("a").unwrap());
        assert_eq!(scorer.len(), 1);
        scorer.clear().unwrap();
        assert!(scorer.is_empty());
    }

    #[test]
    fn blend_reproduces_each_ranking_at_the_extremes() {
        let lexical = vec![
            LexicalHit {
                key: "b".to_owned(),
                score: 10.0,
            },
            LexicalHit {
                key: "a".to_owned(),
                score: 5.0,
            },
        ];
        let mut embedding = BTreeMap::new();
        embedding.insert("a".to_owned(), 0.9_f64);
        embedding.insert("b".to_owned(), 0.0_f64);

        let pure_lexical = blend_scores(&lexical, &embedding, 0.0);
        assert_eq!(
            pure_lexical,
            vec![
                LexicalHit {
                    key: "b".to_owned(),
                    score: 1.0
                },
                LexicalHit {
                    key: "a".to_owned(),
                    score: 0.5
                },
            ]
        );

        let pure_embedding = blend_scores(&lexical, &embedding, 1.0);
        assert_eq!(
            pure_embedding,
            vec![
                LexicalHit {
                    key: "a".to_owned(),
                    score: 0.95
                },
                LexicalHit {
                    key: "b".to_owned(),
                    score: 0.5
                },
            ]
        );

        // Clamped and non-finite weights behave like the extremes.
        assert_eq!(blend_scores(&lexical, &embedding, -2.0), pure_lexical);
        assert_eq!(blend_scores(&lexical, &embedding, f64::NAN), pure_lexical);
        assert_eq!(blend_scores(&lexical, &embedding, 9.0), pure_embedding);
    }

    #[test]
    fn blend_unions_the_key_sets_and_ties_break_on_the_key() {
        let lexical = vec![LexicalHit {
            key: "shared".to_owned(),
            score: 0.0,
        }];
        let mut embedding = BTreeMap::new();
        embedding.insert("only-embedding".to_owned(), 0.0_f64);
        // A zero maximum lexical score normalizes everything to 0.0, so the two keys tie
        // and the tie breaks on the key.
        let blended = blend_scores(&lexical, &embedding, 0.0);
        assert_eq!(
            blended,
            vec![
                LexicalHit {
                    key: "only-embedding".to_owned(),
                    score: 0.0
                },
                LexicalHit {
                    key: "shared".to_owned(),
                    score: 0.0
                },
            ]
        );
        let blended = blend_scores(&[], &embedding, 0.5);
        assert_eq!(blended.len(), 1);
        assert_eq!(blended[0].key, "only-embedding");
        assert_eq!(blended[0].score, 0.25);
        assert_eq!(blend_scores(&[], &BTreeMap::new(), 0.5), vec![]);
    }
}
