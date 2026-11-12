//! In-memory cosine vector index over memory nodes.
//!
//! Ports `openjiuwen/extensions/context_evolver/core/vector_store/memory_vector_store.py`,
//! the `numpy`-backed store Python uses for development and tests. The Rust
//! port keeps the observable behavior (cosine similarity, optional metadata
//! filtering, `top_k` selection) without the `numpy` dependency or any
//! embedding/Milvus call: embeddings arrive on [`VectorNode`] and the caller
//! owns similarity ranking semantics.
//!
//! Unlike Python's `dict` index, nodes are kept in an insertion-ordered
//! [`Vec`], so equal-score ties are returned in insertion order and results are
//! deterministic across runs.
//!
//! # Deviation from Python
//!
//! Python scores an empty query against an empty vector as `0.0` (the zero-norm
//! short-circuit fires first). This port returns `1.0` for two empty vectors:
//! they are identical, and the contract for this crate states that behavior
//! explicitly.
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

/// A single memory node together with its embedding and metadata.
///
/// The Python `VectorNode` also carries the source `content`; the Rust contract
/// keeps identifiers, vectors, and metadata only, because content retrieval is
/// the caller's concern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VectorNode {
    /// Unique identifier; an upsert replaces the node with the same id.
    pub id: String,
    /// Embedding vector compared by cosine similarity.
    pub embedding: Vec<f32>,
    /// Arbitrary JSON metadata used by [`MemoryVectorStore::search_with_metadata`].
    pub metadata: BTreeMap<String, Value>,
}

impl VectorNode {
    /// Builds a node with an empty metadata map.
    #[must_use]
    pub fn new(id: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            embedding,
            metadata: BTreeMap::new(),
        }
    }
}

/// A node returned by a search together with its cosine score.
///
/// The score is the raw cosine similarity in `[-1.0, 1.0]` (higher means more
/// similar); results are returned in descending score order. Non-finite scores
/// are clamped to `0.0`, so a scored result never contains `NaN` or infinity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ScoredNode {
    /// The matched node.
    pub node: VectorNode,
    /// Cosine similarity between the query and [`VectorNode::embedding`].
    pub score: f32,
}

/// An insertion-ordered, in-memory cosine index over [`VectorNode`]s.
///
/// Cloning a store shares the same underlying index (the state sits behind an
/// [`Arc`]); use [`MemoryVectorStore::new`] for an independent store.
#[derive(Debug, Clone)]
pub struct MemoryVectorStore {
    nodes: Arc<RwLock<Vec<VectorNode>>>,
}

impl MemoryVectorStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Inserts or replaces nodes by id, returning the number processed.
    ///
    /// A node whose [`VectorNode::id`] is empty or whitespace-only is rejected
    /// with [`Error::Validation`]; the whole batch is validated before any node
    /// is written, so a rejected call leaves the store unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when any node in `nodes` has an empty id.
    pub async fn upsert(&self, nodes: Vec<VectorNode>) -> Result<usize> {
        for (index, node) in nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                let id = &node.id;
                return Err(Error::Validation(format!(
                    "vector node at index {index} has an empty id ({id:?}); every node needs a non-empty id"
                )));
            }
        }
        let count = nodes.len();
        if count == 0 {
            return Ok(0);
        }
        let mut stored = self.nodes.write().await;
        for node in nodes {
            if let Some(position) = stored.iter().position(|existing| existing.id == node.id) {
                stored[position] = node;
            } else {
                stored.push(node);
            }
        }
        Ok(count)
    }

    /// Searches every node for the vectors closest to `query`.
    ///
    /// Results are sorted by descending cosine score with a stable sort, so
    /// equal scores keep insertion order. `top_k` is clamped to the number of
    /// stored nodes; `top_k == 0` returns an empty vector.
    ///
    /// # Errors
    ///
    /// The signature mirrors Python's fallible store API and the module
    /// contract; this implementation has no failure path.
    #[allow(clippy::unnecessary_wraps)] // Contract: the search surface is fallible.
    pub async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredNode>> {
        let filter = BTreeMap::new();
        self.search_with_metadata(query, top_k, &filter).await
    }

    /// Searches only nodes whose metadata matches every entry in `metadata`.
    ///
    /// Keys are exact dotted paths: `"workspace.id"` walks the nested object
    /// `{"workspace": {"id": ...}}`. A missing path, or a non-object value in
    /// the middle of a path, is a non-match. Values compare with `==` on
    /// [`Value`](serde_json::Value), and an empty filter matches every node.
    ///
    /// # Errors
    ///
    /// The signature mirrors Python's fallible store API and the module
    /// contract; this implementation has no failure path.
    #[allow(clippy::unnecessary_wraps)] // Contract: the search surface is fallible.
    pub async fn search_with_metadata(
        &self,
        query: &[f32],
        top_k: usize,
        metadata: &BTreeMap<String, Value>,
    ) -> Result<Vec<ScoredNode>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let stored = self.nodes.read().await;
        let mut ranked: Vec<ScoredNode> = stored
            .iter()
            .filter(|node| metadata_matches(node, metadata))
            .map(|node| ScoredNode {
                node: node.clone(),
                score: cosine_similarity(query, node.embedding.as_slice()),
            })
            .collect();
        drop(stored);
        ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Returns the number of stored nodes.
    #[must_use]
    pub async fn len(&self) -> usize {
        self.nodes.read().await.len()
    }

    /// Returns `true` when the store holds no nodes.
    #[must_use]
    pub async fn is_empty(&self) -> bool {
        self.nodes.read().await.is_empty()
    }

    /// Removes every stored node.
    pub async fn clear(&self) {
        self.nodes.write().await.clear();
    }
}

impl Default for MemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity with the module's documented edge-case semantics.
///
/// - two empty vectors are identical, so the score is `1.0`;
/// - exactly one empty vector, or differing dimensions, scores `0.0`;
/// - a zero-norm vector scores `0.0`;
/// - otherwise the raw cosine `dot / (‖query‖·‖vector‖)` is returned;
/// - a non-finite result (`NaN`, infinity) is clamped to `0.0`.
#[allow(clippy::float_cmp)] // Exact zero is the intended zero-norm test.
fn cosine_similarity(query: &[f32], vector: &[f32]) -> f32 {
    if query.is_empty() && vector.is_empty() {
        return 1.0;
    }
    if query.is_empty() || vector.is_empty() || query.len() != vector.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut query_squared = 0.0_f32;
    let mut vector_squared = 0.0_f32;
    for (query_component, vector_component) in query.iter().zip(vector) {
        dot += query_component * vector_component;
        query_squared += query_component * query_component;
        vector_squared += vector_component * vector_component;
    }
    if query_squared == 0.0 || vector_squared == 0.0 {
        return 0.0;
    }
    let score = dot / (query_squared.sqrt() * vector_squared.sqrt());
    if score.is_finite() {
        // Normalize `-0.0` to `0.0` so ordering is deterministic.
        score + 0.0
    } else {
        0.0
    }
}

/// Resolves an exact dotted path inside a node's metadata.
fn lookup_path<'a>(metadata: &'a BTreeMap<String, Value>, path: &str) -> Option<&'a Value> {
    let mut segments = path.split('.');
    let mut current = metadata.get(segments.next()?)?;
    for segment in segments {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// Returns `true` when every filter entry matches the node's metadata exactly.
fn metadata_matches(node: &VectorNode, filter: &BTreeMap<String, Value>) -> bool {
    filter
        .iter()
        .all(|(path, expected)| lookup_path(&node.metadata, path) == Some(expected))
}

#[cfg(test)]
mod tests {
    // Python counterpart: `tests/unit_tests/extensions/context_evolver/`
    // exercises the store only through the live flows, which are skipped in
    // Python without an API key; the Rust tests pin the ported pure behavior.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    const EPSILON: f32 = 1e-6;

    fn assert_score(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn node(id: &str, embedding: Vec<f32>) -> VectorNode {
        VectorNode::new(id, embedding)
    }

    fn ids(results: &[ScoredNode]) -> Vec<&str> {
        results
            .iter()
            .map(|scored| scored.node.id.as_str())
            .collect()
    }

    #[tokio::test]
    async fn orders_by_descending_cosine() {
        let store = MemoryVectorStore::new();
        store
            .upsert(vec![
                node("far", vec![0.0, 1.0]),
                node("near", vec![1.0, 0.0]),
                node("middle", vec![1.0, 1.0]),
            ])
            .await
            .unwrap();
        let results = store.search(&[1.0, 0.0], 10).await.unwrap();
        assert_eq!(ids(&results), ["near", "middle", "far"]);
        assert_score(results[0].score, 1.0);
        assert_score(results[1].score, std::f32::consts::FRAC_1_SQRT_2);
        assert_score(results[2].score, 0.0);
    }

    #[tokio::test]
    async fn equal_scores_keep_insertion_order() {
        let store = MemoryVectorStore::new();
        store
            .upsert(vec![
                node("first", vec![1.0, 0.0]),
                node("second", vec![2.0, 0.0]),
                node("third", vec![3.0, 0.0]),
            ])
            .await
            .unwrap();
        let results = store.search(&[1.0, 0.0], 10).await.unwrap();
        assert_eq!(ids(&results), ["first", "second", "third"]);
        for scored in &results {
            assert_score(scored.score, 1.0);
        }
    }

    #[tokio::test]
    async fn top_k_clamps_to_candidates_and_zero_returns_empty() {
        let store = MemoryVectorStore::new();
        store
            .upsert(vec![
                node("a", vec![1.0]),
                node("b", vec![0.5]),
                node("c", vec![0.25]),
            ])
            .await
            .unwrap();
        assert_eq!(store.search(&[1.0], 10).await.unwrap().len(), 3);
        assert_eq!(store.search(&[1.0], 2).await.unwrap().len(), 2);
        assert_eq!(store.search(&[1.0], 0).await.unwrap().len(), 0);
        assert_eq!(
            MemoryVectorStore::new()
                .search(&[1.0], 5)
                .await
                .unwrap()
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn empty_vectors_score_one_against_each_other() {
        let store = MemoryVectorStore::new();
        store.upsert(vec![node("empty", Vec::new())]).await.unwrap();
        let results = store.search(&[], 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_score(results[0].score, 1.0);
    }

    #[tokio::test]
    async fn empty_versus_non_empty_or_mismatched_dimensions_score_zero() {
        let store = MemoryVectorStore::new();
        store
            .upsert(vec![node("two", vec![1.0, 2.0]), node("empty", Vec::new())])
            .await
            .unwrap();

        let empty_query = store.search(&[], 10).await.unwrap();
        let empty_scores: BTreeMap<&str, f32> = empty_query
            .iter()
            .map(|scored| (scored.node.id.as_str(), scored.score))
            .collect();
        assert_score(empty_scores["two"], 0.0);
        assert_score(empty_scores["empty"], 1.0);

        let mismatched = store.search(&[1.0, 2.0, 3.0], 10).await.unwrap();
        for scored in mismatched {
            assert_score(scored.score, 0.0);
        }
    }

    #[tokio::test]
    async fn zero_vector_and_nan_components_score_zero() {
        let store = MemoryVectorStore::new();
        store
            .upsert(vec![
                node("zero", vec![0.0, 0.0]),
                node("nan", vec![f32::NAN, 1.0]),
            ])
            .await
            .unwrap();
        let results = store.search(&[1.0, 0.0], 10).await.unwrap();
        assert_eq!(results.len(), 2);
        for scored in &results {
            assert!(scored.score.is_finite());
            assert_score(scored.score, 0.0);
        }
    }

    #[tokio::test]
    async fn metadata_filter_supports_exact_dotted_paths() {
        let store = MemoryVectorStore::new();
        let mut first = node("first", vec![1.0, 0.0]);
        first
            .metadata
            .insert("workspace".into(), json!({"id": "w1"}));
        first.metadata.insert("tag".into(), json!("alpha"));
        let mut second = node("second", vec![1.0, 0.0]);
        second
            .metadata
            .insert("workspace".into(), json!({"id": "w2"}));
        second.metadata.insert("tag".into(), json!("beta"));
        store.upsert(vec![first, second]).await.unwrap();

        let mut filter = BTreeMap::new();
        filter.insert("workspace.id".to_string(), json!("w1"));
        let results = store
            .search_with_metadata(&[1.0, 0.0], 10, &filter)
            .await
            .unwrap();
        assert_eq!(ids(&results), ["first"]);

        filter.insert("tag".to_string(), json!("alpha"));
        let both = store
            .search_with_metadata(&[1.0, 0.0], 10, &filter)
            .await
            .unwrap();
        assert_eq!(ids(&both), ["first"]);

        filter.insert("tag".to_string(), json!("beta"));
        assert_eq!(
            store
                .search_with_metadata(&[1.0, 0.0], 10, &filter)
                .await
                .unwrap()
                .len(),
            0
        );

        let mut non_matching = BTreeMap::new();
        non_matching.insert("workspace.id".to_string(), json!("missing"));
        assert_eq!(
            store
                .search_with_metadata(&[1.0, 0.0], 10, &non_matching)
                .await
                .unwrap()
                .len(),
            0
        );

        let mut through_scalar = BTreeMap::new();
        through_scalar.insert("tag.nested".to_string(), json!("alpha"));
        assert_eq!(
            store
                .search_with_metadata(&[1.0, 0.0], 10, &through_scalar)
                .await
                .unwrap()
                .len(),
            0
        );

        let unfiltered = store
            .search_with_metadata(&[1.0, 0.0], 10, &BTreeMap::new())
            .await
            .unwrap();
        assert_eq!(ids(&unfiltered), ["first", "second"]);
    }

    #[tokio::test]
    async fn upsert_replaces_by_id_and_counts_batch() {
        let store = MemoryVectorStore::new();
        assert_eq!(
            store
                .upsert(vec![node("a", vec![1.0, 0.0]), node("b", vec![0.0, 1.0])])
                .await
                .unwrap(),
            2
        );
        assert_eq!(store.len().await, 2);
        assert_eq!(
            store.upsert(vec![node("a", vec![0.0, 1.0])]).await.unwrap(),
            1
        );
        assert_eq!(store.len().await, 2);
        let results = store.search(&[0.0, 1.0], 10).await.unwrap();
        let embeddings: BTreeMap<&str, &Vec<f32>> = results
            .iter()
            .map(|scored| (scored.node.id.as_str(), &scored.node.embedding))
            .collect();
        let expected_embedding = [0.0_f32, 1.0];
        assert_eq!(embeddings["a"].as_slice(), expected_embedding.as_slice());
        assert_score(store.search(&[0.0, 1.0], 1).await.unwrap()[0].score, 1.0);
        assert_eq!(store.upsert(Vec::new()).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn upsert_rejects_empty_ids_without_partial_write() {
        let store = MemoryVectorStore::new();
        let error = store
            .upsert(vec![node("valid", vec![1.0]), node("   ", vec![2.0])])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("index 1"), "{}", error.message());
        assert!(store.is_empty().await);

        let error = store.upsert(vec![node("", Vec::new())]).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(store.is_empty().await);
    }

    #[tokio::test]
    async fn clear_and_lengths() {
        let store = MemoryVectorStore::new();
        assert!(store.is_empty().await);
        assert_eq!(store.len().await, 0);
        store
            .upsert(vec![node("a", vec![1.0]), node("b", vec![2.0])])
            .await
            .unwrap();
        assert_eq!(store.len().await, 2);
        assert!(!store.is_empty().await);
        store.clear().await;
        assert_eq!(store.len().await, 0);
        assert!(store.is_empty().await);
    }

    #[tokio::test]
    async fn node_and_score_round_trip_through_json() {
        let mut vector = node("id-1", vec![0.5, 0.25]);
        vector.metadata.insert("k".into(), json!({"nested": true}));
        let encoded = serde_json::to_string(&vector).unwrap();
        let decoded: VectorNode = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, vector);

        let scored = ScoredNode {
            node: vector,
            score: 0.75,
        };
        let encoded = serde_json::to_string(&scored).unwrap();
        let decoded: ScoredNode = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, scored);
    }

    #[tokio::test]
    async fn default_store_is_empty() {
        let store = MemoryVectorStore::default();
        assert!(store.is_empty().await);
    }
}
