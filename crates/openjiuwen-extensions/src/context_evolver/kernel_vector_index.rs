//! The kernel-`VectorStore`-backed memory index.
//!
//! Ports the *storage* half of Python's
//! `openjiuwen/extensions/context_evolver/core/vector_store/` package: Python
//! keeps per-user memory embeddings in Milvus and scores retrieval with the
//! vector database's similarity search. Rust keeps the same seam
//! ([`VectorIndex`](super::memory_store::VectorIndex)) and implements it over
//! any [`VectorStore`] the kernel offers, so a memory index can live in process
//! memory ([`InMemoryVectorStore`]), in local JSON files
//! ([`LocalFileVectorStore`]), in `Elasticsearch`
//! ([`ElasticsearchVectorStore`](crate::store::elasticsearch::ElasticsearchVectorStore)),
//! or in a vendor `Milvus` adapter registered with the kernel — the evolver
//! itself stays offline and dependency-free.
//!
//! # What the collection holds
//!
//! One document per [`VectorNode`]:
//!
//! | field | type | meaning |
//! | --- | --- | --- |
//! | [`ID_FIELD`] | `varchar`, primary key | [`VectorNode::id`] |
//! | [`EMBEDDING_FIELD`] | `float_vector`, `dim` from the first upsert | [`VectorNode::embedding`] |
//! | [`METADATA_FIELD`] | `json` | [`VectorNode::metadata`] as one object |
//!
//! The collection is created lazily on the first
//! [`VectorIndex::upsert`](super::memory_store::VectorIndex::upsert), from that
//! batch's embedding dimension; a later batch of a different dimension fails
//! with [`Error::Validation`] naming both dimensions, so one collection never
//! mixes embedding models.
//!
//! # Scores
//!
//! The kernel normalizes every similarity into `[0, 1]` (`(cosine + 1) / 2` for
//! the cosine metric), while the evolver's contract is the *raw* cosine in
//! `[-1.0, 1.0]` — the value
//! [`HashingEmbedding`](crate::context_evolver::HashingEmbedding) blends with
//! the lexical score. This index inverts the kernel's normalization, so a
//! service ranks and blends identically over both backends.
//!
//! # Limitations (documented, never silent)
//!
//! * Only [`Metric::Cosine`] is supported; [`KernelVectorIndex::with_metric`]
//!   rejects the other metrics with [`Error::Unsupported`], because their
//!   scores are not cosines and could not be inverted.
//! * Metadata filtering uses the kernel's shared filter language on
//!   `<metadata>.<dotted path>`, so a backend that cannot express an equality
//!   (for example `Elasticsearch` on an object value) reports its own typed
//!   error instead of returning a wrong match set.
//! * The kernel's document contract is insert-only, so [`VectorIndex::upsert`]
//!   deletes the named primary keys and re-inserts them; the two steps are not
//!   transactional, and the batch is validated (ids, dimensions) before the
//!   first delete.
//! * A backend that does not echo the stored vector in its search hits yields
//!   nodes with an empty [`VectorNode::embedding`]; the score and id are always
//!   authoritative.
//! * Equal scores are ordered by primary key ascending, the kernel's documented
//!   tie-break, where the in-process index keeps insertion order. A service
//!   re-sorts through its shared `finalize_hits`, so service output never
//!   depends on which backend answered.
//!
//! Python counterparts: `core/vector_store/milvus_vector_store.py` and the
//! `retrieve/` pipelines that score through it.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, Metric, VectorDataType, VectorQuery, VectorSearchResult,
    VectorStore,
};
use serde_json::{json, Map, Value};

use super::memory_store::{ScoredNode, VectorIndex, VectorNode};

/// The document field holding a node's id (the collection's primary key).
pub const ID_FIELD: &str = "id";
/// The document field holding a node's embedding.
pub const EMBEDDING_FIELD: &str = "embedding";
/// The document field holding a node's metadata object.
pub const METADATA_FIELD: &str = "metadata";

/// A [`VectorIndex`] whose nodes live in a kernel [`VectorStore`] collection.
///
/// ```no_run
/// # async fn demo() -> Result<(), openjiuwen_common::error::Error> {
/// use std::sync::Arc;
/// use openjiuwen_core::foundation::store::InMemoryVectorStore;
/// use openjiuwen_extensions::context_evolver::{KernelVectorIndex, VectorIndex, VectorNode};
///
/// let index = KernelVectorIndex::new(Arc::new(InMemoryVectorStore::new()), "memories")?;
/// index
///     .upsert(vec![VectorNode::new("memory-1", vec![1.0, 0.0])])
///     .await?;
/// let hits = index.search(&[1.0, 0.0], 1).await?;
/// assert_eq!(hits[0].node.id, "memory-1");
/// assert_eq!(hits[0].score, 1.0);
/// # Ok(())
/// # }
/// ```
///
/// The store is held behind the [`Arc`] the caller supplies, so several
/// indexes (and several services) can share one store instance; use the same
/// collection name to share the nodes.
#[derive(Clone)]
pub struct KernelVectorIndex {
    /// The backing store every operation goes through.
    store: Arc<dyn VectorStore>,
    /// The collection this index owns inside `store`.
    collection: String,
}

impl fmt::Debug for KernelVectorIndex {
    /// Reports the collection only: the backing store's configuration (URLs,
    /// credentials) is its own business and is never printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KernelVectorIndex")
            .field("collection", &self.collection)
            .finish_non_exhaustive()
    }
}

impl KernelVectorIndex {
    /// Builds an index over `collection` in `store`, using the cosine metric.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `collection` is blank.
    pub fn new(store: Arc<dyn VectorStore>, collection: impl Into<String>) -> Result<Self> {
        Self::with_metric(store, collection, Metric::Cosine)
    }

    /// Builds an index over `collection` for a store configured with `metric`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `collection` is blank and
    /// [`Error::Unsupported`] for [`Metric::Dot`] and [`Metric::Euclidean`]:
    /// their kernel scores are not cosines, so they cannot be converted back to
    /// the raw similarity this crate's services blend with.
    pub fn with_metric(
        store: Arc<dyn VectorStore>,
        collection: impl Into<String>,
        metric: Metric,
    ) -> Result<Self> {
        let collection = collection.into();
        if collection.trim().is_empty() {
            return Err(Error::validation(
                "a store-backed memory index needs a non-empty collection name",
            ));
        }
        if metric != Metric::Cosine {
            return Err(Error::unsupported(format!(
                "the store-backed memory index reports raw cosine similarities, so it supports \
                 the cosine metric only; {metric:?} cannot be converted — configure the backing \
                 vector store with Metric::Cosine"
            )));
        }
        Ok(Self { store, collection })
    }

    /// The collection this index reads and writes.
    #[must_use]
    pub fn collection(&self) -> &str {
        &self.collection
    }

    /// The backing store, for a host that wants to manage the collection.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn VectorStore> {
        &self.store
    }

    /// The schema created for a `dim`-dimensional collection.
    fn schema(dim: usize) -> CollectionSchema {
        CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                ..FieldSchema::new(ID_FIELD, VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(dim),
                ..FieldSchema::new(EMBEDDING_FIELD, VectorDataType::FloatVector)
            },
            FieldSchema::new(METADATA_FIELD, VectorDataType::Json),
        ])
    }

    /// Creates the collection for `dim` buckets, or checks an existing one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a collection of the same name stores
    /// a different dimension (or has no vector field at all), and the store's
    /// own construction error when creation fails.
    async fn ensure_collection(&self, dim: usize) -> Result<()> {
        match self.store.get_schema(&self.collection).await? {
            None => {
                self.store
                    .create_collection(&self.collection, Self::schema(dim))
                    .await
            }
            Some(schema) => {
                let Some(field) = schema.default_vector_field() else {
                    return Err(Error::validation(format!(
                        "vector collection {:?} declares no FLOAT_VECTOR field; a memory index \
                         needs one {EMBEDDING_FIELD:?} field",
                        self.collection
                    )));
                };
                let stored = field.dim.unwrap_or(0);
                if stored != dim {
                    return Err(Error::validation(format!(
                        "the memory index embeds {dim}-dimension vectors but vector collection {:?} \
                         stores {stored}-dimension vectors (field {:?}); use one embedding model \
                         per collection",
                        self.collection, field.name
                    )));
                }
                Ok(())
            }
        }
    }

    /// Fails with the kernel's not-found error when the collection is absent.
    async fn require_collection(&self) -> Result<()> {
        if self.store.collection_exists(&self.collection).await? {
            Ok(())
        } else {
            Err(Error::not_found(format!(
                "vector collection {:?} not found; the store-backed memory index creates it on \
                 the first upsert",
                self.collection
            )))
        }
    }

    /// Runs one similarity search, translating the kernel's result shape.
    async fn run_query(&self, query: &VectorQuery) -> Result<Vec<ScoredNode>> {
        self.require_collection().await?;
        let results = self.store.search(&self.collection, query).await?;
        results.into_iter().map(|hit| scored_node(&hit)).collect()
    }
}

#[async_trait]
impl VectorIndex for KernelVectorIndex {
    async fn upsert(&self, nodes: Vec<VectorNode>) -> Result<usize> {
        for (position, node) in nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(Error::validation(format!(
                    "vector node at index {position} has an empty id ({:?}); every node needs a \
                     non-empty id",
                    node.id
                )));
            }
        }
        if nodes.is_empty() {
            return Ok(0);
        }
        let dim = nodes[0].embedding.len();
        if dim == 0 {
            return Err(Error::validation(format!(
                "vector node {:?} has an empty embedding; a store-backed index needs the \
                 embedding dimension to create its collection",
                nodes[0].id
            )));
        }
        if let Some((position, node)) = nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.embedding.len() != dim)
        {
            return Err(Error::validation(format!(
                "vector node {:?} (index {position}) has {} embedding components but node {:?} \
                 has {dim}; every node in one index must share the embedding dimension",
                node.id,
                node.embedding.len(),
                nodes[0].id
            )));
        }
        self.ensure_collection(dim).await?;
        // Last occurrence wins, matching the in-process index's replace-by-id.
        let mut unique: BTreeMap<&str, &VectorNode> = BTreeMap::new();
        for node in &nodes {
            unique.insert(node.id.as_str(), node);
        }
        let ids: Vec<String> = unique.keys().map(|id| (*id).to_string()).collect();
        // The kernel document contract is insert-only: an upsert is a delete of
        // the named primary keys followed by an insert of the new documents.
        self.store
            .delete_docs_by_ids(&self.collection, &ids)
            .await?;
        let documents: Vec<Map<String, Value>> =
            unique.values().map(|node| node_document(node)).collect();
        self.store.add_docs(&self.collection, documents).await?;
        Ok(nodes.len())
    }

    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredNode>> {
        if top_k == 0 {
            self.require_collection().await?;
            return Ok(Vec::new());
        }
        self.run_query(&VectorQuery::new(query.to_vec()).with_top_k(top_k))
            .await
    }

    async fn search_with_metadata(
        &self,
        query: &[f32],
        top_k: usize,
        metadata: &BTreeMap<String, Value>,
    ) -> Result<Vec<ScoredNode>> {
        if top_k == 0 {
            self.require_collection().await?;
            return Ok(Vec::new());
        }
        self.run_query(
            &VectorQuery::new(query.to_vec())
                .with_top_k(top_k)
                .with_filters(metadata_filter(metadata)),
        )
        .await
    }

    async fn search_ids(&self, query: &[f32], ids: &[String]) -> Result<Vec<ScoredNode>> {
        if ids.is_empty() {
            self.require_collection().await?;
            return Ok(Vec::new());
        }
        self.run_query(
            &VectorQuery::new(query.to_vec())
                .with_top_k(ids.len())
                .with_filters(json!({ ID_FIELD: { "$in": ids } })),
        )
        .await
    }

    async fn clear(&self) -> Result<()> {
        if !self.store.collection_exists(&self.collection).await? {
            return Ok(());
        }
        // A `null` filter matches every document, so this empties the
        // collection while keeping its schema.
        self.store
            .delete_docs_by_filters(&self.collection, &Value::Null)
            .await?;
        Ok(())
    }
}

/// Renders one node as the stored document.
fn node_document(node: &VectorNode) -> Map<String, Value> {
    let mut document = Map::new();
    document.insert(ID_FIELD.to_string(), Value::from(node.id.clone()));
    document.insert(
        EMBEDDING_FIELD.to_string(),
        Value::Array(
            node.embedding
                .iter()
                .map(|component| Value::from(*component))
                .collect(),
        ),
    );
    document.insert(
        METADATA_FIELD.to_string(),
        Value::Object(
            node.metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
    );
    document
}

/// Translates the evolver's dotted-path filter into the kernel filter language.
///
/// An empty filter becomes `null`, which matches every document.
fn metadata_filter(metadata: &BTreeMap<String, Value>) -> Value {
    if metadata.is_empty() {
        return Value::Null;
    }
    let mut filter = Map::new();
    for (path, expected) in metadata {
        filter.insert(format!("{METADATA_FIELD}.{path}"), expected.clone());
    }
    Value::Object(filter)
}

/// Converts one kernel hit into a scored node.
fn scored_node(result: &VectorSearchResult) -> Result<ScoredNode> {
    let id = match result.fields.get(ID_FIELD) {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        Some(other) => {
            return Err(Error::validation(format!(
                "vector store hit has a non-string {ID_FIELD:?} field: {other}"
            )));
        }
        None => {
            return Err(Error::validation(format!(
                "vector store hit carries no {ID_FIELD:?} field"
            )));
        }
    };
    Ok(ScoredNode {
        node: VectorNode {
            id,
            embedding: embedding_of(&result.fields)?,
            metadata: metadata_of(&result.fields),
        },
        score: raw_cosine(result.score),
    })
}

/// Reads the embedding a hit echoes, or an empty vector when it echoes none.
///
/// The cast is the kernel's own `f64`-component storage narrowed back to the
/// evolver's `f32` embeddings, exactly as the kernel's search path stores them.
#[allow(clippy::cast_possible_truncation)]
fn embedding_of(fields: &Map<String, Value>) -> Result<Vec<f32>> {
    match fields.get(EMBEDDING_FIELD) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(components)) => components
            .iter()
            .enumerate()
            .map(|(position, component)| {
                component.as_f64().map(|value| value as f32).ok_or_else(|| {
                    Error::validation(format!(
                        "vector store hit has a non-numeric {EMBEDDING_FIELD:?} component at \
                         position {position}: {component}"
                    ))
                })
            })
            .collect(),
        Some(other) => Err(Error::validation(format!(
            "vector store hit has a {EMBEDDING_FIELD:?} field that is not an array: {other}"
        ))),
    }
}

/// Reads the metadata object a hit carries.
fn metadata_of(fields: &Map<String, Value>) -> BTreeMap<String, Value> {
    fields
        .get(METADATA_FIELD)
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Inverts the kernel's `(cosine + 1) / 2` normalization back to the raw cosine.
fn raw_cosine(score: f32) -> f32 {
    let raw = 2.0 * score - 1.0;
    if raw.is_finite() {
        // Normalize `-0.0` to `0.0` so ordering is deterministic.
        raw.clamp(-1.0, 1.0) + 0.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    // Python counterpart: `tests/unit_tests/extensions/context_evolver/` runs
    // its scoring only through Milvus-backed live flows, which are skipped
    // without credentials; these tests pin the ported seam offline.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;
    use crate::context_evolver::MemoryVectorStore;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_core::foundation::store::{InMemoryVectorStore, LocalFileVectorStore};

    const EPSILON: f32 = 1e-6;

    fn index(collection: &str) -> KernelVectorIndex {
        KernelVectorIndex::new(Arc::new(InMemoryVectorStore::new()), collection).unwrap()
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

    fn assert_score(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[tokio::test]
    async fn upsert_creates_the_collection_and_ranks_like_the_in_process_index() {
        let nodes = vec![
            node("far", vec![0.0, 1.0]),
            node("near", vec![1.0, 0.0]),
            node("middle", vec![1.0, 1.0]),
        ];
        let store_backed = index("memories");
        assert_eq!(store_backed.upsert(nodes.clone()).await.unwrap(), 3);
        let in_process = MemoryVectorStore::new();
        in_process.upsert(nodes).await.unwrap();

        let stored = store_backed.search(&[1.0, 0.0], 10).await.unwrap();
        let memory = in_process.search(&[1.0, 0.0], 10).await.unwrap();
        assert_eq!(ids(&stored), ["near", "middle", "far"]);
        assert_eq!(ids(&stored), ids(&memory));
        for (left, right) in stored.iter().zip(&memory) {
            assert_score(left.score, right.score);
        }
        assert_score(stored[1].score, std::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(stored[0].node.embedding, vec![1.0, 0.0]);
    }

    #[tokio::test]
    async fn upsert_replaces_by_id_and_collapses_a_repeated_id() {
        let store_backed = index("memories");
        store_backed
            .upsert(vec![node("a", vec![1.0, 0.0]), node("b", vec![0.0, 1.0])])
            .await
            .unwrap();
        store_backed
            .upsert(vec![node("a", vec![0.0, 1.0])])
            .await
            .unwrap();
        // The id appears twice in one batch: the last occurrence wins, as in
        // the in-process index.
        assert_eq!(
            store_backed
                .upsert(vec![node("c", vec![1.0, 0.0]), node("c", vec![0.5, 0.5])])
                .await
                .unwrap(),
            2
        );
        let hits = store_backed
            .search_ids(&[1.0, 0.0], &["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        // `c` is the replaced node; `a` and `b` tie at `0.0`, and the kernel
        // breaks ties by primary key ascending (the in-process index keeps
        // insertion order instead).
        assert_eq!(ids(&hits), ["c", "a", "b"]);
        assert_score(hits[0].score, std::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(hits[0].node.embedding, vec![0.5, 0.5]);
        assert_score(hits[1].score, 0.0);
        assert_eq!(hits[1].node.embedding, vec![0.0, 1.0]);
    }

    #[tokio::test]
    async fn upsert_rejects_bad_ids_and_inconsistent_embeddings() {
        let store_backed = index("memories");
        let error = store_backed
            .upsert(vec![node("valid", vec![1.0]), node("  ", vec![2.0])])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("index 1"), "{}", error.message());
        assert!(store_backed.search(&[1.0], 1).await.is_err());

        let error = index("memories")
            .upsert(vec![node("a", Vec::new())])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("empty embedding"),
            "{}",
            error.message()
        );

        let error = index("memories")
            .upsert(vec![node("a", vec![1.0, 0.0]), node("b", vec![1.0])])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("share the embedding dimension"),
            "{}",
            error.message()
        );

        assert_eq!(index("memories").upsert(Vec::new()).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn upsert_rejects_a_dimension_the_collection_does_not_store() {
        let store_backed = index("memories");
        store_backed
            .upsert(vec![node("a", vec![1.0, 0.0])])
            .await
            .unwrap();
        let error = store_backed
            .upsert(vec![node("b", vec![1.0, 0.0, 0.0])])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.message().contains("3-dimension") && error.message().contains("2-dimension"),
            "{}",
            error.message()
        );
    }

    #[tokio::test]
    async fn searching_an_unknown_collection_reports_not_found() {
        let store_backed = index("memories");
        let error = store_backed.search(&[1.0], 1).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.message().contains("memories"), "{}", error.message());

        let error = store_backed.clear().await;
        assert!(error.is_ok(), "clearing an empty index is a no-op");
        let mut filter = BTreeMap::new();
        filter.insert("tag".to_string(), json!("alpha"));
        assert_eq!(
            store_backed
                .search_with_metadata(&[1.0], 1, &filter)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::NotFound
        );
        assert_eq!(
            store_backed
                .search_ids(&[1.0], &["a".into()])
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::NotFound
        );
    }

    #[tokio::test]
    async fn searching_with_the_wrong_query_dimension_is_rejected() {
        let store_backed = index("memories");
        store_backed
            .upsert(vec![node("a", vec![1.0, 0.0])])
            .await
            .unwrap();
        let error = store_backed.search(&[1.0, 0.0, 0.0], 1).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("dimension"), "{}", error.message());
    }

    #[tokio::test]
    async fn metadata_filters_use_the_same_dotted_paths_as_the_in_process_index() {
        let store_backed = index("memories");
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
        store_backed
            .upsert(vec![first.clone(), second.clone()])
            .await
            .unwrap();
        let in_process = MemoryVectorStore::new();
        in_process.upsert(vec![first, second]).await.unwrap();

        let mut filter = BTreeMap::new();
        filter.insert("workspace.id".to_string(), json!("w1"));
        let stored = store_backed
            .search_with_metadata(&[1.0, 0.0], 10, &filter)
            .await
            .unwrap();
        assert_eq!(ids(&stored), ["first"]);
        assert_eq!(
            ids(&in_process
                .search_with_metadata(&[1.0, 0.0], 10, &filter)
                .await
                .unwrap()),
            ["first"]
        );
        assert_eq!(stored[0].node.metadata.len(), 2);

        filter.insert("tag".to_string(), json!("beta"));
        assert_eq!(
            store_backed
                .search_with_metadata(&[1.0, 0.0], 10, &filter)
                .await
                .unwrap()
                .len(),
            0
        );
        let unfiltered = store_backed
            .search_with_metadata(&[1.0, 0.0], 10, &BTreeMap::new())
            .await
            .unwrap();
        assert_eq!(ids(&unfiltered), ["first", "second"]);
    }

    #[tokio::test]
    async fn search_ids_scores_only_the_named_nodes() {
        let store_backed = index("memories");
        store_backed
            .upsert(vec![
                node("a", vec![1.0, 0.0]),
                node("b", vec![0.0, 1.0]),
                node("c", vec![1.0, 1.0]),
            ])
            .await
            .unwrap();
        let none: [&str; 0] = [];
        assert_eq!(
            ids(&store_backed.search_ids(&[1.0, 0.0], &[]).await.unwrap()),
            none
        );
        let hits = store_backed
            .search_ids(&[1.0, 0.0], &["c".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(ids(&hits), ["c", "b"]);
        assert_score(hits[0].score, std::f32::consts::FRAC_1_SQRT_2);
        // An id the index does not hold is simply absent.
        let hits = store_backed
            .search_ids(&[1.0, 0.0], &["a".into(), "ghost".into()])
            .await
            .unwrap();
        assert_eq!(ids(&hits), ["a"]);
    }

    #[tokio::test]
    async fn clear_empties_the_collection_and_keeps_its_schema() {
        let store_backed = index("memories");
        store_backed
            .upsert(vec![node("a", vec![1.0, 0.0])])
            .await
            .unwrap();
        store_backed.clear().await.unwrap();
        store_backed.clear().await.unwrap();
        assert_eq!(
            store_backed.search(&[1.0, 0.0], 10).await.unwrap(),
            Vec::new()
        );
        // The schema survives, so the same dimension can be written again.
        store_backed
            .upsert(vec![node("b", vec![0.0, 1.0])])
            .await
            .unwrap();
        assert_eq!(
            ids(&store_backed.search(&[0.0, 1.0], 10).await.unwrap()),
            ["b"]
        );
    }

    #[tokio::test]
    async fn construction_rejects_blank_collections_and_non_cosine_metrics() {
        let error = KernelVectorIndex::new(Arc::new(InMemoryVectorStore::new()), "  ").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        for metric in [Metric::Dot, Metric::Euclidean] {
            let error = KernelVectorIndex::with_metric(
                Arc::new(InMemoryVectorStore::new()),
                "memories",
                metric,
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(
                error.message().contains("cosine metric only"),
                "{}",
                error.message()
            );
        }
        let index = KernelVectorIndex::with_metric(
            Arc::new(InMemoryVectorStore::new()),
            "memories",
            Metric::Cosine,
        )
        .unwrap();
        assert_eq!(index.collection(), "memories");
        assert!(format!("{index:?}").contains("memories"));
    }

    #[tokio::test]
    async fn a_local_file_store_keeps_the_index_across_a_reopen() {
        let root = tempfile::tempdir().unwrap();
        let built = KernelVectorIndex::new(
            Arc::new(LocalFileVectorStore::new(root.path()).await.unwrap()),
            "memories",
        )
        .unwrap();
        built
            .upsert(vec![node("a", vec![1.0, 0.0]), node("b", vec![0.0, 1.0])])
            .await
            .unwrap();

        // A second store opened on the same root sees the committed nodes,
        // which is what makes the index outlive the service that built it.
        let reopened = KernelVectorIndex::new(
            Arc::new(LocalFileVectorStore::new(root.path()).await.unwrap()),
            "memories",
        )
        .unwrap();
        let hits = reopened.search(&[1.0, 0.0], 10).await.unwrap();
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_score(hits[0].score, 1.0);
        assert_eq!(hits[0].node.embedding, vec![1.0, 0.0]);
    }
}
