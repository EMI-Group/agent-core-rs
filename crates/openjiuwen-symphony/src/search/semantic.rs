//! Embedding-backed semantic capability search.
//!
//! Python's symphony engine has no vector path: it selects capabilities by walking an
//! LLM-built tree. The Rust port adds an additive, offline-capable alternative on top
//! of `openjiuwen-retrieval`: one [`openjiuwen_retrieval::VectorRecord`] per capability, keyed by its
//! qualified id and carrying the descriptor JSON in its metadata, so a hit can be
//! turned back into a full [`CapabilityDescriptor`] without a catalog lookup.
//!
//! Determinism is the caller's choice of [`openjiuwen_retrieval::Embedding`]: production uses a model,
//! tests use [`openjiuwen_retrieval::HashEmbedding`], which derives vectors purely from
//! the input text.

use std::fmt;
use std::sync::Arc;

use openjiuwen_retrieval::{
    Embedding, InMemoryVectorStore, SearchResult, VectorQuery, VectorRecord, VectorStore,
};
use tracing::warn;

use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;
use crate::search::SearchHit;

/// The metadata key holding the capability type of an indexed record.
pub const CAPABILITY_TYPE_FIELD: &str = "capability_type";
/// The metadata key holding the capability id of an indexed record.
pub const CAPABILITY_ID_FIELD: &str = "capability_id";
/// The metadata key holding the serialized descriptor of an indexed record.
pub const DESCRIPTOR_FIELD: &str = "descriptor";

/// A semantic index over capability cards, backed by an embedding model.
///
/// The store is a per-index [`InMemoryVectorStore`], so each index owns its records
/// and `clear` cannot affect other indexes.
pub struct SemanticIndex {
    /// The embedder used for documents and queries.
    embedding: Arc<dyn Embedding>,
    /// The record store: one record per capability, keyed by qualified id.
    store: InMemoryVectorStore,
}

impl fmt::Debug for SemanticIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticIndex")
            .field("model", &self.embedding.model_name())
            .field("store", &self.store)
            .finish()
    }
}

impl SemanticIndex {
    /// Creates an empty index that embeds with `embedding`.
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self {
            embedding,
            store: InMemoryVectorStore::new(),
        }
    }

    /// Embeds and indexes every descriptor, returning how many records were written.
    ///
    /// Each descriptor becomes one record whose id is its qualified id, whose text is
    /// [`CapabilityDescriptor::retrieval_text`], and whose metadata carries the
    /// capability type, the capability id, and the descriptor JSON. An existing record
    /// with the same id is replaced.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error (including [`Error::Validation`] for a blank
    /// retrieval text) or the store's error.
    pub async fn index(&self, descriptors: &[CapabilityDescriptor]) -> Result<usize> {
        if descriptors.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = descriptors
            .iter()
            .map(CapabilityDescriptor::retrieval_text)
            .collect();
        let vectors = self.embedding.embed_documents(&texts, None).await?;
        let mut records = Vec::with_capacity(descriptors.len());
        for ((descriptor, text), vector) in descriptors.iter().zip(texts).zip(vectors) {
            records.push(vector_record(descriptor, text, vector)?);
        }
        self.store.upsert(records).await
    }

    /// Embeds and stores one descriptor, replacing any record with the same id.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error or the store's error.
    pub async fn upsert(&self, descriptor: &CapabilityDescriptor) -> Result<()> {
        let text = descriptor.retrieval_text();
        let vector = self.embedding.embed_query(&text).await?;
        let record = vector_record(descriptor, text, vector)?;
        self.store.upsert(vec![record]).await?;
        Ok(())
    }

    /// Removes the record of `descriptor`, reporting whether it existed.
    ///
    /// # Errors
    ///
    /// Returns the store's error.
    pub async fn remove(&self, descriptor: &CapabilityDescriptor) -> Result<bool> {
        let removed = self.store.delete(&[descriptor.qualified_id()]).await?;
        Ok(removed > 0)
    }

    /// Removes every indexed record.
    ///
    /// # Errors
    ///
    /// Returns the store's error.
    pub async fn clear(&self) -> Result<()> {
        self.store.clear().await
    }

    /// Embeds `query` and returns the closest capabilities.
    ///
    /// `top_k == 0` means unlimited. Hits carry the reconstructed descriptor and an
    /// empty [`SearchHit::matched_terms`], and are ordered by descending score, then
    /// ascending qualified id. A blank query yields no hits, mirroring the keyword
    /// search.
    ///
    /// Records whose `descriptor` metadata is missing or malformed are skipped with a
    /// warning instead of failing the search.
    ///
    /// # Errors
    ///
    /// Returns the embedding model's error or the store's error.
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = if top_k == 0 {
            self.store.len().await?
        } else {
            top_k
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let vector = self.embedding.embed_query(query).await?;
        let results = self
            .store
            .search(VectorQuery::new(vector).with_top_k(limit))
            .await?;
        let mut hits = Vec::with_capacity(results.len());
        for result in results {
            if let Some(hit) = hit_from_result(&result) {
                hits.push(hit);
            }
        }
        sort_hits(&mut hits);
        Ok(hits)
    }
}

/// Builds the stored record for `descriptor`.
///
/// # Errors
///
/// Returns [`Error::External`] when the descriptor cannot be serialized into metadata.
fn vector_record(
    descriptor: &CapabilityDescriptor,
    text: String,
    vector: Vec<f32>,
) -> Result<VectorRecord> {
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        CAPABILITY_TYPE_FIELD.to_string(),
        serde_json::Value::String(descriptor.capability_type.clone()),
    );
    metadata.insert(
        CAPABILITY_ID_FIELD.to_string(),
        serde_json::Value::String(descriptor.capability_id.clone()),
    );
    metadata.insert(
        DESCRIPTOR_FIELD.to_string(),
        serde_json::to_value(descriptor).map_err(Error::external)?,
    );
    Ok(VectorRecord {
        id: descriptor.qualified_id(),
        text,
        vector,
        metadata,
        doc_id: None,
        chunk_id: None,
    })
}

/// Reconstructs a hit from a store result, skipping records without a usable
/// descriptor.
fn hit_from_result(result: &SearchResult) -> Option<SearchHit> {
    let Some(value) = result.metadata.get(DESCRIPTOR_FIELD) else {
        warn!(
            record_id = %result.id,
            "semantic search skipped a record without a descriptor"
        );
        return None;
    };
    match serde_json::from_value::<CapabilityDescriptor>(value.clone()) {
        Ok(descriptor) => Some(SearchHit {
            descriptor,
            score: result.score,
            matched_terms: Vec::new(),
        }),
        Err(error) => {
            warn!(
                record_id = %result.id,
                %error,
                "semantic search skipped a record with a malformed descriptor"
            );
            None
        }
    }
}

/// Orders hits by descending score, then ascending qualified id.
fn sort_hits(hits: &mut [SearchHit]) {
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.descriptor
                    .qualified_id()
                    .cmp(&right.descriptor.qualified_id())
            })
    });
}
