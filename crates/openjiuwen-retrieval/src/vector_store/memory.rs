//! Offline in-memory vector store with cosine search and optional JSON persistence.
//!
//! Python has no in-memory store; this implementation exists so the retrieval
//! pipeline and its tests can run without an external database. Scores follow the
//! Chroma/Milvus `cosine` convention: `(cos + 1) / 2` in `[0, 1]`.
use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::{
    cosine_similarity, normalize_cosine, VectorQuery, VectorRecord, VectorStore, CHUNK_ID_FIELD,
    DOC_ID_FIELD,
};
use crate::common::{Metadata, MetadataFilter, SearchResult};

/// Snapshot format version; bump when the persisted shape changes.
const PERSISTENCE_VERSION: u32 = 1;
/// Collection name used by [`InMemoryVectorStore::new`].
const DEFAULT_COLLECTION_NAME: &str = "default";

/// Persisted store payload: `{"version":1,"name":"...","records":[...]}`.
#[derive(Debug, Serialize, Deserialize)]
struct Snapshot {
    /// Snapshot format version.
    version: u32,
    /// Collection name at write time.
    name: String,
    /// Records in insertion order.
    records: Vec<VectorRecord>,
}

/// An in-memory [`VectorStore`] with optional atomic JSON persistence.
///
/// Records keep their insertion order, which defines both `list` order and the
/// tie-break for equal search scores. `upsert` replaces the payload of an
/// existing id without moving it to the end.
#[derive(Debug)]
pub struct InMemoryVectorStore {
    name: String,
    records: RwLock<IndexMap<String, VectorRecord>>,
    path: Option<PathBuf>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    /// Creates an empty store named `DEFAULT_COLLECTION_NAME` with no persistence.
    pub fn new() -> Self {
        Self {
            name: DEFAULT_COLLECTION_NAME.to_owned(),
            records: RwLock::new(IndexMap::new()),
            path: None,
        }
    }

    /// Creates an empty store with `name`; a blank name falls back to the default.
    pub fn with_collection(name: impl Into<String>) -> Self {
        Self {
            name: normalize_name(&name.into()),
            records: RwLock::new(IndexMap::new()),
            path: None,
        }
    }

    /// Opens a store backed by `path`, loading the JSON snapshot when it exists.
    ///
    /// A missing file yields an empty store; the collection name comes from the
    /// snapshot (or the default when there is no file). The store auto-persists
    /// after every mutation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] for I/O failures and [`Error::Retrieval`] for
    /// malformed or unsupported snapshots.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_inner(path.into(), None).await
    }

    /// Like [`Self::open`], but always names the collection `name`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::open`].
    pub async fn open_with_collection(
        path: impl Into<PathBuf>,
        name: impl Into<String>,
    ) -> Result<Self> {
        Self::open_inner(path.into(), Some(normalize_name(&name.into()))).await
    }

    async fn open_inner(path: PathBuf, name: Option<String>) -> Result<Self> {
        let mut records = IndexMap::new();
        let mut persisted_name = None;
        match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let snapshot: Snapshot = serde_json::from_slice(&bytes).map_err(|error| {
                    Error::retrieval(format!(
                        "parsing vector store snapshot {}: {error}",
                        path.display()
                    ))
                })?;
                if snapshot.version != PERSISTENCE_VERSION {
                    return Err(Error::retrieval(format!(
                        "unsupported vector store snapshot version {} in {} (expected {})",
                        snapshot.version,
                        path.display(),
                        PERSISTENCE_VERSION
                    )));
                }
                records.extend(snapshot.records.into_iter().map(|r| (r.id.clone(), r)));
                // An explicit collection name wins over the persisted one.
                persisted_name = Some(normalize_name(&snapshot.name));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Error::transport(format!(
                    "reading vector store snapshot {}: {error}",
                    path.display()
                )));
            }
        }
        Ok(Self {
            name: name
                .or(persisted_name)
                .unwrap_or_else(|| DEFAULT_COLLECTION_NAME.to_owned()),
            records: RwLock::new(records),
            path: Some(path),
        })
    }

    /// The persistence path, when this store was opened from a file.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Writes the current records to [`Self::path`].
    ///
    /// A store without a path is a no-op. The write goes to a temporary file in
    /// the same directory and is then renamed, so readers never see a partial
    /// snapshot. The lock is released before any `.await`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Retrieval`] when serialization fails and
    /// [`Error::Transport`] when the file cannot be written.
    pub async fn persist(&self) -> Result<()> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let snapshot = {
            let records = self.records.read();
            Snapshot {
                version: PERSISTENCE_VERSION,
                name: self.name.clone(),
                records: records.values().cloned().collect(),
            }
        };
        let json = serde_json::to_vec_pretty(&snapshot).map_err(|error| {
            Error::retrieval(format!(
                "serializing vector store snapshot {}: {error}",
                path.display()
            ))
        })?;
        let temp = temp_path(&path);
        if let Err(error) = tokio::fs::write(&temp, &json).await {
            return Err(Error::transport(format!(
                "writing vector store snapshot {}: {error}",
                temp.display()
            )));
        }
        if let Err(error) = tokio::fs::rename(&temp, &path).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(Error::transport(format!(
                "replacing vector store snapshot {}: {error}",
                path.display()
            )));
        }
        Ok(())
    }

    async fn persist_if_configured(&self) -> Result<()> {
        if self.path.is_some() {
            self.persist().await?;
        }
        Ok(())
    }
}

/// Trims a collection name, falling back to the default when blank.
fn normalize_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        DEFAULT_COLLECTION_NAME.to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().map_or_else(
        || "vector_store.json".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    path.with_file_name(format!(".{file_name}.tmp-{}", uuid::Uuid::new_v4()))
}

/// Metadata as seen by filters: record metadata plus `doc_id`/`chunk_id`.
fn effective_metadata(record: &VectorRecord) -> Cow<'_, Metadata> {
    let missing_doc_id = record.doc_id.is_some() && !record.metadata.contains_key(DOC_ID_FIELD);
    let missing_chunk_id =
        record.chunk_id.is_some() && !record.metadata.contains_key(CHUNK_ID_FIELD);
    if !missing_doc_id && !missing_chunk_id {
        return Cow::Borrowed(&record.metadata);
    }
    let mut metadata = record.metadata.clone();
    if let Some(doc_id) = &record.doc_id {
        metadata.insert(DOC_ID_FIELD.to_owned(), Value::String(doc_id.clone()));
    }
    if let Some(chunk_id) = &record.chunk_id {
        metadata.insert(CHUNK_ID_FIELD.to_owned(), Value::String(chunk_id.clone()));
    }
    Cow::Owned(metadata)
}

fn record_matches(record: &VectorRecord, filter: &MetadataFilter) -> bool {
    filter.matches(&effective_metadata(record))
}

/// Converts a record into a hit, injecting the ids into the result metadata.
fn search_result(record: &VectorRecord, score: f32) -> SearchResult {
    let metadata = effective_metadata(record).into_owned();
    SearchResult {
        id: record.id.clone(),
        text: record.text.clone(),
        score,
        metadata,
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    fn name(&self) -> &str {
        &self.name
    }

    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize> {
        let supplied = records.len();
        {
            let mut stored = self.records.write();
            for record in records {
                // IndexMap::insert keeps an existing key in its original position.
                stored.insert(record.id.clone(), record);
            }
        }
        self.persist_if_configured().await?;
        Ok(supplied)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorRecord>> {
        Ok(self.records.read().get(id).cloned())
    }

    async fn get_many(&self, ids: &[String]) -> Result<Vec<VectorRecord>> {
        // A single read lock serves the whole batch; unknown ids are skipped.
        let stored = self.records.read();
        Ok(ids
            .iter()
            .filter_map(|id| stored.get(id.as_str()).cloned())
            .collect())
    }

    async fn list(&self, limit: Option<usize>) -> Result<Vec<VectorRecord>> {
        let limit = limit.unwrap_or(usize::MAX);
        Ok(self.records.read().values().take(limit).cloned().collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<usize> {
        let removed = {
            let mut stored = self.records.write();
            ids.iter()
                .filter(|id| stored.shift_remove(id.as_str()).is_some())
                .count()
        };
        self.persist_if_configured().await?;
        Ok(removed)
    }

    async fn delete_by_filter(&self, filter: &MetadataFilter) -> Result<usize> {
        let removed = {
            let mut stored = self.records.write();
            let matches: Vec<String> = stored
                .values()
                .filter(|record| record_matches(record, filter))
                .map(|record| record.id.clone())
                .collect();
            matches
                .into_iter()
                .filter(|id| stored.shift_remove(id).is_some())
                .count()
        };
        self.persist_if_configured().await?;
        Ok(removed)
    }

    async fn search(&self, query: VectorQuery) -> Result<Vec<SearchResult>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }
        let mut hits: Vec<SearchResult> = {
            let stored = self.records.read();
            stored
                .values()
                .filter(|record| record.vector.len() == query.vector.len())
                .filter(|record| {
                    query
                        .filter
                        .as_ref()
                        .is_none_or(|filter| record_matches(record, filter))
                })
                .map(|record| {
                    let score = normalize_cosine(cosine_similarity(&record.vector, &query.vector));
                    (record, score)
                })
                .filter(|(_, score)| query.score_threshold.is_none_or(|min| *score >= min))
                .map(|(record, score)| search_result(record, score))
                .collect()
        };
        // Rust's sort is stable, so equal scores keep their insertion order.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        hits.truncate(query.top_k);
        Ok(hits)
    }

    async fn len(&self) -> Result<usize> {
        Ok(self.records.read().len())
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.records.read().is_empty())
    }

    async fn clear(&self) -> Result<()> {
        self.records.write().clear();
        self.persist_if_configured().await
    }
}

#[cfg(test)]
mod tests;
