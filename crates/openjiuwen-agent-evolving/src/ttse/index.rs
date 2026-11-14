//! Embedding-backed retrieval index over the tip bank.
//!
//! Ports the portable part of Python `ttse/index.py`: a content-addressed vector
//! store keyed by [`TipRecord::tip_id`](crate::ttse::TipRecord), cold-filled in
//! batches from an [`Embedding`] provider and queried with cosine similarity.
//! Where Python binds a Chroma collection to the record store, this index keeps an
//! in-process [`BTreeMap`] and persists it as one JSON document, so the ranking is
//! deterministic and needs no external service.
//!
//! Persistence is a plain JSON document:
//!
//! ```json
//! {"version": 1, "model_name": "hash-embedding", "dimension": 64,
//!  "vectors": {"<tip_id>": [0.0, 1.0]}}
//! ```
//!
//! A stored index is only reusable with the embedder that built it: loading one
//! through [`EmbeddingIndex::open`] validates the model name and, when the
//! embedder reports one up front, the dimension. A mismatch is
//! [`Error::Validation`] rather than a silent truncation, because mixing vector
//! spaces would make every cosine score meaningless.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result, ResultExt};
use openjiuwen_core::retrieval::embedding::Embedding;
use serde::{Deserialize, Serialize};

use crate::ttse::bank::TipBank;

/// The persisted schema version written by [`EmbeddingIndex::save`].
const INDEX_VERSION: u32 = 1;
/// The suffix appended to the index path for the in-flight write.
const TEMP_SUFFIX: &str = ".tmp";
/// How many tips one cold-fill embedding request carries.
const EMBED_BATCH_SIZE: usize = 32;

/// One scored hit returned by [`EmbeddingIndex::query`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexMatch {
    /// The content-addressed tip id.
    pub tip_id: String,
    /// The cosine similarity to the query vector.
    pub score: f64,
}

/// A vector index over the rules of a [`TipBank`].
///
/// The index owns the embedder it was built with, the vectors keyed by tip id, the
/// stored schema metadata and the path a [`EmbeddingIndex::save_default`] writes
/// to. The text every vector was computed from is tracked in memory (it is not
/// persisted), so [`EmbeddingIndex::sync`] can refresh a vector whose rule changed
/// while the process has been running. Since tip ids are content-addressed, a
/// reloaded vector normally still matches its tip.
#[derive(Clone)]
pub struct EmbeddingIndex {
    /// The provider used for cold fills and queries.
    embedding: Arc<dyn Embedding>,
    /// Vectors keyed by tip id, ordered for deterministic iteration.
    vectors: BTreeMap<String, Vec<f32>>,
    /// The model name the vectors belong to, as reported by the embedder.
    model_name: String,
    /// The vector length; `0` until the first embedding settles it.
    dimension: usize,
    /// The rule text each in-memory vector was computed from.
    texts: BTreeMap<String, String>,
    /// Where [`EmbeddingIndex::save_default`] writes, when bound to a file.
    path: Option<PathBuf>,
}

impl fmt::Debug for EmbeddingIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbeddingIndex")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .field("vectors", &self.vectors.len())
            .field("path", &self.path)
            .finish()
    }
}

/// The persisted JSON document, as read back.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct IndexFile {
    /// The schema version; missing means the current one.
    version: u32,
    /// The model name that produced the vectors.
    model_name: String,
    /// The vector length shared by every stored vector.
    dimension: usize,
    /// The vectors keyed by tip id.
    vectors: BTreeMap<String, Vec<f32>>,
}

/// The persisted JSON document, borrowed for writing.
#[derive(Serialize)]
struct IndexFileRef<'a> {
    /// The schema version written.
    version: u32,
    /// The model name that produced the vectors.
    model_name: &'a str,
    /// The vector length shared by every stored vector.
    dimension: usize,
    /// The vectors keyed by tip id.
    vectors: &'a BTreeMap<String, Vec<f32>>,
}

impl EmbeddingIndex {
    /// Builds an empty index over `embedding`.
    ///
    /// The dimension comes from [`Embedding::dimension_hint`] when the model
    /// reports one; otherwise it is learned from the first embedding and every
    /// later vector must match it.
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self {
            model_name: embedding.model_name().to_owned(),
            dimension: embedding.dimension_hint().unwrap_or(0),
            embedding,
            vectors: BTreeMap::new(),
            texts: BTreeMap::new(),
            path: None,
        }
    }

    /// Binds the index to the file [`EmbeddingIndex::save_default`] writes.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Opens the index persisted at `path`, creating an empty one when the file is
    /// absent.
    ///
    /// The stored `model_name` must equal [`Embedding::model_name`], and a
    /// dimension the embedder reports up front must equal the stored one; either
    /// mismatch is [`Error::Validation`], never a silent truncation. With an
    /// embedder that only reports its dimension after the first call (a `None`
    /// hint) the stored dimension is adopted and every loaded vector is checked
    /// against it.
    pub fn open(path: impl Into<PathBuf>, embedding: Arc<dyn Embedding>) -> Result<Self> {
        let path = path.into();
        let mut index = Self::new(embedding);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let file: IndexFile = serde_json::from_slice(&bytes).map_err(|error| {
                    Error::validation(format!(
                        "invalid ttse embedding index {}: {error}",
                        path.display()
                    ))
                })?;
                index.load_file(&file, &path)?;
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Error::from(error)
                    .with_context(format!("reading ttse embedding index {}", path.display())));
            }
        }
        index.path = Some(path);
        Ok(index)
    }

    /// Applies a loaded document, validating it against the supplied embedder.
    fn load_file(&mut self, file: &IndexFile, path: &Path) -> Result<()> {
        if file.version != INDEX_VERSION {
            return Err(Error::validation(format!(
                "ttse embedding index {} uses unsupported version {}, expected {INDEX_VERSION}",
                path.display(),
                file.version
            )));
        }
        if file.model_name != self.model_name {
            return Err(Error::validation(format!(
                "ttse embedding index {} was built with model `{}`, but the supplied embedder is `{}`",
                path.display(),
                file.model_name,
                self.model_name
            )));
        }
        if let Some(hint) = self.embedding.dimension_hint() {
            if hint != file.dimension {
                return Err(Error::validation(format!(
                    "ttse embedding index {} stores {}-dimensional vectors, but model `{}` produces {}",
                    path.display(),
                    file.dimension,
                    self.model_name,
                    hint
                )));
            }
        }
        for (tip_id, vector) in &file.vectors {
            if vector.len() != file.dimension {
                return Err(Error::validation(format!(
                    "ttse embedding index {} holds a {}-dimensional vector for `{tip_id}`, expected {}",
                    path.display(),
                    vector.len(),
                    file.dimension
                )));
            }
        }
        self.dimension = file.dimension;
        self.vectors.clone_from(&file.vectors);
        self.texts.clear();
        Ok(())
    }

    /// The model name the vectors belong to.
    #[must_use]
    pub fn embedding_model(&self) -> &str {
        &self.model_name
    }

    /// The vector length; `0` until the first embedding settles it.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// How many tips are indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether no tip is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Whether `tip_id` has a stored vector.
    #[must_use]
    pub fn contains(&self, tip_id: &str) -> bool {
        self.vectors.contains_key(tip_id)
    }

    /// The vector stored for `tip_id`, when there is one.
    #[must_use]
    pub fn vector(&self, tip_id: &str) -> Option<&[f32]> {
        self.vectors.get(tip_id).map(Vec::as_slice)
    }

    /// The file this index was opened from or bound to.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Embeds the tips that are missing or whose rule changed, and drops vectors
    /// of tips the bank no longer holds.
    ///
    /// Cold fills go through [`Embedding::embed_documents`] in batches of at most
    /// 32 rules, in ascending `tip_id` order, so identical banks produce identical
    /// requests. Returns how many vectors were (re)computed.
    pub async fn sync(&mut self, bank: &TipBank) -> Result<usize> {
        let live: BTreeSet<&str> = bank.tips.iter().map(|tip| tip.tip_id.as_str()).collect();
        self.vectors
            .retain(|tip_id, _| live.contains(tip_id.as_str()));
        self.texts
            .retain(|tip_id, _| live.contains(tip_id.as_str()));

        let mut stale: Vec<(String, String)> = Vec::new();
        for tip in &bank.tips {
            if self.texts.get(&tip.tip_id).map(String::as_str) != Some(tip.rule.as_str()) {
                stale.push((tip.tip_id.clone(), tip.rule.clone()));
            }
        }
        stale.sort_by(|left, right| left.0.cmp(&right.0));
        let embedded = stale.len();
        for batch in stale.chunks(EMBED_BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|(_, rule)| rule.clone()).collect();
            let vectors = self
                .embedding
                .embed_documents(&texts, Some(EMBED_BATCH_SIZE))
                .await?;
            if vectors.len() != texts.len() {
                return Err(Error::validation(format!(
                    "ttse embedder `{}` returned {} vectors for {} documents",
                    self.model_name,
                    vectors.len(),
                    texts.len()
                )));
            }
            for ((tip_id, rule), vector) in batch.iter().zip(vectors) {
                ensure_dimension(&mut self.dimension, &self.model_name, vector.len())?;
                self.vectors.insert(tip_id.clone(), vector);
                self.texts.insert(tip_id.clone(), rule.clone());
            }
        }
        Ok(embedded)
    }

    /// Embeds one tip and stores its vector, replacing an existing one.
    pub async fn upsert(
        &mut self,
        tip_id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<()> {
        let tip_id = tip_id.into();
        let text = text.into();
        let texts = [text.clone()];
        let vectors = self.embedding.embed_documents(&texts, None).await?;
        let vector = vectors.into_iter().next().ok_or_else(|| {
            Error::validation(format!(
                "ttse embedder `{}` returned no vector for tip `{tip_id}`",
                self.model_name
            ))
        })?;
        ensure_dimension(&mut self.dimension, &self.model_name, vector.len())?;
        self.vectors.insert(tip_id.clone(), vector);
        self.texts.insert(tip_id, text);
        Ok(())
    }

    /// Drops the vector of `tip_id`; returns whether one was stored.
    pub fn remove(&mut self, tip_id: &str) -> bool {
        self.texts.remove(tip_id);
        self.vectors.remove(tip_id).is_some()
    }

    /// Drops every vector, keeping the embedder, the metadata and the path.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.texts.clear();
    }

    /// Ranks the stored vectors against `text` by cosine similarity.
    ///
    /// The result is descending by score, ties broken by ascending `tip_id`, and
    /// truncated to `limit`. A blank query, an empty index or a zero limit yields
    /// nothing without calling the embedder. A query vector whose length differs
    /// from the stored one is [`Error::Validation`].
    pub async fn query(&self, text: &str, limit: usize) -> Result<Vec<IndexMatch>> {
        if limit == 0 || self.vectors.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let query = self.embedding.embed_query(text).await?;
        if self.dimension != 0 && query.len() != self.dimension {
            return Err(Error::validation(format!(
                "ttse embedder `{}` produced a {}-dimensional query vector, expected {}",
                self.model_name,
                query.len(),
                self.dimension
            )));
        }
        let mut matches: Vec<IndexMatch> = self
            .vectors
            .iter()
            .map(|(tip_id, vector)| IndexMatch {
                tip_id: tip_id.clone(),
                score: cosine(&query, vector),
            })
            .collect();
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.tip_id.cmp(&right.tip_id))
        });
        matches.truncate(limit);
        Ok(matches)
    }

    /// Writes the index to `path` atomically: `<path>.tmp` first, then a rename.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = IndexFileRef {
            version: INDEX_VERSION,
            model_name: &self.model_name,
            dimension: self.dimension,
            vectors: &self.vectors,
        };
        let encoded = serde_json::to_vec_pretty(&file)?;
        write_atomic(path.as_ref(), &encoded).await
    }

    /// Writes the index to the path it was opened from or bound to.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the index has no path, plus the errors of
    /// [`EmbeddingIndex::save`].
    pub async fn save_default(&self) -> Result<()> {
        let path = self.path.as_deref().ok_or_else(|| {
            Error::validation(
                "ttse embedding index has no path; bind one with `with_path` or `open`",
            )
        })?;
        self.save(path).await
    }
}

/// The cosine similarity of two vectors, prefix-limited to the shorter one.
///
/// Returns `0.0` when either vector is empty or has a zero norm, so a degenerate
/// input ranks last instead of poisoning the order with `NaN`.
#[must_use]
pub fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (a, b) in left.iter().zip(right) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

/// Settles the index dimension from the first vector and rejects later mismatches.
fn ensure_dimension(dimension: &mut usize, model_name: &str, produced: usize) -> Result<()> {
    if *dimension == 0 {
        *dimension = produced;
        return Ok(());
    }
    if *dimension == produced {
        return Ok(());
    }
    Err(Error::validation(format!(
        "ttse embedding dimension mismatch: the index holds {}-dimensional vectors but model `{model_name}` produced {produced}",
        *dimension
    )))
}

/// Writes `bytes` to `path` through a sibling temporary file and a rename.
pub(crate) async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        tokio::fs::create_dir_all(parent)
            .await
            .context(format!("creating directory {}", parent.display()))?;
    }
    let mut temp = path.as_os_str().to_owned();
    temp.push(TEMP_SUFFIX);
    let temp = PathBuf::from(temp);
    tokio::fs::write(&temp, bytes)
        .await
        .context(format!("writing temporary file {}", temp.display()))?;
    if let Err(error) = tokio::fs::rename(&temp, path).await {
        // Best effort: a leftover `.tmp` must not shadow the real file.
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(Error::from(error).with_context(format!(
            "renaming {} to {}",
            temp.display(),
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openjiuwen_core::retrieval::embedding::HashEmbedding;

    use super::*;
    use crate::ttse::models::{TipCategory, TipRecord};

    fn embedder(dimension: usize) -> Arc<dyn Embedding> {
        Arc::new(HashEmbedding::new(dimension))
    }

    fn bank_with(rules: &[&str]) -> TipBank {
        let mut bank = TipBank::with_capacity(16);
        for rule in rules {
            bank.add(TipRecord::new(*rule, TipCategory::Tool).expect("valid rule"));
        }
        bank
    }

    #[test]
    fn cosine_is_prefix_limited_and_zero_safe() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert!(cosine(&[], &[1.0]).abs() < 1e-9);
        assert!(cosine(&[0.0, 0.0], &[1.0, 1.0]).abs() < 1e-9);
        assert!((cosine(&[1.0, 1.0, 9.0], &[1.0, 1.0]) - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn sync_embeds_only_the_missing_tips() {
        let mut index = EmbeddingIndex::new(embedder(64));
        let mut bank = bank_with(&["use rg instead of grep", "run the tests offline"]);

        assert_eq!(index.sync(&bank).await.expect("sync"), 2);
        assert_eq!(index.len(), 2);
        assert_eq!(index.sync(&bank).await.expect("second sync"), 0);
        assert_eq!(index.dimension(), 64);

        bank.add(TipRecord::new("quote every path", TipCategory::Tool).expect("valid rule"));
        assert_eq!(index.sync(&bank).await.expect("third sync"), 1);
        assert_eq!(index.len(), 3);

        let removed = bank.tips()[0].tip_id.clone();
        bank.remove(&removed);
        assert_eq!(index.sync(&bank).await.expect("fourth sync"), 0);
        assert!(!index.contains(&removed));
        assert_eq!(index.len(), 2);
    }

    #[tokio::test]
    async fn query_ranks_by_cosine_with_a_deterministic_tie_break() {
        let mut index = EmbeddingIndex::new(embedder(64));
        let bank = bank_with(&["use rg instead of grep", "run the tests offline"]);
        index.sync(&bank).await.expect("sync");

        let matches = index
            .query("use rg instead of grep", 8)
            .await
            .expect("query");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].tip_id, bank.tips()[0].tip_id);
        assert!(matches[0].score > matches[1].score);
        assert!(matches[0].score <= 1.0 + 1e-9);

        assert!(index.query("  ", 8).await.expect("blank query").is_empty());
        assert!(index
            .query("anything", 0)
            .await
            .expect("zero limit")
            .is_empty());
    }

    #[tokio::test]
    async fn upsert_remove_and_clear_manage_the_map() {
        let mut index = EmbeddingIndex::new(embedder(32));
        index.upsert("t-1", "first rule").await.expect("upsert");
        assert!(index.contains("t-1"));
        assert_eq!(index.len(), 1);
        index
            .upsert("t-1", "first rule again")
            .await
            .expect("upsert");
        assert_eq!(index.len(), 1);

        assert!(index.remove("t-1"));
        assert!(!index.remove("t-1"));
        assert!(index.is_empty());

        index.upsert("t-2", "second rule").await.expect("upsert");
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.path(), None);
    }

    #[tokio::test]
    async fn save_and_open_round_trip() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("nested").join("index.json");
        let mut index = EmbeddingIndex::new(embedder(64)).with_path(&path);
        let bank = bank_with(&["use rg instead of grep", "run the tests offline"]);
        index.sync(&bank).await.expect("sync");
        index.save_default().await.expect("save");

        let restored = EmbeddingIndex::open(&path, embedder(64)).expect("open");
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.dimension(), 64);
        assert_eq!(restored.embedding_model(), "hash-embedding");
        assert_eq!(restored.path(), Some(path.as_path()));
        let expected = index.query("use rg", 8).await.expect("query");
        let actual = restored.query("use rg", 8).await.expect("query");
        assert_eq!(actual, expected);

        let file: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        assert_eq!(file["version"], INDEX_VERSION);
        assert_eq!(file["dimension"], 64);
        assert_eq!(file["model_name"], "hash-embedding");
        assert!(file["vectors"].is_object());

        let missing = directory.path().join("absent.json");
        let empty = EmbeddingIndex::open(&missing, embedder(64)).expect("open missing");
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn open_rejects_a_model_or_dimension_mismatch() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("index.json");
        let mut index = EmbeddingIndex::new(embedder(64));
        index
            .upsert("t-1", "use rg instead of grep")
            .await
            .expect("upsert");
        index.save(&path).await.expect("save");

        let wrong_dimension = EmbeddingIndex::open(&path, embedder(32)).expect_err("dimension");
        assert!(matches!(wrong_dimension, Error::Validation(_)));

        let renamed: Arc<dyn Embedding> =
            Arc::new(HashEmbedding::new(64).with_model_name("other-model"));
        let wrong_model = EmbeddingIndex::open(&path, renamed).expect_err("model");
        assert!(matches!(wrong_model, Error::Validation(_)));

        std::fs::write(&path, b"not json").expect("write");
        let malformed = EmbeddingIndex::open(&path, embedder(64)).expect_err("malformed");
        assert!(matches!(malformed, Error::Validation(_)));

        std::fs::write(&path, br#"{"version": 2, "model_name": "hash-embedding"}"#).expect("write");
        let future = EmbeddingIndex::open(&path, embedder(64)).expect_err("version");
        assert!(matches!(future, Error::Validation(_)));
    }

    #[tokio::test]
    async fn save_default_requires_a_path() {
        let index = EmbeddingIndex::new(embedder(16));
        let error = index.save_default().await.expect_err("no path");
        assert!(matches!(error, Error::Validation(_)));
    }
}
