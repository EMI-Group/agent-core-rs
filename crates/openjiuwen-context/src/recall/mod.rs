//! Compression recall: archive compressed history, then retrieve it with BM25.
//!
//! Maps to Python's `core.context_engine.processor.forked.compressor.recall`.
//! When a compressor replaces old dialogue with a summary it keeps the exact
//! replaced messages in a [`RecallArchive`]: conversation turns are grouped,
//! rendered into a section transcript, and split into overlapping byte-window
//! chunks. Later, [`recall_compressed_context`] ranks turns and chunks with
//! [`crate::recall::bm25::Bm25Index`] and returns a token-budgeted slice of the archived text,
//! which is what lets an agent recover details the summary dropped.
//!
//! ```
//! use openjiuwen_context::recall::{
//!     archive_from_messages, recall_compressed_context, InMemoryRecallArchiveStore, RecallOptions,
//!     RecallArchiveStore,
//! };
//! use openjiuwen_context::{CompressionRecallConfig, OffloadScope};
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
//! let messages = vec![
//!     BaseMessage::user("how does borrow checking work"),
//!     BaseMessage::assistant("the borrow checker enforces aliasing rules"),
//! ];
//! let archive = archive_from_messages(
//!     "mem_1",
//!     "session",
//!     "main",
//!     "2026-01-01T00:00:00Z",
//!     &messages,
//!     &CompressionRecallConfig::default(),
//! );
//! let store = InMemoryRecallArchiveStore::new();
//! let scope = OffloadScope::new("session", "main");
//! store.put(&scope, archive).await.unwrap();
//! let result = recall_compressed_context(
//!     &store,
//!     &scope,
//!     Some("mem_1"),
//!     "borrow checker",
//!     &RecallOptions::default(),
//! )
//! .await
//! .unwrap();
//! assert!(!result.chunks.is_empty());
//! # });
//! ```
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, PoisonError, RwLock};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::store::{KvStore, KvValue};
use serde::{Deserialize, Serialize};

use crate::config::CompressionRecallConfig;
use crate::offload::{sanitize_key, OffloadScope};

pub mod bm25;
pub mod chunk;

pub use bm25::{rank_bm25, tokenize, Bm25Index};
pub use chunk::{
    estimate_tokens, group_turns, is_real_user_message, render_turn, split_text, TurnRecord,
    INTERNAL_USER_PREFIXES,
};

/// Prefix that opens a compression-recall marker.
pub const COMPRESSION_RECALL_MARKER_PREFIX: &str = "[[COMPRESSION_RECALL:";

/// Renders the marker that points at archive `memory_id`.
#[must_use]
pub fn recall_marker(memory_id: &str) -> String {
    format!("{COMPRESSION_RECALL_MARKER_PREFIX} id={memory_id}]]")
}

/// Extracts the archive id from the first recall marker in `text`.
///
/// Returns `None` when no well-formed marker is present.
#[must_use]
pub fn parse_recall_marker(text: &str) -> Option<String> {
    let start = text.find(COMPRESSION_RECALL_MARKER_PREFIX)?;
    let remainder = &text[start + COMPRESSION_RECALL_MARKER_PREFIX.len()..];
    let end = remainder.find("]]")?;
    let (key, value) = remainder[..end].split_once('=')?;
    if key.trim() != "id" {
        return None;
    }
    let memory_id = value.trim();
    if memory_id.is_empty() {
        None
    } else {
        Some(memory_id.to_owned())
    }
}

/// One chunk of archived transcript text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallChunk {
    /// Stable chunk identifier (`turn_0001_chunk_000`).
    pub chunk_id: String,
    /// Turn the chunk belongs to.
    pub turn_id: String,
    /// Zero-based position of the chunk inside its turn.
    pub chunk_index: usize,
    /// The chunk's rendered transcript text.
    pub text: String,
    /// Estimated token count, `chars / 3` with a floor of one.
    pub tokens: usize,
    /// Whether [`RecallChunk::tokens`] is an estimate rather than a measured count.
    ///
    /// Every chunk this crate archives is sized by [`chunk::estimate_tokens`], so
    /// this is `true` for all of them; the field exists so a caller that skips
    /// chunks on a token budget can see that the number it skipped on is an
    /// estimate. Deserialized chunks without the field default to `true`,
    /// because no tokenizer produced their `tokens` either.
    pub estimated: bool,
}

impl Default for RecallChunk {
    /// The defaults of an archived chunk: its size is an estimate.
    fn default() -> Self {
        Self {
            chunk_id: String::new(),
            turn_id: String::new(),
            chunk_index: 0,
            text: String::new(),
            tokens: 0,
            estimated: true,
        }
    }
}

impl RecallChunk {
    /// Builds a chunk whose `tokens` is the character-based estimate.
    ///
    /// The estimate is marked with [`RecallChunk::estimated`], so a budget that
    /// skips this chunk knows the number was never measured by a tokenizer.
    #[must_use]
    pub fn estimated(
        chunk_id: impl Into<String>,
        turn_id: impl Into<String>,
        chunk_index: usize,
        text: impl Into<String>,
    ) -> Self {
        let text = text.into();
        Self {
            chunk_id: chunk_id.into(),
            turn_id: turn_id.into(),
            chunk_index,
            tokens: chunk::estimate_tokens(&text),
            text,
            estimated: true,
        }
    }
}

/// One archived conversation turn and the chunks it produced.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallTurn {
    /// Stable turn identifier (`turn_0001`).
    pub turn_id: String,
    /// The turn's user query.
    pub query: String,
    /// The turn's answer text.
    pub answer: String,
    /// Ids of the chunks produced from this turn, in order.
    pub chunk_ids: Vec<String>,
}

/// Everything archived by one compression.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallArchive {
    /// Archive identifier used in markers and lookups.
    pub memory_id: String,
    /// Session that produced the archive.
    pub session_id: String,
    /// Context that produced the archive.
    pub context_id: String,
    /// Creation timestamp, supplied by the caller for determinism.
    pub created_at: String,
    /// Archived transcript chunks.
    pub chunks: Vec<RecallChunk>,
    /// Archived conversation turns.
    pub turns: Vec<RecallTurn>,
}

impl RecallArchive {
    /// Creates an empty archive with a fresh random `memory_id`.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        context_id: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Self {
        Self::with_id(
            uuid::Uuid::new_v4().simple().to_string(),
            session_id,
            context_id,
            created_at,
        )
    }

    /// Creates an empty archive with an explicit `memory_id`.
    #[must_use]
    pub fn with_id(
        memory_id: impl Into<String>,
        session_id: impl Into<String>,
        context_id: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Self {
        Self {
            memory_id: memory_id.into(),
            session_id: session_id.into(),
            context_id: context_id.into(),
            created_at: created_at.into(),
            chunks: Vec::new(),
            turns: Vec::new(),
        }
    }

    /// Looks up a chunk by id.
    #[must_use]
    pub fn chunk(&self, chunk_id: &str) -> Option<&RecallChunk> {
        self.chunks.iter().find(|chunk| chunk.chunk_id == chunk_id)
    }

    /// Looks up a turn by id.
    #[must_use]
    pub fn turn(&self, turn_id: &str) -> Option<&RecallTurn> {
        self.turns.iter().find(|turn| turn.turn_id == turn_id)
    }

    /// Whether the archive holds no turns or chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty() && self.chunks.is_empty()
    }
}

/// Durable storage for recall archives, scoped to a session and context.
#[async_trait]
pub trait RecallArchiveStore: Send + Sync {
    /// Stores or replaces `archive` under its `memory_id`.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the archive cannot be persisted.
    async fn put(&self, scope: &OffloadScope, archive: RecallArchive) -> Result<()>;

    /// Reads one archive, or `None` when the scope holds no such archive.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the archive exists but cannot be read.
    async fn get(&self, scope: &OffloadScope, memory_id: &str) -> Result<Option<RecallArchive>>;

    /// Lists the archive ids stored for `scope`.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the listing cannot be read.
    async fn list(&self, scope: &OffloadScope) -> Result<Vec<String>>;

    /// Removes one archive; removing a missing archive succeeds.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the removal cannot be persisted.
    async fn remove(&self, scope: &OffloadScope, memory_id: &str) -> Result<()>;
}

/// Process-local archive storage, used by tests and short-lived engines.
#[derive(Debug, Default)]
pub struct InMemoryRecallArchiveStore {
    archives: RwLock<HashMap<(String, String), BTreeMap<String, RecallArchive>>>,
}

impl InMemoryRecallArchiveStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of archives across every scope.
    #[must_use]
    pub fn len(&self) -> usize {
        let archives = self.archives.read().unwrap_or_else(PoisonError::into_inner);
        archives.values().map(BTreeMap::len).sum()
    }

    /// Whether the store holds no archives in any scope.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Identity of `scope` inside the store.
    fn scope_key(scope: &OffloadScope) -> (String, String) {
        (scope.session_id.clone(), scope.context_id.clone())
    }
}

#[async_trait]
impl RecallArchiveStore for InMemoryRecallArchiveStore {
    async fn put(&self, scope: &OffloadScope, archive: RecallArchive) -> Result<()> {
        let mut archives = self
            .archives
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        archives
            .entry(Self::scope_key(scope))
            .or_default()
            .insert(archive.memory_id.clone(), archive);
        Ok(())
    }

    async fn get(&self, scope: &OffloadScope, memory_id: &str) -> Result<Option<RecallArchive>> {
        let archives = self.archives.read().unwrap_or_else(PoisonError::into_inner);
        Ok(archives
            .get(&Self::scope_key(scope))
            .and_then(|scoped| scoped.get(memory_id))
            .cloned())
    }

    async fn list(&self, scope: &OffloadScope) -> Result<Vec<String>> {
        let archives = self.archives.read().unwrap_or_else(PoisonError::into_inner);
        Ok(archives
            .get(&Self::scope_key(scope))
            .map(|scoped| scoped.keys().cloned().collect())
            .unwrap_or_default())
    }

    async fn remove(&self, scope: &OffloadScope, memory_id: &str) -> Result<()> {
        let mut archives = self
            .archives
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(scoped) = archives.get_mut(&Self::scope_key(scope)) {
            scoped.remove(memory_id);
            if scoped.is_empty() {
                archives.remove(&Self::scope_key(scope));
            }
        }
        Ok(())
    }
}

/// Archive storage backed by foundation's [`KvStore`].
///
/// Keys are `prefix + sanitized session + "_" + sanitized context + "_" +
/// sanitized memory id`, matching [`crate::offload::KvOffloadStore`]'s isolation
/// scheme. Archives are stored as JSON bytes.
pub struct KvRecallArchiveStore {
    store: Arc<dyn KvStore>,
    prefix: String,
}

impl std::fmt::Debug for KvRecallArchiveStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KvRecallArchiveStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl KvRecallArchiveStore {
    /// The default key prefix.
    pub const DEFAULT_PREFIX: &'static str = "context:recall:";

    /// Wraps `store` with the default key prefix.
    #[must_use]
    pub fn new(store: Arc<dyn KvStore>) -> Self {
        Self::with_prefix(store, Self::DEFAULT_PREFIX)
    }

    /// Wraps `store` with an explicit key prefix.
    #[must_use]
    pub fn with_prefix(store: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
        }
    }

    /// The key prefix shared by every archive of `scope`.
    #[must_use]
    fn scope_prefix(&self, scope: &OffloadScope) -> String {
        format!(
            "{}{}_{}_",
            self.prefix,
            sanitize_key(&scope.session_id),
            sanitize_key(&scope.context_id)
        )
    }

    /// The storage key for one `(scope, memory_id)` pair.
    #[must_use]
    pub fn key(&self, scope: &OffloadScope, memory_id: &str) -> String {
        format!("{}{}", self.scope_prefix(scope), sanitize_key(memory_id))
    }
}

#[async_trait]
impl RecallArchiveStore for KvRecallArchiveStore {
    async fn put(&self, scope: &OffloadScope, archive: RecallArchive) -> Result<()> {
        let key = self.key(scope, &archive.memory_id);
        let bytes = serde_json::to_vec(&archive).map_err(|error| {
            Error::Validation(format!("cannot serialize recall archive: {error}"))
        })?;
        self.store.set(&key, KvValue::bytes(bytes)).await
    }

    async fn get(&self, scope: &OffloadScope, memory_id: &str) -> Result<Option<RecallArchive>> {
        let Some(value) = self.store.get(&self.key(scope, memory_id)).await? else {
            return Ok(None);
        };
        let archive = serde_json::from_slice(value.as_bytes())
            .map_err(|error| Error::Validation(format!("cannot decode recall archive: {error}")))?;
        Ok(Some(archive))
    }

    async fn list(&self, scope: &OffloadScope) -> Result<Vec<String>> {
        let prefix = self.scope_prefix(scope);
        let entries = self.store.get_by_prefix(&prefix).await?;
        Ok(entries
            .keys()
            .filter_map(|key| key.strip_prefix(&prefix).map(str::to_owned))
            .collect())
    }

    async fn remove(&self, scope: &OffloadScope, memory_id: &str) -> Result<()> {
        self.store.delete(&self.key(scope, memory_id)).await
    }
}

/// Budget and ranking knobs for one recall call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallOptions {
    /// Maximum tokens of chunk text the result may return.
    pub max_content_tokens: usize,
    /// Maximum number of conversation turns matched per archive.
    pub turn_top_k: usize,
    /// Maximum number of chunks matched per turn.
    pub chunk_top_k: usize,
    /// Turns scoring below `turn_score_ratio * top_turn_score` are dropped.
    pub turn_score_ratio: f32,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            max_content_tokens: 8_000,
            turn_top_k: 2,
            chunk_top_k: 2,
            turn_score_ratio: 0.3,
        }
    }
}

/// One matched archived chunk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallChunkMatch {
    /// Chunk identifier.
    pub chunk_id: String,
    /// Turn the chunk belongs to.
    pub turn_id: String,
    /// Archive the chunk came from.
    pub memory_id: String,
    /// Score used for ranking (normalized across archives, otherwise raw BM25).
    pub score: f32,
    /// Unnormalized BM25 score.
    pub raw_score: f32,
    /// Estimated tokens of [`RecallChunkMatch::text`].
    pub tokens: usize,
    /// The chunk's transcript text.
    pub text: String,
}

/// One matched archived turn.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallTurnMatch {
    /// Turn identifier.
    pub turn_id: String,
    /// Archive the turn came from.
    pub memory_id: String,
    /// The turn's user query.
    pub query: String,
    /// Score used for ranking (normalized across archives, otherwise raw BM25).
    pub score: f32,
    /// Unnormalized BM25 score.
    pub raw_score: f32,
}

/// Outcome of one recall call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RecallResult {
    /// Archive searched when the caller named one, else `None`.
    pub memory_id: Option<String>,
    /// Matched turns, best first.
    pub matched_turns: Vec<RecallTurnMatch>,
    /// Budgeted matched chunks, best first.
    pub chunks: Vec<RecallChunkMatch>,
    /// Whether at least one otherwise-matching chunk was skipped by the budget.
    pub truncated: bool,
    /// Total tokens of the returned chunks.
    pub returned_tokens: usize,
    /// Archive ids that were searched, sorted.
    pub archives: Vec<String>,
}

/// A turn candidate with raw BM25 scoring.
#[derive(Debug, Clone)]
struct ScoredTurn {
    turn_id: String,
    memory_id: String,
    query: String,
    score: f64,
    raw_score: f64,
}

/// A chunk candidate with raw BM25 and normalized scoring.
#[derive(Debug, Clone)]
struct ScoredChunk {
    chunk_id: String,
    turn_id: String,
    memory_id: String,
    score: f64,
    raw_score: f64,
    tokens: usize,
    text: String,
}

/// Groups, chunks, and archives `messages`.
///
/// Returns an empty archive for empty input. When the recall configuration is
/// invalid (`chunk_overlap_tokens >= chunk_size_tokens`) the problem is logged
/// and an empty archive is returned; use [`try_archive_from_messages`] to get
/// [`Error::Config`] instead.
#[must_use]
pub fn archive_from_messages(
    memory_id: &str,
    session_id: &str,
    context_id: &str,
    created_at: &str,
    messages: &[BaseMessage],
    config: &CompressionRecallConfig,
) -> RecallArchive {
    match try_archive_from_messages(
        memory_id, session_id, context_id, created_at, messages, config,
    ) {
        Ok(archive) => archive,
        Err(error) => {
            tracing::warn!(
                memory_id,
                error = %error,
                "cannot archive messages for compression recall"
            );
            RecallArchive::with_id(memory_id, session_id, context_id, created_at)
        }
    }
}

/// Groups, chunks, and archives `messages`, reporting invalid configuration.
///
/// Returns an empty archive for empty input.
///
/// # Errors
///
/// Returns [`Error::Config`] when `chunk_overlap_tokens >= chunk_size_tokens`.
pub fn try_archive_from_messages(
    memory_id: &str,
    session_id: &str,
    context_id: &str,
    created_at: &str,
    messages: &[BaseMessage],
    config: &CompressionRecallConfig,
) -> Result<RecallArchive> {
    let mut archive = RecallArchive::with_id(memory_id, session_id, context_id, created_at);
    if messages.is_empty() {
        return Ok(archive);
    }
    for turn in chunk::group_turns(messages) {
        let body = chunk::render_turn(messages, &turn.message_indices);
        let bodies =
            chunk::split_text(&body, config.chunk_size_tokens, config.chunk_overlap_tokens)?;
        let mut chunk_ids = Vec::with_capacity(bodies.len());
        for (chunk_index, text) in bodies.into_iter().enumerate() {
            let chunk_id = format!("{}_chunk_{:03}", turn.turn_id, chunk_index);
            chunk_ids.push(chunk_id.clone());
            archive.chunks.push(RecallChunk::estimated(
                chunk_id,
                turn.turn_id.clone(),
                chunk_index,
                text,
            ));
        }
        archive.turns.push(RecallTurn {
            turn_id: turn.turn_id,
            query: turn.query,
            answer: turn.answer,
            chunk_ids,
        });
    }
    Ok(archive)
}

/// Two-stage BM25 search inside one archive.
///
/// Turns are ranked over `query + "\n" + answer`; the top
/// [`RecallOptions::turn_top_k`] turns above the ratio cutoff are kept. Each
/// kept turn's chunks are then ranked over their own text and the top
/// [`RecallOptions::chunk_top_k`] positive-scoring chunks are returned. Scores
/// are raw BM25 values.
fn search_archive(
    archive: &RecallArchive,
    query: &str,
    options: &RecallOptions,
) -> (Vec<ScoredTurn>, Vec<ScoredChunk>) {
    if archive.turns.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let turn_documents: Vec<String> = archive
        .turns
        .iter()
        .map(|turn| format!("{}\n{}", turn.query, turn.answer))
        .collect();
    let turn_texts: Vec<&str> = turn_documents.iter().map(String::as_str).collect();
    let turn_scores = Bm25Index::build(&turn_texts).scores(query);
    let mut ranked_turns: Vec<(f64, usize)> = turn_scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| (score, index))
        .filter(|(score, _)| *score > 0.0)
        .collect();
    ranked_turns.sort_by(|(left_score, left_index), (right_score, right_index)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });
    let Some((top_score, _)) = ranked_turns.first().copied() else {
        return (Vec::new(), Vec::new());
    };
    let cutoff = f64::from(options.turn_score_ratio) * top_score;
    ranked_turns.truncate(options.turn_top_k);
    let chunks_by_id: HashMap<&str, &RecallChunk> = archive
        .chunks
        .iter()
        .map(|chunk| (chunk.chunk_id.as_str(), chunk))
        .collect();
    let mut matched_turns = Vec::new();
    let mut matched_chunks = Vec::new();
    for (turn_score, turn_index) in ranked_turns {
        if turn_score < cutoff {
            continue;
        }
        let turn = &archive.turns[turn_index];
        matched_turns.push(ScoredTurn {
            turn_id: turn.turn_id.clone(),
            memory_id: archive.memory_id.clone(),
            query: turn.query.clone(),
            score: turn_score,
            raw_score: turn_score,
        });
        let turn_chunks: Vec<&RecallChunk> = turn
            .chunk_ids
            .iter()
            .filter_map(|chunk_id| chunks_by_id.get(chunk_id.as_str()).copied())
            .collect();
        if turn_chunks.is_empty() {
            continue;
        }
        let chunk_texts: Vec<&str> = turn_chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect();
        let chunk_scores = Bm25Index::build(&chunk_texts).scores(query);
        let mut ranked_chunks: Vec<(f64, usize)> = chunk_scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| (score, index))
            .filter(|(score, _)| *score > 0.0)
            .collect();
        ranked_chunks.sort_by(|(left_score, left_index), (right_score, right_index)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });
        ranked_chunks.truncate(options.chunk_top_k);
        for (chunk_score, chunk_index) in ranked_chunks {
            let chunk = turn_chunks[chunk_index];
            matched_chunks.push(ScoredChunk {
                chunk_id: chunk.chunk_id.clone(),
                turn_id: chunk.turn_id.clone(),
                memory_id: archive.memory_id.clone(),
                score: chunk_score,
                raw_score: chunk_score,
                tokens: chunk.tokens,
                text: chunk.text.clone(),
            });
        }
    }
    (matched_turns, matched_chunks)
}

/// Normalizes turn scores so the archive's best turn scores `1.0`.
fn normalize_turns(turns: &mut [ScoredTurn]) {
    let top_score = turns.iter().map(|turn| turn.score).fold(0.0, f64::max);
    if top_score > 0.0 {
        for turn in turns {
            turn.score /= top_score;
        }
    }
}

/// Normalizes chunk scores so the archive's best chunk scores `1.0`.
fn normalize_chunks(chunks: &mut [ScoredChunk]) {
    let top_score = chunks.iter().map(|chunk| chunk.score).fold(0.0, f64::max);
    if top_score > 0.0 {
        for chunk in chunks {
            chunk.score /= top_score;
        }
    }
}

/// Deterministic cross-archive chunk ordering.
fn compare_chunks(left: &ScoredChunk, right: &ScoredChunk) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.raw_score.total_cmp(&left.raw_score))
        .then_with(|| left.chunk_id.cmp(&right.chunk_id))
}

/// Deterministic cross-archive turn ordering.
fn compare_turns(left: &ScoredTurn, right: &ScoredTurn) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.memory_id.cmp(&right.memory_id))
        .then_with(|| left.turn_id.cmp(&right.turn_id))
}

/// Keeps chunks until `budget` tokens are used; over-budget chunks are skipped.
fn apply_budget(
    candidates: Vec<ScoredChunk>,
    budget: usize,
) -> (Vec<RecallChunkMatch>, bool, usize) {
    let mut chunks = Vec::new();
    let mut returned_tokens: usize = 0;
    let mut truncated = false;
    for candidate in candidates {
        if returned_tokens.saturating_add(candidate.tokens) > budget {
            truncated = true;
            continue;
        }
        returned_tokens += candidate.tokens;
        chunks.push(RecallChunkMatch {
            chunk_id: candidate.chunk_id,
            turn_id: candidate.turn_id,
            memory_id: candidate.memory_id,
            score: candidate.score as f32,
            raw_score: candidate.raw_score as f32,
            tokens: candidate.tokens,
            text: candidate.text,
        });
    }
    (chunks, truncated, returned_tokens)
}

/// Recalls budget-capped chunks of compressed conversation content.
///
/// With `memory_id` the search is confined to that archive and scores stay raw
/// BM25 values. Without it, every archive listed for `scope` is searched, each
/// archive's best score is normalized to `1.0` (raw scores are kept), and the
/// results are merged deterministically. A blank query or an absent archive is
/// an empty result, never an error.
///
/// # Errors
///
/// Propagates store failures from [`RecallArchiveStore::get`] and
/// [`RecallArchiveStore::list`], and surfaces malformed archives as
/// [`Error::Validation`].
pub async fn recall_compressed_context(
    store: &dyn RecallArchiveStore,
    scope: &OffloadScope,
    memory_id: Option<&str>,
    query: &str,
    options: &RecallOptions,
) -> Result<RecallResult> {
    let mut result = RecallResult {
        memory_id: memory_id.map(str::to_owned),
        ..RecallResult::default()
    };
    if query.trim().is_empty() {
        return Ok(result);
    }
    let Some(memory_id) = memory_id else {
        let mut memory_ids = store.list(scope).await?;
        memory_ids.sort();
        result.archives.clone_from(&memory_ids);
        let mut turns = Vec::new();
        let mut chunks = Vec::new();
        for memory_id in &memory_ids {
            let Some(archive) = store.get(scope, memory_id).await? else {
                tracing::warn!(
                    memory_id = memory_id.as_str(),
                    session_id = scope.session_id.as_str(),
                    context_id = scope.context_id.as_str(),
                    "recall archive listed but missing; skipping"
                );
                continue;
            };
            let (mut archive_turns, mut archive_chunks) = search_archive(&archive, query, options);
            normalize_turns(&mut archive_turns);
            normalize_chunks(&mut archive_chunks);
            turns.extend(archive_turns);
            chunks.extend(archive_chunks);
        }
        turns.sort_by(compare_turns);
        chunks.sort_by(compare_chunks);
        let (chunks, truncated, returned_tokens) = apply_budget(chunks, options.max_content_tokens);
        result.matched_turns = turns.into_iter().map(into_turn_match).collect();
        result.chunks = chunks;
        result.truncated = truncated;
        result.returned_tokens = returned_tokens;
        return Ok(result);
    };
    let Some(archive) = store.get(scope, memory_id).await? else {
        return Ok(result);
    };
    result.archives.push(memory_id.to_owned());
    let (turns, mut chunks) = search_archive(&archive, query, options);
    chunks.sort_by(compare_chunks);
    let (chunks, truncated, returned_tokens) = apply_budget(chunks, options.max_content_tokens);
    result.matched_turns = turns.into_iter().map(into_turn_match).collect();
    result.chunks = chunks;
    result.truncated = truncated;
    result.returned_tokens = returned_tokens;
    Ok(result)
}

/// Converts an internal turn candidate into the public match type.
fn into_turn_match(turn: ScoredTurn) -> RecallTurnMatch {
    RecallTurnMatch {
        turn_id: turn.turn_id,
        memory_id: turn.memory_id,
        query: turn.query,
        score: turn.score as f32,
        raw_score: turn.raw_score as f32,
    }
}
