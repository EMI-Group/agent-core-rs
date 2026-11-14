//! Integration coverage of the opt-in local coding-memory stack.
//!
//! `tests/lite_local.rs` owns the default single-document, keyword-only store; the `lite`
//! submodules unit-test the pieces in isolation. This suite drives the surfaces layered on
//! top of that default from the outside: the chunked embedding index and its content-hash
//! cache (semantic and hybrid recall), the per-fact `facts/` layout with the
//! `MEMORY.md`/`USER.md` index split, JSONL session recall, the caller-driven sync triggers,
//! `daily_memory` rollups, and the write modes with bounded conflict retries.
//!
//! Every test opens its own [`tempfile::TempDir`] root, so the Markdown documents, the
//! derived `.memory/` state, the `facts/` directory, the `sessions/` logs, and the
//! `daily_memory/` rollups of one test can never leak into another.
//!
//! Everything is offline and deterministic: the embedding model is
//! [`HashEmbedding`] (or a counting wrapper delegating to it), the dates are fixed
//! `NaiveDate`s, the retry policy uses millisecond backoff, the background interval task is
//! started with an hour-long interval and stopped immediately, and no assertion depends on
//! the wall clock. The persistent-cache assertion is gated on the `sqlite` feature, so the
//! suite passes in both builds.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_memory::config::CodingMemoryConfig;
use openjiuwen_memory::lite::daily;
use openjiuwen_memory::lite::embed::{blend, EmbeddingStats};
use openjiuwen_memory::lite::facts::{file_name_for, parse_level, render_level, IndexLevel};
use openjiuwen_memory::lite::local::LocalCodingMemory;
use openjiuwen_memory::lite::markdown::MarkdownMemory;
use openjiuwen_memory::lite::session::{SessionRecord, SessionStore};
use openjiuwen_memory::lite::sync::SyncTrigger;
use openjiuwen_memory::lite::write::{WriteMode, WritePolicy};
use openjiuwen_memory::lite::{CodingFact, CodingFactType};
use openjiuwen_memory::record::{MemoryScope, RecallHit, SearchMode, SearchOptions};
use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
use tempfile::TempDir;

/// Vector dimension of the test embedder.
///
/// Identical texts hash to identical vectors (a similarity of `1.0`), while unrelated
/// short texts land near a cosine of `0.0` (a normalized similarity around `0.5`), which
/// keeps the ranking assertions stable.
const DIMENSION: usize = 16;

/// The weight the store gives the semantic signal in a hybrid score.
const VECTOR_WEIGHT: f32 = 0.7;

/// The weight the store gives the keyword signal in a hybrid score.
const TEXT_WEIGHT: f32 = 0.3;

/// The scope every store of this suite is opened in.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// A project fact in [`scope`] named `name` with `content` as its body.
fn fact(name: &str, content: &str) -> CodingFact {
    CodingFact::new(&scope(), name, content)
}

/// The fixed day the session, rollup, and sync assertions use.
fn day() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 2).expect("a valid date")
}

/// The day before [`day`].
fn previous_day() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).expect("a valid date")
}

/// 09:00 UTC on `date`, the pinned time of every session line and fact of this suite.
fn midday(date: NaiveDate) -> DateTime<Utc> {
    date.and_hms_opt(9, 0, 0)
        .expect("a valid clock time")
        .and_utc()
}

/// The coding-fact name carried in a hit's record metadata.
fn hit_name(hit: &RecallHit) -> &str {
    hit.record
        .metadata
        .get("name")
        .and_then(serde_json::Value::as_str)
        .expect("a coding fact name")
}

/// Reads a file as text.
fn read(path: &Path) -> String {
    std::fs::read_to_string(path).expect("the file is readable")
}

/// Opens a keyword-only (no embedding) store rooted at `temp`.
async fn open_keyword(temp: &TempDir) -> LocalCodingMemory {
    LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
        .await
        .expect("open the keyword-only store")
}

/// Opens a store rooted at `temp` whose semantic recall is backed by `counter`.
async fn open_embedded_with(temp: &TempDir, counter: &Arc<CountingEmbedding>) -> LocalCodingMemory {
    let embedding: Arc<dyn Embedding> = counter.clone();
    LocalCodingMemory::builder(temp.path(), CodingMemoryConfig::default())
        .scope(scope())
        .embedding(embedding)
        .open()
        .await
        .expect("open the embedded store")
}

/// Opens a store rooted at `temp` plus its counting embedding model.
async fn open_embedded(temp: &TempDir) -> (LocalCodingMemory, Arc<CountingEmbedding>) {
    let counter = Arc::new(CountingEmbedding::new(DIMENSION));
    let store = open_embedded_with(temp, &counter).await;
    (store, counter)
}

/// Replaces the store's `MEMORY.md` document with `facts`.
///
/// The single-document layout is the source of truth, so this is the documented way to
/// pin timestamps the write path would stamp with "now". Nothing is reindexed here: the
/// store notices the changed stamp and reloads on its next read or refresh.
fn write_markdown(store: &LocalCodingMemory, facts: &[CodingFact]) {
    let mut document = MarkdownMemory::new().with_scope(scope());
    for fact in facts {
        let _replaced = document.upsert_fact(fact);
    }
    std::fs::write(store.memory_path(), document.render()).expect("write MEMORY.md");
}

/// Seeds one fact on [`day`], one on [`previous_day`], and one session log per day.
///
/// A rollup selects a fact by its `updated_at` date, which the write path stamps with
/// "now", so both facts are pinned through the source-of-truth document; each session line
/// carries its own pinned timestamp.
async fn seed_rollup_inputs(store: &LocalCodingMemory) {
    let _build = store
        .write(fact("build", "cargo build"))
        .await
        .expect("the write");
    let _older = store
        .write(fact("older", "an older body"))
        .await
        .expect("the write");
    let pinned: Vec<CodingFact> = store
        .list()
        .await
        .expect("the list")
        .into_iter()
        .map(|mut fact| {
            fact.updated_at = if fact.name == "build" {
                midday(day())
            } else {
                midday(previous_day())
            };
            fact
        })
        .collect();
    write_markdown(store, &pinned);
    store.reindex().await.expect("the reindex");
    let _today = store
        .append_session(
            "one",
            day(),
            &[SessionRecord::new("ran the build").with_timestamp(midday(day()))],
        )
        .await
        .expect("the session append");
    let _yesterday = store
        .append_session(
            "two",
            previous_day(),
            &[SessionRecord::new("yesterday's turn").with_timestamp(midday(previous_day()))],
        )
        .await
        .expect("the session append");
}

/// A [`HashEmbedding`] that counts the calls the store makes on it.
///
/// It delegates every call to the deterministic hash embedder, so the produced vectors
/// are the real ones; only the call counters are test-only.
struct CountingEmbedding {
    /// The real embedder producing deterministic vectors.
    inner: HashEmbedding,
    /// Number of `embed_documents` calls.
    documents: AtomicUsize,
    /// Number of texts handed to `embed_documents`.
    texts: AtomicUsize,
    /// Number of `embed_query` calls.
    queries: AtomicUsize,
}

impl CountingEmbedding {
    /// Creates a counting embedder around a `dimension`-sized hash embedder.
    fn new(dimension: usize) -> Self {
        Self {
            inner: HashEmbedding::new(dimension),
            documents: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
        }
    }

    /// The `embed_documents` call count.
    fn document_calls(&self) -> usize {
        self.documents.load(Ordering::SeqCst)
    }

    /// The number of texts passed to `embed_documents`.
    fn embedded_texts(&self) -> usize {
        self.texts.load(Ordering::SeqCst)
    }

    /// The `embed_query` call count.
    fn query_calls(&self) -> usize {
        self.queries.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Embedding for CountingEmbedding {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn dimension_hint(&self) -> Option<usize> {
        self.inner.dimension_hint()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.embed_query(text).await
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.documents.fetch_add(1, Ordering::SeqCst);
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.embed_documents(texts, batch_size).await
    }
}

/// Semantic recall ranks the exact match first and reports every score component.
///
/// Ports the semantic half of Python `core/memory/lite/manager.py` on top of the local
/// store: a fact's semantic score is the best similarity of its own chunks, and a hit
/// carries the raw components the ranking blended.
#[tokio::test]
async fn semantic_search_ranks_the_closest_fact_and_exposes_the_components() {
    let temp = TempDir::new().expect("temp dir");
    let (store, counter) = open_embedded(&temp).await;
    assert!(store.has_embedding());
    assert!(store.embedding().is_some());
    assert_eq!(store.chunking().target_tokens, 256);
    for (name, body) in [
        ("theme", "the editor uses a dark theme by default"),
        ("build", "cargo build --workspace"),
        ("deploy", "ship the release with the deploy script"),
    ] {
        let _stored = store
            .write(fact(name, body))
            .await
            .expect("the fact is written");
    }
    assert_eq!(
        store.embedding_chunks().await,
        3,
        "every fact contributes one chunk"
    );
    let options = SearchOptions::default()
        .with_mode(SearchMode::Semantic)
        .with_threshold(0.0)
        .with_top_k(10);
    let hits = store
        .search("cargo build --workspace", &options)
        .await
        .expect("the semantic search succeeds");
    assert_eq!(hits.len(), 3, "a zero threshold keeps every fact");
    assert_eq!(hit_name(&hits[0]), "build");
    assert!(
        hits[0].similarity > 0.99,
        "an identical body scores 1.0, got {}",
        hits[0].similarity
    );
    assert!(
        hits[0].score > hits[1].score,
        "the semantic ranking is strict: {hits:?}"
    );
    assert!(
        hits.windows(2).all(|pair| pair[0].score >= pair[1].score),
        "hits are ordered by score descending: {hits:?}"
    );
    for hit in &hits {
        assert!(
            hit.similarity > 0.0 && hit.similarity <= 1.0,
            "the semantic component is reported in 0.0..=1.0, got {}",
            hit.similarity
        );
        assert!(
            (0.0..=1.0).contains(&hit.keyword),
            "the keyword component is reported in 0.0..=1.0, got {}",
            hit.keyword
        );
        assert!(
            (hit.importance - 0.5).abs() < f32::EPSILON,
            "a coding fact reports the fixed importance, got {}",
            hit.importance
        );
        assert!(
            hit.recency > 0.0 && hit.recency <= 1.0,
            "the recency component is reported in 0.0..=1.0, got {}",
            hit.recency
        );
        assert!(
            (hit.score - hit.similarity).abs() < f32::EPSILON,
            "semantic mode scores by the semantic signal alone, got {} vs {}",
            hit.score,
            hit.similarity
        );
    }
    assert!(counter.query_calls() >= 1, "the query is embedded");
}

/// Hybrid recall orders the hits by the documented blend of the reported components.
///
/// The blend is `(0.7 * semantic + 0.3 * keyword) / 1.0`
/// ([`blend`](openjiuwen_memory::lite::embed::blend)); the store reports both components on
/// every [`RecallHit`], so the ordering is fully reconstructible from the outside. A store
/// without an embedding model answers [`SearchMode::Semantic`] with
/// [`Error::Unsupported`].
#[tokio::test]
async fn hybrid_search_orders_by_the_documented_blend() {
    let temp = TempDir::new().expect("temp dir");
    let (store, counter) = open_embedded(&temp).await;
    // The keyword signal prefers `release-notes` (both query tokens match its name)
    // while the semantic signal prefers `shipping` (its body is the query verbatim).
    for (name, body) in [
        ("release-notes", "unrelated words here"),
        ("shipping", "release notes"),
        ("other", "nothing else"),
    ] {
        let _stored = store
            .write(fact(name, body))
            .await
            .expect("the fact is written");
    }
    let keyword = store
        .search(
            "release notes",
            &SearchOptions::default()
                .with_mode(SearchMode::Keyword)
                .with_threshold(0.0)
                .with_top_k(10),
        )
        .await
        .expect("the keyword search succeeds");
    assert_eq!(hit_name(&keyword[0]), "release-notes");
    assert_eq!(
        counter.query_calls(),
        0,
        "keyword mode never embeds the query"
    );

    let hybrid = store
        .search(
            "release notes",
            &SearchOptions::default()
                .with_mode(SearchMode::Hybrid)
                .with_threshold(0.0)
                .with_top_k(10),
        )
        .await
        .expect("the hybrid search succeeds");
    assert_eq!(hybrid.len(), 3);
    for hit in &hybrid {
        let expected = blend(hit.similarity, hit.keyword, VECTOR_WEIGHT, TEXT_WEIGHT);
        assert!(
            (hit.score - expected).abs() < 1e-6,
            "score {} is not the documented blend {expected}",
            hit.score
        );
    }
    let mut ordered: Vec<&RecallHit> = hybrid.iter().collect();
    ordered.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| hit_name(left).cmp(hit_name(right)))
    });
    let actual: Vec<&str> = hybrid.iter().map(hit_name).collect();
    let expected: Vec<&str> = ordered.iter().map(|hit| hit_name(hit)).collect();
    assert_eq!(
        actual, expected,
        "hits are ordered by the blend descending and then by name"
    );
    assert!(
        hybrid
            .iter()
            .any(|hit| (hit.score - hit.keyword).abs() > 1e-6),
        "a hybrid score is not the keyword score alone: {hybrid:?}"
    );
    assert_eq!(
        counter.query_calls(),
        1,
        "the hybrid query is embedded exactly once"
    );

    let plain_temp = TempDir::new().expect("temp dir");
    let plain = open_keyword(&plain_temp).await;
    let error = plain
        .search(
            "anything",
            &SearchOptions::default().with_mode(SearchMode::Semantic),
        )
        .await
        .expect_err("semantic mode needs an embedding model");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("embedding"), "{message}");
    assert!(message.contains("LocalCodingMemory::builder"), "{message}");
    assert_eq!(
        plain
            .refresh_embeddings()
            .await
            .expect("the keyword refresh"),
        EmbeddingStats::default(),
        "a store without an embedding index has nothing to refresh"
    );
}

/// The content-hash cache embeds a chunk once and then serves it from the cache.
///
/// A refresh over unchanged text makes no model call; a third fact embeds exactly one new
/// chunk. Reopening the same root reuses the persisted cache with the `sqlite` feature and
/// re-embeds with the default in-memory case.
#[tokio::test]
async fn the_embedding_cache_populates_once_and_serves_every_later_refresh() {
    let temp = TempDir::new().expect("temp dir");
    let (store, counter) = open_embedded(&temp).await;
    assert_eq!(store.embedding_chunks().await, 0);
    assert_eq!(counter.document_calls(), 0, "an empty store embeds nothing");
    assert_eq!(
        store.embedding_cache_len().await.expect("the cache length"),
        0
    );

    // `MEMORY.md` is the source of truth, so an external write is a caller edit the next
    // refresh embeds.
    write_markdown(
        &store,
        &[fact("build", "cargo build"), fact("test", "cargo test")],
    );
    let first = store.refresh_embeddings().await.expect("the first refresh");
    assert_eq!(first.chunks, 2, "the refresh populates the chunk index");
    assert_eq!(first.embedded, 2, "both chunks are new");
    assert_eq!(first.cached, 0);
    assert_eq!(first.dimension, Some(DIMENSION));
    assert_eq!(
        counter.document_calls(),
        1,
        "the misses are embedded in one batch"
    );
    assert_eq!(counter.embedded_texts(), 2);
    assert_eq!(store.embedding_chunks().await, 2);
    assert_eq!(
        store.embedding_cache_len().await.expect("the cache length"),
        2
    );

    let calls = counter.document_calls();
    let second = store
        .refresh_embeddings()
        .await
        .expect("the second refresh");
    assert_eq!(second.chunks, 2);
    assert_eq!(second.embedded, 0, "an unchanged document embeds nothing");
    assert_eq!(second.cached, 2, "every chunk comes from the cache");
    assert_eq!(
        counter.document_calls(),
        calls,
        "a cached refresh makes no model call"
    );
    assert_eq!(counter.embedded_texts(), 2);
    assert_eq!(
        store.embedding_cache_len().await.expect("the cache length"),
        2,
        "the cache did not grow"
    );

    let cached = store.embedding_cache_len().await.expect("the cache length");
    write_markdown(
        &store,
        &[
            fact("build", "cargo build"),
            fact("test", "cargo test"),
            fact("lint", "cargo clippy"),
        ],
    );
    let third = store.refresh_embeddings().await.expect("the third refresh");
    assert_eq!(third.chunks, 3);
    assert_eq!(third.embedded, 1, "only the new chunk is embedded");
    assert_eq!(third.cached, 2, "the unchanged chunks stay cached");
    assert!(
        store.embedding_cache_len().await.expect("the cache length") > cached,
        "the cache grows by the new chunk"
    );
    drop(store);

    let reopened_counter = Arc::new(CountingEmbedding::new(DIMENSION));
    let reopened = open_embedded_with(&temp, &reopened_counter).await;
    assert_eq!(reopened.embedding_chunks().await, 3);
    assert_eq!(
        reopened
            .embedding_cache_len()
            .await
            .expect("the cache length"),
        3
    );
    #[cfg(feature = "sqlite")]
    assert_eq!(
        reopened_counter.document_calls(),
        0,
        "the persisted cache feeds the reopened index"
    );
    #[cfg(not(feature = "sqlite"))]
    assert_eq!(
        reopened_counter.document_calls(),
        1,
        "the in-memory cache is empty after a reopen, so opening embeds the chunks"
    );
    assert_eq!(reopened_counter.query_calls(), 0);
}

/// The per-fact layout exports one file per fact and splits the two index documents.
///
/// `facts/<slug>-<hash>.md` is authoritative for content, `MEMORY.md` renders the non-user
/// facts, and `USER.md` renders the `user`-level ones; importing the exported documents
/// reproduces the same facts.
#[tokio::test]
async fn per_fact_files_split_the_user_and_memory_indexes() {
    let temp = TempDir::new().expect("temp dir");
    let store = LocalCodingMemory::builder(temp.path(), CodingMemoryConfig::default())
        .scope(scope())
        .fact_files(true)
        .open()
        .await
        .expect("open the fact-file store");
    assert!(store.fact_files_enabled());
    assert_eq!(store.facts_dir(), temp.path().join("facts"));
    assert_eq!(store.user_path(), temp.path().join("USER.md"));
    let _user = store
        .write(
            fact("theme", "prefers a dark theme")
                .with_type(CodingFactType::User)
                .with_description("Theme preference"),
        )
        .await
        .expect("the user fact is written");
    let _project = store
        .write(fact("layout", "crates/ holds every workspace member").with_description("Layout"))
        .await
        .expect("the project fact is written");

    let exported = store.export_index().await.expect("the export succeeds");
    assert_eq!(exported, 2);
    for name in ["theme", "layout"] {
        let path = store.facts_dir().join(file_name_for(name));
        assert!(path.is_file(), "the fact file for {name} exists");
        let text = read(&path);
        assert!(text.contains(&format!("name: {name}")), "{text}");
    }
    let memory = read(store.memory_path());
    assert!(memory.contains("### layout"), "{memory}");
    assert!(
        !memory.contains("theme"),
        "the user fact stays out of MEMORY.md: {memory}"
    );
    let user = read(&store.user_path());
    assert!(user.contains("# User"), "{user}");
    assert!(user.contains("### theme"), "{user}");
    assert!(
        !user.contains("layout"),
        "the project fact stays out of USER.md: {user}"
    );

    let imported = store.import_index().await.expect("the import succeeds");
    assert_eq!(imported, 2);
    assert_eq!(store.list().await.expect("the list").len(), 2);
    let theme = store
        .read("theme")
        .await
        .expect("the read succeeds")
        .expect("the user fact exists");
    assert_eq!(theme.fact_type, CodingFactType::User);
    assert_eq!(theme.description, "Theme preference");
    assert_eq!(theme.content, "prefers a dark theme");
}

/// A hand-edited index document wins over the per-fact files on import.
///
/// The fact files are authoritative for content, but an edit of `MEMORY.md`/`USER.md` is
/// explicit caller intent: `import_index` reads both documents, imports them, and rewrites
/// the fact files to match.
#[tokio::test]
async fn a_hand_edited_index_document_wins_over_the_fact_files() {
    let temp = TempDir::new().expect("temp dir");
    let store = LocalCodingMemory::builder(temp.path(), CodingMemoryConfig::default())
        .scope(scope())
        .fact_files(true)
        .open()
        .await
        .expect("open the fact-file store");
    let _project = store
        .write(fact("layout", "crates/ holds every workspace member").with_description("Layout"))
        .await
        .expect("the project fact is written");
    store.export_index().await.expect("the export succeeds");
    let fact_file = store.facts_dir().join(file_name_for("layout"));
    let stale = read(&fact_file).replace("description: Layout", "description: stale text");
    std::fs::write(&fact_file, stale).expect("the stale fact file is written");

    let mut facts = parse_level(&read(store.memory_path())).expect("parse MEMORY.md");
    assert_eq!(facts.len(), 1, "MEMORY.md holds the project fact");
    facts[0].description = "Edited by hand".to_owned();
    std::fs::write(
        store.memory_path(),
        render_level(&facts, IndexLevel::Memory),
    )
    .expect("the hand edit is written");

    let imported = store.import_index().await.expect("the import succeeds");
    assert_eq!(imported, 1);
    let layout = store
        .read("layout")
        .await
        .expect("the read succeeds")
        .expect("the project fact exists");
    assert_eq!(
        layout.description, "Edited by hand",
        "the index document wins over the fact file"
    );
    let rewritten = read(&fact_file);
    assert!(
        rewritten.contains("description: Edited by hand"),
        "{rewritten}"
    );
    assert!(!rewritten.contains("stale text"), "{rewritten}");
}

/// Appends are additive and session recall keeps the deterministic `(path, line)` order.
#[tokio::test]
async fn jsonl_sessions_append_number_and_rank_in_path_and_line_order() {
    let temp = TempDir::new().expect("temp dir");
    let store = open_keyword(&temp).await;
    assert_eq!(store.sessions_dir(), temp.path().join("sessions"));
    assert_eq!(
        store.session_records().await.expect("the records"),
        Vec::new(),
        "a missing sessions directory is an empty log set"
    );
    let first = store
        .append_session(
            "s-1",
            day(),
            &[SessionRecord::new("alpha beta gamma")
                .with_role("user")
                .with_timestamp(midday(day()))],
        )
        .await
        .expect("the first append");
    assert_eq!(
        first,
        temp.path()
            .join("sessions")
            .join("2026-01-02")
            .join("s-1.jsonl")
    );
    let second = store
        .append_session("s-1", day(), &[SessionRecord::new("alpha only")])
        .await
        .expect("the second append");
    assert_eq!(second, first, "an append reuses the same log file");

    let files = SessionStore::new(store.root())
        .files()
        .await
        .expect("the session files");
    assert_eq!(files, vec![first.clone()], "one log holds both appends");
    let chunks = store.session_records().await.expect("the decoded chunks");
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| (chunk.path.as_str(), chunk.line))
            .collect::<Vec<_>>(),
        vec![
            ("sessions/2026-01-02/s-1.jsonl", 1),
            ("sessions/2026-01-02/s-1.jsonl", 2),
        ],
        "chunks are numbered 1-based in (path, line) order"
    );
    assert_eq!(chunks[0].record.role.as_deref(), Some("user"));
    assert_eq!(chunks[0].record.timestamp, Some(midday(day())));

    let hits = store
        .search_sessions(
            "alpha beta gamma",
            &SearchOptions::default()
                .with_mode(SearchMode::Keyword)
                .with_threshold(0.0)
                .with_top_k(10),
        )
        .await
        .expect("the session search succeeds");
    assert_eq!(
        hits.iter().map(|hit| hit.line).collect::<Vec<_>>(),
        vec![1, 2],
        "the fully covered line ranks first: {hits:?}"
    );
    assert!((hits[0].score - 1.0).abs() < f32::EPSILON);
    assert!((hits[1].score - (1.0 / 3.0)).abs() < 1e-6);
    assert!(
        (hits[0].similarity - 0.0).abs() < f32::EPSILON,
        "keyword mode has no semantic term"
    );
    assert!((hits[1].keyword - hits[1].score).abs() < f32::EPSILON);
}

/// A hostile session id is sanitized into the day directory, and a blank one is rejected.
///
/// `sanitize_session_id` keeps ASCII alphanumerics plus `-`, `_`, and `.` and folds every
/// other character to `-`, so `"../x"` becomes the file name `..-x.jsonl` inside
/// `sessions/<date>/` rather than a traversal; only an id that sanitizes to nothing is
/// [`Error::Validation`].
#[tokio::test]
async fn hostile_session_ids_are_sanitized_and_blank_ids_are_rejected() {
    let temp = TempDir::new().expect("temp dir");
    let store = open_keyword(&temp).await;
    let path = store
        .append_session("../x", day(), &[SessionRecord::new("traversal attempt")])
        .await
        .expect("the id is sanitized instead of rejected");
    assert_eq!(
        path,
        temp.path()
            .join("sessions")
            .join("2026-01-02")
            .join("..-x.jsonl")
    );
    assert!(
        path.starts_with(store.sessions_dir()),
        "the log stays inside the session directory: {}",
        path.display()
    );
    assert!(path.is_file());
    for blank in ["", "   ", "\t"] {
        let error = store
            .append_session(blank, day(), &[SessionRecord::new("ignored")])
            .await
            .expect_err("a blank session id is rejected");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }
    let error = store
        .append_session("empty", day(), &[SessionRecord::new("   ")])
        .await
        .expect_err("an append without a non-empty record is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}

/// A daily rollup renders only, then writes idempotently, and selects one day's inputs.
#[tokio::test]
async fn daily_rollups_render_then_write_idempotently_for_one_day() {
    let temp = TempDir::new().expect("temp dir");
    let store = open_keyword(&temp).await;
    seed_rollup_inputs(&store).await;

    assert_eq!(store.daily_dir(), temp.path().join("daily_memory"));
    let rollup = store.rollup_day(day()).await.expect("the dry run renders");
    assert!(!rollup.written, "a dry run writes nothing");
    assert_eq!(rollup.id, "daily:2026-01-02");
    assert_eq!(rollup.date, day());
    assert_eq!(rollup.facts, vec!["build".to_owned()]);
    assert_eq!(
        rollup.sessions,
        vec!["sessions/2026-01-02/one.jsonl:1".to_owned()]
    );
    assert!(
        rollup.content.contains("- [Project] build"),
        "{}",
        rollup.content
    );
    assert!(
        rollup.content.contains("ran the build"),
        "{}",
        rollup.content
    );
    assert!(
        !rollup.content.contains("yesterday's turn"),
        "only the day's session lines are selected: {}",
        rollup.content
    );
    assert!(!rollup.content.contains("older"), "{}", rollup.content);
    assert!(
        std::fs::metadata(store.daily_dir()).is_err(),
        "a dry run creates no rollup directory"
    );

    let written = store
        .write_daily_rollup(day())
        .await
        .expect("the rollup is written");
    assert!(written.written, "the first write changes the file");
    assert_eq!(
        written.content, rollup.content,
        "both passes render the same document"
    );
    let path = daily::path_for(temp.path(), day());
    assert_eq!(path, temp.path().join("daily_memory").join("2026-01-02.md"));
    assert_eq!(read(&path), written.content);

    let again = store
        .write_daily_rollup(day())
        .await
        .expect("the second write");
    assert!(!again.written, "an unchanged day is not rewritten");
    assert_eq!(again.content, written.content);
    assert_eq!(read(&path), again.content, "the file is byte-identical");

    let other = store
        .rollup_day(previous_day())
        .await
        .expect("the other day renders");
    assert_eq!(other.facts, vec!["older".to_owned()]);
    assert_eq!(
        other.sessions,
        vec!["sessions/2026-01-01/two.jsonl:1".to_owned()]
    );
    assert!(other.content.contains("yesterday's turn"));
    assert!(!other.content.contains("ran the build"));
}

/// Every write mode applies its documented effect on an existing fact.
#[tokio::test]
async fn write_modes_produce_the_documented_outcomes() {
    let temp = TempDir::new().expect("temp dir");
    let store = open_keyword(&temp).await;
    let policy = WritePolicy::default();
    let created = store
        .write_mode(fact("build", "cargo build"), WriteMode::Create, policy)
        .await
        .expect("a create of a new fact succeeds");
    assert_eq!(created.mode, WriteMode::Create);
    assert!(created.created);
    assert_eq!(created.attempts, 1);
    assert!(!created.conflict);
    assert_eq!(created.note, None);
    assert_eq!(created.fact.content, "cargo build");

    let reloads = store.reload_count();
    let skipped = store
        .write_mode(fact("build", "ignored body"), WriteMode::Skip, policy)
        .await
        .expect("a skip of an existing fact succeeds");
    assert_eq!(skipped.mode, WriteMode::Skip);
    assert!(!skipped.created);
    assert_eq!(
        skipped.fact.content, "cargo build",
        "the stored fact is returned"
    );
    assert!(
        skipped
            .note
            .as_deref()
            .is_some_and(|note| note.contains("already exists")),
        "{:?}",
        skipped.note
    );
    assert_eq!(store.reload_count(), reloads, "a skip touches no file");
    assert_eq!(
        store
            .read("build")
            .await
            .expect("the read")
            .expect("the fact")
            .content,
        "cargo build"
    );

    let merged = store
        .write_mode(
            fact("build", "cargo build --release").with_description("build the workspace"),
            WriteMode::Merge,
            policy,
        )
        .await
        .expect("a merge of an existing fact succeeds");
    assert_eq!(merged.mode, WriteMode::Merge);
    assert!(!merged.created);
    assert_eq!(merged.attempts, 1);
    assert_eq!(merged.fact.content, "cargo build --release");
    assert_eq!(merged.fact.description, "build the workspace");
    assert_eq!(
        merged.fact.fact_type,
        CodingFactType::Project,
        "the stored fact type survives a merge"
    );

    let appended = store
        .write_mode(
            fact("build", "then run the tests"),
            WriteMode::Append,
            policy,
        )
        .await
        .expect("an append of an existing fact succeeds");
    assert_eq!(appended.mode, WriteMode::Append);
    assert!(!appended.created);
    assert_eq!(
        appended.fact.content, "cargo build --release\n\nthen run the tests",
        "an append keeps both bodies"
    );

    let replaced = store
        .write_mode(
            fact("build", "totally different"),
            WriteMode::Replace,
            policy,
        )
        .await
        .expect("a replace of an existing fact succeeds");
    assert_eq!(replaced.mode, WriteMode::Replace);
    assert!(!replaced.created);
    assert_eq!(replaced.fact.content, "totally different");
    assert_eq!(
        store
            .read("build")
            .await
            .expect("the read")
            .expect("the fact")
            .content,
        "totally different"
    );
}

/// An exhausted `create` conflict is a typed error that leaves the store untouched.
#[tokio::test]
async fn an_exhausted_create_conflict_is_a_typed_error() {
    let temp = TempDir::new().expect("temp dir");
    let store = open_keyword(&temp).await;
    let _stored = store
        .write(fact("build", "cargo build"))
        .await
        .expect("the write");
    let policy = WritePolicy::default()
        .with_max_attempts(2)
        .with_backoff(Duration::from_millis(1), Duration::from_millis(2));
    let error = store
        .write_mode(fact("build", "other body"), WriteMode::Create, policy)
        .await
        .expect_err("a create collision never overwrites the stored fact");
    assert!(matches!(error, Error::Memory(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("build"), "{message}");
    assert!(
        message.contains('2'),
        "the attempt budget is reported: {message}"
    );
    assert!(
        message.contains("conflict"),
        "the message explains the conflict: {message}"
    );
    assert_eq!(
        store
            .read("build")
            .await
            .expect("the read")
            .expect("the fact")
            .content,
        "cargo build",
        "the stored fact is unchanged"
    );
}

/// Sync is caller-driven: `sync_now` reports every counter, the triggers respect their
/// preconditions, and the interval task is opt-in and per handle.
#[tokio::test]
async fn sync_triggers_report_counters_and_own_the_interval_task() {
    let temp = TempDir::new().expect("temp dir");
    let (store, counter) = open_embedded(&temp).await;
    let _build = store
        .write(fact("build", "cargo build"))
        .await
        .expect("the write");
    let _test = store
        .write(fact("test", "cargo test --workspace"))
        .await
        .expect("the write");
    let _log = store
        .append_session(
            "s-1",
            day(),
            &[
                SessionRecord::new("ran the build").with_timestamp(midday(day())),
                SessionRecord::new("and the tests").with_timestamp(midday(day())),
            ],
        )
        .await
        .expect("the session append");
    let calls = counter.document_calls();

    let report = store.sync_now().await.expect("the sync succeeds");
    assert!(report.ran);
    assert_eq!(report.trigger, "sync_now");
    assert_eq!(report.facts, 2);
    assert_eq!(report.chunks, 2);
    assert_eq!(report.embedded, 0, "the writes already cached the vectors");
    assert_eq!(report.sessions, 2, "both session lines are counted");
    assert_eq!(
        counter.document_calls(),
        calls,
        "a sync over unchanged text makes no model call"
    );
    let again = store.sync_now().await.expect("the second sync succeeds");
    assert!(again.ran);
    assert_eq!(again.facts, report.facts);
    assert_eq!(again.chunks, report.chunks);
    assert_eq!(again.sessions, report.sessions);
    assert_eq!(again.embedded, 0);

    let start = store
        .on_trigger(SyncTrigger::OnSessionStart)
        .await
        .expect("the session-start trigger succeeds");
    assert!(start.ran, "a session start always runs a full sync");
    assert_eq!(start.trigger, "on_session_start");
    assert_eq!(start.facts, 2);
    assert_eq!(start.sessions, 2);

    let write = store
        .on_trigger(SyncTrigger::OnWrite)
        .await
        .expect("the write trigger succeeds");
    assert!(!write.ran, "nothing changed on disk");
    assert_eq!(write.trigger, "on_write");

    let interval = Duration::from_secs(3600);
    let first = store
        .on_trigger(SyncTrigger::OnInterval(interval))
        .await
        .expect("the first interval poll succeeds");
    assert!(first.ran, "the interval gate is due the first time");
    assert_eq!(first.trigger, "on_interval");
    let second = store
        .on_trigger(SyncTrigger::OnInterval(interval))
        .await
        .expect("the second interval poll succeeds");
    assert!(!second.ran, "the interval gate is not due again");
    assert_eq!(second.trigger, "on_interval");
    assert_eq!(second.facts, 0);

    assert!(!store.is_interval_sync_running().await);
    store
        .start_interval_sync(interval)
        .await
        .expect("the interval task starts");
    assert!(store.is_interval_sync_running().await);
    store.stop_interval_sync().await;
    assert!(!store.is_interval_sync_running().await);

    let error = store
        .on_trigger(SyncTrigger::OnInterval(Duration::ZERO))
        .await
        .expect_err("a zero interval is rejected");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert!(!store.is_interval_sync_running().await);
}
