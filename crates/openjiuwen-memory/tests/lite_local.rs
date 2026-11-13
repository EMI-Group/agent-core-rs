//! Integration coverage of the local, file-backed coding-memory stack.
//!
//! [`LocalCodingMemory`](openjiuwen_memory::lite::local::LocalCodingMemory) owns a
//! human-editable `MEMORY.md` as its source of truth, a derived index under `.memory/`,
//! and a debounced watcher. The `lite` submodules unit-test the pieces in isolation
//! (Markdown parse/render, `FTS5`, the advisory lock, the watcher); this suite drives
//! the assembled store from the outside: write/edit/delete through the facade, the
//! rendered document on disk, a second instance on the same root, a hand-written
//! Markdown edit followed by `reindex()`, and the watcher.
//!
//! Every test opens its own [`tempfile::TempDir`] root, so the Markdown document, the
//! state directory, and the lock file of one test can never leak into another.
//!
//! Everything is offline and deterministic: search runs without an embedding model, so
//! recall is keyword scoring (with the `FTS5` index merged in when the `sqlite` feature
//! is on); hand-written Markdown carries a fixed timestamp derived from `1_700_000_000`
//! rather than the wall clock, and the watcher is polled with a hard deadline instead of
//! an unbounded wait. Without the `sqlite` feature the keyword scan answers every query,
//! so the same suite passes in both builds.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_memory::config::CodingMemoryConfig;
use openjiuwen_memory::lite::local::LocalCodingMemory;
use openjiuwen_memory::lite::{CodingFact, CodingFactType};
use openjiuwen_memory::record::{MemoryScope, RecallHit, SearchMode, SearchOptions};
use serde_json::Value;
use tempfile::TempDir;

/// Upper bound on every watcher poll, so a broken watcher fails instead of hanging.
const WATCH_BUDGET: Duration = Duration::from_secs(5);

/// Delay between two watcher polls.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// RFC 3339 rendering of the shared test epoch ([`ts`] with offset `0`); hand-written
/// Markdown uses it verbatim so the parse stays deterministic.
const EPOCH_STAMP: &str = "2023-11-14T22:13:20Z";

/// A fixed instant `offset` seconds after the shared test epoch.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("a valid timestamp")
}

/// The scope every store of this suite is opened in.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// A fact in [`scope`] with a pinned creation time (the store re-stamps it on write).
fn fact(name: &str, content: &str) -> CodingFact {
    let mut fact = CodingFact::new(&scope(), name, content);
    fact.updated_at = ts(0);
    fact
}

/// Opens a store on `temp` in [`scope`].
async fn store_at(temp: &TempDir) -> LocalCodingMemory {
    LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
        .await
        .expect("open the local coding-memory store")
}

/// Keyword search options that keep every candidate, so a body-only match survives the
/// default `threshold = 0.3`.
fn keyword_options() -> SearchOptions {
    SearchOptions::default()
        .with_mode(SearchMode::Keyword)
        .with_threshold(0.0)
        .with_top_k(10)
}

/// The coding-fact name carried in a hit's record metadata.
fn hit_name(hit: &RecallHit) -> &str {
    hit.record
        .metadata
        .get("name")
        .and_then(Value::as_str)
        .expect("a coding fact name")
}

/// The names of the hits for `query`, best match first.
async fn search_names(store: &LocalCodingMemory, query: &str) -> Vec<String> {
    let hits = store
        .search(query, &keyword_options())
        .await
        .expect("local search");
    hits.iter().map(hit_name).map(str::to_owned).collect()
}

/// Reads `MEMORY.md` of `store` back from disk.
fn document(store: &LocalCodingMemory) -> String {
    std::fs::read_to_string(store.memory_path()).expect("read the memory document")
}

/// Feature-gated checks that the derived `FTS5` index holds the same facts, is rebuilt
/// by `reindex()`, and reproduces the keyword ranking.
#[cfg(feature = "sqlite")]
async fn assert_fts_backed_store(store: &LocalCodingMemory) {
    assert_eq!(
        store.indexed_count().await.expect("indexed row count"),
        2,
        "the derived index must hold both facts"
    );
    assert_eq!(store.reindex().await.expect("reindex"), 2);
    assert_eq!(
        store.indexed_count().await.expect("count after reindex"),
        2,
        "reindex must rebuild the same two rows"
    );
    assert_eq!(
        search_names(store, "ranking").await,
        vec!["ranking-strong-fact", "weak-observation-fact"],
        "the FTS-backed search must stay best-match-first"
    );
}

#[tokio::test]
async fn the_markdown_document_round_trips_to_a_fresh_instance() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _written = store
        .write(
            fact("round-trip-fact", "cargo build the workspace")
                .with_description("Persisted in MEMORY.md")
                .with_tags(vec!["build".to_owned()])
                .with_path("docs/BUILD.md"),
        )
        .await
        .expect("write");

    let text = document(&store);
    assert!(
        text.starts_with("---\nschema: openjiuwen-memory-lite"),
        "the document must carry the schema front matter, got {text:?}"
    );
    assert!(text.contains("### round-trip-fact"), "got {text:?}");
    assert!(
        text.contains("- description: Persisted in MEMORY.md"),
        "got {text:?}"
    );
    assert!(text.contains("cargo build the workspace"), "got {text:?}");

    let snapshot = store.markdown().await.expect("markdown snapshot");
    let in_snapshot = snapshot.fact("round-trip-fact").expect("the fact");
    assert_eq!(in_snapshot.content, "cargo build the workspace");
    assert_eq!(in_snapshot.description, "Persisted in MEMORY.md");
    assert_eq!(in_snapshot.fact_type, CodingFactType::Project);

    let reopened = LocalCodingMemory::open(temp.path(), CodingMemoryConfig::default())
        .await
        .expect("reopen the store");
    let read = reopened
        .read("round-trip-fact")
        .await
        .expect("read")
        .expect("the committed fact");
    assert_eq!(read.content, "cargo build the workspace");
    assert_eq!(read.tags, vec!["build".to_owned()]);
    assert_eq!(read.path.as_deref(), Some("docs/BUILD.md"));
    let listed = reopened.list().await.expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, "round-trip-fact");
    assert_eq!(reopened.scope(), &MemoryScope::default());
    assert_eq!(
        reopened.reload_count(),
        0,
        "opening a store already loads the document, so no reload was needed"
    );
}

#[tokio::test]
async fn search_ranks_the_stronger_keyword_match_first() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _strong = store
        .write(
            fact("ranking-strong-fact", "the strongest match of this corpus")
                .with_description("ranked first")
                .with_tags(vec!["ranking".to_owned()]),
        )
        .await
        .expect("write the strong fact");
    let _weak = store
        .write(fact(
            "weak-observation-fact",
            "a single passing mention of ranking buried in a long body",
        ))
        .await
        .expect("write the weak fact");

    let hits = store
        .search("ranking", &keyword_options())
        .await
        .expect("search");
    assert_eq!(hits.len(), 2, "both facts match, got {hits:?}");
    assert_eq!(hit_name(&hits[0]), "ranking-strong-fact");
    assert_eq!(hit_name(&hits[1]), "weak-observation-fact");
    assert!(
        hits[0].score > hits[1].score,
        "the name/tag match must outrank the body-only match: {:?} vs {:?}",
        hits[0].score,
        hits[1].score
    );
    assert!(
        hits[0].score > 0.0 && hits[0].score <= 1.0,
        "scores stay normalized, got {:?}",
        hits[0].score
    );

    // The keyword scan alone answers the query in both builds; the derived index adds
    // `FTS5` candidates on top of it when the feature is on.
    assert_eq!(
        search_names(&store, "corpus").await,
        vec!["ranking-strong-fact"]
    );
    #[cfg(feature = "sqlite")]
    assert_fts_backed_store(&store).await;
}

#[tokio::test]
async fn edit_rewrites_the_markdown_body_and_the_search_hits() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _written = store
        .write(
            fact("editable-fact", "cargo build the debug workspace")
                .with_description("How to build"),
        )
        .await
        .expect("write");
    assert!(document(&store).contains("debug workspace"));

    let edited = store
        .edit("editable-fact", "cargo test the release workspace")
        .await
        .expect("edit");
    assert_eq!(edited.content, "cargo test the release workspace");
    assert_eq!(
        edited.description, "How to build",
        "an edit preserves the description"
    );

    let text = document(&store);
    assert!(text.contains("release workspace"), "got {text:?}");
    assert!(
        !text.contains("debug workspace"),
        "the replaced body must be gone, got {text:?}"
    );

    assert_eq!(search_names(&store, "release").await, vec!["editable-fact"]);
    assert!(
        search_names(&store, "debug").await.is_empty(),
        "the old wording must no longer be recalled"
    );
}

#[tokio::test]
async fn delete_drops_the_fact_from_the_markdown_and_the_search_index() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _written = store
        .write(fact("temporary-fact", "throwaway body about indexing"))
        .await
        .expect("write");
    assert!(document(&store).contains("### temporary-fact"));

    assert!(
        store.delete("temporary-fact").await.expect("first delete"),
        "deleting an existing fact reports the removal"
    );
    assert!(
        !store.delete("temporary-fact").await.expect("second delete"),
        "deleting an absent fact reports no removal"
    );

    let text = document(&store);
    assert!(!text.contains("temporary-fact"), "got {text:?}");
    assert!(!text.contains("throwaway body"), "got {text:?}");
    let remaining = store.list().await.expect("list");
    assert!(
        remaining.is_empty(),
        "the store must hold no fact, got {remaining:?}"
    );
    assert!(
        search_names(&store, "throwaway").await.is_empty(),
        "a deleted fact must not be recalled"
    );
    assert_eq!(
        store.render_index().await.expect("index"),
        "# Memory\n",
        "the summary index keeps only its title"
    );
    #[cfg(feature = "sqlite")]
    assert_eq!(
        store.indexed_count().await.expect("indexed row count"),
        0,
        "the derived index must drop the deleted row"
    );
}

#[tokio::test]
async fn reindex_picks_up_a_hand_written_markdown_entry() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _written = store
        .write(fact("store-fact", "written by the store"))
        .await
        .expect("write");

    let text = document(&store);
    let edited = format!(
        "{text}\n## Feedback\n\n\
         ### editorial-note\n\n\
         - description: Added by hand\n\
         - tags: []\n\
         - updated_at: {EPOCH_STAMP}\n\n\
         written by an editor\n"
    );
    std::fs::write(store.memory_path(), edited).expect("write the edited document");

    let before = store.reload_count();
    let reindexed = store.reindex().await.expect("reindex");
    assert_eq!(reindexed, 2, "the file now holds two facts");
    assert_eq!(
        store.reload_count(),
        before + 1,
        "reindex always re-reads the file"
    );

    let note = store
        .read("editorial-note")
        .await
        .expect("read")
        .expect("the hand-written fact");
    assert_eq!(note.content, "written by an editor");
    assert_eq!(note.description, "Added by hand");
    assert_eq!(note.fact_type, CodingFactType::Feedback);
    assert_eq!(note.updated_at, ts(0), "the attribute timestamp is parsed");
    assert_eq!(store.list().await.expect("list").len(), 2);
    assert_eq!(
        search_names(&store, "editorial").await,
        vec!["editorial-note"]
    );
    #[cfg(feature = "sqlite")]
    assert_eq!(store.indexed_count().await.expect("indexed row count"), 2);
}

#[tokio::test]
async fn a_second_instance_sees_a_fact_committed_before_it_opened() {
    let temp = TempDir::new().expect("temp dir");
    let first = store_at(&temp).await;
    let _written = first
        .write(fact("shared-fact", "committed by the first instance"))
        .await
        .expect("write");

    let second = store_at(&temp).await;
    let read = second
        .read("shared-fact")
        .await
        .expect("read")
        .expect("the committed fact");
    assert_eq!(read.content, "committed by the first instance");
    assert_eq!(second.list().await.expect("list").len(), 1);
    assert_eq!(
        second.reload_count(),
        0,
        "the document was already there when the second instance opened"
    );
    assert_eq!(
        search_names(&second, "committed").await,
        vec!["shared-fact"]
    );

    let _second_write = second
        .write(fact("late-fact", "committed by the second instance"))
        .await
        .expect("second write");
    assert_eq!(first.list().await.expect("first list").len(), 2);
    assert_eq!(
        first.reload_count(),
        1,
        "the first instance reloads when it notices the new commit"
    );
}

#[tokio::test]
async fn the_store_layout_and_summary_index_describe_the_facts() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    assert_eq!(store.root(), temp.path());
    assert_eq!(store.memory_path(), temp.path().join("MEMORY.md"));
    assert_eq!(store.lock_path(), temp.path().join(".memory/index.lock"));
    assert_eq!(store.scope(), &scope());
    assert_eq!(store.config(), &CodingMemoryConfig::default());
    assert!(
        store.lock_path().exists(),
        "opening the store takes the lock"
    );
    assert_eq!(
        store.render_index().await.expect("empty index"),
        "# Memory\n"
    );

    let _project = store
        .write(
            fact("project-layout", "crates/ holds every workspace member")
                .with_description("Layout")
                .with_path("docs/ARCHITECTURE.md"),
        )
        .await
        .expect("write the project fact");
    let _feedback = store
        .write(
            fact("tone-rule", "answer concisely")
                .with_type(CodingFactType::Feedback)
                .with_description("Tone"),
        )
        .await
        .expect("write the feedback fact");

    assert_eq!(
        store.render_index().await.expect("index"),
        concat!(
            "# Memory\n",
            "- [Feedback] tone-rule — Tone\n",
            "- [Project] project-layout — Layout (`docs/ARCHITECTURE.md`)\n",
        )
    );
}

#[cfg(unix)]
#[tokio::test]
async fn watching_reindexes_an_external_markdown_edit() {
    let temp = TempDir::new().expect("temp dir");
    let store = store_at(&temp).await;
    let _written = store
        .write(fact("watched-fact", "written before watching"))
        .await
        .expect("write");
    if store.start_watching().await.is_err() {
        // No usable filesystem watcher on this platform; the `watch` unit tests own
        // that path, so there is nothing to assert here.
        return;
    }
    assert!(store.is_watching().await, "the watcher must report itself");

    let before = store.reload_count();
    let text = document(&store);
    let edited = format!(
        "{text}\n## Project\n\n\
         ### editor-fact\n\n\
         - description: Edited while watching\n\
         - tags: []\n\
         - updated_at: {EPOCH_STAMP}\n\n\
         picked up by the watcher\n"
    );
    std::fs::write(store.memory_path(), edited).expect("write the edited document");

    let deadline = tokio::time::Instant::now() + WATCH_BUDGET;
    let mut reloaded = false;
    while tokio::time::Instant::now() < deadline {
        if store.reload_count() > before {
            reloaded = true;
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    assert!(
        reloaded,
        "the watcher never reloaded the document within {WATCH_BUDGET:?}"
    );
    assert_eq!(
        search_names(&store, "watcher").await,
        vec!["editor-fact"],
        "search must see the externally added fact"
    );

    store.stop_watching().await;
    assert!(
        !store.is_watching().await,
        "the watcher must report its stop"
    );
}
