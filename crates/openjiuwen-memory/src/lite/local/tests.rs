//! Unit tests of the local store: open/layout, the mutation round trip, typed errors,
//! keyword + FTS search, the summary index, staleness visibility, reindexing, document
//! round trips, concurrent writers, the watcher, and the handle's thread-safety.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use tempfile::TempDir;

use openjiuwen_common::error::Error as MemoryError;

use super::*;
use crate::config::CodingMemoryConfig;
use crate::lite::fact::CodingFact;
use crate::lite::markdown::MarkdownMemory;
use crate::lite::CodingFactType;
use crate::record::{MemoryScope, MemoryType, RecallHit, SearchMode, SearchOptions};

fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

async fn store_at(temp: &TempDir) -> LocalCodingMemory {
    LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
        .await
        .unwrap()
}

fn fact(name: &str, content: &str) -> CodingFact {
    CodingFact::new(&scope(), name, content)
}

fn keyword_options() -> SearchOptions {
    SearchOptions::default().with_mode(SearchMode::Keyword)
}

#[tokio::test]
async fn open_creates_the_document_and_the_state_directory() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    assert!(store.memory_path().exists());
    assert!(store.lock_path().exists());
    let text = std::fs::read_to_string(store.memory_path()).unwrap();
    assert!(text.contains("# Memory"), "{text}");
    assert_eq!(store.list().await.unwrap(), Vec::<CodingFact>::new());
    assert_eq!(store.reload_count(), 0);
    #[cfg(feature = "sqlite")]
    assert_eq!(store.indexed_count().await.unwrap(), 0);
}

#[tokio::test]
async fn write_read_edit_delete_round_trip() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let written = store
        .write(fact("build", "cargo build").with_description("How to build"))
        .await
        .unwrap();
    assert_eq!(written.name, "build");
    assert_eq!(written.scope, scope());

    let read = store.read("build").await.unwrap().unwrap();
    assert_eq!(read.content, "cargo build");
    assert_eq!(read.description, "How to build");

    let edited = store.edit("build", "cargo build --release").await.unwrap();
    assert_eq!(edited.content, "cargo build --release");
    assert_eq!(
        store.read("build").await.unwrap().unwrap().content,
        "cargo build --release"
    );

    assert!(store.delete("build").await.unwrap());
    assert!(!store.delete("build").await.unwrap());
    assert!(store.read("build").await.unwrap().is_none());
    assert_eq!(store.list().await.unwrap(), Vec::<CodingFact>::new());
}

#[tokio::test]
async fn edit_and_delete_missing_facts_are_typed() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let error = store.edit("ghost", "body").await.unwrap_err();
    assert!(matches!(error, MemoryError::NotFound(_)), "{error:?}");
    assert!(!store.delete("ghost").await.unwrap());
    let error = store.edit("ghost", "   ").await.unwrap_err();
    assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
}

#[tokio::test]
async fn search_finds_a_fact_by_name_body_description_and_tag() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let _ = store
        .write(
            fact("release-checklist", "run the workspace test suite")
                .with_description("how to ship")
                .with_tags(vec!["ship".to_owned()]),
        )
        .await
        .unwrap();
    let _ = store
        .write(fact("unrelated", "nothing to see"))
        .await
        .unwrap();

    for query in ["release-checklist", "workspace", "ship"] {
        let hits = store.search(query, &keyword_options()).await.unwrap();
        assert!(!hits.is_empty(), "no hit for {query}");
        assert_eq!(
            hits[0]
                .record
                .metadata
                .get("name")
                .and_then(serde_json::Value::as_str),
            Some("release-checklist"),
            "query {query}"
        );
    }
    let hits = store.search("release", &keyword_options()).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].score > 0.0 && hits[0].score <= 1.0);
    assert_eq!(
        store
            .search("absent-term", &keyword_options())
            .await
            .unwrap(),
        Vec::<RecallHit>::new()
    );
}

#[tokio::test]
async fn search_rejects_blank_queries_and_semantic_mode() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let error = store.search("   ", &keyword_options()).await.unwrap_err();
    assert!(matches!(error, MemoryError::Validation(_)));
    let error = store
        .search(
            "anything",
            &SearchOptions::default().with_mode(SearchMode::Semantic),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
    let hits = store
        .search(
            "anything",
            &keyword_options().with_kinds(vec![MemoryType::Summary]),
        )
        .await
        .unwrap();
    assert_eq!(hits, Vec::<RecallHit>::new());
}

#[tokio::test]
async fn the_summary_index_uses_the_documented_shape() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let _ = store
        .write(
            fact("build", "cargo build")
                .with_description("How to build")
                .with_path("docs/BUILD.md"),
        )
        .await
        .unwrap();
    let index = store.render_index().await.unwrap();
    assert!(index.starts_with("# Memory\n"), "{index}");
    assert_eq!(
        index.lines().nth(1),
        Some("- [Project] build — How to build (`docs/BUILD.md`)")
    );
}

#[tokio::test]
async fn a_second_instance_sees_committed_facts() {
    let temp = TempDir::new().unwrap();
    let first = store_at(&temp).await;
    let second = store_at(&temp).await;
    let _ = first.write(fact("build", "cargo build")).await.unwrap();

    let seen = second.read("build").await.unwrap().unwrap();
    assert_eq!(seen.content, "cargo build");
    assert_eq!(second.reload_count(), 1);

    let _ = first.write(fact("test", "cargo test")).await.unwrap();
    assert_eq!(second.list().await.unwrap().len(), 2);
    assert_eq!(second.reload_count(), 2);
}

#[tokio::test]
async fn external_markdown_edits_are_reloaded_and_reindexed() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let _ = store.write(fact("build", "cargo build")).await.unwrap();

    let mut document =
        MarkdownMemory::parse(&std::fs::read_to_string(store.memory_path()).unwrap())
            .unwrap()
            .with_scope(scope());
    let _ = document.upsert_fact(&fact("hand-written", "edited outside the store"));
    std::fs::write(store.memory_path(), document.render()).unwrap();

    let read = store.read("hand-written").await.unwrap().unwrap();
    assert_eq!(read.content, "edited outside the store");
    #[cfg(feature = "sqlite")]
    assert_eq!(store.indexed_count().await.unwrap(), 2);
    assert_eq!(
        store
            .search("edited", &keyword_options())
            .await
            .unwrap()
            .len(),
        1
    );

    // A write after the external edit must keep the external fact.
    let _ = store.write(fact("test", "cargo test")).await.unwrap();
    let text = std::fs::read_to_string(store.memory_path()).unwrap();
    assert!(text.contains("hand-written"), "{text}");
    assert_eq!(store.list().await.unwrap().len(), 3);
}

#[tokio::test]
async fn reindex_rebuilds_the_derived_index_from_the_file() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let reindexed = store.reindex().await.unwrap();
    assert_eq!(reindexed, 0);
    assert_eq!(store.reload_count(), 1);

    let mut document = MarkdownMemory::new().with_scope(scope());
    let _ = document.upsert_fact(&fact("only-in-file", "written by an editor"));
    std::fs::write(store.memory_path(), document.render()).unwrap();
    assert_eq!(store.reindex().await.unwrap(), 1);
    assert_eq!(store.reload_count(), 2);
    #[cfg(feature = "sqlite")]
    assert_eq!(store.indexed_count().await.unwrap(), 1);
    assert_eq!(store.list().await.unwrap().len(), 1);
}

#[tokio::test]
async fn the_markdown_document_round_trips_through_the_store() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let _ = store
        .write(fact("build", "cargo build").with_type(CodingFactType::Project))
        .await
        .unwrap();
    let _ = store
        .write(
            fact("tone", "be concise")
                .with_type(CodingFactType::Feedback)
                .with_tags(vec!["style".to_owned()]),
        )
        .await
        .unwrap();
    let snapshot = store.markdown().await.unwrap();
    let text = std::fs::read_to_string(store.memory_path()).unwrap();
    let parsed = MarkdownMemory::parse(&text).unwrap().with_scope(scope());
    assert_eq!(parsed.render(), text);
    assert_eq!(parsed.facts().len(), 2);
    assert_eq!(snapshot.facts().len(), 2);
    assert!(text.contains("## Feedback"), "{text}");
}

#[tokio::test]
async fn concurrent_writers_are_serialized() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let mut tasks = Vec::new();
    for index in 0..8 {
        let handle = store.clone();
        tasks.push(tokio::spawn(async move {
            handle
                .write(fact(&format!("fact-{index}"), &format!("body {index}")))
                .await
                .unwrap();
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    let facts = store.list().await.unwrap();
    assert_eq!(facts.len(), 8);
    let text = std::fs::read_to_string(store.memory_path()).unwrap();
    assert_eq!(text.matches("### fact-").count(), 8, "{text}");
}

#[cfg(unix)]
#[tokio::test]
async fn watching_picks_up_external_edits() {
    let temp = TempDir::new().unwrap();
    let store = store_at(&temp).await;
    let _ = store.write(fact("build", "cargo build")).await.unwrap();
    store.start_watching().await.unwrap();
    assert!(store.is_watching().await);
    let before = store.reload_count();

    let mut document =
        MarkdownMemory::parse(&std::fs::read_to_string(store.memory_path()).unwrap())
            .unwrap()
            .with_scope(scope());
    let _ = document.upsert_fact(&fact("from-editor", "written while watching"));
    std::fs::write(store.memory_path(), document.render()).unwrap();

    let mut observed = false;
    for _ in 0..100 {
        if store.reload_count() > before {
            observed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(observed, "the watcher never reloaded the document");
    assert_eq!(
        store.read("from-editor").await.unwrap().unwrap().content,
        "written while watching"
    );

    store.stop_watching().await;
    assert!(!store.is_watching().await);
    let after = store.reload_count();
    let _ = document.upsert_fact(&fact("after-stop", "ignored"));
    std::fs::write(store.memory_path(), document.render()).unwrap();
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(store.reload_count(), after);
}

#[test]
fn the_store_handle_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LocalCodingMemory>();
}
