// `missing_docs` is allowed before the `cfg` gate: when the `sqlite` feature is off the
// whole crate (including `//!` docs) is stripped, and a later `allow` would be stripped
// with it, leaving the denied `missing_docs` lint to fire on the empty crate root.
#![allow(missing_docs)]
#![cfg(feature = "sqlite")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage of the SQLite-backed stores.
//!
//! [`MemoryStores::sqlite`] keeps records, variables, scope configs, and message
//! history in one SQLite database while vectors live in a local-file store under
//! `<path>.vectors` and graph entities under `<path>.graph`. This suite proves the
//! committed state is observable after the facade and the stores are dropped and
//! reopened on the same path, and that two databases stay independent.
//!
//! Every test derives its database path from its own [`tempfile::TempDir`], which is
//! kept alive for the whole test, and pins record timestamps so nothing depends on the
//! wall clock.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_memory::config::{AgentMemoryConfig, MemoryEngineConfig, MemoryScopeConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::graph::GraphAddOptions;
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, NewMemory, SearchMode, SearchOptions,
};
use openjiuwen_memory::store::{keys, MemoryStores};
use openjiuwen_retrieval::embedding::HashEmbedding;
use serde_json::json;

/// The content of the record written by the persistence tests.
const RECORD_TEXT: &str = "Alice prefers dark mode";
/// The content of the conversation message written by the persistence tests.
const MESSAGE_TEXT: &str = "remember the release checklist";

/// The engine configuration shared by every test.
fn config() -> MemoryEngineConfig {
    MemoryEngineConfig::default()
}

/// A fixed creation time so every assertion is wall-clock independent.
fn fixed_timestamp() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("fixed timestamp")
}

/// The scope shared by the record and the message; it carries a session so the session
/// variable lives in the session namespace and the history is session-filtered.
fn session_scope() -> MemoryScope {
    MemoryScope::new("alice", "ws").session("s1")
}

/// A session-free scope used for the user-scoped variable.
fn user_scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// The per-scope configuration written by the persistence test.
fn scope_config() -> MemoryScopeConfig {
    MemoryScopeConfig {
        user_profile_definition: Some("stable identity facts".to_owned()),
        semantic_memory_definition: Some("durable project facts".to_owned()),
        episodic_memory_definition: None,
    }
}

/// An agent config that only appends history, writing no extracted memories.
fn history_only() -> AgentMemoryConfig {
    AgentMemoryConfig {
        enable_episodic_memory: false,
        enable_summary_memory: false,
        ..Default::default()
    }
}

/// Adds one record with dedupe disabled and returns the engine-minted id.
async fn add_record(ltm: &LongTermMemory, scope: &MemoryScope, memory: NewMemory) -> String {
    let options = AddOptions::default().with_dedupe(openjiuwen_memory::record::DedupePolicy::Add);
    let outcome = ltm.add(scope, memory, &options).await.expect("add record");
    assert_eq!(
        outcome.added.len(),
        1,
        "the write must add exactly one record: {outcome:?}"
    );
    outcome.added[0].id.clone()
}

/// The plain text of every message, in order.
fn texts(messages: &[BaseMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| message.text().into_owned())
        .collect()
}

/// Asserts the record, variables, scope config, and message history of the fixture are
/// observable through `ltm`.
async fn assert_snapshot(
    ltm: &LongTermMemory,
    session: &MemoryScope,
    user: &MemoryScope,
    id: &str,
) {
    let record = ltm
        .get(session, id)
        .await
        .expect("get record")
        .expect("the record must survive");
    assert_eq!(record.id, id);
    assert_eq!(record.content, RECORD_TEXT);
    assert_eq!(record.metadata.get("theme"), Some(&json!("dark")));
    assert_eq!(record.timestamp, fixed_timestamp());

    let listed = ltm.list(session, &[]).await.expect("list records");
    assert!(
        listed.iter().any(|item| item.id == id),
        "the listing must contain the record: {listed:?}"
    );

    let session_variables = ltm
        .get_variables(session, None)
        .await
        .expect("session variables");
    assert_eq!(
        session_variables.get("theme").map(String::as_str),
        Some("dark")
    );
    let user_variables = ltm.get_variables(user, None).await.expect("user variables");
    assert_eq!(
        user_variables.get("timezone").map(String::as_str),
        Some("UTC")
    );

    assert_eq!(
        ltm.get_scope_config("ws").await.expect("scope config"),
        Some(scope_config())
    );

    assert_eq!(
        texts(
            &ltm.get_recent_messages(session, 10)
                .await
                .expect("messages")
        ),
        [MESSAGE_TEXT.to_owned()]
    );
}

#[tokio::test]
async fn sqlite_persists_records_variables_configs_and_messages_across_reopen() {
    // `TempDir` deletes its directory on drop, so keep the binding alive for the whole
    // test and only derive the database path from it.
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let session = session_scope();
    let user = user_scope();

    let ltm = LongTermMemory::sqlite(config(), &path)
        .await
        .expect("open sqlite facade");
    let id = add_record(
        &ltm,
        &session,
        NewMemory::semantic(RECORD_TEXT)
            .with_metadata("theme", json!("dark"))
            .with_timestamp(fixed_timestamp()),
    )
    .await;
    ltm.set_variable(&session, "theme", "dark")
        .await
        .expect("session variable");
    ltm.set_variable(&user, "timezone", "UTC")
        .await
        .expect("user variable");
    assert!(
        !ltm.set_scope_config("ws", scope_config())
            .await
            .expect("write scope config"),
        "the first scope-config write replaces nothing"
    );
    ltm.add_messages(
        &session,
        &[BaseMessage::user(MESSAGE_TEXT)],
        &history_only(),
        &AddOptions::default(),
    )
    .await
    .expect("append message");

    assert_snapshot(&ltm, &session, &user, &id).await;
    drop(ltm);

    let reopened = LongTermMemory::sqlite(config(), &path)
        .await
        .expect("reopen sqlite facade");
    assert_snapshot(&reopened, &session, &user, &id).await;
}

#[tokio::test]
async fn delete_and_clear_remain_visible_after_reopen() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let scope = user_scope();

    let ltm = LongTermMemory::sqlite(config(), &path)
        .await
        .expect("open sqlite facade");
    let deleted = add_record(&ltm, &scope, NewMemory::semantic("record to delete")).await;
    let kept = add_record(&ltm, &scope, NewMemory::semantic("record to keep")).await;
    assert!(ltm.delete(&scope, &deleted).await.expect("delete record"));
    drop(ltm);

    let reopened = LongTermMemory::sqlite(config(), &path)
        .await
        .expect("reopen sqlite facade");
    assert!(
        reopened
            .get(&scope, &deleted)
            .await
            .expect("get deleted")
            .is_none(),
        "the delete must survive the reopen"
    );
    assert!(
        reopened
            .get(&scope, &kept)
            .await
            .expect("get kept")
            .is_some(),
        "the untouched record must survive the reopen"
    );

    let cleared = reopened
        .clear(&MemoryFilter::for_scope(&scope), &CallOptions::default())
        .await
        .expect("clear records");
    assert_eq!(cleared, 1, "only the surviving record is cleared");
    drop(reopened);

    let reopened_after_clear = LongTermMemory::sqlite(config(), &path)
        .await
        .expect("reopen after clear");
    assert!(
        reopened_after_clear
            .list(&scope, &[])
            .await
            .expect("list")
            .is_empty(),
        "the clear must survive the reopen"
    );
}

#[tokio::test]
async fn reopened_sqlite_stores_preserve_committed_state() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let scope = user_scope();

    // The write goes through the facade; the store handle is cloned so the record key
    // can be asserted directly at the KV layer before the stores are dropped.
    let id = {
        let stores = MemoryStores::sqlite(&path).await.expect("open stores");
        let ltm = LongTermMemory::with_stores(config(), stores.clone())
            .await
            .expect("facade over stores");
        let id = add_record(
            &ltm,
            &scope,
            NewMemory::semantic("written through the store layer"),
        )
        .await;
        ltm.set_variable(&scope, "theme", "dark")
            .await
            .expect("user variable");
        ltm.add_messages(
            &scope,
            &[BaseMessage::user(MESSAGE_TEXT)],
            &history_only(),
            &AddOptions::default(),
        )
        .await
        .expect("append message");
        let value = stores
            .kv
            .get(&keys::record_key(&scope, &id))
            .await
            .expect("read record key")
            .expect("the record key exists in the KV store");
        assert!(
            value.as_str().expect("record is text").contains(&id),
            "the KV value carries the record JSON"
        );
        id
    };

    let reopened = MemoryStores::sqlite(&path).await.expect("reopen stores");
    let ltm = LongTermMemory::with_stores(config(), reopened.clone())
        .await
        .expect("facade over reopened stores");
    let record = ltm
        .get(&scope, &id)
        .await
        .expect("get record")
        .expect("the record survives in the reopened stores");
    assert_eq!(record.content, "written through the store layer");
    assert_eq!(
        ltm.get_variables(&scope, None)
            .await
            .expect("variables")
            .get("theme")
            .map(String::as_str),
        Some("dark")
    );
    assert_eq!(
        texts(&ltm.get_recent_messages(&scope, 10).await.expect("messages")),
        [MESSAGE_TEXT.to_owned()]
    );
    assert!(
        reopened
            .kv
            .get(&keys::record_key(&scope, &id))
            .await
            .expect("read record key")
            .is_some(),
        "the reopened KV store sees the committed record"
    );
}

#[tokio::test]
async fn embeddings_and_graph_entities_persist_across_reopen() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("memory.db");
    let scope = user_scope();

    let mut settings = config();
    settings.graph.enabled = true;
    let embedding = Arc::new(HashEmbedding::new(16));

    let id = {
        let stores = MemoryStores::sqlite(&path).await.expect("open stores");
        let ltm =
            LongTermMemory::builder(settings.clone())
                .stores(stores)
                .embedding(
                    Arc::clone(&embedding) as Arc<dyn openjiuwen_retrieval::embedding::Embedding>
                )
                .build()
                .expect("build facade");
        let id = add_record(
            &ltm,
            &scope,
            NewMemory::semantic("tokio async runtime task scheduling"),
        )
        .await;
        let graph = ltm.graph().expect("graph memory is enabled");
        let extraction = graph
            .add(
                &scope,
                "I met Bob in New York; Bob uses Rust.",
                &GraphAddOptions::default(),
            )
            .await
            .expect("add graph memory");
        assert!(
            !extraction.entities.is_empty(),
            "heuristic extraction must find entities"
        );
        id
    };

    // Vectors live under `<path>.vectors`; the local-file backend rewrites a JSON
    // snapshot per mutation, so a reopened facade sees the committed embeddings.
    let reopened = MemoryStores::sqlite(&path).await.expect("reopen stores");
    let ltm = LongTermMemory::builder(settings)
        .stores(reopened)
        .embedding(Arc::clone(&embedding) as Arc<dyn openjiuwen_retrieval::embedding::Embedding>)
        .build()
        .expect("reopen facade");

    let options = SearchOptions::default()
        .with_mode(SearchMode::Semantic)
        .with_threshold(0.0)
        .with_top_k(5);
    let hits = ltm
        .search(&scope, "tokio async runtime", &options)
        .await
        .expect("semantic search");
    assert_eq!(
        hits.len(),
        1,
        "the persisted vector must be found: {hits:?}"
    );
    assert_eq!(hits[0].record.id, id);
    assert!(
        hits[0].similarity > 0.0,
        "semantic recall must use the persisted embedding: {hits:?}"
    );

    let graph = ltm.graph().expect("graph memory is enabled");
    assert!(
        graph
            .entity_named(&scope, "Rust")
            .await
            .expect("lookup entity")
            .is_some(),
        "the graph entity must survive the reopen"
    );
}

#[tokio::test]
async fn independent_databases_do_not_share_state() {
    let first_directory = tempfile::tempdir().expect("first tempdir");
    let second_directory = tempfile::tempdir().expect("second tempdir");
    let first_path = first_directory.path().join("memory.db");
    let second_path = second_directory.path().join("memory.db");
    let scope = user_scope();

    let first = LongTermMemory::sqlite(config(), &first_path)
        .await
        .expect("first facade");
    let second = LongTermMemory::sqlite(config(), &second_path)
        .await
        .expect("second facade");

    add_record(
        &first,
        &scope,
        NewMemory::semantic("only in the first database"),
    )
    .await;
    first
        .set_variable(&scope, "theme", "first")
        .await
        .expect("first variable");
    assert_eq!(first.list(&scope, &[]).await.expect("first list").len(), 1);
    assert!(
        second
            .list(&scope, &[])
            .await
            .expect("second list")
            .is_empty(),
        "a separate database must not see the first record"
    );
    assert!(
        second
            .get_variables(&scope, None)
            .await
            .expect("second variables")
            .is_empty(),
        "a separate database must not see the first variable"
    );

    add_record(
        &second,
        &scope,
        NewMemory::semantic("only in the second database"),
    )
    .await;
    assert_eq!(
        second.list(&scope, &[]).await.expect("second list").len(),
        1
    );
    assert_eq!(
        first.list(&scope, &[]).await.expect("first list").len(),
        1,
        "the second write must not leak into the first database"
    );
}
