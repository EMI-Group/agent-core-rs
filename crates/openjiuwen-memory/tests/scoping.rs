//! Scoping integration tests of the [`openjiuwen_memory::facade::LongTermMemory`] surface.
//!
//! Ports the scoping coverage of the Python `tests/unit_tests/core/memory` suites: the
//! user / scope / session / agent isolation of records, the user and session variable
//! namespaces, the per-scope configuration, the conversation history, and the
//! Python-parity aliases (`search_user_mem`, `search_user_history_summary`,
//! `update_mem_by_id`, `delete_mem_by_*`, `user_mem_total_num`,
//! `get_user_mem_by_page`).
//!
//! Every test builds its own in-memory facade and seeds explicit timestamps, so nothing
//! depends on wall-clock values or on the other tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_memory::config::{AgentMemoryConfig, MemoryEngineConfig, MemoryScopeConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{
    AddOptions, CallOptions, DedupePolicy, MemoryFilter, MemoryRecord, MemoryScope, MemoryType,
    NewMemory,
};

/// The content every scope of the isolation tests receives.
const SHARED_CONTENT: &str = "the same shared note repeated in every scope";

/// The ids one seeded record per scope produced.
struct Seeded {
    workspace: String,
    other_scope: String,
    bob: String,
    first_session: String,
    second_session: String,
    agent: String,
}

/// A fixed creation time, offset by `offset` seconds.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed timestamp")
}

/// Builds the default in-memory facade (no embedding model, no LLM).
async fn facade() -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("in-memory facade")
}

/// Adds one record with dedupe disabled and returns its engine-minted id.
async fn add_record(ltm: &LongTermMemory, scope: &MemoryScope, memory: NewMemory) -> String {
    let options = AddOptions::default().with_dedupe(DedupePolicy::Add);
    let outcome = ltm.add(scope, memory, &options).await.expect("add record");
    assert_eq!(
        outcome.added.len(),
        1,
        "expected one new record: {outcome:?}"
    );
    outcome.added[0].id.clone()
}

/// Appends one message to the scope's history without writing memory records.
async fn append_message(ltm: &LongTermMemory, scope: &MemoryScope, message: BaseMessage) {
    let agent = AgentMemoryConfig {
        enable_episodic_memory: false,
        enable_summary_memory: false,
        ..Default::default()
    };
    let messages = std::slice::from_ref(&message);
    let result = ltm
        .add_messages(scope, messages, &agent, &AddOptions::default())
        .await;
    let result = result.expect("append message");
    assert!(
        result.is_empty(),
        "the history-only config wrote: {result:?}"
    );
}

/// Seeds [`SHARED_CONTENT`] under alice/ws, alice/other, bob/ws, two sessions, and an
/// agent scope.
async fn seed_shared_content(ltm: &LongTermMemory) -> Seeded {
    let semantic = || NewMemory::semantic(SHARED_CONTENT);
    let workspace = MemoryScope::new("alice", "ws");
    Seeded {
        workspace: add_record(ltm, &workspace, semantic()).await,
        other_scope: add_record(ltm, &MemoryScope::new("alice", "other"), semantic()).await,
        bob: add_record(ltm, &MemoryScope::new("bob", "ws"), semantic()).await,
        first_session: add_record(ltm, &workspace.clone().session("s1"), semantic()).await,
        second_session: add_record(ltm, &workspace.clone().session("s2"), semantic()).await,
        agent: add_record(ltm, &workspace.agent("a1"), semantic()).await,
    }
}

/// The ids of `records`, in their returned order.
fn ids(records: &[MemoryRecord]) -> Vec<String> {
    records.iter().map(|record| record.id.clone()).collect()
}

/// The text of `messages`, in their returned order.
fn texts(messages: &[BaseMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| message.text().into_owned())
        .collect()
}

/// Asserts that listing `scope` yields exactly `expected` (order-insensitive).
async fn assert_listing(ltm: &LongTermMemory, scope: &MemoryScope, expected: &[&str]) {
    let mut listed = ids(&ltm.list(scope, &[]).await.expect("list"));
    let mut wanted: Vec<String> = expected.iter().map(|id| (*id).to_owned()).collect();
    listed.sort_unstable();
    wanted.sort_unstable();
    assert_eq!(listed, wanted, "listing of {scope:?}");
}

/// Asserts the record count `scope` reports statistics for.
async fn assert_stats(ltm: &LongTermMemory, scope: &MemoryScope, total: usize) {
    let stats = ltm.stats(scope, &[]).await.expect("stats");
    assert_eq!(stats.total, total, "statistics of {scope:?}");
}

/// Asserts whether `id` is readable in `scope`.
async fn assert_get(ltm: &LongTermMemory, scope: &MemoryScope, id: &str, present: bool) {
    let found = ltm.get(scope, id).await.expect("get").is_some();
    assert_eq!(found, present, "presence of {id} in {scope:?}");
}

/// The `user_mem_total_num` count of one user in one scope.
async fn total(ltm: &LongTermMemory, user_id: &str, scope_id: &str) -> usize {
    ltm.user_mem_total_num(user_id, scope_id)
        .await
        .expect("user_mem_total_num")
}

/// Every variable of `scope`, in `BTreeMap` order.
async fn variables(ltm: &LongTermMemory, scope: &MemoryScope) -> Vec<(String, String)> {
    let variables = ltm.get_variables(scope, None).await.expect("variables");
    variables.into_iter().collect()
}

/// The names of every variable of `scope`, in sorted order.
async fn variable_names(ltm: &LongTermMemory, scope: &MemoryScope) -> Vec<String> {
    variables(ltm, scope)
        .await
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

#[tokio::test]
async fn records_are_isolated_across_user_scope_session_and_agent() {
    // `NewMemory` carries no id (the engine mints one), so every scope receives the
    // same content; `DedupePolicy::Add` keeps the writes from being merged.
    let ltm = facade().await;
    let workspace = MemoryScope::new("alice", "ws");
    let other_scope = MemoryScope::new("alice", "other");
    let bob = MemoryScope::new("bob", "ws");
    let first_session = workspace.clone().session("s1");
    let second_session = workspace.clone().session("s2");
    let agent_scope = workspace.clone().agent("a1");
    let seeded = seed_shared_content(&ltm).await;

    let minted = [
        &seeded.workspace,
        &seeded.other_scope,
        &seeded.bob,
        &seeded.first_session,
        &seeded.second_session,
        &seeded.agent,
    ];
    let unique: HashSet<&String> = minted.iter().copied().collect();
    assert_eq!(unique.len(), minted.len(), "every write mints its own id");

    // A narrow scope only lists its own records.
    assert_listing(&ltm, &other_scope, &[&seeded.other_scope]).await;
    assert_listing(&ltm, &bob, &[&seeded.bob]).await;
    assert_listing(&ltm, &first_session, &[&seeded.first_session]).await;
    assert_listing(&ltm, &second_session, &[&seeded.second_session]).await;
    assert_listing(&ltm, &agent_scope, &[&seeded.agent]).await;
    // `MemoryScope::matches`: an unbound scope selects every session and agent of the
    // same user and scope, so the session-free listing sees all four of them.
    let workspace_records: [&str; 4] = [
        &seeded.workspace,
        &seeded.first_session,
        &seeded.second_session,
        &seeded.agent,
    ];
    assert_listing(&ltm, &workspace, &workspace_records).await;

    for scope in [
        &first_session,
        &second_session,
        &agent_scope,
        &other_scope,
        &bob,
    ] {
        assert_stats(&ltm, scope, 1).await;
    }
    assert_stats(&ltm, &workspace, 4).await;

    // A point read addresses `UMD/{user_id}/{scope_id}/{id}`, so only the user/scope
    // dimensions isolate `get`; session and agent isolation is a listing contract.
    assert_get(&ltm, &workspace, &seeded.workspace, true).await;
    assert_get(&ltm, &first_session, &seeded.first_session, true).await;
    assert_get(&ltm, &other_scope, &seeded.workspace, false).await;
    assert_get(&ltm, &bob, &seeded.workspace, false).await;
    assert_get(&ltm, &workspace, "000000000000000000000000", false).await;
}

#[tokio::test]
async fn clearing_one_scope_leaves_the_others_intact() {
    let ltm = facade().await;
    let workspace = MemoryScope::new("alice", "ws");
    let first_session = workspace.clone().session("s1");
    let second_session = workspace.clone().session("s2");
    let agent_scope = workspace.clone().agent("a1");
    let other_scope = MemoryScope::new("alice", "other");
    let bob = MemoryScope::new("bob", "ws");
    let _seeded = seed_shared_content(&ltm).await;
    let options = CallOptions::default();

    for scope in [&first_session, &second_session, &agent_scope] {
        let filter = MemoryFilter::for_scope(scope);
        assert_eq!(ltm.clear(&filter, &options).await.expect("clear"), 1);
        assert_eq!(ltm.list(scope, &[]).await.expect("list").len(), 0);
    }
    assert_stats(&ltm, &workspace, 1).await;

    // Other users and sibling scopes are never touched.
    for scope in [&other_scope, &bob] {
        let filter = MemoryFilter::for_scope(scope);
        assert_eq!(ltm.clear(&filter, &options).await.expect("clear"), 1);
    }
    assert_stats(&ltm, &workspace, 1).await;
    let filter = MemoryFilter::for_scope(&workspace);
    assert_eq!(ltm.clear(&filter, &options).await.expect("clear"), 1);
    assert_stats(&ltm, &workspace, 0).await;
}

#[tokio::test]
async fn variables_are_scoped_sorted_and_validated() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    for (name, value) in [("zeta", "last"), ("alpha", "first"), ("mike", "middle")] {
        ltm.set_variable(&scope, name, value)
            .await
            .expect("set variable");
    }

    // `get_variables(None)` returns every variable in `BTreeMap` (sorted) order.
    assert_eq!(
        variable_names(&ltm, &scope).await,
        ["alpha", "mike", "zeta"]
    );
    let all = ltm
        .get_variables(&scope, None)
        .await
        .expect("all variables");
    assert_eq!(all.get("alpha").map(String::as_str), Some("first"));

    // Named reads return only the names that exist.
    let requested = ["mike".to_owned(), "missing".to_owned()];
    let named = ltm
        .get_variables(&scope, Some(&requested))
        .await
        .expect("named");
    assert_eq!(named.keys().cloned().collect::<Vec<_>>(), ["mike"]);
    assert_eq!(named.get("mike").map(String::as_str), Some("middle"));

    // Session variables live in their own namespace, even with the same name.
    let first_session = scope.clone().session("s1");
    ltm.set_variable(&first_session, "zeta", "session last")
        .await
        .expect("session variable");
    let user_variables = variables(&ltm, &scope).await;
    assert_eq!(user_variables.len(), 3);
    assert!(user_variables.contains(&("zeta".to_owned(), "last".to_owned())));
    let session_variables = variables(&ltm, &first_session).await;
    assert_eq!(
        session_variables,
        [("zeta".to_owned(), "session last".to_owned())]
    );
    assert_eq!(variables(&ltm, &scope.clone().session("s2")).await.len(), 0);

    // Deletes report the removed count and never cross the user/session boundary.
    let doomed = ["alpha".to_owned(), "missing".to_owned()];
    assert_eq!(
        ltm.delete_variables(&scope, &doomed).await.expect("delete"),
        1
    );
    assert_eq!(variable_names(&ltm, &scope).await, ["mike", "zeta"]);
    assert_eq!(variables(&ltm, &first_session).await.len(), 1);
    assert_eq!(ltm.delete_variables(&scope, &[]).await.expect("none"), 0);

    // Invalid names are rejected before anything is written.
    for name in ["", "   ", "a/b", "nul\0byte"] {
        let error = ltm
            .set_variable(&scope, name, "value")
            .await
            .expect_err("bad name");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }
    let requested = ["a/b".to_owned()];
    let error = ltm
        .get_variables(&scope, Some(&requested))
        .await
        .expect_err("bad read");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(variable_names(&ltm, &scope).await, ["mike", "zeta"]);
}

#[tokio::test]
async fn scope_configs_round_trip_and_are_deleted() {
    let ltm = facade().await;
    let config = MemoryScopeConfig {
        user_profile_definition: Some("stable identity facts".to_owned()),
        semantic_memory_definition: Some("durable project facts".to_owned()),
        episodic_memory_definition: Some("dated events".to_owned()),
    };

    let created = ltm
        .set_scope_config("ws", config.clone())
        .await
        .expect("create");
    assert!(!created);
    let stored = ltm.get_scope_config("ws").await.expect("get config");
    assert_eq!(stored, Some(config.clone()));
    let replaced = ltm
        .set_scope_config("ws", config.clone())
        .await
        .expect("replace");
    assert!(replaced, "a second write must report the replacement");
    let stored = ltm.get_scope_config("ws").await.expect("replaced config");
    assert_eq!(stored, Some(config));
    assert_eq!(ltm.get_scope_config("other").await.expect("absent"), None);

    assert!(ltm.delete_scope_config("ws").await.expect("delete config"));
    assert!(!ltm.delete_scope_config("ws").await.expect("delete again"));
    assert_eq!(ltm.get_scope_config("ws").await.expect("deleted"), None);

    for scope_id in ["bad/scope", ""] {
        let default_config = MemoryScopeConfig::default();
        let set = ltm.set_scope_config(scope_id, default_config).await;
        assert!(matches!(set, Err(Error::Validation(_))), "{set:?}");
        let get = ltm.get_scope_config(scope_id).await;
        assert!(matches!(get, Err(Error::Validation(_))), "{get:?}");
        let delete = ltm.delete_scope_config(scope_id).await;
        assert!(matches!(delete, Err(Error::Validation(_))), "{delete:?}");
    }
}

#[tokio::test]
async fn message_history_respects_the_session_filter() {
    let ltm = facade().await;
    let workspace = MemoryScope::new("alice", "ws");
    let first_session = workspace.clone().session("s1");
    let second_session = workspace.clone().session("s2");

    let first_turns = [
        BaseMessage::system("be careful"),
        BaseMessage::user("first turn"),
        BaseMessage::assistant("first reply"),
    ];
    for message in first_turns {
        append_message(&ltm, &first_session, message).await;
    }
    let second_turns = [BaseMessage::user("other session turn")];
    for message in second_turns {
        append_message(&ltm, &second_session, message).await;
    }

    // History is recorded even when long-term memory is switched off for the agent.
    let disabled = AgentMemoryConfig {
        enable_long_term_mem: false,
        ..Default::default()
    };
    let messages = [BaseMessage::user("free turn")];
    let options = AddOptions::default();
    let result = ltm
        .add_messages(&workspace, &messages, &disabled, &options)
        .await;
    let result = result.expect("add messages");
    assert!(result.is_empty(), "extraction is disabled: {result:?}");
    assert_eq!(ltm.list(&workspace, &[]).await.expect("records").len(), 0);

    let first_history = ltm
        .get_recent_messages(&first_session, 10)
        .await
        .expect("first");
    let expected = ["be careful", "first turn", "first reply"];
    assert_eq!(
        texts(&first_history),
        expected,
        "a session-bound read is chronological and only sees its own session"
    );
    assert!(first_history[1].role.is_user());
    assert!(first_history[2].role.is_assistant());
    let second_history = ltm
        .get_recent_messages(&second_session, 10)
        .await
        .expect("second");
    assert_eq!(texts(&second_history), ["other session turn"]);

    // A session-free scope reads every session's history (documented store semantics).
    let all_history = ltm
        .get_recent_messages(&workspace, 10)
        .await
        .expect("workspace");
    assert_eq!(
        texts(&all_history),
        [
            "be careful",
            "first turn",
            "first reply",
            "other session turn",
            "free turn"
        ]
    );
}

#[tokio::test]
async fn message_history_clamps_to_the_configured_limit() {
    let config = MemoryEngineConfig {
        message_history_limit: 2,
        ..Default::default()
    };
    let ltm = LongTermMemory::in_memory(config).await.expect("facade");
    let scope = MemoryScope::new("alice", "ws");
    for text in ["oldest turn", "middle turn", "newest turn"] {
        append_message(&ltm, &scope, BaseMessage::user(text)).await;
    }

    // `n` above `message_history_limit` is clamped to the newest turns, oldest first.
    let clamped = ltm.get_recent_messages(&scope, 10).await.expect("clamped");
    assert_eq!(texts(&clamped), ["middle turn", "newest turn"]);
    let only_one = ltm.get_recent_messages(&scope, 1).await.expect("one");
    assert_eq!(texts(&only_one), ["newest turn"]);
    assert_eq!(
        ltm.get_recent_messages(&scope, 0)
            .await
            .expect("zero")
            .len(),
        0
    );
}

#[tokio::test]
async fn search_user_mem_alias_is_limited_to_its_user_and_scope() {
    let ltm = facade().await;
    let alice = MemoryScope::new("alice", "ws");
    let bob = MemoryScope::new("bob", "ws");
    let alice_other = MemoryScope::new("alice", "other");

    let note = |who: &str| NewMemory::semantic(format!("{who} keeps the pineapple notes"));
    let alice_id = add_record(&ltm, &alice, note("alice")).await;
    add_record(&ltm, &bob, note("bob")).await;
    add_record(&ltm, &alice_other, note("another scope")).await;

    let hits = ltm
        .search_user_mem("pineapple", 5, "alice", "ws", 0.0)
        .await;
    let hits = hits.expect("search_user_mem");
    assert_eq!(hits.len(), 1, "only the named user/scope is searched");
    assert_eq!(hits[0].record.id, alice_id);
    assert!(hits
        .iter()
        .all(|hit| hit.record.scope.user_id == "alice" && hit.record.scope.scope_id == "ws"));

    let limited = ltm.search_user_mem("pineapple", 1, "alice", "ws", 0.0);
    assert_eq!(limited.await.expect("limited").len(), 1);
    let bob_hits = ltm.search_user_mem("pineapple", 5, "bob", "ws", 0.0).await;
    let bob_hits = bob_hits.expect("bob search_user_mem");
    assert_eq!(bob_hits.len(), 1);
    assert!(bob_hits.iter().all(|hit| hit.record.scope.user_id == "bob"));
    let unknown = ltm.search_user_mem("pineapple", 5, "carol", "ws", 0.0);
    assert_eq!(unknown.await.expect("unknown user").len(), 0);
}

#[tokio::test]
async fn search_user_history_summary_alias_only_returns_summaries() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    let text = "the pipeline moved to nightly";
    let summary_id = add_record(&ltm, &scope, NewMemory::summary(text)).await;
    add_record(&ltm, &scope, NewMemory::semantic(text)).await;
    add_record(
        &ltm,
        &MemoryScope::new("bob", "ws"),
        NewMemory::summary(text),
    )
    .await;

    let hits = ltm.search_user_history_summary("pipeline", 5, "alice", "ws", 0.0);
    let hits = hits.await.expect("search_user_history_summary");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, summary_id);
    assert!(hits
        .iter()
        .all(|hit| hit.record.kind == MemoryType::Summary));
    assert!(hits
        .iter()
        .all(|hit| hit.record.scope.user_id == "alice" && hit.record.scope.scope_id == "ws"));
    let empty = ltm.search_user_history_summary("pipeline", 5, "bob", "other", 0.0);
    assert_eq!(empty.await.expect("empty scope").len(), 0);
}

#[tokio::test]
async fn update_mem_by_id_alias_replaces_content_and_reports_not_found() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    let memory = NewMemory::semantic("alice prefers dark mode");
    let id = add_record(&ltm, &scope, memory).await;

    let update = ltm.update_mem_by_id(&id, "alice prefers light mode", "alice", "ws");
    update.await.expect("update_mem_by_id");
    let stored = ltm.get(&scope, &id).await.expect("get").expect("present");
    assert_eq!(stored.content, "alice prefers light mode");
    assert_eq!(stored.kind, MemoryType::SemanticMemory);

    let missing = ltm.update_mem_by_id("000000000000000000000000", "new", "alice", "ws");
    let error = missing.await.expect_err("unknown id");
    assert!(matches!(error, Error::NotFound(_)), "{error:?}");
}

#[tokio::test]
async fn delete_aliases_clear_one_id_one_user_or_one_scope() {
    let ltm = facade().await;
    let alice = MemoryScope::new("alice", "ws");
    let bob = MemoryScope::new("bob", "ws");
    let alice_other = MemoryScope::new("alice", "other");
    let carol = MemoryScope::new("carol", "ws");

    let first = add_record(&ltm, &alice, NewMemory::semantic("alice record one")).await;
    add_record(&ltm, &alice, NewMemory::semantic("alice record two")).await;
    add_record(&ltm, &bob, NewMemory::semantic("bob record one")).await;
    add_record(&ltm, &alice_other, NewMemory::semantic("alice elsewhere")).await;

    // `delete_mem_by_id` reports whether the record existed.
    let deleted = ltm.delete_mem_by_id(&first, "alice", "ws").await;
    assert!(deleted.expect("delete_mem_by_id"));
    let deleted = ltm.delete_mem_by_id(&first, "alice", "ws").await;
    assert!(!deleted.expect("delete_mem_by_id again"));
    assert_eq!(total(&ltm, "alice", "ws").await, 1);

    // `delete_mem_by_user_id` clears one user inside one scope.
    add_record(&ltm, &alice, NewMemory::semantic("alice record three")).await;
    let removed = ltm.delete_mem_by_user_id("alice", "ws").await;
    assert_eq!(removed.expect("delete_mem_by_user_id"), 2);
    assert_eq!(total(&ltm, "alice", "ws").await, 0);
    assert_eq!(total(&ltm, "bob", "ws").await, 1);
    assert_eq!(total(&ltm, "alice", "other").await, 1);

    // `delete_mem_by_scope` clears every user inside one scope.
    add_record(&ltm, &alice, NewMemory::semantic("alice record four")).await;
    add_record(&ltm, &bob, NewMemory::semantic("bob record two")).await;
    add_record(&ltm, &carol, NewMemory::semantic("carol record one")).await;
    let removed = ltm.delete_mem_by_scope("ws").await;
    assert_eq!(removed.expect("delete_mem_by_scope"), 4);
    assert_eq!(total(&ltm, "alice", "other").await, 1);
    assert_eq!(total(&ltm, "bob", "ws").await, 0);

    let error = ltm
        .delete_mem_by_scope("bad/scope")
        .await
        .expect_err("bad id");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}

#[tokio::test]
async fn user_mem_total_num_counts_records_but_not_variables() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    ltm.set_variable(&scope, "theme", "dark")
        .await
        .expect("variable");
    let session = scope.clone().session("s1");
    ltm.set_variable(&session, "theme", "light")
        .await
        .expect("session var");
    assert_eq!(total(&ltm, "alice", "ws").await, 0);

    add_record(&ltm, &scope, NewMemory::semantic("alice semantic record")).await;
    add_record(&ltm, &scope, NewMemory::summary("alice summary record")).await;
    assert_eq!(total(&ltm, "alice", "ws").await, 2);
    assert_eq!(total(&ltm, "bob", "ws").await, 0);
    assert_stats(&ltm, &scope, 2).await;
}

#[tokio::test]
async fn get_user_mem_by_page_is_one_based_and_filters_by_kind() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    for offset in 0..5 {
        let memory = NewMemory::semantic(format!("paged semantic {offset}"));
        add_record(&ltm, &scope, memory.with_timestamp(ts(offset))).await;
    }
    for offset in 5..7 {
        let memory = NewMemory::summary(format!("paged summary {offset}"));
        add_record(&ltm, &scope, memory.with_timestamp(ts(offset))).await;
    }
    let newest_first = ids(&ltm.list(&scope, &[]).await.expect("list"));
    assert_eq!(newest_first.len(), 7);

    let mut pages: Vec<Vec<String>> = Vec::new();
    for page_idx in 1..=5 {
        let page = ltm.get_user_mem_by_page("alice", "ws", 2, page_idx, MemoryType::Unknown);
        pages.push(ids(&page.await.expect("page")));
    }
    let sizes: Vec<usize> = pages.iter().map(Vec::len).collect();
    assert_eq!(
        sizes,
        [2, 2, 2, 1, 0],
        "pages are 1-based; a far page is empty"
    );
    let flattened: Vec<String> = pages.iter().flatten().cloned().collect();
    assert_eq!(flattened, newest_first, "pages tile the listing in order");
    assert_eq!(flattened.iter().collect::<HashSet<_>>().len(), 7);

    let page_zero = ltm.get_user_mem_by_page("alice", "ws", 2, 0, MemoryType::Unknown);
    assert_eq!(
        ids(&page_zero.await.expect("zero")),
        pages[0],
        "0 means page 1"
    );
    let far = ltm.get_user_mem_by_page("alice", "ws", 2, 99, MemoryType::Unknown);
    assert_eq!(far.await.expect("far page").len(), 0);

    let summaries = ltm.get_user_mem_by_page("alice", "ws", 10, 1, MemoryType::Summary);
    let summaries = summaries.await.expect("summary page");
    assert_eq!(summaries.len(), 2);
    assert!(summaries
        .iter()
        .all(|record| record.kind == MemoryType::Summary));
    let coding = ltm.get_user_mem_by_page("alice", "ws", 10, 1, MemoryType::Coding);
    assert_eq!(coding.await.expect("coding page").len(), 0);

    let error = ltm.get_user_mem_by_page("alice", "ws", 0, 1, MemoryType::Unknown);
    let error = error.await.expect_err("zero page size");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}

#[tokio::test]
async fn delete_aliases_leave_variables_and_messages_intact() {
    let ltm = facade().await;
    let scope = MemoryScope::new("alice", "ws");
    ltm.set_variable(&scope, "theme", "dark")
        .await
        .expect("variable");
    let turn = BaseMessage::user("remember the release checklist");
    append_message(&ltm, &scope, turn).await;
    add_record(&ltm, &scope, NewMemory::semantic("alice releasable record")).await;

    let removed = ltm.delete_mem_by_user_id("alice", "ws").await;
    assert_eq!(removed.expect("delete_mem_by_user_id"), 1);
    let stored = ltm
        .get_variables(&scope, None)
        .await
        .expect("variables survive");
    assert_eq!(stored.get("theme").map(String::as_str), Some("dark"));
    let history = ltm.get_recent_messages(&scope, 10).await.expect("history");
    assert_eq!(texts(&history), ["remember the release checklist"]);

    add_record(
        &ltm,
        &scope,
        NewMemory::semantic("alice releasable record again"),
    )
    .await;
    let removed = ltm.delete_mem_by_scope("ws").await;
    assert_eq!(removed.expect("delete_mem_by_scope"), 1);
    assert_eq!(variables(&ltm, &scope).await.len(), 1);
    assert_eq!(
        ltm.get_recent_messages(&scope, 10)
            .await
            .expect("history")
            .len(),
        1
    );
    assert_stats(&ltm, &scope, 0).await;
}
