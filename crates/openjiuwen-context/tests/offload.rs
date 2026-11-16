#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for the offload storage layer.
//!
//! These tests pin the contract callers rely on: the `[[OFFLOAD: ...]]` marker
//! wire format and its parser, the three shipped stores (in-process, key-value,
//! session state) including their scope isolation and error paths, key
//! sanitisation, `rehydrate`, and the end-to-end path where
//! [`ToolResultProcessor`] offloads tool output into a session and `rehydrate`
//! recovers it from the placeholder.
//!
//! Everything is offline and deterministic: stores are in-process
//! (`InMemoryKvStore`, `AgentSession`), token counts come from the
//! `ApproxTokenCounter` with one character per token, and no model is involved.

use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_context::{
    rehydrate, sanitize_key, ContextConfig, ContextEventKind, ContextProcessor,
    InMemoryOffloadStore, KvOffloadStore, OffloadKind, OffloadMarker, OffloadRecord, OffloadScope,
    OffloadStore, ProcessorContext, SessionOffloadStore, ToolResultConfig, ToolResultProcessor,
    OFFLOAD_MARKER_PREFIX, PERSISTED_OUTPUT_TAG,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use openjiuwen_foundation::store::{InMemoryKvStore, KvStore, KvValue};
use openjiuwen_session::{AgentSession, Session};
use serde_json::{json, Map};
use tokio_util::sync::CancellationToken;

/// A scope over `session`/`context`.
fn scope(session: &str, context: &str) -> OffloadScope {
    OffloadScope::new(session, context)
}

/// A marker with an optional filesystem path.
fn marker(handle: &str, kind: OffloadKind, path: Option<&str>) -> OffloadMarker {
    OffloadMarker {
        handle: handle.to_owned(),
        kind,
        path: path.map(str::to_owned),
    }
}

/// One offloaded record carrying a single tool message.
fn record(handle: &str, kind: OffloadKind, text: &str) -> OffloadRecord {
    OffloadRecord {
        handle: handle.to_owned(),
        kind,
        messages: vec![BaseMessage::tool(handle, text)],
        metadata: Map::new(),
    }
}

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// The marker text renders byte-for-byte like Python's.
#[test]
fn marker_renders_python_wire_format() {
    let in_memory = marker("abc", OffloadKind::InMemory, None);
    assert_eq!(
        in_memory.to_marker_string(),
        "[[OFFLOAD: handle=abc, type=in_memory]]"
    );

    let on_disk = marker("abc", OffloadKind::Filesystem, Some("/tmp/out.txt"));
    let rendered = on_disk.to_marker_string();
    assert_eq!(
        rendered,
        "[[OFFLOAD: handle=abc, type=filesystem, path=/tmp/out.txt]]"
    );
    assert!(rendered.ends_with(", path=/tmp/out.txt]]"));

    // The published prefix opens every rendered marker.
    assert!(in_memory
        .to_marker_string()
        .starts_with(OFFLOAD_MARKER_PREFIX));
    assert!(rendered.starts_with(OFFLOAD_MARKER_PREFIX));
}

/// `parse` restores handle, kind, and path, even inside surrounding prose.
#[test]
fn marker_round_trips_through_parse() {
    let cases = [
        marker("h-1", OffloadKind::InMemory, None),
        marker("h-2", OffloadKind::Filesystem, None),
        marker("h-3", OffloadKind::Filesystem, Some("/var/offload/h-3.txt")),
    ];
    for expected in cases {
        let rendered = expected.to_marker_string();
        assert_eq!(OffloadMarker::parse(&rendered), Some(expected.clone()));

        let prose =
            format!("<{PERSISTED_OUTPUT_TAG}>head\n...\n{rendered}\n</{PERSISTED_OUTPUT_TAG}>");
        assert_eq!(OffloadMarker::parse(&prose), Some(expected));
    }
}

/// `parse` returns `None` for text without a marker, a truncated marker, or a
/// marker whose handle/type cannot be trusted.
#[test]
fn parse_rejects_missing_incomplete_and_unknown_markers() {
    assert_eq!(OffloadMarker::parse(""), None);
    assert_eq!(OffloadMarker::parse("nothing to see here"), None);
    assert_eq!(
        OffloadMarker::parse("[[OFFLOAD: handle=abc, type=in_memory"),
        None
    );
    assert_eq!(
        OffloadMarker::parse("[[OFFLOAD: handle=abc, type=in_memory]"),
        None
    );
    assert_eq!(OffloadMarker::parse("[[OFFLOAD: handle=abc]]"), None);
    assert_eq!(
        OffloadMarker::parse("[[OFFLOAD: handle=, type=filesystem]]"),
        None
    );
    assert_eq!(
        OffloadMarker::parse("[[OFFLOAD: handle=abc, type=network]]"),
        None
    );
}

/// The in-process store round-trips, misses as `Ok(None)`, and removes
/// idempotently.
#[tokio::test]
async fn in_memory_store_round_trips_and_removes_idempotently() {
    let store = InMemoryOffloadStore::new();
    assert_eq!(store.kind(), OffloadKind::InMemory);
    assert!(store.is_empty());

    let scoped = scope("s1", "main");
    let stored = record("h1", OffloadKind::InMemory, "original output");
    store.put(&scoped, stored.clone()).await.unwrap();

    assert_eq!(store.len(), 1);
    assert_eq!(store.get(&scoped, "h1").await.unwrap(), Some(stored));
    assert_eq!(store.get(&scoped, "unknown").await.unwrap(), None);

    store.remove(&scoped, "h1").await.unwrap();
    assert_eq!(store.get(&scoped, "h1").await.unwrap(), None);
    assert!(store.is_empty());

    // Removing a missing record is a no-op, not an error.
    store.remove(&scoped, "h1").await.unwrap();
    store.remove(&scoped, "never-there").await.unwrap();
    assert!(store.is_empty());
}

/// The in-process store keeps identically named handles apart per session and
/// per context.
#[tokio::test]
async fn in_memory_store_isolates_records_across_scopes() {
    let store = InMemoryOffloadStore::new();
    let main = scope("s1", "main");
    let other_context = scope("s1", "other");
    let other_session = scope("s2", "main");

    store
        .put(&main, record("h1", OffloadKind::InMemory, "one"))
        .await
        .unwrap();
    store
        .put(&other_context, record("h1", OffloadKind::InMemory, "two"))
        .await
        .unwrap();
    store
        .put(&other_session, record("h1", OffloadKind::InMemory, "three"))
        .await
        .unwrap();

    assert_eq!(store.len(), 3);
    for (scoped, expected) in [
        (&main, "one"),
        (&other_context, "two"),
        (&other_session, "three"),
    ] {
        let loaded = store.get(scoped, "h1").await.unwrap().unwrap();
        assert_eq!(loaded.messages[0].text(), expected);
    }

    // Removing one scope leaves the others intact.
    store.remove(&main, "h1").await.unwrap();
    assert_eq!(store.get(&main, "h1").await.unwrap(), None);
    assert!(store.get(&other_context, "h1").await.unwrap().is_some());
    assert!(store.get(&other_session, "h1").await.unwrap().is_some());
}

/// The key-value store round-trips records, reports misses, and removes them.
#[tokio::test]
async fn kv_store_round_trips_over_in_memory_kv() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvOffloadStore::new(Arc::clone(&kv));
    assert_eq!(store.kind(), OffloadKind::Filesystem);

    let scoped = scope("s1", "main");
    let stored = record("h1", OffloadKind::Filesystem, "kv payload");
    store.put(&scoped, stored.clone()).await.unwrap();

    assert_eq!(store.get(&scoped, "h1").await.unwrap(), Some(stored));
    assert_eq!(store.get(&scoped, "unknown").await.unwrap(), None);

    store.remove(&scoped, "h1").await.unwrap();
    assert_eq!(store.get(&scoped, "h1").await.unwrap(), None);
    // A repeated remove is harmless.
    store.remove(&scoped, "h1").await.unwrap();
}

/// Keys use the documented prefix and separator shape, and `with_prefix`
/// rewrites the prefix without touching the key body.
#[tokio::test]
async fn kv_store_uses_default_and_custom_prefixes() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let default_store = KvOffloadStore::new(Arc::clone(&kv));
    let scoped = scope("s1", "main");
    let expected_key = "context:offload:s1_main_h1";
    assert_eq!(default_store.key(&scoped, "h1"), expected_key);

    let stored = record("h1", OffloadKind::Filesystem, "prefixed payload");
    default_store.put(&scoped, stored.clone()).await.unwrap();

    // The record really lives under that raw key in the backing store.
    let raw = kv.get(expected_key).await.unwrap().unwrap();
    let decoded: OffloadRecord = serde_json::from_slice(raw.as_bytes()).unwrap();
    assert_eq!(decoded, stored);

    let custom = KvOffloadStore::with_prefix(Arc::clone(&kv), "tests:offload:");
    assert_eq!(custom.key(&scoped, "h1"), "tests:offload:s1_main_h1");
    let custom_record = record("h1", OffloadKind::Filesystem, "custom payload");
    custom.put(&scoped, custom_record.clone()).await.unwrap();

    let raw_custom = kv.get("tests:offload:s1_main_h1").await.unwrap().unwrap();
    let decoded_custom: OffloadRecord = serde_json::from_slice(raw_custom.as_bytes()).unwrap();
    assert_eq!(decoded_custom, custom_record);

    // Different prefixes do not see each other's records.
    assert_eq!(
        default_store.get(&scoped, "h1").await.unwrap(),
        Some(stored)
    );
    assert_eq!(
        custom.get(&scoped, "h1").await.unwrap(),
        Some(custom_record)
    );
}

/// The key-value store scopes records by session and context, so one handle can
/// hold different payloads in different scopes.
#[tokio::test]
async fn kv_store_isolates_scopes_for_the_same_handle() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvOffloadStore::new(Arc::clone(&kv));
    let main = scope("s1", "main");
    let other_session = scope("s2", "main");

    store
        .put(&main, record("h1", OffloadKind::Filesystem, "one"))
        .await
        .unwrap();
    store
        .put(&other_session, record("h1", OffloadKind::Filesystem, "two"))
        .await
        .unwrap();

    assert_eq!(
        store.get(&main, "h1").await.unwrap().unwrap().messages[0].text(),
        "one"
    );
    assert_eq!(
        store
            .get(&other_session, "h1")
            .await
            .unwrap()
            .unwrap()
            .messages[0]
            .text(),
        "two"
    );

    store.remove(&main, "h1").await.unwrap();
    assert_eq!(store.get(&main, "h1").await.unwrap(), None);
    assert!(store.get(&other_session, "h1").await.unwrap().is_some());
}

/// Undecodable stored bytes surface as an error instead of a panic.
#[tokio::test]
async fn kv_store_reports_malformed_records_as_errors() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvOffloadStore::new(Arc::clone(&kv));
    let scoped = scope("s1", "main");

    kv.set(
        &store.key(&scoped, "broken"),
        KvValue::bytes(b"this is not an offload record".to_vec()),
    )
    .await
    .unwrap();

    let error = store.get(&scoped, "broken").await.unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    // The failed decode does not remove or poison the entry.
    assert!(kv.exists(&store.key(&scoped, "broken")).await.unwrap());
}

/// The session store round-trips records and persists them as JSON in the
/// session's global state.
#[tokio::test]
async fn session_store_round_trips_through_session_state() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("sess-1"));
    let store = SessionOffloadStore::new(Arc::clone(&session));
    assert_eq!(store.kind(), OffloadKind::Filesystem);

    let scoped = scope("sess-1", "main");
    let stored = record("h1", OffloadKind::Filesystem, "payload");
    store.put(&scoped, stored.clone()).await.unwrap();

    assert_eq!(store.get(&scoped, "h1").await.unwrap(), Some(stored));
    assert_eq!(store.get(&scoped, "absent").await.unwrap(), None);

    let key = store.key(&scoped, "h1");
    assert_eq!(key, "context_offload.sess-1_main_h1");

    // The record is durable JSON under the documented session-state path.
    let persisted = session.get_state(&key).await.unwrap().unwrap();
    assert_eq!(persisted["handle"], json!("h1"));
    assert_eq!(persisted["kind"], json!("filesystem"));
    assert_eq!(persisted["messages"].as_array().unwrap().len(), 1);
    assert_eq!(persisted["messages"][0]["content"], json!("payload"));
    assert!(persisted["metadata"].is_object());

    let lane = session
        .get_state(SessionOffloadStore::STATE_PREFIX)
        .await
        .unwrap()
        .unwrap();
    assert!(lane.get("sess-1_main_h1").is_some());
}

/// Removing a session record clears the session-state entry, so a later read
/// misses instead of decoding a placeholder.
#[tokio::test]
async fn session_store_remove_clears_the_record() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("sess-2"));
    let store = SessionOffloadStore::new(Arc::clone(&session));
    let scoped = scope("sess-2", "main");
    let key = store.key(&scoped, "h1");

    store
        .put(&scoped, record("h1", OffloadKind::Filesystem, "gone soon"))
        .await
        .unwrap();
    assert!(session.get_state(&key).await.unwrap().is_some());

    store.remove(&scoped, "h1").await.unwrap();
    assert_eq!(store.get(&scoped, "h1").await.unwrap(), None);
    // The write of `null` removes the entry from the committed global lane.
    assert_eq!(session.get_state(&key).await.unwrap(), None);

    // Removing again (and removing an absent handle) stays a no-op.
    store.remove(&scoped, "h1").await.unwrap();
    store.remove(&scoped, "absent").await.unwrap();
    assert_eq!(store.get(&scoped, "h1").await.unwrap(), None);
}

/// Two sessions never read each other's offloaded records, even for the same
/// context id and handle.
#[tokio::test]
async fn session_store_isolates_records_across_sessions() {
    let first: Arc<dyn Session> = Arc::new(AgentSession::new("sess-a"));
    let second: Arc<dyn Session> = Arc::new(AgentSession::new("sess-b"));
    let first_store = SessionOffloadStore::new(Arc::clone(&first));
    let second_store = SessionOffloadStore::new(Arc::clone(&second));

    let first_scope = scope("sess-a", "main");
    let second_scope = scope("sess-b", "main");
    first_store
        .put(&first_scope, record("h1", OffloadKind::Filesystem, "one"))
        .await
        .unwrap();
    second_store
        .put(&second_scope, record("h1", OffloadKind::Filesystem, "two"))
        .await
        .unwrap();

    assert_eq!(
        first_store
            .get(&first_scope, "h1")
            .await
            .unwrap()
            .unwrap()
            .messages[0]
            .text(),
        "one"
    );
    assert_eq!(
        second_store
            .get(&second_scope, "h1")
            .await
            .unwrap()
            .unwrap()
            .messages[0]
            .text(),
        "two"
    );

    // A session store cannot reach the other session's record.
    assert_eq!(first_store.get(&second_scope, "h1").await.unwrap(), None);
    assert_eq!(second_store.get(&first_scope, "h1").await.unwrap(), None);
}

/// `sanitize_key` replaces unsafe characters and keeps safe ones.
#[test]
fn sanitize_key_replaces_unsafe_characters() {
    assert_eq!(sanitize_key("a-b_c.1/2:3"), "a-b_c_1_2_3");
    assert_eq!(sanitize_key(""), "");
    assert_eq!(sanitize_key("AZaz09-_-"), "AZaz09-_-");
    assert_eq!(sanitize_key("sess/../etc"), "sess____etc");
    // Non-ASCII characters are replaced too, keeping keys filesystem/JSON safe.
    assert_eq!(sanitize_key("café"), "caf_");
}

/// `rehydrate` returns exactly the messages stored under the marker's handle.
#[tokio::test]
async fn rehydrate_returns_the_stored_messages() {
    let store = InMemoryOffloadStore::new();
    let scoped = scope("s", "main");
    let stored = OffloadRecord {
        handle: "h1".to_owned(),
        kind: OffloadKind::InMemory,
        messages: vec![
            BaseMessage::user("the question"),
            BaseMessage::tool("call_1", "the original output"),
        ],
        metadata: Map::new(),
    };
    store.put(&scoped, stored.clone()).await.unwrap();

    let rendered = marker("h1", OffloadKind::InMemory, None).to_marker_string();
    let text = format!("trimmed head\n...\n{rendered}\n</{PERSISTED_OUTPUT_TAG}>");
    let restored = rehydrate(&store, &scoped, &text).await.unwrap().unwrap();
    assert_eq!(restored, stored.messages);
    assert_eq!(restored[1].text(), "the original output");
    assert_eq!(restored[1].tool_call_id.as_deref(), Some("call_1"));
}

/// Text without a marker rehydrates to `Ok(None)` rather than an error.
#[tokio::test]
async fn rehydrate_returns_none_without_a_marker() {
    let store = InMemoryOffloadStore::new();
    let scoped = scope("s", "main");
    assert_eq!(rehydrate(&store, &scoped, "").await.unwrap(), None);
    assert_eq!(
        rehydrate(&store, &scoped, "plain tool output")
            .await
            .unwrap(),
        None
    );
}

/// A marker whose record was removed rehydrates to `Ok(None)` and only logs a
/// warning, so callers can keep the placeholder instead of failing.
#[tokio::test]
async fn rehydrate_returns_none_when_the_record_is_gone() {
    let store = InMemoryOffloadStore::new();
    let scoped = scope("s", "main");
    store
        .put(&scoped, record("h1", OffloadKind::InMemory, "short lived"))
        .await
        .unwrap();

    let rendered = marker("h1", OffloadKind::InMemory, None).to_marker_string();
    let text = format!("head\n...\n{rendered}");
    assert!(rehydrate(&store, &scoped, &text).await.unwrap().is_some());

    store.remove(&scoped, "h1").await.unwrap();
    assert_eq!(rehydrate(&store, &scoped, &text).await.unwrap(), None);
}

/// The tool-result processor writes to a session-backed store, and the value it
/// puts in the placeholder is enough for `rehydrate` to recover the original.
#[tokio::test]
async fn tool_result_processor_offloads_into_the_session_store() {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("sess-e2e"));
    let session_store = Arc::new(SessionOffloadStore::new(Arc::clone(&session)));
    let store: Arc<dyn OffloadStore> = session_store.clone();

    let scoped = scope("sess-e2e", "main");
    let original = "o".repeat(200);
    let mut messages = vec![
        BaseMessage::user("run the grep"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep").with_id("call_big")]),
        BaseMessage::tool("call_big", original.clone()),
    ];

    let processor = ToolResultProcessor::new(
        ToolResultConfig {
            offload_over_chars: Some(50),
            keep_recent_results: 0,
            head_chars: 20,
            tail_chars: 0,
            ..ToolResultConfig::default()
        },
        Arc::clone(&store),
    );

    let config = ContextConfig::default();
    let token_counter = counter();
    let cancellation = CancellationToken::new();
    let context = ProcessorContext {
        config: &config,
        counter: &token_counter,
        model: None,
        offload: store.as_ref(),
        scope: &scoped,
        budget: 100_000,
        trigger_tokens: 80_000,
        manual: false,
        cancellation: &cancellation,
        recall: None,
        session_memory: None,
        session_memory_agent: None,
    };

    let event = processor
        .process(&mut messages, &context)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![2]);
    let handle = event.offload_handle.clone().unwrap();
    assert_ne!(handle, "");

    // The placeholder carries a filesystem marker for this handle.
    let placeholder = messages[2].text().to_string();
    assert!(placeholder.starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
    assert!(placeholder.ends_with(&format!("</{PERSISTED_OUTPUT_TAG}>")));
    let parsed = OffloadMarker::parse(&placeholder).unwrap();
    assert_eq!(parsed.handle, handle);
    assert_eq!(parsed.kind, OffloadKind::Filesystem);

    // The record landed in the session's global state under the scoped key.
    let key = session_store.key(&scoped, &handle);
    assert!(session.get_state(&key).await.unwrap().is_some());

    // Scanning the placeholder recovers the original tool message.
    let restored = rehydrate(store.as_ref(), &scoped, &placeholder)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].text(), original);
    assert_eq!(restored[0].tool_call_id.as_deref(), Some("call_big"));

    // A second pass leaves the placeholder alone (no marker duplication).
    assert!(processor
        .process(&mut messages, &context)
        .await
        .unwrap()
        .is_none());
}
