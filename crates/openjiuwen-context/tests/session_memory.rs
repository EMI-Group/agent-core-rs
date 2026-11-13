#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for the session-memory seam.
//!
//! These tests pin the store contract (in-process and key-value, both scoped by
//! `(session_id, context_id)`), the notes envelope, the exact boundary/summary
//! message shapes, the `reinject_session_memory` replacement rules (anchor
//! resolution, strict shortening, tool-round extension), and `stamp_message_id`.
//!
//! Everything is offline: the key-value backend is `InMemoryKvStore`, messages
//! carry ids written by the tests, and no model or filesystem is involved.

use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_context::{
    boundary_message, is_boundary_message, reinject_session_memory, stamp_message_id,
    summary_message, validate_tool_pairs, InMemorySessionMemoryStore, KvSessionMemoryStore,
    OffloadScope, SessionMemoryNotes, SessionMemoryStore, CONTEXT_MESSAGE_ID_KEY,
    RECENT_MESSAGES_NOTICE, SESSION_MEMORY_BOUNDARY, SESSION_MEMORY_SUMMARY_INTRO,
};
use openjiuwen_foundation::prompt::{BaseMessage, Role, ToolCall};
use openjiuwen_foundation::store::{InMemoryKvStore, KvStore, KvValue};
use serde_json::{json, Value};

/// A scope over `session`/`context`.
fn scope(session: &str, context: &str) -> OffloadScope {
    OffloadScope::new(session, context)
}

/// Notes carrying `text` and an optional anchor.
fn notes(text: &str, anchor: Option<&str>) -> SessionMemoryNotes {
    SessionMemoryNotes {
        notes: text.to_owned(),
        anchor_message_id: anchor.map(str::to_owned),
    }
}

/// A message carrying the stable context id `id`.
fn with_id(mut message: BaseMessage, id: &str) -> BaseMessage {
    message.metadata.insert(
        CONTEXT_MESSAGE_ID_KEY.to_owned(),
        Value::String(id.to_owned()),
    );
    message
}

/// The conversation ids in order.
fn ids(messages: &[BaseMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| {
            message
                .metadata
                .get(CONTEXT_MESSAGE_ID_KEY)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        })
        .collect()
}

/// The plain texts in order.
fn texts(messages: &[BaseMessage]) -> Vec<String> {
    messages
        .iter()
        .map(|message| message.text().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// Store round-trips and scope isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_store_round_trips_and_isolates_scopes() {
    let store = InMemorySessionMemoryStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    let first = scope("s1", "main");
    let second = scope("s2", "main");
    let other_context = scope("s1", "other");

    store
        .put(&first, notes("first notes", Some("m1")))
        .await
        .unwrap();
    store.put(&second, notes("", None)).await.unwrap();
    assert_eq!(store.len(), 2);
    assert!(!store.is_empty());

    let loaded = store.get(&first).await.unwrap().unwrap();
    assert_eq!(loaded, notes("first notes", Some("m1")));
    assert_eq!(
        store.get(&second).await.unwrap().unwrap(),
        notes("", None),
        "empty notes still round-trip"
    );
    assert!(store.get(&other_context).await.unwrap().is_none());
    assert!(store.get(&scope("s3", "main")).await.unwrap().is_none());

    store.remove(&first).await.unwrap();
    assert!(store.get(&first).await.unwrap().is_none());
    assert!(store.get(&second).await.unwrap().is_some());
    // Removing missing notes is a no-op.
    store.remove(&first).await.unwrap();
    assert_eq!(store.len(), 1);
}

#[tokio::test]
async fn kv_store_round_trips_and_isolates_scopes() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvSessionMemoryStore::new(Arc::clone(&kv));

    let first = scope("s1", "main");
    let second = scope("s2", "main");
    store
        .put(&first, notes("first notes", Some("m1")))
        .await
        .unwrap();
    store
        .put(&second, notes("second notes", None))
        .await
        .unwrap();

    assert_eq!(store.key(&first), "context:session_memory:s1_main");
    assert_eq!(
        store.get(&first).await.unwrap().unwrap(),
        notes("first notes", Some("m1"))
    );
    assert_eq!(
        store.get(&second).await.unwrap().unwrap(),
        notes("second notes", None)
    );
    assert!(store.get(&scope("s1", "other")).await.unwrap().is_none());
    assert!(store.get(&scope("absent", "main")).await.unwrap().is_none());

    store.remove(&first).await.unwrap();
    assert!(store.get(&first).await.unwrap().is_none());
    assert!(store.get(&second).await.unwrap().is_some());
    store.remove(&first).await.unwrap();
}

#[tokio::test]
async fn kv_store_honours_a_custom_prefix() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvSessionMemoryStore::with_prefix(Arc::clone(&kv), "harness:memory:");
    let target = scope("s1", "main");

    assert_eq!(store.key(&target), "harness:memory:s1_main");
    store.put(&target, notes("custom", None)).await.unwrap();
    assert!(kv.get("harness:memory:s1_main").await.unwrap().is_some());
    assert_eq!(store.get(&target).await.unwrap().unwrap().notes, "custom");
}

#[tokio::test]
async fn kv_store_reports_undecodable_payloads() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvSessionMemoryStore::new(Arc::clone(&kv));
    let broken = scope("s1", "broken");

    kv.set(&store.key(&broken), KvValue::bytes(b"not json".to_vec()))
        .await
        .unwrap();
    let error = store.get(&broken).await.unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("session memory notes")),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// Notes envelope
// ---------------------------------------------------------------------------

#[test]
fn notes_round_trip_through_serde_and_tolerate_unknown_fields() {
    let full = notes("summary text", Some("m7"));
    let encoded = serde_json::to_string(&full).unwrap();
    let decoded: SessionMemoryNotes = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, full);

    let partial: SessionMemoryNotes = serde_json::from_str(r#"{"notes":"only"}"#).unwrap();
    assert_eq!(partial, notes("only", None));

    let unknown: SessionMemoryNotes =
        serde_json::from_str(r#"{"notes":"kept","anchor_message_id":"m1","extra":true}"#).unwrap();
    assert_eq!(unknown, notes("kept", Some("m1")));

    assert_eq!(
        serde_json::to_value(SessionMemoryNotes::default()).unwrap(),
        json!({"notes": "", "anchor_message_id": null})
    );
}

#[test]
fn is_empty_treats_missing_and_blank_notes_as_empty() {
    assert!(SessionMemoryNotes::default().is_empty());
    assert!(notes("", None).is_empty());
    assert!(notes("   \n\t ", Some("m1")).is_empty());
    assert!(!notes("content", None).is_empty());
}

// ---------------------------------------------------------------------------
// Message shapes
// ---------------------------------------------------------------------------

#[test]
fn boundary_message_matches_the_python_shape() {
    let boundary = boundary_message();
    assert_eq!(boundary.role, Role::System);
    assert_eq!(
        boundary.text(),
        format!("{SESSION_MEMORY_BOUNDARY}\nEarlier conversation replaced with session memory")
    );
    assert!(boundary.text().starts_with(SESSION_MEMORY_BOUNDARY));
    assert!(is_boundary_message(&boundary));

    assert!(!is_boundary_message(&BaseMessage::user(
        "[SESSION_MEMORY_BOUNDARY]\nwrong role"
    )));
    assert!(!is_boundary_message(&BaseMessage::system("persona")));
    assert!(!is_boundary_message(&summary_message(
        &notes("n", None),
        false
    )));
}

#[test]
fn summary_message_carries_intro_notes_and_optional_recent_notice() {
    let payload = notes("the notes body", Some("m1"));

    let without_tail = summary_message(&payload, false);
    assert_eq!(without_tail.role, Role::User);
    assert_eq!(
        without_tail.text(),
        format!("{SESSION_MEMORY_SUMMARY_INTRO}\n\nthe notes body")
    );
    assert!(!without_tail.text().contains(RECENT_MESSAGES_NOTICE));

    let with_tail = summary_message(&payload, true);
    assert_eq!(with_tail.role, Role::User);
    assert_eq!(
        with_tail.text(),
        format!("{SESSION_MEMORY_SUMMARY_INTRO}\n\nthe notes body\n\n{RECENT_MESSAGES_NOTICE}")
    );
}

// ---------------------------------------------------------------------------
// reinject_session_memory
// ---------------------------------------------------------------------------

#[test]
fn reinject_replaces_the_covered_prefix_and_keeps_the_tail() {
    let mut history = vec![
        with_id(BaseMessage::user("q0"), "c0"),
        with_id(BaseMessage::assistant("a0"), "c1"),
        with_id(BaseMessage::user("q1"), "c2"),
        with_id(BaseMessage::assistant("a1"), "c3"),
        with_id(BaseMessage::user("q2"), "c4"),
        with_id(BaseMessage::assistant("a2"), "c5"),
    ];
    let tail = history[3..].to_vec();

    let removed = reinject_session_memory(&mut history, &notes("notes body", Some("c2")));

    assert_eq!(removed, Some(3));
    assert_eq!(history.len(), 5);
    assert!(is_boundary_message(&history[0]));
    assert_eq!(history[1].role, Role::User);
    assert!(history[1].text().starts_with(SESSION_MEMORY_SUMMARY_INTRO));
    assert!(history[1].text().contains("notes body"));
    assert!(history[1].text().ends_with(RECENT_MESSAGES_NOTICE));
    assert_eq!(
        history[2..],
        tail[..],
        "messages after the anchor stay intact"
    );
    assert_eq!(ids(&history[2..]), vec!["c3", "c4", "c5"]);
}

#[test]
fn reinject_refuses_without_notes_or_a_resolvable_anchor() {
    let original = vec![
        with_id(BaseMessage::user("q0"), "c0"),
        with_id(BaseMessage::assistant("a0"), "c1"),
        with_id(BaseMessage::user("q1"), "c2"),
        with_id(BaseMessage::assistant("a1"), "c3"),
    ];

    let mut blank_notes = original.clone();
    assert_eq!(
        reinject_session_memory(&mut blank_notes, &notes("  ", Some("c3"))),
        None
    );
    assert_eq!(blank_notes, original);

    let mut missing_anchor = original.clone();
    assert_eq!(
        reinject_session_memory(&mut missing_anchor, &notes("content", Some("nope"))),
        None
    );
    assert_eq!(missing_anchor, original);

    let mut no_anchor = original.clone();
    assert_eq!(
        reinject_session_memory(&mut no_anchor, &notes("content", None)),
        None
    );
    assert_eq!(no_anchor, original);

    let mut empty_anchor = original.clone();
    assert_eq!(
        reinject_session_memory(&mut empty_anchor, &notes("content", Some(""))),
        None
    );
    assert_eq!(empty_anchor, original);

    // Ids exist only in metadata; matching never falls back to role+content.
    let mut unstamped = vec![
        BaseMessage::user("q0"),
        BaseMessage::assistant("a0"),
        BaseMessage::user("q1"),
    ];
    let before = unstamped.clone();
    assert_eq!(
        reinject_session_memory(&mut unstamped, &notes("content", Some("q0"))),
        None
    );
    assert_eq!(unstamped, before);
}

#[test]
fn reinject_never_grows_the_history() {
    // A two-message prefix would be replaced by two synthesized messages.
    let mut two = vec![
        with_id(BaseMessage::user("q0"), "c0"),
        with_id(BaseMessage::assistant("a0"), "c1"),
    ];
    let before = two.clone();
    assert_eq!(
        reinject_session_memory(&mut two, &notes("content", Some("c1"))),
        None
    );
    assert_eq!(two, before);

    // A one-message prefix would grow the history.
    let mut one = vec![with_id(BaseMessage::user("q0"), "c0")];
    assert_eq!(
        reinject_session_memory(&mut one, &notes("content", Some("c0"))),
        None
    );
    assert_eq!(one.len(), 1);

    // Three covered messages make it strictly shorter.
    let mut three = vec![
        with_id(BaseMessage::user("q0"), "c0"),
        with_id(BaseMessage::assistant("a0"), "c1"),
        with_id(BaseMessage::user("q1"), "c2"),
    ];
    assert_eq!(
        reinject_session_memory(&mut three, &notes("content", Some("c2"))),
        Some(3)
    );
    assert_eq!(three.len(), 2);
    assert_eq!(texts(&three)[0], boundary_message().text());
}

#[test]
fn reinject_moves_an_anchor_inside_a_tool_round_to_the_round_end() {
    let tool_round = || {
        vec![
            with_id(BaseMessage::user("q1"), "c0"),
            with_id(
                BaseMessage::assistant("")
                    .with_tool_calls(vec![ToolCall::new("search").with_id("call_1")]),
                "c1",
            ),
            with_id(BaseMessage::tool("call_1", "tool output"), "c2"),
            with_id(BaseMessage::assistant("a1"), "c3"),
            with_id(BaseMessage::user("q2"), "c4"),
            with_id(BaseMessage::assistant("a2"), "c5"),
        ]
    };

    // The anchor on the assistant that requests the tool is moved past the
    // result and the closing assistant.
    let mut history = tool_round();
    assert_eq!(
        reinject_session_memory(&mut history, &notes("content", Some("c1"))),
        Some(4)
    );
    assert_eq!(
        texts(&history),
        vec![
            boundary_message().text().into_owned(),
            summary_message(&notes("content", Some("c1")), true)
                .text()
                .into_owned(),
            "q2".to_owned(),
            "a2".to_owned(),
        ]
    );
    assert_eq!(validate_tool_pairs(&history), Vec::<usize>::new());
    assert!(!history.iter().any(|message| message.role == Role::Tool));

    // An anchor on the tool result itself moves the same way.
    let mut history = tool_round();
    assert_eq!(
        reinject_session_memory(&mut history, &notes("content", Some("c2"))),
        Some(4)
    );
    assert_eq!(validate_tool_pairs(&history), Vec::<usize>::new());
    assert!(!history.iter().any(|message| message.role == Role::Tool));

    // A round without a tool exchange never moves the anchor.
    let mut plain = vec![
        with_id(BaseMessage::user("q0"), "c0"),
        with_id(BaseMessage::user("q1"), "c1"),
        with_id(BaseMessage::user("q2"), "c2"),
        with_id(BaseMessage::assistant("a2"), "c3"),
    ];
    assert_eq!(
        reinject_session_memory(&mut plain, &notes("content", Some("c2"))),
        Some(3)
    );
    assert_eq!(texts(&plain)[2], "a2");
}

// ---------------------------------------------------------------------------
// stamp_message_id
// ---------------------------------------------------------------------------

#[test]
fn stamp_message_id_sets_once_and_is_stable() {
    let mut message = BaseMessage::user("hi");
    let first = stamp_message_id(&mut message);
    assert_eq!(first.len(), 32, "simple hex uuid");
    assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
    assert_eq!(
        message.metadata.get(CONTEXT_MESSAGE_ID_KEY).unwrap(),
        &Value::String(first.clone())
    );

    let second = stamp_message_id(&mut message);
    assert_eq!(second, first);
    assert_eq!(
        message.metadata.get(CONTEXT_MESSAGE_ID_KEY).unwrap(),
        &Value::String(first)
    );
}

#[test]
fn stamp_message_id_preserves_an_existing_id_and_replaces_an_empty_one() {
    let mut fixed = BaseMessage::user("x").with_metadata(CONTEXT_MESSAGE_ID_KEY, json!("fixed"));
    assert_eq!(stamp_message_id(&mut fixed), "fixed");
    assert_eq!(stamp_message_id(&mut fixed), "fixed");

    let mut empty = BaseMessage::user("x").with_metadata(CONTEXT_MESSAGE_ID_KEY, json!(""));
    let stamped = stamp_message_id(&mut empty);
    assert_ne!(stamped, "");
    assert_eq!(stamp_message_id(&mut empty), stamped);
}
