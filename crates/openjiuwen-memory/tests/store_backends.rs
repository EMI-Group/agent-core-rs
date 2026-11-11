#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration coverage for the public [`MemoryStores`] surface.
//!
//! The unit tests beside the store modules can reach the crate-internal
//! `MemoryStorage`; this suite deliberately stays on the public API: it writes one
//! record through `stores.kv` and one message through `stores.document`, reads both
//! back, and — for the local-file backend — proves both survive closing and reopening
//! the stores on the same directory.

use chrono::{DateTime, TimeZone, Utc};
use openjiuwen_foundation::store::{Document, KvValue};
use openjiuwen_memory::record::{MemoryRecord, MemoryScope, MemoryType};
use openjiuwen_memory::store::{keys, MemoryStores, StoredMessage};

/// Id of the record this suite writes and reads back.
const RECORD_ID: &str = "record-1";
/// Id of the message this suite writes and reads back.
const MESSAGE_ID: &str = "msg-fixed";
/// Content of the record this suite writes and reads back.
const RECORD_TEXT: &str = "Alice prefers dark mode";

/// The scope shared by the record and the message.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws").session("s1")
}

/// A fixed timestamp so every assertion is wall-clock independent.
fn timestamp() -> DateTime<Utc> {
    Utc.timestamp_opt(1_700_000_000, 0)
        .single()
        .expect("fixed time")
}

/// The record written by [`write`].
fn record() -> MemoryRecord {
    MemoryRecord::new(scope(), RECORD_TEXT, MemoryType::SemanticMemory)
        .with_id(RECORD_ID)
        .with_timestamp(timestamp())
}

/// The message written by [`write`].
fn message() -> StoredMessage {
    StoredMessage {
        id: MESSAGE_ID.to_owned(),
        user_id: "alice".to_owned(),
        scope_id: "ws".to_owned(),
        session_id: Some("s1".to_owned()),
        role: "user".to_owned(),
        content: "hello".to_owned(),
        timestamp: timestamp(),
    }
}

/// Writes the record through the KV store and the message through the document store.
async fn write(stores: &MemoryStores) {
    let record = record();
    let record_json = serde_json::to_string(&record).expect("serialize record");
    stores
        .kv
        .set(
            &keys::record_key(&record.scope, &record.id),
            KvValue::text(record_json),
        )
        .await
        .expect("write record");

    let message = message();
    let document =
        Document::from_value(serde_json::to_value(&message).expect("serialize the stored message"))
            .expect("message document shape");
    stores
        .document
        .upsert(keys::MESSAGES_COLLECTION, vec![document])
        .await
        .expect("write message");
}

/// Asserts both rows are still readable with their identity and content intact.
async fn assert_written(stores: &MemoryStores) {
    let value = stores
        .kv
        .get(&keys::record_key(&scope(), RECORD_ID))
        .await
        .expect("read record")
        .expect("record exists");
    let record_json: serde_json::Value =
        serde_json::from_str(value.as_str().expect("record is text")).expect("record JSON");
    assert_eq!(record_json["id"], serde_json::json!(RECORD_ID));
    assert_eq!(record_json["text"], serde_json::json!(RECORD_TEXT));
    assert_eq!(record_json["user_id"], serde_json::json!("alice"));

    let document = stores
        .document
        .get(keys::MESSAGES_COLLECTION, MESSAGE_ID)
        .await
        .expect("read message")
        .expect("message exists");
    let restored: StoredMessage =
        serde_json::from_value(document.to_value()).expect("message shape");
    assert_eq!(restored, message());
}

#[tokio::test]
async fn in_memory_round_trips_records_and_messages() {
    let stores = MemoryStores::in_memory();
    write(&stores).await;
    assert_written(&stores).await;
}

#[tokio::test]
async fn local_file_persists_records_and_messages_across_reopen() {
    let directory = tempfile::tempdir().expect("tempdir");
    {
        let stores = MemoryStores::local_file(directory.path())
            .await
            .expect("open local-file stores");
        write(&stores).await;
        assert_written(&stores).await;
    }
    let reopened = MemoryStores::local_file(directory.path())
        .await
        .expect("reopen local-file stores");
    assert_written(&reopened).await;
}
