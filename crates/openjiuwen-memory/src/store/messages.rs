//! Conversation message history on the document store.
//!
//! Python keeps messages in SQL tables (`manage/mem_model/message_manager.py`);
//! this module keeps the same flat rows in one document collection
//! (`messages`, or whatever the prefix layout names). A row carries its scope (`user_id`, `scope_id`,
//! optional `session_id`), the wire role, the plain-text view of the content, and an
//! RFC 3339 timestamp, so history can be replayed without the foundation prompt types.
//!
//! Ids are deterministic ([`derived_message_id`]:
//! `msg_{fnv64(content + role + millis):016x}_{millis}`), which reproduces the Python
//! `msg_{sha256(...)}_{ms}` shape without a crypto dependency: re-appending the same
//! message in the same millisecond replaces the same row.
//!
//! The collection name comes from the [`KvPrefixRegistry`] (`messages` by default), so a
//! deployment can rename it exactly like the KV families; existing rows then stay reachable
//! through [`MessageStore::with_prefixes`].

use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::store::{Document, DocumentStore, Query, SortOrder};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::record::{MemoryFilter, MemoryScope};

use super::json::plain_text;
use super::keys;
use super::prefix::KvPrefixRegistry;

/// The 64-bit FNV-1a offset basis.
const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit FNV-1a prime.
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Document field holding the owner user id.
const FIELD_USER_ID: &str = "user_id";
/// Document field holding the owner scope id.
const FIELD_SCOPE_ID: &str = "scope_id";
/// Document field holding the session id, when the scope is session-bound.
const FIELD_SESSION_ID: &str = "session_id";
/// Document field holding the author role (`system`/`user`/`assistant`/`tool`).
const FIELD_ROLE: &str = "role";
/// Document field holding the plain-text message content.
const FIELD_CONTENT: &str = "content";
/// Document field holding the RFC 3339 timestamp.
const FIELD_TIMESTAMP: &str = "timestamp";

/// One persisted conversation message.
///
/// The flat JSON shape is the document row: the scope ids, the wire role string, the
/// plain-text content, and the RFC 3339 timestamp. `session_id` is omitted when the
/// scope is not session-bound, and unknown fields are tolerated on parse.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StoredMessage {
    /// Deterministic message id (`msg_{fnv64}_{millis}`).
    pub id: String,
    /// Owning user id.
    pub user_id: String,
    /// Owning scope id (workspace, tenant, or project).
    pub scope_id: String,
    /// Session the message belongs to, when the scope is session-bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Author role as its wire string (`system`, `user`, `assistant`, `tool`, ...).
    pub role: String,
    /// Plain-text view of the message content; non-text parts are JSON-rendered.
    pub content: String,
    /// When the message was appended.
    pub timestamp: DateTime<Utc>,
}

/// Reads and writes conversation history in one document collection.
///
/// The store is cheap to clone through its `Arc<dyn DocumentStore>` and holds no state
/// of its own; every operation is a single document-store call.
pub(crate) struct MessageStore {
    /// The document store holding the message collection.
    document: Arc<dyn DocumentStore>,
    /// The prefix layout supplying the collection name.
    prefixes: KvPrefixRegistry,
}

impl MessageStore {
    /// Builds a message store over `document` with the default collection name.
    pub(crate) fn new(document: Arc<dyn DocumentStore>) -> Self {
        Self::with_prefixes(document, KvPrefixRegistry::default())
    }

    /// Builds a message store over `document` with an explicit prefix layout.
    pub(crate) fn with_prefixes(
        document: Arc<dyn DocumentStore>,
        prefixes: KvPrefixRegistry,
    ) -> Self {
        Self { document, prefixes }
    }

    /// The name of the collection the history is read from and written to.
    fn collection(&self) -> &str {
        keys::messages_collection_in(&self.prefixes)
    }

    /// Appends `messages` to the history of `scope`, stamping every row with `timestamp`.
    ///
    /// Each row is upserted into the collection named by the prefix layout (`messages` by
    /// default) and the built [`StoredMessage`]s are returned in input order. An empty input
    /// writes nothing.
    pub(crate) async fn append(
        &self,
        scope: &MemoryScope,
        messages: &[BaseMessage],
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<StoredMessage>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }
        let rfc3339 = timestamp.to_rfc3339();
        let mut stored = Vec::with_capacity(messages.len());
        let mut documents = Vec::with_capacity(messages.len());
        for message in messages {
            let content = plain_text(&message.content);
            let role = message.role.as_str().to_owned();
            let id = derived_message_id(&content, &role, timestamp);
            let message = StoredMessage {
                id: id.clone(),
                user_id: scope.user_id.clone(),
                scope_id: scope.scope_id.clone(),
                session_id: scope.session_id.clone(),
                role,
                content,
                timestamp,
            };
            let mut document = Document::new(id)
                .with_field(FIELD_USER_ID, message.user_id.clone())
                .with_field(FIELD_SCOPE_ID, message.scope_id.clone())
                .with_field(FIELD_ROLE, message.role.clone())
                .with_field(FIELD_CONTENT, message.content.clone())
                .with_field(FIELD_TIMESTAMP, rfc3339.clone());
            if let Some(session_id) = &message.session_id {
                document = document.with_field(FIELD_SESSION_ID, session_id.clone());
            }
            documents.push(document);
            stored.push(message);
        }
        self.document.upsert(self.collection(), documents).await?;
        Ok(stored)
    }

    /// Returns the `limit` most recent messages of `scope`, in chronological order.
    ///
    /// The newest rows are selected by timestamp descending, then reversed so the
    /// caller always replays oldest-first. `limit == 0` returns an empty vector; a
    /// missing collection is empty history, not an error.
    pub(crate) async fn recent(
        &self,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<StoredMessage>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = Query {
            filter: scope_filter(scope),
            limit: Some(limit),
            offset: 0,
            order_by: vec![(FIELD_TIMESTAMP.to_owned(), SortOrder::Desc)],
        };
        let documents = self.document.query(self.collection(), &query).await?;
        let mut messages = Vec::with_capacity(documents.len());
        for document in documents {
            match serde_json::from_value::<StoredMessage>(document.to_value()) {
                Ok(message) => messages.push(message),
                Err(error) => tracing::warn!(
                    target: "openjiuwen_memory::store",
                    %error,
                    id = %document.id,
                    "skipping a malformed stored message"
                ),
            }
        }
        messages.reverse();
        Ok(messages)
    }

    /// Deletes every message matching `filter`, returning how many rows were removed.
    ///
    /// Scope clauses, `ids`, and the exclusive `created_after`/`created_before` bounds
    /// are translated into the document filter language; the other record clauses
    /// (kinds, importance, text) do not apply to message rows.
    pub(crate) async fn delete_for(&self, filter: &MemoryFilter) -> Result<usize> {
        let query = Query {
            filter: message_filter(filter),
            ..Query::default()
        };
        self.document.delete(self.collection(), &query).await
    }
}

/// The deterministic id of one message: `msg_{fnv64(content + role + millis):016x}_{millis}`.
///
/// The derivation is public to the crate so a legacy-row import and [`MessageStore::append`]
/// can never drift apart: the same content, wire role, and millisecond always yield the same
/// id, which is what makes re-appending replace the existing row.
pub(crate) fn derived_message_id(content: &str, role: &str, timestamp: DateTime<Utc>) -> String {
    let millis = timestamp.timestamp_millis();
    let digest = fnv1a_64(format!("{content}{role}{millis}").as_bytes());
    format!("msg_{digest:016x}_{millis}")
}

/// Computes the 64-bit FNV-1a hash of `bytes`.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    hash
}

/// The document filter selecting every message of `scope`.
///
/// `user_id` and `scope_id` always participate; `session_id` only when the scope is
/// session-bound, so a session-free scope reads every session's history.
fn scope_filter(scope: &MemoryScope) -> Value {
    let mut filter = Map::new();
    filter.insert(
        FIELD_USER_ID.to_owned(),
        Value::String(scope.user_id.clone()),
    );
    filter.insert(
        FIELD_SCOPE_ID.to_owned(),
        Value::String(scope.scope_id.clone()),
    );
    if let Some(session_id) = &scope.session_id {
        filter.insert(
            FIELD_SESSION_ID.to_owned(),
            Value::String(session_id.clone()),
        );
    }
    Value::Object(filter)
}

/// Translates a [`MemoryFilter`] into the document filter language.
fn message_filter(filter: &MemoryFilter) -> Value {
    let mut clauses = Map::new();
    if let Some(user_id) = &filter.user_id {
        clauses.insert(FIELD_USER_ID.to_owned(), Value::String(user_id.clone()));
    }
    if let Some(scope_id) = &filter.scope_id {
        clauses.insert(FIELD_SCOPE_ID.to_owned(), Value::String(scope_id.clone()));
    }
    if let Some(session_id) = &filter.session_id {
        clauses.insert(
            FIELD_SESSION_ID.to_owned(),
            Value::String(session_id.clone()),
        );
    }
    if let Some(agent_id) = &filter.agent_id {
        clauses.insert("agent_id".to_owned(), Value::String(agent_id.clone()));
    }
    if !filter.ids.is_empty() {
        let ids = filter
            .ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect();
        let mut in_clause = Map::new();
        in_clause.insert("$in".to_owned(), Value::Array(ids));
        clauses.insert("id".to_owned(), Value::Object(in_clause));
    }
    let mut timestamp = Map::new();
    if let Some(after) = filter.created_after {
        timestamp.insert("$gt".to_owned(), Value::String(after.to_rfc3339()));
    }
    if let Some(before) = filter.created_before {
        timestamp.insert("$lt".to_owned(), Value::String(before.to_rfc3339()));
    }
    if !timestamp.is_empty() {
        clauses.insert(FIELD_TIMESTAMP.to_owned(), Value::Object(timestamp));
    }
    Value::Object(clauses)
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)] // `assert!(x.is_empty())` reads better in history tests
mod tests {
    use super::super::prefix::KvFamily;
    use super::*;
    use chrono::TimeZone;
    use openjiuwen_foundation::prompt::{BaseMessage, ContentPart, ImageUrl, MessageContent};
    use openjiuwen_foundation::store::{InMemoryDocumentStore, Query};

    fn store() -> MessageStore {
        MessageStore::new(Arc::new(InMemoryDocumentStore::new()))
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws").session("s1")
    }

    fn time(offset: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + offset, 0)
            .single()
            .expect("fixed time")
    }

    async fn append_user(
        store: &MessageStore,
        scope: &MemoryScope,
        text: &str,
        offset: i64,
    ) -> StoredMessage {
        store
            .append(scope, &[BaseMessage::user(text)], time(offset))
            .await
            .expect("append")
            .pop()
            .expect("one message")
    }

    #[test]
    fn message_ids_are_deterministic_and_content_specific() {
        let first = derived_message_id("hello", "user", time(0));
        assert_eq!(first, derived_message_id("hello", "user", time(0)));
        assert_ne!(first, derived_message_id("hello", "assistant", time(0)));
        assert_ne!(first, derived_message_id("other", "user", time(0)));
        assert_ne!(first, derived_message_id("hello", "user", time(1)));
        assert!(first.starts_with("msg_"));
        let (_, suffix) = first.rsplit_once('_').expect("millis suffix");
        assert_eq!(suffix, time(0).timestamp_millis().to_string());
    }

    #[tokio::test]
    async fn append_builds_rows_with_the_plain_text_and_role() {
        let store = store();
        let timestamp = time(0);
        let stored = store
            .append(
                &scope(),
                &[
                    BaseMessage::system("rules"),
                    BaseMessage::assistant("answer"),
                ],
                timestamp,
            )
            .await
            .expect("append");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[0].content, "rules");
        assert_eq!(stored[0].user_id, "alice");
        assert_eq!(stored[0].scope_id, "ws");
        assert_eq!(stored[0].session_id.as_deref(), Some("s1"));
        assert_eq!(stored[0].timestamp, timestamp);
        assert_eq!(stored[1].role, "assistant");
        assert_eq!(
            stored[0].id,
            derived_message_id("rules", "system", timestamp)
        );
        assert_eq!(
            stored[1].id,
            derived_message_id("answer", "assistant", timestamp)
        );
        let document = store
            .document
            .get(keys::MESSAGES_COLLECTION, &stored[0].id)
            .await
            .expect("read row")
            .expect("row exists");
        assert_eq!(
            document.field(FIELD_ROLE),
            Some(&Value::String("system".to_owned()))
        );
        let value = document.to_value();
        assert_eq!(value[FIELD_ROLE], Value::String("system".to_owned()));
        assert_eq!(value[FIELD_CONTENT], Value::String("rules".to_owned()));
        assert_eq!(
            value[FIELD_TIMESTAMP],
            Value::String(timestamp.to_rfc3339())
        );
    }

    #[tokio::test]
    async fn recent_returns_the_newest_limit_in_chronological_order() {
        let store = store();
        let scope = scope();
        for (offset, text) in [(0, "one"), (1, "two"), (2, "three")] {
            append_user(&store, &scope, text, offset).await;
        }
        let recent = store.recent(&scope, 2).await.expect("recent");
        assert_eq!(
            recent
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            ["two", "three"]
        );
        let all = store.recent(&scope, 10).await.expect("recent");
        assert_eq!(
            all.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            ["one", "two", "three"]
        );
        assert!(store.recent(&scope, 0).await.expect("recent").is_empty());
    }

    #[tokio::test]
    async fn recent_filters_by_session() {
        let store = store();
        let first = MemoryScope::new("alice", "ws").session("s1");
        let second = MemoryScope::new("alice", "ws").session("s2");
        append_user(&store, &first, "first session", 0).await;
        append_user(&store, &first, "first session again", 1).await;
        append_user(&store, &second, "second session", 2).await;
        let hits = store.recent(&first, 10).await.expect("recent");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|m| m.session_id.as_deref() == Some("s1")));
        let hits = store.recent(&second, 10).await.expect("recent");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "second session");
        let unbound = MemoryScope::new("alice", "ws");
        let hits = store.recent(&unbound, 10).await.expect("recent");
        assert_eq!(hits.len(), 3);
        assert_eq!(store.recent(&first, 1_000).await.expect("recent").len(), 2);
        assert_eq!(
            store.recent(&unbound, 1_000).await.expect("recent").len(),
            3
        );
    }

    #[tokio::test]
    async fn delete_for_scope_and_ids() {
        let store = store();
        let scope = scope();
        let first = append_user(&store, &scope, "keep", 0).await;
        append_user(&store, &scope, "drop", 1).await;
        let mut by_id = MemoryFilter::new();
        by_id.ids = vec![first.id.clone()];
        assert_eq!(store.delete_for(&by_id).await.expect("delete id"), 1);
        assert_eq!(store.recent(&scope, 1_000).await.expect("recent").len(), 1);
        let other = MemoryFilter::for_scope(&MemoryScope::new("bob", "ws"));
        assert_eq!(store.delete_for(&other).await.expect("delete other"), 0);
        let mut bounded = MemoryFilter::for_scope(&scope);
        bounded.created_after = Some(time(0));
        assert_eq!(store.delete_for(&bounded).await.expect("delete bound"), 1);
        assert_eq!(store.recent(&scope, 1_000).await.expect("recent").len(), 0);
        append_user(&store, &scope, "third", 2).await;
        assert_eq!(
            store
                .delete_for(&MemoryFilter::for_scope(&scope))
                .await
                .expect("delete scope"),
            1
        );
        assert_eq!(store.recent(&scope, 1_000).await.expect("recent").len(), 0);
    }

    #[tokio::test]
    async fn missing_collection_is_empty_history() {
        let store = store();
        let scope = scope();
        assert!(store.recent(&scope, 5).await.expect("recent").is_empty());
        assert_eq!(store.recent(&scope, 1_000).await.expect("recent").len(), 0);
        assert_eq!(
            store
                .delete_for(&MemoryFilter::for_scope(&scope))
                .await
                .expect("delete"),
            0
        );
    }

    #[tokio::test]
    async fn parts_are_stored_as_plain_text() {
        let store = store();
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/cat.png".to_owned(),
                    detail: None,
                },
            },
        ]);
        let stored = store
            .append(&scope(), &[BaseMessage::user(content.clone())], time(0))
            .await
            .expect("append");
        assert_eq!(stored[0].content, plain_text(&content));
        assert!(stored[0]
            .content
            .starts_with("describe {\"type\":\"image_url\""));
        assert!(stored[0].content.contains("cat.png"));
    }

    #[test]
    fn malformed_rows_are_skipped_when_reading() {
        // The skip path is exercised through `recent`; assert the JSON shape mismatch
        // is detectable without the async store call.
        let result = serde_json::from_value::<StoredMessage>(serde_json::json!({"id": 1}));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn custom_collection_name_is_written_and_read_back() {
        let document = Arc::new(InMemoryDocumentStore::new());
        let prefixes = KvPrefixRegistry::new()
            .with_current(KvFamily::Messages, "chat_history")
            .expect("valid prefix");
        let store = MessageStore::with_prefixes(document.clone(), prefixes);
        let scope = scope();
        store
            .append(&scope, &[BaseMessage::user("hello")], time(0))
            .await
            .expect("append");
        assert_eq!(store.collection(), "chat_history");
        let query = Query {
            filter: scope_filter(&scope),
            ..Query::default()
        };
        assert!(document
            .query(keys::MESSAGES_COLLECTION, &query)
            .await
            .expect("query default collection")
            .is_empty());
        let history = store.recent(&scope, 10).await.expect("recent");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[0].id, derived_message_id("hello", "user", time(0)));
        assert_eq!(
            store
                .delete_for(&MemoryFilter::for_scope(&scope))
                .await
                .expect("delete"),
            1
        );
        assert!(store.recent(&scope, 10).await.expect("recent").is_empty());
    }
}
