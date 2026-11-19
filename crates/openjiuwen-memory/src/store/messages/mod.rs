//! Conversation message history on the document store.
//!
//! Python keeps messages in SQL tables (`manage/mem_model/message_manager.py`);
//! this module keeps the same flat rows in one document collection
//! (`messages`, or whatever the prefix layout names). A row carries its scope (`user_id`, `scope_id`,
//! optional `session_id`), the wire role, the plain-text view of the content, and an
//! RFC 3339 timestamp, so history can be replayed without the foundation prompt types.
//!
//! Ids are deterministic ([`derived_message_id`]:
//! `msg_{fnv64(user + scope + session + role + millis + ordinal + content):016x}_{millis}`), which
//! reproduces the Python `msg_{sha256(...)}_{ms}` shape without a crypto dependency. The
//! preimage folds the row's `user_id`, `scope_id`, `session_id`, and its position in the
//! appended batch, so two tenants (or two identical turns of one batch) never share a
//! row, while re-appending the same batch in the same millisecond replaces its own rows.
//!
//! The collection name comes from the [`KvPrefixRegistry`] (`messages` by default), so a
//! deployment can rename it exactly like the KV families; existing rows then stay reachable
//! through [`MessageStore::with_prefixes`].
//!
//! History beyond "the newest N" ([`MessageStore::recent`]) is a keyset page
//! ([`MessageStore::page`]) ordered by `(timestamp, id)`: the position travels as an
//! opaque [`MessageCursor`] token, so pages survive a store reopen and need no offsets.

use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::store::{Document, DocumentStore, Query, SortOrder};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::record::{MemoryFilter, MemoryScope};

use super::json::plain_text;
use super::keys;
use super::prefix::KvPrefixRegistry;

mod cursor;

pub use cursor::{MessageCursor, MessagePage};

use cursor::{message_filter, page_filter, scope_filter};

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
/// Document field holding the message id (the document primary key).
const FIELD_ID: &str = "id";
/// Document field holding the author role (`system`/`user`/`assistant`/`tool`).
const FIELD_ROLE: &str = "role";
/// Document field holding the plain-text message content.
const FIELD_CONTENT: &str = "content";
/// Document field holding the RFC 3339 timestamp.
const FIELD_TIMESTAMP: &str = "timestamp";

/// Renders `timestamp` in the one RFC 3339 form the message rows and the pagination
/// bounds share: UTC with an explicit `+00:00` offset and exactly six fractional digits
/// (`2023-11-14T22:13:20.000123+00:00`).
///
/// One fixed precision matters because the document filter compares these strings
/// lexicographically: a fixed-width fraction keeps lexical order chronological, and using
/// the same width for the stored row and for a [`MessageCursor`] bound means the bound is
/// exactly the row it points at instead of a truncated (or rounded) neighbour. Nanoseconds
/// below the microsecond are dropped, which the row and the cursor drop identically.
pub(crate) fn rfc3339_micros(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::Micros, false)
}

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
    ///
    /// The id of a row folds the scope and the row's position in the batch
    /// ([`derived_message_id`]), so two identical turns of one batch (which share the batch
    /// `timestamp`) stay two rows and the same text in two scopes never overwrites another
    /// tenant's row.
    pub(crate) async fn append(
        &self,
        scope: &MemoryScope,
        messages: &[BaseMessage],
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<StoredMessage>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }
        let mut stored = Vec::with_capacity(messages.len());
        let mut documents = Vec::with_capacity(messages.len());
        for (ordinal, message) in messages.iter().enumerate() {
            let content = plain_text(&message.content);
            let role = message.role.as_str().to_owned();
            let id = derived_message_id(DerivedMessageId {
                user_id: &scope.user_id,
                scope_id: &scope.scope_id,
                session_id: scope.session_id.as_deref(),
                content: &content,
                role: &role,
                timestamp,
                ordinal,
            });
            let message = StoredMessage {
                user_id: scope.user_id.clone(),
                scope_id: scope.scope_id.clone(),
                session_id: scope.session_id.clone(),
                role,
                content,
                timestamp,
                id,
            };
            documents.push(stored_document(&message));
            stored.push(message);
        }
        self.document.upsert(self.collection(), documents).await?;
        Ok(stored)
    }

    /// Returns the `limit` most recent messages of `scope`, in chronological order.
    ///
    /// The newest rows are selected by `(timestamp, id)` descending — the same total order
    /// [`MessageStore::page`] walks ascending — then reversed so the caller always replays
    /// oldest-first. Rows sharing a timestamp (one appended batch carries exactly one)
    /// therefore come back in ascending id order, which makes `recent(n)` exactly the tail
    /// of `page()`. `limit == 0` returns an empty vector; a missing collection is empty
    /// history, not an error.
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
            order_by: vec![
                (FIELD_TIMESTAMP.to_owned(), SortOrder::Desc),
                (FIELD_ID.to_owned(), SortOrder::Desc),
            ],
        };
        let documents = self.document.query(self.collection(), &query).await?;
        let mut messages = decode(documents);
        messages.reverse();
        Ok(messages)
    }

    /// Returns one keyset page of the history of `scope`, oldest row first.
    ///
    /// Rows are ordered strictly ascending by `(timestamp, id)`, so rows sharing a
    /// timestamp paginate in ascending id order with no duplicates and no gaps. The
    /// page resumes strictly after `cursor` (a token from
    /// [`MessageCursor::encode`] or [`MessagePage::next_cursor`]); `filter` narrows the
    /// page further and is `AND`ed with the scope clauses rather than replacing them.
    /// One extra row is read to decide `has_more`, and `next_cursor` is the position
    /// of the last returned row (`None` on the last page).
    ///
    /// `page_size == 0` and a missing collection both yield an empty page with
    /// `has_more == false`; a malformed `cursor` is [`Error::Validation`].
    ///
    /// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
    pub(crate) async fn page(
        &self,
        scope: &MemoryScope,
        page_size: usize,
        cursor: Option<&str>,
        filter: Option<&MemoryFilter>,
    ) -> Result<MessagePage> {
        if page_size == 0 {
            return Ok(MessagePage::default());
        }
        let boundary = cursor.map(MessageCursor::decode).transpose()?;
        let query = Query {
            filter: page_filter(scope, filter, boundary.as_ref()),
            offset: 0,
            order_by: vec![
                (FIELD_TIMESTAMP.to_owned(), SortOrder::Asc),
                (FIELD_ID.to_owned(), SortOrder::Asc),
            ],
            ..Query::default()
        };
        // The filter language has no `$or`, so the cursor bound is inclusive and the
        // rows at the cursor's timestamp that sort at or before it are dropped here.
        // Their number is unknown up front, so the read window grows until it holds a
        // full page plus the look-ahead row or the history ends.
        let growth = page_size.saturating_add(1);
        let mut window = growth;
        let mut rows = loop {
            let windowed = Query {
                limit: Some(window),
                ..query.clone()
            };
            let documents = self.document.query(self.collection(), &windowed).await?;
            let exhausted = documents.len() < window;
            let rows: Vec<StoredMessage> = decode(documents)
                .into_iter()
                .filter(|message| boundary.as_ref().is_none_or(|at| at.is_before(message)))
                .collect();
            if rows.len() > page_size || exhausted {
                break rows;
            }
            window = window.saturating_add(growth);
        };
        let has_more = rows.len() > page_size;
        rows.truncate(page_size);
        let next_cursor = match rows.last() {
            Some(last) if has_more => Some(MessageCursor::of(last).encode()),
            _ => None,
        };
        Ok(MessagePage {
            items: rows,
            next_cursor,
            has_more,
        })
    }

    /// Deletes every message of one tenant scope matching `filter`, returning how many
    /// rows were removed.
    ///
    /// The delete is always scoped: `filter` must name a non-empty `user_id` and
    /// `scope_id` ([`require_message_scope`]), and those clauses are `AND`ed into the
    /// document filter, so no filter shape can reach another tenant's rows or wipe the
    /// whole collection. `session_id`, `ids`, and the exclusive `created_after`/
    /// `created_before` bounds narrow the delete further; the other record clauses
    /// (kinds, importance, text) do not apply to message rows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `filter` names no `user_id` or no `scope_id`,
    /// before anything is deleted, and the storage error when the delete fails.
    ///
    /// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
    pub(crate) async fn delete_for(&self, filter: &MemoryFilter) -> Result<usize> {
        let query = Query {
            filter: delete_filter(filter)?,
            ..Query::default()
        };
        self.document.delete(self.collection(), &query).await
    }
}

/// Rejects a message query that names no tenant scope.
///
/// Conversation history is never deleted without an explicit `user_id` and `scope_id`:
/// an empty [`MemoryFilter`] renders `{}`,
/// which the document filter language reads as "match every row", and Python's history
/// API has no cross-tenant delete. Callers build the scope with
/// [`MemoryFilter::for_scope`].
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the missing field(s).
///
/// [`Error::Validation`]: openjiuwen_common::error::Error::Validation
pub(crate) fn require_message_scope(filter: &MemoryFilter) -> Result<()> {
    let mut missing = Vec::new();
    if filter.user_id.as_deref().is_none_or(str::is_empty) {
        missing.push("user_id");
    }
    if filter.scope_id.as_deref().is_none_or(str::is_empty) {
        missing.push("scope_id");
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(Error::validation(format!(
        "message history requires an explicit tenant scope: filter.{} is missing \
         (build the filter with MemoryFilter::for_scope)",
        missing.join(" and filter.")
    )))
}

/// Builds the document filter of one scoped message delete.
///
/// The validated `user_id`/`scope_id` are written into the filter unconditionally, so the
/// scope clauses are always part of the delete no matter what the caller's filter carries.
fn delete_filter(filter: &MemoryFilter) -> Result<Value> {
    require_message_scope(filter)?;
    let mut clauses = match message_filter(filter) {
        Value::Object(clauses) => clauses,
        _ => Map::new(),
    };
    if let Some(user_id) = &filter.user_id {
        clauses.insert(FIELD_USER_ID.to_owned(), Value::String(user_id.clone()));
    }
    if let Some(scope_id) = &filter.scope_id {
        clauses.insert(FIELD_SCOPE_ID.to_owned(), Value::String(scope_id.clone()));
    }
    Ok(Value::Object(clauses))
}

/// The fields [`derived_message_id`] hashes into a message id.
///
/// The scope parts make the id tenant-unique, and `ordinal` disambiguates identical turns
/// of one batch (which shares a single [`DateTime`]).
pub(crate) struct DerivedMessageId<'a> {
    /// Owning user id.
    pub user_id: &'a str,
    /// Owning scope id.
    pub scope_id: &'a str,
    /// Session id, when the scope is session-bound.
    pub session_id: Option<&'a str>,
    /// Plain-text content.
    pub content: &'a str,
    /// Author role as its wire string.
    pub role: &'a str,
    /// The batch timestamp.
    pub timestamp: DateTime<Utc>,
    /// The row's position in the appended batch.
    pub ordinal: usize,
}

/// The deterministic id of one message:
/// `msg_{fnv64(scope + role + millis + ordinal + content):016x}_{millis}`.
///
/// The derivation is public to the crate so a legacy-row import and [`MessageStore::append`]
/// can never drift apart: the same scope, content, wire role, millisecond, and batch
/// position always yield the same id, which is what makes re-appending replace the existing
/// row. The scope is part of the preimage because the document collection is shared by every
/// tenant: without it the same text in two scopes would collapse into one row.
pub(crate) fn derived_message_id(fields: DerivedMessageId<'_>) -> String {
    let millis = fields.timestamp.timestamp_millis();
    // A unit separator keeps the parts unambiguous: no id, role, or content can forge the
    // boundary between two neighbouring fields.
    let preimage = format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{millis}\u{1f}{}\u{1f}{}",
        fields.user_id,
        fields.scope_id,
        fields.session_id.unwrap_or_default(),
        fields.role,
        fields.ordinal,
        fields.content
    );
    let digest = fnv1a_64(preimage.as_bytes());
    format!("msg_{digest:016x}_{millis}")
}

/// Builds the stored document of one message (the shape [`MessageStore::append`] writes).
///
/// This is the single source of the document layout: the append path and the
/// `migration::message` legacy-row import both go through it, so the two can never
/// disagree on field names, the timestamp encoding, or the optional `session_id`.
pub(crate) fn stored_document(message: &StoredMessage) -> Document {
    let mut document = Document::new(message.id.clone())
        .with_field(FIELD_USER_ID, message.user_id.clone())
        .with_field(FIELD_SCOPE_ID, message.scope_id.clone())
        .with_field(FIELD_ROLE, message.role.clone())
        .with_field(FIELD_CONTENT, message.content.clone())
        .with_field(FIELD_TIMESTAMP, rfc3339_micros(message.timestamp));
    if let Some(session_id) = &message.session_id {
        document = document.with_field(FIELD_SESSION_ID, session_id.clone());
    }
    document
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

/// Decodes stored documents into messages, warning about and skipping malformed rows.
///
/// A corrupt row never fails a history read; it is logged with its id and left out.
fn decode(documents: Vec<Document>) -> Vec<StoredMessage> {
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
    messages
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)] // `assert!(x.is_empty())` reads better in history tests
mod tests {
    use super::super::prefix::KvFamily;
    use super::*;
    use chrono::TimeZone;
    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::prompt::{BaseMessage, ContentPart, ImageUrl, MessageContent};
    use openjiuwen_foundation::store::{InMemoryDocumentStore, LocalFileDocumentStore, Query};
    use serde_json::Value;

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
        fn id(
            user_id: &str,
            scope_id: &str,
            session_id: &str,
            content: &str,
            role: &str,
            ordinal: usize,
            timestamp: DateTime<Utc>,
        ) -> String {
            derived_message_id(DerivedMessageId {
                user_id,
                scope_id,
                session_id: Some(session_id),
                content,
                role,
                timestamp,
                ordinal,
            })
        }
        let first = id("alice", "ws", "s1", "hello", "user", 0, time(0));
        assert_eq!(first, id("alice", "ws", "s1", "hello", "user", 0, time(0)));
        assert_ne!(
            first,
            id("alice", "ws", "s1", "hello", "assistant", 0, time(0))
        );
        assert_ne!(first, id("alice", "ws", "s1", "other", "user", 0, time(0)));
        assert_ne!(
            first,
            id("alice", "ws", "s1", "hello", "user", 1, time(0)),
            "a batch ordinal is part of the id"
        );
        assert_ne!(
            first,
            id("alice", "ws", "s1", "hello", "user", 0, time(1)),
            "the millisecond is part of the id"
        );
        assert_ne!(
            first,
            id("bob", "ws", "s1", "hello", "user", 0, time(0)),
            "the owning user is part of the id"
        );
        assert_ne!(
            first,
            id("alice", "other", "s1", "hello", "user", 0, time(0)),
            "the scope is part of the id"
        );
        assert_ne!(
            first,
            id("alice", "ws", "s2", "hello", "user", 0, time(0)),
            "the session is part of the id"
        );
        assert_ne!(
            first,
            derived_message_id(DerivedMessageId {
                user_id: "alice",
                scope_id: "ws",
                session_id: None,
                content: "hello",
                role: "user",
                timestamp: time(0),
                ordinal: 0,
            }),
            "a session-free row is not a session-bound one"
        );
        assert!(first.starts_with("msg_"));
        let (_, suffix) = first.rsplit_once('_').expect("millis suffix");
        assert_eq!(suffix, time(0).timestamp_millis().to_string());
        let (hash, _) = first
            .strip_prefix("msg_")
            .and_then(|rest| rest.split_once('_'))
            .expect("hash segment");
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn a_batch_of_identical_messages_keeps_every_row() {
        let store = store();
        let scope = scope();
        let stored = store
            .append(
                &scope,
                &[
                    BaseMessage::user("same text"),
                    BaseMessage::user("same text"),
                ],
                time(0),
            )
            .await
            .expect("append");
        assert_eq!(stored.len(), 2);
        assert_ne!(stored[0].id, stored[1].id, "one batch, two rows");
        let history = store.recent(&scope, 10).await.expect("recent");
        assert_eq!(
            history
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>(),
            ["same text", "same text"],
            "both rows are readable"
        );
        let mut expected: Vec<String> = stored.iter().map(|message| message.id.clone()).collect();
        expected.sort_unstable();
        assert_eq!(
            history
                .iter()
                .map(|message| message.id.clone())
                .collect::<Vec<_>>(),
            expected,
            "both rows share one timestamp and paginate in ascending id order"
        );
        // Re-appending the identical batch replaces its own rows instead of adding more.
        let again = store
            .append(
                &scope,
                &[
                    BaseMessage::user("same text"),
                    BaseMessage::user("same text"),
                ],
                time(0),
            )
            .await
            .expect("re-append");
        assert_eq!(
            again.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            stored.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
        assert_eq!(store.recent(&scope, 10).await.expect("recent").len(), 2);
    }

    #[tokio::test]
    async fn the_same_text_in_two_scopes_stays_two_rows() {
        let store = store();
        let alice = MemoryScope::new("alice", "ws").session("s1");
        let bob = MemoryScope::new("bob", "ws").session("s1");
        let other_session = MemoryScope::new("alice", "ws").session("s2");
        for scope in [&alice, &bob, &other_session] {
            store
                .append(scope, &[BaseMessage::user("hello")], time(0))
                .await
                .expect("append");
        }
        let mut ids: Vec<String> = Vec::new();
        for scope in [&alice, &bob, &other_session] {
            let history = store.recent(scope, 10).await.expect("recent");
            assert_eq!(history.len(), 1, "each scope owns exactly its own row");
            assert_eq!(history[0].content, "hello");
            assert_eq!(history[0].user_id, scope.user_id);
            assert_eq!(history[0].scope_id, scope.scope_id);
            assert_eq!(history[0].session_id, scope.session_id);
            ids.push(history[0].id.clone());
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3, "no two scopes share a row");
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
            derived_message_id(DerivedMessageId {
                user_id: "alice",
                scope_id: "ws",
                session_id: Some("s1"),
                content: "rules",
                role: "system",
                timestamp,
                ordinal: 0,
            })
        );
        assert_eq!(
            stored[1].id,
            derived_message_id(DerivedMessageId {
                user_id: "alice",
                scope_id: "ws",
                session_id: Some("s1"),
                content: "answer",
                role: "assistant",
                timestamp,
                ordinal: 1,
            })
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
            Value::String("2023-11-14T22:13:20.000000+00:00".to_owned()),
            "the stored row carries the canonical microsecond timestamp"
        );
        assert_eq!(
            rfc3339_micros(timestamp),
            "2023-11-14T22:13:20.000000+00:00"
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
    async fn recent_is_the_tail_of_the_page_order_within_one_batch() {
        let store = store();
        let scope = scope();
        // One append stamps the whole batch with the same timestamp, so only the id
        // tie-break can order these rows.
        let batch: Vec<BaseMessage> = (0..5)
            .map(|index| BaseMessage::user(format!("row {index}")))
            .collect();
        let stored = store.append(&scope, &batch, time(0)).await.expect("append");
        let mut expected: Vec<String> = stored.iter().map(|message| message.id.clone()).collect();
        expected.sort_unstable();

        // `page` defines the total order (timestamp, then id ascending).
        let mut paged = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let page = store
                .page(&scope, 2, cursor.as_deref(), None)
                .await
                .expect("page");
            paged.extend(page.items.into_iter().map(|message| message.id));
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        assert_eq!(paged, expected, "the page order is (timestamp, id)");

        // `recent(n)` is always the tail of that order, never an arbitrary subset.
        for limit in 1..=expected.len() {
            let recent = store.recent(&scope, limit).await.expect("recent");
            let ids: Vec<String> = recent.into_iter().map(|message| message.id).collect();
            assert_eq!(
                ids,
                expected[expected.len() - limit..],
                "recent({limit}) is the tail of the page order"
            );
        }
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
        let mut by_id = MemoryFilter::for_scope(&scope);
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
    async fn delete_for_never_reaches_another_tenant() {
        let store = store();
        let alice = scope();
        let bob = MemoryScope::new("bob", "ws").session("s1");
        append_user(&store, &alice, "alice one", 0).await;
        append_user(&store, &alice, "alice two", 1).await;
        append_user(&store, &bob, "bob one", 0).await;
        assert_eq!(
            store
                .delete_for(&MemoryFilter::for_scope(&alice))
                .await
                .expect("delete alice"),
            2
        );
        assert!(store.recent(&alice, 10).await.expect("alice").is_empty());
        let bob_history = store.recent(&bob, 10).await.expect("bob");
        assert_eq!(bob_history.len(), 1, "bob's history is untouched");
        assert_eq!(bob_history[0].content, "bob one");
    }

    #[tokio::test]
    async fn delete_for_without_a_scope_is_a_validation_error() {
        let store = store();
        let alice = scope();
        let bob = MemoryScope::new("bob", "ws").session("s1");
        append_user(&store, &alice, "alice one", 0).await;
        append_user(&store, &bob, "bob one", 0).await;
        for filter in [
            MemoryFilter::new(),
            MemoryFilter {
                user_id: Some("alice".to_owned()),
                ..MemoryFilter::default()
            },
            MemoryFilter {
                scope_id: Some("ws".to_owned()),
                ..MemoryFilter::default()
            },
            MemoryFilter {
                user_id: Some("alice".to_owned()),
                scope_id: Some(String::new()),
                ..MemoryFilter::default()
            },
        ] {
            let error = store
                .delete_for(&filter)
                .await
                .expect_err("an unscoped delete is refused");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
        assert_eq!(store.recent(&alice, 10).await.expect("alice").len(), 1);
        assert_eq!(store.recent(&bob, 10).await.expect("bob").len(), 1);
    }

    #[tokio::test]
    async fn delete_for_rejects_a_missing_field_naming_it() {
        let store = store();
        let error = store
            .delete_for(&MemoryFilter::new())
            .await
            .expect_err("unscoped");
        let message = error.to_string();
        assert!(message.contains("user_id"), "{message}");
        assert!(message.contains("scope_id"), "{message}");
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
        assert_eq!(
            history[0].id,
            derived_message_id(DerivedMessageId {
                user_id: "alice",
                scope_id: "ws",
                session_id: Some("s1"),
                content: "hello",
                role: "user",
                timestamp: time(0),
                ordinal: 0,
            })
        );
        assert_eq!(
            store
                .delete_for(&MemoryFilter::for_scope(&scope))
                .await
                .expect("delete"),
            1
        );
        assert!(store.recent(&scope, 10).await.expect("recent").is_empty());
    }

    fn page_ids(page: &MessagePage) -> Vec<&str> {
        page.items.iter().map(|m| m.id.as_str()).collect()
    }

    #[tokio::test]
    async fn page_walks_the_whole_history_in_ascending_order() {
        let store = store();
        let scope = scope();
        let mut expected = Vec::new();
        for (offset, text) in [
            (0, "one"),
            (1, "two"),
            (2, "three"),
            (3, "four"),
            (4, "five"),
        ] {
            expected.push(append_user(&store, &scope, text, offset).await);
        }
        let mut seen = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0;
        loop {
            let page = store
                .page(&scope, 2, cursor.as_deref(), None)
                .await
                .expect("page");
            pages += 1;
            seen.extend(page_ids(&page).into_iter().map(str::to_owned));
            if let Some(next) = page.next_cursor {
                assert!(page.has_more);
                assert_eq!(page.items.len(), 2);
                cursor = Some(next);
            } else {
                assert!(!page.has_more);
                break;
            }
            assert!(pages <= 5, "pagination must terminate");
        }
        assert_eq!(pages, 3);
        assert_eq!(
            seen,
            expected.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
        let first = store.page(&scope, 2, None, None).await.expect("first page");
        assert_eq!(
            MessageCursor::decode(first.next_cursor.as_deref().expect("cursor")).expect("decode"),
            MessageCursor::of(&expected[1])
        );
        let last = store
            .page(&scope, 2, first.next_cursor.as_deref(), None)
            .await
            .expect("second page")
            .next_cursor
            .expect("second cursor");
        let third = store
            .page(&scope, 2, Some(&last), None)
            .await
            .expect("third");
        assert_eq!(third.items.len(), 1);
        assert!(!third.has_more);
        assert!(third.next_cursor.is_none());
    }

    #[tokio::test]
    async fn identical_timestamps_page_in_ascending_id_order() {
        let store = store();
        let scope = scope();
        let mut expected = Vec::new();
        for text in ["a", "b", "c", "d", "e"] {
            expected.push(append_user(&store, &scope, text, 0).await);
        }
        expected.sort_by(|left, right| {
            (left.timestamp, left.id.as_str()).cmp(&(right.timestamp, right.id.as_str()))
        });
        let mut seen = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let page = store
                .page(&scope, 2, cursor.as_deref(), None)
                .await
                .expect("page");
            assert!(page.items.len() <= 2);
            seen.extend(page_ids(&page).into_iter().map(str::to_owned));
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        assert_eq!(
            seen,
            expected.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
        let mut unique = seen.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), seen.len(), "no row may repeat");
    }

    #[tokio::test]
    async fn appends_within_one_millisecond_page_without_duplicates_or_gaps() {
        let store = store();
        let scope = scope();
        let mut expected = Vec::new();
        for (index, text) in ["one", "two", "three", "four", "five"].iter().enumerate() {
            // One millisecond, six orders of magnitude of room: every row shares the
            // millisecond and its id suffix, so only the sub-millisecond stamp (and the
            // id tie-break) can order them.
            let timestamp = Utc
                .timestamp_opt(1_700_000_000, 123_000_000 + index as u32 * 1_000)
                .single()
                .expect("fixed time");
            let stored = store
                .append(&scope, &[BaseMessage::user(*text)], timestamp)
                .await
                .expect("append")
                .pop()
                .expect("one message");
            expected.push(stored);
        }
        let millis = expected[0].timestamp.timestamp_millis();
        assert!(expected
            .iter()
            .all(|message| message.timestamp.timestamp_millis() == millis));
        let mut seen = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0;
        loop {
            let page = store
                .page(&scope, 2, cursor.as_deref(), None)
                .await
                .expect("page");
            pages += 1;
            assert!(page.items.len() <= 2);
            seen.extend(page_ids(&page).into_iter().map(str::to_owned));
            match page.next_cursor {
                Some(next) => {
                    assert!(page.has_more);
                    let decoded = MessageCursor::decode(&next).expect("decode");
                    assert_eq!(
                        decoded,
                        MessageCursor::of(&expected[seen.len() - 1]),
                        "the token names the last returned row"
                    );
                    cursor = Some(next);
                }
                None => break,
            }
            assert!(pages <= 5, "pagination must terminate");
        }
        assert_eq!(pages, 3);
        assert_eq!(
            seen,
            expected.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn malformed_cursors_are_rejected_by_page() {
        let store = store();
        let scope = scope();
        append_user(&store, &scope, "one", 0).await;
        let valid = MessageCursor::new(time(0), "msg_x_1700000000000").encode();
        let truncated = &valid[..valid.len() - 2];
        for token in ["nonsense", truncated, "mc1.1.0000000000000000."] {
            let error = store
                .page(&scope, 2, Some(token), None)
                .await
                .expect_err(token);
            assert!(matches!(error, Error::Validation(_)), "{token}: {error:?}");
        }
    }

    #[tokio::test]
    async fn zero_page_size_and_empty_history_are_empty_pages() {
        let store = store();
        let scope = scope();
        append_user(&store, &scope, "one", 0).await;
        let zero = store.page(&scope, 0, None, None).await.expect("zero size");
        assert_eq!(zero, MessagePage::default());
        let missing = store
            .page(&MemoryScope::new("bob", "ws"), 5, None, None)
            .await
            .expect("missing history");
        assert_eq!(missing, MessagePage::default());
        assert!(!missing.has_more);
        assert!(missing.next_cursor.is_none());
    }

    #[tokio::test]
    async fn page_composes_the_scope_with_the_caller_filter() {
        let store = store();
        let scope = scope();
        let first = append_user(&store, &scope, "one", 0).await;
        let second = append_user(&store, &scope, "two", 1).await;
        let third = append_user(&store, &scope, "three", 2).await;
        append_user(
            &store,
            &MemoryScope::new("bob", "ws").session("s1"),
            "other user",
            1,
        )
        .await;
        append_user(
            &store,
            &MemoryScope::new("alice", "other").session("s1"),
            "other scope",
            1,
        )
        .await;
        let mut by_ids = MemoryFilter::new();
        by_ids.ids = vec![first.id.clone(), third.id.clone()];
        let page = store
            .page(&scope, 10, None, Some(&by_ids))
            .await
            .expect("page ids");
        assert_eq!(page_ids(&page), [first.id.as_str(), third.id.as_str()]);
        assert!(!page.has_more);
        assert!(page.next_cursor.is_none());
        let mut after = MemoryFilter::new();
        after.created_after = Some(time(0));
        let page = store
            .page(&scope, 10, None, Some(&after))
            .await
            .expect("page after");
        assert_eq!(page_ids(&page), [second.id.as_str(), third.id.as_str()]);
        // A conflicting scope clause narrows the page to nothing; it never widens it.
        let mut foreign = MemoryFilter::new();
        foreign.user_id = Some("bob".to_owned());
        let page = store
            .page(&scope, 10, None, Some(&foreign))
            .await
            .expect("page foreign");
        assert!(page.items.is_empty());
        assert!(!page.has_more);
    }

    #[tokio::test]
    async fn cursor_tokens_survive_a_store_reopen() {
        let directory = tempfile::tempdir().expect("tempdir");
        let root = directory.path().join("documents");
        let scope = scope();
        let cursor = {
            let document = Arc::new(
                LocalFileDocumentStore::new(&root)
                    .await
                    .expect("open documents"),
            );
            let store = MessageStore::new(document);
            for (offset, text) in [(0, "one"), (1, "two"), (2, "three")] {
                append_user(&store, &scope, text, offset).await;
            }
            let page = store.page(&scope, 2, None, None).await.expect("first page");
            assert!(page.has_more);
            page.next_cursor.expect("cursor")
        };
        // The token carries the position of the last returned row (`timestamp`, `id`), not
        // an offset, so it stays valid against a store instance this process did not open.
        assert!(cursor.starts_with("mc2."), "{cursor}");
        let reopened = MessageStore::new(Arc::new(
            LocalFileDocumentStore::new(&root)
                .await
                .expect("reopen documents"),
        ));
        // Appending an older row after the first page shifts every offset by one; an
        // offset-based resume would repeat `two`, a position-based one still yields `three`.
        append_user(&reopened, &scope, "zero", -1).await;
        let page = reopened
            .page(&scope, 2, Some(&cursor), None)
            .await
            .expect("resumed page");
        assert_eq!(
            page.items
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            ["three"]
        );
        assert!(!page.has_more);
    }
}
