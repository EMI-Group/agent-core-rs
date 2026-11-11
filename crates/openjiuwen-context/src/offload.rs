//! Offload storage for content that leaves the context window.
//!
//! When a tool result is too large to keep in the prompt, the processor writes
//! the original message to an [`OffloadStore`] and replaces it with a short
//! placeholder carrying an [`OffloadMarker`]. Markers keep Python's shape:
//!
//! ```text
//! [[OFFLOAD: handle=3f2a..., type=in_memory]]
//! [[OFFLOAD: handle=3f2a..., type=filesystem, path=/tmp/…]]
//! ```
//!
//! Three stores are provided: an in-process [`InMemoryOffloadStore`], a
//! [`KvOffloadStore`] over foundation's key-value abstraction, and a
//! [`SessionOffloadStore`] that persists through a session's global state. Every
//! key is scoped by `(session_id, context_id, handle)`, so concurrent sessions
//! cannot read each other's payloads even when handle ids collide.
use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::store::{KvStore, KvValue};
use openjiuwen_session::Session;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Prefix that opens an offload marker.
pub const OFFLOAD_MARKER_PREFIX: &str = "[[OFFLOAD:";

/// Where an offloaded record lives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadKind {
    /// In-process memory; lost when the engine drops.
    #[default]
    InMemory,
    /// Any durable backend (key-value store, session state, filesystem).
    Filesystem,
}

impl OffloadKind {
    /// The marker spelling (`in_memory` / `filesystem`).
    #[must_use]
    pub fn as_marker_str(self) -> &'static str {
        match self {
            Self::InMemory => "in_memory",
            Self::Filesystem => "filesystem",
        }
    }

    /// Parses a marker spelling.
    #[must_use]
    pub fn from_marker_str(value: &str) -> Option<Self> {
        match value {
            "in_memory" => Some(Self::InMemory),
            "filesystem" => Some(Self::Filesystem),
            _ => None,
        }
    }
}

/// Identity of the context an offloaded record belongs to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OffloadScope {
    /// Owning session.
    pub session_id: String,
    /// Owning context within the session.
    pub context_id: String,
}

impl OffloadScope {
    /// Creates a scope from a session and context id.
    #[must_use]
    pub fn new(session_id: impl Into<String>, context_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            context_id: context_id.into(),
        }
    }
}

/// One offloaded unit of history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OffloadRecord {
    /// Opaque handle referenced by the marker.
    pub handle: String,
    /// Where the record is stored.
    pub kind: OffloadKind,
    /// The messages removed from the window.
    pub messages: Vec<BaseMessage>,
    /// Free-form annotations (for example the tool name or original length).
    pub metadata: Map<String, Value>,
}

impl Default for OffloadRecord {
    fn default() -> Self {
        Self {
            handle: String::new(),
            kind: OffloadKind::InMemory,
            messages: Vec::new(),
            metadata: Map::new(),
        }
    }
}

/// The placeholder text that points at an offloaded record.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OffloadMarker {
    /// Opaque handle of the record.
    pub handle: String,
    /// Where the record is stored.
    pub kind: OffloadKind,
    /// Filesystem path, when the record lives on disk.
    pub path: Option<String>,
}

impl OffloadMarker {
    /// Renders the marker exactly as Python writes it.
    #[must_use]
    pub fn to_marker_string(&self) -> String {
        match (&self.kind, &self.path) {
            (OffloadKind::Filesystem, Some(path)) => format!(
                "{OFFLOAD_MARKER_PREFIX} handle={}, type=filesystem, path={path}]]",
                self.handle
            ),
            (kind, _) => format!(
                "{OFFLOAD_MARKER_PREFIX} handle={}, type={}]]",
                self.handle,
                kind.as_marker_str()
            ),
        }
    }

    /// Parses the first marker found anywhere in `text`.
    ///
    /// Returns `None` when there is no marker, or when the marker does not carry
    /// a handle and a known type.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let start = text.find(OFFLOAD_MARKER_PREFIX)?;
        let inner_start = start + OFFLOAD_MARKER_PREFIX.len();
        let rest = &text[inner_start..];
        let end = rest.find("]]")?;
        let inner = &rest[..end];

        let mut handle = None;
        let mut kind = None;
        let mut path = None;
        for field in inner.split(',') {
            let Some((key, value)) = field.split_once('=') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "handle" if !value.is_empty() => handle = Some(value.to_owned()),
                "type" => kind = OffloadKind::from_marker_str(value),
                "path" if !value.is_empty() => path = Some(value.to_owned()),
                _ => {}
            }
        }
        Some(Self {
            handle: handle?,
            kind: kind?,
            path,
        })
    }
}

/// Durable storage for offloaded records.
#[async_trait]
pub trait OffloadStore: Send + Sync {
    /// Which storage class this store represents.
    fn kind(&self) -> OffloadKind;

    /// Stores `record` under `scope` and the record's handle.
    ///
    /// # Errors
    ///
    /// Returns a backend error (serialization, I/O, or session persistence).
    async fn put(&self, scope: &OffloadScope, record: OffloadRecord) -> Result<()>;

    /// Reads the record for `handle`, or `None` when it was removed.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the entry exists but cannot be decoded.
    async fn get(&self, scope: &OffloadScope, handle: &str) -> Result<Option<OffloadRecord>>;

    /// Removes the record for `handle`; removing a missing record is a no-op.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the removal cannot be persisted.
    async fn remove(&self, scope: &OffloadScope, handle: &str) -> Result<()>;
}

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`, so a scope can be
/// embedded in a storage key or a session state path.
#[must_use]
pub fn sanitize_key(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// The isolation key for one `(scope, handle)` pair.
fn isolation_key(scope: &OffloadScope, handle: &str) -> String {
    format!(
        "{}_{}_{}",
        sanitize_key(&scope.session_id),
        sanitize_key(&scope.context_id),
        sanitize_key(handle)
    )
}

/// Process-local offload storage, used by tests and short-lived engines.
#[derive(Debug, Default)]
pub struct InMemoryOffloadStore {
    records: RwLock<HashMap<String, OffloadRecord>>,
}

impl InMemoryOffloadStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether the store holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl OffloadStore for InMemoryOffloadStore {
    fn kind(&self) -> OffloadKind {
        OffloadKind::InMemory
    }

    async fn put(&self, scope: &OffloadScope, record: OffloadRecord) -> Result<()> {
        let key = isolation_key(scope, &record.handle);
        self.records
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, record);
        Ok(())
    }

    async fn get(&self, scope: &OffloadScope, handle: &str) -> Result<Option<OffloadRecord>> {
        let key = isolation_key(scope, handle);
        Ok(self
            .records
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned())
    }

    async fn remove(&self, scope: &OffloadScope, handle: &str) -> Result<()> {
        let key = isolation_key(scope, handle);
        self.records
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&key);
        Ok(())
    }
}

/// Offload storage backed by foundation's [`KvStore`].
#[derive(Clone)]
pub struct KvOffloadStore {
    store: Arc<dyn KvStore>,
    prefix: String,
}

impl std::fmt::Debug for KvOffloadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvOffloadStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl KvOffloadStore {
    /// The default key prefix.
    pub const DEFAULT_PREFIX: &'static str = "context:offload:";

    /// Wraps `store` with the default key prefix.
    #[must_use]
    pub fn new(store: Arc<dyn KvStore>) -> Self {
        Self::with_prefix(store, Self::DEFAULT_PREFIX)
    }

    /// Wraps `store` with an explicit key prefix.
    #[must_use]
    pub fn with_prefix(store: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
        }
    }

    /// The storage key for one `(scope, handle)` pair.
    #[must_use]
    pub fn key(&self, scope: &OffloadScope, handle: &str) -> String {
        format!("{}{}", self.prefix, isolation_key(scope, handle))
    }
}

#[async_trait]
impl OffloadStore for KvOffloadStore {
    fn kind(&self) -> OffloadKind {
        OffloadKind::Filesystem
    }

    async fn put(&self, scope: &OffloadScope, record: OffloadRecord) -> Result<()> {
        let key = self.key(scope, &record.handle);
        let bytes = serde_json::to_vec(&record).map_err(|error| {
            openjiuwen_common::error::Error::Validation(format!(
                "cannot serialize offload record: {error}"
            ))
        })?;
        self.store.set(&key, KvValue::bytes(bytes)).await
    }

    async fn get(&self, scope: &OffloadScope, handle: &str) -> Result<Option<OffloadRecord>> {
        let key = self.key(scope, handle);
        let Some(value) = self.store.get(&key).await? else {
            return Ok(None);
        };
        let record = serde_json::from_slice(value.as_bytes()).map_err(|error| {
            openjiuwen_common::error::Error::Validation(format!(
                "cannot decode offload record: {error}"
            ))
        })?;
        Ok(Some(record))
    }

    async fn remove(&self, scope: &OffloadScope, handle: &str) -> Result<()> {
        self.store.delete(&self.key(scope, handle)).await
    }
}

/// Offload storage persisted in a session's global state.
pub struct SessionOffloadStore {
    session: Arc<dyn Session>,
}

impl std::fmt::Debug for SessionOffloadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionOffloadStore")
            .field("session_id", &self.session.id())
            .finish_non_exhaustive()
    }
}

impl SessionOffloadStore {
    /// The session-state path prefix records live under.
    pub const STATE_PREFIX: &'static str = "context_offload";

    /// Wraps `session`.
    #[must_use]
    pub fn new(session: Arc<dyn Session>) -> Self {
        Self { session }
    }

    /// The session-state path for one `(scope, handle)` pair.
    #[must_use]
    pub fn key(&self, scope: &OffloadScope, handle: &str) -> String {
        format!("{}.{}", Self::STATE_PREFIX, isolation_key(scope, handle))
    }

    /// Writes `value` under `key` in the session's global state.
    async fn write(&self, key: &str, value: Value) -> Result<()> {
        let patch = Value::Object(Map::from_iter([(key.to_owned(), value)]));
        self.session.update_state(patch).await
    }
}

#[async_trait]
impl OffloadStore for SessionOffloadStore {
    fn kind(&self) -> OffloadKind {
        OffloadKind::Filesystem
    }

    async fn put(&self, scope: &OffloadScope, record: OffloadRecord) -> Result<()> {
        let key = self.key(scope, &record.handle);
        let value = serde_json::to_value(&record).map_err(|error| {
            openjiuwen_common::error::Error::Validation(format!(
                "cannot serialize offload record: {error}"
            ))
        })?;
        self.write(&key, value).await
    }

    async fn get(&self, scope: &OffloadScope, handle: &str) -> Result<Option<OffloadRecord>> {
        let key = self.key(scope, handle);
        let Some(value) = self.session.get_state(&key).await? else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        let record = serde_json::from_value(value).map_err(|error| {
            openjiuwen_common::error::Error::Validation(format!(
                "cannot decode offload record: {error}"
            ))
        })?;
        Ok(Some(record))
    }

    async fn remove(&self, scope: &OffloadScope, handle: &str) -> Result<()> {
        let key = self.key(scope, handle);
        self.write(&key, Value::Null).await
    }
}

/// Reads the record a marker in `text` points at.
///
/// Returns `None` when `text` carries no marker. When the marker exists but the
/// record was removed, the miss is logged and `Ok(None)` is returned so a caller
/// can keep the placeholder instead of failing the whole request.
///
/// # Errors
///
/// Propagates backend errors from [`OffloadStore::get`].
pub async fn rehydrate(
    store: &dyn OffloadStore,
    scope: &OffloadScope,
    text: &str,
) -> Result<Option<Vec<BaseMessage>>> {
    let Some(marker) = OffloadMarker::parse(text) else {
        return Ok(None);
    };
    let Some(record) = store.get(scope, &marker.handle).await? else {
        tracing::warn!(
            handle = marker.handle.as_str(),
            session_id = scope.session_id.as_str(),
            context_id = scope.context_id.as_str(),
            "offload record is missing; keeping the placeholder"
        );
        return Ok(None);
    };
    Ok(Some(record.messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::store::InMemoryKvStore;
    use openjiuwen_session::AgentSession;

    fn scope(session: &str, context: &str) -> OffloadScope {
        OffloadScope::new(session, context)
    }

    fn record(handle: &str, text: &str) -> OffloadRecord {
        OffloadRecord {
            handle: handle.to_owned(),
            kind: OffloadKind::InMemory,
            messages: vec![BaseMessage::tool(handle, text)],
            metadata: Map::new(),
        }
    }

    #[test]
    fn marker_format_and_parse_round_trip() {
        let in_memory = OffloadMarker {
            handle: "abc123".to_owned(),
            kind: OffloadKind::InMemory,
            path: None,
        };
        assert_eq!(
            in_memory.to_marker_string(),
            "[[OFFLOAD: handle=abc123, type=in_memory]]"
        );

        let on_disk = OffloadMarker {
            handle: "abc123".to_owned(),
            kind: OffloadKind::Filesystem,
            path: Some("/tmp/out.txt".to_owned()),
        };
        assert_eq!(
            on_disk.to_marker_string(),
            "[[OFFLOAD: handle=abc123, type=filesystem, path=/tmp/out.txt]]"
        );

        let surrounded = format!("before {} after", on_disk.to_marker_string());
        assert_eq!(OffloadMarker::parse(&surrounded), Some(on_disk));
        assert_eq!(
            OffloadMarker::parse(&in_memory.to_marker_string()),
            Some(in_memory)
        );

        assert_eq!(OffloadMarker::parse("no marker here"), None);
        assert_eq!(OffloadMarker::parse("[[OFFLOAD: handle=x]]"), None);
        assert_eq!(
            OffloadMarker::parse("[[OFFLOAD: handle=, type=filesystem]]"),
            None
        );
        assert_eq!(
            OffloadMarker::parse("[[OFFLOAD: handle=x, type=network]]"),
            None
        );
    }

    #[tokio::test]
    async fn in_memory_store_round_trips_and_scopes_records() {
        let store = InMemoryOffloadStore::new();
        assert!(store.is_empty());
        assert_eq!(store.kind(), OffloadKind::InMemory);

        let first = scope("s1", "main");
        let second = scope("s2", "main");
        store.put(&first, record("h1", "one")).await.unwrap();
        store.put(&second, record("h1", "two")).await.unwrap();
        assert_eq!(store.len(), 2);

        let a = store.get(&first, "h1").await.unwrap().unwrap();
        let b = store.get(&second, "h1").await.unwrap().unwrap();
        assert_eq!(a.messages[0].text(), "one");
        assert_eq!(b.messages[0].text(), "two");

        store.remove(&first, "h1").await.unwrap();
        assert!(store.get(&first, "h1").await.unwrap().is_none());
        assert!(store.get(&second, "h1").await.unwrap().is_some());
        store.remove(&first, "missing").await.unwrap();
    }

    #[tokio::test]
    async fn kv_store_round_trips_and_scopes_records() {
        let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
        let store = KvOffloadStore::new(Arc::clone(&kv));
        assert_eq!(store.kind(), OffloadKind::Filesystem);

        let first = scope("s1", "main");
        let second = scope("s2", "main");
        store.put(&first, record("h1", "one")).await.unwrap();
        store.put(&second, record("h1", "two")).await.unwrap();

        assert_eq!(
            store.get(&first, "h1").await.unwrap().unwrap().messages[0].text(),
            "one"
        );
        assert_eq!(
            store.get(&second, "h1").await.unwrap().unwrap().messages[0].text(),
            "two"
        );
        assert_eq!(store.key(&first, "h1"), "context:offload:s1_main_h1");

        store.remove(&first, "h1").await.unwrap();
        assert!(store.get(&first, "h1").await.unwrap().is_none());
        assert!(store.get(&second, "h1").await.unwrap().is_some());
        assert!(store.get(&first, "absent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn session_store_round_trips_through_session_state() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("sess-1"));
        let store = SessionOffloadStore::new(Arc::clone(&session));
        assert_eq!(store.kind(), OffloadKind::Filesystem);

        let scoped = scope("sess-1", "main");
        store.put(&scoped, record("h1", "payload")).await.unwrap();
        let loaded = store.get(&scoped, "h1").await.unwrap().unwrap();
        assert_eq!(loaded.messages[0].text(), "payload");
        assert_eq!(store.key(&scoped, "h1"), "context_offload.sess-1_main_h1");

        // A different context cannot see the record.
        assert!(store
            .get(&scope("sess-1", "other"), "h1")
            .await
            .unwrap()
            .is_none());

        store.remove(&scoped, "h1").await.unwrap();
        assert!(store.get(&scoped, "h1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rehydrate_returns_messages_or_none() {
        let store = InMemoryOffloadStore::new();
        let scoped = scope("s", "main");
        let record = record("h1", "original output");
        let marker = OffloadMarker {
            handle: record.handle.clone(),
            kind: OffloadKind::InMemory,
            path: None,
        }
        .to_marker_string();
        store.put(&scoped, record).await.unwrap();

        let text = format!("trimmed head\n...\n{marker}\n</persisted-output>");
        let messages = rehydrate(&store, &scoped, &text).await.unwrap().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "original output");

        assert!(rehydrate(&store, &scoped, "no marker")
            .await
            .unwrap()
            .is_none());
        assert!(
            rehydrate(&store, &scoped, "[[OFFLOAD: handle=gone, type=in_memory]]")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sanitizes_keys_without_losing_safe_characters() {
        assert_eq!(sanitize_key("a-b_c.1/2:3"), "a-b_c_1_2_3");
        assert_eq!(sanitize_key(""), "");
    }
}
