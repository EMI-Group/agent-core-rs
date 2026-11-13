//! The on-disk session metadata schema: `sessions.json` and `state.data`.
//!
//! Ports `openjiuwen/core/session/session_controller/schema.py` (v0.1.20).
//! [`SessionMeta`] and [`ScopeSessionsMeta`] are the `sessions.json` shape,
//! [`SessionsIndex`] is the per-agent index, and [`SessionRecord`] is the
//! `state.data` document (`{"meta": …, "data": <snapshot>}`).
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::session::SessionSnapshot;

/// The default data container type recorded for new sessions (`"agent"`).
pub const DEFAULT_DATA_CONTAINER_TYPE: &str = "agent";

/// The smallest step used to keep metadata timestamps strictly monotonic.
pub(crate) const TIMESTAMP_EPSILON: f64 = 0.000_001;

/// Returns the current time as UTC seconds since the Unix epoch.
///
/// A clock set before the epoch yields `0.0`; that cannot happen on a
/// correctly configured host and keeps the signature infallible.
pub(crate) fn now_seconds() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64(),
        Err(_) => 0.0,
    }
}

/// Metadata for one persisted session (an entry in `sessions.json`).
///
/// Field names are the JSON contract (`snake_case`), unknown fields are
/// tolerated, and `data_container_type` falls back to
/// [`DEFAULT_DATA_CONTAINER_TYPE`] when absent, matching Python's `from_dict`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionMeta {
    /// The session identifier.
    pub session_id: String,
    /// Creation time, as UTC seconds since the Unix epoch.
    pub created_at: f64,
    /// Last update time, as UTC seconds since the Unix epoch.
    pub updated_at: f64,
    /// The session data version, bumped on every save.
    pub version: u64,
    /// Whether this is the active session in its scope; at most one per scope.
    pub is_active: bool,
    /// The data container type; [`DEFAULT_DATA_CONTAINER_TYPE`] for agents.
    pub data_container_type: String,
    /// An optional human-readable title.
    pub title: Option<String>,
    /// Opaque additional metadata.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for SessionMeta {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            created_at: 0.0,
            updated_at: 0.0,
            version: 1,
            is_active: false,
            data_container_type: DEFAULT_DATA_CONTAINER_TYPE.to_string(),
            title: None,
            metadata: BTreeMap::new(),
        }
    }
}

impl SessionMeta {
    /// Creates metadata for a new, active session starting at version 1.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = now_seconds();
        Self {
            session_id: session_id.into(),
            created_at: now,
            updated_at: now,
            is_active: true,
            ..Self::default()
        }
    }

    /// Marks the session as updated: bumps `updated_at` (strictly) and
    /// `version`.
    pub fn touch(&mut self) {
        self.updated_at = now_seconds().max(self.updated_at + TIMESTAMP_EPSILON);
        self.version = self.version.saturating_add(1);
    }
}

/// Bookkeeping for all sessions sharing one scope under an agent.
///
/// Sessions are kept newest-updated first; ties are broken by session id so
/// the order is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ScopeSessionsMeta {
    /// The [`super::SessionScopeKey`] string this collection belongs to.
    pub session_scope_key: String,
    /// The active session id in this scope, when one is active.
    pub active_session: Option<String>,
    /// All sessions in the scope, newest-updated first.
    pub sessions: Vec<SessionMeta>,
}

impl ScopeSessionsMeta {
    /// Creates an empty collection for `session_scope_key`.
    #[must_use]
    pub fn new(session_scope_key: impl Into<String>) -> Self {
        Self {
            session_scope_key: session_scope_key.into(),
            ..Self::default()
        }
    }

    /// Looks up a session's metadata by id.
    #[must_use]
    pub fn get_session(&self, session_id: &str) -> Option<&SessionMeta> {
        self.sessions
            .iter()
            .find(|session| session.session_id == session_id)
    }

    /// Inserts or replaces `meta`.
    ///
    /// An active `meta` deactivates every other session and becomes the
    /// scope's `active_session`; the list is then re-sorted.
    pub fn upsert(&mut self, meta: SessionMeta) {
        if meta.is_active {
            for session in &mut self.sessions {
                session.is_active = false;
            }
            self.active_session = Some(meta.session_id.clone());
        }
        match self
            .sessions
            .iter_mut()
            .find(|session| session.session_id == meta.session_id)
        {
            Some(existing) => *existing = meta,
            None => self.sessions.push(meta),
        }
        self.sort_sessions();
    }

    /// Removes a session's metadata, clearing `active_session` when it matched.
    pub fn remove(&mut self, session_id: &str) -> Option<SessionMeta> {
        let index = self
            .sessions
            .iter()
            .position(|session| session.session_id == session_id)?;
        let removed = self.sessions.remove(index);
        if self.active_session.as_deref() == Some(session_id) {
            self.active_session = None;
        }
        Some(removed)
    }

    /// Returns the active session's metadata, if any.
    #[must_use]
    pub fn active(&self) -> Option<&SessionMeta> {
        let active = self.active_session.as_deref()?;
        self.get_session(active)
    }

    /// Sorts the sessions newest-updated first (ties by session id).
    pub fn sort_sessions(&mut self) {
        self.sessions.sort_by(|left, right| {
            right
                .updated_at
                .total_cmp(&left.updated_at)
                .then_with(|| left.session_id.cmp(&right.session_id))
        });
    }
}

/// The per-agent index persisted as `sessions/sessions.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionsIndex {
    /// The agent this index belongs to.
    pub agent_id: String,
    /// Last update time, as UTC seconds since the Unix epoch.
    pub updated_at: f64,
    /// Session collections keyed by [`super::SessionScopeKey`] string.
    pub scopes: BTreeMap<String, ScopeSessionsMeta>,
}

/// One persisted session: its metadata plus the snapshot stored in
/// `state.data`.
///
/// The snapshot is serialized under the `data` key, so the JSON document is
/// exactly `{"meta": …, "data": <SessionSnapshot>}` and a `data` payload loads
/// directly through [`crate::Session::restore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionRecord {
    /// Session metadata.
    pub meta: SessionMeta,
    /// The full session snapshot.
    #[serde(rename = "data")]
    pub snapshot: SessionSnapshot,
}

impl SessionRecord {
    /// Creates a record from `meta` and `snapshot`.
    #[must_use]
    pub fn new(meta: SessionMeta, snapshot: SessionSnapshot) -> Self {
        Self { meta, snapshot }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use serde_json::json;

    use super::{ScopeSessionsMeta, SessionMeta, DEFAULT_DATA_CONTAINER_TYPE};

    #[test]
    fn new_meta_is_active_at_version_one() {
        let meta = SessionMeta::new("s1");
        assert_eq!(meta.session_id, "s1");
        assert_eq!(meta.version, 1);
        assert!(meta.is_active);
        assert_eq!(meta.data_container_type, DEFAULT_DATA_CONTAINER_TYPE);
        assert!(meta.created_at > 0.0);
    }

    #[test]
    fn touch_bumps_version_and_timestamp_strictly() {
        let mut meta = SessionMeta::new("s1");
        let before = meta.updated_at;
        meta.touch();
        assert_eq!(meta.version, 2);
        assert!(meta.updated_at > before);
    }

    #[test]
    fn missing_container_type_defaults_when_deserializing() {
        let meta: SessionMeta =
            serde_json::from_value(json!({"session_id": "s1", "version": 3})).unwrap();
        assert_eq!(meta.data_container_type, DEFAULT_DATA_CONTAINER_TYPE);
    }

    #[test]
    fn upsert_deactivates_other_sessions_and_sorts() {
        let mut scope = ScopeSessionsMeta::new("agent:a:main");
        let mut older = SessionMeta::new("older");
        older.updated_at = 10.0;
        older.is_active = false;
        scope.upsert(older);
        let mut newer = SessionMeta::new("newer");
        newer.updated_at = 20.0;
        newer.is_active = true;
        scope.upsert(newer);
        let ids: Vec<&str> = scope
            .sessions
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        assert_eq!(ids, ["newer", "older"]);
        assert_eq!(scope.active_session.as_deref(), Some("newer"));
        assert_eq!(scope.active().unwrap().session_id, "newer");
        assert!(!scope.get_session("older").unwrap().is_active);
    }

    #[test]
    fn remove_clears_active_session() {
        let mut scope = ScopeSessionsMeta::new("agent:a:main");
        scope.upsert(SessionMeta::new("s1"));
        assert!(scope.remove("s1").is_some());
        assert!(scope.remove("s1").is_none());
        assert!(scope.active_session.is_none());
    }
}
