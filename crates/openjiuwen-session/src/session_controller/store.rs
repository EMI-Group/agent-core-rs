//! Disk-backed session storage and the per-agent controller.
//!
//! Ports the storage half of `openjiuwen/core/session/session_controller/`
//! (v0.1.20): [`DiskSessionStore`] owns the layout
//! `<base>/<agent_id>/sessions/…`, and [`SessionController`] keeps Python's
//! per-agent in-memory cache in front of it.
//!
//! Layout:
//!
//! - `<base>/<agent_id>/sessions/sessions.json` — the [`SessionsIndex`], keyed
//!   by [`SessionScopeKey`].
//! - `<base>/<agent_id>/sessions/<session_id>/state.data` — a [`SessionRecord`]
//!   (`{"meta": …, "data": <SessionSnapshot>}`).
//! - `<base>/<agent_id>/sessions/<session_id>/history/` — version history, owned
//!   by [`crate::vcs::SessionHistory`].
//! - `<base>/<agent_id>/links.json` — the agent's session link graph
//!   ([`SessionLinks`]), described in [`super::links`].
//!
//! Every write goes through a temporary file in the destination directory and a
//! rename, so readers never observe a partially written document and a failed
//! write leaves no `.tmp` file behind.
use std::collections::BTreeMap;
use std::io::ErrorKind as IoErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;

use crate::checkpoint::validate_id;
use crate::session::{Session, SessionSnapshot, SessionType};
use crate::vcs::SessionHistory;

use super::links::{links_path, read_links, update_links, write_links, SessionLinks};
use super::schema::{
    now_seconds, ScopeSessionsMeta, SessionMeta, SessionRecord, SessionsIndex,
    DEFAULT_DATA_CONTAINER_TYPE, TIMESTAMP_EPSILON,
};
use super::scope::{SessionScope, SessionScopeKey};

/// A session store rooted at a base directory.
///
/// The store is cheap to clone: clones share the same base path, the same
/// version-history commit lock, the same link-graph write lock, and the same
/// session-index write lock, so concurrent writers coordinate through one mutex
/// each.
///
/// Two stores constructed independently over the same base directory do not
/// share those locks, so writers over one base must be store clones.
#[derive(Debug, Clone)]
pub struct DiskSessionStore {
    base: PathBuf,
    history_lock: Arc<AsyncMutex<()>>,
    links_lock: Arc<AsyncMutex<()>>,
    index_lock: Arc<AsyncMutex<()>>,
}

/// The file name of the per-agent session index.
const SESSIONS_INDEX_FILE: &str = "sessions.json";
/// The directory holding per-session state files.
const SESSIONS_DIR: &str = "sessions";
/// The file name of a session's state document.
const STATE_FILE: &str = "state.data";
/// The directory holding a session's version history.
const HISTORY_DIR: &str = "history";

impl DiskSessionStore {
    /// Creates a store rooted at `base`; directories are created lazily on the
    /// first write.
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self {
            base: base.into(),
            history_lock: Arc::new(AsyncMutex::new(())),
            links_lock: Arc::new(AsyncMutex::new(())),
            index_lock: Arc::new(AsyncMutex::new(())),
        }
    }

    /// The store's base directory.
    #[must_use]
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// `<base>/<agent_id>`.
    #[must_use]
    pub fn agent_dir(&self, agent_id: &str) -> PathBuf {
        self.base.join(agent_id)
    }

    /// `<base>/<agent_id>/sessions`.
    #[must_use]
    pub fn sessions_dir(&self, agent_id: &str) -> PathBuf {
        self.agent_dir(agent_id).join(SESSIONS_DIR)
    }

    /// `<base>/<agent_id>/sessions/sessions.json`.
    #[must_use]
    pub fn sessions_index_path(&self, agent_id: &str) -> PathBuf {
        self.sessions_dir(agent_id).join(SESSIONS_INDEX_FILE)
    }

    /// `<base>/<agent_id>/sessions/<session_id>`.
    #[must_use]
    pub fn session_dir(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.sessions_dir(agent_id).join(session_id)
    }

    /// `<session_dir>/state.data`.
    #[must_use]
    pub fn state_path(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.session_dir(agent_id, session_id).join(STATE_FILE)
    }

    /// `<session_dir>/history`, the root of the session's version history.
    #[must_use]
    pub fn history_dir(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.session_dir(agent_id, session_id).join(HISTORY_DIR)
    }

    /// Returns the version history facade over this store.
    #[must_use]
    pub fn history(&self) -> SessionHistory {
        SessionHistory::new(self.clone())
    }

    /// `<base>/<agent_id>/links.json`, the agent's session link graph.
    #[must_use]
    pub fn links_path(&self, agent_id: &str) -> PathBuf {
        links_path(&self.base, agent_id)
    }

    /// The lock that serializes version-id allocation across store clones.
    pub(crate) fn history_lock(&self) -> Arc<AsyncMutex<()>> {
        Arc::clone(&self.history_lock)
    }

    /// Persists `record` under `scope` and returns the stored metadata.
    ///
    /// A new session starts at version 1, becomes the scope's active session,
    /// and deactivates its siblings. An existing record keeps `created_at` and
    /// its version is bumped. Timestamps are made strictly monotonic across
    /// saves for one agent, so "newest first" listings are deterministic.
    ///
    /// The index update is a read-modify-write cycle serialized across store
    /// clones, so concurrent saves of distinct sessions keep every entry; both
    /// the state document and the index are written atomically.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent or session id, and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// or serialization failures.
    pub async fn save(
        &self,
        agent_id: &str,
        scope: &SessionScope,
        record: SessionRecord,
    ) -> Result<SessionMeta> {
        validate_id("agent", agent_id)?;
        validate_id("session", &record.meta.session_id)?;

        // The index is a read-modify-write document: without this lock two
        // concurrent saves of distinct sessions could each read the same index
        // and the later write would drop the other's entry.
        let _index_guard = self.index_lock.lock().await;

        let mut index = self.read_index(agent_id).await?;
        let now = now_seconds().max(index.updated_at + TIMESTAMP_EPSILON);
        let scope_key = SessionScopeKey {
            agent_id: agent_id.to_string(),
            scope: scope.clone(),
        }
        .to_string();

        let scope_meta = index
            .scopes
            .entry(scope_key.clone())
            .or_insert_with(|| ScopeSessionsMeta::new(scope_key));

        let mut meta = if let Some(existing) = scope_meta.get_session(&record.meta.session_id) {
            let mut meta = existing.clone();
            meta.title.clone_from(&record.meta.title);
            meta.metadata.clone_from(&record.meta.metadata);
            if !record.meta.data_container_type.is_empty() {
                meta.data_container_type
                    .clone_from(&record.meta.data_container_type);
            }
            meta.touch();
            meta
        } else {
            let mut meta = record.meta.clone();
            if meta.created_at <= 0.0 {
                meta.created_at = now;
            }
            meta.version = meta.version.max(1);
            if meta.data_container_type.is_empty() {
                DEFAULT_DATA_CONTAINER_TYPE.clone_into(&mut meta.data_container_type);
            }
            meta
        };
        meta.updated_at = now;
        meta.is_active = true;
        let mut snapshot = record.snapshot;
        if snapshot.session_id.is_empty() {
            snapshot.session_id = meta.session_id.clone();
        }
        let stored = SessionRecord::new(meta.clone(), snapshot);

        scope_meta.upsert(meta.clone());
        index.updated_at = now;

        self.write_record(agent_id, &stored).await?;
        self.write_index(agent_id, &index).await?;
        Ok(meta)
    }

    /// Persists `snapshot` for `session_id` under `scope`.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::save`].
    pub async fn save_snapshot(
        &self,
        agent_id: &str,
        scope: &SessionScope,
        session_id: &str,
        snapshot: SessionSnapshot,
        title: Option<&str>,
    ) -> Result<SessionMeta> {
        let mut meta = SessionMeta::new(session_id);
        meta.title = title.map(str::to_string);
        let mut snapshot = snapshot;
        if snapshot.session_id.is_empty() {
            snapshot.session_id = session_id.to_string();
        }
        self.save(agent_id, scope, SessionRecord::new(meta, snapshot))
            .await
    }

    /// Snapshots `session` and persists it under `scope`.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::snapshot`] failures and [`DiskSessionStore::save`]
    /// errors.
    pub async fn save_from_session(
        &self,
        agent_id: &str,
        scope: &SessionScope,
        session: &dyn Session,
        title: Option<&str>,
    ) -> Result<SessionMeta> {
        let snapshot = session.snapshot().await?;
        self.save_snapshot(agent_id, scope, session.id(), snapshot, title)
            .await
    }

    /// Loads a session record, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// stored document is unreadable or malformed.
    pub async fn load(&self, agent_id: &str, session_id: &str) -> Result<Option<SessionRecord>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        read_json_optional(&self.state_path(agent_id, session_id)).await
    }

    /// Loads just the snapshot of a session, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::load`].
    pub async fn load_snapshot(
        &self,
        agent_id: &str,
        session_id: &str,
    ) -> Result<Option<SessionSnapshot>> {
        Ok(self
            .load(agent_id, session_id)
            .await?
            .map(|record| record.snapshot))
    }

    /// Loads a stored session into `session` via [`Session::restore`].
    ///
    /// Returns `false` when no record exists.
    ///
    /// # Errors
    ///
    /// Propagates [`DiskSessionStore::load`] and [`Session::restore`] failures.
    pub async fn load_into_session(
        &self,
        agent_id: &str,
        session_id: &str,
        session: &dyn Session,
    ) -> Result<bool> {
        match self.load(agent_id, session_id).await? {
            Some(record) => {
                session.restore(record.snapshot).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Returns whether a state document exists for the session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// filesystem lookup fails for a reason other than absence.
    pub async fn exists(&self, agent_id: &str, session_id: &str) -> Result<bool> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        match tokio::fs::metadata(self.state_path(agent_id, session_id)).await {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == IoErrorKind::NotFound => Ok(false),
            Err(error) => Err(io_error(
                "stat",
                &self.state_path(agent_id, session_id),
                &error,
            )),
        }
    }

    /// Deletes a session's directory, its index entry, and its link entry.
    ///
    /// Returns `false` when the session directory does not exist; nothing is
    /// changed in that case.
    ///
    /// Only that session's [`SessionLink`](super::links::SessionLink) is dropped:
    /// its parent stops listing it as a child, its children lose the parent
    /// pointer (see [`SessionLinks::remove`]), and every other entry survives.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures.
    pub async fn delete(&self, agent_id: &str, session_id: &str) -> Result<bool> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let session_dir = self.session_dir(agent_id, session_id);
        match tokio::fs::metadata(&session_dir).await {
            Ok(_) => {}
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(io_error("stat", &session_dir, &error)),
        }
        tokio::fs::remove_dir_all(&session_dir)
            .await
            .map_err(|error| io_error("remove", &session_dir, &error))?;

        let index_path = self.sessions_index_path(agent_id);
        if tokio::fs::metadata(&index_path).await.is_ok() {
            // Same read-modify-write lock as `save`: a concurrent save must not
            // resurrect an entry this delete just removed (or vice versa).
            let _index_guard = self.index_lock.lock().await;
            let mut index = self.read_index(agent_id).await?;
            let mut changed = false;
            index.scopes.retain(|_, scope| {
                if scope.remove(session_id).is_some() {
                    changed = true;
                }
                !scope.sessions.is_empty()
            });
            if changed {
                index.updated_at = now_seconds().max(index.updated_at + TIMESTAMP_EPSILON);
                self.write_index(agent_id, &index).await?;
            }
        }

        let links_path = self.links_path(agent_id);
        if tokio::fs::metadata(&links_path).await.is_ok() {
            let _guard = self.links_lock.lock().await;
            let mut links = self.read_links(agent_id).await?;
            if links.remove(session_id).is_some() {
                self.write_links(agent_id, &links).await?;
            }
        }
        Ok(true)
    }

    /// Lists session metadata, newest-updated first.
    ///
    /// With `Some(scope)` only that scope's sessions are returned; with `None`
    /// every session of the agent is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// index is unreadable or malformed.
    pub async fn list(
        &self,
        agent_id: &str,
        scope: Option<&SessionScope>,
    ) -> Result<Vec<SessionMeta>> {
        validate_id("agent", agent_id)?;
        let index = self.read_index(agent_id).await?;
        let mut sessions = match scope {
            Some(scope) => {
                let key = SessionScopeKey {
                    agent_id: agent_id.to_string(),
                    scope: scope.clone(),
                }
                .to_string();
                index
                    .scopes
                    .get(&key)
                    .map(|scope| scope.sessions.clone())
                    .unwrap_or_default()
            }
            None => index
                .scopes
                .values()
                .flat_map(|scope| scope.sessions.iter().cloned())
                .collect(),
        };
        sessions.sort_by(|left, right| {
            right
                .updated_at
                .total_cmp(&left.updated_at)
                .then_with(|| left.session_id.cmp(&right.session_id))
        });
        Ok(sessions)
    }

    /// Lists every session metadata of an agent (see [`DiskSessionStore::list`]).
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::list`].
    pub async fn list_all(&self, agent_id: &str) -> Result<Vec<SessionMeta>> {
        self.list(agent_id, None).await
    }

    /// Lists the agent ids that have a `sessions/sessions.json` under the base
    /// directory, sorted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`](openjiuwen_common::error::ErrorKind::Session)
    /// when the base directory exists but cannot be read.
    pub async fn list_agents(&self) -> Result<Vec<String>> {
        let mut reader = match tokio::fs::read_dir(&self.base).await {
            Ok(reader) => reader,
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error("read dir", &self.base, &error)),
        };
        let mut agents = Vec::new();
        loop {
            let entry = match reader.next_entry().await {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => return Err(io_error("read dir", &self.base, &error)),
            };
            let path = entry.path();
            let is_dir = match entry.file_type().await {
                Ok(file_type) => file_type.is_dir(),
                Err(error) => return Err(io_error("stat", &path, &error)),
            };
            if !is_dir {
                continue;
            }
            let index_path = path.join(SESSIONS_DIR).join(SESSIONS_INDEX_FILE);
            if tokio::fs::metadata(&index_path).await.is_ok() {
                if let Some(name) = entry.file_name().to_str() {
                    agents.push(name.to_string());
                }
            }
        }
        agents.sort();
        Ok(agents)
    }

    /// Reads the agent's session index; a missing index yields an empty one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// index is unreadable or malformed.
    pub async fn read_index(&self, agent_id: &str) -> Result<SessionsIndex> {
        validate_id("agent", agent_id)?;
        Ok(read_json_optional(&self.sessions_index_path(agent_id))
            .await?
            .unwrap_or_else(|| SessionsIndex {
                agent_id: agent_id.to_string(),
                updated_at: 0.0,
                scopes: BTreeMap::new(),
            }))
    }

    /// Atomically writes the agent's session index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// or serialization failures.
    pub async fn write_index(&self, agent_id: &str, index: &SessionsIndex) -> Result<()> {
        validate_id("agent", agent_id)?;
        write_json_atomic(&self.sessions_index_path(agent_id), index).await
    }

    /// Atomically writes one session's `state.data`.
    async fn write_record(&self, agent_id: &str, record: &SessionRecord) -> Result<()> {
        write_json_atomic(&self.state_path(agent_id, &record.meta.session_id), record).await
    }

    /// Reads the agent's session link graph; a missing document yields an empty
    /// graph.
    ///
    /// The document is returned as stored, so a hand-written cycle surfaces when
    /// it is traversed (see [`SessionLinks::chain`]) rather than on read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when
    /// `links.json` is unreadable or malformed.
    pub async fn read_links(&self, agent_id: &str) -> Result<SessionLinks> {
        read_links(&self.base, agent_id).await
    }

    /// Atomically writes the agent's session link graph (`links.json`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent or session id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// or serialization failures.
    pub async fn write_links(&self, agent_id: &str, links: &SessionLinks) -> Result<()> {
        write_links(&self.base, agent_id, links).await
    }

    /// Records `child` as created from `parent`, then persists the graph.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id or a link that would create a cycle, and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures or a stored cycle.
    pub async fn link_parent(&self, agent_id: &str, child: &str, parent: &str) -> Result<()> {
        update_links(&self.base, &self.links_lock, agent_id, |links| {
            links.record_parent(child, parent)
        })
        .await
    }

    /// Records `child` as forked from `origin`'s state, then persists the graph.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::link_parent`].
    pub async fn link_fork(&self, agent_id: &str, child: &str, origin: &str) -> Result<()> {
        update_links(&self.base, &self.links_lock, agent_id, |links| {
            links.record_fork(child, origin)
        })
        .await
    }

    /// The session `session_id` was created from, or `None` when the graph has
    /// no such parent link.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::read_links`].
    pub async fn parent_of(&self, agent_id: &str, session_id: &str) -> Result<Option<String>> {
        Ok(self
            .read_links(agent_id)
            .await?
            .parent_of(session_id)
            .map(str::to_string))
    }

    /// The sessions created from `session_id`, in insertion order.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::read_links`].
    pub async fn children_of(&self, agent_id: &str, session_id: &str) -> Result<Vec<String>> {
        Ok(self
            .read_links(agent_id)
            .await?
            .children_of(session_id)
            .to_vec())
    }

    /// The root→parent path of `session_id`.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::read_links`] and [`SessionLinks::ancestors`].
    pub async fn ancestors(&self, agent_id: &str, session_id: &str) -> Result<Vec<String>> {
        self.read_links(agent_id).await?.ancestors(session_id)
    }

    /// Every session downstream of `session_id`, breadth-first.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::read_links`] and [`SessionLinks::descendants`].
    pub async fn descendants(&self, agent_id: &str, session_id: &str) -> Result<Vec<String>> {
        self.read_links(agent_id).await?.descendants(session_id)
    }

    /// The full root→session path (inclusive) of `session_id`.
    ///
    /// This is the traversal a `ChainSession`-style caller needs: it names every
    /// session between a root and the requested one, which is the order to
    /// replay or inherit state in.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::read_links`] and [`SessionLinks::chain`].
    pub async fn chain(&self, agent_id: &str, session_id: &str) -> Result<Vec<String>> {
        self.read_links(agent_id).await?.chain(session_id)
    }
}

/// An in-memory cache in front of one agent's [`DiskSessionStore`].
///
/// Mirrors Python's per-agent `SessionController`: reads populate the cache,
/// writes persist immediately, and [`SessionController::flush`] re-persists
/// every cached record.
///
/// [`SessionController::upsert`] and [`SessionController::flush`] re-file a
/// record in the scope the index already records for it, so they reject a
/// session the index does not know; [`SessionController::upsert_in`],
/// [`SessionController::save`] and [`SessionController::save_in`] take the scope
/// explicitly.
#[derive(Debug, Clone)]
pub struct SessionController {
    store: DiskSessionStore,
    agent_id: String,
    cache: Arc<Mutex<BTreeMap<String, SessionRecord>>>,
}

impl SessionController {
    /// Creates a controller for `agent_id` over `store`.
    pub fn new(store: DiskSessionStore, agent_id: impl Into<String>) -> Self {
        Self {
            store,
            agent_id: agent_id.into(),
            cache: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// The agent this controller manages.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// The underlying store.
    #[must_use]
    pub fn store(&self) -> &DiskSessionStore {
        &self.store
    }

    /// Creates and persists a new empty session in `scope`.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::save_snapshot`].
    pub async fn create(&self, scope: &SessionScope, session_id: &str) -> Result<SessionMeta> {
        let snapshot = SessionSnapshot::new(session_id, SessionType::Agent);
        let meta = self
            .store
            .save_snapshot(&self.agent_id, scope, session_id, snapshot.clone(), None)
            .await?;
        self.cache.lock().insert(
            session_id.to_string(),
            SessionRecord::new(meta.clone(), snapshot),
        );
        Ok(meta)
    }

    /// Persists `record` in the scope it already belongs to and caches it.
    ///
    /// The scope is read from the agent's index, so **the session must already be
    /// stored**: a session the index does not know is
    /// [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation),
    /// because there is no scope to file it under. Use
    /// [`SessionController::upsert_in`] when the scope is known.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::save`], plus
    /// [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation) for
    /// a session with no scope and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for an
    /// unparseable scope key in the index.
    pub async fn upsert(&self, record: SessionRecord) -> Result<SessionMeta> {
        let session_id = record.meta.session_id.clone();
        let scope = self.require_scope(&session_id).await?;
        self.upsert_in(&scope, record).await
    }

    /// Persists `record` in `scope` and caches it.
    ///
    /// The explicit-scope spelling of [`SessionController::upsert`]: it works for
    /// a session the index does not know yet, and a new session starts at
    /// version 1 and becomes the scope's active session.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::save`].
    pub async fn upsert_in(
        &self,
        scope: &SessionScope,
        record: SessionRecord,
    ) -> Result<SessionMeta> {
        let session_id = record.meta.session_id.clone();
        let meta = self
            .store
            .save(&self.agent_id, scope, record.clone())
            .await?;
        self.cache.lock().insert(
            session_id,
            SessionRecord::new(meta.clone(), record.snapshot),
        );
        Ok(meta)
    }

    /// Reads a session record, consulting the cache first.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::load`].
    pub async fn get(&self, session_id: &str) -> Result<Option<SessionRecord>> {
        if let Some(record) = self.cache.lock().get(session_id).cloned() {
            return Ok(Some(record));
        }
        let record = self.store.load(&self.agent_id, session_id).await?;
        if let Some(record) = &record {
            self.cache
                .lock()
                .insert(session_id.to_string(), record.clone());
        }
        Ok(record)
    }

    /// Snapshots `session`, persists it under `scope`, and caches the record.
    ///
    /// This is the explicit-scope form used by
    /// [`SessionController::save_in`]; unlike [`SessionController::flush`] it
    /// needs no previously stored scope.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::snapshot`] and [`DiskSessionStore::save`] failures.
    pub async fn save(
        &self,
        session: &dyn Session,
        scope: &SessionScope,
        title: Option<&str>,
    ) -> Result<SessionMeta> {
        self.save_in(session, scope, title).await
    }

    /// Snapshots `session`, persists it under `scope`, and caches the record.
    ///
    /// The explicit-scope spelling of [`SessionController::save`], kept so the
    /// "save this live session here" path is distinct from the
    /// known-session-only [`SessionController::upsert`] / [`SessionController::flush`]
    /// pair.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::snapshot`] and [`DiskSessionStore::save`] failures.
    pub async fn save_in(
        &self,
        session: &dyn Session,
        scope: &SessionScope,
        title: Option<&str>,
    ) -> Result<SessionMeta> {
        let session_id = session.id().to_string();
        let mut meta = SessionMeta::new(session_id.clone());
        meta.title = title.map(str::to_string);
        let snapshot = session.snapshot().await?;
        let meta = self
            .store
            .save(
                &self.agent_id,
                scope,
                SessionRecord::new(meta, snapshot.clone()),
            )
            .await?;
        self.cache
            .lock()
            .insert(session_id, SessionRecord::new(meta.clone(), snapshot));
        Ok(meta)
    }

    /// Re-persists every cached record, each in the scope it already belongs to.
    ///
    /// Every cached session must already be stored: a cached record the index
    /// does not know is
    /// [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// rather than a silent move into the main scope.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::save`], plus
    /// [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation) for
    /// a session with no scope and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for an
    /// unparseable scope key in the index.
    pub async fn flush(&self) -> Result<()> {
        let records: Vec<SessionRecord> = self.cache.lock().values().cloned().collect();
        for record in records {
            let session_id = record.meta.session_id.clone();
            let scope = self.require_scope(&session_id).await?;
            let meta = self
                .store
                .save(&self.agent_id, &scope, record.clone())
                .await?;
            self.cache
                .lock()
                .insert(session_id, SessionRecord::new(meta, record.snapshot));
        }
        Ok(())
    }

    /// Deletes a session from the cache and the store.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::delete`].
    pub async fn delete(&self, session_id: &str) -> Result<bool> {
        self.cache.lock().remove(session_id);
        self.store.delete(&self.agent_id, session_id).await
    }

    /// Lists session metadata, merging records that are only cached.
    ///
    /// A scope-filtered listing delegates to the store; the scope-less listing
    /// additionally reports cached records the index does not know about yet.
    ///
    /// # Errors
    ///
    /// See [`DiskSessionStore::list`].
    pub async fn list(&self, scope: Option<&SessionScope>) -> Result<Vec<SessionMeta>> {
        let mut sessions = self.store.list(&self.agent_id, scope).await?;
        if scope.is_none() {
            let cached: Vec<SessionRecord> = self.cache.lock().values().cloned().collect();
            for record in cached {
                if !sessions
                    .iter()
                    .any(|meta| meta.session_id == record.meta.session_id)
                {
                    sessions.push(record.meta);
                }
            }
            sessions.sort_by(|left, right| {
                right
                    .updated_at
                    .total_cmp(&left.updated_at)
                    .then_with(|| left.session_id.cmp(&right.session_id))
            });
        }
        Ok(sessions)
    }

    /// Finds the scope a session already belongs to.
    ///
    /// Returns `Ok(None)` when the index does not know `session_id`. An index key
    /// that cannot be parsed as a [`SessionScopeKey`] is a corrupt index and is
    /// reported as [`Error::Session`] naming the key — never silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`](openjiuwen_common::error::ErrorKind::Session)
    /// when the index is unreadable, malformed, or carries an unparseable scope
    /// key for this session.
    async fn scope_of(&self, session_id: &str) -> Result<Option<SessionScope>> {
        let index = self.store.read_index(&self.agent_id).await?;
        for (key, scope) in &index.scopes {
            if scope.get_session(session_id).is_some() {
                let parsed = key.parse::<SessionScopeKey>().map_err(|error| {
                    Error::session(format!(
                        "invalid session scope key {key:?} in sessions.json: {error}"
                    ))
                })?;
                return Ok(Some(parsed.scope));
            }
        }
        Ok(None)
    }

    /// The scope a session already belongs to, or a validation error naming the
    /// session when the index does not know it.
    async fn require_scope(&self, session_id: &str) -> Result<SessionScope> {
        self.scope_of(session_id).await?.ok_or_else(|| {
            Error::validation(format!(
                "session {session_id} has no scope: save it with an explicit scope first"
            ))
        })
    }
}

/// Builds a [`Error::Session`] from an I/O failure.
fn io_error(action: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::session(format!("failed to {action} {}: {error}", path.display()))
}

/// Reads and deserializes a JSON document, returning `None` when it is absent.
///
/// # Errors
///
/// Returns [`Error::Session`] when the document is unreadable or malformed.
pub(crate) async fn read_json_optional<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error("read", path, &error)),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| Error::session(format!("malformed JSON at {}: {error}", path.display())))
}

/// Atomically writes `value` as JSON: write a temporary sibling, then rename.
///
/// The temporary file is removed when either step fails, so a failed write
/// leaves no `.tmp` artifact behind.
///
/// # Errors
///
/// Returns [`Error::Session`] for serialization or I/O failures.
pub(crate) async fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::session(format!("failed to serialize {}: {error}", path.display()))
    })?;
    let parent = path.parent().ok_or_else(|| {
        Error::validation(format!("path has no parent directory: {}", path.display()))
    })?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| io_error("create directory", parent, &error))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data");
    let temp = parent.join(format!(".{file_name}.{}.tmp", uuid_hex()));

    if let Err(error) = tokio::fs::write(&temp, &bytes).await {
        remove_temp(&temp).await;
        return Err(io_error("write", &temp, &error));
    }
    if let Err(error) = tokio::fs::rename(&temp, path).await {
        remove_temp(&temp).await;
        return Err(io_error("rename", path, &error));
    }
    Ok(())
}

/// Best-effort removal of a temporary file after a failed write.
async fn remove_temp(path: &Path) {
    if let Err(error) = tokio::fs::remove_file(path).await {
        if error.kind() != IoErrorKind::NotFound {
            tracing::debug!(path = %path.display(), %error, "failed to remove temporary file");
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::DiskSessionStore;

    #[test]
    fn path_builders_follow_the_python_layout() {
        let store = DiskSessionStore::new("/tmp/base");
        assert_eq!(store.agent_dir("a1"), std::path::Path::new("/tmp/base/a1"));
        assert_eq!(
            store.sessions_index_path("a1"),
            std::path::Path::new("/tmp/base/a1/sessions/sessions.json")
        );
        assert_eq!(
            store.state_path("a1", "s1"),
            std::path::Path::new("/tmp/base/a1/sessions/s1/state.data")
        );
        assert_eq!(
            store.history_dir("a1", "s1"),
            std::path::Path::new("/tmp/base/a1/sessions/s1/history")
        );
    }
}
