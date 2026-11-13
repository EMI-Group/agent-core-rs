//! File-backed session version history.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20) on top of [`DiskSessionStore`]:
//!
//! - `<session_dir>/history/versions/v000001.json` — one [`Version`] per
//!   commit.
//! - `<session_dir>/history/index.json` — an optional version metadata cache;
//!   when it is absent or corrupt the versions directory is scanned.
//!
//! Version ids are sequential. Allocation happens under a store-level
//! `tokio::sync::Mutex`, so concurrent commits (even through cloned stores or
//! histories) never reuse an id. Every write is atomic (temp file + rename).
use std::io::ErrorKind as IoErrorKind;
use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};

use crate::checkpoint::validate_id;
use crate::session::Session;

use super::diff::{diff_snapshots, JsonChange};
use super::version::{Version, VersionId, VersionMeta};
use crate::session_controller::store::{read_json_optional, write_json_atomic};
use crate::session_controller::DiskSessionStore;

/// The `index.json` cache document: the version metadata list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct VersionIndex {
    versions: Vec<VersionMeta>,
}

/// A session's local version history.
///
/// Each [`SessionHistory::commit`] snapshots a session (or accepts a snapshot
/// directly) and appends an immutable version; [`SessionHistory::restore`]
/// applies an old snapshot back to a live session without creating a version.
#[derive(Debug, Clone)]
pub struct SessionHistory {
    store: DiskSessionStore,
}

impl SessionHistory {
    /// Creates a history over `store`.
    #[must_use]
    pub fn new(store: DiskSessionStore) -> Self {
        Self { store }
    }

    /// The store backing this history.
    #[must_use]
    pub fn store(&self) -> &DiskSessionStore {
        &self.store
    }

    /// `<session_dir>/history`.
    #[must_use]
    pub fn history_dir(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.store.history_dir(agent_id, session_id)
    }

    /// `<history>/versions`.
    #[must_use]
    pub fn versions_dir(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.history_dir(agent_id, session_id).join("versions")
    }

    /// `<history>/versions/v000001.json`.
    #[must_use]
    pub fn version_path(&self, agent_id: &str, session_id: &str, id: VersionId) -> PathBuf {
        self.versions_dir(agent_id, session_id)
            .join(format!("{id}.json"))
    }

    /// `<history>/index.json`.
    #[must_use]
    pub fn index_path(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.history_dir(agent_id, session_id).join("index.json")
    }

    /// Commits `snapshot` as the next version, returning its metadata.
    ///
    /// The new version's `parent` is the previously latest version and its id
    /// is the next sequential number. The history directory is created lazily.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent or session id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// or serialization failures.
    pub async fn commit(
        &self,
        agent_id: &str,
        session_id: &str,
        snapshot: &crate::session::SessionSnapshot,
        label: Option<&str>,
    ) -> Result<VersionMeta> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;

        let existing = self.list_unlocked(agent_id, session_id).await?;
        let id = existing.last().map_or(1, |version| version.id + 1);
        let parent = existing.last().map(|version| version.id);
        let version = Version {
            id,
            label: label.map(str::to_string),
            created_at: now_rfc3339(),
            parent,
            snapshot: snapshot.clone(),
        };
        write_json_atomic(
            &self.version_path(agent_id, session_id, VersionId(id)),
            &version,
        )
        .await?;

        let mut metas = existing;
        metas.push(version.meta());
        metas.sort_by_key(|meta| meta.id);
        self.write_index_unlocked(agent_id, session_id, &metas)
            .await?;
        Ok(version.meta())
    }

    /// Snapshots `session` and commits it.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::snapshot`] and [`SessionHistory::commit`] failures.
    pub async fn commit_from_session(
        &self,
        agent_id: &str,
        session: &dyn Session,
        label: Option<&str>,
    ) -> Result<VersionMeta> {
        let snapshot = session.snapshot().await?;
        self.commit(agent_id, session.id(), &snapshot, label).await
    }

    /// Lists version metadata in ascending id order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when a
    /// version document is unreadable or malformed.
    pub async fn list(&self, agent_id: &str, session_id: &str) -> Result<Vec<VersionMeta>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;
        self.list_unlocked(agent_id, session_id).await
    }

    /// Loads one version, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// document is malformed.
    pub async fn load(
        &self,
        agent_id: &str,
        session_id: &str,
        id: VersionId,
    ) -> Result<Option<Version>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        read_json_optional(&self.version_path(agent_id, session_id, id)).await
    }

    /// Loads the latest version, or `None` when the history is empty.
    ///
    /// # Errors
    ///
    /// See [`SessionHistory::list`].
    pub async fn latest(&self, agent_id: &str, session_id: &str) -> Result<Option<Version>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;
        let latest = self
            .list_unlocked(agent_id, session_id)
            .await?
            .into_iter()
            .next_back();
        match latest {
            Some(meta) => self.load(agent_id, session_id, VersionId(meta.id)).await,
            None => Ok(None),
        }
    }

    /// Deletes one version, returning `false` when it did not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures.
    pub async fn delete(&self, agent_id: &str, session_id: &str, id: VersionId) -> Result<bool> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;

        let path = self.version_path(agent_id, session_id, id);
        match tokio::fs::metadata(&path).await {
            Ok(_) => {}
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(Error::session(format!(
                    "failed to stat version {}: {error}",
                    path.display()
                )))
            }
        }
        tokio::fs::remove_file(&path).await.map_err(|error| {
            Error::session(format!(
                "failed to delete version {}: {error}",
                path.display()
            ))
        })?;
        let metas = self.scan_metas(agent_id, session_id).await?;
        self.write_index_unlocked(agent_id, session_id, &metas)
            .await?;
        Ok(true)
    }

    /// Removes the whole history directory; a missing history is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures.
    pub async fn clear(&self, agent_id: &str, session_id: &str) -> Result<()> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;
        let dir = self.history_dir(agent_id, session_id);
        match tokio::fs::remove_dir_all(&dir).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::session(format!(
                "failed to clear history {}: {error}",
                dir.display()
            ))),
        }
    }

    /// Diffs two versions' snapshots; see [`diff_snapshots`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`](openjiuwen_common::error::ErrorKind::Session)
    /// when either version is missing or malformed.
    pub async fn diff(
        &self,
        agent_id: &str,
        session_id: &str,
        left: VersionId,
        right: VersionId,
    ) -> Result<Vec<JsonChange>> {
        let left_version = self
            .load(agent_id, session_id, left)
            .await?
            .ok_or_else(|| missing_version(agent_id, session_id, left))?;
        let right_version = self
            .load(agent_id, session_id, right)
            .await?
            .ok_or_else(|| missing_version(agent_id, session_id, right))?;
        Ok(diff_snapshots(
            &left_version.snapshot,
            &right_version.snapshot,
        ))
    }

    /// Restores `session` to version `id`, without committing a new version.
    ///
    /// Returns `false` when the version does not exist.
    ///
    /// # Errors
    ///
    /// Propagates [`SessionHistory::load`] and [`Session::restore`] failures.
    pub async fn restore(
        &self,
        agent_id: &str,
        session_id: &str,
        id: VersionId,
        session: &dyn Session,
    ) -> Result<bool> {
        match self.load(agent_id, session_id, id).await? {
            Some(version) => {
                session.restore(version.snapshot).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Lists version metadata in ascending id order without taking the lock.
    async fn list_unlocked(&self, agent_id: &str, session_id: &str) -> Result<Vec<VersionMeta>> {
        let mut metas = match self.read_index(agent_id, session_id).await? {
            Some(metas) => metas,
            None => self.scan_metas(agent_id, session_id).await?,
        };
        metas.sort_by_key(|meta| meta.id);
        Ok(metas)
    }

    /// Reads the index cache; a missing or corrupt index yields `None` so the
    /// caller scans the versions directory instead.
    async fn read_index(
        &self,
        agent_id: &str,
        session_id: &str,
    ) -> Result<Option<Vec<VersionMeta>>> {
        let path = self.index_path(agent_id, session_id);
        match read_json_optional::<VersionIndex>(&path).await {
            Ok(Some(index)) => Ok(Some(index.versions)),
            Ok(None) => Ok(None),
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "ignoring unreadable version index; scanning versions"
                );
                Ok(None)
            }
        }
    }

    /// Scans `<history>/versions` for version documents.
    async fn scan_metas(&self, agent_id: &str, session_id: &str) -> Result<Vec<VersionMeta>> {
        let dir = self.versions_dir(agent_id, session_id);
        let mut reader = match tokio::fs::read_dir(&dir).await {
            Ok(reader) => reader,
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(Error::session(format!(
                    "failed to read versions directory {}: {error}",
                    dir.display()
                )))
            }
        };
        let mut metas = Vec::new();
        loop {
            let entry = match reader.next_entry().await {
                Ok(Some(entry)) => entry,
                Ok(None) => break,
                Err(error) => {
                    return Err(Error::session(format!(
                        "failed to read versions directory {}: {error}",
                        dir.display()
                    )))
                }
            };
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            if let Some(version) = read_json_optional::<Version>(&path).await? {
                metas.push(version.meta());
            }
        }
        metas.sort_by_key(|meta| meta.id);
        Ok(metas)
    }

    /// Atomically writes (or removes, when empty) the index cache.
    async fn write_index_unlocked(
        &self,
        agent_id: &str,
        session_id: &str,
        metas: &[VersionMeta],
    ) -> Result<()> {
        let path = self.index_path(agent_id, session_id);
        if metas.is_empty() {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
                Err(error) => Err(Error::session(format!(
                    "failed to remove version index {}: {error}",
                    path.display()
                ))),
            }
        } else {
            write_json_atomic(
                &path,
                &VersionIndex {
                    versions: metas.to_vec(),
                },
            )
            .await
        }
    }
}

fn missing_version(agent_id: &str, session_id: &str, id: VersionId) -> Error {
    Error::session(format!(
        "version {id} not found for session {session_id:?} of agent {agent_id:?}"
    ))
}
