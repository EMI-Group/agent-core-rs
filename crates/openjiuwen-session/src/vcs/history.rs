//! File-backed session version history.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20) on top of [`DiskSessionStore`]:
//!
//! - `<session_dir>/history/versions/v000001.json` — one [`Version`] per
//!   commit.
//! - `<session_dir>/history/index.json` — an optional version metadata cache;
//!   when it is absent or corrupt the versions directory is scanned.
//! - `<session_dir>/history/refs.json` — the local refs document
//!   ([`Refs`]): `HEAD` (advanced by every commit, reset by a rewind) and
//!   `main` (the latest committed version), plus any caller-defined ref.
//!
//! Version ids are sequential. Allocation happens under a store-level
//! `tokio::sync::Mutex`, so concurrent commits (even through cloned stores or
//! histories) never reuse an id. Every write is atomic (temp file + rename).
//!
//! [`SessionHistory::rewind`] restores the live session to a version and moves
//! `HEAD` back (a soft rewind that keeps the newer versions); its
//! [`SessionHistory::rewind_hard`] form also deletes them and re-points `main`.
//! [`SessionHistory::fork`] clones a version into a brand-new session record
//! plus a fresh one-version history for it.
use std::io::ErrorKind as IoErrorKind;
use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};

use crate::checkpoint::validate_id;
use crate::session::Session;

use super::diff::{diff_snapshots, JsonChange};
use super::refs::{RefEntry, RefName, Refs};
use super::version::{Version, VersionId, VersionMeta};
use crate::session_controller::store::{read_json_optional, write_json_atomic};
use crate::session_controller::{DiskSessionStore, SessionScope, SessionScopeKey};

/// The `index.json` cache document: the version metadata list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct VersionIndex {
    versions: Vec<VersionMeta>,
}

/// How [`SessionHistory::rewind_with`] treats versions newer than the target.
///
/// The default ([`RewindOptions::SOFT`]) keeps them, mirroring the read-only
/// replay Python's `VersioningManager::restore` performs; [`RewindOptions::HARD`]
/// discards them, mirroring Python's overwrite `rewind` (which truncates the
/// WAL after the target).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RewindOptions {
    /// Delete every version newer than the target and re-point `main` at it.
    pub discard_newer: bool,
}

impl RewindOptions {
    /// Keep the newer versions: only the live session and `HEAD` move back.
    pub const SOFT: Self = Self {
        discard_newer: false,
    };

    /// Delete the newer versions and re-point `main` at the target.
    pub const HARD: Self = Self {
        discard_newer: true,
    };
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
        self.advance_refs_unlocked(agent_id, session_id, VersionId(id))
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

    /// `<history>/refs.json`.
    #[must_use]
    pub fn refs_path(&self, agent_id: &str, session_id: &str) -> PathBuf {
        self.history_dir(agent_id, session_id).join("refs.json")
    }

    /// Reads the session's refs document.
    ///
    /// A missing document is empty; an unreadable or malformed one is logged and
    /// treated as empty, so a corrupt `refs.json` never blocks the history (the
    /// same tolerance [`SessionHistory::list`] applies to `index.json`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed agent or session id.
    pub async fn refs(&self, agent_id: &str, session_id: &str) -> Result<Refs> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        self.read_refs_unlocked(agent_id, session_id).await
    }

    /// Points the ref `name` at `version`, creating it when needed.
    ///
    /// Refs are local metadata over the version history; `HEAD` and `main` are
    /// created by the first commit rather than reserved.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id or ref name and for a version that does not exist, and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures.
    pub async fn set_ref(
        &self,
        agent_id: &str,
        session_id: &str,
        name: &str,
        version: VersionId,
    ) -> Result<RefEntry> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let name = RefName::new(name)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;

        if self.load(agent_id, session_id, version).await?.is_none() {
            return Err(Error::validation(format!(
                "cannot point ref {:?} at {version}: no such version for session {session_id:?} of agent {agent_id:?}",
                name.as_str()
            )));
        }
        let mut refs = self.read_refs_unlocked(agent_id, session_id).await?;
        let entry = refs.set(&name, version);
        self.write_refs_unlocked(agent_id, session_id, &refs)
            .await?;
        Ok(entry)
    }

    /// Removes the ref `name`, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id or ref name and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) for I/O
    /// failures.
    pub async fn delete_ref(&self, agent_id: &str, session_id: &str, name: &str) -> Result<bool> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let name = RefName::new(name)?;
        let lock = self.store.history_lock();
        let _guard = lock.lock().await;

        let mut refs = self.read_refs_unlocked(agent_id, session_id).await?;
        if refs.remove(&name).is_none() {
            return Ok(false);
        }
        self.write_refs_unlocked(agent_id, session_id, &refs)
            .await?;
        Ok(true)
    }

    /// Resolves a ref to its version's metadata.
    ///
    /// `None` means the ref does not exist, or points at a version that has
    /// since been deleted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id or ref name and
    /// [`Error::Session`](openjiuwen_common::error::ErrorKind::Session) when the
    /// version document is malformed.
    pub async fn resolve_ref(
        &self,
        agent_id: &str,
        session_id: &str,
        name: &str,
    ) -> Result<Option<VersionMeta>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        let name = RefName::new(name)?;
        let refs = self.read_refs_unlocked(agent_id, session_id).await?;
        let Some(entry) = refs.get(&name) else {
            return Ok(None);
        };
        let target = entry.target;
        Ok(self
            .load(agent_id, session_id, target)
            .await?
            .map(|version| version.meta()))
    }

    /// Lists every ref of the session, in name order.
    ///
    /// # Errors
    ///
    /// See [`SessionHistory::refs`].
    pub async fn list_refs(&self, agent_id: &str, session_id: &str) -> Result<Vec<RefEntry>> {
        Ok(self
            .refs(agent_id, session_id)
            .await?
            .list()
            .into_iter()
            .cloned()
            .collect())
    }

    /// The version the history currently points at: `HEAD`, falling back to the
    /// latest version for a history that predates the refs document.
    ///
    /// A `HEAD` pointing at a deleted version is logged and ignored, so the
    /// fallback also covers a partially deleted history.
    ///
    /// # Errors
    ///
    /// See [`SessionHistory::list`].
    pub async fn head(&self, agent_id: &str, session_id: &str) -> Result<Option<VersionMeta>> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;
        match self.resolve_target_unlocked(agent_id, session_id).await? {
            Some(id) => Ok(self.load(agent_id, session_id, id).await?.map(|v| v.meta())),
            None => Ok(None),
        }
    }

    /// Rewinds `session` to `target` (default: the current `HEAD`).
    ///
    /// The target version's snapshot is restored into the live session through
    /// [`Session::restore`], then `HEAD` is reset to the target. The newer
    /// versions stay in the history and `main` keeps tracking the latest
    /// commit, so this is a *soft* rewind; use [`SessionHistory::rewind_hard`]
    /// to discard them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`](openjiuwen_common::error::ErrorKind::Session)
    /// when the target version does not exist (or the history is empty and no
    /// target was given), [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id, and propagates [`Session::restore`] failures.
    pub async fn rewind(
        &self,
        agent_id: &str,
        session_id: &str,
        target: Option<VersionId>,
        session: &dyn Session,
    ) -> Result<VersionMeta> {
        self.rewind_with(agent_id, session_id, target, session, RewindOptions::SOFT)
            .await
    }

    /// Rewinds like [`SessionHistory::rewind`] and discards the newer versions.
    ///
    /// Every version newer than the target is deleted, the version index is
    /// rewritten, and `main` is re-pointed at the target.
    ///
    /// # Errors
    ///
    /// See [`SessionHistory::rewind`].
    pub async fn rewind_hard(
        &self,
        agent_id: &str,
        session_id: &str,
        target: Option<VersionId>,
        session: &dyn Session,
    ) -> Result<VersionMeta> {
        self.rewind_with(agent_id, session_id, target, session, RewindOptions::HARD)
            .await
    }

    /// Rewinds `session` to `target` with explicit [`RewindOptions`].
    ///
    /// The live session is restored before anything is deleted, so a failed
    /// restore leaves the history untouched. Rewinding to the current `HEAD` is
    /// idempotent.
    ///
    /// # Errors
    ///
    /// See [`SessionHistory::rewind`].
    pub async fn rewind_with(
        &self,
        agent_id: &str,
        session_id: &str,
        target: Option<VersionId>,
        session: &dyn Session,
        options: RewindOptions,
    ) -> Result<VersionMeta> {
        validate_id("agent", agent_id)?;
        validate_id("session", session_id)?;

        let target = match target {
            Some(target) => target,
            None => self
                .resolve_target_unlocked(agent_id, session_id)
                .await?
                .ok_or_else(|| {
                    Error::session(format!(
                        "cannot rewind session {session_id:?} of agent {agent_id:?}: the history has no versions"
                    ))
                })?,
        };
        let version = self
            .load(agent_id, session_id, target)
            .await?
            .ok_or_else(|| missing_version(agent_id, session_id, target))?;
        let meta = version.meta();
        session.restore(version.snapshot).await?;

        let lock = self.store.history_lock();
        let _guard = lock.lock().await;
        if options.discard_newer {
            self.truncate_unlocked(agent_id, session_id, target).await?;
        }
        let mut refs = self.read_refs_unlocked(agent_id, session_id).await?;
        refs.set_head(target);
        if options.discard_newer {
            refs.set_main(target);
        }
        self.write_refs_unlocked(agent_id, session_id, &refs)
            .await?;
        Ok(meta)
    }

    /// Forks `source_session_id` at `from` (default: the source's `HEAD`) into
    /// `new_session_id`.
    ///
    /// The chosen version's snapshot becomes a brand-new session record in the
    /// source session's scope, and the fork gets its own history whose first
    /// version is labelled `fork of <source>@v000003`.
    /// A fork starts a new lineage, so that first version's `parent` is `None`,
    /// and the fork's `HEAD`/`main` refs point at it.
    /// The source session, its history and its refs are left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// for a malformed id, a `new_session_id` that already exists, and a source
    /// session that the store does not know; [`Error::Session`](openjiuwen_common::error::ErrorKind::Session)
    /// when the source version does not exist (or the source history is empty
    /// and no `from` was given); propagates store I/O failures.
    pub async fn fork(
        &self,
        agent_id: &str,
        source_session_id: &str,
        from: Option<VersionId>,
        new_session_id: &str,
        title: Option<&str>,
    ) -> Result<VersionMeta> {
        validate_id("agent", agent_id)?;
        validate_id("session", source_session_id)?;
        validate_id("session", new_session_id)?;
        if new_session_id == source_session_id {
            return Err(Error::validation(format!(
                "cannot fork session {source_session_id:?} onto itself: pass a new session id"
            )));
        }
        // Resolve the source scope first, so an unknown source is a validation
        // error rather than a silent fallback to another session's scope.
        let scope = self.source_scope(agent_id, source_session_id).await?;

        let from = match from {
            Some(from) => from,
            None => self
                .resolve_target_unlocked(agent_id, source_session_id)
                .await?
                .ok_or_else(|| {
                    Error::session(format!(
                        "cannot fork session {source_session_id:?} of agent {agent_id:?}: the history has no versions"
                    ))
                })?,
        };
        let version = self
            .load(agent_id, source_session_id, from)
            .await?
            .ok_or_else(|| missing_version(agent_id, source_session_id, from))?;
        if self.store.exists(agent_id, new_session_id).await? {
            return Err(Error::validation(format!(
                "cannot fork into session {new_session_id:?}: it already exists for agent {agent_id:?}"
            )));
        }

        let mut snapshot = version.snapshot;
        snapshot.session_id = new_session_id.to_string();
        self.store
            .save_snapshot(agent_id, &scope, new_session_id, snapshot.clone(), title)
            .await?;
        let label = format!("fork of {source_session_id}@{from}");
        self.commit(agent_id, new_session_id, &snapshot, Some(&label))
            .await
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

    /// Reads the refs document; a missing or malformed document is empty.
    async fn read_refs_unlocked(&self, agent_id: &str, session_id: &str) -> Result<Refs> {
        let path = self.refs_path(agent_id, session_id);
        match read_json_optional::<Refs>(&path).await {
            Ok(Some(refs)) => Ok(refs),
            Ok(None) => Ok(Refs::new()),
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "ignoring unreadable refs document; treating it as empty"
                );
                Ok(Refs::new())
            }
        }
    }

    /// Atomically writes (or removes, when empty) the refs document.
    async fn write_refs_unlocked(
        &self,
        agent_id: &str,
        session_id: &str,
        refs: &Refs,
    ) -> Result<()> {
        let path = self.refs_path(agent_id, session_id);
        if refs.is_empty() {
            match tokio::fs::remove_file(&path).await {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == IoErrorKind::NotFound => Ok(()),
                Err(error) => Err(Error::session(format!(
                    "failed to remove refs document {}: {error}",
                    path.display()
                ))),
            }
        } else {
            write_json_atomic(&path, refs).await
        }
    }

    /// Advances `HEAD` and `main` to a freshly written version.
    async fn advance_refs_unlocked(
        &self,
        agent_id: &str,
        session_id: &str,
        id: VersionId,
    ) -> Result<()> {
        let mut refs = self.read_refs_unlocked(agent_id, session_id).await?;
        refs.set_head(id);
        refs.set_main(id);
        self.write_refs_unlocked(agent_id, session_id, &refs).await
    }

    /// The version `HEAD` points at, falling back to the latest version.
    async fn resolve_target_unlocked(
        &self,
        agent_id: &str,
        session_id: &str,
    ) -> Result<Option<VersionId>> {
        let refs = self.read_refs_unlocked(agent_id, session_id).await?;
        if let Some(entry) = refs.head() {
            let version = entry.target;
            if self.load(agent_id, session_id, version).await?.is_some() {
                return Ok(Some(version));
            }
            tracing::warn!(
                version = %version,
                "HEAD points at a missing version; falling back to the latest version"
            );
        }
        Ok(self
            .list_unlocked(agent_id, session_id)
            .await?
            .last()
            .map(|meta| VersionId(meta.id)))
    }

    /// Deletes every version newer than `target` and rewrites the index.
    async fn truncate_unlocked(
        &self,
        agent_id: &str,
        session_id: &str,
        target: VersionId,
    ) -> Result<()> {
        let mut metas = self.list_unlocked(agent_id, session_id).await?;
        for meta in metas.iter().filter(|meta| meta.id > target.0) {
            let path = self.version_path(agent_id, session_id, VersionId(meta.id));
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {}
                Err(error) if error.kind() == IoErrorKind::NotFound => {}
                Err(error) => {
                    return Err(Error::session(format!(
                        "failed to delete version {}: {error}",
                        path.display()
                    )))
                }
            }
        }
        metas.retain(|meta| meta.id <= target.0);
        self.write_index_unlocked(agent_id, session_id, &metas)
            .await
    }

    /// The scope a stored session belongs to, resolved from the agent's index.
    async fn source_scope(&self, agent_id: &str, session_id: &str) -> Result<SessionScope> {
        let index = self.store.read_index(agent_id).await?;
        for (key, scope) in &index.scopes {
            if scope.get_session(session_id).is_none() {
                continue;
            }
            let parsed: SessionScopeKey = key.parse()?;
            return Ok(parsed.scope);
        }
        Err(Error::validation(format!(
            "unknown session {session_id:?} for agent {agent_id:?}: the session index has no entry, so its scope cannot be resolved"
        )))
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
