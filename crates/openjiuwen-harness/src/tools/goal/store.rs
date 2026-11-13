//! JSON persistence of the session goal: the `goal.json` layout, atomic saves and the lock.
//!
//! Ports `openjiuwen/harness/goal/store.py`, where Python keeps the record inside `Session`
//! state under `harness.goal.record`. Here the record is one JSON object in a file the caller
//! names ([`GoalStore::for_workspace`] uses the workspace `goal` node), which keeps the harness
//! independent of a particular session backend.
//!
//! Deviation from Python: Python's `SessionGoalStore` clears a record it cannot deserialize;
//! here a malformed file is an [`Error::Validation`] naming the path, so the caller decides
//! whether to start over. A missing file is `Ok(None)`.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::{Mutex, MutexGuard};

use openjiuwen_common::error::{Error, Result};

use super::record::GoalRecord;
use super::schema::DEFAULT_SESSION_ID;
use crate::workspace::Workspace;

/// Name of the workspace node holding the goal file.
const GOAL_NODE: &str = "goal";

/// File name of the persisted record.
const GOAL_FILE: &str = "goal.json";

/// Persists at most one [`GoalRecord`] at a fixed path.
///
/// The store owns no process-global state: each instance has its own async lock, and only the
/// file at [`Self::path`] is shared. [`Self::update`] is the atomic read-modify-write primitive;
/// [`Self::load`] and [`Self::save`] take the lock individually but are not atomic together.
pub struct GoalStore {
    /// File holding the record.
    path: PathBuf,
    /// Session id stamped into records created by [`Self::start`].
    session_id: String,
    /// Serializes the operations of this store; never held across a caller's `.await`.
    lock: Mutex<()>,
}

impl GoalStore {
    /// Builds a store over `path` with the Python default session id (`default`).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Arc<Self> {
        Self::for_session(path, DEFAULT_SESSION_ID)
    }

    /// Builds a store over `path` that stamps `session_id` into created records.
    #[must_use]
    pub fn for_session(path: impl Into<PathBuf>, session_id: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            path: path.into(),
            session_id: session_id.into(),
            lock: Mutex::new(()),
        })
    }

    /// Builds a store at the workspace `goal` node (`<goal node>/goal.json`).
    ///
    /// A workspace whose schema has no `goal` node falls back to `<root_path>/goal/goal.json`,
    /// the same rule the todo store uses for its directory.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Arc<Self> {
        let dir = workspace
            .get_node_path(GOAL_NODE)
            .unwrap_or_else(|| workspace.root_path.join(GOAL_NODE));
        Self::new(dir.join(GOAL_FILE))
    }

    /// The file holding the record.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The session id stamped into records created here.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Acquires the store lock; hold the guard for a whole read-modify-write span.
    ///
    /// [`Self::load`], [`Self::save`], [`Self::start`], [`Self::update`] and [`Self::clear`]
    /// take this lock themselves, so a caller only needs it when it drives several of those
    /// calls as one unit (and must then use the unlocked internals through [`Self::update`]).
    pub async fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().await
    }

    /// Loads the persisted record; a missing file yields `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the path when the file is not a JSON object or does
    /// not hold a valid [`GoalRecord`], and the underlying error when the file cannot be read.
    pub async fn load(&self) -> Result<Option<GoalRecord>> {
        let _guard = self.lock.lock().await;
        self.load_unlocked().await
    }

    /// Atomically replaces the persisted record.
    ///
    /// # Errors
    ///
    /// Returns an error when the parent directory cannot be created, the record cannot be
    /// serialized, or the temporary file cannot be written or renamed over [`Self::path`].
    pub async fn save(&self, record: &GoalRecord) -> Result<()> {
        let _guard = self.lock.lock().await;
        self.save_unlocked(record).await
    }

    /// Creates and persists a new active record for `objective` in this store's session.
    ///
    /// An existing record is replaced: Python's `GoalManager.set` asks for an explicit
    /// `overwrite_confirmed` before calling the store, so that policy belongs to the caller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with Python's message when the trimmed objective is empty,
    /// plus every error [`Self::save`] can report.
    pub async fn start(
        &self,
        objective: impl Into<String>,
        now: DateTime<Utc>,
    ) -> Result<GoalRecord> {
        let objective = objective.into();
        let objective = objective.trim();
        if objective.is_empty() {
            return Err(Error::validation("goal objective must not be empty"));
        }
        let record = GoalRecord::new(objective, now).with_session_id(self.session_id.clone());
        self.save(&record).await?;
        Ok(record)
    }

    /// Applies `mutate` to the persisted record, stamps `updated_at` with `now` and saves it.
    ///
    /// Returns `Ok(None)` when no record exists; nothing is created. The whole read-modify-write
    /// span holds the store lock, so concurrent updates cannot lose a change.
    ///
    /// # Errors
    ///
    /// Returns every error [`Self::load`] and [`Self::save`] can report.
    pub async fn update<F>(&self, mutate: F, now: DateTime<Utc>) -> Result<Option<GoalRecord>>
    where
        F: FnOnce(&mut GoalRecord),
    {
        let _guard = self.lock.lock().await;
        let Some(mut record) = self.load_unlocked().await? else {
            return Ok(None);
        };
        mutate(&mut record);
        record.touch(now);
        self.save_unlocked(&record).await?;
        Ok(Some(record))
    }

    /// Removes the persisted record, returning it when one existed.
    ///
    /// # Errors
    ///
    /// Returns every error [`Self::load`] can report, and the underlying error when the file
    /// exists but cannot be removed.
    pub async fn clear(&self) -> Result<Option<GoalRecord>> {
        let _guard = self.lock.lock().await;
        let record = self.load_unlocked().await?;
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => Ok(record),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(record),
            Err(error) => Err(error.into()),
        }
    }

    /// Loads the record without taking the store lock.
    async fn load_unlocked(&self) -> Result<Option<GoalRecord>> {
        let bytes = match tokio::fs::read(&self.path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| self.invalid(format_args!("invalid JSON: {error}")))?;
        let record = GoalRecord::from_dict(&value)
            .map_err(|error| self.invalid(format_args!("{}", error.message())))?;
        Ok(Some(record))
    }

    /// Saves the record without taking the store lock, through a temporary file and a rename.
    async fn save_unlocked(&self, record: &GoalRecord) -> Result<()> {
        let dir = self
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        tokio::fs::create_dir_all(&dir).await?;
        let rendered = serde_json::to_string_pretty(&record.to_dict())?;
        let temporary = dir.join(format!("{GOAL_FILE}.tmp-{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, rendered.as_bytes()).await?;
        if let Err(error) = tokio::fs::rename(&temporary, &self.path).await {
            // Windows cannot replace the target in place; remove it and retry once.
            tracing::debug!(%error, path = %self.path.display(), "retrying goal rename");
            let _ = tokio::fs::remove_file(&self.path).await;
            if let Err(error) = tokio::fs::rename(&temporary, &self.path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Builds the validation error naming the store path.
    fn invalid(&self, details: std::fmt::Arguments<'_>) -> Error {
        Error::validation(format!(
            "invalid goal record at {}: {details}",
            self.path.display()
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;
    use crate::tools::goal::schema::{GoalAssessment, GoalAssessmentStatus, GoalStatus};
    use crate::workspace::Workspace;

    /// A fixed clock.
    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap()
    }

    /// A clock one minute later.
    fn later() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 2, 3, 5, 10).unwrap()
    }

    /// A store inside a fresh temporary directory.
    fn store(dir: &tempfile::TempDir) -> Arc<GoalStore> {
        GoalStore::for_session(dir.path().join("goal").join("goal.json"), "s-1")
    }

    #[test]
    fn workspace_store_lives_in_the_goal_node() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = Workspace::new(dir.path());
        let store = GoalStore::for_workspace(&workspace);
        assert_eq!(
            store.path(),
            dir.path().join("goal").join("goal.json").as_path()
        );
        assert_eq!(store.session_id(), DEFAULT_SESSION_ID);
        assert_eq!(
            GoalStore::new("a/goal.json").path(),
            Path::new("a/goal.json")
        );
    }

    #[tokio::test]
    async fn missing_file_loads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        assert_eq!(store.load().await.unwrap(), None);
        // Clearing an absent record is a no-op.
        assert_eq!(store.clear().await.unwrap(), None);
        assert!(!store.path().exists());
    }

    #[tokio::test]
    async fn start_load_and_persist_across_instances() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("goal").join("goal.json");
        let store = GoalStore::for_session(&path, "s-1");
        let record = store.start("  ship it  ", now()).await.unwrap();
        assert_eq!(record.objective, "ship it");
        assert_eq!(record.session_id, "s-1");
        assert_eq!(record.status, GoalStatus::Active);

        let loaded = store.load().await.unwrap().unwrap();
        assert_eq!(loaded, record);

        // A second store over the same file sees the same record.
        let second = GoalStore::for_session(&path, "s-1");
        assert_eq!(second.load().await.unwrap(), Some(record.clone()));

        let persisted: Value =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert_eq!(persisted, record.to_dict());
        assert_eq!(persisted["objective"], json!("ship it"));
    }

    #[tokio::test]
    async fn start_rejects_an_empty_objective() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let error = store.start("   ", now()).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(error.message(), "goal objective must not be empty");
        assert!(!store.path().exists());
    }

    #[tokio::test]
    async fn start_replaces_an_existing_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let first = store.start("first goal", now()).await.unwrap();
        let second = store.start("second goal", later()).await.unwrap();
        assert_ne!(first.goal_id, second.goal_id);
        assert_eq!(store.load().await.unwrap().unwrap(), second);
    }

    #[tokio::test]
    async fn update_applies_the_closure_and_stamps_the_clock() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        assert_eq!(store.session_id(), "s-1");

        let updated = store
            .update(
                |record| {
                    record.bump_revision(later());
                    record.attempt_count += 1;
                    record.record_progress(
                        &GoalAssessment::new(GoalAssessmentStatus::Complete, "done", None, None),
                        later(),
                    );
                },
                later(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.revision, 1);
        assert_eq!(updated.attempt_count, 1);
        assert_eq!(updated.status, GoalStatus::Completed);
        assert_eq!(
            updated.updated_at,
            crate::tools::goal::schema::now_iso(later())
        );
        assert_eq!(updated.last_stop_reason.as_deref(), Some("completed"));

        let reloaded = GoalStore::for_session(store.path(), "s-1")
            .load()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reloaded, updated);
    }

    #[tokio::test]
    async fn update_without_a_record_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let outcome = store
            .update(|record| record.attempt_count = 3, now())
            .await
            .unwrap();
        assert_eq!(outcome, None);
        assert!(!store.path().exists());
    }

    #[tokio::test]
    async fn clear_removes_the_file_and_returns_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let record = store.start("ship it", now()).await.unwrap();
        assert_eq!(store.clear().await.unwrap(), Some(record));
        assert!(!store.path().exists());
        assert_eq!(store.load().await.unwrap(), None);
    }

    #[tokio::test]
    async fn malformed_json_is_a_validation_error_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("goal").join("goal.json");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&path, b"{not json").await.unwrap();

        let store = GoalStore::for_session(&path, "s-1");
        let error = store.load().await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert!(
            error
                .message()
                .starts_with(&format!("invalid goal record at {}", path.display())),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn an_invalid_record_is_a_validation_error_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("goal.json");
        tokio::fs::write(
            &path,
            serde_json::to_vec(&json!({
                "goal_id": "abc",
                "session_id": "s-1",
                "objective": "ship it",
                "created_at": "yesterday",
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let store = GoalStore::for_session(&path, "s-1");
        let error = store.load().await.unwrap_err();
        assert_eq!(
            error.message(),
            format!(
                "invalid goal record at {}: invalid persisted GoalRecord created_at",
                path.display()
            )
        );
    }

    #[tokio::test]
    async fn concurrent_updates_both_apply() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        let first = Arc::clone(&store);
        let second = Arc::clone(&store);
        let (a, b) = tokio::join!(
            async move {
                first
                    .update(|record| record.attempt_count += 1, now())
                    .await
            },
            async move {
                second
                    .update(|record| record.attempt_count += 1, now())
                    .await
            }
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(store.load().await.unwrap().unwrap().attempt_count, 2);
    }

    #[tokio::test]
    async fn save_writes_a_readable_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let record = GoalRecord::new("ship it", now()).with_session_id("s-9");
        store.save(&record).await.unwrap();
        let persisted: PathBuf = store.path().to_path_buf();
        assert!(persisted.exists());
        assert_eq!(store.load().await.unwrap(), Some(record));
    }
}
