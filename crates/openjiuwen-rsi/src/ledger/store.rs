//! Persistence backends for the run ledger.
//!
//! The [`LedgerStore`] seam separates the ledger's deduplication and
//! aggregation logic from where records live. Two backends ship here:
//!
//! - [`InMemoryLedgerStore`] — a process-local map, the reference backend for
//!   tests and ephemeral runs.
//! - [`FileLedgerStore`] — one JSONL file per task under a root directory,
//!   matching Python's `model_calls.jsonl` layout (one flat record per line).
//!
//! Both are usable behind `Arc<dyn LedgerStore>` and neither ever silently drops
//! data: a malformed JSONL line is reported as [`Error::session`], not skipped.
// Only the file backend awaits; the in-memory backend keeps the trait's async shape.
#![allow(clippy::unused_async_trait_impl)]
use crate::ledger::record::RunEvent;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock};
use tokio::io::AsyncWriteExt;

/// Durable storage for one task's run events.
///
/// Backends append single events, load the whole history in write order, and
/// remove a task's history. Loading a task with no history returns an empty
/// vector rather than [`Error::NotFound`]; a missing history is a normal state
/// for a fresh run.
#[async_trait]
pub trait LedgerStore: Send + Sync {
    /// Appends one event to `task_id`'s history.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the event cannot be serialised or the
    /// backend cannot write it.
    async fn append(&self, task_id: &str, event: &RunEvent) -> Result<()>;

    /// Loads `task_id`'s events in write order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the backend cannot read the history or a
    /// line is not a valid [`RunEvent`].
    async fn load(&self, task_id: &str) -> Result<Vec<RunEvent>>;

    /// Removes `task_id`'s history, ignoring a missing history.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the backend cannot delete the history.
    async fn remove(&self, task_id: &str) -> Result<()>;
}

/// A [`LedgerStore`] keeping events in process memory.
///
/// The map lives behind a [`std::sync::RwLock`] held only for map operations,
/// never across an `.await`.
#[derive(Debug, Default)]
pub struct InMemoryLedgerStore {
    events: RwLock<BTreeMap<String, Vec<RunEvent>>>,
}

impl InMemoryLedgerStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of `task_id`'s events (empty when none were recorded).
    #[must_use]
    pub fn events(&self, task_id: &str) -> Vec<RunEvent> {
        self.read().get(task_id).cloned().unwrap_or_default()
    }

    /// Returns the ids of every task with recorded events, sorted.
    #[must_use]
    pub fn tasks(&self) -> Vec<String> {
        self.read().keys().cloned().collect()
    }

    /// Locks the map for reading, recovering from a poisoned lock.
    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, Vec<RunEvent>>> {
        self.events.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the map for writing, recovering from a poisoned lock.
    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Vec<RunEvent>>> {
        self.events.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl LedgerStore for InMemoryLedgerStore {
    async fn append(&self, task_id: &str, event: &RunEvent) -> Result<()> {
        self.write()
            .entry(task_id.to_owned())
            .or_default()
            .push(event.clone());
        Ok(())
    }

    async fn load(&self, task_id: &str) -> Result<Vec<RunEvent>> {
        Ok(self.events(task_id))
    }

    async fn remove(&self, task_id: &str) -> Result<()> {
        self.write().remove(task_id);
        Ok(())
    }
}

/// A [`LedgerStore`] writing one JSONL file per task under `root`.
///
/// The file name is the task id with every character outside
/// `[A-Za-z0-9_-]` replaced by `_`, so a task id can never escape `root`.
/// Appends open the file with create-and-append, write exactly one JSON line,
/// and flush; a reader therefore never rewrites the history. This matches
/// Python's append-only `model_calls.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLedgerStore {
    root: PathBuf,
}

impl FileLedgerStore {
    /// Creates a store rooted at `root`; the directory is created on first append.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the JSONL path used for `task_id`.
    #[must_use]
    pub fn path_for(&self, task_id: &str) -> PathBuf {
        self.root
            .join(format!("{}.jsonl", sanitize_task_id(task_id)))
    }
}

#[async_trait]
impl LedgerStore for FileLedgerStore {
    async fn append(&self, task_id: &str, event: &RunEvent) -> Result<()> {
        let path = self.path_for(task_id);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::session(format!(
                    "creating run ledger directory `{}`: {error}",
                    parent.display()
                ))
            })?;
        }
        let line = serde_json::to_string(event)
            .map_err(|error| Error::session(format!("serialising a run ledger event: {error}")))?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|error| {
                Error::session(format!("opening run ledger `{}`: {error}", path.display()))
            })?;
        let mut payload = line;
        payload.push('\n');
        file.write_all(payload.as_bytes()).await.map_err(|error| {
            Error::session(format!("writing run ledger `{}`: {error}", path.display()))
        })?;
        file.flush().await.map_err(|error| {
            Error::session(format!("flushing run ledger `{}`: {error}", path.display()))
        })
    }

    async fn load(&self, task_id: &str) -> Result<Vec<RunEvent>> {
        let path = self.path_for(task_id);
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(Error::session(format!(
                    "reading run ledger `{}`: {error}",
                    path.display()
                )));
            }
        };
        let mut events = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event = serde_json::from_str::<RunEvent>(line).map_err(|error| {
                Error::session(format!(
                    "run ledger line {} of task `{task_id}` is not a valid event: {error}",
                    index + 1
                ))
            })?;
            events.push(event);
        }
        Ok(events)
    }

    async fn remove(&self, task_id: &str) -> Result<()> {
        let path = self.path_for(task_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::session(format!(
                "removing run ledger `{}`: {error}",
                path.display()
            ))),
        }
    }
}

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`.
///
/// The result is never empty and never contains a path separator or `..`, so a
/// caller-controlled task id cannot point outside the store root.
fn sanitize_task_id(task_id: &str) -> String {
    let mut name: String = task_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if name.is_empty() {
        name.push('_');
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_task_id_keeps_names_inside_the_root() {
        assert_eq!(sanitize_task_id("6f0d0b3e-1c2a"), "6f0d0b3e-1c2a");
        assert_eq!(sanitize_task_id("run_1"), "run_1");
        assert_eq!(sanitize_task_id("../evil"), "___evil");
        assert_eq!(sanitize_task_id("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_task_id(""), "_");
        assert_eq!(sanitize_task_id("таск"), "____");
    }

    #[test]
    fn path_for_stays_under_the_root() {
        let store = FileLedgerStore::new("/tmp/ledger-root");
        assert_eq!(
            store.path_for("../evil"),
            PathBuf::from("/tmp/ledger-root/___evil.jsonl")
        );
        assert_eq!(store.root(), Path::new("/tmp/ledger-root"));
    }
}
