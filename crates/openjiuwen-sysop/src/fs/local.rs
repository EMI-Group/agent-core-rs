//! The local filesystem backend: `LocalFsOperation`.
//!
//! This module is the Rust counterpart of Python's `local/fs_operation.py` and implements
//! [`FsOperation`] (the ten Python methods) plus [`FsExtOperation`] (the additive Rust
//! helpers). The sandbox backend implements the same traits, so agent code never needs to
//! know where an operation runs.
//!
//! # Path policy
//!
//! Every caller-supplied string goes through [`PathPolicy`] before any I/O: reads use
//! [`PathPolicy::resolve_existing`], writes and upload targets use
//! [`PathPolicy::resolve_for_write`]. That applies lexical `..` removal, resolves symbolic
//! links for the longest existing prefix, and enforces the sandbox roots. The one
//! deliberate exception is the *host* side of a transfer (the upload source and the
//! download destination), which Python never sandbox-checks; those paths are only
//! tilde-expanded and canonicalized.
//!
//! # In-process locks
//!
//! `LocalFsOperation::locks` maps a resolved path to a [`tokio::sync::RwLock`]. Readers
//! take the shared lock, writers the exclusive one, and a transfer locks both paths in a
//! deterministic order (sorted by path, `Write` winning over `Read`) so two transfers can
//! never deadlock. These are **in-process** locks: Python's `ReadWriteLockManager` uses
//! on-disk SQLite locks shared between processes, which is not ported because a single
//! process owns a workspace here. A read with `only_read = true` takes no lock at all
//! (Python `skip_lock`).
//!
//! # Blocking work
//!
//! All file I/O is awaited `tokio::fs`; the only synchronous calls are `stat`/`chmod` on
//! already-resolved paths (`set_permissions` for the best-effort permission application),
//! which are cheap metadata operations and never traverse a directory tree.
//!
//! # Failures
//!
//! A failure is a Python-compatible [`OpResult`] value with code `199003` (built by
//! [`crate::result::fs_failure`]) carrying the Python method name in the `execution` slot, or code
//! `199011` (`file version conflict`) for a rejected conditional write. Nothing panics and
//! no error is silently dropped.
//!
//! # Module layout
//!
//! This file is the facade plus the seam every filesystem operation shares: the path
//! policy, the configuration, and the per-path lock map. The operation bodies live in
//! sibling modules and are reused here as one-line delegations, so the trait impls stay
//! readable:
//!
//! - `fs::read` — `read_file`, `read_file_stream`, and the read preconditions
//! - `fs::write` — `write_file`, the conditional compare-and-swap, permissions
//! - `fs::transfer` — `upload_file`, `download_file`, their streams, and copying
//! - [`crate::fs::slice`] — pure read validation/slicing and write-content composition
//! - [`crate::fs::list`], [`crate::fs::search`], [`crate::fs::ext`] — listing, searching,
//!   and the extension operations
//!
//! ```
//! use openjiuwen_sysop::config::LocalWorkConfig;
//! use openjiuwen_sysop::fs::local::LocalFsOperation;
//! use openjiuwen_sysop::FsOperation;
//!
//! let backend = LocalFsOperation::new(LocalWorkConfig::default()).unwrap();
//! assert!(backend.supports_conditional_write());
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::config::LocalWorkConfig;
use crate::fs::types::{
    CopyData, CopyParams, DeleteData, DeleteParams, DownloadFileData, DownloadFileParams, FileStat,
    ListDirsParams, ListFilesData, ListFilesParams, MkdirData, MkdirParams, MoveData, MoveParams,
    ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
    SearchPatternMode, TransferChunkData, UploadFileData, UploadFileParams, WriteFileData,
    WriteFileParams,
};
use crate::fs::{ext, list, read, search, transfer, write};
use crate::operation::{FsExtOperation, FsOperation, OpStream};
use crate::result::OpResult;
use crate::workspace::PathPolicy;

/// The local filesystem backend (Python `local/fs_operation.py`).
///
/// Built by [`LocalFsOperation::new`] or through [`crate::fs::local_fs`]; it owns the
/// resolved [`PathPolicy`], the configuration it was built from, and the per-path lock
/// map. Cheap to share: wrap it in an `Arc` and clone that.
#[derive(Debug)]
pub struct LocalFsOperation {
    policy: PathPolicy,
    config: LocalWorkConfig,
    locks: Mutex<HashMap<PathBuf, Arc<RwLock<()>>>>,
}

impl LocalFsOperation {
    /// Builds a backend for `config`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when the configuration is
    /// invalid (a blank allowlist entry or sandbox root, an uncompilable dangerous pattern,
    /// a zero `max_output_bytes`, or a blank `workspace_root`).
    pub fn new(config: LocalWorkConfig) -> openjiuwen_common::error::Result<Self> {
        config.validate()?;
        let policy = PathPolicy::new(&config);
        Ok(Self {
            policy,
            config,
            locks: Mutex::new(HashMap::new()),
        })
    }

    /// The path policy every operation resolves through.
    #[must_use]
    pub fn policy(&self) -> &PathPolicy {
        &self.policy
    }

    /// The configuration this backend was built from.
    #[must_use]
    pub fn config(&self) -> &LocalWorkConfig {
        &self.config
    }

    // ----- locking -------------------------------------------------------------------

    /// Returns the lock guarding `path`, creating it on first use.
    ///
    /// A poisoned mutex is recovered rather than panicking: the map itself is never left in
    /// an inconsistent state, so the worst case is a stale per-path lock.
    fn lock_for(&self, path: &Path) -> Arc<RwLock<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(locks.entry(path.to_path_buf()).or_default())
    }

    /// Takes the shared lock for `path` (Python's default read lock).
    pub(crate) async fn read_guard(&self, path: &Path) -> OwnedRwLockReadGuard<()> {
        let lock = self.lock_for(path);
        lock.read_owned().await
    }

    /// Takes the exclusive lock for `path`.
    pub(crate) async fn write_guard(&self, path: &Path) -> OwnedRwLockWriteGuard<()> {
        let lock = self.lock_for(path);
        lock.write_owned().await
    }

    /// Takes several locks in a deadlock-free order.
    ///
    /// Requests are sorted by path and deduplicated, keeping the stronger mode when the same
    /// path is requested twice (a transfer whose source and destination coincide). The
    /// returned guards must be kept alive for the duration of the operation.
    pub(crate) async fn acquire_ordered(
        &self,
        mut requests: Vec<(PathBuf, LockMode)>,
    ) -> Vec<PathGuard> {
        requests.sort_by(|left, right| left.0.cmp(&right.0));
        let mut merged: Vec<(PathBuf, LockMode)> = Vec::with_capacity(requests.len());
        for (path, mode) in requests {
            match merged.last_mut() {
                Some((last_path, last_mode)) if *last_path == path => {
                    if mode == LockMode::Write {
                        *last_mode = LockMode::Write;
                    }
                }
                _ => merged.push((path, mode)),
            }
        }
        let mut guards = Vec::with_capacity(merged.len());
        for (path, mode) in merged {
            let lock = self.lock_for(&path);
            guards.push(match mode {
                LockMode::Read => PathGuard::Read(lock.read_owned().await),
                LockMode::Write => PathGuard::Write(lock.write_owned().await),
            });
        }
        guards
    }
}

#[async_trait]
impl FsOperation for LocalFsOperation {
    fn supports_conditional_write(&self) -> bool {
        true
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        read::read_file(self, params).await
    }

    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData> {
        read::read_file_stream(self, params)
    }

    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        write::write_file(self, params).await
    }

    fn upload_file_stream(
        self: Arc<Self>,
        params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        transfer::upload_file_stream(self, params)
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        transfer::upload_file(self, params).await
    }

    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        transfer::download_file_stream(self, params)
    }

    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData> {
        transfer::download_file(self, params).await
    }

    async fn list_files(&self, params: ListFilesParams) -> OpResult<ListFilesData> {
        list::list_files(self.policy(), &params).await
    }

    async fn list_directories(&self, params: ListDirsParams) -> OpResult<ListFilesData> {
        list::list_directories(self.policy(), &params).await
    }

    async fn search_files(&self, params: SearchFilesParams) -> OpResult<SearchFilesData> {
        search::search_files(self.policy(), &params, SearchPatternMode::Glob).await
    }
}

#[async_trait]
impl FsExtOperation for LocalFsOperation {
    async fn stat(&self, path: &str) -> OpResult<FileStat> {
        ext::stat(self.policy(), path).await
    }

    async fn exists(&self, path: &str) -> OpResult<bool> {
        ext::exists(self.policy(), path).await
    }

    async fn mkdir(&self, params: MkdirParams) -> OpResult<MkdirData> {
        ext::mkdir(self.policy(), &params).await
    }

    async fn delete(&self, params: DeleteParams) -> OpResult<DeleteData> {
        ext::delete(self.policy(), &params).await
    }

    async fn copy(&self, params: CopyParams) -> OpResult<CopyData> {
        ext::copy(self.policy(), &params).await
    }

    async fn move_path(&self, params: MoveParams) -> OpResult<MoveData> {
        ext::move_path(self.policy(), &params).await
    }

    async fn search_files_with(
        &self,
        params: SearchFilesParams,
        mode: SearchPatternMode,
    ) -> OpResult<SearchFilesData> {
        search::search_files(self.policy(), &params, mode).await
    }
}

/// How a path is locked for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LockMode {
    /// Shared access, used by readers.
    Read,
    /// Exclusive access, used by writers.
    Write,
}

/// An acquired per-path lock, held for as long as the value lives.
#[allow(dead_code)] // RAII holder: the guards exist only to keep the locks held
pub(crate) enum PathGuard {
    /// A shared guard.
    Read(OwnedRwLockReadGuard<()>),
    /// An exclusive guard.
    Write(OwnedRwLockWriteGuard<()>),
}

/// Converts a byte count to `u64`, saturating instead of wrapping.
pub(crate) fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
pub(crate) mod test_util {
    //! Fixtures shared by the filesystem backend unit tests.

    use std::sync::Arc;

    use tempfile::TempDir;

    use super::LocalFsOperation;
    use crate::config::LocalWorkConfig;
    use crate::util::format_path;

    /// A backend rooted at `dir` without sandbox restrictions.
    pub(crate) fn backend(dir: &TempDir) -> Arc<LocalFsOperation> {
        let config = LocalWorkConfig::default().with_workspace(format_path(dir.path()));
        Arc::new(LocalFsOperation::new(config).expect("the default workspace config is valid"))
    }

    /// A backend whose paths must stay inside `dir`.
    pub(crate) fn restricted(dir: &TempDir) -> Arc<LocalFsOperation> {
        let config = LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_sandbox_roots([format_path(dir.path())]);
        Arc::new(LocalFsOperation::new(config).expect("the restricted workspace config is valid"))
    }

    /// Canonical path of `dir/name`, matching what `PathPolicy` resolves to.
    pub(crate) async fn canonical(dir: &TempDir, name: &str) -> String {
        format_path(
            &tokio::fs::canonicalize(dir.path().join(name))
                .await
                .expect("the path exists"),
        )
    }

    /// The bytes written to `dir/name`.
    pub(crate) fn read_back(dir: &TempDir, name: &str) -> Vec<u8> {
        std::fs::read(dir.path().join(name)).expect("the file exists")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::test_util::backend;
    use super::*;
    use crate::fs::types::{ListFilesParams, SearchFilesParams};

    // ----- delegated operations -----------------------------------------------------

    #[tokio::test]
    async fn list_and_search_delegate_to_the_shared_helpers() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"a").unwrap();
        std::fs::write(dir.path().join("sub/b.rs"), b"b").unwrap();
        let op = backend(&dir);

        let listing = op
            .list_files(ListFilesParams::new("."))
            .await
            .into_result()
            .unwrap();
        assert_eq!(listing.list_items.len(), 1);
        assert_eq!(listing.list_items[0].name, "a.txt");

        let glob = op
            .search_files(SearchFilesParams::new(".", "**/*.rs"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(glob.total_matches, 1);

        let regex = op
            .search_files_with(
                SearchFilesParams::new(".", r"^b\."),
                SearchPatternMode::Regex,
            )
            .await
            .into_result()
            .unwrap();
        assert_eq!(regex.total_matches, 1);

        assert!(op.exists("a.txt").await.into_result().unwrap());
        assert_eq!(op.stat("a.txt").await.into_result().unwrap().size, 1);
        let made = op
            .mkdir(MkdirParams::new("fresh"))
            .await
            .into_result()
            .unwrap();
        assert!(made.created);
        op.delete(DeleteParams::new("fresh"))
            .await
            .into_result()
            .unwrap();
        let copied = op
            .copy(CopyParams::new("a.txt", "b.txt"))
            .await
            .into_result()
            .unwrap();
        assert!(copied.target_path.ends_with("b.txt"));
        op.move_path(MoveParams::new("b.txt", "c.txt"))
            .await
            .into_result()
            .unwrap();
        assert!(dir.path().join("c.txt").exists());
    }

    #[test]
    fn new_rejects_an_invalid_configuration() {
        let config = LocalWorkConfig {
            max_output_bytes: 0,
            ..LocalWorkConfig::default()
        };
        assert!(LocalFsOperation::new(config).is_err());
        let backend = LocalFsOperation::new(LocalWorkConfig::default()).unwrap();
        assert_eq!(backend.config(), &LocalWorkConfig::default());
        assert!(!backend.policy().is_restricted());
    }

    #[test]
    fn local_fs_factory_returns_a_working_backend() {
        let dir = tempfile::tempdir().unwrap();
        let config =
            LocalWorkConfig::default().with_workspace(crate::util::format_path(dir.path()));
        let backend = crate::fs::local_fs(config).unwrap();
        assert!(backend.supports_conditional_write());
    }
}
