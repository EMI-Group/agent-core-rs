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
//! [`LocalFsOperation::locks`] maps a resolved path to a [`tokio::sync::RwLock`]. Readers
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
//! [`fs_failure`]) carrying the Python method name in the `execution` slot, or code
//! `199011` (`file version conflict`) for a rejected conditional write. Nothing panics and
//! no error is silently dropped.
//!
//! # Why this file is large
//!
//! The crate keeps one file per backend responsibility, which fixes this file as the seam
//! for the whole local backend: the facade, the lock map, read/write/transfer, and both
//! streaming implementations. The pure slicing logic lives in [`crate::fs::slice`], and
//! listing, searching, and the extension operations live in
//! [`crate::fs::list`], [`crate::fs::search`], and [`crate::fs::ext`] — none of them owns a
//! transfer helper, so the copy/stream machinery has to stay here. The implementation is
//! about a thousand lines and the unit tests below add roughly as many.
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
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::config::LocalWorkConfig;
use crate::constants::{
    AUTO_ENCODING, DEFAULT_READ_STREAM_CHUNK_SIZE, DEFAULT_TRANSFER_CHUNK_SIZE,
};
use crate::fs::types::{
    CopyData, CopyParams, DeleteData, DeleteParams, DownloadFileData, DownloadFileParams,
    FileContent, FileStat, ListDirsParams, ListFilesData, ListFilesParams, MkdirData, MkdirParams,
    MoveData, MoveParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode, SearchFilesData,
    SearchFilesParams, SearchPatternMode, TransferChunkData, UploadFileData, UploadFileParams,
    WriteFileData, WriteFileParams,
};
use crate::fs::{ext, list, search, slice};
use crate::operation::{FsExtOperation, FsOperation, OpStream};
use crate::result::{fs_failure, OpResult};
use crate::status::{file_version_conflict_message, SysOpStatus};
use crate::util::{format_path, sha256_hex};
use crate::workspace::{canonicalize_containing, PathPolicy};

/// `execution` slot of a `read_file` failure.
const READ_FILE: &str = "read_file";
/// `execution` slot of a `read_file_stream` failure.
const READ_FILE_STREAM: &str = "read_file_stream";
/// `execution` slot of a `write_file` failure.
const WRITE_FILE: &str = "write_file";
/// `execution` slot of an `upload_file` failure.
const UPLOAD_FILE: &str = "upload_file";
/// `execution` slot of an `upload_file_stream` failure.
const UPLOAD_FILE_STREAM: &str = "upload_file_stream";
/// `execution` slot of a `download_file` failure.
const DOWNLOAD_FILE: &str = "download_file";
/// `execution` slot of a `download_file_stream` failure.
const DOWNLOAD_FILE_STREAM: &str = "download_file_stream";

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
    async fn read_guard(&self, path: &Path) -> OwnedRwLockReadGuard<()> {
        let lock = self.lock_for(path);
        lock.read_owned().await
    }

    /// Takes the exclusive lock for `path`.
    async fn write_guard(&self, path: &Path) -> OwnedRwLockWriteGuard<()> {
        let lock = self.lock_for(path);
        lock.write_owned().await
    }

    /// Takes several locks in a deadlock-free order.
    ///
    /// Requests are sorted by path and deduplicated, keeping the stronger mode when the same
    /// path is requested twice (a transfer whose source and destination coincide). The
    /// returned guards must be kept alive for the duration of the operation.
    async fn acquire_ordered(&self, mut requests: Vec<(PathBuf, LockMode)>) -> Vec<PathGuard> {
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

    // ----- shared preparation --------------------------------------------------------

    /// Applies every read precondition and returns the file contents.
    ///
    /// The line-selection validation is *not* part of this step; callers run
    /// [`slice::validate_read_params`] first so a bad request is rejected before the path is
    /// even resolved (Python's order).
    async fn load_for_read(&self, params: &ReadFileParams) -> Result<LoadedFile, String> {
        let path = self
            .policy
            .resolve_existing(&params.path)
            .await
            .map_err(|error| error.message().to_string())?;
        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|error| metadata_reason(&path, &error))?;
        if !metadata.is_file() {
            return Err(if metadata.is_dir() {
                format!("Is a directory: {}", format_path(&path))
            } else {
                format!("not a file: {}", format_path(&path))
            });
        }
        if let Some(limit) = self.policy.max_file_size() {
            if metadata.len() > limit {
                return Err(format!(
                    "file size {} exceeds the configured limit {limit} bytes",
                    metadata.len()
                ));
            }
        }
        let bytes = self.read_bytes_locked(&path, params.only_read).await?;
        Ok(LoadedFile {
            path,
            size: metadata.len(),
            bytes,
        })
    }

    /// Reads `path`, holding the shared lock unless `only_read` skips it.
    async fn read_bytes_locked(&self, path: &Path, only_read: bool) -> Result<Vec<u8>, String> {
        if only_read {
            return tokio::fs::read(path)
                .await
                .map_err(|error| error.to_string());
        }
        let guard = self.read_guard(path).await;
        let result = tokio::fs::read(path).await;
        drop(guard);
        result.map_err(|error| error.to_string())
    }

    /// Resolves and validates an upload (host → workspace) transfer.
    async fn prepare_upload(
        &self,
        params: &UploadFileParams,
    ) -> Result<(PathBuf, PathBuf), String> {
        if params.local_path.trim().is_empty() {
            return Err("source path must not be empty".to_string());
        }
        let source = canonicalize_containing(&expand_host_path(&params.local_path)).await;
        let destination = self
            .policy
            .resolve(&params.target_path, params.create_parent_dirs)
            .await
            .map_err(|error| error.message().to_string())?;
        ensure_source_file(&source).await?;
        ensure_target_free(&destination, params.overwrite).await?;
        Ok((source, destination))
    }

    /// Resolves and validates a download (workspace → host) transfer.
    async fn prepare_download(
        &self,
        params: &DownloadFileParams,
    ) -> Result<(PathBuf, PathBuf), String> {
        if params.local_path.trim().is_empty() {
            return Err("target path must not be empty".to_string());
        }
        let source = self
            .policy
            .resolve_existing(&params.source_path)
            .await
            .map_err(|error| error.message().to_string())?;
        let destination = canonicalize_containing(&expand_host_path(&params.local_path)).await;
        ensure_source_file(&source).await?;
        ensure_target_free(&destination, params.overwrite).await?;
        if params.create_parent_dirs {
            create_parent_dirs(&destination).await?;
        }
        Ok((source, destination))
    }

    /// Copies one file into another under both paths' locks, returning the bytes copied.
    async fn transfer(
        &self,
        source: &Path,
        destination: &Path,
        chunk_size: u64,
        preserve_permissions: bool,
    ) -> Result<u64, String> {
        let guards = self
            .acquire_ordered(vec![
                (source.to_path_buf(), LockMode::Read),
                (destination.to_path_buf(), LockMode::Write),
            ])
            .await;
        let metadata = source_metadata(source, preserve_permissions).await;
        let copied = copy_file(source, destination, chunk_size).await;
        drop(guards);
        if copied.is_ok() {
            apply_copied_permissions(metadata.as_ref(), destination);
        }
        copied
    }

    /// Builds a streamed transfer, resolving and validating before the first chunk.
    fn transfer_stream(
        self: Arc<Self>,
        source: PathBuf,
        destination: PathBuf,
        chunk_size: u64,
        preserve_permissions: bool,
        execution: &'static str,
    ) -> OpStream<TransferChunkData> {
        Box::pin(async_stream::stream! {
            let guards = self
                .acquire_ordered(vec![
                    (source.clone(), LockMode::Read),
                    (destination.clone(), LockMode::Write),
                ])
                .await;
            let mut chunks = copy_chunks(source, destination, chunk_size, preserve_permissions, execution);
            while let Some(chunk) = chunks.next().await {
                yield chunk;
            }
            drop(guards);
        })
    }

    /// Prepares a streamed read: validation, resolution, decoding, and selection.
    async fn prepare_read_stream(&self, params: &ReadFileParams) -> Result<ReadStreamPlan, String> {
        let selection = slice::validate_read_params(params)?;
        let loaded = self.load_for_read(params).await?;
        let path = format_path(&loaded.path);
        match params.mode {
            ReadMode::Text => {
                let text =
                    crate::encoding::decode_text(&loaded.bytes, &params.encoding, params.binary)
                        .map_err(|error| error.message().to_string())?;
                let encoding = if uses_auto_encoding(&params.encoding) {
                    crate::encoding::detected_encoding_name(&loaded.bytes)
                } else {
                    params.encoding.clone()
                };
                let lines = slice::split_lines_keep_ends(&text);
                let selected = slice::select_lines(&lines, selection)
                    .into_iter()
                    .map(str::to_string)
                    .collect();
                Ok(ReadStreamPlan {
                    path,
                    mode: params.mode,
                    body: StreamBody::Lines {
                        lines: selected,
                        encoding,
                        tail: matches!(selection, Some(slice::LineSelection::Tail(_))),
                    },
                })
            }
            ReadMode::Bytes => Ok(ReadStreamPlan {
                path,
                mode: params.mode,
                body: StreamBody::Bytes {
                    bytes: loaded.bytes,
                    chunk_size: slice::effective_chunk_size(
                        params.chunk_size,
                        DEFAULT_READ_STREAM_CHUNK_SIZE,
                    )
                    .max(1),
                },
            }),
        }
    }
}

#[async_trait]
impl FsOperation for LocalFsOperation {
    fn supports_conditional_write(&self) -> bool {
        true
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        let selection = match slice::validate_read_params(&params) {
            Ok(selection) => selection,
            Err(reason) => return fs_failure(READ_FILE, &reason),
        };
        let loaded = match self.load_for_read(&params).await {
            Ok(loaded) => loaded,
            Err(reason) => return fs_failure(READ_FILE, &reason),
        };
        let path = format_path(&loaded.path);
        let (content, encoding, truncated) = match params.mode {
            ReadMode::Text => {
                let text = match crate::encoding::decode_text(
                    &loaded.bytes,
                    &params.encoding,
                    params.binary,
                ) {
                    Ok(text) => text,
                    Err(error) => return fs_failure(READ_FILE, &error.message()),
                };
                let lines = slice::split_lines_keep_ends(&text);
                let content = FileContent::Text(slice::select_lines(&lines, selection).concat());
                let encoding = uses_auto_encoding(&params.encoding)
                    .then(|| crate::encoding::detected_encoding_name(&loaded.bytes));
                (content, encoding, false)
            }
            ReadMode::Bytes => {
                let (chunk, truncated) = slice::take_first_chunk(&loaded.bytes, params.chunk_size);
                (FileContent::Bytes(chunk.to_vec()), None, truncated)
            }
        };
        OpResult::success(ReadFileData {
            path,
            content,
            mode: params.mode,
            encoding,
            truncated,
            total_bytes: loaded.size,
        })
    }

    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData> {
        Box::pin(async_stream::stream! {
            let plan = match self.prepare_read_stream(&params).await {
                Ok(plan) => plan,
                Err(reason) => {
                    yield fs_failure(READ_FILE_STREAM, &reason);
                    return;
                }
            };
            let ReadStreamPlan { path, mode, body } = plan;
            match body {
                StreamBody::Lines { lines, encoding, tail } => {
                    if lines.is_empty() {
                        if tail {
                            yield OpResult::success(ReadChunkData {
                                path,
                                chunk: FileContent::Text(String::new()),
                                mode,
                                chunk_size: 0,
                                chunk_index: 0,
                                is_last_chunk: true,
                            });
                        }
                        return;
                    }
                    let last = lines.len() - 1;
                    for (index, line) in lines.into_iter().enumerate() {
                        let chunk_size = slice::encoded_len(&line, &encoding);
                        yield OpResult::success(ReadChunkData {
                            path: path.clone(),
                            chunk: FileContent::Text(line),
                            mode,
                            chunk_size,
                            chunk_index: len_u64(index),
                            is_last_chunk: index == last,
                        });
                    }
                }
                StreamBody::Bytes { bytes, chunk_size } => {
                    let total = bytes.len().div_ceil(chunk_size);
                    for (index, chunk) in bytes.chunks(chunk_size).enumerate() {
                        yield OpResult::success(ReadChunkData {
                            path: path.clone(),
                            chunk: FileContent::Bytes(chunk.to_vec()),
                            mode,
                            chunk_size: len_u64(chunk.len()),
                            chunk_index: len_u64(index),
                            is_last_chunk: index + 1 == total,
                        });
                    }
                }
            }
        })
    }

    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        if let Err(reason) = validate_write(&params) {
            return fs_failure(WRITE_FILE, &reason);
        }
        let path = match self.policy.resolve_for_write(&params.path).await {
            Ok(path) => path,
            Err(error) => return fs_failure(WRITE_FILE, &error.message()),
        };
        let metadata = tokio::fs::metadata(&path).await.ok();
        if metadata.as_ref().is_some_and(std::fs::Metadata::is_dir) {
            return fs_failure(
                WRITE_FILE,
                &format!("Target path is a directory: {}", format_path(&path)),
            );
        }
        let exists = metadata.is_some();
        if !params.create_if_not_exist
            && !exists
            && params.expected_content_sha256.is_none()
            && !params.create_if_absent
        {
            return fs_failure(
                WRITE_FILE,
                &format!("File does not exist: {}", format_path(&path)),
            );
        }
        let guard = self.write_guard(&path).await;
        let result = write_locked(&path, &params).await;
        drop(guard);
        result
    }

    fn upload_file_stream(
        self: Arc<Self>,
        params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(async_stream::stream! {
            let (source, destination) = match self.prepare_upload(&params).await {
                Ok(pair) => pair,
                Err(reason) => {
                    yield fs_failure(UPLOAD_FILE_STREAM, &reason);
                    return;
                }
            };
            let chunk_size = transfer_chunk_size(params.chunk_size);
            let mut chunks = self.transfer_stream(
                source,
                destination,
                chunk_size,
                params.preserve_permissions,
                UPLOAD_FILE_STREAM,
            );
            while let Some(chunk) = chunks.next().await {
                yield chunk;
            }
        })
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        let (source, destination) = match self.prepare_upload(&params).await {
            Ok(pair) => pair,
            Err(reason) => return fs_failure(UPLOAD_FILE, &reason),
        };
        let size = match self
            .transfer(
                &source,
                &destination,
                params.chunk_size,
                params.preserve_permissions,
            )
            .await
        {
            Ok(size) => size,
            Err(reason) => return fs_failure(UPLOAD_FILE, &reason),
        };
        OpResult::success(UploadFileData {
            local_path: format_path(&source),
            target_path: format_path(&destination),
            size,
        })
    }

    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        Box::pin(async_stream::stream! {
            let (source, destination) = match self.prepare_download(&params).await {
                Ok(pair) => pair,
                Err(reason) => {
                    yield fs_failure(DOWNLOAD_FILE_STREAM, &reason);
                    return;
                }
            };
            let chunk_size = transfer_chunk_size(params.chunk_size);
            let mut chunks = self.transfer_stream(
                source,
                destination,
                chunk_size,
                params.preserve_permissions,
                DOWNLOAD_FILE_STREAM,
            );
            while let Some(chunk) = chunks.next().await {
                yield chunk;
            }
        })
    }

    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData> {
        let (source, destination) = match self.prepare_download(&params).await {
            Ok(pair) => pair,
            Err(reason) => return fs_failure(DOWNLOAD_FILE, &reason),
        };
        let size = match self
            .transfer(
                &source,
                &destination,
                params.chunk_size,
                params.preserve_permissions,
            )
            .await
        {
            Ok(size) => size,
            Err(reason) => return fs_failure(DOWNLOAD_FILE, &reason),
        };
        OpResult::success(DownloadFileData {
            source_path: format_path(&source),
            local_path: format_path(&destination),
            size,
        })
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

/// A file read and validated for streaming.
struct LoadedFile {
    /// Resolved absolute path.
    path: PathBuf,
    /// Full file size in bytes, captured before the read.
    size: u64,
    /// The whole file contents.
    bytes: Vec<u8>,
}

/// What a streamed read will emit, decided before the first item.
enum StreamBody {
    /// One chunk per selected line.
    Lines {
        /// Selected lines, each keeping its terminator.
        lines: Vec<String>,
        /// Encoding used to size the chunks.
        encoding: String,
        /// Whether a tail selection produced the lines (Python emits one empty chunk when
        /// a tail selects nothing).
        tail: bool,
    },
    /// Fixed-size byte chunks.
    Bytes {
        /// The whole file.
        bytes: Vec<u8>,
        /// Normalized chunk size, always at least one.
        chunk_size: usize,
    },
}

/// A prepared streamed read.
struct ReadStreamPlan {
    /// Resolved path reported on every chunk.
    path: String,
    /// Text or bytes.
    mode: ReadMode,
    /// The chunks to emit.
    body: StreamBody,
}

/// How a path is locked for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockMode {
    /// Shared access, used by readers.
    Read,
    /// Exclusive access, used by writers.
    Write,
}

/// An acquired per-path lock, held for as long as the value lives.
#[allow(dead_code)] // RAII holder: the guards exist only to keep the locks held
enum PathGuard {
    /// A shared guard.
    Read(OwnedRwLockReadGuard<()>),
    /// An exclusive guard.
    Write(OwnedRwLockWriteGuard<()>),
}

/// Converts a byte count to `u64`, saturating instead of wrapping.
fn len_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Returns `true` when the request selects encoding detection (Python `_uses_auto_encoding`).
fn uses_auto_encoding(encoding: &str) -> bool {
    encoding.trim().eq_ignore_ascii_case(AUTO_ENCODING)
}

/// Builds the reason for a failed `metadata` call, using Python's missing-file wording.
fn metadata_reason(path: &Path, error: &std::io::Error) -> String {
    if error.kind() == ErrorKind::NotFound {
        return format!("File does not exist: {}: {error}", format_path(path));
    }
    error.to_string()
}

/// Builds Python's `file version conflict` failure (code `199011`).
fn version_conflict<T>() -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::FileVersionConflict,
        file_version_conflict_message(),
    )
}

/// Rejects the write flag combinations Python refuses before touching the filesystem.
fn validate_write(params: &WriteFileParams) -> Result<(), String> {
    let conditional = params.expected_content_sha256.is_some() || params.create_if_absent;
    let incompatible = (params.expected_content_sha256.is_some() && params.create_if_absent)
        || (conditional && params.append)
        || (params.create_if_absent && !params.create_if_not_exist);
    if incompatible {
        return Err("conditional write options are incompatible".to_string());
    }
    Ok(())
}

/// Writes `params` into `path` while the exclusive lock is held.
///
/// The conditional checks run *before* the file is opened for writing, so a rejected
/// compare-and-swap leaves the target byte-identical.
async fn write_locked(path: &Path, params: &WriteFileParams) -> OpResult<WriteFileData> {
    let exists = tokio::fs::metadata(path).await.is_ok();
    if params.create_if_absent && exists {
        return version_conflict();
    }
    if let Some(expected) = params.expected_content_sha256.as_deref() {
        match tokio::fs::read(path).await {
            Ok(current) => {
                if sha256_hex(&current) != expected {
                    return version_conflict();
                }
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return version_conflict(),
            Err(error) => return fs_failure(WRITE_FILE, &error.to_string()),
        }
    }
    let bytes = match slice::write_bytes(params) {
        Ok(bytes) => bytes,
        Err(reason) => return fs_failure(WRITE_FILE, &reason),
    };
    let mut options = tokio::fs::OpenOptions::new();
    if params.create_if_absent {
        options.create_new(true).write(true);
    } else if params.append {
        options.append(true).create(true);
    } else {
        options.write(true).create(true).truncate(true);
    }
    let mut file = match options.open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::AlreadyExists && params.create_if_absent => {
            return version_conflict();
        }
        Err(error) => return fs_failure(WRITE_FILE, &error.to_string()),
    };
    if let Err(error) = file.write_all(&bytes).await {
        return fs_failure(WRITE_FILE, &error.to_string());
    }
    // `tokio::fs::File` can complete a write into an internal buffer, so the flush must be
    // awaited before the file is closed and read back.
    if let Err(error) = file.flush().await {
        return fs_failure(WRITE_FILE, &error.to_string());
    }
    drop(file);
    apply_permissions(path, params.permissions.as_deref());
    OpResult::success(WriteFileData {
        path: format_path(path),
        size: len_u64(bytes.len()),
        mode: params.content.mode(),
    })
}

/// Applies an octal permission string, best-effort, on Unix only.
fn apply_permissions(path: &Path, permissions: Option<&str>) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let Some(permissions) = permissions else {
            return;
        };
        let trimmed = permissions.trim();
        let Ok(mode) = u32::from_str_radix(trimmed.trim_start_matches('0'), 8)
            .or_else(|_| u32::from_str_radix(trimmed, 8))
        else {
            tracing::warn!(
                path = %format_path(path),
                permissions = trimmed,
                "ignoring an invalid octal permission string"
            );
            return;
        };
        if let Err(error) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)) {
            tracing::warn!(
                path = %format_path(path),
                permissions = trimmed,
                %error,
                "failed to apply permissions"
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (path, permissions);
    }
}

/// Captures the source metadata before a copy so its permissions can be applied afterwards.
async fn source_metadata(source: &Path, preserve: bool) -> Option<std::fs::Metadata> {
    if preserve {
        return tokio::fs::metadata(source).await.ok();
    }
    None
}

/// Applies copied permissions to `destination`, best-effort, on Unix only.
fn apply_copied_permissions(source: Option<&std::fs::Metadata>, destination: &Path) {
    #[cfg(unix)]
    {
        let Some(source) = source else {
            return;
        };
        if let Err(error) = std::fs::set_permissions(destination, source.permissions()) {
            tracing::warn!(
                path = %format_path(destination),
                %error,
                "failed to preserve permissions"
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (source, destination);
    }
}

/// Expands a leading `~` or `~/…` with `$HOME`.
///
/// Python calls `pathlib.Path.expanduser()`; only the bare `~` and `~/` forms are supported
/// here, so a `~user` prefix is treated as a literal path segment.
fn expand_host_path(raw: &str) -> PathBuf {
    let raw = raw.trim();
    let expanded = if raw == "~" {
        std::env::var_os("HOME").map(PathBuf::from)
    } else if let Some(rest) = raw.strip_prefix("~/") {
        std::env::var_os("HOME").map(|home| {
            let mut path = PathBuf::from(home);
            path.push(rest);
            path
        })
    } else {
        None
    };
    expanded.unwrap_or_else(|| PathBuf::from(raw))
}

/// Normalizes a streaming transfer chunk size: `0` means the 1 MiB default.
fn transfer_chunk_size(requested: u64) -> u64 {
    if requested == 0 {
        DEFAULT_TRANSFER_CHUNK_SIZE
    } else {
        requested
    }
}

/// Rejects a source that is missing or is not a regular file.
async fn ensure_source_file(source: &Path) -> Result<(), String> {
    let is_file = tokio::fs::metadata(source)
        .await
        .is_ok_and(|metadata| metadata.is_file());
    if is_file {
        return Ok(());
    }
    Err(format!("Source not found: {}", format_path(source)))
}

/// Rejects an existing destination unless the caller allows overwriting it.
async fn ensure_target_free(destination: &Path, overwrite: bool) -> Result<(), String> {
    if !overwrite && tokio::fs::metadata(destination).await.is_ok() {
        return Err(format!("Target exists: {}", format_path(destination)));
    }
    Ok(())
}

/// Creates the missing parent directories of `destination`.
async fn create_parent_dirs(destination: &Path) -> Result<(), String> {
    let Some(parent) = destination.parent() else {
        return Ok(());
    };
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| format!("create parent dirs: {error}"))
}

/// Copies `source` to `destination`, returning the number of bytes written.
///
/// `chunk_size == 0` reads the whole file in one pass (Python `read()`), otherwise the copy
/// runs in `chunk_size` slices. The destination is created (and truncated) even for an
/// empty source.
async fn copy_file(source: &Path, destination: &Path, chunk_size: u64) -> Result<u64, String> {
    let mut input = tokio::fs::File::open(source)
        .await
        .map_err(|error| error.to_string())?;
    let mut output = tokio::fs::File::create(destination)
        .await
        .map_err(|error| error.to_string())?;
    if chunk_size == 0 {
        let mut bytes = Vec::new();
        input
            .read_to_end(&mut bytes)
            .await
            .map_err(|error| error.to_string())?;
        output
            .write_all(&bytes)
            .await
            .map_err(|error| error.to_string())?;
        output.flush().await.map_err(|error| error.to_string())?;
        return Ok(len_u64(bytes.len()));
    }
    let mut buffer = vec![0_u8; usize::try_from(chunk_size).unwrap_or(usize::MAX).max(1)];
    let mut total = 0_u64;
    loop {
        let read = input
            .read(&mut buffer)
            .await
            .map_err(|error| error.to_string())?;
        if read == 0 {
            output.flush().await.map_err(|error| error.to_string())?;
            return Ok(total);
        }
        output
            .write_all(&buffer[..read])
            .await
            .map_err(|error| error.to_string())?;
        total += len_u64(read);
    }
}

/// Streams a copy, yielding one [`TransferChunkData`] per written chunk.
///
/// An empty source yields exactly one terminal chunk (`chunk_size = 0`,
/// `is_last_chunk = true`) — Python yields nothing at all here, which is a bug this port
/// deliberately fixes.
fn copy_chunks(
    source: PathBuf,
    destination: PathBuf,
    chunk_size: u64,
    preserve_permissions: bool,
    execution: &'static str,
) -> OpStream<TransferChunkData> {
    Box::pin(async_stream::stream! {
        let source_path = format_path(&source);
        let target_path = format_path(&destination);
        let metadata = source_metadata(&source, preserve_permissions).await;
        let mut input = match tokio::fs::File::open(&source).await {
            Ok(file) => file,
            Err(error) => {
                yield fs_failure(execution, &error.to_string());
                return;
            }
        };
        let mut output = match tokio::fs::File::create(&destination).await {
            Ok(file) => file,
            Err(error) => {
                yield fs_failure(execution, &error.to_string());
                return;
            }
        };
        let mut buffer = vec![0_u8; usize::try_from(chunk_size).unwrap_or(usize::MAX).max(1)];
        let mut current = match input.read(&mut buffer).await {
            Ok(read) => read,
            Err(error) => {
                yield fs_failure(execution, &error.to_string());
                return;
            }
        };
        if current == 0 {
            yield OpResult::success(TransferChunkData {
                source_path: source_path.clone(),
                target_path: target_path.clone(),
                chunk_size: 0,
                chunk_index: 0,
                is_last_chunk: true,
            });
        } else {
            let mut index = 0_u64;
            loop {
                if let Err(error) = output.write_all(&buffer[..current]).await {
                    yield fs_failure(execution, &error.to_string());
                    return;
                }
                let next = match input.read(&mut buffer).await {
                    Ok(read) => read,
                    Err(error) => {
                        yield fs_failure(execution, &error.to_string());
                        return;
                    }
                };
                let is_last = next == 0;
                yield OpResult::success(TransferChunkData {
                    source_path: source_path.clone(),
                    target_path: target_path.clone(),
                    chunk_size: len_u64(current),
                    chunk_index: index,
                    is_last_chunk: is_last,
                });
                if is_last {
                    break;
                }
                current = next;
                index += 1;
            }
        }
        if let Err(error) = output.flush().await {
            yield fs_failure(execution, &error.to_string());
            return;
        }
        drop(output);
        apply_copied_permissions(metadata.as_ref(), &destination);
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::BinaryPolicy;
    use tempfile::TempDir;

    /// A backend rooted at `dir` without sandbox restrictions.
    fn backend(dir: &TempDir) -> Arc<LocalFsOperation> {
        let config = LocalWorkConfig::default().with_workspace(format_path(dir.path()));
        Arc::new(LocalFsOperation::new(config).unwrap())
    }

    /// A backend whose paths must stay inside `dir`.
    fn restricted(dir: &TempDir) -> Arc<LocalFsOperation> {
        let config = LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_sandbox_roots([format_path(dir.path())]);
        Arc::new(LocalFsOperation::new(config).unwrap())
    }

    /// Canonical path of `dir/name`, matching what [`PathPolicy`] resolves to.
    async fn canonical(dir: &TempDir, name: &str) -> String {
        format_path(
            &tokio::fs::canonicalize(dir.path().join(name))
                .await
                .unwrap(),
        )
    }

    fn read_text(path: &str) -> ReadFileParams {
        ReadFileParams::text(path)
    }

    // ----- read_file ----------------------------------------------------------------

    #[tokio::test]
    async fn read_text_uses_python_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"one\ntwo\n").unwrap();
        let data = backend(&dir)
            .read_file(read_text("a.txt"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(data.content, FileContent::Text("one\ntwo\n".to_string()));
        assert_eq!(data.mode, ReadMode::Text);
        assert!(
            data.encoding.is_none(),
            "an explicit encoding is not reported"
        );
        assert!(!data.truncated);
        assert_eq!(data.total_bytes, 8);
        assert!(data.path.ends_with("a.txt"));
    }

    #[tokio::test]
    async fn read_auto_reports_the_detected_encoding() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        let op = backend(&dir);
        let auto = op
            .read_file(read_text("a.txt").with_encoding("auto"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(auto.encoding.as_deref(), Some("utf-8"));
        assert_eq!(auto.content, FileContent::Text("hello".to_string()));

        let explicit = op
            .read_file(read_text("a.txt").with_encoding("utf-8"))
            .await
            .into_result()
            .unwrap();
        assert!(explicit.encoding.is_none());
    }

    #[tokio::test]
    async fn read_bytes_mode_truncates_and_keeps_the_full_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), b"0123456789").unwrap();
        let mut params = ReadFileParams::bytes("a.bin");
        params.chunk_size = 4;
        let data = backend(&dir).read_file(params).await.into_result().unwrap();
        assert_eq!(data.content, FileContent::Bytes(b"0123".to_vec()));
        assert!(data.truncated);
        assert_eq!(data.total_bytes, 10);
    }

    #[tokio::test]
    async fn read_applies_head_tail_and_line_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lines.txt");
        std::fs::write(&path, b"l1\nl2\nl3\nl4\n").unwrap();
        let op = backend(&dir);

        let head = op
            .read_file(read_text("lines.txt").with_head(2))
            .await
            .into_result()
            .unwrap();
        assert_eq!(head.content, FileContent::Text("l1\nl2\n".to_string()));

        let tail = op
            .read_file(read_text("lines.txt").with_tail(2))
            .await
            .into_result()
            .unwrap();
        assert_eq!(tail.content, FileContent::Text("l3\nl4\n".to_string()));

        let range = op
            .read_file(read_text("lines.txt").with_line_range(2, 3))
            .await
            .into_result()
            .unwrap();
        assert_eq!(range.content, FileContent::Text("l2\nl3\n".to_string()));

        let clamped = op
            .read_file(read_text("lines.txt").with_line_range(3, 99))
            .await
            .into_result()
            .unwrap();
        assert_eq!(clamped.content, FileContent::Text("l3\nl4\n".to_string()));

        let empty = op
            .read_file(read_text("lines.txt").with_line_range(0, 2))
            .await
            .into_result()
            .unwrap();
        assert_eq!(empty.content, FileContent::Text(String::new()));
    }

    #[tokio::test]
    async fn read_rejects_line_selections_in_bytes_mode() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), b"xyz").unwrap();
        let mut params = ReadFileParams::bytes("a.bin");
        params.head = Some(1);
        let failure = backend(&dir).read_file(params).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure.message.contains("only supported in text mode"),
            "{}",
            failure.message
        );
        assert!(failure.message.contains("execution: read_file"));
    }

    #[tokio::test]
    async fn read_rejects_two_selections() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        let mut params = read_text("a.txt");
        params.head = Some(1);
        params.tail = Some(1);
        let failure = backend(&dir).read_file(params).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure
                .message
                .contains("cannot be specified simultaneously"),
            "{}",
            failure.message
        );
    }

    #[tokio::test]
    async fn read_rejects_missing_files_and_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let op = backend(&dir);

        let missing = op.read_file(read_text("missing.txt")).await;
        assert_eq!(missing.code, 199_003);
        assert!(
            missing.message.contains("File does not exist"),
            "{}",
            missing.message
        );

        let directory = op.read_file(read_text("sub")).await;
        assert_eq!(directory.code, 199_003);
        assert!(
            directory.message.contains("Is a directory"),
            "{}",
            directory.message
        );
    }

    #[tokio::test]
    async fn read_enforces_the_configured_max_file_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"0123456789").unwrap();
        let config = LocalWorkConfig {
            max_file_size_bytes: Some(4),
            ..LocalWorkConfig::default().with_workspace(format_path(dir.path()))
        };
        let op = Arc::new(LocalFsOperation::new(config).unwrap());
        let failure = op.read_file(read_text("a.txt")).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure
                .message
                .contains("exceeds the configured limit 4 bytes"),
            "{}",
            failure.message
        );
    }

    #[tokio::test]
    async fn read_rejects_binary_content_under_the_reject_policy() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), [0_u8, 1, 2, 3]).unwrap();
        let mut params = read_text("a.bin");
        params.binary = BinaryPolicy::Reject;
        let failure = backend(&dir).read_file(params).await;
        assert_eq!(failure.code, 199_003);
        assert!(failure.message.contains("binary"), "{}", failure.message);
    }

    #[tokio::test]
    async fn restricted_backend_denies_traversal_for_read_and_write() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("escape.txt"), b"secret").unwrap();
        let op = restricted(&root);
        let escaped = format_path(&outside.path().join("escape.txt"));

        let read = op.read_file(read_text(&escaped)).await;
        assert_eq!(read.code, 199_003);
        assert!(read.message.contains("Access denied"), "{}", read.message);

        let write = op.read_file(read_text("../escape.txt")).await;
        assert_eq!(write.code, 199_003);
        assert!(write.message.contains("Access denied"), "{}", write.message);

        let refused = op
            .write_file(WriteFileParams::raw("../escape.txt", "x"))
            .await;
        assert_eq!(refused.code, 199_003);
        assert!(
            refused.message.contains("Access denied"),
            "{}",
            refused.message
        );
        assert!(!root.path().join("../escape.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restricted_backend_denies_symlink_escapes() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();

        let failure = restricted(&root)
            .read_file(read_text("link/secret.txt"))
            .await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure.message.contains("Access denied"),
            "{}",
            failure.message
        );
    }

    // ----- write_file ---------------------------------------------------------------

    fn read_back(dir: &TempDir, name: &str) -> Vec<u8> {
        std::fs::read(dir.path().join(name)).unwrap()
    }

    #[tokio::test]
    async fn write_prepends_a_newline_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let data = backend(&dir)
            .write_file(WriteFileParams::new("a.txt", "hi"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(read_back(&dir, "a.txt"), b"\nhi".to_vec());
        assert_eq!(data.size, 3);
        assert_eq!(data.mode, ReadMode::Text);
        assert!(data.path.ends_with("a.txt"));
    }

    #[tokio::test]
    async fn write_raw_writes_exact_content() {
        let dir = tempfile::tempdir().unwrap();
        backend(&dir)
            .write_file(WriteFileParams::raw("a.txt", "hi"))
            .await
            .into_result()
            .unwrap();
        assert_eq!(read_back(&dir, "a.txt"), b"hi".to_vec());
    }

    #[tokio::test]
    async fn write_honors_append_newline_and_append_mode() {
        let dir = tempfile::tempdir().unwrap();
        let op = backend(&dir);
        let mut params = WriteFileParams::raw("a.txt", "hi");
        params.append_newline = true;
        op.write_file(params).await.into_result().unwrap();
        assert_eq!(read_back(&dir, "a.txt"), b"hi\n".to_vec());

        // `append` adds no newline of its own and creates a missing file.
        op.write_file(WriteFileParams::raw("a.txt", "more").with_append())
            .await
            .into_result()
            .unwrap();
        assert_eq!(read_back(&dir, "a.txt"), b"hi\nmore".to_vec());

        op.write_file(WriteFileParams::raw("b.txt", "fresh").with_append())
            .await
            .into_result()
            .unwrap();
        assert_eq!(read_back(&dir, "b.txt"), b"fresh".to_vec());
    }

    #[tokio::test]
    async fn write_rejects_missing_files_directories_and_bad_flags() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let op = backend(&dir);

        let mut missing = WriteFileParams::raw("missing.txt", "x");
        missing.create_if_not_exist = false;
        let failure = op.write_file(missing).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure.message.contains("File does not exist"),
            "{}",
            failure.message
        );
        assert!(!dir.path().join("missing.txt").exists());

        let directory = op.write_file(WriteFileParams::raw("sub", "x")).await;
        assert_eq!(directory.code, 199_003);
        assert!(
            directory.message.contains("is a directory"),
            "{}",
            directory.message
        );

        let mut incompatible = WriteFileParams::raw("c.txt", "x");
        incompatible.create_if_absent = true;
        incompatible.append = true;
        let failure = op.write_file(incompatible).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure
                .message
                .contains("conditional write options are incompatible"),
            "{}",
            failure.message
        );
    }

    #[tokio::test]
    async fn write_rejects_a_stale_digest_without_touching_the_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"abc").unwrap();
        let op = backend(&dir);

        let stale = WriteFileParams::raw("a.txt", "new")
            .with_expected_sha256(sha256_hex(b"something else"));
        let failure = op.write_file(stale).await;
        assert_eq!(failure.code, 199_011);
        assert_eq!(failure.message, "file version conflict");
        assert_eq!(read_back(&dir, "a.txt"), b"abc".to_vec());

        let fresh = WriteFileParams::raw("a.txt", "new").with_expected_sha256(sha256_hex(b"abc"));
        op.write_file(fresh).await.into_result().unwrap();
        assert_eq!(read_back(&dir, "a.txt"), b"new".to_vec());
    }

    #[tokio::test]
    async fn write_create_if_absent_rejects_existing_files_including_empty_ones() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        let op = backend(&dir);

        let mut params = WriteFileParams::raw("empty.txt", "x");
        params.create_if_absent = true;
        let failure = op.write_file(params).await;
        assert_eq!(failure.code, 199_011);
        assert_eq!(failure.message, "file version conflict");
        assert_eq!(read_back(&dir, "empty.txt"), Vec::<u8>::new());

        let mut created = WriteFileParams::raw("fresh.txt", "x");
        created.create_if_absent = true;
        op.write_file(created).await.into_result().unwrap();
        assert_eq!(read_back(&dir, "fresh.txt"), b"x".to_vec());
    }

    // ----- transfers ----------------------------------------------------------------

    /// A directory outside the workspace holding a host-side source file.
    fn host_source(content: &[u8]) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("host.txt");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[tokio::test]
    async fn upload_creates_parents_and_reports_both_paths() {
        let dir = tempfile::tempdir().unwrap();
        let (_host, source) = host_source(b"hello");
        let params = UploadFileParams {
            local_path: format_path(&source),
            target_path: "nested/deep/out.txt".to_string(),
            ..UploadFileParams::default()
        };
        let data = backend(&dir)
            .upload_file(params)
            .await
            .into_result()
            .unwrap();
        assert_eq!(read_back(&dir, "nested/deep/out.txt"), b"hello".to_vec());
        assert_eq!(data.size, 5);
        assert_eq!(data.local_path, format_path(&source));
        assert_eq!(
            data.target_path,
            canonical(&dir, "nested/deep/out.txt").await
        );
    }

    #[tokio::test]
    async fn upload_refuses_existing_targets_unless_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let (_host, source) = host_source(b"new");
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("nested/out.txt"), b"old").unwrap();
        let op = backend(&dir);

        let params = UploadFileParams {
            local_path: format_path(&source),
            target_path: "nested/out.txt".to_string(),
            ..UploadFileParams::default()
        };
        let failure = op.upload_file(params).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure.message.contains("Target exists"),
            "{}",
            failure.message
        );
        assert_eq!(read_back(&dir, "nested/out.txt"), b"old".to_vec());

        let params = UploadFileParams {
            local_path: format_path(&source),
            target_path: "nested/out.txt".to_string(),
            overwrite: true,
            ..UploadFileParams::default()
        };
        op.upload_file(params).await.into_result().unwrap();
        assert_eq!(read_back(&dir, "nested/out.txt"), b"new".to_vec());
    }

    #[tokio::test]
    async fn download_mirrors_upload() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"payload").unwrap();
        let host = tempfile::tempdir().unwrap();
        let destination = host.path().join("nested/got.txt");
        let params = DownloadFileParams {
            source_path: "a.txt".to_string(),
            local_path: format_path(&destination),
            ..DownloadFileParams::default()
        };
        let data = backend(&dir)
            .download_file(params)
            .await
            .into_result()
            .unwrap();
        assert_eq!(data.size, 7);
        assert_eq!(data.source_path, canonical(&dir, "a.txt").await);
        assert_eq!(
            data.local_path,
            format_path(&tokio::fs::canonicalize(&destination).await.unwrap())
        );
        assert_eq!(std::fs::read(&destination).unwrap(), b"payload".to_vec());

        let params = DownloadFileParams {
            source_path: "a.txt".to_string(),
            local_path: format_path(&destination),
            ..DownloadFileParams::default()
        };
        let failure = backend(&dir).download_file(params).await;
        assert_eq!(failure.code, 199_003);
        assert!(
            failure.message.contains("Target exists"),
            "{}",
            failure.message
        );
    }

    #[tokio::test]
    async fn transfers_reject_blank_host_paths_and_missing_sources() {
        let dir = tempfile::tempdir().unwrap();
        let op = backend(&dir);

        let blank_upload = op
            .upload_file(UploadFileParams {
                local_path: "  ".to_string(),
                target_path: "a.txt".to_string(),
                ..UploadFileParams::default()
            })
            .await;
        assert_eq!(blank_upload.code, 199_003);
        assert!(
            blank_upload
                .message
                .contains("source path must not be empty"),
            "{}",
            blank_upload.message
        );

        let blank_download = op
            .download_file(DownloadFileParams {
                source_path: "a.txt".to_string(),
                local_path: String::new(),
                ..DownloadFileParams::default()
            })
            .await;
        assert_eq!(blank_download.code, 199_003);
        assert!(
            blank_download
                .message
                .contains("target path must not be empty"),
            "{}",
            blank_download.message
        );

        let missing_source = op
            .upload_file(UploadFileParams {
                local_path: format_path(&dir.path().join("absent.txt")),
                target_path: "a.txt".to_string(),
                ..UploadFileParams::default()
            })
            .await;
        assert_eq!(missing_source.code, 199_003);
        assert!(
            missing_source.message.contains("Source not found"),
            "{}",
            missing_source.message
        );

        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let missing_download = op
            .download_file(DownloadFileParams {
                source_path: "sub".to_string(),
                local_path: format_path(&dir.path().join("out.txt")),
                ..DownloadFileParams::default()
            })
            .await;
        assert_eq!(missing_download.code, 199_003);
        assert!(
            missing_download.message.contains("Source not found"),
            "{}",
            missing_download.message
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upload_preserves_source_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let (_host, source) = host_source(b"secret");
        std::fs::set_permissions(&source, std::fs::Permissions::from_mode(0o600)).unwrap();
        backend(&dir)
            .upload_file(UploadFileParams {
                local_path: format_path(&source),
                target_path: "out.txt".to_string(),
                ..UploadFileParams::default()
            })
            .await
            .into_result()
            .unwrap();
        let mode = std::fs::metadata(dir.path().join("out.txt"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    // ----- streams ------------------------------------------------------------------

    /// Drains a read stream into its items.
    async fn read_stream(
        op: &Arc<LocalFsOperation>,
        params: ReadFileParams,
    ) -> Vec<OpResult<ReadChunkData>> {
        op.clone().read_file_stream(params).collect().await
    }

    #[tokio::test]
    async fn text_stream_emits_one_chunk_per_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"a\nbb\nccc\n").unwrap();
        let items = read_stream(&backend(&dir), read_text("a.txt")).await;
        assert_eq!(items.len(), 3);
        let sizes: Vec<u64> = items
            .iter()
            .map(|item| item.data.as_ref().unwrap().chunk_size)
            .collect();
        assert_eq!(sizes, vec![2, 3, 4]);
        let indices: Vec<u64> = items
            .iter()
            .map(|item| item.data.as_ref().unwrap().chunk_index)
            .collect();
        assert_eq!(indices, vec![0, 1, 2]);
        let flags: Vec<bool> = items
            .iter()
            .map(|item| item.data.as_ref().unwrap().is_last_chunk)
            .collect();
        assert_eq!(flags, vec![false, false, true]);
        assert_eq!(
            items[1].data.as_ref().unwrap().chunk,
            FileContent::Text("bb\n".to_string())
        );
    }

    #[tokio::test]
    async fn text_stream_emits_nothing_for_an_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        let items = read_stream(&backend(&dir), read_text("empty.txt")).await;
        assert!(items.is_empty(), "an empty file yields no chunks");
    }

    #[tokio::test]
    async fn tail_stream_without_lines_yields_one_terminal_chunk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        let items = read_stream(&backend(&dir), read_text("empty.txt").with_tail(5)).await;
        assert_eq!(items.len(), 1, "Python emits one empty terminal chunk");
        let data = items[0].data.as_ref().unwrap();
        assert_eq!(data.chunk, FileContent::Text(String::new()));
        assert_eq!(data.chunk_size, 0);
        assert_eq!(data.chunk_index, 0);
        assert!(data.is_last_chunk);
    }

    #[tokio::test]
    async fn bytes_stream_chunks_by_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), b"0123456789").unwrap();
        let mut params = ReadFileParams::bytes("a.bin");
        params.chunk_size = 4;
        let items = read_stream(&backend(&dir), params).await;
        assert_eq!(items.len(), 3);
        let sizes: Vec<u64> = items
            .iter()
            .map(|item| item.data.as_ref().unwrap().chunk_size)
            .collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        assert!(items[2].data.as_ref().unwrap().is_last_chunk);
        assert_eq!(
            items[0].data.as_ref().unwrap().chunk,
            FileContent::Bytes(b"0123".to_vec())
        );
    }

    #[tokio::test]
    async fn read_stream_failure_is_a_single_item() {
        let dir = tempfile::tempdir().unwrap();
        let items = read_stream(&backend(&dir), read_text("missing.txt")).await;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].code, 199_003);
        assert!(
            items[0].message.contains("File does not exist"),
            "{}",
            items[0].message
        );
    }

    #[tokio::test]
    async fn upload_stream_emits_a_terminal_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let (_host, source) = host_source(b"hello");
        let items: Vec<OpResult<TransferChunkData>> = backend(&dir)
            .upload_file_stream(UploadFileParams {
                local_path: format_path(&source),
                target_path: "out.txt".to_string(),
                ..UploadFileParams::default()
            })
            .collect()
            .await;
        assert_eq!(items.len(), 1, "a 5-byte file fits in the default chunk");
        let data = items[0].data.as_ref().unwrap();
        assert_eq!(data.chunk_size, 5);
        assert_eq!(data.chunk_index, 0);
        assert!(data.is_last_chunk);
        assert_eq!(read_back(&dir, "out.txt"), b"hello".to_vec());
    }

    #[tokio::test]
    async fn upload_stream_reports_each_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let (_host, source) = host_source(b"0123456789");
        let items: Vec<OpResult<TransferChunkData>> = backend(&dir)
            .upload_file_stream(UploadFileParams {
                local_path: format_path(&source),
                target_path: "out.txt".to_string(),
                chunk_size: 4,
                ..UploadFileParams::default()
            })
            .collect()
            .await;
        let sizes: Vec<u64> = items
            .iter()
            .map(|item| item.data.as_ref().unwrap().chunk_size)
            .collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        assert!(items[2].data.as_ref().unwrap().is_last_chunk);
        assert!(!items[0].data.as_ref().unwrap().is_last_chunk);
        assert_eq!(read_back(&dir, "out.txt"), b"0123456789".to_vec());
    }

    #[tokio::test]
    async fn transfer_stream_of_an_empty_source_yields_one_chunk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        let host = tempfile::tempdir().unwrap();
        let items: Vec<OpResult<TransferChunkData>> = backend(&dir)
            .download_file_stream(DownloadFileParams {
                source_path: "empty.txt".to_string(),
                local_path: format_path(&host.path().join("got.txt")),
                ..DownloadFileParams::default()
            })
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        let data = items[0].data.as_ref().unwrap();
        assert_eq!(data.chunk_size, 0);
        assert_eq!(data.chunk_index, 0);
        assert!(data.is_last_chunk);
    }

    #[tokio::test]
    async fn transfer_stream_failure_is_a_single_item() {
        let dir = tempfile::tempdir().unwrap();
        let items: Vec<OpResult<TransferChunkData>> = backend(&dir)
            .upload_file_stream(UploadFileParams {
                local_path: format_path(&dir.path().join("absent.txt")),
                target_path: "out.txt".to_string(),
                ..UploadFileParams::default()
            })
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].code, 199_003);
        assert!(
            items[0].message.contains("Source not found"),
            "{}",
            items[0].message
        );
    }

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
        let config = LocalWorkConfig::default().with_workspace(format_path(dir.path()));
        let backend = crate::fs::local_fs(config).unwrap();
        assert!(backend.supports_conditional_write());
    }
}
