//! Uploads and downloads for the local backend: host ↔ workspace transfers.
//!
//! This module is the transfer half of Python's `local/fs_operation.py`. The facade in
//! [`crate::fs::local`] owns the path policy and the per-path locks; the functions here
//! resolve both sides, validate them, and copy bytes (or stream them) under an ordered
//! pair of locks.
//!
//! # Host paths
//!
//! Python never sandbox-checks the *host* side of a transfer (the upload source and the
//! download destination), so neither does this port: those paths are only tilde-expanded
//! and canonicalized. The workspace side always goes through the
//! [`PathPolicy`](crate::workspace::PathPolicy).
//!
//! # Locking
//!
//! A transfer locks the source `Read` and the destination `Write` through
//! [`LocalFsOperation::acquire_ordered`], which sorts the paths so two transfers can never
//! deadlock and takes both lock layers (in-process `RwLock` and the cross-process `flock` in
//! [`crate::fs::lock`]) in that same order. The guards are held until the last byte is
//! flushed. A lock that cannot be taken (timeout or an unusable sidecar file) fails the
//! transfer with code `199003` instead of copying unlocked.
//!
//! # Failures
//!
//! Failures are Python-compatible [`OpResult`] values with code `199003`, carrying the
//! Python method name (`upload_file`, `upload_file_stream`, `download_file`,
//! `download_file_stream`) in the `execution` slot.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::constants::DEFAULT_TRANSFER_CHUNK_SIZE;
use crate::fs::local::{len_u64, LocalFsOperation, LockMode};
use crate::fs::types::{
    DownloadFileData, DownloadFileParams, TransferChunkData, UploadFileData, UploadFileParams,
};
use crate::operation::OpStream;
use crate::result::{fs_failure, OpResult};
use crate::util::format_path;
use crate::workspace::canonicalize_containing;

/// `execution` slot of an `upload_file` failure.
const UPLOAD_FILE: &str = "upload_file";
/// `execution` slot of an `upload_file_stream` failure.
const UPLOAD_FILE_STREAM: &str = "upload_file_stream";
/// `execution` slot of a `download_file` failure.
const DOWNLOAD_FILE: &str = "download_file";
/// `execution` slot of a `download_file_stream` failure.
const DOWNLOAD_FILE_STREAM: &str = "download_file_stream";

/// Copies one host file into the workspace (Python `upload_file`).
pub(crate) async fn upload_file(
    op: &LocalFsOperation,
    params: UploadFileParams,
) -> OpResult<UploadFileData> {
    let (source, destination) = match prepare_upload(op, &params).await {
        Ok(pair) => pair,
        Err(reason) => return fs_failure(UPLOAD_FILE, &reason),
    };
    let size = match transfer(
        op,
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

/// Streams one host file into the workspace, one chunk per read.
pub(crate) fn upload_file_stream(
    op: Arc<LocalFsOperation>,
    params: UploadFileParams,
) -> OpStream<TransferChunkData> {
    Box::pin(async_stream::stream! {
        let (source, destination) = match prepare_upload(&op, &params).await {
            Ok(pair) => pair,
            Err(reason) => {
                yield fs_failure(UPLOAD_FILE_STREAM, &reason);
                return;
            }
        };
        let chunk_size = transfer_chunk_size(params.chunk_size);
        let mut chunks = transfer_stream(
            Arc::clone(&op),
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

/// Copies one workspace file out to the host (Python `download_file`).
pub(crate) async fn download_file(
    op: &LocalFsOperation,
    params: DownloadFileParams,
) -> OpResult<DownloadFileData> {
    let (source, destination) = match prepare_download(op, &params).await {
        Ok(pair) => pair,
        Err(reason) => return fs_failure(DOWNLOAD_FILE, &reason),
    };
    let size = match transfer(
        op,
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

/// Streams one workspace file out to the host, one chunk per read.
pub(crate) fn download_file_stream(
    op: Arc<LocalFsOperation>,
    params: DownloadFileParams,
) -> OpStream<TransferChunkData> {
    Box::pin(async_stream::stream! {
        let (source, destination) = match prepare_download(&op, &params).await {
            Ok(pair) => pair,
            Err(reason) => {
                yield fs_failure(DOWNLOAD_FILE_STREAM, &reason);
                return;
            }
        };
        let chunk_size = transfer_chunk_size(params.chunk_size);
        let mut chunks = transfer_stream(
            Arc::clone(&op),
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

/// Resolves and validates an upload (host → workspace) transfer.
///
/// # Errors
///
/// Returns the Python reason for a blank source, a denied target, a missing source file, or
/// a destination that exists without `overwrite`.
async fn prepare_upload(
    op: &LocalFsOperation,
    params: &UploadFileParams,
) -> Result<(PathBuf, PathBuf), String> {
    if params.local_path.trim().is_empty() {
        return Err("source path must not be empty".to_string());
    }
    let source = canonicalize_containing(&expand_host_path(&params.local_path)).await;
    let destination = op
        .policy()
        .resolve(&params.target_path, params.create_parent_dirs)
        .await
        .map_err(|error| error.message().to_string())?;
    ensure_source_file(&source).await?;
    ensure_target_free(&destination, params.overwrite).await?;
    Ok((source, destination))
}

/// Resolves and validates a download (workspace → host) transfer.
///
/// # Errors
///
/// Returns the Python reason for a blank target, a denied source, a missing source file, or
/// a destination that exists without `overwrite`.
async fn prepare_download(
    op: &LocalFsOperation,
    params: &DownloadFileParams,
) -> Result<(PathBuf, PathBuf), String> {
    if params.local_path.trim().is_empty() {
        return Err("target path must not be empty".to_string());
    }
    let source = op
        .policy()
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
///
/// # Errors
///
/// Returns the reason when a cross-process lock times out or its sidecar file cannot be
/// opened, or the I/O error message when the copy fails.
async fn transfer(
    op: &LocalFsOperation,
    source: &Path,
    destination: &Path,
    chunk_size: u64,
    preserve_permissions: bool,
) -> Result<u64, String> {
    let guards = op
        .acquire_ordered(vec![
            (source.to_path_buf(), LockMode::Read),
            (destination.to_path_buf(), LockMode::Write),
        ])
        .await?;
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
    op: Arc<LocalFsOperation>,
    source: PathBuf,
    destination: PathBuf,
    chunk_size: u64,
    preserve_permissions: bool,
    execution: &'static str,
) -> OpStream<TransferChunkData> {
    Box::pin(async_stream::stream! {
        let guards = match op
            .acquire_ordered(vec![
                (source.clone(), LockMode::Read),
                (destination.clone(), LockMode::Write),
            ])
            .await
        {
            Ok(guards) => guards,
            Err(reason) => {
                yield fs_failure(execution, &reason);
                return;
            }
        };
        let mut chunks = copy_chunks(source, destination, chunk_size, preserve_permissions, execution);
        while let Some(chunk) = chunks.next().await {
            yield chunk;
        }
        drop(guards);
    })
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
///
/// # Errors
///
/// Returns `Source not found: <path>` when the source is missing or not a regular file.
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
///
/// # Errors
///
/// Returns `Target exists: <path>` when the destination exists without `overwrite`.
async fn ensure_target_free(destination: &Path, overwrite: bool) -> Result<(), String> {
    if !overwrite && tokio::fs::metadata(destination).await.is_ok() {
        return Err(format!("Target exists: {}", format_path(destination)));
    }
    Ok(())
}

/// Creates the missing parent directories of `destination`.
///
/// # Errors
///
/// Returns the reason when the directories cannot be created.
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
///
/// # Errors
///
/// Returns the I/O error message when reading or writing fails.
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
    use crate::fs::local::test_util::{backend, canonical, read_back};
    use crate::operation::FsOperation;
    use tempfile::TempDir;

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
}
