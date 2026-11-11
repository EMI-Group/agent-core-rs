//! Writes for the local backend: plain, append, and conditional compare-and-swap.
//!
//! This module is the write half of Python's `local/fs_operation.py`. The facade in
//! [`crate::fs::local`] resolves the target path through the
//! [`PathPolicy`](crate::workspace::PathPolicy) and holds the exclusive per-path lock;
//! the functions here validate the flag combination, run the conditional checks, and
//! compose the content bytes.
//!
//! # Order of operations
//!
//! Python's order is preserved and matters for callers: the flag combination is rejected
//! first (no filesystem access), then the target is resolved and its kind checked, then the
//! conditional checks run *before* the file is opened for writing, so a rejected
//! compare-and-swap leaves the target byte-identical.
//!
//! # Failures
//!
//! Failures are Python-compatible [`OpResult`] values: code `199003` for an I/O or
//! parameter failure and code `199011` (`file version conflict`) for a rejected
//! conditional write.

use std::io::ErrorKind;
use std::path::Path;

use tokio::io::AsyncWriteExt;

use crate::fs::local::{len_u64, LocalFsOperation};
use crate::fs::slice;
use crate::fs::types::{WriteFileData, WriteFileParams};
use crate::result::{fs_failure, OpResult};
use crate::status::{file_version_conflict_message, SysOpStatus};
use crate::util::{format_path, sha256_hex};

/// `execution` slot of a `write_file` failure.
const WRITE_FILE: &str = "write_file";

/// Writes `params` to its target (Python `write_file`).
pub(crate) async fn write_file(
    op: &LocalFsOperation,
    params: WriteFileParams,
) -> OpResult<WriteFileData> {
    if let Err(reason) = validate_write(&params) {
        return fs_failure(WRITE_FILE, &reason);
    }
    let path = match op.policy().resolve_for_write(&params.path).await {
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
    let guard = op.write_guard(&path).await;
    let result = write_locked(&path, &params).await;
    drop(guard);
    result
}

/// Builds Python's `file version conflict` failure (code `199011`).
fn version_conflict<T>() -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::FileVersionConflict,
        file_version_conflict_message(),
    )
}

/// Rejects the write flag combinations Python refuses before touching the filesystem.
///
/// # Errors
///
/// Returns the Python reason when two conditional options are combined incompatibly.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::fs::local::test_util::{backend, read_back};
    use crate::fs::types::ReadMode;
    use crate::operation::FsOperation;

    // ----- write_file ---------------------------------------------------------------

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
}
