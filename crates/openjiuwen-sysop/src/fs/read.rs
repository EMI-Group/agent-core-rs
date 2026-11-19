//! Whole-file and streaming reads for the local backend.
//!
//! This module is the read half of Python's `local/fs_operation.py`; the facade in
//! [`crate::fs::local`] owns the [`PathPolicy`](crate::workspace::PathPolicy) and the
//! per-path locks, and the functions here resolve, validate, decode, and slice one file.
//! Python's order is preserved: the line selection is validated before the path is
//! resolved, so a bad request never touches the filesystem.
//!
//! # Read contract
//!
//! Two read paths exist, and the difference is the memory they may use:
//!
//! - [`read_file`] / [`read_file_stream`] load the **whole** file. They therefore refuse a
//!   file above the configured ceiling
//!   ([`PathPolicy::check_read_size`](crate::workspace::PathPolicy::check_read_size),
//!   `LocalWorkConfig::max_file_size_bytes`, 64 MiB by default) before reading a byte, and
//!   report a reason naming the file size and the limit. Python has no such ceiling; setting
//!   `max_file_size_bytes` to `None` restores its unlimited behavior.
//! - [`read_bytes_limited`] reads at most the requested window from a requested offset by
//!   seeking and taking, so a caller that wants the first lines of a huge file never
//!   allocates the file. It reports the real total length and whether bytes remained.
//!
//! # Failures
//!
//! Failures are Python-compatible [`OpResult`] values with code `199003`, carrying the
//! Python method name (`read_file` or `read_file_stream`) in the `execution` slot. The
//! bounded read returns the typed [`openjiuwen_common::error::Result`] instead, because it
//! is a Rust addition: a path or I/O failure is
//! [`ErrorKind::SysOp`](openjiuwen_common::error::ErrorKind::SysOp) and a rejected window is
//! [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation).

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::constants::{AUTO_ENCODING, DEFAULT_READ_STREAM_CHUNK_SIZE};
use crate::fs::local::{len_u64, LocalFsOperation};
use crate::fs::slice;
use crate::fs::types::{
    FileContent, ReadBytesData, ReadBytesParams, ReadChunkData, ReadFileData, ReadFileParams,
    ReadMode,
};
use crate::operation::OpStream;
use crate::result::{fs_failure, OpResult};
use crate::util::format_path;

/// `execution` slot of a `read_file` failure.
const READ_FILE: &str = "read_file";
/// `execution` slot of a `read_file_stream` failure.
const READ_FILE_STREAM: &str = "read_file_stream";
/// Bytes reserved up front for a bounded read window (the buffer grows only as data arrives).
const WINDOW_RESERVE_BYTES: u64 = 8 * 1024;

/// Reads a whole file in text or bytes mode (Python `read_file`).
pub(crate) async fn read_file(
    op: &LocalFsOperation,
    params: ReadFileParams,
) -> OpResult<ReadFileData> {
    let selection = match slice::validate_read_params(&params) {
        Ok(selection) => selection,
        Err(reason) => return fs_failure(READ_FILE, &reason),
    };
    let loaded = match load_for_read(op, &params).await {
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

/// Streams a file, one chunk per line in text mode and fixed-size chunks in bytes mode.
pub(crate) fn read_file_stream(
    op: Arc<LocalFsOperation>,
    params: ReadFileParams,
) -> OpStream<ReadChunkData> {
    Box::pin(async_stream::stream! {
        let plan = match prepare_read_stream(&op, &params).await {
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
                    // The chunk size is the encoded length; a line the declared encoding
                    // cannot re-encode is reported instead of a fabricated size.
                    let chunk_size = match slice::try_encoded_len(&line, &encoding) {
                        Ok(size) => size,
                        Err(reason) => {
                            yield fs_failure(READ_FILE_STREAM, &reason);
                            return;
                        }
                    };
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

/// Prepares a streamed read: validation, resolution, decoding, and selection.
async fn prepare_read_stream(
    op: &LocalFsOperation,
    params: &ReadFileParams,
) -> Result<ReadStreamPlan, String> {
    let selection = slice::validate_read_params(params)?;
    let loaded = load_for_read(op, params).await?;
    let path = format_path(&loaded.path);
    match params.mode {
        ReadMode::Text => {
            let text = crate::encoding::decode_text(&loaded.bytes, &params.encoding, params.binary)
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

/// Applies every read precondition and returns the file contents.
///
/// The line-selection validation is *not* part of this step; callers run
/// [`slice::validate_read_params`] first so a bad request is rejected before the path is
/// even resolved (Python's order).
///
/// # Errors
///
/// Returns the failure reason when the path is denied, missing, not a regular file, over
/// the configured size limit (see [`PathPolicy::check_read_size`](crate::workspace::PathPolicy::check_read_size)),
/// or unreadable. The size check runs on the metadata, before the file is opened, so an
/// oversized file is never loaded.
async fn load_for_read(
    op: &LocalFsOperation,
    params: &ReadFileParams,
) -> Result<LoadedFile, String> {
    let path = op
        .policy()
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
    if let Err(error) = op.policy().check_read_size(metadata.len()) {
        return Err(error.message().to_string());
    }
    let bytes = read_bytes_locked(op, &path, params.only_read).await?;
    Ok(LoadedFile {
        path,
        size: metadata.len(),
        bytes,
    })
}

/// Reads at most `params.max_bytes` bytes from `params.offset` without loading the file.
///
/// The bounded counterpart of [`read_file`]: it resolves and inspects the path, then seeks
/// to the requested offset and reads no more than the window, so the memory used is bounded
/// by [`ReadBytesParams::max_bytes`] rather than by the file size. The whole-file ceiling is
/// not applied to the file (a large file is exactly what this read is for); it *is* applied
/// to the requested window, so a caller cannot ask this path for an allocation the size
/// ceiling would refuse.
///
/// # Errors
///
/// Returns [`Error::validation`] when the window is wider than the configured ceiling, when
/// the path is a directory, and [`Error::sys_op`] when the path is denied, missing, cannot
/// be locked, or cannot be read.
pub(crate) async fn read_bytes_limited(
    op: &LocalFsOperation,
    params: ReadBytesParams,
) -> Result<ReadBytesData> {
    if let Some(limit) = op.policy().max_file_size() {
        if params.max_bytes > limit {
            return Err(Error::validation(format!(
                "requested read window {} exceeds the configured limit {limit} bytes",
                params.max_bytes
            )));
        }
    }
    let path = op.policy().resolve_existing(&params.path).await?;
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|error| Error::sys_op(metadata_reason(&path, &error)))?;
    if !metadata.is_file() {
        return Err(Error::validation(if metadata.is_dir() {
            format!("Is a directory: {}", format_path(&path))
        } else {
            format!("not a file: {}", format_path(&path))
        }));
    }
    let total_bytes = metadata.len();
    let bytes = read_window(op, &path, &params).await?;
    let end = params.offset.saturating_add(len_u64(bytes.len()));
    Ok(ReadBytesData {
        path: format_path(&path),
        bytes,
        total_bytes,
        truncated: end < total_bytes,
    })
}

/// Seeks to `params.offset` and reads at most `params.max_bytes` bytes, holding the read lock.
///
/// The buffer is reserved only up to the window (capped at [`WINDOW_RESERVE_BYTES`]), so a
/// window of a huge file never sizes an allocation from the file's length.
async fn read_window(
    op: &LocalFsOperation,
    path: &Path,
    params: &ReadBytesParams,
) -> Result<Vec<u8>> {
    let guard = if params.only_read {
        None
    } else {
        Some(op.read_guard(path).await.map_err(Error::sys_op)?)
    };
    let result = read_window_locked(path, params.offset, params.max_bytes).await;
    drop(guard);
    result
}

/// The unlocked seek-and-read half of [`read_window`].
async fn read_window_locked(path: &Path, offset: u64, max_bytes: u64) -> Result<Vec<u8>> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| Error::sys_op(error.to_string()))?;
    if offset > 0 {
        file.seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(|error| Error::sys_op(error.to_string()))?;
    }
    let reserve = usize::try_from(max_bytes.min(WINDOW_RESERVE_BYTES)).unwrap_or(0);
    let mut bytes = Vec::with_capacity(reserve);
    // `take` bounds both the bytes read and the buffer's growth: the window, not the file.
    file.take(max_bytes)
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| Error::sys_op(error.to_string()))?;
    Ok(bytes)
}

/// Reads `path`, holding the shared lock unless `only_read` skips it.
///
/// `only_read` (Python `skip_lock`) takes neither the in-process lock nor the
/// cross-process `flock`; everything else goes through
/// [`LocalFsOperation::read_guard`](crate::fs::local::LocalFsOperation), which takes both.
///
/// # Errors
///
/// Returns the reason when the cross-process lock times out, or the I/O error message when
/// the file cannot be read.
async fn read_bytes_locked(
    op: &LocalFsOperation,
    path: &Path,
    only_read: bool,
) -> Result<Vec<u8>, String> {
    if only_read {
        return tokio::fs::read(path)
            .await
            .map_err(|error| error.to_string());
    }
    let guard = op.read_guard(path).await?;
    let result = tokio::fs::read(path).await;
    drop(guard);
    result.map_err(|error| error.to_string())
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::{BinaryPolicy, LocalWorkConfig};
    use crate::fs::local::test_util::{backend, restricted};
    use crate::fs::types::WriteFileParams;
    use crate::operation::FsOperation;
    use futures::StreamExt;

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
}
