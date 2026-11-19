//! Bounded reads and the whole-file read ceiling.
//!
//! Two contracts are guarded here:
//!
//! 1. `read_file` / `read_file_stream` load the whole file, so a file above
//!    `LocalWorkConfig::max_file_size_bytes` is refused from its metadata — before a byte is
//!    allocated — with a typed `Validation` reason naming the size and the limit. The default
//!    configuration carries a ceiling; `unrestricted` (and `with_max_file_size(None)`) opt out.
//! 2. `FsOperation::read_bytes_limited` reads a caller-sized window from a caller-chosen
//!    offset without loading the file, reporting the real total length and whether bytes
//!    remained. This is the path a caller takes instead of reading a huge file whole.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::constants::DEFAULT_MAX_FILE_SIZE_BYTES;
use openjiuwen_sysop::fs::local::LocalFsOperation;
use openjiuwen_sysop::fs::types::{FileContent, ReadBytesParams, ReadFileParams};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::util::format_path;

/// A fresh temporary workspace.
fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// A backend for `config` (the local backend is `Send + Sync`, so it is shared by `Arc`).
fn backend(config: LocalWorkConfig) -> Arc<dyn FsOperation> {
    openjiuwen_sysop::fs::local_fs(config).expect("the configuration is valid")
}

/// The default configuration rooted at `dir` (its 64 MiB whole-file ceiling applies).
fn default_backend(dir: &tempfile::TempDir) -> Arc<dyn FsOperation> {
    backend(LocalWorkConfig::default().with_workspace(format_path(dir.path())))
}

#[tokio::test]
async fn the_default_configuration_carries_a_whole_file_ceiling() {
    assert_eq!(
        LocalWorkConfig::default().max_file_size_bytes,
        Some(DEFAULT_MAX_FILE_SIZE_BYTES),
        "a whole-file read must be bounded unless the caller opts out"
    );
    assert_eq!(
        LocalWorkConfig::unrestricted("/tmp").max_file_size_bytes,
        None,
        "unrestricted keeps Python's unlimited whole-file read"
    );
}

#[tokio::test]
async fn a_whole_file_read_refuses_a_file_above_the_default_ceiling() {
    let dir = workspace();
    // A sparse file one byte above the ceiling: the metadata check must reject it without
    // reading (or allocating) the 64 MiB body.
    let file = std::fs::File::create(dir.path().join("huge.log")).unwrap();
    file.set_len(DEFAULT_MAX_FILE_SIZE_BYTES + 1).unwrap();
    drop(file);

    let failure = default_backend(&dir)
        .read_file(ReadFileParams::text("huge.log"))
        .await;
    assert_eq!(failure.code, 199_003, "{}", failure.message);
    let limit = DEFAULT_MAX_FILE_SIZE_BYTES;
    assert!(
        failure.message.contains(&format!(
            "file size {} exceeds the configured limit {limit} bytes",
            limit + 1
        )),
        "the rejection must name the size and the limit, got: {}",
        failure.message
    );
    assert!(
        failure.message.contains("execution: read_file"),
        "{}",
        failure.message
    );
}

#[tokio::test]
async fn a_whole_file_stream_also_refuses_an_oversized_file() {
    let dir = workspace();
    std::fs::write(dir.path().join("big.txt"), b"0123456789").unwrap();
    let config = LocalWorkConfig::default()
        .with_workspace(format_path(dir.path()))
        .with_max_file_size(Some(4));
    let op = backend(config);
    let items: Vec<_> = {
        use futures::StreamExt;
        op.clone()
            .read_file_stream(ReadFileParams::text("big.txt"))
            .collect()
            .await
    };
    assert_eq!(items.len(), 1, "a failed stream yields one item");
    assert_eq!(items[0].code, 199_003);
    assert!(
        items[0]
            .message
            .contains("exceeds the configured limit 4 bytes"),
        "{}",
        items[0].message
    );
}

#[tokio::test]
async fn the_size_gate_is_a_typed_validation_error() {
    let dir = workspace();
    let config = LocalWorkConfig::default()
        .with_workspace(format_path(dir.path()))
        .with_max_file_size(Some(4));
    let op = backend(config);
    let error = op
        .read_bytes_limited(ReadBytesParams::new("x", 0, 0))
        .await
        .err();
    // `x` does not exist, so the path failure is a typed sys-op error: the gate is not a
    // panic and not a silent empty success.
    assert_eq!(
        error.expect("a missing file is an error").kind(),
        ErrorKind::SysOp
    );

    std::fs::write(dir.path().join("a.txt"), b"0123456789").unwrap();
    let config = LocalWorkConfig::default()
        .with_workspace(format_path(dir.path()))
        .with_max_file_size(Some(4));
    let local = LocalFsOperation::new(config).unwrap();
    let typed = local.policy().check_read_size(10).unwrap_err();
    assert_eq!(typed.kind(), ErrorKind::Validation);
    assert_eq!(
        typed.message(),
        "file size 10 exceeds the configured limit 4 bytes"
    );
}

#[tokio::test]
async fn clearing_the_ceiling_restores_the_unlimited_read() {
    let dir = workspace();
    std::fs::write(dir.path().join("a.txt"), b"0123456789").unwrap();
    let bounded = backend(
        LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_max_file_size(Some(4)),
    );
    assert_eq!(
        bounded.read_file(ReadFileParams::text("a.txt")).await.code,
        199_003
    );
    let unlimited = backend(
        LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_max_file_size(None),
    );
    let data = unlimited
        .read_file(ReadFileParams::text("a.txt"))
        .await
        .into_result()
        .expect("an unlimited read succeeds");
    assert_eq!(data.content, FileContent::Text("0123456789".to_string()));
    assert_eq!(data.total_bytes, 10);
}

#[tokio::test]
async fn a_bounded_read_returns_a_window_and_the_real_length() {
    let dir = workspace();
    let body: Vec<u8> = (0..=u8::MAX).cycle().take(1_048_576).collect();
    std::fs::write(dir.path().join("big.bin"), &body).unwrap();

    // The whole-file read refuses this file under a 1 KiB ceiling...
    let bounded = backend(
        LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_max_file_size(Some(1024)),
    );
    assert_eq!(
        bounded
            .read_file(ReadFileParams::bytes("big.bin"))
            .await
            .code,
        199_003
    );

    // ...while a 64-byte window of it succeeds and reports the real length.
    let window = bounded
        .read_bytes_limited(ReadBytesParams::new("big.bin", 0, 64))
        .await
        .expect("a bounded read never loads the file");
    assert_eq!(window.bytes, body[..64]);
    assert_eq!(window.total_bytes, 1_048_576);
    assert!(
        window.truncated,
        "the window ended before the end of the file"
    );
    assert!(window.path.ends_with("big.bin"));
}

#[tokio::test]
async fn a_bounded_read_honours_the_offset_and_the_end_of_the_file() {
    let dir = workspace();
    std::fs::write(dir.path().join("a.bin"), b"0123456789").unwrap();
    let op = default_backend(&dir);

    let middle = op
        .read_bytes_limited(ReadBytesParams::new("a.bin", 4, 3))
        .await
        .expect("a window inside the file");
    assert_eq!(middle.bytes, b"456");
    assert_eq!(middle.total_bytes, 10);
    assert!(middle.truncated);

    let tail = op
        .read_bytes_limited(ReadBytesParams::new("a.bin", 8, 64))
        .await
        .expect("a window past the requested size");
    assert_eq!(tail.bytes, b"89");
    assert!(
        !tail.truncated,
        "a window that reaches EOF is not truncated"
    );

    let past_end = op
        .read_bytes_limited(ReadBytesParams::new("a.bin", 99, 16))
        .await
        .expect("an offset past EOF is an empty window, not an error");
    assert_eq!(past_end.bytes, Vec::<u8>::new());
    assert_eq!(past_end.total_bytes, 10);
    assert!(!past_end.truncated);

    let empty_window = op
        .read_bytes_limited(ReadBytesParams::new("a.bin", 0, 0))
        .await
        .expect("a zero-byte window is allowed");
    assert_eq!(empty_window.bytes, Vec::<u8>::new());
    assert!(empty_window.truncated);
}

#[tokio::test]
async fn a_bounded_read_rejects_a_window_above_the_ceiling() {
    let dir = workspace();
    std::fs::write(dir.path().join("a.bin"), b"0123456789").unwrap();
    let op = backend(
        LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_max_file_size(Some(16)),
    );
    let error = op
        .read_bytes_limited(ReadBytesParams::new("a.bin", 0, 32))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .message()
            .contains("requested read window 32 exceeds the configured limit 16 bytes"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn a_bounded_read_reports_path_and_lock_failures_typed() {
    let dir = workspace();
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("secret.bin"), b"secret").unwrap();
    let op = backend(
        LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_sandbox_roots([format_path(dir.path())]),
    );

    let missing = op
        .read_bytes_limited(ReadBytesParams::new("missing.bin", 0, 8))
        .await
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::SysOp);
    assert!(missing.message().contains("File does not exist"));

    let escaped = format_path(&root.path().join("secret.bin"));
    let denied = op
        .read_bytes_limited(ReadBytesParams::new(escaped, 0, 8))
        .await
        .unwrap_err();
    assert_eq!(denied.kind(), ErrorKind::SysOp);
    assert!(
        denied.message().contains("Access denied"),
        "{}",
        denied.message()
    );

    let directory = op
        .read_bytes_limited(ReadBytesParams::new(".", 0, 8))
        .await
        .unwrap_err();
    assert_eq!(directory.kind(), ErrorKind::Validation);
    assert!(directory.message().contains("Is a directory"));
}

#[tokio::test]
async fn a_bounded_read_can_skip_the_locks() {
    let dir = workspace();
    std::fs::write(dir.path().join("a.bin"), b"0123456789").unwrap();
    let window = default_backend(&dir)
        .read_bytes_limited(ReadBytesParams::new("a.bin", 0, 4).with_only_read(true))
        .await
        .expect("an unlocked read succeeds");
    assert_eq!(window.bytes, b"0123");
}
