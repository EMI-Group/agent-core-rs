//! Ports `tests/unit_tests/core/sys_operation/local/test_fs_operation.py` (read slicing, byte mode, limits, conditional write, streaming).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_sysop::config::{BinaryPolicy, LocalWorkConfig};
use openjiuwen_sysop::fs::local_fs;
use openjiuwen_sysop::fs::types::{
    FileContent, ReadChunkData, ReadFileData, ReadFileParams, WriteFileParams,
};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::result::OpResult;
use openjiuwen_sysop::status::SysOpStatus;
use tempfile::TempDir;

/// The five-line fixture every slicing test reads.
const FIVE_LINES: &str = "one\ntwo\nthree\nfour\nfive\n";

/// Builds an unrestricted local backend rooted at `dir`.
fn backend(dir: &TempDir) -> Arc<dyn FsOperation> {
    let config = LocalWorkConfig::unrestricted(dir.path().to_string_lossy().into_owned());
    local_fs(config).unwrap()
}

/// Writes `bytes` to `name` inside `dir`.
fn write(dir: &TempDir, name: &str, bytes: &[u8]) {
    std::fs::write(dir.path().join(name), bytes).unwrap();
}

/// Writes the five-line fixture and returns a backend rooted at `dir`.
fn fixture(dir: &TempDir) -> Arc<dyn FsOperation> {
    write(dir, "lines.txt", FIVE_LINES.as_bytes());
    backend(dir)
}

/// Reads `params` and returns the decoded text, requiring success.
async fn read_text(op: &Arc<dyn FsOperation>, params: ReadFileParams) -> String {
    let data = op.read_file(params).await.into_result().unwrap();
    data.content.as_text().map(str::to_string).unwrap()
}

/// Reads `params` and requires a `199003` failure, returning the raw result.
async fn expect_read_failure(
    op: &Arc<dyn FsOperation>,
    params: ReadFileParams,
) -> OpResult<ReadFileData> {
    let result = op.read_file(params).await;
    assert_eq!(
        result.code,
        SysOpStatus::FsExecutionError.code(),
        "expected a filesystem failure, got code {} ({})",
        result.code,
        result.message
    );
    assert!(result.data.is_none(), "a failure carries no payload");
    result
}

/// Drains a read stream into its items.
async fn drain_stream(
    op: &Arc<dyn FsOperation>,
    params: ReadFileParams,
) -> Vec<OpResult<ReadChunkData>> {
    Arc::clone(op).read_file_stream(params).collect().await
}

#[tokio::test]
async fn head_keeps_first_lines() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_head(2)).await;
    assert_eq!(content, "one\ntwo\n");
}

#[tokio::test]
async fn tail_keeps_last_lines() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_tail(2)).await;
    assert_eq!(content, "four\nfive\n");
}

#[tokio::test]
async fn head_zero_reads_whole_file() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_head(0)).await;
    assert_eq!(
        content, FIVE_LINES,
        "head=0 means 'not specified' in Python"
    );
}

#[tokio::test]
async fn tail_zero_reads_whole_file() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_tail(0)).await;
    assert_eq!(content, FIVE_LINES);
}

#[tokio::test]
async fn head_larger_than_file_reads_everything() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_head(99)).await;
    assert_eq!(content, FIVE_LINES);
}

#[tokio::test]
async fn line_range_is_one_based_and_inclusive() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let content = read_text(&op, ReadFileParams::text("lines.txt").with_line_range(2, 3)).await;
    assert_eq!(content, "two\nthree\n");
}

#[tokio::test]
async fn line_range_past_eof_is_empty_success() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let result = op
        .read_file(ReadFileParams::text("lines.txt").with_line_range(10, 20))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    let data = result.data.unwrap();
    assert_eq!(data.content, FileContent::Text(String::new()));
}

#[tokio::test]
async fn line_range_start_greater_than_end_is_empty_success() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let result = op
        .read_file(ReadFileParams::text("lines.txt").with_line_range(3, 1))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    let data = result.data.unwrap();
    assert_eq!(data.content, FileContent::Text(String::new()));
}

#[tokio::test]
async fn bytes_mode_rejects_head_with_exact_message() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let result = expect_read_failure(&op, ReadFileParams::bytes("lines.txt").with_head(1)).await;
    assert!(
        result.message.contains(
            "Parameters 'head', 'tail', and 'line_range' are only supported in text mode"
        ),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn head_and_tail_are_mutually_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let params = ReadFileParams::text("lines.txt").with_head(1).with_tail(1);
    let result = expect_read_failure(&op, params).await;
    assert!(
        result
            .message
            .contains("tail and head cannot be specified simultaneously"),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn head_and_line_range_are_mutually_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let params = ReadFileParams::text("lines.txt")
        .with_head(1)
        .with_line_range(1, 2);
    let result = expect_read_failure(&op, params).await;
    assert!(
        result
            .message
            .contains("head and line_range cannot be specified simultaneously"),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn tail_and_line_range_are_mutually_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let params = ReadFileParams::text("lines.txt")
        .with_tail(1)
        .with_line_range(1, 2);
    let result = expect_read_failure(&op, params).await;
    assert!(
        result
            .message
            .contains("tail and line_range cannot be specified simultaneously"),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn chunk_size_truncates_bytes_mode() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "data.bin", b"0123456789");
    let op = backend(&dir);

    let mut params = ReadFileParams::bytes("data.bin");
    params.chunk_size = 4;
    let data = op.read_file(params).await.into_result().unwrap();
    assert_eq!(data.content.as_bytes().unwrap().len(), 4);
    assert_eq!(data.content, FileContent::Bytes(b"0123".to_vec()));
    assert!(data.truncated, "a short chunk reports the truncation");
    assert_eq!(data.total_bytes, 10);

    let data = op
        .read_file(ReadFileParams::bytes("data.bin"))
        .await
        .into_result()
        .unwrap();
    assert_eq!(data.content, FileContent::Bytes(b"0123456789".to_vec()));
    assert!(!data.truncated, "chunk_size=0 reads the whole file");
    assert_eq!(data.total_bytes, 10);
}

#[tokio::test]
async fn text_mode_ignores_chunk_size() {
    let dir = tempfile::tempdir().unwrap();
    let op = fixture(&dir);
    let mut params = ReadFileParams::text("lines.txt");
    params.chunk_size = 2;
    let data = op.read_file(params).await.into_result().unwrap();
    assert_eq!(data.content, FileContent::Text(FIVE_LINES.to_string()));
    assert!(!data.truncated, "chunk_size applies to bytes mode only");
}

#[tokio::test]
async fn max_file_size_limit_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "data.bin", b"0123456789");
    let mut config = LocalWorkConfig::unrestricted(dir.path().to_string_lossy().into_owned());
    config.max_file_size_bytes = Some(4);
    let op = local_fs(config).unwrap();

    let result = expect_read_failure(&op, ReadFileParams::text("data.bin")).await;
    assert!(
        result.message.contains("exceeds the configured limit"),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn binary_content_is_rejected_under_reject_policy() {
    let dir = tempfile::tempdir().unwrap();
    let op = backend(&dir);
    op.write_file(WriteFileParams::raw(
        "blob.bin",
        FileContent::bytes(vec![0_u8, 1, 2, 3]),
    ))
    .await
    .into_result()
    .unwrap();

    let mut params = ReadFileParams::text("blob.bin");
    params.binary = BinaryPolicy::Reject;
    let result = expect_read_failure(&op, params).await;
    assert!(
        result.message.contains("binary"),
        "unexpected message: {}",
        result.message
    );
}

#[tokio::test]
async fn conditional_write_rejects_a_stale_digest() {
    let dir = tempfile::tempdir().unwrap();
    let op = backend(&dir);
    op.write_file(WriteFileParams::raw("notes.txt", "initial"))
        .await
        .into_result()
        .unwrap();

    let stale = WriteFileParams::raw("notes.txt", "new").with_expected_sha256("00".repeat(32));
    let result = op.write_file(stale).await;
    assert_eq!(result.code, SysOpStatus::FileVersionConflict.code());
    assert_eq!(result.code, 199_011);
    assert_eq!(result.message, "file version conflict");
    assert!(result.data.is_none());

    let unchanged = read_text(&op, ReadFileParams::text("notes.txt")).await;
    assert_eq!(
        unchanged, "initial",
        "a rejected write leaves the file intact"
    );
}

#[tokio::test]
async fn create_if_absent_rejects_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let op = backend(&dir);
    write(&dir, "existing.txt", b"");

    let mut params = WriteFileParams::raw("existing.txt", "new");
    params.create_if_absent = true;
    let result = op.write_file(params).await;
    assert_eq!(result.code, SysOpStatus::FileVersionConflict.code());
    assert_eq!(result.message, "file version conflict");
    assert_eq!(
        std::fs::read(dir.path().join("existing.txt")).unwrap(),
        b"",
        "the existing file is left unchanged"
    );

    let mut params = WriteFileParams::raw("fresh.txt", "new");
    params.create_if_absent = true;
    op.write_file(params).await.into_result().unwrap();
    assert_eq!(
        std::fs::read(dir.path().join("fresh.txt")).unwrap(),
        b"new",
        "create_if_absent creates a missing file"
    );
}

#[tokio::test]
async fn empty_file_text_stream_emits_zero_chunks() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "empty.txt", b"");
    let op = backend(&dir);
    let items = drain_stream(&op, ReadFileParams::text("empty.txt")).await;
    assert!(
        items.is_empty(),
        "an empty file yields no chunks, got {items:?}"
    );
}

#[tokio::test]
async fn text_stream_emits_one_chunk_per_line_and_terminal_flag() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "abc.txt", b"a\nb\nc\n");
    let op = backend(&dir);
    let items = drain_stream(&op, ReadFileParams::text("abc.txt")).await;
    assert_eq!(items.len(), 3);
    for item in &items {
        assert_eq!(item.code, SysOpStatus::Success.code(), "{}", item.message);
    }
    let chunks: Vec<FileContent> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap().chunk.clone())
        .collect();
    assert_eq!(
        chunks,
        vec![
            FileContent::Text("a\n".to_string()),
            FileContent::Text("b\n".to_string()),
            FileContent::Text("c\n".to_string()),
        ]
    );
    let indices: Vec<u64> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap().chunk_index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
    let last_flags: Vec<bool> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap().is_last_chunk)
        .collect();
    assert_eq!(last_flags, vec![false, false, true]);

    let invalid = ReadFileParams::text("abc.txt").with_head(1).with_tail(1);
    let failures = drain_stream(&op, invalid).await;
    assert_eq!(failures.len(), 1, "a bad request yields one error chunk");
    assert_eq!(failures[0].code, SysOpStatus::FsExecutionError.code());
    assert!(failures[0].data.is_none());
}
