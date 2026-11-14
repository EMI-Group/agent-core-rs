//! Conditional-write behavior of the provider-neutral filesystem wrapper.
//!
//! Python parity: the reference providers in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` do not implement
//! `expected_content_sha256` / `create_if_absent` at all — only the local provider does — so
//! these cases pin the additive Rust capability described in
//! `crate::sys_operation::remote::conditional`: [`RemoteFs`] evaluates the precondition itself
//! over the backend's read and write primitives, and only when the backend opts in through
//! [`SandboxBackend::supports_conditional_write`].

use futures::future::poll_immediate;
use openjiuwen_core::sys_operation::fs::types::WriteFileParams;

use crate::sys_operation::remote::conditional::{lock_path, sha256_hex};

use super::*;

/// A `create_if_absent` write request; the kernel's default `create_if_not_exist`
/// (`true`) satisfies the flag combination the preconditions accept.
fn create_if_absent(path: &str, content: &str) -> WriteFileParams {
    let mut params = WriteFileParams::raw(path, content);
    params.create_if_absent = true;
    params
}

/// A write request that requires the current remote bytes to hash to `digest`.
fn expecting(path: &str, content: &str, digest: &str) -> WriteFileParams {
    WriteFileParams::raw(path, content).with_expected_sha256(digest)
}

#[tokio::test]
async fn the_capability_is_the_backends_opt_in() {
    let plain = Arc::new(FakeBackend::new());
    assert!(!fs_backend(&plain).supports_conditional_write());
    let capable = Arc::new(FakeBackend::conditional());
    assert!(fs_backend(&capable).supports_conditional_write());
}

#[tokio::test]
async fn a_plain_write_takes_no_conditional_path() {
    let backend = Arc::new(FakeBackend::conditional());
    let fs = fs_backend(&backend);

    // Hold the per-path lock a conditional write of this path would take: a plain write must
    // not wait for it, so it completes within a few polls while the guard is still alive.
    let guard = lock_path("fake", "/work/plain.txt").await;
    let mut write = Box::pin(fs.write_file(WriteFileParams::raw("/work/plain.txt", "hello")));
    let mut completed = None;
    for _ in 0..32 {
        if let Some(result) = poll_immediate(write.as_mut()).await {
            completed = Some(result);
            break;
        }
        tokio::task::yield_now().await;
    }
    let result = completed.expect("a plain write must not take the conditional-write lock");
    drop(guard);

    assert_eq!(result.code, 0);
    assert_eq!(result.data.as_ref().unwrap().size, 5);
    assert_eq!(backend.contents("/work/plain.txt").await.unwrap(), b"hello");
    // A plain write with the kernel's default `create_if_not_exist` never reads, and it is the
    // read-check-write sequence that takes the lock.
    assert_eq!(backend.read_count().await, 0, "a plain write reads nothing");
    assert_eq!(
        backend.writes().await,
        vec![("/work/plain.txt".to_string(), 5)]
    );
}

#[tokio::test]
async fn create_if_absent_writes_a_missing_path() {
    let backend = Arc::new(FakeBackend::conditional());
    let fs = fs_backend(&backend);

    let result = fs
        .write_file(create_if_absent("/work/new.txt", "hello"))
        .await;

    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(result.data.as_ref().unwrap().size, 5);
    assert_eq!(backend.contents("/work/new.txt").await.unwrap(), b"hello");
    assert_eq!(
        backend.writes().await,
        vec![("/work/new.txt".to_string(), 5)]
    );
}

#[tokio::test]
async fn create_if_absent_rejects_an_existing_path_without_writing() {
    let backend = Arc::new(FakeBackend::conditional());
    backend.put("/work/existing.txt", b"original").await;
    let fs = fs_backend(&backend);

    let result = fs
        .write_file(create_if_absent("/work/existing.txt", "replacement"))
        .await;

    assert_eq!(result.code, 199_003);
    assert!(
        result.message.contains("/work/existing.txt"),
        "{}",
        result.message
    );
    assert!(
        result.message.contains("create_if_absent"),
        "{}",
        result.message
    );
    assert_eq!(
        backend.contents("/work/existing.txt").await.unwrap(),
        b"original"
    );
    assert!(
        backend.writes().await.is_empty(),
        "a rejected precondition issues no write request"
    );
}

#[tokio::test]
async fn expected_sha256_accepts_matching_content() {
    let backend = Arc::new(FakeBackend::conditional());
    backend.put("/work/log.txt", b"hello").await;
    let fs = fs_backend(&backend);

    let result = fs
        .write_file(expecting("/work/log.txt", "goodbye", &sha256_hex(b"hello")))
        .await;

    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(backend.contents("/work/log.txt").await.unwrap(), b"goodbye");
    assert_eq!(
        backend.writes().await,
        vec![("/work/log.txt".to_string(), 7)]
    );
}

#[tokio::test]
async fn expected_sha256_rejects_a_mismatch_without_writing() {
    let backend = Arc::new(FakeBackend::conditional());
    backend.put("/work/log.txt", b"hello").await;
    let fs = fs_backend(&backend);
    let expected = sha256_hex(b"goodbye");

    let result = fs
        .write_file(expecting("/work/log.txt", "replacement", &expected))
        .await;

    assert_eq!(result.code, 199_003);
    assert!(
        result.message.contains("/work/log.txt"),
        "{}",
        result.message
    );
    assert!(result.message.contains(&expected), "{}", result.message);
    assert!(
        result.message.contains(&sha256_hex(b"hello")),
        "the message names the digest the remote file has: {}",
        result.message
    );
    assert_eq!(backend.contents("/work/log.txt").await.unwrap(), b"hello");
    assert!(backend.writes().await.is_empty(), "no write on a mismatch");
}

#[tokio::test]
async fn expected_sha256_rejects_a_missing_path_without_writing() {
    let backend = Arc::new(FakeBackend::conditional());
    let fs = fs_backend(&backend);
    let expected = sha256_hex(b"hello");

    let result = fs
        .write_file(expecting("/work/absent.txt", "hello", &expected))
        .await;

    assert_eq!(result.code, 199_003);
    assert!(
        result.message.contains("/work/absent.txt"),
        "{}",
        result.message
    );
    assert!(
        result.message.contains("does not exist"),
        "{}",
        result.message
    );
    assert!(backend.contents("/work/absent.txt").await.is_none());
    assert!(
        backend.writes().await.is_empty(),
        "no write on a missing file"
    );
}

#[tokio::test]
async fn incompatible_conditional_options_are_refused_before_any_io() {
    let backend = Arc::new(FakeBackend::conditional());
    let fs = fs_backend(&backend);

    let mut both = WriteFileParams::raw("/work/a.txt", "x").with_expected_sha256("deadbeef");
    both.create_if_absent = true;
    let absent_and_append = create_if_absent("/work/a.txt", "x").with_append();
    let digest_and_append = WriteFileParams::raw("/work/a.txt", "x")
        .with_expected_sha256("deadbeef")
        .with_append();
    let mut absent_without_create = create_if_absent("/work/a.txt", "x");
    absent_without_create.create_if_not_exist = false;

    for params in [
        both,
        absent_and_append,
        digest_and_append,
        absent_without_create,
    ] {
        let result = fs.write_file(params).await;
        assert_eq!(result.code, 199_003);
        assert!(
            result
                .message
                .contains("conditional write options are incompatible"),
            "{}",
            result.message
        );
    }
    assert_eq!(
        backend.read_count().await,
        0,
        "the flag combination is rejected before the read"
    );
    assert!(backend.writes().await.is_empty(), "and before the write");
}
