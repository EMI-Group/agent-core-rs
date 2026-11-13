//! Cross-process advisory lock behavior for the local filesystem backend.
//!
//! These tests prove the `flock` layer added on top of the in-process `RwLock` map:
//!
//! - same-process contention through two independent descriptors (a writer holds, a reader
//!   or `try_acquire` waits, and the lock is free again after the guard drops);
//! - genuine cross-process contention, proven by re-executing this test binary as a helper
//!   process (and, when the util-linux `flock(1)` CLI is installed, by it as well);
//! - the timeout -> `OpResult` 199003 mapping, including the Python `execution` slot;
//! - the canonical two-path acquisition order: two transfers locking the same pair of paths
//!   in opposite roles complete without deadlock;
//! - `cross_process_locks = false` creating no lock files at all.
//!
//! Every test uses `tempfile` directories, bounded waits, and no network.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tempfile::TempDir;

use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::fs::local::LocalFsOperation;
use openjiuwen_sysop::fs::lock::{
    default_lock_root, lock_path_for, FileLockManager, LockError, LockMode,
};
use openjiuwen_sysop::fs::types::{
    DownloadFileParams, FileContent, ReadFileParams, UploadFileParams, WriteFileParams,
};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::util::format_path;

/// Upper bound for every await that could otherwise hang the suite.
const BOUND: Duration = Duration::from_secs(10);

/// Environment variable that turns a test-binary run into the cross-process lock helper.
const HELPER_ENV: &str = "SYSOP_LOCK_HELPER";
/// Environment variable selecting the helper's lock mode (`read` or `write`).
const HELPER_MODE_ENV: &str = "SYSOP_LOCK_HELPER_MODE";
/// The helper test's name, used as the libtest `--exact` filter.
const HELPER_TEST: &str = "lock_helper_process";
/// Exit code of the helper when it took the lock.
const EXIT_ACQUIRED: i32 = 0;
/// Exit code of the helper when another process holds the lock.
const EXIT_CONTENDED: i32 = 3;
/// Exit code of the helper for any unexpected error.
const EXIT_ERROR: i32 = 4;

/// A configuration rooted at `dir` with the crate defaults (cross-process locks on).
fn workspace_config(dir: &TempDir) -> LocalWorkConfig {
    LocalWorkConfig::default().with_workspace(format_path(dir.path()))
}

/// A manager rooted at `dir/locks` so each test keeps its sidecar files to itself.
fn isolated_manager(dir: &TempDir) -> FileLockManager {
    FileLockManager::with_root(&LocalWorkConfig::default(), dir.path().join("locks"))
}

/// Skips the current test when the platform has no `flock`, reporting why.
fn skip_without_flock(manager: &FileLockManager) -> bool {
    if manager.is_enabled() {
        return false;
    }
    eprintln!("skipping: this platform has no flock (cross-process locks are disabled)");
    true
}

// ----- cross-process helper ---------------------------------------------------------

/// Child-process entry point used by [`a_second_process_observes_the_lock`].
///
/// Runs as an ordinary (passing) test when [`HELPER_ENV`] is unset. When it is set to
/// `<lock root>\n<target>`, this process tries a *non-blocking* lock through the public
/// manager and exits with [`EXIT_ACQUIRED`], [`EXIT_CONTENDED`], or [`EXIT_ERROR`], which is
/// what makes the contention observed by the parent genuinely cross-process.
#[test]
fn lock_helper_process() {
    let Some(spec) = std::env::var_os(HELPER_ENV) else {
        return;
    };
    let spec = spec.to_string_lossy().into_owned();
    let (root, target) = spec
        .split_once('\n')
        .expect("the helper spec carries a lock root and a target");
    let mode = match std::env::var(HELPER_MODE_ENV).as_deref() {
        Ok("read") => LockMode::Read,
        _ => LockMode::Write,
    };
    let manager = FileLockManager::with_root(&LocalWorkConfig::default(), root);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the helper runtime builds");
    let code = match runtime.block_on(manager.try_acquire(Path::new(target), mode)) {
        Ok(guard) => {
            let active = guard.is_active();
            drop(guard);
            if active {
                EXIT_ACQUIRED
            } else {
                eprintln!("helper: the platform took no lock");
                EXIT_ERROR
            }
        }
        Err(LockError::Contended { .. }) => EXIT_CONTENDED,
        Err(error) => {
            eprintln!("helper: lock error: {error}");
            EXIT_ERROR
        }
    };
    std::process::exit(code);
}

/// Runs the helper as a separate process and returns its exit code.
fn run_helper(root: &Path, target: &Path, mode: LockMode) -> i32 {
    let executable = std::env::current_exe().expect("the test binary path");
    let spec = format!("{}\n{}", root.display(), target.display());
    let output = std::process::Command::new(executable)
        .args(["--exact", HELPER_TEST, "--test-threads", "1"])
        .env(HELPER_ENV, spec)
        .env(HELPER_MODE_ENV, mode.label())
        .output()
        .expect("the helper process starts");
    output.status.code().unwrap_or(-1)
}

#[tokio::test]
async fn a_second_process_observes_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("locks");
    let target = dir.path().join("shared.txt");
    let manager = FileLockManager::with_root(&LocalWorkConfig::default(), &root);
    if skip_without_flock(&manager) {
        return;
    }

    let held = manager.acquire(&target, LockMode::Write).await.unwrap();
    let (root_for_helper, target_for_helper) = (root.clone(), target.clone());
    let contended = tokio::task::spawn_blocking(move || {
        run_helper(&root_for_helper, &target_for_helper, LockMode::Read)
    })
    .await
    .unwrap();
    assert_eq!(
        contended, EXIT_CONTENDED,
        "a second process must observe this process's exclusive lock"
    );

    drop(held);
    let (root_for_helper, target_for_helper) = (root.clone(), target.clone());
    let acquired = tokio::task::spawn_blocking(move || {
        run_helper(&root_for_helper, &target_for_helper, LockMode::Read)
    })
    .await
    .unwrap();
    assert_eq!(
        acquired, EXIT_ACQUIRED,
        "the second process must acquire the lock after the guard is dropped"
    );
}

/// Whether the util-linux `flock(1)` command is installed.
fn flock_cli_available() -> bool {
    std::process::Command::new("flock")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Runs `flock -n <lock file> true` and reports whether it could take the lock.
fn flock_cli_takes(lock_file: &Path) -> bool {
    std::process::Command::new("flock")
        .arg("-n")
        .arg(lock_file)
        .arg("true")
        .status()
        .expect("the flock command runs")
        .success()
}

#[tokio::test]
async fn the_flock_cli_observes_the_lock() {
    if !flock_cli_available() {
        eprintln!("skipping: the flock(1) CLI is not installed");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("cli.txt");
    let manager = isolated_manager(&dir);
    if skip_without_flock(&manager) {
        return;
    }

    let held = manager.acquire(&target, LockMode::Write).await.unwrap();
    let lock_file = manager.lock_path(&target);
    assert!(
        lock_file.is_file(),
        "the sidecar exists while the lock is held"
    );
    let contended = {
        let lock_file = lock_file.clone();
        tokio::task::spawn_blocking(move || flock_cli_takes(&lock_file))
            .await
            .unwrap()
    };
    assert!(
        !contended,
        "flock -n must fail while this process holds the exclusive lock"
    );

    drop(held);
    let free = tokio::task::spawn_blocking(move || flock_cli_takes(&lock_file))
        .await
        .unwrap();
    assert!(free, "the lock file is free once the guard is dropped");
}

// ----- same-process contention ------------------------------------------------------

#[tokio::test]
async fn a_reader_waits_for_an_exclusive_holder() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("contended.txt");
    let manager = Arc::new(isolated_manager(&dir));
    if skip_without_flock(&manager) {
        return;
    }

    let held = manager.acquire(&target, LockMode::Write).await.unwrap();
    let contended = manager.try_acquire(&target, LockMode::Read).await;
    assert!(
        matches!(contended, Err(LockError::Contended { .. })),
        "try_acquire must report the held lock instead of waiting"
    );

    let writer_released = Arc::new(AtomicBool::new(false));
    let waiter = {
        let manager = Arc::clone(&manager);
        let target = target.clone();
        let writer_released = Arc::clone(&writer_released);
        tokio::spawn(async move {
            let guard = manager.acquire(&target, LockMode::Read).await.unwrap();
            assert!(
                writer_released.load(Ordering::SeqCst),
                "the reader must acquire only after the writer released"
            );
            drop(guard);
        })
    };

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !waiter.is_finished(),
        "the reader must still be waiting while the writer holds the lock"
    );
    writer_released.store(true, Ordering::SeqCst);
    drop(held);
    tokio::time::timeout(BOUND, waiter)
        .await
        .expect("the reader proceeds once the writer releases")
        .unwrap();

    manager
        .try_acquire(&target, LockMode::Read)
        .await
        .expect("try_acquire succeeds once every guard is dropped");
}

// ----- failure mapping --------------------------------------------------------------

#[tokio::test]
async fn a_blocked_read_reports_199003_with_the_method_name_and_a_timeout() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"payload").unwrap();
    let config = LocalWorkConfig {
        lock_timeout_secs: 1,
        ..workspace_config(&dir)
    };
    let backend = LocalFsOperation::new(config).unwrap();
    let manager = FileLockManager::new(&LocalWorkConfig::default());
    if skip_without_flock(&manager) {
        return;
    }

    let resolved = tokio::fs::canonicalize(dir.path().join("a.txt"))
        .await
        .unwrap();
    let held = manager.acquire(&resolved, LockMode::Write).await.unwrap();
    let failure = backend.read_file(ReadFileParams::text("a.txt")).await;
    assert_eq!(failure.code, 199_003);
    assert!(
        failure.message.contains("execution: read_file"),
        "{}",
        failure.message
    );
    assert!(failure.message.contains("timeout"), "{}", failure.message);
    assert!(
        failure.message.contains(&format_path(&resolved)),
        "{}",
        failure.message
    );
    drop(held);

    // With the lock free again the same read succeeds.
    let data = backend
        .read_file(ReadFileParams::text("a.txt"))
        .await
        .into_result()
        .unwrap();
    assert_eq!(data.content, FileContent::Text("payload".to_string()));
}

#[tokio::test]
async fn a_blocked_write_reports_199003_with_the_method_name_and_a_timeout() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"old").unwrap();
    let config = LocalWorkConfig {
        lock_timeout_secs: 1,
        ..workspace_config(&dir)
    };
    let backend = LocalFsOperation::new(config).unwrap();
    let manager = FileLockManager::new(&LocalWorkConfig::default());
    if skip_without_flock(&manager) {
        return;
    }

    let resolved = tokio::fs::canonicalize(dir.path().join("a.txt"))
        .await
        .unwrap();
    let held = manager.acquire(&resolved, LockMode::Read).await.unwrap();
    let failure = backend
        .write_file(WriteFileParams::raw("a.txt", "new"))
        .await;
    assert_eq!(failure.code, 199_003);
    assert!(
        failure.message.contains("execution: write_file"),
        "{}",
        failure.message
    );
    assert!(failure.message.contains("timeout"), "{}", failure.message);
    drop(held);
    assert_eq!(
        std::fs::read(dir.path().join("a.txt")).unwrap(),
        b"old".to_vec()
    );
}

// ----- canonical two-path ordering --------------------------------------------------

#[tokio::test]
async fn reversed_transfers_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let host = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"workspace").unwrap();
    std::fs::write(host.path().join("a.txt"), b"host").unwrap();
    let backend = Arc::new(LocalFsOperation::new(workspace_config(&dir)).unwrap());
    let host_file = format_path(&host.path().join("a.txt"));

    // A -> B and B -> A on the same pair of paths; the canonical ordering must prevent the
    // classic two-lock deadlock.
    let upload = {
        let backend = Arc::clone(&backend);
        let host_file = host_file.clone();
        tokio::spawn(async move {
            backend
                .upload_file(UploadFileParams {
                    local_path: host_file,
                    target_path: "a.txt".to_string(),
                    overwrite: true,
                    ..UploadFileParams::default()
                })
                .await
        })
    };
    let download = {
        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            backend
                .download_file(DownloadFileParams {
                    source_path: "a.txt".to_string(),
                    local_path: host_file,
                    overwrite: true,
                    ..DownloadFileParams::default()
                })
                .await
        })
    };

    let (upload, download) = tokio::time::timeout(BOUND, async {
        (upload.await.unwrap(), download.await.unwrap())
    })
    .await
    .expect("reversed transfers must not deadlock");
    upload.into_result().unwrap();
    download.into_result().unwrap();
}

#[tokio::test]
async fn reversed_transfer_streams_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let host = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"workspace").unwrap();
    std::fs::write(host.path().join("a.txt"), b"host").unwrap();
    let backend = Arc::new(LocalFsOperation::new(workspace_config(&dir)).unwrap());
    let host_file = format_path(&host.path().join("a.txt"));

    let upload = {
        let backend = Arc::clone(&backend);
        let host_file = host_file.clone();
        tokio::spawn(async move {
            backend
                .upload_file_stream(UploadFileParams {
                    local_path: host_file,
                    target_path: "a.txt".to_string(),
                    overwrite: true,
                    chunk_size: 2,
                    ..UploadFileParams::default()
                })
                .collect::<Vec<_>>()
                .await
        })
    };
    let download = {
        let backend = Arc::clone(&backend);
        tokio::spawn(async move {
            backend
                .download_file_stream(DownloadFileParams {
                    source_path: "a.txt".to_string(),
                    local_path: host_file,
                    overwrite: true,
                    chunk_size: 2,
                    ..DownloadFileParams::default()
                })
                .collect::<Vec<_>>()
                .await
        })
    };

    let (upload, download) = tokio::time::timeout(BOUND, async {
        (upload.await.unwrap(), download.await.unwrap())
    })
    .await
    .expect("reversed transfer streams must not deadlock");
    assert_ne!(upload.len(), 0, "the upload stream yields chunks");
    assert_ne!(download.len(), 0, "the download stream yields chunks");
    assert!(upload.iter().all(|item| item.code == 0), "{upload:?}");
    assert!(download.iter().all(|item| item.code == 0), "{download:?}");
}

// ----- configuration switch ---------------------------------------------------------

#[tokio::test]
async fn enabling_cross_process_locks_creates_the_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let backend = LocalFsOperation::new(workspace_config(&dir)).unwrap();
    let manager = FileLockManager::new(&LocalWorkConfig::default());
    if skip_without_flock(&manager) {
        return;
    }

    backend
        .write_file(WriteFileParams::raw("a.txt", "hi"))
        .await
        .into_result()
        .unwrap();
    let resolved = tokio::fs::canonicalize(dir.path().join("a.txt"))
        .await
        .unwrap();
    let lock_file = lock_path_for(&default_lock_root(), &resolved);
    assert!(
        lock_file.is_file(),
        "the default configuration must create {}",
        lock_file.display()
    );
}

#[tokio::test]
async fn disabling_cross_process_locks_creates_no_lock_files() {
    let dir = tempfile::tempdir().unwrap();
    let config = LocalWorkConfig {
        cross_process_locks: false,
        ..workspace_config(&dir)
    };
    let backend = LocalFsOperation::new(config).unwrap();
    backend
        .write_file(WriteFileParams::raw("a.txt", "hi"))
        .await
        .into_result()
        .unwrap();
    let read = backend
        .read_file(ReadFileParams::text("a.txt"))
        .await
        .into_result()
        .unwrap();
    assert_eq!(read.content, FileContent::Text("hi".to_string()));

    let resolved = tokio::fs::canonicalize(dir.path().join("a.txt"))
        .await
        .unwrap();
    let lock_file = lock_path_for(&default_lock_root(), &resolved);
    assert!(
        !lock_file.exists(),
        "a disabled manager must not create {}",
        lock_file.display()
    );
}
