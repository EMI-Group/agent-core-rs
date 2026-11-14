//! Remote-container teardown across the crate boundary.
//!
//! Python parity: `delete_jiuwenbox_sandbox` / `delete_yuanrong_sandbox` in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/`. Both pop the matching
//! entries out of the process-wide container cache and then release the containers,
//! treating "already gone" (`404`) as success; `aio` has no container-delete API, so
//! its teardown is a typed `Unsupported` that names the gateway.
//!
//! Every case enters through `openjiuwen_extensions::sys_operation::{shutdown_remote,
//! shutdown_remote_all, SandboxSelector}` and a loopback `wiremock` peer. The providers
//! keep process-global container caches shared with the rest of this binary, so a case
//! only ever asserts on its own entries (one isolation key each) and the whole-cache
//! drain runs in a child process.

use std::sync::atomic::{AtomicUsize, Ordering};

use openjiuwen_extensions::sys_operation::{shutdown_remote, shutdown_remote_all, SandboxSelector};

use super::*;

/// Environment variable that turns a test-binary run into the whole-cache drain helper.
const TEARDOWN_HELPER_ENV: &str = "OPENJIUWEN_EXTENSIONS_SANDBOX_TEARDOWN_HELPER";

/// The shared cache key a release reports for `server` + `isolation`: the normalized
/// base URL, the separator, then the isolation key.
fn scope_key(server: &MockServer, isolation: &str) -> String {
    format!("{}|{isolation}", server.uri())
}

// ─── jiuwenbox: DELETE /api/v1/sandboxes/{id} ───────────────────────────────
// Python: `delete_jiuwenbox_sandbox` releases the cached container and its scope.
#[tokio::test]
async fn jiuwenbox_teardown_deletes_the_container_and_drains_the_cache() {
    let server = MockServer::start().await;
    // Two containers, handed out in creation order, so the second read can prove the
    // cache was drained by the teardown rather than reused.
    let creates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&creates);
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(move |_: &Request| {
            let index = counter.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({"id": format!("sbx-teardown-{index}")}))
        })
        .mount(&server)
        .await;
    for (sandbox_id, remote_path) in [
        ("sbx-teardown-0", "/workspace/first.txt"),
        ("sbx-teardown-1", "/workspace/second.txt"),
    ] {
        Mock::given(method("GET"))
            .and(path(format!("/api/v1/sandboxes/{sandbox_id}/download")))
            .and(query_param("sandbox_path", remote_path))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
            .mount(&server)
            .await;
    }
    Mock::given(method("DELETE"))
        .and(path("/api/v1/sandboxes/sbx-teardown-0"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let isolation = "session_jiuwenbox_teardown";
    let config = jiuwenbox_config(&server, json!({}));
    let fs = fs_operation(
        &registry(),
        "jiuwenbox",
        &endpoint(&server, None, isolation),
        &config,
    );
    let first = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/first.txt")),
    )
    .await
    .unwrap();
    assert_eq!(first.code, 0, "{}", first.message);
    assert_eq!(
        creates.load(Ordering::SeqCst),
        1,
        "the read creates a container"
    );

    let selector = SandboxSelector::scope(&server.uri(), Some(isolation)).with_reason("shutdown");
    let report = tokio::time::timeout(BOUND, shutdown_remote("jiuwenbox", &selector))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(report.sandbox_type, "jiuwenbox");
    assert_eq!(report.released_keys, vec![scope_key(&server, isolation)]);
    assert_eq!(report.deleted, vec!["sbx-teardown-0".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert!(report.unsupported.is_none());

    let recorded = requests(&server).await;
    let deletes: Vec<&Request> = recorded
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1, "one container, one delete");
    assert_eq!(deletes[0].url.path(), "/api/v1/sandboxes/sbx-teardown-0");
    assert!(query(deletes[0]).is_empty(), "the delete carries no query");

    // A later operation from a fresh provider creates a second container: the shared
    // cache no longer holds the released scope.
    let later = fs_operation(
        &registry(),
        "jiuwenbox",
        &endpoint(&server, None, isolation),
        &config,
    );
    let second = tokio::time::timeout(
        BOUND,
        later.read_file(ReadFileParams::text("/workspace/second.txt")),
    )
    .await
    .unwrap();
    assert_eq!(second.code, 0, "{}", second.message);
    assert_eq!(
        creates.load(Ordering::SeqCst),
        2,
        "the drained cache must force a second create"
    );
}

// Python: a `404` from the provider means the container is already gone (released,
// not failed) — `delete_jiuwenbox_sandbox` logs it instead of raising.
#[tokio::test]
async fn jiuwenbox_teardown_tolerates_a_container_the_server_already_dropped() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-gone/download"))
        .and(query_param("sandbox_path", "/workspace/a.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/api/v1/sandboxes/sbx-gone"))
        .respond_with(ResponseTemplate::new(404).set_body_string("Sandbox not found"))
        .mount(&server)
        .await;

    let isolation = "session_jiuwenbox_teardown_404";
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-gone"}));
    let fs = fs_operation(
        &registry(),
        "jiuwenbox",
        &endpoint(&server, None, isolation),
        &config,
    );
    let read = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(read.code, 0, "{}", read.message);

    let selector = SandboxSelector::scope(&server.uri(), Some(isolation));
    let report = tokio::time::timeout(BOUND, shutdown_remote("jiuwenbox", &selector))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(report.released_keys, vec![scope_key(&server, isolation)]);
    assert_eq!(report.deleted, vec!["sbx-gone".to_string()]);
    assert!(
        report.failed.is_empty(),
        "a 404 is a released container, not a failure: {:?}",
        report.failed
    );
}

// ─── yuanrong: DELETE /api/environment/v1/environments/{id} ─────────────────
// Python: `delete_yuanrong_sandbox` releases the cached ADX environment.
#[tokio::test]
async fn yuanrong_teardown_deletes_the_environment_with_the_adx_delete() {
    let server = MockServer::start().await;
    mount_yuanrong_create(&server, "env-teardown", "docker").await;
    mount_yuanrong_invoke(&server, "env-teardown", 200, json!({"data": hex("hello")})).await;
    Mock::given(method("DELETE"))
        .and(path("/api/environment/v1/environments/env-teardown"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code": 0})))
        .mount(&server)
        .await;

    let isolation = "session_yuanrong_teardown";
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(
        &registry(),
        "yuanrong",
        &endpoint(&server, None, isolation),
        &config,
    );
    let read = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(read.code, 0, "{}", read.message);

    let selector = SandboxSelector::scope(&server.uri(), Some(isolation));
    let report = tokio::time::timeout(BOUND, shutdown_remote("yuanrong", &selector))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(report.sandbox_type, "yuanrong");
    assert_eq!(report.released_keys, vec![scope_key(&server, isolation)]);
    assert_eq!(report.deleted, vec!["env-teardown".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert!(report.unsupported.is_none());

    let recorded = requests(&server).await;
    let deletes: Vec<&Request> = recorded
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1, "one environment, one delete");
    assert_eq!(
        deletes[0].url.path(),
        "/api/environment/v1/environments/env-teardown"
    );
    assert_eq!(header(deletes[0], "accept"), "application/json");
}

// ─── refusals ───────────────────────────────────────────────────────────────
// Python: the AIO gateway owns the container lifetime; it exposes no delete API.
#[tokio::test]
async fn aio_teardown_is_unsupported_and_issues_no_request() {
    let server = MockServer::start().await;
    let selector = SandboxSelector::scope(&server.uri(), Some("session_aio_teardown"));
    let error = tokio::time::timeout(BOUND, shutdown_remote("aio", &selector))
        .await
        .unwrap()
        .expect_err("the aio gateway has no container-delete API");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message().to_string();
    assert!(message.contains("aio"), "{message}");
    assert!(message.contains("gateway"), "{message}");
    assert!(message.contains("no container-delete API"), "{message}");
    assert!(
        requests(&server).await.is_empty(),
        "the aio teardown must not contact the gateway at all"
    );
}

// Python: `delete_*_sandbox` accepts only the three remote provider names, and an
// empty selection has nothing to release.
#[tokio::test]
async fn teardown_names_the_three_types_and_needs_a_selection() {
    let error = shutdown_remote("local", &SandboxSelector::all())
        .await
        .expect_err("'local' is not a remote sandbox type");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error
            .message()
            .contains("supported types: aio, jiuwenbox, yuanrong"),
        "{}",
        error.message()
    );

    let error = shutdown_remote("jiuwenbox", &SandboxSelector::default())
        .await
        .expect_err("the default selector names nothing to release");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("jiuwenbox"),
        "the rejection names the provider: {}",
        error.message()
    );

    let error = shutdown_remote("yuanrong", &SandboxSelector::default())
        .await
        .expect_err("the default selector names nothing to release");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

// ─── whole-cache drain ──────────────────────────────────────────────────────
/// The drain reports every remote type, in [`REMOTE_SANDBOX_TYPES`] order.
///
/// `shutdown_remote_all` drains the process-global container caches of `jiuwenbox` and
/// `yuanrong`, so it runs in a child process: the cases above run in parallel and
/// assert on their own cache entries, which an in-process drain would release first.
#[test]
fn shutdown_remote_all_reports_every_type_in_order() {
    let executable = std::env::current_exe().expect("the test binary path");
    let output = std::process::Command::new(executable)
        .args([
            "--exact",
            "teardown::shutdown_remote_all_helper",
            "--test-threads",
            "1",
        ])
        .env(TEARDOWN_HELPER_ENV, "1")
        .output()
        .expect("the helper process starts");
    assert!(
        output.status.success(),
        "the teardown drain case failed in a child process:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child-process entry point of [`shutdown_remote_all_reports_every_type_in_order`].
///
/// Runs as an ordinary (passing) test when [`TEARDOWN_HELPER_ENV`] is unset.
#[test]
fn shutdown_remote_all_helper() {
    if std::env::var_os(TEARDOWN_HELPER_ENV).is_none() {
        return;
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the helper runtime builds");
    runtime.block_on(async {
        let reports = shutdown_remote_all().await.unwrap();
        assert_eq!(reports.len(), REMOTE_SANDBOX_TYPES.len());
        for (report, sandbox_type) in reports.iter().zip(REMOTE_SANDBOX_TYPES) {
            assert_eq!(report.sandbox_type, sandbox_type);
        }
        let aio = &reports[0];
        let reason = aio
            .unsupported
            .as_deref()
            .expect("the aio report carries the reason it cannot be torn down");
        assert!(reason.contains("gateway"), "{reason}");
        assert!(aio.released_keys.is_empty(), "{:?}", aio.released_keys);
        assert!(aio.deleted.is_empty(), "{:?}", aio.deleted);
        // The other two have a teardown path, so their reports carry no reason. A fresh
        // process holds no cached container, so they release nothing either.
        for report in &reports[1..] {
            assert!(
                report.unsupported.is_none(),
                "{} has a teardown path",
                report.sandbox_type
            );
            assert!(
                report.released_keys.is_empty(),
                "{:?}",
                report.released_keys
            );
            assert!(report.deleted.is_empty(), "{:?}", report.deleted);
            assert!(report.failed.is_empty(), "{:?}", report.failed);
        }
    });
}
