//! Teardown tests for the `yuanrong` ADX backend.
//!
//! Python counterpart: `delete_yuanrong_sandbox` in
//! `tests/unit_tests/extensions/sys_operation/sandbox/test_yuanrong.py`.
//!
//! # Intentional difference from Python
//!
//! `delete_yuanrong_sandbox` returns the released environment ids and cannot
//! report a per-environment failure: its `_delete_environment` helper swallows
//! the transport error (logging it at `warning`) so a failing environment looks
//! exactly like a released one. This implementation pops the entry out of the
//! cache and records the rendered transport reason in `TeardownReport::failed`,
//! so a caller can distinguish "released" from "tried and failed" without
//! scraping logs (see [`a_500_delete_is_recorded_instead_of_swallowed`]).
//!
//! Every test talks to a loopback `wiremock` server and uses an isolation key
//! unique to the test, because the environment cache is process-global and the
//! suite runs in parallel: whole-cache drains only assert on this test's own
//! entries, never on global emptiness.
// The parent module already allows these; repeated here so the file stays
// self-contained if it is ever moved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WireRequest, ResponseTemplate};

use crate::sys_operation::remote::backend::SandboxBackend;
use crate::sys_operation::SandboxEndpoint;

use super::super::{shared_key, teardown, SandboxSelector, YuanRongBackend};
use super::{gateway_config, mount_create, mount_invoke, CREATE_PATH, NEXT_ISOLATION};

/// An endpoint bound to `base_url` under a caller-chosen isolation key.
fn scoped_endpoint(base_url: &str, isolation: &str) -> SandboxEndpoint {
    SandboxEndpoint {
        base_url: base_url.to_string(),
        sandbox_id: None,
        isolation_key: Some(isolation.to_string()),
    }
}

/// A backend for one isolation scope of `server`.
///
/// Each call builds a fresh backend, so the per-backend environment slot starts
/// empty and the operation has to consult (and then repopulate) the shared cache.
fn scoped_backend(server: &MockServer, isolation: &str) -> YuanRongBackend {
    YuanRongBackend::new(
        &scoped_endpoint(&server.uri(), isolation),
        &gateway_config(&[]),
    )
    .expect("backend")
}

/// A fresh isolation key, unique per test so the process-global cache cannot leak
/// an entry between tests (the `endpoint` helper in the parent module does the
/// same for the operation tests).
fn isolation(label: &str) -> String {
    format!(
        "teardown_{label}_{}",
        NEXT_ISOLATION.fetch_add(1, Ordering::SeqCst)
    )
}

/// Serializes the teardown tests of this module.
///
/// The environment cache is process-global and [`SandboxSelector::all`] drains
/// all of it, so two teardown tests running in parallel would delete each
/// other's entries and every "the cache still/again holds my entry" assertion
/// would race. Holding this guard for the whole test body keeps the drains
/// deterministic without touching production code.
///
/// Waits until no other teardown test is running.
async fn exclusive_teardown() -> tokio::sync::MutexGuard<'static, ()> {
    crate::sys_operation::remote::teardown_guard().lock().await
}

/// Mounts one `DELETE` responder answering `status` with `body` for every path.
async fn mount_delete(server: &MockServer, status: u16, body: &'static str) {
    Mock::given(method("DELETE"))
        .respond_with(move |_request: &WireRequest| {
            ResponseTemplate::new(status).set_body_string(body)
        })
        .mount(server)
        .await;
}

/// The ADX management-plane create requests the server has answered so far.
async fn create_count(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .filter(|request| request.url.path() == CREATE_PATH)
        .count()
}

/// The `DELETE` requests the server recorded.
async fn delete_paths(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("requests")
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .map(|request| request.url.path().to_string())
        .collect()
}

#[tokio::test]
async fn teardown_deletes_the_cached_environment_and_drains_the_cache() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    // A slashed id proves the delete path percent-encodes the id.
    mount_create(&server, "env/1").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 200, "released").await;

    let isolation = isolation("drain");
    let backend = scoped_backend(&server, &isolation);
    assert_eq!(backend.read_bytes("/a.txt").await.expect("read"), b"hi");
    assert_eq!(create_count(&server).await, 1);

    let selector =
        SandboxSelector::scope(&server.uri(), Some(&isolation)).with_reason("test drain");
    let report = teardown(&selector).await.expect("teardown");

    assert_eq!(report.sandbox_type, "yuanrong");
    assert_eq!(
        report.released_keys,
        vec![shared_key(&server.uri(), Some(&isolation))]
    );
    assert_eq!(report.deleted, vec!["env/1".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert!(report.unsupported.is_none());
    assert!(!report.is_empty());

    let paths = delete_paths(&server).await;
    assert_eq!(paths, vec!["/api/environment/v1/environments/env%2F1"]);
    let delete = server
        .received_requests()
        .await
        .expect("requests")
        .into_iter()
        .find(|request| request.method.as_str() == "DELETE")
        .expect("delete request");
    assert_eq!(
        delete
            .headers
            .get("accept")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );

    // The cache entry is gone, so a fresh backend of the same scope creates again.
    let fresh = scoped_backend(&server, &isolation);
    assert_eq!(fresh.read_bytes("/b.txt").await.expect("read"), b"hi");
    assert_eq!(create_count(&server).await, 2);
}

#[tokio::test]
async fn a_404_delete_counts_as_released() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    mount_create(&server, "env-404").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    // Python treats an environment the management plane no longer knows as gone.
    mount_delete(&server, 404, "Sandbox 'env-404' not found").await;

    let isolation = isolation("gone");
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/a")
            .await
            .expect("read"),
        b"hi"
    );

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&isolation)))
        .await
        .expect("teardown");
    assert_eq!(report.deleted, vec!["env-404".to_string()]);
    assert_eq!(
        report.released_keys,
        vec![shared_key(&server.uri(), Some(&isolation))]
    );
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert_eq!(
        delete_paths(&server).await,
        vec!["/api/environment/v1/environments/env-404"]
    );

    // A tolerated 404 still drains the cache.
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/b")
            .await
            .expect("read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);
}

#[tokio::test]
async fn a_500_delete_is_recorded_instead_of_swallowed() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    mount_create(&server, "env-500").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 500, "adx exploded").await;

    let isolation = isolation("failed");
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/a")
            .await
            .expect("read"),
        b"hi"
    );

    let report = teardown(
        &SandboxSelector::scope(&server.uri(), Some(&isolation)).with_reason("test failure"),
    )
    .await
    .expect("teardown");

    // Python's `_delete_environment` swallows this failure and returns the id as
    // released; here the id is reported as a failure instead of a release.
    assert!(report.deleted.is_empty(), "{:?}", report.deleted);
    assert!(
        report.released_keys.is_empty(),
        "{:?}",
        report.released_keys
    );
    assert_eq!(report.failed.len(), 1, "{:?}", report.failed);
    let failure = &report.failed[0];
    assert_eq!(failure.key, shared_key(&server.uri(), Some(&isolation)));
    assert_eq!(failure.container_id, "env-500");
    assert!(failure.reason.contains("500"), "{}", failure.reason);
    assert!(failure.reason.contains("delete_environment"), "{failure:?}");
    assert!(
        failure.reason.contains("adx exploded"),
        "{}",
        failure.reason
    );

    // The entry leaves the cache before the delete is attempted, so a fresh
    // backend of the same scope creates another environment.
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/b")
            .await
            .expect("read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);
}

#[tokio::test]
async fn a_base_url_selector_releases_every_scope_of_that_server() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    mount_create(&server, "env-base").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 200, "released").await;

    let first = isolation("base_a");
    let second = isolation("base_b");
    assert_eq!(
        scoped_backend(&server, &first)
            .read_bytes("/a")
            .await
            .expect("first read"),
        b"hi"
    );
    assert_eq!(
        scoped_backend(&server, &second)
            .read_bytes("/b")
            .await
            .expect("second read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);

    let report = teardown(&SandboxSelector::for_base_url(server.uri()))
        .await
        .expect("teardown");

    // The cache is process-global and other tests run concurrently, so only this
    // test's own entries are asserted.
    for isolation in [&first, &second] {
        let key = shared_key(&server.uri(), Some(isolation));
        assert!(report.released_keys.contains(&key), "{report:?}");
        assert!(
            !report.failed.iter().any(|failure| failure.key == key),
            "{report:?}"
        );
    }
    assert!(
        report.deleted.iter().any(|id| id == "env-base"),
        "{report:?}"
    );

    // Both scopes are drained.
    for isolation in [&first, &second] {
        assert_eq!(
            scoped_backend(&server, isolation)
                .read_bytes("/c")
                .await
                .expect("read"),
            b"hi"
        );
    }
    assert_eq!(create_count(&server).await, 4);
}

#[tokio::test]
async fn delete_all_releases_this_servers_entries() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    mount_create(&server, "env-all").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 200, "released").await;

    let isolation = isolation("all");
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/a")
            .await
            .expect("read"),
        b"hi"
    );

    let report = teardown(&SandboxSelector::all()).await.expect("teardown");
    let key = shared_key(&server.uri(), Some(&isolation));
    // Own entries only: `all()` drains other tests' scopes too, and asserting
    // global emptiness would race with them.
    assert!(report.released_keys.contains(&key), "{report:?}");
    assert!(
        !report.failed.iter().any(|failure| failure.key == key),
        "{report:?}"
    );
    assert!(
        report.deleted.iter().any(|id| id == "env-all"),
        "{report:?}"
    );

    // The entry is drained, so a fresh backend of the same scope creates again.
    assert_eq!(
        scoped_backend(&server, &isolation)
            .read_bytes("/b")
            .await
            .expect("read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);
}

#[tokio::test]
async fn a_scope_selector_leaves_another_isolation_key_cached() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    mount_create(&server, "env-scope").await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 200, "released").await;

    let released = isolation("scope_released");
    let kept = isolation("scope_kept");
    assert_eq!(
        scoped_backend(&server, &released)
            .read_bytes("/a")
            .await
            .expect("released read"),
        b"hi"
    );
    assert_eq!(
        scoped_backend(&server, &kept)
            .read_bytes("/b")
            .await
            .expect("kept read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&released)))
        .await
        .expect("teardown");
    assert_eq!(
        report.released_keys,
        vec![shared_key(&server.uri(), Some(&released))]
    );
    assert_eq!(report.deleted, vec!["env-scope".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);

    // The untouched scope still hits its cache entry: no third create.
    assert_eq!(
        scoped_backend(&server, &kept)
            .read_bytes("/c")
            .await
            .expect("kept read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 2);

    // The released scope has to create again.
    assert_eq!(
        scoped_backend(&server, &released)
            .read_bytes("/d")
            .await
            .expect("released read"),
        b"hi"
    );
    assert_eq!(create_count(&server).await, 3);
}

#[tokio::test]
async fn a_sandbox_id_narrows_a_base_url_selection_to_one_environment() {
    let _guard = exclusive_teardown().await;
    let server = MockServer::start().await;
    let creates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&creates);
    Mock::given(method("POST"))
        .and(path(CREATE_PATH))
        .respond_with(move |_request: &WireRequest| {
            let index = counter.fetch_add(1, Ordering::SeqCst);
            // Ids no other test uses: a base-URL selection reads the whole
            // process cache, and a reused loopback port can leave another
            // test's entry (with its own environment id) under this base URL.
            let environment_id = if index == 0 {
                "narrow-env-first"
            } else {
                "narrow-env-second"
            };
            ResponseTemplate::new(200)
                .set_body_json(json!({"data": {"environmentId": environment_id}}))
        })
        .mount(&server)
        .await;
    mount_invoke(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({"data": "6869"})),
    )
    .await;
    mount_delete(&server, 200, "released").await;

    let first = isolation("narrow_a");
    let second = isolation("narrow_b");
    assert_eq!(
        scoped_backend(&server, &first)
            .read_bytes("/a")
            .await
            .expect("first read"),
        b"hi"
    );
    assert_eq!(
        scoped_backend(&server, &second)
            .read_bytes("/b")
            .await
            .expect("second read"),
        b"hi"
    );
    assert_eq!(creates.load(Ordering::SeqCst), 2);

    let mut selector = SandboxSelector::for_base_url(server.uri());
    selector.sandbox_id = Some("narrow-env-second".to_string());
    selector.reason = "narrowed".to_string();
    let report = teardown(&selector).await.expect("teardown");

    assert!(report
        .released_keys
        .contains(&shared_key(&server.uri(), Some(&second))));
    assert!(!report
        .released_keys
        .contains(&shared_key(&server.uri(), Some(&first))));
    assert_eq!(report.deleted, vec!["narrow-env-second".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert_eq!(
        delete_paths(&server).await,
        vec!["/api/environment/v1/environments/narrow-env-second"]
    );

    // The other environment of the same base URL is untouched and still cached.
    assert_eq!(
        scoped_backend(&server, &first)
            .read_bytes("/c")
            .await
            .expect("first read"),
        b"hi"
    );
    assert_eq!(creates.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_sandbox_id_only_selector_is_unsupported() {
    // An ADX environment id carries no base URL, so there is no host to send the
    // `DELETE` to; Python's `delete_yuanrong_sandbox` requires a shared key or a
    // base URL too, and never reaches this path.
    let error = teardown(&SandboxSelector::from_sandbox_id("env-1"))
        .await
        .expect_err("unsupported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message().to_string();
    assert!(message.contains("base URL"), "{message}");
    assert!(message.contains("shared key"), "{message}");
}

#[tokio::test]
async fn an_empty_selector_is_a_validation_error() {
    let error = teardown(&SandboxSelector::default())
        .await
        .expect_err("validation");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("yuanrong"), "{}", error.message());
}
