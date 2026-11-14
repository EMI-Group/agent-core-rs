//! Teardown cases for the `jiuwenbox` provider against a loopback mock server.
//!
//! Python counterpart: `delete_jiuwenbox_sandbox` in
//! `tests/unit_tests/extensions/sys_operation/sandbox/providers/jiuwenbox.py`.
//! It pops the selected entries out of the process-wide `_shared_sandbox_ids`
//! cache and issues `DELETE /api/v1/sandboxes/{id}`, counting `404` as "already
//! gone" and logging — never raising — a release failure.
//!
//! The container cache is process-global, so every case here takes one lock:
//! [`SandboxSelector::all`] drains the whole cache, and a parallel case would
//! otherwise lose the entry it is about to release. Assertions therefore only
//! ever describe this case's own entries, never the whole cache.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
use serde_json::json;
use tokio::sync::Mutex;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::super::{teardown, JiuwenBoxBackend, SandboxBackend, SandboxSelector};
use super::{mount_create, mount_download, unique_isolation_key, NO_RECREATE, SANDBOX_ID};

/// Serialises the cases: [`SandboxSelector::all`] drains the whole process cache.
static TEARDOWN_CASES: Mutex<()> = Mutex::const_new(());

/// `/api/v1/sandboxes/{id}`: the container-delete path.
fn container_path(sandbox_id: &str) -> String {
    format!("/api/v1/sandboxes/{sandbox_id}")
}

/// The cache key a backend of `server` with `isolation_key` writes.
fn cache_key(server: &MockServer, isolation_key: &str) -> String {
    format!("{}|{isolation_key}", server.uri())
}

/// A backend bound to a known isolation key, so a selector can address its entry.
fn scoped_backend(
    server: &MockServer,
    isolation_key: &str,
    config: &SandboxGatewayConfig,
) -> JiuwenBoxBackend {
    let endpoint = SandboxEndpoint {
        base_url: server.uri(),
        sandbox_id: None,
        isolation_key: Some(isolation_key.to_string()),
    };
    JiuwenBoxBackend::new(&endpoint, config)
        .unwrap()
        .with_bearer_token(None)
        .with_recreate(NO_RECREATE.0, NO_RECREATE.1)
}

/// A card carrying one explicit `Authorization` header.
fn card_config() -> SandboxGatewayConfig {
    let mut config = SandboxGatewayConfig::default();
    config
        .auth_headers
        .insert("authorization".to_string(), "Bearer card-token".to_string());
    config
}

/// A download mock for one named container.
async fn mount_download_for(server: &MockServer, sandbox_id: &str, body: &'static [u8]) {
    Mock::given(method("GET"))
        .and(path(format!("{}/download", container_path(sandbox_id))))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body.to_vec()))
        .mount(server)
        .await;
}

/// A create mock that hands out `sandbox_id` exactly once.
async fn mount_create_once(server: &MockServer, sandbox_id: &str) {
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": sandbox_id })))
        .up_to_n_times(1)
        .mount(server)
        .await;
}

/// How many `POST /api/v1/sandboxes` creates the server saw.
async fn create_count(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| {
            request.method == reqwest::Method::POST && request.url.path() == "/api/v1/sandboxes"
        })
        .count()
}

/// The `DELETE` paths the server saw, in request order.
async fn delete_paths(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|request| request.method == reqwest::Method::DELETE)
        .map(|request| request.url.path().to_string())
        .collect()
}

// --- releasing one cached container ---------------------------------------

#[tokio::test]
async fn teardown_deletes_the_cached_container_and_drains_the_cache() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation = unique_isolation_key();
    mount_create(&server, SANDBOX_ID).await;
    mount_download(&server, b"x").await;
    let config = card_config();
    let backend = scoped_backend(&server, &isolation, &config);
    assert_eq!(backend.read_bytes("/a.txt").await.unwrap(), b"x");
    Mock::given(method("DELETE"))
        .and(path(container_path(SANDBOX_ID)))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&isolation)))
        .await
        .unwrap();

    assert_eq!(report.sandbox_type, "jiuwenbox");
    assert_eq!(report.released_keys, vec![cache_key(&server, &isolation)]);
    assert_eq!(report.deleted, vec![SANDBOX_ID.to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert!(report.unsupported.is_none());

    // The delete went to the container path with the card's auth header.
    let requests = server.received_requests().await.unwrap();
    let deletes: Vec<_> = requests
        .iter()
        .filter(|request| request.method == reqwest::Method::DELETE)
        .collect();
    assert_eq!(deletes.len(), 1, "{requests:?}");
    assert_eq!(deletes[0].url.path(), container_path(SANDBOX_ID));
    assert_eq!(
        deletes[0]
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer card-token"
    );

    // The cache entry is gone, so a fresh backend of this scope creates again.
    let fresh = scoped_backend(&server, &isolation, &config);
    assert_eq!(fresh.read_bytes("/b.txt").await.unwrap(), b"x");
    assert_eq!(
        create_count(&server).await,
        2,
        "a drained cache entry makes the scope create a new container"
    );
}

#[tokio::test]
async fn a_404_delete_counts_as_released() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation = unique_isolation_key();
    mount_create(&server, SANDBOX_ID).await;
    mount_download(&server, b"x").await;
    let config = SandboxGatewayConfig::default();
    scoped_backend(&server, &isolation, &config)
        .read_bytes("/a.txt")
        .await
        .unwrap();
    Mock::given(method("DELETE"))
        .and(path(container_path(SANDBOX_ID)))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_json(json!({ "error": "Sandbox 'sb-1' not found" })),
        )
        .mount(&server)
        .await;

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&isolation)))
        .await
        .unwrap();

    assert_eq!(report.released_keys, vec![cache_key(&server, &isolation)]);
    assert_eq!(report.deleted, vec![SANDBOX_ID.to_string()]);
    assert!(
        report.failed.is_empty(),
        "an already-gone container is a release, not a failure: {:?}",
        report.failed
    );
}

#[tokio::test]
async fn a_500_delete_is_recorded_without_panicking() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation = unique_isolation_key();
    mount_create(&server, SANDBOX_ID).await;
    mount_download(&server, b"x").await;
    let config = SandboxGatewayConfig::default();
    scoped_backend(&server, &isolation, &config)
        .read_bytes("/a.txt")
        .await
        .unwrap();
    Mock::given(method("DELETE"))
        .and(path(container_path(SANDBOX_ID)))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({ "error": "boom" })))
        .mount(&server)
        .await;

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&isolation)))
        .await
        .unwrap();

    assert_eq!(report.deleted, Vec::<String>::new());
    assert_eq!(report.failed.len(), 1, "{:?}", report.failed);
    let failure = &report.failed[0];
    assert_eq!(failure.key, cache_key(&server, &isolation));
    assert_eq!(failure.container_id, SANDBOX_ID);
    assert!(
        failure.reason.contains("500"),
        "the reason names the status: {}",
        failure.reason
    );

    // A failed release still drained the entry, so the next call creates again.
    let fresh = scoped_backend(&server, &isolation, &config);
    assert_eq!(fresh.read_bytes("/b.txt").await.unwrap(), b"x");
    assert_eq!(create_count(&server).await, 2);
}

// --- what a selector releases ---------------------------------------------

#[tokio::test]
async fn a_sandbox_id_selector_releases_the_matching_cache_value() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation = unique_isolation_key();
    // A distinct id keeps the cache-value lookup away from other cases' `sb-1`.
    let sandbox_id = "sb-by-id";
    mount_create(&server, sandbox_id).await;
    mount_download_for(&server, sandbox_id, b"x").await;
    let config = SandboxGatewayConfig::default();
    scoped_backend(&server, &isolation, &config)
        .read_bytes("/a.txt")
        .await
        .unwrap();
    Mock::given(method("DELETE"))
        .and(path(container_path(sandbox_id)))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;

    let report = teardown(&SandboxSelector::from_sandbox_id(sandbox_id))
        .await
        .unwrap();

    assert_eq!(report.released_keys, vec![cache_key(&server, &isolation)]);
    assert_eq!(report.deleted, vec![sandbox_id.to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert_eq!(
        delete_paths(&server).await,
        vec![container_path(sandbox_id)]
    );

    // The value lookup removed the entry, so this scope creates again.
    let fresh = scoped_backend(&server, &isolation, &config);
    assert_eq!(fresh.read_bytes("/b.txt").await.unwrap(), b"x");
    assert_eq!(create_count(&server).await, 2);
}

#[tokio::test]
async fn delete_all_releases_every_entry_built_here() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation_a = unique_isolation_key();
    let isolation_b = unique_isolation_key();
    mount_create_once(&server, "sb-all-a").await;
    mount_create_once(&server, "sb-all-b").await;
    mount_download_for(&server, "sb-all-a", b"a").await;
    mount_download_for(&server, "sb-all-b", b"b").await;
    // Other suites' entries may be drained too; answer them all benignly.
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let config = SandboxGatewayConfig::default();
    scoped_backend(&server, &isolation_a, &config)
        .read_bytes("/a.txt")
        .await
        .unwrap();
    scoped_backend(&server, &isolation_b, &config)
        .read_bytes("/b.txt")
        .await
        .unwrap();

    let report = teardown(&SandboxSelector::all()).await.unwrap();

    // The process cache also holds other cases' entries, so only ours are asserted.
    assert!(
        report
            .released_keys
            .contains(&cache_key(&server, &isolation_a)),
        "{:?}",
        report.released_keys
    );
    assert!(
        report
            .released_keys
            .contains(&cache_key(&server, &isolation_b)),
        "{:?}",
        report.released_keys
    );
    let deletes = delete_paths(&server).await;
    for sandbox_id in ["sb-all-a", "sb-all-b"] {
        assert!(
            report.deleted.iter().any(|released| released == sandbox_id),
            "{sandbox_id} was not released: {:?}",
            report.deleted
        );
        assert!(
            deletes.contains(&container_path(sandbox_id)),
            "{sandbox_id} was not deleted"
        );
    }
}

#[tokio::test]
async fn a_scope_selector_leaves_the_other_isolation_key_alone() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation_a = unique_isolation_key();
    let isolation_b = unique_isolation_key();
    // Three creates in order: scope A, scope B, then scope A again.
    mount_create_once(&server, "sb-scope-a").await;
    mount_create_once(&server, "sb-scope-b").await;
    mount_create(&server, "sb-scope-a2").await;
    mount_download_for(&server, "sb-scope-a", b"a").await;
    mount_download_for(&server, "sb-scope-b", b"b").await;
    mount_download_for(&server, "sb-scope-a2", b"a").await;
    Mock::given(method("DELETE"))
        .and(path(container_path("sb-scope-a")))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let config = SandboxGatewayConfig::default();
    assert_eq!(
        scoped_backend(&server, &isolation_a, &config)
            .read_bytes("/a.txt")
            .await
            .unwrap(),
        b"a"
    );
    assert_eq!(
        scoped_backend(&server, &isolation_b, &config)
            .read_bytes("/b.txt")
            .await
            .unwrap(),
        b"b"
    );

    let report = teardown(&SandboxSelector::scope(&server.uri(), Some(&isolation_a)))
        .await
        .unwrap();

    assert_eq!(report.released_keys, vec![cache_key(&server, &isolation_a)]);
    assert_eq!(report.deleted, vec!["sb-scope-a".to_string()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);

    // Scope B still resolves its cached container: no create, no delete.
    assert_eq!(
        scoped_backend(&server, &isolation_b, &config)
            .read_bytes("/b.txt")
            .await
            .unwrap(),
        b"b"
    );
    assert_eq!(
        create_count(&server).await,
        2,
        "the untouched scope must keep its cached container"
    );
    assert_eq!(
        delete_paths(&server).await,
        vec![container_path("sb-scope-a")]
    );

    // Scope A was drained, so it creates a new container.
    assert_eq!(
        scoped_backend(&server, &isolation_a, &config)
            .read_bytes("/a.txt")
            .await
            .unwrap(),
        b"a"
    );
    assert_eq!(create_count(&server).await, 3);
}

// --- selectors that name nothing or miss the cache ------------------------

#[tokio::test]
async fn a_default_selector_is_a_validation_error() {
    let _serial = TEARDOWN_CASES.lock().await;

    let error = teardown(&SandboxSelector::default()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("jiuwenbox"), "{}", error.message());
}

#[tokio::test]
async fn a_shared_key_miss_with_a_sandbox_id_deletes_through_the_key_base_url() {
    let _serial = TEARDOWN_CASES.lock().await;
    let server = MockServer::start().await;
    let isolation = unique_isolation_key();
    // Never created here, so neither the key nor the id is in the cache.
    let sandbox_id = format!("sb-fallback-{isolation}");
    Mock::given(method("DELETE"))
        .and(path(container_path(&sandbox_id)))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let key = cache_key(&server, &isolation);
    let selector = SandboxSelector {
        shared_key: Some(key.clone()),
        sandbox_id: Some(sandbox_id.clone()),
        ..SandboxSelector::default()
    };

    let report = teardown(&selector).await.unwrap();

    assert_eq!(report.released_keys, vec![key]);
    assert_eq!(report.deleted, vec![sandbox_id.clone()]);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert_eq!(
        delete_paths(&server).await,
        vec![container_path(&sandbox_id)]
    );
}
