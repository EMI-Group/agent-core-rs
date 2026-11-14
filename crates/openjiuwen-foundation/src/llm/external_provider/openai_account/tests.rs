//! Tests for the `OpenAIAccount` credential manager and model catalog.
//!
//! Pure parsing, cache round trips, and the live → cache → defaults fallback are
//! driven through [`MockServer`] on loopback and [`tempfile`] directories, so the
//! suite needs no network, no credential, and never touches `~/.openjiuwen`.
//! Ports `tests/unit_tests/core/foundation/llm/test_openai_account_models.py`
//! and the credential-state cases of the Python `openai_account_auth` tests.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use chrono::{DateTime, Utc};
use openjiuwen_common::config::env;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::llm::oauth::{
    FileTokenStore, InMemoryTokenStore, OAuthToken, OpenAiOAuthClient, TokenStore, AUTH_FILE_ENV,
};

use super::super::base::{ExternalProviderConfig, ModelCatalogRequest};
use super::{
    default_cache_path, parse_model_ids, OpenAiAccountAuthManager, OpenAiAccountModelCatalog,
    DEFAULT_MODELS, INVALID_STORE_CODE, MISSING_AUTH_CODE, MODELS_CACHE_FILENAME,
    MODELS_CACHE_FILE_ENV, PROVIDER_NAME,
};

/// Account path prefix the mock server stands in for.
const CODEX_PATH: &str = "/backend-api/codex";

/// Model-list path served by the account backend.
const MODELS_PATH: &str = "/backend-api/codex/models";

/// Returns an expiry `seconds` from now, truncated to whole seconds.
fn expiry_in(seconds: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(Utc::now().timestamp() + seconds, 0).expect("an in-range timestamp")
}

/// Returns the cache file a catalog stores under `directory`.
fn cache_path_in(directory: &tempfile::TempDir) -> PathBuf {
    directory.path().join(MODELS_CACHE_FILENAME)
}

/// Builds a catalog pointed at `server` with its cache inside `directory`.
///
/// The account host is `https://chatgpt.com`, so the mocked server carries the
/// `/backend-api/codex` prefix in its configured base URL; the clock is pinned so
/// the cache payload is deterministic.
fn catalog_at(server: &MockServer, directory: &tempfile::TempDir) -> OpenAiAccountModelCatalog {
    catalog_with_cache(cache_path_in(directory))
        .with_base_url(format!("{}{CODEX_PATH}", server.uri()))
}

/// Builds a catalog over an explicit cache file, with a harmless base URL.
fn catalog_with_cache(cache_path: PathBuf) -> OpenAiAccountModelCatalog {
    OpenAiAccountModelCatalog::with_config(
        &ExternalProviderConfig::new()
            .with_cache_path(cache_path)
            .with_base_url(format!("https://example.invalid{CODEX_PATH}")),
    )
    .expect("the catalog builds")
    .with_fixed_time(1000.0)
}

/// Builds a file-backed manager for `auth_path`.
fn manager_at(auth_path: &Path) -> OpenAiAccountAuthManager {
    OpenAiAccountAuthManager::with_config(&ExternalProviderConfig::new().with_auth_path(auth_path))
        .expect("the manager builds")
}

/// Builds an in-memory manager, pointing refreshes at `server` when given.
fn manager_over(
    store: Arc<InMemoryTokenStore>,
    server: Option<&MockServer>,
) -> OpenAiAccountAuthManager {
    let manager = OpenAiAccountAuthManager::with_store(store).expect("the manager builds");
    match server {
        Some(server) => manager.with_oauth_client(
            OpenAiOAuthClient::new()
                .expect("the oauth client builds")
                .with_base_url(server.uri()),
        ),
        None => manager,
    }
}

/// Stores an `access`/`refresh` token set expiring `seconds` from now.
async fn seed(manager: &OpenAiAccountAuthManager, seconds: i64) {
    manager
        .save_tokens(
            &OAuthToken::new("access")
                .with_refresh_token("refresh")
                .with_expires_at(expiry_in(seconds)),
        )
        .await
        .expect("the seeded token");
}

/// Mounts the model endpoint answering `body` with `status`.
async fn mount_models(server: &MockServer, status: u16, body: Value) {
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

/// Fetches with a throwaway token and returns the failure message.
async fn fetch_error(catalog: &OpenAiAccountModelCatalog) -> String {
    let error = catalog
        .fetch_models(&SecretString::new("access-token"))
        .await
        .expect_err("the discovery fails");
    assert_eq!(error.kind(), ErrorKind::Llm);
    error.to_string()
}

/// Mounts the token endpoint answering `body` exactly `hits` times.
async fn mount_refresh(server: &MockServer, body: Value, hits: u64) {
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(hits)
        .mount(server)
        .await;
}

// --- model-list parsing (pure) ---------------------------------------------

#[test]
fn parse_model_ids_keeps_api_disabled_models_and_drops_hidden_ones() {
    // ports test_parse_openai_account_model_ids_does_not_filter_supported_in_api
    let payload = json!({
        "models": [
            {"slug": "hidden-model", "visibility": "hide", "priority": 0},
            {"slug": "gpt-5.4", "priority": 20},
            {"slug": "api-disabled-visible", "supported_in_api": false, "priority": 10},
            {"slug": "gpt-5.3-codex-spark", "supported_in_api": false, "priority": 1},
        ]
    });

    let model_ids = parse_model_ids(&payload);

    assert_eq!(
        &model_ids[..3],
        ["gpt-5.3-codex-spark", "api-disabled-visible", "gpt-5.4"],
        "entries sort by priority, not by payload order: {model_ids:?}"
    );
    assert!(
        !model_ids.contains(&"hidden-model".to_owned()),
        "a hidden model never reaches the caller: {model_ids:?}"
    );
    assert!(
        model_ids.contains(&"gpt-5.5".to_owned()),
        "the forward-compatibility alias is appended: {model_ids:?}"
    );
}

#[test]
fn parse_model_ids_accepts_every_documented_entry_shape() {
    // `models` as an object keyed by id: the key is the `slug` fallback.
    assert_eq!(
        parse_model_ids(&json!({"models": {"m-alpha": {}, "m-beta": {"priority": 5}}})),
        ["m-beta", "m-alpha"],
        "an entry without metadata still carries the map key"
    );
    // `data` covers the deployments that never renamed the field, in both shapes.
    assert_eq!(
        parse_model_ids(&json!({"data": [{"id": "data-model"}]})),
        ["data-model"]
    );
    assert_eq!(
        parse_model_ids(&json!({"data": {"data-keyed": {}}})),
        ["data-keyed"]
    );
    // `models` wins when both are present.
    assert_eq!(
        parse_model_ids(&json!({
            "models": [{"slug": "from-models"}],
            "data": [{"slug": "from-data"}],
        })),
        ["from-models"]
    );
}

#[test]
fn parse_model_ids_prefers_slug_then_id_then_name_then_model() {
    let model_ids = parse_model_ids(&json!({"models": [
        {"slug": "slug-wins", "id": "id-loses", "name": "name-loses", "model": "model-loses"},
        {"id": "id-wins", "name": "name-loses", "model": "model-loses"},
        {"name": "name-wins", "model": "model-loses"},
        {"model": "model-wins"},
        {"slug": "   ", "id": "blank-slug-falls-through"},
    ]}));

    for expected in [
        "slug-wins",
        "id-wins",
        "name-wins",
        "model-wins",
        "blank-slug-falls-through",
    ] {
        assert!(
            model_ids.contains(&expected.to_owned()),
            "{expected} is missing from {model_ids:?}"
        );
    }
    assert_eq!(
        model_ids.len(),
        5,
        "one ID per entry, no synthetic additions: {model_ids:?}"
    );
}

#[test]
fn parse_model_ids_dedupes_skips_and_orders_by_priority() {
    let unusable = json!({"models": [
        1,
        "string-entry",
        null,
        {"slug": "dup", "priority": 1},
        {"slug": "dup", "priority": 2},
        {"slug": "visible", "visibility": "VISIBLE"},
        {"slug": "hidden-trimmed", "visibility": " HIDDEN "},
        {"slug": "hidden-lower", "visibility": "hide"},
    ]});
    assert_eq!(
        parse_model_ids(&unusable),
        ["dup", "visible"],
        "duplicates collapse, non-objects and hidden entries are skipped"
    );

    let priorities = json!({"models": [
        {"slug": "string-priority", "priority": "5"},
        {"slug": "boolean-priority", "priority": true},
        {"slug": "absent-priority"},
        {"slug": "float-priority", "priority": 2.9},
    ]});
    assert_eq!(
        parse_model_ids(&priorities),
        [
            "float-priority",
            "string-priority",
            "absent-priority",
            "boolean-priority",
        ],
        "a numeric string orders, a boolean falls back to the default priority"
    );
}

// --- live discovery ---------------------------------------------------------

#[tokio::test]
async fn list_model_ids_fetches_models_and_writes_the_cache() {
    // ports test_catalog_fetches_models_with_openai_account_token_and_writes_cache
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .and(query_param("client_version", "1.0.0"))
        .and(header("authorization", "Bearer access-token"))
        .and(header("accept", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {"slug": "gpt-5.4", "priority": 10},
                {"slug": "gpt-5.3-codex", "priority": 20},
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let model_ids = catalog_at(&server, &directory)
        .list_model_ids(&ModelCatalogRequest::new().with_access_token("access-token"))
        .await
        .expect("the live model list");

    assert_eq!(
        &model_ids[..2],
        ["gpt-5.4", "gpt-5.3-codex"],
        "the endpoint order is preserved as the priority order"
    );
    assert!(
        model_ids.contains(&"gpt-5.5".to_owned()),
        "the forward-compatibility alias is appended: {model_ids:?}"
    );

    let cache_path = cache_path_in(&directory);
    let cache: Value = serde_json::from_slice(
        &std::fs::read(&cache_path).expect("the live list writes the cache file"),
    )
    .expect("the cache file is JSON");
    assert_eq!(cache["provider"], json!(PROVIDER_NAME));
    assert_eq!(
        cache["base_url"],
        json!(format!("{}{CODEX_PATH}", server.uri())),
        "the cache records the base URL the answer came from"
    );
    assert_eq!(cache["client_version"], json!("1.0.0"));
    assert_eq!(
        cache["fetched_at"],
        json!(1000.0),
        "the pinned clock shows up"
    );
    let cached_ids = cache["model_ids"]
        .as_array()
        .expect("the cache stores the model IDs");
    assert_eq!(&cached_ids[..2], [json!("gpt-5.4"), json!("gpt-5.3-codex")]);
    assert!(cached_ids.contains(&json!("gpt-5.5")));
    assert_eq!(cache["payload"]["models"].as_array().map(Vec::len), Some(2));
    assert!(
        !cache_path
            .with_file_name(format!("{MODELS_CACHE_FILENAME}.tmp"))
            .exists(),
        "the atomic rename leaves no temporary file behind"
    );
}

#[tokio::test]
async fn fetch_models_maps_a_non_200_status_to_an_llm_error() {
    let server = MockServer::start().await;
    mount_models(&server, 500, json!({"error": "failed"})).await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let error = catalog_at(&server, &directory)
        .fetch_models(&SecretString::new("access-token"))
        .await
        .expect_err("a 500 response is a discovery failure");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(
        error.to_string().contains("status 500"),
        "the status is named: {error}"
    );
}

#[tokio::test]
async fn fetch_models_rejects_an_unusable_body() {
    let directory = tempfile::tempdir().expect("a temp directory");

    let invalid_json = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&invalid_json)
        .await;
    assert!(
        fetch_error(&catalog_at(&invalid_json, &directory))
            .await
            .contains("invalid JSON"),
        "the body is not JSON"
    );

    let array_body = MockServer::start().await;
    mount_models(&array_body, 200, json!([1, 2])).await;
    assert!(
        fetch_error(&catalog_at(&array_body, &directory))
            .await
            .contains("unsupported payload"),
        "a JSON array is not a payload"
    );

    let no_visible = MockServer::start().await;
    mount_models(
        &no_visible,
        200,
        json!({"models": [{"slug": "x", "visibility": "hide"}]}),
    )
    .await;
    assert!(
        fetch_error(&catalog_at(&no_visible, &directory))
            .await
            .contains("no visible models"),
        "every entry is hidden"
    );
}

#[tokio::test]
async fn fetch_models_refuses_an_empty_access_token_without_a_request() {
    let server = MockServer::start().await;
    let directory = tempfile::tempdir().expect("a temp directory");

    let error = catalog_at(&server, &directory)
        .fetch_models(&SecretString::new(""))
        .await
        .expect_err("an empty bearer token cannot authenticate");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(
        error.to_string().contains("requires an access token"),
        "message: {error}"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "the endpoint is never called without a credential"
    );
}

// --- fallbacks --------------------------------------------------------------

#[tokio::test]
async fn a_live_failure_falls_back_to_the_cache() {
    // ports test_catalog_uses_cache_when_live_request_fails
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "failed"})))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempfile::tempdir().expect("a temp directory");
    std::fs::write(
        cache_path_in(&directory),
        br#"{"model_ids": ["cached-model"]}"#,
    )
    .expect("the seeded cache");

    let model_ids = catalog_at(&server, &directory)
        .list_model_ids(&ModelCatalogRequest::new().with_access_token("access-token"))
        .await
        .expect("the cache answers when the endpoint fails");

    assert_eq!(model_ids, ["cached-model"]);
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        1,
        "the live attempt is made exactly once"
    );
}

#[tokio::test]
async fn without_a_token_or_a_cache_the_built_in_defaults_answer() {
    // ports test_catalog_uses_fallback_without_token_or_cache
    let server = MockServer::start().await;
    mount_models(&server, 200, json!({"models": [{"slug": "live"}]})).await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let model_ids = catalog_at(&server, &directory)
        .list_model_ids(&ModelCatalogRequest::new())
        .await
        .expect("the defaults always answer");

    assert_eq!(
        model_ids,
        DEFAULT_MODELS.map(str::to_owned).to_vec(),
        "no credential and no cache means the built-in list"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "an undiscoverable list is never requested"
    );
}

#[tokio::test]
async fn write_cache_and_read_cached_round_trip() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let catalog = catalog_with_cache(cache_path_in(&directory));

    catalog
        .write_cache(
            &json!({"models": [{"slug": "cached-model"}]}),
            &["cached-model".to_owned()],
        )
        .await
        .expect("the cache write");

    assert_eq!(
        catalog.read_cached().await,
        Some(vec!["cached-model".to_owned()]),
        "the written cache reads back"
    );
    assert_eq!(
        catalog_with_cache(directory.path().join("absent.json"))
            .read_cached()
            .await,
        None,
        "an absent cache is `None`"
    );

    let corrupt_path = directory.path().join("corrupt.json");
    std::fs::write(&corrupt_path, b"not json").expect("the corrupt cache");
    assert_eq!(
        catalog_with_cache(corrupt_path).read_cached().await,
        None,
        "a corrupt cache degrades instead of failing"
    );
}

// --- credential manager -----------------------------------------------------

#[tokio::test]
async fn a_valid_token_is_reported_and_resolved_without_a_refresh() {
    let manager = manager_over(Arc::new(InMemoryTokenStore::new()), None);
    seed(&manager, 3600).await;

    let status = manager.status().await.expect("the status");
    assert!(status.authenticated);
    assert!(status.has_refresh_token);
    assert!(!status.needs_refresh, "an hour of validity is enough");
    assert_eq!(status.auth_path, None, "a custom store has no file path");
    assert_eq!(status.error, None);
    assert!(status.expires_at.is_some());

    // The manager keeps the default OAuth client, which would have to reach the
    // real issuer: answering at all proves no refresh round trip happens.
    assert_eq!(
        manager
            .resolve_access_token(false)
            .await
            .expect("the stored token")
            .expose_secret(),
        "access"
    );
}

#[tokio::test]
async fn an_expiring_token_is_refreshed_and_persisted() {
    let server = MockServer::start().await;
    mount_refresh(
        &server,
        json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 600}),
        1,
    )
    .await;

    let store = Arc::new(InMemoryTokenStore::new());
    let manager = manager_over(store.clone(), Some(&server));
    // A 60 s lifetime sits inside the 120 s default skew.
    seed(&manager, 60).await;

    let token = manager
        .resolve_access_token(false)
        .await
        .expect("the refreshed token");
    assert_eq!(token.expose_secret(), "new-access");

    let stored = store
        .load()
        .await
        .expect("the store read")
        .expect("the refreshed token is persisted");
    assert_eq!(stored.access_token.expose_secret(), "new-access");
    assert_eq!(
        stored
            .refresh_token
            .as_ref()
            .map(SecretString::expose_secret),
        Some("new-refresh"),
        "a rotated refresh token is kept"
    );
}

#[tokio::test]
async fn a_forced_refresh_replaces_a_still_valid_token() {
    let server = MockServer::start().await;
    mount_refresh(
        &server,
        json!({"access_token": "forced-access", "refresh_token": "forced", "expires_in": 600}),
        1,
    )
    .await;

    let manager = manager_over(Arc::new(InMemoryTokenStore::new()), Some(&server));
    seed(&manager, 3600).await;

    let token = manager
        .resolve_access_token(true)
        .await
        .expect("the forced refresh");
    assert_eq!(token.expose_secret(), "forced-access");
}

#[tokio::test]
async fn logout_removes_only_the_openai_account_entry() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let manager = manager_at(&auth_path);
    seed(&manager, 3600).await;

    // Another provider shares the auth file; logout must not touch it.
    let mut auth_file: Value =
        serde_json::from_slice(&std::fs::read(&auth_path).expect("the auth file"))
            .expect("the auth file is JSON");
    let other = json!({"tokens": {"access_token": "other-token"}, "auth_mode": "other"});
    auth_file["providers"]["other"] = other.clone();
    std::fs::write(
        &auth_path,
        serde_json::to_vec_pretty(&auth_file).expect("serializing the auth file"),
    )
    .expect("rewriting the auth file");

    assert!(
        manager.logout().await.expect("the logout"),
        "a token was stored"
    );
    assert!(
        !manager.logout().await.expect("the second logout"),
        "nothing is left to remove"
    );

    let after: Value = serde_json::from_slice(
        &std::fs::read(&auth_path).expect("the auth file survives the logout"),
    )
    .expect("the auth file is JSON");
    assert_eq!(
        after["providers"]["other"], other,
        "another provider's entry survives"
    );
    assert!(
        after["providers"].get(PROVIDER_NAME).is_none(),
        "the account entry is gone: {after}"
    );
    assert!(
        FileTokenStore::new(&auth_path)
            .load()
            .await
            .expect("the store read")
            .is_none(),
        "the store reports no credential after the logout"
    );
}

#[tokio::test]
async fn status_reports_the_credential_state_of_the_auth_file() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");

    let status = manager_at(&auth_path).status().await.expect("the status");
    assert!(!status.authenticated, "the file does not exist yet");
    assert_eq!(status.error.as_deref(), Some(MISSING_AUTH_CODE));
    assert_eq!(status.auth_path.as_deref(), Some(auth_path.as_path()));
    assert!(!status.has_refresh_token);
    assert!(!status.needs_refresh);

    std::fs::write(&auth_path, b"not json").expect("the corrupt store");
    let status = manager_at(&auth_path).status().await.expect("the status");
    assert!(!status.authenticated);
    assert_eq!(status.error.as_deref(), Some(INVALID_STORE_CODE));

    // A fresh store: `save` merges into the existing document, so a corrupt file
    // would reject the seed instead of exercising the skew check.
    let seeded_directory = tempfile::tempdir().expect("a temp directory");
    let manager = manager_at(&seeded_directory.path().join("auth.json"));
    seed(&manager, 60).await;
    let status = manager.status().await.expect("the status");
    assert!(status.authenticated);
    assert!(
        status.needs_refresh,
        "60 s of validity is inside the 120 s skew"
    );
    assert_eq!(status.error, None);
}

#[tokio::test]
async fn the_catalog_resolves_its_token_through_the_auth_manager() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(MODELS_PATH))
        .and(header("authorization", "Bearer access"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"slug": "gpt-5.4", "priority": 10}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let manager = manager_over(Arc::new(InMemoryTokenStore::new()), Some(&server));
    seed(&manager, 3600).await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let model_ids = catalog_at(&server, &directory)
        .list_model_ids(&ModelCatalogRequest::new().with_auth_manager(Arc::new(manager)))
        .await
        .expect("the live list through the manager");

    assert_eq!(
        model_ids[0], "gpt-5.4",
        "the live list is returned: {model_ids:?}"
    );
}

// --- path resolution --------------------------------------------------------

/// Serializes tests that mutate the process environment.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Every variable [`default_cache_path`] may read.
const ENV_VARS: &[&str] = &[MODELS_CACHE_FILE_ENV, AUTH_FILE_ENV, env::OPENJIUWEN_HOME];

/// Clears every variable the cache-path resolver reads and restores it on drop.
struct EnvGuard {
    /// Keeps the mutation exclusive for the guard's lifetime.
    _lock: MutexGuard<'static, ()>,
    /// The values to restore.
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    /// Clears the resolver's inputs.
    fn clear() -> Self {
        let lock = env_lock();
        let saved: Vec<(&'static str, Option<String>)> = ENV_VARS
            .iter()
            .map(|name| (*name, std::env::var(name).ok()))
            .collect();
        for name in ENV_VARS {
            std::env::remove_var(name);
        }
        Self { _lock: lock, saved }
    }

    /// Sets `name` for the guard's lifetime.
    #[allow(clippy::unused_self)]
    fn set(&self, name: &str, value: &str) {
        std::env::set_var(name, value);
    }

    /// Removes `name`.
    #[allow(clippy::unused_self)]
    fn remove(&self, name: &str) {
        std::env::remove_var(name);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

#[test]
fn default_cache_path_follows_the_environment() {
    // ports test_default_openai_account_models_cache_path_uses_openjiuwen_home
    let guard = EnvGuard::clear();
    let directory = tempfile::tempdir().expect("a temp directory");
    let home = directory.path().join("home");

    let override_path = directory.path().join("models-cache.json");
    guard.set(MODELS_CACHE_FILE_ENV, &override_path.to_string_lossy());
    guard.set(env::OPENJIUWEN_HOME, &home.to_string_lossy());
    assert_eq!(
        default_cache_path().expect("the override wins"),
        override_path,
        "the explicit cache file is used verbatim"
    );

    guard.remove(MODELS_CACHE_FILE_ENV);
    assert_eq!(
        default_cache_path().expect("the openJiuwen home"),
        home.join(MODELS_CACHE_FILENAME),
        "the cache defaults to the auth-store directory"
    );
}
