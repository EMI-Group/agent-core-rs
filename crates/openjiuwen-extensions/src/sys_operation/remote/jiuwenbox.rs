//! The `jiuwenbox` remote sandbox backend.
//!
//! Python parity: `extensions/sys_operation/sandbox/providers/jiuwenbox.py`
//! (v0.1.20) — the filesystem, shell, and code operations of a `JiuwenBox`
//! container, driven through the container's native REST API under
//! `/api/v1/...` instead of the kernel gateway envelope that `aio` speaks.
//!
//! # Endpoint and credentials
//!
//! [`SandboxHttp::new`] takes the base URL from the endpoint and falls back to
//! `launcher_config.base_url`; a bare `host:port` gains `http://` and trailing
//! slashes are stripped, so `127.0.0.1:8080/` and `http://127.0.0.1:8080`
//! address the same container. The shared transport injects the card's
//! `auth_headers` and `auth_query_params` on every request and adds
//! `Authorization: Bearer <token>` from `JIUWENBOX_API_TOKEN` when that variable
//! holds a non-blank token (a configured `Authorization` header wins).
//!
//! # Container lifecycle
//!
//! The kernel has no launcher or lifecycle API — it resolves a
//! [`SandboxEndpoint`] and calls the factories — so the container is created
//! lazily and cached. [`JiuwenBoxBackend::sandbox_id`] resolves, in order:
//! `JIUWENBOX_SANDBOX_ID`, `launcher.extra_params["sandbox_id"]`,
//! `endpoint.sandbox_id` as captured at construction, the process-global cache
//! entry for `base_url|isolation_key`, and finally `POST /api/v1/sandboxes`.
//! Resolution runs under one per-backend lock, so concurrent first calls issue a
//! single create, and the resulting id is published to the cache that Python
//! keeps in `_shared_sandbox_ids` (fs, shell, and code handles of one scope
//! share one container).
//!
//! Before the first create of a base URL the server idle policy is written once
//! through `PUT /api/v1/timeout` (`launcher.idle_ttl_seconds` plus
//! `extra_params["idle_check_interval"]`); the request is skipped when neither is
//! configured, deduplicated per base URL like Python's `_idle_timeout_cache`, and
//! failures are logged at `debug` instead of failing the create.
//!
//! # Recreate on a lost container
//!
//! Every request runs in a bounded retry loop (`JIUWENBOX_SANDBOX_RECREATE_RETRIES`,
//! default `3`, `0` disables it). A 404 whose body reads `Sandbox '<id>' not
//! found` is [`RemoteError::ContainerGone`]: the cached id is dropped, the stale
//! container is deleted best-effort (`404` tolerated), the backend waits one
//! second, and the next attempt creates a fresh container. Every other failure —
//! including a plain file-level 404, which stays [`RemoteError::NotFound`] —
//! returns immediately.
//!
//! # Deliberately not ported
//!
//! Python's hybrid local/remote shell routing (`plan_command_rewrite`,
//! tree-sitter leaf classification, `exclude_commands`), the host fallback of
//! `_run_exec_pipeline` (`fallback_on_failure`, `_run_local_subprocess`),
//! lifecycle hooks (`before_create` / `after_create` / `before_recreate` /
//! `before_delete` / `after_delete`), the `preserve_files_upload` re-upload after
//! a recreate, and Unix-domain-socket transport (`unix:///...`,
//! `JIUWENBOX_LISTEN`).
//!
//! # Teardown
//!
//! [`teardown`] ports `delete_jiuwenbox_sandbox`: it pops the selected cache
//! entries and issues `DELETE /api/v1/sandboxes/{id}` (`2xx` and `404` are both
//! success). The kernel never calls a provider on release, so a host drives the
//! teardown through `crate::sys_operation::shutdown_remote`; without it only the
//! server's idle reaper collects the container.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::sys_operation::sandbox::config::{
    SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio::sync::Mutex;

use super::backend::{
    provider_set_from, ExecCommand, ExecRequest, ExecResult, ListRequest, RemoteEntry,
    SandboxBackend, SearchRequest,
};
use super::semantics::basename;
use super::teardown::{
    require_selection, send_delete, shared_key, SandboxSelector, TeardownReport,
};
use super::transport::{base64_encode, RawResponse, RemoteError, SandboxHttp};
use crate::sys_operation::SandboxProviderSet;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;

/// The sandbox type this backend serves.
const PROVIDER: &str = "jiuwenbox";

/// Log target shared with the rest of the remote sandbox subtree.
const LOG_TARGET: &str = "openjiuwen_extensions::sys_operation";

/// Bearer token environment variable (Python `_ENV_API_TOKEN`).
const API_TOKEN_ENV: &str = "JIUWENBOX_API_TOKEN";

/// Pre-provisioned sandbox id environment variable.
const SANDBOX_ID_ENV: &str = "JIUWENBOX_SANDBOX_ID";

/// Recreate-attempt environment variable (Python `JIUWENBOX_SANDBOX_RECREATE_RETRIES`).
const RECREATE_RETRIES_ENV: &str = "JIUWENBOX_SANDBOX_RECREATE_RETRIES";

/// Recreate attempts when the environment says nothing.
const DEFAULT_RECREATE_RETRIES: u32 = 3;

/// Wait between a lost container and the recreate (Python `_SANDBOX_RECREATE_RETRY_SLEEP_SECONDS`).
const DEFAULT_RECREATE_DELAY: Duration = Duration::from_secs(1);

/// Request timeout floor for `exec`, matching Python's `max(timeout or 30, 30)`.
const DEFAULT_EXEC_TIMEOUT_SECONDS: u64 = 30;

/// Exit code the container reports when it killed a command on timeout.
const EXIT_CODE_TIMEOUT: i32 = 124;

/// Parent-directory-creating append used when `write_bytes(append = true)`.
const APPEND_SCRIPT: &str = "set -euo pipefail; target=\"$1\"; parent=$(dirname -- \"$target\"); mkdir -p -- \"$parent\"; base64 -d >> \"$target\"";

/// The filesystem, shell, and code factories of the `jiuwenbox` sandbox type.
#[must_use]
pub(crate) fn provider_set() -> SandboxProviderSet {
    provider_set_from(JiuwenBoxBackend::new)
}

/// One cache entry a teardown popped.
struct PoppedSandbox {
    /// Cache key the entry lived under.
    key: String,
    /// Container id to release.
    sandbox_id: String,
    /// Client bound to the entry's base URL and auth.
    http: SandboxHttp,
}

/// Releases cached `JiuwenBox` containers.
///
/// Python parity: `delete_jiuwenbox_sandbox` in
/// `extensions/sys_operation/sandbox/providers/jiuwenbox.py`. The selector is
/// applied in Python's precedence order — `delete_all` (drain the whole process
/// cache), then the shared key, then the base URL — and `sandbox_id` selects by
/// container identity, falling back to the base URL of a given shared key when
/// the cache no longer holds that scope.
///
/// Entries leave the cache before the delete request, so a failed delete cannot
/// resurrect a stale entry; `2xx` and `404` both count as released and every
/// other outcome lands in [`TeardownReport::failed`].
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// when the selector names nothing to release.
pub(crate) async fn teardown(selector: &SandboxSelector) -> Result<TeardownReport> {
    require_selection(PROVIDER, selector)?;
    let mut report = TeardownReport::new(PROVIDER);
    for entry in pop_cached(selector, &mut report).await {
        let request = entry
            .http
            .request(reqwest::Method::DELETE, &sandbox_url(&entry.sandbox_id));
        match send_delete(&entry.http, request, "delete_sandbox").await {
            Ok(()) => {
                tracing::info!(
                    target: LOG_TARGET,
                    provider = PROVIDER,
                    key = %entry.key,
                    reason = %selector.reason,
                    "jiuwenbox sandbox released"
                );
                report.record_released(entry.key, entry.sandbox_id);
            }
            Err(reason) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    provider = PROVIDER,
                    key = %entry.key,
                    reason = %reason,
                    "jiuwenbox sandbox release failed"
                );
                report.record_failure(entry.key, entry.sandbox_id, reason);
            }
        }
    }
    Ok(report)
}

/// Pops the cache entries `selector` names, in key order.
///
/// `sandbox_id` additionally matches a cache value; when a shared key is given
/// but its entry is gone, the id is released through a client rebuilt from that
/// key's base URL (Python's fallback, including the environment bearer token).
/// A client that cannot be rebuilt there is recorded in `report` as a failed
/// release instead of being dropped: the caller asked for that container to be
/// deleted, so an empty report would claim a release that never happened.
async fn pop_cached(selector: &SandboxSelector, report: &mut TeardownReport) -> Vec<PoppedSandbox> {
    let mut popped = Vec::new();
    {
        let mut shared = shared_sandbox_ids().lock().await;
        let keys: Vec<String> = shared
            .keys()
            .filter(|key| selector.matches_key(key))
            .cloned()
            .collect();
        for key in keys {
            if let Some(entry) = shared.remove(&key) {
                popped.push(PoppedSandbox {
                    key,
                    sandbox_id: entry.sandbox_id,
                    http: entry.http,
                });
            }
        }
        if let Some(id) = non_blank(selector.sandbox_id.as_deref()) {
            if !popped.iter().any(|entry| entry.sandbox_id == id) {
                let matching = shared
                    .iter()
                    .find(|(_, entry)| entry.sandbox_id == id)
                    .map(|(key, _)| key.clone());
                if let Some(key) = matching {
                    if let Some(entry) = shared.remove(&key) {
                        popped.push(PoppedSandbox {
                            key,
                            sandbox_id: entry.sandbox_id,
                            http: entry.http,
                        });
                    }
                }
            }
        }
    }
    if popped.is_empty() {
        if let (Some(key), Some(id)) = (
            selector.shared_key.as_deref(),
            non_blank(selector.sandbox_id.as_deref()),
        ) {
            match client_for_base_url(key.split('|').next().unwrap_or_default()) {
                Ok(http) => popped.push(PoppedSandbox {
                    key: key.to_string(),
                    sandbox_id: id,
                    http,
                }),
                Err(error) => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        provider = PROVIDER,
                        key = %key,
                        %error,
                        "jiuwenbox sandbox release could not build a client for the uncached id"
                    );
                    report.record_failure(
                        key.to_string(),
                        id,
                        format!("could not build the delete client: {error}"),
                    );
                }
            }
        }
    }
    popped
}

/// Rebuilds a client from a cache key's base URL, for a container no longer cached.
///
/// The card's auth headers are not part of the cache key, so such a delete
/// carries only the bearer token from `JIUWENBOX_API_TOKEN` (Python behaves the
/// same way: it builds the client from the base URL alone).
fn client_for_base_url(base_url: &str) -> Result<SandboxHttp> {
    let endpoint = SandboxEndpoint {
        base_url: base_url.to_string(),
        ..SandboxEndpoint::default()
    };
    SandboxHttp::new(PROVIDER, &endpoint, &SandboxGatewayConfig::default())
        .map(|http| http.with_bearer_token(api_token_from_env()))
        .map_err(|error| Error::config(error.message().to_string()))
}

/// The idle policy a base URL was configured with: `(idle_timeout, idle_check_interval)`.
type IdleTimeout = (Option<u64>, Option<u64>);

/// Base URL to the idle policy already written to that server.
type IdleTimeoutCache = Mutex<BTreeMap<String, IdleTimeout>>;

/// One cached container: its id plus the client that created it.
///
/// The client is cached with the id so a teardown can issue
/// `DELETE /api/v1/sandboxes/{id}` with the same base URL, auth headers, and
/// bearer token the create used (Python rebuilds a client from the base URL and
/// the environment token, which loses the card's auth headers).
#[derive(Clone)]
struct CachedSandbox {
    /// The container id the server assigned.
    sandbox_id: String,
    /// Client bound to the container's base URL and auth.
    http: SandboxHttp,
}

/// Containers shared by every backend with the same `base_url|isolation_key`
/// (Python's process-global `_shared_sandbox_ids`).
fn shared_sandbox_ids() -> &'static Mutex<BTreeMap<String, CachedSandbox>> {
    static SHARED: OnceLock<Mutex<BTreeMap<String, CachedSandbox>>> = OnceLock::new();
    SHARED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Base URLs whose idle policy was already written, with the pair that was written
/// (Python's process-global `_idle_timeout_cache`).
fn idle_timeouts() -> &'static IdleTimeoutCache {
    static WRITTEN: OnceLock<IdleTimeoutCache> = OnceLock::new();
    WRITTEN.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The `jiuwenbox` wire backend behind the generic remote operation wrappers.
pub(crate) struct JiuwenBoxBackend {
    /// HTTP client with the endpoint, auth headers, and timeout resolved.
    http: SandboxHttp,
    /// Launcher configuration: create options, idle policy, pre-provisioned id.
    launcher: Option<SandboxLauncherConfig>,
    /// `endpoint.sandbox_id` as captured at construction.
    endpoint_id: Option<String>,
    /// Isolation key this backend serves; half of the shared cache key.
    isolation_key: Option<String>,
    /// The id this backend resolved or created, guarded so one lock yields one create.
    id: Mutex<Option<String>>,
    /// Recreate attempts after a lost container (`0` disables recreating).
    recreate_retries: u32,
    /// Wait between a lost container and the recreate.
    recreate_delay: Duration,
}

impl JiuwenBoxBackend {
    /// Builds the backend for one endpoint, reading the bearer token and the
    /// recreate budget from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config)
    /// when neither the endpoint nor the launcher configuration carries a base
    /// URL, or when the HTTP client cannot be built.
    pub(crate) fn new(endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig) -> Result<Self> {
        let http =
            SandboxHttp::new(PROVIDER, endpoint, config)?.with_bearer_token(api_token_from_env());
        Ok(Self {
            http,
            launcher: config.launcher_config.clone(),
            endpoint_id: non_blank(endpoint.sandbox_id.as_deref()),
            isolation_key: non_blank(endpoint.isolation_key.as_deref()),
            id: Mutex::new(None),
            recreate_retries: recreate_retries_from_env(),
            recreate_delay: DEFAULT_RECREATE_DELAY,
        })
    }

    /// Overrides the bearer token, so tests do not depend on the process environment.
    #[cfg(test)]
    pub(crate) fn with_bearer_token(mut self, token: Option<String>) -> Self {
        self.http = self.http.with_bearer_token(token);
        self
    }

    /// Overrides the recreate budget and delay, so tests do not sleep.
    #[cfg(test)]
    pub(crate) fn with_recreate(mut self, retries: u32, delay: Duration) -> Self {
        self.recreate_retries = retries;
        self.recreate_delay = delay;
        self
    }

    /// One launcher `extra_params` entry, re-read on every call so a hot-updated
    /// launcher configuration takes effect (Python parity).
    fn extra_param(&self, key: &str) -> Option<&Value> {
        self.launcher.as_ref()?.extra_params.get(key)
    }

    /// `JIUWENBOX_SANDBOX_ID`, then `extra_params["sandbox_id"]`, then
    /// `endpoint.sandbox_id`: the ids a caller pre-provisioned for this backend.
    fn configured_sandbox_id(&self) -> Option<String> {
        non_blank(std::env::var(SANDBOX_ID_ENV).ok().as_deref())
            .or_else(|| non_blank(self.extra_param("sandbox_id").and_then(Value::as_str)))
            .or_else(|| self.endpoint_id.clone())
    }

    /// The shared cache key: one container per `base_url|isolation_key`.
    fn scope_key(&self) -> String {
        shared_key(self.http.base_url(), self.isolation_key.as_deref())
    }

    /// The container id, creating a sandbox when none is known.
    ///
    /// Resolution follows Python's `_get_sandbox_id`: environment, launcher
    /// `extra_params`, endpoint id, the shared cache, then a create. The whole
    /// resolution runs under [`Self::id`], so concurrent first calls create once.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::ContainerGone`] or [`RemoteError::Status`] /
    /// [`RemoteError::Transport`] when the create fails, and
    /// [`RemoteError::Transport`] when the create response carries no usable id.
    async fn sandbox_id(&self) -> std::result::Result<String, RemoteError> {
        let mut current = self.id.lock().await;
        if let Some(id) = self.configured_sandbox_id() {
            *current = Some(id.clone());
            shared_sandbox_ids().lock().await.insert(
                self.scope_key(),
                CachedSandbox {
                    sandbox_id: id.clone(),
                    http: self.http.clone(),
                },
            );
            return Ok(id);
        }
        let scope_key = self.scope_key();
        let cached = shared_sandbox_ids()
            .lock()
            .await
            .get(&scope_key)
            .map(|entry| entry.sandbox_id.clone());
        if let Some(id) = cached {
            *current = Some(id.clone());
            return Ok(id);
        }
        if let Some(id) = current.clone() {
            return Ok(id);
        }
        let id = self.create().await?;
        *current = Some(id.clone());
        shared_sandbox_ids().lock().await.insert(
            scope_key,
            CachedSandbox {
                sandbox_id: id.clone(),
                http: self.http.clone(),
            },
        );
        Ok(id)
    }

    /// The body of `POST /api/v1/sandboxes`, built from the launcher
    /// `extra_params` (Python `_sandbox_create_options_from_launcher_extra_params`).
    fn create_body(&self) -> Value {
        let mut body = Map::new();
        if let Some(policy) = self.extra_param("policy") {
            body.insert("policy".to_string(), policy.clone());
        }
        if let Some(mode) = self
            .extra_param("policy_mode")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|mode| !mode.is_empty())
        {
            body.insert("policy_mode".to_string(), Value::String(mode.to_string()));
        }
        // Swarm maps `type=jiuwenbox-conch` onto `api_sandbox_runtime=conch`.
        let runtime = self
            .extra_param("api_sandbox_runtime")
            .filter(|value| is_truthy(value))
            .or_else(|| self.extra_param("sandbox_runtime"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|runtime| !runtime.is_empty());
        if let Some(runtime) = runtime {
            body.insert(
                "sandbox_runtime".to_string(),
                Value::String(runtime.to_string()),
            );
        }
        Value::Object(body)
    }

    /// `POST /api/v1/sandboxes`; the idle policy is written once per base URL first.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::ContainerGone`] / [`RemoteError::Status`] /
    /// [`RemoteError::Transport`] when the create fails, and
    /// [`RemoteError::Transport`] when the answer carries no sandbox id.
    async fn create(&self) -> std::result::Result<String, RemoteError> {
        self.configure_idle_timeout().await;
        let body = self.create_body();
        let body = &body;
        let request = self
            .http
            .request(reqwest::Method::POST, "/api/v1/sandboxes")
            .json(body);
        let response = self.http.send_raw(request, "create_sandbox").await?;
        ensure_success(&response, "create_sandbox")?;
        let payload = decode_json(&response, "create_sandbox")?;
        let id = payload
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                RemoteError::Transport(
                    "create_sandbox: jiuwenbox returned no sandbox id".to_string(),
                )
            })?;
        Ok(id.to_string())
    }

    /// Writes the idle reaper settings once per base URL; failures are logged only
    /// (Python `_configure_server_idle_timeout`).
    async fn configure_idle_timeout(&self) {
        let idle_timeout = self
            .launcher
            .as_ref()
            .and_then(|launcher| launcher.idle_ttl_seconds);
        let idle_check_interval = self
            .extra_param("idle_check_interval")
            .and_then(number_as_u64);
        if idle_timeout.is_none() && idle_check_interval.is_none() {
            return;
        }
        let cache_key = self.http.base_url().to_string();
        let target: IdleTimeout = (idle_timeout, idle_check_interval);
        if idle_timeouts().lock().await.get(&cache_key) == Some(&target) {
            return;
        }
        let mut body = Map::new();
        if let Some(idle_timeout) = idle_timeout {
            body.insert("idle_timeout".to_string(), Value::from(idle_timeout));
        }
        if let Some(idle_check_interval) = idle_check_interval {
            body.insert(
                "idle_check_interval".to_string(),
                Value::from(idle_check_interval),
            );
        }
        let body = Value::Object(body);
        let request = self
            .http
            .request(reqwest::Method::PUT, "/api/v1/timeout")
            .json(&body);
        match self.http.send_raw(request, "set_idle_timeout").await {
            Ok(response) if response.is_success() => {
                idle_timeouts().lock().await.insert(cache_key, target);
            }
            Ok(response) => tracing::debug!(
                target: LOG_TARGET,
                provider = PROVIDER,
                status = response.status,
                "jiuwenbox idle-timeout update failed"
            ),
            Err(error) => tracing::debug!(
                target: LOG_TARGET,
                provider = PROVIDER,
                "jiuwenbox idle-timeout update failed: {}",
                error.message()
            ),
        }
    }

    /// Runs `op` against the current container id, recreating a lost container.
    ///
    /// # Errors
    ///
    /// Returns the first failure that is not [`RemoteError::ContainerGone`]
    /// unchanged, and the last container-gone error once the recreate budget is
    /// exhausted.
    async fn with_sandbox<T, F, Fut>(&self, mut op: F) -> std::result::Result<T, RemoteError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = std::result::Result<T, RemoteError>> + Send,
    {
        let attempts = self.recreate_retries.saturating_add(1);
        let mut last_error = None;
        for attempt in 1..=attempts {
            let sandbox_id = self.sandbox_id().await?;
            match op(sandbox_id.clone()).await {
                Ok(value) => return Ok(value),
                Err(error) if error.is_container_gone() => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        provider = PROVIDER,
                        attempt,
                        attempts,
                        "jiuwenbox sandbox is gone; recreating it"
                    );
                    self.discard_sandbox(&sandbox_id).await;
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
            if attempt < attempts {
                tokio::time::sleep(self.recreate_delay).await;
            }
        }
        Err(last_error
            .unwrap_or_else(|| RemoteError::Transport("jiuwenbox operation failed".to_string())))
    }

    /// Forgets a container the server reported gone and deletes it best-effort.
    async fn discard_sandbox(&self, stale: &str) {
        {
            let mut current = self.id.lock().await;
            if current.as_deref() == Some(stale) {
                *current = None;
            }
        }
        let scope_key = self.scope_key();
        {
            let mut shared = shared_sandbox_ids().lock().await;
            if shared
                .get(&scope_key)
                .is_some_and(|entry| entry.sandbox_id == stale)
            {
                shared.remove(&scope_key);
            }
        }
        let request = self
            .http
            .request(reqwest::Method::DELETE, &sandbox_url(stale));
        match self.http.send_raw(request, "delete_sandbox").await {
            Ok(response) if !response.is_success() && response.status != 404 => {
                tracing::debug!(
                    target: LOG_TARGET,
                    provider = PROVIDER,
                    status = response.status,
                    "jiuwenbox stale sandbox cleanup failed"
                );
            }
            Ok(_) => {}
            Err(error) => tracing::debug!(
                target: LOG_TARGET,
                provider = PROVIDER,
                "jiuwenbox stale sandbox cleanup failed: {}",
                error.message()
            ),
        }
    }
}

#[async_trait]
impl SandboxBackend for JiuwenBoxBackend {
    fn provider_name(&self) -> &'static str {
        PROVIDER
    }

    /// The generic wrapper evaluates `create_if_absent` /
    /// `expected_content_sha256` host-side over `read_bytes` and `write_bytes`,
    /// under a per-path in-process lock.
    fn supports_conditional_write(&self) -> bool {
        true
    }

    async fn read_bytes(&self, path: &str) -> std::result::Result<Vec<u8>, RemoteError> {
        self.with_sandbox(|sandbox_id| async move {
            let request = self
                .http
                .request(
                    reqwest::Method::GET,
                    &sandbox_url_with(&sandbox_id, "download"),
                )
                .query(&[("sandbox_path", path)]);
            let response = self.http.send_raw(request, "read_file").await?;
            ensure_success(&response, "read_file")?;
            Ok(response.body)
        })
        .await
    }

    async fn write_bytes(
        &self,
        path: &str,
        data: &[u8],
        append: bool,
    ) -> std::result::Result<(), RemoteError> {
        if append {
            let body = append_body(path, data);
            let body = &body;
            return self
                .with_sandbox(|sandbox_id| async move {
                    let request = self
                        .http
                        .request(
                            reqwest::Method::POST,
                            &sandbox_url_with(&sandbox_id, "exec"),
                        )
                        .json(body);
                    let response = self.http.send_raw(request, "write_file").await?;
                    ensure_success(&response, "write_file")?;
                    let payload = ExecPayload::decode(&response, "write_file")?;
                    if payload.exit_code() != 0 {
                        return Err(RemoteError::Status {
                            status: 0,
                            message: payload
                                .error_message()
                                .unwrap_or_else(|| "append file failed".to_string()),
                        });
                    }
                    Ok(())
                })
                .await;
        }
        self.with_sandbox(|sandbox_id| async move {
            let form = reqwest::multipart::Form::new().part(
                "file",
                reqwest::multipart::Part::bytes(data.to_vec()).file_name(upload_file_name(path)),
            );
            let request = self
                .http
                .request(
                    reqwest::Method::POST,
                    &sandbox_url_with(&sandbox_id, "upload"),
                )
                .query(&[("sandbox_path", path)])
                .multipart(form);
            let response = self.http.send_raw(request, "write_file").await?;
            ensure_success(&response, "write_file")
        })
        .await
    }

    async fn list(
        &self,
        request: &ListRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        self.with_sandbox(|sandbox_id| async move {
            let mut query = vec![
                ("sandbox_path".to_string(), request.path.clone()),
                ("recursive".to_string(), request.recursive.to_string()),
                (
                    "include_files".to_string(),
                    request.include_files.to_string(),
                ),
                ("include_dirs".to_string(), request.include_dirs.to_string()),
            ];
            if let Some(max_depth) = request.max_depth {
                query.push(("max_depth".to_string(), max_depth.to_string()));
            }
            let http_request = self
                .http
                .request(
                    reqwest::Method::GET,
                    &sandbox_url_with(&sandbox_id, "files"),
                )
                .query(&query);
            let response = self.http.send_raw(http_request, "list_files").await?;
            ensure_success(&response, "list_files")?;
            decode_items(&response, "list_files")
        })
        .await
    }

    async fn search(
        &self,
        request: &SearchRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        self.with_sandbox(|sandbox_id| async move {
            let mut query = vec![
                ("sandbox_path".to_string(), request.path.clone()),
                ("pattern".to_string(), request.pattern.clone()),
            ];
            for pattern in request.exclude_patterns.iter().flatten() {
                query.push(("exclude_patterns".to_string(), pattern.clone()));
            }
            let http_request = self
                .http
                .request(
                    reqwest::Method::GET,
                    &sandbox_url_with(&sandbox_id, "search"),
                )
                .query(&query);
            let response = self.http.send_raw(http_request, "search_files").await?;
            ensure_success(&response, "search_files")?;
            decode_items(&response, "search_files")
        })
        .await
    }

    async fn execute(&self, request: &ExecRequest) -> std::result::Result<ExecResult, RemoteError> {
        let body = exec_body(request);
        let body = &body;
        let timeout = self.http.timeout().max(Duration::from_secs(
            request.timeout_secs.unwrap_or(DEFAULT_EXEC_TIMEOUT_SECONDS),
        ));
        self.with_sandbox(|sandbox_id| async move {
            let http_request = self
                .http
                .request(
                    reqwest::Method::POST,
                    &sandbox_url_with(&sandbox_id, "exec"),
                )
                .timeout(timeout)
                .json(body);
            let response = self.http.send_raw(http_request, "exec").await?;
            ensure_success(&response, "exec")?;
            let payload = ExecPayload::decode(&response, "exec")?;
            let exit_code = payload.exit_code();
            Ok(ExecResult {
                exit_code,
                stdout: payload.stdout.unwrap_or_default(),
                stderr: payload.stderr.unwrap_or_default(),
                timed_out: exit_code == EXIT_CODE_TIMEOUT,
            })
        })
        .await
    }
}

/// The decoded `POST /api/v1/sandboxes/{id}/exec` answer.
#[derive(Debug, Default, Deserialize)]
struct ExecPayload {
    /// Exit status; absent or `null` means success.
    #[serde(default)]
    exit_code: Option<Value>,
    /// Captured standard output.
    #[serde(default)]
    stdout: Option<String>,
    /// Captured standard error.
    #[serde(default)]
    stderr: Option<String>,
}

impl ExecPayload {
    /// Decodes an exec answer, tolerating a missing `exit_code` and null strings.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::Transport`] when the body is not JSON or does not
    /// decode as an exec answer.
    fn decode(response: &RawResponse, operation: &str) -> std::result::Result<Self, RemoteError> {
        let value = decode_json(response, operation)?;
        serde_json::from_value(value).map_err(|error| {
            RemoteError::Transport(format!(
                "{operation}: could not decode the jiuwenbox exec response: {error}"
            ))
        })
    }

    /// The exit status, with Python's `int(result.get("exit_code") or 0)` defaults.
    fn exit_code(&self) -> i32 {
        exit_code_of(self.exit_code.as_ref())
    }

    /// The first non-blank of `stderr` and `stdout`, as Python's `or` chain does.
    fn error_message(&self) -> Option<String> {
        [self.stderr.as_deref(), self.stdout.as_deref()]
            .into_iter()
            .flatten()
            .map(str::trim)
            .find(|text| !text.is_empty())
            .map(str::to_string)
    }
}

/// One `{"items": [...]}` listing answer.
#[derive(Debug, Default, Deserialize)]
struct ItemList {
    /// The reported entries; an absent list is empty, like Python's `.get("items", [])`.
    #[serde(default)]
    items: Vec<Item>,
}

/// One remote entry as the container reports it.
///
/// Every field is optional because the container omits `type` for directories and
/// may send `null` for the rest (Python `_item_from_payload`).
#[derive(Debug, Default, Deserialize)]
struct Item {
    /// Entry name.
    #[serde(default)]
    name: Option<String>,
    /// Absolute path of the entry.
    #[serde(default)]
    path: Option<String>,
    /// Size in bytes.
    #[serde(default)]
    size: Option<u64>,
    /// Whether the entry is a directory.
    #[serde(default)]
    is_directory: Option<bool>,
    /// Modification time, formatted like a Python datetime.
    #[serde(default)]
    modified_time: Option<String>,
    /// The reported entry type, when the container sends one.
    #[serde(default, rename = "type")]
    file_type: Option<String>,
}

impl From<Item> for RemoteEntry {
    fn from(item: Item) -> Self {
        Self {
            name: item.name.unwrap_or_default(),
            path: item.path.unwrap_or_default(),
            size: item.size.unwrap_or(0),
            modified_time: item
                .modified_time
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| "0".to_string()),
            is_directory: item.is_directory.unwrap_or(false),
            file_type: item.file_type,
        }
    }
}

/// Trims a value and drops it when nothing is left.
fn non_blank(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// The bearer token from [`API_TOKEN_ENV`], trimmed, `None` when blank or absent.
fn api_token_from_env() -> Option<String> {
    non_blank(std::env::var(API_TOKEN_ENV).ok().as_deref())
}

/// The recreate budget from [`RECREATE_RETRIES_ENV`], falling back to the default
/// for a missing or unparsable value (Python `_resolve_recreate_retries`).
fn recreate_retries_from_env() -> u32 {
    let Ok(raw) = std::env::var(RECREATE_RETRIES_ENV) else {
        return DEFAULT_RECREATE_RETRIES;
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return DEFAULT_RECREATE_RETRIES;
    }
    raw.parse::<u32>().unwrap_or(DEFAULT_RECREATE_RETRIES)
}

/// Python truthiness for a JSON value (`if not api_sandbox_runtime`).
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// A JSON number as an unsigned integer, rejecting booleans (Python's
/// `isinstance(raw_check, bool)` guard).
fn number_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

/// The exec exit status with Python's defaults.
fn exit_code_of(value: Option<&Value>) -> i32 {
    match value {
        Some(Value::Number(number)) => number_exit_code(number),
        Some(Value::String(text)) => text.trim().parse::<i32>().unwrap_or(0),
        Some(Value::Bool(flag)) => i32::from(u8::from(*flag)),
        Some(Value::Null | Value::Array(_) | Value::Object(_)) | None => 0,
    }
}

/// An exit status reported as a JSON number, truncated like Python's `int()`.
fn number_exit_code(number: &serde_json::Number) -> i32 {
    if let Some(code) = number.as_i64() {
        return i32::try_from(code).unwrap_or(0);
    }
    let Some(real) = number.as_f64() else {
        return 0;
    };
    #[allow(clippy::cast_possible_truncation)] // Python's `int(1.9)` truncates as well.
    let code = real.trunc() as i32;
    code
}

/// The `command`, `workdir`, `env`, `stdin`, and `timeout_seconds` of an exec
/// request, omitting the fields the caller left unset (Python `_JiuwenBoxClient.exec`).
fn exec_body(request: &ExecRequest) -> Value {
    let mut body = Map::new();
    let command = match &request.command {
        ExecCommand::Argv(argv) => argv.iter().cloned().map(Value::String).collect(),
        ExecCommand::Shell(command) => vec![
            Value::String("bash".to_string()),
            Value::String("-lc".to_string()),
            Value::String(command.clone()),
        ],
    };
    body.insert("command".to_string(), Value::Array(command));
    if let Some(cwd) = request
        .cwd
        .as_deref()
        .filter(|cwd| !cwd.is_empty() && *cwd != ".")
    {
        body.insert("workdir".to_string(), Value::String(cwd.to_string()));
    }
    if !request.environment.is_empty() {
        let mut environment = Map::new();
        for (key, value) in &request.environment {
            environment.insert(
                key.clone(),
                value
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            );
        }
        body.insert("env".to_string(), Value::Object(environment));
    }
    if let Some(stdin) = &request.stdin {
        body.insert("stdin".to_string(), Value::String(stdin.clone()));
    }
    if let Some(timeout_seconds) = request.timeout_secs.filter(|seconds| *seconds > 0) {
        body.insert("timeout_seconds".to_string(), Value::from(timeout_seconds));
    }
    Value::Object(body)
}

/// The `stdin`-fed append request: a `bash` script that creates the parent
/// directory and appends the decoded payload (Python `_JiuwenBoxClient.append_bytes`).
fn append_body(path: &str, data: &[u8]) -> Value {
    let mut body = Map::new();
    body.insert(
        "command".to_string(),
        Value::Array(
            ["bash", "-lc", APPEND_SCRIPT, "jiuwenbox-append", path]
                .into_iter()
                .map(|part| Value::String(part.to_string()))
                .collect(),
        ),
    );
    body.insert("stdin".to_string(), Value::String(base64_encode(data)));
    Value::Object(body)
}

/// The multipart file name of an upload: the path's basename, `upload.bin` when empty.
fn upload_file_name(path: &str) -> String {
    let name = basename(path);
    if name.is_empty() {
        "upload.bin".to_string()
    } else {
        name
    }
}

/// `/api/v1/sandboxes/{id}`.
fn sandbox_url(sandbox_id: &str) -> String {
    format!("/api/v1/sandboxes/{sandbox_id}")
}

/// `/api/v1/sandboxes/{id}/{suffix}`.
fn sandbox_url_with(sandbox_id: &str, suffix: &str) -> String {
    format!("{}/{suffix}", sandbox_url(sandbox_id))
}

/// Maps a non-2xx answer onto the matching [`RemoteError`], distinguishing a lost
/// container from a missing remote path.
fn ensure_success(response: &RawResponse, operation: &str) -> std::result::Result<(), RemoteError> {
    if response.is_success() {
        return Ok(());
    }
    let status = response.status;
    let detail = error_detail(response);
    let message = if detail.is_empty() {
        format!("HTTP {status} for {operation}")
    } else {
        format!("HTTP {status} for {operation}: {detail}")
    };
    if status != 404 {
        return Err(RemoteError::Status { status, message });
    }
    if is_sandbox_not_found(response) {
        return Err(RemoteError::ContainerGone(message));
    }
    Err(RemoteError::NotFound(message))
}

/// The provider's explanation of a failed response (Python `_response_error_detail`).
fn error_detail(response: &RawResponse) -> String {
    if let Some(value) = response.json() {
        if let Some(map) = value.as_object() {
            for key in ["error", "detail", "message"] {
                match map.get(key) {
                    Some(Value::String(text)) if !text.trim().is_empty() => {
                        return text.clone();
                    }
                    Some(Value::Null | Value::String(_)) | None => {}
                    Some(other) => return other.to_string(),
                }
            }
            return value.to_string();
        }
        if let Value::String(text) = &value {
            if !text.trim().is_empty() {
                return text.clone();
            }
        } else if !value.is_null() {
            return value.to_string();
        }
    }
    response.text().trim().to_string()
}

/// Whether a 404 reports the sandbox container itself as gone (Python
/// `_is_sandbox_not_found_error`), rather than a missing file or directory.
fn is_sandbox_not_found(response: &RawResponse) -> bool {
    if response.status != 404 {
        return false;
    }
    match response.json() {
        Some(Value::Object(map)) => ["error", "detail", "message"].iter().any(|key| {
            map.get(*key)
                .and_then(Value::as_str)
                .is_some_and(is_sandbox_not_found_text)
        }),
        Some(_) => false,
        None => is_sandbox_not_found_text(&response.text()),
    }
}

/// Matches Python's `^\s*Sandbox\b.*\bnot found\b` (case-insensitive).
fn is_sandbox_not_found_text(text: &str) -> bool {
    const PREFIX: &str = "Sandbox";
    let trimmed = text.trim_start();
    let Some(head) = trimmed.get(..PREFIX.len()) else {
        return false;
    };
    if !head.eq_ignore_ascii_case(PREFIX) {
        return false;
    }
    let rest = &trimmed[PREFIX.len()..];
    if rest
        .chars()
        .next()
        .is_some_and(|character| character.is_alphanumeric() || character == '_')
    {
        return false;
    }
    rest.to_ascii_lowercase().contains("not found")
}

/// The response body as JSON.
///
/// # Errors
///
/// Returns [`RemoteError::Transport`] when the body is empty or not JSON.
fn decode_json(response: &RawResponse, operation: &str) -> std::result::Result<Value, RemoteError> {
    response.json().ok_or_else(|| {
        RemoteError::Transport(format!(
            "{operation}: jiuwenbox returned a non-JSON response"
        ))
    })
}

/// Decodes a `{"items": [...]}` listing answer.
///
/// # Errors
///
/// Returns [`RemoteError::Transport`] when the body is not a decodable listing.
fn decode_items(
    response: &RawResponse,
    operation: &str,
) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
    let value = decode_json(response, operation)?;
    let items: ItemList = serde_json::from_value(value).map_err(|error| {
        RemoteError::Transport(format!(
            "{operation}: could not decode the jiuwenbox listing: {error}"
        ))
    })?;
    Ok(items.items.into_iter().map(RemoteEntry::from).collect())
}
