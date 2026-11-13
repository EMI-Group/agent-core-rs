//! The `yuanrong` remote sandbox backend.
//!
//! Python parity: `extensions/sys_operation/sandbox/providers/yuanrong.py`
//! (v0.1.20). The backend drives a `YuanRong` **ADX** environment over three HTTP
//! endpoints rather than the kernel's gateway envelope:
//!
//! - `POST {base}/api/environment/v1/environments` creates an environment,
//! - `DELETE {base}/api/environment/v1/environments/{id}` releases one, and
//! - `POST {base}/direct/{id}/invoke` runs one `process.exec` / `file.*` action.
//!
//! # Configuration
//!
//! `endpoint.base_url` (or the launcher's `base_url`) is required; without it
//! construction fails with `Error::config("yuanrong provider requires
//! endpoint.base_url")` from the shared `SandboxHttp` transport. The provider
//! has no authentication of its own: the transport injects the card's
//! `auth_headers` and `auth_query_params` into every request, and the ADX calls
//! add only `Accept: application/json`. `launcher_config.extra_params` carries
//! the create fields (`rootfs`, `cpu`, `memory`, `idleTimeoutSeconds` /
//! `idle_timeout`, `image`, `workdir`, `mounts`, `user`) and the normalized
//! `executor` (`supervisor`, `docker`, or `conch`; anything else falls back to
//! `supervisor`).
//!
//! # Lifecycle
//!
//! The kernel has no launcher or lifecycle API, so the environment is created
//! lazily. The `(environment_id, runtime)` pair is cached in a documented
//! process-global map keyed by `"{base_url}|{isolation_key}"`, with an absent
//! isolation key contributing the empty string, exactly like Python's
//! `_shared_instances`, and mirrored in a per-backend slot. Resolution order is
//! the endpoint's `sandbox_id` captured at construction, then the shared cache,
//! then a fresh create.
//!
//! A `404` from the invoke endpoint means the environment is gone: the shared
//! entry is invalidated when it still points at the stale id, a best-effort
//! `DELETE` releases that environment, and the action retries against a new
//! one. Attempts are bounded by `min(recreate_retries + 1, 6)` — the default
//! five retries gives six attempts, matching Python's
//! `YUANRONG_RECREATE_RETRIES = 5` plus the first attempt — with a one-second
//! pause between attempts.
//!
//! `SandboxLauncherConfig::on_stop`, `idle_ttl_seconds`, and the gateway's
//! `timeout_seconds`-based idle eviction are not applied: the kernel never calls
//! a teardown hook, so an environment outlives the process that created it
//! unless the ADX deployment evicts it itself. The provider does not interpret
//! `on_stop` / `idle_ttl_seconds` at all.
//!
//! # Deliberate deviations from Python
//!
//! - `file.write` always sends binary hex (`"binary": true` with a hex `data`
//!   payload) so the remote bytes are exactly the caller's bytes; Python's text
//!   path sends a UTF-8 string and relies on the server's text handling.
//! - `file.read` assumes a binary payload: base64 when `content_encoding` is
//!   `"base64"`, hex otherwise. Python additionally accepts plain text when
//!   `binary` is unset, which this backend never requests.
//! - Streaming operations are buffered: the generic wrappers read or execute
//!   once and split the result host-side, as Python's `YuanRong` provider does.
//! - `list_files().file_types` filtering keeps Python's host-side semantics:
//!   only entries whose `type` field matches a requested value exactly survive,
//!   and unknown ADX `type` values pass through unchanged.
//! - `upload_file` / `download_file` `preserve_permissions` and
//!   `create_parent_dirs` flags are ignored; the ADX API has no channel for
//!   them.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::sys_operation::code::types::Language;
use serde_json::{json, Map, Value};

use super::backend::{
    provider_set_from, ExecCommand, ExecRequest, ExecResult, ListRequest, RemoteEntry,
    SandboxBackend, SearchRequest,
};
use super::semantics::posix_quote;
use super::teardown::{
    require_selection, send_delete, shared_key, SandboxSelector, TeardownReport,
};
use super::transport::{
    base64_decode, base64_encode, hex_decode, hex_encode, RemoteError, SandboxHttp,
};
use crate::sys_operation::{SandboxEndpoint, SandboxGatewayConfig, SandboxProviderSet};

/// The sandbox type this provider registers.
const PROVIDER: &str = "yuanrong";

/// ADX management-plane path for environments.
const ENVIRONMENTS_PATH: &str = "/api/environment/v1/environments";

/// Log target shared with the other remote sandbox providers.
const LOG_TARGET: &str = "openjiuwen_extensions::sys_operation";

/// The default ADX container executor.
const DEFAULT_EXECUTOR: &str = "supervisor";

/// Default CPU request sent on create.
const DEFAULT_CPU: i64 = 1000;

/// Default memory request (MiB) sent on create.
const DEFAULT_MEMORY: i64 = 1024;

/// The SDK's recursion cap, translated to an ADX list depth of `cap + 1`.
const UNLIMITED_LIST_DEPTH: u64 = 21;

/// The largest ADX list depth the provider asks for.
const MAX_LIST_DEPTH: u64 = 64;

/// Default recreate retries (Python `YUANRONG_RECREATE_RETRIES`).
const DEFAULT_RECREATE_RETRIES: u32 = 5;

/// Upper bound on attempts, including the first one.
const MAX_RECREATE_ATTEMPTS: u32 = 6;

/// Pause between recreate attempts (Python `YUANRONG_RECREATE_SLEEP_SECONDS`).
const DEFAULT_RECREATE_DELAY: Duration = Duration::from_secs(1);

/// Message used for a `file.search` that exceeded the server's result limit.
const SEARCH_LIMIT_MESSAGE: &str = "file search max_results exceeded; narrow the search";

/// The filesystem, shell, and code factories of the `yuanrong` sandbox type.
///
/// Python parity: `@SandboxRegistry.provider("yuanrong", "fs" | "shell" | "code")`.
pub(crate) fn provider_set() -> SandboxProviderSet {
    provider_set_from(YuanRongBackend::new)
}

/// One cache entry a teardown popped.
struct PoppedEnvironment {
    /// Cache key the entry lived under.
    key: String,
    /// The environment to release.
    instance: Instance,
    /// Client bound to the entry's ADX base URL and auth.
    http: SandboxHttp,
}

/// Releases cached ADX environments.
///
/// Python parity: `delete_yuanrong_sandbox` in
/// `extensions/sys_operation/sandbox/providers/yuanrong.py`. The selector is
/// applied to the process-wide cache in Python's precedence order — `delete_all`
/// (a Rust extension: drain the whole cache), then the shared key, then the base
/// URL — and a configured `sandbox_id` narrows the selection to that one
/// environment. Entries leave the cache before the delete request, so a failed
/// delete cannot leave a stale entry behind.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// when the selector names nothing to release, and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// when it names only an environment id: an ADX environment id carries no base
/// URL, so there is no host to send the `DELETE` to. Python has no such path
/// either (`delete_yuanrong_sandbox` requires `shared_key` or `base_url`).
pub(crate) async fn teardown(selector: &SandboxSelector) -> Result<TeardownReport> {
    require_selection(PROVIDER, selector)?;
    if selector.shared_key.is_none() && selector.base_url.is_none() && !selector.delete_all {
        return Err(Error::unsupported(
            "the yuanrong provider cannot delete an environment by sandbox id alone: an ADX environment id carries no base URL to send the delete to; select a shared key or a base URL instead",
        ));
    }
    let mut report = TeardownReport::new(PROVIDER);
    for entry in pop_cached(selector).await {
        let path = format!(
            "{ENVIRONMENTS_PATH}/{}",
            percent_encode_path(&entry.instance.environment_id)
        );
        let request = entry
            .http
            .request(reqwest::Method::DELETE, &path)
            .header("Accept", "application/json");
        match send_delete(&entry.http, request, "delete_environment").await {
            Ok(()) => {
                tracing::info!(
                    target: LOG_TARGET,
                    provider = PROVIDER,
                    key = %entry.key,
                    reason = %selector.reason,
                    "yuanrong environment released"
                );
                report.record_released(entry.key, entry.instance.environment_id);
            }
            Err(reason) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    provider = PROVIDER,
                    key = %entry.key,
                    reason = %reason,
                    "yuanrong environment release failed"
                );
                report.record_failure(entry.key, entry.instance.environment_id, reason);
            }
        }
    }
    Ok(report)
}

/// Pops the cache entries `selector` names, in key order.
///
/// `selector.sandbox_id` narrows the key/base-URL selection to that one
/// environment id.
async fn pop_cached(selector: &SandboxSelector) -> Vec<PoppedEnvironment> {
    let wanted = selector
        .sandbox_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let mut shared = shared_instances().lock().await;
    let keys: Vec<String> = shared
        .keys()
        .filter(|key| selector.matches_key(key))
        .filter(|key| {
            wanted.is_none_or(|id| {
                shared
                    .get(*key)
                    .is_some_and(|entry| entry.instance.environment_id == id)
            })
        })
        .cloned()
        .collect();
    let mut popped = Vec::with_capacity(keys.len());
    for key in keys {
        if let Some(entry) = shared.remove(&key) {
            popped.push(PoppedEnvironment {
                key,
                instance: entry.instance,
                http: entry.http,
            });
        }
    }
    popped
}

/// One ADX environment: its id and the executor it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Instance {
    /// The `environmentId` returned by the management plane.
    environment_id: String,
    /// The runtime the environment reported, or the configured executor.
    runtime: String,
}

/// The environments created by this process, keyed by `base_url|isolation_key`.
///
/// A process-global by design (Python keeps the same data in
/// `_YuanrongProviderMixin._shared_instances`): every provider instance built for
/// the same base URL and isolation key must share one environment, and the kernel
/// hands each factory a fresh `SandboxEndpoint` and cannot propagate a discovered
/// environment id back through it.
static SHARED_INSTANCES: OnceLock<tokio::sync::Mutex<BTreeMap<String, CachedEnvironment>>> =
    OnceLock::new();

/// One cached environment: its `(id, runtime)` pair plus the client that created
/// it, so a teardown can issue the matching `DELETE` with the same base URL and
/// auth.
#[derive(Clone)]
struct CachedEnvironment {
    /// The environment the management plane returned.
    instance: Instance,
    /// Client bound to the environment's ADX base URL and auth.
    http: SandboxHttp,
}

/// The process-global environment cache, initialized on first use.
fn shared_instances() -> &'static tokio::sync::Mutex<BTreeMap<String, CachedEnvironment>> {
    SHARED_INSTANCES.get_or_init(|| tokio::sync::Mutex::new(BTreeMap::new()))
}

/// The ADX backend behind the `yuanrong` provider factories.
pub(crate) struct YuanRongBackend {
    /// HTTP client bound to the ADX base URL, with auth already injected.
    http: SandboxHttp,
    /// `launcher_config.extra_params`, the create fields and the executor hint.
    extra: BTreeMap<String, Value>,
    /// The normalized executor (`supervisor`, `docker`, or `conch`).
    executor: String,
    /// The endpoint's isolation key, part of the shared cache key.
    isolation_key: Option<String>,
    /// The endpoint's pre-assigned sandbox id, when the kernel already resolved one.
    endpoint_id: Option<String>,
    /// The environment this backend last bound to, mirroring Python's `_instance`.
    instance: tokio::sync::Mutex<Option<Instance>>,
    /// Recreate attempts after the first one, before giving up.
    recreate_retries: u32,
    /// Pause between recreate attempts.
    recreate_delay: Duration,
}

impl YuanRongBackend {
    /// Builds the backend for one sandbox endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config)
    /// when no base URL is configured, when the HTTP client cannot be built, or
    /// when `idleTimeoutSeconds` is a boolean or not an integer.
    pub(crate) fn new(endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig) -> Result<Self> {
        let http = SandboxHttp::new(PROVIDER, endpoint, config)?;
        let extra = config
            .launcher_config
            .as_ref()
            .map(|launcher| launcher.extra_params.clone())
            .unwrap_or_default();
        // Validate the create field eagerly so a bad card fails construction, not
        // the first remote call (Python raises from `_idle_timeout_seconds`).
        idle_timeout_seconds(&extra).map_err(Error::config)?;
        Ok(Self {
            http,
            executor: normalize_executor(extra.get("executor")),
            isolation_key: endpoint.isolation_key.clone().filter(|key| !key.is_empty()),
            endpoint_id: endpoint
                .sandbox_id
                .clone()
                .filter(|id| !id.trim().is_empty()),
            extra,
            instance: tokio::sync::Mutex::new(None),
            recreate_retries: DEFAULT_RECREATE_RETRIES,
            recreate_delay: DEFAULT_RECREATE_DELAY,
        })
    }

    /// Overrides the recreate policy; tests use it to avoid the one-second pause.
    #[cfg(test)]
    pub(crate) fn with_recreate(mut self, retries: u32, delay: Duration) -> Self {
        self.recreate_retries = retries;
        self.recreate_delay = delay;
        self
    }

    /// The create body sent to the ADX management plane.
    fn create_body(&self) -> Value {
        let mut body = Map::new();
        body.insert(
            "rootfs".to_string(),
            Value::String(rootfs_string(&self.extra)),
        );
        body.insert(
            "cpu".to_string(),
            json!(integer_extra(&self.extra, "cpu", DEFAULT_CPU)),
        );
        body.insert(
            "memory".to_string(),
            json!(integer_extra(&self.extra, "memory", DEFAULT_MEMORY)),
        );
        body.insert(
            "idleTimeoutSeconds".to_string(),
            json!(idle_timeout_seconds(&self.extra).unwrap_or(0)),
        );
        body.insert("runtime".to_string(), Value::String(self.executor.clone()));
        if let Some(user) = self.extra.get("user").and_then(Value::as_str) {
            let user = user.trim();
            if !user.is_empty() {
                body.insert("extra_config".to_string(), json!({ "user": user }));
            }
        }
        Value::Object(body)
    }

    /// The shared cache key of this backend: `base_url|isolation_key`.
    fn cache_key(&self) -> String {
        shared_key(self.http.base_url(), self.isolation_key.as_deref())
    }

    /// Returns the environment to invoke, creating one when needed.
    ///
    /// Resolution order: the per-backend slot, the endpoint's `sandbox_id`
    /// captured at construction, the process-global cache, then a fresh create.
    async fn ensure_instance(&self) -> std::result::Result<Instance, RemoteError> {
        {
            let local = self.instance.lock().await;
            if let Some(instance) = local.as_ref() {
                return Ok(instance.clone());
            }
        }
        if let Some(environment_id) = &self.endpoint_id {
            return Ok(Instance {
                environment_id: environment_id.clone(),
                runtime: self.executor.clone(),
            });
        }
        let key = self.cache_key();
        {
            let shared = shared_instances().lock().await;
            if let Some(entry) = shared.get(&key) {
                let instance = entry.instance.clone();
                drop(shared);
                *self.instance.lock().await = Some(instance.clone());
                return Ok(instance);
            }
        }
        let created = self.create_instance().await?;
        let existing = {
            let mut shared = shared_instances().lock().await;
            let existing = shared.get(&key).map(|entry| entry.instance.clone());
            if existing.is_none() {
                shared.insert(
                    key,
                    CachedEnvironment {
                        instance: created.clone(),
                        http: self.http.clone(),
                    },
                );
            }
            existing
        };
        let instance = if let Some(existing) = existing {
            tracing::debug!(
                target: LOG_TARGET,
                "discarding a duplicate yuanrong environment"
            );
            self.delete_environment(&created.environment_id).await;
            existing
        } else {
            created
        };
        *self.instance.lock().await = Some(instance.clone());
        Ok(instance)
    }

    /// Creates one ADX environment and returns its id and runtime.
    async fn create_instance(&self) -> std::result::Result<Instance, RemoteError> {
        let request = self
            .http
            .request(reqwest::Method::POST, ENVIRONMENTS_PATH)
            .header("Accept", "application/json")
            .json(&self.create_body());
        let response = self.http.send_raw(request, "create_environment").await?;
        if response.status != 200 {
            let payload = response.json();
            let detail = adx_error_message(payload.as_ref());
            return Err(RemoteError::Status {
                status: response.status,
                message: format!("HTTP {} for create_environment: {detail}", response.status),
            });
        }
        let payload = response.json().ok_or_else(|| {
            RemoteError::Transport(
                "create_environment: the ADX response is not valid JSON".to_string(),
            )
        })?;
        let data = decode_management_data(&payload)?;
        let environment_id = data
            .get("environmentId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                RemoteError::Transport("ADX create response has no environmentId".to_string())
            })?
            .to_string();
        let runtime = data
            .get("runtime")
            .and_then(Value::as_str)
            .filter(|runtime| !runtime.is_empty())
            .unwrap_or(&self.executor)
            .to_string();
        let instance = Instance {
            environment_id,
            runtime,
        };
        tracing::debug!(
            target: LOG_TARGET,
            runtime = %instance.runtime,
            "yuanrong environment created"
        );
        Ok(instance)
    }

    /// Drops the stale environment from the caches and releases it.
    ///
    /// The shared entry is removed only when it still points at `stale`; the
    /// remote delete then follows Python's `_delete_environment`, tolerating
    /// `200` and `404` and logging anything else at `debug`.
    async fn invalidate(&self, stale: &str) {
        *self.instance.lock().await = None;
        let key = self.cache_key();
        let should_delete = {
            let mut shared = shared_instances().lock().await;
            match shared.get(&key) {
                Some(entry) if entry.instance.environment_id == stale => {
                    shared.remove(&key);
                    true
                }
                _ => false,
            }
        };
        if should_delete {
            self.delete_environment(stale).await;
        }
    }

    /// Best-effort `DELETE` of one environment.
    async fn delete_environment(&self, environment_id: &str) {
        let path = format!(
            "{ENVIRONMENTS_PATH}/{}",
            percent_encode_path(environment_id)
        );
        let request = self
            .http
            .request(reqwest::Method::DELETE, &path)
            .header("Accept", "application/json");
        match self.http.send_raw(request, "delete_environment").await {
            Ok(response) if response.status == 200 || response.status == 404 => {}
            Ok(response) => tracing::debug!(
                target: LOG_TARGET,
                status = response.status,
                "yuanrong environment delete failed"
            ),
            Err(error) => tracing::debug!(
                target: LOG_TARGET,
                reason = %error.message(),
                "yuanrong environment delete failed"
            ),
        }
    }

    /// Runs one action against the bound environment, recreating on `404`.
    async fn invoke(&self, action: &str, args: &Value) -> std::result::Result<Value, RemoteError> {
        let attempts = self
            .recreate_retries
            .saturating_add(1)
            .clamp(1, MAX_RECREATE_ATTEMPTS);
        let mut last = None;
        for attempt in 0..attempts {
            if attempt > 0 {
                tokio::time::sleep(self.recreate_delay).await;
            }
            let instance = self.ensure_instance().await?;
            match self.invoke_once(&instance, action, args).await {
                Ok(value) => return Ok(value),
                Err(error) if error.is_container_gone() => {
                    tracing::debug!(
                        target: LOG_TARGET,
                        action,
                        "yuanrong environment is gone; recreating"
                    );
                    self.invalidate(&instance.environment_id).await;
                    last = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| {
            RemoteError::Transport(format!("{action}: yuanrong recreate exhausted"))
        }))
    }

    /// Invokes one action against a known environment.
    async fn invoke_once(
        &self,
        instance: &Instance,
        action: &str,
        args: &Value,
    ) -> std::result::Result<Value, RemoteError> {
        let path = format!(
            "/direct/{}/invoke",
            percent_encode_path(&instance.environment_id)
        );
        let body = json!({ "action": action, "args": args });
        let request = self
            .http
            .request(reqwest::Method::POST, &path)
            .header("Accept", "application/json")
            .json(&body);
        let response = self.http.send_raw(request, action).await?;
        let payload = response.json();
        if response.status == 429 {
            let message = payload
                .as_ref()
                .and_then(Value::as_object)
                .and_then(|map| map.get("error"))
                .and_then(Value::as_str)
                .filter(|message| !message.trim().is_empty())
                .map_or_else(|| SEARCH_LIMIT_MESSAGE.to_string(), str::to_string);
            return Err(RemoteError::SearchLimit(message));
        }
        if response.status >= 400 {
            return Err(response_error(response.status, payload.as_ref(), action));
        }
        Ok(payload.unwrap_or_else(|| Value::Object(Map::new())))
    }
}

#[async_trait]
impl SandboxBackend for YuanRongBackend {
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
        let payload = self
            .invoke("file.read", &json!({ "path": path, "binary": true }))
            .await?;
        decode_read_payload(&payload)
    }

    async fn write_bytes(
        &self,
        path: &str,
        data: &[u8],
        append: bool,
    ) -> std::result::Result<(), RemoteError> {
        let payload = self
            .invoke(
                "file.write",
                &json!({
                    "path": path,
                    "data": hex_encode(data),
                    "binary": true,
                    "append": append,
                }),
            )
            .await?;
        if let Some(message) = action_error(&payload) {
            return Err(RemoteError::Status { status: 0, message });
        }
        Ok(())
    }

    async fn list(
        &self,
        request: &ListRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        let payload = self
            .invoke(
                "file.list",
                &json!({
                    "path": request.path,
                    "depth": list_depth(request),
                }),
            )
            .await?;
        Ok(normalize_list_items(&payload)?
            .into_iter()
            .map(|item| item_from_payload(&item))
            .filter(|entry| {
                if entry.is_directory {
                    request.include_dirs
                } else {
                    request.include_files
                }
            })
            .collect())
    }

    async fn search(
        &self,
        request: &SearchRequest,
    ) -> std::result::Result<Vec<RemoteEntry>, RemoteError> {
        let mut args = Map::new();
        args.insert("path".to_string(), Value::String(request.path.clone()));
        args.insert(
            "pattern".to_string(),
            Value::String(request.pattern.clone()),
        );
        if let Some(exclude_patterns) = &request.exclude_patterns {
            args.insert("exclude_patterns".to_string(), json!(exclude_patterns));
        }
        let payload = self.invoke("file.search", &Value::Object(args)).await?;
        Ok(normalize_list_items(&payload)?
            .into_iter()
            .map(|item| item_from_payload(&item))
            .collect())
    }

    async fn execute(&self, request: &ExecRequest) -> std::result::Result<ExecResult, RemoteError> {
        let mut command = request.command.to_shell_command();
        if !request.environment.is_empty() {
            let mut parts = Vec::with_capacity(request.environment.len());
            for (key, value) in &request.environment {
                match value {
                    Some(value) => {
                        parts.push(format!("export {key}={}", posix_quote(value)));
                    }
                    None => parts.push(format!("unset -- {}", posix_quote(key))),
                }
            }
            command = format!("{} && {command}", parts.join(" && "));
        }
        let mut args = Map::new();
        args.insert("cmd".to_string(), Value::String(command));
        if let Some(cwd) = request
            .cwd
            .as_deref()
            .map(str::trim)
            .filter(|cwd| !cwd.is_empty() && *cwd != ".")
        {
            args.insert("cwd".to_string(), Value::String(cwd.to_string()));
        }
        if let Some(timeout) = request.timeout_secs.filter(|timeout| *timeout > 0) {
            args.insert("timeout".to_string(), json!(timeout));
        }
        let payload = self.invoke("process.exec", &Value::Object(args)).await?;
        Ok(exec_result(&payload))
    }

    /// Reproduces Python's `YuanRong` snippet commands: the payload is base64
    /// encoded and decoded inside the interpreter (inline) or written to a
    /// temporary file first (`force_file`).
    fn code_command(&self, code: &str, language: Language, force_file: bool) -> ExecCommand {
        let encoded = base64_encode(code.as_bytes());
        if force_file {
            return ExecCommand::Shell(format!(
                "tmp=$(mktemp /tmp/ojw_code_XXXXXX{suffix}) && printf %s {payload} | base64 -d > \"$tmp\" && {executable} \"$tmp\"; status=$?; rm -f \"$tmp\"; exit $status",
                suffix = language.file_suffix(),
                payload = posix_quote(&encoded),
                executable = language.executable(),
            ));
        }
        match language {
            Language::Python => ExecCommand::Shell(format!(
                "python3 -c {}",
                posix_quote(&format!(
                    "import base64; exec(base64.b64decode('{encoded}').decode('utf-8'))"
                ))
            )),
            Language::Javascript => ExecCommand::Shell(format!(
                "node -e {}",
                posix_quote(&format!(
                    "eval(Buffer.from('{encoded}','base64').toString('utf8'))"
                ))
            )),
        }
    }
}

/// Builds the `rootfs` field of the create body.
///
/// A non-blank string passes through, an object is shallow-cloned with `image`
/// aliased to `imageurl` when the latter is absent, and anything else is
/// assembled from `image` / `workdir` / `mounts` (missing keys become JSON
/// `null`, never omitted).
fn rootfs_string(extra: &BTreeMap<String, Value>) -> String {
    if let Some(Value::String(text)) = extra.get("rootfs") {
        if !text.trim().is_empty() {
            return text.clone();
        }
    }
    if let Some(Value::Object(map)) = extra.get("rootfs") {
        let mut payload = map.clone();
        if !payload.contains_key("imageurl") {
            if let Some(image) = payload.remove("image") {
                payload.insert("imageurl".to_string(), image);
            }
        }
        return Value::Object(payload).to_string();
    }
    json!({
        "type": "image",
        "imageurl": extra.get("image").cloned().unwrap_or(Value::Null),
        "workdir": extra.get("workdir").cloned().unwrap_or(Value::Null),
        "mounts": extra.get("mounts").cloned().unwrap_or(Value::Null),
    })
    .to_string()
}

/// Reads `idleTimeoutSeconds`, falling back to `idle_timeout`, else `0`.
///
/// Negatives become `0`; a boolean or a non-integer is an error, matching
/// Python's `_idle_timeout_seconds`.
fn idle_timeout_seconds(extra: &BTreeMap<String, Value>) -> std::result::Result<i64, String> {
    let raw = extra
        .get("idleTimeoutSeconds")
        .or_else(|| extra.get("idle_timeout"));
    let invalid = || "idleTimeoutSeconds must be an integer".to_string();
    match raw {
        None | Some(Value::Null) => Ok(0),
        Some(Value::Bool(_)) => Err(invalid()),
        Some(value) => integer_value(value)
            .map(|seconds| seconds.max(0))
            .ok_or_else(invalid),
    }
}

/// Reads an integer create field, falling back to `default`.
fn integer_extra(extra: &BTreeMap<String, Value>, key: &str, default: i64) -> i64 {
    extra.get(key).and_then(integer_value).unwrap_or(default)
}

/// Converts a JSON number or numeric string to an integer.
#[allow(clippy::cast_possible_truncation)]
fn integer_value(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| number.as_f64().map(|value| value.trunc() as i64)),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Normalizes `launcher_config.extra_params["executor"]`.
///
/// The value is lowercased and trimmed, one wrapping quote pair is stripped,
/// and only `supervisor`, `docker`, and `conch` are kept; everything else falls
/// back to `supervisor`.
fn normalize_executor(value: Option<&Value>) -> String {
    let mut text = match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => DEFAULT_EXECUTOR.to_string(),
        Some(other) => other.to_string(),
    };
    text = text.trim().to_lowercase();
    let bytes = text.as_bytes();
    if bytes.len() >= 2 && bytes[0] == bytes[bytes.len() - 1] && matches!(bytes[0], b'\'' | b'"') {
        text = text[1..text.len() - 1].trim().to_string();
    }
    let text = text.trim_matches(['\'', '"']);
    match text {
        "supervisor" | "docker" | "conch" => text.to_string(),
        _ => DEFAULT_EXECUTOR.to_string(),
    }
}

/// Decodes the `data` field of a management-plane response.
///
/// `data` is either the object itself or a base64 string holding its JSON.
fn decode_management_data(payload: &Value) -> std::result::Result<Map<String, Value>, RemoteError> {
    let map = payload.as_object().ok_or_else(|| {
        RemoteError::Transport("ADX management response is not an object".to_string())
    })?;
    match map.get("data") {
        Some(Value::Object(data)) => Ok(data.clone()),
        Some(Value::String(encoded)) => {
            let bytes = base64_decode(encoded).map_err(|error| {
                RemoteError::Transport(format!(
                    "ADX management response data is not valid base64: {}",
                    error.message()
                ))
            })?;
            let decoded: Value = serde_json::from_slice(&bytes).map_err(|error| {
                RemoteError::Transport(format!(
                    "ADX management response data is not valid JSON: {error}"
                ))
            })?;
            decoded.as_object().cloned().ok_or_else(|| {
                RemoteError::Transport("ADX management response has no data".to_string())
            })
        }
        _ => Err(RemoteError::Transport(
            "ADX management response has no data".to_string(),
        )),
    }
}

/// Extracts a human-readable message from an ADX error body.
fn adx_error_message(body: Option<&Value>) -> String {
    if let Some(map) = body.and_then(Value::as_object) {
        for key in ["message", "error"] {
            if let Some(text) = map.get(key).and_then(error_text) {
                return text;
            }
        }
    }
    "ADX request failed".to_string()
}

/// Renders one error field, unwrapping a nested `message` object.
fn error_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => (!text.trim().is_empty()).then(|| text.clone()),
        Value::Object(inner) => match inner.get("message") {
            Some(Value::String(text)) if !text.trim().is_empty() => Some(text.clone()),
            _ => Some(Value::Object(inner.clone()).to_string()),
        },
        other => Some(other.to_string()),
    }
}

/// Builds the mapped error for a non-200 invoke response.
fn response_error(status: u16, body: Option<&Value>, action: &str) -> RemoteError {
    let message = format!("HTTP {status} for {action}: {}", adx_error_message(body));
    if status == 404 {
        RemoteError::ContainerGone(message)
    } else {
        RemoteError::Status { status, message }
    }
}

/// Returns the non-empty `error` field of an action response.
fn action_error(value: &Value) -> Option<String> {
    match value.as_object()?.get("error")? {
        Value::Null => None,
        Value::String(text) if text.trim().is_empty() => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

/// Decodes a `file.read` payload (hex, or base64 when announced).
fn decode_read_payload(payload: &Value) -> std::result::Result<Vec<u8>, RemoteError> {
    let Some(map) = payload.as_object() else {
        return Ok(match payload {
            Value::Null => Vec::new(),
            Value::String(text) => text.clone().into_bytes(),
            other => other.to_string().into_bytes(),
        });
    };
    if let Some(message) = action_error(payload) {
        return Err(if message.to_lowercase().contains("no such file") {
            RemoteError::NotFound(message)
        } else {
            RemoteError::Status { status: 0, message }
        });
    }
    let data = map
        .get("data")
        .or_else(|| map.get("content"))
        .ok_or_else(|| RemoteError::Transport("file.read response has no data".to_string()))?;
    if map.get("content_encoding").and_then(Value::as_str) == Some("base64") {
        let encoded = data.as_str().ok_or_else(|| {
            RemoteError::Transport("file.read base64 payload is not a string".to_string())
        })?;
        return base64_decode(encoded);
    }
    match data {
        Value::String(text) => hex_decode(text),
        other => Ok(other.to_string().into_bytes()),
    }
}

/// The ADX `file.list` depth for an SDK listing request.
fn list_depth(request: &ListRequest) -> u64 {
    if !request.recursive {
        return 1;
    }
    match request.max_depth {
        Some(depth) => u64::try_from(depth)
            .unwrap_or(u64::MAX)
            .saturating_add(1)
            .min(MAX_LIST_DEPTH),
        None => UNLIMITED_LIST_DEPTH,
    }
}

/// Extracts the entry objects of a `file.list` / `file.search` payload.
///
/// Accepts `{"entries": [...]}`, `{"items": [...]}`, and a bare array; an
/// `error` field becomes a [`RemoteError::Status`] with status `0`.
fn normalize_list_items(
    payload: &Value,
) -> std::result::Result<Vec<Map<String, Value>>, RemoteError> {
    if let Some(message) = action_error(payload) {
        return Err(RemoteError::Status { status: 0, message });
    }
    Ok(match payload {
        Value::Object(map) => map
            .get("entries")
            .or_else(|| map.get("items"))
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_object).cloned().collect())
            .unwrap_or_default(),
        Value::Array(items) => items.iter().filter_map(Value::as_object).cloned().collect(),
        _ => Vec::new(),
    })
}

/// Maps one ADX list/search entry onto [`RemoteEntry`].
fn item_from_payload(item: &Map<String, Value>) -> RemoteEntry {
    let is_directory = match item.get("is_directory") {
        Some(value) => json_truthy(value),
        None => matches!(
            item.get("type").and_then(Value::as_str),
            Some("dir" | "directory")
        ),
    };
    let modified_time = match item.get("modified_time") {
        None | Some(Value::Null | Value::Bool(_)) => "0".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    };
    let raw_type = item.get("type");
    let file_type = match raw_type {
        None | Some(Value::Null) => None,
        _ if is_directory => None,
        Some(Value::String(text)) if matches!(text.as_str(), "file" | "dir" | "directory") => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(other) => Some(other.to_string()),
    };
    RemoteEntry {
        name: text_field(item, "name"),
        path: text_field(item, "path"),
        size: item
            .get("size")
            .and_then(integer_value)
            .and_then(|size| u64::try_from(size).ok())
            .unwrap_or(0),
        is_directory,
        modified_time,
        file_type,
    }
}

/// Reads a string field, stringifying non-string values and treating null as empty.
fn text_field(item: &Map<String, Value>, key: &str) -> String {
    match item.get(key) {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Normalizes a `process.exec` response.
fn exec_result(payload: &Value) -> ExecResult {
    let Some(map) = payload.as_object() else {
        return ExecResult {
            exit_code: -1,
            stdout: String::new(),
            stderr: "unexpected result type".to_string(),
            timed_out: false,
        };
    };
    let exit_code = map
        .get("exit_code")
        .or_else(|| map.get("returncode"))
        .and_then(integer_value)
        .and_then(|code| i32::try_from(code).ok())
        .unwrap_or(-1);
    let stdout = text_field(map, "stdout");
    let stderr = text_field(map, "stderr");
    let timed_out = exit_code == 124 || stderr.to_lowercase().contains("timed out");
    ExecResult {
        exit_code,
        stdout,
        stderr,
        timed_out,
    }
}

/// Percent-encodes one path segment, keeping only RFC 3986 unreserved bytes.
fn percent_encode_path(value: &str) -> String {
    const HEX: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    ];
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(HEX[usize::from(byte >> 4)]);
            out.push(HEX[usize::from(byte & 0x0f)]);
        }
    }
    out
}

#[cfg(test)]
mod tests;
