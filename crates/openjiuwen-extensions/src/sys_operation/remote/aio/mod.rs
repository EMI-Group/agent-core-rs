//! The `aio` remote sandbox backend.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/aio.py`
//! (v0.1.20), which registers one provider per operation type
//! (`@SandboxRegistry.provider("aio", "fs" | "shell" | "code")`). This module
//! builds the same three factories.
//!
//! # The transport decision
//!
//! Python's provider wraps the third-party `agent_sandbox` SDK, which is not
//! vendored and whose private REST contract is not part of the reference
//! repository. The only wire contract defined there is the kernel gateway
//! envelope: a [`GatewayInvokeRequest`] (`op_type` `"fs"` / `"shell"` /
//! `"code"`, `method`, `params`, `isolation_key`) answered by a
//! [`GatewayResponse`] (`code`, `message`, `data`) where `code == 0` means
//! success (`core/sys_operation/sandbox/gateway/gateway.py` and
//! `gateway_client.py`, which dispatch `params` as the provider method's
//! keyword arguments). This provider is therefore a real HTTP client for that
//! envelope:
//!
//! 1. it POSTs the envelope to `{base_url}{invoke_path}`, where `invoke_path`
//!    defaults to [`DEFAULT_INVOKE_PATH`] and can be overridden through
//!    `launcher_config.extra_params["invoke_path"]` (a leading `/` is added
//!    when missing);
//! 2. it unwraps the [`GatewayResponse`] tolerantly (`code` defaults to `0`,
//!    `message` to `""`, `data` is optional) and returns `data` (`null` when
//!    absent) on success, or a [`RemoteError::Status`] carrying
//!    `aio gateway code <code>: <message>` otherwise;
//! 3. it maps `data` into the kernel result structs.
//!
//! Every failure caused by the gateway or the transport — a `code != 0`
//! envelope rejection, a non-2xx answer, a timeout, or an undecodable body —
//! becomes the kernel's sandbox gateway status (code `199007`) with
//! [`sandbox_gateway_message`](openjiuwen_core::sys_operation::status::sandbox_gateway_message),
//! for filesystem, shell, and code operations alike. Operation-specific codes
//! are reserved for parameter validation and execution outcomes.
//!
//! The `params` field names mirror the Python provider signatures exactly
//! (`timeout`, not `timeout_secs`; `line_range` as a `[start, end]` array;
//! bytes content as a padded base64 string with `encoding: "base64"`), because
//! the gateway dispatches them as keyword arguments.
//!
//! # What stays host-side
//!
//! The envelope has **no streaming variant**, so the three stream methods
//! perform their buffered call and chunk the result on the host: text reads by
//! line, byte reads by `chunk_size` (or [`DEFAULT_STREAM_CHUNK_SIZE`]) and
//! shell/code output by line, always ending in a terminal frame. Host-side
//! work is otherwise limited to local file transfer (upload/download) and path
//! handling; the *remote* provider performs text selection
//! (`head`/`tail`/`line_range`) and newline transforms
//! (`prepend_newline`/`append_newline`), which is why those arrive as
//! parameters instead of being applied here.
//!
//! Two further consequences are deliberate:
//!
//! - `upload_file` cannot hand the container a `local_path` (the envelope sees
//!   the provider's filesystem, not the caller's), so the host file is read and
//!   inlined as base64 bytes in a `write_file` call;
//! - `execute_argv` has no Python counterpart — it is sent as `execute_cmd`
//!   with the POSIX-quoted argv joined by single spaces, the same shape
//!   [`posix_quote`] gives the generic remote backend.
//!
//! Container lifecycle (`launcher_config.on_stop`, `idle_ttl_seconds`) is not
//! applied: the kernel has no launcher or lifecycle API, so a provider only
//! talks to the already resolved [`SandboxEndpoint`].
//!
//! # Notes on the mapping
//!
//! - Shell completion is success even for a non-zero exit code (Python
//!   parity); only a timeout becomes a failure (`199004`,
//!   `execution timeout after <n> seconds` with the output as `data`), and a
//!   `124` exit code — `timeout(1)`'s status — is the timeout signal. A blank
//!   command is `199004` `command can not be empty`.
//! - Code completion follows the same rule (`199005`), detecting a timeout
//!   from exit code `124` or a `stderr` containing "timed out"; a blank
//!   snippet is `199005` `code can not be empty`.
//! - Gateway and transport failures are always `199007` (see above), so they
//!   never take the operation-specific code of the call.
//! - `write_file` honours `expected_content_sha256` and `create_if_absent`
//!   host-side: it reads the current remote bytes, checks the precondition, and
//!   only then posts the `write_file` envelope, all while holding a per-path
//!   in-process lock (the crate's shared conditional-write support). Python's
//!   provider has no such options, so this is an additive Rust capability, and
//!   the guarantee is process-local: the envelope write itself is not atomic
//!   against a concurrent reader. `supports_conditional_write` is therefore
//!   always `true`.

//! # Layout
//!
//! `mod.rs` holds the envelope client, the shared `params` / `data` helpers, and
//! the three factories; the kernel operation traits live in `fs.rs`, `shell.rs`,
//! and `code.rs`, one wrapper each, and the wire-contract tests live in the
//! sibling `tests.rs` (with per-operation child modules).

use std::sync::Arc;

use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_core::sys_operation::operation::{CodeOperation, FsOperation, ShellOperation};
use openjiuwen_core::sys_operation::sandbox::config::{
    GatewayInvokeRequest, GatewayResponse, SandboxEndpoint, SandboxGatewayConfig,
};

use super::transport::{RemoteError, SandboxHttp};
use crate::sys_operation::SandboxProviderSet;

mod code;
mod fs;
mod shell;

#[cfg(test)]
mod tests;

pub(crate) use code::AioCode;
pub(crate) use fs::AioFs;
pub(crate) use shell::AioShell;

/// The sandbox type this module implements.
const PROVIDER: &str = "aio";

/// The envelope path a stock AIO gateway listens on.
const DEFAULT_INVOKE_PATH: &str = "/api/v1/invoke";

/// The `launcher_config.extra_params` key overriding [`DEFAULT_INVOKE_PATH`].
const INVOKE_PATH_PARAM: &str = "invoke_path";

/// The timeout applied when the caller omits one (Python `timeout or 300`).
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// `timeout(1)`'s exit status, which both providers treat as a timeout.
const TIMEOUT_EXIT_CODE: i32 = 124;

/// The encoding name that marks bytes payloads in the envelope.
const BASE64_ENCODING: &str = "base64";

/// The filesystem, shell, and code factories of the `aio` sandbox type.
///
/// Every factory builds its own [`AioClient`] from the endpoint and
/// configuration the registry hands it: the registry calls each factory
/// separately, so the three operation types of one sandbox may legitimately be
/// resolved against different endpoints.
#[must_use]
pub(crate) fn provider_set() -> SandboxProviderSet {
    SandboxProviderSet::new()
        .with_fs(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn FsOperation> = Arc::new(AioFs { client });
                Ok(operation)
            },
        ))
        .with_shell(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn ShellOperation> = Arc::new(AioShell { client });
                Ok(operation)
            },
        ))
        .with_code(Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let client = Arc::new(AioClient::new(endpoint, config)?);
                let operation: Arc<dyn CodeOperation> = Arc::new(AioCode { client });
                Ok(operation)
            },
        ))
}

/// The envelope client shared by the three operation wrappers.
pub(crate) struct AioClient {
    /// HTTP client carrying the base URL, auth headers, and auth query.
    http: SandboxHttp,
    /// Path the envelope is posted to.
    invoke_path: String,
    /// Resolved isolation key sent with every request (`null` when absent).
    isolation_key: Option<String>,
}

impl AioClient {
    /// Builds a client for `endpoint`, honoring `launcher_config` overrides.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config)
    /// when neither the endpoint nor the launcher configuration carries a base
    /// URL, or when the HTTP client cannot be built.
    pub(crate) fn new(endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig) -> Result<Self> {
        let http = SandboxHttp::new(PROVIDER, endpoint, config)?;
        Ok(Self {
            invoke_path: invoke_path(config),
            isolation_key: endpoint.isolation_key.clone(),
            http,
        })
    }

    /// Posts one envelope and returns its `data` payload.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::Status`] with status `0` and the message
    /// `aio gateway code <code>: <message>` when the gateway answers a
    /// non-zero `code`, [`RemoteError::NotFound`] / [`RemoteError::Status`] for
    /// a non-2xx HTTP answer, and [`RemoteError::Transport`] when no usable
    /// response arrives.
    async fn invoke(
        &self,
        op_type: &str,
        method: &str,
        params: Value,
    ) -> std::result::Result<Value, RemoteError> {
        let envelope = GatewayInvokeRequest {
            op_type: op_type.to_string(),
            method: method.to_string(),
            params,
            isolation_key: self.isolation_key.clone(),
        };
        let request = self
            .http
            .request(reqwest::Method::POST, &self.invoke_path)
            .json(&envelope);
        let body = self.http.send_value(request, method).await?;
        let response: GatewayResponse = serde_json::from_value(body).map_err(|error| {
            RemoteError::Transport(format!(
                "{method}: could not decode the {PROVIDER} gateway response: {error}"
            ))
        })?;
        if response.is_success() {
            Ok(response.data.unwrap_or(Value::Null))
        } else {
            Err(RemoteError::Status {
                status: 0,
                message: format!(
                    "{PROVIDER} gateway code {}: {}",
                    response.code, response.message
                ),
            })
        }
    }
}

/// The envelope path, honoring the launcher's `invoke_path` override.
fn invoke_path(config: &SandboxGatewayConfig) -> String {
    let configured = config
        .launcher_config
        .as_ref()
        .and_then(|launcher| launcher.extra_params.get(INVOKE_PATH_PARAM))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty());
    match configured {
        Some(path) if path.starts_with('/') => path.to_string(),
        Some(path) => format!("/{path}"),
        None => DEFAULT_INVOKE_PATH.to_string(),
    }
}

/// Whether a probe failure means the remote path is absent.
///
/// The envelope signals a missing path as `code != 0` rather than as HTTP
/// `404`, so besides the transport status the gateway message is the only
/// signal available.
fn is_absent(error: &RemoteError) -> bool {
    if error.is_not_found() {
        return true;
    }
    let message = error.message().to_ascii_lowercase();
    [
        "not found",
        "no such file",
        "does not exist",
        "not exist",
        "no exists",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

/// A non-empty string field, if present.
fn text_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|text| !text.is_empty())
}

/// An unsigned integer field, accepting a numeric string too.
fn json_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|value| u64::try_from(value).ok())),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// A signed 32-bit integer field, accepting a numeric string too.
fn json_i32(value: Option<&Value>) -> Option<i32> {
    match value? {
        Value::Number(number) => number.as_i64().and_then(|value| i32::try_from(value).ok()),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}
