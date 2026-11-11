//! HTTP sandbox gateway client (feature `sandbox`).
//!
//! This module is the deferred seam to an external AIO container service. The reference
//! Python SDK does not speak HTTP inside `openjiuwen/core/sys_operation/`: its
//! `SandboxGateway` is an in-process async router and the real transports live in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/{aio,jiuwenbox,yuanrong}.py`.
//! The wire contract of those providers is an external service protocol this rewrite has
//! no specification for.
//!
//! What this module provides:
//!
//! - [`HttpSandboxGateway`]: builds a [`reqwest::Client`] from
//!   [`SandboxGatewayConfig`] and carries the [`SandboxEndpoint`] plus the request
//!   envelope ([`GatewayInvokeRequest`] → [`GatewayResponse`]).
//! - Three forwarding backends ([`HttpFsOperation`], [`HttpShellOperation`],
//!   [`HttpCodeOperation`]) implementing the same traits as the local backend and mapping
//!   every call onto one gateway invocation with the Python method name preserved.
//! - [`install`], which registers those backends in a [`SandboxProviderRegistry`], so a
//!   sandbox card compiles, validates, and fails loudly instead of silently doing nothing.
//!
//! The configuration and envelope types in [`crate::sandbox::config`] always compile; only
//! this client is behind `#[cfg(feature = "sandbox")]`.
//!
//! # Network behavior
//!
//! [`HttpSandboxGateway::probe`] is the **only** method that performs network I/O: it
//! sends a `GET` to the endpoint `base_url` and reports whether the gateway answers with a
//! 2xx status. [`HttpSandboxGateway::invoke`] — and therefore every backend method —
//! returns [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
//! without touching the network, naming the gap.
//!
//! # Deferred gaps
//!
//! - AIO container launcher lifecycle: create / pause / resume / delete.
//! - Endpoint discovery and `sandbox_id` assignment (a resolved
//!   [`SandboxEndpoint::sandbox_id`] is forwarded verbatim but never populated here).
//! - Isolation-key → container mapping (the key is only forwarded as
//!   [`GatewayInvokeRequest::isolation_key`]).
//! - Provider-specific request/response payload schemas: the parameter structs are
//!   serialized with their Python field names, and a successful response `data` is
//!   deserialized into the matching [`crate::fs::types`] / [`crate::shell::types`] /
//!   [`crate::code::types`] result type.
//! - The streaming wire format: a streamed call invokes provider method
//!   `"<method>_stream"` and expects a JSON array of chunks, which no provider in this
//!   build can return, so in practice exactly one error chunk is yielded.
//! - Auth injection policy: [`SandboxGatewayConfig::auth_headers`] and
//!   [`SandboxGatewayConfig::auth_query_params`] are carried but never applied, printed,
//!   or logged; where and how they attach to a request is part of the external contract.
//!
//! Registering real providers is `openjiuwen-extensions`' job through
//! [`crate::sandbox::provider`]; [`install`] exists so this feature can be exercised
//! end-to-end with the stub gateway.
//!
//! # Parameter mapping
//!
//! Every unary method serializes its parameter struct with `serde_json` and sends it as
//! [`GatewayInvokeRequest::params`], keyed by `op_type` (`"fs"`, `"shell"`, `"code"`) and
//! the Python method name (`"read_file"`, `"execute_cmd"`, ...). The sole exception is
//! [`ReadFileParams::only_read`], an in-process lock hint with no remote equivalent, which
//! is dropped before serialization. A successful [`GatewayResponse`] (`code == 0`) must
//! carry the result payload in `data`, deserialized into the Python-shaped data type.

// Deferred Python sources (read-only reference; no Rust equivalent implemented yet):
//   openjiuwen/extensions/sys_operation/sandbox/providers/aio.py
//   openjiuwen/extensions/sys_operation/sandbox/providers/jiuwenbox.py
//   openjiuwen/extensions/sys_operation/sandbox/providers/yuanrong.py
//   openjiuwen/extensions/sys_operation/sandbox/gateway/*.py
//   openjiuwen/extensions/sys_operation/sandbox/launchers/*.py

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::code::types::{CodeChunk, CodeExecParams, CodeOutput};
use crate::fs::types::{
    DownloadFileData, DownloadFileParams, ListDirsParams, ListFilesData, ListFilesParams,
    ReadChunkData, ReadFileData, ReadFileParams, SearchFilesData, SearchFilesParams,
    TransferChunkData, UploadFileData, UploadFileParams, WriteFileData, WriteFileParams,
};
use crate::operation::{CodeOperation, FsOperation, OpStream, ShellOperation};
use crate::result::OpResult;
use crate::sandbox::config::{
    GatewayInvokeRequest, GatewayResponse, SandboxEndpoint, SandboxGatewayConfig,
};
use crate::sandbox::provider::{
    CodeProviderFactory, FsProviderFactory, SandboxProviderRegistry, ShellProviderFactory,
};
use crate::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
};
use crate::status::{sandbox_gateway_message, SysOpStatus};

/// Timeout applied when [`SandboxGatewayConfig::timeout_seconds`] is `0`.
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 30;

/// Operation kind of filesystem calls.
const OP_TYPE_FS: &str = "fs";

/// Operation kind of shell calls.
const OP_TYPE_SHELL: &str = "shell";

/// Operation kind of code calls.
const OP_TYPE_CODE: &str = "code";

/// The effective client timeout; `0` means "use the 30 s default".
const fn effective_timeout_secs(config: &SandboxGatewayConfig) -> u64 {
    if config.timeout_seconds == 0 {
        DEFAULT_HTTP_TIMEOUT_SECS
    } else {
        config.timeout_seconds
    }
}

/// An HTTP client bound to one resolved sandbox endpoint.
///
/// The gateway owns the [`SandboxEndpoint`] (including the resolved isolation key) and the
/// [`SandboxGatewayConfig`] used to build the client. The config's `auth_headers` and
/// `auth_query_params` are never logged.
pub struct HttpSandboxGateway {
    endpoint: SandboxEndpoint,
    config: SandboxGatewayConfig,
    client: reqwest::Client,
}

impl HttpSandboxGateway {
    /// Builds a client for `endpoint`, applying the configured timeout.
    ///
    /// The client uses a `openjiuwen-sysop/<version>` user agent and the timeout from
    /// `config` (`0` falls back to the 30 s default). No request is sent here.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport)
    /// when the underlying [`reqwest::Client`] cannot be built (for example an invalid
    /// user agent or TLS backend initialization failure).
    pub fn new(endpoint: SandboxEndpoint, config: SandboxGatewayConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(effective_timeout_secs(&config)))
            .user_agent(concat!("openjiuwen-sysop/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| {
                Error::transport(format!(
                    "failed to build sandbox gateway HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            endpoint,
            config,
            client,
        })
    }

    /// The endpoint this gateway forwards to.
    #[must_use]
    pub const fn endpoint(&self) -> &SandboxEndpoint {
        &self.endpoint
    }

    /// The configuration this gateway was built from.
    #[must_use]
    pub const fn config(&self) -> &SandboxGatewayConfig {
        &self.config
    }

    /// Forwards one request to the sandbox provider.
    ///
    /// The remote protocol is not implemented, so this always fails with
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) and
    /// performs **no** network I/O. See the module documentation for the concrete gaps.
    ///
    /// # Errors
    ///
    /// Always returns
    /// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) until
    /// the external AIO container contract is implemented.
    // The async signature is the eventual provider-call shape; today it returns at once.
    #[allow(clippy::unused_async)]
    #[allow(clippy::unused_async_trait_impl)]
    pub async fn invoke(&self, request: GatewayInvokeRequest) -> Result<GatewayResponse> {
        tracing::debug!(
            op_type = %request.op_type,
            method = %request.method,
            isolation_key_present = %request.isolation_key.is_some(),
            "sandbox gateway invoke is deferred; no network I/O is performed"
        );
        Err(Error::unsupported(format!(
            "sandbox gateway protocol is not implemented: the AIO container HTTP contract \
             is an external service protocol (deferred: launcher lifecycle \
             create/pause/resume/delete, endpoint discovery, provider-specific payload \
             mapping); op_type: {}, method: {}, isolation_key present: {}",
            request.op_type,
            request.method,
            request.isolation_key.is_some()
        )))
    }

    /// Probes the gateway by issuing a `GET` on the endpoint `base_url`.
    ///
    /// This is the **only** network call in this module; [`Self::invoke`] never sends a
    /// request. The configured timeout applies to the whole probe.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport) when
    ///   `base_url` is empty or blank, when the request cannot be sent, or when the
    ///   gateway answers with a non-2xx status.
    /// - [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout) when the
    ///   request exceeds the configured timeout.
    pub async fn probe(&self) -> Result<SandboxEndpoint> {
        let url = self.endpoint.base_url.trim();
        if url.is_empty() {
            return Err(Error::transport("sandbox endpoint base_url is empty"));
        }
        let response = self.client.get(url).send().await.map_err(|error| {
            if error.is_timeout() {
                Error::timeout(format!("sandbox gateway probe timed out at {url}: {error}"))
            } else {
                Error::transport(format!("sandbox gateway probe failed at {url}: {error}"))
            }
        })?;
        let status = response.status();
        if status.is_success() {
            Ok(self.endpoint.clone())
        } else {
            Err(Error::transport(format!(
                "sandbox gateway probe failed: HTTP {status} at {url}"
            )))
        }
    }
}

/// Forwards filesystem operations to an [`HttpSandboxGateway`].
pub struct HttpFsOperation {
    gateway: Arc<HttpSandboxGateway>,
}

impl HttpFsOperation {
    /// Binds an operation backend to `gateway`.
    #[must_use]
    pub fn new(gateway: Arc<HttpSandboxGateway>) -> Self {
        Self { gateway }
    }

    /// The gateway this backend forwards to.
    #[must_use]
    pub const fn gateway(&self) -> &Arc<HttpSandboxGateway> {
        &self.gateway
    }
}

#[async_trait]
impl FsOperation for HttpFsOperation {
    fn supports_conditional_write(&self) -> bool {
        self.gateway.config().supports_conditional_write
    }

    async fn read_file(&self, params: ReadFileParams) -> OpResult<ReadFileData> {
        invoke_param_stripping(
            &self.gateway,
            OP_TYPE_FS,
            "read_file",
            &params,
            &["only_read"],
        )
        .await
    }

    fn read_file_stream(self: Arc<Self>, params: ReadFileParams) -> OpStream<ReadChunkData> {
        stream_param::<ReadFileParams, ReadChunkData>(
            Arc::clone(&self.gateway),
            OP_TYPE_FS,
            "read_file",
            params,
            &["only_read"],
        )
    }

    async fn write_file(&self, params: WriteFileParams) -> OpResult<WriteFileData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "write_file", &params).await
    }

    async fn upload_file(&self, params: UploadFileParams) -> OpResult<UploadFileData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "upload_file", &params).await
    }

    fn upload_file_stream(
        self: Arc<Self>,
        params: UploadFileParams,
    ) -> OpStream<TransferChunkData> {
        stream_param::<UploadFileParams, TransferChunkData>(
            Arc::clone(&self.gateway),
            OP_TYPE_FS,
            "upload_file",
            params,
            &[],
        )
    }

    async fn download_file(&self, params: DownloadFileParams) -> OpResult<DownloadFileData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "download_file", &params).await
    }

    fn download_file_stream(
        self: Arc<Self>,
        params: DownloadFileParams,
    ) -> OpStream<TransferChunkData> {
        stream_param::<DownloadFileParams, TransferChunkData>(
            Arc::clone(&self.gateway),
            OP_TYPE_FS,
            "download_file",
            params,
            &[],
        )
    }

    async fn list_files(&self, params: ListFilesParams) -> OpResult<ListFilesData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "list_files", &params).await
    }

    async fn list_directories(&self, params: ListDirsParams) -> OpResult<ListFilesData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "list_directories", &params).await
    }

    async fn search_files(&self, params: SearchFilesParams) -> OpResult<SearchFilesData> {
        invoke_param(&self.gateway, OP_TYPE_FS, "search_files", &params).await
    }
}

/// Forwards shell operations to an [`HttpSandboxGateway`].
pub struct HttpShellOperation {
    gateway: Arc<HttpSandboxGateway>,
}

impl HttpShellOperation {
    /// Binds an operation backend to `gateway`.
    #[must_use]
    pub fn new(gateway: Arc<HttpSandboxGateway>) -> Self {
        Self { gateway }
    }

    /// The gateway this backend forwards to.
    #[must_use]
    pub const fn gateway(&self) -> &Arc<HttpSandboxGateway> {
        &self.gateway
    }
}

#[async_trait]
impl ShellOperation for HttpShellOperation {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        invoke_param(&self.gateway, OP_TYPE_SHELL, "execute_cmd", &params).await
    }

    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk> {
        stream_param::<ShellExecParams, ShellChunk>(
            Arc::clone(&self.gateway),
            OP_TYPE_SHELL,
            "execute_cmd",
            params,
            &[],
        )
    }

    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        invoke_param(
            &self.gateway,
            OP_TYPE_SHELL,
            "execute_cmd_background",
            &params,
        )
        .await
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        invoke_param(&self.gateway, OP_TYPE_SHELL, "execute_argv", &params).await
    }
}

/// Forwards code operations to an [`HttpSandboxGateway`].
pub struct HttpCodeOperation {
    gateway: Arc<HttpSandboxGateway>,
}

impl HttpCodeOperation {
    /// Binds an operation backend to `gateway`.
    #[must_use]
    pub fn new(gateway: Arc<HttpSandboxGateway>) -> Self {
        Self { gateway }
    }

    /// The gateway this backend forwards to.
    #[must_use]
    pub const fn gateway(&self) -> &Arc<HttpSandboxGateway> {
        &self.gateway
    }
}

#[async_trait]
impl CodeOperation for HttpCodeOperation {
    async fn execute_code(&self, params: CodeExecParams) -> OpResult<CodeOutput> {
        invoke_param(&self.gateway, OP_TYPE_CODE, "execute_code", &params).await
    }

    fn execute_code_stream(self: Arc<Self>, params: CodeExecParams) -> OpStream<CodeChunk> {
        stream_param::<CodeExecParams, CodeChunk>(
            Arc::clone(&self.gateway),
            OP_TYPE_CODE,
            "execute_code",
            params,
            &[],
        )
    }
}

/// Maps one gateway round-trip onto an [`OpResult`].
///
/// Failure mapping (every failure is a value with code
/// [`SysOpStatus::SandboxGatewayError`] and the Python gateway message template):
///
/// | Cause | `reason` in `sandbox gateway error, operation: <method>, reason: ...` |
/// |---|---|
/// | parameter serialization | `failed to serialize params: <error>` |
/// | [`HttpSandboxGateway::invoke`] error | the shared error's message |
/// | `response.code != 0` | `provider returned code <code>: <message>` |
/// | `data` cannot be deserialized | `invalid provider payload: <error>` |
/// | `data` is absent | `invalid provider payload: provider returned no data` |
async fn invoke_value<D: DeserializeOwned>(
    gateway: &HttpSandboxGateway,
    op_type: &str,
    method: &str,
    params: Value,
) -> OpResult<D> {
    let request = GatewayInvokeRequest {
        op_type: op_type.to_string(),
        method: method.to_string(),
        params,
        isolation_key: gateway.endpoint().isolation_key.clone(),
    };
    let response = match gateway.invoke(request).await {
        Ok(response) => response,
        Err(error) => {
            return OpResult::failure(
                SysOpStatus::SandboxGatewayError,
                sandbox_gateway_message(method, &error.message()),
            );
        }
    };
    if !response.is_success() {
        return OpResult::failure(
            SysOpStatus::SandboxGatewayError,
            sandbox_gateway_message(
                method,
                &format!(
                    "provider returned code {}: {}",
                    response.code, response.message
                ),
            ),
        );
    }
    let Some(data) = response.data else {
        return OpResult::failure(
            SysOpStatus::SandboxGatewayError,
            sandbox_gateway_message(
                method,
                "invalid provider payload: provider returned no data",
            ),
        );
    };
    match serde_json::from_value::<D>(data) {
        Ok(value) => OpResult::success(value),
        Err(error) => OpResult::failure(
            SysOpStatus::SandboxGatewayError,
            sandbox_gateway_message(method, &format!("invalid provider payload: {error}")),
        ),
    }
}

/// Serializes `params` and forwards them through [`invoke_value`].
async fn invoke_param<P, D>(
    gateway: &HttpSandboxGateway,
    op_type: &str,
    method: &str,
    params: &P,
) -> OpResult<D>
where
    P: Serialize + ?Sized,
    D: DeserializeOwned,
{
    invoke_param_stripping(gateway, op_type, method, params, &[]).await
}

/// Like [`invoke_param`], dropping the named top-level fields before serialization.
async fn invoke_param_stripping<P, D>(
    gateway: &HttpSandboxGateway,
    op_type: &str,
    method: &str,
    params: &P,
    strip: &[&str],
) -> OpResult<D>
where
    P: Serialize + ?Sized,
    D: DeserializeOwned,
{
    let mut value = match serde_json::to_value(params) {
        Ok(value) => value,
        Err(error) => {
            return OpResult::failure(
                SysOpStatus::SandboxGatewayError,
                sandbox_gateway_message(method, &format!("failed to serialize params: {error}")),
            );
        }
    };
    if let Value::Object(map) = &mut value {
        for key in strip {
            map.remove(*key);
        }
    }
    invoke_value(gateway, op_type, method, value).await
}

/// Builds a stream that invokes provider method `"<method>_stream"` and replays the
/// returned chunk array.
///
/// The remote streaming wire format is deferred, so in practice the single invocation
/// fails and exactly one error chunk is yielded. On success each array element is yielded
/// as its own [`OpResult::success`], and the stream ends; it never yields a failure after
/// a successful element.
fn stream_param<P, T>(
    gateway: Arc<HttpSandboxGateway>,
    op_type: &'static str,
    method: &'static str,
    params: P,
    strip: &'static [&'static str],
) -> OpStream<T>
where
    P: Serialize + Send + Sync + 'static,
    T: DeserializeOwned + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let stream_method = format!("{method}_stream");
        let result = invoke_param_stripping::<P, Vec<T>>(
            &gateway,
            op_type,
            &stream_method,
            &params,
            strip,
        )
        .await;
        if result.is_success() {
            if let Some(items) = result.data {
                for item in items {
                    yield OpResult::success(item);
                }
            }
        } else {
            yield OpResult {
                code: result.code,
                message: result.message,
                data: None,
            };
        }
    })
}

/// Registers the HTTP gateway providers for `sandbox_type` in `registry`.
///
/// A filesystem, shell, and code factory are installed unless one is already registered for
/// `sandbox_type`; a duplicate-registration error is treated as success, so repeated calls
/// are idempotent. The factories ignore the install-time `endpoint`/`config` prototype:
/// each gateway is rebuilt from the endpoint and config handed to
/// [`SandboxProviderRegistry::create_fs`] and friends, i.e. the creation-time endpoint
/// (including the resolved isolation key) wins. The install-time arguments are validated
/// for a non-blank `base_url` only.
///
/// # Errors
///
/// Returns [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport) when
/// `endpoint.base_url` is empty or blank, and propagates any registration error that is
/// not a duplicate.
pub fn install(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    endpoint: &SandboxEndpoint,
    config: &SandboxGatewayConfig,
) -> Result<()> {
    tracing::debug!(
        sandbox_type,
        timeout_seconds = effective_timeout_secs(config),
        "installing HTTP sandbox gateway providers"
    );
    if endpoint.base_url.trim().is_empty() {
        return Err(Error::transport("sandbox gateway base_url is empty"));
    }
    if !registry.has_fs(sandbox_type) {
        let factory: FsProviderFactory = Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let gateway = Arc::new(HttpSandboxGateway::new(endpoint.clone(), config.clone())?);
                Ok(Arc::new(HttpFsOperation::new(gateway)) as Arc<dyn FsOperation>)
            },
        );
        ignore_already_registered(registry.register_fs(sandbox_type, factory))?;
    }
    if !registry.has_shell(sandbox_type) {
        let factory: ShellProviderFactory = Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let gateway = Arc::new(HttpSandboxGateway::new(endpoint.clone(), config.clone())?);
                Ok(Arc::new(HttpShellOperation::new(gateway)) as Arc<dyn ShellOperation>)
            },
        );
        ignore_already_registered(registry.register_shell(sandbox_type, factory))?;
    }
    if !registry.has_code(sandbox_type) {
        let factory: CodeProviderFactory = Arc::new(
            |endpoint: &SandboxEndpoint, config: &SandboxGatewayConfig| {
                let gateway = Arc::new(HttpSandboxGateway::new(endpoint.clone(), config.clone())?);
                Ok(Arc::new(HttpCodeOperation::new(gateway)) as Arc<dyn CodeOperation>)
            },
        );
        ignore_already_registered(registry.register_code(sandbox_type, factory))?;
    }
    Ok(())
}

/// Treats a duplicate registration as success so [`install`] is idempotent.
fn ignore_already_registered(result: Result<()>) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.message().contains("already registered") => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;

    fn endpoint() -> SandboxEndpoint {
        SandboxEndpoint {
            base_url: "http://127.0.0.1:1".to_string(),
            sandbox_id: Some("sandbox-1".to_string()),
            isolation_key: Some("session_pre_deploy_aio_s1".to_string()),
        }
    }

    fn gateway() -> HttpSandboxGateway {
        HttpSandboxGateway::new(endpoint(), SandboxGatewayConfig::default()).unwrap()
    }

    #[test]
    fn new_applies_timeout_and_exposes_config() {
        let gateway = gateway();
        assert_eq!(gateway.config().timeout_seconds, 30);
        assert_eq!(gateway.endpoint().base_url, "http://127.0.0.1:1");
        assert_eq!(gateway.endpoint().sandbox_id.as_deref(), Some("sandbox-1"));
        assert_eq!(
            gateway.endpoint().isolation_key.as_deref(),
            Some("session_pre_deploy_aio_s1")
        );

        let config = SandboxGatewayConfig {
            timeout_seconds: 7,
            ..SandboxGatewayConfig::default()
        };
        let gateway = HttpSandboxGateway::new(endpoint(), config).unwrap();
        assert_eq!(gateway.config().timeout_seconds, 7);

        // A zero timeout is normalized to the default when the client is built.
        assert_eq!(effective_timeout_secs(gateway.config()), 7);
        assert_eq!(
            effective_timeout_secs(&SandboxGatewayConfig {
                timeout_seconds: 0,
                ..SandboxGatewayConfig::default()
            }),
            DEFAULT_HTTP_TIMEOUT_SECS
        );
    }

    #[tokio::test]
    async fn invoke_reports_unsupported_protocol() {
        // `base_url` points at a closed local port: `invoke` must never contact it.
        let error = gateway()
            .invoke(GatewayInvokeRequest {
                op_type: OP_TYPE_FS.to_string(),
                method: "read_file".to_string(),
                params: serde_json::json!({"path": "x"}),
                isolation_key: Some("key".to_string()),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().contains("not implemented"));
        assert!(error.message().contains("read_file"));
        assert!(error.message().contains("isolation_key present: true"));
    }

    #[tokio::test]
    async fn install_registers_all_three_providers() {
        let registry = SandboxProviderRegistry::new();
        let config = SandboxGatewayConfig {
            supports_conditional_write: true,
            ..SandboxGatewayConfig::default()
        };
        install(&registry, "aio_http_mock", &endpoint(), &config).unwrap();
        assert!(registry.has_fs("aio_http_mock"));
        assert!(registry.has_shell("aio_http_mock"));
        assert!(registry.has_code("aio_http_mock"));

        let fs = registry
            .create_fs("aio_http_mock", &endpoint(), &config)
            .unwrap();
        assert!(fs.supports_conditional_write());
        let result = fs.read_file(ReadFileParams::text("x")).await;
        assert_eq!(result.code, 199_007);
        assert!(result
            .message
            .contains("sandbox gateway error, operation: read_file"));
        assert!(result.message.contains("not implemented"));

        let without_conditional = SandboxGatewayConfig::default();
        let fs = registry
            .create_fs("aio_http_mock", &endpoint(), &without_conditional)
            .unwrap();
        assert!(!fs.supports_conditional_write());
    }

    #[tokio::test]
    async fn install_is_idempotent() {
        let registry = SandboxProviderRegistry::new();
        install(
            &registry,
            "aio_http_mock",
            &endpoint(),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();
        install(
            &registry,
            "aio_http_mock",
            &endpoint(),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();
        assert!(registry.has_fs("aio_http_mock"));
        assert!(registry.has_shell("aio_http_mock"));
        assert!(registry.has_code("aio_http_mock"));
    }

    #[test]
    fn install_rejects_empty_base_url() {
        let registry = SandboxProviderRegistry::new();
        let error = install(
            &registry,
            "aio_http_mock",
            &SandboxEndpoint::default(),
            &SandboxGatewayConfig::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.message().contains("base_url is empty"));
        assert!(!registry.has_fs("aio_http_mock"));
    }

    #[tokio::test]
    async fn probe_rejects_empty_base_url() {
        let gateway =
            HttpSandboxGateway::new(SandboxEndpoint::default(), SandboxGatewayConfig::default())
                .unwrap();
        let error = gateway.probe().await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.message().contains("base_url is empty"));
    }

    #[tokio::test]
    async fn stream_yields_single_error_chunk() {
        let registry = SandboxProviderRegistry::new();
        install(
            &registry,
            "aio_http_mock",
            &endpoint(),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();

        let shell = registry
            .create_shell(
                "aio_http_mock",
                &endpoint(),
                &SandboxGatewayConfig::default(),
            )
            .unwrap();
        let mut chunks = shell.execute_cmd_stream(ShellExecParams::new("echo hi"));
        let first = chunks.next().await.unwrap();
        assert_eq!(first.code, 199_007);
        assert!(first
            .message
            .contains("sandbox gateway error, operation: execute_cmd_stream"));
        assert!(chunks.next().await.is_none());

        let fs = registry
            .create_fs(
                "aio_http_mock",
                &endpoint(),
                &SandboxGatewayConfig::default(),
            )
            .unwrap();
        let mut chunks = fs.read_file_stream(ReadFileParams::text("x"));
        let first = chunks.next().await.unwrap();
        assert_eq!(first.code, 199_007);
        assert!(first
            .message
            .contains("sandbox gateway error, operation: read_file_stream"));
        assert!(chunks.next().await.is_none());
    }

    /// The single network test in this module; ignored so the offline suite stays hermetic.
    ///
    /// Point `base_url` at a reachable gateway and run
    /// `cargo test -p openjiuwen-sysop --features sandbox -- --ignored
    /// probe_against_live_gateway`. Every other test in this module never opens a socket.
    #[tokio::test]
    #[ignore = "requires a live sandbox gateway on the network"]
    async fn probe_against_live_gateway() {
        let gateway = HttpSandboxGateway::new(
            SandboxEndpoint {
                base_url: "http://127.0.0.1:8080".to_string(),
                ..SandboxEndpoint::default()
            },
            SandboxGatewayConfig::default(),
        )
        .unwrap();
        let probed = gateway.probe().await.unwrap();
        assert_eq!(probed.base_url.trim(), gateway.endpoint().base_url.trim());
    }
}
