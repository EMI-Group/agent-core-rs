//! MCP clients: server configuration, the [`McpClient`] seam, reconnect policy,
//! and the `rmcp`-backed [`RmcpClient`].
//!
//! Ports `openjiuwen/core/foundation/tool/mcp/` (`base.py` `McpServerConfig`,
//! `client/mcp_client.py` `McpClient`, `client/reconnect.py`).
//!
//! [`McpClient`] is the seam every tool uses: [`RmcpClient`] implements it on
//! top of `rmcp` (stdio through `rmcp`'s child-process transport, HTTP and SSE
//! through the transports in `super::http`), while tests substitute
//! `FakeMcpClient`. Connections are lazy — the first operation connects — and
//! transport failures are retried with the reconnect policy below.
//!
//! Resource subscriptions (`resources/subscribe`) are optional in MCP, so they
//! are defaulted on the trait ([`McpClient::subscribe_resource`],
//! [`McpClient::unsubscribe_resource`], [`McpClient::supports_resource_subscriptions`])
//! and gated on the capability the server negotiated in its `initialize` result;
//! the gate and the request construction live in `client::subscription`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::config::env::mcp_reconnect_attempts;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::redact::sanitize_headers;
use openjiuwen_common::utils::id::uuid_hex;
use rmcp::model::{CallToolRequestParam, ReadResourceRequestParam};
use rmcp::service::{serve_client, Peer, RoleClient, RunningService, ServiceError};
use rmcp::transport::TokioChildProcess;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

mod subscription;

use super::http::sse::SseTransport;
use super::http::{ServerStreamState, StreamableHttpTransport};
use super::result::{normalize_call_tool_result, McpToolResult};

/// The kind of transport used to reach an MCP server.
///
/// Python parity: the `client_type` string of `McpServerConfig`
/// (`"stdio"`, `"sse"`, `"streamable-http"`, `"openapi"`, `"playwright"`).
/// Values are matched case-insensitively and both dash and underscore spellings
/// are accepted; `"http"` is an accepted alias for
/// [`McpClientType::StreamableHttp`], which is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum McpClientType {
    /// Spawn a local process and speak MCP over its stdio.
    Stdio,
    /// Connect to a Server-Sent-Events endpoint (legacy HTTP+SSE transport).
    Sse,
    /// Connect to a streamable-HTTP MCP endpoint (the default).
    #[default]
    StreamableHttp,
    /// Derive one tool per operation from an `OpenAPI` document (no JSON-RPC).
    #[serde(rename = "openapi")]
    OpenApi,
    /// Drive a Playwright browser session — deferred, construction is unsupported.
    #[serde(rename = "playwright")]
    Playwright,
}

impl McpClientType {
    /// Returns the canonical `snake_case` spelling of this transport.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Sse => "sse",
            Self::StreamableHttp => "streamable_http",
            Self::OpenApi => "openapi",
            Self::Playwright => "playwright",
        }
    }

    /// Resolves a transport name, accepting the Python aliases.
    ///
    /// Accepts `stdio`, `sse`, `streamable_http`, `streamable-http`, `http`,
    /// `openapi`, `open_api`, and `playwright` in any ASCII case.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown transport name.
    pub fn parse(value: &str) -> Result<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "stdio" => Ok(Self::Stdio),
            "sse" => Ok(Self::Sse),
            "streamable_http" | "http" => Ok(Self::StreamableHttp),
            "openapi" | "open_api" => Ok(Self::OpenApi),
            "playwright" => Ok(Self::Playwright),
            _ => Err(Error::validation(format!(
                "unsupported MCP client type `{value}`; expected stdio, sse, streamable-http, \
                 openapi, or playwright"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for McpClientType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// Configuration of one MCP server.
///
/// Python parity: `McpServerConfig` in
/// `openjiuwen/core/foundation/tool/mcp/base.py`. Every field has a serde
/// default so partial JSON deserializes; `server_id` defaults to a fresh
/// `uuid4().hex`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpServerConfig {
    /// Resource-manager key; first segment of every generated tool id.
    pub server_id: String,
    /// Human-readable server name; second segment of every generated tool id.
    pub server_name: String,
    /// Executable (stdio) or endpoint URL (SSE / streamable HTTP).
    pub server_path: String,
    /// Transport kind.
    pub client_type: McpClientType,
    /// Transport parameters: `command`, `args`, `env`, `cwd` for stdio.
    pub params: Value,
    /// Headers sent with every HTTP request; never logged.
    pub auth_headers: BTreeMap<String, String>,
    /// Query parameters appended to HTTP endpoint URLs.
    pub auth_query_params: BTreeMap<String, String>,
    /// Whether image blocks are forwarded as multimodal data instead of text.
    pub include_image_content: bool,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            server_id: uuid_hex(),
            server_name: String::new(),
            server_path: String::new(),
            client_type: McpClientType::StreamableHttp,
            params: json!({}),
            auth_headers: BTreeMap::new(),
            auth_query_params: BTreeMap::new(),
            include_image_content: false,
        }
    }
}

impl McpServerConfig {
    /// Creates a stdio configuration for `server_name` running `command`.
    #[must_use]
    pub fn stdio(server_name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            server_path: command.into(),
            client_type: McpClientType::Stdio,
            ..Self::default()
        }
    }

    /// Creates an HTTP configuration for `server_name` at `url`.
    #[must_use]
    pub fn http(server_name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            server_path: url.into(),
            client_type: McpClientType::StreamableHttp,
            ..Self::default()
        }
    }

    /// Returns this configuration with a fixed `server_id`.
    #[must_use]
    pub fn with_server_id(mut self, server_id: impl Into<String>) -> Self {
        self.server_id = server_id.into();
        self
    }

    /// Returns this configuration with one extra auth header.
    #[must_use]
    pub fn with_auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth_headers.insert(name.into(), value.into());
        self
    }

    /// Returns this configuration with one extra auth query parameter.
    #[must_use]
    pub fn with_auth_query_param(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.auth_query_params.insert(name.into(), value.into());
        self
    }

    /// Returns this configuration with image forwarding enabled.
    #[must_use]
    pub fn with_include_image_content(mut self, include: bool) -> Self {
        self.include_image_content = include;
        self
    }

    /// Returns the stdio executable: `params.command`, else `server_path`.
    #[must_use]
    pub fn command(&self) -> Option<&str> {
        self.params
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .or_else(|| {
                let path = self.server_path.trim();
                (!path.is_empty()).then_some(path)
            })
    }

    /// Returns the stdio arguments from `params.args` (non-string entries ignored).
    #[must_use]
    pub fn args(&self) -> Vec<String> {
        self.params
            .get("args")
            .and_then(Value::as_array)
            .map(|args| {
                args.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the stdio environment from `params.env` (non-string values ignored).
    #[must_use]
    pub fn env(&self) -> BTreeMap<String, String> {
        self.params
            .get("env")
            .and_then(Value::as_object)
            .map(|env| {
                env.iter()
                    .filter_map(|(key, value)| {
                        value.as_str().map(|value| (key.clone(), value.to_owned()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the stdio working directory from `params.cwd`.
    #[must_use]
    pub fn cwd(&self) -> Option<String> {
        self.params
            .get("cwd")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    }

    /// Returns sanitized auth headers, safe for logs.
    #[must_use]
    pub fn redacted_headers(&self) -> BTreeMap<String, String> {
        sanitize_headers(&self.auth_headers)
    }
}

/// A tool advertised by an MCP server.
///
/// Python parity: the `McpToolCard` projection built by every Python transport's
/// `list_tools` (`name`, `server_name`, `description`, `input_params`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpRemoteTool {
    /// Remote tool name, as the server advertises it.
    pub name: String,
    /// Human-readable description (`""` when the server omits it).
    pub description: String,
    /// JSON Schema of the accepted arguments (`{}` when the server omits it).
    pub input_schema: Value,
}

impl McpRemoteTool {
    /// Creates a remote tool description.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// A notification pushed by an MCP server over its server-initiated stream.
///
/// Python parity: the payload a transport hands to its registered notification
/// handlers (`method` plus the raw `params`). Notifications are buffered by the
/// client (bounded to 64 entries, the oldest dropped with a `tracing::warn!`) and
/// drained by [`McpClient::take_notifications`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpNotification {
    /// JSON-RPC method of the notification, e.g. `notifications/tools/list_changed`.
    pub method: String,
    /// Raw `params` object of the notification (`null` when the server sent none).
    pub params: Value,
}

/// A provider-neutral MCP client.
///
/// Python parity: `McpClient` in
/// `openjiuwen/core/foundation/tool/mcp/client/mcp_client.py`. Implementations
/// connect lazily: every operation ensures a session first.
#[async_trait]
pub trait McpClient: Send + Sync + 'static {
    /// Returns the configured server name.
    fn server_name(&self) -> &str;

    /// Returns the configured server id (`""` when the implementation has none).
    #[allow(clippy::unnecessary_literal_bound)]
    fn server_id(&self) -> &str {
        ""
    }

    /// Returns whether image blocks are forwarded as multimodal data.
    fn include_image_content(&self) -> bool {
        false
    }

    /// Opens the session if it is not open yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the transport cannot be established and
    /// [`Error::Validation`] when the configuration is unusable.
    async fn connect(&self) -> Result<()>;

    /// Closes the session, releasing any process or socket.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Mcp`] when shutdown fails after the session was already
    /// unusable; the session is dropped either way.
    async fn disconnect(&self) -> Result<()>;

    /// Returns whether a session object is currently held.
    async fn is_connected(&self) -> bool;

    /// Lists every tool the server advertises, following pagination.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the session cannot be (re)established
    /// and [`Error::Mcp`] when the server rejects a request.
    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>>;

    /// Calls `tool_name` with `arguments`.
    ///
    /// `timeout` bounds a single attempt; a timed-out call is never retried.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] on a deadline, [`Error::Transport`] on
    /// reconnectable transport failures, and [`Error::Mcp`] on rejections.
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
    ) -> Result<McpToolResult> {
        self.call_tool_with(tool_name, arguments, timeout, self.include_image_content())
            .await
    }

    /// Calls `tool_name`, choosing whether image blocks become multimodal data.
    ///
    /// # Errors
    ///
    /// Same as [`McpClient::call_tool`].
    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
        include_image_content: bool,
    ) -> Result<McpToolResult>;

    /// Lists the resources the server advertises.
    ///
    /// # Errors
    ///
    /// Defaults to [`Error::Unsupported`]; transports without resource support
    /// keep that default.
    async fn list_resources(&self) -> Result<Vec<Value>> {
        Err(Error::unsupported(
            "this MCP client does not support listing resources",
        ))
    }

    /// Reads one resource by URI.
    ///
    /// # Errors
    ///
    /// Defaults to [`Error::Unsupported`].
    async fn read_resource(&self, uri: &str) -> Result<Value> {
        let _ = uri;
        Err(Error::unsupported(
            "this MCP client does not support reading resources",
        ))
    }

    /// Subscribes to update notifications for the resource `uri`.
    ///
    /// Python parity: the `resources/subscribe` request of the MCP resource
    /// specification. Subscribing is optional in MCP, so this is a defaulted
    /// method: the default reports [`Error::Unsupported`] naming the transport,
    /// and implementations that speak the request additionally report
    /// [`McpClient::supports_resource_subscriptions`].
    ///
    /// The server must have advertised `capabilities.resources.subscribe` in its
    /// `initialize` result; a server that did not is an [`Error::Unsupported`]
    /// naming the missing capability, raised without sending the request.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a transport (or a server) without the
    /// capability, [`Error::Validation`] for a blank URI, [`Error::Transport`]
    /// when the session cannot be (re)established, and [`Error::Mcp`] when the
    /// server rejects the request.
    async fn subscribe_resource(&self, uri: &str) -> Result<()> {
        let _ = uri;
        Err(Error::unsupported(
            "this MCP transport does not support resource subscriptions: no \
             resources/subscribe request is implemented for it",
        ))
    }

    /// Cancels the update subscription of the resource `uri`.
    ///
    /// # Errors
    ///
    /// Same as [`McpClient::subscribe_resource`]; the default reports
    /// [`Error::Unsupported`] naming the transport.
    async fn unsubscribe_resource(&self, uri: &str) -> Result<()> {
        let _ = uri;
        Err(Error::unsupported(
            "this MCP transport does not support resource subscriptions: no \
             resources/unsubscribe request is implemented for it",
        ))
    }

    /// Reports whether the server negotiated the `resources/subscribe` capability.
    ///
    /// This is the negotiated capability, not a property of the transport: it is
    /// `false` before a session is established, when the server's `initialize`
    /// result does not advertise `capabilities.resources.subscribe`, and for
    /// transports without the request (the default).
    fn supports_resource_subscriptions(&self) -> bool {
        false
    }

    /// Drains the notifications the server pushed over its server-initiated stream.
    ///
    /// Notifications are removed by the call, oldest first, so each one is
    /// returned exactly once. The buffer is bounded (64 entries) and drops the
    /// oldest notification with a `tracing::warn!` when it overflows. The default
    /// implementation buffers nothing and returns an empty vector.
    ///
    /// # Errors
    ///
    /// The default implementation never fails; implementations that raise a
    /// transport error propagate it.
    async fn take_notifications(&self) -> Result<Vec<McpNotification>> {
        Ok(Vec::new())
    }

    /// Reports whether the server accepted the server-initiated `GET` stream.
    ///
    /// This is the negotiated capability: it is `false` before a session is
    /// established, when the server refused the channel (405/404/501), and for
    /// transports without such a channel (the default).
    fn supports_server_stream(&self) -> bool {
        false
    }
}

/// Builds a client for `config` without connecting.
///
/// Python parity: `ToolMgr._create_client` plus the client registry
/// (`"stdio"`, `"sse"`, `"streamable-http"`, `"openapi"`, `"playwright"`). The
/// returned client connects on its first operation. The Playwright transport is
/// deferred: it needs a browser runtime, so it is rejected here.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `config` cannot address its transport (a
/// stdio config without a command, an HTTP config without an absolute
/// http/https URL, or an `OpenAPI` config without a spec source), and
/// [`Error::Unsupported`] for the deferred Playwright transport.
pub fn connect_mcp_server(config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
    let client: Arc<dyn McpClient> = match config.client_type {
        McpClientType::OpenApi => Arc::new(super::openapi::OpenApiClient::new(config)?),
        McpClientType::Playwright => {
            tracing::debug!(
                server_name = %config.server_name,
                server_id = %config.server_id,
                client_type = config.client_type.as_str(),
                "playwright MCP transport is deferred"
            );
            return Err(Error::unsupported(
                "playwright MCP transport requires a browser runtime",
            ));
        }
        McpClientType::Stdio | McpClientType::Sse | McpClientType::StreamableHttp => {
            Arc::new(RmcpClient::new(config)?)
        }
    };
    tracing::debug!(
        server_name = %config.server_name,
        server_id = %config.server_id,
        client_type = config.client_type.as_str(),
        auth_headers = ?config.redacted_headers(),
        "created MCP client"
    );
    Ok(client)
}

/// The number of total connect attempts (first try included).
///
/// Python parity: `reconnect.py` attempt budget — `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`
/// when set, else 3. The value is never below 1.
#[must_use]
pub fn reconnect_attempt_budget() -> u32 {
    mcp_reconnect_attempts().max(1)
}

/// The backoff before connect attempt `attempt + 1`.
///
/// Python parity: `min(120, 5 * 2 ** min(attempt - 1, 5))` seconds, without the
/// Python jitter so the policy stays deterministic (and testable). Attempt `1`
/// is the first retry and waits 5 s; the delay saturates at 120 s.
#[must_use]
pub fn reconnect_delay(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(5);
    let seconds = 5_u64 << exponent;
    Duration::from_secs(seconds.min(120))
}

/// Returns whether `error` is worth another connect attempt.
///
/// Only transport failures reconnect: timeouts, cancellation, protocol
/// rejections, validation, and unsupported operations are final.
fn is_retryable(error: &Error) -> bool {
    matches!(error, Error::Transport(_))
}

/// The error a transport this client does not speak reports.
fn unsupported_transport(client_type: McpClientType) -> Error {
    Error::unsupported(format!(
        "`{}` is not an rmcp transport; build it through `connect_mcp_server`",
        client_type.as_str()
    ))
}
/// Maps a `rmcp` service error onto the shared error type.
fn map_service_error(error: ServiceError) -> Error {
    match error {
        ServiceError::McpError(error) => Error::mcp(error.to_string()),
        ServiceError::TransportSend(error) => Error::transport(error.to_string()),
        ServiceError::TransportClosed => Error::transport("transport closed"),
        ServiceError::UnexpectedResponse => Error::mcp("unexpected response type"),
        ServiceError::Cancelled { reason } => {
            let _ = reason;
            Error::Cancelled
        }
        ServiceError::Timeout { timeout } => {
            Error::timeout(format!("MCP request timed out after {timeout:?}"))
        }
        other => Error::mcp(other.to_string()),
    }
}

/// An [`McpClient`] implemented on top of `rmcp`.
///
/// The session is created lazily on the first operation and held until
/// [`McpClient::disconnect`]; after a transport failure the session is dropped so
/// the next attempt reconnects.
pub struct RmcpClient {
    config: McpServerConfig,
    connection: Mutex<Option<RunningService<RoleClient, ()>>>,
    connect_lock: Mutex<()>,
    server_stream: parking_lot::Mutex<Option<ServerStreamState>>,
    /// The `resources.subscribe` capability negotiated by the held session.
    ///
    /// Cached when the session is established because
    /// [`McpClient::supports_resource_subscriptions`] is synchronous while the
    /// session lives behind an async mutex; cleared whenever the session drops.
    resource_subscriptions: AtomicBool,
}

impl RmcpClient {
    /// Validates `config` and creates a client that connects on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unusable transport configuration and
    /// [`Error::Unsupported`] for the transports this client does not speak
    /// (`openapi`, `playwright`); [`connect_mcp_server`] routes those elsewhere,
    /// so the unsupported arms are defensive only.
    pub fn new(config: &McpServerConfig) -> Result<Self> {
        match config.client_type {
            McpClientType::Stdio => {
                if config.command().is_none() {
                    return Err(Error::validation(
                        "MCP stdio transport requires `server_path` or `params.command`",
                    ));
                }
            }
            McpClientType::Sse | McpClientType::StreamableHttp => {
                super::http::validate_endpoint(&config.server_path)?;
            }
            McpClientType::OpenApi | McpClientType::Playwright => {
                return Err(unsupported_transport(config.client_type));
            }
        }
        Ok(Self {
            config: config.clone(),
            connection: Mutex::new(None),
            connect_lock: Mutex::new(()),
            server_stream: parking_lot::Mutex::new(None),
            resource_subscriptions: AtomicBool::new(false),
        })
    }

    /// Returns the configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Builds and initializes a fresh `rmcp` session.
    ///
    /// Returns the session together with the server-initiated stream handle when
    /// the transport has one (`streamable_http` only). The `initialize` result is
    /// read here to record the negotiated `resources.subscribe` capability.
    async fn spawn_session(
        &self,
    ) -> Result<(RunningService<RoleClient, ()>, Option<ServerStreamState>)> {
        tracing::debug!(
            server_name = %self.config.server_name,
            client_type = self.config.client_type.as_str(),
            "connecting MCP server"
        );
        let failure = |error: String| Error::transport(format!("MCP connect failed: {error}"));
        let established = match self.config.client_type {
            McpClientType::Stdio => {
                let transport = build_stdio(&self.config)?;
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
                    .map(|session| (session, None))
            }
            McpClientType::StreamableHttp => {
                let transport = StreamableHttpTransport::new(&self.config)?;
                let state = transport.server_stream();
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
                    .map(|session| (session, Some(state)))
            }
            McpClientType::Sse => {
                let transport = SseTransport::new(&self.config)?;
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
                    .map(|session| (session, None))
            }
            McpClientType::OpenApi | McpClientType::Playwright => {
                Err(unsupported_transport(self.config.client_type))
            }
        }?;
        let (session, stream_state) = established;
        self.resource_subscriptions.store(
            subscription::advertises_subscriptions(session.peer().peer_info()),
            Ordering::SeqCst,
        );
        Ok((session, stream_state))
    }

    /// Drops the held session, cancelling its worker and clearing the stream state.
    async fn drop_session(&self) {
        self.resource_subscriptions.store(false, Ordering::SeqCst);
        let state = {
            let mut guard = self.server_stream.lock();
            guard.take()
        };
        if let Some(state) = state {
            state.reset();
        }
        let mut guard = self.connection.lock().await;
        if let Some(service) = guard.take() {
            if let Err(error) = service.cancel().await {
                tracing::debug!(%error, "MCP session shutdown reported an error");
            }
        }
    }

    /// Returns a live peer, connecting when no session is held.
    async fn peer(&self) -> Result<Peer<RoleClient>> {
        if let Some(service) = self.connection.lock().await.as_ref() {
            return Ok(service.peer().clone());
        }
        self.connect().await?;
        let guard = self.connection.lock().await;
        guard
            .as_ref()
            .map(|service| service.peer().clone())
            .ok_or_else(|| Error::transport("MCP session was not established"))
    }

    /// Runs `operation` with reconnect retries.
    async fn with_reconnect<T, F, Fut>(&self, operation: &'static str, operation_fn: F) -> Result<T>
    where
        F: Fn(Peer<RoleClient>) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let attempts = reconnect_attempt_budget();
        let mut attempt: u32 = 1;
        loop {
            let outcome = match self.peer().await {
                Ok(peer) => operation_fn(peer).await,
                Err(error) => Err(error),
            };
            match outcome {
                Ok(value) => return Ok(value),
                Err(error) => {
                    if attempt >= attempts || !is_retryable(&error) {
                        return Err(error);
                    }
                    let delay = reconnect_delay(attempt);
                    tracing::warn!(
                        operation,
                        attempt,
                        attempts,
                        delay_ms = delay.as_millis(),
                        error = %error,
                        "MCP operation failed; reconnecting"
                    );
                    self.drop_session().await;
                    tokio::time::sleep(delay).await;
                    attempt = attempt.saturating_add(1);
                }
            }
        }
    }
}

#[async_trait]
impl McpClient for RmcpClient {
    fn server_name(&self) -> &str {
        &self.config.server_name
    }

    fn server_id(&self) -> &str {
        &self.config.server_id
    }

    fn include_image_content(&self) -> bool {
        self.config.include_image_content
    }

    async fn connect(&self) -> Result<()> {
        if self.connection.lock().await.is_some() {
            return Ok(());
        }
        let _guard = self.connect_lock.lock().await;
        if self.connection.lock().await.is_some() {
            return Ok(());
        }
        let (session, stream_state) = self.spawn_session().await?;
        *self.server_stream.lock() = stream_state;
        *self.connection.lock().await = Some(session);
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.drop_session().await;
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connection.lock().await.is_some()
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        self.with_reconnect("list_tools", |peer| async move {
            let tools = peer.list_all_tools().await.map_err(map_service_error)?;
            Ok(tools
                .into_iter()
                .map(|tool| McpRemoteTool {
                    name: tool.name.to_string(),
                    description: tool
                        .description
                        .map(|text| text.to_string())
                        .unwrap_or_default(),
                    input_schema: Value::Object((*tool.input_schema).clone()),
                })
                .collect())
        })
        .await
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
        include_image_content: bool,
    ) -> Result<McpToolResult> {
        let name = tool_name.to_owned();
        let arguments = arguments.as_object().cloned().unwrap_or_default();
        self.with_reconnect("call_tool", |peer| {
            let name = name.clone();
            let arguments: Map<String, Value> = arguments.clone();
            async move {
                let request = peer.call_tool(CallToolRequestParam {
                    name: name.clone().into(),
                    arguments: Some(arguments),
                });
                let result = match timeout {
                    Some(limit) => tokio::time::timeout(limit, request)
                        .await
                        .map_err(|_| {
                            Error::timeout(format!("MCP call_tool `{name}` exceeded {limit:?}"))
                        })?
                        .map_err(map_service_error)?,
                    None => request.await.map_err(map_service_error)?,
                };
                Ok(normalize_call_tool_result(&result, include_image_content))
            }
        })
        .await
    }

    async fn list_resources(&self) -> Result<Vec<Value>> {
        self.with_reconnect("list_resources", |peer| async move {
            let resources = peer.list_all_resources().await.map_err(map_service_error)?;
            Ok(resources
                .into_iter()
                .map(|resource| serde_json::to_value(&resource).unwrap_or(Value::Null))
                .collect())
        })
        .await
    }

    async fn read_resource(&self, uri: &str) -> Result<Value> {
        let uri = uri.to_owned();
        self.with_reconnect("read_resource", |peer| {
            let uri = uri.clone();
            async move {
                let result = peer
                    .read_resource(ReadResourceRequestParam { uri })
                    .await
                    .map_err(map_service_error)?;
                Ok(serde_json::to_value(&result).unwrap_or(Value::Null))
            }
        })
        .await
    }

    async fn subscribe_resource(&self, uri: &str) -> Result<()> {
        let params = subscription::subscribe_params(uri)?;
        self.with_reconnect("subscribe_resource", |peer| {
            let params = params.clone();
            async move {
                subscription::ensure_capability(peer.peer_info(), subscription::METHOD_SUBSCRIBE)?;
                peer.subscribe(params).await.map_err(map_service_error)
            }
        })
        .await
    }

    async fn unsubscribe_resource(&self, uri: &str) -> Result<()> {
        let params = subscription::unsubscribe_params(uri)?;
        self.with_reconnect("unsubscribe_resource", |peer| {
            let params = params.clone();
            async move {
                subscription::ensure_capability(
                    peer.peer_info(),
                    subscription::METHOD_UNSUBSCRIBE,
                )?;
                peer.unsubscribe(params).await.map_err(map_service_error)
            }
        })
        .await
    }

    fn supports_resource_subscriptions(&self) -> bool {
        self.resource_subscriptions.load(Ordering::SeqCst)
    }

    // Draining the buffer needs no I/O and never creates a session.
    #[allow(clippy::unused_async_trait_impl)]
    async fn take_notifications(&self) -> Result<Vec<McpNotification>> {
        let state = self.server_stream.lock().clone();
        match state {
            Some(state) => Ok(state.take_notifications()),
            None => Ok(Vec::new()),
        }
    }

    fn supports_server_stream(&self) -> bool {
        self.server_stream
            .lock()
            .as_ref()
            .is_some_and(|state| state.supports_server_stream() && !state.is_refused())
    }
}

/// Spawns the stdio server described by `config`.
fn build_stdio(config: &McpServerConfig) -> Result<TokioChildProcess> {
    let command_path = config
        .command()
        .ok_or_else(|| Error::validation("MCP stdio transport requires a command"))?;
    let mut command = tokio::process::Command::new(command_path);
    command.args(config.args());
    if let Some(cwd) = config.cwd() {
        command.current_dir(cwd);
    }
    for (key, value) in config.env() {
        command.env(key, value);
    }
    command.kill_on_drop(true);
    TokioChildProcess::new(command).map_err(|error| {
        Error::transport(format!(
            "failed to spawn MCP stdio server `{command_path}`: {error}"
        ))
    })
}

#[cfg(test)]
mod fake;

#[cfg(test)]
pub use fake::{FakeCall, FakeMcpClient, FakeSubscriptionAction, FakeSubscriptionCall};
