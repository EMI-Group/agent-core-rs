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

use std::collections::BTreeMap;
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

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::http::{SseTransport, StreamableHttpTransport};
use super::result::{normalize_call_tool_result, McpToolResult};

/// The kind of transport used to reach an MCP server.
///
/// Python parity: the `client_type` string of `McpServerConfig`
/// (`"stdio"`, `"sse"`, `"streamable-http"`). Values are matched
/// case-insensitively and both dash and underscore spellings are accepted;
/// `"http"` is an accepted alias for [`McpClientType::StreamableHttp`], which is
/// the default.
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
}

impl McpClientType {
    /// Returns the canonical `snake_case` spelling of this transport.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Sse => "sse",
            Self::StreamableHttp => "streamable_http",
        }
    }

    /// Resolves a transport name, accepting the Python aliases.
    ///
    /// Accepts `stdio`, `sse`, `streamable_http`, `streamable-http`, and `http`
    /// in any ASCII case.
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
            _ => Err(Error::validation(format!(
                "unsupported MCP client type `{value}`; expected stdio, sse, or streamable-http"
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
}

/// Builds a client for `config` without connecting.
///
/// Python parity: `ToolMgr._create_client` plus the client registry
/// (`"stdio"`, `"sse"`, `"streamable-http"`). The returned client connects on
/// its first operation.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `config` cannot address its transport (a
/// stdio config without a command, or an HTTP config without an absolute
/// http/https URL).
pub fn connect_mcp_server(config: &McpServerConfig) -> Result<Arc<dyn McpClient>> {
    let client = RmcpClient::new(config)?;
    tracing::debug!(
        server_name = %config.server_name,
        server_id = %config.server_id,
        client_type = config.client_type.as_str(),
        auth_headers = ?config.redacted_headers(),
        "created MCP client"
    );
    Ok(Arc::new(client))
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
}

impl RmcpClient {
    /// Validates `config` and creates a client that connects on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unusable transport configuration.
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
        }
        Ok(Self {
            config: config.clone(),
            connection: Mutex::new(None),
            connect_lock: Mutex::new(()),
        })
    }

    /// Returns the configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Builds and initializes a fresh `rmcp` session.
    async fn spawn_session(&self) -> Result<RunningService<RoleClient, ()>> {
        tracing::debug!(
            server_name = %self.config.server_name,
            client_type = self.config.client_type.as_str(),
            "connecting MCP server"
        );
        let failure = |error: String| Error::transport(format!("MCP connect failed: {error}"));
        match self.config.client_type {
            McpClientType::Stdio => {
                let transport = build_stdio(&self.config)?;
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
            }
            McpClientType::StreamableHttp => {
                let transport = StreamableHttpTransport::new(&self.config)?;
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
            }
            McpClientType::Sse => {
                let transport = SseTransport::new(&self.config)?;
                serve_client((), transport)
                    .await
                    .map_err(|error| failure(error.to_string()))
            }
        }
    }

    /// Drops the held session, cancelling its worker.
    async fn drop_session(&self) {
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
        let session = self.spawn_session().await?;
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

/// An in-memory [`McpClient`] for tests: records calls, returns scripted results.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct FakeMcpClient {
    server_id: String,
    server_name: String,
    include_image_content: bool,
    tools: Mutex<Vec<McpRemoteTool>>,
    replies: Mutex<HashMap<String, FakeReply>>,
    calls: Mutex<Vec<FakeCall>>,
    connected: AtomicBool,
    connect_count: AtomicUsize,
}

/// One recorded [`FakeMcpClient`] call.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct FakeCall {
    /// Tool that was called.
    pub tool_name: String,
    /// Arguments passed through.
    pub arguments: Value,
    /// Per-call deadline.
    pub timeout: Option<Duration>,
    /// Whether image blocks were requested as multimodal data.
    pub include_image_content: bool,
}

#[cfg(test)]
#[derive(Debug, Clone)]
enum FakeReply {
    Result(McpToolResult),
    Failure(String),
}

#[cfg(test)]
impl FakeMcpClient {
    /// Creates a fake client for `server_id`/`server_name` with no tools.
    #[must_use]
    pub fn new(server_id: impl Into<String>, server_name: impl Into<String>) -> Self {
        Self {
            server_id: server_id.into(),
            server_name: server_name.into(),
            include_image_content: false,
            tools: Mutex::new(Vec::new()),
            replies: Mutex::new(HashMap::new()),
            calls: Mutex::new(Vec::new()),
            connected: AtomicBool::new(false),
            connect_count: AtomicUsize::new(0),
        }
    }

    /// Sets the tools this client advertises.
    pub async fn set_tools(&self, tools: Vec<McpRemoteTool>) {
        *self.tools.lock().await = tools;
    }

    /// Scripts a successful reply for `tool_name`.
    pub async fn set_result(&self, tool_name: &str, result: McpToolResult) {
        self.replies
            .lock()
            .await
            .insert(tool_name.to_owned(), FakeReply::Result(result));
    }

    /// Scripts a failure reply for `tool_name`.
    pub async fn set_failure(&self, tool_name: &str, message: impl Into<String>) {
        self.replies
            .lock()
            .await
            .insert(tool_name.to_owned(), FakeReply::Failure(message.into()));
    }

    /// Returns the recorded calls, oldest first.
    pub async fn calls(&self) -> Vec<FakeCall> {
        self.calls.lock().await.clone()
    }

    /// Returns how many times [`McpClient::connect`] established a session.
    #[must_use]
    pub fn connect_count(&self) -> usize {
        self.connect_count.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl McpClient for FakeMcpClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn include_image_content(&self) -> bool {
        self.include_image_content
    }

    async fn connect(&self) -> Result<()> {
        if !self.connected.swap(true, Ordering::SeqCst) {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        self.connect().await?;
        Ok(self.tools.lock().await.clone())
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
        include_image_content: bool,
    ) -> Result<McpToolResult> {
        self.connect().await?;
        self.calls.lock().await.push(FakeCall {
            tool_name: tool_name.to_owned(),
            arguments,
            timeout,
            include_image_content,
        });
        match self.replies.lock().await.get(tool_name).cloned() {
            Some(FakeReply::Result(result)) => Ok(result),
            Some(FakeReply::Failure(message)) => Err(Error::mcp(message)),
            None => Ok(McpToolResult::success(Map::new())),
        }
    }
}
