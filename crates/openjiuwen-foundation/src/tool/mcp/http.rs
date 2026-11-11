//! HTTP transports for MCP: streamable HTTP and legacy HTTP+SSE.
//!
//! Both implement [`rmcp::transport::Transport`] directly on `reqwest` instead of
//! using `rmcp`'s own HTTP clients: the workspace enables `rmcp` without its
//! `transport-*-reqwest` features, and this crate may not change those features.
//! Framing, session handling, and request ids above the transport still come from
//! `rmcp` through [`rmcp::serve_client`].
//!
//! - [`StreamableHttpTransport`] POSTs every JSON-RPC message to the endpoint and
//!   accepts either a JSON response, a `text/event-stream` response carrying the
//!   replies, or `202 Accepted` for notifications. A `Mcp-Session-Id` response
//!   header is captured and replayed on later requests; `close` sends `DELETE`.
//! - [`SseTransport`] opens one `GET` event stream, learns the POST endpoint from
//!   its `endpoint` event, and receives every reply on that stream.
//!
//! Configured `auth_headers` are sent with every request and
//! `auth_query_params` are appended to the endpoint URL, so servers that
//! authenticate through headers or query parameters both work. Header values are
//! never logged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Response, StatusCode};
use rmcp::model::ServerJsonRpcMessage;
use rmcp::service::TxJsonRpcMessage;
use rmcp::{RoleClient, transport::Transport};
use tokio::sync::Notify;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::client::McpServerConfig;

/// Header carrying the MCP session id in streamable-HTTP responses.
const HEADER_SESSION_ID: &str = "Mcp-Session-Id";
/// MIME type of an SSE response body.
const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
/// MIME type of a single JSON-RPC response body.
const JSON_MIME_TYPE: &str = "application/json";
/// How long an SSE transport waits for the server's `endpoint` event.
const SSE_ENDPOINT_TIMEOUT: Duration = Duration::from_secs(30);

/// Transport-level failures raised by the HTTP transports.
#[derive(Debug, thiserror::Error)]
pub(crate) enum HttpTransportError {
    /// The request could not be sent or the body could not be read.
    #[error("MCP request failed: {0}")]
    Request(#[from] reqwest::Error),
    /// The endpoint answered with a non-success status.
    #[error("MCP endpoint rejected the request with status {0}")]
    Status(u16),
    /// The response did not follow the MCP framing.
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    /// The endpoint URL could not be parsed or resolved.
    #[error("invalid MCP endpoint URL: {0}")]
    Url(#[from] url::ParseError),
}

impl HttpTransportError {
    /// Builds a protocol error with a human-readable `message`.
    fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }
}

/// Validates that `endpoint` is an absolute `http`/`https` URL with a host.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the URL is relative, uses another scheme,
/// or has no host.
pub(crate) fn validate_endpoint(endpoint: &str) -> Result<()> {
    let parsed = Url::parse(endpoint.trim())
        .map_err(|error| Error::validation(format!("invalid MCP endpoint `{endpoint}`: {error}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(Error::validation(format!(
            "MCP endpoint `{endpoint}` must use http or https"
        )));
    }
    if parsed.host_str().is_none() {
        return Err(Error::validation(format!(
            "MCP endpoint `{endpoint}` has no host"
        )));
    }
    Ok(())
}

/// A resolved endpoint: HTTP client, URL, and the headers to replay.
#[derive(Clone)]
struct Endpoint {
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
}

impl Endpoint {
    /// Builds the endpoint for `config`, appending auth query parameters.
    fn new(config: &McpServerConfig) -> Result<Self> {
        validate_endpoint(&config.server_path)?;
        let mut url = Url::parse(config.server_path.trim())
            .map_err(|error| Error::validation(format!("invalid MCP endpoint: {error}")))?;
        if !config.auth_query_params.is_empty() {
            {
                let mut pairs = url.query_pairs_mut();
                for (name, value) in &config.auth_query_params {
                    pairs.append_pair(name, value);
                }
            }
        }
        let mut headers = HeaderMap::new();
        for (name, value) in &config.auth_headers {
            let header_name = HeaderName::from_bytes(name.trim().as_bytes()).map_err(|error| {
                Error::validation(format!("invalid MCP auth header name: {error}"))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|_| {
                Error::validation(format!("invalid MCP auth header value for `{header_name}`"))
            })?;
            headers.insert(header_name, header_value);
        }
        let client = reqwest::Client::builder().build().map_err(|error| {
            Error::transport(format!("failed to build MCP HTTP client: {error}"))
        })?;
        Ok(Self {
            client,
            url: url.to_string(),
            headers,
        })
    }
}

/// Returns the content type of `response`, lowercased and without parameters.
fn content_type(response: &Response) -> Option<String> {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
        })
}

/// Returns the `Mcp-Session-Id` header value of `response`.
fn session_header(response: &Response) -> Option<String> {
    response
        .headers()
        .get(HEADER_SESSION_ID)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

/// A streamable-HTTP transport.
pub(crate) struct StreamableHttpTransport {
    endpoint: Endpoint,
    session_id: Arc<Mutex<Option<String>>>,
    inbound: UnboundedReceiver<ServerJsonRpcMessage>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
}

impl StreamableHttpTransport {
    /// Creates a transport for `config`; no request is made yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the URL or the auth headers are invalid.
    pub(crate) fn new(config: &McpServerConfig) -> Result<Self> {
        let (outbound, inbound) = unbounded_channel();
        Ok(Self {
            endpoint: Endpoint::new(config)?,
            session_id: Arc::new(Mutex::new(None)),
            inbound,
            outbound,
            cancellation: CancellationToken::new(),
        })
    }
}

impl Transport<RoleClient> for StreamableHttpTransport {
    type Error = HttpTransportError;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl std::future::Future<Output = std::result::Result<(), Self::Error>> + Send + 'static
    {
        let endpoint = self.endpoint.clone();
        let session_id = self.session_id.clone();
        let outbound = self.outbound.clone();
        let cancellation = self.cancellation.clone();
        async move {
            if cancellation.is_cancelled() {
                return Err(HttpTransportError::protocol("transport closed"));
            }
            let body = serde_json::to_string(&item)
                .map_err(|error| HttpTransportError::protocol(error.to_string()))?;
            let mut request = endpoint
                .client
                .post(&endpoint.url)
                .header(ACCEPT, format!("{JSON_MIME_TYPE}, {EVENT_STREAM_MIME_TYPE}"))
                .header(CONTENT_TYPE, JSON_MIME_TYPE)
                .headers(endpoint.headers.clone())
                .body(body);
            if let Some(session) = session_id.lock().ok().and_then(|guard| guard.clone()) {
                request = request.header(HEADER_SESSION_ID, session);
            }
            let response = request.send().await?;
            let status = response.status();
            if let Some(session) = session_header(&response) {
                if let Ok(mut guard) = session_id.lock() {
                    *guard = Some(session);
                }
            }
            if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
                return Ok(());
            }
            if !status.is_success() {
                return Err(HttpTransportError::Status(status.as_u16()));
            }
            match content_type(&response) {
                Some(mime) if mime == EVENT_STREAM_MIME_TYPE => {
                    tokio::spawn(async move {
                        if let Err(error) = read_message_stream(response, outbound, cancellation).await
                        {
                            tracing::debug!(%error, "MCP streamable HTTP stream ended");
                        }
                    });
                    Ok(())
                }
                Some(mime) if mime == JSON_MIME_TYPE => {
                    let bytes = response.bytes().await?;
                    let message: ServerJsonRpcMessage = serde_json::from_slice(&bytes)
                        .map_err(|error| HttpTransportError::protocol(error.to_string()))?;
                    let _ = outbound.send(message);
                    Ok(())
                }
                other => Err(HttpTransportError::protocol(format!(
                    "unexpected MCP response content type {other:?}"
                ))),
            }
        }
    }

    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        self.inbound.recv().await
    }

    async fn close(&mut self) -> std::result::Result<(), Self::Error> {
        self.cancellation.cancel();
        let session = self.session_id.lock().ok().and_then(|guard| guard.clone());
        if let Some(session) = session {
            let _ = self
                .endpoint
                .client
                .delete(&self.endpoint.url)
                .headers(self.endpoint.headers.clone())
                .header(HEADER_SESSION_ID, session)
                .send()
                .await;
        }
        Ok(())
    }
}

/// Reads an SSE response body, forwarding every JSON-RPC message to `outbound`.
async fn read_message_stream(
    response: Response,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
) -> std::result::Result<(), HttpTransportError> {
    let mut stream = response.bytes_stream();
    let mut parser = SseParser::default();
    loop {
        tokio::select! {
            () = cancellation.cancelled() => return Ok(()),
            chunk = stream.next() => match chunk {
                Some(Ok(bytes)) => {
                    for frame in parser.push(&bytes) {
                        forward_frame(&frame, &outbound, None);
                    }
                }
                Some(Err(error)) => return Err(HttpTransportError::Request(error)),
                None => return Ok(()),
            },
        }
    }
}

/// Parses one SSE frame into a JSON-RPC message and forwards it when it is one.
///
/// `endpoint_slot` is set when the frame announces the POST endpoint (SSE
/// transport); other frames are parsed as server messages.
fn forward_frame(
    frame: &SseFrame,
    outbound: &UnboundedSender<ServerJsonRpcMessage>,
    endpoint_slot: Option<&Arc<Mutex<Option<String>>>>,
) {
    if frame.event.as_deref() == Some("endpoint") {
        if let Some(slot) = endpoint_slot {
            if let Ok(mut guard) = slot.lock() {
                *guard = Some(frame.data.trim().to_owned());
            }
        }
        return;
    }
    if frame.data.trim().is_empty() {
        return;
    }
    match serde_json::from_str::<ServerJsonRpcMessage>(&frame.data) {
        Ok(message) => {
            let _ = outbound.send(message);
        }
        Err(error) => {
            tracing::debug!(%error, "ignoring non JSON-RPC MCP event data");
        }
    }
}

/// A legacy HTTP+SSE transport.
pub(crate) struct SseTransport {
    endpoint: Endpoint,
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
    started: Arc<AtomicBool>,
    inbound: UnboundedReceiver<ServerJsonRpcMessage>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
}

impl SseTransport {
    /// Creates a transport for `config`; the event stream opens on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the URL or the auth headers are invalid.
    pub(crate) fn new(config: &McpServerConfig) -> Result<Self> {
        let (outbound, inbound) = unbounded_channel();
        Ok(Self {
            endpoint: Endpoint::new(config)?,
            post_endpoint: Arc::new(Mutex::new(None)),
            endpoint_ready: Arc::new(Notify::new()),
            started: Arc::new(AtomicBool::new(false)),
            inbound,
            outbound,
            cancellation: CancellationToken::new(),
        })
    }
}

impl Transport<RoleClient> for SseTransport {
    type Error = HttpTransportError;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl std::future::Future<Output = std::result::Result<(), Self::Error>> + Send + 'static
    {
        let endpoint = self.endpoint.clone();
        let post_endpoint = self.post_endpoint.clone();
        let endpoint_ready = self.endpoint_ready.clone();
        let started = self.started.clone();
        let outbound = self.outbound.clone();
        let cancellation = self.cancellation.clone();
        async move {
            if cancellation.is_cancelled() {
                return Err(HttpTransportError::protocol("transport closed"));
            }
            if !started.swap(true, Ordering::SeqCst) {
                spawn_sse_stream(
                    endpoint.clone(),
                    post_endpoint.clone(),
                    endpoint_ready.clone(),
                    outbound,
                    cancellation.clone(),
                );
            }
            let target = wait_post_endpoint(post_endpoint, endpoint_ready).await?;
            let target = resolve_endpoint(&endpoint.url, &target)?;
            let body = serde_json::to_string(&item)
                .map_err(|error| HttpTransportError::protocol(error.to_string()))?;
            let response = endpoint
                .client
                .post(target)
                .header(CONTENT_TYPE, JSON_MIME_TYPE)
                .headers(endpoint.headers.clone())
                .body(body)
                .send()
                .await?;
            let status = response.status();
            if status.is_success() {
                return Ok(());
            }
            Err(HttpTransportError::Status(status.as_u16()))
        }
    }

    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        self.inbound.recv().await
    }

    async fn close(&mut self) -> std::result::Result<(), Self::Error> {
        self.cancellation.cancel();
        Ok(())
    }
}

/// Resolves the (possibly relative) POST endpoint against the SSE URL.
fn resolve_endpoint(base: &str, endpoint: &str) -> std::result::Result<String, HttpTransportError> {
    let base = Url::parse(base)?;
    let resolved = base.join(endpoint.trim())?;
    Ok(resolved.to_string())
}

/// Waits for the SSE `endpoint` event, bounded by [`SSE_ENDPOINT_TIMEOUT`].
async fn wait_post_endpoint(
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
) -> std::result::Result<String, HttpTransportError> {
    let deadline = tokio::time::Instant::now() + SSE_ENDPOINT_TIMEOUT;
    loop {
        if let Some(endpoint) = post_endpoint.lock().ok().and_then(|slot| slot.clone()) {
            return Ok(endpoint);
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(HttpTransportError::protocol(
                "timed out waiting for the MCP SSE endpoint event",
            ));
        }
        let _ = tokio::time::timeout(remaining, endpoint_ready.notified()).await;
    }
}

/// Spawns the long-lived GET event stream of an SSE transport.
fn spawn_sse_stream(
    endpoint: Endpoint,
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
) {
    tokio::spawn(async move {
        let request = endpoint
            .client
            .get(&endpoint.url)
            .header(ACCEPT, EVENT_STREAM_MIME_TYPE)
            .headers(endpoint.headers.clone());
        let response = match request.send().await {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                tracing::debug!(
                    status = response.status().as_u16(),
                    "MCP SSE endpoint rejected the event stream"
                );
                return;
            }
            Err(error) => {
                tracing::debug!(%error, "MCP SSE endpoint could not be reached");
                return;
            }
        };
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::default();
        loop {
            tokio::select! {
                () = cancellation.cancelled() => break,
                chunk = stream.next() => match chunk {
                    Some(Ok(bytes)) => {
                        for frame in parser.push(&bytes) {
                            forward_frame(&frame, &outbound, Some(&post_endpoint));
                        }
                        endpoint_ready.notify_waiters();
                    }
                    Some(Err(error)) => {
                        tracing::debug!(%error, "MCP SSE event stream failed");
                        break;
                    }
                    None => break,
                },
            }
        }
        // Let the next send start a fresh stream.
        if let Ok(mut slot) = post_endpoint.lock() {
            *slot = None;
        }
    });
}

/// One parsed Server-Sent-Events frame.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// Value of the `event` field, when present.
    pub event: Option<String>,
    /// Joined `data` field lines.
    pub data: String,
}

/// An incremental Server-Sent-Events parser.
///
/// Feed raw body chunks to [`SseParser::push`]; complete frames come back as they
/// finish (a blank line delimits frames). Unknown fields and comment lines are
/// ignored. The parser tolerates `\n` and `\r\n` line endings and split chunks.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buffer: String,
    event: Option<String>,
    data: String,
    pending: bool,
}

impl SseParser {
    /// Appends `bytes` and returns every frame completed by them.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<SseFrame> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut frames = Vec::new();
        while let Some(index) = self.buffer.find('\n') {
            let line = self.buffer[..index].trim_end_matches('\r').to_owned();
            self.buffer.drain(..=index);
            if let Some(frame) = self.push_line(&line) {
                frames.push(frame);
            }
        }
        frames
    }

    /// Consumes one line, returning a frame when the line terminates one.
    fn push_line(&mut self, line: &str) -> Option<SseFrame> {
        if line.is_empty() {
            if !self.pending {
                return None;
            }
            let frame = SseFrame {
                event: self.event.take(),
                data: std::mem::take(&mut self.data),
            };
            self.pending = false;
            return Some(frame);
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => {
                self.event = Some(value.to_owned());
                self.pending = true;
            }
            "data" => {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.pending = true;
            }
            _ => {}
        }
        None
    }
}
