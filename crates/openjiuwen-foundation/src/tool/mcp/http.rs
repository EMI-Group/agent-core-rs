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
//!   Once the session id is known, the transport also opens the server-initiated
//!   `GET` event stream (see [`ServerStreamState`]); server notifications and
//!   server requests arrive on it.
//! - [`SseTransport`] opens one `GET` event stream, learns the POST endpoint from
//!   its `endpoint` event, and receives every reply on that stream.
//!
//! Configured `auth_headers` are sent with every request and
//! `auth_query_params` are appended to the endpoint URL, so servers that
//! authenticate through headers or query parameters both work. Header values are
//! never logged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use reqwest::{Response, StatusCode};
use rmcp::model::{ProtocolVersion, ServerJsonRpcMessage};
use rmcp::service::TxJsonRpcMessage;
use rmcp::{transport::Transport, RoleClient};
use serde_json::Value;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::client::{reconnect_attempt_budget, reconnect_delay, McpNotification, McpServerConfig};

/// Header carrying the MCP session id in streamable-HTTP responses.
const HEADER_SESSION_ID: &str = "Mcp-Session-Id";
/// MIME type of an SSE response body.
const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
/// MIME type of a single JSON-RPC response body.
const JSON_MIME_TYPE: &str = "application/json";
/// How long an SSE transport waits for the server's `endpoint` event.
const SSE_ENDPOINT_TIMEOUT: Duration = Duration::from_secs(30);
/// Header carrying the MCP protocol version negotiated with the server.
const HEADER_PROTOCOL_VERSION: &str = "MCP-Protocol-Version";
/// Header carrying the id of the last event the client accepted.
const HEADER_LAST_EVENT_ID: &str = "Last-Event-ID";
/// Retry delay used when the server sends no `retry` field on its event stream.
pub(crate) const DEFAULT_STREAM_RETRY: Duration = Duration::from_secs(1);
/// Cap on the server-stream reconnect delay, whatever the server asks for.
const MAX_STREAM_RETRY: Duration = Duration::from_secs(120);
/// Maximum number of server notifications buffered for [`super::client::McpClient::take_notifications`].
pub(crate) const NOTIFICATION_BUFFER_BOUND: usize = 64;

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
    let parsed = Url::parse(endpoint.trim()).map_err(|error| {
        Error::validation(format!("invalid MCP endpoint `{endpoint}`: {error}"))
    })?;
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

/// Shared handle to the state of one transport's server-initiated stream.
///
/// The transport owns the loop that reads the `GET` event stream; a clone of this
/// handle lets the client read what the loop learned: whether the server supports
/// the channel, the last event id to replay, and the server notifications the loop
/// buffered. Every field is `Arc`-shared, so clones observe the same state.
#[derive(Clone, Default)]
pub(crate) struct ServerStreamState {
    supported: Arc<AtomicBool>,
    refused: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    last_event_id: Arc<parking_lot::Mutex<Option<String>>>,
    notifications: Arc<parking_lot::Mutex<VecDeque<McpNotification>>>,
}

impl ServerStreamState {
    /// Returns whether the server accepted the `GET` channel at least once.
    pub(crate) fn supports_server_stream(&self) -> bool {
        self.supported.load(Ordering::SeqCst)
    }

    /// Records that the server accepted the `GET` channel.
    pub(crate) fn mark_supported(&self) {
        self.supported.store(true, Ordering::SeqCst);
    }

    /// Returns whether the server refused the `GET` channel (405/404/501).
    pub(crate) fn is_refused(&self) -> bool {
        self.refused.load(Ordering::SeqCst)
    }

    /// Records that the server refused the `GET` channel with `status`.
    ///
    /// Refusal is permanent for a session: the channel is never retried, so the
    /// capability stays reported as unsupported.
    pub(crate) fn mark_refused(&self, status: u16) {
        if !self.refused.swap(true, Ordering::SeqCst) {
            tracing::debug!(
                status,
                "MCP server refused the server-initiated event stream"
            );
        }
    }

    /// Records that the transport started the `GET` loop for this session.
    pub(crate) fn mark_started(&self) {
        self.started.store(true, Ordering::SeqCst);
    }

    /// Returns whether the `GET` loop was already started for this session.
    pub(crate) fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    /// Returns the id of the last event received on the `GET` channel.
    pub(crate) fn last_event_id(&self) -> Option<String> {
        self.last_event_id.lock().clone()
    }

    /// Records the id of the last event received on the `GET` channel.
    pub(crate) fn set_last_event_id(&self, id: impl Into<String>) {
        *self.last_event_id.lock() = Some(id.into());
    }

    /// Buffers one server notification, dropping the oldest when the buffer is full.
    pub(crate) fn push_notification(&self, notification: McpNotification) {
        let mut buffer = self.notifications.lock();
        if buffer.len() >= NOTIFICATION_BUFFER_BOUND {
            buffer.pop_front();
            tracing::warn!(
                bound = NOTIFICATION_BUFFER_BOUND,
                "MCP notification buffer is full; dropping the oldest notification"
            );
        }
        buffer.push_back(notification);
    }

    /// Drains the buffered server notifications, oldest first.
    pub(crate) fn take_notifications(&self) -> Vec<McpNotification> {
        self.notifications.lock().drain(..).collect()
    }

    /// Clears the per-session stream state; called when the session is dropped.
    pub(crate) fn reset(&self) {
        self.started.store(false, Ordering::SeqCst);
        self.refused.store(false, Ordering::SeqCst);
        self.supported.store(false, Ordering::SeqCst);
        *self.last_event_id.lock() = None;
    }
}

/// Builds the headers of the server-initiated `GET` event stream request.
///
/// Always advertises `text/event-stream`, the negotiated session id, the protocol
/// version, and `no-store` caching; `Last-Event-ID` is added when the client
/// already accepted an event so the server can replay the tail of its stream.
pub(crate) fn server_stream_headers(
    session_id: &str,
    last_event_id: Option<&str>,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        ("Accept", EVENT_STREAM_MIME_TYPE.to_owned()),
        (HEADER_SESSION_ID, session_id.to_owned()),
        (HEADER_PROTOCOL_VERSION, ProtocolVersion::LATEST.to_string()),
        ("Cache-Control", "no-store".to_owned()),
    ];
    if let Some(id) = last_event_id {
        headers.push((HEADER_LAST_EVENT_ID, id.to_owned()));
    }
    headers
}

/// Computes the delay before retrying the server-initiated event stream.
///
/// `attempt` is 1-based. The server's `retry` hint wins over
/// [`DEFAULT_STREAM_RETRY`], but never exceeds the shared reconnect policy delay
/// for that attempt nor [`MAX_STREAM_RETRY`], so a misbehaving server cannot hold
/// the reconnect loop off for longer than the rest of the client.
pub(crate) fn stream_reconnect_delay(attempt: u32, hint: Option<Duration>) -> Duration {
    hint.unwrap_or(DEFAULT_STREAM_RETRY)
        .min(reconnect_delay(attempt))
        .min(MAX_STREAM_RETRY)
}

/// A streamable-HTTP transport.
pub(crate) struct StreamableHttpTransport {
    endpoint: Endpoint,
    session_id: Arc<Mutex<Option<String>>>,
    inbound: UnboundedReceiver<ServerJsonRpcMessage>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
    state: ServerStreamState,
    stream_started: Arc<AtomicBool>,
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
            state: ServerStreamState::default(),
            stream_started: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns a handle to this transport's server-initiated stream state.
    ///
    /// Clones the handle so a caller (the client) can read the negotiated
    /// capability and drain notifications after the transport has been moved into
    /// `rmcp`.
    pub(crate) fn server_stream(&self) -> ServerStreamState {
        self.state.clone()
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
        let state = self.state.clone();
        let stream_started = self.stream_started.clone();
        async move {
            if cancellation.is_cancelled() {
                return Err(HttpTransportError::protocol("transport closed"));
            }
            let body = serde_json::to_string(&item)
                .map_err(|error| HttpTransportError::protocol(error.to_string()))?;
            let mut request = endpoint
                .client
                .post(&endpoint.url)
                .header(
                    ACCEPT,
                    format!("{JSON_MIME_TYPE}, {EVENT_STREAM_MIME_TYPE}"),
                )
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
                    *guard = Some(session.clone());
                }
                // The initialize handshake response is where the session id first
                // appears: that is the only point at which the server-initiated
                // stream can be addressed, so open it once per transport.
                if !stream_started.swap(true, Ordering::SeqCst) {
                    spawn_server_stream(
                        endpoint.clone(),
                        session,
                        outbound.clone(),
                        state.clone(),
                        cancellation.clone(),
                    );
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
                        if let Err(error) =
                            read_message_stream(response, outbound, cancellation).await
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

/// Why one attempt at reading the server-initiated event stream ended.
enum StreamOutcome {
    /// The server refused the channel (405/404/501); never retried.
    Refused,
    /// The channel ended or failed and may be opened again.
    Retryable,
    /// The transport was closed; the loop must stop.
    Cancelled,
}

/// Spawns the long-lived `GET` event stream of a streamable-HTTP transport.
///
/// `state` is the transport's shared handle: the loop records the negotiated
/// capability, the last event id, the reconnect hint, and every server
/// notification it sees. Frames are forwarded to `outbound` unchanged, so `rmcp`
/// correlates responses by id and handles server-initiated requests itself.
fn spawn_server_stream(
    endpoint: Endpoint,
    session_id: String,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    state: ServerStreamState,
    cancellation: CancellationToken,
) {
    if state.is_started() {
        return;
    }
    state.mark_started();
    tokio::spawn(async move {
        let attempts = reconnect_attempt_budget();
        let mut attempt: u32 = 1;
        let mut hint: Option<Duration> = None;
        loop {
            if cancellation.is_cancelled() {
                return;
            }
            match run_server_stream(
                &endpoint,
                &session_id,
                &outbound,
                &state,
                &cancellation,
                &mut hint,
            )
            .await
            {
                StreamOutcome::Refused | StreamOutcome::Cancelled => return,
                StreamOutcome::Retryable => {}
            }
            if attempt >= attempts {
                tracing::debug!(
                    attempt,
                    attempts,
                    "MCP server stream retry budget exhausted"
                );
                return;
            }
            let delay = stream_reconnect_delay(attempt, hint);
            tracing::debug!(
                attempt,
                delay_ms = delay.as_millis(),
                "MCP server stream ended; reconnecting"
            );
            tokio::select! {
                () = cancellation.cancelled() => return,
                () = tokio::time::sleep(delay) => {}
            }
            attempt = attempt.saturating_add(1);
        }
    });
}

/// Runs one attempt at the server-initiated `GET` event stream.
///
/// The body is read until it ends, fails, or the transport is cancelled; the
/// accumulated event id stays in `state` so a retry replays from the last event
/// the client accepted.
async fn run_server_stream(
    endpoint: &Endpoint,
    session_id: &str,
    outbound: &UnboundedSender<ServerJsonRpcMessage>,
    state: &ServerStreamState,
    cancellation: &CancellationToken,
    hint: &mut Option<Duration>,
) -> StreamOutcome {
    let mut request = endpoint
        .client
        .get(&endpoint.url)
        .headers(endpoint.headers.clone());
    for (name, value) in server_stream_headers(session_id, state.last_event_id().as_deref()) {
        request = request.header(name, value);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(%error, "MCP server stream could not be reached");
            return StreamOutcome::Retryable;
        }
    };
    let status = response.status();
    if matches!(
        status,
        StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_FOUND | StatusCode::NOT_IMPLEMENTED
    ) {
        state.mark_refused(status.as_u16());
        return StreamOutcome::Refused;
    }
    if !status.is_success() {
        tracing::debug!(
            status = status.as_u16(),
            "MCP server stream rejected the request"
        );
        return StreamOutcome::Retryable;
    }
    state.mark_supported();
    let mut stream = response.bytes_stream();
    let mut parser = SseParser::default();
    loop {
        tokio::select! {
            () = cancellation.cancelled() => return StreamOutcome::Cancelled,
            chunk = stream.next() => match chunk {
                Some(Ok(bytes)) => {
                    for frame in parser.push(&bytes) {
                        if let Some(id) = frame.id.clone() {
                            state.set_last_event_id(id);
                        }
                        if let Some(retry) = frame.retry {
                            *hint = Some(Duration::from_millis(retry));
                        }
                        handle_server_stream_frame(&frame, outbound, state);
                    }
                }
                Some(Err(error)) => {
                    tracing::debug!(%error, "MCP server stream failed");
                    return StreamOutcome::Retryable;
                }
                None => return StreamOutcome::Retryable,
            },
        }
    }
}

/// Buffers one server notification and forwards the frame's JSON-RPC message.
///
/// A frame is buffered when `rmcp` parses it as a JSON-RPC notification, i.e. when
/// it is one of the notification methods `rmcp` models; unknown methods are only
/// logged, exactly as on the POST stream. `method`/`params` come from the raw JSON
/// because `rmcp` models several notifications without parameters and would drop
/// them.
fn handle_server_stream_frame(
    frame: &SseFrame,
    outbound: &UnboundedSender<ServerJsonRpcMessage>,
    state: &ServerStreamState,
) {
    if frame.event.as_deref() == Some("endpoint") || frame.data.trim().is_empty() {
        return;
    }
    let payload: Value = match serde_json::from_str(&frame.data) {
        Ok(payload) => payload,
        Err(error) => {
            tracing::debug!(%error, "ignoring non JSON-RPC MCP event data");
            return;
        }
    };
    let message: ServerJsonRpcMessage = match serde_json::from_value(payload.clone()) {
        Ok(message) => message,
        Err(error) => {
            tracing::debug!(%error, "ignoring non JSON-RPC MCP event data");
            return;
        }
    };
    if matches!(message, ServerJsonRpcMessage::Notification(_)) {
        if let Some(method) = payload.get("method").and_then(Value::as_str) {
            let params = payload.get("params").cloned().unwrap_or(Value::Null);
            state.push_notification(McpNotification {
                method: method.to_owned(),
                params,
            });
        }
    }
    let _ = outbound.send(message);
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
                    started.clone(),
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

    // No `.await` is needed here, but the transport contract is async.
    #[allow(clippy::unused_async_trait_impl)]
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
    started: Arc<AtomicBool>,
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
        started.store(false, Ordering::SeqCst);
    });
}

/// One parsed Server-Sent-Events frame.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// Value of the `event` field, when present.
    pub event: Option<String>,
    /// Joined `data` field lines.
    pub data: String,
    /// Value of the `id` field of the frame, when present.
    pub id: Option<String>,
    /// Value of the `retry` field (reconnect delay in milliseconds), when present.
    pub retry: Option<u64>,
}

/// An incremental Server-Sent-Events parser.
///
/// Feed raw body chunks to [`SseParser::push`]; complete frames come back as they
/// finish (a blank line delimits frames). Unknown fields and comment lines are
/// ignored. The parser tolerates `\n` and `\r\n` line endings and split chunks.
///
/// `id` and `retry` are captured into the next completed frame; an unparsable
/// `retry` value is ignored. Frames are still delimited by `event`/`data` lines
/// alone, so a lone `id` or `retry` line never emits a frame by itself.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buffer: String,
    event: Option<String>,
    data: String,
    id: Option<String>,
    retry: Option<u64>,
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
                id: self.id.take(),
                retry: self.retry.take(),
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
            "id" => {
                self.id = Some(value.to_owned());
            }
            "retry" => {
                if let Ok(milliseconds) = value.trim().parse::<u64>() {
                    self.retry = Some(milliseconds);
                }
            }
            _ => {}
        }
        None
    }
}
