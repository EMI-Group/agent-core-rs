//! The A2A JSON-RPC client and the remote-client adapter.
//!
//! Port of `openjiuwen/extensions/a2a/a2a_client.py` and
//! `openjiuwen/extensions/a2a/a2a_remote_client.py` (v0.1.20). The Python code
//! is a thin wrapper over `a2a-sdk`; this module speaks the wire protocol
//! itself, so the envelopes, method names, SSE framing, and error mapping match
//! what `a2a-sdk` 1.0 puts on the wire.
//!
//! # Wire protocol
//!
//! Every call is one HTTP `POST` to [`A2aClientConfig::rpc_url`] with a
//! JSON-RPC 2.0 envelope (`{"jsonrpc":"2.0","id":N,"method":...,"params":...}`)
//! and a `Content-Type: application/json` body. [`A2aClient::invoke`] uses
//! `SendMessage`, [`A2aClient::stream`] uses `SendStreamingMessage` with
//! `Accept: text/event-stream`, and [`A2aClient::cancel_task`] uses
//! `CancelTask`. Request ids start at 1 and increase monotonically per client;
//! extra headers from [`A2aClientConfig::headers`] are sent with every request.
//!
//! A non-2xx status is [`Error::Transport`] (`"A2A <operation> failed with
//! status <code>"`), an unparseable body is [`Error::Transport`], and a
//! JSON-RPC `error` member becomes the crate error selected by its code
//! (`-32001` is [`Error::NotFound`], `-32601` is [`Error::Unsupported`], the
//! rest are [`Error::Validation`]).
//!
//! # Server-sent events
//!
//! A streaming response is a sequence of SSE frames, each terminated by a blank
//! line and carrying one `data:` payload that is a complete [`JsonRpcResponse`].
//! Frames may be split at any byte across HTTP chunks, so the parser
//! (`SseFramer`) is incremental over `bytes_stream()`: it never buffers the
//! whole body, accepts `\n`, `\r`, and `\r\n` line endings (including a `\r\n`
//! pair split across two chunks), joins multiple `data:` lines of one frame
//! with `\n`, and ignores comment lines (`:`), `event:`, `id:`, and `retry:`
//! fields as well as empty frames. At end of stream the only thing flushed is a
//! trailing frame that already carries a `data:` payload; a frame with no data
//! lines is discarded.
//!
//! # Timeouts, stop-on-error, and cancellation
//!
//! [`A2aClientConfig::timeout`] is a per-request budget (60 s when `None`). A
//! unary call ([`A2aClient::invoke`], [`A2aClient::cancel_task`]) spends it on
//! the whole exchange, body included; a streaming call
//! ([`A2aClient::stream`]) spends it on connecting and then as a per-chunk
//! *idle* deadline, so a long healthy stream is never cut off while a silent
//! peer still fails within the budget. Either budget maps onto
//! [`Error::Timeout`] (`"<method> timed out after <x>s"`), which is what a retry
//! policy keys on.
//!
//! [`A2aRemoteClient`] adds the Python `A2ARemoteClient` semantics on top: an
//! optional caller budget per call, and `stop()` before re-raising any failure
//! so a broken remote client is not reused. [`Error::Cancelled`] is the one
//! exception — it propagates without stopping the client, exactly like Python's
//! `asyncio.CancelledError` handling.
//!
//! # Documented deviations from Python
//!
//! 1. Starting a remote client performs no I/O: Python's `start()` only flips a
//!    flag as well, and this side builds the HTTP client eagerly in
//!    [`A2aRemoteClient::new`] (lazy start). [`A2aRemoteClient::start`] and
//!    [`A2aRemoteClient::stop`] are idempotent.
//! 2. [`A2aClient::stop`] is a documented no-op: `reqwest` pools connections
//!    inside the shared client and owns no SDK session to close. The method
//!    exists so the Python call sequence (`await client.stop()`) ports over.
//! 3. The supplied [`AgentCard`] is used as-is. Python rebuilds the card through
//!    the card adapter to embed the derived interface URL; here
//!    [`A2aRemoteClient::card`] returns the caller's card unchanged and the
//!    resolved endpoint is exposed separately by [`A2aRemoteClient::rpc_url`].
//! 4. [`RemoteClientConfig::default`] selects the only supported protocol,
//!    `A2A`, so a config assembled field by field does not have to repeat it.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_stream::try_stream;
use futures::stream::{BoxStream, Stream, StreamExt};
use openjiuwen_common::error::{Error, Result};
use reqwest::header::ACCEPT;
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use super::jsonrpc::{method, JsonRpcRequest, JsonRpcResponse};
use super::transform::{
    merge_results, resolve_session_id, with_session_id, A2aTransformer, AgentResult,
};
use super::types::{AgentCard, CancelTaskRequest};

/// The per-request timeout applied when [`A2aClientConfig::timeout`] is `None`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// The protocol binding this client speaks.
const JSONRPC_BINDING: &str = "JSONRPC";

/// The operation name the streaming timeout errors carry.
const STREAM_OPERATION: &str = "stream";

/// Settings for [`A2aClient`].
///
/// ```
/// use std::time::Duration;
/// use openjiuwen_extensions::a2a::client::A2aClientConfig;
///
/// let config = A2aClientConfig::new("http://127.0.0.1:8080/a2a/jsonrpc/")
///     .with_timeout(Duration::from_secs(30))
///     .with_header("x-api-key", "secret");
/// assert_eq!(config.timeout, Some(Duration::from_secs(30)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A2aClientConfig {
    /// The JSON-RPC endpoint, for example `http://127.0.0.1:8080/a2a/jsonrpc/`.
    ///
    /// An empty (or whitespace-only) URL is rejected by [`A2aClient::new`] with
    /// [`Error::Config`].
    pub rpc_url: String,
    /// The per-request timeout; 60 seconds when `None`.
    pub timeout: Option<Duration>,
    /// Extra HTTP headers sent with every request, keyed by header name.
    ///
    /// The values are never logged; header names are validated when a request is
    /// built, and an invalid one is [`Error::Config`].
    pub headers: BTreeMap<String, String>,
}

impl Default for A2aClientConfig {
    /// An empty configuration: no URL, the default timeout, no extra headers.
    fn default() -> Self {
        Self {
            rpc_url: String::new(),
            timeout: None,
            headers: BTreeMap::new(),
        }
    }
}

impl A2aClientConfig {
    /// A configuration posting to `rpc_url` with the default timeout and no
    /// extra headers.
    #[must_use]
    pub fn new(rpc_url: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            ..Self::default()
        }
    }

    /// Sets the per-request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Adds one HTTP header sent with every request, replacing a header of the
    /// same name.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// The per-request timeout actually applied.
    fn effective_timeout(&self) -> Duration {
        self.timeout.unwrap_or(DEFAULT_TIMEOUT)
    }
}

/// An incremental server-sent-events parser for one JSON-RPC streaming call.
///
/// Feed it HTTP body chunks with [`SseFramer::push`] and it returns the `data:`
/// payload of every frame completed by that chunk. Bytes are consumed one line
/// at a time and only an unterminated line is retained, so a frame split across
/// any number of chunks is reassembled without buffering the whole body.
///
/// The frame rules implemented here are the ones the A2A streams use (see the
/// [module docs](self)): a blank line ends a frame, several `data:` lines are
/// joined with `\n`, a `\r\n` split across two chunks counts as one terminator,
/// and comment, `event:`, `id:`, and `retry:` lines are ignored.
#[derive(Debug, Default)]
pub(crate) struct SseFramer {
    /// The bytes of the line currently being assembled.
    line: Vec<u8>,
    /// The `data:` values of the frame currently being assembled.
    data: Vec<String>,
    /// Whether the previous byte was a `\r` whose line terminator may still be
    /// completed by a `\n` arriving in the next chunk.
    pending_cr: bool,
}

impl SseFramer {
    /// An empty framer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Consumes one body chunk and returns the payloads of the frames it
    /// completed, in order.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut frames = Vec::new();
        for &byte in chunk {
            if self.pending_cr {
                self.pending_cr = false;
                // The `\r` ended the previous line; a following `\n` belongs to
                // the same `\r\n` terminator and adds nothing.
                self.complete_line(&mut frames);
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' => self.pending_cr = true,
                b'\n' => self.complete_line(&mut frames),
                other => self.line.push(other),
            }
        }
        frames
    }

    /// Flushes the end of the stream.
    ///
    /// Returns the payload of a trailing frame that already carried `data:`
    /// lines but was never terminated by a blank line; returns `None` when
    /// nothing is pending. Calling it twice is safe.
    pub(crate) fn finish(&mut self) -> Option<String> {
        let mut frames = Vec::new();
        if self.pending_cr {
            self.pending_cr = false;
            self.complete_line(&mut frames);
        }
        if !self.line.is_empty() {
            self.complete_line(&mut frames);
        }
        self.flush_frame(&mut frames);
        frames.into_iter().next()
    }

    /// Applies one complete line to the frame being assembled.
    fn complete_line(&mut self, frames: &mut Vec<String>) {
        let line = std::mem::take(&mut self.line);
        let text = String::from_utf8_lossy(&line).into_owned();
        if text.is_empty() {
            self.flush_frame(frames);
            return;
        }
        if text.starts_with(':') {
            // A comment: the usual keep-alive heartbeat, deliberately ignored.
            return;
        }
        let (field, value) = match text.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (text.as_str(), ""),
        };
        if field == "data" {
            self.data.push(value.to_owned());
        }
        // `event`, `id`, `retry`, and unknown fields cannot change a JSON-RPC
        // payload, so they are dropped.
    }

    /// Emits the frame being assembled, when it carried any `data:` line.
    fn flush_frame(&mut self, frames: &mut Vec<String>) {
        if self.data.is_empty() {
            return;
        }
        frames.push(self.data.join("\n"));
        self.data.clear();
    }
}

/// Parses one SSE `data:` payload into a JSON-RPC response.
///
/// # Errors
///
/// [`Error::Transport`] when the payload is not a JSON-RPC response object.
pub(crate) fn parse_frame(payload: &str) -> Result<JsonRpcResponse> {
    serde_json::from_str(payload).map_err(|error| {
        Error::transport(format!(
            "A2A stream frame is not a valid JSON-RPC response: {error}"
        ))
    })
}

/// Unwraps a JSON-RPC response into its `result`, mapping a peer error.
///
/// # Errors
///
/// The crate error selected by the JSON-RPC error code when the response carried
/// an `error`, or [`Error::Transport`] when it carried no `result` at all.
fn decode_response(response: JsonRpcResponse, operation: &str) -> Result<Value> {
    if let Some(error) = response.error {
        return Err(error.into_error());
    }
    response
        .result
        .ok_or_else(|| Error::transport(format!("A2A {operation} response is missing result")))
}

/// Decodes one SSE frame into an openJiuwen agent result.
///
/// # Errors
///
/// The same errors as [`parse_frame`] and [`decode_response`].
fn decode_frame(payload: &str, operation: &str) -> Result<AgentResult> {
    let response = parse_frame(payload)?;
    let result = decode_response(response, operation)?;
    Ok(A2aTransformer::from_response(&result))
}

/// Serializes the params of a JSON-RPC call.
///
/// # Errors
///
/// [`Error::Validation`] when `params` cannot be serialized (a programming
/// error for the wire types, which are plain JSON structs).
fn encode_params<T: Serialize>(params: &T, operation: &str) -> Result<Value> {
    serde_json::to_value(params).map_err(|error| {
        Error::validation(format!(
            "failed to encode the A2A {operation} params: {error}"
        ))
    })
}

/// Turns a non-2xx response into [`Error::Transport`].
///
/// # Errors
///
/// [`Error::Transport`] naming `operation` and the HTTP status code.
fn ensure_success(response: &reqwest::Response, operation: &str) -> Result<()> {
    if response.status().is_success() {
        return Ok(());
    }
    Err(Error::transport(format!(
        "A2A {operation} failed with status {}",
        response.status().as_u16()
    )))
}

/// Builds a [`Error::Timeout`] naming the operation and its budget.
fn timeout_error(operation: &str, budget: Option<Duration>) -> Error {
    match budget {
        Some(budget) => Error::timeout(format!(
            "A2A {operation} timed out after {}s",
            budget.as_secs_f64()
        )),
        None => Error::timeout(format!("A2A {operation} timed out")),
    }
}

/// Reads the next chunk of a streaming response, failing when the peer goes
/// quiet for longer than `idle`.
///
/// The deadline restarts on every chunk, so a stream may run for as long as the
/// peer keeps sending while a stalled one still fails within the configured
/// budget. A body error that the transport reports as a timeout keeps the
/// [`Error::Timeout`] kind instead of degrading into a transport error.
///
/// # Errors
///
/// [`Error::Timeout`] when `idle` elapses with no chunk, and [`Error::Transport`]
/// for any other body failure.
async fn next_body_chunk<T, S>(body: &mut S, idle: Duration) -> Result<Option<T>>
where
    S: Stream<Item = reqwest::Result<T>> + Unpin,
{
    match tokio::time::timeout(idle, body.next()).await {
        Ok(Some(Ok(chunk))) => Ok(Some(chunk)),
        Ok(Some(Err(error))) => Err(if error.is_timeout() {
            timeout_error(STREAM_OPERATION, Some(idle))
        } else {
            Error::transport(format!("A2A stream body failed: {error}"))
        }),
        Ok(None) => Ok(None),
        Err(_) => {
            tracing::debug!(
                idle_ms = u64::try_from(idle.as_millis()).unwrap_or(u64::MAX),
                "the A2A stream sent nothing before its idle deadline"
            );
            Err(timeout_error(STREAM_OPERATION, Some(idle)))
        }
    }
}

/// The shared state of one [`A2aClient`], behind an [`Arc`] so `stream` can
/// outlive the borrow of the client while keeping one request-id counter.
#[derive(Debug)]
struct ClientState {
    /// The pooled HTTP client.
    http: reqwest::Client,
    /// The configuration this client was built from.
    config: A2aClientConfig,
    /// The id of the next JSON-RPC request.
    next_id: AtomicU64,
}

/// The JSON-RPC client for one A2A agent endpoint.
///
/// Cloning is cheap: every clone shares the same connection pool, configuration,
/// and request-id counter, which is how [`A2aClient::stream`] returns a
/// `'static` stream.
#[derive(Debug, Clone)]
pub struct A2aClient {
    /// The shared state.
    state: Arc<ClientState>,
}

impl A2aClient {
    /// Builds a client for `config`.
    ///
    /// No I/O happens here; the HTTP client is created eagerly and connections
    /// are opened by the first call.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] when `config.rpc_url` is empty or whitespace-only, or
    /// when the HTTP client cannot be built (for example with a broken TLS
    /// backend).
    pub fn new(config: A2aClientConfig) -> Result<Self> {
        if config.rpc_url.trim().is_empty() {
            return Err(Error::config("A2A client requires a non-empty rpc_url"));
        }
        // The connect step is bounded on the client, because a streaming call
        // keeps reading long after the headers arrive and therefore cannot carry
        // a total request timeout.
        let http = reqwest::Client::builder()
            .connect_timeout(config.effective_timeout())
            .build()
            .map_err(|error| {
                Error::config(format!("failed to build the A2A HTTP client: {error}"))
            })?;
        Ok(Self {
            state: Arc::new(ClientState {
                http,
                config,
                next_id: AtomicU64::new(1),
            }),
        })
    }

    /// The configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &A2aClientConfig {
        &self.state.config
    }

    /// Sends one message and folds the response into a single result.
    ///
    /// The `SendMessage` result is mapped with
    /// [`A2aTransformer::from_response`] and folded onto an empty result with
    /// [`merge_results`]; the request's session id (`conversation_id`, then
    /// `sessionId`) is stamped on afterwards.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `inputs` is not a JSON object,
    /// [`Error::Transport`] for HTTP/JSON failures, [`Error::Timeout`] when the
    /// request exceeds the configured timeout, and the crate error selected by
    /// the peer's JSON-RPC error code.
    pub async fn invoke(&self, inputs: Value) -> Result<AgentResult> {
        let request = A2aTransformer::to_request(&inputs)?;
        let params = encode_params(&request, "send")?;
        let result = self.call(method::SEND_MESSAGE, params, "send").await?;
        let folded = merge_results(
            AgentResult::default(),
            A2aTransformer::from_response(&result),
        );
        Ok(with_session_id(
            folded,
            resolve_session_id(&inputs).as_deref(),
        ))
    }

    /// Streams one message and yields one [`AgentResult`] per SSE frame.
    ///
    /// The request is sent lazily (when the stream is first polled), and the
    /// HTTP response lives inside the returned stream, so dropping the stream
    /// drops the response and its connection instead of leaking the task.
    ///
    /// A frame carrying a JSON-RPC `error` yields `Err` and ends the stream; the
    /// same happens for a non-2xx status, a malformed frame payload, or a body
    /// read failure.
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentResult>> {
        let client = self.clone();
        let session_id = resolve_session_id(&inputs);
        Box::pin(try_stream! {
            let request = A2aTransformer::to_request(&inputs)?;
            let params = encode_params(&request, "stream")?;
            let response = client
                .post(method::SEND_STREAMING_MESSAGE, params, true)
                .await?;
            ensure_success(&response, "stream")?;
            let idle = client.state.config.effective_timeout();
            let mut body = Box::pin(response.bytes_stream());
            let mut framer = SseFramer::new();
            while let Some(chunk) = next_body_chunk(&mut body, idle).await? {
                for payload in framer.push(&chunk) {
                    let result = decode_frame(&payload, "stream")?;
                    yield with_session_id(result, session_id.as_deref());
                }
            }
            if let Some(payload) = framer.finish() {
                let result = decode_frame(&payload, "stream")?;
                yield with_session_id(result, session_id.as_deref());
            }
        })
    }

    /// Asks the agent to cancel a task.
    ///
    /// `tenant` is forwarded only when set, matching the Python client.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] for HTTP/JSON failures, [`Error::Timeout`] when the
    /// request exceeds the configured timeout, and the crate error selected by
    /// the peer's JSON-RPC error code (for example [`Error::NotFound`] when the
    /// task id is unknown).
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult> {
        let request = CancelTaskRequest {
            id: task_id.to_owned(),
            tenant: tenant.map(str::to_owned),
            ..CancelTaskRequest::default()
        };
        let params = encode_params(&request, "cancel")?;
        let result = self.call(method::CANCEL_TASK, params, "cancel").await?;
        Ok(A2aTransformer::from_response(&result))
    }

    /// Releases the resources held by the client.
    ///
    /// This is a documented no-op: `reqwest` pools connections inside the shared
    /// client and there is no SDK session to close. The method exists — and
    /// stays `async` — so the Python `await client.stop()` sequence ports over.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn stop(&self) {
        tracing::debug!(url = %self.state.config.rpc_url, "A2A client stop is a no-op");
    }

    /// Posts one JSON-RPC request and returns the raw HTTP response.
    ///
    /// Applies the configured headers, the per-request timeout, and a fresh
    /// request id; `accept_sse` asks for an event stream.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] for an invalid configured header name or value,
    /// [`Error::Timeout`] when the transport reports a timeout, and
    /// [`Error::Transport`] for every other transport failure.
    async fn post(
        &self,
        method_name: &str,
        params: Value,
        accept_sse: bool,
    ) -> Result<reqwest::Response> {
        let id = self.state.next_id.fetch_add(1, Ordering::Relaxed);
        let envelope = JsonRpcRequest::new(id, method_name, Some(params));
        let timeout = self.state.config.effective_timeout();
        tracing::debug!(
            method = method_name,
            id,
            url = %self.state.config.rpc_url,
            "sending an A2A JSON-RPC request"
        );
        // A streaming response is read chunk by chunk long after the headers
        // arrive, so a total request timeout would kill a healthy long stream;
        // only the unary path is bounded as a whole. The connect step is bounded
        // on the client itself, and the stream applies `timeout` as an idle
        // deadline between chunks.
        let mut builder = self
            .state
            .http
            .post(&self.state.config.rpc_url)
            .json(&envelope);
        if accept_sse {
            builder = builder.header(ACCEPT, "text/event-stream");
        } else {
            builder = builder.timeout(timeout);
        }
        for (name, value) in &self.state.config.headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                    Error::config(format!("invalid A2A header name `{name}`: {error}"))
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
                Error::config(format!("invalid A2A header value for `{name}`: {error}"))
            })?;
            builder = builder.header(header_name, header_value);
        }
        builder.send().await.map_err(|error| {
            if error.is_timeout() {
                timeout_error(method_name, Some(timeout))
            } else {
                Error::transport(format!("A2A {method_name} request failed: {error}"))
            }
        })
    }

    /// Performs one non-streaming JSON-RPC call and returns its `result`.
    ///
    /// # Errors
    ///
    /// [`Error::Transport`] for a non-2xx status, an unreadable body, or a body
    /// that is not a JSON-RPC response; the crate error selected by the peer's
    /// JSON-RPC error code; [`Error::Transport`] when no `result` is present.
    async fn call(&self, method_name: &str, params: Value, operation: &str) -> Result<Value> {
        let response = self.post(method_name, params, false).await?;
        ensure_success(&response, operation)?;
        let body = response.text().await.map_err(|error| {
            Error::transport(format!(
                "A2A {operation} response could not be read: {error}"
            ))
        })?;
        let envelope: JsonRpcResponse = serde_json::from_str(&body).map_err(|error| {
            Error::transport(format!(
                "A2A {operation} returned an invalid JSON-RPC response: {error}"
            ))
        })?;
        decode_response(envelope, operation)
    }
}

/// Settings for [`A2aRemoteClient`].
///
/// Mirrors the Python `RemoteClientConfig`: identity fields plus the protocol,
/// the agent card, the base URL, an optional timeout, and free-form `kwargs`
/// (Python reads its card out of `kwargs`, here [`RemoteClientConfig::card`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteClientConfig {
    /// The remote client id, used in logs and diagnostics.
    pub id: String,
    /// The version of the remote agent.
    pub version: String,
    /// The display name of the remote agent.
    pub name: String,
    /// The description of the remote agent.
    pub description: String,
    /// The transport protocol; only `A2A` (case-insensitive) is supported.
    pub protocol: String,
    /// The agent card published by the remote agent.
    pub card: Option<AgentCard>,
    /// The base URL of the remote agent, used when the card advertises no
    /// usable JSON-RPC interface.
    pub url: String,
    /// The default per-call timeout; `None` leaves calls unbounded beyond the
    /// client's own request timeout.
    pub timeout: Option<Duration>,
    /// Free-form provider-specific settings.
    pub kwargs: serde_json::Map<String, Value>,
}

impl Default for RemoteClientConfig {
    /// An empty configuration that already selects the only supported protocol,
    /// `A2A`, so a config assembled field by field does not have to repeat it.
    fn default() -> Self {
        Self {
            id: String::new(),
            version: String::new(),
            name: String::new(),
            description: String::new(),
            protocol: String::from("A2A"),
            card: None,
            url: String::new(),
            timeout: None,
            kwargs: serde_json::Map::new(),
        }
    }
}

impl RemoteClientConfig {
    /// A configuration for the remote client `id` reachable at `url`, with the
    /// `A2A` protocol and no card yet.
    #[must_use]
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            ..Self::default()
        }
    }
}

/// The Python-compatible remote-client adapter over [`A2aClient`].
///
/// Resolves the endpoint from the agent card (or derives it from the configured
/// base URL), keeps the started flag, and applies the caller's timeout with
/// stop-on-error semantics.
#[derive(Debug, Clone)]
pub struct A2aRemoteClient {
    /// The configuration this client was built from.
    config: RemoteClientConfig,
    /// The agent card supplied by the caller, returned unchanged.
    card: AgentCard,
    /// The underlying JSON-RPC client.
    client: A2aClient,
    /// Whether the client has been started; shared with the stream so a failed
    /// stream can clear it.
    started: Arc<AtomicBool>,
}

impl A2aRemoteClient {
    /// Builds a remote client from `config`.
    ///
    /// The endpoint is the first `supportedInterfaces` entry with a JSON-RPC
    /// binding and a non-empty URL; otherwise it is derived from `config.url`
    /// (see the [module docs](self)). No I/O happens here.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the protocol is not `A2A`, when no card is
    /// supplied, when neither the card nor `config.url` names an endpoint, or
    /// when the derived [`A2aClient`] rejects the configuration.
    pub fn new(config: RemoteClientConfig) -> Result<Self> {
        if !config.protocol.eq_ignore_ascii_case("A2A") {
            return Err(Error::validation(format!(
                "unsupported protocol '{}', expected A2A",
                config.protocol
            )));
        }
        let card = config
            .card
            .clone()
            .ok_or_else(|| Error::validation("A2A remote client requires an agent card"))?;
        let rpc_url = resolve_rpc_url(&card, &config.url)?;
        let client = A2aClient::new(A2aClientConfig {
            rpc_url,
            timeout: config.timeout,
            headers: BTreeMap::new(),
        })?;
        tracing::debug!(
            client = %config.id,
            url = %config.url,
            "initialized the A2A remote client"
        );
        Ok(Self {
            config,
            card,
            client,
            started: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &RemoteClientConfig {
        &self.config
    }

    /// The agent card supplied by the caller.
    #[must_use]
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// The JSON-RPC endpoint the client posts to.
    #[must_use]
    pub fn rpc_url(&self) -> &str {
        self.client.config().rpc_url.as_str()
    }

    /// Marks the client as started.
    ///
    /// Starting is lazy and idempotent: no connection is opened, matching
    /// Python's `A2ARemoteClient.start`. The method stays `async` so the Python
    /// `await client.start()` sequence ports over.
    ///
    /// # Errors
    ///
    /// This implementation cannot fail; the [`Result`] is part of the async
    /// lifecycle contract shared with the other remote-client implementations.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn start(&self) -> Result<()> {
        self.started.store(true, Ordering::Release);
        tracing::debug!(client = %self.config.id, "started the A2A remote client");
        Ok(())
    }

    /// Whether the client has been started.
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    /// Stops the client.
    ///
    /// Idempotent: the underlying client is stopped only when the client was
    /// started, and the flag is cleared either way.
    pub async fn stop(&self) {
        if self.started.swap(false, Ordering::AcqRel) {
            self.client.stop().await;
            tracing::debug!(client = %self.config.id, "stopped the A2A remote client");
        }
    }

    /// Invokes the remote agent and returns its folded result.
    ///
    /// `timeout` is an optional total budget for the call. Any failure stops the
    /// client before the error is returned; [`Error::Cancelled`] is the
    /// exception and propagates without stopping it.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when `timeout` elapses, plus every error
    /// [`A2aClient::invoke`] can return.
    pub async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentResult> {
        tracing::debug!(client = %self.config.id, "invoking the A2A remote agent");
        let session_id = resolve_session_id(&inputs);
        let outcome = match timeout {
            Some(budget) => match tokio::time::timeout(budget, self.client.invoke(inputs)).await {
                Ok(outcome) => outcome,
                Err(_) => Err(timeout_error("invoke", Some(budget))),
            },
            None => self.client.invoke(inputs).await,
        };
        let result = self.stop_on_error(outcome).await?;
        Ok(with_session_id(result, session_id.as_deref()))
    }

    /// Streams results from the remote agent.
    ///
    /// `timeout` bounds the whole stream: the deadline is computed once when the
    /// stream is created and only the remaining budget is given to each
    /// `next()`. Every chunk carries the request's session id.
    ///
    /// A failed stream clears the started flag so the client is not reused;
    /// calling `stop()` is unnecessary because the underlying client holds no
    /// session. [`Error::Cancelled`] propagates without touching the flag.
    /// Exceeding the deadline yields [`Error::Timeout`].
    pub fn stream(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentResult>> {
        let session_id = resolve_session_id(&inputs);
        let client = self.client.clone();
        let started = Arc::clone(&self.started);
        let deadline = timeout.map(|budget| Instant::now() + budget);
        Box::pin(try_stream! {
            let mut inner = client.stream(inputs);
            loop {
                match next_chunk(&mut inner, deadline, timeout).await {
                    Ok(Some(result)) => yield with_session_id(result, session_id.as_deref()),
                    Ok(None) => break,
                    Err(error) => {
                        if !error.is_cancelled() {
                            started.store(false, Ordering::Release);
                        }
                        tracing::debug!(
                            error = %error,
                            "the A2A remote stream stopped with an error"
                        );
                        Err::<(), Error>(error)?;
                    }
                }
            }
        })
    }

    /// Cancels a remote task.
    ///
    /// The result is returned as-is (no session id is stamped on it, matching
    /// Python). Any failure stops the client before the error is returned, with
    /// [`Error::Cancelled`] propagating without stopping it.
    ///
    /// # Errors
    ///
    /// Every error [`A2aClient::cancel_task`] can return.
    pub async fn cancel_task(&self, task_id: &str, tenant: Option<&str>) -> Result<AgentResult> {
        tracing::debug!(client = %self.config.id, task_id, "cancelling an A2A remote task");
        let outcome = self.client.cancel_task(task_id, tenant).await;
        self.stop_on_error(outcome).await
    }

    /// Stops the client before re-raising a failure.
    ///
    /// [`Error::Cancelled`] is returned unchanged without stopping the client,
    /// mirroring Python's `CancelledError` handling.
    ///
    /// # Errors
    ///
    /// The error carried by `outcome`, when it is one.
    async fn stop_on_error<T>(&self, outcome: Result<T>) -> Result<T> {
        match outcome {
            Ok(value) => Ok(value),
            Err(error) if error.is_cancelled() => Err(error),
            Err(error) => {
                tracing::warn!(
                    client = %self.config.id,
                    "an A2A remote call failed; stopping the remote client"
                );
                tracing::debug!(client = %self.config.id, error = %error, "A2A failure detail");
                self.stop().await;
                Err(error)
            }
        }
    }
}

/// Advances a boxed stream, honouring an optional deadline.
///
/// # Errors
///
/// [`Error::Timeout`] when `deadline` has passed, plus any error the stream
/// itself yields.
async fn next_chunk(
    stream: &mut BoxStream<'static, Result<AgentResult>>,
    deadline: Option<Instant>,
    budget: Option<Duration>,
) -> Result<Option<AgentResult>> {
    match deadline {
        Some(deadline) => match tokio::time::timeout_at(deadline, stream.next()).await {
            Ok(next) => next.transpose(),
            Err(_) => Err(timeout_error("stream", budget)),
        },
        None => stream.next().await.transpose(),
    }
}

/// Resolves the JSON-RPC endpoint of a remote client.
///
/// # Errors
///
/// [`Error::Validation`] when the card has no usable JSON-RPC interface and
/// `url` is empty.
fn resolve_rpc_url(card: &AgentCard, url: &str) -> Result<String> {
    if let Some(interface_url) = jsonrpc_interface_url(card) {
        return Ok(interface_url.to_owned());
    }
    if !url.is_empty() {
        return Ok(derive_interface_url(url));
    }
    Err(Error::validation("A2A remote client requires a url"))
}

/// The first JSON-RPC interface URL advertised by `card`.
///
/// Matching is case-insensitive on the binding and skips entries with an empty
/// URL, like the Python `supported_interfaces` scan.
fn jsonrpc_interface_url(card: &AgentCard) -> Option<&str> {
    card.supported_interfaces
        .iter()
        .find(|interface| {
            interface
                .protocol_binding
                .eq_ignore_ascii_case(JSONRPC_BINDING)
                && !interface.url.is_empty()
        })
        .map(|interface| interface.url.as_str())
}

/// Derives the published interface URL from a base URL, exactly like Python.
///
/// A trailing `/` is stripped; a URL that already ends in `/a2a/jsonrpc` gets
/// one `/` appended, anything else gets `/a2a/jsonrpc/`.
fn derive_interface_url(url: &str) -> String {
    let normalized = url.trim_end_matches('/');
    if normalized.ends_with("/a2a/jsonrpc") {
        format!("{normalized}/")
    } else {
        format!("{normalized}/a2a/jsonrpc/")
    }
}

#[cfg(test)]
mod tests;
