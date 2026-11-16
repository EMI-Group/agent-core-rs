//! The legacy HTTP+SSE transport of [`SseTransport`].
//!
//! Split out of `http.rs` to keep both files well under the crate's size
//! baseline. The transport implements [`rmcp::transport::Transport`] directly on
//! `reqwest`: one `GET` event stream, the POST endpoint learned from the
//! `endpoint` event, and `Last-Event-ID` replay when the stream is reopened.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::Result;
use reqwest::header::CONTENT_TYPE;
use rmcp::model::ServerJsonRpcMessage;
use rmcp::service::TxJsonRpcMessage;
use rmcp::{transport::Transport, RoleClient};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::super::client::{reconnect_attempt_budget, McpServerConfig};
use super::{
    forward_frame, stream_reconnect_delay, Endpoint, HttpTransportError, SseParser,
    EVENT_STREAM_MIME_TYPE, HEADER_LAST_EVENT_ID, JSON_MIME_TYPE,
};

/// How long an SSE transport waits for the server's `endpoint` event.
const SSE_ENDPOINT_TIMEOUT: Duration = Duration::from_secs(30);

/// A legacy HTTP+SSE transport.
pub(crate) struct SseTransport {
    endpoint: Endpoint,
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
    started: Arc<AtomicBool>,
    inbound: UnboundedReceiver<ServerJsonRpcMessage>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
    /// The id of the last event this transport accepted, replayed on every `GET`.
    last_event_id: Arc<Mutex<Option<String>>>,
    /// Why the `GET` loop stopped for good, when it stopped before a `send` saw it.
    stop: Arc<Mutex<Option<SseStop>>>,
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
            last_event_id: Arc::new(Mutex::new(None)),
            stop: Arc::new(Mutex::new(None)),
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
        let last_event_id = self.last_event_id.clone();
        let stop = self.stop.clone();
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
                    last_event_id.clone(),
                    stop.clone(),
                );
            }
            let target = match wait_post_endpoint(post_endpoint, endpoint_ready, stop).await {
                PostEndpoint::Ready(target) => target,
                // The loop records why it could not (re)establish the stream, so a
                // post that would otherwise wait for a bare timeout reports it and
                // says what was replayed.
                PostEndpoint::Stopped(stop) => {
                    let last = last_event_id.lock().ok().and_then(|slot| slot.clone());
                    return Err(sse_resume_error(stop, last.as_deref()));
                }
                PostEndpoint::TimedOut => {
                    return Err(HttpTransportError::protocol(
                        "timed out waiting for the MCP SSE endpoint event",
                    ));
                }
            };
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

/// Why a `send` could not obtain the POST endpoint of the SSE session.
enum PostEndpoint {
    /// The stream announced the endpoint.
    Ready(String),
    /// The stream loop stopped for good; the value says why.
    Stopped(SseStop),
    /// No endpoint arrived within [`SSE_ENDPOINT_TIMEOUT`].
    TimedOut,
}

/// Waits for the SSE `endpoint` event, bounded by [`SSE_ENDPOINT_TIMEOUT`].
///
/// Returns early with [`PostEndpoint::Stopped`] when the stream loop recorded an
/// [`SseStop`], so a request whose session could not be established reports the
/// typed reason instead of waiting out the deadline.
async fn wait_post_endpoint(
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
    stop: Arc<Mutex<Option<SseStop>>>,
) -> PostEndpoint {
    let deadline = tokio::time::Instant::now() + SSE_ENDPOINT_TIMEOUT;
    loop {
        if let Some(endpoint) = post_endpoint.lock().ok().and_then(|slot| slot.clone()) {
            return PostEndpoint::Ready(endpoint);
        }
        if let Some(stop) = stop.lock().ok().and_then(|slot| *slot) {
            return PostEndpoint::Stopped(stop);
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return PostEndpoint::TimedOut;
        }
        let _ = tokio::time::timeout(remaining, endpoint_ready.notified()).await;
    }
}

/// Why the legacy SSE `GET` loop stopped for good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SseStop {
    /// The server answered the `GET` stream with a non-success status; never retried.
    Rejected(u16),
    /// The reconnect budget ran out (the value is the number of attempts made).
    Exhausted(u32),
}

/// Builds the headers of a legacy SSE `GET` request.
///
/// Always advertises `text/event-stream`; `Last-Event-ID` is added once this
/// transport accepted an event, so the server replays everything after it instead
/// of restarting (or dropping) the stream.
pub(crate) fn sse_stream_headers(last_event_id: Option<&str>) -> Vec<(&'static str, String)> {
    let mut headers = vec![("Accept", EVENT_STREAM_MIME_TYPE.to_owned())];
    if let Some(id) = last_event_id {
        headers.push((HEADER_LAST_EVENT_ID, id.to_owned()));
    }
    headers
}

/// Builds the error a legacy SSE transport reports for a stream it cannot resume.
///
/// A rejected stream keeps the typed status; an exhausted budget names the last
/// event the client accepted, so an operator knows how far the replay got.
pub(crate) fn sse_resume_error(stop: SseStop, last_event_id: Option<&str>) -> HttpTransportError {
    match stop {
        SseStop::Rejected(status) => HttpTransportError::Status(status),
        SseStop::Exhausted(attempts) => HttpTransportError::protocol(format!(
            "MCP SSE event stream could not be resumed{}: the reconnect budget of {attempts} \
             attempt(s) ran out",
            match last_event_id {
                Some(id) => format!(" after event id `{id}`"),
                None => String::new(),
            }
        )),
    }
}

/// Spawns the long-lived GET event stream of an SSE transport.
///
/// The loop owns reconnection: a body that ends or fails is reopened up to
/// [`reconnect_attempt_budget`] attempts, each carrying the `Last-Event-ID` of the
/// last accepted event, with [`stream_reconnect_delay`] between attempts (the
/// server's `retry` hint wins, but never exceeds the shared policy). A non-success
/// status ends the loop without a retry, and an exhausted budget ends it too; both
/// record an [`SseStop`] the next [`Transport::send`] reports as a typed error
/// instead of a bare endpoint timeout, and both clear the start-once flag so that
/// send can open a fresh stream.
fn spawn_sse_stream(
    endpoint: Endpoint,
    post_endpoint: Arc<Mutex<Option<String>>>,
    endpoint_ready: Arc<Notify>,
    started: Arc<AtomicBool>,
    outbound: UnboundedSender<ServerJsonRpcMessage>,
    cancellation: CancellationToken,
    last_event_id: Arc<Mutex<Option<String>>>,
    stop: Arc<Mutex<Option<SseStop>>>,
) {
    tokio::spawn(async move {
        let attempts = reconnect_attempt_budget();
        let mut attempt: u32 = 1;
        let mut hint: Option<Duration> = None;
        loop {
            if let Ok(mut slot) = stop.lock() {
                *slot = None;
            }
            let outcome = run_sse_stream(
                &endpoint,
                &post_endpoint,
                &endpoint_ready,
                &outbound,
                &cancellation,
                &last_event_id,
                &mut hint,
            )
            .await;
            let reason = match outcome {
                SseOutcome::Cancelled => return,
                SseOutcome::Rejected(status) => SseStop::Rejected(status),
                SseOutcome::Retryable => {
                    if attempt >= attempts {
                        SseStop::Exhausted(attempts)
                    } else {
                        let delay = stream_reconnect_delay(attempt, hint);
                        tracing::debug!(
                            attempt,
                            attempts,
                            delay_ms = delay.as_millis(),
                            "MCP SSE event stream ended; reconnecting"
                        );
                        tokio::select! {
                            () = cancellation.cancelled() => return,
                            () = tokio::time::sleep(delay) => {}
                        }
                        attempt = attempt.saturating_add(1);
                        continue;
                    }
                }
            };
            tracing::debug!(
                ?reason,
                "MCP SSE event stream stopped and cannot be resumed by this transport"
            );
            if let Ok(mut slot) = stop.lock() {
                *slot = Some(reason);
            }
            // Let the next send start a fresh stream.
            if let Ok(mut slot) = post_endpoint.lock() {
                *slot = None;
            }
            started.store(false, Ordering::SeqCst);
            // Wake a send that is still waiting for the endpoint event.
            endpoint_ready.notify_waiters();
            return;
        }
    });
}

/// Why one attempt at reading the legacy SSE stream ended.
enum SseOutcome {
    /// The server answered with a non-success status; never retried.
    Rejected(u16),
    /// The body ended or failed; it may be resumed from the last event id.
    Retryable,
    /// The transport was closed.
    Cancelled,
}

/// Runs one attempt at the legacy SSE `GET` stream.
///
/// Every accepted event id is recorded in `last_event_id` (and the server's
/// `retry` hint in `hint`) so the next attempt replays from the last event the
/// client saw.
async fn run_sse_stream(
    endpoint: &Endpoint,
    post_endpoint: &Arc<Mutex<Option<String>>>,
    endpoint_ready: &Arc<Notify>,
    outbound: &UnboundedSender<ServerJsonRpcMessage>,
    cancellation: &CancellationToken,
    last_event_id: &Arc<Mutex<Option<String>>>,
    hint: &mut Option<Duration>,
) -> SseOutcome {
    let mut request = endpoint
        .client
        .get(&endpoint.url)
        .headers(endpoint.headers.clone());
    let replay = last_event_id.lock().ok().and_then(|slot| slot.clone());
    for (name, value) in sse_stream_headers(replay.as_deref()) {
        request = request.header(name, value);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(%error, "MCP SSE endpoint could not be reached");
            return SseOutcome::Retryable;
        }
    };
    let status = response.status();
    if !status.is_success() {
        tracing::debug!(
            status = status.as_u16(),
            "MCP SSE endpoint rejected the event stream"
        );
        return SseOutcome::Rejected(status.as_u16());
    }
    let mut stream = response.bytes_stream();
    let mut parser = SseParser::default();
    loop {
        tokio::select! {
            () = cancellation.cancelled() => return SseOutcome::Cancelled,
            chunk = stream.next() => match chunk {
                Some(Ok(bytes)) => {
                    for frame in parser.push(&bytes) {
                        if let Some(id) = frame.id.clone() {
                            if let Ok(mut slot) = last_event_id.lock() {
                                *slot = Some(id);
                            }
                        }
                        if let Some(retry) = frame.retry {
                            *hint = Some(Duration::from_millis(retry));
                        }
                        forward_frame(&frame, outbound, Some(post_endpoint));
                    }
                    endpoint_ready.notify_waiters();
                }
                Some(Err(error)) => {
                    tracing::debug!(%error, "MCP SSE event stream failed");
                    return SseOutcome::Retryable;
                }
                None => return SseOutcome::Retryable,
            },
        }
    }
}
