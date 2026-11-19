//! A loopback OpenAI-compatible HTTP server for tests.
//!
//! [`MockLlmServer`] binds `127.0.0.1:0`, lets the OS assign the port, and
//! speaks the chat-completions wire format so the real
//! [`OpenAiCompatibleClient`] exercises its transport, SSE decoder, retry, and
//! error mapping against a socket instead of a stub.
//!
//! The scripted queue mirrors `tests/unit_tests/fixtures/mock_llm.py`: answers
//! are popped in order with no prompt matching, and once the queue is empty the
//! fallback answer is reused without ever being consumed.
//!
//! ```
//! # use openjiuwen_test_support::{MockLlmServer, ScriptedResponse};
//! # async fn example() {
//! let server = MockLlmServer::with_script([ScriptedResponse::text("hello")]).await;
//! assert_eq!(server.base_url(), format!("http://{}/v1", server.addr()));
//! assert_eq!(server.request_count(), 0);
//! server.shutdown().await;
//! # }
//! ```
//!
//! [`OpenAiCompatibleClient`]: openjiuwen_foundation::llm::OpenAiCompatibleClient

mod routes;
mod state;
#[cfg(test)]
mod tests;
mod wire;

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::Router;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ModelClientConfig, ModelRequestConfig, OpenAiCompatibleClient};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tower_service::Service as _;

use state::{ServerConfig, ServerState};

pub use state::{RecordedRequest, ScriptedResponse};
pub use wire::MockHttpError;

/// The API key every [`MockLlmServer`] client uses.
pub const TEST_API_KEY: &str = "test-api-key";

/// Upper bound on how long [`MockLlmServer::shutdown`] waits for the server task.
///
/// A client that left a keep-alive connection open must not hang a test, so the
/// wait is bounded and the task is aborted when the bound expires.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// How often a drain re-checks the live connection tasks.
const DRAIN_POLL: Duration = Duration::from_millis(5);

/// The connection tasks one server spawned, so they can be aborted on `Drop`.
///
/// `axum::serve` spawns one task per accepted connection and hands no handle
/// back, so aborting it left every in-flight connection — and its SSE stream —
/// running after the test finished. The mock server owns its accept loop instead
/// and records each task it spawns here; [`MockLlmServer`]'s `Drop` aborts them
/// all.
///
/// A handle obtained with [`MockLlmServer::connections`] stays usable after the
/// server is dropped, which is how a test asserts that nothing is still running.
#[derive(Debug, Clone, Default)]
pub struct ConnectionTasks {
    /// Every connection task that has not been pruned yet.
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl ConnectionTasks {
    /// Number of connection tasks that have not finished yet.
    #[must_use]
    pub fn live(&self) -> usize {
        state::lock(&self.handles)
            .iter()
            .filter(|handle| !handle.is_finished())
            .count()
    }

    /// Number of connection tasks this server ever spawned.
    #[must_use]
    pub fn spawned(&self) -> usize {
        state::lock(&self.handles).len()
    }

    /// Records one spawned connection task, dropping the handles that finished.
    fn track(&self, handle: JoinHandle<()>) {
        let mut handles = state::lock(&self.handles);
        handles.retain(|tracked| !tracked.is_finished());
        handles.push(handle);
    }

    /// Aborts every connection task that is still running.
    fn abort_all(&self) {
        for handle in state::lock(&self.handles).drain(..) {
            handle.abort();
        }
    }

    /// Waits up to `timeout` for every connection task to finish on its own.
    async fn wait_idle(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while self.live() > 0 && Instant::now() < deadline {
            tokio::time::sleep(DRAIN_POLL).await;
        }
    }
}

/// A running mock server plus the handle to its task.
///
/// Dropping the handle aborts the server task and every connection it spawned;
/// call [`MockLlmServer::shutdown`] for a graceful stop that drains in-flight
/// responses first.
pub struct MockLlmServer {
    /// Shared script queue, recordings, and configuration.
    state: Arc<ServerState>,
    /// Loopback address the server is listening on.
    addr: SocketAddr,
    /// `http://127.0.0.1:<port>/v1`.
    base_url: String,
    /// `base_url` plus `/chat/completions`.
    endpoint: String,
    /// Wakes the accept loop out of `accept`.
    shutdown_signal: Arc<Notify>,
    /// The accept-loop task, taken by [`MockLlmServer::shutdown`] and `Drop`.
    task: Mutex<Option<JoinHandle<()>>>,
    /// The connection tasks the accept loop spawned.
    connections: Arc<ConnectionTasks>,
}

impl MockLlmServer {
    /// Creates a builder with an empty script and the default fallback.
    #[must_use]
    pub fn builder() -> MockLlmServerBuilder {
        MockLlmServerBuilder::default()
    }

    /// Starts a server with an empty script and the default fallback.
    ///
    /// # Panics
    ///
    /// Panics when the loopback listener cannot be bound. Use
    /// [`MockLlmServer::try_start`] to handle that failure instead.
    pub async fn start() -> Self {
        match Self::builder().try_start().await {
            Ok(server) => server,
            Err(error) => panic!("the mock LLM server could not bind 127.0.0.1:0: {error}"),
        }
    }

    /// Starts a server with an empty script, returning a bind failure instead of
    /// panicking.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the loopback listener cannot be bound.
    pub async fn try_start() -> Result<Self> {
        Self::builder().try_start().await
    }

    /// Starts a server whose script holds `items`, in order.
    ///
    /// # Panics
    ///
    /// Panics when the loopback listener cannot be bound, like
    /// [`MockLlmServer::start`].
    pub async fn with_script(items: impl IntoIterator<Item = ScriptedResponse>) -> Self {
        match Self::builder().script(items).try_start().await {
            Ok(server) => server,
            Err(error) => panic!("the mock LLM server could not bind 127.0.0.1:0: {error}"),
        }
    }

    /// The API base the client should be configured with:
    /// `http://127.0.0.1:<port>/v1`.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The chat-completions endpoint (`base_url` plus `/chat/completions`).
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The loopback address the server is listening on.
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// A client configuration pointed at this server.
    ///
    /// Provider `openai`, API key [`TEST_API_KEY`], and `max_retries = 0`, so a
    /// scripted 5xx fails on the first attempt instead of retrying.
    #[must_use]
    pub fn client_config(&self) -> ModelClientConfig {
        ModelClientConfig::new("openai")
            .with_api_key(TEST_API_KEY)
            .with_api_base(self.base_url())
            .with_max_retries(0)
    }

    /// A per-call request configuration for `model`.
    #[must_use]
    pub fn request_config(&self, model: &str) -> ModelRequestConfig {
        ModelRequestConfig::new(model)
    }

    /// Builds an [`OpenAiCompatibleClient`] pointed at this server.
    ///
    /// # Errors
    ///
    /// Returns the configuration or transport errors of
    /// [`OpenAiCompatibleClient::new`].
    pub fn openai_client(&self, model: &str) -> Result<OpenAiCompatibleClient> {
        OpenAiCompatibleClient::new(self.client_config(), self.request_config(model))
    }

    /// Appends one scripted answer to the queue.
    pub fn push(&self, response: ScriptedResponse) {
        self.state.push(response);
    }

    /// Appends several scripted answers, preserving order.
    pub fn push_many(&self, items: impl IntoIterator<Item = ScriptedResponse>) {
        self.state.push_many(items);
    }

    /// Drops every queued answer; the fallback and the recordings stay.
    pub fn clear_script(&self) {
        self.state.clear_script();
    }

    /// Replaces the answer served once the queue is empty.
    pub fn set_fallback(&self, response: ScriptedResponse) {
        self.state.set_fallback(response);
    }

    /// Number of queued answers.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.state.queued()
    }

    /// Number of requests seen so far.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.state.request_count()
    }

    /// Every recorded request, in arrival order.
    #[must_use]
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state.requests()
    }

    /// The most recent recorded request, if any.
    #[must_use]
    pub fn last_request(&self) -> Option<RecordedRequest> {
        self.state.last_request()
    }

    /// Waits until at least `count` requests have been recorded, bounded by
    /// [`DEFAULT_EVENTUALLY_TIMEOUT`](crate::DEFAULT_EVENTUALLY_TIMEOUT).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the bound expires first; the message
    /// reports the expected and the observed count, so a script mismatch fails
    /// the test instead of hanging it. Use
    /// [`MockLlmServer::wait_for_requests_timeout`] for another bound.
    pub async fn wait_for_requests(&self, count: usize) -> Result<()> {
        self.state
            .wait_for_requests_timeout(count, crate::DEFAULT_EVENTUALLY_TIMEOUT)
            .await
    }

    /// Waits until at least `count` requests have been recorded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when `timeout` expires first; the message
    /// reports how many requests the server saw.
    pub async fn wait_for_requests_timeout(&self, count: usize, timeout: Duration) -> Result<()> {
        self.state.wait_for_requests_timeout(count, timeout).await
    }

    /// The connection tasks this server spawned.
    ///
    /// The handle stays valid after the server is dropped, so a test can assert
    /// that no connection task outlives it.
    #[must_use]
    pub fn connections(&self) -> ConnectionTasks {
        ConnectionTasks {
            handles: Arc::clone(&self.connections.handles),
        }
    }

    /// Number of connection tasks that have not finished yet.
    #[must_use]
    pub fn live_connections(&self) -> usize {
        self.connections.live()
    }

    /// Stops the server gracefully, aborting the task if it does not wind down
    /// within `SHUTDOWN_TIMEOUT`.
    ///
    /// In-flight connections are given `SHUTDOWN_TIMEOUT` to finish on their
    /// own; whatever is still running after that is aborted, so no connection
    /// task survives the call.
    pub async fn shutdown(self) {
        self.shutdown_signal.notify_one();
        let handle = state::lock(&self.task).take();
        if let Some(mut handle) = handle {
            if tokio::time::timeout(SHUTDOWN_TIMEOUT, &mut handle)
                .await
                .is_err()
            {
                handle.abort();
            }
        }
        self.connections.wait_idle(SHUTDOWN_TIMEOUT).await;
        self.connections.abort_all();
    }
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        self.shutdown_signal.notify_one();
        if let Some(handle) = state::lock(&self.task).take() {
            handle.abort();
        }
        // The accept task only owns the listener: every connection it spawned is
        // a separate task, so aborting them is what actually stops the server.
        self.connections.abort_all();
    }
}

/// Accepts connections until `shutdown` is signalled, tracking every task spawned.
///
/// This replaces `axum::serve`, which spawns connection tasks internally and
/// hands out no handle: an aborted `axum::serve` left its connections running.
async fn serve(
    listener: TcpListener,
    app: Router,
    shutdown: Arc<Notify>,
    connections: Arc<ConnectionTasks>,
) {
    // The graceful signal every connection watches: once it flips, a connection
    // finishes the response it is writing and closes, which is what `axum::serve`'s
    // graceful shutdown did.
    let (graceful, _) = tokio::sync::watch::channel(false);
    loop {
        let accepted = tokio::select! {
            accepted = listener.accept() => accepted,
            () = shutdown.notified() => break,
        };
        let Ok((stream, _peer)) = accepted else {
            // The listener is gone; there is nothing left to accept.
            break;
        };
        spawn_connection(stream, app.clone(), &connections, graceful.subscribe());
    }
    // Nothing is accepted any more: tell the open connections to wind down.
    let _ = graceful.send(true);
}

/// Serves one accepted connection on a task recorded in `connections`.
fn spawn_connection(
    stream: TcpStream,
    app: Router,
    connections: &ConnectionTasks,
    mut graceful: tokio::sync::watch::Receiver<bool>,
) {
    let io = TokioIo::new(stream);
    let handle = tokio::spawn(async move {
        let service = service_fn(move |request: hyper::Request<Incoming>| {
            let mut app = app.clone();
            async move {
                match app.call(request.map(Body::new)).await {
                    Ok(response) => Ok::<_, std::convert::Infallible>(response),
                    Err(never) => match never {},
                }
            }
        });
        let builder = Builder::new(TokioExecutor::new());
        let mut connection = std::pin::pin!(builder.serve_connection_with_upgrades(io, service));
        loop {
            tokio::select! {
                // A client that disconnects mid-stream is the normal end of a
                // test, so the connection error is deliberately dropped.
                _ = connection.as_mut() => break,
                changed = graceful.changed() => {
                    if changed.is_err() {
                        // The server is gone; nothing will close this connection.
                        break;
                    }
                    connection.as_mut().graceful_shutdown();
                }
            }
        }
    });
    connections.track(handle);
}

/// Builder for [`MockLlmServer`].
///
/// Every method consumes and returns `Self`; [`MockLlmServerBuilder::start`] and
/// [`MockLlmServerBuilder::try_start`] launch the server.
#[derive(Debug, Default)]
pub struct MockLlmServerBuilder {
    /// Script, fallback, id prefix, default model, and chunk delay.
    config: ServerConfig,
}

impl MockLlmServerBuilder {
    /// Replaces the script with `items`, in order.
    #[must_use]
    pub fn script(mut self, items: impl IntoIterator<Item = ScriptedResponse>) -> Self {
        self.config.script = items.into_iter().collect();
        self
    }

    /// Appends one scripted answer.
    #[must_use]
    pub fn item(mut self, response: ScriptedResponse) -> Self {
        self.config.script.push(response);
        self
    }

    /// Sets the answer served when the queue is empty.
    ///
    /// Defaults to the Python parity text `"Default mock response"`.
    #[must_use]
    pub fn fallback(mut self, response: ScriptedResponse) -> Self {
        self.config.fallback = response;
        self
    }

    /// Sets the prefix of the deterministic ids (`chatcmpl-<prefix><seq>`).
    ///
    /// Defaults to `"mock-"`, which yields `chatcmpl-mock-0001` for the first
    /// popped answer.
    #[must_use]
    pub fn id_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.id_prefix = prefix.into();
        self
    }

    /// Sets the model reported when a scripted turn names no model.
    ///
    /// Defaults to `"mock-model"`.
    #[must_use]
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Delays every SSE frame by `delay` (default: no delay at all).
    #[must_use]
    pub fn chunk_delay(mut self, delay: Duration) -> Self {
        self.config.chunk_delay = delay;
        self
    }

    /// Starts the server.
    ///
    /// # Panics
    ///
    /// Panics when the loopback listener cannot be bound; use
    /// [`MockLlmServerBuilder::try_start`] to handle that failure instead.
    pub async fn start(self) -> MockLlmServer {
        match self.try_start().await {
            Ok(server) => server,
            Err(error) => panic!("the mock LLM server could not bind 127.0.0.1:0: {error}"),
        }
    }

    /// Starts the server, returning a bind failure instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] when the loopback listener cannot be bound or
    /// its address cannot be read.
    pub async fn try_start(self) -> Result<MockLlmServer> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .map_err(|error| {
                Error::transport(format!(
                    "binding the mock LLM server to 127.0.0.1:0 failed: {error}"
                ))
            })?;
        let addr = listener.local_addr().map_err(|error| {
            Error::transport(format!(
                "reading the mock LLM server address failed: {error}"
            ))
        })?;

        let state = Arc::new(ServerState::new(self.config));
        let app = routes::router(Arc::clone(&state));
        let shutdown_signal = Arc::new(Notify::new());
        let connections = Arc::new(ConnectionTasks::default());
        let signal = Arc::clone(&shutdown_signal);
        let tracked = Arc::clone(&connections);
        let task = tokio::spawn(async move {
            serve(listener, app, signal, tracked).await;
        });

        let base_url = format!("http://{addr}/v1");
        let endpoint = format!("{base_url}/chat/completions");
        Ok(MockLlmServer {
            state,
            addr,
            base_url,
            endpoint,
            shutdown_signal,
            task: Mutex::new(Some(task)),
            connections,
        })
    }
}
