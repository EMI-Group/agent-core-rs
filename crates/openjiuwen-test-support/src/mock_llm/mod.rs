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
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ModelClientConfig, ModelRequestConfig, OpenAiCompatibleClient};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

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

/// A running mock server plus the handle to its task.
///
/// Dropping the handle aborts the server task; call [`MockLlmServer::shutdown`]
/// for a graceful stop that drains in-flight responses first.
pub struct MockLlmServer {
    /// Shared script queue, recordings, and configuration.
    state: Arc<ServerState>,
    /// Loopback address the server is listening on.
    addr: SocketAddr,
    /// `http://127.0.0.1:<port>/v1`.
    base_url: String,
    /// `base_url` plus `/chat/completions`.
    endpoint: String,
    /// Wakes the graceful-shutdown future of `axum::serve`.
    shutdown_signal: Arc<Notify>,
    /// The server task, taken by [`MockLlmServer::shutdown`] and `Drop`.
    task: Mutex<Option<JoinHandle<()>>>,
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

    /// Waits indefinitely until at least `count` requests have been recorded.
    pub async fn wait_for_requests(&self, count: usize) {
        self.state.wait_for_requests(count).await;
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

    /// Stops the server gracefully, aborting the task if it does not wind down
    /// within [`SHUTDOWN_TIMEOUT`].
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
    }
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        self.shutdown_signal.notify_one();
        if let Some(handle) = state::lock(&self.task).take() {
            handle.abort();
        }
    }
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
        let signal = Arc::clone(&shutdown_signal);
        let task = tokio::spawn(async move {
            // A serve error means the listener went away; there is nothing left
            // to report to the test, so it is deliberately dropped.
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move { signal.notified().await })
                .await;
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
        })
    }
}
