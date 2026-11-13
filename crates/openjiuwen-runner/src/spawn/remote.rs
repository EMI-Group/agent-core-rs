//! The HTTP remote-agent client of the spawn seam.
//!
//! Ports the HTTP half of Python `openjiuwen/core/runner/drunner/remote_client/`
//! (`remote_client.py`'s `RemoteClient` ABC, `remote_client_config.py`'s
//! `ProtocolEnum`, and `remote_agent.py`'s start/stop/invoke/stream/cancel
//! lifecycle) onto `reqwest`, and exposes it through the crate's spawn seam so a
//! runner can address a remote agent exactly like a subprocess one.
//!
//! [`RemoteAgent`] reuses the subprocess NDJSON wire ([`SpawnMessage`]): the
//! stream endpoint answers one frame per line, a `stream_chunk` frame carries
//! one [`AgentChunk`], an `error` frame is the stream's only `Err`, and
//! `done`/`output` end it. Non-JSON line noise (keep-alives, blank lines) is
//! skipped at `trace`.
//!
//! [`RemoteProtocol::Mq`] and [`RemoteProtocol::A2a`] are typed as
//! [`Error::Unsupported`]: the Pulsar/ZeroMQ reply-topic transport and the A2A
//! remote transport are deferred, and neither degrades silently into HTTP.
//!
//! # Example
//!
//! ```no_run
//! use openjiuwen_common::error::Result;
//! use openjiuwen_runner::spawn::remote::{RemoteAgent, RemoteAgentConfig, RemoteClient};
//! use serde_json::json;
//!
//! # async fn example() -> Result<()> {
//! let config =
//!     RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080").with_session_id("session-1");
//! let agent = RemoteAgent::new(config);
//! agent.start().await?;
//! let output = agent.invoke(json!({"query": "hi"}), None).await?;
//! assert!(!output.output.is_empty());
//! agent.stop().await?;
//! # Ok(())
//! # }
//! ```

use crate::error;
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Default invoke path, relative to [`RemoteAgentConfig::base_url`].
pub const DEFAULT_INVOKE_PATH: &str = "agents/{agent_id}/invoke";
/// Default stream path, relative to [`RemoteAgentConfig::base_url`].
pub const DEFAULT_STREAM_PATH: &str = "agents/{agent_id}/stream";
/// Default cancel path, relative to [`RemoteAgentConfig::base_url`].
pub const DEFAULT_CANCEL_PATH: &str = "agents/{agent_id}/cancel";
/// Timeout applied when [`RemoteAgentConfig::timeout_secs`] is unset or unusable.
pub const DEFAULT_REMOTE_TIMEOUT_SECS: f64 = 60.0;

/// The placeholder a configured path may use for the agent id.
const AGENT_ID_PLACEHOLDER: &str = "{agent_id}";
/// Longest error-body snippet echoed back in an error message.
const BODY_SNIPPET_CHARS: usize = 256;

/// Transport a remote agent is reached over (Python `ProtocolEnum`).
///
/// Only [`RemoteProtocol::Http`] is implemented; the other variants exist so a
/// host's configuration round-trips and so the deferral is a typed error rather
/// than a guess.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProtocol {
    /// Plain HTTP(S) JSON + NDJSON stream against [`RemoteAgentConfig::base_url`].
    #[default]
    Http,
    /// Pulsar/ZeroMQ reply-topic transport (deferred).
    Mq,
    /// A2A remote transport (deferred; owned by `openjiuwen-extensions`).
    A2a,
}

impl RemoteProtocol {
    /// The `snake_case` wire name (`http`, `mq`, `a2a`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Mq => "mq",
            Self::A2a => "a2a",
        }
    }

    /// Builds the client for this transport.
    ///
    /// Shorthand for [`create_remote_client`].
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for the deferred [`Mq`](Self::Mq) and
    /// [`A2a`](Self::A2a) transports.
    pub fn client(self, config: RemoteAgentConfig) -> Result<Arc<dyn RemoteClient>> {
        create_remote_client(self, config)
    }
}

impl fmt::Display for RemoteProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Builds the remote client for `protocol`.
///
/// # Errors
///
/// [`Error::Unsupported`] for [`RemoteProtocol::Mq`] and [`RemoteProtocol::A2a`],
/// whose transports are deferred.
pub fn create_remote_client(
    protocol: RemoteProtocol,
    config: RemoteAgentConfig,
) -> Result<Arc<dyn RemoteClient>> {
    ensure_supported(protocol)?;
    Ok(Arc::new(RemoteAgent::with_protocol(config, protocol)?))
}

/// Rejects the deferred transports with a message naming them.
fn ensure_supported(protocol: RemoteProtocol) -> Result<()> {
    match protocol {
        RemoteProtocol::Http => Ok(()),
        RemoteProtocol::Mq => Err(error::unsupported(
            "remote client protocol 'mq'",
            "the Pulsar/ZeroMQ reply-topic transport is deferred; use RemoteProtocol::Http",
        )),
        RemoteProtocol::A2a => Err(error::unsupported(
            "remote client protocol 'a2a'",
            "the A2A remote transport is deferred (see openjiuwen-extensions' a2a feature); use RemoteProtocol::Http",
        )),
    }
}

/// Where and how to reach one remote agent.
///
/// The URL layout is `{base_url}/{path}` with `{agent_id}` substituted in the
/// path; extra slashes are collapsed and a missing slash between base and path
/// is tolerated, so `("http://host:8080/", "/agents/{agent_id}/invoke")` and
/// `("http://host:8080", "agents/{agent_id}/invoke")` build the same URL.
///
/// Everything is serde (`snake_case`, defaults, unknown fields tolerated) so a
/// host can persist its remote-agent table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RemoteAgentConfig {
    /// Id of the remote agent; substituted into the configured paths.
    pub agent_id: String,
    /// Origin the agent is reached at, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// Path of the invoke endpoint (default [`DEFAULT_INVOKE_PATH`]).
    pub invoke_path: String,
    /// Path of the stream endpoint (default [`DEFAULT_STREAM_PATH`]).
    pub stream_path: String,
    /// Path of the cancel endpoint, or `None` when the agent is not cancellable.
    pub cancel_path: Option<String>,
    /// Session the agent binds to; sent as `session_id` on every invocation.
    pub session_id: Option<String>,
    /// Default request deadline in seconds (default 60).
    pub timeout_secs: f64,
    /// Extra request headers, e.g. an authorization token.
    pub headers: BTreeMap<String, String>,
}

impl Default for RemoteAgentConfig {
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            base_url: String::new(),
            invoke_path: DEFAULT_INVOKE_PATH.to_owned(),
            stream_path: DEFAULT_STREAM_PATH.to_owned(),
            cancel_path: Some(DEFAULT_CANCEL_PATH.to_owned()),
            session_id: None,
            timeout_secs: DEFAULT_REMOTE_TIMEOUT_SECS,
            headers: BTreeMap::new(),
        }
    }
}

impl RemoteAgentConfig {
    /// A config for `agent_id` at `base_url`, with the default paths and timeout.
    #[must_use]
    pub fn new(agent_id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            base_url: base_url.into(),
            ..Self::default()
        }
    }

    /// Replaces the invoke and stream paths.
    #[must_use]
    pub fn with_paths(
        mut self,
        invoke_path: impl Into<String>,
        stream_path: impl Into<String>,
    ) -> Self {
        self.invoke_path = invoke_path.into();
        self.stream_path = stream_path.into();
        self
    }

    /// Replaces the cancel path; `None` marks the agent as not cancellable.
    #[must_use]
    pub fn with_cancel_path(mut self, cancel_path: Option<String>) -> Self {
        self.cancel_path = cancel_path;
        self
    }

    /// Sets the session the agent binds to.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the default request deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Adds one request header.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// The configured deadline as a [`Duration`], falling back to
    /// [`DEFAULT_REMOTE_TIMEOUT_SECS`] when the value is zero, negative,
    /// non-finite, or too large to represent.
    #[must_use]
    pub fn effective_timeout(&self) -> Duration {
        if self.timeout_secs.is_finite() && self.timeout_secs > 0.0 {
            Duration::try_from_secs_f64(self.timeout_secs)
                .unwrap_or_else(|_| Duration::from_secs_f64(DEFAULT_REMOTE_TIMEOUT_SECS))
        } else {
            Duration::from_secs_f64(DEFAULT_REMOTE_TIMEOUT_SECS)
        }
    }

    /// The absolute invoke URL.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when [`Self::base_url`] is empty or when
    /// [`Self::invoke_path`] needs an [`Self::agent_id`] that is empty.
    pub fn invoke_url(&self) -> Result<String> {
        build_url(&self.base_url, &self.invoke_path, &self.agent_id)
    }

    /// The absolute stream URL.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when [`Self::base_url`] is empty or when
    /// [`Self::stream_path`] needs an [`Self::agent_id`] that is empty.
    pub fn stream_url(&self) -> Result<String> {
        build_url(&self.base_url, &self.stream_path, &self.agent_id)
    }

    /// The absolute cancel URL, or `None` when no cancel path is configured.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when [`Self::base_url`] is empty or when the
    /// configured path needs an [`Self::agent_id`] that is empty.
    pub fn cancel_url(&self) -> Result<Option<String>> {
        match self.cancel_path.as_deref() {
            Some(path) => build_url(&self.base_url, path, &self.agent_id).map(Some),
            None => Ok(None),
        }
    }
}

/// Joins `base_url` and `path`, substituting `agent_id`.
fn build_url(base_url: &str, path: &str, agent_id: &str) -> Result<String> {
    require_base_url(base_url)?;
    let base = base_url.trim().trim_end_matches('/');
    let mut path = path.trim().to_owned();
    if path.contains(AGENT_ID_PLACEHOLDER) {
        let agent_id = agent_id.trim();
        if agent_id.is_empty() {
            return Err(Error::validation(format!(
                "remote agent agent_id cannot be empty: path '{path}' requires one"
            )));
        }
        path = path.replace(AGENT_ID_PLACEHOLDER, agent_id);
    }
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        return Ok(base.to_owned());
    }
    Ok(collapse_slashes(&format!("{base}/{path}")))
}

/// Rejects an empty base URL.
fn require_base_url(base_url: &str) -> Result<()> {
    if base_url.trim().is_empty() {
        return Err(Error::validation(
            "remote agent base_url cannot be empty: set RemoteAgentConfig::base_url",
        ));
    }
    Ok(())
}

/// Collapses runs of `/` outside the scheme separator.
fn collapse_slashes(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => format!("{scheme}://{}", collapse_slash_runs(rest)),
        None => collapse_slash_runs(url),
    }
}

/// Collapses every run of `/` in `text` to one.
fn collapse_slash_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_slash = false;
    for character in text.chars() {
        if character == '/' {
            if !previous_slash {
                out.push(character);
            }
            previous_slash = true;
        } else {
            out.push(character);
            previous_slash = false;
        }
    }
    out
}

/// The remote-client seam (Python `RemoteClient` ABC).
///
/// A client is started explicitly or on first use; [`Self::stream`] returns the
/// stream immediately and performs the HTTP work inside it, so a transport
/// failure is a stream item rather than a call-site error.
#[async_trait]
pub trait RemoteClient: Send + Sync {
    /// Opens the client's transport; idempotent.
    ///
    /// # Errors
    ///
    /// A client that cannot be constructed (an invalid configured header, a TLS
    /// backend failure) fails here.
    async fn start(&self) -> Result<()>;
    /// Closes the client's transport; idempotent.
    ///
    /// # Errors
    ///
    /// A backend that cannot release its resources cleanly fails here; the HTTP
    /// client never does.
    async fn stop(&self) -> Result<()>;
    /// Returns `true` between [`Self::start`] and [`Self::stop`].
    fn is_started(&self) -> bool;
    /// Returns `true` outside the started window (Python parity).
    fn is_stopped(&self) -> bool {
        !self.is_started()
    }
    /// Invokes the agent once and returns its terminal output.
    ///
    /// `timeout` overrides the client's configured deadline for this request.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an unusable URL or a body that is not an output,
    /// [`Error::NotFound`] when the agent is unknown, [`Error::Timeout`] when the
    /// deadline elapses, and [`Error::Transport`] for every other failure.
    async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentOutput>;
    /// Streams one invocation as [`AgentChunk`] frames.
    fn stream(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>>;
    /// Asks the agent to cancel `task_id`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the client has no cancel endpoint,
    /// [`Error::Validation`] for an unusable URL, [`Error::NotFound`] when the
    /// task is unknown, [`Error::Timeout`] when the deadline elapses, and
    /// [`Error::Transport`] for every other failure.
    async fn cancel_task(&self, task_id: &str) -> Result<()>;
}

/// The mutable part of a [`RemoteAgent`], shared across clones.
#[derive(Debug)]
struct RemoteState {
    /// Where and how to reach the agent.
    config: RemoteAgentConfig,
    /// Transport the agent is reached over (always HTTP today).
    protocol: RemoteProtocol,
    /// The lazily built client; `None` means stopped.
    client: Mutex<Option<reqwest::Client>>,
}

/// One remote agent reached over HTTP, usable as a [`SpawnedAgent`].
///
/// Python `remote_agent.py`: the same start/stop/invoke/stream/cancel lifecycle,
/// with the NDJSON stream wire of the subprocess backend. Cloning is cheap (the
/// state is shared); the stream entry points clone the handle so the request
/// lives inside the returned stream.
#[derive(Debug, Clone)]
pub struct RemoteAgent {
    /// Config, protocol, and the lazily built HTTP client.
    state: Arc<RemoteState>,
}

impl RemoteAgent {
    /// A remote agent reached over [`RemoteProtocol::Http`].
    #[must_use]
    pub fn new(config: RemoteAgentConfig) -> Self {
        Self::with_state(config, RemoteProtocol::Http)
    }

    /// A remote agent reached over `protocol`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for the deferred [`RemoteProtocol::Mq`] and
    /// [`RemoteProtocol::A2a`] transports.
    pub fn with_protocol(config: RemoteAgentConfig, protocol: RemoteProtocol) -> Result<Self> {
        ensure_supported(protocol)?;
        Ok(Self::with_state(config, protocol))
    }

    /// The shared state constructor both public builders funnel through.
    fn with_state(config: RemoteAgentConfig, protocol: RemoteProtocol) -> Self {
        Self {
            state: Arc::new(RemoteState {
                config,
                protocol,
                client: Mutex::new(None),
            }),
        }
    }

    /// The agent's configuration.
    #[must_use]
    pub fn config(&self) -> &RemoteAgentConfig {
        &self.state.config
    }

    /// The configured agent id.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        &self.state.config.agent_id
    }

    /// The transport the agent is reached over.
    #[must_use]
    pub fn protocol(&self) -> RemoteProtocol {
        self.state.protocol
    }

    /// Builds the client when it is not started; synchronous and idempotent.
    fn ensure_started(&self) -> Result<()> {
        let started = self.state.client.lock().is_some();
        if started {
            return Ok(());
        }
        let client = self.build_client()?;
        *self.state.client.lock() = Some(client);
        Ok(())
    }

    /// Clones the started client out of the lock, auto-starting when needed.
    ///
    /// The guard is released before the caller awaits, so no lock is ever held
    /// across a request.
    fn http_client(&self) -> Result<reqwest::Client> {
        self.ensure_started()?;
        self.state
            .client
            .lock()
            .clone()
            .ok_or_else(|| error::not_started("remote agent"))
    }

    /// Builds the `reqwest` client from the configured timeout and headers.
    fn build_client(&self) -> Result<reqwest::Client> {
        let config = self.config();
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                Error::config(format!(
                    "invalid remote agent header name '{name}': {error}"
                ))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|error| {
                Error::config(format!(
                    "invalid remote agent header value for '{name}': {error}"
                ))
            })?;
            headers.insert(header_name, header_value);
        }
        reqwest::Client::builder()
            .timeout(config.effective_timeout())
            .default_headers(headers)
            .build()
            .map_err(|error| {
                error::transport(format!(
                    "cannot build the remote agent HTTP client: {error}"
                ))
            })
    }

    /// The JSON body of an invocation: `session_id` always, `inputs` verbatim.
    fn request_body(&self, inputs: &Value) -> Value {
        serde_json::json!({
            "session_id": self.config().session_id,
            "inputs": inputs,
        })
    }

    /// Maps a `reqwest` failure onto the crate's error vocabulary.
    fn transport_error(
        &self,
        label: &str,
        url: &str,
        timeout: Duration,
        error: &reqwest::Error,
    ) -> Error {
        if error.is_timeout() {
            return error::run_timeout(label, timeout);
        }
        error::transport(format!(
            "{label} ('{}') to {url} failed: {error}",
            self.agent_id()
        ))
    }
}

#[async_trait]
impl RemoteClient for RemoteAgent {
    #[allow(clippy::unused_async_trait_impl)]
    async fn start(&self) -> Result<()> {
        self.ensure_started()
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn stop(&self) -> Result<()> {
        self.state.client.lock().take();
        Ok(())
    }

    fn is_started(&self) -> bool {
        self.state.client.lock().is_some()
    }

    async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentOutput> {
        let url = self.config().invoke_url()?;
        let client = self.http_client()?;
        let deadline = timeout.unwrap_or_else(|| self.config().effective_timeout());
        let payload = self.request_body(&inputs);
        tracing::debug!(
            agent_id = %self.agent_id(),
            url = %url,
            "invoking a remote agent over HTTP"
        );
        let response = client
            .post(&url)
            .timeout(deadline)
            .json(&payload)
            .send()
            .await
            .map_err(|error| self.transport_error("remote agent invoke", &url, deadline, &error))?;
        let status = response.status();
        let body = response.text().await.map_err(|error| {
            error::transport(format!(
                "remote agent '{}' invoke body from {url} cannot be read: {error}",
                self.agent_id()
            ))
        })?;
        if !status.is_success() {
            return Err(map_status(self.agent_id(), status.as_u16(), &url, &body));
        }
        let value: Value = serde_json::from_str(&body).map_err(|error| {
            Error::validation(format!(
                "remote agent '{}' returned a non-JSON body from {url}: {error}: {}",
                self.agent_id(),
                body_snippet(&body)
            ))
        })?;
        AgentOutput::from_value(value).map_err(|error| {
            Error::validation(format!(
                "remote agent '{}' returned an invalid output object from {url}: {error}",
                self.agent_id()
            ))
        })
    }

    fn stream(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let agent = self.clone();
        Box::pin(async_stream::stream! {
            let url = match agent.config().stream_url() {
                Ok(url) => url,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let client = match agent.http_client() {
                Ok(client) => client,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            let deadline = timeout.unwrap_or_else(|| agent.config().effective_timeout());
            let payload = agent.request_body(&inputs);
            tracing::debug!(
                agent_id = %agent.agent_id(),
                url = %url,
                "streaming a remote agent over HTTP"
            );
            let response = match client
                .post(&url)
                .timeout(deadline)
                .json(&payload)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    yield Err(agent.transport_error("remote agent stream", &url, deadline, &error));
                    return;
                }
            };
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let body = response.text().await.unwrap_or_default();
                yield Err(map_status(agent.agent_id(), status, &url, &body));
                return;
            }
            let mut body = Box::pin(response.bytes_stream());
            let mut buffer: Vec<u8> = Vec::new();
            while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Err(error::transport(format!(
                            "remote agent '{}' stream from {url} failed: {error}",
                            agent.agent_id()
                        )));
                        return;
                    }
                };
                buffer.extend_from_slice(&chunk);
                while let Some(line) = take_line(&mut buffer) {
                    match parse_stream_line(&line) {
                        StreamLine::Skip => {}
                        StreamLine::Chunk(chunk) => yield Ok(chunk),
                        StreamLine::Error(error) => {
                            yield Err(error);
                            return;
                        }
                        StreamLine::End => return,
                    }
                }
            }
            if !buffer.is_empty() {
                let line = String::from_utf8_lossy(&buffer).into_owned();
                match parse_stream_line(&line) {
                    StreamLine::Skip => {}
                    StreamLine::Chunk(chunk) => yield Ok(chunk),
                    StreamLine::Error(error) => yield Err(error),
                    StreamLine::End => {}
                }
            }
        })
    }

    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        let Some(url) = self.config().cancel_url()? else {
            return Err(error::unsupported(
                "remote agent cancel_task",
                "no cancel_path is configured for this RemoteAgent",
            ));
        };
        let client = self.http_client()?;
        let deadline = self.config().effective_timeout();
        let response = client
            .post(&url)
            .timeout(deadline)
            .json(&serde_json::json!({"task_id": task_id}))
            .send()
            .await
            .map_err(|error| {
                self.transport_error("remote agent cancel_task", &url, deadline, &error)
            })?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let body = response.text().await.unwrap_or_default();
        Err(map_status(self.agent_id(), status.as_u16(), &url, &body))
    }
}

#[async_trait]
impl SpawnedAgent for RemoteAgent {
    fn id(&self) -> &str {
        self.agent_id()
    }

    /// A remote agent has no OS process: liveness is the client's state.
    fn is_alive(&self) -> bool {
        RemoteClient::is_started(self)
    }

    /// Always `None`: the agent runs behind an HTTP endpoint, not as a child.
    fn pid(&self) -> Option<u32> {
        None
    }

    fn backend(&self) -> SpawnBackend {
        SpawnBackend::RemoteHttp
    }

    /// The client's started state: there is no probe endpoint to ask.
    fn is_healthy(&self) -> bool {
        RemoteClient::is_started(self)
    }

    /// The HTTP wire contract defines only invoke/stream/cancel, so there is no
    /// health endpoint to probe: the answer is the client's own started state
    /// (`false` until the first [`RemoteClient::invoke`]/[`RemoteClient::stream`]
    /// auto-starts the client, or an explicit [`RemoteClient::start`]).
    #[allow(clippy::unused_async_trait_impl)]
    async fn health(&self) -> Result<bool> {
        Ok(RemoteClient::is_started(self))
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        RemoteClient::invoke(self, inputs, None).await
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        RemoteClient::stream(self, inputs, None)
    }

    /// Releases the client and reports a graceful exit; there is nothing to kill.
    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        RemoteClient::stop(self).await?;
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        RemoteClient::stop(self).await
    }

    /// Forwards to [`RemoteClient::cancel_task`]: the cancel endpoint is the one
    /// place a remote backend owns a real task handle.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] when no `cancel_path` is
    /// configured, [`openjiuwen_common::error::Error::NotFound`] when the task is
    /// unknown, [`openjiuwen_common::error::Error::Timeout`] when the deadline
    /// elapses, and [`openjiuwen_common::error::Error::Transport`] for every other
    /// failure.
    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        RemoteClient::cancel_task(self, task_id).await
    }
}

/// An [`AgentSpawner`] that addresses remote agents over HTTP.
///
/// The spawned handle is a [`RemoteAgent`]; the agent id comes from the spawn
/// spec, so a runner registered with this spawner can route a class-agent spawn
/// to a remote endpoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RemoteSpawner {
    /// Origin every spawned agent is reached at.
    base_url: String,
    /// Default request deadline, when the caller set one.
    timeout: Option<Duration>,
    /// Headers applied to every spawned agent's client.
    headers: BTreeMap<String, String>,
}

impl RemoteSpawner {
    /// A spawner for agents served under `base_url`.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Self::default()
        }
    }

    /// Sets the default request deadline of every spawned agent.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Adds one header to every spawned agent's client.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// The configured origin.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Builds the [`RemoteAgentConfig`] this spawner hands to `agent_id`.
    #[must_use]
    pub fn config_for(&self, agent_id: &str) -> RemoteAgentConfig {
        let mut config = RemoteAgentConfig::new(agent_id, self.base_url.clone());
        config.headers.clone_from(&self.headers);
        if let Some(timeout) = self.timeout {
            config = config.with_timeout(timeout);
        }
        config
    }
}

#[async_trait]
impl AgentSpawner for RemoteSpawner {
    fn backend(&self) -> SpawnBackend {
        SpawnBackend::RemoteHttp
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let agent_id = remote_agent_id(&spec)?;
        let mut config = self.config_for(&agent_id);
        config.session_id.clone_from(&spec.session_id);
        // Fail at spawn time rather than at the first invoke.
        require_base_url(&config.base_url)?;
        Ok(Arc::new(RemoteAgent::new(config)))
    }
}

/// The remote agent id a spawn spec addresses: `agent_class`, then
/// `agent_module`, then a non-empty `init_kwargs["agent_id"]`.
fn remote_agent_id(spec: &SpawnSpec) -> Result<String> {
    let candidates = [
        spec.agent_class.as_deref(),
        spec.agent_module.as_deref(),
        spec.init_kwargs.get("agent_id").and_then(Value::as_str),
    ];
    for candidate in candidates.into_iter().flatten() {
        let trimmed = candidate.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_owned());
        }
    }
    Err(Error::validation(
        "spawn spec has no agent_class/agent_module/agent_id to address a remote agent",
    ))
}

/// Maps a non-2xx status onto an error; `404` means the agent is unknown.
fn map_status(agent_id: &str, status: u16, url: &str, body: &str) -> Error {
    let snippet = body_snippet(body);
    if status == 404 {
        return Error::NotFound(with_snippet(
            format!("remote agent '{agent_id}' is not registered at {url}"),
            &snippet,
        ));
    }
    error::transport(with_snippet(
        format!("remote agent '{agent_id}' returned status {status} from {url}"),
        &snippet,
    ))
}

/// Appends `: snippet` when there is a snippet to show.
fn with_snippet(message: String, snippet: &str) -> String {
    if snippet.is_empty() {
        message
    } else {
        format!("{message}: {snippet}")
    }
}

/// A whitespace-collapsed, at-most-[`BODY_SNIPPET_CHARS`]-char body snippet.
fn body_snippet(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= BODY_SNIPPET_CHARS {
        return collapsed;
    }
    let mut snippet: String = collapsed.chars().take(BODY_SNIPPET_CHARS - 1).collect();
    snippet.push('…');
    snippet
}

/// What one line of a remote stream means.
#[derive(Debug)]
enum StreamLine {
    /// One chunk item.
    Chunk(AgentChunk),
    /// One `Err` item; the stream ends afterwards.
    Error(Error),
    /// The stream is complete.
    End,
    /// Noise or a frame this direction should not send: ignore it.
    Skip,
}

/// Parses one NDJSON line of a remote stream.
///
/// Non-JSON noise is skipped at `trace`; valid JSON that is not a spawn frame is
/// a [`Error::Validation`] that ends the stream.
fn parse_stream_line(line: &str) -> StreamLine {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return StreamLine::Skip;
    }
    if serde_json::from_str::<Value>(trimmed).is_err() {
        tracing::trace!(
            target: "openjiuwen_runner::spawn",
            line = %body_snippet(trimmed),
            "skipping a non-JSON line of a remote agent stream"
        );
        return StreamLine::Skip;
    }
    let frame = match SpawnMessage::from_line(trimmed) {
        Ok(frame) => frame,
        Err(error) => return StreamLine::Error(error),
    };
    match frame.kind {
        SpawnMessageKind::StreamChunk => match AgentChunk::from_value(frame.payload) {
            Ok(chunk) => StreamLine::Chunk(chunk),
            Err(error) => StreamLine::Error(Error::validation(format!(
                "remote agent stream_chunk is not a valid AgentChunk frame: {error}"
            ))),
        },
        SpawnMessageKind::Done | SpawnMessageKind::Output => StreamLine::End,
        SpawnMessageKind::Error => StreamLine::Error(error::transport(format!(
            "remote agent stream reported an error: {}",
            frame_error_message(&frame.payload)
        ))),
        other => {
            tracing::trace!(
                target: "openjiuwen_runner::spawn",
                kind = ?other,
                "ignoring a remote agent stream frame"
            );
            StreamLine::Skip
        }
    }
}

/// The human-readable message of an `error` frame's payload.
fn frame_error_message(payload: &Value) -> String {
    match payload {
        Value::String(text) => text.clone(),
        Value::Object(map) => match map.get("message") {
            Some(Value::String(text)) => text.clone(),
            Some(other) => other.to_string(),
            None => payload.to_string(),
        },
        Value::Null => "no error detail was provided".to_owned(),
        other => other.to_string(),
    }
}

/// Removes and returns the next complete `\n`-terminated line of `buffer`.
fn take_line(buffer: &mut Vec<u8>) -> Option<String> {
    let position = buffer.iter().position(|byte| *byte == b'\n')?;
    let line: Vec<u8> = buffer.drain(..=position).collect();
    Some(String::from_utf8_lossy(&line).into_owned())
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
