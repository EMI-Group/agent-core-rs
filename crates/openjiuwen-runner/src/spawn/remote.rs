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
use crate::spawn::{AgentSpawner, SpawnSpec, SpawnedAgent};
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Serializes `frame` as one NDJSON line (without the trailing newline).
    fn frame_line(frame: &SpawnMessage) -> String {
        frame.to_line().unwrap().trim_end().to_owned()
    }

    /// Serializes `chunk` as a `stream_chunk` line.
    fn chunk_line(index: i64, text: &str) -> String {
        frame_line(&SpawnMessage::new(
            SpawnMessageKind::StreamChunk,
            serde_json::to_value(AgentChunk::llm_output(index, text)).unwrap(),
        ))
    }

    fn server_agent(server: &MockServer, agent_id: &str) -> RemoteAgent {
        RemoteAgent::new(RemoteAgentConfig::new(agent_id, server.uri()))
    }

    #[test]
    fn config_defaults_match_the_python_layout() {
        let config = RemoteAgentConfig::default();
        assert_eq!(config.agent_id, "");
        assert_eq!(config.base_url, "");
        assert_eq!(config.invoke_path, "agents/{agent_id}/invoke");
        assert_eq!(config.stream_path, "agents/{agent_id}/stream");
        assert_eq!(
            config.cancel_path.as_deref(),
            Some("agents/{agent_id}/cancel")
        );
        assert_eq!(config.session_id, None);
        assert!((config.timeout_secs - 60.0).abs() < f64::EPSILON);
        assert!(config.headers.is_empty());
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080")
            .with_paths("a/{agent_id}/run", "a/{agent_id}/events")
            .with_cancel_path(Some("a/{agent_id}/stop".to_owned()))
            .with_session_id("s-1")
            .with_timeout(Duration::from_secs(5))
            .with_header("Authorization", "Bearer token");
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["agent_id"], json!("assistant"));
        assert_eq!(value["invoke_path"], json!("a/{agent_id}/run"));
        assert_eq!(value["cancel_path"], json!("a/{agent_id}/stop"));
        assert_eq!(value["timeout_secs"], json!(5.0));
        let restored: RemoteAgentConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn config_tolerates_unknown_fields_and_partial_json() {
        let config: RemoteAgentConfig = serde_json::from_value(json!({
            "agent_id": "assistant",
            "base_url": "http://127.0.0.1:8080",
            "retry_policy": {"max": 3},
        }))
        .unwrap();
        assert_eq!(config.agent_id, "assistant");
        assert_eq!(config.invoke_path, DEFAULT_INVOKE_PATH);
        assert_eq!(config.stream_path, DEFAULT_STREAM_PATH);
        assert!((config.timeout_secs - 60.0).abs() < f64::EPSILON);
        assert!(config.headers.is_empty());
    }

    #[test]
    fn cancel_path_none_round_trips() {
        let config = RemoteAgentConfig::new("a", "http://host").with_cancel_path(None);
        let restored: RemoteAgentConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored.cancel_path, None);
        assert_eq!(restored.cancel_url().unwrap(), None);
    }

    #[test]
    fn effective_timeout_sanitizes_unusable_values() {
        let mut config = RemoteAgentConfig::default();
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
        config.timeout_secs = 0.0;
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
        config.timeout_secs = -1.0;
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
        config.timeout_secs = f64::NAN;
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
        config.timeout_secs = f64::INFINITY;
        assert_eq!(config.effective_timeout(), Duration::from_secs(60));
        config.timeout_secs = 1.5;
        assert_eq!(config.effective_timeout(), Duration::from_millis(1_500));
    }

    #[test]
    fn protocol_wire_names_and_default() {
        assert_eq!(RemoteProtocol::default(), RemoteProtocol::Http);
        for (protocol, name) in [
            (RemoteProtocol::Http, "http"),
            (RemoteProtocol::Mq, "mq"),
            (RemoteProtocol::A2a, "a2a"),
        ] {
            assert_eq!(serde_json::to_value(protocol).unwrap(), json!(name));
            let parsed: RemoteProtocol = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(parsed, protocol);
            assert_eq!(protocol.as_str(), name);
            assert_eq!(protocol.to_string(), name);
        }
    }

    #[test]
    fn urls_follow_the_documented_layout() {
        let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080");
        assert_eq!(
            config.invoke_url().unwrap(),
            "http://127.0.0.1:8080/agents/assistant/invoke"
        );
        assert_eq!(
            config.stream_url().unwrap(),
            "http://127.0.0.1:8080/agents/assistant/stream"
        );
        assert_eq!(
            config.cancel_url().unwrap().unwrap(),
            "http://127.0.0.1:8080/agents/assistant/cancel"
        );
    }

    #[test]
    fn urls_tolerate_extra_and_missing_slashes() {
        let joined = |base: &str, path: &str| {
            RemoteAgentConfig::new("a", base)
                .with_paths(path, path)
                .invoke_url()
                .unwrap()
        };
        assert_eq!(
            joined("http://host:8080/", "/agents/{agent_id}/invoke"),
            "http://host:8080/agents/a/invoke"
        );
        assert_eq!(
            joined("http://host:8080//", "//agents//{agent_id}//invoke"),
            "http://host:8080/agents/a/invoke"
        );
        assert_eq!(
            joined("https://host/api/v1", "agents/{agent_id}/invoke"),
            "https://host/api/v1/agents/a/invoke"
        );
    }

    #[test]
    fn empty_base_url_is_a_validation_error() {
        for config in [
            RemoteAgentConfig::new("a", ""),
            RemoteAgentConfig::new("a", "   "),
        ] {
            for error in [
                config.invoke_url().unwrap_err(),
                config.stream_url().unwrap_err(),
                config.cancel_url().unwrap_err(),
            ] {
                assert_eq!(error.kind(), ErrorKind::Validation);
                assert!(
                    error.to_string().contains("base_url"),
                    "unexpected message: {error}"
                );
            }
        }
    }

    #[test]
    fn empty_agent_id_needing_a_placeholder_is_a_validation_error() {
        let config = RemoteAgentConfig::new("", "http://host:8080");
        let error = config.invoke_url().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("agent_id cannot be empty"),
            "unexpected message: {error}"
        );
        assert!(config.stream_url().is_err());
        assert!(config.cancel_url().is_err());
    }

    #[test]
    fn paths_without_a_placeholder_need_no_agent_id() {
        let config = RemoteAgentConfig::new("", "http://host:8080")
            .with_paths("/healthz", "/events")
            .with_cancel_path(Some("/stop".to_owned()));
        assert_eq!(config.invoke_url().unwrap(), "http://host:8080/healthz");
        assert_eq!(config.stream_url().unwrap(), "http://host:8080/events");
        assert_eq!(
            config.cancel_url().unwrap().as_deref(),
            Some("http://host:8080/stop")
        );
    }

    #[test]
    fn unsupported_protocols_name_the_deferred_transport() {
        let error = ensure_supported(RemoteProtocol::Mq).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("remote client protocol 'mq'"), "{message}");
        assert!(message.contains("Pulsar/ZeroMQ"), "{message}");
        assert!(message.contains("RemoteProtocol::Http"), "{message}");

        let error = ensure_supported(RemoteProtocol::A2a).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(
            message.contains("remote client protocol 'a2a'"),
            "{message}"
        );
        assert!(message.contains("A2A"), "{message}");
        assert!(message.contains("openjiuwen-extensions"), "{message}");

        assert!(ensure_supported(RemoteProtocol::Http).is_ok());
    }

    #[test]
    fn factory_builds_http_and_rejects_deferred_transports() {
        let config = RemoteAgentConfig::new("assistant", "http://127.0.0.1:8080");
        for protocol in [RemoteProtocol::Mq, RemoteProtocol::A2a] {
            let error = create_remote_client(protocol, config.clone())
                .err()
                .expect("a deferred transport cannot build a client");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            let error = protocol
                .client(config.clone())
                .err()
                .expect("a deferred transport cannot build a client");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            let error = RemoteAgent::with_protocol(config.clone(), protocol).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
        }
        let client = create_remote_client(RemoteProtocol::Http, config.clone()).unwrap();
        assert!(!client.is_started());
        let agent = RemoteAgent::with_protocol(config, RemoteProtocol::Http).unwrap();
        assert_eq!(agent.protocol(), RemoteProtocol::Http);
    }

    #[test]
    fn status_mapping_splits_not_found_from_transport_failures() {
        let not_found = map_status("a", 404, "http://host/agents/a/invoke", "");
        assert_eq!(not_found.kind(), ErrorKind::NotFound);
        assert!(
            not_found
                .to_string()
                .contains("http://host/agents/a/invoke"),
            "unexpected message: {not_found}"
        );
        let not_found = map_status("a", 404, "http://host/x", "{\"message\":\"no such agent\"}");
        assert!(not_found.to_string().contains("no such agent"));

        for status in [400, 401, 500, 503] {
            let error = map_status("a", status, "http://host/x", "boom");
            assert_eq!(error.kind(), ErrorKind::Transport, "for status {status}");
            let message = error.to_string();
            assert!(message.contains(&status.to_string()), "{message}");
            assert!(message.contains("http://host/x"), "{message}");
            assert!(message.contains("boom"), "{message}");
        }
    }

    #[test]
    fn body_snippets_collapse_whitespace_and_truncate() {
        assert_eq!(body_snippet("  a\n\tb   c  "), "a b c");
        assert_eq!(body_snippet(""), "");
        let long = "é".repeat(BODY_SNIPPET_CHARS * 2);
        let snippet = body_snippet(&long);
        assert_eq!(snippet.chars().count(), BODY_SNIPPET_CHARS);
        assert!(snippet.ends_with('…'));
        let exact = "x".repeat(BODY_SNIPPET_CHARS);
        assert_eq!(body_snippet(&exact), exact);
    }

    #[test]
    fn stream_lines_map_to_items_ends_and_errors() {
        match parse_stream_line(&chunk_line(0, "he")) {
            StreamLine::Chunk(chunk) => {
                assert_eq!(chunk.index, 0);
                assert_eq!(chunk.kind, openjiuwen_agent::output::LLM_OUTPUT);
            }
            other => panic!("expected a chunk, got {other:?}"),
        }
        assert!(matches!(
            parse_stream_line(&frame_line(&SpawnMessage::new(
                SpawnMessageKind::Done,
                Value::Null
            ))),
            StreamLine::End
        ));
        assert!(matches!(
            parse_stream_line(&frame_line(&SpawnMessage::new(
                SpawnMessageKind::Output,
                json!({"output": "done"})
            ))),
            StreamLine::End
        ));
        assert!(matches!(parse_stream_line(""), StreamLine::Skip));
        assert!(matches!(parse_stream_line("   \r\n"), StreamLine::Skip));
        assert!(matches!(
            parse_stream_line("keep-alive: ping"),
            StreamLine::Skip
        ));
        assert!(matches!(
            parse_stream_line(&frame_line(&SpawnMessage::new(
                SpawnMessageKind::HealthCheck,
                Value::Null
            ))),
            StreamLine::Skip
        ));
    }

    #[test]
    fn error_frames_carry_the_remote_message() {
        let message = |payload: Value| match parse_stream_line(&frame_line(&SpawnMessage::new(
            SpawnMessageKind::Error,
            payload,
        ))) {
            StreamLine::Error(error) => error.to_string(),
            other => panic!("expected an error, got {other:?}"),
        };
        assert!(message(json!({"message": "boom"})).contains("boom"));
        assert!(message(json!("plain boom")).contains("plain boom"));
        assert!(message(json!({"code": 7})).contains('7'));
        assert!(message(Value::Null).contains("no error detail"));
    }

    #[test]
    fn valid_json_that_is_not_a_frame_is_a_validation_error() {
        for line in ["42", "{\"not\":\"a frame\"}", "{\"kind\":\"teleport\"}"] {
            match parse_stream_line(line) {
                StreamLine::Error(error) => assert_eq!(error.kind(), ErrorKind::Validation),
                other => panic!("expected a validation error for {line}, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_stream_chunk_without_an_agent_chunk_payload_is_a_validation_error() {
        let line = frame_line(&SpawnMessage::new(
            SpawnMessageKind::StreamChunk,
            json!({"nope": true}),
        ));
        match parse_stream_line(&line) {
            // Any JSON object deserializes into the defaulted `AgentChunk`; a
            // non-object payload is the case that must be rejected.
            StreamLine::Chunk(chunk) => assert_eq!(chunk.index, 0),
            other => panic!("unexpected outcome: {other:?}"),
        }
        let line = frame_line(&SpawnMessage::new(SpawnMessageKind::StreamChunk, json!(5)));
        match parse_stream_line(&line) {
            StreamLine::Error(error) => assert_eq!(error.kind(), ErrorKind::Validation),
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[test]
    fn take_line_splits_on_newlines_and_keeps_the_partial_tail() {
        let mut buffer = b"one\ntwo\r\npartial".to_vec();
        assert_eq!(take_line(&mut buffer).as_deref(), Some("one\n"));
        assert_eq!(take_line(&mut buffer).as_deref(), Some("two\r\n"));
        assert_eq!(take_line(&mut buffer), None);
        assert_eq!(buffer, b"partial");
        buffer.push(b'\n');
        assert_eq!(take_line(&mut buffer).as_deref(), Some("partial\n"));
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn remote_ids_come_from_class_module_or_init_kwargs() {
        let spec = SpawnSpec::class_agent("pkg.mod", "HelperAgent");
        assert_eq!(remote_agent_id(&spec).unwrap(), "HelperAgent");

        let spec = SpawnSpec {
            agent_class: Some("  ".to_owned()),
            agent_module: Some("pkg.mod".to_owned()),
            ..SpawnSpec::default()
        };
        assert_eq!(remote_agent_id(&spec).unwrap(), "pkg.mod");

        let spec = SpawnSpec {
            init_kwargs: json!({"agent_id": "remote-1"}),
            ..SpawnSpec::default()
        };
        assert_eq!(remote_agent_id(&spec).unwrap(), "remote-1");

        let spec = SpawnSpec {
            init_kwargs: json!({"agent_id": "  "}),
            ..SpawnSpec::default()
        };
        let error = remote_agent_id(&spec).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("no agent_class/agent_module/agent_id"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn spawner_config_for_applies_defaults_headers_and_timeout() {
        let spawner = RemoteSpawner::new("http://127.0.0.1:8080")
            .with_timeout(Duration::from_secs(5))
            .with_header("Authorization", "Bearer token")
            .with_header("X-Team", "red");
        assert_eq!(spawner.base_url(), "http://127.0.0.1:8080");
        let config = spawner.config_for("assistant");
        assert_eq!(config.agent_id, "assistant");
        assert_eq!(config.base_url, "http://127.0.0.1:8080");
        assert_eq!(config.session_id, None);
        assert_eq!(
            config.headers.get("Authorization").map(String::as_str),
            Some("Bearer token")
        );
        assert_eq!(
            config.headers.get("X-Team").map(String::as_str),
            Some("red")
        );
        assert_eq!(config.effective_timeout(), Duration::from_secs(5));
        assert_eq!(
            config.invoke_url().unwrap(),
            "http://127.0.0.1:8080/agents/assistant/invoke"
        );

        let plain = RemoteSpawner::new("http://host").config_for("a");
        assert!((plain.timeout_secs - DEFAULT_REMOTE_TIMEOUT_SECS).abs() < f64::EPSILON);
        assert!(plain.headers.is_empty());
    }

    #[tokio::test]
    async fn spawner_spawns_a_remote_agent_from_the_spec() {
        let spawner = RemoteSpawner::new("http://127.0.0.1:9").with_header("X-Team", "red");
        let spec = SpawnSpec::class_agent("pkg.mod", "HelperAgent").with_session_id("s-1");
        let agent = spawner.spawn(spec).await.unwrap();
        assert_eq!(agent.id(), "HelperAgent");
        assert_eq!(agent.pid(), None);
        assert!(!agent.is_alive());
    }

    #[tokio::test]
    async fn spawner_reports_a_spec_without_an_address() {
        let spawner = RemoteSpawner::new("http://127.0.0.1:9");
        let error = spawner
            .spawn(SpawnSpec::team_agent(json!({"name": "crew"})))
            .await
            .err()
            .expect("a team payload carries no remote agent id");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("no agent_class/agent_module/agent_id"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn spawner_rejects_an_empty_base_url() {
        let spawner = RemoteSpawner::new("");
        let error = spawner
            .spawn(SpawnSpec::class_agent("pkg.mod", "HelperAgent"))
            .await
            .err()
            .expect("an empty base url cannot address an agent");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn lifecycle_toggles_is_started_and_needs_no_network() {
        let agent = RemoteAgent::new(RemoteAgentConfig::new("a", "http://127.0.0.1:9"));
        assert!(!agent.is_started());
        assert!(agent.is_stopped());
        agent.start().await.unwrap();
        assert!(agent.is_started());
        agent.start().await.unwrap();
        assert!(agent.is_started());
        assert!(agent.is_alive());
        assert_eq!(agent.pid(), None);
        agent.stop().await.unwrap();
        assert!(!agent.is_started());
        agent.stop().await.unwrap();
        assert!(agent.shutdown(Duration::from_secs(1)).await.unwrap());
        assert!(!agent.is_alive());
        agent.start().await.unwrap();
        agent.force_kill().await.unwrap();
        assert!(!agent.is_started());
        assert_eq!(agent.agent_id(), "a");
        assert_eq!(agent.protocol(), RemoteProtocol::Http);
    }

    #[tokio::test]
    async fn invoke_posts_the_session_body_and_parses_the_output() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/assistant/invoke"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"output": "hi there", "result_type": "answer"})),
            )
            .mount(&server)
            .await;
        let agent = RemoteAgent::new(
            RemoteAgentConfig::new("assistant", server.uri()).with_session_id("s-1"),
        );
        let output = agent.invoke(json!({"query": "ping"}), None).await.unwrap();
        assert_eq!(output.output, "hi there");

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["session_id"], json!("s-1"));
        assert_eq!(body["inputs"], json!({"query": "ping"}));

        let output = SpawnedAgent::send(&agent, json!({"query": "ping"}))
            .await
            .unwrap();
        assert_eq!(output.output, "hi there");
    }

    #[tokio::test]
    async fn invoke_sends_a_null_session_when_unset() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/a/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output": "ok"})))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        agent.invoke(json!({}), None).await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["session_id"], Value::Null);
    }

    #[tokio::test]
    async fn invoke_maps_http_failures_onto_error_kinds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/missing/invoke"))
            .respond_with(ResponseTemplate::new(404).set_body_string("no such agent"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/agents/boom/invoke"))
            .respond_with(ResponseTemplate::new(500).set_body_string("kaboom"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/agents/garbage/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>nope</html>"))
            .mount(&server)
            .await;

        let error = server_agent(&server, "missing")
            .invoke(json!({}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("no such agent"), "{error}");
        assert!(
            error.to_string().contains("/agents/missing/invoke"),
            "{error}"
        );

        let error = server_agent(&server, "boom")
            .invoke(json!({}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("500"), "{error}");
        assert!(error.to_string().contains("kaboom"), "{error}");

        let error = server_agent(&server, "garbage")
            .invoke(json!({}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("/agents/garbage/invoke"),
            "{error}"
        );
        assert!(error.to_string().contains("<html>nope</html>"), "{error}");
    }

    #[tokio::test]
    async fn invoke_reports_an_unreachable_agent_as_a_transport_error() {
        // Port 1 on loopback is closed: the request fails without any DNS.
        let agent = RemoteAgent::new(RemoteAgentConfig::new("a", "http://127.0.0.1:1"));
        let error = agent.invoke(json!({}), None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("remote agent invoke"), "{error}");
    }

    #[tokio::test]
    async fn stream_yields_chunks_and_ends_on_done() {
        let server = MockServer::start().await;
        let body = [
            chunk_line(0, "he"),
            chunk_line(1, "llo"),
            "keep-alive".to_owned(),
            frame_line(&SpawnMessage::new(SpawnMessageKind::Done, Value::Null)),
        ]
        .join("\n");
        Mock::given(method("POST"))
            .and(path("/agents/a/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let agent = server_agent(&server, "a");
        let items: Vec<_> = RemoteClient::stream(&agent, json!({"query": "hi"}), None)
            .collect()
            .await;
        assert_eq!(items.len(), 2);
        let texts: Vec<Value> = items
            .into_iter()
            .map(|item| item.unwrap().payload)
            .collect();
        assert_eq!(texts[0], json!({"content": "he"}));
        assert_eq!(texts[1], json!({"content": "llo"}));
    }

    #[tokio::test]
    async fn stream_parses_a_trailing_line_without_a_newline() {
        let server = MockServer::start().await;
        let body = chunk_line(0, "only");
        Mock::given(method("POST"))
            .and(path("/agents/a/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert_eq!(
            items.into_iter().next().unwrap().unwrap().payload,
            json!({"content": "only"})
        );
    }

    #[tokio::test]
    async fn stream_maps_an_error_frame_to_one_error_item() {
        let server = MockServer::start().await;
        let body = [
            chunk_line(0, "he"),
            frame_line(&SpawnMessage::new(
                SpawnMessageKind::Error,
                json!({"message": "boom"}),
            )),
            chunk_line(1, "never"),
        ]
        .join("\n");
        Mock::given(method("POST"))
            .and(path("/agents/a/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
            .collect()
            .await;
        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok());
        let error = items[1].as_ref().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("boom"), "{error}");
    }

    #[tokio::test]
    async fn stream_maps_a_bad_status_to_one_error_item() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/a/stream"))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("503"), "{error}");
        assert!(error.to_string().contains("overloaded"), "{error}");
    }

    #[tokio::test]
    async fn stream_rejects_an_unusable_url_as_one_error_item() {
        let agent = RemoteAgent::new(RemoteAgentConfig::new("", "http://127.0.0.1:9"));
        let items: Vec<_> = RemoteClient::stream(&agent, json!({}), None)
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().unwrap_err().kind(), ErrorKind::Validation);
    }

    #[tokio::test]
    async fn cancel_task_posts_the_task_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/a/cancel"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        agent.cancel_task("task-1").await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body, json!({"task_id": "task-1"}));
    }

    #[tokio::test]
    async fn cancel_task_without_a_cancel_path_is_unsupported() {
        let agent = RemoteAgent::new(
            RemoteAgentConfig::new("a", "http://127.0.0.1:9").with_cancel_path(None),
        );
        let error = agent.cancel_task("task-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("remote agent cancel_task"),
            "{error}"
        );
        assert!(error.to_string().contains("cancel_path"), "{error}");
    }

    #[tokio::test]
    async fn cancel_task_maps_a_missing_task_to_not_found() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/agents/a/cancel"))
            .respond_with(ResponseTemplate::new(404).set_body_string("unknown task"))
            .mount(&server)
            .await;
        let agent = server_agent(&server, "a");
        let error = agent.cancel_task("task-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("unknown task"), "{error}");
    }

    #[tokio::test]
    async fn custom_paths_are_honoured() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/run/assistant"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output": "ok"})))
            .mount(&server)
            .await;
        let agent = RemoteAgent::new(
            RemoteAgentConfig::new("assistant", server.uri())
                .with_paths("v2/run/{agent_id}", "v2/events/{agent_id}"),
        );
        let output = agent.invoke(json!({}), None).await.unwrap();
        assert_eq!(output.output, "ok");
    }
}
