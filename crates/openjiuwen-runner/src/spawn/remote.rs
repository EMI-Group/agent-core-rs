//! The HTTP remote-agent client of the spawn seam.
//!
//! Ports the HTTP half of Python `openjiuwen/core/runner/drunner/remote_client/`
//! (`remote_client.py`'s `RemoteClient` ABC, `remote_client_config.py`'s
//! `ProtocolEnum`, and `remote_agent.py`'s start/stop/invoke/stream/cancel
//! lifecycle) onto `reqwest`, and exposes it through the crate's spawn seam so a
//! runner can address a remote agent exactly like a subprocess one.
//!
//! [`RemoteAgent`] owns the configuration, the URL layout, and the start/stop
//! lifecycle; every request is executed by the private `remote_io` child module,
//! which owns client construction, the `session_id`/`inputs` body, the status
//! mapping, and the NDJSON stream grammar of
//! [`crate::spawn::protocol::SpawnMessage`].
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

#[path = "remote_io.rs"]
mod remote_io;

use crate::error;
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

// The unit tests in `remote_tests.rs` reach the transport helpers through
// `use super::*`, so they stay importable in the test build only.
#[cfg(test)]
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
#[cfg(test)]
use futures::StreamExt;
#[cfg(test)]
use remote_io::{
    body_snippet, map_status, parse_stream_line, take_line, StreamLine, BODY_SNIPPET_CHARS,
};

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
    /// Asks the agent to cancel `task_id`, with no reason.
    ///
    /// The convenience form of [`Self::cancel_task_with_request_id`].
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the client has no cancel endpoint,
    /// [`Error::Validation`] for a blank id or an unusable URL,
    /// [`Error::NotFound`] when the task is unknown, [`Error::Timeout`] when the
    /// deadline elapses, and [`Error::Transport`] for every other failure.
    async fn cancel_task(&self, task_id: &str) -> Result<()>;

    /// Asks the agent to cancel `request_id`, carrying `reason` when known.
    ///
    /// Defaults to [`Self::cancel_task`]: a client whose wire cannot say more than
    /// the id still correlates by id, so nothing is lost by the default.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::cancel_task`] reports for this client.
    async fn cancel_task_with_request_id(
        &self,
        request_id: &str,
        _reason: Option<&str>,
    ) -> Result<()> {
        self.cancel_task(request_id).await
    }
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
    ///
    /// The transport is built lazily on first use, so an agent constructed
    /// directly reports `is_alive() == false` until it is started;
    /// [`RemoteSpawner::spawn`] returns one that is already started.
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
        let client = remote_io::build_client(self.config())?;
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
        remote_io::invoke_http(self, inputs, timeout).await
    }

    fn stream(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        remote_io::stream_http(self, inputs, timeout)
    }

    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        remote_io::cancel_http(self, task_id, None).await
    }

    /// The same endpoint, with the optional reason in the body.
    async fn cancel_task_with_request_id(
        &self,
        request_id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        remote_io::cancel_http(self, request_id, reason).await
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
    /// health endpoint to probe: the answer is the client's own started state. A
    /// handle built by [`RemoteSpawner::spawn`] is started at spawn (so this is
    /// `true` immediately), while one built with [`RemoteAgent::new`] is `false`
    /// until the first [`RemoteClient::invoke`]/[`RemoteClient::stream`]
    /// auto-starts it, or an explicit [`RemoteClient::start`].
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

    /// Bounds this one request by `timeout`; `None` keeps the configured
    /// `timeout_secs`.
    async fn send_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> Result<AgentOutput> {
        RemoteClient::invoke(self, inputs, timeout).await
    }

    /// Bounds this one stream's request by `timeout`; `None` keeps the configured
    /// `timeout_secs`.
    fn stream_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        RemoteClient::stream(self, inputs, timeout)
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
    /// configured, [`openjiuwen_common::error::Error::Validation`] for a blank id,
    /// [`openjiuwen_common::error::Error::NotFound`] when the task is unknown,
    /// [`openjiuwen_common::error::Error::Timeout`] when the deadline elapses, and
    /// [`openjiuwen_common::error::Error::Transport`] for every other failure.
    async fn cancel_task(&self, task_id: &str) -> Result<()> {
        RemoteClient::cancel_task(self, task_id).await
    }

    /// Posts the id **and** the optional reason to the cancel endpoint.
    ///
    /// The body is this crate's own layout (`{"task_id": …, "request_id": …,
    /// "reason": … | null}`), so the id travels under both the original
    /// `task_id` key and its correlation name, and the reason travels with it —
    /// nothing about a remote cancel is dropped on the way out.
    ///
    /// # Errors
    ///
    /// The same errors as [`SpawnedAgent::cancel_task`].
    async fn cancel_task_with_request_id(
        &self,
        request_id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        RemoteClient::cancel_task_with_request_id(self, request_id, reason).await
    }
}

/// An [`AgentSpawner`] that addresses remote agents over HTTP.
///
/// The spawned handle is a [`RemoteAgent`]; the agent id comes from the spawn
/// spec, so a runner registered with this spawner can route a class-agent spawn
/// to a remote endpoint.
///
/// Unlike a directly constructed [`RemoteAgent`], the handle this spawner
/// returns is already started: readiness polled through
/// [`SpawnedAgent::is_alive`]/[`SpawnedAgent::is_healthy`]/[`SpawnedAgent::health`]
/// is `true` immediately, and a client that cannot be built (an unusable base
/// URL, an invalid configured header) fails the spawn rather than the first
/// invocation.
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

    /// The origin, the default deadline, and the header *names* — never a header
    /// value, because that is where an `Authorization` bearer token lives.
    fn describe(&self) -> Value {
        json!({
            "backend": SpawnBackend::RemoteHttp.as_str(),
            "base_url": self.base_url,
            "timeout_secs": self.timeout.map(|timeout| timeout.as_secs_f64()),
            "header_names": self.headers.keys().collect::<Vec<_>>(),
        })
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let agent_id = remote_agent_id(&spec)?;
        let mut config = self.config_for(&agent_id);
        config.session_id.clone_from(&spec.session_id);
        // Fail at spawn time rather than at the first invoke.
        require_base_url(&config.base_url)?;
        require_url(&config)?;
        let agent = RemoteAgent::new(config);
        // A spawned remote child is started, so a host that polls readiness
        // through the seam sees a live handle, and a client that cannot be built
        // fails here instead of inside the first invocation.
        RemoteClient::start(&agent).await?;
        Ok(Arc::new(agent))
    }
}

/// Rejects a base URL the invoker could never use.
///
/// An empty base URL is caught by [`require_base_url`]; this catches a
/// non-empty one that is still not an absolute URL (a missing scheme, an
/// unparsable host), which would otherwise surface only inside a request.
fn require_url(config: &RemoteAgentConfig) -> Result<()> {
    let url = config.invoke_url()?;
    reqwest::Url::parse(&url).map_err(|error| {
        Error::validation(format!(
            "remote agent base_url '{}' is not a usable URL: {error}",
            config.base_url
        ))
    })?;
    Ok(())
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

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;

#[cfg(test)]
mod describe_tests {
    use super::*;

    #[test]
    fn describe_reports_the_origin_and_header_names_only() {
        let spawner = RemoteSpawner::new("http://127.0.0.1:8080")
            .with_timeout(Duration::from_secs(15))
            .with_header("Authorization", "Bearer super-secret-value")
            .with_header("X-Tenant", "acme");
        let described = spawner.describe();
        assert_eq!(described["backend"], json!("remote_http"));
        assert_eq!(described["base_url"], json!("http://127.0.0.1:8080"));
        assert_eq!(described["timeout_secs"], json!(15.0));
        assert_eq!(
            described["header_names"],
            json!(["Authorization", "X-Tenant"]),
            "headers are reported by name only"
        );
        let rendered = described.to_string();
        assert!(
            !rendered.contains("super-secret-value") && !rendered.contains("Bearer"),
            "no header value leaks: {rendered}"
        );
        assert!(
            !rendered.contains("acme"),
            "no header value leaks: {rendered}"
        );

        // An unset deadline is JSON null; an empty header set is [].
        let plain = RemoteSpawner::new("http://127.0.0.1:8080").describe();
        assert_eq!(plain["timeout_secs"], Value::Null);
        assert_eq!(plain["header_names"], json!([]));
    }
}
