//! The spawn seam of the runner: drive an agent that lives behind a process or
//! an HTTP endpoint.
//!
//! The seam is transport-neutral. [`crate::spawn::AgentSpawner`] turns a
//! JSON-safe [`crate::spawn::SpawnSpec`] into a
//! [`crate::spawn::SpawnedAgent`] handle that sends one invocation, streams its
//! `AgentChunk` frames, reports liveness, a health snapshot and a process id, and
//! shuts down. Four backends implement it, each identified by a
//! [`crate::spawn::SpawnBackend`]:
//!
//! - [`crate::spawn::SubprocessSpawner`] launches a real child with
//!   `tokio::process`, writes a bootstrap frame to its stdin, and returns a
//!   [`crate::spawn::SpawnedProcess`] handle; dropping the last handle reference
//!   kills and reaps the child.
//! - [`crate::spawn::ExternalCommandSpawner`] resolves a host-configured program
//!   on `PATH` and hands it to the same subprocess backend, so a vendor CLI or
//!   any child speaking the protocol needs no separate implementation; its
//!   children are labelled
//!   [`crate::spawn::SpawnBackend::ExternalCommand`] rather than `Subprocess`.
//! - [`crate::spawn::InProcessSpawner`] wraps a host-bound
//!   [`crate::spawn::InProcessHandler`]: the agent lives in this process and
//!   there is no child to signal.
//! - [`crate::spawn::RemoteSpawner`] and [`crate::spawn::RemoteAgent`] address an
//!   agent served over HTTP with `reqwest`, reusing the same frame vocabulary on
//!   an NDJSON stream endpoint.
//!
//! Every backend speaks the wire in [`crate::spawn::protocol`] (one JSON object
//! per line, [`crate::spawn::SpawnMessage`]) over the bounded line framing in
//! [`crate::spawn::frames`]: non-JSON stdout/body lines are skipped as noise,
//! while a line that is JSON but not a frame — or that is not UTF-8, or that
//! exceeds the frame cap — is a validation error.
//!
//! [`crate::spawn::SpawnerRegistry`] hosts several backends at once and routes a
//! [`crate::spawn::SpawnSpec`] to one of them (its `backend` field, then the
//! registry default, then the sole registered spawner), with
//! [`crate::spawn::DeferredSpawner`] answering for every backend that is not
//! registered. [`crate::spawn::UnsupportedSpawner`] stays the documented default
//! until a host installs a backend with `Runner::set_spawner`; it reports
//! `openjiuwen_common::error::Error::Unsupported`.
//!
//! [`crate::spawn::MonitoredSpawnedAgent`] decorates any handle so one spawned
//! child's lifecycle — admission, health observations, and the terminal
//! shutdown/kill/drop — is reported into a [`crate::monitor::RunMonitor`] as a
//! [`crate::monitor::RunKind::Spawn`] run.
//!
//! Deferred transports are typed, never silent: `RemoteProtocol::Mq`
//! (Pulsar/ZeroMQ reply-topic routing) and `RemoteProtocol::A2a` (the A2A remote
//! transport) fail with `Unsupported` from
//! [`crate::spawn::create_remote_client`], and the same two backends are the
//! [`crate::spawn::SpawnBackend`] variants a [`crate::spawn::SpawnerRegistry`]
//! answers with a deferred spawner. Python's `python -m ...` child, its
//! `process_manager` pool and health-check escalation machinery, and the drunner
//! reply-topic plumbing are not ported.
//!
//! Ports the surface of Python `openjiuwen/core/runner/spawn/` (`agent_config.py`,
//! `protocol.py`, `process_manager.py`) plus the HTTP surface of
//! `openjiuwen/core/runner/drunner/remote_client/`.

pub mod external;
pub mod frames;
pub mod in_process;
pub mod monitored;
pub mod protocol;
pub mod registry;
pub mod remote;
pub mod subprocess;

pub use external::ExternalCommandSpawner;
pub use frames::{frame_bytes, parse_frame_line, read_frame, DEFAULT_MAX_FRAME_BYTES};
pub use in_process::{InProcessAgent, InProcessFactory, InProcessHandler, InProcessSpawner};
pub use monitored::MonitoredSpawnedAgent;
pub use protocol::{SpawnMessage, SpawnMessageKind};
pub use registry::{DeferredSpawner, SpawnerRegistry};
pub use remote::{
    create_remote_client, RemoteAgent, RemoteAgentConfig, RemoteClient, RemoteProtocol,
    RemoteSpawner,
};
pub use subprocess::{SpawnedProcess, SubprocessConfig, SubprocessSpawner};

use crate::error;
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Bootstrap kind of a spawned agent (Python `SpawnAgentKind`).
///
/// The kind selects which spawn configuration is meaningful: [`ClassAgent`]
/// carries `agent_module` + `agent_class` + `init_kwargs`, [`TeamAgent`] carries
/// a `payload` describing the team.
///
/// [`ClassAgent`]: SpawnAgentKind::ClassAgent
/// [`TeamAgent`]: SpawnAgentKind::TeamAgent
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawnAgentKind {
    /// Construct a class from `agent_module`/`agent_class` in the child
    /// (Python `CLASS_AGENT`).
    #[default]
    ClassAgent,
    /// Bootstrap a team from a JSON payload (Python `TEAM_AGENT`).
    TeamAgent,
}

/// The transport a [`SpawnedAgent`] was launched over.
///
/// This is the routing key of a [`crate::spawn::SpawnerRegistry`] and the
/// identity a handle reports from [`SpawnedAgent::backend`], so a host can tell a
/// real child from a host-bound handler or an HTTP endpoint without downcasting.
///
/// [`Subprocess`](SpawnBackend::Subprocess),
/// [`ExternalCommand`](SpawnBackend::ExternalCommand),
/// [`InProcess`](SpawnBackend::InProcess) and
/// [`RemoteHttp`](SpawnBackend::RemoteHttp) are implemented;
/// [`RemoteMq`](SpawnBackend::RemoteMq) is Python's drunner Pulsar/ZeroMQ
/// reply-topic transport and [`RemoteA2a`](SpawnBackend::RemoteA2a) the A2A
/// remote transport, both deferred and typed as
/// [`openjiuwen_common::error::Error::Unsupported`] rather than degrading into
/// another backend. [`Unknown`](SpawnBackend::Unknown) is what a handle that does
/// not name its backend reports.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SpawnBackend {
    /// The handle does not name its backend (also: no spawner is registered).
    #[default]
    Unknown,
    /// A child process this crate launched and supervises.
    Subprocess,
    /// A host-configured external program resolved on `PATH`, supervised through
    /// the subprocess backend.
    ExternalCommand,
    /// An agent bound to this process, driven directly without a child.
    InProcess,
    /// An agent served over HTTP(S) (invoke/stream/cancel).
    RemoteHttp,
    /// Python's drunner Pulsar/ZeroMQ reply-topic transport (deferred).
    RemoteMq,
    /// The A2A remote transport (deferred; it lives in `openjiuwen-extensions`).
    RemoteA2a,
}

impl SpawnBackend {
    /// The backend's wire/config name (`snake_case`, the serde name).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Subprocess => "subprocess",
            Self::ExternalCommand => "external_command",
            Self::InProcess => "in_process",
            Self::RemoteHttp => "remote_http",
            Self::RemoteMq => "remote_mq",
            Self::RemoteA2a => "remote_a2a",
        }
    }

    /// Whether this crate implements the backend.
    ///
    /// [`SpawnBackend::RemoteMq`] and [`SpawnBackend::RemoteA2a`] are deferred and
    /// report `false`; so does [`SpawnBackend::Unknown`], which names nothing.
    #[must_use]
    pub const fn is_implemented(self) -> bool {
        matches!(
            self,
            Self::Subprocess | Self::ExternalCommand | Self::InProcess | Self::RemoteHttp
        )
    }
}

impl fmt::Display for SpawnBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A point-in-time health snapshot of one [`SpawnedAgent`].
///
/// Returned by [`SpawnedAgent::health_snapshot`] so a host or a run monitor can
/// copy the backend, liveness, the last observed probe result, the process id,
/// and the last recorded failure without holding the handle or awaiting a probe.
/// It is a snapshot, not a subscription: nothing is refreshed until the next
/// `health`/`is_alive` call.
///
/// The type serializes (`snake_case`, unknown fields tolerated, every field
/// defaulted) so a monitor can log or forward it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SpawnHealth {
    /// Backend that launched the handle.
    pub backend: SpawnBackend,
    /// Whether the handle was alive when the snapshot was taken.
    pub alive: bool,
    /// The last observed probe result.
    pub healthy: bool,
    /// OS process id, when the backend reports one.
    pub pid: Option<u32>,
    /// Last failure the backend recorded, when any.
    pub last_error: Option<String>,
}

impl SpawnHealth {
    /// A snapshot reporting `backend` as alive and healthy, running as `pid`.
    ///
    /// For a backend that knows it is up (an in-process handler, a started HTTP
    /// client) but has no probe result to report.
    #[must_use]
    pub fn healthy(backend: SpawnBackend, pid: Option<u32>) -> Self {
        Self {
            backend,
            alive: true,
            healthy: true,
            pid,
            last_error: None,
        }
    }
}

/// Everything a backend needs to launch one child agent.
///
/// Mirrors Python's `SpawnAgentConfig`/`ClassAgentSpawnConfig` union flattened
/// into one struct: `agent_module`/`agent_class`/`init_kwargs` are only read for
/// [`SpawnAgentKind::ClassAgent`], `payload` only for
/// [`SpawnAgentKind::TeamAgent`]. The type is serde-serializable so it can be
/// handed to a child as JSON, and it tolerates unknown fields and partial JSON
/// (missing fields fall back to [`SpawnSpec::default`]).
///
/// [`SpawnSpec::backend`] is the routing hint
/// [`crate::spawn::SpawnerRegistry`] reads; it is additive metadata a child
/// ignores (unknown fields are tolerated) and it never changes the wire frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SpawnSpec {
    /// Which bootstrap the child performs.
    pub agent_kind: SpawnAgentKind,
    /// Import path of the agent class (Python `ClassAgentSpawnConfig.agent_module`).
    pub agent_module: Option<String>,
    /// Name of the agent class (Python `ClassAgentSpawnConfig.agent_class`).
    pub agent_class: Option<String>,
    /// Keyword arguments for the agent constructor; defaults to `{}`.
    pub init_kwargs: Value,
    /// Team payload (Python `TeamAgentSpawnConfig.payload`); defaults to `null`.
    pub payload: Value,
    /// Backend this spawn should be routed to, when the caller cares; defaults
    /// to `None` (let the spawner or the registry default decide).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<SpawnBackend>,
    /// Session the child binds to, when the caller already owns one.
    pub session_id: Option<String>,
    /// Interval between health checks, in seconds (default `5.0`).
    pub health_check_interval_secs: f64,
    /// Deadline for a graceful shutdown before a kill, in seconds (default `10.0`).
    pub shutdown_timeout_secs: f64,
    /// Environment overrides handed to the child.
    pub envs: BTreeMap<String, String>,
}

impl Default for SpawnSpec {
    fn default() -> Self {
        Self {
            agent_kind: SpawnAgentKind::default(),
            agent_module: None,
            agent_class: None,
            init_kwargs: Value::Object(serde_json::Map::new()),
            payload: Value::Null,
            backend: None,
            session_id: None,
            health_check_interval_secs: 5.0,
            shutdown_timeout_secs: 10.0,
            envs: BTreeMap::new(),
        }
    }
}

impl SpawnSpec {
    /// A [`SpawnAgentKind::ClassAgent`] spec for `module::class`.
    #[must_use]
    pub fn class_agent(module: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            agent_kind: SpawnAgentKind::ClassAgent,
            agent_module: Some(module.into()),
            agent_class: Some(class.into()),
            ..Self::default()
        }
    }

    /// A [`SpawnAgentKind::TeamAgent`] spec carrying `payload`.
    #[must_use]
    pub fn team_agent(payload: Value) -> Self {
        Self {
            agent_kind: SpawnAgentKind::TeamAgent,
            payload,
            ..Self::default()
        }
    }

    /// Sets the session the child binds to.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Replaces the constructor keyword arguments.
    #[must_use]
    pub fn with_init_kwargs(mut self, init_kwargs: Value) -> Self {
        self.init_kwargs = init_kwargs;
        self
    }

    /// Adds one environment override.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.envs.insert(key.into(), value.into());
        self
    }

    /// Routes this spawn to `backend` when a
    /// [`crate::spawn::SpawnerRegistry`] resolves it.
    #[must_use]
    pub fn with_backend(mut self, backend: SpawnBackend) -> Self {
        self.backend = Some(backend);
        self
    }
}

/// A live child agent handle returned by [`AgentSpawner::spawn`].
///
/// The handle is the runner-visible half of Python's `ChildProcessManager`: it
/// sends input, streams output frames, reports liveness and health, and shuts the
/// child down. Implementations own the process; nothing in this crate does.
///
/// Only the transport methods are required. [`Self::backend`],
/// [`Self::is_healthy`], [`Self::health_snapshot`], [`Self::health`],
/// [`Self::cancel_task`], [`Self::send_with_timeout`] and
/// [`Self::stream_with_timeout`] carry defaults so a minimal handle — and every
/// backend that predates them — stays object-safe and compiles unchanged.
#[async_trait]
pub trait SpawnedAgent: Send + Sync {
    /// The handle's stable id (not necessarily the OS process id).
    fn id(&self) -> &str;
    /// Whether the child is still running.
    fn is_alive(&self) -> bool;
    /// The OS process id, when the backend can report one.
    fn pid(&self) -> Option<u32>;
    /// Sends one invocation to the child and awaits its terminal output.
    ///
    /// # Errors
    ///
    /// Returns an error when the child is gone, the transport fails, or the
    /// child reports an error message.
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    /// Sends one invocation and yields [`AgentChunk`] frames as they arrive.
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    /// Sends one invocation, asking the backend to bound it by `timeout`.
    ///
    /// The deadline is a hint, not part of the wire: a backend that cannot bound
    /// a single request ignores it and calls [`Self::send`], and `None` means
    /// "the backend's own default". [`SpawnedProcess`] (the configured
    /// `request_timeout_secs`) and [`RemoteAgent`] (the configured
    /// `timeout_secs`) honour it.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::send`] returns, plus
    /// [`openjiuwen_common::error::Error::Timeout`] from a backend that honours
    /// the deadline.
    ///
    /// [`SpawnedProcess`]: crate::spawn::SpawnedProcess
    /// [`RemoteAgent`]: crate::spawn::RemoteAgent
    async fn send_with_timeout(
        &self,
        inputs: Value,
        _timeout: Option<Duration>,
    ) -> Result<AgentOutput> {
        self.send(inputs).await
    }
    /// Streams one invocation, asking the backend to bound it by `timeout`.
    ///
    /// The same advisory rule as [`Self::send_with_timeout`]: the default ignores
    /// the hint and returns [`Self::stream`].
    fn stream_with_timeout(
        &self,
        inputs: Value,
        _timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        self.stream(inputs)
    }
    /// Asks the child to exit gracefully.
    ///
    /// Returns `true` when it exited within `timeout` and `false` when it had to
    /// be killed.
    ///
    /// # Errors
    ///
    /// Returns an error when the child cannot be signalled at all.
    async fn shutdown(&self, timeout: Duration) -> Result<bool>;
    /// Kills the child unconditionally.
    ///
    /// # Errors
    ///
    /// Returns an error when the process cannot be killed.
    async fn force_kill(&self) -> Result<()>;

    /// The backend that launched this handle.
    ///
    /// Defaults to [`SpawnBackend::Unknown`]: a handle that does not name its
    /// backend is not misreported as one of the real ones.
    fn backend(&self) -> SpawnBackend {
        SpawnBackend::Unknown
    }

    /// The last observed health state of the child.
    ///
    /// Defaults to [`Self::is_alive`] — a backend with no probe has nothing
    /// better to report, so liveness is the answer.
    fn is_healthy(&self) -> bool {
        self.is_alive()
    }

    /// A point-in-time [`SpawnHealth`] snapshot of this handle.
    ///
    /// Composed from [`Self::backend`], [`Self::is_alive`], [`Self::is_healthy`]
    /// and [`Self::pid`], with no fabricated failure; a backend that recorded a
    /// real failure overrides it.
    fn health_snapshot(&self) -> SpawnHealth {
        SpawnHealth {
            backend: self.backend(),
            alive: self.is_alive(),
            healthy: self.is_healthy(),
            pid: self.pid(),
            last_error: None,
        }
    }

    /// Runs one health probe against the child.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] for a backend without a
    /// probe — the message names the backend and points at
    /// [`Self::is_alive`]/[`Self::is_healthy`] — plus whatever the backend's own
    /// probe can fail with: a probe deadline
    /// ([`openjiuwen_common::error::Error::Timeout`]), a dead child or a failed
    /// transport ([`openjiuwen_common::error::Error::Transport`]).
    #[allow(clippy::unused_async_trait_impl)]
    async fn health(&self) -> Result<bool> {
        Err(error::unsupported(
            "agent health probe",
            &format!(
                "the '{}' backend has no probe; use `is_alive`/`is_healthy` or `health_snapshot`",
                self.backend()
            ),
        ))
    }

    /// Asks the backend to cancel the task identified by `task_id`.
    ///
    /// The NDJSON protocol carries no request correlation id, so only a backend
    /// that owns a real task handle (an HTTP endpoint, a broker) can honour this;
    /// the default reports the seam instead of pretending to cancel.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Unsupported`] unless the backend
    /// implements task cancellation — the message names the backend — plus the
    /// backend's own failures: an unknown task
    /// ([`openjiuwen_common::error::Error::NotFound`]), a deadline
    /// ([`openjiuwen_common::error::Error::Timeout`]), or a transport failure
    /// ([`openjiuwen_common::error::Error::Transport`]).
    #[allow(clippy::unused_async_trait_impl)]
    async fn cancel_task(&self, _task_id: &str) -> Result<()> {
        Err(error::unsupported(
            "spawned task cancellation",
            &format!(
                "the '{}' backend owns no cancellable task handle",
                self.backend()
            ),
        ))
    }
}

/// The backend seam: builds a [`SpawnedAgent`] from a [`SpawnSpec`].
///
/// Installed on a runner with `Runner::set_spawner`, or hosted beside other
/// backends in a [`crate::spawn::SpawnerRegistry`]; [`UnsupportedSpawner`] is the
/// default.
#[async_trait]
pub trait AgentSpawner: Send + Sync {
    /// Launches the agent described by `spec`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Unsupported`] when the backend
    /// does not implement spawning, and a transport/validation error when the
    /// launch itself fails.
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;

    /// The backend this spawner launches, so a registry key and a handle agree.
    ///
    /// Defaults to [`SpawnBackend::Unknown`] for a spawner that does not say.
    fn backend(&self) -> SpawnBackend {
        SpawnBackend::Unknown
    }
}

/// The built-in [`AgentSpawner`]: spawns nothing and reports the seam.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsupportedSpawner;

#[async_trait]
impl AgentSpawner for UnsupportedSpawner {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        Err(error::unsupported(
            "subprocess agent spawning",
            "install an AgentSpawner with Runner::set_spawner",
        ))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
