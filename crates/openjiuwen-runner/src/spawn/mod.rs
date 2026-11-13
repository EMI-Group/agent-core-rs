//! The spawn seam of the runner: drive an agent that lives behind a process or
//! an HTTP endpoint.
//!
//! The seam is transport-neutral. [`crate::spawn::AgentSpawner`] turns a
//! JSON-safe [`crate::spawn::SpawnSpec`] into a
//! [`crate::spawn::SpawnedAgent`] handle that sends one invocation, streams its
//! `AgentChunk` frames, reports liveness and a process id, and shuts down. Two
//! backends implement it:
//!
//! - [`crate::spawn::SubprocessSpawner`] launches a real child with
//!   `tokio::process`, writes a bootstrap frame to its stdin, and returns a
//!   [`crate::spawn::SpawnedProcess`] handle; dropping the last handle reference
//!   kills and reaps the child.
//! - [`crate::spawn::RemoteSpawner`] and [`crate::spawn::RemoteAgent`] address an
//!   agent served over HTTP with `reqwest`, reusing the same frame vocabulary on
//!   an NDJSON stream endpoint.
//!
//! Both speak the wire in [`crate::spawn::protocol`] (one JSON object per line,
//! [`crate::spawn::SpawnMessage`]) over the bounded line framing in
//! [`crate::spawn::frames`]: non-JSON stdout/body lines are skipped as noise,
//! while a line that is JSON but not a frame — or that is not UTF-8, or that
//! exceeds the frame cap — is a validation error.
//!
//! [`crate::spawn::UnsupportedSpawner`] stays the documented default until a host
//! installs a backend with `Runner::set_spawner`; it reports
//! `openjiuwen_common::error::Error::Unsupported`.
//!
//! Deferred transports are typed, never silent: `RemoteProtocol::Mq`
//! (Pulsar/ZeroMQ reply-topic routing) and `RemoteProtocol::A2a` (the A2A remote
//! transport) fail with `Unsupported` from
//! [`crate::spawn::create_remote_client`]. Python's `python -m ...` child, its
//! `process_manager` pool and health-check escalation machinery, and the drunner
//! reply-topic plumbing are not ported.
//!
//! Ports the surface of Python `openjiuwen/core/runner/spawn/` (`agent_config.py`,
//! `protocol.py`, `process_manager.py`) plus the HTTP surface of
//! `openjiuwen/core/runner/drunner/remote_client/`.

pub mod frames;
pub mod protocol;
pub mod remote;
pub mod subprocess;

pub use frames::{frame_bytes, parse_frame_line, read_frame, DEFAULT_MAX_FRAME_BYTES};
pub use protocol::{SpawnMessage, SpawnMessageKind};
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

/// Everything a backend needs to launch one child agent.
///
/// Mirrors Python's `SpawnAgentConfig`/`ClassAgentSpawnConfig` union flattened
/// into one struct: `agent_module`/`agent_class`/`init_kwargs` are only read for
/// [`SpawnAgentKind::ClassAgent`], `payload` only for
/// [`SpawnAgentKind::TeamAgent`]. The type is serde-serializable so it can be
/// handed to a child as JSON, and it tolerates unknown fields and partial JSON
/// (missing fields fall back to [`SpawnSpec::default`]).
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
}

/// A live child agent handle returned by [`AgentSpawner::spawn`].
///
/// The handle is the runner-visible half of Python's `ChildProcessManager`: it
/// sends input, streams output frames, reports liveness, and shuts the child
/// down. Implementations own the process; nothing in this crate does.
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
}

/// The backend seam: builds a [`SpawnedAgent`] from a [`SpawnSpec`].
///
/// Installed on a runner with `Runner::set_spawner`; [`UnsupportedSpawner`] is
/// the default.
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    #[test]
    fn spec_defaults_match_python() {
        let spec = SpawnSpec::default();
        assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
        assert_eq!(spec.agent_module, None);
        assert_eq!(spec.agent_class, None);
        assert_eq!(spec.init_kwargs, json!({}));
        assert_eq!(spec.payload, Value::Null);
        assert_eq!(spec.session_id, None);
        assert!((spec.health_check_interval_secs - 5.0).abs() < f64::EPSILON);
        assert!((spec.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
        assert!(spec.envs.is_empty());
    }

    #[test]
    fn kind_wire_names_are_snake_case() {
        assert_eq!(
            serde_json::to_value(SpawnAgentKind::ClassAgent).unwrap(),
            json!("class_agent")
        );
        assert_eq!(
            serde_json::to_value(SpawnAgentKind::TeamAgent).unwrap(),
            json!("team_agent")
        );
        let parsed: SpawnAgentKind = serde_json::from_value(json!("team_agent")).unwrap();
        assert_eq!(parsed, SpawnAgentKind::TeamAgent);
        assert_eq!(SpawnAgentKind::default(), SpawnAgentKind::ClassAgent);
    }

    #[test]
    fn class_agent_builder_fills_module_and_class() {
        let spec = SpawnSpec::class_agent("my_pkg.agents", "HelperAgent");
        assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
        assert_eq!(spec.agent_module.as_deref(), Some("my_pkg.agents"));
        assert_eq!(spec.agent_class.as_deref(), Some("HelperAgent"));
        assert_eq!(spec.init_kwargs, json!({}));
    }

    #[test]
    fn team_agent_builder_carries_the_payload() {
        let spec = SpawnSpec::team_agent(json!({"name": "crew"}));
        assert_eq!(spec.agent_kind, SpawnAgentKind::TeamAgent);
        assert_eq!(spec.payload, json!({"name": "crew"}));
        assert_eq!(spec.agent_module, None);
    }

    #[test]
    fn builders_chain() {
        let spec = SpawnSpec::class_agent("mod", "Cls")
            .with_session_id("session-1")
            .with_init_kwargs(json!({"a": 1}))
            .with_env("OPENJIUWEN_LOG_LEVEL", "debug")
            .with_env("API_KEY", "secret");
        assert_eq!(spec.session_id.as_deref(), Some("session-1"));
        assert_eq!(spec.init_kwargs, json!({"a": 1}));
        assert_eq!(spec.envs.len(), 2);
        assert_eq!(spec.envs.get("API_KEY").map(String::as_str), Some("secret"));
        assert_eq!(
            spec.envs.get("OPENJIUWEN_LOG_LEVEL").map(String::as_str),
            Some("debug")
        );
    }

    #[test]
    fn spec_round_trips_through_json() {
        let spec = SpawnSpec::class_agent("mod", "Cls")
            .with_session_id("s")
            .with_init_kwargs(json!({"n": 2}))
            .with_env("K", "V");
        let value = serde_json::to_value(&spec).unwrap();
        assert_eq!(value["agent_kind"], json!("class_agent"));
        assert_eq!(value["health_check_interval_secs"], json!(5.0));
        assert_eq!(value["shutdown_timeout_secs"], json!(10.0));
        let restored: SpawnSpec = serde_json::from_value(value).unwrap();
        assert_eq!(restored, spec);
    }

    #[test]
    fn spec_tolerates_unknown_fields() {
        let spec: SpawnSpec = serde_json::from_value(json!({
            "agent_kind": "team_agent",
            "payload": {"name": "crew"},
            "extra_field": "ignored",
        }))
        .unwrap();
        assert_eq!(spec.agent_kind, SpawnAgentKind::TeamAgent);
        assert_eq!(spec.payload, json!({"name": "crew"}));
        assert!((spec.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn spec_accepts_partial_json() {
        let spec: SpawnSpec = serde_json::from_value(json!({
            "agent_module": "mod",
            "agent_class": "Cls",
            "session_id": "s-1",
        }))
        .unwrap();
        assert_eq!(spec.agent_kind, SpawnAgentKind::ClassAgent);
        assert_eq!(spec.agent_module.as_deref(), Some("mod"));
        assert_eq!(spec.agent_class.as_deref(), Some("Cls"));
        assert_eq!(spec.session_id.as_deref(), Some("s-1"));
        assert_eq!(spec.init_kwargs, json!({}));
        assert!(spec.envs.is_empty());
    }

    #[tokio::test]
    async fn unsupported_spawner_reports_the_seam() {
        let error = UnsupportedSpawner
            .spawn(SpawnSpec::class_agent("mod", "Cls"))
            .await
            .err()
            .expect("the built-in spawner never spawns");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("subprocess agent spawning"),
            "unexpected message: {error}"
        );
        assert!(
            error.to_string().contains("Runner::set_spawner"),
            "unexpected message: {error}"
        );
    }
}
