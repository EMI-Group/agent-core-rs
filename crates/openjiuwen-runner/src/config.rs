//! Runner configuration: the [`RunnerConfig`] tree, message-queue settings, and
//! the defaults Python's `runner_config.py` prescribes.
//!
//! The tree is process-local data plus declarative backend selection: a
//! checkpointer is described by [`openjiuwen_session::checkpoint::CheckpointerConfig`]
//! and built by [`crate::Runner::start`], never carried as a live trait object.
//! Python parity: `RunnerConfig(distributed_mode=False,
//! distributed_config=DistributedConfig(message_queue_config=MessageQueueConfig(type="fake")))`
//! is the shipped default (`DEFAULT_RUNNER_CONFIG`), i.e. **distributed mode is
//! off by default**.

use openjiuwen_session::checkpoint::CheckpointerConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default session id Python uses when neither the caller nor `inputs`
/// provides one (`_DEFAULT_AGENT_SESSION_ID`).
pub const DEFAULT_SESSION_ID: &str = "default_session";

/// Reserved input key carrying the agent conversation id (Python
/// `_AGENT_CONVERSATION_ID`).
pub const CONVERSATION_ID_KEY: &str = "conversation_id";

/// Default runner id (`_DEFAULT_RUNNER_ID`).
pub const DEFAULT_RUNNER_ID: &str = "global";

/// Default bounded capacity of one in-process topic.
pub const DEFAULT_TOPIC_CAPACITY: usize = 1024;

/// Default deadline of one request/reply message on the in-process queue,
/// in seconds (Python `DistributedConfig.request_timeout = 30.0`).
pub const DEFAULT_REQUEST_TIMEOUT_SECS: f64 = 30.0;

/// Transport backing [`crate::MessageQueue`].
///
/// Python's `MessageQueueType` has `pulsar` and `fake`; the in-process backend
/// is spelled `in_memory` here (Rust naming) and is the only implemented one.
/// `Pulsar` and `ZeroMq` are the documented seams a future backend implements;
/// [`crate::create_message_queue`] returns
/// [`openjiuwen_common::error::Error::Unsupported`] for both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageQueueKind {
    /// Tokio-native, topic-per-worker queue ([`crate::InMemoryMessageQueue`]).
    #[default]
    InMemory,
    /// Apache Pulsar (deferred; `openjiuwen-extensions` owns the client).
    Pulsar,
    /// `ZeroMQ` (deferred).
    #[serde(rename = "zeromq")]
    ZeroMq,
}

impl MessageQueueKind {
    /// The `snake_case` wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InMemory => "in_memory",
            Self::Pulsar => "pulsar",
            Self::ZeroMq => "zeromq",
        }
    }
}

/// Settings for the message queue the runner owns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageQueueConfig {
    /// Which transport to build.
    pub kind: MessageQueueKind,
    /// Bounded per-topic capacity (in-flight deliveries).
    pub capacity: usize,
    /// Deadline applied by `request` when the caller passes none; `None` waits
    /// forever, `0.0` disables the default deadline.
    pub request_timeout_secs: Option<f64>,
    /// Broker URL for `pulsar` / `zeromq` (ignored by `in_memory`).
    pub url: Option<String>,
    /// Broker worker threads for the deferred backends.
    pub max_workers: usize,
}

impl Default for MessageQueueConfig {
    fn default() -> Self {
        Self {
            kind: MessageQueueKind::InMemory,
            capacity: DEFAULT_TOPIC_CAPACITY,
            request_timeout_secs: Some(DEFAULT_REQUEST_TIMEOUT_SECS),
            url: None,
            max_workers: 8,
        }
    }
}

impl MessageQueueConfig {
    /// Builds an in-process configuration with the documented defaults.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Returns the default request deadline as a [`std::time::Duration`].
    ///
    /// `None` when the config disables it (`None` or a non-positive value).
    #[must_use]
    pub fn request_timeout(&self) -> Option<std::time::Duration> {
        match self.request_timeout_secs {
            Some(secs) if secs > 0.0 => Some(std::time::Duration::from_secs_f64(secs)),
            _ => None,
        }
    }

    /// Sets the topic capacity.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Sets the default request deadline in seconds.
    #[must_use]
    pub fn with_request_timeout_secs(mut self, secs: Option<f64>) -> Self {
        self.request_timeout_secs = secs;
        self
    }
}

/// Declarative description of the distributed backend the runner would use.
///
/// Distributed execution is **deferred** in this port: the struct preserves
/// Python's field names and defaults so configuration round-trips, but
/// [`crate::Runner`] never starts a broker. See `docs/PARITY.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DistributedConfig {
    /// Deadline of one remote request, in seconds (Python 30.0).
    pub request_timeout_secs: f64,
    /// Upper bound of concurrent remote requests (Python 10000).
    pub max_request_concurrency: usize,
    /// Topic template for one agent (Python
    /// `openjiuwen.single_agent.{agent_id}.{version}`).
    pub agent_topic_template: String,
    /// Topic template for runner replies (Python
    /// `openjiuwen.reply.runner.{instance_id}`).
    pub reply_topic_template: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_request_concurrency: 10_000,
            agent_topic_template: "openjiuwen.single_agent.{agent_id}.{version}".to_owned(),
            reply_topic_template: "openjiuwen.reply.runner.{instance_id}".to_owned(),
        }
    }
}

/// The runner's configuration tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunnerConfig {
    /// Identifier of the runner instance; `global` for the process singleton.
    pub runner_id: String,
    /// Whether distributed execution was configured (deferred; see
    /// [`DistributedConfig`]).
    pub distributed_mode: bool,
    /// Distributed settings (kept for parity and future use).
    pub distributed: DistributedConfig,
    /// Settings of the runner-owned local message queue.
    pub message_queue: MessageQueueConfig,
    /// Session id used when neither the caller nor `inputs["conversation_id"]`
    /// provides one.
    pub default_session_id: String,
    /// Checkpointer built by [`crate::Runner::start`] and injected into every
    /// session the runner creates.
    pub checkpointer: Option<CheckpointerConfig>,
    /// Runner-level default run deadline in seconds; `None` means no deadline.
    pub default_run_timeout_secs: Option<f64>,
    /// How long `stop` waits for in-flight queue deliveries before returning.
    pub shutdown_timeout_secs: f64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            runner_id: DEFAULT_RUNNER_ID.to_owned(),
            distributed_mode: false,
            distributed: DistributedConfig::default(),
            message_queue: MessageQueueConfig::default(),
            default_session_id: DEFAULT_SESSION_ID.to_owned(),
            checkpointer: None,
            default_run_timeout_secs: None,
            shutdown_timeout_secs: 5.0,
        }
    }
}

impl RunnerConfig {
    /// Builds the documented default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the session id used when the caller supplies none.
    #[must_use]
    pub fn with_default_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.default_session_id = session_id.into();
        self
    }

    /// Overrides the message-queue settings.
    #[must_use]
    pub fn with_message_queue(mut self, config: MessageQueueConfig) -> Self {
        self.message_queue = config;
        self
    }

    /// Selects the checkpointer the runner builds on `start`.
    #[must_use]
    pub fn with_checkpointer(mut self, config: CheckpointerConfig) -> Self {
        self.checkpointer = Some(config);
        self
    }

    /// Sets the runner-level default run deadline in seconds.
    #[must_use]
    pub fn with_default_run_timeout_secs(mut self, secs: Option<f64>) -> Self {
        self.default_run_timeout_secs = secs;
        self
    }

    /// Enables the (deferred) distributed mode; configuration only.
    #[must_use]
    pub fn with_distributed_mode(mut self, enabled: bool) -> Self {
        self.distributed_mode = enabled;
        self
    }

    /// Returns the default run deadline, if one is configured.
    #[must_use]
    pub fn default_run_timeout(&self) -> Option<std::time::Duration> {
        match self.default_run_timeout_secs {
            Some(secs) if secs > 0.0 => Some(std::time::Duration::from_secs_f64(secs)),
            _ => None,
        }
    }

    /// Returns the shutdown grace period.
    #[must_use]
    pub fn shutdown_timeout(&self) -> std::time::Duration {
        if self.shutdown_timeout_secs > 0.0 {
            std::time::Duration::from_secs_f64(self.shutdown_timeout_secs)
        } else {
            std::time::Duration::from_secs(0)
        }
    }

    /// Returns the file-system path of the configured checkpointer, if any
    /// (convenience for hosts that log their storage root).
    #[must_use]
    pub fn checkpointer_path(&self) -> Option<&PathBuf> {
        self.checkpointer.as_ref().and_then(|config| config.path.as_ref())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let config = RunnerConfig::default();
        assert_eq!(config.runner_id, "global");
        assert!(!config.distributed_mode);
        assert_eq!(config.default_session_id, "default_session");
        assert_eq!(config.message_queue.kind, MessageQueueKind::InMemory);
        assert_eq!(config.message_queue.capacity, 1024);
        assert_eq!(
            config.message_queue.request_timeout(),
            Some(std::time::Duration::from_secs(30))
        );
        assert_eq!(config.shutdown_timeout_secs, 5.0);
        assert!(config.checkpointer.is_none());
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = RunnerConfig::default()
            .with_default_session_id("conv-1")
            .with_distributed_mode(true);
        let text = serde_json::to_string(&config).expect("serialize");
        let back: RunnerConfig = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, config);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let value = serde_json::json!({"runner_id": "custom", "future_field": 7});
        let config: RunnerConfig = serde_json::from_value(value).expect("deserialize");
        assert_eq!(config.runner_id, "custom");
        assert_eq!(config.message_queue.kind, MessageQueueKind::InMemory);
    }

    #[test]
    fn message_queue_kind_wire_names() {
        let kinds = [
            (MessageQueueKind::InMemory, "in_memory"),
            (MessageQueueKind::Pulsar, "pulsar"),
            (MessageQueueKind::ZeroMq, "zeromq"),
        ];
        for (kind, name) in kinds {
            assert_eq!(kind.as_str(), name);
            assert_eq!(serde_json::to_value(kind).expect("serialize"), name);
        }
    }

    #[test]
    fn timeout_helpers_reject_non_positive_values() {
        let queue = MessageQueueConfig::default().with_request_timeout_secs(Some(0.0));
        assert!(queue.request_timeout().is_none());
        let runner = RunnerConfig::default().with_default_run_timeout_secs(Some(-1.0));
        assert!(runner.default_run_timeout().is_none());
        let runner = RunnerConfig::default().with_default_run_timeout_secs(Some(2.5));
        assert_eq!(
            runner.default_run_timeout(),
            Some(std::time::Duration::from_millis(2500))
        );
    }

    #[test]
    fn capacity_is_clamped_to_one() {
        let queue = MessageQueueConfig::default().with_capacity(0);
        assert_eq!(queue.capacity, 1);
    }
}
