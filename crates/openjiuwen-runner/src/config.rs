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

/// Default upper bound of concurrent runs a [`TeamPoolConfig`] admits.
pub const DEFAULT_TEAM_POOL_MAX_RUNS: usize = 8;

/// Default idle deadline in seconds after which a pooled team run is evicted.
pub const DEFAULT_TEAM_POOL_IDLE_TIMEOUT_SECS: f64 = 300.0;

/// Default capacity of one pooled run's inbound human-payload buffer.
pub const DEFAULT_TEAM_POOL_PAYLOAD_CAPACITY: usize = 64;

/// Default number of finished pool entries retained for late observers.
pub const DEFAULT_TEAM_POOL_RETAIN_FINISHED: usize = 32;

/// Default broadcast capacity of one pooled run's frame channel.
pub const DEFAULT_TEAM_POOL_FRAME_CAPACITY: usize = 256;

/// Default interval in seconds between two idle-eviction scans.
pub const DEFAULT_TEAM_POOL_REAPER_INTERVAL_SECS: f64 = 30.0;

/// Default maximum await in seconds of a stopped pool run task.
pub const DEFAULT_TEAM_POOL_STOP_TIMEOUT_SECS: f64 = 5.0;

/// Settings of the in-process team runtime pool ([`crate::pool::TeamPool`]).
///
/// Python correspondence: `openjiuwen/core/runner/team_runner.py` keeps these
/// bounds as module constants of `TeamRuntimeManager` and its
/// `TeamRuntimePool` (max active runtimes, orphan reclaim, per-run interaction
/// gate capacity). The struct keeps every value declarative so a host can
/// persist its runner configuration; the pool reads it once at construction.
///
/// Serialized with `snake_case` field names, every missing field defaulted, and
/// unknown fields tolerated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamPoolConfig {
    /// Upper bound of concurrent pooled team runs; the effective value is
    /// clamped to at least one ([`TeamPoolConfig::max_runs`]).
    pub max_concurrent_runs: usize,
    /// Idle deadline after which a pooled run is evicted, in seconds; a
    /// non-positive value disables eviction ([`TeamPoolConfig::idle_timeout`]).
    pub idle_timeout_secs: f64,
    /// Deadline applied to one pooled run, in seconds; `None` means no deadline.
    pub run_timeout_secs: Option<f64>,
    /// Capacity of one pooled run's inbound human-payload buffer, clamped to at
    /// least one ([`TeamPoolConfig::payload_capacity`]).
    pub payload_capacity: usize,
    /// How many finished pool entries are retained for late observers (best
    /// effort: an entry with live leases is never pruned).
    pub retain_finished: usize,
    /// Broadcast capacity of one pooled run's frame channel, clamped to at least
    /// one ([`TeamPoolConfig::frame_capacity`]).
    pub frame_capacity: usize,
    /// Interval between two idle-eviction scans, in seconds; a non-positive
    /// value falls back to one second ([`TeamPoolConfig::reaper_interval`]).
    pub reaper_interval_secs: f64,
    /// Upper bound of the await on a stopped pool run task, in seconds
    /// ([`TeamPoolConfig::stop_timeout`]).
    pub stop_timeout_secs: f64,
}

impl Default for TeamPoolConfig {
    fn default() -> Self {
        Self {
            max_concurrent_runs: DEFAULT_TEAM_POOL_MAX_RUNS,
            idle_timeout_secs: DEFAULT_TEAM_POOL_IDLE_TIMEOUT_SECS,
            run_timeout_secs: None,
            payload_capacity: DEFAULT_TEAM_POOL_PAYLOAD_CAPACITY,
            retain_finished: DEFAULT_TEAM_POOL_RETAIN_FINISHED,
            frame_capacity: DEFAULT_TEAM_POOL_FRAME_CAPACITY,
            reaper_interval_secs: DEFAULT_TEAM_POOL_REAPER_INTERVAL_SECS,
            stop_timeout_secs: DEFAULT_TEAM_POOL_STOP_TIMEOUT_SECS,
        }
    }
}

impl TeamPoolConfig {
    /// Builds the documented default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the concurrency bound, clamped to at least one.
    #[must_use]
    pub fn with_max_concurrent_runs(mut self, max_concurrent_runs: usize) -> Self {
        self.max_concurrent_runs = max_concurrent_runs.max(1);
        self
    }

    /// Sets the idle deadline in seconds; a non-positive value disables eviction.
    #[must_use]
    pub fn with_idle_timeout_secs(mut self, secs: f64) -> Self {
        self.idle_timeout_secs = secs;
        self
    }

    /// Sets the per-run deadline in seconds (`None` disables it).
    #[must_use]
    pub fn with_run_timeout_secs(mut self, secs: Option<f64>) -> Self {
        self.run_timeout_secs = secs;
        self
    }

    /// Sets the inbound payload buffer capacity, clamped to at least one.
    #[must_use]
    pub fn with_payload_capacity(mut self, capacity: usize) -> Self {
        self.payload_capacity = capacity.max(1);
        self
    }

    /// Sets how many finished entries are retained.
    #[must_use]
    pub fn with_retain_finished(mut self, retain_finished: usize) -> Self {
        self.retain_finished = retain_finished;
        self
    }

    /// Sets the frame channel capacity, clamped to at least one.
    #[must_use]
    pub fn with_frame_capacity(mut self, capacity: usize) -> Self {
        self.frame_capacity = capacity.max(1);
        self
    }

    /// Sets the idle-eviction scan interval in seconds.
    #[must_use]
    pub fn with_reaper_interval_secs(mut self, secs: f64) -> Self {
        self.reaper_interval_secs = secs;
        self
    }

    /// Sets the upper bound of the await on a stopped run task, in seconds.
    #[must_use]
    pub fn with_stop_timeout_secs(mut self, secs: f64) -> Self {
        self.stop_timeout_secs = secs;
        self
    }

    /// The effective concurrency bound: never below one.
    #[must_use]
    pub fn max_runs(&self) -> usize {
        self.max_concurrent_runs.max(1)
    }

    /// The idle deadline, or `None` when eviction is disabled (`<= 0`).
    #[must_use]
    pub fn idle_timeout(&self) -> Option<std::time::Duration> {
        positive_duration(self.idle_timeout_secs)
    }

    /// The per-run deadline, or `None` when none is configured (`<= 0`).
    #[must_use]
    pub fn run_timeout(&self) -> Option<std::time::Duration> {
        self.run_timeout_secs.and_then(positive_duration)
    }

    /// The idle-eviction scan interval; never zero, `<= 0` falls back to one
    /// second.
    #[must_use]
    pub fn reaper_interval(&self) -> std::time::Duration {
        positive_duration(self.reaper_interval_secs)
            .unwrap_or_else(|| std::time::Duration::from_secs(1))
    }

    /// The upper bound of the await on a stopped run task.
    #[must_use]
    pub fn stop_timeout(&self) -> std::time::Duration {
        positive_duration(self.stop_timeout_secs).unwrap_or(std::time::Duration::ZERO)
    }

    /// The inbound payload buffer capacity: never below one.
    #[must_use]
    pub fn payload_capacity(&self) -> usize {
        self.payload_capacity.max(1)
    }

    /// The frame channel capacity: never below one.
    #[must_use]
    pub fn frame_capacity(&self) -> usize {
        self.frame_capacity.max(1)
    }
}

/// A positive, finite number of seconds, or `None` when the value disables the
/// deadline.
fn positive_duration(secs: f64) -> Option<std::time::Duration> {
    if secs.is_finite() && secs > 0.0 {
        Some(std::time::Duration::from_secs_f64(secs))
    } else {
        None
    }
}

/// Transport backing [`crate::MessageQueue`].
///
/// Python's `MessageQueueType` has `pulsar` and `fake`; the in-process backend
/// is spelled `in_memory` here (Rust naming) and is the only implemented one.
/// `Pulsar` and `ZeroMq` are the documented seams a future backend implements:
/// [`crate::create_message_queue`] returns
/// [`openjiuwen_common::error::Error::Unsupported`] for both, and
/// [`crate::mq::build_message_queue`] yields a typed
/// [`DeferredMessageQueue`](crate::mq::DeferredMessageQueue) placeholder rather
/// than an in-memory queue.
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
///
/// The runner builds the backend from this tree in [`crate::Runner::with_config`]:
/// [`MessageQueueKind::InMemory`] yields the in-process queue, and any deferred
/// kind (`pulsar` / `zeromq`) yields the typed
/// [`DeferredMessageQueue`](crate::mq::DeferredMessageQueue) placeholder whose
/// every operation is `Error::Unsupported` — the configured kind is never
/// silently served by the in-memory queue. [`crate::Runner::start`] returns that
/// same typed error for a deferred kind.
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
/// [`crate::Runner`] never starts a broker — [`crate::Runner::start`] refuses a
/// [`RunnerConfig::distributed_mode`] runner with `Error::Unsupported` naming the
/// missing `drunner/` transports (Pulsar/ZeroMQ reply-topic routing,
/// `RemoteProtocol::Mq`/`A2a`, cross-process attach). The values are persisted
/// and inspected by hosts; nothing in-process reads them. See `docs/PARITY.md`.
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
    /// Whether distributed execution was configured (Python `distributed_mode`).
    ///
    /// Persisted and inspected by hosts, but **execution is refused**: this build
    /// ships no distributed transport (`drunner/` — Pulsar/ZeroMQ reply-topic
    /// routing, `RemoteProtocol::Mq`/`A2a`, cross-process attach), so
    /// [`crate::Runner::start`] returns `Error::Unsupported` naming it.
    /// [`crate::Runner::with_config`] still accepts the tree (a host can build,
    /// serialize, and inspect it) and every in-process path — runs, callbacks,
    /// registries, the team pool — is unaffected by the flag.
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
    /// Settings of the in-process team runtime pool
    /// ([`crate::pool::TeamPool`]).
    pub team_pool: TeamPoolConfig,
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
            team_pool: TeamPoolConfig::default(),
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

    /// Enables distributed mode: configuration only.
    ///
    /// The flag round-trips and is visible to hosts, but [`crate::Runner::start`]
    /// refuses a distributed runner (`Error::Unsupported`) until a distributed
    /// transport lands; see [`RunnerConfig::distributed_mode`].
    #[must_use]
    pub fn with_distributed_mode(mut self, enabled: bool) -> Self {
        self.distributed_mode = enabled;
        self
    }

    /// Overrides the team-pool settings.
    #[must_use]
    pub fn with_team_pool(mut self, config: TeamPoolConfig) -> Self {
        self.team_pool = config;
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
        self.checkpointer
            .as_ref()
            .and_then(|config| config.path.as_ref())
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

    #[test]
    fn team_pool_defaults_match_the_constants() {
        let pool = TeamPoolConfig::default();
        assert_eq!(pool, TeamPoolConfig::new());
        assert_eq!(pool.max_concurrent_runs, DEFAULT_TEAM_POOL_MAX_RUNS);
        assert_eq!(pool.max_runs(), 8);
        assert_eq!(pool.idle_timeout_secs, DEFAULT_TEAM_POOL_IDLE_TIMEOUT_SECS);
        assert_eq!(
            pool.idle_timeout(),
            Some(std::time::Duration::from_secs(300))
        );
        assert!(pool.run_timeout_secs.is_none());
        assert!(pool.run_timeout().is_none());
        assert_eq!(pool.payload_capacity, DEFAULT_TEAM_POOL_PAYLOAD_CAPACITY);
        assert_eq!(pool.payload_capacity(), 64);
        assert_eq!(pool.retain_finished, DEFAULT_TEAM_POOL_RETAIN_FINISHED);
        assert_eq!(pool.frame_capacity(), DEFAULT_TEAM_POOL_FRAME_CAPACITY);
        assert_eq!(pool.reaper_interval(), std::time::Duration::from_secs(30));
        assert_eq!(pool.stop_timeout(), std::time::Duration::from_secs(5));
        assert_eq!(RunnerConfig::default().team_pool, pool);
    }

    #[test]
    fn team_pool_round_trips_through_json() {
        let pool = TeamPoolConfig::default()
            .with_max_concurrent_runs(3)
            .with_run_timeout_secs(Some(1.5))
            .with_retain_finished(4);
        let text = serde_json::to_string(&pool).expect("serialize");
        let back: TeamPoolConfig = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, pool);

        let config = RunnerConfig::default().with_team_pool(pool.clone());
        let text = serde_json::to_string(&config).expect("serialize");
        let back: RunnerConfig = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, config);
        assert_eq!(back.team_pool, pool);
    }

    #[test]
    fn a_partial_team_pool_object_fills_the_rest_from_defaults() {
        let config: RunnerConfig =
            serde_json::from_value(serde_json::json!({"team_pool": {"max_concurrent_runs": 2}}))
                .expect("deserialize");
        assert_eq!(config.team_pool.max_concurrent_runs, 2);
        assert_eq!(
            config.team_pool.idle_timeout_secs,
            DEFAULT_TEAM_POOL_IDLE_TIMEOUT_SECS
        );
        assert_eq!(
            config.team_pool.payload_capacity,
            DEFAULT_TEAM_POOL_PAYLOAD_CAPACITY
        );
        assert_eq!(
            config.team_pool.frame_capacity,
            DEFAULT_TEAM_POOL_FRAME_CAPACITY
        );
        assert!(config.team_pool.run_timeout_secs.is_none());
        // The rest of the runner configuration is untouched by the nested block.
        assert_eq!(config.runner_id, "global");
        assert_eq!(config.message_queue.kind, MessageQueueKind::InMemory);
    }

    #[test]
    fn team_pool_tolerates_unknown_fields() {
        let pool: TeamPoolConfig =
            serde_json::from_value(serde_json::json!({"retain_finished": 2, "future_field": true}))
                .expect("deserialize");
        assert_eq!(pool.retain_finished, 2);
        assert_eq!(pool.max_runs(), 8);

        let config: RunnerConfig = serde_json::from_value(serde_json::json!({
            "team_pool": {"stop_timeout_secs": 0.25, "unknown": [1, 2]},
        }))
        .expect("deserialize");
        assert_eq!(
            config.team_pool.stop_timeout(),
            std::time::Duration::from_millis(250)
        );
    }

    #[test]
    fn team_pool_builders_clamp_and_helpers_reject_non_positive_values() {
        let pool = TeamPoolConfig::default()
            .with_max_concurrent_runs(0)
            .with_payload_capacity(0)
            .with_frame_capacity(0);
        assert_eq!(pool.max_concurrent_runs, 1);
        assert_eq!(pool.payload_capacity, 1);
        assert_eq!(pool.frame_capacity, 1);

        let pool = TeamPoolConfig::default()
            .with_idle_timeout_secs(0.0)
            .with_run_timeout_secs(Some(-2.0))
            .with_reaper_interval_secs(-1.0)
            .with_stop_timeout_secs(f64::NAN);
        assert!(pool.idle_timeout().is_none(), "eviction is disabled");
        assert!(pool.run_timeout().is_none());
        assert_eq!(pool.reaper_interval(), std::time::Duration::from_secs(1));
        assert_eq!(pool.stop_timeout(), std::time::Duration::ZERO);

        // A defensive accessor clamps a deserialized zero even though the
        // builder never stores one.
        let raw: TeamPoolConfig =
            serde_json::from_value(serde_json::json!({"max_concurrent_runs": 0})).expect("parse");
        assert_eq!(raw.max_concurrent_runs, 0);
        assert_eq!(raw.max_runs(), 1);

        let pool = TeamPoolConfig::default()
            .with_idle_timeout_secs(2.5)
            .with_run_timeout_secs(Some(1.25));
        assert_eq!(
            pool.idle_timeout(),
            Some(std::time::Duration::from_millis(2500))
        );
        assert_eq!(
            pool.run_timeout(),
            Some(std::time::Duration::from_millis(1250))
        );
    }

    #[test]
    fn the_team_pool_block_has_the_documented_wire_names() {
        let value = serde_json::to_value(TeamPoolConfig::default()).expect("serialize");
        for field in [
            "max_concurrent_runs",
            "idle_timeout_secs",
            "run_timeout_secs",
            "payload_capacity",
            "retain_finished",
            "frame_capacity",
            "reaper_interval_secs",
            "stop_timeout_secs",
        ] {
            assert!(value.get(field).is_some(), "missing field {field}");
        }
        assert_eq!(value["run_timeout_secs"], serde_json::Value::Null);
    }
}
