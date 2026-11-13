//! The subprocess backend of the spawn seam: a real child process speaking the
//! NDJSON protocol over its stdio pipes.
//!
//! This module owns the spawner, the child handle, and the state they share; the
//! serde configuration itself lives in the private `subprocess_config` sibling and
//! is re-exported here as [`SubprocessConfig`], and the wire exchanges plus the
//! supervision live in the private `subprocess_io` child module.
//!
//! [`SubprocessSpawner`] launches `program` + `args` with piped
//! stdin/stdout/stderr, writes the bootstrap frame
//! (`{"spawn_spec": <SpawnSpec>}`), and returns a [`SpawnedProcess`] handle.
//! Each child inherits its spawner's [`SpawnBackend`] label — `subprocess` for
//! [`SubprocessSpawner::new`], the wrapper's own name for a spawner built with
//! [`SubprocessSpawner::with_backend`] — so a handle, its spawner, and a runner's
//! monitor detail agree on which backend produced the child.
//! [`SpawnedProcess`] exposes the seam surface and hands every exchange to
//! `subprocess_io`: one request frame down stdin followed by frames up stdout,
//! with the wire in [`crate::spawn::protocol`] and the bounded line framing in
//! [`crate::spawn::frames`]. A child implements no task cancellation (the protocol
//! has no task handle to address): stop one with `shutdown` or `force_kill`.
//!
//! **No child outlives its handle.** Dropping the last reference kills and reaps
//! the process, and the stderr drain and health tasks are aborted with it;
//! `Command::kill_on_drop` is the final net for a runtime already shutting down.

#[path = "subprocess_config.rs"]
mod subprocess_config;
#[path = "subprocess_io.rs"]
mod subprocess_io;

pub use subprocess_config::SubprocessConfig;
// The exchange layer is a child module, so the helpers it owns are bound
// explicitly; the rest of it is reached through them.
use subprocess_io::{
    force_kill_impl, health_impl, is_alive_impl, kill_child, send_impl, shutdown_impl,
    spawn_health_task, spawn_stderr_task, stream_impl, write_frame,
};

use crate::error;
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnHealth, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::BufReader;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

// The unit tests in `subprocess_tests.rs` reach the module's private defaults and
// helpers through `use super::*`, so they stay importable in the test build only.
#[cfg(test)]
use crate::spawn::frames;
#[cfg(test)]
use crate::spawn::frames::DEFAULT_MAX_FRAME_BYTES;
#[cfg(test)]
use subprocess_config::{secs, MAX_TIMEOUT_SECS};
#[cfg(test)]
use subprocess_io::{error_text, health_payload, invoke_frame, line_preview, STDERR_PREVIEW_CHARS};

/// The subprocess [`AgentSpawner`]: launches [`SubprocessConfig::program`] for
/// each [`SpawnSpec`].
///
/// Cheap to clone and state-free: every spawn starts an independent child.
///
/// The spawner also carries the [`SpawnBackend`] every child it launches reports:
/// [`SpawnBackend::Subprocess`] for [`SubprocessSpawner::new`], or the wrapper's
/// own backend when a backend reuses this protocol under another name (see
/// [`SubprocessSpawner::with_backend`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SubprocessSpawner {
    config: SubprocessConfig,
    backend: SpawnBackend,
}

impl Default for SubprocessSpawner {
    /// A spawner over the default [`SubprocessConfig`], labelled
    /// [`SpawnBackend::Subprocess`].
    fn default() -> Self {
        Self::new(SubprocessConfig::default())
    }
}

impl SubprocessSpawner {
    /// A spawner that launches children according to `config`.
    ///
    /// Children it launches report [`SpawnBackend::Subprocess`].
    #[must_use]
    pub fn new(config: SubprocessConfig) -> Self {
        Self {
            config,
            backend: SpawnBackend::Subprocess,
        }
    }

    /// A spawner that launches children according to `config` and labels them
    /// `backend`.
    ///
    /// For a backend that is the subprocess protocol plus a step of its own — an
    /// external program resolved on `PATH`, say — so its spawner, its handles, and
    /// a runner's monitor detail all name that wrapper instead of
    /// [`SpawnBackend::Subprocess`].
    #[must_use]
    pub fn with_backend(config: SubprocessConfig, backend: SpawnBackend) -> Self {
        Self { config, backend }
    }

    /// The launch configuration every child inherits.
    #[must_use]
    pub fn config(&self) -> &SubprocessConfig {
        &self.config
    }

    /// The backend every child of this spawner reports.
    #[must_use]
    pub fn backend(&self) -> SpawnBackend {
        self.backend
    }

    /// Launches one child and writes its bootstrap frame.
    ///
    /// The handle is concrete-typed so tests and hosts can call
    /// [`SpawnedProcess::health`] / [`SpawnedProcess::pid`]; use
    /// [`SpawnedProcess::into_agent`] to coerce it to a [`SpawnedAgent`]. The
    /// call returns once the process exists and its bootstrap frame is written —
    /// it does not wait for a handshake from the child.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when
    /// [`SubprocessConfig::program`] is empty,
    /// [`openjiuwen_common::error::Error::Transport`] when the process cannot be
    /// launched or its stdin written, and
    /// [`openjiuwen_common::error::Error::Validation`] when the spec cannot be
    /// serialized.
    pub async fn spawn_process(&self, spec: SpawnSpec) -> Result<Arc<SpawnedProcess>> {
        if self.config.program.trim().is_empty() {
            return Err(Error::Config(
                "subprocess spawner requires a program: SubprocessConfig::program is empty"
                    .to_owned(),
            ));
        }

        let mut command = Command::new(&self.config.program);
        command.args(&self.config.args);
        if let Some(cwd) = &self.config.cwd {
            command.current_dir(cwd);
        }
        command.envs(merged_env(&self.config, &spec));
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|error| {
            error::transport(format!(
                "failed to launch subprocess '{}': {error}",
                self.config.program
            ))
        })?;
        let pid = child.id();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().map(BufReader::new);
        let stderr = child.stderr.take();
        let id = match pid {
            Some(pid) => format!("subprocess-{pid}"),
            None => format!("subprocess-{}", uuid_v4()),
        };

        let state = Arc::new(ProcessState {
            id,
            pid,
            backend: self.backend,
            config: self.config.clone(),
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(stdout),
            request: Mutex::new(()),
            desynchronized: AtomicBool::new(false),
            healthy: AtomicBool::new(true),
            health_failures: AtomicU32::new(0),
            health_task: Mutex::new(None),
            stderr_task: Mutex::new(None),
        });

        if let Some(stderr) = stderr {
            *state.stderr_task.lock().await = Some(spawn_stderr_task(pid, stderr));
        }
        if let Err(error) = write_frame(&state, &bootstrap_frame(&spec)?).await {
            // A child that cannot take its bootstrap frame is unusable: do not
            // hand back a handle that is broken from the start.
            let _ = kill_child(&state).await;
            return Err(error);
        }
        spawn_health_task(&state).await;
        Ok(Arc::new(SpawnedProcess { state }))
    }
}

#[async_trait]
impl AgentSpawner for SubprocessSpawner {
    fn backend(&self) -> SpawnBackend {
        self.backend
    }

    /// The launch line, the deadlines, and the environment *names* — never an
    /// environment value, because a child's environment routinely carries a
    /// credential.
    fn describe(&self) -> Value {
        let mut summary = describe_settings(&self.config);
        summary.insert("backend".to_owned(), json!(self.backend.as_str()));
        Value::Object(summary)
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let process = self.spawn_process(spec).await?;
        Ok(process)
    }
}

/// The launch settings of `config` as a JSON object, for
/// [`AgentSpawner::describe`](crate::spawn::AgentSpawner::describe).
///
/// Shared with the wrapper backends so a backend that reuses this protocol
/// reports the settings it inherits: the program, its arguments, the working
/// directory, the environment **names** (never their values), and the configured
/// deadlines and caps.
pub(super) fn describe_settings(config: &SubprocessConfig) -> Map<String, Value> {
    let mut summary = Map::new();
    summary.insert("program".to_owned(), json!(config.program));
    summary.insert("args".to_owned(), json!(config.args));
    summary.insert(
        "cwd".to_owned(),
        config
            .cwd
            .as_ref()
            .map_or(Value::Null, |cwd| json!(cwd.to_string_lossy().into_owned())),
    );
    summary.insert(
        "env_names".to_owned(),
        json!(config.env.keys().cloned().collect::<Vec<_>>()),
    );
    summary.insert(
        "request_timeout_secs".to_owned(),
        json!(config.request_timeout_secs),
    );
    summary.insert(
        "shutdown_timeout_secs".to_owned(),
        json!(config.shutdown_timeout_secs),
    );
    summary.insert(
        "health_check_interval_secs".to_owned(),
        json!(config.health_check_interval_secs),
    );
    summary.insert(
        "health_check_timeout_secs".to_owned(),
        json!(config.health_check_timeout_secs),
    );
    summary.insert(
        "max_health_failures".to_owned(),
        json!(config.max_health_failures),
    );
    summary.insert("max_frame_bytes".to_owned(), json!(config.max_frame_bytes));
    summary
}

/// A live child process speaking the NDJSON protocol.
///
/// Obtained from [`SubprocessSpawner::spawn_process`] (or as a [`SpawnedAgent`]
/// through the [`AgentSpawner`] trait). The handle owns the process: dropping the
/// last reference kills it. Its [`SpawnedAgent::backend`] is the label of the
/// spawner that launched it — [`SpawnBackend::Subprocess`] unless that spawner was
/// built with [`SubprocessSpawner::with_backend`].
pub struct SpawnedProcess {
    state: Arc<ProcessState>,
}

impl SpawnedProcess {
    /// The handle's stable id (`subprocess-<pid>` when the OS reports one).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.state.id
    }

    /// The backend that launched this child, as its spawner labelled it.
    #[must_use]
    pub fn backend(&self) -> SpawnBackend {
        self.state.backend
    }

    /// The child's OS process id, when the platform reported one.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.state.pid
    }

    /// Whether the child is still running.
    ///
    /// A non-blocking probe; lock contention or a probe error reports `true`, so
    /// a caller never declares a healthy child dead.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        is_alive_impl(&self.state)
    }

    /// The last observed health state, refreshed by `health` and the background
    /// task; `true` for a fresh child.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state.healthy.load(Ordering::SeqCst)
    }

    /// Runs one health probe: `health_check` in, `health_check_response` out
    /// within `health_check_timeout_secs`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Timeout`] when the probe deadline
    /// expires, [`openjiuwen_common::error::Error::Transport`] when the transport
    /// fails, and [`openjiuwen_common::error::Error::Validation`] when the handle
    /// is out of sync.
    ///
    /// A probe that hits its deadline leaves the child's response frame unread,
    /// which desynchronizes the handle like an abandoned stream: every later
    /// exchange fails with [`openjiuwen_common::error::Error::Validation`], so the
    /// handle must be dropped and the child respawned.
    pub async fn health(&self) -> Result<bool> {
        health_impl(&self.state).await
    }

    /// Sends one invocation and awaits its terminal output.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Transport`] when the child reports an
    /// `error` frame, exits before replying, or a pipe fails;
    /// [`openjiuwen_common::error::Error::Timeout`] when the request deadline
    /// expires (which also marks the handle out of sync); and
    /// [`openjiuwen_common::error::Error::Validation`] when the handle is out of
    /// sync or the child's output payload is malformed.
    pub async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        send_impl(&self.state, inputs, None).await
    }

    /// Sends one invocation bounded by `timeout`; `None` keeps the configured
    /// `request_timeout_secs`.
    ///
    /// # Errors
    ///
    /// The same as [`SpawnedProcess::send`] — a caller deadline that expires is
    /// [`openjiuwen_common::error::Error::Timeout`] and also marks the handle out
    /// of sync.
    pub async fn send_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> Result<AgentOutput> {
        send_impl(&self.state, inputs, timeout).await
    }

    /// Sends one invocation and yields its [`AgentChunk`] frames.
    ///
    /// The stream ends at the child's `done` (or a terminal `output`) frame, and
    /// yields exactly one error — then ends — on an `error` frame, on expiring
    /// the request deadline, or on losing the child mid-stream. Dropping the
    /// stream before a terminator abandons the exchange and marks the handle out
    /// of sync.
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs, None)
    }

    /// Streams one invocation bounded by `timeout`; `None` keeps the configured
    /// `request_timeout_secs`.
    pub fn stream_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs, timeout)
    }

    /// Asks the child to exit and waits up to `timeout` for it.
    ///
    /// `timeout` of [`Duration::ZERO`] means "the configured
    /// [`SubprocessConfig::with_shutdown_timeout`]"; a non-zero caller timeout
    /// wins over the configuration.
    ///
    /// Returns `true` when it exited after the `shutdown`/`shutdown_ack`
    /// exchange (or was already gone) and `false` when it had to be killed.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Transport`] when the child cannot be
    /// signalled at all.
    pub async fn shutdown(&self, timeout: Duration) -> Result<bool> {
        shutdown_impl(&self.state, timeout).await
    }

    /// Kills the child, bounded by an internal reap timeout. Idempotent.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Transport`] when the kill cannot be
    /// delivered or the child cannot be reaped.
    pub async fn force_kill(&self) -> Result<()> {
        force_kill_impl(&self.state).await
    }

    /// Coerces the handle to a [`SpawnedAgent`] trait object.
    #[must_use]
    pub fn into_agent(self: Arc<Self>) -> Arc<dyn SpawnedAgent> {
        self
    }
}

impl fmt::Debug for SpawnedProcess {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpawnedProcess")
            .field("id", &self.state.id)
            .field("backend", &self.state.backend)
            .field("pid", &self.state.pid)
            .field("alive", &self.is_alive())
            .field("healthy", &self.is_healthy())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SpawnedAgent for SpawnedProcess {
    fn id(&self) -> &str {
        &self.state.id
    }

    fn is_alive(&self) -> bool {
        is_alive_impl(&self.state)
    }

    fn pid(&self) -> Option<u32> {
        self.state.pid
    }

    fn backend(&self) -> SpawnBackend {
        self.state.backend
    }

    /// The same atomic the background health task updates.
    fn is_healthy(&self) -> bool {
        self.state.healthy.load(Ordering::SeqCst)
    }

    /// Carries the handle's own backend; `last_error` stays `None` because a
    /// failed probe is an `Err` from `health`, never a fabricated field.
    fn health_snapshot(&self) -> SpawnHealth {
        SpawnHealth {
            backend: self.state.backend,
            alive: is_alive_impl(&self.state),
            healthy: self.state.healthy.load(Ordering::SeqCst),
            pid: self.state.pid,
            last_error: None,
        }
    }

    /// One probe; see [`SpawnedProcess::health`] for the deadline, desync and
    /// error rules.
    async fn health(&self) -> Result<bool> {
        health_impl(&self.state).await
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        send_impl(&self.state, inputs, None).await
    }

    /// Bounds this one exchange by `timeout`; `None` keeps the configured
    /// `request_timeout_secs`. A timed-out exchange desynchronizes the handle.
    async fn send_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> Result<AgentOutput> {
        send_impl(&self.state, inputs, timeout).await
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs, None)
    }

    /// Bounds this one stream by `timeout`; `None` keeps the configured
    /// `request_timeout_secs`.
    fn stream_with_timeout(
        &self,
        inputs: Value,
        timeout: Option<Duration>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs, timeout)
    }

    async fn shutdown(&self, timeout: Duration) -> Result<bool> {
        shutdown_impl(&self.state, timeout).await
    }

    async fn force_kill(&self) -> Result<()> {
        force_kill_impl(&self.state).await
    }
}

/// Everything one live child owns.
///
/// Held behind an [`Arc`]: the handle owns it, and the stderr drain plus health
/// tasks reference it weakly. Its [`Drop`] is the last-chance kill.
struct ProcessState {
    /// Stable handle id.
    id: String,
    /// OS process id captured at spawn.
    pid: Option<u32>,
    /// Backend label inherited from the spawner that launched the child.
    backend: SpawnBackend,
    /// Launch/policy configuration, cloned from the spawner.
    config: SubprocessConfig,
    /// The OS process; `None` only after [`Drop`] took it for reaping.
    child: Mutex<Option<Child>>,
    /// Write half of the protocol.
    stdin: Mutex<Option<ChildStdin>>,
    /// Read half of the protocol.
    stdout: Mutex<Option<BufReader<ChildStdout>>>,
    /// Serialises exchanges (the protocol has no request correlation id).
    request: Mutex<()>,
    /// Set when an exchange ended without consuming its terminator.
    desynchronized: AtomicBool,
    /// Last observed health state.
    healthy: AtomicBool,
    /// Consecutive failed health probes.
    health_failures: AtomicU32,
    /// The background health task, aborted when the handle stops.
    health_task: Mutex<Option<JoinHandle<()>>>,
    /// The stderr drain task, aborted when the handle stops.
    stderr_task: Mutex<Option<JoinHandle<()>>>,
}

impl ProcessState {
    /// A human description used in error messages.
    fn describe(&self) -> String {
        match self.pid {
            Some(pid) => format!("child process {pid}"),
            None => format!("child process '{}'", self.id),
        }
    }

    /// The request deadline from the configuration.
    fn request_timeout(&self) -> Duration {
        self.config.request_timeout()
    }

    /// The graceful-shutdown deadline from the configuration.
    fn shutdown_timeout(&self) -> Duration {
        self.config.shutdown_timeout()
    }

    /// The health-probe deadline from the configuration.
    fn health_timeout(&self) -> Duration {
        self.config.health_timeout()
    }

    /// Records a probe outcome: a healthy child resets the failure run.
    fn record_health(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
        if healthy {
            self.health_failures.store(0, Ordering::SeqCst);
        }
    }

    /// Aborts the background health task and waits for it to stop.
    async fn stop_health_task(&self) {
        if let Some(handle) = self.health_task.lock().await.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    /// Aborts the stderr drain task and waits for it to stop.
    async fn stop_stderr_task(&self) {
        if let Some(handle) = self.stderr_task.lock().await.take() {
            handle.abort();
            let _ = handle.await;
        }
    }
}

impl Drop for ProcessState {
    fn drop(&mut self) {
        if let Some(handle) = self.health_task.get_mut().take() {
            handle.abort();
        }
        if let Some(handle) = self.stderr_task.get_mut().take() {
            handle.abort();
        }
        let Some(mut child) = self.child.get_mut().take() else {
            return;
        };
        // Already exited (and reaped): nothing to kill.
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        // `Child::drop` does not kill, so arm the kill here. With a runtime in
        // scope a detached task also reaps the zombie; without one (a
        // synchronous drop) the signal is still delivered and `kill_on_drop`
        // covers the reap.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                });
            }
            Err(_) => {
                let _ = child.start_kill();
            }
        }
    }
}

/// The bootstrap frame: the spawn spec, sent before any invocation.
fn bootstrap_frame(spec: &SpawnSpec) -> Result<SpawnMessage> {
    let spawn_spec = serde_json::to_value(spec)
        .map_err(|error| Error::validation(format!("spawn spec is not serializable: {error}")))?;
    Ok(SpawnMessage::new(
        SpawnMessageKind::Input,
        json!({ "spawn_spec": spawn_spec }),
    ))
}

/// The child's environment: the parent's, then the config's, then the spec's
/// (later wins).
fn merged_env(config: &SubprocessConfig, spec: &SpawnSpec) -> BTreeMap<String, String> {
    let mut env = config.env.clone();
    for (key, value) in &spec.envs {
        env.insert(key.clone(), value.clone());
    }
    env
}

#[cfg(test)]
#[path = "subprocess_tests.rs"]
mod tests;
