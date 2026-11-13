//! The subprocess backend of the spawn seam: a real child process speaking the
//! NDJSON protocol over its stdio pipes.
//!
//! This module owns the spawner, the child handle, and the supervision; the serde
//! configuration itself lives in the private `subprocess_config` sibling and is
//! re-exported here as [`SubprocessConfig`].
//!
//! [`SubprocessSpawner`] launches `program` + `args` with piped
//! stdin/stdout/stderr, writes the bootstrap frame
//! (`{"spawn_spec": <SpawnSpec>}`), and returns a [`SpawnedProcess`] handle.
//! Each child inherits its spawner's [`SpawnBackend`] label — `subprocess` for
//! [`SubprocessSpawner::new`], the wrapper's own name for a spawner built with
//! [`SubprocessSpawner::with_backend`] — so a handle, its spawner, and a runner's
//! monitor detail agree on which backend produced the child.
//! An exchange is one request frame down stdin followed by frames up stdout; the
//! wire lives in [`crate::spawn::protocol`] and the bounded line framing in
//! [`crate::spawn::frames`].
//!
//! **One exchange at a time.** The protocol carries no request correlation id,
//! so a handle serialises exchanges with one `tokio::sync::Mutex` request guard
//! (a fair queue: callers wait instead of interleaving frames). A handle whose
//! stream was dropped before a terminator is marked out of sync and refuses
//! further exchanges with [`openjiuwen_common::error::Error::Validation`],
//! because the next read would consume that abandoned exchange's frames. For the
//! same reason a child implements no task cancellation (there is no task handle
//! to address): stop one with `shutdown` or `force_kill`.
//!
//! **Timing** (`request_timeout_secs`, `shutdown_timeout_secs`,
//! `health_check_*`) comes from [`SubprocessConfig`]; the matching [`SpawnSpec`]
//! fields travel to the child in the bootstrap payload, where the child's own
//! runtime may honour them. A caller can bound one exchange with
//! [`SpawnedProcess::send_with_timeout`] /
//! [`SpawnedProcess::stream_with_timeout`] (`None` keeps the configured request
//! deadline), and [`SpawnedProcess::shutdown`] treats a zero caller timeout as
//! "the configured `shutdown_timeout_secs`", so the configuration is never
//! silently ignored.
//!
//! **No child outlives its handle.** Dropping the last reference kills and reaps
//! the process, and the stderr drain and health tasks are aborted with it;
//! `Command::kill_on_drop` is the final net for a runtime already shutting down.

#[path = "subprocess_config.rs"]
mod subprocess_config;

pub use subprocess_config::SubprocessConfig;
// The unit tests in `subprocess_tests.rs` reach the module's private defaults and
// helpers through `use super::*`, so they stay importable in the test build only.
#[cfg(test)]
use crate::spawn::frames::DEFAULT_MAX_FRAME_BYTES;
#[cfg(test)]
use subprocess_config::{secs, MAX_TIMEOUT_SECS};

use crate::error;
use crate::spawn::frames;
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnHealth, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{debug, trace, warn};

/// `tracing` target for the spawn seam.
const TARGET: &str = "openjiuwen_runner::spawn";

/// How long a killed child is given to be reaped.
const KILL_REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest prefix of a child's stderr line echoed into the debug log.
const STDERR_PREVIEW_CHARS: usize = 512;

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

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let process = self.spawn_process(spec).await?;
        Ok(process)
    }
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

    /// Takes the request lock, refusing to start when the handle is out of sync.
    async fn acquire(&self) -> Result<RequestGuard<'_>> {
        let request = self.request.lock().await;
        if self.desynchronized.load(Ordering::SeqCst) {
            return Err(out_of_sync(self));
        }
        Ok(RequestGuard {
            state: self,
            _request: request,
            finished: false,
        })
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

/// Holds the exchange lock for one request/reply conversation.
///
/// Dropping it without [`RequestGuard::finish`] means the exchange ended without
/// consuming its terminator (its stream was abandoned), so the handle is marked
/// out of sync *before* the lock is released: the next caller cannot start on a
/// half-consumed stream.
struct RequestGuard<'a> {
    state: &'a ProcessState,
    /// The held exchange lock; the field exists for its guard.
    _request: tokio::sync::MutexGuard<'a, ()>,
    finished: bool,
}

impl RequestGuard<'_> {
    /// Marks the exchange as cleanly terminated.
    fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.state.desynchronized.store(true, Ordering::SeqCst);
        }
    }
}

/// One step of a streaming read.
enum StreamStep {
    /// A chunk to yield.
    Chunk(AgentChunk),
    /// A terminator (`done`, or the terminal `output` that only `send` reports).
    End,
    /// The child reported an error: yield it and end.
    Failed(Error),
    /// A frame that does not belong to this exchange.
    Ignore,
}

/// Sends one invocation and awaits its terminal output.
///
/// `timeout` bounds this exchange; `None` falls back to the configured
/// `request_timeout_secs`.
async fn send_impl(
    state: &ProcessState,
    inputs: Value,
    timeout: Option<Duration>,
) -> Result<AgentOutput> {
    let mut guard = state.acquire().await?;
    let timeout = timeout.unwrap_or_else(|| state.request_timeout());
    match tokio::time::timeout(timeout, send_exchange(state, inputs)).await {
        Ok(Ok(output)) => {
            guard.finish();
            Ok(output)
        }
        Ok(Err(error)) => {
            guard.finish();
            Err(error)
        }
        // A timed-out exchange leaves unread frames behind, so the guard is
        // dropped unfinished and marks the handle out of sync.
        Err(_) => Err(error::run_timeout("subprocess request", timeout)),
    }
}

/// The request/reply body of [`send_impl`].
async fn send_exchange(state: &ProcessState, inputs: Value) -> Result<AgentOutput> {
    write_frame(state, &invoke_frame(&inputs)).await?;
    loop {
        let Some(frame) = read_frame(state).await? else {
            return Err(gone(state, "before replying"));
        };
        match frame.kind {
            SpawnMessageKind::Output => return AgentOutput::from_value(frame.payload),
            SpawnMessageKind::Done => {
                if frame.payload.is_null() {
                    return Err(error::transport(format!(
                        "{} completed without an output",
                        state.describe()
                    )));
                }
                return AgentOutput::from_value(frame.payload);
            }
            SpawnMessageKind::Error => return Err(child_error(state, &frame.payload)),
            // A stream frame before the terminal output is noise for `send`
            // (Python's reader tolerates stray frames too).
            other => trace!(
                target: TARGET,
                kind = ?other,
                "ignoring an unexpected frame while awaiting a subprocess reply"
            ),
        }
    }
}

/// Streams one invocation's frames.
///
/// `timeout` bounds this exchange; `None` falls back to the configured
/// `request_timeout_secs`.
fn stream_impl(
    state: Arc<ProcessState>,
    inputs: Value,
    timeout: Option<Duration>,
) -> BoxStream<'static, Result<AgentChunk>> {
    let stream: BoxStream<'static, Result<AgentChunk>> = Box::pin(async_stream::try_stream! {
        let mut guard = state.acquire().await?;
        let timeout = timeout.unwrap_or_else(|| state.request_timeout());
        let deadline = Instant::now() + timeout;
        if let Err(error) = write_frame(&state, &invoke_frame(&inputs)).await {
            guard.finish();
            Err(error)?;
        }
        loop {
            match read_stream_step(&state, deadline, timeout).await {
                Ok(StreamStep::Chunk(chunk)) => yield chunk,
                Ok(StreamStep::Ignore) => {}
                Ok(StreamStep::End) => {
                    guard.finish();
                    break;
                }
                Ok(StreamStep::Failed(error)) => {
                    guard.finish();
                    Err(error)?;
                }
                // A timeout or a lost child leaves unread frames behind: the
                // guard drops unfinished and marks the handle out of sync.
                Err(error) => Err(error)?,
            }
        }
    });
    stream
}

/// Reads one frame of a streaming exchange.
async fn read_stream_step(
    state: &ProcessState,
    deadline: Instant,
    timeout: Duration,
) -> Result<StreamStep> {
    let frame = match tokio::time::timeout_at(deadline, read_frame(state)).await {
        Ok(Ok(Some(frame))) => frame,
        Ok(Ok(None)) => return Err(gone(state, "mid-stream")),
        Ok(Err(error)) => return Err(error),
        Err(_) => return Err(error::run_timeout("subprocess stream", timeout)),
    };
    match frame.kind {
        SpawnMessageKind::StreamChunk => {
            AgentChunk::from_value(frame.payload).map(StreamStep::Chunk)
        }
        SpawnMessageKind::Done | SpawnMessageKind::Output => Ok(StreamStep::End),
        SpawnMessageKind::Error => Ok(StreamStep::Failed(child_error(state, &frame.payload))),
        other => {
            trace!(
                target: TARGET,
                kind = ?other,
                "ignoring an unexpected frame in a subprocess stream"
            );
            Ok(StreamStep::Ignore)
        }
    }
}

/// Runs one health probe.
async fn health_impl(state: &ProcessState) -> Result<bool> {
    if !is_alive_impl(state) {
        state.record_health(false);
        return Ok(false);
    }
    let mut guard = state.acquire().await?;
    let timeout = state.health_timeout();
    let probe = async {
        write_frame(
            state,
            &SpawnMessage::new(SpawnMessageKind::HealthCheck, Value::Null),
        )
        .await?;
        loop {
            let Some(frame) = read_frame(state).await? else {
                // EOF: the child is gone, which is not healthy.
                return Ok(false);
            };
            if frame.kind == SpawnMessageKind::HealthCheckResponse {
                return Ok(health_payload(&frame.payload));
            }
            trace!(
                target: TARGET,
                kind = ?frame.kind,
                "ignoring an unexpected frame while awaiting a health response"
            );
        }
    };
    match tokio::time::timeout(timeout, probe).await {
        Ok(Ok(healthy)) => {
            guard.finish();
            state.record_health(healthy);
            Ok(healthy)
        }
        Ok(Err(error)) => {
            guard.finish();
            state.record_health(false);
            Err(error)
        }
        Err(_) => {
            state.record_health(false);
            Err(error::run_timeout("subprocess health check", timeout))
        }
    }
}

/// Shuts the child down gracefully, escalating to a kill.
///
/// A zero `timeout` means "the configured
/// [`SubprocessConfig::with_shutdown_timeout`]"; a non-zero caller deadline wins
/// over it.
async fn shutdown_impl(state: &ProcessState, timeout: Duration) -> Result<bool> {
    let timeout = if timeout.is_zero() {
        state.shutdown_timeout()
    } else {
        timeout
    };
    state.stop_health_task().await;
    if !is_alive_impl(state) {
        state.record_health(false);
        state.stop_stderr_task().await;
        return Ok(true);
    }

    // Recovery must work on a desynchronized handle, so the lock is taken
    // directly instead of through `acquire`.
    let graceful = {
        let _request = state.request.lock().await;
        let exchange = async {
            write_frame(
                state,
                &SpawnMessage::new(SpawnMessageKind::Shutdown, Value::Null),
            )
            .await?;
            // Closing stdin lets a child that reads to EOF leave on its own.
            drop(state.stdin.lock().await.take());
            loop {
                match read_frame(state).await {
                    Ok(Some(frame)) if frame.kind == SpawnMessageKind::ShutdownAck => break,
                    Ok(Some(frame)) => trace!(
                        target: TARGET,
                        kind = ?frame.kind,
                        "ignoring an unexpected frame while awaiting a shutdown ack"
                    ),
                    // EOF: it exited before acknowledging, which is just as good.
                    Ok(None) => break,
                    Err(error) => return Err(error),
                }
            }
            let mut child = state.child.lock().await;
            match child.as_mut() {
                Some(child) => child.wait().await.map(|_status| ()).map_err(|error| {
                    error::transport(format!(
                        "waiting for {} to exit failed: {error}",
                        state.describe()
                    ))
                }),
                None => Ok(()),
            }
        };
        tokio::time::timeout(timeout, exchange).await
    };

    match graceful {
        Ok(Ok(())) => {
            state.record_health(false);
            state.stop_stderr_task().await;
            Ok(true)
        }
        Ok(Err(error)) => {
            debug!(
                target: TARGET,
                %error,
                "graceful shutdown failed; killing the child instead"
            );
            kill_child(state).await?;
            state.stop_stderr_task().await;
            Ok(false)
        }
        Err(_) => {
            warn!(
                target: TARGET,
                timeout_ms = timeout.as_millis(),
                "child did not shut down within the deadline; killing it"
            );
            kill_child(state).await?;
            state.stop_stderr_task().await;
            Ok(false)
        }
    }
}

/// Kills the child unconditionally.
async fn force_kill_impl(state: &ProcessState) -> Result<()> {
    state.stop_health_task().await;
    let result = kill_child(state).await;
    state.stop_stderr_task().await;
    result
}

/// Signals the child to die and reaps it, tolerating an already-exited child.
async fn kill_child(state: &ProcessState) -> Result<()> {
    state.record_health(false);
    let mut guard = state.child.lock().await;
    let Some(child) = guard.as_mut() else {
        return Ok(());
    };
    if matches!(child.try_wait(), Ok(Some(_))) {
        return Ok(());
    }
    match child.start_kill() {
        Ok(()) => {}
        // The process exited between the check and the signal.
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => return Ok(()),
        Err(error) => {
            return Err(error::transport(format!(
                "killing {} failed: {error}",
                state.describe()
            )))
        }
    }
    match tokio::time::timeout(KILL_REAP_TIMEOUT, child.wait()).await {
        Ok(Ok(_status)) => Ok(()),
        Ok(Err(error)) => Err(error::transport(format!(
            "reaping {} failed: {error}",
            state.describe()
        ))),
        Err(_) => Err(error::transport(format!(
            "{} did not exit within {}ms of being killed",
            state.describe(),
            KILL_REAP_TIMEOUT.as_millis()
        ))),
    }
}

/// Whether the child is still running.
fn is_alive_impl(state: &ProcessState) -> bool {
    let Ok(mut guard) = state.child.try_lock() else {
        // Contended: unknown is treated as alive.
        return true;
    };
    match guard.as_mut() {
        Some(child) => !matches!(child.try_wait(), Ok(Some(_))),
        // Taken for reaping by `Drop`: definitely not alive.
        None => false,
    }
}

/// Writes one frame to the child's stdin.
async fn write_frame(state: &ProcessState, frame: &SpawnMessage) -> Result<()> {
    let bytes = frames::frame_bytes(frame)?;
    let mut stdin = state.stdin.lock().await;
    let Some(pipe) = stdin.as_mut() else {
        return Err(error::transport(format!(
            "{} stdin is closed",
            state.describe()
        )));
    };
    pipe.write_all(&bytes).await.map_err(|error| {
        error::transport(format!("writing to {} failed: {error}", state.describe()))
    })?;
    pipe.flush().await.map_err(|error| {
        error::transport(format!(
            "flushing {} stdin failed: {error}",
            state.describe()
        ))
    })
}

/// Reads one frame from the child's stdout.
async fn read_frame(state: &ProcessState) -> Result<Option<SpawnMessage>> {
    let mut stdout = state.stdout.lock().await;
    let Some(reader) = stdout.as_mut() else {
        return Ok(None);
    };
    frames::read_frame(reader, state.config.max_frame_bytes).await
}

/// Drains the child's stderr into the debug log so a full pipe can never block
/// it.
fn spawn_stderr_task(pid: Option<u32>, stderr: ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => debug!(
                    target: TARGET,
                    pid = ?pid,
                    line = %line_preview(&line),
                    "child stderr"
                ),
                Ok(None) => return,
                Err(error) => {
                    trace!(target: TARGET, pid = ?pid, %error, "child stderr could not be read");
                    return;
                }
            }
        }
    })
}

/// Starts the background health loop, if the configuration enables it.
///
/// The task holds only a [`Weak`] reference, so it never keeps the handle alive,
/// and it exits when the child dies or a probe finds it unusable.
async fn spawn_health_task(state: &Arc<ProcessState>) {
    let interval = state.config.health_interval();
    if interval.is_zero() {
        return;
    }
    let weak = Arc::downgrade(state);
    let handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick completes immediately; consume it so the interval is
        // the delay before the first probe.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let Some(state) = Weak::upgrade(&weak) else {
                return;
            };
            if !is_alive_impl(&state) {
                state.record_health(false);
                return;
            }
            let healthy = match health_impl(&state).await {
                Ok(healthy) => healthy,
                Err(error) => {
                    debug!(target: TARGET, pid = ?state.pid, %error, "health probe failed");
                    false
                }
            };
            if healthy {
                continue;
            }
            let failures = state.health_failures.fetch_add(1, Ordering::SeqCst) + 1;
            let limit = state.config.max_health_failures;
            if limit > 0 && failures >= limit {
                warn!(
                    target: TARGET,
                    pid = ?state.pid,
                    failures,
                    "child failed {failures} health checks in a row; killing it"
                );
                // The low-level kill is used on purpose: this task owns the
                // health handle that `stop_health_task` would join.
                let _ = kill_child(&state).await;
                return;
            }
        }
    });
    *state.health_task.lock().await = Some(handle);
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

/// The invocation frame: one request's inputs.
fn invoke_frame(inputs: &Value) -> SpawnMessage {
    SpawnMessage::new(SpawnMessageKind::Input, json!({ "inputs": inputs }))
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

/// The child's message from an `error` frame payload: a `{"message": ...}`
/// object, a bare string, or the whole payload as a fallback.
fn error_text(payload: &Value) -> String {
    match payload {
        Value::String(message) => message.clone(),
        Value::Object(map) => map
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| payload.to_string(), str::to_owned),
        Value::Null => "unspecified error".to_owned(),
        other => other.to_string(),
    }
}

/// Whether a `health_check_response` payload reports a healthy child.
///
/// The frame is itself proof of liveness, so `null`, a bare `true`, or a payload
/// without a boolean `healthy` field all mean healthy.
fn health_payload(payload: &Value) -> bool {
    match payload {
        Value::Bool(healthy) => *healthy,
        Value::Object(map) => map.get("healthy").and_then(Value::as_bool).unwrap_or(true),
        _ => true,
    }
}

/// The out-of-sync error: the next read would consume stale frames.
fn out_of_sync(state: &ProcessState) -> Error {
    Error::validation(format!(
        "{} is out of sync after an abandoned stream; respawn the agent",
        state.describe()
    ))
}

/// The transport error for a child that disappeared.
fn gone(state: &ProcessState, when: &str) -> Error {
    error::transport(format!("{} exited {when}", state.describe()))
}

/// The transport error for an `error` frame.
fn child_error(state: &ProcessState, payload: &Value) -> Error {
    error::transport(format!(
        "{} reported an error: {}",
        state.describe(),
        error_text(payload)
    ))
}

/// A short, char-boundary-safe prefix of `text` for the debug log.
fn line_preview(text: &str) -> String {
    let mut shown: String = text.chars().take(STDERR_PREVIEW_CHARS).collect();
    if text.chars().count() > STDERR_PREVIEW_CHARS {
        shown.push('…');
    }
    shown
}

#[cfg(test)]
#[path = "subprocess_tests.rs"]
mod tests;
