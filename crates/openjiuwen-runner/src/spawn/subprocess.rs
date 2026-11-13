//! The subprocess backend of the spawn seam: a real child process speaking the
//! NDJSON protocol over its stdio pipes.
//!
//! [`SubprocessSpawner`] launches `program` + `args` with piped
//! stdin/stdout/stderr, writes the bootstrap frame
//! (`{"spawn_spec": <SpawnSpec>}`), and returns a [`SpawnedProcess`] handle.
//! An exchange is one request frame down stdin followed by frames up stdout; the
//! wire lives in [`crate::spawn::protocol`] and the bounded line framing in
//! [`crate::spawn::frames`].
//!
//! **One exchange at a time.** The protocol carries no request correlation id,
//! so a handle serialises exchanges with one `tokio::sync::Mutex` request guard
//! (a fair queue: callers wait instead of interleaving frames). A handle whose
//! stream was dropped before a terminator is marked out of sync and refuses
//! further exchanges with [`openjiuwen_common::error::Error::Validation`],
//! because the next read would consume that abandoned exchange's frames.
//!
//! **Timing** (`request_timeout_secs`, `shutdown_timeout_secs`,
//! `health_check_*`) comes from [`SubprocessConfig`]; the matching [`SpawnSpec`]
//! fields travel to the child in the bootstrap payload, where the child's own
//! runtime may honour them.
//!
//! **No child outlives its handle.** Dropping the last reference kills and reaps
//! the process, and the stderr drain and health tasks are aborted with it;
//! `Command::kill_on_drop` is the final net for a runtime already shutting down.

use crate::error;
use crate::spawn::frames::{self, DEFAULT_MAX_FRAME_BYTES};
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use crate::spawn::{AgentSpawner, SpawnSpec, SpawnedAgent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
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
/// Default deadline for one request/reply exchange, in seconds.
const DEFAULT_REQUEST_TIMEOUT_SECS: f64 = 30.0;
/// Default deadline for a graceful shutdown before a kill, in seconds.
const DEFAULT_SHUTDOWN_TIMEOUT_SECS: f64 = 10.0;
/// Default interval between background health probes, in seconds.
const DEFAULT_HEALTH_CHECK_INTERVAL_SECS: f64 = 5.0;
/// Default deadline for one health probe, in seconds.
const DEFAULT_HEALTH_CHECK_TIMEOUT_SECS: f64 = 5.0;
/// Consecutive failed health probes before the child is killed.
const DEFAULT_MAX_HEALTH_FAILURES: u32 = 3;
/// Upper bound on any configured timeout: one day.
///
/// Clamping keeps `Instant + Duration` (which panics on overflow) infallible and
/// makes nonsense configuration mean "effectively no deadline", not a panic.
const MAX_TIMEOUT_SECS: f64 = 86_400.0;
/// How long a killed child is given to be reaped.
const KILL_REAP_TIMEOUT: Duration = Duration::from_secs(5);
/// Longest prefix of a child's stderr line echoed into the debug log.
const STDERR_PREVIEW_CHARS: usize = 512;

/// How a [`SubprocessSpawner`] launches and supervises its children.
///
/// Serializes (`snake_case`, unknown fields ignored, every field defaulted) so a
/// host can keep it beside the rest of its runner configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubprocessConfig {
    /// Program to launch (an interpreter or an agent binary); empty is an error.
    pub program: String,
    /// Arguments passed to [`SubprocessConfig::program`].
    pub args: Vec<String>,
    /// Working directory; the parent's is inherited when `None`.
    pub cwd: Option<PathBuf>,
    /// Environment overrides added on top of the parent's environment.
    pub env: BTreeMap<String, String>,
    /// Deadline for one request/reply exchange, in seconds.
    pub request_timeout_secs: f64,
    /// Deadline for a graceful shutdown before a kill, in seconds.
    pub shutdown_timeout_secs: f64,
    /// Interval between background health probes, in seconds.
    ///
    /// `0.0` disables the background task; explicit [`SpawnedProcess::health`]
    /// calls still work.
    pub health_check_interval_secs: f64,
    /// Deadline for one health probe, in seconds.
    pub health_check_timeout_secs: f64,
    /// Consecutive failed health probes before the child is killed.
    ///
    /// `0` never kills on health, whatever the failure count.
    pub max_health_failures: u32,
    /// Cap on one frame line read from the child, in bytes.
    pub max_frame_bytes: usize,
}

impl Default for SubprocessConfig {
    fn default() -> Self {
        Self {
            program: String::new(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            shutdown_timeout_secs: DEFAULT_SHUTDOWN_TIMEOUT_SECS,
            health_check_interval_secs: DEFAULT_HEALTH_CHECK_INTERVAL_SECS,
            health_check_timeout_secs: DEFAULT_HEALTH_CHECK_TIMEOUT_SECS,
            max_health_failures: DEFAULT_MAX_HEALTH_FAILURES,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }
}

impl SubprocessConfig {
    /// A configuration for `program`.
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Appends one argument.
    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Replaces the argument list.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the child's working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Adds one environment override (below a [`SpawnSpec`]'s own overrides).
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the deadline for one request/reply exchange.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the deadline for a graceful shutdown before a kill.
    #[must_use]
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the background health-check policy.
    ///
    /// A zero `interval` disables the background task; a zero `max_failures`
    /// disables killing on health.
    #[must_use]
    pub fn with_health_check(
        mut self,
        interval: Duration,
        timeout: Duration,
        max_failures: u32,
    ) -> Self {
        self.health_check_interval_secs = interval.as_secs_f64();
        self.health_check_timeout_secs = timeout.as_secs_f64();
        self.max_health_failures = max_failures;
        self
    }

    /// Sets the cap on one frame line read from the child.
    #[must_use]
    pub fn with_max_frame_bytes(mut self, max_frame_bytes: usize) -> Self {
        self.max_frame_bytes = max_frame_bytes;
        self
    }

    /// The configured request deadline, clamped to a representable range.
    fn request_timeout(&self) -> Duration {
        secs(self.request_timeout_secs)
    }

    /// The configured health-probe deadline, clamped to a representable range.
    fn health_timeout(&self) -> Duration {
        secs(self.health_check_timeout_secs)
    }

    /// The configured background health interval, clamped to a representable
    /// range.
    fn health_interval(&self) -> Duration {
        secs(self.health_check_interval_secs)
    }
}

/// The subprocess [`AgentSpawner`]: launches [`SubprocessConfig::program`] for
/// each [`SpawnSpec`].
///
/// Cheap to clone and state-free: every spawn starts an independent child.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubprocessSpawner {
    config: SubprocessConfig,
}

impl SubprocessSpawner {
    /// A spawner that launches children according to `config`.
    #[must_use]
    pub fn new(config: SubprocessConfig) -> Self {
        Self { config }
    }

    /// The launch configuration every child inherits.
    #[must_use]
    pub fn config(&self) -> &SubprocessConfig {
        &self.config
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
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let process = self.spawn_process(spec).await?;
        Ok(process)
    }
}

/// A live child process speaking the NDJSON protocol.
///
/// Obtained from [`SubprocessSpawner::spawn_process`] (or as a [`SpawnedAgent`]
/// through the [`AgentSpawner`] trait). The handle owns the process: dropping the
/// last reference kills it.
pub struct SpawnedProcess {
    state: Arc<ProcessState>,
}

impl SpawnedProcess {
    /// The handle's stable id (`subprocess-<pid>` when the OS reports one).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.state.id
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

    /// The last observed health state: `true` for a fresh child, refreshed by
    /// [`SpawnedProcess::health`] and the background task.
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
    /// is out of sync after an abandoned stream.
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
        send_impl(&self.state, inputs).await
    }

    /// Sends one invocation and yields its [`AgentChunk`] frames.
    ///
    /// The stream ends at the child's `done` (or a terminal `output`) frame, and
    /// yields exactly one error — then ends — on an `error` frame, on expiring
    /// the request deadline, or on losing the child mid-stream. Dropping the
    /// stream before a terminator abandons the exchange and marks the handle out
    /// of sync.
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs)
    }

    /// Asks the child to exit and waits up to `timeout` for it.
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

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        send_impl(&self.state, inputs).await
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        stream_impl(Arc::clone(&self.state), inputs)
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
async fn send_impl(state: &ProcessState, inputs: Value) -> Result<AgentOutput> {
    let mut guard = state.acquire().await?;
    let timeout = state.request_timeout();
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
fn stream_impl(state: Arc<ProcessState>, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
    let stream: BoxStream<'static, Result<AgentChunk>> = Box::pin(async_stream::try_stream! {
        let mut guard = state.acquire().await?;
        let timeout = state.request_timeout();
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
async fn shutdown_impl(state: &ProcessState, timeout: Duration) -> Result<bool> {
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

/// A configured number of seconds as a [`Duration`], clamped to
/// `[0, MAX_TIMEOUT_SECS]`.
///
/// Negative and NaN values become zero (`f64::max` ignores NaN) and huge values
/// become one day, so `Instant + Duration` cannot overflow.
fn secs(value: f64) -> Duration {
    // `clamp` panics on a NaN bound but returns NaN for a NaN input, and
    // `Duration::from_secs_f64` panics on NaN, so handle it first.
    let clamped = if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, MAX_TIMEOUT_SECS)
    };
    Duration::from_secs_f64(clamped)
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    fn assert_backend<T: SpawnedAgent>() {}
    fn assert_spawner<T: AgentSpawner>() {}

    #[test]
    fn the_backend_types_satisfy_the_seam() {
        assert_backend::<SpawnedProcess>();
        assert_spawner::<SubprocessSpawner>();
    }

    #[test]
    fn config_defaults_match_the_contract() {
        let config = SubprocessConfig::default();
        assert_eq!(config.program, "");
        assert_eq!(config.args, Vec::<String>::new());
        assert_eq!(config.cwd, None);
        assert_eq!(config.env, BTreeMap::new());
        assert!((config.request_timeout_secs - 30.0).abs() < f64::EPSILON);
        assert!((config.shutdown_timeout_secs - 10.0).abs() < f64::EPSILON);
        assert!((config.health_check_interval_secs - 5.0).abs() < f64::EPSILON);
        assert!((config.health_check_timeout_secs - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.max_health_failures, 3);
        assert_eq!(config.max_frame_bytes, DEFAULT_MAX_FRAME_BYTES);
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = SubprocessConfig::new("python")
            .with_args(["-m", "agent"])
            .with_cwd("/tmp/agent")
            .with_env("API_KEY", "secret")
            .with_request_timeout(Duration::from_secs(12))
            .with_shutdown_timeout(Duration::from_secs(4))
            .with_health_check(Duration::from_secs(2), Duration::from_millis(500), 7)
            .with_max_frame_bytes(4096);
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["program"], json!("python"));
        assert_eq!(value["args"], json!(["-m", "agent"]));
        assert_eq!(value["cwd"], json!("/tmp/agent"));
        assert_eq!(value["max_frame_bytes"], json!(4096));
        assert_eq!(
            serde_json::from_value::<SubprocessConfig>(value).unwrap(),
            config
        );
    }

    #[test]
    fn config_tolerates_unknown_fields_and_partial_json() {
        let config: SubprocessConfig = serde_json::from_value(json!({
            "program": "node",
            "unknown_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(config.program, "node");
        assert_eq!(config.request_timeout_secs, 30.0);
        assert_eq!(config.max_frame_bytes, DEFAULT_MAX_FRAME_BYTES);
        let empty: SubprocessConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, SubprocessConfig::default());
    }

    #[test]
    fn builders_chain() {
        let config = SubprocessConfig::new("agent")
            .with_arg("--verbose")
            .with_arg("--json");
        assert_eq!(
            config.args,
            vec!["--verbose".to_owned(), "--json".to_owned()]
        );
        assert_eq!(
            config.with_args(["a", "b"]).args,
            vec!["a".to_owned(), "b".to_owned()]
        );
    }

    #[test]
    fn timeout_seconds_are_clamped() {
        assert_eq!(secs(-1.0), Duration::ZERO);
        assert_eq!(secs(f64::NAN), Duration::ZERO);
        assert_eq!(secs(0.0), Duration::ZERO);
        assert_eq!(secs(1.5), Duration::from_secs_f64(1.5));
        assert_eq!(
            secs(MAX_TIMEOUT_SECS * 4.0),
            Duration::from_secs_f64(MAX_TIMEOUT_SECS)
        );
        assert!(secs(f64::INFINITY) <= Duration::from_secs_f64(MAX_TIMEOUT_SECS));
    }

    #[test]
    fn merged_env_lets_the_spec_win() {
        let config = SubprocessConfig::new("agent")
            .with_env("SHARED", "config")
            .with_env("CONFIG_ONLY", "1");
        let spec = SpawnSpec::default()
            .with_env("SHARED", "spec")
            .with_env("SPEC_ONLY", "2");
        let env = merged_env(&config, &spec);
        assert_eq!(env.get("SHARED").map(String::as_str), Some("spec"));
        assert_eq!(env.get("CONFIG_ONLY").map(String::as_str), Some("1"));
        assert_eq!(env.get("SPEC_ONLY").map(String::as_str), Some("2"));
        assert_eq!(env.len(), 3);
    }

    #[test]
    fn bootstrap_frame_carries_the_spawn_spec() {
        let spec = SpawnSpec::class_agent("pkg.agents", "Helper").with_session_id("s-1");
        let frame = bootstrap_frame(&spec).unwrap();
        assert_eq!(frame.kind, SpawnMessageKind::Input);
        assert_eq!(
            frame.payload["spawn_spec"]["agent_module"],
            json!("pkg.agents")
        );
        assert_eq!(frame.payload["spawn_spec"]["agent_class"], json!("Helper"));
        assert_eq!(frame.payload["spawn_spec"]["session_id"], json!("s-1"));
        let line = frame.to_line().unwrap();
        assert_eq!(SpawnMessage::from_line(&line).unwrap(), frame);
    }

    #[test]
    fn invoke_frame_wraps_the_inputs() {
        let frame = invoke_frame(&json!({"query": "hi"}));
        assert_eq!(frame.kind, SpawnMessageKind::Input);
        assert_eq!(frame.payload, json!({"inputs": {"query": "hi"}}));
    }

    #[test]
    fn error_text_understands_objects_strings_and_null() {
        assert_eq!(error_text(&json!({"message": "boom"})), "boom");
        assert_eq!(error_text(&json!("boom")), "boom");
        assert_eq!(error_text(&Value::Null), "unspecified error");
        assert_eq!(
            error_text(&json!({"code": 3})),
            json!({"code": 3}).to_string()
        );
    }

    #[test]
    fn health_payload_defaults_to_healthy() {
        assert!(health_payload(&Value::Null));
        assert!(health_payload(&json!({"healthy": true})));
        assert!(!health_payload(&json!({"healthy": false})));
        assert!(health_payload(&json!({})));
        assert!(health_payload(&json!(true)));
        assert!(!health_payload(&json!(false)));
    }

    #[test]
    fn line_preview_is_bounded() {
        let long = "x".repeat(STDERR_PREVIEW_CHARS + 10);
        assert_eq!(
            line_preview(&long).chars().count(),
            STDERR_PREVIEW_CHARS + 1
        );
        assert_eq!(line_preview("short"), "short");
    }

    #[tokio::test]
    async fn an_empty_program_is_a_config_error() {
        let spawner = SubprocessSpawner::new(SubprocessConfig::default());
        let error = spawner
            .spawn_process(SpawnSpec::class_agent("pkg", "Agent"))
            .await
            .expect_err("an empty program cannot launch");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error.to_string().contains("SubprocessConfig::program"),
            "unexpected message: {error}"
        );

        // The trait entry point reports the same thing (`Arc<dyn SpawnedAgent>`
        // has no `Debug`, so the Ok arm is matched rather than expect_err'd).
        match SubprocessSpawner::new(SubprocessConfig::new("   "))
            .spawn(SpawnSpec::default())
            .await
        {
            Ok(_) => panic!("a blank program cannot launch"),
            Err(error) => assert_eq!(error.kind(), ErrorKind::Config),
        }
    }

    #[test]
    fn the_spawner_exposes_its_config_including_the_frame_cap() {
        let config = SubprocessConfig::new("agent").with_max_frame_bytes(1024);
        let spawner = SubprocessSpawner::new(config.clone());
        assert_eq!(spawner.config(), &config);
        assert_eq!(spawner.config().max_frame_bytes, 1024);
        assert_eq!(
            SubprocessSpawner::default().config().max_frame_bytes,
            frames::DEFAULT_MAX_FRAME_BYTES
        );
    }
}
