//! The exchange and supervision layer of the subprocess backend: one
//! request/reply conversation at a time over the child's stdio pipes, plus the
//! health loop, the stderr drain, and the shutdown/kill escalation.
//!
//! An exchange is one request frame down stdin followed by frames up stdout; the
//! wire lives in [`crate::spawn::protocol`] and the bounded line framing in
//! [`crate::spawn::frames`]. Every helper here is reached through the handle in
//! [`crate::spawn::subprocess::SpawnedProcess`], which owns the state these
//! functions mutate.
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
//! `health_check_*`) comes from
//! [`crate::spawn::subprocess::SubprocessConfig`]; the matching
//! [`crate::spawn::SpawnSpec`] fields travel to the child in the bootstrap
//! payload, where the child's own runtime may honour them. A caller can bound one
//! exchange with
//! [`crate::spawn::subprocess::SpawnedProcess::send_with_timeout`] /
//! [`crate::spawn::subprocess::SpawnedProcess::stream_with_timeout`] (`None`
//! keeps the configured request deadline), and
//! [`crate::spawn::subprocess::SpawnedProcess::shutdown`] treats a zero caller
//! timeout as "the configured `shutdown_timeout_secs`", so the configuration is
//! never silently ignored.
//!
//! **Background tasks hold a [`Weak`] and are owned by the state.** The health
//! loop and the stderr drain upgrade the state per tick and are aborted when the
//! handle's [`crate::spawn::subprocess::SpawnedProcess`] state is dropped, so
//! neither can outlive or resurrect a child.

use super::{HealthTask, ProcessState};
use crate::error;
use crate::spawn::frames;
use crate::spawn::protocol::{SpawnMessage, SpawnMessageKind};
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::ChildStderr;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, trace, warn};

/// `tracing` target for the spawn seam.
const TARGET: &str = "openjiuwen_runner::spawn";

/// How long a killed child is given to be reaped.
const KILL_REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a background health loop is given to stop cooperatively before it is
/// aborted.
///
/// A loop waiting for the request lock behind a caller's exchange can take up to
/// the configured `request_timeout_secs`, which is too long to make a re-tune
/// wait: past this bound the loop is aborted, which is the one case where a
/// re-tune can leave a stray frame behind.
const HEALTH_TASK_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest prefix of a child's stderr line echoed into the debug log.
pub(super) const STDERR_PREVIEW_CHARS: usize = 512;

/// Takes the request lock, refusing to start when the handle is out of sync.
async fn acquire(state: &ProcessState) -> Result<RequestGuard<'_>> {
    let request = state.request.lock().await;
    if state.desynchronized.load(Ordering::SeqCst) {
        return Err(out_of_sync(state));
    }
    Ok(RequestGuard {
        state,
        _request: request,
        finished: false,
    })
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
pub(super) async fn send_impl(
    state: &ProcessState,
    inputs: Value,
    timeout: Option<Duration>,
) -> Result<AgentOutput> {
    let mut guard = acquire(state).await?;
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
pub(super) fn stream_impl(
    state: Arc<ProcessState>,
    inputs: Value,
    timeout: Option<Duration>,
) -> BoxStream<'static, Result<AgentChunk>> {
    let stream: BoxStream<'static, Result<AgentChunk>> = Box::pin(async_stream::try_stream! {
        let mut guard = acquire(&state).await?;
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
pub(super) async fn health_impl(state: &ProcessState) -> Result<bool> {
    if !is_alive_impl(state) {
        state.record_health(false);
        return Ok(false);
    }
    let mut guard = acquire(state).await?;
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
/// [`crate::spawn::subprocess::SubprocessConfig::with_shutdown_timeout`]"; a
/// non-zero caller deadline wins over it.
pub(super) async fn shutdown_impl(state: &ProcessState, timeout: Duration) -> Result<bool> {
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
pub(super) async fn force_kill_impl(state: &ProcessState) -> Result<()> {
    state.stop_health_task().await;
    let result = kill_child(state).await;
    state.stop_stderr_task().await;
    result
}

/// Signals the child to die and reaps it, tolerating an already-exited child.
pub(super) async fn kill_child(state: &ProcessState) -> Result<()> {
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
pub(super) fn is_alive_impl(state: &ProcessState) -> bool {
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
pub(super) async fn write_frame(state: &ProcessState, frame: &SpawnMessage) -> Result<()> {
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
pub(super) async fn read_frame(state: &ProcessState) -> Result<Option<SpawnMessage>> {
    let mut stdout = state.stdout.lock().await;
    let Some(reader) = stdout.as_mut() else {
        return Ok(None);
    };
    frames::read_frame(reader, state.config.max_frame_bytes).await
}

/// Drains the child's stderr into the debug log so a full pipe can never block
/// it.
pub(super) fn spawn_stderr_task(pid: Option<u32>, stderr: ChildStderr) -> JoinHandle<()> {
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

/// Starts the background health loop, if the handle's live interval enables it.
///
/// The interval is read from the handle
/// ([`ProcessState::health_interval`]), not from
/// [`crate::spawn::subprocess::SubprocessConfig`], so a restart from
/// [`crate::spawn::subprocess::SpawnedProcess::set_health_interval`] takes
/// effect. The task holds only a [`Weak`] reference, so it never keeps the handle
/// alive, and it exits when the child dies or a probe finds it unusable.
pub(super) async fn spawn_health_task(state: &Arc<ProcessState>) {
    let interval = state.health_interval();
    if interval.is_zero() {
        return;
    }
    let stop = CancellationToken::new();
    let task_stop = stop.clone();
    let weak = Arc::downgrade(state);
    let handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick completes immediately; consume it so the interval is
        // the delay before the first probe.
        ticker.tick().await;
        loop {
            tokio::select! {
                // The stop signal only ever cancels the wait *between* probes, so
                // a re-tune never abandons an exchange half-way.
                () = task_stop.cancelled() => return,
                _ = ticker.tick() => {}
            }
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
    *state.health_task.lock().await = Some(HealthTask { handle, stop });
}

/// Re-tunes the background health interval of a live child.
///
/// The new interval is stored first, so the restarted loop reads it; the previous
/// loop is then stopped cooperatively and replaced. `Duration::ZERO` leaves no
/// background task running (explicit `health` probes still work); a non-zero value
/// starts one.
///
/// The child is not touched, so a re-tune never desynchronises the handle, never
/// clears the `healthy` flag and never resets the consecutive-failure count: a
/// `health_fail` child that already failed twice keeps that count.
pub(super) async fn set_health_interval_impl(
    state: &Arc<ProcessState>,
    interval: Duration,
) -> Result<()> {
    // One re-tune at a time, so two callers cannot leave two loops behind.
    let _tuning = state.health_tuning.lock().await;
    if !is_alive_impl(state) {
        return Err(stopped(state));
    }
    state.store_health_interval(interval);
    stop_health_task_gracefully(state).await;
    spawn_health_task(state).await;
    Ok(())
}

/// Stops the background health loop without abandoning an exchange.
///
/// The loop's stop signal cancels only its wait between probes: a probe already in
/// flight runs to completion (it is bounded by `health_check_timeout_secs`), so its
/// reply is consumed and the handle stays in sync. A loop that cannot stop within
/// [`HEALTH_TASK_STOP_TIMEOUT`] — it is queued behind a caller's exchange for the
/// request lock — is aborted as a last resort.
async fn stop_health_task_gracefully(state: &ProcessState) {
    let Some(task) = state.health_task.lock().await.take() else {
        return;
    };
    task.stop.cancel();
    let mut handle = task.handle;
    if tokio::time::timeout(HEALTH_TASK_STOP_TIMEOUT, &mut handle)
        .await
        .is_err()
    {
        debug!(
            target: TARGET,
            pid = ?state.pid,
            "the background health loop did not stop in time; aborting it"
        );
        handle.abort();
        let _ = handle.await;
    }
}

/// The invocation frame: one request's inputs.
pub(super) fn invoke_frame(inputs: &Value) -> SpawnMessage {
    SpawnMessage::new(SpawnMessageKind::Input, json!({ "inputs": inputs }))
}

/// The child's message from an `error` frame payload: a `{"message": ...}`
/// object, a bare string, or the whole payload as a fallback.
pub(super) fn error_text(payload: &Value) -> String {
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
pub(super) fn health_payload(payload: &Value) -> bool {
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

/// The error for a re-tune of a handle whose child already stopped.
///
/// A stopped handle has nothing left to supervise: starting a loop would probe a
/// dead child and, for a killed one, only inflate the failure count.
fn stopped(state: &ProcessState) -> Error {
    Error::validation(format!(
        "{} has stopped; the background health interval can only be re-tuned while the child is alive",
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
pub(super) fn line_preview(text: &str) -> String {
    let mut shown: String = text.chars().take(STDERR_PREVIEW_CHARS).collect();
    if text.chars().count() > STDERR_PREVIEW_CHARS {
        shown.push('…');
    }
    shown
}
