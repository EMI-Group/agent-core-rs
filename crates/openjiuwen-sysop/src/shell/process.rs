//! Shared child-process engine used by the shell backend and the code backend.
//!
//! It owns the three behaviors every command or snippet run needs: a hardened spawn (no
//! stdin, piped output, own process group), bounded output capture, and timeout or
//! cancellation handling that kills the child **and its process group** exactly once and
//! always reaps the direct child.
//!
//! Process-group management: [`spawn_child`] puts every child into a fresh process group
//! whose id is the child's pid (Python's `start_new_session=True`, minus the controlling
//! terminal detach — `setsid()` needs a `pre_exec` hook, which the crate-level
//! `forbid(unsafe_code)` rules out). A kill therefore reaches the shell's descendants
//! through [`kill_group`], which is what Python's `os.killpg` does in
//! `shell_process_registry.py`/`utils.py`.

use std::future::Future;
use std::pin::Pin;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::constants;
use crate::shell::types::StreamKind;

/// Marker appended to backgrounded stderr that exceeded the retention budget.
const STDERR_TRUNCATION_MARKER: &str = "\n[stderr truncated]";
/// How long the early-exit path waits for the stderr reader to finish.
const STDERR_JOIN_GRACE: Duration = Duration::from_millis(100);
/// Bytes read from a pipe per `read` call while capturing.
const READ_CHUNK_SIZE: usize = 65_536;
/// How long to wait for a pipe reader once the child has been reaped.
const READER_GRACE: Duration = Duration::from_secs(5);
/// Capacity of the streamed-event channel.
pub(crate) const EVENT_CHANNEL_CAPACITY: usize = 64;
/// Program that switches a POSIX child to line buffering (Python's Linux wrapper).
const LINE_BUFFERING_WRAPPER: &str = "stdbuf";
/// Flags passed to [`LINE_BUFFERING_WRAPPER`]: line-buffered stdout, line-buffered stderr.
const LINE_BUFFERING_FLAGS: [&str; 2] = ["-oL", "-eL"];

/// A boxed, sendable future used for the deadline and cancellation branches.
type EngineFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Text captured from one child stream.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct CapturedText {
    pub(crate) text: String,
    pub(crate) truncated: bool,
}

/// Limits and cancellation tokens applied to one child run.
#[derive(Debug, Clone, Default)]
pub(crate) struct RunOptions {
    pub(crate) timeout_secs: u64,
    pub(crate) max_output_bytes: u64,
    pub(crate) cancellation: Vec<CancellationToken>,
}

/// The outcome of a captured child run (`exit_code` is `-1` when killed by a signal).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RunOutcome {
    pub(crate) exit_code: i32,
    pub(crate) stdout: CapturedText,
    pub(crate) stderr: CapturedText,
    pub(crate) timed_out: bool,
    pub(crate) cancelled: bool,
    pub(crate) duration_ms: u64,
}

/// One event of a streamed child run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EngineEvent {
    /// Output arrived on a pipe.
    Data { kind: StreamKind, text: String },
    /// The child exited; `code` is its exit status.
    Exit { code: i32 },
    /// The timeout expired and the child was killed.
    TimedOut { code: i32 },
    /// A cancellation token fired and the child was killed.
    Cancelled { code: i32 },
}

/// Spawns `command` with the hardened child configuration: stdin closed, both output
/// pipes captured, killed on drop, and (on Unix) in its own process group so the
/// descendants of a shell die with it.
pub(crate) fn spawn_child(mut command: Command) -> std::io::Result<Child> {
    command
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    command.spawn()
}

/// Replaces the child environment with `pairs` (already policy-scoped by the caller).
pub(crate) fn apply_environment(command: &mut Command, pairs: &[(String, String)]) {
    let envs = pairs
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()));
    command.env_clear().envs(envs);
}

/// Kills the process group led by `pid` with `SIGKILL`, best effort.
///
/// Returns `true` when the signal was delivered. A group that no longer exists (`ESRCH`)
/// is not an error: the child may have exited between the timeout firing and the kill.
/// Non-positive ids are rejected outright — `killpg(0)` would signal the caller's own
/// group — and non-Unix platforms have no process groups, so they return `false` and the
/// caller's `Child::start_kill` fallback still runs.
#[cfg(unix)]
pub(crate) fn kill_group(pid: u32) -> bool {
    let Ok(raw) = i32::try_from(pid) else {
        return false;
    };
    if raw <= 0 {
        return false;
    }
    match nix::sys::signal::killpg(
        nix::unistd::Pid::from_raw(raw),
        nix::sys::signal::Signal::SIGKILL,
    ) {
        Ok(()) => true,
        // The whole group is already gone: nothing left to kill.
        Err(nix::errno::Errno::ESRCH) => false,
        Err(error) => {
            tracing::debug!(pid, %error, "process-group kill failed");
            false
        }
    }
}

/// Kills the process group led by `pid`, best effort; always `false` without POSIX groups.
#[cfg(not(unix))]
pub(crate) fn kill_group(pid: u32) -> bool {
    tracing::debug!(pid, "process groups are not available on this platform");
    false
}

/// The process group of one child, killed at most once and never after the reap.
///
/// The guard exists because a pid can be recycled: signalling the group of a reaped child
/// could hit an unrelated process. Callers therefore report whether the child was already
/// reaped when the run ended, and the guard's kill-once flag makes a second timeout or
/// cancellation branch a no-op.
#[derive(Debug)]
pub(crate) struct ProcessGroup {
    /// Group id (= the child pid), `None` once the child has been reaped.
    pid: Option<u32>,
    /// Set by the first successful or attempted group kill.
    killed: bool,
}

impl ProcessGroup {
    /// Records the process group of `pid` (`None` when the child was already reaped).
    pub(crate) fn new(pid: Option<u32>) -> Self {
        Self { pid, killed: false }
    }

    /// Kills the group unless it was already killed or `reaped` is set.
    ///
    /// Returns `true` when a kill was attempted.
    pub(crate) fn terminate(&mut self, reaped: bool) -> bool {
        if reaped || self.killed {
            return false;
        }
        self.killed = true;
        let Some(pid) = self.pid.take() else {
            return false;
        };
        kill_group(pid)
    }
}

/// Prepends the line-buffering wrapper to `argv` when line buffering is on and available.
///
/// Python parity: `_build_shell_command` wraps Linux commands in `stdbuf -oL -eL` so a
/// child writing to a pipe flushes per line. Wrapping happens at spawn time, after the
/// allowlist and sandbox gates, so the filters always inspect the user's command and
/// [`crate::shell::plan::resolve_plan`] keeps returning the plain interpreter.
///
/// The wrapper is skipped — never an error — when line buffering is disabled, `argv` is
/// empty, `stdbuf` is not on `PATH`, or the host is not Unix. (`macOS` has no `stdbuf`;
/// Python's `script -q /dev/null` alternative is not used because it changes the output
/// transport to a terminal and would duplicate the PTY path.)
pub(crate) fn line_buffered_argv(argv: Vec<String>, enabled: bool) -> Vec<String> {
    if !enabled {
        return argv;
    }
    wrap_with(resolve_wrapper().as_deref(), argv)
}

/// Resolves the line-buffering wrapper on `PATH`; `None` when it is unavailable.
#[cfg(unix)]
fn resolve_wrapper() -> Option<String> {
    match which::which(LINE_BUFFERING_WRAPPER) {
        Ok(path) => Some(crate::util::format_path(&path)),
        Err(error) => {
            tracing::debug!(%error, "line-buffering wrapper unavailable; spawning unwrapped");
            None
        }
    }
}

/// No line-buffering wrapper exists outside Unix.
#[cfg(not(unix))]
fn resolve_wrapper() -> Option<String> {
    None
}

/// Applies the wrapper decision given an already-resolved wrapper path.
fn wrap_with(wrapper: Option<&str>, mut argv: Vec<String>) -> Vec<String> {
    let Some(wrapper) = wrapper else {
        return argv;
    };
    if argv.is_empty() {
        return argv;
    }
    let mut prefixed = Vec::with_capacity(argv.len() + LINE_BUFFERING_FLAGS.len() + 1);
    prefixed.push(wrapper.to_string());
    prefixed.extend(LINE_BUFFERING_FLAGS.iter().map(|flag| (*flag).to_string()));
    prefixed.append(&mut argv);
    prefixed
}

/// Runs `child` to completion, capturing bounded output.
///
/// Output is drained concurrently with the wait, so a chatty child cannot deadlock on
/// a full pipe. A timeout or a cancellation token kills the child, keeps the partial
/// output, and sets the matching flag.
pub(crate) async fn run_captured(mut child: Child, options: RunOptions) -> RunOutcome {
    let started = Instant::now();
    let max_output_bytes = options.max_output_bytes;
    let mut group = ProcessGroup::new(child.id());
    let stdout_task = tokio::spawn(read_bounded(child.stdout.take(), max_output_bytes));
    let stderr_task = tokio::spawn(read_bounded(child.stderr.take(), max_output_bytes));
    let deadline = deadline_future(options.timeout_secs);
    let cancel = cancellation_future(&options.cancellation);
    let mut timed_out = false;
    let mut cancelled = false;
    let mut reaped = false;
    // The branch handlers must not touch `child`: the wait future borrows it.
    tokio::select! {
        _ = child.wait() => reaped = true,
        () = deadline => timed_out = true,
        () = cancel => cancelled = true,
    }
    if !reaped {
        // Kill the whole tree before reaping: the group is still addressable while the
        // unreaped child leads it, so grandchildren die with it.
        group.terminate(false);
        let _ = child.start_kill();
    }
    let exit_code = exit_code_of(child.wait().await);
    let stdout = join_reader(stdout_task).await;
    let stderr = join_reader(stderr_task).await;
    let duration_ms = elapsed_ms(started);
    RunOutcome {
        exit_code,
        stdout,
        stderr,
        timed_out,
        cancelled,
        duration_ms,
    }
}

/// Runs `child`, yielding output events until it exits or the run is interrupted.
///
/// The stream always ends with exactly one terminal event: [`EngineEvent::Exit`],
/// [`EngineEvent::TimedOut`], or [`EngineEvent::Cancelled`]. A reader failure ends the
/// affected pipe instead of panicking, and a per-stream byte budget emits
/// [`constants::OUTPUT_TRUNCATION_MARKER`] once before discarding further bytes.
pub(crate) fn stream_child(
    mut child: Child,
    options: RunOptions,
) -> BoxStream<'static, EngineEvent> {
    let max_output_bytes = options.max_output_bytes;
    let (sender, receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    if let Some(pipe) = child.stdout.take() {
        tokio::spawn(stream_reader(
            pipe,
            StreamKind::Stdout,
            max_output_bytes,
            sender.clone(),
        ));
    }
    if let Some(pipe) = child.stderr.take() {
        tokio::spawn(stream_reader(
            pipe,
            StreamKind::Stderr,
            max_output_bytes,
            sender.clone(),
        ));
    }
    // Only the reader tasks may keep the channel open.
    drop(sender);
    drive_child(child, options.timeout_secs, options.cancellation, receiver)
}

/// Forwards the events of `receiver` until it closes, then emits one terminal event.
///
/// The readers feeding `receiver` are already running; this driver owns the timeout, the
/// cancellation tokens, and the process-group kill, so the pipe path and the PTY path
/// share exactly one set of run semantics.
pub(crate) fn drive_child(
    mut child: Child,
    timeout_secs: u64,
    cancellation: Vec<CancellationToken>,
    mut receiver: mpsc::Receiver<EngineEvent>,
) -> BoxStream<'static, EngineEvent> {
    Box::pin(async_stream::stream! {
        let mut group = ProcessGroup::new(child.id());
        let mut deadline = deadline_future(timeout_secs);
        let mut cancel = cancellation_future(&cancellation);
        let mut interrupted = None;
        loop {
            let mut event: Option<EngineEvent> = None;
            let mut finished = false;
            // `yield` is not allowed inside a branch, so the event is buffered.
            tokio::select! {
                incoming = receiver.recv() => match incoming {
                    Some(item) => event = Some(item),
                    None => finished = true,
                },
                () = &mut deadline => {
                    interrupted = Some(EngineEvent::TimedOut { code: -1 });
                    finished = true;
                }
                () = &mut cancel => {
                    interrupted = Some(EngineEvent::Cancelled { code: -1 });
                    finished = true;
                }
            }
            if let Some(item) = event {
                yield item;
            }
            if finished {
                break;
            }
        }
        if interrupted.is_some() {
            // The child has not been reaped here, so its group is still addressable.
            group.terminate(false);
            let _ = child.start_kill();
        }
        let code = exit_code_of(child.wait().await);
        match interrupted {
            Some(event) => yield event,
            None => yield EngineEvent::Exit { code },
        }
    })
}

/// Reads `pipe`, sending bounded UTF-8 chunks to `sender`.
async fn stream_reader<R>(
    mut pipe: R,
    kind: StreamKind,
    max_output_bytes: u64,
    sender: mpsc::Sender<EngineEvent>,
) where
    R: AsyncRead + Unpin,
{
    let limit = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
    let mut produced = 0_usize;
    let mut marked = false;
    let mut decoder = Utf8Decoder::default();
    let mut buffer = vec![0_u8; constants::DEFAULT_TEXT_CHUNK_SIZE];
    loop {
        let read = match pipe.read(&mut buffer).await {
            // EOF and read failures both mean "no more data"; dropping the sender
            // signals the end of this pipe to the generator.
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        if marked {
            // Past the budget: keep draining so the child never blocks on a full pipe.
            continue;
        }
        let take = limit.saturating_sub(produced).min(read);
        if take > 0 {
            produced += take;
            let text = decoder.decode(&buffer[..take]);
            if !text.is_empty() && sender.send(EngineEvent::Data { kind, text }).await.is_err() {
                // The consumer went away: stop reading.
                return;
            }
        }
        if take < read {
            marked = true;
            let mut tail = decoder.flush();
            tail.push_str(constants::OUTPUT_TRUNCATION_MARKER);
            if sender
                .send(EngineEvent::Data { kind, text: tail })
                .await
                .is_err()
            {
                return;
            }
        }
    }
    let tail = decoder.flush();
    if !marked && !tail.is_empty() {
        let _ = sender.send(EngineEvent::Data { kind, text: tail }).await;
    }
}

/// Incremental UTF-8 decoder: a character split across reads is reassembled and
/// invalid sequences decode to `U+FFFD` (Python `errors="replace"`).
#[derive(Debug, Default, Clone)]
pub(crate) struct Utf8Decoder {
    /// Bytes of an incomplete trailing sequence carried to the next call.
    pending: Vec<u8>,
}

impl Utf8Decoder {
    /// Decodes as much of `bytes` as possible, keeping a partial trailing sequence.
    pub(crate) fn decode(&mut self, bytes: &[u8]) -> String {
        let mut combined = std::mem::take(&mut self.pending);
        combined.extend_from_slice(bytes);
        let mut text = String::with_capacity(combined.len());
        let mut rest = combined.as_slice();
        loop {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    text.push_str(valid);
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if let Ok(valid) = std::str::from_utf8(&rest[..valid_up_to]) {
                        text.push_str(valid);
                    }
                    let Some(length) = error.error_len() else {
                        self.pending = rest[valid_up_to..].to_vec();
                        break;
                    };
                    text.push('\u{FFFD}');
                    rest = &rest[valid_up_to + length..];
                }
            }
        }
        text
    }

    /// Decodes the carried partial sequence, replacing undecodable bytes.
    pub(crate) fn flush(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return String::new();
        }
        String::from_utf8_lossy(&pending).into_owned()
    }
}

/// A future that completes when `timeout_secs` elapses (`0` = never).
pub(crate) fn deadline_future(timeout_secs: u64) -> EngineFuture {
    if timeout_secs == 0 {
        return Box::pin(std::future::pending());
    }
    let duration = Duration::from_secs(timeout_secs);
    Box::pin(async move { tokio::time::sleep(duration).await })
}

/// A future that completes when any of `tokens` is cancelled (`pending` when empty).
pub(crate) fn cancellation_future(tokens: &[CancellationToken]) -> EngineFuture {
    let futures: Vec<EngineFuture> = tokens
        .iter()
        .map(|token| {
            let token = token.clone();
            Box::pin(async move { token.cancelled_owned().await }) as EngineFuture
        })
        .collect();
    if futures.is_empty() {
        return Box::pin(std::future::pending());
    }
    Box::pin(async move {
        let _ = futures::future::select_all(futures).await;
    })
}

/// Reads `pipe` to EOF, retaining at most `max_output_bytes` bytes.
async fn read_bounded<R>(pipe: Option<R>, max_output_bytes: u64) -> CapturedText
where
    R: AsyncRead + Unpin,
{
    let Some(mut pipe) = pipe else {
        return CapturedText::default();
    };
    let limit = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
    let mut buffer = vec![0_u8; READ_CHUNK_SIZE];
    let mut kept: Vec<u8> = Vec::new();
    let mut truncated = false;
    loop {
        match pipe.read(&mut buffer).await {
            // EOF and I/O errors both end the capture; whatever was read is kept.
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let take = limit.saturating_sub(kept.len()).min(read);
                kept.extend_from_slice(&buffer[..take]);
                truncated |= take < read;
            }
        }
    }
    let mut text = String::from_utf8_lossy(&kept).into_owned();
    if truncated {
        text.push_str(constants::OUTPUT_TRUNCATION_MARKER);
    }
    CapturedText { text, truncated }
}

/// Awaits a pipe reader, giving up after [`READER_GRACE`].
async fn join_reader(mut task: tokio::task::JoinHandle<CapturedText>) -> CapturedText {
    if let Ok(Ok(captured)) = tokio::time::timeout(READER_GRACE, &mut task).await {
        return captured;
    }
    task.abort();
    CapturedText::default()
}

/// Wall-clock milliseconds since `started`, saturating at `u64::MAX`.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The exit code of `status`, or `-1` when the child died from a signal.
pub(crate) fn exit_code_of(status: std::io::Result<ExitStatus>) -> i32 {
    status.ok().and_then(|status| status.code()).unwrap_or(-1)
}

/// Builds Python's early-exit message (`process exited early with code <n>: <stderr>`).
pub(crate) async fn background_early_exit(
    pid: Option<u32>,
    status: ExitStatus,
    stderr_task: Option<tokio::task::JoinHandle<String>>,
    sink: &Arc<Mutex<StderrSink>>,
) -> String {
    let code = status.code().unwrap_or(-1);
    let stderr = collect_stderr(stderr_task, sink).await;
    let trimmed = stderr.trim();
    let message = if trimmed.is_empty() {
        format!("process exited early with code {code}")
    } else {
        format!("process exited early with code {code}: {trimmed}")
    };
    tracing::warn!(pid, reason = %message, "background command exited early");
    message
}

/// Retained stderr of a backgrounded command, shared with its reader task.
#[derive(Debug, Default)]
pub(crate) struct StderrSink {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Locks the stderr sink, recovering from a poisoned lock.
fn lock_sink(sink: &Mutex<StderrSink>) -> std::sync::MutexGuard<'_, StderrSink> {
    sink.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drains `pipe` into `sink`, retaining at most `limit` bytes, and returns the final text.
pub(crate) async fn drain_stderr<R>(mut pipe: R, limit: u64, sink: Arc<Mutex<StderrSink>>) -> String
where
    R: AsyncRead + Unpin,
{
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut buffer = vec![0_u8; constants::DEFAULT_TEXT_CHUNK_SIZE];
    loop {
        let read = match pipe.read(&mut buffer).await {
            // EOF or a read failure ends the capture and keeps what was retained.
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        let mut state = lock_sink(&sink);
        let take = limit.saturating_sub(state.bytes.len()).min(read);
        if take > 0 {
            state.bytes.extend_from_slice(&buffer[..take]);
        }
        state.truncated |= take < read;
    }
    let mut text = String::from_utf8_lossy(&lock_sink(&sink).bytes).into_owned();
    if lock_sink(&sink).truncated {
        text.push_str(STDERR_TRUNCATION_MARKER);
    }
    text
}

/// Awaits the stderr reader, falling back to the shared buffer when it is still busy.
async fn collect_stderr(
    task: Option<tokio::task::JoinHandle<String>>,
    sink: &Arc<Mutex<StderrSink>>,
) -> String {
    let Some(mut task) = task else {
        return String::new();
    };
    if let Ok(Ok(text)) = tokio::time::timeout(STDERR_JOIN_GRACE, &mut task).await {
        return text;
    }
    task.abort();
    let state = lock_sink(sink);
    let mut text = String::from_utf8_lossy(&state.bytes).into_owned();
    if state.truncated {
        text.push_str(STDERR_TRUNCATION_MARKER);
    }
    text
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn incremental_decoder_handles_split_characters() {
        let text = "hello 中中 end";
        let bytes = text.as_bytes();
        for split in 0..=bytes.len() {
            let mut decoder = Utf8Decoder::default();
            let mut out = decoder.decode(&bytes[..split]);
            out.push_str(&decoder.decode(&bytes[split..]));
            out.push_str(&decoder.flush());
            assert_eq!(out, text, "split at byte {split}");
        }
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.decode(b""), "");
        assert_eq!(decoder.flush(), "");
    }

    #[test]
    fn decoder_replaces_invalid_and_incomplete_bytes() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.decode(&[0xFF, 0x41]), "\u{FFFD}A");
        let mut decoder = Utf8Decoder::default();
        // An incomplete three-byte sequence is carried, then flushed lossily.
        assert_eq!(decoder.decode(&[0xE4, 0xB8]), "");
        assert_eq!(decoder.flush(), "\u{FFFD}");
    }

    /// The plan of `echo hi` under `/bin/sh`, as a command plan resolves it.
    fn sh_argv() -> Vec<String> {
        ["/bin/sh", "-c", "echo hi"]
            .iter()
            .map(|part| (*part).to_string())
            .collect()
    }

    #[test]
    fn wrapper_is_prepended_when_a_wrapper_exists() {
        let argv = sh_argv();
        let wrapped = wrap_with(Some("/usr/bin/stdbuf"), argv.clone());
        assert_eq!(
            wrapped,
            vec!["/usr/bin/stdbuf", "-oL", "-eL", "/bin/sh", "-c", "echo hi"]
        );
        // The plan's own argv is preserved verbatim after the flags.
        assert_eq!(&wrapped[3..], &argv[..]);
    }

    #[test]
    fn wrapper_is_skipped_without_a_wrapper_or_a_request() {
        let argv = sh_argv();
        assert_eq!(wrap_with(None, argv.clone()), argv);
        assert_eq!(line_buffered_argv(argv.clone(), false), argv);
        assert_eq!(
            wrap_with(Some("/usr/bin/stdbuf"), Vec::new()),
            Vec::<String>::new(),
            "an empty argv has nothing to wrap"
        );
    }

    #[test]
    fn enabled_line_buffering_matches_the_resolved_wrapper() {
        let argv = sh_argv();
        let wrapped = line_buffered_argv(argv.clone(), true);
        if which::which(LINE_BUFFERING_WRAPPER).is_ok() {
            assert_eq!(wrapped.len(), argv.len() + LINE_BUFFERING_FLAGS.len() + 1);
            assert_eq!(&wrapped[1..3], &LINE_BUFFERING_FLAGS);
            assert_eq!(&wrapped[3..], &argv[..]);
        } else {
            // No wrapper on this host: the command still runs, unwrapped.
            assert_eq!(wrapped, argv);
        }
    }

    #[test]
    fn process_group_is_never_killed_after_the_reap() {
        let pid = u32::try_from(i32::MAX).unwrap();
        let mut group = ProcessGroup::new(Some(pid));
        assert!(!group.terminate(true), "a reaped child is never signalled");
        assert!(
            !group.killed,
            "the reap guard must not consume the kill-once flag"
        );
        assert_eq!(group.pid, Some(pid));
    }

    #[test]
    fn process_group_is_killed_at_most_once() {
        // No live group can own this pid, so the signal is attempted and not delivered.
        let pid = u32::try_from(i32::MAX).unwrap();
        let mut group = ProcessGroup::new(Some(pid));
        assert!(!group.terminate(false), "there is no such process group");
        assert!(group.killed, "the attempt sets the kill-once flag");
        assert_eq!(group.pid, None);
        assert!(!group.terminate(false), "a group is never killed twice");

        let mut reaped_twice = ProcessGroup::new(None);
        assert!(!reaped_twice.terminate(false), "an unknown pid is a no-op");
        assert!(reaped_twice.killed);
    }

    #[test]
    fn kill_group_rejects_a_non_process_id() {
        // pid 0 would signal the caller's own process group.
        assert!(!kill_group(0));
    }
}
