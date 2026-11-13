//! The child-process member runtime: spawn one member process and drive it over the JSON-lines
//! protocol.
//!
//! [`StdioMemberRuntime`] is the [`MemberRuntime`] for [`MemberSpawnKind::Process`],
//! [`MemberSpawnKind::ExternalCli`] and `MemberSpawnKind::Ssh` members: every child-process spawn
//! kind launches a program whose stdin/stdout carry the [`crate::member::protocol`] frames.
//! Only the launch details differ, so this runtime owns the process, the pipes, the reader tasks
//! and the lifecycle shared by all three kinds.
//! Python parity source: `openjiuwen/agent_teams/spawn/inprocess_spawn.py` and
//! `spawn/process_spawn.py` (the `asyncio.subprocess` member wrapper).
//!
//! # Rules
//!
//! The runtime owns exactly one child process and never blocks the async runtime on it.
//! One background task owns the [`Child`] and reaps it; a second reads stdout frame by frame; a
//! third drains stderr into a bounded diagnostics tail.
//! Garbage on stdout is logged and skipped, so a child that prints log noise is not mistaken for a
//! protocol failure.
//! The reader task gives up after `MAX_GARBAGE_LINES` consecutive unparseable lines and marks the
//! runtime terminated, so a hostile child cannot spin the reader forever.
//! `ready` never reaches observers; it only completes the pending [`MemberRuntime::start`]
//! handshake.
//! An `error` frame fails the pending `start` handshake and is otherwise logged; it never moves the
//! lifecycle state.
//! A child exit stores [`HarnessState::Terminated`], forwards it to every observer exactly once and
//! turns every later call into a typed "member process ... exited" error.
//! Every write and every handshake wait is bounded by [`StdioRuntimeOptions::io_timeout`].
//! Cancellation uses a [`CancellationToken`]; cancelling it kills and reaps the child exactly like
//! [`MemberRuntime::stop`] does.
//! [`Drop`] kills the child without blocking and aborts the background tasks; it cannot reap the
//! exit status, which tokio's orphan reaper handles.
//!
//! [`MemberSpawnKind::Process`]: crate::member::spawn_config::MemberSpawnKind::Process
//! [`MemberSpawnKind::ExternalCli`]: crate::member::spawn_config::MemberSpawnKind::ExternalCli
//!
//! ```no_run
//! use openjiuwen_agent_teams::member::spawn_config::StdioCommand;
//! use openjiuwen_agent_teams::member::stdio::StdioMemberRuntime;
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let runtime = StdioMemberRuntime::spawn(
//!     &StdioCommand::new("team-member").with_arg("serve"),
//!     "team-session",
//! )
//! .await?;
//! runtime.stop().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt as _, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::member::protocol::{MemberRequest, MemberResponse};
use crate::member::runtime::{MemberObserver, MemberRuntime, ObserverSet};
use crate::member::spawn_config::StdioCommand;

/// Default bound for every stdin write, handshake wait and graceful teardown.
pub const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Number of consecutive unparseable stdout lines after which the reader declares the member
/// terminated.
const MAX_GARBAGE_LINES: usize = 100;

/// Number of stderr lines retained for diagnostics.
const MAX_STDERR_LINES: usize = 40;

/// Maximum number of stderr characters retained for diagnostics.
const MAX_STDERR_CHARS: usize = 8000;

/// Longest stdout prefix echoed into a warning for an unparseable line.
const STDOUT_PREVIEW_CHARS: usize = 200;

/// How the child is driven: the bound for every pipe write and handshake wait, plus the optional
/// cancellation token.
///
/// The default is [`DEFAULT_IO_TIMEOUT`] and no token; the runtime mints a fresh token in that case
/// so [`StdioMemberRuntime::cancellation_token`] always returns something usable.
#[derive(Debug, Clone)]
pub struct StdioRuntimeOptions {
    /// Bound applied to one stdin write and to one handshake wait.
    pub io_timeout: Duration,
    /// Token that, when cancelled, kills and reaps the child like [`MemberRuntime::stop`] does.
    pub cancellation: Option<CancellationToken>,
}

impl Default for StdioRuntimeOptions {
    fn default() -> Self {
        Self {
            io_timeout: DEFAULT_IO_TIMEOUT,
            cancellation: None,
        }
    }
}

impl StdioRuntimeOptions {
    /// Creates the default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the I/O bound.
    #[must_use]
    pub fn with_io_timeout(mut self, timeout: Duration) -> Self {
        self.io_timeout = timeout;
        self
    }

    /// Attaches a cancellation token.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }
}

/// A member whose brain is a child process speaking the JSON-lines member protocol.
///
/// The runtime is constructed by [`StdioMemberRuntime::spawn`] or
/// [`StdioMemberRuntime::spawn_with`]; every call is bounded by the configured I/O timeout and
/// every call fails fast once the child has exited.
/// The configured session id is fixed at construction because [`MemberRuntime::session_id`] returns
/// a stable `&str` for the lifetime of the runtime.
pub struct StdioMemberRuntime {
    /// Program configured for the child, used in every diagnostic message.
    program: String,
    /// Session id reported by [`MemberRuntime::session_id`].
    session_id: String,
    /// I/O bound and the configured cancellation token.
    options: StdioRuntimeOptions,
    /// Operating-system process id captured at spawn time.
    pid: Option<u32>,
    /// State shared with the reader and supervisor tasks.
    shared: Arc<RuntimeState>,
    /// Token exposed by [`Self::cancellation_token`]; cancelling it tears the child down.
    cancel: CancellationToken,
    /// Token the supervisor waits on; cancelling it makes the supervisor kill and reap the child.
    kill: CancellationToken,
    /// Task that reads stdout frames.
    reader: JoinHandle<()>,
    /// Task that drains stderr, when the child gave us a stderr pipe.
    stderr_reader: Option<JoinHandle<()>>,
    /// Task that owns the [`Child`] and reaps it.
    supervisor: JoinHandle<()>,
    /// Task that turns a cancellation into a kill.
    watcher: JoinHandle<()>,
    /// Serializes `start` handshakes so at most one is pending at a time.
    start_lock: tokio::sync::Mutex<()>,
}

impl StdioMemberRuntime {
    /// Spawns `command` and returns a runtime for it, using [`StdioRuntimeOptions::default`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the program cannot be resolved, [`Error::transport`] when the
    /// child cannot be spawned or one of its pipes is unavailable.
    pub async fn spawn(command: &StdioCommand, session_id: impl Into<String>) -> Result<Self> {
        Self::spawn_with(command, session_id, StdioRuntimeOptions::default()).await
    }

    /// Spawns `command` with explicit options and returns a runtime for it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the program cannot be resolved, [`Error::transport`] when the
    /// child cannot be spawned or one of its pipes is unavailable.
    pub async fn spawn_with(
        command: &StdioCommand,
        session_id: impl Into<String>,
        options: StdioRuntimeOptions,
    ) -> Result<Self> {
        let program = command.program().to_owned();
        let mut command = command.to_command()?;
        let mut child = command.spawn().map_err(|error| {
            Error::transport(format!(
                "failed to start member process '{program}': {error}"
            ))
        })?;
        let pid = child.id();
        let Some(stdin) = child.stdin.take() else {
            let _ = child.start_kill();
            return Err(Error::transport(format!(
                "member process '{program}' stdin is unavailable"
            )));
        };
        let Some(stdout) = child.stdout.take() else {
            let _ = child.start_kill();
            return Err(Error::transport(format!(
                "member process '{program}' stdout is unavailable"
            )));
        };
        let stderr = child.stderr.take();

        let cancel = options.cancellation.clone().unwrap_or_default();
        let kill = CancellationToken::new();
        let shared = Arc::new(RuntimeState::new(program.clone(), session_id.into(), stdin));

        let reader = spawn_reader(stdout, Arc::clone(&shared));
        let stderr_reader = stderr.map(|pipe| spawn_stderr_reader(pipe, Arc::clone(&shared)));
        let supervisor = spawn_supervisor(child, Arc::clone(&shared), kill.clone());
        let watcher = spawn_cancellation_watcher(
            Arc::clone(&shared),
            cancel.clone(),
            kill.clone(),
            options.io_timeout,
        );
        // Yield once so the pipe readers and the supervisor are scheduled before the caller writes
        // its first frame; nothing depends on the order, but it keeps a `start` write off the same
        // scheduler tick as the spawn.
        tokio::task::yield_now().await;
        let configured_session = shared.session_id.clone();
        Ok(Self {
            program,
            session_id: configured_session,
            options,
            pid,
            shared,
            cancel,
            kill,
            reader,
            stderr_reader,
            supervisor,
            watcher,
            start_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Returns the session id configured for this runtime.
    ///
    /// This is the same value [`MemberRuntime::session_id`] returns; it exists so callers that do
    /// not import the trait can still read it.
    #[must_use]
    pub fn session_id_value(&self) -> &str {
        &self.session_id
    }

    /// Returns the operating-system process id captured at spawn time.
    #[must_use]
    pub fn child_pid(&self) -> Option<u32> {
        self.pid
    }

    /// Returns `true` while the runtime has not observed the child's termination.
    ///
    /// The answer flips to `false` as soon as the supervisor (or the stdout reader) records the
    /// termination, not necessarily the instant the operating system reaps the process.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.shared.is_terminated()
    }

    /// Returns the child's exit code once it exited normally.
    ///
    /// A child killed by a signal (or one the runtime declared terminated before reaping) has no
    /// exit code, so this returns `None` even after termination.
    #[must_use]
    pub fn exit_code(&self) -> Option<i32> {
        self.shared
            .termination
            .lock()
            .as_ref()
            .and_then(|termination| termination.code)
    }

    /// Returns the bounded stderr tail captured so far: at most 40 lines and 8000 characters.
    #[must_use]
    pub fn stderr_tail(&self) -> String {
        self.shared.stderr_text()
    }

    /// Returns the cancellation token this runtime watches.
    ///
    /// When [`StdioRuntimeOptions::cancellation`] was `None`, this is a token the runtime minted;
    /// cancelling it kills and reaps the child.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Returns the options the runtime was built with.
    ///
    /// The configured `cancellation` field is left as the caller passed it; the token the runtime
    /// actually watches is available from [`Self::cancellation_token`].
    #[must_use]
    pub fn options(&self) -> &StdioRuntimeOptions {
        &self.options
    }

    /// Returns the typed error every call fails with once the child exited, if it has.
    fn terminated_error(&self) -> Option<Error> {
        self.shared.terminated_error()
    }

    /// Writes one protocol frame, bounded by the configured I/O timeout.
    async fn write_frame(&self, request: &MemberRequest) -> Result<()> {
        if let Some(error) = self.terminated_error() {
            return Err(error);
        }
        let line = request.to_line()?;
        let frame = request.frame_type();
        let io_timeout = self.options.io_timeout;
        let mut stdin = tokio::time::timeout(io_timeout, self.shared.stdin.lock())
            .await
            .map_err(|_| {
                Error::timeout(format!(
                    "member process '{}': timed out after {io_timeout:?} waiting to write the '{frame}' frame",
                    self.program
                ))
            })?;
        let Some(stdin) = stdin.as_mut() else {
            return Err(self.terminated_error().unwrap_or_else(|| {
                Error::transport(format!("member process '{}' stdin is closed", self.program))
            }));
        };
        let write = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        tokio::time::timeout(io_timeout, write)
            .await
            .map_err(|_| {
                Error::timeout(format!(
                    "member process '{}' did not accept the '{frame}' frame within {io_timeout:?}",
                    self.program
                ))
            })?
            .map_err(|error| {
                Error::transport(format!(
                    "failed to write the '{frame}' frame to member process '{}': {error}",
                    self.program
                ))
            })
    }

    /// Waits up to `bound` for the child's termination to be recorded.
    async fn wait_terminated(&self, bound: Duration) -> bool {
        tokio::time::timeout(bound, wait_for_termination(&self.shared))
            .await
            .is_ok()
    }
}

#[async_trait]
impl MemberRuntime for StdioMemberRuntime {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn state(&self) -> HarnessState {
        self.shared.state()
    }

    /// Sends the `start` frame and waits for the child's `ready` frame.
    ///
    /// A child process cannot receive an [`Arc<dyn Session>`](Session), so only the configured
    /// session id crosses the process boundary; the `team_session` argument is accepted and
    /// ignored, and harness state persistence belongs to the child.
    /// This call does not move the lifecycle state itself — state changes come from the child's
    /// `state` frames (and from termination).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Timeout`](openjiuwen_common::error::ErrorKind::Timeout) when the child
    /// does not answer within the I/O bound, [`ErrorKind::External`](openjiuwen_common::error::ErrorKind::External)
    /// when the child answers with an `error` frame or exits first, and
    /// [`ErrorKind::Transport`](openjiuwen_common::error::ErrorKind::Transport) when the start frame
    /// cannot be written.
    async fn start(&self, _team_session: Option<Arc<dyn Session>>) -> Result<()> {
        let _serial = self.start_lock.lock().await;
        if let Some(error) = self.terminated_error() {
            return Err(error);
        }
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = self.shared.start_pending.lock();
            *pending = Some(sender);
        }
        let request = MemberRequest::Start {
            session_id: self.session_id.clone(),
        };
        if let Err(error) = self.write_frame(&request).await {
            self.shared.start_pending.lock().take();
            return Err(error);
        }
        let io_timeout = self.options.io_timeout;
        match tokio::time::timeout(io_timeout, receiver).await {
            Err(_) => {
                self.shared.start_pending.lock().take();
                Err(Error::timeout(format!(
                    "member process '{}' did not answer the start frame with a ready frame within {io_timeout:?}",
                    self.program
                )))
            }
            Ok(Err(_)) => Err(self.terminated_error().unwrap_or_else(|| {
                Error::transport(format!(
                    "member process '{}' dropped the start handshake",
                    self.program
                ))
            })),
            Ok(Ok(StartReply::Ready)) => Ok(()),
            Ok(Ok(StartReply::Refused(message))) => Err(Error::external(std::io::Error::other(
                format!("member process refused to start: {message}"),
            ))),
            Ok(Ok(StartReply::Exited(error))) => Err(*error),
        }
    }

    /// Stops the child: a best-effort `stop` frame, then a kill, then a bounded wait for the exit.
    ///
    /// The call is idempotent and returns `Ok(())` when the runtime already terminated; teardown
    /// loops depend on that. A `stop` frame that cannot be written is logged, never returned.
    /// The stdin handle is dropped last so the child observes end-of-file.
    ///
    /// # Errors
    ///
    /// This implementation never reports a teardown failure: every step is best effort and the
    /// terminal state is recorded even when the child ignores the kill.
    async fn stop(&self) -> Result<()> {
        if self.shared.is_terminated() {
            return Ok(());
        }
        if let Err(error) = self.write_frame(&MemberRequest::Stop).await {
            debug!(
                program = %self.program,
                %error,
                "could not write the stop frame; killing the member process"
            );
        }
        self.kill.cancel();
        if !self.wait_terminated(self.options.io_timeout).await {
            warn!(
                program = %self.program,
                bound = ?self.options.io_timeout,
                "member process did not exit after being killed; escalating"
            );
            self.kill.cancel();
            if !self.shared.is_terminated() {
                self.shared
                    .terminate(Termination::stop_timeout(&self.program));
            }
        }
        if let Ok(mut stdin) =
            tokio::time::timeout(self.options.io_timeout, self.shared.stdin.lock()).await
        {
            stdin.take();
        }
        Ok(())
    }

    /// Writes one `send` frame, passing the `immediate` flag through.
    ///
    /// # Errors
    ///
    /// Returns a typed termination error once the child exited, [`Error::timeout`] when the write
    /// exceeds the I/O bound, and [`Error::transport`] on a pipe failure.
    async fn send(&self, content: String, immediate: bool) -> Result<()> {
        self.write_frame(&MemberRequest::Send { content, immediate })
            .await
    }

    /// Writes one `abort` frame, passing the `immediate` flag through.
    ///
    /// # Errors
    ///
    /// Returns a typed termination error once the child exited, [`Error::timeout`] when the write
    /// exceeds the I/O bound, and [`Error::transport`] on a pipe failure.
    async fn abort(&self, immediate: bool) -> Result<()> {
        self.write_frame(&MemberRequest::Abort { immediate }).await
    }

    /// Writes one `pause` frame.
    ///
    /// # Errors
    ///
    /// Returns a typed termination error once the child exited, [`Error::timeout`] when the write
    /// exceeds the I/O bound, and [`Error::transport`] on a pipe failure.
    async fn pause(&self) -> Result<()> {
        self.write_frame(&MemberRequest::Pause).await
    }

    /// Writes one `resume` frame, carrying the optional query.
    ///
    /// # Errors
    ///
    /// Returns a typed termination error once the child exited, [`Error::timeout`] when the write
    /// exceeds the I/O bound, and [`Error::transport`] on a pipe failure.
    async fn resume(&self, query: Option<String>) -> Result<()> {
        self.write_frame(&MemberRequest::Resume { query }).await
    }

    /// Registers an observer for later state, round and chunk frames.
    ///
    /// A `ready` frame never reaches observers; it completes [`Self::start`] instead.
    fn subscribe(&self, observer: Arc<dyn MemberObserver>) {
        self.shared.observers.add(observer);
    }
}

impl Drop for StdioMemberRuntime {
    /// Kills the child without blocking and aborts the background tasks.
    ///
    /// Dropping cannot await, so the exit status is not reaped here: aborting the supervisor drops
    /// the [`Child`], whose `kill_on_drop` sends the kill and leaves reaping to tokio's orphan
    /// reaper.
    fn drop(&mut self) {
        self.kill.cancel();
        self.reader.abort();
        if let Some(reader) = self.stderr_reader.as_ref() {
            reader.abort();
        }
        self.watcher.abort();
        self.supervisor.abort();
    }
}

impl std::fmt::Debug for StdioMemberRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StdioMemberRuntime")
            .field("program", &self.program)
            .field("pid", &self.pid)
            .field("session_id", &self.session_id)
            .field("state", &self.shared.state())
            .finish_non_exhaustive()
    }
}

/// The answer one pending `start` handshake receives from the child.
enum StartReply {
    /// The child sent `ready`.
    Ready,
    /// The child sent an `error` frame with this message.
    Refused(String),
    /// The child exited before answering.
    Exited(Box<Error>),
}

/// Why and how the child ended, recorded once so every later call can name it.
#[derive(Debug, Clone)]
struct Termination {
    /// Exit code when the process exited normally.
    code: Option<i32>,
    /// Human-readable description used in error messages.
    reason: String,
}

impl Termination {
    /// Describes a collected exit status.
    fn from_status(status: ExitStatus) -> Self {
        match status.code() {
            Some(code) => Self {
                code: Some(code),
                reason: format!("exited with status {code}"),
            },
            None => Self {
                code: None,
                reason: format!("ended without an exit status ({status})"),
            },
        }
    }

    /// Describes a failed wait.
    fn from_wait_error(error: &std::io::Error) -> Self {
        Self {
            code: None,
            reason: format!("ended without an exit status: {error}"),
        }
    }

    /// Describes a reader that gave up on unparseable stdout.
    fn garbage() -> Self {
        Self {
            code: None,
            reason: format!(
                "was terminated after {MAX_GARBAGE_LINES} consecutive unparseable stdout lines"
            ),
        }
    }

    /// Describes a stop request that did not finish within the I/O bound.
    fn stop_timeout(program: &str) -> Self {
        Self {
            code: None,
            reason: format!(
                "was marked terminated after a stop request to '{program}' did not exit within the I/O timeout"
            ),
        }
    }

    /// Describes a termination recorded without an exit status.
    fn unrecorded() -> Self {
        Self {
            code: None,
            reason: "reported the terminated state without an exit status".to_owned(),
        }
    }
}

/// State shared between the runtime and its reader, stderr, supervisor and watcher tasks.
///
/// Every field is a short critical section: no `parking_lot` guard is ever held across an `await`.
struct RuntimeState {
    /// Program the child was launched from, used in diagnostics.
    program: String,
    /// Session id configured for the member; the child's `ready` id never replaces it.
    session_id: String,
    /// Lifecycle phase last reported by the child (or by termination).
    state: RwLock<HarnessState>,
    /// Termination detail, recorded once.
    termination: Mutex<Option<Termination>>,
    /// Child stdin, absent once the runtime drops it.
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
    /// Observers that receive state, round and chunk callbacks.
    observers: ObserverSet,
    /// Pending `start` handshake, if one is in flight.
    start_pending: Mutex<Option<oneshot::Sender<StartReply>>>,
    /// Broadcasts termination to waiters; `send_replace` keeps the value without receivers.
    terminated: watch::Sender<bool>,
    /// Bounded stderr tail for diagnostics.
    stderr: Mutex<StderrTail>,
}

impl RuntimeState {
    /// Builds the shared state for a freshly spawned child.
    fn new(program: String, session_id: String, stdin: ChildStdin) -> Self {
        let (terminated, _receiver) = watch::channel(false);
        Self {
            program,
            session_id,
            state: RwLock::new(HarnessState::Idle),
            termination: Mutex::new(None),
            stdin: tokio::sync::Mutex::new(Some(stdin)),
            observers: ObserverSet::new(),
            start_pending: Mutex::new(None),
            terminated,
            stderr: Mutex::new(StderrTail::default()),
        }
    }

    /// Returns the current lifecycle phase.
    fn state(&self) -> HarnessState {
        *self.state.read()
    }

    /// Returns `true` once the terminal phase was recorded.
    fn is_terminated(&self) -> bool {
        matches!(self.state(), HarnessState::Terminated)
    }

    /// Returns the bounded stderr tail.
    fn stderr_text(&self) -> String {
        self.stderr.lock().text()
    }

    /// Builds the typed error every call fails with after termination.
    fn terminated_error(&self) -> Option<Error> {
        if !self.is_terminated() {
            return None;
        }
        let termination = {
            let slot = self.termination.lock();
            slot.clone()
        };
        let termination = termination.unwrap_or_else(Termination::unrecorded);
        Some(self.error_for(&termination))
    }

    /// Renders `termination` as a typed error, appending the stderr tail when there is one.
    fn error_for(&self, termination: &Termination) -> Error {
        let mut message = format!("member process {}", termination.reason);
        let tail = self.stderr_text();
        if !tail.is_empty() {
            message.push_str("; stderr: ");
            message.push_str(&tail);
        }
        match termination.code {
            Some(_) => Error::external(std::io::Error::other(message)),
            None => Error::transport(message),
        }
    }

    /// Records the terminal phase, waking the pending `start` handshake, the waiters and the
    /// observers exactly once.
    fn terminate(&self, termination: Termination) {
        let first = {
            let mut state = self.state.write();
            let first = *state != HarnessState::Terminated;
            *state = HarnessState::Terminated;
            first
        };
        let stored = {
            let mut slot = self.termination.lock();
            if slot.is_none() {
                *slot = Some(termination);
            }
            slot.clone().unwrap_or_else(Termination::unrecorded)
        };
        let error = self.error_for(&stored);
        if let Some(sender) = self.start_pending.lock().take() {
            let _ = sender.send(StartReply::Exited(Box::new(error)));
        }
        if first {
            debug!(program = %self.program, reason = %stored.reason, "member process terminated");
            // Notify observers before the watch signal: a `stop` waiter that resumes on the signal
            // must already see the terminal fan-out.
            self.observers.state(HarnessState::Terminated);
        }
        self.terminated.send_replace(true);
    }

    /// Applies one frame read from the child's stdout.
    fn handle_response(&self, response: MemberResponse) {
        if self.is_terminated() {
            return;
        }
        match response {
            MemberResponse::Ready { session_id } => {
                if session_id != self.session_id {
                    warn!(
                        configured = %self.session_id,
                        child = %session_id,
                        "member process reported a different session id; keeping the configured id"
                    );
                }
                if let Some(sender) = self.start_pending.lock().take() {
                    let _ = sender.send(StartReply::Ready);
                } else {
                    debug!(program = %self.program, "member process sent ready with no pending start handshake");
                }
            }
            MemberResponse::State { state } => {
                *self.state.write() = state;
                self.observers.state(state);
            }
            MemberResponse::Round { outcome } => self.observers.round(outcome),
            MemberResponse::Chunk { chunk } => self.observers.chunk(chunk),
            MemberResponse::Error { message } => {
                if let Some(sender) = self.start_pending.lock().take() {
                    let _ = sender.send(StartReply::Refused(message));
                } else {
                    warn!(program = %self.program, %message, "member process reported an error");
                }
            }
        }
    }
}

/// A bounded tail of the child's stderr: the last [`MAX_STDERR_LINES`] lines, at most
/// [`MAX_STDERR_CHARS`] characters.
#[derive(Debug, Default)]
struct StderrTail {
    /// Retained lines, oldest first.
    lines: VecDeque<String>,
    /// Approximate retained character count, counting one separator per line.
    chars: usize,
}

impl StderrTail {
    /// Appends one line, trimming the front to stay inside the bounds on a character boundary.
    fn push(&mut self, line: String) {
        let line = truncate_chars(line, MAX_STDERR_CHARS);
        self.chars += line.chars().count() + 1;
        self.lines.push_back(line);
        while self.lines.len() > MAX_STDERR_LINES
            || (self.chars > MAX_STDERR_CHARS && self.lines.len() > 1)
        {
            let Some(front) = self.lines.pop_front() else {
                break;
            };
            self.chars = self.chars.saturating_sub(front.chars().count() + 1);
        }
    }

    /// Renders the retained lines as one newline-joined string.
    fn text(&self) -> String {
        self.lines
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Truncates `text` to at most `limit` characters on a character boundary.
fn truncate_chars(text: String, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text;
    }
    text.chars().take(limit).collect()
}

/// A short, char-boundary-safe prefix of one stdout line for a warning.
fn preview(line: &str) -> String {
    let mut shown: String = line.chars().take(STDOUT_PREVIEW_CHARS).collect();
    if line.chars().count() > STDOUT_PREVIEW_CHARS {
        shown.push('…');
    }
    shown
}

/// Resolves when the terminal phase has been recorded.
async fn wait_for_termination(state: &RuntimeState) {
    let mut receiver = state.terminated.subscribe();
    let _ = receiver.wait_for(|terminated| *terminated).await;
}

/// Owns the [`Child`] and reaps it, either after a natural exit or after a kill request.
fn spawn_supervisor(
    mut child: Child,
    state: Arc<RuntimeState>,
    kill: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let status = tokio::select! {
            biased;
            () = kill.cancelled() => {
                if let Err(error) = child.start_kill() {
                    warn!(
                        program = %state.program,
                        %error,
                        "failed to kill member process"
                    );
                }
                child.wait().await
            }
            status = child.wait() => status,
        };
        let termination = match status {
            Ok(status) => Termination::from_status(status),
            Err(error) => Termination::from_wait_error(&error),
        };
        state.terminate(termination);
    })
}

/// Reads stdout line by line and applies each frame.
fn spawn_reader(stdout: ChildStdout, state: Arc<RuntimeState>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        let mut garbage = 0_usize;
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => match MemberResponse::parse_noisy_line(&line) {
                    Ok(None) => {}
                    Ok(Some(response)) => {
                        garbage = 0;
                        state.handle_response(response);
                    }
                    Err(error) => {
                        garbage += 1;
                        warn!(
                            program = %state.program,
                            line = %preview(&line),
                            %error,
                            "ignoring unparseable member stdout line"
                        );
                        if garbage >= MAX_GARBAGE_LINES {
                            warn!(
                                program = %state.program,
                                limit = MAX_GARBAGE_LINES,
                                "member process produced too many unparseable stdout lines; terminating"
                            );
                            state.terminate(Termination::garbage());
                            return;
                        }
                    }
                },
                Ok(None) => {
                    debug!(program = %state.program, "member stdout closed");
                    return;
                }
                Err(error) => {
                    warn!(
                        program = %state.program,
                        %error,
                        "member stdout read failed"
                    );
                    return;
                }
            }
        }
    })
}

/// Drains stderr into the bounded diagnostics tail; stderr is never protocol.
fn spawn_stderr_reader(stderr: ChildStderr, state: Arc<RuntimeState>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => state.stderr.lock().push(line),
                Ok(None) => return,
                Err(error) => {
                    debug!(
                        program = %state.program,
                        %error,
                        "member stderr read failed"
                    );
                    return;
                }
            }
        }
    })
}

/// Watches the cancellation token and tears the child down when it fires.
fn spawn_cancellation_watcher(
    state: Arc<RuntimeState>,
    cancel: CancellationToken,
    kill: CancellationToken,
    bound: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        tokio::select! {
            () = cancel.cancelled() => {
                debug!(
                    program = %state.program,
                    "member process cancelled; killing it"
                );
                kill.cancel();
                if tokio::time::timeout(bound, wait_for_termination(&state)).await.is_err() {
                    warn!(
                        program = %state.program,
                        ?bound,
                        "cancelled member process did not exit within the bound"
                    );
                    if !state.is_terminated() {
                        state.terminate(Termination::stop_timeout(&state.program));
                    }
                }
            }
            () = wait_for_termination(&state) => {}
        }
    })
}
