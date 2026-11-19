//! The Claude Code subprocess transport.
//!
//! [`ClaudeDriver`] implements [`VendorDriver`] over the `claude` CLI in its stream-json mode: it
//! spawns one child process per start/stop cycle, writes one JSON request per line to stdin, reads
//! one JSON message per line from stdout, and maps those frames onto the harness SPI. The generic
//! [`VendorHarness`](crate::vendor_cli::VendorHarness) shell owns everything provider-independent
//! (turn queue, lifecycle events, capability gating, checkpoints), so this module only speaks the
//! vendor protocol.
//!
//! # Transport contract (Rust)
//!
//! The CLI is started once per cycle with this argv, in this order:
//!
//! 1. `--print --input-format stream-json --output-format stream-json --verbose` — one-shot
//!    streaming mode;
//! 2. `--include-partial-messages` when `include_partial_messages` is set;
//! 3. `--permission-mode <wire>` — the configured enum value (`default`, `acceptEdits`, `plan`,
//!    `bypassPermissions`, `dontAsk`, `auto`), overridden to `default` whenever the host declared an
//!    interactive capability so the CLI consults the stdio permission tool instead of its own
//!    bypass mode;
//! 4. `--permission-prompt-tool stdio` when the host declared
//!    [`HostCapability::ToolApproval`](openjiuwen_harness_protocol::HostCapability::ToolApproval) or
//!    [`HostCapability::UserInput`](openjiuwen_harness_protocol::HostCapability::UserInput);
//! 5. `--model <model>` and `--effort <effort>` from `config.model`, when set;
//! 6. `--resume <id>` when resuming, otherwise `--session-id <id>`;
//! 7. `--max-turns <n>` when configured;
//! 8. one `--add-dir <dir>` per additional directory;
//! 9. `--append-system-prompt <text>` (append mode) or `--system-prompt <text>` (replace mode) when
//!    the host supplied a system prompt;
//! 10. `--settings <compact json>` when the endpoint/`settings_env` pair or `config.settings`
//!     produces anything: the generated environment goes under `{"env": {…}}` and a
//!     `config.settings` object is merged over it (its top-level keys, and its own `env` entries,
//!     win).
//!
//! The process environment is inherited when `inherit_process_env` is set, then `config.env`, then
//! `context.env`. The working directory is `config.cwd`, else `context.cwd`.
//!
//! # Wire contract
//!
//! Every turn writes one user frame and then reads until the turn settles:
//!
//! ```json
//! {"type":"user","message":{"role":"user","content":[{"type":"text","text":"…"}]},
//!  "parent_tool_use_id":null,"uuid":"message-…","session_id":"…"}
//! ```
//!
//! Frames are handled as follows: `system/init` captures and emits the provider session id,
//! `assistant` projects text/thinking outputs, tool calls and usage (and classifies a reported
//! `error`), `user` projects tool results, `stream_event` projects delta outputs (partial messages
//! only), `control_request/can_use_tool` answers the host's permission policy, `control_response` is
//! ignored, `result` records the turn's terminal outcome, and any other frame — including the
//! `command_lifecycle` delivery receipts — is forwarded as a provider event (nothing is silently
//! dropped).
//!
//! # Turn boundary
//!
//! A `result` frame alone does not end the turn. The CLI acknowledges every user message with a
//! `command_lifecycle` receipt keyed by the frame `uuid`, and a message the host steered into an
//! idle CLI is answered in a *new* cycle carrying a result of its own; ending at the first result
//! would leave that cycle's output outside the turn. The loop therefore feeds every inbound frame to
//! a [`TurnCycleTracker`] and stops when the tracker says the turn settled: a result arrived and no
//! submitted message is still outstanding. A CLI build that reports no receipts at all settles on
//! the first result, exactly as before. After `lifecycle_ack_timeout_s` a message the CLI never
//! acknowledged stops blocking the turn, and the abandonment is published as a `warning` diagnostic.
//!
//! # Cancellation
//!
//! A turn owns the child process for its whole duration and multiplexes between the next stdout
//! line, its control channel (steer, interrupt, close) and the acknowledgement deadline. A control
//! command therefore wins over a blocked read. `ChildProcess::read_json` is resumable after such a
//! cancellation, but a partially read line is discarded: commands normally arrive while the CLI is
//! idle, and the deadline only arms when the CLI has acknowledged nothing at all, so no frame is in
//! flight on either path.
//!
//! # Files
//!
//! This module owns the driver lifecycle and the turn loop; `frames` projects the CLI's frames
//! onto harness observations, `lifecycle` decides the turn boundary, `failure` classifies CLI
//! failures and `wire` builds the process invocation (argv, `--settings`, environment).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    empty_json_object, AbortMode, CheckpointReason, DiagnosticLevel, HarnessCheckpoint,
    HarnessContext, HarnessError, HarnessInput, HarnessInteractionHandler, HarnessResult,
    JsonObject, JsonValue, TurnError, TurnResult, TurnTerminationKind,
};
use serde_json::json;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::input::harness_input_text;
use crate::skills::install_for_provider;
use crate::vendor_cli::child::DEFAULT_MAX_LINE_BYTES;
use crate::vendor_cli::{
    checkpoint_str, derive_session_id, failed_turn, interrupted_turn, resolve_resume_id,
    ChildProcess, ChildProcessConfig, PermissionPolicy, TurnContext, TurnRequest, VendorDriver,
};

use super::failure::{classify_cli_failure, FailurePhase};
use super::frames::TurnAccumulator;
use super::lifecycle::TurnCycleTracker;
use super::wire::{build_args, build_env};
use super::{ClaudeCodeHarnessConfig, CARD_NAME, DEFAULT_CLAUDE_MAX_BUFFER_SIZE};

/// Human-readable harness label used in every error message.
const LABEL: &str = "Claude Code";

/// Provider name stamped on provider extension events.
pub(super) const PROVIDER_EVENT_NAME: &str = CARD_NAME;

/// Schema version of the provider's checkpoint payload.
const CHECKPOINT_SCHEMA_VERSION: &str = "1";

/// Normalized failure code of a stream that ended without a terminal result.
const MISSING_RESULT_CODE: &str = "CLAUDE_MISSING_RESULT";

/// Normalized failure code of a malformed CLI stream.
const TRANSPORT_CODE: &str = "CLAUDE_TRANSPORT";

/// Error type name reported when the CLI's message stream ends before the turn does.
const STREAM_ENDED_TYPE: &str = "CLIConnectionError";

/// Error type name reported when a stdout line cannot be read or decoded.
const READ_FAILED_TYPE: &str = "ProcessError";

/// How long `close` waits for a running turn to release the child process.
const CLOSE_LOCK_TIMEOUT: Duration = Duration::from_secs(4);

/// Command accepted by a running turn.
#[derive(Debug)]
enum TurnCommand {
    /// Write `text` as another user frame (host steering).
    Steer {
        /// Frame id minted for the steered message, so its receipts are attributable.
        message_id: String,
        /// Text projection of the steered input.
        text: String,
    },
    /// Interrupt the turn with a `control_request`.
    Interrupt,
    /// Release the turn so `close` can reap the process.
    Close,
}

/// Control channel of the active turn, plus the closing flag.
#[derive(Default)]
struct TurnLink {
    /// Sender of the running turn; `None` when no turn is active.
    sender: Option<mpsc::UnboundedSender<TurnCommand>>,
    /// Whether `close` asked the harness to stop.
    closing: bool,
}

/// Session facts learned at `open`.
#[derive(Default)]
pub(super) struct DriverState {
    /// Provider session id of the current cycle.
    session_id: Option<String>,
    /// Host session the cycle belongs to, for checkpoint envelopes.
    host_session_id: Option<String>,
    /// Agent the cycle belongs to, for checkpoint envelopes.
    agent_id: Option<String>,
    /// What the host allows the CLI to ask this cycle.
    pub(super) policy: PermissionPolicy,
    /// Host interaction handler answering the CLI's permission prompts.
    pub(super) handler: Option<std::sync::Arc<dyn HarnessInteractionHandler>>,
    /// Whether the "different session id" warning was already emitted.
    adopted_warned: bool,
}

/// The Claude Code subprocess driver.
///
/// One instance backs one [`VendorHarness`](crate::vendor_cli::VendorHarness) handle: the CLI is
/// spawned by [`open`](VendorDriver::open), spoken to by [`run_turn`](VendorDriver::run_turn) and
/// reaped by [`close`](VendorDriver::close).
pub struct ClaudeDriver {
    /// Validated configuration decoded by the provider's `create`.
    pub(super) config: ClaudeCodeHarnessConfig,
    /// Resolved `claude` executable.
    binary: PathBuf,
    /// The live child process, owned by the running turn while one is active.
    process: tokio::sync::Mutex<Option<ChildProcess>>,
    /// Control channel and closing flag of the current turn.
    turn_link: Mutex<TurnLink>,
    /// Session facts learned at `open`.
    pub(super) state: Mutex<DriverState>,
    /// Whether an interrupt was requested; the turn observes it even before its channel is read.
    interrupt_requested: AtomicBool,
    /// Last allocated checkpoint sequence.
    sequence: AtomicU64,
    /// Number of steered messages the current cycle wrote, for their distinct frame ids.
    steers: AtomicU64,
    /// Receipt tracker of the current cycle; taken by the running turn and restored when it ends.
    cycle_tracker: Mutex<Option<TurnCycleTracker>>,
}

impl ClaudeDriver {
    /// Builds the driver over a validated configuration and a resolved executable.
    #[must_use]
    pub fn new(config: ClaudeCodeHarnessConfig, binary: PathBuf) -> Self {
        Self {
            config,
            binary,
            process: tokio::sync::Mutex::new(None),
            turn_link: Mutex::new(TurnLink::default()),
            state: Mutex::new(DriverState::default()),
            interrupt_requested: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
            steers: AtomicU64::new(0),
            cycle_tracker: Mutex::new(None),
        }
    }

    /// Returns the current provider session id, when the cycle has one.
    fn session_id(&self) -> Option<String> {
        lock(&self.state).session_id.clone()
    }

    /// Stores `id` as the provider session id, warning once when it changes.
    pub(super) fn adopt_session(&self, id: &str) {
        let mut state = lock(&self.state);
        let current = state.session_id.clone();
        match current.as_deref() {
            Some(current) if current == id => {}
            Some(current) => {
                if !state.adopted_warned {
                    state.adopted_warned = true;
                    tracing::warn!(
                        configured = current,
                        reported = id,
                        "the Claude Code CLI reported a different session id; adopting it"
                    );
                }
                state.session_id = Some(id.to_owned());
            }
            None => state.session_id = Some(id.to_owned()),
        }
    }

    /// Returns the next checkpoint sequence.
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Builds the checkpoint payload of the current session.
    fn checkpoint(&self, session_id: &str, resumed: bool) -> HarnessCheckpoint {
        let state = lock(&self.state);
        let mut data = JsonObject::new();
        data.insert("session_id".to_owned(), json!(session_id));
        data.insert("resumed".to_owned(), json!(resumed));
        HarnessCheckpoint {
            provider: CARD_NAME.to_owned(),
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            agent_id: state.agent_id.clone().unwrap_or_default(),
            host_session_id: state.host_session_id.clone().unwrap_or_default(),
            // The checkpoint id is the provider session id, so a later resume resolves to it.
            checkpoint_id: session_id.to_owned(),
            sequence: self.next_sequence(),
            data,
            provider_session_id: Some(session_id.to_owned()),
            revision: None,
        }
    }

    /// Returns the user frame written for one input (`_submit` in Python).
    fn user_frame(&self, message_id: &str, text: &str) -> JsonValue {
        let session_id = self.session_id().map_or(JsonValue::Null, JsonValue::String);
        json!({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            "parent_tool_use_id": JsonValue::Null,
            "uuid": message_id,
            "session_id": session_id,
        })
    }

    /// Sends `command` to the running turn, if there is one.
    fn send_command(&self, command: TurnCommand) -> HarnessResult<()> {
        let sender = lock(&self.turn_link).sender.clone();
        match sender {
            Some(sender) => sender
                .send(command)
                .map_err(|_| HarnessError::state("the Claude Code turn control channel is closed")),
            None => Err(HarnessError::state("there is no active Claude Code turn")),
        }
    }

    /// Returns the acknowledgement timeout of an unacknowledged message.
    fn ack_timeout(&self) -> Duration {
        Duration::try_from_secs_f64(self.config.lifecycle_ack_timeout_s)
            .unwrap_or_else(|_| Duration::from_secs_f64(DEFAULT_ACK_TIMEOUT_S))
    }

    /// Starts a fresh receipt tracker for a new CLI cycle (Python's `_connect`).
    fn install_cycle_tracker(&self) {
        *lock(&self.cycle_tracker) = Some(TurnCycleTracker::new(self.ack_timeout()));
    }

    /// Takes the cycle's tracker and starts tracking `turn_id`.
    ///
    /// The tracker is owned by the turn while it runs so its state can be advanced across awaits;
    /// [`TurnCycleGuard`] puts it back when the turn ends.
    fn begin_cycle(&self, turn_id: &str) -> TurnCycleGuard<'_> {
        let mut tracker = lock(&self.cycle_tracker)
            .take()
            .unwrap_or_else(|| TurnCycleTracker::new(self.ack_timeout()));
        tracker.begin_turn(turn_id);
        TurnCycleGuard {
            driver: self,
            tracker: Some(tracker),
        }
    }

    /// Mints the frame id of a steered message.
    ///
    /// The CLI echoes this id back as the `command_uuid` of its delivery receipts, which is what
    /// ties a receipt to the message it answers; reusing the turn's own id would make a steered
    /// message indistinguishable from the one that opened the turn.
    fn mint_steer_id(&self, turn_message_id: &str) -> String {
        let sequence = self.steers.fetch_add(1, Ordering::SeqCst) + 1;
        format!("{turn_message_id}-steer-{sequence}")
    }

    /// Writes one user frame, recording its id for the CLI's receipts before it goes out.
    async fn submit(
        &self,
        cycle: &mut TurnCycleGuard<'_>,
        process: &mut ChildProcess,
        message_id: &str,
        text: &str,
    ) -> HarnessResult<()> {
        let frame = self.user_frame(message_id, text);
        if let (Some(object), Some(tracker)) = (frame.as_object(), cycle.tracker.as_mut()) {
            tracker.note_outbound(object);
        }
        process.write_json(&frame).await
    }

    /// Writes the interrupt request and waits for the CLI's terminal acknowledgement.
    async fn finish_interrupt(
        &self,
        process: &mut ChildProcess,
        receiver: &mut mpsc::UnboundedReceiver<TurnCommand>,
    ) -> HarnessResult<TurnResult> {
        let frame = json!({
            "type": "control_request",
            "request_id": Uuid::new_v4().to_string(),
            "request": {"subtype": "interrupt"},
        });
        process.write_json(&frame).await?;
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs_f64(self.config.lifecycle_ack_timeout_s);
        loop {
            let read = process.read_json();
            tokio::pin!(read);
            let outcome = tokio::select! {
                () = tokio::time::sleep_until(deadline) => InterruptOutcome::Timeout,
                command = receiver.recv() => InterruptOutcome::Command(command),
                frame = &mut read => InterruptOutcome::Frame(frame),
            };
            match outcome {
                InterruptOutcome::Timeout => {
                    return Ok(interrupted_turn(
                        TurnTerminationKind::Timeout,
                        "Claude Code did not acknowledge the interrupt",
                    ))
                }
                InterruptOutcome::Command(None | Some(TurnCommand::Close))
                | InterruptOutcome::Frame(Ok(None) | Err(_)) => {
                    return Ok(interrupted_turn(
                        TurnTerminationKind::UserAbort,
                        INTERRUPTED_MESSAGE,
                    ))
                }
                InterruptOutcome::Command(Some(
                    TurnCommand::Interrupt | TurnCommand::Steer { .. },
                )) => {}
                InterruptOutcome::Frame(Ok(Some(value))) => {
                    if value.get("type").and_then(JsonValue::as_str) == Some("result") {
                        return Ok(interrupted_turn(
                            TurnTerminationKind::UserAbort,
                            INTERRUPTED_MESSAGE,
                        ));
                    }
                }
            }
        }
    }
}

/// Message of a turn the host interrupted.
const INTERRUPTED_MESSAGE: &str = "Claude Code interrupted the turn";

/// Acknowledgement timeout used when the configuration carries no usable value.
const DEFAULT_ACK_TIMEOUT_S: f64 = 10.0;

/// Builds the failure of a CLI stream that died inside a turn.
///
/// The classification is the shared turn-phase one; the code keeps the adapter's normalized
/// identifier, which a host can match without knowing the vendor's exception names.
fn stream_failure(code: &str, message: &str, error_type: &str, stderr_tail: &str) -> TurnError {
    let mut error =
        classify_cli_failure(FailurePhase::Turn, error_type, message, Some(stderr_tail));
    error.code = Some(code.to_owned());
    error
}

/// Outcome of one interrupt-acknowledgement read.
enum InterruptOutcome {
    /// The acknowledgement deadline elapsed.
    Timeout,
    /// A control command arrived.
    Command(Option<TurnCommand>),
    /// A frame (or read failure) arrived.
    Frame(HarnessResult<Option<JsonValue>>),
}

/// Outcome of one main turn-loop iteration.
enum TurnStep {
    /// A control command arrived.
    Command(Option<TurnCommand>),
    /// A frame (or read failure) arrived.
    Frame(HarnessResult<Option<JsonValue>>),
    /// The acknowledgement deadline of an unanswered message elapsed.
    Deadline,
}

/// Waits for the next stdout frame, control command or acknowledgement deadline.
///
/// The read future is scoped to this call so cancelling it releases the process borrow; the
/// control channel is polled first so a queued command always wins over a blocked read, and the
/// deadline last so a frame that is already available is never traded for a timeout.
async fn next_step(
    process: &mut ChildProcess,
    receiver: &mut mpsc::UnboundedReceiver<TurnCommand>,
    deadline: Option<Duration>,
) -> TurnStep {
    let read = process.read_json();
    tokio::pin!(read);
    let expiry = async {
        match deadline {
            Some(wait) => tokio::time::sleep(wait).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(expiry);
    tokio::select! {
        biased;
        command = receiver.recv() => TurnStep::Command(command),
        frame = &mut read => TurnStep::Frame(frame),
        () = &mut expiry => TurnStep::Deadline,
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Clears the active turn's control channel when the turn ends.
struct TurnLinkGuard<'a> {
    /// Driver whose link is cleared.
    driver: &'a ClaudeDriver,
}

impl Drop for TurnLinkGuard<'_> {
    fn drop(&mut self) {
        lock(&self.driver.turn_link).sender = None;
    }
}

/// Ends the tracked turn and hands the receipt tracker back when the turn ends.
struct TurnCycleGuard<'a> {
    /// Driver the tracker belongs to.
    driver: &'a ClaudeDriver,
    /// Tracker owned by the running turn.
    tracker: Option<TurnCycleTracker>,
}

impl TurnCycleGuard<'_> {
    /// Returns the tracker of the running turn.
    fn tracker(&self) -> &TurnCycleTracker {
        self.tracker
            .as_ref()
            .expect("the turn owns its cycle tracker")
    }

    /// Returns the tracker of the running turn, mutably.
    fn tracker_mut(&mut self) -> &mut TurnCycleTracker {
        self.tracker
            .as_mut()
            .expect("the turn owns its cycle tracker")
    }
}

impl Drop for TurnCycleGuard<'_> {
    fn drop(&mut self) {
        if let Some(mut tracker) = self.tracker.take() {
            tracker.end_turn();
            *lock(&self.driver.cycle_tracker) = Some(tracker);
        }
    }
}

#[async_trait]
impl VendorDriver for ClaudeDriver {
    fn label(&self) -> &'static str {
        LABEL
    }

    /// Installs skills, spawns the CLI, publishes the session checkpoint and returns the session id.
    async fn open(
        &self,
        context: &HarnessContext,
        _session: &TurnContext,
    ) -> HarnessResult<Option<String>> {
        let cwd = self
            .config
            .cwd
            .clone()
            .or_else(|| context.cwd.as_deref().map(PathBuf::from));
        if !self.config.skills.is_empty() {
            let sources = self.config.skills.clone();
            let conflict = self.config.skill_conflict;
            let project = cwd.clone().unwrap_or_else(|| PathBuf::from("."));
            let installed = tokio::task::spawn_blocking(move || {
                install_for_provider(&sources, "claudecode", Some(&project), conflict)
            })
            .await
            .map_err(|error| {
                HarnessError::protocol(format!("{LABEL} skill installation task failed: {error}"))
            })?;
            installed.map_err(|error| {
                HarnessError::protocol(format!("{LABEL} skills could not be installed: {error}"))
            })?;
        }
        let policy = PermissionPolicy::from_context(context);
        let session_id =
            self.config.session_id.clone().unwrap_or_else(|| {
                derive_session_id(&context.host_session_id, &context.agent_name)
            });
        // Our own checkpoints name themselves after the provider session id; older payloads carry
        // it in `data.session_id`, which wins when both are present.
        let normalized = context.checkpoint.as_ref().map(|checkpoint| {
            match checkpoint_str(checkpoint, "session_id") {
                Some(session) => HarnessCheckpoint {
                    checkpoint_id: session,
                    ..checkpoint.clone()
                },
                None => checkpoint.clone(),
            }
        });
        let resume = resolve_resume_id(
            context.resume_policy,
            normalized.as_ref(),
            Some(session_id.clone()),
            "Claude Code cannot resume without a checkpoint or derived session id",
        )?;
        let resumed = resume.is_some();
        let args = build_args(
            &self.config,
            context,
            policy,
            resume.as_deref(),
            &session_id,
        );
        let provider_session_id = resume.unwrap_or_else(|| session_id.clone());
        let env = build_env(&self.config, context);
        // `max_buffer_size` is the per-line stdout ceiling. A `u64` that does not fit `usize` (only
        // possible on a 32-bit target) falls back to the transport's documented 32 MiB default
        // instead of disabling the ceiling.
        let max_line_bytes = usize::try_from(
            self.config
                .max_buffer_size
                .unwrap_or(DEFAULT_CLAUDE_MAX_BUFFER_SIZE),
        )
        .unwrap_or(DEFAULT_MAX_LINE_BYTES);
        {
            let mut state = lock(&self.state);
            state.session_id = Some(provider_session_id.clone());
            state.host_session_id = Some(context.host_session_id.clone());
            state.agent_id = Some(context.agent_id.clone());
            state.policy = policy;
            state.handler.clone_from(&context.interactions);
            state.adopted_warned = false;
        }
        lock(&self.turn_link).closing = false;
        self.interrupt_requested.store(false, Ordering::SeqCst);
        let process = ChildProcess::spawn(ChildProcessConfig {
            program: &self.binary,
            args: &args,
            cwd: cwd.as_deref(),
            env: &env,
            inherit_env: self.config.inherit_process_env,
            label: LABEL,
            max_line_bytes,
        })
        .await?;
        *self.process.lock().await = Some(process);
        self.install_cycle_tracker();
        if let Some(sink) = context.checkpoint_sink.as_ref() {
            let checkpoint = self.checkpoint(&provider_session_id, resumed);
            if let Err(error) = sink
                .save(&checkpoint, CheckpointReason::SessionActivated, None)
                .await
            {
                tracing::warn!(%error, "failed to publish the Claude Code session checkpoint");
            }
        }
        Ok(Some(provider_session_id))
    }

    /// Releases the child process, unblocking a running turn first.
    async fn close(&self) -> HarnessResult<()> {
        {
            let mut link = lock(&self.turn_link);
            link.closing = true;
            if let Some(sender) = &link.sender {
                let _ = sender.send(TurnCommand::Close);
            }
        }
        lock(&self.cycle_tracker).take();
        let Ok(mut guard) = tokio::time::timeout(CLOSE_LOCK_TIMEOUT, self.process.lock()).await
        else {
            return Err(HarnessError::state(format!(
                "the {LABEL} harness did not stop cleanly"
            )));
        };
        let Some(mut process) = guard.take() else {
            return Ok(());
        };
        drop(guard);
        process.close_stdin().await;
        if process.wait().await.is_none() {
            process.kill().await;
        }
        Ok(())
    }

    /// Runs one turn: writes the user frame and reads until the delivery receipts settle the turn.
    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        let text = harness_input_text(&HarnessInput {
            content: turn.content.clone(),
            metadata: turn.metadata.clone(),
        });
        let mut process_guard = self.process.lock().await;
        let Some(process) = process_guard.as_mut() else {
            return Err(HarnessError::protocol(format!(
                "the {LABEL} harness is not started"
            )));
        };
        let (sender, mut receiver) = mpsc::unbounded_channel();
        {
            let mut link = lock(&self.turn_link);
            if link.closing {
                return Ok(interrupted_turn(
                    TurnTerminationKind::HarnessStop,
                    format!("the {LABEL} harness is stopping"),
                ));
            }
            link.sender = Some(sender);
        }
        let _link_guard = TurnLinkGuard { driver: self };
        let mut cycle = self.begin_cycle(&turn.turn_id);
        self.submit(&mut cycle, process, &turn.message_id, &text)
            .await?;
        let mut accumulator = TurnAccumulator::default();
        if self.interrupt_requested.swap(false, Ordering::SeqCst) {
            return self.finish_interrupt(process, &mut receiver).await;
        }
        loop {
            let deadline = cycle.tracker().straggler_timeout();
            let outcome = next_step(process, &mut receiver, deadline).await;
            let frame = match outcome {
                TurnStep::Deadline => {
                    // A message the CLI never acknowledged stops blocking the turn; a settle here
                    // means nothing else is coming for it.
                    if cycle.tracker_mut().abandon_unacknowledged() {
                        break;
                    }
                    continue;
                }
                TurnStep::Command(Some(TurnCommand::Steer { message_id, text })) => {
                    self.submit(&mut cycle, process, &message_id, &text).await?;
                    continue;
                }
                TurnStep::Command(Some(TurnCommand::Interrupt)) => {
                    return self.finish_interrupt(process, &mut receiver).await;
                }
                TurnStep::Command(Some(TurnCommand::Close) | None) => {
                    return Ok(interrupted_turn(
                        TurnTerminationKind::HarnessStop,
                        format!("the {LABEL} harness is stopping"),
                    ));
                }
                TurnStep::Frame(frame) => frame,
            };
            if self.interrupt_requested.swap(false, Ordering::SeqCst) {
                return self.finish_interrupt(process, &mut receiver).await;
            }
            match frame {
                Ok(None) => {
                    return Ok(failed_turn(stream_failure(
                        MISSING_RESULT_CODE,
                        "Claude Code ended the response stream without a result",
                        STREAM_ENDED_TYPE,
                        &process.stderr_tail(),
                    )));
                }
                Err(error) => {
                    return Ok(failed_turn(stream_failure(
                        TRANSPORT_CODE,
                        error.message(),
                        READ_FAILED_TYPE,
                        &process.stderr_tail(),
                    )));
                }
                Ok(Some(value)) => {
                    let Some(frame) = value.as_object() else {
                        session
                            .provider_event(&turn.turn_id, PROVIDER_EVENT_NAME, "unknown", value)
                            .await;
                        continue;
                    };
                    self.dispatch(frame, &turn.turn_id, session, &mut accumulator, process)
                        .await?;
                    let settled = cycle.tracker_mut().note_inbound(frame);
                    if accumulator.has_result() && settled {
                        break;
                    }
                }
            }
        }
        let diagnostics = cycle.tracker_mut().drain_diagnostics();
        let Some(result) = accumulator.terminal_result() else {
            return Ok(failed_turn(stream_failure(
                MISSING_RESULT_CODE,
                "Claude Code ended the response stream without a result",
                STREAM_ENDED_TYPE,
                &process.stderr_tail(),
            )));
        };
        for reported in diagnostics {
            session
                .diagnostic(
                    &turn.turn_id,
                    DiagnosticLevel::Warning,
                    &format!("Claude Code {reported}"),
                    JsonObject::new(),
                )
                .await;
        }
        Ok(result)
    }

    /// Writes the steered content as another user frame.
    async fn steer(
        &self,
        turn: &TurnRequest,
        content: &JsonValue,
        _session: &TurnContext,
    ) -> HarnessResult<()> {
        let text = harness_input_text(&HarnessInput {
            content: content.clone(),
            metadata: empty_json_object(),
        });
        self.send_command(TurnCommand::Steer {
            message_id: self.mint_steer_id(&turn.message_id),
            text,
        })
    }

    /// Sends the CLI's interrupt request.
    ///
    /// Both abort modes send the same control request: the CLI exposes no harder stop than the
    /// interrupt it acknowledges with a terminal `result`.
    async fn interrupt(&self, _mode: AbortMode, _session: &TurnContext) -> HarnessResult<()> {
        self.interrupt_requested.store(true, Ordering::SeqCst);
        if lock(&self.turn_link).sender.is_some() {
            self.send_command(TurnCommand::Interrupt)?;
        }
        Ok(())
    }

    /// Returns the current provider session as a checkpoint, when a session is known.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        match self.session_id() {
            Some(session_id) => Ok(Some(self.checkpoint(&session_id, false))),
            None => Ok(None),
        }
    }
}
