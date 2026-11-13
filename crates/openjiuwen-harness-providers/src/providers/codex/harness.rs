//! The Codex subprocess transport.
//!
//! [`CodexDriver`] implements [`VendorDriver`] over the `codex app-server` JSON-RPC protocol: it
//! spawns one child process per start/stop cycle, sends one JSON message per line, and maps the
//! app-server's notifications and server requests onto the harness SPI. The generic
//! [`VendorHarness`](crate::vendor_cli::VendorHarness) shell owns everything provider-independent
//! (turn queue, lifecycle events, capability gating, checkpoints), so this module only speaks the
//! vendor protocol.
//!
//! # Transport contract (Rust)
//!
//! The executable is `config.codex_bin` when set, otherwise the `codex` on `PATH`; a missing binary
//! is a typed [`HarnessError::Protocol`] naming `codex`, never an unimplemented capability. The
//! argv is:
//!
//! 1. `app-server --listen stdio://` — the newline-delimited JSON-RPC app-server;
//! 2. one `-c key=value` per configuration override, in this order: the external endpoint's
//!    `model_provider` / `model_providers.<key>.{name,base_url,env_key}` group,
//!    `features.default_mode_request_user_input=true` when the host declared
//!    [`HostCapability::UserInput`], one `mcp_servers.<key>.{command,args,url}` group per host MCP
//!    server, and finally `config.config_overrides` verbatim. Values are JSON-encoded so Codex's
//!    TOML parser reads them as strings or arrays instead of bare tokens; `<key>` decays `-` to
//!    `_` because Codex reads the dotted segment as a TOML bare key.
//!
//! The process environment is inherited when `inherit_process_env` is set, then `config.env`, then
//! `context.env`, then `OPENJIUWEN_CODEX_API_KEY` from the endpoint's `api_key`. The working
//! directory is `config.cwd`, else `context.cwd`.
//!
//! # Wire contract
//!
//! [`open`](VendorDriver::open) performs the handshake — `initialize` with this adapter's
//! `clientInfo`, then the `initialized` notification — and creates the thread:
//! `thread/start` (or `thread/resume` when a checkpoint is resumed) with the ephemeral flag, the
//! working directory, the model and provider label, the host system prompt as developer
//! instructions, the approval policy and sandbox mode, and a `config` object carrying the thread
//! configuration plus `model_reasoning_effort`. An external endpoint (or
//! `bypass_approvals_and_sandbox`) selects `never` / `danger-full-access`, everything else
//! `on-request` / `workspace-write`. A resumed thread whose echo names another thread id is
//! `Protocol("Codex resumed unexpected thread '<id>'")`.
//!
//! A turn is one `turn/start` request carrying the text input, followed by a read loop over
//! [`JsonRpcClient::next_incoming`]:
//!
//! - `item/agentMessage/delta` projects answer deltas, `item/reasoning/summaryTextDelta` and
//!   `item/reasoning/textDelta` reasoning deltas;
//! - `item/started` / `item/completed` project the tool lifecycle of `commandExecution` (`shell`),
//!   `dynamicToolCall`, `fileChange` (`apply_patch`) and `mcpToolCall` (`<server>.<tool>`), keyed by
//!   the item id, plus the completed `agentMessage` / `reasoning` blocks;
//! - `thread/tokenUsage/updated` publishes delta usage;
//! - `error` records a pending failure and publishes a diagnostic, downgraded to a warning while
//!   Codex retries;
//! - `turn/completed` terminates the turn (`completed`, `interrupted`, `failed`);
//! - `rawResponseItem/completed` and `rawResponse/completed` are dropped, every other notification
//!   is forwarded as a `codex` provider event;
//! - server-to-client requests are answered here: the two `…/requestApproval` methods through the
//!   host's tool-approval policy, `item/tool/requestUserInput` through the host's user-input
//!   policy, and `item/tool/call` (host dynamic tools) with a JSON-RPC "method not found" error —
//!   as is every unknown method.
//!
//! A stream that ends before `turn/completed` fails the turn with
//! [`MISSING_TURN_COMPLETED_CODE`](frames::MISSING_TURN_COMPLETED_CODE) and the stderr tail.
//!
//! Steering sends `turn/steer` with the active native turn id; a "no active turn" peer error
//! becomes `State("the Codex turn ended before the steer was accepted")`. Aborting sends
//! `turn/interrupt` under a five-second bound and marks the turn host-aborted, so the terminal
//! result is an `Aborted` turn even when the stream ends first.
//!
//! # Session and checkpoints
//!
//! The thread id is the provider session id and the checkpoint payload is
//! `{"thread_id": <id>, "resumed": <bool>}`. [`open`](VendorDriver::open) publishes it through
//! `context.checkpoint_sink` with [`CheckpointReason::SessionActivated`] (a sink failure is logged,
//! never fatal) and resumes from `checkpoint_str(checkpoint, "thread_id")`.
//!
//! # Cancellation
//!
//! The app-server is a long-lived child: [`close`](VendorDriver::close) closes the JSON-RPC client,
//! which aborts the reader task and drops the child process (reaped by `kill_on_drop`). An
//! in-flight turn then observes end of stream and reports an `Aborted` turn with
//! [`TurnTerminationKind::HarnessStop`], so `stop` joins its supervisor promptly.
//!
//! # Files
//!
//! This module owns the driver lifecycle and the turn loop; `frames` projects the app-server's
//! messages onto harness observations and `wire` builds the invocation and thread options.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    empty_json_object, AbortMode, CheckpointReason, HarnessCheckpoint, HarnessContext,
    HarnessError, HarnessInput, HarnessInteractionHandler, HarnessResult, JsonObject, JsonValue,
    ResumePolicy, TurnResult, TurnTerminationKind,
};
use serde_json::json;
use tokio::sync::Mutex as AsyncMutex;

use crate::input::harness_input_text;
use crate::skills::install_for_provider;
use crate::vendor_cli::{
    checkpoint_str, failed_turn, interrupted_turn, turn_error, ChildProcess, ChildProcessConfig,
    JsonRpcClient, PermissionPolicy, TurnContext, TurnRequest, VendorDriver,
};

use super::frames::{
    FrameAction, TurnAccumulator, INTERRUPTED_MESSAGE, MISSING_TURN_COMPLETED_CODE,
};
use super::wire::{build_args, build_env, params_value, thread_params};
use super::{CodexHarnessConfig, CARD_NAME};

/// Human-readable harness label used in every error message.
pub(super) const LABEL: &str = "Codex";

/// Provider name stamped on provider extension events.
pub(super) const PROVIDER_EVENT_NAME: &str = CARD_NAME;

/// Coarse failure category of an SDK/transport level failure.
pub(super) const SDK_ERROR_CATEGORY: &str = "sdk_error";

/// Schema version of the provider's checkpoint payload.
const CHECKPOINT_SCHEMA_VERSION: &str = "1";

/// Client name this adapter reports in the app-server handshake.
const CLIENT_NAME: &str = "openjiuwen-harness";

/// Client version this adapter reports in the app-server handshake.
const CLIENT_VERSION: &str = "1";

/// Upper bound of one `turn/interrupt` request.
const INTERRUPT_TIMEOUT: Duration = Duration::from_secs(5);

/// Session facts learned at `open`.
#[derive(Default)]
pub(super) struct DriverState {
    /// Thread id of the current cycle, which is also the provider session id.
    thread_id: Option<String>,
    /// Native turn id of the running turn, used by steer and interrupt.
    active_turn_id: Option<String>,
    /// Host session the cycle belongs to, for checkpoint envelopes.
    host_session_id: Option<String>,
    /// Agent the cycle belongs to, for checkpoint envelopes.
    agent_id: Option<String>,
    /// What the host allows the app-server to ask this cycle.
    pub(super) policy: PermissionPolicy,
    /// Host interaction handler answering the app-server's prompts.
    pub(super) handler: Option<std::sync::Arc<dyn HarnessInteractionHandler>>,
}

/// The Codex app-server driver.
///
/// One instance backs one [`VendorHarness`](crate::vendor_cli::VendorHarness) handle: the CLI is
/// spawned by [`open`](VendorDriver::open), spoken to by [`run_turn`](VendorDriver::run_turn) and
/// reaped by [`close`](VendorDriver::close).
pub struct CodexDriver {
    /// Validated configuration decoded by the provider's `create`.
    config: CodexHarnessConfig,
    /// Resolved `codex` executable.
    binary: PathBuf,
    /// The JSON-RPC client of the current cycle.
    rpc: AsyncMutex<Option<std::sync::Arc<JsonRpcClient>>>,
    /// Session facts learned at `open`.
    pub(super) state: Mutex<DriverState>,
    /// Whether the host asked for the running turn to stop.
    host_abort: AtomicBool,
    /// Whether the cycle is closing, so an ending stream reports a harness stop.
    closing: AtomicBool,
    /// Last allocated checkpoint sequence.
    sequence: AtomicU64,
}

impl CodexDriver {
    /// Builds the driver over a validated configuration and a resolved executable.
    #[must_use]
    pub fn new(config: CodexHarnessConfig, binary: PathBuf) -> Self {
        Self {
            config,
            binary,
            rpc: AsyncMutex::new(None),
            state: Mutex::new(DriverState::default()),
            host_abort: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
        }
    }

    /// Returns the JSON-RPC client of the current cycle.
    async fn client(&self) -> HarnessResult<std::sync::Arc<JsonRpcClient>> {
        self.rpc
            .lock()
            .await
            .clone()
            .ok_or_else(|| HarnessError::protocol(format!("the {LABEL} harness is not started")))
    }

    /// Returns the thread id of the current cycle.
    fn thread_id(&self) -> HarnessResult<String> {
        lock(&self.state)
            .thread_id
            .clone()
            .ok_or_else(|| HarnessError::protocol(format!("the {LABEL} harness is not started")))
    }

    /// Takes the host-abort flag, clearing it.
    pub(super) fn take_host_abort(&self) -> bool {
        self.host_abort.swap(false, Ordering::SeqCst)
    }

    /// Returns the next checkpoint sequence.
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Builds the checkpoint payload of thread `thread_id`.
    fn checkpoint(&self, thread_id: &str, resumed: bool) -> HarnessCheckpoint {
        let state = lock(&self.state);
        let mut data = JsonObject::new();
        data.insert("thread_id".to_owned(), json!(thread_id));
        data.insert("resumed".to_owned(), json!(resumed));
        HarnessCheckpoint {
            provider: CARD_NAME.to_owned(),
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            agent_id: state.agent_id.clone().unwrap_or_default(),
            host_session_id: state.host_session_id.clone().unwrap_or_default(),
            // The checkpoint id is the thread id, so a later resume resolves to the same thread.
            checkpoint_id: thread_id.to_owned(),
            sequence: self.next_sequence(),
            data,
            provider_session_id: Some(thread_id.to_owned()),
            revision: None,
        }
    }

    /// Resolves the thread id to resume, applying the host's resume policy.
    fn resume_thread_id(&self, context: &HarnessContext) -> HarnessResult<Option<String>> {
        let checkpoint = context
            .checkpoint
            .as_ref()
            .and_then(|checkpoint| checkpoint_str(checkpoint, "thread_id"));
        match context.resume_policy {
            ResumePolicy::New => Ok(None),
            ResumePolicy::ResumeIfAvailable => Ok(checkpoint),
            ResumePolicy::RequireResume => checkpoint.map(Some).ok_or_else(|| {
                HarnessError::protocol("Codex checkpoint does not carry a thread id to resume")
            }),
        }
    }

    /// Performs the handshake and creates or resumes the thread, returning its id.
    async fn open_thread(
        &self,
        client: &std::sync::Arc<JsonRpcClient>,
        context: &HarnessContext,
        cwd: Option<&std::path::Path>,
        resume: Option<&str>,
    ) -> HarnessResult<String> {
        client
            .request(
                "initialize",
                json!({"clientInfo": {
                    "name": CLIENT_NAME,
                    "title": self.config.client_title,
                    "version": CLIENT_VERSION,
                }}),
            )
            .await?;
        client.notify("initialized", json!({})).await?;
        let params = thread_params(&self.config, context, cwd, resume);
        let method = if resume.is_some() {
            "thread/resume"
        } else {
            "thread/start"
        };
        let result = client.request(method, params_value(&params)).await?;
        let thread_id = result
            .get("thread")
            .and_then(|thread| thread.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        let Some(thread_id) = thread_id else {
            return Err(HarnessError::protocol(format!(
                "Codex did not report the thread id of '{method}'"
            )));
        };
        if let Some(expected) = resume {
            if thread_id != expected {
                return Err(HarnessError::protocol(format!(
                    "Codex resumed unexpected thread '{thread_id}'"
                )));
            }
        }
        Ok(thread_id)
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Clears the active turn id when the turn ends.
struct ActiveTurnGuard<'a> {
    /// Driver whose turn id is cleared.
    driver: &'a CodexDriver,
}

impl Drop for ActiveTurnGuard<'_> {
    fn drop(&mut self) {
        lock(&self.driver.state).active_turn_id = None;
    }
}

/// Appends `tail` to `message` when the CLI produced stderr output.
pub(super) fn with_tail(message: String, tail: &str) -> String {
    if tail.is_empty() {
        message
    } else {
        format!("{message}\n{tail}")
    }
}

#[async_trait]
impl VendorDriver for CodexDriver {
    fn label(&self) -> &'static str {
        LABEL
    }

    /// Installs skills, spawns the app-server, performs the handshake and opens the thread.
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
                install_for_provider(&sources, "codex", Some(&project), conflict)
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
        let resume = self.resume_thread_id(context)?;
        let args = build_args(&self.config, context);
        let env = build_env(&self.config, context);
        let process = ChildProcess::spawn(ChildProcessConfig {
            program: &self.binary,
            args: &args,
            cwd: cwd.as_deref(),
            env: &env,
            inherit_env: self.config.inherit_process_env,
            label: LABEL,
            max_line_bytes: 0,
        })
        .await?;
        let client = JsonRpcClient::spawn(Box::new(process), LABEL);
        let opened = self
            .open_thread(&client, context, cwd.as_deref(), resume.as_deref())
            .await;
        let thread_id = match opened {
            Ok(thread_id) => thread_id,
            Err(error) => {
                client.close().await;
                return Err(error);
            }
        };
        {
            let mut state = lock(&self.state);
            state.thread_id = Some(thread_id.clone());
            state.active_turn_id = None;
            state.host_session_id = Some(context.host_session_id.clone());
            state.agent_id = Some(context.agent_id.clone());
            state.policy = policy;
            state.handler.clone_from(&context.interactions);
        }
        self.closing.store(false, Ordering::SeqCst);
        self.host_abort.store(false, Ordering::SeqCst);
        *self.rpc.lock().await = Some(std::sync::Arc::clone(&client));
        if let Some(sink) = context.checkpoint_sink.as_ref() {
            let checkpoint = self.checkpoint(&thread_id, resume.is_some());
            if let Err(error) = sink
                .save(&checkpoint, CheckpointReason::SessionActivated, None)
                .await
            {
                tracing::warn!(%error, "failed to publish the Codex session checkpoint");
            }
        }
        Ok(Some(thread_id))
    }

    /// Closes the JSON-RPC client, which reaps the child process and ends a running turn.
    async fn close(&self) -> HarnessResult<()> {
        self.closing.store(true, Ordering::SeqCst);
        let client = self.rpc.lock().await.take();
        {
            let mut state = lock(&self.state);
            state.active_turn_id = None;
            state.thread_id = None;
        }
        if let Some(client) = client {
            client.close().await;
        }
        Ok(())
    }

    /// Runs one turn: starts it and folds every notification until `turn/completed`.
    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        let text = harness_input_text(&HarnessInput {
            content: turn.content.clone(),
            metadata: turn.metadata.clone(),
        });
        let client = self.client().await?;
        let thread_id = self.thread_id()?;
        if self.closing.load(Ordering::SeqCst) {
            return Ok(interrupted_turn(
                TurnTerminationKind::HarnessStop,
                format!("the {LABEL} harness is stopping"),
            ));
        }
        self.host_abort.store(false, Ordering::SeqCst);
        let started = client
            .request(
                "turn/start",
                json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": text}],
                }),
            )
            .await?;
        let native_turn_id = started
            .get("turn")
            .and_then(|turn| turn.get("id"))
            .and_then(JsonValue::as_str)
            .map(str::to_owned);
        lock(&self.state).active_turn_id = native_turn_id;
        let _guard = ActiveTurnGuard { driver: self };
        let mut accumulator = TurnAccumulator::default();
        loop {
            let Some(message) = client.next_incoming().await else {
                if self.closing.load(Ordering::SeqCst) {
                    return Ok(interrupted_turn(
                        TurnTerminationKind::HarnessStop,
                        format!("the {LABEL} harness is stopping"),
                    ));
                }
                if self.take_host_abort() {
                    return Ok(interrupted_turn(
                        TurnTerminationKind::UserAbort,
                        INTERRUPTED_MESSAGE,
                    ));
                }
                let message = with_tail(
                    "Codex ended the stream without completing the turn".to_owned(),
                    &client.stderr_tail(),
                );
                return Ok(failed_turn(turn_error(
                    message,
                    MISSING_TURN_COMPLETED_CODE,
                    SDK_ERROR_CATEGORY,
                    false,
                )));
            };
            match self
                .dispatch(&message, &turn.turn_id, session, &mut accumulator, &client)
                .await?
            {
                FrameAction::Continue => {}
                FrameAction::Terminal(result) => return Ok(*result),
            }
        }
    }

    /// Sends `turn/steer` with the active native turn id.
    async fn steer(
        &self,
        _turn: &TurnRequest,
        content: &JsonValue,
        _session: &TurnContext,
    ) -> HarnessResult<()> {
        let text = harness_input_text(&HarnessInput {
            content: content.clone(),
            metadata: empty_json_object(),
        });
        let client = self.client().await?;
        let thread_id = self.thread_id()?;
        let Some(native_turn_id) = lock(&self.state).active_turn_id.clone() else {
            return Err(HarnessError::state("there is no active Codex turn"));
        };
        client
            .request(
                "turn/steer",
                json!({
                    "threadId": thread_id,
                    "turnId": native_turn_id,
                    "input": [{"type": "text", "text": text}],
                }),
            )
            .await
            .map_err(|error| {
                if error
                    .message()
                    .to_ascii_lowercase()
                    .contains("no active turn")
                {
                    HarnessError::state("the Codex turn ended before the steer was accepted")
                } else {
                    error
                }
            })?;
        Ok(())
    }

    /// Sends `turn/interrupt` for the running turn and marks it host-aborted.
    ///
    /// Both abort modes use the same request: the app-server exposes no harder stop than the
    /// interrupt it acknowledges with an `interrupted` turn.
    async fn interrupt(&self, _mode: AbortMode, _session: &TurnContext) -> HarnessResult<()> {
        self.host_abort.store(true, Ordering::SeqCst);
        let client = self.client().await?;
        let (thread_id, native_turn_id) = {
            let state = lock(&self.state);
            (state.thread_id.clone(), state.active_turn_id.clone())
        };
        let (Some(thread_id), Some(native_turn_id)) = (thread_id, native_turn_id) else {
            return Ok(());
        };
        if let Err(error) = client
            .request_with_timeout(
                "turn/interrupt",
                json!({"threadId": thread_id, "turnId": native_turn_id}),
                INTERRUPT_TIMEOUT,
            )
            .await
        {
            tracing::warn!(%error, "the Codex interrupt request failed");
        }
        Ok(())
    }

    /// Returns the current thread as a checkpoint, when a cycle is open.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        match lock(&self.state).thread_id.clone() {
            Some(thread_id) => Ok(Some(self.checkpoint(&thread_id, false))),
            None => Ok(None),
        }
    }
}
