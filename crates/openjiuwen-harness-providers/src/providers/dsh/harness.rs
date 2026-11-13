//! The `DeepSeek` Harness (DSH) subprocess transport.
//!
//! [`DshDriver`] implements [`VendorDriver`] over the `dsh` runtime in its JSON-RPC 2.0 stdio
//! mode: it spawns one child process per start/stop cycle, writes id-correlated requests, reads
//! notifications, and maps the session's events onto the harness SPI. The generic
//! [`VendorHarness`](crate::vendor_cli::VendorHarness) shell owns everything provider-independent
//! (turn queue, lifecycle events, capability gating), so this module only speaks the vendor
//! protocol.
//!
//! # Transport contract (Rust)
//!
//! The runtime is started once per cycle as
//!
//! ```text
//! dsh [--profile <profile>] [--patch <absolute path>]…
//! ```
//!
//! A configured `launch_args_override` replaces that vector verbatim. The executable is
//! `config.dsh_bin` when set, else `dsh` on `PATH`. The process environment is inherited, then
//! `config.env`, then `context.env`, then the values this adapter owns: `DSH_HOME` (mandatory —
//! `create` reports it missing instead of guessing `~/.dsh`), `DEEPSEEK_BASE_URL` /
//! `DEEPSEEK_API_KEY` from the configured endpoint, and the system prompt under
//! `system_prompt_env_var` when a custom composition declared one. The working directory is
//! `config.runtime_cwd`, else `context.cwd`, else `config.cwd`.
//!
//! # Wire contract
//!
//! Requests and notifications are newline-delimited JSON-RPC 2.0 through
//! [`JsonRpcClient`]. Opening a cycle sends
//!
//! ```json
//! {"id":1,"method":"initialize","params":{"cwd":"…","provider":"…","model":"…"}}
//! ```
//!
//! (plus `reasoningEffort` and `maxTokens` when configured) and then mints the provider session id
//! `dsh-<uuid>`. The server creates the session lazily on the first prompt, so no separate session
//! call exists. Every turn sends
//!
//! ```json
//! {"id":2,"method":"session/prompt",
//!  "params":{"sessionId":"dsh-…","contentBlocks":[{"type":"text","text":"…"}]}}
//! ```
//!
//! and then consumes `session.event` notifications until a `turn/end` arrives (see
//! [`frames`](super::frames) for the mapping). A server-to-client request is answered with a
//! `-32601` error, because the adapter exposes no client-side capability the runtime may call.
//!
//! # Cancellation
//!
//! The pinned protocol has no interrupt request, so [`VendorDriver::interrupt`] keeps its default
//! `UnsupportedCapability` body and the card declares no abort capability. [`close`] is the only
//! way to stop a turn: it sends a bounded `shutdown` request, closes the JSON-RPC client (which
//! unblocks the running turn with a `HarnessStop` interruption) and reaps the process.
//!
//! [`close`]: VendorDriver::close
//! [`JsonRpcClient`]: crate::vendor_cli::JsonRpcClient

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    json_object_to_value, HarnessContext, HarnessError, HarnessInput, HarnessResult, JsonObject,
    JsonValue, ResumePolicy, TurnResult,
};
use serde_json::json;
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

use crate::input::harness_input_text;
use crate::skills::install_for_provider;
use crate::vendor_cli::{
    ChildProcess, ChildProcessConfig, JsonRpcClient, RpcTransport, TurnContext, TurnRequest,
    VendorDriver,
};

use super::frames::DshAccumulator;
use super::wire;
use super::DshHarnessConfig;

/// Human-readable harness label used in every error message.
const LABEL: &str = "DeepSeek Harness";

/// Provider name stamped on provider extension events.
pub(super) const PROVIDER_EVENT_NAME: &str = "dsh";

/// Message of a turn the runtime blocked.
pub(super) const BLOCKED_MESSAGE: &str = "DeepSeek Harness blocked the turn";

/// Message of a turn the runtime interrupted.
pub(super) const INTERRUPTED_MESSAGE: &str = "DeepSeek Harness interrupted the turn";

/// Message of a turn the runtime reported as failed.
pub(super) const FAILED_TURN_MESSAGE: &str = "DeepSeek Harness reported a failed turn";

/// Message of a stream that ended before the native turn ended.
pub(super) const MISSING_TURN_END_MESSAGE: &str =
    "DeepSeek Harness became idle without ending a native turn";

/// Message of a turn this cycle stopped.
pub(super) const STOPPED_MESSAGE: &str = "DeepSeek Harness stopped before the turn completed";

/// Message of a turn ending the adapter does not recognize.
pub(super) const UNKNOWN_FINISH_MESSAGE: &str = "DeepSeek Harness returned an unknown turn ending";

/// Normalized failure code of a stream that ended without a native turn ending.
pub(super) const MISSING_TURN_END_CODE: &str = "DSH_MISSING_TURN_END";

/// Normalized failure code of a turn ending the adapter does not recognize.
pub(super) const UNKNOWN_FINISH_CODE: &str = "DSH_UNKNOWN_FINISH_REASON";

/// Normalized failure code of a JSON-RPC transport failure.
pub(super) const TRANSPORT_CODE: &str = "DSH_TRANSPORT";

/// JSON-RPC code of an unsupported server-to-client request.
const METHOD_NOT_FOUND: i64 = -32601;

/// How long `close` waits for the runtime to acknowledge `shutdown`.
const SHUTDOWN_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

/// How long `close` waits for the aborted reader to release the child process.
const REAP_LOCK_TIMEOUT: Duration = Duration::from_secs(5);

/// The child process, shared between the JSON-RPC transport and the driver's teardown.
type SharedChild = Arc<AsyncMutex<Option<ChildProcess>>>;

/// Session facts learned at `open`.
#[derive(Debug, Default)]
pub(super) struct DriverState {
    /// Provider session id of the current cycle.
    pub(super) session_id: Option<String>,
    /// Whether `close` asked the harness to stop.
    pub(super) closing: bool,
}

/// The `DeepSeek` Harness subprocess driver.
///
/// One instance backs one [`VendorHarness`](crate::vendor_cli::VendorHarness) handle: the runtime
/// is spawned by [`open`](VendorDriver::open), prompted by
/// [`run_turn`](VendorDriver::run_turn) and reaped by [`close`](VendorDriver::close).
pub struct DshDriver {
    /// Validated configuration decoded by the provider's `create`.
    pub(super) config: DshHarnessConfig,
    /// Resolved `dsh` executable.
    binary: PathBuf,
    /// Mandatory DSH home directory, exported into the child.
    dsh_home: PathBuf,
    /// Launch arguments resolved (and validated) by the provider's `create`.
    launch: Vec<String>,
    /// The live child process, shared with the JSON-RPC transport.
    process: SharedChild,
    /// JSON-RPC client of the open cycle.
    client: AsyncMutex<Option<Arc<JsonRpcClient>>>,
    /// Session facts learned at `open`.
    pub(super) state: Mutex<DriverState>,
}

impl DshDriver {
    /// Builds the driver over a validated configuration, a resolved executable, its home directory
    /// and the launch arguments resolved by the provider.
    #[must_use]
    pub fn new(
        config: DshHarnessConfig,
        binary: PathBuf,
        dsh_home: PathBuf,
        launch: Vec<String>,
    ) -> Self {
        Self {
            config,
            binary,
            dsh_home,
            launch,
            process: Arc::new(AsyncMutex::new(None)),
            client: AsyncMutex::new(None),
            state: Mutex::new(DriverState::default()),
        }
    }

    /// Returns the provider session id of the current cycle.
    fn session_id(&self) -> Option<String> {
        lock(&self.state).session_id.clone()
    }

    /// Returns the client of the open cycle, when one is open.
    async fn client(&self) -> Option<Arc<JsonRpcClient>> {
        self.client.lock().await.clone()
    }

    /// Installs the configured skills into the project's DSH discovery directory.
    async fn install_skills(&self, context: &HarnessContext) -> HarnessResult<()> {
        if self.config.skills.is_empty() {
            return Ok(());
        }
        let sources = self.config.skills.clone();
        let conflict = self.config.skill_conflict;
        let project = wire::skills_cwd(&self.config, context);
        let installed = tokio::task::spawn_blocking(move || {
            install_for_provider(&sources, "dsh", Some(&project), conflict)
        })
        .await
        .map_err(|error| {
            HarnessError::protocol(format!("{LABEL} skill installation task failed: {error}"))
        })?;
        installed.map(|_| ()).map_err(|error| {
            HarnessError::protocol(format!("{LABEL} skills could not be installed: {error}"))
        })
    }

    /// Sends `initialize` and waits for its acknowledgement.
    async fn initialize(
        &self,
        client: &JsonRpcClient,
        cwd: Option<&std::path::Path>,
    ) -> HarnessResult<()> {
        let mut params = serde_json::Map::new();
        if let Some(cwd) = cwd {
            params.insert("cwd".to_owned(), json!(cwd.to_string_lossy()));
        }
        params.insert("provider".to_owned(), json!(self.config.provider.as_str()));
        params.insert("model".to_owned(), json!(self.config.model.as_str()));
        if let Some(effort) = self.config.reasoning_effort.as_deref() {
            params.insert("reasoningEffort".to_owned(), json!(effort));
        }
        if let Some(max_tokens) = self.config.max_tokens {
            params.insert("maxTokens".to_owned(), json!(max_tokens));
        }
        let params = JsonValue::Object(params);
        match self.config.initialize_timeout_seconds {
            Some(seconds) => client
                .request_with_timeout("initialize", params, Duration::from_secs_f64(seconds))
                .await
                .map(|_| ()),
            None => client.request("initialize", params).await.map(|_| ()),
        }
    }

    /// Routes one inbound JSON-RPC notification, returning the terminal result when it ends the
    /// turn.
    async fn dispatch(
        &self,
        message: &JsonObject,
        accumulator: &mut DshAccumulator,
        client: &JsonRpcClient,
        session: &TurnContext,
    ) -> Option<TurnResult> {
        let method = message
            .get("method")
            .and_then(JsonValue::as_str)
            .unwrap_or_default();
        if let Some(id) = message.get("id") {
            let reason = format!("unsupported {LABEL} server request '{method}'");
            if let Err(error) = client
                .respond_error(id.clone(), METHOD_NOT_FOUND, &reason)
                .await
            {
                tracing::debug!(%error, "failed to answer a DeepSeek Harness server request");
            }
            return None;
        }
        let params = message
            .get("params")
            .map(crate::json::to_json_object)
            .unwrap_or_default();
        match method {
            "subagent.started" | "subagent.finished" => {
                accumulator.consume_subagent(session, method, &params).await;
                None
            }
            "session.event" => {
                let Some(event) = params.get("event") else {
                    accumulator
                        .forward(
                            session,
                            method,
                            &json!({"params": json_object_to_value(&params)}),
                        )
                        .await;
                    return None;
                };
                let event = crate::json::to_json_object(event);
                let session_id = params.get("sessionId").and_then(JsonValue::as_str);
                if accumulator.is_root(session_id) {
                    accumulator.consume(session, &event).await
                } else {
                    accumulator.consume_child(session, session_id, &event).await;
                    None
                }
            }
            other => {
                let event_type = if other.is_empty() {
                    "unknown-notification"
                } else {
                    other
                };
                accumulator
                    .forward(session, event_type, &json_object_to_value(&params))
                    .await;
                None
            }
        }
    }

    /// Reaps the child process: closes stdin, waits, then kills a wedged runtime.
    async fn reap(&self) {
        let Ok(mut guard) = tokio::time::timeout(REAP_LOCK_TIMEOUT, self.process.lock()).await
        else {
            tracing::warn!("the DeepSeek Harness process lock was not released in time");
            return;
        };
        let Some(mut process) = guard.take() else {
            return;
        };
        drop(guard);
        process.close_stdin().await;
        if process.wait().await.is_none() {
            process.kill().await;
        }
    }

    /// Stops the JSON-RPC client and reaps the process.
    async fn shutdown(&self) {
        let client = self.client.lock().await.take();
        if let Some(client) = client {
            let _ = client
                .request_with_timeout("shutdown", json!({}), SHUTDOWN_REQUEST_TIMEOUT)
                .await;
            client.close().await;
        }
        self.reap().await;
    }
}

/// Outcome of one turn-loop iteration.
enum TurnStep {
    /// The `session/prompt` request was answered.
    Prompted,
    /// The `session/prompt` request failed.
    PromptFailed(HarnessError),
    /// A notification arrived, or the stream ended ([`None`]).
    Message(Option<JsonObject>),
}

#[async_trait]
impl VendorDriver for DshDriver {
    fn label(&self) -> &'static str {
        LABEL
    }

    /// Installs skills, spawns the runtime, initializes the session and returns its id.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`] when the host asks to resume a checkpoint
    /// (the pinned protocol cannot restore one) before anything is spawned, and the transport's
    /// own failure when the process cannot be started or initialized.
    async fn open(
        &self,
        context: &HarnessContext,
        _session: &TurnContext,
    ) -> HarnessResult<Option<String>> {
        if context.resume_policy == ResumePolicy::RequireResume || context.checkpoint.is_some() {
            return Err(HarnessError::unsupported_capability(
                "the DSH SDK server cannot restore protocol checkpoints",
            ));
        }
        self.install_skills(context).await?;
        let cwd = wire::resolve_cwd(&self.config, context);
        let env = wire::build_env(&self.config, context, &self.dsh_home);
        let process = ChildProcess::spawn(ChildProcessConfig {
            program: &self.binary,
            args: &self.launch,
            cwd: cwd.as_deref(),
            env: &env,
            inherit_env: true,
            label: LABEL,
            max_line_bytes: 0,
        })
        .await?;
        *self.process.lock().await = Some(process);
        let client = JsonRpcClient::spawn(
            Box::new(SharedProcess {
                process: Arc::clone(&self.process),
            }),
            LABEL,
        );
        if let Err(error) = self.initialize(&client, cwd.as_deref()).await {
            tracing::warn!(%error, "the DeepSeek Harness runtime failed to initialize");
            client.close().await;
            self.reap().await;
            return Err(error);
        }
        let session_id = format!("dsh-{}", Uuid::new_v4());
        {
            let mut state = lock(&self.state);
            state.session_id = Some(session_id.clone());
            state.closing = false;
        }
        *self.client.lock().await = Some(client);
        Ok(Some(session_id))
    }

    /// Sends a bounded `shutdown` request, closes the client and reaps the runtime.
    ///
    /// # Errors
    ///
    /// This teardown is best effort and never fails: a runtime that ignores the request is killed
    /// after the bounded wait.
    async fn close(&self) -> HarnessResult<()> {
        lock(&self.state).closing = true;
        self.shutdown().await;
        lock(&self.state).session_id = None;
        Ok(())
    }

    /// Runs one turn: sends `session/prompt` and consumes notifications until `turn/end`.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when no cycle is open. A provider-level failure (a blocked
    /// or failed native turn, a stream that ended early) is an `Ok(`[`failed_turn`]`)` instead, so
    /// the turn carries its normalized error.
    ///
    /// [`failed_turn`]: crate::vendor_cli::failed_turn
    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        let text = harness_input_text(&HarnessInput {
            content: turn.content.clone(),
            metadata: turn.metadata.clone(),
        });
        let Some(client) = self.client().await else {
            return Err(HarnessError::protocol(format!(
                "the {LABEL} harness is not started"
            )));
        };
        let Some(session_id) = self.session_id() else {
            return Err(HarnessError::protocol(format!(
                "the {LABEL} harness has no open session"
            )));
        };
        let mut accumulator = DshAccumulator::new(&turn.turn_id, &session_id);
        if lock(&self.state).closing {
            return Ok(accumulator.build_open_ended(true));
        }
        let prompt = client.request(
            "session/prompt",
            json!({
                "sessionId": session_id,
                "contentBlocks": [{"type": "text", "text": text}],
            }),
        );
        tokio::pin!(prompt);
        let mut prompted = false;
        loop {
            let step = tokio::select! {
                biased;
                response = &mut prompt, if !prompted => {
                    prompted = true;
                    match response {
                        Ok(_) => TurnStep::Prompted,
                        Err(error) => TurnStep::PromptFailed(error),
                    }
                }
                message = client.next_incoming() => TurnStep::Message(message),
            };
            match step {
                TurnStep::Prompted => {}
                TurnStep::PromptFailed(error) => {
                    tracing::debug!(%error, "the DeepSeek Harness prompt request failed");
                    return Ok(accumulator.build_transport_failure(error.message()));
                }
                TurnStep::Message(None) => {
                    let stopped = lock(&self.state).closing;
                    return Ok(accumulator.build_open_ended(stopped));
                }
                TurnStep::Message(Some(message)) => {
                    if let Some(result) = self
                        .dispatch(&message, &mut accumulator, &client, session)
                        .await
                    {
                        return Ok(result);
                    }
                }
            }
        }
    }
}

/// Locks `mutex`, recovering the guard when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The JSON-RPC transport of one cycle: the child process plus the driver's own handle to it.
///
/// [`JsonRpcClient`] owns the boxed transport, but teardown (close stdin, wait, kill) belongs to
/// the driver. Sharing the process through an [`Arc`] keeps both sides on the same child without
/// duplicating the transport.
struct SharedProcess {
    /// The child process, shared with the driver.
    process: SharedChild,
}

#[async_trait]
impl RpcTransport for SharedProcess {
    async fn write_line(&mut self, line: &str) -> HarnessResult<()> {
        let mut guard = self.process.lock().await;
        let Some(process) = guard.as_mut() else {
            return Err(HarnessError::protocol(format!(
                "the {LABEL} CLI exited before the message was written"
            )));
        };
        process.write_line(line).await
    }

    async fn read_json(&mut self) -> HarnessResult<Option<JsonValue>> {
        let mut guard = self.process.lock().await;
        match guard.as_mut() {
            Some(process) => process.read_json().await,
            None => Ok(None),
        }
    }

    fn stderr_tail(&self) -> String {
        match self.process.try_lock() {
            Ok(guard) => guard
                .as_ref()
                .map(ChildProcess::stderr_tail)
                .unwrap_or_default(),
            Err(_) => String::new(),
        }
    }
}
