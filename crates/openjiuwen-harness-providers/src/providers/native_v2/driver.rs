//! The `native_v2` driver: one agent-teams team driven turn by turn.
//!
//! [`NativeV2Driver`] is the vendor-free half of the harness: [`open`](VendorDriver::open)
//! activates a team through `TeamRuntimeManager::activate`, [`run_turn`](VendorDriver::run_turn)
//! streams one team run and projects its chunks onto the harness observations, and the remaining
//! commands delegate to the leader runtime (`send`, `abort`, `pause`, `resume`) or to
//! `TeamAgent::stop`. The generic shell
//! ([`VendorHarness`](crate::vendor_cli::VendorHarness)) owns everything the protocol requires, so
//! this file only describes *one* team session and *one* turn.
//!
//! # Turn projection
//!
//! | Team chunk kind | Observation |
//! | --- | --- |
//! | `llm_output` | answer-channel [`OutputEvent`] (also accumulated into the final answer) |
//! | `llm_reasoning` | reasoning-channel [`OutputEvent`] |
//! | `tool_call` | `tool_call` item `Started` |
//! | `tool_result` | `tool_call` item `Completed` |
//! | anything else | [`ProviderEvent`] under the `native_v2` namespace |
//!
//! # Interruption
//!
//! `TeamAgent::stream` has no interruption channel, so the driver keeps one: an
//! [`AbortMode`] watch value. [`interrupt`](VendorDriver::interrupt) records the requested mode
//! (after delegating to the leader runtime's `abort`) and the running turn returns
//! [`interrupted_turn`] as soon as it observes that record. The turn value is reset at the start
//! of every turn, so a stale abort can never end a later one.
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use futures::StreamExt;
use openjiuwen_agent_teams::runtime::{TeamAgent, TeamBuildDeps, TeamRuntimeManager};
use openjiuwen_common::error::Error;
use openjiuwen_harness_protocol::{
    AbortMode, ContentBlock, HarnessContext, HarnessError, HarnessResult, JsonObject, JsonValue,
    MessageRole, OutputChannel, OutputEvent, OutputKind, OutputOperation, TurnMessage, TurnResult,
    TurnTerminationKind,
};
use serde_json::json;
use tokio::sync::watch;

use crate::harness_input_text;
use crate::providers::native_v2::NativeV2Config;
use crate::vendor_cli::{
    completed_turn, failed_turn, interrupted_turn, turn_error, TurnContext, TurnRequest,
    VendorDriver,
};

use super::PROVIDER_NAME;

/// Human label the shell uses in its messages.
const LABEL: &str = "native_v2";

/// Team stream chunk kind of answer-channel model output.
const CHUNK_LLM_OUTPUT: &str = "llm_output";

/// Team stream chunk kind of reasoning-channel model output.
const CHUNK_LLM_REASONING: &str = "llm_reasoning";

/// Team stream chunk kind of a started tool call.
const CHUNK_TOOL_CALL: &str = "tool_call";

/// Team stream chunk kind of a finished tool call.
const CHUNK_TOOL_RESULT: &str = "tool_result";

/// Termination message of an interrupted turn.
const INTERRUPTED_MESSAGE: &str = "native_v2 interrupted the turn";

/// Machine-readable code of a failed team run.
const FAILED_CODE: &str = "team_failure";

/// Failure category of a team stream failure.
const FAILED_CATEGORY: &str = "team";

/// The `native_v2` driver: a team runtime behind the [`VendorDriver`] seam.
pub struct NativeV2Driver {
    /// The decoded create mapping.
    config: NativeV2Config,
    /// Runtime dependencies of the team (real ones by default, fakes in tests).
    deps: TeamBuildDeps,
    /// Live manager and team of the open cycle.
    state: Mutex<DriverState>,
    /// Requested abort of the running turn, observed by [`VendorDriver::run_turn`].
    aborts: watch::Sender<Option<AbortMode>>,
}

/// Live runtime handles of one start/stop cycle.
#[derive(Default)]
struct DriverState {
    /// Manager the team was activated through; it owns the live-team pool.
    manager: Option<Arc<TeamRuntimeManager>>,
    /// Activated team handle.
    team: Option<Arc<TeamAgent>>,
}

impl fmt::Debug for NativeV2Driver {
    /// Prints the team identity and the live cycle, never the blueprint (it may carry model
    /// credentials) and never the injected dependencies.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock(&self.state);
        formatter
            .debug_struct("NativeV2Driver")
            .field("team", &self.config.team.team_name)
            .field("language", &self.config.language)
            .field("event_buffer_capacity", &self.config.event_buffer_capacity)
            .field(
                "team_session_id",
                &state.team.as_ref().map(|team| team.session_id().to_owned()),
            )
            .finish_non_exhaustive()
    }
}

impl NativeV2Driver {
    /// Builds the driver over `config` and the in-process [`TeamBuildDeps::default`].
    ///
    /// The default dependencies resolve member models from the environment (through
    /// `NativeMemberFactory::from_env`), so an offline host passes its own through
    /// [`NativeV2Driver::with_deps`].
    #[must_use]
    pub fn new(config: NativeV2Config) -> Self {
        Self::with_deps(config, TeamBuildDeps::default())
    }

    /// Builds the driver over `config` and explicit `deps`.
    ///
    /// A configured [`language`](NativeV2Config::language) is written onto `deps`, matching the
    /// precedence of `openjiuwen-agent-teams` (`TeamBuildDeps::language` wins over the spec).
    #[must_use]
    pub fn with_deps(config: NativeV2Config, deps: TeamBuildDeps) -> Self {
        let deps = match config.language.clone() {
            Some(language) => deps.with_language(language),
            None => deps,
        };
        Self {
            config,
            deps,
            state: Mutex::new(DriverState::default()),
            aborts: watch::channel(None).0,
        }
    }

    /// Returns the configuration the driver was built with.
    #[must_use]
    pub fn config(&self) -> &NativeV2Config {
        &self.config
    }

    /// Returns the activated team, or [`HarnessError::State`] when no cycle is open.
    fn team(&self) -> HarnessResult<Arc<TeamAgent>> {
        lock(&self.state)
            .team
            .clone()
            .ok_or_else(|| HarnessError::state("the native_v2 harness is not started"))
    }

    /// Sends `mode` to the running turn and removes every other pending abort.
    fn record_abort(&self, mode: AbortMode) {
        let _ = self.aborts.send_replace(Some(mode));
    }

    /// Clears a recorded abort so a later turn cannot observe a stale one.
    fn clear_abort(&self) {
        let _ = self.aborts.send_replace(None);
    }
}

#[async_trait::async_trait]
impl VendorDriver for NativeV2Driver {
    /// Returns `"native_v2"`, the name used in every shell error message.
    fn label(&self) -> &'static str {
        LABEL
    }

    /// Activates the team and reports its session id as the provider session id.
    ///
    /// The activation session is [`NativeV2Config::session_id`] when configured and the host
    /// session id of `context` otherwise; the request payload is an empty object because the
    /// driver never consumes run inputs at activation time.
    ///
    /// # Errors
    ///
    /// Returns the activation failure unchanged (`Validation` for a rejected blueprint or
    /// session, `Config`/`External` for a build or start failure).
    async fn open(
        &self,
        context: &HarnessContext,
        session: &TurnContext,
    ) -> HarnessResult<Option<String>> {
        let _ = session;
        let session_id = self
            .config
            .session_id
            .clone()
            .unwrap_or_else(|| context.host_session_id.clone());
        let manager = Arc::new(TeamRuntimeManager::new(self.deps.clone()));
        let team = manager
            .activate(&self.config.team, &session_id, &json!({}))
            .await
            .map_err(team_error)?;
        let provider_session = team.session_id().to_owned();
        let mut state = lock(&self.state);
        state.manager = Some(manager);
        state.team = Some(team);
        self.clear_abort();
        Ok(Some(provider_session))
    }

    /// Stops the activated team and drops the runtime handles.
    ///
    /// `TeamAgent::stop` never fails, so teardown problems surface as a logged warning inside the
    /// team runtime rather than as a driver error.
    async fn close(&self) -> HarnessResult<()> {
        let (team, manager) = {
            let mut state = lock(&self.state);
            self.clear_abort();
            (state.team.take(), state.manager.take())
        };
        if let Some(team) = team {
            team.stop().await.map_err(team_error)?;
        }
        drop(manager);
        Ok(())
    }

    /// Runs one team request and projects the team stream onto the harness observations.
    ///
    /// The turn ends `Completed` with the accumulated answer, `Interrupted` when an abort is
    /// observed, and `Failed` with category `"team"` when the run reports an error.
    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        let team = self.team()?;
        self.clear_abort();
        let mut aborts = self.aborts.subscribe();
        let mut aborts_open = true;
        let mut stream = team.stream(turn.content.clone(), None);
        let mut answer = String::new();
        let mut item_index: u64 = 0;
        loop {
            tokio::select! {
                biased;
                changed = aborts.changed(), if aborts_open => {
                    if changed.is_err() {
                        // No abort can reach this turn any more; keep draining the run.
                        aborts_open = false;
                        continue;
                    }
                    if let Some(mode) = *aborts.borrow_and_update() {
                        return Ok(interrupted_turn(termination_kind(mode), INTERRUPTED_MESSAGE));
                    }
                }
                next = stream.next() => match next {
                    None => break,
                    Some(Err(error)) => {
                        if let Some(mode) = *aborts.borrow() {
                            return Ok(interrupted_turn(
                                termination_kind(mode),
                                INTERRUPTED_MESSAGE,
                            ));
                        }
                        return Ok(failed_turn(turn_error(
                            format!("the native_v2 team run failed: {error}"),
                            FAILED_CODE,
                            FAILED_CATEGORY,
                            false,
                        )));
                    }
                    Some(Ok(chunk)) => {
                        project_chunk(session, &turn.turn_id, &chunk, &mut answer, &mut item_index)
                            .await;
                    }
                },
            }
        }
        Ok(completed_turn(
            JsonValue::String(answer.clone()),
            answer_messages(&answer),
            None,
        ))
    }

    /// Injects `content` into the running leader turn.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when no cycle is open, and the leader runtime's own error when it
    /// rejects the input (a terminated member, for example).
    async fn steer(
        &self,
        turn: &TurnRequest,
        content: &JsonValue,
        session: &TurnContext,
    ) -> HarnessResult<()> {
        let _ = (turn, session);
        let team = self.team()?;
        let text = TurnRequest::new("", "", content.clone(), JsonObject::new()).text();
        team.runtime().send(text, true).await.map_err(team_error)
    }

    /// Aborts the running team run in `mode`.
    ///
    /// The leader runtime performs the abort and the driver records the mode so the running turn
    /// returns [`interrupted_turn`] as soon as it observes it.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when no cycle is open, and the leader runtime's own error when the
    /// abort cannot be delivered.
    async fn interrupt(&self, mode: AbortMode, session: &TurnContext) -> HarnessResult<()> {
        let _ = session;
        let team = self.team()?;
        team.runtime()
            .abort(mode == AbortMode::Force)
            .await
            .map_err(team_error)?;
        self.record_abort(mode);
        Ok(())
    }

    /// Pauses the running leader turn.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when no cycle is open, and the leader runtime's own error.
    async fn pause(&self, session: &TurnContext) -> HarnessResult<()> {
        let _ = session;
        let team = self.team()?;
        team.runtime().pause().await.map_err(team_error)
    }

    /// Resumes a paused leader turn, forwarding `query` as the continuing input.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when no cycle is open, and the leader runtime's own error.
    async fn resume(
        &self,
        query: Option<&openjiuwen_harness_protocol::HarnessInput>,
        session: &TurnContext,
    ) -> HarnessResult<()> {
        let _ = session;
        let team = self.team()?;
        team.runtime()
            .resume(query.map(harness_input_text))
            .await
            .map_err(team_error)
    }
}

/// Projects one team stream chunk onto the harness observations.
///
/// `answer` accumulates the answer-channel text and `item_index` numbers the tool items that carry
/// no id of their own.
async fn project_chunk(
    session: &TurnContext,
    turn_id: &str,
    chunk: &openjiuwen_agent::AgentChunk,
    answer: &mut String,
    item_index: &mut u64,
) {
    let kind = chunk.kind.as_str();
    let payload = chunk_payload(&chunk.payload);
    match kind {
        CHUNK_LLM_OUTPUT => {
            let text = payload_text(&payload);
            if !text.is_empty() {
                answer.push_str(&text);
            }
            session
                .output(turn_id, output_event(&text, OutputChannel::Answer))
                .await;
        }
        CHUNK_LLM_REASONING => {
            let text = payload_text(&payload);
            session
                .output(turn_id, output_event(&text, OutputChannel::Reasoning))
                .await;
        }
        CHUNK_TOOL_CALL => {
            *item_index += 1;
            let item_id = item_id_of(&payload, *item_index);
            session
                .tool_started(
                    turn_id,
                    &item_id,
                    field_str(&payload, "tool_call_id"),
                    field_str(&payload, "name").unwrap_or("tool"),
                    field_value(&payload, "arguments").unwrap_or(JsonValue::Null),
                )
                .await;
        }
        CHUNK_TOOL_RESULT => {
            let item_id = item_id_of(&payload, (*item_index).max(1));
            session
                .tool_completed(
                    turn_id,
                    &item_id,
                    field_str(&payload, "tool_call_id"),
                    field_str(&payload, "tool_name").unwrap_or("tool"),
                    field_value(&payload, "result").unwrap_or_else(|| payload.clone()),
                    field_str(&payload, "rendered_result").map(str::to_owned),
                )
                .await;
        }
        other => {
            session
                .provider_event(turn_id, PROVIDER_NAME, other, payload)
                .await;
        }
    }
}

/// Returns the payload of one team stream chunk.
///
/// The leader tags every forwarded frame as `{"source_member", "role", "chunk"}` and the chunk
/// itself is `{"type", "payload"}`; both wrappers are stripped so a projection only ever sees the
/// provider payload. A frame with no `chunk` key is used as-is.
fn chunk_payload(payload: &JsonValue) -> JsonValue {
    let inner = payload.get("chunk").unwrap_or(payload);
    inner
        .get("payload")
        .cloned()
        .unwrap_or_else(|| inner.clone())
}

/// Returns the text of one output chunk payload (`content`, else `text`).
fn payload_text(payload: &JsonValue) -> String {
    payload
        .get("content")
        .and_then(JsonValue::as_str)
        .or_else(|| payload.get("text").and_then(JsonValue::as_str))
        .unwrap_or_default()
        .to_owned()
}

/// Returns the string `key` of a chunk payload's `data` object.
///
/// A tool item carries its fields under `data` (the item lifecycle shape); a flat payload is read
/// directly so an already-unwrapped observation still projects.
fn field_str<'a>(payload: &'a JsonValue, key: &str) -> Option<&'a str> {
    let data = payload.get("data").unwrap_or(payload);
    data.get(key).and_then(JsonValue::as_str)
}

/// Returns the JSON value of `key` from a chunk payload's `data` object, or from the payload
/// itself when it carries no `data`.
fn field_value(payload: &JsonValue, key: &str) -> Option<JsonValue> {
    let data = payload.get("data").unwrap_or(payload);
    data.get(key).cloned()
}

/// Returns the item id of a tool chunk, falling back to `fallback`.
fn item_id_of(payload: &JsonValue, fallback: u64) -> String {
    payload
        .get("item_id")
        .and_then(JsonValue::as_str)
        .map_or_else(|| format!("item-{fallback}"), str::to_owned)
}

/// Builds one answer or reasoning output observation.
fn output_event(text: &str, channel: OutputChannel) -> OutputEvent {
    OutputEvent {
        output_id: format!("native-v2-{}", channel.as_str()),
        kind: OutputKind::Text,
        content: JsonValue::String(text.to_owned()),
        operation: OutputOperation::Delta,
        channel,
        content_index: 0,
        content_type: None,
        data: JsonObject::new(),
    }
}

/// Builds the normalized message history of a completed turn: one assistant text message.
///
/// An empty answer yields no message, so a silent turn reports an empty history instead of an
/// empty text block.
fn answer_messages(answer: &str) -> Vec<TurnMessage> {
    if answer.is_empty() {
        return Vec::new();
    }
    vec![TurnMessage {
        message_id: "message-native-v2-answer".to_owned(),
        role: MessageRole::Assistant,
        content: vec![ContentBlock {
            block_id: "block-native-v2-answer".to_owned(),
            kind: "text".to_owned(),
            content: JsonValue::String(answer.to_owned()),
            data: JsonObject::new(),
        }],
        data: JsonObject::new(),
    }]
}

/// Maps the requested abort mode onto the termination kind of the interrupted turn.
///
/// Both modes are host-requested aborts, so both report
/// [`TurnTerminationKind::UserAbort`]; the requested mode itself is already recorded by
/// [`VendorDriver::interrupt`].
const fn termination_kind(mode: AbortMode) -> TurnTerminationKind {
    match mode {
        AbortMode::Graceful | AbortMode::Force => TurnTerminationKind::UserAbort,
    }
}

/// Translates a team-runtime error into the harness SPI error.
///
/// `Validation` and `Unsupported` keep their own kinds (a bad blueprint, a deferred capability);
/// every other failure is a runtime protocol problem, reported as [`HarnessError::Protocol`] with
/// the workspace error's rendered text.
fn team_error(error: Error) -> HarnessError {
    match error {
        Error::Validation(message) => HarnessError::validation(message),
        Error::Unsupported(message) => HarnessError::unsupported_capability(message),
        other => HarnessError::protocol(other.to_string()),
    }
}

/// Locks `mutex`, recovering from a poisoned lock.
///
/// A panic in one team callback must not brick the driver for the rest of the process; the guarded
/// value is two `Option<Arc<…>>` handles, so no invariant can be left half-updated.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
