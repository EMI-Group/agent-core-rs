//! The per-hop endpoint agent that drives one handoff hop.
//!
//! Python parity: `ContainerAgent` in
//! `openjiuwen/core/multi_agent/teams/handoff/container_agent.py`. One container
//! belongs to one member and one team: the strategy registers it in the runtime
//! under `__handoff_ep_{team_id}_{agent_id}` and subscribes that endpoint to the
//! pub/sub topic `container_{agent_id}`. Delivering a [`HandoffRequest`] to the
//! endpoint runs one hop:
//!
//! 1. resolve the target member from its provider and mount the
//!    `transfer_to_<id>` tools of every allowed next target on it (once per
//!    endpoint);
//! 2. invoke the target in the child team session the runtime built;
//! 3. mirror the team's context history into the child session and back;
//! 4. write the target's result to the team stream and append it to the hop
//!    history;
//! 5. hand the result to the [`HandoffOrchestrator`]: complete the chain, forward
//!    an interrupt, or publish the next hop.
//!
//! ## Return value and re-entrancy
//!
//! Every success path returns `{}`: pub/sub delivery discards the output, so the
//! results reach consumers only through the team stream. The next hop is *not*
//! published inline: a container runs inside the per-topic FIFO worker of the
//! session's pub/sub transport topic, and awaiting `publish` there would enqueue
//! onto the same topic and deadlock (the worker cannot drain while it awaits the
//! handler). The next hop is therefore published from a spawned task, and its
//! failure is logged rather than propagated.
//!
//! A failing target never raises: the coordinator is resolved with the target's
//! error and the container still returns `{}` (Python parity).
//!
//! ## Events
//!
//! The container owns the hop-level notifications of a chain, all emitted on
//! the runtime's event bus while the chain runs:
//!
//! - `handoff_started` right before the target is invoked, with the previous
//!   member as the subject (the target itself for the first hop), the target as
//!   the peer, the team session as the session, and the 1-based hop index under
//!   the `hop` detail;
//! - `handoff_completed` when a hop actually resolves the chain, with the
//!   resolving member in both agent slots, the approved-transfer count under the
//!   `handoffs` detail, and `interrupted: true` when the hop forwarded a team
//!   interrupt;
//! - `error` when the target invocation fails, with the failure message and an
//!   `operation` detail of `handoff` (the chain is then resolved with that error,
//!   as before).
//!
//! A container whose runtime is gone emits nothing: reporting is best effort and
//! never fails a hop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::Tool;
use openjiuwen_session::{AgentTeamSession, Session, StreamItem};
use serde_json::{json, Value};

use super::signal::DEFAULT_CONTEXT_ID;
use super::{
    extract_handoff_signal, extract_handoff_signal_from_observations,
    extract_handoff_signal_from_session, flush_team_session, HandoffHistoryEntry,
    HandoffOrchestrator, HandoffRegistry, HandoffRequest, HandoffTool, CONTEXT_HISTORY_STATE_KEY,
    HISTORY_STATE_KEY,
};

use crate::error::TeamError;
use crate::events::TeamEvent;
use crate::member::{AgentProvider, TeamAgent};
use crate::runtime::TeamRuntime;

/// The answer every successful container path returns.
///
/// An empty JSON object is Python's `{}`: pub/sub delivery discards it, and the
/// hop result reaches consumers through the team stream instead.
const EMPTY_ANSWER: &str = "{}";

/// Team-session state key holding the mirrored conversation messages.
const CONTEXT_STATE_KEY: &str = "context";

/// Message key carrying the tool calls of an assistant turn.
const TOOL_CALLS_KEY: &str = "tool_calls";

/// Message key carrying the id of the tool call an observation answers.
const TOOL_CALL_ID_KEY: &str = "tool_call_id";

/// The endpoint agent of one handoff hop.
///
/// A container is cheap to clone (every field is an `Arc`, a `String`, or a
/// plain card), so the strategy registers one prototype per member and hands out
/// clones through the member provider.
#[derive(Clone)]
pub struct ContainerAgent {
    /// Card of the agent this endpoint runs (the card the runtime registers the
    /// member under).
    target_card: AgentCard,
    /// Provider of the target member instance.
    target_provider: AgentProvider,
    /// Ids of the agents the target may hand off to, sorted.
    allowed_targets: Vec<String>,
    /// Id of the owning team card, used in log fields.
    team_card_id: String,
    /// Registry id of this endpoint (`__handoff_ep_{team_id}_{agent_id}`).
    endpoint_id: String,
    /// Runtime that dispatches members and owns the bound team sessions.
    runtime: Weak<TeamRuntime>,
    /// Live coordinators keyed by team session id.
    registry: Arc<HandoffRegistry>,
    /// Whether the transfer tools were already injected into the target.
    tools_injected: Arc<AtomicBool>,
}

impl std::fmt::Debug for ContainerAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerAgent")
            .field("endpoint_id", &self.endpoint_id)
            .field("team_card_id", &self.team_card_id)
            .field("target", &self.target_card.id)
            .field("allowed_targets", &self.allowed_targets)
            .field(
                "tools_injected",
                &self.tools_injected.load(Ordering::SeqCst),
            )
            .finish_non_exhaustive()
    }
}

impl ContainerAgent {
    /// Builds the endpoint of one member.
    ///
    /// `runtime` is a [`Weak`] reference so the endpoint the runtime dispatches
    /// to never keeps the runtime (and therefore the bus) alive. `registry` is
    /// shared with the strategy: it maps a team session id to the coordinator of
    /// that session's chain.
    #[must_use]
    pub fn new(
        target_card: AgentCard,
        target_provider: AgentProvider,
        allowed_targets: Vec<String>,
        team_card_id: String,
        endpoint_id: String,
        runtime: Weak<TeamRuntime>,
        registry: Arc<HandoffRegistry>,
    ) -> Self {
        Self {
            target_card,
            target_provider,
            allowed_targets,
            team_card_id,
            endpoint_id,
            runtime,
            registry,
            tools_injected: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns the runtime registry id of this endpoint.
    #[must_use]
    pub fn endpoint_id(&self) -> &str {
        &self.endpoint_id
    }

    /// Returns the ids of the agents this endpoint injected a transfer tool for.
    #[must_use]
    pub fn allowed_targets(&self) -> &[String] {
        &self.allowed_targets
    }

    /// Emits `event` on the runtime's event bus.
    ///
    /// `runtime` is the hop runtime once it is resolved; without it the
    /// container's own weak handle is upgraded. A dropped runtime emits nothing,
    /// because event reporting must never fail a hop.
    fn emit(&self, runtime: Option<&Arc<TeamRuntime>>, event: TeamEvent) {
        let bus = runtime
            .map(|runtime| runtime.events())
            .or_else(|| self.runtime.upgrade().map(|runtime| runtime.events()));
        if let Some(bus) = bus {
            bus.emit(event);
        }
    }

    /// Emits the `handoff_started` event of the hop that is about to run.
    ///
    /// The subject is the member recorded last in the hop history — the one that
    /// handed control over; the first hop has no predecessor, so it names the
    /// target itself. The hop index is 1-based and counts the accumulated
    /// history, so it survives a resumed chain.
    fn emit_hop_started(&self, hop: &HopContext, history: &[HandoffHistoryEntry]) {
        let source = history
            .last()
            .map_or_else(|| self.target_card.id.clone(), |entry| entry.agent.clone());
        let index = u64::try_from(history.len().saturating_add(1)).unwrap_or(u64::MAX);
        let event =
            TeamEvent::handoff_started(hop.runtime.team_id(), source, self.target_card.id.as_str())
                .with_session(hop.session_id.as_str())
                .with_detail("hop", Value::from(index));
        self.emit(Some(&hop.runtime), event);
    }

    /// Emits the `handoff_completed` event of a resolved chain.
    ///
    /// The agent slots hold the coordinator's current member (the one that
    /// produced the terminal result) and the `handoffs` detail counts the
    /// approved transfers of the chain. An interrupted chain is marked with
    /// `interrupted: true`.
    fn emit_completion(&self, hop: &HopContext, interrupted: bool) {
        let last_agent = hop.coordinator.current_agent_id();
        let handoffs = u64::try_from(hop.coordinator.handoff_count()).unwrap_or(u64::MAX);
        let mut event =
            TeamEvent::handoff_completed(hop.runtime.team_id(), last_agent.clone(), last_agent)
                .with_session(hop.session_id.as_str())
                .with_detail("handoffs", Value::from(handoffs));
        if interrupted {
            event = event.with_detail("interrupted", Value::Bool(true));
        }
        self.emit(Some(&hop.runtime), event);
    }

    /// Emits the `error` event of a hop whose member invocation failed.
    ///
    /// The `handoff` operation detail tells this failure apart from other team
    /// errors; the chain is resolved with the same error afterwards.
    fn emit_hop_error(&self, hop: &HopContext, error: &Error) {
        let event = TeamEvent::error(
            hop.runtime.team_id(),
            self.target_card.id.as_str(),
            error.to_string(),
        )
        .with_session(hop.session_id.as_str())
        .with_detail("operation", Value::String("handoff".to_owned()));
        self.emit(Some(&hop.runtime), event);
    }

    /// Resolves the chain with `result` and reports the completion.
    ///
    /// The event is emitted only when this call resolved the chain, so a late
    /// completion of an already-resolved chain stays silent.
    async fn complete_chain(&self, hop: &HopContext, result: Value, interrupted: bool) {
        if hop.coordinator.complete(result).await {
            self.emit_completion(hop, interrupted);
        }
    }

    /// Mounts the `transfer_to_<id>` tools of the allowed targets on `handle`.
    ///
    /// Injection happens at most once per endpoint: the flag is set *before* the
    /// tool host is probed, so a host-less member is probed exactly once, matching
    /// Python. A rejected mount is logged and otherwise ignored.
    fn inject_tools(&self, runtime: &Arc<TeamRuntime>, handle: &TeamAgent) {
        if self.tools_injected.swap(true, Ordering::SeqCst) {
            return;
        }
        let Some(host) = handle.tool_host() else {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                target = %self.target_card.id,
                "handoff target has no tool host; skipping transfer tool injection"
            );
            return;
        };
        for target in &self.allowed_targets {
            let description = runtime
                .agent_card(target)
                .map(|card| card.description)
                .unwrap_or_default();
            let tool: Arc<dyn Tool> = Arc::new(HandoffTool::new(target.clone(), &description));
            let name = tool.card().name.clone();
            if !host.mount_tool(tool) {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    target = %self.target_card.id,
                    tool = %name,
                    "handoff transfer tool was rejected by the tool host"
                );
            }
        }
    }

    /// Shapes the target input from the request message and the hop history.
    ///
    /// Without history the message passes through unchanged. With history an
    /// object message gets a `handoff_history` key, and any other message is
    /// wrapped as `{"query": <message>, "handoff_history": [...]}` (Python
    /// parity).
    fn build_agent_input(&self, request: &HandoffRequest) -> Value {
        if request.history.is_empty() {
            return request.input_message.clone();
        }
        let history = serde_json::to_value(&request.history).unwrap_or_else(|error| {
            tracing::warn!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff hop history could not be serialized"
            );
            Value::Array(Vec::new())
        });
        match &request.input_message {
            Value::Object(message) => {
                let mut object = message.clone();
                object.insert("handoff_history".to_owned(), history);
                Value::Object(object)
            }
            message => json!({"query": message, "handoff_history": history}),
        }
    }

    /// Copies the team's mirrored context history into the hop's child session.
    ///
    /// Best effort: a missing history or a failing state write is logged at debug
    /// level and never fails the hop.
    async fn inject_context_history(
        &self,
        team_session: &Arc<AgentTeamSession>,
        child: Option<&Arc<dyn Session>>,
    ) {
        let Some(child) = child else {
            return;
        };
        let stored = match team_session.get_state(CONTEXT_HISTORY_STATE_KEY).await {
            Ok(stored) => stored,
            Err(error) => {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    %error,
                    "handoff: failed to read the team context history"
                );
                return;
            }
        };
        let Some(messages) = stored
            .as_ref()
            .and_then(Value::as_array)
            .filter(|messages| !messages.is_empty())
        else {
            return;
        };
        let patch = json!({
            CONTEXT_STATE_KEY: {
                DEFAULT_CONTEXT_ID: {
                    "messages": messages,
                    "offload_messages": {},
                }
            }
        });
        if let Err(error) = child.update_state(patch).await {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff: failed to inject the context history into the hop session"
            );
        }
    }

    /// Merges the hop's conversation messages into the team context history.
    ///
    /// Tool observations and assistant turns carrying tool calls are dropped, and
    /// messages already present in the team history are not appended again. Best
    /// effort: every failure is logged at debug level and ignored.
    async fn save_context_history(
        &self,
        team_session: &Arc<AgentTeamSession>,
        child: Option<&Arc<dyn Session>>,
    ) {
        let Some(child) = child else {
            return;
        };
        let state = match child.get_state(CONTEXT_STATE_KEY).await {
            Ok(state) => state,
            Err(error) => {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    %error,
                    "handoff: failed to read the hop context state"
                );
                return;
            }
        };
        let Some(messages) = state
            .as_ref()
            .and_then(|state| state.get(DEFAULT_CONTEXT_ID))
            .and_then(|context| context.get("messages"))
            .and_then(Value::as_array)
        else {
            return;
        };
        let mut merged = match team_session.get_state(CONTEXT_HISTORY_STATE_KEY).await {
            Ok(Some(Value::Array(existing))) => existing,
            Ok(_) => Vec::new(),
            Err(error) => {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    %error,
                    "handoff: failed to read the team context history"
                );
                return;
            }
        };
        let known: Vec<(String, String, String, String)> = merged.iter().map(message_key).collect();
        let mut appended = 0_usize;
        for message in messages {
            if is_transient_message(message) {
                continue;
            }
            let key = message_key(message);
            if known.contains(&key) {
                continue;
            }
            merged.push(message.clone());
            appended += 1;
        }
        if appended == 0 {
            return;
        }
        if let Err(error) = team_session
            .update_state(json!({CONTEXT_HISTORY_STATE_KEY: merged}))
            .await
        {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff: failed to save the team context history"
            );
        }
    }

    /// Writes the hop result to the team stream as `message` frames.
    ///
    /// An object result is one frame; an array result is one frame per object
    /// element, indexed by its position; any other value is one frame. A `null`
    /// result writes no frame because the stream hub rejects a `null` payload.
    async fn write_result(&self, team_session: &Arc<AgentTeamSession>, result: &Value) {
        let mut frames: Vec<(Option<i64>, Value)> = Vec::new();
        match result {
            Value::Null => {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    "handoff hop produced a null result; no stream frame was written"
                );
                return;
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    if item.is_object() {
                        frames.push((Some(i64::try_from(index).unwrap_or(i64::MAX)), item.clone()));
                    }
                }
            }
            other => frames.push((Some(0), other.clone())),
        }
        for (index, payload) in frames {
            let item = StreamItem::output("message", index, payload);
            if let Err(error) = team_session.write_stream(item).await {
                tracing::warn!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    %error,
                    "handoff: failed to write a hop result to the team stream"
                );
            }
        }
    }

    /// Persists the coordinator and the hop history after an interrupt, then
    /// completes the chain with the interrupt result.
    async fn forward_interrupt(
        &self,
        hop: &HopContext,
        history: &[HandoffHistoryEntry],
        result: Value,
    ) {
        let coordinator = &hop.coordinator;
        let team_session = &hop.team_session;
        if let Err(error) = coordinator.save_to_session(team_session).await {
            tracing::warn!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff: failed to save the coordinator after an interrupt"
            );
        }
        let history_value = serde_json::to_value(history).unwrap_or_else(|error| {
            tracing::warn!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff hop history could not be serialized"
            );
            Value::Array(Vec::new())
        });
        if let Err(error) = team_session
            .update_state(json!({HISTORY_STATE_KEY: history_value}))
            .await
        {
            tracing::warn!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %error,
                "handoff: failed to save the hop history after an interrupt"
            );
        }
        flush_team_session(Some(team_session)).await;
        self.complete_chain(hop, result, true).await;
    }

    /// Publishes the next hop from a spawned task.
    ///
    /// Publishing inline would enqueue onto the very topic worker that is running
    /// this hop and deadlock, so the request is handed to a task and a failure is
    /// logged instead of raised (the chain then ends by timeout).
    fn publish_next_hop(
        &self,
        runtime: &Arc<TeamRuntime>,
        session_id: &str,
        target: &str,
        request: HandoffRequest,
    ) {
        let runtime = Arc::clone(runtime);
        let topic = format!("container_{target}");
        let sender = self.endpoint_id.clone();
        let session_id = session_id.to_owned();
        let team_card_id = self.team_card_id.clone();
        tokio::spawn(async move {
            if let Err(error) = runtime
                .publish(request.to_value(), &topic, &sender, Some(&session_id))
                .await
            {
                tracing::warn!(%error, %topic, team = %team_card_id, "handoff hop publish failed");
            }
        });
    }

    /// Resolves the runtime, coordinator, and team session of one hop.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::External`] when the hop was invoked
    /// without a team session, or when its coordinator or bound team session is
    /// gone (the chain ended or the endpoint outlived its strategy).
    fn locate(&self, session: Option<&Arc<dyn Session>>) -> Result<HopContext> {
        let session_id = session
            .and_then(|session| session.parent_id())
            .filter(|id| !id.is_empty())
            .map(ToOwned::to_owned);
        let (Some(runtime), Some(session_id)) = (self.runtime.upgrade(), session_id) else {
            tracing::error!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                "handoff endpoint invoked without a team session"
            );
            return Err(Error::external(TeamError::execution(
                "ContainerAgent invoked without a HandoffTeam session (session_id is empty)",
            )));
        };
        let coordinator = self.registry.get(&session_id);
        let team_session = runtime.team_session(&session_id);
        let (Some(coordinator), Some(team_session)) = (coordinator, team_session) else {
            tracing::error!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                %session_id,
                "handoff coordinator lookup failed"
            );
            return Err(Error::external(TeamError::execution(format!(
                "coordinator not found for session_id='{session_id}'; session may have already \
                 ended or never been registered"
            ))));
        };
        Ok(HopContext {
            runtime,
            session_id,
            coordinator,
            team_session,
        })
    }

    /// Runs the target of one hop and hands its result to the coordinator.
    async fn run_hop(
        &self,
        hop: &HopContext,
        request: HandoffRequest,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let handle = (self.target_provider)();
        handle.bind(&hop.runtime, &self.target_card.id);
        self.inject_tools(&hop.runtime, &handle);

        let agent_input = self.build_agent_input(&request);
        let mut history = request.history.clone();
        tracing::debug!(
            team = %self.team_card_id,
            endpoint = %self.endpoint_id,
            session_id = %hop.session_id,
            target = %self.target_card.id,
            hop = history.len(),
            "handoff hop started"
        );

        self.inject_context_history(&hop.team_session, session.as_ref())
            .await;
        self.emit_hop_started(hop, &history);
        let output = match handle.agent().invoke(agent_input, session.clone()).await {
            Ok(output) => output,
            Err(error) => {
                tracing::warn!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    session_id = %hop.session_id,
                    target = %self.target_card.id,
                    %error,
                    "handoff target invocation failed"
                );
                self.emit_hop_error(hop, &error);
                hop.coordinator.error(error).await;
                return Ok(AgentOutput::answer(EMPTY_ANSWER));
            }
        };
        self.save_context_history(&hop.team_session, session.as_ref())
            .await;

        let result = output.to_value();
        history.push(HandoffHistoryEntry {
            agent: self.target_card.id.clone(),
            output: result.clone(),
        });
        self.write_result(&hop.team_session, &result).await;

        if output.is_interrupt() || is_interrupt_result(&result) {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                session_id = %hop.session_id,
                "handoff hop interrupted; persisting the chain state"
            );
            self.forward_interrupt(hop, &history, result).await;
            return Ok(AgentOutput::answer(EMPTY_ANSWER));
        }
        self.dispatch_result(
            hop,
            &output,
            &request.input_message,
            history,
            result,
            session.as_ref(),
        )
        .await
    }

    /// Completes the chain, forwards an interrupt, or publishes the next hop.
    async fn dispatch_result(
        &self,
        hop: &HopContext,
        output: &AgentOutput,
        input_message: &Value,
        history: Vec<HandoffHistoryEntry>,
        result: Value,
        session: Option<&Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let mut signal = extract_handoff_signal(&result)
            .or_else(|| extract_handoff_signal_from_observations(&output.observations));
        if signal.is_none() {
            if let Some(child) = session {
                signal = extract_handoff_signal_from_session(&result, child).await;
            }
        }
        let Some(signal) = signal else {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                session_id = %hop.session_id,
                "handoff chain completed without a handoff signal"
            );
            self.complete_chain(hop, result, false).await;
            return Ok(AgentOutput::answer(EMPTY_ANSWER));
        };
        let target = signal.target.clone();
        if !hop
            .coordinator
            .request_handoff(&target, signal.reason.as_deref())
            .await?
        {
            tracing::debug!(
                team = %self.team_card_id,
                endpoint = %self.endpoint_id,
                session_id = %hop.session_id,
                %target,
                "handoff blocked; completing with the current result"
            );
            self.complete_chain(hop, result, false).await;
            return Ok(AgentOutput::answer(EMPTY_ANSWER));
        }
        let next_input = signal
            .message
            .map_or_else(|| input_message.clone(), Value::String);
        tracing::debug!(
            team = %self.team_card_id,
            endpoint = %self.endpoint_id,
            session_id = %hop.session_id,
            from = %self.target_card.id,
            %target,
            "handoff approved; publishing the next hop"
        );
        self.publish_next_hop(
            &hop.runtime,
            &hop.session_id,
            &target,
            HandoffRequest::new(next_input, history),
        );
        Ok(AgentOutput::answer(EMPTY_ANSWER))
    }
}

/// The resolved context one hop runs in.
struct HopContext {
    /// Runtime that dispatches members and owns the bound sessions.
    runtime: Arc<TeamRuntime>,
    /// Id of the team session the chain runs in.
    session_id: String,
    /// Coordinator of that chain.
    coordinator: Arc<HandoffOrchestrator>,
    /// The bound team session the hop writes its frames to.
    team_session: Arc<AgentTeamSession>,
}

/// Returns the dedupe key of one conversation message.
///
/// Python keys on `(role, str(content), str(tool_calls), tool_call_id)`; the Rust
/// projection strings non-string fields with their JSON rendering.
fn message_key(message: &Value) -> (String, String, String, String) {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let content = message.get("content").map_or_else(String::new, |content| {
        content
            .as_str()
            .map_or_else(|| content.to_string(), ToOwned::to_owned)
    });
    let tool_calls = message
        .get(TOOL_CALLS_KEY)
        .map_or_else(String::new, Value::to_string);
    let tool_call_id = message
        .get(TOOL_CALL_ID_KEY)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    (role, content, tool_calls, tool_call_id)
}

/// Returns `true` for messages that must not enter the team context history.
///
/// Tool observations are per-hop records, and an assistant turn carrying tool
/// calls is meaningless without the observations it requested (Python parity).
fn is_transient_message(message: &Value) -> bool {
    if message.get("role").and_then(Value::as_str) == Some("tool") {
        return true;
    }
    let is_assistant = message.get("role").and_then(Value::as_str) == Some("assistant");
    if !is_assistant {
        return false;
    }
    message
        .get(TOOL_CALLS_KEY)
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
}

/// Returns `true` when a result value is an interrupt result.
fn is_interrupt_result(result: &Value) -> bool {
    result.get("result_type").and_then(Value::as_str) == Some("interrupt")
}

#[async_trait]
impl Agent for ContainerAgent {
    fn card(&self) -> &AgentCard {
        &self.target_card
    }

    /// Runs one hop and answers `{}`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when the hop
    /// cannot find its team session or its coordinator (the chain already ended);
    /// a failing target is *not* an error here — the coordinator is resolved with
    /// it instead.
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let request = match HandoffRequest::from_value(&inputs) {
            Ok(request) => request,
            Err(error) => {
                tracing::debug!(
                    team = %self.team_card_id,
                    endpoint = %self.endpoint_id,
                    %error,
                    "handoff endpoint received a non-handoff payload; ignoring it"
                );
                return Ok(AgentOutput::answer(EMPTY_ANSWER));
            }
        };
        let hop = self.locate(session.as_ref())?;
        self.run_hop(&hop, request, session).await
    }

    /// Streams the single answer of [`Agent::invoke`] as one chunk.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let container = self.clone();
        futures::stream::once(async move {
            let output = container.invoke(inputs, session).await?;
            Ok(AgentChunk::answer(&output))
        })
        .boxed()
    }
}

#[cfg(test)]
#[path = "container_tests.rs"]
mod tests;
