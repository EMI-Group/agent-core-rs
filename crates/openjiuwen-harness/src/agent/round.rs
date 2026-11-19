//! Input normalization, round execution and stream plumbing of the
//! `DeepAgent` task loop.
//!
//! [`normalize_inputs`] turns the host payload into the canonical
//! `{"query": …}` shape the kernel [`AgentInput`] accepts;
//! [`RoundExecutorHandle`] adapts one inner `ReActAgent` turn to the
//! object-safe [`RoundExecutor`] seam of [`crate::task_loop`], firing the
//! harness rails around it; [`RoundState`] remembers the last output and the
//! cumulative token usage for stop-condition evaluation.
//!
//! The handle holds only `Arc`s and owned values, so it can be cloned into the
//! loop controller and never borrows the `DeepAgent`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use openjiuwen_agent::output::{ANSWER, INTERACTION, LLM_OUTPUT};
use openjiuwen_agent::{
    Agent, AgentChunk, AgentInput, AgentOutput, OutputFrame, RailContext, RailEvent, ReActAgent,
    ResultType,
};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;

use crate::agent::rail_events::{drain_rail_events, RailEventBus};
use crate::rails::{
    HarnessRailContext, HarnessRailRegistry, TaskIterationInfo, TaskIterationPhase,
};
use crate::schema::{RoundOutcome, RoundWorkItem};
use crate::task_loop::{LoopQueues, RoundExecutor};

/// Message of the Python `DEEPAGENT_INPUT_PARAM_ERROR`.
const INPUT_PARAM_ERROR: &str = "Input must be dict with 'query', str, or InteractiveInput.";

/// Error code of a round that failed inside the inner agent.
const DEEPAGENT_ERROR_CODE: &str = "deepagent_error";

/// Error code of a round whose execution returned an `Err`.
const ROUND_EXECUTION_ERROR_CODE: &str = "round_execution_error";

/// Error code of a round whose task-iteration hook failed.
const RAIL_ERROR_CODE: &str = "rail_error";

/// Message returned by [`RoundExecutorHandle::execute`] after an abort.
const ABORTED_MESSAGE: &str = "DeepAgent invocation aborted";

/// Message returned by [`RoundExecutorHandle::execute`] when the completion timeout expires.
const COMPLETION_TIMEOUT_MESSAGE: &str = "DeepAgent invocation exceeded its completion timeout";

/// Keys the task-loop consumes rather than forwarding to the kernel agent.
const RESERVED_KEYS: [&str; 14] = [
    "query",
    "conversation_id",
    "parent_session_id",
    "user_id",
    "run_kind",
    "max_iterations",
    "extra",
    "invocation_id",
    "parent_invocation_id",
    "delegation_id",
    "agent_path",
    "depth",
    "raw_query",
    "_turn_number",
];

/// Keys forwarded to the kernel unchanged when present.
const FORWARDED_KEYS: [&str; 5] = [
    "conversation_id",
    "parent_session_id",
    "user_id",
    "run_kind",
    "max_iterations",
];

/// Normalizes a host invocation payload.
///
/// A bare string becomes `{"query": <string>}`. An object keeps `query` (text
/// or an interactive payload), forwards the reserved invocation keys, and
/// collects every other key under `extra`. When `query` is absent a string
/// `raw_query` supplies it.
///
/// # Errors
///
/// Returns [`Error::validation`] with the Python
/// `DEEPAGENT_INPUT_PARAM_ERROR` text when the payload is neither a string nor
/// an object with a usable `query`.
pub fn normalize_inputs(inputs: &Value) -> Result<Value> {
    match inputs {
        Value::String(text) => Ok(json!({ "query": text })),
        Value::Object(map) => {
            let query = match map.get("query") {
                Some(Value::String(text)) => Value::String(text.clone()),
                Some(value @ Value::Object(_)) => value.clone(),
                Some(_) => return Err(Error::validation(INPUT_PARAM_ERROR)),
                None => match map.get("raw_query") {
                    Some(Value::String(text)) => Value::String(text.clone()),
                    _ => return Err(Error::validation(INPUT_PARAM_ERROR)),
                },
            };
            let mut normalized = Map::new();
            normalized.insert("query".to_string(), query);
            for key in FORWARDED_KEYS {
                if let Some(value) = map.get(key) {
                    normalized.insert(key.to_string(), value.clone());
                }
            }
            let mut extra = Map::new();
            for (key, value) in map {
                if RESERVED_KEYS.contains(&key.as_str()) {
                    continue;
                }
                extra.insert(key.clone(), value.clone());
            }
            if !extra.is_empty() {
                normalized.insert("extra".to_string(), Value::Object(extra));
            }
            Ok(Value::Object(normalized))
        }
        _ => Err(Error::validation(INPUT_PARAM_ERROR)),
    }
}

/// Last output and cumulative token usage of one `DeepAgent`.
#[derive(Debug, Default)]
pub struct RoundState {
    last_output: Mutex<Option<AgentOutput>>,
    token_usage: AtomicU64,
}

impl RoundState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the Python result dict of the last output (`output`,
    /// `result_type`, and `finish_reason` when present).
    #[must_use]
    pub fn last_result(&self) -> Option<Value> {
        self.last_output.lock().as_ref().map(|output| {
            let value = output.to_value();
            let mut result = Map::new();
            for key in ["output", "result_type", "finish_reason"] {
                if let Some(field) = value.get(key) {
                    result.insert(key.to_string(), field.clone());
                }
            }
            Value::Object(result)
        })
    }

    /// Stores `output` and adds its token usage to the running total.
    pub fn record(&self, output: &AgentOutput) {
        *self.last_output.lock() = Some(output.clone());
        self.token_usage
            .fetch_add(read_token_usage(output), Ordering::Relaxed);
    }

    /// Returns the cumulative token usage recorded so far.
    #[must_use]
    pub fn token_usage(&self) -> u64 {
        self.token_usage.load(Ordering::Relaxed)
    }

    /// Drops the stored output and resets the token counter.
    pub fn reset(&self) {
        *self.last_output.lock() = None;
        self.token_usage.store(0, Ordering::Relaxed);
    }

    /// Returns a clone of the last output.
    #[must_use]
    pub fn last_output(&self) -> Option<AgentOutput> {
        self.last_output.lock().clone()
    }
}

/// Adapter between the task-loop [`RoundExecutor`] seam and one inner
/// [`ReActAgent`].
///
/// Cloning is cheap and shares the inner agent, rails, queues, cancellation
/// token and recorded state.
#[derive(Clone)]
pub struct RoundExecutorHandle {
    inner: Arc<ReActAgent>,
    rails: HarnessRailRegistry,
    /// Shared steering queue of the owning agent; [`PromptRail`] drains it, the
    /// handle only carries it so later wiring can reuse the same queue.
    ///
    /// [`PromptRail`]: crate::agent::PromptRail
    #[allow(dead_code)]
    queues: LoopQueues,
    cancel: CancellationToken,
    session: Option<Arc<dyn Session>>,
    state: Arc<RoundState>,
    stream_tx: Option<Sender<Result<AgentChunk>>>,
    rail_events: Option<Arc<RailEventBus>>,
    /// Deadline of this turn, when the caller bounded it.
    deadline: Option<tokio::time::Instant>,
}

impl RoundExecutorHandle {
    /// Builds a handle around `inner`.
    ///
    /// `rails` receives the outer task-loop hooks, `queues` is drained for
    /// steering, `cancel` aborts an in-flight turn, `session` is handed to the
    /// inner agent, and `state` records the last output and token usage.
    #[must_use]
    pub fn new(
        inner: Arc<ReActAgent>,
        rails: HarnessRailRegistry,
        queues: LoopQueues,
        cancel: CancellationToken,
        session: Option<Arc<dyn Session>>,
        state: Arc<RoundState>,
    ) -> Self {
        Self {
            inner,
            rails,
            queues,
            cancel,
            session,
            state,
            stream_tx: None,
            rail_events: None,
            deadline: None,
        }
    }

    /// Returns this handle forwarding every stream frame to `tx`.
    #[must_use]
    pub fn with_stream_sender(mut self, tx: Sender<Result<AgentChunk>>) -> Self {
        self.stream_tx = Some(tx);
        self
    }

    /// Returns this handle bounded by `timeout`, counted from now.
    ///
    /// A zero timeout leaves the handle unbounded (Python's "0 disables the timeout").
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        if !timeout.is_zero() {
            self.deadline = Some(tokio::time::Instant::now() + timeout);
        }
        self
    }

    /// Returns this handle draining rail-queued events into `bus`.
    ///
    /// The [`RailEventDrain`](crate::agent::RailEventDrain) rail already drains
    /// the contexts of both layers; this handle drains the contexts it builds
    /// itself after every rail firing point, so events remain observable when a
    /// registry has no drain rail (and no event is drained twice: the rail takes
    /// them out of `extra` first).
    #[must_use]
    pub fn with_rail_events(mut self, bus: Arc<RailEventBus>) -> Self {
        self.rail_events = Some(bus);
        self
    }

    /// Returns the last recorded output, if any.
    #[must_use]
    pub fn last_output(&self) -> Option<AgentOutput> {
        self.state.last_output()
    }

    /// Runs one whole inner-agent user turn on `inputs`.
    ///
    /// Fires the harness user-message/steering hooks, then either invokes the
    /// agent directly or drains its stream (forwarding every non-terminal frame
    /// and ending with one `answer` frame). Expected model failures are
    /// reported inside the returned [`AgentOutput`], not as `Err`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] after an abort and propagates agent-level
    /// failures (invalid input, unconfigured model, cancellation) from the
    /// inner agent.
    pub async fn execute(&self, inputs: &Value) -> Result<AgentOutput> {
        let run = async {
            self.fire_pre_hooks(inputs).await?;
            let output = if self.stream_tx.is_some() {
                self.stream_turn(inputs).await?
            } else {
                self.inner
                    .invoke(inputs.clone(), self.session.clone())
                    .await?
            };
            self.state.record(&output);
            Ok(output)
        };
        // The turn races its own cancellation and deadline. Dropping the inner future stops the
        // model request instead of letting it finish into a receiver nobody reads.
        match self.deadline {
            Some(deadline) => tokio::select! {
                biased;
                _ = self.cancel.cancelled() => Err(Error::validation(ABORTED_MESSAGE)),
                result = tokio::time::timeout_at(deadline, run) => match result {
                    Ok(result) => result,
                    Err(_) => Err(Error::timeout(COMPLETION_TIMEOUT_MESSAGE)),
                },
            },
            None => tokio::select! {
                biased;
                _ = self.cancel.cancelled() => Err(Error::validation(ABORTED_MESSAGE)),
                result = run => result,
            },
        }
    }

    /// Fires the user-message and steering hooks for one turn.
    async fn fire_pre_hooks(&self, inputs: &Value) -> Result<()> {
        let mut ctx = HarnessRailContext::new(self.base_context(inputs));
        let result = async {
            self.rails.fire_user_message(&mut ctx).await?;
            self.rails.fire_steering_drain(&mut ctx).await
        }
        .await;
        self.drain_events(&mut ctx);
        result
    }

    /// Moves the events `ctx` queued into the rail event bus, when one is wired.
    ///
    /// Nothing is touched without a bus, and the rail contexts this handle
    /// builds are drained exactly once: the entries are removed from `extra`.
    fn drain_events(&self, ctx: &mut HarnessRailContext) {
        let Some(bus) = &self.rail_events else {
            return;
        };
        for event in drain_rail_events(ctx) {
            bus.push(event);
        }
    }

    /// Drains the inner stream, forwarding frames and returning the terminal
    /// output.
    async fn stream_turn(&self, inputs: &Value) -> Result<AgentOutput> {
        let mut stream = self.inner.stream(inputs.clone(), self.session.clone());
        let mut answer: Option<AgentOutput> = None;
        let mut frames: Vec<OutputFrame> = Vec::new();
        let mut llm_text = String::new();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            match chunk.kind.as_str() {
                ANSWER => answer = Some(decode_answer(&chunk.payload)),
                INTERACTION => {
                    if let Ok(frame) = OutputFrame::from_value(chunk.payload.clone()) {
                        frames.push(frame);
                    }
                    if !self.send(chunk).await {
                        return Err(Error::cancelled());
                    }
                }
                LLM_OUTPUT => {
                    if let Some(text) = chunk.payload.get("content").and_then(Value::as_str) {
                        llm_text.push_str(text);
                    }
                    if !self.send(chunk).await {
                        return Err(Error::cancelled());
                    }
                }
                _ => {
                    if !self.send(chunk).await {
                        return Err(Error::cancelled());
                    }
                }
            }
        }

        let output = match answer {
            Some(output) => output,
            None if frames.is_empty() => AgentOutput::answer(llm_text),
            None => {
                let ids = interrupt_ids(&frames);
                AgentOutput::interrupt(frames, ids)
            }
        };
        // A consumer that went away ends the turn: its terminal frame has nowhere to go.
        if !self.send(AgentChunk::answer(&output)).await {
            return Err(Error::cancelled());
        }
        Ok(output)
    }

    /// Builds a kernel context carrying the session and parsed input.
    fn base_context(&self, inputs: &Value) -> RailContext {
        let mut base = RailContext::new(RailEvent::BeforeInvoke);
        base.session.clone_from(&self.session);
        base.input = AgentInput::parse(inputs).ok();
        base
    }

    /// Forwards `chunk` to the stream sender, when one is configured.
    ///
    /// Returns `false` when the receiver was dropped: the turn can then stop instead of
    /// producing frames nobody will read.
    async fn send(&self, chunk: AgentChunk) -> bool {
        match &self.stream_tx {
            Some(tx) => tx.send(Ok(chunk)).await.is_ok(),
            None => true,
        }
    }
}

#[async_trait]
impl RoundExecutor for RoundExecutorHandle {
    async fn run_round(&self, work: RoundWorkItem) -> Result<RoundOutcome> {
        if self.cancel.is_cancelled() {
            return Ok(RoundOutcome::abort());
        }

        let inputs = round_inputs(&work);
        let info = TaskIterationInfo::new(0, 0);
        let info = if work.is_follow_up {
            info.with_follow_up()
        } else {
            info
        };

        let mut before = HarnessRailContext::new(self.base_context(&inputs));
        let fired = self
            .rails
            .fire_task_iteration(&mut before, info, TaskIterationPhase::Before)
            .await;
        self.drain_events(&mut before);
        if let Err(error) = fired {
            return Ok(RoundOutcome::failed(RAIL_ERROR_CODE, error.to_string()));
        }

        let (outcome, output) = match self.execute(&inputs).await {
            Ok(output) => {
                let outcome = match output.result_type {
                    ResultType::Answer | ResultType::Interrupt => RoundOutcome::answer(None),
                    ResultType::Error => {
                        RoundOutcome::failed(DEEPAGENT_ERROR_CODE, output.output.clone())
                    }
                };
                (outcome, Some(output))
            }
            Err(error) => {
                let outcome = if self.cancel.is_cancelled() {
                    RoundOutcome::abort()
                } else {
                    RoundOutcome::failed(ROUND_EXECUTION_ERROR_CODE, error.to_string())
                };
                (outcome, None)
            }
        };

        let mut after_base = self.base_context(&inputs);
        after_base.output = output;
        let mut after = HarnessRailContext::new(after_base);
        let fired = self
            .rails
            .fire_task_iteration(&mut after, info, TaskIterationPhase::After)
            .await;
        self.drain_events(&mut after);
        if let Err(error) = fired {
            tracing::warn!(error = %error, "after_task_iteration hook failed");
        }

        Ok(outcome)
    }

    fn last_round_result(&self) -> Option<Value> {
        self.state.last_result()
    }

    fn token_usage(&self) -> u64 {
        self.state.token_usage()
    }
}

/// Builds the inner-agent inputs of one round.
fn round_inputs(work: &RoundWorkItem) -> Value {
    let mut inputs: Map<String, Value> = work.inputs.clone().into_iter().collect();
    if let Some(query) = work.query() {
        inputs.insert("query".to_string(), query.clone());
    }
    Value::Object(inputs)
}

/// Decodes an `answer` frame payload, falling back to a plain-text output.
fn decode_answer(payload: &Value) -> AgentOutput {
    match AgentOutput::from_value(payload.clone()) {
        Ok(output) => output,
        Err(_) => AgentOutput::answer(payload.to_string()),
    }
}

/// Collects the interrupt ids carried by `frames`.
fn interrupt_ids(frames: &[OutputFrame]) -> Vec<String> {
    frames
        .iter()
        .filter_map(|frame| {
            frame
                .payload
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

/// Reads token usage from `output.data` or `output.extra`.
fn read_token_usage(output: &AgentOutput) -> u64 {
    if let Some(usage) = output.data.as_ref().and_then(token_usage_from_object) {
        return usage;
    }
    output
        .extra
        .get("token_usage")
        .and_then(Value::as_u64)
        .or_else(|| output.extra.get("usage").and_then(total_tokens))
        .unwrap_or(0)
}

/// Reads `token_usage` / `usage.total_tokens` from a JSON object.
fn token_usage_from_object(value: &Value) -> Option<u64> {
    let map = value.as_object()?;
    map.get("token_usage")
        .and_then(Value::as_u64)
        .or_else(|| map.get("usage").and_then(total_tokens))
}

/// Reads `total_tokens` from a usage object.
fn total_tokens(value: &Value) -> Option<u64> {
    value
        .as_object()?
        .get("total_tokens")
        .and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use openjiuwen_agent::{AgentCard, ReActAgentConfig};
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::{normalize_inputs, RoundExecutorHandle, RoundState};
    use crate::rails::HarnessRailRegistry;
    use crate::schema::{RoundWorkItem, RoundWorkKind};
    use crate::task_loop::{LoopQueues, RoundExecutor};

    fn agent(model: ScriptedModel) -> std::sync::Arc<openjiuwen_agent::ReActAgent> {
        let agent = openjiuwen_agent::ReActAgent::new(AgentCard::new("round-test"));
        agent.set_model(std::sync::Arc::new(model));
        agent.configure(ReActAgentConfig::default());
        std::sync::Arc::new(agent)
    }

    fn handle(model: ScriptedModel) -> RoundExecutorHandle {
        RoundExecutorHandle::new(
            agent(model),
            HarnessRailRegistry::new(),
            LoopQueues::default(),
            CancellationToken::new(),
            None,
            RoundState::new(),
        )
    }

    fn work(query: &str) -> RoundWorkItem {
        RoundWorkItem::user(
            None,
            BTreeMap::from([("query".to_string(), json!(query))]),
            false,
            true,
        )
    }

    #[test]
    fn normalize_string_wraps_the_query() {
        assert_eq!(
            normalize_inputs(&json!("hello")).unwrap(),
            json!({"query": "hello"})
        );
    }

    #[test]
    fn normalize_object_forwards_reserved_keys_and_collects_extra() {
        let normalized = normalize_inputs(&json!({
            "query": "hello",
            "conversation_id": "conv-1",
            "user_id": "u-1",
            "max_iterations": 3,
            "mode": "fast",
            "payload": {"a": 1}
        }))
        .unwrap();
        assert_eq!(
            normalized,
            json!({
                "query": "hello",
                "conversation_id": "conv-1",
                "user_id": "u-1",
                "max_iterations": 3,
                "extra": {"mode": "fast", "payload": {"a": 1}}
            })
        );
    }

    #[test]
    fn normalize_raw_query_supplies_a_missing_query() {
        assert_eq!(
            normalize_inputs(&json!({"raw_query": "from raw", "mode": "fast"})).unwrap(),
            json!({"query": "from raw", "extra": {"mode": "fast"}})
        );
    }

    #[test]
    fn normalize_interactive_object_passes_through() {
        let interactive = json!({"interrupt_id": "i-1", "answers": {"a": "yes"}});
        assert_eq!(
            normalize_inputs(&json!({"query": interactive})).unwrap(),
            json!({"query": {"interrupt_id": "i-1", "answers": {"a": "yes"}}})
        );
    }

    #[test]
    fn normalize_rejects_missing_or_invalid_queries() {
        for invalid in [
            json!({}),
            json!({"query": 5}),
            json!({"query": null}),
            json!(7),
        ] {
            let error = normalize_inputs(&invalid).unwrap_err();
            assert!(error
                .to_string()
                .contains("Input must be dict with 'query', str, or InteractiveInput."));
        }
    }

    #[test]
    fn round_state_filters_the_result_and_accumulates_usage() {
        let state = RoundState::new();
        assert_eq!(state.last_result(), None);
        assert_eq!(state.token_usage(), 0);

        state.record(&openjiuwen_agent::AgentOutput::answer("first"));
        assert_eq!(
            state.last_result(),
            Some(json!({"output": "first", "result_type": "answer"}))
        );

        let with_usage = openjiuwen_agent::AgentOutput::answer("second")
            .with_data(json!({"usage": {"total_tokens": 11}}));
        state.record(&with_usage);
        assert_eq!(state.token_usage(), 11);

        let with_extra =
            openjiuwen_agent::AgentOutput::answer("third").with_extra("token_usage", json!(4));
        state.record(&with_extra);
        assert_eq!(state.token_usage(), 15);

        state.reset();
        assert_eq!(state.last_result(), None);
        assert_eq!(state.token_usage(), 0);
    }

    #[tokio::test]
    async fn run_round_answers_and_records_the_output() {
        let handle = handle(ScriptedModel::text("the answer"));
        let outcome = handle.run_round(work("hello")).await.unwrap();
        assert!(!outcome.is_error());
        assert_eq!(outcome.next_work, None);

        let result = handle.last_round_result().unwrap();
        assert_eq!(result.get("output"), Some(&json!("the answer")));
        assert_eq!(result.get("result_type"), Some(&json!("answer")));
        assert!(handle.last_output().is_some());
    }

    #[tokio::test]
    async fn run_round_maps_a_model_error_to_deepagent_error() {
        // The inner agent surfaces a model failure as a `result_type: "error"`
        // terminal frame while streaming, so the round reports `deepagent_error`
        // rather than a transport-level `round_execution_error`.
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let handle =
            handle(ScriptedModel::new([ScriptedTurn::fail("boom")])).with_stream_sender(tx);
        let outcome = handle.run_round(work("hello")).await.unwrap();
        assert!(outcome.is_error());
        assert_eq!(outcome.error_code.as_deref(), Some("deepagent_error"));
    }

    #[tokio::test]
    async fn run_round_aborts_when_the_token_is_cancelled() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let handle = RoundExecutorHandle::new(
            agent(ScriptedModel::text("unused")),
            HarnessRailRegistry::new(),
            LoopQueues::default(),
            cancel,
            None,
            RoundState::new(),
        );
        let outcome = handle.run_round(work("hello")).await.unwrap();
        assert_eq!(outcome.error_code.as_deref(), Some("aborted"));
    }

    #[tokio::test]
    async fn execute_rejects_an_aborted_invocation() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let handle = RoundExecutorHandle::new(
            agent(ScriptedModel::text("unused")),
            HarnessRailRegistry::new(),
            LoopQueues::default(),
            cancel,
            None,
            RoundState::new(),
        );
        let error = handle
            .execute(&json!({"query": "hello"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("DeepAgent invocation aborted"));
    }

    #[tokio::test]
    async fn streaming_forwards_frames_and_returns_the_same_output() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let handle = handle(ScriptedModel::text("streamed answer")).with_stream_sender(tx);

        let output = handle.execute(&json!({"query": "hello"})).await.unwrap();
        assert_eq!(output.output, "streamed answer");

        let mut kinds = Vec::new();
        let mut terminal = None;
        while let Ok(item) = rx.try_recv() {
            let chunk = item.unwrap();
            if chunk.kind == openjiuwen_agent::output::ANSWER {
                terminal = openjiuwen_agent::AgentOutput::from_value(chunk.payload).ok();
            }
            kinds.push(chunk.kind);
        }
        assert_eq!(
            kinds.last().map(String::as_str),
            Some(openjiuwen_agent::output::ANSWER)
        );
        assert_eq!(terminal, Some(output));
    }

    #[test]
    fn work_item_keeps_its_kind_and_follow_up_flag() {
        let item = RoundWorkItem::user(None, BTreeMap::new(), true, true);
        assert_eq!(item.kind, RoundWorkKind::User);
        assert!(item.is_follow_up);
    }
}
