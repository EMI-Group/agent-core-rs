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
use futures::stream::BoxStream;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::Sender;
use tokio_util::sync::CancellationToken;

use openjiuwen_agent::output::{ANSWER, INTERACTION};
use openjiuwen_agent::{
    Agent, AgentChunk, AgentInput, AgentOutput, OutputFrame, RailContext, RailEvent, ResultType,
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

/// Longest raw frame preview quoted in a decode error, in characters.
const PREVIEW_LIMIT: usize = 256;

/// Replacement text for a redacted secret in a raw frame preview.
const REDACTED: &str = "***";

/// Key fragments whose values are redacted from a raw frame preview.
const SECRET_KEY_FRAGMENTS: [&str; 7] = [
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "credential",
];

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
/// `ReActAgent`.
///
/// Cloning is cheap and shares the inner agent, rails, queues, cancellation
/// token and recorded state.
#[derive(Clone)]
pub struct RoundExecutorHandle {
    inner: Arc<dyn Agent>,
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
        inner: Arc<dyn Agent>,
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
        if let Some(deadline) = self.deadline {
            tokio::select! {
                biased;
                () = self.cancel.cancelled() => Err(Error::validation(ABORTED_MESSAGE)),
                result = tokio::time::timeout_at(deadline, run) => match result {
                    Ok(result) => result,
                    Err(_) => Err(Error::timeout(COMPLETION_TIMEOUT_MESSAGE)),
                },
            }
        } else {
            tokio::select! {
                biased;
                () = self.cancel.cancelled() => Err(Error::validation(ABORTED_MESSAGE)),
                result = run => result,
            }
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
        let stream = self.inner.stream(inputs.clone(), self.session.clone());
        self.consume_stream(stream).await
    }

    /// Consumes one agent frame stream, forwarding every frame and returning the
    /// terminal output.
    ///
    /// A stream that ends without a terminal `answer` frame and without any
    /// interaction frame violated the frame protocol, so it is reported as an
    /// error rather than turned into an empty answer.
    async fn consume_stream(
        &self,
        mut stream: BoxStream<'static, Result<AgentChunk>>,
    ) -> Result<AgentOutput> {
        let mut answer: Option<AgentOutput> = None;
        let mut frames: Vec<OutputFrame> = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            match chunk.kind.as_str() {
                ANSWER => answer = Some(decode_answer(&chunk.payload)?),
                INTERACTION => {
                    frames.push(decode_interaction(chunk.index, &chunk.payload)?);
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
            None if frames.is_empty() => {
                return Err(frame_decode_error(
                    ANSWER,
                    "the stream ended without a terminal answer frame",
                    &Value::Null,
                ))
            }
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

/// Decodes an `answer` frame payload into the terminal output of the turn.
///
/// The payload is normally the JSON object of an [`AgentOutput`]; a JSON string
/// is parsed first because some transports hand the payload through as text.
///
/// # Errors
///
/// Returns a typed error carrying the decode reason and a redacted, truncated
/// raw preview when the payload is not decodable, carries no `output` /
/// `result_type` field, or is text that is not JSON. A malformed terminal frame
/// is never turned into a plausible answer.
fn decode_answer(payload: &Value) -> Result<AgentOutput> {
    let frame = match payload {
        Value::String(text) => serde_json::from_str::<Value>(text)
            .map_err(|error| frame_decode_error(ANSWER, &error.to_string(), payload))?,
        other => other.clone(),
    };
    if !carries_answer_fields(&frame) {
        return Err(frame_decode_error(
            ANSWER,
            "missing 'output' and 'result_type'",
            payload,
        ));
    }
    AgentOutput::from_value(frame)
        .map_err(|error| frame_decode_error(ANSWER, &error.to_string(), payload))
}

/// Decodes an `__interaction__` frame payload into an [`OutputFrame`].
///
/// The kernel publishes the frame *payload* (`{"id", "value"}`) rather than the
/// full `{"type", "index", "payload"}` wire frame, so a payload without a `type`
/// key is wrapped into an interaction frame at the position `index`.
///
/// # Errors
///
/// Returns a typed error when a wire-shaped payload does not decode or the bare
/// payload is not a JSON object.
fn decode_interaction(index: i64, payload: &Value) -> Result<OutputFrame> {
    if payload.get("type").is_some() {
        return OutputFrame::from_value(payload.clone())
            .map_err(|error| frame_decode_error(INTERACTION, &error.to_string(), payload));
    }
    if !payload.is_object() {
        return Err(frame_decode_error(
            INTERACTION,
            "expected an object payload",
            payload,
        ));
    }
    Ok(OutputFrame {
        kind: INTERACTION.to_owned(),
        index,
        payload: payload.clone(),
    })
}

/// Reports whether a decoded value carries the fields of an answer output.
fn carries_answer_fields(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    map.contains_key("output") || map.contains_key("result_type")
}

/// Builds the typed error of a frame that could not be decoded.
///
/// The message carries the frame kind, the decode reason and a truncated raw
/// preview with secret-looking values redacted, so a malformed provider frame is
/// diagnosable without leaking credentials into logs.
fn frame_decode_error(kind: &str, reason: &str, payload: &Value) -> Error {
    Error::validation(format!(
        "malformed {kind} frame: {reason}; raw payload: {}",
        frame_preview(payload)
    ))
}

/// Renders a bounded, secret-redacted preview of `payload`.
///
/// The preview is bounded *before* it is redacted, so a multi-megabyte frame
/// costs one bounded scan; a string value that truncation cut in half is still
/// redacted, because an unterminated secret value swallows the rest of the
/// preview.
fn frame_preview(payload: &Value) -> String {
    let text = match payload {
        Value::String(text) => truncate_chars(text, PREVIEW_LIMIT),
        other => truncate_chars(&redact_secrets(other).to_string(), PREVIEW_LIMIT),
    };
    redact_text(&text)
}

/// Redacts `"key": "value"` pairs whose key names a secret in unparsed text.
///
/// A frame that failed to parse cannot be redacted key by key, so the preview
/// is scanned textually: the value after a secret-looking key is replaced with
/// [`REDACTED`] up to its closing quote (or to the end of a truncated preview).
fn redact_text(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for fragment in SECRET_KEY_FRAGMENTS {
        let mut from = 0;
        while let Some(offset) = lower[from..].find(fragment) {
            let key_at = from + offset;
            from = key_at + fragment.len();
            if let Some(range) = secret_value_range(text, key_at) {
                ranges.push(range);
            }
        }
    }
    if ranges.is_empty() {
        return text.to_owned();
    }
    ranges.sort_unstable();
    ranges.dedup();
    let mut redacted = String::with_capacity(text.len());
    let mut cursor = 0;
    for (start, end) in ranges {
        if start < cursor {
            continue;
        }
        redacted.push_str(&text[cursor..start]);
        redacted.push_str(REDACTED);
        cursor = end;
    }
    redacted.push_str(&text[cursor..]);
    redacted
}

/// Returns the byte range of the value of the key at `key_at`.
///
/// The range covers the content between the first pair of quotes after the
/// key's colon, so it is empty for a `null`/numeric value; an unterminated
/// value extends to the end of `text`.
fn secret_value_range(text: &str, key_at: usize) -> Option<(usize, usize)> {
    let colon = text.get(key_at..)?.find(':')?;
    let value_at = key_at + colon + 1;
    let start = value_at + text.get(value_at..)?.find('"')? + 1;
    match text.get(start..).and_then(|tail| tail.find('"')) {
        Some(len) => Some((start, start + len)),
        None => Some((start, text.len())),
    }
}

/// Replaces the values of secret-looking keys and nested JSON text.
fn redact_secrets(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| {
                    let item = if is_secret_key(key) {
                        Value::String(REDACTED.to_owned())
                    } else {
                        redact_secrets(item)
                    };
                    (key.clone(), item)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(redact_secrets).collect()),
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(nested) => Value::String(redact_secrets(&nested).to_string()),
            Err(_) => value.clone(),
        },
        other => other.clone(),
    }
}

/// Reports whether a JSON key names a secret whose value must not be previewed.
fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    SECRET_KEY_FRAGMENTS
        .iter()
        .any(|fragment| key.contains(fragment))
}

/// Truncates `text` to [`PREVIEW_LIMIT`] characters on a char boundary.
fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(limit).collect();
    truncated.push('…');
    truncated
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
    use openjiuwen_agent::ResultType;

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

    /// A fake inner agent that streams a fixed frame list.
    struct FrameAgent {
        card: openjiuwen_agent::AgentCard,
        chunks: Vec<openjiuwen_agent::AgentChunk>,
    }

    #[async_trait::async_trait]
    impl openjiuwen_agent::Agent for FrameAgent {
        fn card(&self) -> &openjiuwen_agent::AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: serde_json::Value,
            _session: Option<std::sync::Arc<dyn openjiuwen_session::Session>>,
        ) -> openjiuwen_common::error::Result<openjiuwen_agent::AgentOutput> {
            Err(openjiuwen_common::error::Error::validation(
                "the fake agent only streams",
            ))
        }

        fn stream(
            &self,
            _inputs: serde_json::Value,
            _session: Option<std::sync::Arc<dyn openjiuwen_session::Session>>,
        ) -> futures::stream::BoxStream<
            'static,
            openjiuwen_common::error::Result<openjiuwen_agent::AgentChunk>,
        > {
            Box::pin(futures::stream::iter(
                self.chunks.clone().into_iter().map(Ok),
            ))
        }
    }

    /// Builds a stream-path handle over a fixed frame list.
    fn frame_handle(chunks: Vec<openjiuwen_agent::AgentChunk>) -> RoundExecutorHandle {
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        RoundExecutorHandle::new(
            std::sync::Arc::new(FrameAgent {
                card: AgentCard::new("frame-agent"),
                chunks,
            }),
            HarnessRailRegistry::new(),
            LoopQueues::default(),
            CancellationToken::new(),
            None,
            RoundState::new(),
        )
        .with_stream_sender(tx)
    }

    #[tokio::test]
    async fn a_malformed_terminal_frame_ends_the_run_with_the_typed_error() {
        let chunk = openjiuwen_agent::AgentChunk::new(
            openjiuwen_agent::output::ANSWER,
            0,
            json!("{not json"),
        );
        let handle = frame_handle(vec![chunk]);
        let error = handle
            .execute(&json!({"query": "hello"}))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("malformed answer frame"),
            "{error}"
        );
        assert_eq!(handle.last_output(), None);
    }

    #[tokio::test]
    async fn a_malformed_terminal_frame_fails_the_round() {
        let chunk = openjiuwen_agent::AgentChunk::new(
            openjiuwen_agent::output::ANSWER,
            0,
            json!({"output": 5}),
        );
        let outcome = frame_handle(vec![chunk])
            .run_round(work("hello"))
            .await
            .unwrap();
        assert!(outcome.is_error());
        assert_eq!(outcome.error_code.as_deref(), Some("round_execution_error"));
        assert!(
            outcome
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("malformed answer frame")),
            "{outcome:?}"
        );
    }

    /// Builds a stream of `chunks` for [`RoundExecutorHandle::consume_stream`].
    fn frames(
        chunks: Vec<openjiuwen_common::error::Result<openjiuwen_agent::AgentChunk>>,
    ) -> futures::stream::BoxStream<
        'static,
        openjiuwen_common::error::Result<openjiuwen_agent::AgentChunk>,
    > {
        Box::pin(futures::stream::iter(chunks))
    }

    #[tokio::test]
    async fn malformed_answer_object_is_a_typed_error_not_an_answer() {
        let handle = handle(ScriptedModel::text("unused"));
        for (payload, expected) in [
            (json!({"foo": 1}), "missing 'output' and 'result_type'"),
            (json!({"result_type": 7}), "invalid type"),
        ] {
            let chunk =
                openjiuwen_agent::AgentChunk::new(openjiuwen_agent::output::ANSWER, 0, payload);
            let error = handle
                .consume_stream(frames(vec![Ok(chunk)]))
                .await
                .unwrap_err();
            let message = error.to_string();
            assert!(message.contains("malformed answer frame"), "{message}");
            assert!(message.contains(expected), "{message}");
        }
    }

    #[tokio::test]
    async fn malformed_answer_text_is_a_typed_error_with_a_redacted_preview() {
        let handle = handle(ScriptedModel::text("unused"));
        let raw = "{\"output\": \"cut off\", \"api_key\": \"sk-super-secret\", ";
        let chunk =
            openjiuwen_agent::AgentChunk::new(openjiuwen_agent::output::ANSWER, 0, json!(raw));
        let error = handle
            .consume_stream(frames(vec![Ok(chunk)]))
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("malformed answer frame"), "{message}");
        assert!(message.contains("EOF while parsing a value"), "{message}");
        assert!(message.contains("cut off"), "{message}");
        assert!(!message.contains("sk-super-secret"), "{message}");
    }

    #[tokio::test]
    async fn malformed_answer_preview_is_redacted_and_truncated() {
        let payload = json!({
            "api_key": "sk-leaked",
            "nested": {"authorization": "Bearer abc", "note": "x".repeat(4096)},
        });
        let message = super::frame_decode_error(super::ANSWER, "boom", &payload).to_string();
        assert!(message.starts_with("validation: malformed answer frame: boom; raw payload: "));
        assert!(!message.contains("sk-leaked"), "{message}");
        assert!(!message.contains("Bearer abc"), "{message}");
        assert!(message.contains("\"api_key\":\"***\""), "{message}");
        assert!(message.chars().count() < 400, "preview is truncated");
    }

    #[tokio::test]
    async fn malformed_interaction_frame_is_a_typed_error() {
        let handle = handle(ScriptedModel::text("unused"));
        let chunk = openjiuwen_agent::AgentChunk::new(
            openjiuwen_agent::output::INTERACTION,
            1,
            json!("not-a-frame"),
        );
        let error = handle
            .consume_stream(frames(vec![Ok(chunk)]))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("malformed __interaction__ frame: expected an object payload"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn bare_interaction_payload_is_wrapped_at_its_chunk_position() {
        let handle = handle(ScriptedModel::text("unused"));
        let chunk = openjiuwen_agent::AgentChunk::new(
            openjiuwen_agent::output::INTERACTION,
            7,
            json!({"id": "ask-1", "value": {"question": "ok?"}}),
        );
        let output = handle
            .consume_stream(frames(vec![Ok(chunk)]))
            .await
            .unwrap();
        assert_eq!(output.result_type, ResultType::Interrupt);
        assert_eq!(output.interrupt_ids, vec!["ask-1".to_string()]);
        assert_eq!(output.state.len(), 1);
        assert_eq!(output.state[0].kind, openjiuwen_agent::output::INTERACTION);
        assert_eq!(output.state[0].index, 7);
    }

    #[tokio::test]
    async fn a_stream_without_a_terminal_frame_is_a_typed_error() {
        let handle = handle(ScriptedModel::text("unused"));
        let chunk = openjiuwen_agent::AgentChunk::new(
            openjiuwen_agent::output::LLM_OUTPUT,
            0,
            json!({"content": "half an answer"}),
        );
        let error = handle
            .consume_stream(frames(vec![Ok(chunk)]))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("the stream ended without a terminal answer frame"),
            "{error}"
        );
    }

    #[test]
    fn work_item_keeps_its_kind_and_follow_up_flag() {
        let item = RoundWorkItem::user(None, BTreeMap::new(), true, true);
        assert_eq!(item.kind, RoundWorkKind::User);
        assert!(item.is_follow_up);
    }
}
