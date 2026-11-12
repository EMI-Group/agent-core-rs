//! Interaction domain objects used by the session-scoped `DeepAgent` loop.
//!
//! Ports `openjiuwen/harness/schema/interaction.py` (v0.1.20): the lifecycle phase
//! ([`InteractionPhase`]), the input routing policy ([`InputDispatchMode`]), the host
//! request ([`SendInputRequest`]), the unit of work consumed by the supervisor
//! ([`RoundWorkItem`]), the result of one round ([`RoundOutcome`]), the bookkeeping
//! record ([`ActiveInteractionRound`]) and the structured events the interaction
//! producer emits ([`InteractionEvent`], [`InteractionEventType`]).
//!
//! These types are pure data: they hold no session, model or stream handles. The
//! output lease/stream machinery of the Python module (`OutputLease`,
//! `OutputLeaseManager`, `InteractionOutputStream`) is runtime concurrency and
//! belongs to the agent streaming layer, not to this schema.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Internal lifecycle phase of one session-scoped interaction loop.
///
/// Python parity: `InteractionPhase(str, Enum)`; wire forms `idle`/`running`/`terminated`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InteractionPhase {
    /// No round is executing and no work item is queued.
    #[default]
    Idle,
    /// A round is currently executing an [`ActiveInteractionRound`].
    Running,
    /// The loop has terminated and accepts no further input.
    Terminated,
}

/// Routing policy for a user input that arrives while a round is active.
///
/// Python parity: `InputDispatchMode(str, Enum)`; wire forms `follow_up`/`steer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InputDispatchMode {
    /// Queue the input behind the active round (default host behaviour).
    FollowUp,
    /// Deliver the input to the running round as steering guidance.
    Steer,
}

/// Interaction-originated event types transported through an output stream.
///
/// Python parity: `InteractionEventType(str, Enum)`; wire forms `goal.updated` and
/// `execution.error`. The dotted spelling needs an explicit rename per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum InteractionEventType {
    /// The persistent goal attached to the session was created or updated.
    #[serde(rename = "goal.updated")]
    GoalUpdated,
    /// The interaction round failed with a structured error.
    #[serde(rename = "execution.error")]
    ExecutionError,
}

/// Kind of work item consumed by the single interaction supervisor.
///
/// Python parity: the `Literal["user", "goal"]` tag on `RoundWorkItem`; wire forms
/// `user`/`goal`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RoundWorkKind {
    /// A host-supplied user turn.
    #[default]
    User,
    /// A round resuming the session's persistent goal.
    Goal,
}

/// A host request to dispatch input through the interaction loop.
///
/// Python parity: frozen `SendInputRequest`. `inputs["query"]` is the only source of
/// user text, but the complete mapping is retained so invocation metadata (conversation
/// id, trusted directories, …) travels with the work item. `mode` applies only to user
/// text. Unknown fields are ignored and every missing field decodes to its default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SendInputRequest {
    /// Caller-chosen correlation id echoed back through the loop.
    pub request_id: String,
    /// The complete host input mapping; `query` carries the user text.
    pub inputs: BTreeMap<String, Value>,
    /// Optional routing policy for input that races with an active round.
    pub mode: Option<InputDispatchMode>,
}

/// One unit of work consumed by the single interaction supervisor.
///
/// Python parity: frozen `RoundWorkItem`. The item owns an isolated copy of the host
/// `inputs`; `context` carries loop bookkeeping (`reset_loop`, and for goals
/// `goal_id`/`revision`/`session_id`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RoundWorkItem {
    /// Which flavour of work this item represents.
    pub kind: RoundWorkKind,
    /// The originating host request id, absent for goal rounds.
    pub request_id: Option<String>,
    /// The host input mapping for this round.
    pub inputs: BTreeMap<String, Value>,
    /// Loop bookkeeping such as `reset_loop`, `goal_id`, `revision`, `session_id`.
    pub context: BTreeMap<String, Value>,
    /// Whether the round was queued by a follow-up instead of the goal loop.
    pub is_follow_up: bool,
}

impl RoundWorkItem {
    /// Builds an isolated user work item from host-supplied inputs.
    ///
    /// Python parity: `RoundWorkItem.user`; the context is exactly
    /// `{"reset_loop": reset_loop}`.
    #[must_use]
    pub fn user(
        request_id: Option<String>,
        inputs: BTreeMap<String, Value>,
        is_follow_up: bool,
        reset_loop: bool,
    ) -> Self {
        let mut context = BTreeMap::new();
        context.insert("reset_loop".to_string(), Value::Bool(reset_loop));
        Self {
            kind: RoundWorkKind::User,
            request_id,
            inputs,
            context,
            is_follow_up,
        }
    }

    /// Builds a goal work item wholly owned by the agent.
    ///
    /// Python parity: `RoundWorkItem.goal`; the context is exactly `goal_id`,
    /// `revision`, `session_id` and `reset_loop: true`.
    #[must_use]
    pub fn goal(
        inputs: BTreeMap<String, Value>,
        goal_id: &str,
        revision: i64,
        session_id: &str,
    ) -> Self {
        let mut context = BTreeMap::new();
        context.insert("goal_id".to_string(), Value::String(goal_id.to_string()));
        context.insert("revision".to_string(), Value::from(revision));
        context.insert(
            "session_id".to_string(),
            Value::String(session_id.to_string()),
        );
        context.insert("reset_loop".to_string(), Value::Bool(true));
        Self {
            kind: RoundWorkKind::Goal,
            request_id: None,
            inputs,
            context,
            is_follow_up: false,
        }
    }

    /// The task-loop query, intentionally derived from `inputs` only.
    ///
    /// Python parity: the `query` property (`inputs["query"]`).
    #[must_use]
    pub fn query(&self) -> Option<&Value> {
        self.inputs.get("query")
    }

    /// Whether the loop should be reset before this round runs.
    ///
    /// Python parity: the `reset_loop` property; a missing or non-boolean
    /// `context["reset_loop"]` yields `true`.
    #[must_use]
    pub fn reset_loop(&self) -> bool {
        self.context
            .get("reset_loop")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }
}

/// Result of one agent-executed interaction round.
///
/// Python parity: frozen `RoundOutcome`. `next_work` is decided by the agent; the
/// `DeepAgent` supervisor is responsible for enqueueing it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RoundOutcome {
    /// Follow-up work decided by the agent, if any.
    pub next_work: Option<RoundWorkItem>,
    /// Machine-readable failure code, absent for a successful round.
    pub error_code: Option<String>,
    /// Human-readable failure message, absent for a successful round.
    pub error_message: Option<String>,
}

impl RoundOutcome {
    /// A successful round, optionally queuing the agent-chosen next work item.
    #[must_use]
    pub fn answer(next_work: Option<RoundWorkItem>) -> Self {
        Self {
            next_work,
            error_code: None,
            error_message: None,
        }
    }

    /// A round that exceeded its completion timeout.
    #[must_use]
    pub fn timeout() -> Self {
        Self::failed(
            "completion_timeout",
            "interaction round exceeded its completion timeout",
        )
    }

    /// A round that was aborted.
    #[must_use]
    pub fn abort() -> Self {
        Self::failed("aborted", "interaction round was aborted")
    }

    /// A failed round with an explicit code and message.
    #[must_use]
    pub fn failed(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            next_work: None,
            error_code: Some(code.into()),
            error_message: Some(message.into()),
        }
    }

    /// Whether the round failed (any `error_code` is present).
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.error_code.is_some()
    }
}

/// Bookkeeping for the work item currently executed by the supervisor.
///
/// Python parity: `ActiveInteractionRound`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ActiveInteractionRound {
    /// The work item being executed.
    pub work: RoundWorkItem,
    /// The task id assigned to the running round, when one exists.
    pub task_id: Option<String>,
}

impl ActiveInteractionRound {
    /// The kind of the running round (Python `run_kind`).
    #[must_use]
    pub fn run_kind(&self) -> RoundWorkKind {
        self.work.kind
    }

    /// The loop context of the running round (Python `run_context`).
    #[must_use]
    pub fn run_context(&self) -> &BTreeMap<String, Value> {
        &self.work.context
    }

    /// The persistent goal id of the running round, if this is a goal round.
    #[must_use]
    pub fn goal_id(&self) -> Option<&str> {
        self.work.context.get("goal_id").and_then(Value::as_str)
    }

    /// The goal revision of the running round, if this is a goal round.
    #[must_use]
    pub fn revision(&self) -> Option<i64> {
        self.work.context.get("revision").and_then(Value::as_i64)
    }
}

/// Structured event emitted by the interaction output producer.
///
/// Python parity: frozen `InteractionEvent`; serializes with the Python key `type`
/// and tolerates a missing `payload` (decoded as an empty map) and unknown fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InteractionEvent {
    /// The event type; serialized under the Python key `type`.
    #[serde(rename = "type")]
    pub event_type: InteractionEventType,
    /// The event payload, always present after deserialization.
    #[serde(default)]
    pub payload: BTreeMap<String, Value>,
}

impl InteractionEvent {
    /// Builds a `goal.updated` event; the payload always carries the `goal` key.
    ///
    /// Python parity: `InteractionEvent.goal_updated` (the goal may be JSON `null`).
    #[must_use]
    pub fn goal_updated(goal: Option<Value>) -> Self {
        let mut payload = BTreeMap::new();
        payload.insert("goal".to_string(), goal.unwrap_or(Value::Null));
        Self {
            event_type: InteractionEventType::GoalUpdated,
            payload,
        }
    }

    /// Builds an `execution.error` event.
    ///
    /// Python parity: `InteractionEvent.execution_error`; the payload holds `code`
    /// and `message`, plus `goal` only when a goal is supplied.
    #[must_use]
    pub fn execution_error(code: &str, message: &str, goal: Option<Value>) -> Self {
        let mut payload = BTreeMap::new();
        payload.insert("code".to_string(), Value::String(code.to_string()));
        payload.insert("message".to_string(), Value::String(message.to_string()));
        if let Some(goal) = goal {
            payload.insert("goal".to_string(), goal);
        }
        Self {
            event_type: InteractionEventType::ExecutionError,
            payload,
        }
    }

    /// Renders the event as the Python `OutputSchema` shape.
    ///
    /// Python parity: `to_output_schema`; the result is exactly
    /// `{"type": <wire form>, "index": 0, "payload": { … }}`.
    #[must_use]
    pub fn to_output_value(&self) -> Value {
        let mut output = serde_json::Map::new();
        output.insert(
            "type".to_string(),
            Value::String(self.event_type.as_wire().to_string()),
        );
        output.insert("index".to_string(), Value::from(0_u8));
        let payload = self
            .payload
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        output.insert("payload".to_string(), Value::Object(payload));
        Value::Object(output)
    }
}

impl InteractionEventType {
    /// The Python value of the event type, as emitted by `to_output_value`.
    fn as_wire(self) -> &'static str {
        match self {
            Self::GoalUpdated => "goal.updated",
            Self::ExecutionError => "execution.error",
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inputs_with_query(query: &str) -> BTreeMap<String, Value> {
        BTreeMap::from([("query".to_string(), Value::String(query.to_string()))])
    }

    #[test]
    fn interaction_phase_wire_values_round_trip() {
        for (phase, wire) in [
            (InteractionPhase::Idle, "idle"),
            (InteractionPhase::Running, "running"),
            (InteractionPhase::Terminated, "terminated"),
        ] {
            assert_eq!(serde_json::to_value(phase).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<InteractionPhase>(json!(wire)).unwrap(),
                phase
            );
        }
        assert_eq!(InteractionPhase::default(), InteractionPhase::Idle);
    }

    #[test]
    fn input_dispatch_mode_wire_values_round_trip() {
        for (mode, wire) in [
            (InputDispatchMode::FollowUp, "follow_up"),
            (InputDispatchMode::Steer, "steer"),
        ] {
            assert_eq!(serde_json::to_value(mode).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<InputDispatchMode>(json!(wire)).unwrap(),
                mode
            );
        }
    }

    #[test]
    fn interaction_event_type_wire_values_round_trip() {
        for (event_type, wire) in [
            (InteractionEventType::GoalUpdated, "goal.updated"),
            (InteractionEventType::ExecutionError, "execution.error"),
        ] {
            assert_eq!(serde_json::to_value(event_type).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<InteractionEventType>(json!(wire)).unwrap(),
                event_type
            );
        }
    }

    #[test]
    fn round_work_kind_wire_values_round_trip() {
        for (kind, wire) in [(RoundWorkKind::User, "user"), (RoundWorkKind::Goal, "goal")] {
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<RoundWorkKind>(json!(wire)).unwrap(),
                kind
            );
        }
        assert_eq!(RoundWorkKind::default(), RoundWorkKind::User);
    }

    #[test]
    fn send_input_request_round_trips_and_tolerates_missing_fields() {
        let request = SendInputRequest {
            request_id: "req-1".to_string(),
            inputs: inputs_with_query("hi"),
            mode: Some(InputDispatchMode::Steer),
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({"request_id": "req-1", "inputs": {"query": "hi"}, "mode": "steer"})
        );
        assert_eq!(
            serde_json::from_value::<SendInputRequest>(value).unwrap(),
            request
        );

        let defaulted: SendInputRequest =
            serde_json::from_value(json!({"unknown_field": true})).unwrap();
        assert_eq!(defaulted, SendInputRequest::default());
        assert_eq!(defaulted.mode, None);
    }

    #[test]
    fn user_work_item_builds_exact_context() {
        let inputs = inputs_with_query("do it");
        let item = RoundWorkItem::user(Some("req-1".to_string()), inputs.clone(), true, false);
        assert_eq!(item.kind, RoundWorkKind::User);
        assert_eq!(item.request_id.as_deref(), Some("req-1"));
        assert_eq!(item.inputs, inputs);
        assert!(item.is_follow_up);
        assert_eq!(
            item.context,
            BTreeMap::from([("reset_loop".to_string(), json!(false))])
        );
        assert!(!item.reset_loop());
        assert_eq!(item.query(), Some(&json!("do it")));
    }

    #[test]
    fn goal_work_item_builds_exact_context() {
        let inputs = inputs_with_query("goal");
        let item = RoundWorkItem::goal(inputs.clone(), "goal-1", 7, "session-1");
        assert_eq!(item.kind, RoundWorkKind::Goal);
        assert_eq!(item.request_id, None);
        assert_eq!(item.inputs, inputs);
        assert!(!item.is_follow_up);
        assert_eq!(
            item.context,
            BTreeMap::from([
                ("goal_id".to_string(), json!("goal-1")),
                ("revision".to_string(), json!(7)),
                ("session_id".to_string(), json!("session-1")),
                ("reset_loop".to_string(), json!(true)),
            ])
        );
        assert!(item.reset_loop());
        assert_eq!(item.query(), Some(&json!("goal")));
    }

    #[test]
    fn query_is_absent_when_inputs_has_no_query() {
        let item = RoundWorkItem::user(None, BTreeMap::new(), false, true);
        assert_eq!(item.query(), None);
    }

    #[test]
    fn reset_loop_defaults_true_for_missing_or_non_bool_values() {
        let mut item = RoundWorkItem::user(None, BTreeMap::new(), false, true);
        assert!(item.reset_loop());

        item.context.clear();
        assert!(item.reset_loop());

        item.context.insert("reset_loop".to_string(), json!("no"));
        assert!(item.reset_loop());

        item.context.insert("reset_loop".to_string(), json!(false));
        assert!(!item.reset_loop());
    }

    #[test]
    fn round_work_item_round_trips_through_json() {
        let item = RoundWorkItem::goal(inputs_with_query("goal"), "goal-1", 2, "session-1");
        let value = serde_json::to_value(&item).unwrap();
        assert_eq!(
            value,
            json!({
                "kind": "goal",
                "request_id": null,
                "inputs": {"query": "goal"},
                "context": {
                    "goal_id": "goal-1",
                    "revision": 2,
                    "session_id": "session-1",
                    "reset_loop": true,
                },
                "is_follow_up": false,
            })
        );
        assert_eq!(
            serde_json::from_value::<RoundWorkItem>(value).unwrap(),
            item
        );
    }

    #[test]
    fn round_outcome_constructors_and_error_detection() {
        let plain = RoundOutcome::answer(None);
        assert!(!plain.is_error());
        assert_eq!(plain.error_code, None);
        assert_eq!(plain.error_message, None);
        assert_eq!(plain, RoundOutcome::default());

        let item = RoundWorkItem::user(None, BTreeMap::new(), false, true);
        let answered = RoundOutcome::answer(Some(item.clone()));
        assert!(!answered.is_error());
        assert_eq!(answered.next_work, Some(item));

        let timeout = RoundOutcome::timeout();
        assert!(timeout.is_error());
        assert_eq!(timeout.error_code.as_deref(), Some("completion_timeout"));
        assert!(timeout.error_message.is_some());
        assert_eq!(timeout.next_work, None);

        let aborted = RoundOutcome::abort();
        assert!(aborted.is_error());
        assert_eq!(aborted.error_code.as_deref(), Some("aborted"));
        assert!(aborted.error_message.is_some());

        let failed = RoundOutcome::failed("boom", "it broke");
        assert!(failed.is_error());
        assert_eq!(failed.error_code.as_deref(), Some("boom"));
        assert_eq!(failed.error_message.as_deref(), Some("it broke"));
    }

    #[test]
    fn round_outcome_round_trips_through_json() {
        let outcome = RoundOutcome::failed("boom", "it broke");
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(
            value,
            json!({"next_work": null, "error_code": "boom", "error_message": "it broke"})
        );
        assert_eq!(
            serde_json::from_value::<RoundOutcome>(value).unwrap(),
            outcome
        );
    }

    #[test]
    fn active_round_accessors_for_goal_and_user_rounds() {
        let goal_round = ActiveInteractionRound {
            work: RoundWorkItem::goal(BTreeMap::new(), "goal-1", 3, "session-1"),
            task_id: Some("task-1".to_string()),
        };
        assert_eq!(goal_round.run_kind(), RoundWorkKind::Goal);
        assert_eq!(goal_round.task_id.as_deref(), Some("task-1"));
        assert_eq!(goal_round.goal_id(), Some("goal-1"));
        assert_eq!(goal_round.revision(), Some(3));
        assert_eq!(
            goal_round.run_context().get("reset_loop"),
            Some(&json!(true))
        );

        let user_round = ActiveInteractionRound {
            work: RoundWorkItem::user(None, BTreeMap::new(), false, true),
            task_id: None,
        };
        assert_eq!(user_round.run_kind(), RoundWorkKind::User);
        assert_eq!(user_round.goal_id(), None);
        assert_eq!(user_round.revision(), None);
    }

    #[test]
    fn goal_updated_event_always_contains_goal_key() {
        let event = InteractionEvent::goal_updated(Some(json!({"title": "ship it"})));
        assert_eq!(event.event_type, InteractionEventType::GoalUpdated);
        assert_eq!(
            event.payload.get("goal"),
            Some(&json!({"title": "ship it"}))
        );

        let without_goal = InteractionEvent::goal_updated(None);
        assert_eq!(without_goal.event_type, InteractionEventType::GoalUpdated);
        assert_eq!(without_goal.payload.get("goal"), Some(&Value::Null));
        assert!(without_goal.payload.contains_key("goal"));
    }

    #[test]
    fn execution_error_event_includes_goal_only_when_present() {
        let with_goal = InteractionEvent::execution_error("boom", "bad", Some(json!({"id": "g"})));
        assert_eq!(with_goal.event_type, InteractionEventType::ExecutionError);
        assert_eq!(with_goal.payload.get("code"), Some(&json!("boom")));
        assert_eq!(with_goal.payload.get("message"), Some(&json!("bad")));
        assert_eq!(with_goal.payload.get("goal"), Some(&json!({"id": "g"})));

        let without_goal = InteractionEvent::execution_error("boom", "bad", None);
        assert_eq!(without_goal.payload.get("code"), Some(&json!("boom")));
        assert_eq!(without_goal.payload.get("message"), Some(&json!("bad")));
        assert!(!without_goal.payload.contains_key("goal"));
    }

    #[test]
    fn to_output_value_uses_the_python_output_schema_shape() {
        let event = InteractionEvent::goal_updated(Some(json!({"title": "ship it"})));
        assert_eq!(
            event.to_output_value(),
            json!({
                "type": "goal.updated",
                "index": 0,
                "payload": {"goal": {"title": "ship it"}},
            })
        );

        let error = InteractionEvent::execution_error("code-1", "message", None);
        assert_eq!(
            error.to_output_value(),
            json!({
                "type": "execution.error",
                "index": 0,
                "payload": {"code": "code-1", "message": "message"},
            })
        );
    }

    #[test]
    fn interaction_event_round_trips_and_defaults_a_missing_payload() {
        let event = InteractionEvent::execution_error("code-1", "message", None);
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(
            value,
            json!({
                "type": "execution.error",
                "payload": {"code": "code-1", "message": "message"},
            })
        );
        assert_eq!(
            serde_json::from_value::<InteractionEvent>(value).unwrap(),
            event
        );

        let missing_payload: InteractionEvent =
            serde_json::from_value(json!({"type": "goal.updated", "unknown": 1})).unwrap();
        assert_eq!(
            missing_payload.event_type,
            InteractionEventType::GoalUpdated
        );
        assert_eq!(missing_payload.payload, BTreeMap::new());
    }
}
