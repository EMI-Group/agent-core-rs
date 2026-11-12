//! Per-invoke runtime state of the `DeepAgent` and its session JSON round-trip.
//!
//! Ports `openjiuwen/harness/schema/state.py`: [`PlanModeState`] (plan-mode
//! bookkeeping) and [`DeepAgentState`] (iteration counter, task plan, stop
//! condition snapshot, queued follow-ups). Both types persist through
//! `to_session_dict` / `from_session_dict`, stored under the session key
//! [`SESSION_STATE_KEY`] or the runtime attribute [`SESSION_RUNTIME_ATTR`].
//!
//! Restoration is deliberately lenient: persisted payloads may come from older
//! Python checkpoints, so malformed subtrees degrade to defaults instead of
//! failing the invoke. The written key set is exact and stable, because Python
//! reads the same payload back.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::schema::task::TaskPlan;

/// Session-state key under which the `DeepAgent` state payload is stored.
pub const SESSION_STATE_KEY: &str = "deepagent";

/// Session runtime attribute holding the live `DeepAgentState` object.
pub const SESSION_RUNTIME_ATTR: &str = "_deepagent_runtime_state";

/// Agent mode assumed when a persisted payload carries no usable mode.
const DEFAULT_MODE: &str = "normal";

/// Plan-mode session state of the `DeepAgent` (Python `PlanModeState`).
///
/// `mode` is the active agent mode (`"normal"` or `"plan"`), `pre_plan_mode`
/// remembers the mode that was active before plan mode was entered, `plan_slug`
/// names the active plan file, and `prompt_context` is a legacy
/// prompt-specialization marker kept for older checkpoints.
///
/// Defaulting is asymmetric, mirroring Python: a fresh [`Self::default`] has
/// `pre_plan_mode = Some("normal")`, while a restored payload whose
/// `pre_plan_mode` key is missing or `null` yields `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct PlanModeState {
    /// Current agent mode — `"normal"` or `"plan"`.
    pub mode: String,
    /// Mode that was active before plan mode was entered; `None` when the
    /// persisted payload dropped it.
    #[serde(default)]
    pub pre_plan_mode: Option<String>,
    /// Short identifier of the active plan file, e.g. `gleaming-brewing-phoenix`.
    pub plan_slug: Option<String>,
    /// Legacy prompt-specialization marker retained for older checkpoints.
    pub prompt_context: Option<String>,
}

impl Default for PlanModeState {
    /// Builds the Python dataclass defaults: normal mode, `pre_plan_mode` `"normal"`.
    fn default() -> Self {
        Self {
            mode: DEFAULT_MODE.to_owned(),
            pre_plan_mode: Some(DEFAULT_MODE.to_owned()),
            plan_slug: None,
            prompt_context: None,
        }
    }
}

impl PlanModeState {
    /// Converts to Python's JSON-serializable dict (`PlanModeState.to_dict`).
    ///
    /// All four keys are always present (`mode`, `pre_plan_mode`, `plan_slug`,
    /// `prompt_context`) and absent values are written as `null`.
    #[must_use]
    pub fn to_session_dict(&self) -> Value {
        let mut map = Map::new();
        map.insert("mode".to_owned(), Value::String(self.mode.clone()));
        map.insert(
            "pre_plan_mode".to_owned(),
            optional_string(self.pre_plan_mode.as_deref()),
        );
        map.insert(
            "plan_slug".to_owned(),
            optional_string(self.plan_slug.as_deref()),
        );
        map.insert(
            "prompt_context".to_owned(),
            optional_string(self.prompt_context.as_deref()),
        );
        Value::Object(map)
    }

    /// Restores a snapshot produced by [`Self::to_session_dict`]
    /// (Python `PlanModeState.from_dict`).
    ///
    /// `None`, a non-object value, or an empty object yields [`Self::default`]
    /// (Python's `if not data`). For a non-empty object, `mode` falls back to
    /// `"normal"` when missing or non-string, while `pre_plan_mode`,
    /// `plan_slug` and `prompt_context` become `None` when missing, `null` or
    /// non-string.
    #[must_use]
    pub fn from_session_dict(data: Option<&Value>) -> Self {
        let Some(map) = data.and_then(Value::as_object) else {
            return Self::default();
        };
        if map.is_empty() {
            return Self::default();
        }
        Self {
            mode: map
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_MODE)
                .to_owned(),
            pre_plan_mode: optional_str_field(map, "pre_plan_mode"),
            plan_slug: optional_str_field(map, "plan_slug"),
            prompt_context: optional_str_field(map, "prompt_context"),
        }
    }
}

/// Mutable per-invoke state of the `DeepAgent` (Python `DeepAgentState`).
///
/// The live object lives on the session while an invoke/stream request runs; a
/// serializable subset is checkpointed through [`Self::to_session_dict`] under
/// [`SESSION_STATE_KEY`]. Unknown fields in a restored payload are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct DeepAgentState {
    /// Number of completed task-loop rounds.
    pub iteration: usize,
    /// Structured plan driving the outer loop, if one has been built.
    pub task_plan: Option<TaskPlan>,
    /// Snapshot of the stop-condition evaluators (opaque to this schema).
    pub stop_condition_state: Option<Value>,
    /// Follow-up user messages queued for later rounds, in arrival order.
    pub pending_follow_ups: Vec<String>,
    /// Plan-mode bookkeeping.
    pub plan_mode: PlanModeState,
}

impl Default for DeepAgentState {
    /// Builds the Python dataclass defaults: iteration 0, no plan, empty queues.
    fn default() -> Self {
        Self {
            iteration: 0,
            task_plan: None,
            stop_condition_state: None,
            pending_follow_ups: Vec::new(),
            plan_mode: PlanModeState::default(),
        }
    }
}

impl DeepAgentState {
    /// Converts to Python's JSON-friendly session payload
    /// (Python `DeepAgentState.to_session_dict`).
    ///
    /// The key set is exactly Python's: `iteration`, `task_plan`,
    /// `stop_condition_state`, `pending_follow_ups`, `plan_mode`; absent values
    /// are written as `null` and no extra keys are introduced.
    #[must_use]
    pub fn to_session_dict(&self) -> Value {
        let mut map = Map::new();
        map.insert("iteration".to_owned(), Value::from(self.iteration));
        map.insert(
            "task_plan".to_owned(),
            self.task_plan
                .as_ref()
                .map_or(Value::Null, TaskPlan::to_dict),
        );
        map.insert(
            "stop_condition_state".to_owned(),
            self.stop_condition_state.clone().unwrap_or(Value::Null),
        );
        map.insert(
            "pending_follow_ups".to_owned(),
            Value::Array(
                self.pending_follow_ups
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        map.insert("plan_mode".to_owned(), self.plan_mode.to_session_dict());
        Value::Object(map)
    }

    /// Builds state from a session snapshot
    /// (Python `DeepAgentState.from_session_dict`).
    ///
    /// Deliberately lenient, because the payload may come from an older
    /// checkpoint: `None`, a non-object value or an empty object yield
    /// [`Self::default`]; `iteration` accepts a non-negative JSON number or a
    /// numeric string (anything else, including negative numbers, becomes `0`);
    /// `task_plan` is restored only from a JSON object and degrades to `None`
    /// when that object is invalid; `stop_condition_state` is cloned when
    /// present and non-`null`; `pending_follow_ups` keeps the string entries of
    /// an array and skips every other element; `plan_mode` goes through
    /// [`PlanModeState::from_session_dict`].
    #[must_use]
    pub fn from_session_dict(data: Option<&Value>) -> Self {
        let Some(map) = data.and_then(Value::as_object) else {
            return Self::default();
        };
        if map.is_empty() {
            return Self::default();
        }
        Self {
            iteration: iteration_from_value(map.get("iteration")),
            task_plan: map
                .get("task_plan")
                .filter(|value| value.is_object())
                .and_then(|value| TaskPlan::from_dict(value).ok()),
            stop_condition_state: map
                .get("stop_condition_state")
                .filter(|value| !value.is_null())
                .cloned(),
            pending_follow_ups: map
                .get("pending_follow_ups")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default(),
            plan_mode: PlanModeState::from_session_dict(map.get("plan_mode")),
        }
    }
}

/// Parses the persisted `iteration` counter.
///
/// Accepts a non-negative JSON number or a numeric string; negative numbers,
/// fractions, junk strings and missing keys all become `0` (no panic). Values
/// that do not fit a `usize` saturate instead of wrapping.
fn iteration_from_value(value: Option<&Value>) -> usize {
    match value {
        Some(Value::Number(number)) => number.as_u64().map_or_else(
            || {
                number
                    .as_i64()
                    .map_or(0, |signed| usize::try_from(signed).unwrap_or(0))
            },
            |unsigned| usize::try_from(unsigned).unwrap_or(usize::MAX),
        ),
        Some(Value::String(text)) => text
            .trim()
            .parse::<u64>()
            .map_or(0, |parsed| usize::try_from(parsed).unwrap_or(usize::MAX)),
        _ => 0,
    }
}

/// Reads an optional string field, treating missing, `null` and non-string values as `None`.
fn optional_str_field(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// Serializes an optional string, keeping `null` for absent values.
fn optional_string(value: Option<&str>) -> Value {
    value.map_or(Value::Null, |text| Value::String(text.to_owned()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;
    use crate::schema::task::{TodoItem, TodoStatus};

    fn keys(value: &Value) -> Vec<&str> {
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect()
    }

    fn sample_plan() -> TaskPlan {
        TaskPlan {
            goal: "ship the parser".to_owned(),
            tasks: vec![TodoItem {
                id: "parse".to_owned(),
                content: "parse tokens".to_owned(),
                active_form: "parsing tokens".to_owned(),
                status: TodoStatus::InProgress,
                ..TodoItem::default()
            }],
            current_task_id: Some("parse".to_owned()),
        }
    }

    #[test]
    fn plan_mode_default_prefers_explicit_normal() {
        let state = PlanModeState::default();
        assert_eq!(state.mode, "normal");
        assert_eq!(state.pre_plan_mode, Some("normal".to_owned()));
        assert_eq!(state.plan_slug, None);
        assert_eq!(state.prompt_context, None);
    }

    #[test]
    fn plan_mode_null_pre_plan_mode_restores_as_none() {
        let state = PlanModeState::from_session_dict(Some(&json!({
            "mode": "plan",
            "pre_plan_mode": null,
            "plan_slug": "brave-otter",
        })));
        assert_eq!(state.mode, "plan");
        assert_eq!(state.pre_plan_mode, None);
        assert_eq!(state.plan_slug.as_deref(), Some("brave-otter"));
        assert_eq!(state.prompt_context, None);
    }

    #[test]
    fn plan_mode_missing_pre_plan_mode_restores_as_none() {
        let state = PlanModeState::from_session_dict(Some(&json!({"mode": "plan"})));
        assert_eq!(state.pre_plan_mode, None);
        assert_eq!(state.mode, "plan");
    }

    #[test]
    fn plan_mode_lenient_values() {
        let state = PlanModeState::from_session_dict(Some(&json!({
            "mode": 7,
            "pre_plan_mode": 3,
            "plan_slug": ["nope"],
            "prompt_context": "legacy",
        })));
        assert_eq!(state.mode, "normal");
        assert_eq!(state.pre_plan_mode, None);
        assert_eq!(state.plan_slug, None);
        assert_eq!(state.prompt_context.as_deref(), Some("legacy"));
    }

    #[test]
    fn plan_mode_empty_and_non_object_fall_back_to_default() {
        assert_eq!(
            PlanModeState::from_session_dict(None),
            PlanModeState::default()
        );
        assert_eq!(
            PlanModeState::from_session_dict(Some(&json!({}))),
            PlanModeState::default()
        );
        assert_eq!(
            PlanModeState::from_session_dict(Some(&json!([1, 2]))),
            PlanModeState::default()
        );
        assert_eq!(
            PlanModeState::from_session_dict(Some(&Value::Null)),
            PlanModeState::default()
        );
        assert_eq!(
            PlanModeState::from_session_dict(Some(&json!("plan"))),
            PlanModeState::default()
        );
    }

    #[test]
    fn plan_mode_dict_keys_and_null_preservation() {
        let state = PlanModeState {
            mode: "plan".to_owned(),
            pre_plan_mode: None,
            plan_slug: Some("brave-otter".to_owned()),
            prompt_context: None,
        };
        let dict = state.to_session_dict();
        assert_eq!(
            keys(&dict),
            vec!["mode", "pre_plan_mode", "plan_slug", "prompt_context"]
        );
        assert_eq!(dict["mode"], json!("plan"));
        assert_eq!(dict["pre_plan_mode"], Value::Null);
        assert_eq!(dict["plan_slug"], json!("brave-otter"));
        assert_eq!(dict["prompt_context"], Value::Null);
        assert_eq!(
            PlanModeState::from_session_dict(Some(&dict)),
            state,
            "round trip through the dict"
        );
    }

    #[test]
    fn deep_agent_state_default_is_empty() {
        let state = DeepAgentState::default();
        assert_eq!(state.iteration, 0);
        assert_eq!(state.task_plan, None);
        assert_eq!(state.stop_condition_state, None);
        assert_eq!(state.pending_follow_ups, Vec::<String>::new());
        assert_eq!(state.plan_mode, PlanModeState::default());
    }

    #[test]
    fn deep_agent_state_full_round_trip() {
        let state = DeepAgentState {
            iteration: 4,
            task_plan: Some(sample_plan()),
            stop_condition_state: Some(json!({"MaxRoundsEvaluator": {"count": 2}})),
            pending_follow_ups: vec!["next".to_owned(), "then".to_owned()],
            plan_mode: PlanModeState {
                mode: "plan".to_owned(),
                pre_plan_mode: Some("normal".to_owned()),
                plan_slug: Some("brave-otter".to_owned()),
                prompt_context: None,
            },
        };
        let dict = state.to_session_dict();
        assert_eq!(
            keys(&dict),
            vec![
                "iteration",
                "task_plan",
                "stop_condition_state",
                "pending_follow_ups",
                "plan_mode",
            ]
        );
        assert_eq!(dict["iteration"], json!(4));
        assert_eq!(dict["task_plan"]["goal"], json!("ship the parser"));
        assert_eq!(
            dict["task_plan"]["tasks"][0]["activeForm"],
            json!("parsing tokens")
        );
        assert_eq!(
            dict["stop_condition_state"],
            json!({"MaxRoundsEvaluator": {"count": 2}})
        );
        assert_eq!(dict["pending_follow_ups"], json!(["next", "then"]));
        assert_eq!(dict["plan_mode"]["plan_slug"], json!("brave-otter"));

        assert_eq!(DeepAgentState::from_session_dict(Some(&dict)), state);
    }

    #[test]
    fn deep_agent_state_none_and_empty_object_are_default() {
        assert_eq!(
            DeepAgentState::from_session_dict(None),
            DeepAgentState::default()
        );
        assert_eq!(
            DeepAgentState::from_session_dict(Some(&json!({}))),
            DeepAgentState::default()
        );
        assert_eq!(
            DeepAgentState::from_session_dict(Some(&json!([]))),
            DeepAgentState::default()
        );
        assert_eq!(
            DeepAgentState::from_session_dict(Some(&Value::Null)),
            DeepAgentState::default()
        );
    }

    #[test]
    fn deep_agent_state_drops_unknown_keys() {
        let state = DeepAgentState::from_session_dict(Some(&json!({
            "iteration": 2,
            "mystery": true,
            "nested": {"anything": [1, 2, 3]},
        })));
        assert_eq!(state.iteration, 2);
        assert_eq!(state.task_plan, None);
        assert!(!state
            .to_session_dict()
            .as_object()
            .unwrap()
            .contains_key("mystery"));
    }

    #[test]
    fn deep_agent_state_rejects_non_object_task_plan() {
        for plan in [json!("goal"), json!(3), json!([{"id": "a"}])] {
            let state = DeepAgentState::from_session_dict(Some(&json!({"task_plan": plan})));
            assert_eq!(state.task_plan, None);
        }
    }

    #[test]
    fn deep_agent_state_degrades_invalid_task_plan() {
        let state = DeepAgentState::from_session_dict(Some(&json!({
            "task_plan": {"goal": "g", "tasks": "nope"},
        })));
        assert_eq!(state.task_plan, None);
    }

    #[test]
    fn deep_agent_state_iteration_parsing() {
        let cases = [
            (json!(7), 7_usize),
            (json!("7"), 7),
            (json!(" 12 "), 12),
            (json!("junk"), 0),
            (json!(-3), 0),
            (json!("-3"), 0),
            (json!(0), 0),
            (Value::Null, 0),
            (json!(1.5), 0),
            (json!(true), 0),
        ];
        for (raw, expected) in cases {
            let state = DeepAgentState::from_session_dict(Some(&json!({"iteration": raw})));
            assert_eq!(state.iteration, expected, "iteration {raw}");
        }
        assert_eq!(
            DeepAgentState::from_session_dict(Some(&json!({}))).iteration,
            0
        );
    }

    #[test]
    fn deep_agent_state_pending_follow_ups_keep_only_strings() {
        let state = DeepAgentState::from_session_dict(Some(&json!({
            "pending_follow_ups": ["first", 2, null, {"a": 1}, "second"],
        })));
        assert_eq!(
            state.pending_follow_ups,
            vec!["first".to_owned(), "second".to_owned()]
        );

        let missing = DeepAgentState::from_session_dict(Some(&json!({"iteration": 1})));
        assert_eq!(missing.pending_follow_ups, Vec::<String>::new());

        let wrong_type = DeepAgentState::from_session_dict(Some(&json!({
            "pending_follow_ups": "not a list",
        })));
        assert_eq!(wrong_type.pending_follow_ups, Vec::<String>::new());
    }

    #[test]
    fn deep_agent_state_stop_condition_state_null_is_none() {
        let state = DeepAgentState::from_session_dict(Some(&json!({
            "stop_condition_state": null,
        })));
        assert_eq!(state.stop_condition_state, None);

        let state = DeepAgentState::from_session_dict(Some(&json!({
            "stop_condition_state": {"rounds": 3},
        })));
        assert_eq!(state.stop_condition_state, Some(json!({"rounds": 3})));
    }

    #[test]
    fn deep_agent_state_plan_mode_restores_through_from_session_dict() {
        let state = DeepAgentState::from_session_dict(Some(&json!({
            "plan_mode": {"mode": "plan", "pre_plan_mode": null, "plan_slug": "x"},
        })));
        assert_eq!(state.plan_mode.mode, "plan");
        assert_eq!(state.plan_mode.pre_plan_mode, None);
        assert_eq!(state.plan_mode.plan_slug.as_deref(), Some("x"));

        let state = DeepAgentState::from_session_dict(Some(&json!({"iteration": 1})));
        assert_eq!(state.plan_mode, PlanModeState::default());
    }

    #[test]
    fn deep_agent_state_serde_round_trip() {
        let state = DeepAgentState {
            iteration: 2,
            task_plan: Some(sample_plan()),
            stop_condition_state: Some(json!({"token_usage": 9})),
            pending_follow_ups: vec!["later".to_owned()],
            plan_mode: PlanModeState {
                mode: "plan".to_owned(),
                ..PlanModeState::default()
            },
        };
        let serialized = serde_json::to_value(&state).unwrap();
        assert_eq!(
            keys(&serialized),
            vec![
                "iteration",
                "task_plan",
                "stop_condition_state",
                "pending_follow_ups",
                "plan_mode",
            ]
        );
        let restored: DeepAgentState = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn deep_agent_state_serde_tolerates_missing_and_unknown_fields() {
        let state: DeepAgentState = serde_json::from_value(json!({
            "iteration": 5,
            "unknown": "ignored",
        }))
        .unwrap();
        assert_eq!(state.iteration, 5);
        assert_eq!(state.task_plan, None);
        assert_eq!(state.plan_mode, PlanModeState::default());
    }

    #[test]
    fn plan_mode_serde_missing_key_is_none_but_default_is_normal() {
        let restored: PlanModeState = serde_json::from_value(json!({"mode": "plan"})).unwrap();
        assert_eq!(restored.pre_plan_mode, None);
        let restored: PlanModeState =
            serde_json::from_value(json!({"pre_plan_mode": null})).unwrap();
        assert_eq!(restored.pre_plan_mode, None);
        assert_eq!(
            PlanModeState::default().pre_plan_mode.as_deref(),
            Some("normal")
        );
    }
}
