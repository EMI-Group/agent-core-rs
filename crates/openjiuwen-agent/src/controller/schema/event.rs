//! Controller events.
//!
//! Ports `openjiuwen/core/controller/schema/event.py`. Python models every
//! event kind as its own subclass of `Event`; here they collapse into one
//! [`Event`] record discriminated by [`EventType`], which keeps the wire
//! shape flat and lets an event round-trip without knowing its subclass.
use super::dataframe::{frames_text, DataFrame, JsonDataFrame};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use uuid::Uuid;
/// Controller event kinds (`input`, `task_interaction`, `task_completion`,
/// `task_failed`, `follow_up`).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// A user input event; the controller's main entry point.
    #[default]
    Input,
    /// A task needs user interaction to continue.
    TaskInteraction,
    /// A task finished successfully.
    TaskCompletion,
    /// A task failed.
    TaskFailed,
    /// Follow-up input that continues the task loop.
    FollowUp,
}
impl EventType {
    /// Returns the Python wire string of the event kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::TaskInteraction => "task_interaction",
            Self::TaskCompletion => "task_completion",
            Self::TaskFailed => "task_failed",
            Self::FollowUp => "follow_up",
        }
    }
}
/// Generates a fresh event id, matching Python's `str(uuid.uuid4())`.
fn new_event_id() -> String {
    Uuid::new_v4().to_string()
}
/// A controller event.
///
/// Combines Python's `Event` base class with its `InputEvent`,
/// `TaskInteractionEvent`, `TaskCompletionEvent`, `TaskFailedEvent`, and
/// `FollowUpEvent` subclasses: [`event_type`](Self::event_type) selects which
/// payload field is meaningful.
///
/// [`Default`] keeps the event id empty so a parsed event without an id stays
/// deterministic; the constructors mint a fresh id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Event {
    /// Event kind.
    pub event_type: EventType,
    /// Unique event id; a missing id on the wire is minted during parsing.
    #[serde(default = "new_event_id")]
    pub event_id: String,
    /// Free-form event metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Input payload of an [`EventType::Input`] or [`EventType::FollowUp`].
    #[serde(rename = "input_data", default, skip_serializing_if = "Vec::is_empty")]
    pub input_data: Vec<DataFrame>,
    /// Result payload of an [`EventType::TaskCompletion`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_result: Vec<DataFrame>,
    /// Interaction payload of an [`EventType::TaskInteraction`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interaction: Vec<DataFrame>,
    /// Failure reason of an [`EventType::TaskFailed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Task the event belongs to, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}
impl Event {
    /// Builds an empty event of `event_type` with a fresh id.
    #[must_use]
    pub fn new(event_type: EventType) -> Self {
        Self {
            event_type,
            event_id: new_event_id(),
            metadata: BTreeMap::new(),
            input_data: Vec::new(),
            task_result: Vec::new(),
            interaction: Vec::new(),
            error_message: None,
            task_id: None,
        }
    }
    /// Builds an input event carrying one text frame.
    #[must_use]
    pub fn user_input(text: impl Into<String>) -> Self {
        let mut event = Self::new(EventType::Input);
        event.input_data = vec![DataFrame::text(text)];
        event
    }
    /// Builds an input event from already-built frames.
    #[must_use]
    pub fn from_frames(frames: Vec<DataFrame>) -> Self {
        let mut event = Self::new(EventType::Input);
        event.input_data = frames;
        event
    }
    /// Builds an input event from the caller's JSON payload.
    ///
    /// Mirrors Python's `InputEvent.from_user_input` plus the kernel's
    /// `AgentInput` shape: strings become text frames, arrays become one text
    /// frame per element, and objects (including `{"query": …}` and
    /// interactive resume payloads) become JSON frames.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for `null` (empty input) and for scalar
    /// numbers or booleans, which are not a controller input shape.
    pub fn from_user_inputs(value: &Value) -> Result<Self> {
        match value {
            Value::String(text) => Ok(Self::user_input(text)),
            Value::Array(items) => Ok(Self::from_frames(
                items
                    .iter()
                    .map(|item| {
                        DataFrame::text(
                            item.as_str()
                                .map_or_else(|| item.to_string(), str::to_owned),
                        )
                    })
                    .collect(),
            )),
            Value::Object(map) => {
                if let Some(query) = map.get("query") {
                    if let Some(text) = query.as_str() {
                        return Ok(Self::user_input(text));
                    }
                    let mut event = Self::new(EventType::Input);
                    event.input_data = vec![DataFrame::Json(JsonDataFrame::from_value(
                        serde_json::json!({"query": query}),
                    ))];
                    return Ok(event);
                }
                let mut event = Self::new(EventType::Input);
                event.input_data = vec![DataFrame::Json(JsonDataFrame::from_value(value.clone()))];
                Ok(event)
            }
            Value::Null => Err(Error::validation("controller input is empty")),
            other => Err(Error::validation(format!(
                "controller input must be a string, array, or object, got {other}"
            ))),
        }
    }
    /// Builds a task interaction event carrying the interaction frames.
    #[must_use]
    pub fn task_interaction(task_id: impl Into<String>, frames: Vec<DataFrame>) -> Self {
        let mut event = Self::new(EventType::TaskInteraction);
        event.task_id = Some(task_id.into());
        event.interaction = frames;
        event
    }
    /// Builds a task completion event carrying the task result.
    ///
    /// Object results are stored as-is; every other value is wrapped under the
    /// `"result"` key.
    #[must_use]
    pub fn task_completion(task_id: impl Into<String>, result: impl Into<Value>) -> Self {
        let data = match result.into() {
            Value::Object(map) => map.into_iter().collect(),
            other => BTreeMap::from([("result".to_owned(), other)]),
        };
        let mut event = Self::new(EventType::TaskCompletion);
        event.task_id = Some(task_id.into());
        event.task_result = vec![DataFrame::Json(JsonDataFrame::new(data))];
        event
    }
    /// Builds a task failed event.
    #[must_use]
    pub fn task_failed(task_id: impl Into<String>, error: impl Into<String>) -> Self {
        let mut event = Self::new(EventType::TaskFailed);
        event.task_id = Some(task_id.into());
        event.error_message = Some(error.into());
        event
    }
    /// Builds a follow-up event carrying one text frame.
    #[must_use]
    pub fn follow_up(text: impl Into<String>) -> Self {
        let mut event = Self::new(EventType::FollowUp);
        event.input_data = vec![DataFrame::text(text)];
        event
    }
    /// Renders the event's payload as text.
    ///
    /// Text and JSON frames of every payload list are joined with `"\n"`;
    /// file frames are ignored.
    #[must_use]
    pub fn text(&self) -> Option<String> {
        let mut frames: Vec<DataFrame> = self.input_data.clone();
        frames.extend(self.task_result.iter().cloned());
        frames.extend(self.interaction.iter().cloned());
        frames_text(&frames)
    }
    /// Returns the payload frames that belong to this event kind.
    ///
    /// [`EventType::Input`] and [`EventType::FollowUp`] expose `input_data`,
    /// [`EventType::TaskCompletion`] exposes `task_result`,
    /// [`EventType::TaskInteraction`] exposes `interaction`, and
    /// [`EventType::TaskFailed`] has no frame list.
    #[must_use]
    pub fn frames(&self) -> &[DataFrame] {
        match self.event_type {
            EventType::Input | EventType::FollowUp => &self.input_data,
            EventType::TaskCompletion => &self.task_result,
            EventType::TaskInteraction => &self.interaction,
            EventType::TaskFailed => &[],
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn user_input_wire_shape() {
        let value = serde_json::to_value(Event::user_input("hi")).unwrap();
        assert_eq!(value["event_type"], json!("input"));
        assert_eq!(value["input_data"], json!([{"type": "text", "text": "hi"}]));
        assert!(value.get("created_at").is_none());
        assert!(value.get("timestamp").is_none());
        assert!(value.get("metadata").is_none());
        assert!(value["event_id"].as_str().is_some_and(|id| !id.is_empty()));
    }
    #[test]
    fn from_user_inputs_string_and_array() {
        let event = Event::from_user_inputs(&json!("hello")).unwrap();
        assert_eq!(event.event_type, EventType::Input);
        assert_eq!(event.input_data, vec![DataFrame::text("hello")]);
        let event = Event::from_user_inputs(&json!(["a", 2])).unwrap();
        assert_eq!(
            event.input_data,
            vec![DataFrame::text("a"), DataFrame::text("2")]
        );
    }
    #[test]
    fn from_user_inputs_query_object() {
        let event =
            Event::from_user_inputs(&json!({"query": {"user_inputs": {"c1": "y"}}})).unwrap();
        assert_eq!(event.event_type, EventType::Input);
        assert_eq!(
            event.input_data,
            vec![DataFrame::json(
                json!({"query": {"user_inputs": {"c1": "y"}}})
            )]
        );
        let text = Event::from_user_inputs(&json!({"query": "hi"})).unwrap();
        assert_eq!(text.input_data, vec![DataFrame::text("hi")]);
    }
    #[test]
    fn from_user_inputs_interactive_and_plain_objects() {
        let interactive = json!({"user_inputs": {"c1": "yes"}, "interrupt_answers": {}});
        let event = Event::from_user_inputs(&interactive).unwrap();
        assert_eq!(event.input_data, vec![DataFrame::json(interactive.clone())]);
        let plain = Event::from_user_inputs(&json!({"k": 1})).unwrap();
        assert_eq!(plain.input_data, vec![DataFrame::json(json!({"k": 1}))]);
    }
    #[test]
    fn from_user_inputs_rejects_empty_and_scalars() {
        assert!(Event::from_user_inputs(&Value::Null).is_err());
        assert!(Event::from_user_inputs(&json!(3)).is_err());
        assert!(Event::from_user_inputs(&json!(true)).is_err());
    }
    #[test]
    fn completion_and_failure_round_trip() {
        let completion = Event::task_completion("t1", json!({"answer": 42}));
        assert_eq!(completion.event_type, EventType::TaskCompletion);
        assert_eq!(completion.task_id.as_deref(), Some("t1"));
        assert_eq!(
            completion.task_result,
            vec![DataFrame::json(json!({"answer": 42}))]
        );
        let wire = serde_json::to_value(&completion).unwrap();
        let parsed: Event = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, completion);
        let scalar = Event::task_completion("t1", json!("done"));
        assert_eq!(
            scalar.task_result,
            vec![DataFrame::json(json!({"result": "done"}))]
        );
        let failed = Event::task_failed("t1", "boom");
        let wire = serde_json::to_value(&failed).unwrap();
        assert_eq!(wire["event_type"], json!("task_failed"));
        assert_eq!(wire["error_message"], json!("boom"));
        let parsed: Event = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, failed);
    }
    #[test]
    fn text_and_frames_follow_the_event_kind() {
        let input = Event::user_input("hello");
        assert_eq!(input.text().as_deref(), Some("hello"));
        assert_eq!(input.frames().len(), 1);
        let completion = Event::task_completion("t1", json!({"a": 1}));
        assert_eq!(completion.text().as_deref(), Some("{\"a\":1}"));
        assert_eq!(completion.frames().len(), 1);
        let failed = Event::task_failed("t1", "boom");
        assert!(failed.text().is_none());
        assert_eq!(failed.frames().len(), 0);
        let follow_up = Event::follow_up("more");
        assert_eq!(follow_up.event_type, EventType::FollowUp);
        assert_eq!(follow_up.text().as_deref(), Some("more"));
    }
    #[test]
    fn event_type_wire_names() {
        assert_eq!(EventType::Input.as_str(), "input");
        assert_eq!(EventType::TaskInteraction.as_str(), "task_interaction");
        assert_eq!(EventType::TaskCompletion.as_str(), "task_completion");
        assert_eq!(EventType::TaskFailed.as_str(), "task_failed");
        assert_eq!(EventType::FollowUp.as_str(), "follow_up");
    }
}
