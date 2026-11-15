//! Controller outputs, streaming and batch.
//!
//! Ports `openjiuwen/core/controller/schema/controller_output.py`: a
//! [`ControllerOutputChunk`] is one `controller_output` stream frame and a
//! [`ControllerOutput`] is the aggregated batch result. The two
//! [`TASK_PROCESSING`] / [`ALL_TASKS_PROCESSED`] sentinels travel as payload
//! kinds, never as task statuses.
use super::dataframe::DataFrame;
use super::task::TaskStatus;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::stream::StreamItem;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// Stream frame type of every controller output chunk.
pub const CONTROLLER_OUTPUT_TYPE: &str = "controller_output";
/// Payload kind sent while a task is still executing.
pub const TASK_PROCESSING: &str = "processing";
/// Payload kind sent once every task has been processed.
pub const ALL_TASKS_PROCESSED: &str = "all_tasks_processed";
/// Payload type discriminator.
///
/// The first three values mirror [`super::event::EventType`];
/// [`OutputKind::Processing`] and [`OutputKind::AllTasksProcessed`] are
/// controller stream sentinels, while [`OutputKind::Reasoning`] marks a frame
/// carrying a task's reasoning text, emitted only when the controller config
/// exposes reasoning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// A task finished successfully.
    #[default]
    TaskCompletion,
    /// A task needs user interaction.
    TaskInteraction,
    /// A task failed.
    TaskFailed,
    /// A task is still executing.
    Processing,
    /// Every queued task has been processed.
    AllTasksProcessed,
    /// A task's reasoning text, exposed only when the controller config asks
    /// for it.
    Reasoning,
}
impl OutputKind {
    /// Returns the Python wire string of the payload kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TaskCompletion => "task_completion",
            Self::TaskInteraction => "task_interaction",
            Self::TaskFailed => "task_failed",
            Self::Processing => TASK_PROCESSING,
            Self::AllTasksProcessed => ALL_TASKS_PROCESSED,
            Self::Reasoning => "reasoning",
        }
    }
}
/// One controller stream payload: its kind, frames, and metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ControllerOutputPayload {
    /// Payload kind, serialized as `"type"` for wire parity.
    #[serde(rename = "type")]
    pub kind: OutputKind,
    /// Frames the payload carries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data: Vec<DataFrame>,
    /// Free-form payload metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}
impl ControllerOutputPayload {
    /// Builds a payload of `kind` carrying `data`.
    #[must_use]
    pub fn new(kind: OutputKind, data: Vec<DataFrame>) -> Self {
        Self {
            kind,
            data,
            metadata: BTreeMap::new(),
        }
    }
    /// Builds the `processing` sentinel payload.
    #[must_use]
    pub fn processing() -> Self {
        Self::new(OutputKind::Processing, Vec::new())
    }
    /// Builds the `all_tasks_processed` sentinel payload.
    #[must_use]
    pub fn all_tasks_processed() -> Self {
        Self::new(OutputKind::AllTasksProcessed, Vec::new())
    }
    /// Builds a payload of `kind` carrying one text frame.
    #[must_use]
    pub fn text(kind: OutputKind, text: impl Into<String>) -> Self {
        Self::new(kind, vec![DataFrame::text(text)])
    }
}
/// Default chunk type (`"controller_output"`).
fn controller_output_type() -> String {
    CONTROLLER_OUTPUT_TYPE.to_owned()
}
/// One chunk of controller streaming output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ControllerOutputChunk {
    /// Position within the controller's chunk sequence.
    pub index: i64,
    /// Frame type, `"controller_output"` unless overridden.
    #[serde(rename = "type", default = "controller_output_type")]
    pub kind: String,
    /// Payload of this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<ControllerOutputPayload>,
    /// Whether this is the final chunk of the stream.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub last_chunk: bool,
}
impl Default for ControllerOutputChunk {
    fn default() -> Self {
        Self {
            index: 0,
            kind: controller_output_type(),
            payload: None,
            last_chunk: false,
        }
    }
}
impl ControllerOutputChunk {
    /// Builds a non-final chunk carrying `payload`.
    #[must_use]
    pub fn new(index: i64, payload: ControllerOutputPayload) -> Self {
        Self {
            index,
            payload: Some(payload),
            ..Self::default()
        }
    }
    /// Builds the final chunk carrying `payload`.
    #[must_use]
    pub fn last(index: i64, payload: ControllerOutputPayload) -> Self {
        Self {
            last_chunk: true,
            ..Self::new(index, payload)
        }
    }
    /// Projects the chunk to its JSON wire form.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
    /// Bridges the chunk to a session stream frame.
    ///
    /// A chunk without a payload serializes the `all_tasks_processed`
    /// sentinel, matching Python's default payload.
    #[must_use]
    pub fn to_stream_item(&self) -> StreamItem {
        let payload = match &self.payload {
            Some(payload) => serde_json::to_value(payload).unwrap_or(Value::Null),
            None => serde_json::to_value(ControllerOutputPayload::all_tasks_processed())
                .unwrap_or(Value::Null),
        };
        StreamItem::output(CONTROLLER_OUTPUT_TYPE, Some(self.index), payload)
    }
    /// Parses a chunk from a session stream frame.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a `controller_output` frame carries a
    /// payload that is not a [`ControllerOutputPayload`].
    pub fn from_stream_item(item: &StreamItem) -> Result<Option<Self>> {
        if item.item_type != CONTROLLER_OUTPUT_TYPE {
            return Ok(None);
        }
        let payload = if item.payload.is_null() {
            None
        } else {
            Some(
                serde_json::from_value(item.payload.clone())
                    .map_err(|error| Error::validation(error.to_string()))?,
            )
        };
        Ok(Some(Self {
            index: item.index.unwrap_or(0),
            kind: item.item_type.clone(),
            payload,
            last_chunk: false,
        }))
    }
}
/// Batch controller result.
///
/// This is the value a controller `invoke` returns: the answer text, the task
/// it came from, the final status, an optional structured payload, and the
/// chunks that produced it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ControllerOutput {
    /// Answer text.
    pub output: String,
    /// Task the output came from, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Final status of the producing task.
    pub status: TaskStatus,
    /// Structured payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Chunks that produced the answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frames: Vec<ControllerOutputChunk>,
}
impl ControllerOutput {
    /// Builds an output with `output` and `status`.
    #[must_use]
    pub fn new(output: impl Into<String>, status: TaskStatus) -> Self {
        Self {
            output: output.into(),
            status,
            ..Self::default()
        }
    }
    /// Builds an empty completed output.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(String::new(), TaskStatus::Completed)
    }
    /// Whether the output pauses for caller input.
    #[must_use]
    pub fn is_interrupt(&self) -> bool {
        self.status == TaskStatus::InputRequired
    }
    /// Projects the output to its JSON wire form.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
    /// Parses an output from its JSON wire form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `value` is not a controller output.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|error| Error::validation(error.to_string()))
    }
    /// Sets the producing task id.
    #[must_use]
    pub fn with_task_id(mut self, id: impl Into<String>) -> Self {
        self.task_id = Some(id.into());
        self
    }
    /// Sets the structured payload.
    #[must_use]
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
    /// Sets the chunks that produced the answer.
    #[must_use]
    pub fn with_frames(mut self, frames: Vec<ControllerOutputChunk>) -> Self {
        self.frames = frames;
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn sentinel_constants() {
        assert_eq!(CONTROLLER_OUTPUT_TYPE, "controller_output");
        assert_eq!(TASK_PROCESSING, "processing");
        assert_eq!(ALL_TASKS_PROCESSED, "all_tasks_processed");
        assert_eq!(OutputKind::Processing.as_str(), TASK_PROCESSING);
        assert_eq!(OutputKind::AllTasksProcessed.as_str(), ALL_TASKS_PROCESSED);
        assert_eq!(OutputKind::TaskCompletion.as_str(), "task_completion");
    }
    #[test]
    fn output_round_trips() {
        let chunk = ControllerOutputChunk::new(
            0,
            ControllerOutputPayload::text(OutputKind::TaskCompletion, "done"),
        );
        let output = ControllerOutput::new("done", TaskStatus::Completed)
            .with_task_id("t1")
            .with_data(json!({"k": 1}))
            .with_frames(vec![chunk]);
        let wire = output.to_value();
        assert_eq!(wire["status"], json!("completed"));
        assert_eq!(wire["frames"][0]["type"], json!("controller_output"));
        assert_eq!(
            wire["frames"][0]["payload"]["type"],
            json!("task_completion")
        );
        let parsed = ControllerOutput::from_value(wire).unwrap();
        assert_eq!(parsed, output);
        assert!(!parsed.is_interrupt());
    }
    #[test]
    fn empty_and_interrupt_outputs() {
        assert_eq!(ControllerOutput::empty().status, TaskStatus::Completed);
        assert_eq!(ControllerOutput::default().output, "");
        let interrupt = ControllerOutput::new("need input", TaskStatus::InputRequired);
        assert!(interrupt.is_interrupt());
        assert_eq!(
            serde_json::to_value(&interrupt).unwrap()["status"],
            json!("input-required")
        );
        assert!(ControllerOutput::from_value(json!("nope")).is_err());
    }
    #[test]
    fn payload_helpers() {
        assert_eq!(
            serde_json::to_value(ControllerOutputPayload::processing()).unwrap(),
            json!({"type": "processing"})
        );
        assert_eq!(
            serde_json::to_value(ControllerOutputPayload::all_tasks_processed()).unwrap(),
            json!({"type": "all_tasks_processed"})
        );
        let text = ControllerOutputPayload::text(OutputKind::TaskFailed, "boom");
        assert_eq!(
            serde_json::to_value(&text).unwrap(),
            json!({"type": "task_failed", "data": [{"type": "text", "text": "boom"}]})
        );
    }
    #[test]
    fn chunk_bridges_to_stream_item() {
        let chunk = ControllerOutputChunk::new(
            2,
            ControllerOutputPayload::text(OutputKind::TaskInteraction, "need input"),
        );
        let item = chunk.to_stream_item();
        assert_eq!(item.item_type, CONTROLLER_OUTPUT_TYPE);
        assert_eq!(item.index, Some(2));
        let back = ControllerOutputChunk::from_stream_item(&item)
            .unwrap()
            .unwrap();
        assert_eq!(back, chunk);
        let foreign = StreamItem::output("llm_output", Some(0), json!({}));
        assert!(ControllerOutputChunk::from_stream_item(&foreign)
            .unwrap()
            .is_none());
        let bare = ControllerOutputChunk::default().to_stream_item();
        assert_eq!(bare.payload["type"], json!("all_tasks_processed"));
        assert_eq!(
            ControllerOutputChunk::from_stream_item(&bare)
                .unwrap()
                .unwrap()
                .payload,
            Some(ControllerOutputPayload::all_tasks_processed())
        );
    }
    #[test]
    fn reasoning_kind_wire_shape() {
        assert_eq!(OutputKind::Reasoning.as_str(), "reasoning");
        assert_eq!(
            serde_json::to_value(OutputKind::Reasoning).unwrap(),
            json!("reasoning")
        );
        let parsed: OutputKind = serde_json::from_value(json!("reasoning")).unwrap();
        assert_eq!(parsed, OutputKind::Reasoning);
    }
    #[test]
    fn reasoning_kind_is_distinct_from_sentinels() {
        assert_ne!(OutputKind::Reasoning.as_str(), TASK_PROCESSING);
        assert_ne!(OutputKind::Reasoning.as_str(), ALL_TASKS_PROCESSED);
        assert_ne!(OutputKind::Reasoning, OutputKind::Processing);
        assert_ne!(OutputKind::Reasoning, OutputKind::AllTasksProcessed);
        assert_ne!(OutputKind::Reasoning, OutputKind::default());
    }
    #[test]
    fn last_chunk_is_serialized_only_when_true() {
        let last = ControllerOutputChunk::last(1, ControllerOutputPayload::processing());
        assert_eq!(
            serde_json::to_value(&last).unwrap()["last_chunk"],
            json!(true)
        );
        let first = ControllerOutputChunk::new(0, ControllerOutputPayload::processing());
        assert!(first.to_value().get("last_chunk").is_none());
        assert_eq!(first.kind, CONTROLLER_OUTPUT_TYPE);
    }
}
