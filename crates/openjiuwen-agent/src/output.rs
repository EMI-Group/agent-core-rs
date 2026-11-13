//! Agent outputs and stream chunks.
//!
//! [`AgentOutput`] is the terminal result of an agent invocation and serializes
//! to the Python result dict; [`AgentChunk`] is one frame of a streaming
//! invocation and bridges to [`openjiuwen_session::StreamItem`].
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::Usage;
use openjiuwen_session::StreamItem;
use openjiuwen_workflow::WorkflowOutput;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
/// Kind of terminal agent result.
///
/// Wire names are `answer`, `error`, and `interrupt`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultType {
    /// The agent produced a final answer.
    #[default]
    Answer,
    /// The agent failed to produce an answer.
    Error,
    /// The agent paused for human input and can be resumed.
    Interrupt,
}
/// One tool observation recorded during a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Observation {
    /// Rendered tool output text.
    pub content: String,
    /// Name of the tool that produced the observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Id of the tool call the observation answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}
/// A persisted output frame; interrupt frames use [`INTERACTION`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OutputFrame {
    /// Frame type, serialized as `"type"` for wire parity.
    #[serde(rename = "type")]
    pub kind: String,
    /// Position within the producer's frame sequence.
    pub index: i64,
    /// Frame payload.
    pub payload: Value,
}
impl OutputFrame {
    /// Builds an `__interaction__` frame carrying `id` and `value`.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)] // the public signature takes ownership
    pub fn interaction(index: i64, id: &str, value: Value) -> Self {
        Self {
            kind: INTERACTION.to_owned(),
            index,
            payload: serde_json::json!({"id": id, "value": value}),
        }
    }
    /// Projects the frame to its JSON wire form.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
    /// Parses a frame from its JSON wire form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `value` is not a frame object.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|error| Error::validation(error.to_string()))
    }
}
/// Terminal result of an agent invocation.
///
/// `None` and empty fields are skipped on serialization so the wire projection
/// matches the Python result dict. `workflow_output` uses Python's
/// `workflow_execution_state` key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentOutput {
    /// Answer text; workflow agents carry the JSON text of the result here.
    pub output: String,
    /// Kind of terminal result.
    pub result_type: ResultType,
    /// Provider finish reason, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Whether the model emitted reasoning/thinking content.
    ///
    /// Python's plain answer dict is only `{"output", "result_type"}`, so the
    /// `false` value is omitted from the wire object.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub reasoning_present: bool,
    /// Why the loop stopped early (`max_iterations`, `token_budget`, `cancelled`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub termination_reason: Option<String>,
    /// Whether the run was cut off by the iteration limit.
    ///
    /// Python only adds this key to the result dict for a truncated run, so the
    /// `false` value is omitted from the wire object.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated_by_max_iterations: bool,
    /// Number of loop iterations consumed, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations_used: Option<usize>,
    /// Tool observations recorded during the run.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<Observation>,
    /// Error raised while finalizing a truncated run, when any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finalization_error: Option<String>,
    /// Interrupt frames (kind [`INTERACTION`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub state: Vec<OutputFrame>,
    /// Ids of the interrupts the run is waiting on.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub interrupt_ids: Vec<String>,
    /// Ids of the workflow components that produced the result.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub component_ids: Vec<String>,
    /// Structured workflow output, serialized as `workflow_execution_state`.
    #[serde(
        rename = "workflow_execution_state",
        skip_serializing_if = "Option::is_none"
    )]
    pub workflow_output: Option<WorkflowOutput>,
    /// Structured payload, e.g. the workflow result object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Extra host-visible fields.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
}
impl AgentOutput {
    /// Builds a successful answer output.
    #[must_use]
    pub fn answer(text: impl Into<String>) -> Self {
        Self {
            output: text.into(),
            result_type: ResultType::Answer,
            ..Self::default()
        }
    }
    /// Builds a failed output carrying `text` as its message.
    #[must_use]
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            output: text.into(),
            result_type: ResultType::Error,
            ..Self::default()
        }
    }
    /// Builds an interrupted output from `frames` and `ids`.
    #[must_use]
    pub fn interrupt(frames: Vec<OutputFrame>, ids: Vec<String>) -> Self {
        Self {
            result_type: ResultType::Interrupt,
            state: frames,
            interrupt_ids: ids,
            ..Self::default()
        }
    }
    /// Sets the provider finish reason.
    #[must_use]
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }
    /// Sets whether reasoning content was present.
    #[must_use]
    pub fn with_reasoning_present(mut self, present: bool) -> Self {
        self.reasoning_present = present;
        self
    }
    /// Sets the number of iterations used.
    #[must_use]
    pub fn with_iterations_used(mut self, iterations: usize) -> Self {
        self.iterations_used = Some(iterations);
        self
    }
    /// Sets why the loop stopped early.
    #[must_use]
    pub fn with_termination_reason(mut self, reason: impl Into<String>) -> Self {
        self.termination_reason = Some(reason.into());
        self
    }
    /// Sets the structured payload.
    #[must_use]
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
    /// Inserts an extra field.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
    /// Replaces the observations.
    #[must_use]
    pub fn with_observations(mut self, observations: Vec<Observation>) -> Self {
        self.observations = observations;
        self
    }
    /// Returns `true` when the run paused for human input.
    #[must_use]
    pub fn is_interrupt(&self) -> bool {
        self.result_type == ResultType::Interrupt
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
    /// Returns [`Error::validation`] when `value` is not an output object.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|error| Error::validation(error.to_string()))
    }
}
/// One streaming frame of an agent invocation.
///
/// `kind` is serialized as `"type"`; see the `*` frame-kind constants.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AgentChunk {
    /// Frame type, serialized as `"type"` for wire parity.
    #[serde(rename = "type")]
    pub kind: String,
    /// Position within the frame's logical sequence.
    pub index: i64,
    /// Frame payload.
    pub payload: Value,
}
/// Frame kind for model output text.
pub const LLM_OUTPUT: &str = "llm_output";
/// Frame kind for model reasoning/thinking text.
pub const LLM_REASONING: &str = "llm_reasoning";
/// Frame kind for model token usage.
pub const LLM_USAGE: &str = "llm_usage";
/// Frame kind for the final answer output.
pub const ANSWER: &str = "answer";
/// Frame kind for an interrupt request.
pub const INTERACTION: &str = "__interaction__";
/// Frame kind for the final workflow output.
pub const WORKFLOW_FINAL: &str = "workflow_final";
/// Frame kind for context/token usage statistics.
pub const CONTEXT_USAGE: &str = "context.usage";
/// Frame kind for a model-route switch (a rail swapped the model for this call).
pub const LLM_ROUTE: &str = "llm_route";
impl AgentChunk {
    /// Builds a frame of `kind`.
    #[must_use]
    pub fn new(kind: impl Into<String>, index: i64, payload: Value) -> Self {
        Self {
            kind: kind.into(),
            index,
            payload,
        }
    }
    /// Builds a model-output text frame.
    #[must_use]
    pub fn llm_output(index: i64, content: impl Into<String>) -> Self {
        let content: String = content.into();
        Self::new(LLM_OUTPUT, index, serde_json::json!({"content": content}))
    }
    /// Builds a model-reasoning text frame.
    #[must_use]
    pub fn llm_reasoning(index: i64, content: impl Into<String>) -> Self {
        let content: String = content.into();
        Self::new(
            LLM_REASONING,
            index,
            serde_json::json!({"content": content}),
        )
    }
    /// Builds a token-usage frame from `usage`.
    #[must_use]
    pub fn llm_usage(index: i64, usage: &Usage) -> Self {
        Self::new(
            LLM_USAGE,
            index,
            serde_json::to_value(usage).unwrap_or(Value::Null),
        )
    }
    /// Builds the final-answer frame for `output`.
    #[must_use]
    pub fn answer(output: &AgentOutput) -> Self {
        Self::new(ANSWER, 0, output.to_value())
    }
    /// Builds an `__interaction__` frame carrying `id` and `value`.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)] // the public signature takes ownership
    pub fn interaction(id: &str, value: Value) -> Self {
        Self::new(
            INTERACTION,
            0,
            serde_json::json!({"id": id, "value": value}),
        )
    }
    /// Builds a final workflow frame carrying `payload`.
    #[must_use]
    pub fn workflow_frame(index: i64, payload: Value) -> Self {
        Self::new(WORKFLOW_FINAL, index, payload)
    }
    /// Projects the frame to its JSON wire form.
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
    /// Parses a frame from its JSON wire form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `value` is not a frame object.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|error| Error::validation(error.to_string()))
    }
    /// Projects the frame to a session [`StreamItem`] in output mode.
    #[must_use]
    pub fn to_stream_item(&self) -> StreamItem {
        StreamItem::output(self.kind.clone(), Some(self.index), self.payload.clone())
    }
    /// Rebuilds a frame from a session [`StreamItem`].
    #[must_use]
    pub fn from_stream_item(item: &StreamItem) -> Self {
        Self {
            kind: item.item_type.clone(),
            index: item.index.unwrap_or(0),
            payload: item.payload.clone(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn result_type_wire_names() {
        assert_eq!(serde_json::to_value(ResultType::Answer).unwrap(), "answer");
        assert_eq!(serde_json::to_value(ResultType::Error).unwrap(), "error");
        assert_eq!(
            serde_json::to_value(ResultType::Interrupt).unwrap(),
            "interrupt"
        );
    }
    #[test]
    fn agent_output_round_trip() {
        let output = AgentOutput::answer("done")
            .with_finish_reason("stop")
            .with_reasoning_present(true)
            .with_iterations_used(2)
            .with_termination_reason("max_iterations")
            .with_data(serde_json::json!({"x": 1}))
            .with_extra("trace", serde_json::json!("t-1"))
            .with_observations(vec![Observation {
                content: "42".to_owned(),
                name: Some("calc".to_owned()),
                tool_call_id: Some("call-1".to_owned()),
            }]);
        let value = output.to_value();
        let restored = AgentOutput::from_value(value).unwrap();
        assert_eq!(restored, output);
    }
    #[test]
    fn agent_output_uses_python_wire_names() {
        let output = AgentOutput::interrupt(
            vec![OutputFrame::interaction(0, "intr_1", serde_json::json!({}))],
            vec!["intr_1".to_owned()],
        );
        let value = output.to_value();
        assert_eq!(value["result_type"], "interrupt");
        let frame = &value["state"][0];
        assert_eq!(frame["type"], "__interaction__");
        assert_eq!(frame["payload"]["id"], "intr_1");
    }
    #[test]
    fn empty_and_none_fields_are_omitted() {
        let value = AgentOutput::answer("hi").to_value();
        assert!(value.get("observations").is_none());
        assert!(value.get("interrupt_ids").is_none());
        assert!(value.get("workflow_execution_state").is_none());
        assert!(value.get("finish_reason").is_none());
    }
    #[test]
    fn reasoning_present_is_absent_unless_true() {
        assert!(AgentOutput::answer("hi")
            .to_value()
            .get("reasoning_present")
            .is_none());
        let present = AgentOutput::answer("hi").with_reasoning_present(true);
        assert_eq!(present.to_value()["reasoning_present"], true);
        let restored = AgentOutput::from_value(AgentOutput::answer("hi").to_value()).unwrap();
        assert!(!restored.reasoning_present);
    }
    #[test]
    fn workflow_output_uses_execution_state_key() {
        let output = AgentOutput {
            workflow_output: Some(WorkflowOutput::default()),
            ..AgentOutput::answer("wf")
        };
        let value = output.to_value();
        assert!(value.get("workflow_execution_state").is_some());
        assert!(value.get("workflow_output").is_none());
        let restored = AgentOutput::from_value(value).unwrap();
        assert_eq!(restored.workflow_output, Some(WorkflowOutput::default()));
    }
    #[test]
    fn output_frame_round_trip() {
        let frame = OutputFrame::interaction(3, "intr_x", serde_json::json!({"k": 1}));
        let value = frame.to_value();
        assert_eq!(value["type"], "__interaction__");
        assert_eq!(value["index"], 3);
        assert_eq!(value["payload"]["value"]["k"], 1);
        assert_eq!(OutputFrame::from_value(value).unwrap(), frame);
    }
    #[test]
    fn output_frame_rejects_non_object() {
        assert!(OutputFrame::from_value(serde_json::json!("nope")).is_err());
    }
    #[test]
    fn chunk_round_trip() {
        let chunk = AgentChunk::llm_output(1, "hello");
        let value = chunk.to_value();
        assert_eq!(value["type"], "llm_output");
        assert_eq!(value["payload"]["content"], "hello");
        assert_eq!(AgentChunk::from_value(value).unwrap(), chunk);
        assert!(AgentChunk::from_value(serde_json::json!(5)).is_err());
    }
    #[test]
    fn chunk_kinds_use_constants() {
        assert_eq!(AgentChunk::llm_reasoning(0, "r").kind, LLM_REASONING);
        assert_eq!(AgentChunk::answer(&AgentOutput::answer("a")).kind, ANSWER);
        assert_eq!(AgentChunk::interaction("i", Value::Null).kind, INTERACTION);
        assert_eq!(
            AgentChunk::workflow_frame(0, Value::Null).kind,
            WORKFLOW_FINAL
        );
        assert_eq!(LLM_OUTPUT, "llm_output");
        assert_eq!(LLM_USAGE, "llm_usage");
        assert_eq!(CONTEXT_USAGE, "context.usage");
        assert_eq!(LLM_ROUTE, "llm_route");
    }
    #[test]
    fn usage_chunk_serializes_usage() {
        let chunk = AgentChunk::llm_usage(0, &Usage::new(3, 4));
        assert_eq!(chunk.payload["input_tokens"], 3);
        assert_eq!(chunk.payload["output_tokens"], 4);
    }
    #[test]
    fn stream_item_round_trip() {
        let chunk = AgentChunk::llm_output(2, "tok");
        let item = chunk.to_stream_item();
        assert_eq!(item.item_type, "llm_output");
        assert_eq!(item.index, Some(2));
        assert_eq!(AgentChunk::from_stream_item(&item), chunk);
        let unindexed = StreamItem::output("answer", None, Value::Null);
        assert_eq!(AgentChunk::from_stream_item(&unindexed).index, 0);
    }
}
