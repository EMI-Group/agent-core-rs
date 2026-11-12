//! Workflow outputs, stream frames, and frame fan-out.
//!
//! Maps the Python `WorkflowOutput`/`WorkflowExecutionState` container and the
//! `OutputSchema`/`CustomSchema`/`TraceSchema` stream payloads onto
//! [`openjiuwen_session::stream::StreamItem`]. [`FrameWriter`] is the single
//! write path used by components and the engine: it forwards every frame to an
//! optional channel sink, to an optional in-memory collector, and to the
//! workflow session's stream.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_session::stream::StreamItem;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::mpsc::UnboundedSender;

/// Frame type used for the buffered output of a streaming end node.
pub const END_NODE_STREAM: &str = "end node stream";

/// Frame type used for the final frame carrying the end component's outputs.
pub const WORKFLOW_FINAL: &str = "workflow_final";

/// Frame type used by the generic output projection.
pub const OUTPUT_FRAME_TYPE: &str = "output";

/// Frame type used for runner-defined custom payloads.
pub const CUSTOM_FRAME_TYPE: &str = "custom";

/// Frame type used for trace payloads.
pub const TRACE_FRAME_TYPE: &str = "trace";

/// One frame of a workflow run stream.
///
/// Python has three schema classes with different fields; Rust uses the single
/// [`StreamItem`] frame whose `mode` selects the consumer. The alias keeps the
/// Python vocabulary in signatures.
pub type WorkflowChunk = StreamItem;

/// In-memory frame buffer used by [`FrameWriter::collect_into`].
pub type FrameCollector = Arc<Mutex<Vec<WorkflowChunk>>>;

/// Possible states of a workflow execution.
///
/// Wire values match Python: `"COMPLETED"`, `"INPUT_REQUIRED"`, and `"ERROR"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkflowExecutionState {
    /// The workflow ran to completion.
    #[default]
    Completed,
    /// The workflow halted waiting for human input and can be resumed.
    InputRequired,
    /// The workflow failed.
    Error,
}

impl WorkflowExecutionState {
    /// The wire name of this state.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "COMPLETED",
            Self::InputRequired => "INPUT_REQUIRED",
            Self::Error => "ERROR",
        }
    }
}

impl std::fmt::Display for WorkflowExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Final output container of a workflow execution.
///
/// `result` is the end component's value, or the list of collected frames when
/// the end node streams. Unknown fields are tolerated so newer Python hosts can
/// feed older consumers.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkflowOutput {
    /// Output data: either the end value or the collected frame list.
    pub result: Value,
    /// Final execution state.
    pub state: WorkflowExecutionState,
}

/// A Python-style `OutputSchema` frame: `{"type", "index", "payload"}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OutputFrame {
    /// Frame type, serialized as `"type"` for wire parity.
    #[serde(rename = "type")]
    pub item_type: String,
    /// Position within the producer's frame sequence.
    pub index: u32,
    /// Frame payload.
    pub payload: Value,
}

impl OutputFrame {
    /// The wire projection `{"type": ..., "index": ..., "payload": ...}`.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        let mut map = Map::new();
        map.insert("type".to_owned(), Value::String(self.item_type.clone()));
        map.insert("index".to_owned(), Value::from(self.index));
        map.insert("payload".to_owned(), self.payload.clone());
        Value::Object(map)
    }

    /// Convert this frame into an output-mode session stream item.
    #[must_use]
    pub fn to_stream_item(&self) -> StreamItem {
        StreamItem::output_at(&self.item_type, i64::from(self.index), self.payload.clone())
    }
}

/// Build an indexed output-mode frame.
#[must_use]
pub fn output_chunk(item_type: impl Into<String>, index: u32, payload: Value) -> WorkflowChunk {
    StreamItem::output_at(item_type, i64::from(index), payload)
}

/// Build a custom-mode frame.
#[must_use]
pub fn custom_chunk(payload: Value) -> WorkflowChunk {
    StreamItem::custom(payload)
}

/// Build a trace-mode frame.
#[must_use]
pub fn trace_chunk(item_type: impl Into<String>, payload: Value) -> WorkflowChunk {
    StreamItem::trace(item_type, payload)
}

/// The wire projection of `frame`.
#[must_use]
pub fn frame_to_wire(frame: &OutputFrame) -> Value {
    frame.to_wire()
}

/// Fan-out writer for workflow stream frames.
///
/// Every frame is offered to the optional channel sink (a closed sink is a
/// debug log, never an error), then appended to the optional collector, and
/// finally written to the session. The session write is the authoritative one:
/// its error is propagated to the caller.
#[derive(Clone)]
pub struct FrameWriter {
    sink: Option<UnboundedSender<WorkflowChunk>>,
    session: Arc<dyn Session>,
    collector: Arc<Mutex<Option<FrameCollector>>>,
}

impl std::fmt::Debug for FrameWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameWriter")
            .field("has_sink", &self.sink.is_some())
            .field("has_collector", &self.collector.lock().is_some())
            .finish_non_exhaustive()
    }
}

impl FrameWriter {
    /// Create a writer forwarding to `sink` (when present) and `session`.
    #[must_use]
    pub fn new(sink: Option<UnboundedSender<WorkflowChunk>>, session: Arc<dyn Session>) -> Self {
        Self {
            sink,
            session,
            collector: Arc::new(Mutex::new(None)),
        }
    }

    /// Attach an in-memory collector that receives a clone of every frame.
    pub fn collect_into(&self, collector: FrameCollector) {
        *self.collector.lock() = Some(collector);
    }

    /// Publish `item` to the sink, the collector, and the session.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::write_stream`] errors (for example a validation
    /// error for a `null` payload).
    pub async fn write(&self, item: WorkflowChunk) -> Result<()> {
        if let Some(sink) = &self.sink {
            if sink.send(item.clone()).is_err() {
                tracing::debug!(
                    item_type = item.item_type.as_str(),
                    "workflow stream sink closed; dropping frame"
                );
            }
        }
        {
            let collector = self.collector.lock().clone();
            if let Some(collector) = collector {
                collector.lock().push(item.clone());
            }
        }
        self.session.write_stream(item).await
    }

    /// Write an indexed output-mode frame.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::write_stream`] errors.
    pub async fn write_output(
        &self,
        item_type: impl Into<String>,
        index: u32,
        payload: Value,
    ) -> Result<()> {
        self.write(output_chunk(item_type, index, payload)).await
    }

    /// Write a custom-mode frame.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::write_stream`] errors.
    pub async fn write_custom(&self, payload: Value) -> Result<()> {
        self.write(custom_chunk(payload)).await
    }

    /// Drain the frames accumulated in the collector.
    ///
    /// Returns an empty vector when no collector was attached. The collector is
    /// left empty, so a second call returns nothing.
    #[must_use]
    pub fn drained(&self) -> Vec<WorkflowChunk> {
        let collector = self.collector.lock().clone();
        let Some(collector) = collector else {
            return Vec::new();
        };
        let mut frames = collector.lock();
        std::mem::take(&mut *frames)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use openjiuwen_session::WorkflowSession;
    use serde_json::json;

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("test-session"))
    }

    #[test]
    fn execution_state_uses_python_wire_names() {
        assert_eq!(
            serde_json::to_value(WorkflowExecutionState::Completed).unwrap(),
            json!("COMPLETED")
        );
        assert_eq!(
            serde_json::to_value(WorkflowExecutionState::InputRequired).unwrap(),
            json!("INPUT_REQUIRED")
        );
        assert_eq!(
            serde_json::to_value(WorkflowExecutionState::Error).unwrap(),
            json!("ERROR")
        );
        let parsed: WorkflowExecutionState =
            serde_json::from_value(json!("INPUT_REQUIRED")).unwrap();
        assert_eq!(parsed, WorkflowExecutionState::InputRequired);
        assert_eq!(
            WorkflowExecutionState::default(),
            WorkflowExecutionState::Completed
        );
        assert_eq!(WorkflowExecutionState::Completed.as_str(), "COMPLETED");
        assert_eq!(
            WorkflowExecutionState::InputRequired.to_string(),
            "INPUT_REQUIRED"
        );
    }

    #[test]
    fn workflow_output_round_trips_and_tolerates_unknown_fields() {
        let output = WorkflowOutput {
            result: json!({"answer": 42}),
            state: WorkflowExecutionState::Completed,
        };
        let wire = serde_json::to_value(&output).unwrap();
        assert_eq!(wire["state"], json!("COMPLETED"));
        let parsed: WorkflowOutput = serde_json::from_value(json!({
            "result": {"answer": 42},
            "state": "COMPLETED",
            "future_field": true
        }))
        .unwrap();
        assert_eq!(parsed, output);
    }

    #[test]
    fn output_frame_projects_python_wire_shape() {
        let frame = OutputFrame {
            item_type: "answer".to_owned(),
            index: 3,
            payload: json!({"text": "hi"}),
        };
        assert_eq!(
            frame.to_wire(),
            json!({"type": "answer", "index": 3, "payload": {"text": "hi"}})
        );
        assert_eq!(frame_to_wire(&frame), frame.to_wire());

        let item = frame.to_stream_item();
        assert_eq!(item.item_type, "answer");
        assert_eq!(item.index, Some(3));
        assert_eq!(item.payload, json!({"text": "hi"}));
    }

    #[test]
    fn chunk_constructors_select_modes() {
        let output = output_chunk(END_NODE_STREAM, 2, json!("x"));
        assert_eq!(output.mode, openjiuwen_session::stream::StreamMode::Output);
        assert_eq!(output.index, Some(2));

        let custom = custom_chunk(json!({"a": 1}));
        assert_eq!(custom.mode, openjiuwen_session::stream::StreamMode::Custom);
        assert_eq!(custom.index, None);

        let trace = trace_chunk(TRACE_FRAME_TYPE, json!({"span": 1}));
        assert_eq!(trace.mode, openjiuwen_session::stream::StreamMode::Trace);
    }

    #[tokio::test]
    async fn frame_writer_forwards_to_sink_session_and_collector() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let writer = FrameWriter::new(Some(tx), session());
        let collector = Arc::new(Mutex::new(Vec::new()));
        writer.collect_into(Arc::clone(&collector));

        writer
            .write_output(WORKFLOW_FINAL, 0, json!({"done": true}))
            .await
            .unwrap();

        let received = rx.try_recv().unwrap();
        assert_eq!(received.item_type, WORKFLOW_FINAL);
        assert_eq!(received.payload, json!({"done": true}));
        assert_eq!(collector.lock().len(), 1);

        let drained = writer.drained();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].item_type, WORKFLOW_FINAL);
        assert!(writer.drained().is_empty());
    }

    #[tokio::test]
    async fn frame_writer_ignores_a_closed_sink() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(rx);
        let writer = FrameWriter::new(Some(tx), session());
        let collector = Arc::new(Mutex::new(Vec::new()));
        writer.collect_into(Arc::clone(&collector));

        writer
            .write_custom(json!({"still": "delivered"}))
            .await
            .unwrap();

        assert_eq!(collector.lock().len(), 1);
    }

    #[tokio::test]
    async fn frame_writer_without_collector_drains_empty() {
        let writer = FrameWriter::new(None, session());
        writer.write_custom(json!({"a": 1})).await.unwrap();
        assert!(writer.drained().is_empty());
    }

    #[tokio::test]
    async fn frame_writer_propagates_session_errors() {
        let writer = FrameWriter::new(None, session());
        assert!(writer
            .write(output_chunk(OUTPUT_FRAME_TYPE, 0, Value::Null))
            .await
            .is_err());
    }
}
