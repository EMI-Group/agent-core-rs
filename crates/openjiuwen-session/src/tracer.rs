//! Trace spans, frames, sinks, and the tracer that mints them.
//!
//! Ports `openjiuwen/core/session/tracer/` (v0.1.20): `data.py` (the
//! [`InvokeType`] and node-status vocabularies), `span.py` (`Span`,
//! `TraceAgentSpan`, `TraceWorkflowSpan`, and `SpanManager`), `handler.py` (the
//! built-in frame handlers), and `tracer.py` (the [`Tracer`] span factory).
//!
//! The [`TracePosition`] type is the resume point persisted in
//! [`crate::checkpoint::Checkpoint`]; it keeps Python's `trace_state` shape
//! (`trace_id`, `last_invoke_id`, `frame_count`).
//!
//! # Deviations from Python
//!
//! - Python's two span classes (`TraceAgentSpan`, `TraceWorkflowSpan`) collapse
//!   into one [`TraceSpan`] with a [`TraceSpanKind`] discriminator; the wire
//!   keys stay `camelCase` and the two frame types (`tracer_agent`,
//!   `tracer_workflow`) are preserved.
//! - Python's two async handler families (agent and workflow, plus a
//!   process-global extension registry) collapse into one synchronous
//!   [`TraceSink`] trait injected per session. A sink never awaits, so
//!   [`Tracer`] and [`SpanHandle`] can emit from synchronous code.
//! - Timestamps are epoch milliseconds (Python uses `datetime`) and
//!   `elapsedTime` is a millisecond count, so a trace frame is plain JSON.
//! - Python's `SpanManager` per-trace span registry is not ported: a
//!   [`SpanHandle`] is itself the live span, and parent/child links are
//!   recorded directly on the handles.
//!
//! ```
//! use std::sync::Arc;
//!
//! use openjiuwen_session::{InvokeType, MemoryTraceSink, Tracer};
//!
//! let sink = Arc::new(MemoryTraceSink::new());
//! let tracer = Tracer::with_sink("session-1", "trace-1", sink.clone());
//! let span = tracer.start_span("answer", InvokeType::Llm);
//! span.finish(serde_json::json!({ "text": "hi" }));
//!
//! assert_eq!(sink.len(), 1);
//! assert_eq!(sink.frames()[0].payload.invoke_id, span.id());
//! ```
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{unix_millis, uuid_hex};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::stream::{StreamItem, StreamWriterManager};

/// The position of the last emitted trace frame within a session.
///
/// Checkpoints persist the position so a resumed run can continue the trace
/// where it stopped instead of starting a new one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TracePosition {
    /// The trace identifier shared by every frame of the run.
    pub trace_id: String,
    /// The invoke id of the most recently emitted span, if any.
    pub last_invoke_id: Option<String>,
    /// The number of frames emitted so far.
    pub frame_count: u64,
}

/// What kind of work an agent span covers.
///
/// Mirrors Python's `InvokeType` (`data.py`) exactly; the `snake_case` wire
/// values are `prompt`, `llm`, `plugin`, `workflow`, `chain`, `retriever`, and
/// `evaluator`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvokeType {
    /// A prompt/template invocation.
    #[default]
    Prompt,
    /// A model invocation.
    Llm,
    /// A tool/plugin invocation.
    Plugin,
    /// A workflow invocation.
    Workflow,
    /// A chain invocation.
    Chain,
    /// A retriever invocation.
    Retriever,
    /// An evaluator invocation.
    Evaluator,
}

impl InvokeType {
    /// The `snake_case` wire name of this invoke type.
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref)] // signature fixed by the crate's API surface
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Llm => "llm",
            Self::Plugin => "plugin",
            Self::Workflow => "workflow",
            Self::Chain => "chain",
            Self::Retriever => "retriever",
            Self::Evaluator => "evaluator",
        }
    }
}

/// Lifecycle status of a span.
///
/// Mirrors Python's `NodeStatus` (`data.py`) with the same wire values
/// (`start`, `running`, `finish`, `interrupted`, `error`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    /// The span was created and has not finished yet.
    #[default]
    Start,
    /// The span is executing.
    Running,
    /// The span completed successfully.
    Finish,
    /// The span was interrupted (human-in-the-loop, cancellation, …).
    Interrupted,
    /// The span failed.
    Error,
}

impl SpanStatus {
    /// The `snake_case` wire name of this status.
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref)] // signature fixed by the crate's API surface
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Running => "running",
            Self::Finish => "finish",
            Self::Interrupted => "interrupted",
            Self::Error => "error",
        }
    }

    /// Whether this status is final, so a span must not be ended again.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Finish | Self::Interrupted | Self::Error)
    }
}

/// Which span flavour a [`TraceSpan`] represents.
///
/// Replaces Python's `TraceAgentSpan`/`TraceWorkflowSpan` class pair. The kind
/// selects the frame type emitted by [`SpanHandle`]: [`Self::Agent`] frames are
/// `tracer_agent`, while [`Self::Workflow`] and [`Self::Component`] frames are
/// `tracer_workflow` (component spans are workflow spans on the wire, matching
/// Python's workflow handlers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceSpanKind {
    /// An agent execution span (`tracer_agent`).
    #[default]
    Agent,
    /// A workflow execution span (`tracer_workflow`).
    Workflow,
    /// A workflow component execution span (`tracer_workflow`).
    Component,
}

/// The stream frame type of a trace frame.
///
/// The wire values are `tracer_agent` and `tracer_workflow`, matching Python's
/// `TraceAgentHandler`/`TraceWorkflowHandler` frame types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceFrameType {
    /// A frame carrying an agent span.
    #[default]
    #[serde(rename = "tracer_agent")]
    Agent,
    /// A frame carrying a workflow or component span.
    #[serde(rename = "tracer_workflow")]
    Workflow,
}

impl TraceFrameType {
    /// The stream frame type written in [`TraceFrame::item_type`].
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref)] // signature fixed by the crate's API surface
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Agent => "tracer_agent",
            Self::Workflow => "tracer_workflow",
        }
    }
}

impl fmt::Display for TraceFrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One trace span: Python's `Span` plus the union of both span subclasses.
///
/// Field names serialize as `camelCase` because they are the canonical wire
/// keys of Python's pydantic aliases (`traceId`, `invokeId`, `childInvokes`,
/// …). `None` values are serialized as `null` rather than omitted so the wire
/// shape stays stable for consumers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TraceSpan {
    /// Trace identifier shared by every span of the run (`traceId`).
    pub trace_id: String,
    /// Session identifier the span belongs to (`sessionId`).
    pub session_id: String,
    /// Unique identifier of this span (`invokeId`).
    pub invoke_id: String,
    /// The enclosing span's invoke id, if any (`parentInvokeId`).
    pub parent_invoke_id: Option<String>,
    /// The invoke ids of direct children (`childInvokes`).
    pub child_invokes: Vec<String>,
    /// Human-readable span name (`name`).
    pub name: String,
    /// Span flavour, serialized as `spanKind`.
    #[serde(rename = "spanKind")]
    pub kind: TraceSpanKind,
    /// The kind of work this span covers (`invokeType`).
    pub invoke_type: Option<InvokeType>,
    /// Lifecycle status (`status`).
    pub status: SpanStatus,
    /// Start timestamp in epoch milliseconds (`startTime`).
    pub start_time: Option<u64>,
    /// End timestamp in epoch milliseconds (`endTime`).
    pub end_time: Option<u64>,
    /// Duration in milliseconds (`elapsedTime`).
    pub elapsed_time: Option<u64>,
    /// Span inputs (`inputs`).
    pub inputs: Option<Value>,
    /// Span outputs (`outputs`).
    pub outputs: Option<Value>,
    /// Failure or interruption reason (`error`).
    pub error: Option<String>,
    /// Free-form span metadata, e.g. token usage (`metaData`).
    pub meta_data: BTreeMap<String, Value>,
    /// Component id for component spans (`componentId`).
    pub component_id: Option<String>,
    /// Component name for component spans (`componentName`).
    pub component_name: Option<String>,
    /// Component type for component spans (`componentType`).
    pub component_type: Option<String>,
    /// Owning workflow id (`workflowId`).
    pub workflow_id: Option<String>,
}

impl TraceSpan {
    /// Create a span in [`SpanStatus::Start`] with the current timestamp.
    #[must_use]
    pub fn new(
        trace_id: impl Into<String>,
        session_id: impl Into<String>,
        invoke_id: impl Into<String>,
        name: impl Into<String>,
        kind: TraceSpanKind,
    ) -> Self {
        Self {
            trace_id: trace_id.into(),
            session_id: session_id.into(),
            invoke_id: invoke_id.into(),
            name: name.into(),
            kind,
            status: SpanStatus::Start,
            start_time: Some(now_millis()),
            ..Self::default()
        }
    }

    /// Set the kind of work this span covers.
    #[must_use]
    pub fn with_invoke_type(mut self, invoke_type: InvokeType) -> Self {
        self.invoke_type = Some(invoke_type);
        self
    }

    /// Record the enclosing span's invoke id.
    #[must_use]
    pub fn with_parent(mut self, parent_invoke_id: impl Into<String>) -> Self {
        self.parent_invoke_id = Some(parent_invoke_id.into());
        self
    }

    /// Mark this span as a component span.
    #[must_use]
    pub fn with_component(
        mut self,
        component_id: impl Into<String>,
        component_name: impl Into<String>,
        component_type: impl Into<String>,
    ) -> Self {
        self.component_id = Some(component_id.into());
        self.component_name = Some(component_name.into());
        self.component_type = Some(component_type.into());
        self
    }

    /// Record the owning workflow id.
    #[must_use]
    pub fn with_workflow_id(mut self, workflow_id: impl Into<String>) -> Self {
        self.workflow_id = Some(workflow_id.into());
        self
    }

    /// The span duration in milliseconds.
    ///
    /// Uses `end_time - start_time` when both are set and the time so far when
    /// the span is still open. Returns `None` when the span was never started,
    /// and saturates at `0` instead of underflowing for an out-of-order clock.
    #[must_use]
    pub fn elapsed(&self) -> Option<u64> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            (Some(start), None) => Some(now_millis().saturating_sub(start)),
            (None, _) => None,
        }
    }
}

/// One trace frame: the stream envelope around a [`TraceSpan`].
///
/// `item_type` serializes as `"type"` and carries a [`TraceFrameType`] wire
/// value, matching Python's `TraceSchema`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TraceFrame {
    /// Stream frame type (`tracer_agent` or `tracer_workflow`), serialized as
    /// `"type"` for wire parity.
    #[serde(rename = "type")]
    pub item_type: String,
    /// The span carried by this frame.
    pub payload: TraceSpan,
}

impl TraceFrame {
    /// Wrap an agent span in a `tracer_agent` frame.
    #[must_use]
    pub fn agent(span: TraceSpan) -> Self {
        Self {
            item_type: TraceFrameType::Agent.as_str().to_owned(),
            payload: span,
        }
    }

    /// Wrap a workflow or component span in a `tracer_workflow` frame.
    #[must_use]
    pub fn workflow(span: TraceSpan) -> Self {
        Self {
            item_type: TraceFrameType::Workflow.as_str().to_owned(),
            payload: span,
        }
    }

    /// Project this frame onto a trace-mode [`StreamItem`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the span cannot be serialized to JSON.
    pub fn to_stream_item(&self) -> Result<StreamItem> {
        let payload = serde_json::to_value(&self.payload)
            .map_err(|error| Error::session(format!("serializing trace payload: {error}")))?;
        Ok(StreamItem::trace(self.item_type.clone(), payload))
    }
}

/// Consumer of emitted [`TraceFrame`]s.
///
/// Replaces Python's async agent/workflow handler families and its
/// process-global registry: a sink is injected per session, is synchronous, and
/// must never block or panic. Implementations are shared across tasks, hence
/// `Send + Sync`.
pub trait TraceSink: Send + Sync + std::fmt::Debug {
    /// Consume one frame.
    fn emit(&self, frame: TraceFrame);

    /// The position of the last frame emitted by this sink, when it tracks one.
    fn position(&self) -> Option<TracePosition> {
        None
    }

    /// Continue the trace from a previously checkpointed position.
    fn set_position(&self, position: TracePosition) {
        let _ = position;
    }
}

/// A sink that discards every frame.
///
/// This is the default sink of [`Tracer::new`]: tracing stays cheap until a
/// real sink (a memory recorder, a stream writer, `OTel`, …) is installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTraceSink;

impl NoopTraceSink {
    /// Create a discarding sink.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl TraceSink for NoopTraceSink {
    fn emit(&self, _frame: TraceFrame) {}
}

/// In-process recording sink for tests and inspection.
///
/// Holds every emitted frame in a lock-guarded vector; clones are not involved
/// because the sink is shared as `Arc<dyn TraceSink>`. Locks are never held
/// across an await point (there is none).
#[derive(Debug, Default)]
pub struct MemoryTraceSink {
    state: Mutex<MemoryState>,
}

/// The recording state behind a [`MemoryTraceSink`].
#[derive(Debug, Default)]
struct MemoryState {
    /// Frames recorded since the last [`MemoryTraceSink::take`].
    frames: Vec<TraceFrame>,
    /// Trace id observed on the first frame (or installed by `set_position`).
    trace_id: Option<String>,
    /// Invoke id of the most recently emitted frame.
    last_invoke_id: Option<String>,
    /// Frame count already accounted for by `set_position`.
    baseline: u64,
    /// Frames emitted since the baseline was captured.
    emitted: u64,
}

impl MemoryTraceSink {
    /// Create an empty recording sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a copy of the recorded frames, oldest first.
    #[must_use]
    pub fn frames(&self) -> Vec<TraceFrame> {
        self.state.lock().frames.clone()
    }

    /// Drain and return the recorded frames, oldest first.
    #[must_use]
    pub fn take(&self) -> Vec<TraceFrame> {
        std::mem::take(&mut self.state.lock().frames)
    }

    /// The number of currently recorded frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().frames.len()
    }

    /// Whether no frames are currently recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.lock().frames.is_empty()
    }

    /// Drop the recorded frames and the derived position.
    pub fn clear(&self) {
        *self.state.lock() = MemoryState::default();
    }
}

impl TraceSink for MemoryTraceSink {
    fn emit(&self, frame: TraceFrame) {
        let mut state = self.state.lock();
        if state.trace_id.is_none() {
            state.trace_id = Some(frame.payload.trace_id.clone());
        }
        state.last_invoke_id = Some(frame.payload.invoke_id.clone());
        state.emitted = state.emitted.saturating_add(1);
        state.frames.push(frame);
    }

    fn position(&self) -> Option<TracePosition> {
        let state = self.state.lock();
        let trace_id = state.trace_id.clone()?;
        Some(TracePosition {
            trace_id,
            last_invoke_id: state.last_invoke_id.clone(),
            frame_count: state.baseline.saturating_add(state.emitted),
        })
    }

    fn set_position(&self, position: TracePosition) {
        let mut state = self.state.lock();
        state.baseline = position.frame_count;
        state.emitted = 0;
        state.last_invoke_id = position.last_invoke_id;
        state.trace_id = Some(position.trace_id);
    }
}

/// Trace sink that publishes frames onto the session stream.
///
/// Ports Python's built-in `TraceSchema` handlers: each frame becomes a
/// trace-mode [`StreamItem`] written through
/// [`StreamWriterManager::write_now`]. Publishing failures are logged at debug
/// level and dropped: observability must never fail the traced run.
#[derive(Debug, Clone)]
pub struct StreamTraceSink {
    writer: Arc<StreamWriterManager>,
}

impl StreamTraceSink {
    /// Create a sink publishing through `writer`.
    #[must_use]
    pub fn new(writer: Arc<StreamWriterManager>) -> Self {
        Self { writer }
    }
}

impl TraceSink for StreamTraceSink {
    fn emit(&self, frame: TraceFrame) {
        let item_type = frame.item_type.clone();
        match frame.to_stream_item() {
            Ok(item) => {
                if let Err(error) = self.writer.write_now(item) {
                    tracing::debug!(
                        item_type = item_type.as_str(),
                        %error,
                        "dropping trace frame: stream write failed"
                    );
                }
            }
            Err(error) => {
                tracing::debug!(
                    item_type = item_type.as_str(),
                    %error,
                    "dropping trace frame: serialization failed"
                );
            }
        }
    }
}

/// Factory for trace spans, bound to one session and one trace.
///
/// Ports Python's `Tracer`: it mints invoke ids, attaches the session/trace
/// identifiers, and hands every frame to the installed [`TraceSink`]. Clones
/// share the same sink slot, so [`Tracer::set_sink`] affects every clone.
#[derive(Debug, Clone)]
pub struct Tracer {
    inner: Arc<TracerInner>,
}

/// Shared, cloneable tracer state.
#[derive(Debug)]
struct TracerInner {
    session_id: String,
    trace_id: String,
    sink: RwLock<Arc<dyn TraceSink>>,
}

impl Tracer {
    /// Create a tracer whose frames go to a [`NoopTraceSink`].
    ///
    /// An empty `trace_id` is replaced by a generated one, as in Python.
    #[must_use]
    pub fn new(session_id: impl Into<String>, trace_id: impl Into<String>) -> Self {
        Self::with_sink(session_id, trace_id, Arc::new(NoopTraceSink::new()))
    }

    /// Create a tracer emitting to `sink`.
    ///
    /// An empty `trace_id` is replaced by a generated one, as in Python.
    #[must_use]
    pub fn with_sink(
        session_id: impl Into<String>,
        trace_id: impl Into<String>,
        sink: Arc<dyn TraceSink>,
    ) -> Self {
        let trace_id = trace_id.into();
        let trace_id = if trace_id.is_empty() {
            uuid_hex()
        } else {
            trace_id
        };
        Self {
            inner: Arc::new(TracerInner {
                session_id: session_id.into(),
                trace_id,
                sink: RwLock::new(sink),
            }),
        }
    }

    /// The session id stamped on every span of this tracer.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// The trace id stamped on every span of this tracer.
    #[must_use]
    pub fn trace_id(&self) -> &str {
        &self.inner.trace_id
    }

    /// The currently installed sink.
    #[must_use]
    pub fn sink(&self) -> Arc<dyn TraceSink> {
        let sink = self.inner.sink.read();
        Arc::clone(&sink)
    }

    /// Replace the sink for spans started afterwards.
    ///
    /// Handles created earlier keep the sink they were started with, so a
    /// half-finished span is always reported to the sink that observed it.
    pub fn set_sink(&self, sink: Arc<dyn TraceSink>) {
        *self.inner.sink.write() = sink;
    }

    /// Start an agent span.
    ///
    /// The invoke id is generated; an empty `name` defaults to `agent`.
    #[must_use]
    pub fn start_span(&self, name: impl Into<String>, invoke_type: InvokeType) -> SpanHandle {
        self.start_span_of_kind(name, TraceSpanKind::Agent, invoke_type)
    }

    /// Start a workflow component span.
    ///
    /// The span kind is [`TraceSpanKind::Component`] and the component id is
    /// also recorded as the workflow id, matching the Python workflow handler's
    /// component frames. An empty `name` defaults to `component`.
    #[must_use]
    pub fn start_workflow_span(
        &self,
        component_id: impl Into<String>,
        name: impl Into<String>,
    ) -> SpanHandle {
        let component_id = component_id.into();
        let mut span = TraceSpan::new(
            self.trace_id(),
            self.session_id(),
            uuid_hex(),
            name,
            TraceSpanKind::Component,
        )
        .with_invoke_type(InvokeType::Workflow);
        if span.name.is_empty() {
            default_span_name(TraceSpanKind::Component).clone_into(&mut span.name);
        }
        span.workflow_id = Some(component_id.clone());
        span.component_id = Some(component_id);
        self.handle(span)
    }

    /// Start a child span under `parent`.
    ///
    /// The child shares the tracer's trace/session ids and kind, records
    /// `parentInvokeId`, uses the tracer's current sink, and is appended to the
    /// parent's `childInvokes`. An empty `name` defaults to the kind's name.
    #[must_use]
    pub fn start_child(
        &self,
        parent: &SpanHandle,
        name: impl Into<String>,
        invoke_type: InvokeType,
    ) -> SpanHandle {
        let mut span = TraceSpan::new(
            self.trace_id(),
            self.session_id(),
            uuid_hex(),
            name,
            parent.kind(),
        )
        .with_parent(parent.id())
        .with_invoke_type(invoke_type);
        if span.name.is_empty() {
            default_span_name(span.kind).clone_into(&mut span.name);
        }
        parent.add_child(&span.invoke_id);
        self.handle(span)
    }

    /// Build a span of `kind` and wrap it in a handle bound to the current sink.
    fn start_span_of_kind(
        &self,
        name: impl Into<String>,
        kind: TraceSpanKind,
        invoke_type: InvokeType,
    ) -> SpanHandle {
        let mut span = TraceSpan::new(self.trace_id(), self.session_id(), uuid_hex(), name, kind)
            .with_invoke_type(invoke_type);
        if span.name.is_empty() {
            default_span_name(kind).clone_into(&mut span.name);
        }
        self.handle(span)
    }

    /// Wrap `span` in a handle bound to the tracer's current sink.
    fn handle(&self, span: TraceSpan) -> SpanHandle {
        SpanHandle::new(self.sink(), span)
    }
}

/// A live span handle: mutate it, end it, and snapshot it.
///
/// Clones share the same span, which is what makes parent linking work: the
/// parent handle observed by [`Tracer::start_child`] is the same span the caller
/// finishes later. Ending a span (finish/fail/interrupted) emits exactly one
/// [`TraceFrame`]; repeated endings are ignored.
#[derive(Debug, Clone)]
pub struct SpanHandle {
    state: Arc<Mutex<SpanState>>,
}

/// The lock-guarded body of a [`SpanHandle`].
#[derive(Debug)]
struct SpanState {
    /// The sink captured when the span was started.
    sink: Arc<dyn TraceSink>,
    /// The mutable span data.
    span: TraceSpan,
}

impl SpanHandle {
    /// Wrap `span` for emission to `sink`.
    fn new(sink: Arc<dyn TraceSink>, span: TraceSpan) -> Self {
        Self {
            state: Arc::new(Mutex::new(SpanState { sink, span })),
        }
    }

    /// The span's invoke id.
    #[must_use]
    pub fn id(&self) -> String {
        self.state.lock().span.invoke_id.clone()
    }

    /// The trace id shared by the whole run.
    #[must_use]
    pub fn trace_id(&self) -> String {
        self.state.lock().span.trace_id.clone()
    }

    /// Record the span inputs.
    pub fn set_inputs(&self, inputs: Value) {
        self.state.lock().span.inputs = Some(inputs);
    }

    /// Record the span outputs.
    pub fn set_outputs(&self, outputs: Value) {
        self.state.lock().span.outputs = Some(outputs);
    }

    /// Record one metadata entry (e.g. token usage).
    pub fn set_metadata(&self, key: &str, value: Value) {
        self.state
            .lock()
            .span
            .meta_data
            .insert(key.to_owned(), value);
    }

    /// Overwrite the span status without ending the span.
    pub fn set_status(&self, status: SpanStatus) {
        self.state.lock().span.status = status;
    }

    /// Finish the span successfully and emit its frame.
    ///
    /// Records `outputs`, sets [`SpanStatus::Finish`], stamps the end time and
    /// elapsed time, and emits exactly one frame. A second call after the span
    /// reached a terminal status is a no-op and emits nothing.
    pub fn finish(&self, outputs: Value) {
        self.complete(SpanStatus::Finish, None, Some(outputs));
    }

    /// Fail the span and emit its frame.
    ///
    /// Records `error`, sets [`SpanStatus::Error`], stamps the times, and emits
    /// exactly one frame. Repeated terminal transitions emit nothing.
    pub fn fail(&self, error: impl Into<String>) {
        self.complete(SpanStatus::Error, Some(error.into()), None);
    }

    /// Mark the span interrupted and emit its frame.
    ///
    /// Records the reason in `error`, sets [`SpanStatus::Interrupted`], stamps
    /// the times, and emits exactly one frame. Repeated terminal transitions
    /// emit nothing.
    pub fn interrupted(&self, reason: impl Into<String>) {
        self.complete(SpanStatus::Interrupted, Some(reason.into()), None);
    }

    /// Take a copy of the current span data.
    #[must_use]
    pub fn snapshot(&self) -> TraceSpan {
        self.state.lock().span.clone()
    }

    /// The span kind, used by [`Tracer::start_child`].
    fn kind(&self) -> TraceSpanKind {
        self.state.lock().span.kind
    }

    /// Append `child_id` to `childInvokes` once.
    fn add_child(&self, child_id: &str) {
        let mut state = self.state.lock();
        if !state
            .span
            .child_invokes
            .iter()
            .any(|existing| existing == child_id)
        {
            state.span.child_invokes.push(child_id.to_owned());
        }
    }

    /// Apply a terminal transition and emit the resulting frame once.
    ///
    /// The lock is released before the sink runs, so a sink may safely read the
    /// span (through a snapshot) without deadlocking.
    fn complete(&self, status: SpanStatus, error: Option<String>, outputs: Option<Value>) {
        let (sink, frame) = {
            let mut state = self.state.lock();
            if state.span.status.is_terminal() {
                return;
            }
            state.span.status = status;
            if let Some(error) = error {
                state.span.error = Some(error);
            }
            if let Some(outputs) = outputs {
                state.span.outputs = Some(outputs);
            }
            stamp_end(&mut state.span);
            (Arc::clone(&state.sink), frame_for(&state.span))
        };
        sink.emit(frame);
    }
}

/// Convert a span into the frame type its kind maps to.
fn frame_for(span: &TraceSpan) -> TraceFrame {
    match span.kind {
        TraceSpanKind::Agent => TraceFrame::agent(span.clone()),
        // Component spans travel as workflow frames, like Python's workflow
        // handler emits component spans.
        TraceSpanKind::Workflow | TraceSpanKind::Component => TraceFrame::workflow(span.clone()),
    }
}

/// The default span name for `kind`, used when the caller passes an empty name.
const fn default_span_name(kind: TraceSpanKind) -> &'static str {
    match kind {
        TraceSpanKind::Agent => "agent",
        TraceSpanKind::Workflow => "workflow",
        TraceSpanKind::Component => "component",
    }
}

/// Stamp the end and elapsed times from the wall clock.
fn stamp_end(span: &mut TraceSpan) {
    let now = now_millis();
    span.end_time = Some(now);
    span.elapsed_time = span.start_time.map(|start| now.saturating_sub(start));
}

/// The current time as non-negative epoch milliseconds.
fn now_millis() -> u64 {
    u64::try_from(unix_millis()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{SpanStatus, TracePosition, TraceSpan, TraceSpanKind};

    #[test]
    fn elapsed_is_none_without_a_start_time() {
        let span = TraceSpan::default();
        assert_eq!(span.elapsed(), None);
    }

    #[test]
    fn elapsed_saturates_instead_of_underflowing() {
        let span = TraceSpan {
            start_time: Some(1_000),
            end_time: Some(400),
            ..TraceSpan::default()
        };
        assert_eq!(span.elapsed(), Some(0));
    }

    #[test]
    fn elapsed_uses_end_minus_start_when_both_are_known() {
        let span = TraceSpan {
            start_time: Some(1_000),
            end_time: Some(1_250),
            ..TraceSpan::default()
        };
        assert_eq!(span.elapsed(), Some(250));
    }

    #[test]
    fn span_status_knows_its_terminal_states() {
        assert!(!SpanStatus::Start.is_terminal());
        assert!(!SpanStatus::Running.is_terminal());
        assert!(SpanStatus::Finish.is_terminal());
        assert!(SpanStatus::Interrupted.is_terminal());
        assert!(SpanStatus::Error.is_terminal());
    }

    #[test]
    fn trace_position_keeps_the_snake_case_checkpoint_shape() {
        let position = TracePosition {
            trace_id: "trace-1".to_owned(),
            last_invoke_id: Some("invoke-1".to_owned()),
            frame_count: 3,
        };
        let value = serde_json::to_value(&position).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "trace_id": "trace-1",
                "last_invoke_id": "invoke-1",
                "frame_count": 3,
            })
        );
    }

    #[test]
    fn span_kind_defaults_to_agent() {
        assert_eq!(TraceSpanKind::default(), TraceSpanKind::Agent);
    }
}
