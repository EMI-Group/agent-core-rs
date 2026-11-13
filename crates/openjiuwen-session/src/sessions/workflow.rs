//! `WorkflowSession` and `NodeSession` — the workflow-side concrete sessions.
//!
//! Ports `openjiuwen/core/session/internal/workflow.py` (v0.1.20):
//! [`WorkflowSession`] owns the workflow state store and shares its config,
//! stream hub, interaction broker, tracer, and checkpointer with
//! [`NodeSession`] children, which are cheap component-scoped views over it.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use parking_lot::RwLock;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::uuid_hex;

use super::shared::{resolve_shared, SessionInner, SessionInnerOptions};
use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::constants::DEFAULT_WORKFLOW_ID;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::state::{StateStore, WorkflowState};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TraceSink, Tracer};

/// Process-local construction options for [`WorkflowSession`].
///
/// Holds trait objects, so deliberately not `serde`-serializable. Every field
/// is optional; the defaults mirror Python's `workflow.Session(session_id)`.
#[derive(Clone, Default)]
pub struct WorkflowSessionOptions {
    /// Session identifier; generated when absent and there is no parent.
    pub session_id: Option<String>,
    /// Workflow id used for the entity metadata only.
    ///
    /// Node executable ids follow Python's `create_executable_id`, not this id.
    pub workflow_id: Option<String>,
    /// Environment entries applied to a freshly created config.
    pub envs: BTreeMap<String, Value>,
    /// Parent session whose config, hub, checkpointer, and trace id are shared.
    pub parent: Option<Arc<dyn Session>>,
    /// Checkpointer backing [`Session::save_checkpoint`] and [`Session::resume`].
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Bounded stream capacity when this session creates its own hub.
    pub stream_capacity: Option<usize>,
    /// Trace id; generated when absent and no parent supplies one.
    pub trace_id: Option<String>,
    /// Opaque metadata copied into [`SessionMetadata::metadata`].
    pub metadata: BTreeMap<String, Value>,
}

/// A workflow (or graph) execution context.
///
/// Owns a [`WorkflowState`] store shared with its [`NodeSession`] children and
/// delegates everything else to the crate-private `SessionInner`.
#[derive(Clone)]
pub struct WorkflowSession {
    inner: Arc<SessionInner>,
    state: Arc<WorkflowState>,
    workflow_id: String,
}

impl std::fmt::Debug for WorkflowSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowSession")
            .field("session_id", &self.inner.id)
            .field("workflow_id", &self.workflow_id)
            .finish_non_exhaustive()
    }
}

impl WorkflowSession {
    /// Creates a session with `session_id` and the default workflow id.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_options(WorkflowSessionOptions {
            session_id: Some(session_id.into()),
            ..WorkflowSessionOptions::default()
        })
    }

    /// Creates a session from `options`.
    #[must_use]
    pub fn with_options(options: WorkflowSessionOptions) -> Self {
        let workflow_id = options
            .workflow_id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| DEFAULT_WORKFLOW_ID.to_string());
        let parent_id = options
            .parent
            .as_ref()
            .map(|parent| parent.id().to_string());
        let session_id = options
            .session_id
            .clone()
            .filter(|id| !id.is_empty())
            .or_else(|| parent_id.clone())
            .unwrap_or_else(uuid_hex);
        let shared = resolve_shared(
            options.parent.as_ref(),
            options.checkpointer,
            options.trace_id,
            options.stream_capacity,
            &options.envs,
        );
        // `with_global_lane(false)` keeps `get_state("")` returning the whole
        // global object; `WorkflowState::new()` is the workflow-only variant
        // that hides that lane, which the session contract does not allow.
        let state = Arc::new(WorkflowState::with_global_lane(false));
        let mut metadata = options.metadata;
        metadata
            .entry("workflow_id".to_string())
            .or_insert_with(|| Value::String(workflow_id.clone()));
        // The workflow identity a `NodeSession` (and a `SubWorkflowSession`)
        // reads back through `SessionMetadata`: a top-level workflow has no
        // enclosing workflow, so it is its own main workflow at depth 0.
        metadata
            .entry("main_workflow_id".to_string())
            .or_insert_with(|| Value::String(workflow_id.clone()));
        metadata
            .entry("workflow_nesting_depth".to_string())
            .or_insert_with(|| Value::from(0));
        let inner = SessionInner::new(SessionInnerOptions {
            id: session_id,
            parent_id,
            kind: SessionType::Workflow,
            entity_id: Some(workflow_id.clone()),
            entity_name: Some(workflow_id.clone()),
            trace_id: shared.trace_id,
            created_at: None,
            config: shared.config,
            state: Arc::clone(&state) as Arc<dyn StateStore>,
            writers: Some(shared.writers),
            stream_capacity: options.stream_capacity,
            sink: None,
            checkpointer: shared.checkpointer,
            parent: options.parent,
            metadata,
        });
        Self {
            inner,
            state,
            workflow_id,
        }
    }

    /// Returns the workflow id used for entity metadata and node ids.
    #[must_use]
    pub fn workflow_id(&self) -> &str {
        &self.workflow_id
    }

    /// Returns the workflow state store shared with child sessions.
    #[must_use]
    pub fn state(&self) -> Arc<WorkflowState> {
        Arc::clone(&self.state)
    }

    /// Returns every committed lane of the workflow state store.
    #[must_use]
    pub fn dump_state(&self) -> Value {
        self.state.dump()
    }

    /// Writes `inputs` into the IO lane and the global lane.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] when `inputs` is
    /// not a JSON object or a key is not a valid nested path.
    pub fn commit_user_inputs(&self, inputs: Value) -> Result<()> {
        self.state.commit_user_inputs(inputs)
    }

    /// Creates a component-scoped [`NodeSession`] child of this session.
    #[must_use]
    pub fn create_node_session(self: &Arc<Self>, component_id: impl Into<String>) -> NodeSession {
        NodeSession::new(Arc::clone(self) as Arc<dyn Session>, component_id)
    }
}

#[async_trait]
impl Session for WorkflowSession {
    fn id(&self) -> &str {
        &self.inner.id
    }

    fn parent_id(&self) -> Option<&str> {
        self.inner.parent_id.as_deref()
    }

    fn session_type(&self) -> SessionType {
        SessionType::Workflow
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        Ok(self.inner.get_state(key))
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        self.inner.update_state(patch)
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        Ok(self.inner.get_component_state(component_id))
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        self.inner.update_component_state(component_id, patch)
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.inner.write_stream(item).await
    }

    async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.inner.request_interaction(request).await
    }

    fn trace(&self) -> &dyn TraceSink {
        self.inner.trace_sink.as_ref()
    }

    fn install_trace_sink(&self, sink: Arc<dyn TraceSink>) -> Result<()> {
        self.inner.install_trace_sink(sink);
        Ok(())
    }

    fn metadata(&self) -> SessionMetadata {
        self.inner.metadata()
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.inner.checkpointer.clone()
    }

    fn tracer(&self) -> Option<&Tracer> {
        Some(&self.inner.tracer)
    }

    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        Some(self.inner.config_handle())
    }

    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        Some(Arc::clone(&self.inner.writers))
    }

    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.inner.subscribe()
    }

    async fn resolve_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.inner.resolve_interaction(response)
    }

    async fn end_stream(&self) -> Result<()> {
        self.inner.end_stream();
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.inner.close();
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        Ok(self.inner.snapshot())
    }

    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        self.inner.restore(snapshot)
    }
}

/// A component-scoped child view of a parent session.
///
/// The executable id follows Python's `create_executable_id`: it is the bare
/// `component_id` when the parent is not itself a node (a workflow or agent
/// session), and `parent executable id + "." + component_id` when nesting node
/// sessions (`a`, `a.b`, `a.b.c`). Component state read/written through this
/// view nests under that id, matching Python's `NodeSession`. Streaming,
/// tracing, interactions, and checkpoints are delegated, and
/// [`Session::end_stream`]/[`Session::close`] are no-ops so a node never closes
/// the shared hub.
#[derive(Clone)]
pub struct NodeSession {
    parent: Arc<dyn Session>,
    component_id: String,
    executable_id: String,
    node_type: Option<String>,
}

impl std::fmt::Debug for NodeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSession")
            .field("executable_id", &self.executable_id)
            .field("component_id", &self.component_id)
            .finish_non_exhaustive()
    }
}

impl NodeSession {
    /// Creates a node view for `component_id` over `parent`.
    #[must_use]
    pub fn new(parent: Arc<dyn Session>, component_id: impl Into<String>) -> Self {
        Self::build(parent, component_id.into(), None)
    }

    /// Creates a node view carrying an explicit node type.
    #[must_use]
    pub fn new_with_type(
        parent: Arc<dyn Session>,
        component_id: impl Into<String>,
        node_type: impl Into<String>,
    ) -> Self {
        Self::build(parent, component_id.into(), Some(node_type.into()))
    }

    /// Returns the executable id (`node` for a workflow's node, `a.b` deeper).
    #[must_use]
    pub fn executable_id(&self) -> &str {
        &self.executable_id
    }

    /// Returns the component id this view was created with.
    #[must_use]
    pub fn component_id(&self) -> &str {
        &self.component_id
    }

    /// Returns the node type this view was created with, when one was supplied.
    #[must_use]
    pub fn node_type(&self) -> Option<&str> {
        self.node_type.as_deref()
    }

    /// Returns the workflow id of the workflow this node belongs to.
    ///
    /// Read from the parent session's `workflow_id` metadata (a workflow
    /// session stamps it at construction, and a node copies its parent's
    /// metadata), defaulting to [`DEFAULT_WORKFLOW_ID`] when absent or empty.
    #[must_use]
    pub fn workflow_id(&self) -> String {
        self.parent_workflow_key("workflow_id")
            .unwrap_or_else(|| DEFAULT_WORKFLOW_ID.to_string())
    }

    /// Returns the id of the outermost workflow in this nesting chain.
    ///
    /// Read from the parent session's `main_workflow_id` metadata, falling back
    /// to [`NodeSession::workflow_id`] for a parent that does not record one.
    #[must_use]
    pub fn main_workflow_id(&self) -> String {
        self.parent_workflow_key("main_workflow_id")
            .unwrap_or_else(|| self.workflow_id())
    }

    /// Returns how many sub-workflows enclose this node.
    ///
    /// Read from the parent session's `workflow_nesting_depth` metadata; a
    /// top-level workflow's node is at depth 0, and
    /// [`crate::sessions::SubWorkflowSession`] adds one level per nesting.
    #[must_use]
    pub fn workflow_nesting_depth(&self) -> u32 {
        self.parent
            .metadata()
            .metadata
            .get("workflow_nesting_depth")
            .and_then(Value::as_u64)
            .and_then(|depth| u32::try_from(depth).ok())
            .unwrap_or(0)
    }

    /// Returns the parent session.
    #[must_use]
    pub fn parent(&self) -> &Arc<dyn Session> {
        &self.parent
    }

    /// Creates a nested node view whose executable id extends this one.
    #[must_use]
    pub fn create_node_session(self: &Arc<Self>, component_id: impl Into<String>) -> NodeSession {
        NodeSession::new(Arc::clone(self) as Arc<dyn Session>, component_id)
    }

    /// Builds the view, deriving the executable id from the parent.
    ///
    /// Ports Python's `create_executable_id`: only a node parent contributes a
    /// prefix, so the first node level is the bare component id.
    fn build(parent: Arc<dyn Session>, component_id: String, node_type: Option<String>) -> Self {
        let base = if parent.session_type() == SessionType::Node {
            parent.metadata().entity_id.unwrap_or_default()
        } else {
            String::new()
        };
        let executable_id = if base.is_empty() {
            component_id.clone()
        } else {
            format!("{base}.{component_id}")
        };
        Self {
            parent,
            component_id,
            executable_id,
            node_type,
        }
    }

    /// Reads a non-empty string workflow key from the parent's metadata.
    fn parent_workflow_key(&self, key: &str) -> Option<String> {
        self.parent
            .metadata()
            .metadata
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    }

    /// Joins the executable id with `component_id` (or returns it when empty).
    fn scoped_component_id(&self, component_id: &str) -> String {
        if component_id.is_empty() {
            self.executable_id.clone()
        } else {
            format!("{}.{}", self.executable_id, component_id)
        }
    }
}

#[async_trait]
impl Session for NodeSession {
    fn id(&self) -> &str {
        self.parent.id()
    }

    fn parent_id(&self) -> Option<&str> {
        Some(self.parent.id())
    }

    fn session_type(&self) -> SessionType {
        SessionType::Node
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        self.parent.get_state(key).await
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        self.parent.update_state(patch).await
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        self.parent
            .get_component_state(&self.scoped_component_id(component_id))
            .await
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        self.parent
            .update_component_state(&self.scoped_component_id(component_id), patch)
            .await
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.parent.write_stream(item).await
    }

    async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        let mut request = request;
        if request.component_id.is_none() {
            request.component_id = Some(self.executable_id.clone());
        }
        self.parent.request_interaction(request).await
    }

    fn trace(&self) -> &dyn TraceSink {
        self.parent.trace()
    }

    fn install_trace_sink(&self, sink: Arc<dyn TraceSink>) -> Result<()> {
        self.parent.install_trace_sink(sink)
    }

    fn metadata(&self) -> SessionMetadata {
        let mut metadata = self.parent.metadata();
        let parent_id = self.parent.id().to_string();
        metadata.session_id.clone_from(&parent_id);
        metadata.parent_session_id = Some(parent_id);
        metadata.kind = SessionType::Node;
        metadata.entity_id = Some(self.executable_id.clone());
        metadata.entity_name = Some(self.component_id.clone());
        metadata.metadata.insert(
            "executable_id".to_string(),
            Value::String(self.executable_id.clone()),
        );
        metadata.metadata.insert(
            "component_id".to_string(),
            Value::String(self.component_id.clone()),
        );
        if let Some(node_type) = &self.node_type {
            metadata
                .metadata
                .insert("node_type".to_string(), Value::String(node_type.clone()));
        }
        metadata
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.parent.checkpointer()
    }

    fn tracer(&self) -> Option<&Tracer> {
        self.parent.tracer()
    }

    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        self.parent.config()
    }

    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        self.parent.writers()
    }

    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.parent.subscribe()
    }

    async fn resolve_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.parent.resolve_interaction(response).await
    }

    async fn end_stream(&self) -> Result<()> {
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        self.parent.snapshot().await
    }

    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        self.parent.restore(snapshot).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use futures::StreamExt;
    use serde_json::json;

    use super::{NodeSession, WorkflowSession};
    use crate::interaction::{
        InteractionKind, InteractionOutput, InteractionRequest, InteractionResponse,
    };
    use crate::session::{Session, SessionType};
    use crate::stream::StreamEvent;

    #[tokio::test]
    async fn state_updates_are_immediately_visible() {
        let session = WorkflowSession::new("s1");
        assert_eq!(session.get_state("").await.unwrap(), Some(json!({})));
        session.update_state(json!({ "step": 1 })).await.unwrap();
        assert_eq!(session.get_state("step").await.unwrap(), Some(json!(1)));
        session
            .update_state(json!({ "step": { "nested": true } }))
            .await
            .unwrap();
        assert_eq!(
            session.get_state("step.nested").await.unwrap(),
            Some(json!(true))
        );
    }

    #[tokio::test]
    async fn node_component_state_nests_under_executable_id() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let node = workflow.create_node_session("a");
        node.update_component_state("b", json!({ "value": 1 }))
            .await
            .unwrap();

        assert_eq!(node.executable_id(), "a");
        assert_eq!(
            node.get_component_state("b").await.unwrap(),
            Some(json!({ "value": 1 }))
        );
        assert_eq!(
            workflow.get_component_state("a.b").await.unwrap(),
            Some(json!({ "value": 1 }))
        );
    }

    #[tokio::test]
    async fn node_executable_ids_chain() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let first = Arc::new(workflow.create_node_session("a"));
        let second = Arc::new(first.create_node_session("b"));
        let third = second.create_node_session("c");

        assert_eq!(
            first.executable_id(),
            "a",
            "a workflow's node has no prefix"
        );
        assert_eq!(
            second.executable_id(),
            "a.b",
            "a nested node extends its node parent's executable id"
        );
        assert_eq!(third.executable_id(), "a.b.c");
        assert_eq!(third.id(), "s1");
        assert_eq!(third.parent_id(), Some("s1"));
        assert_eq!(third.session_type(), SessionType::Node);
        let metadata = third.metadata();
        assert_eq!(metadata.entity_id.as_deref(), Some("a.b.c"));
        assert_eq!(metadata.entity_name.as_deref(), Some("c"));
        assert_eq!(metadata.parent_session_id.as_deref(), Some("s1"));
        assert_eq!(metadata.metadata["executable_id"], json!("a.b.c"));
    }

    #[tokio::test]
    async fn node_end_stream_leaves_parent_open() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let node = workflow.create_node_session("a");
        node.end_stream().await.unwrap();
        node.close().await.unwrap();

        let writers = workflow.writers().unwrap();
        assert!(!writers.is_closed());
        assert!(workflow.parent_id().is_none());
    }

    #[tokio::test]
    async fn node_interaction_frame_carries_executable_id() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let node = workflow.create_node_session("n1");
        let mut events = node.subscribe();

        let request = InteractionRequest::new(InteractionKind::Confirm, json!({ "q": "ok?" }));
        let request_id = request.id.clone();
        let waiter_node: NodeSession = node.clone();
        let waiter = tokio::spawn(async move { waiter_node.request_interaction(request).await });

        let event = events.next().await.unwrap().unwrap();
        let StreamEvent::Item(item) = event else {
            panic!("expected an interaction frame");
        };
        assert_eq!(item.item_type, "__interaction__");
        let output: InteractionOutput = serde_json::from_value(item.payload).unwrap();
        assert_eq!(output.id, "n1");

        node.resolve_interaction(InteractionResponse::approve(request_id))
            .await
            .unwrap();
        let response = waiter.await.unwrap().unwrap();
        assert!(response.is_approved());
    }
}
