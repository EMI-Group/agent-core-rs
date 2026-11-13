//! `SubWorkflowSession` — the session a nested workflow runs with.
//!
//! Ports `SubWorkflowSession` from
//! `openjiuwen/core/session/internal/workflow.py` (v0.1.20). Python builds it
//! from the host component's [`NodeSession`] and re-parents it:
//!
//! ```text
//! SubWorkflowSession(session=host, workflow_id)
//!     = NodeSession(session=host.parent(), node_id=host.node_id(),
//!                   node_type=host.node_type())
//!     + workflow_id = <inner workflow id>
//!     + workflow_nesting_depth = host.workflow_nesting_depth() + 1
//!     + main_workflow_id = host.main_workflow_id()
//! ```
//!
//! Two consequences carry over, and both are pinned by tests:
//!
//! - The executable id is **identical** to the host node's. Re-parenting to
//!   `host.parent()` and reusing `host.node_id()` reproduces Python's
//!   `create_executable_id` result exactly (the bare component id below a
//!   workflow, `a.b` below a node), so the sub-workflow's internal components
//!   keep landing under the host component's lane and a checkpoint written by
//!   the host stays readable.
//! - The state store is the parent workflow's, not the host node's, so nothing
//!   is duplicated: reads and writes through the sub-workflow go through the
//!   same lane the parent workflow exposes as `<host executable id>`.
//!
//! `SubWorkflowSession` only overrides the workflow placement
//! (`workflow_id`, `main_workflow_id`, `workflow_nesting_depth`) and adds the
//! end-output slot. Everything else — state, streams, HITL, tracing,
//! snapshots — delegates to the inner node view, exactly like [`NodeSession`],
//! and `end_stream`/`close` stay no-ops so a nested workflow never closes the
//! shared hub.
//!
//! # End output
//!
//! Python's nested-workflow component copies the child result into the host
//! node's state so the parent's `${node.output}` reference resolves.
//! [`SubWorkflowSession::set_end_output`] records the value (the slot is shared
//! across clones) and [`SubWorkflowSession::propagate_end_output`] merges
//! `{"output": <value>}` into the host node's component lane.
//!
//! # Not ported
//!
//! Python's `RouterSession` sibling lives in [`crate::sessions::router`];
//! Python's per-sub-workflow `ActorManager` (streaming actors) is not part of
//! this crate, so it has no counterpart here.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;

use super::workflow::NodeSession;
use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TraceSink, Tracer};

/// A nested workflow's session: a host node re-parented to the host's parent.
///
/// See the module docs for the Python mapping. Clones share the end-output
/// slot, so a child task may record the value while the spawning task
/// propagates it.
#[derive(Clone)]
pub struct SubWorkflowSession {
    /// The re-parented node view; its executable id equals the host node's.
    node: NodeSession,
    /// The inner workflow's id (overrides the enclosing workflow's id).
    workflow_id: String,
    /// The outermost workflow's id, inherited from the host node.
    main_workflow_id: String,
    /// Enclosing sub-workflow count: host depth + 1.
    workflow_nesting_depth: u32,
    /// The value the nested workflow produced, shared across clones.
    end_output: Arc<Mutex<Option<Value>>>,
}

impl std::fmt::Debug for SubWorkflowSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubWorkflowSession")
            .field("executable_id", &self.node.executable_id())
            .field("component_id", &self.node.component_id())
            .field("workflow_id", &self.workflow_id)
            .field("workflow_nesting_depth", &self.workflow_nesting_depth)
            .finish_non_exhaustive()
    }
}

impl SubWorkflowSession {
    /// Builds the sub-workflow session for `host`, running `workflow_id`.
    ///
    /// An empty `workflow_id` keeps the host node's workflow id, so
    /// [`SubWorkflowSession::workflow_id`] is never empty.
    #[must_use]
    pub fn new(host: &NodeSession, workflow_id: impl Into<String>) -> Self {
        let node_type = host.node_type().map(str::to_string);
        let node = match node_type {
            Some(node_type) => NodeSession::new_with_type(
                Arc::clone(host.parent()),
                host.component_id(),
                node_type,
            ),
            None => NodeSession::new(Arc::clone(host.parent()), host.component_id()),
        };
        let workflow_id = {
            let workflow_id = workflow_id.into();
            if workflow_id.is_empty() {
                host.workflow_id()
            } else {
                workflow_id
            }
        };
        let main_workflow_id = host.main_workflow_id();
        let workflow_nesting_depth = host.workflow_nesting_depth().saturating_add(1);
        Self {
            node,
            workflow_id,
            main_workflow_id,
            workflow_nesting_depth,
            end_output: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns the host component's id.
    #[must_use]
    pub fn component_id(&self) -> &str {
        self.node.component_id()
    }

    /// Returns the executable id, identical to the host node's.
    #[must_use]
    pub fn executable_id(&self) -> &str {
        self.node.executable_id()
    }

    /// Returns the host node's node type, when it had one.
    #[must_use]
    pub fn node_type(&self) -> Option<&str> {
        self.node.node_type()
    }

    /// Returns the inner workflow's id.
    #[must_use]
    pub fn workflow_id(&self) -> &str {
        &self.workflow_id
    }

    /// Returns the outermost workflow's id, inherited from the host node.
    #[must_use]
    pub fn main_workflow_id(&self) -> &str {
        &self.main_workflow_id
    }

    /// Returns the nesting depth: the host node's depth plus one.
    #[must_use]
    pub fn workflow_nesting_depth(&self) -> u32 {
        self.workflow_nesting_depth
    }

    /// Returns the enclosing session (the host node's parent).
    #[must_use]
    pub fn parent(&self) -> &Arc<dyn Session> {
        self.node.parent()
    }

    /// Records the value the nested workflow produced.
    pub fn set_end_output(&self, value: Value) {
        *self.end_output.lock() = Some(value);
    }

    /// Returns a copy of the recorded end output, if any.
    #[must_use]
    pub fn end_output(&self) -> Option<Value> {
        self.end_output.lock().clone()
    }

    /// Takes the recorded end output, leaving the slot empty.
    #[must_use]
    pub fn take_end_output(&self) -> Option<Value> {
        self.end_output.lock().take()
    }

    /// Merges `{"output": <end_output>}` into the host node's component lane so
    /// the parent's `${node.output}` reference resolves.
    ///
    /// This is the Rust realization of Python's child-workflow
    /// end-output propagation: the host component's lane (the sub-workflow's
    /// own executable id) receives an `output` key, and the slot is left intact
    /// so the caller can still inspect it — use
    /// [`SubWorkflowSession::take_end_output`] to consume it. A session with no
    /// recorded end output is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates [`Session::update_component_state`] failures (for example a
    /// state store that rejects the patch).
    pub async fn propagate_end_output(&self) -> Result<()> {
        let Some(value) = self.end_output() else {
            return Ok(());
        };
        // An empty component id addresses the host node's own lane, which is
        // exactly where the parent's `${<host>.output}` reads from.
        self.node
            .update_component_state("", json!({ "output": value }))
            .await
    }
}

#[async_trait]
impl Session for SubWorkflowSession {
    fn id(&self) -> &str {
        self.node.id()
    }

    fn parent_id(&self) -> Option<&str> {
        self.node.parent_id()
    }

    fn session_type(&self) -> SessionType {
        SessionType::Node
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        self.node.get_state(key).await
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        self.node.update_state(patch).await
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        self.node.get_component_state(component_id).await
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        self.node.update_component_state(component_id, patch).await
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        self.node.write_stream(item).await
    }

    async fn request_interaction(
        &self,
        request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.node.request_interaction(request).await
    }

    fn trace(&self) -> &dyn TraceSink {
        self.node.trace()
    }

    fn install_trace_sink(&self, sink: Arc<dyn TraceSink>) -> Result<()> {
        self.node.install_trace_sink(sink)
    }

    fn metadata(&self) -> SessionMetadata {
        let mut metadata = self.node.metadata();
        metadata.kind = SessionType::Node;
        metadata.metadata.insert(
            "workflow_id".to_string(),
            Value::String(self.workflow_id.clone()),
        );
        metadata.metadata.insert(
            "main_workflow_id".to_string(),
            Value::String(self.main_workflow_id.clone()),
        );
        metadata.metadata.insert(
            "workflow_nesting_depth".to_string(),
            Value::from(self.workflow_nesting_depth),
        );
        metadata.metadata.insert(
            "executable_id".to_string(),
            Value::String(self.executable_id().to_string()),
        );
        metadata.metadata.insert(
            "component_id".to_string(),
            Value::String(self.component_id().to_string()),
        );
        metadata
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.node.checkpointer()
    }

    fn tracer(&self) -> Option<&Tracer> {
        self.node.tracer()
    }

    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        self.node.config()
    }

    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        self.node.writers()
    }

    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.node.subscribe()
    }

    async fn resolve_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.node.resolve_interaction(response).await
    }

    async fn end_stream(&self) -> Result<()> {
        // A nested workflow shares the parent hub: never close it here.
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        let mut snapshot = self.node.snapshot().await?;
        snapshot.kind = SessionType::Node;
        Ok(snapshot)
    }

    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()> {
        self.node.restore(snapshot).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use serde_json::json;

    use super::SubWorkflowSession;
    use crate::constants::DEFAULT_WORKFLOW_ID;
    use crate::session::{Session, SessionType};
    use crate::sessions::workflow::WorkflowSession;

    #[tokio::test]
    async fn reuses_the_host_executable_id_and_bumps_the_depth() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let host = workflow.create_node_session("sub");
        let sub = SubWorkflowSession::new(&host, "inner");

        assert_eq!(sub.executable_id(), host.executable_id());
        assert_eq!(sub.component_id(), "sub");
        assert_eq!(sub.workflow_id(), "inner");
        assert_eq!(sub.main_workflow_id(), DEFAULT_WORKFLOW_ID);
        assert_eq!(sub.workflow_nesting_depth(), 1);
        assert_eq!(sub.id(), "s1");
        assert_eq!(sub.parent().id(), "s1");
        assert_eq!(sub.session_type(), SessionType::Node);
        assert_eq!(sub.metadata().kind, SessionType::Node);
    }

    #[tokio::test]
    async fn nested_hosts_keep_their_chained_executable_id() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let outer = Arc::new(workflow.create_node_session("a"));
        let host = outer.create_node_session("b");
        let sub = SubWorkflowSession::new(&host, "inner");

        assert_eq!(host.executable_id(), "a.b");
        assert_eq!(sub.executable_id(), "a.b");
        assert_eq!(sub.parent().id(), "s1");
    }

    #[tokio::test]
    async fn component_state_lands_in_the_parent_lane() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let host = workflow.create_node_session("sub");
        let sub = SubWorkflowSession::new(&host, "inner");

        sub.update_component_state("x", json!({ "value": 1 }))
            .await
            .unwrap();
        assert_eq!(
            workflow.get_component_state("sub.x").await.unwrap(),
            Some(json!({ "value": 1 }))
        );
        assert_eq!(
            sub.get_component_state("x").await.unwrap(),
            Some(json!({ "value": 1 }))
        );
    }

    #[tokio::test]
    async fn end_output_propagates_and_can_be_taken() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let host = workflow.create_node_session("sub");
        let sub = SubWorkflowSession::new(&host, "inner");

        sub.propagate_end_output().await.unwrap();
        assert_eq!(workflow.get_component_state("sub").await.unwrap(), None);

        sub.update_component_state("", json!({ "kept": true }))
            .await
            .unwrap();
        sub.set_end_output(json!({ "answer": 42 }));
        assert_eq!(sub.end_output(), Some(json!({ "answer": 42 })));
        sub.propagate_end_output().await.unwrap();
        assert_eq!(
            workflow.get_component_state("sub").await.unwrap(),
            Some(json!({ "kept": true, "output": { "answer": 42 } }))
        );
        assert_eq!(sub.take_end_output(), Some(json!({ "answer": 42 })));
        assert_eq!(sub.end_output(), None);
    }

    #[tokio::test]
    async fn clone_shares_the_end_output_slot() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let host = workflow.create_node_session("sub");
        let sub = SubWorkflowSession::new(&host, "inner");
        let clone = sub.clone();

        clone.set_end_output(json!("done"));
        assert_eq!(sub.end_output(), Some(json!("done")));
    }

    #[tokio::test]
    async fn end_stream_does_not_close_the_parent_hub() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let host = workflow.create_node_session("sub");
        let sub = SubWorkflowSession::new(&host, "inner");

        sub.end_stream().await.unwrap();
        sub.close().await.unwrap();
        assert!(!workflow.writers().unwrap().is_closed());
    }
}
