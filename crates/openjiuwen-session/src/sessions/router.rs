//! `RouterSession` — the read-only session handed to conditional-edge routers.
//!
//! Ports `RouterSession` from
//! `openjiuwen/core/session/internal/wrapper.py` (v0.1.20). Python builds it as
//! a `StateSession` subclass that reads through to the wrapped session and
//! selectively no-ops everything a router must not do: state writes, stream
//! writes, interaction, configuration lookups (`get_workflow_config`,
//! `get_agent_config`, `get_env`), and `base()`.
//!
//! The Rust port is a thin `Arc<dyn Session>` wrapper with the same shape:
//!
//! | Operation | Behavior |
//! | --- | --- |
//! | `id`, `parent_id`, `get_state`, `get_component_state`, `trace`, `snapshot` | delegate to the inner session |
//! | `update_state`, `update_component_state`, `write_stream`, `restore` | deliberate no-ops returning `Ok(())` (logged at debug) |
//! | `subscribe` | an immediately empty stream (a router never consumes the run's output) |
//! | `request_interaction` | `ErrorKind::Unsupported` (`"router sessions cannot interact"`) |
//! | `install_trace_sink` | `ErrorKind::Unsupported` (`"router sessions are read-only"`) |
//! | `config`, `writers`, `checkpointer`, `tracer` | `None` |
//! | `metadata` | forwarded, with `kind` forced to [`SessionType::Node`] |
//!
//! A routing callback can therefore inspect the state a branch decision needs
//! without being able to mutate it, close the run's stream, or swap the trace
//! sink out from under the workflow.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use parking_lot::RwLock;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::checkpoint::Checkpointer;
use crate::config::SessionConfig;
use crate::interaction::{InteractionRequest, InteractionResponse};
use crate::session::{Session, SessionMetadata, SessionSnapshot, SessionType};
use crate::stream::{StreamEvent, StreamItem, StreamWriterManager};
use crate::tracer::{TraceSink, Tracer};

/// A read-only view over another session, for router callbacks.
#[derive(Clone)]
pub struct RouterSession {
    /// The wrapped session.
    inner: Arc<dyn Session>,
    /// The inner session's entity id, captured at construction.
    executable_id: Option<String>,
}

impl std::fmt::Debug for RouterSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterSession")
            .field("session_id", &self.inner.id())
            .field("executable_id", &self.executable_id)
            .finish_non_exhaustive()
    }
}

impl RouterSession {
    /// Wraps `inner` for a routing callback.
    #[must_use]
    pub fn new(inner: Arc<dyn Session>) -> Self {
        Self {
            executable_id: inner.metadata().entity_id,
            inner,
        }
    }

    /// Returns the wrapped session.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn Session> {
        &self.inner
    }

    /// Returns the wrapped session's entity id (`None` when it has none).
    #[must_use]
    pub fn executable_id(&self) -> Option<&str> {
        self.executable_id.as_deref()
    }

    /// Returns the wrapped session's id.
    #[must_use]
    pub fn session_id(&self) -> &str {
        self.inner.id()
    }
}

#[async_trait]
impl Session for RouterSession {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn parent_id(&self) -> Option<&str> {
        self.inner.parent_id()
    }

    fn session_type(&self) -> SessionType {
        SessionType::Node
    }

    async fn get_state(&self, key: &str) -> Result<Option<Value>> {
        self.inner.get_state(key).await
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        tracing::debug!(
            target: "openjiuwen_session::sessions::router",
            session_id = self.inner.id(),
            patch = %patch,
            "router session ignores a global state update"
        );
        Ok(())
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        self.inner.get_component_state(component_id).await
    }

    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
        tracing::debug!(
            target: "openjiuwen_session::sessions::router",
            session_id = self.inner.id(),
            component_id,
            patch = %patch,
            "router session ignores a component state update"
        );
        Ok(())
    }

    async fn write_stream(&self, item: StreamItem) -> Result<()> {
        tracing::debug!(
            target: "openjiuwen_session::sessions::router",
            session_id = self.inner.id(),
            item_type = item.item_type,
            "router session ignores a stream write"
        );
        Ok(())
    }

    async fn request_interaction(
        &self,
        _request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        Err(Error::unsupported("router sessions cannot interact"))
    }

    fn trace(&self) -> &dyn TraceSink {
        self.inner.trace()
    }

    fn install_trace_sink(&self, _sink: Arc<dyn TraceSink>) -> Result<()> {
        Err(Error::unsupported("router sessions are read-only"))
    }

    fn metadata(&self) -> SessionMetadata {
        let mut metadata = self.inner.metadata();
        metadata.kind = SessionType::Node;
        metadata
    }

    fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        None
    }

    fn tracer(&self) -> Option<&Tracer> {
        None
    }

    fn config(&self) -> Option<Arc<RwLock<SessionConfig>>> {
        None
    }

    fn writers(&self) -> Option<Arc<StreamWriterManager>> {
        None
    }

    fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        Box::pin(futures::stream::empty())
    }

    async fn resolve_interaction(&self, _response: InteractionResponse) -> Result<()> {
        Ok(())
    }

    async fn end_stream(&self) -> Result<()> {
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        let mut snapshot = self.inner.snapshot().await?;
        snapshot.kind = SessionType::Node;
        Ok(snapshot)
    }

    async fn restore(&self, _snapshot: SessionSnapshot) -> Result<()> {
        tracing::debug!(
            target: "openjiuwen_session::sessions::router",
            session_id = self.inner.id(),
            "router session ignores a state restore"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use futures::StreamExt;
    use serde_json::json;

    use super::RouterSession;
    use crate::interaction::{InteractionKind, InteractionRequest};
    use crate::session::{Session, SessionType};
    use crate::sessions::workflow::WorkflowSession;
    use crate::stream::StreamItem;

    fn router_over(workflow: &Arc<WorkflowSession>) -> RouterSession {
        RouterSession::new(Arc::clone(workflow) as Arc<dyn Session>)
    }

    #[tokio::test]
    async fn reads_pass_through_and_writes_are_ignored() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        workflow.update_state(json!({ "step": 1 })).await.unwrap();
        let node = workflow.create_node_session("r");
        node.update_component_state("", json!({ "branch": "left" }))
            .await
            .unwrap();
        let router = RouterSession::new(Arc::new(node) as Arc<dyn Session>);

        assert_eq!(router.session_id(), "s1");
        assert_eq!(router.id(), "s1");
        assert_eq!(router.executable_id(), Some("r"));
        assert_eq!(router.get_state("step").await.unwrap(), Some(json!(1)));
        assert_eq!(
            router.get_component_state("branch").await.unwrap(),
            Some(json!("left"))
        );

        router.update_state(json!({ "step": 99 })).await.unwrap();
        router
            .update_component_state("branch", json!("right"))
            .await
            .unwrap();
        assert_eq!(workflow.get_state("step").await.unwrap(), Some(json!(1)));
        assert_eq!(
            workflow.get_component_state("r.branch").await.unwrap(),
            Some(json!("left"))
        );
    }

    #[tokio::test]
    async fn streaming_and_interaction_are_disabled() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let router = router_over(&workflow);
        let mut events = workflow.subscribe();

        router
            .write_stream(StreamItem::output_at("chunk", 0, json!({ "x": 1 })))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), events.next())
                .await
                .is_err(),
            "a router write must not reach the session stream"
        );
        assert!(router.subscribe().next().await.is_none());

        let error = router
            .request_interaction(InteractionRequest::new(InteractionKind::Confirm, json!({})))
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(router.end_stream().await.is_ok());
        assert!(!workflow.writers().unwrap().is_closed());
        assert_eq!(router.parent_id(), None);
        assert_eq!(router.session_type(), SessionType::Node);
        assert_eq!(router.metadata().kind, SessionType::Node);
    }

    #[tokio::test]
    async fn shared_handles_are_hidden() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let router = router_over(&workflow);

        assert!(router.config().is_none());
        assert!(router.writers().is_none());
        assert!(router.tracer().is_none());
        assert!(router.checkpointer().is_none());
        assert!(std::ptr::eq(router.trace(), workflow.trace()));
        assert_eq!(router.inner().id(), "s1");
    }

    #[tokio::test]
    async fn snapshot_forwards_and_restore_is_ignored() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        workflow.update_state(json!({ "step": 1 })).await.unwrap();
        let router = router_over(&workflow);

        let mut snapshot = router.snapshot().await.unwrap();
        assert_eq!(snapshot.session_id, "s1");
        assert_eq!(snapshot.kind, SessionType::Node);
        snapshot.global = json!({ "injected": true });
        router.restore(snapshot).await.unwrap();
        assert_eq!(workflow.get_state("injected").await.unwrap(), None);
        assert_eq!(workflow.get_state("step").await.unwrap(), Some(json!(1)));
    }

    #[tokio::test]
    async fn trace_sink_installation_is_rejected() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        let router = router_over(&workflow);
        let error = router
            .install_trace_sink(Arc::new(crate::tracer::MemoryTraceSink::new()))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(
            error.to_string().contains("router sessions are read-only"),
            "unexpected message: {error}"
        );
    }
}
