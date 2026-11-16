//! `RouterSession` — the read-only session handed to conditional-edge routers.
//!
//! Ports `RouterSession` from
//! `openjiuwen/core/session/internal/wrapper.py` (v0.1.20). Python builds it as
//! a `StateSession` subclass that reads through to the wrapped session and
//! selectively no-ops everything a router must not do: state writes, stream
//! writes, interaction, configuration lookups (`get_workflow_config`,
//! `get_agent_config`, `get_env`), and `base()`.
//!
//! The Rust port is a thin `Arc<dyn Session>` wrapper with the same shape, but
//! a rejected write is a typed error rather than a silent no-op: a routing
//! callback that tries to mutate state, publish a frame, or restore a snapshot
//! gets [`ErrorKind::Unsupported`] naming the operation, instead of believing a
//! write it never performed succeeded.
//!
//! | Operation | Behavior |
//! | --- | --- |
//! | `id`, `parent_id`, `get_state`, `get_component_state`, `trace`, `snapshot` | delegate to the inner session |
//! | `update_state`, `update_component_state`, `write_stream`, `restore`, `resolve_interaction` | [`ErrorKind::Unsupported`], with a per-method reason saying the router is read-only and pointing at [`RouterSession::inner`] |
//! | `end_stream`, `close` | deliberate `Ok(())` lifecycle no-ops: a router is a *nested* view, so ending or closing it would end the parent run's stream |
//! | `subscribe` | an immediately empty stream (a router never consumes the run's output) |
//! | `request_interaction` | [`ErrorKind::Unsupported`] (`"router sessions cannot interact"`) |
//! | `install_trace_sink` | [`ErrorKind::Unsupported`] (`"router sessions are read-only"`) |
//! | `config`, `writers`, `checkpointer`, `tracer` | `None` |
//! | `metadata` | forwarded, with `kind` forced to [`SessionType::Node`] |
//!
//! A routing callback can therefore inspect the state a branch decision needs
//! without being able to mutate it, close the run's stream, or swap the trace
//! sink out from under the workflow. `end_stream`/`close` stay silent successes
//! on purpose — they are lifecycle bookkeeping the surrounding workflow may run
//! over every session it holds, and a nested router must never close a stream it
//! does not own; the write paths have no such excuse, so they report the
//! rejection.
//!
//! [`ErrorKind::Unsupported`]: openjiuwen_common::error::ErrorKind::Unsupported

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

    async fn update_state(&self, _patch: Value) -> Result<()> {
        Err(Error::unsupported(
            "router sessions are read-only: global state updates are rejected; use \
             RouterSession::inner() to write through",
        ))
    }

    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
        self.inner.get_component_state(component_id).await
    }

    async fn update_component_state(&self, _component_id: &str, _patch: Value) -> Result<()> {
        Err(Error::unsupported(
            "router sessions are read-only: component state updates are rejected; use \
             RouterSession::inner() to write through",
        ))
    }

    async fn write_stream(&self, _item: StreamItem) -> Result<()> {
        Err(Error::unsupported(
            "router sessions are read-only: stream writes are rejected; use \
             RouterSession::inner() to publish through",
        ))
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
        Err(Error::unsupported(
            "router sessions are read-only: interactions cannot be resolved on a router session",
        ))
    }

    /// A deliberate lifecycle no-op.
    ///
    /// A router is a nested view over a live run, so ending the stream here
    /// would end the *parent* session's stream; the run's owner decides when
    /// that happens. Returns `Ok(())` and publishes nothing.
    async fn end_stream(&self) -> Result<()> {
        Ok(())
    }

    /// A deliberate lifecycle no-op.
    ///
    /// Like [`Session::end_stream`], closing a router must not close the shared
    /// hub of the session it wraps. Returns `Ok(())` and leaves the inner
    /// session untouched.
    async fn close(&self) -> Result<()> {
        Ok(())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        let mut snapshot = self.inner.snapshot().await?;
        snapshot.kind = SessionType::Node;
        Ok(snapshot)
    }

    async fn restore(&self, _snapshot: SessionSnapshot) -> Result<()> {
        Err(Error::unsupported(
            "router sessions are read-only: state restores are rejected; use \
             RouterSession::inner() to restore",
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use futures::StreamExt;
    use serde_json::json;

    use super::RouterSession;
    use crate::interaction::{InteractionKind, InteractionRequest, InteractionResponse};
    use crate::session::{Session, SessionType};
    use crate::sessions::workflow::WorkflowSession;
    use crate::stream::StreamItem;

    fn router_over(workflow: &Arc<WorkflowSession>) -> RouterSession {
        RouterSession::new(Arc::clone(workflow) as Arc<dyn Session>)
    }

    #[tokio::test]
    async fn reads_pass_through_and_writes_are_rejected() {
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

        let error = router
            .update_state(json!({ "step": 99 }))
            .await
            .expect_err("a router cannot write global state");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(
            error.to_string().contains("global state updates"),
            "{error}"
        );
        let error = router
            .update_component_state("branch", json!("right"))
            .await
            .expect_err("a router cannot write component state");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(
            error.to_string().contains("component state updates"),
            "{error}"
        );
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

        let error = router
            .write_stream(StreamItem::output_at("chunk", 0, json!({ "x": 1 })))
            .await
            .expect_err("a router cannot publish frames");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("stream writes"), "{error}");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), events.next())
                .await
                .is_err(),
            "a rejected router write must not reach the session stream"
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
        let error = router
            .resolve_interaction(InteractionResponse::cancel("intr_1"))
            .await
            .expect_err("a router cannot resolve interactions");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("read-only"), "{error}");
        assert!(router.end_stream().await.is_ok());
        assert!(router.close().await.is_ok());
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
    async fn snapshot_forwards_and_restore_is_rejected() {
        let workflow = Arc::new(WorkflowSession::new("s1"));
        workflow.update_state(json!({ "step": 1 })).await.unwrap();
        let router = router_over(&workflow);

        let mut snapshot = router.snapshot().await.unwrap();
        assert_eq!(snapshot.session_id, "s1");
        assert_eq!(snapshot.kind, SessionType::Node);
        snapshot.global = json!({ "injected": true });
        let error = router
            .restore(snapshot)
            .await
            .expect_err("a router cannot restore state");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("state restores"), "{error}");
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
