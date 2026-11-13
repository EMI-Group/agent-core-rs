//! Behavior tests for the wrapper sessions and the per-session trace hook.
//!
//! Ports the wrapper semantics of `openjiuwen/core/session/internal/workflow.py`
//! (`SubWorkflowSession`) and `openjiuwen/core/session/internal/wrapper.py`
//! (`StateSession`, `RouterSession`) from openJiuwen v0.1.20, plus the Rust
//! equivalent of Python's `BaseSession.set_tracer`: a session owns one stable
//! [`TraceSink`] handle whose inner sink can be swapped after construction.
//!
//! The three suites are:
//!
//! - **Sub-workflow** — a `SubWorkflowSession` built from a host `NodeSession`
//!   keeps the host's executable id, re-parents to the host's parent, overrides
//!   only its workflow placement, and merges its end output into the host lane.
//! - **Router** — every read passes through, every write is a deliberate no-op,
//!   interaction and sink installation are `Unsupported`.
//! - **Trace hook** — installing a sink reroutes spans started afterwards;
//!   `otel::install_trace_sink` is a documented no-op without the `otel`
//!   feature and installs with it.
//!
//! Everything is offline and deterministic; the only time-based assertions are
//! short bounded "nothing arrives" guards.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::constants::DEFAULT_WORKFLOW_ID;
use openjiuwen_session::interaction::InteractionKind;
use openjiuwen_session::interaction::InteractionRequest;
use openjiuwen_session::otel;
use openjiuwen_session::session::{Session, SessionType};
use openjiuwen_session::sessions::{
    NodeSession, RouterSession, SubWorkflowSession, WorkflowSession, WorkflowSessionOptions,
};
use openjiuwen_session::stream::StreamItem;
use openjiuwen_session::trace_hook::SwappableTraceSink;
use openjiuwen_session::tracer::{InvokeType, MemoryTraceSink, TracePosition, TraceSink};
use serde_json::{json, Value};

/// Upper bound on how long a deterministic in-memory receive may take.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Builds a workflow session carrying a custom workflow id.
fn workflow_with_id(workflow_id: &str) -> Arc<WorkflowSession> {
    Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some("s1".to_string()),
        workflow_id: Some(workflow_id.to_string()),
        ..WorkflowSessionOptions::default()
    }))
}

// ---------------------------------------------------------------------------
// SubWorkflowSession
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sub_workflow_keeps_the_host_executable_id_and_reparents() {
    let workflow = workflow_with_id("outer");
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    assert_eq!(
        sub.executable_id(),
        host.executable_id(),
        "`SubWorkflowSession` re-parents to the host's parent, so \
         `create_executable_id` yields the host node's id"
    );
    assert_eq!(sub.executable_id(), "sub");
    assert_eq!(sub.component_id(), "sub");
    assert_eq!(sub.node_type(), None);
    assert_eq!(sub.id(), "s1");
    assert_eq!(sub.session_type(), SessionType::Node);
    assert_eq!(sub.parent().id(), "s1");
}

#[tokio::test]
async fn sub_workflow_overrides_workflow_placement() {
    let workflow = workflow_with_id("outer");
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    assert_eq!(host.workflow_id(), "outer");
    assert_eq!(host.main_workflow_id(), "outer");
    assert_eq!(host.workflow_nesting_depth(), 0);

    assert_eq!(sub.workflow_id(), "inner");
    assert_eq!(
        sub.main_workflow_id(),
        "outer",
        "the main workflow is inherited, not overridden"
    );
    assert_eq!(sub.workflow_nesting_depth(), 1);

    let metadata = sub.metadata();
    assert_eq!(metadata.kind, SessionType::Node);
    assert_eq!(metadata.entity_id.as_deref(), Some("sub"));
    assert_eq!(metadata.metadata["workflow_id"], json!("inner"));
    assert_eq!(metadata.metadata["main_workflow_id"], json!("outer"));
    assert_eq!(metadata.metadata["workflow_nesting_depth"], json!(1));
    assert_eq!(metadata.metadata["executable_id"], json!("sub"));
    assert_eq!(metadata.metadata["component_id"], json!("sub"));
}

#[tokio::test]
async fn sub_workflow_defaults_workflow_id_to_the_host_workflow() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "");

    assert_eq!(sub.workflow_id(), DEFAULT_WORKFLOW_ID);
    assert_eq!(sub.main_workflow_id(), DEFAULT_WORKFLOW_ID);
}

#[tokio::test]
async fn nested_host_keeps_the_chained_executable_id() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let outer = Arc::new(workflow.create_node_session("a"));
    let host = outer.create_node_session("b");
    let sub = SubWorkflowSession::new(&host, "inner");

    assert_eq!(host.executable_id(), "a.b");
    assert_eq!(sub.executable_id(), "a.b");
    assert_eq!(sub.component_id(), "b");
    assert_eq!(sub.parent().id(), "s1");
    assert_eq!(sub.workflow_nesting_depth(), 1);
}

#[tokio::test]
async fn host_node_type_is_inherited() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = NodeSession::new_with_type(
        Arc::clone(&workflow) as Arc<dyn Session>,
        "sub",
        "sub_workflow",
    );
    let sub = SubWorkflowSession::new(&host, "inner");

    assert_eq!(host.node_type(), Some("sub_workflow"));
    assert_eq!(sub.node_type(), Some("sub_workflow"));
    assert_eq!(sub.metadata().metadata["node_type"], json!("sub_workflow"));
}

#[tokio::test]
async fn component_state_writes_land_in_the_parent_lane() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    sub.update_component_state("x", json!({ "value": 1 }))
        .await
        .unwrap();

    assert_eq!(
        sub.get_component_state("x").await.unwrap(),
        Some(json!({ "value": 1 }))
    );
    assert_eq!(
        workflow.get_component_state("sub.x").await.unwrap(),
        Some(json!({ "value": 1 })),
        "the sub-workflow shares the host node's lane in the parent workflow"
    );
    assert_eq!(
        host.get_component_state("x").await.unwrap(),
        Some(json!({ "value": 1 })),
        "the host node sees the same lane"
    );
}

#[tokio::test]
async fn global_state_passes_through_to_the_parent_workflow() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    sub.update_state(json!({ "step": 1 })).await.unwrap();
    assert_eq!(sub.get_state("step").await.unwrap(), Some(json!(1)));
    assert_eq!(workflow.get_state("step").await.unwrap(), Some(json!(1)));
}

#[tokio::test]
async fn end_output_propagates_into_the_host_component_lane() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    // Without a recorded end output propagation is a no-op.
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
        Some(json!({ "kept": true, "output": { "answer": 42 } })),
        "the end output merges under the host lane so `${{node.output}}` resolves"
    );

    // `take_end_output` consumes the slot.
    assert_eq!(sub.take_end_output(), Some(json!({ "answer": 42 })));
    assert_eq!(sub.end_output(), None);
    assert_eq!(sub.take_end_output(), None);
    sub.propagate_end_output().await.unwrap();
    assert_eq!(
        workflow.get_component_state("sub").await.unwrap(),
        Some(json!({ "kept": true, "output": { "answer": 42 } })),
        "a no-op propagation leaves the previously merged value intact"
    );
}

#[tokio::test]
async fn end_output_slot_is_shared_across_clones() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");
    let clone = sub.clone();

    clone.set_end_output(json!("done"));
    assert_eq!(sub.end_output(), Some(json!("done")));
    assert_eq!(sub.take_end_output(), Some(json!("done")));
    assert_eq!(clone.end_output(), None);
}

#[tokio::test]
async fn snapshot_and_streaming_delegate_to_the_host_node() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let host = workflow.create_node_session("sub");
    let sub = SubWorkflowSession::new(&host, "inner");

    workflow.update_state(json!({ "step": 1 })).await.unwrap();
    let snapshot = sub.snapshot().await.unwrap();
    assert_eq!(snapshot.session_id, "s1");
    assert_eq!(snapshot.kind, SessionType::Node);
    assert_eq!(snapshot.global, json!({ "step": 1 }));

    let mut events = workflow.subscribe();
    sub.write_stream(StreamItem::output_at("chunk", 0, json!({ "x": 1 })))
        .await
        .unwrap();
    let event = tokio::time::timeout(RECV_TIMEOUT, events.next())
        .await
        .expect("the frame arrives before the timeout")
        .expect("the stream is open")
        .expect("the event is Ok");
    let openjiuwen_session::stream::StreamEvent::Item(item) = event else {
        panic!("expected a stream item");
    };
    assert_eq!(item.item_type, "chunk");

    // A nested workflow never closes the shared hub.
    sub.end_stream().await.unwrap();
    sub.close().await.unwrap();
    assert!(!workflow.writers().unwrap().is_closed());
}

// ---------------------------------------------------------------------------
// RouterSession
// ---------------------------------------------------------------------------

#[tokio::test]
async fn router_reads_pass_through_and_identity_is_forwarded() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    workflow.update_state(json!({ "step": 1 })).await.unwrap();
    let node = workflow.create_node_session("branch");
    node.update_component_state("", json!({ "route": "left" }))
        .await
        .unwrap();
    let router = RouterSession::new(Arc::new(node) as Arc<dyn Session>);

    assert_eq!(router.session_id(), "s1");
    assert_eq!(router.id(), "s1");
    assert_eq!(router.parent_id(), Some("s1"));
    assert_eq!(router.executable_id(), Some("branch"));
    assert_eq!(router.session_type(), SessionType::Node);
    assert_eq!(router.metadata().kind, SessionType::Node);
    assert_eq!(router.get_state("step").await.unwrap(), Some(json!(1)));
    assert_eq!(
        router.get_component_state("route").await.unwrap(),
        Some(json!("left"))
    );
}

#[tokio::test]
async fn router_updates_are_deliberate_noops() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    workflow.update_state(json!({ "step": 1 })).await.unwrap();
    let node = workflow.create_node_session("branch");
    node.update_component_state("", json!({ "route": "left" }))
        .await
        .unwrap();
    let router = RouterSession::new(Arc::new(node) as Arc<dyn Session>);

    router.update_state(json!({ "step": 99 })).await.unwrap();
    router
        .update_component_state("route", json!("right"))
        .await
        .unwrap();

    assert_eq!(workflow.get_state("step").await.unwrap(), Some(json!(1)));
    assert_eq!(
        workflow.get_component_state("branch.route").await.unwrap(),
        Some(json!("left"))
    );
}

#[tokio::test]
async fn router_stream_writes_reach_nothing() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);
    let mut events = workflow.subscribe();

    router
        .write_stream(StreamItem::output_at("chunk", 0, json!({ "x": 1 })))
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(RECV_TIMEOUT, events.next())
            .await
            .is_err(),
        "a router session never publishes frames"
    );
    assert!(
        router.subscribe().next().await.is_none(),
        "a router session exposes an empty stream"
    );

    router.end_stream().await.unwrap();
    router.close().await.unwrap();
    assert!(!workflow.writers().unwrap().is_closed());
}

#[tokio::test]
async fn router_interaction_is_unsupported() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);

    let error = router
        .request_interaction(InteractionRequest::new(InteractionKind::Confirm, json!({})))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error
            .to_string()
            .contains("router sessions cannot interact"),
        "unexpected message: {error}"
    );
    assert!(router
        .resolve_interaction(openjiuwen_session::interaction::InteractionResponse::cancel(""))
        .await
        .is_ok());
}

#[tokio::test]
async fn router_hides_shared_handles() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);

    assert!(router.config().is_none());
    assert!(router.writers().is_none());
    assert!(router.tracer().is_none());
    assert!(router.checkpointer().is_none());
    assert!(
        std::ptr::eq(router.trace(), workflow.trace()),
        "the router exposes the inner session's trace sink"
    );
    assert_eq!(router.inner().id(), "s1");
}

#[tokio::test]
async fn router_snapshot_forwards_and_restore_is_ignored() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    workflow.update_state(json!({ "step": 1 })).await.unwrap();
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);

    let mut snapshot = router.snapshot().await.unwrap();
    assert_eq!(snapshot.session_id, "s1");
    assert_eq!(snapshot.kind, SessionType::Node);
    assert_eq!(snapshot.global, json!({ "step": 1 }));

    snapshot.global = json!({ "injected": true });
    router.restore(snapshot).await.unwrap();
    assert_eq!(workflow.get_state("injected").await.unwrap(), None);
    assert_eq!(workflow.get_state("step").await.unwrap(), Some(json!(1)));
}

#[tokio::test]
async fn router_rejects_trace_sink_installation() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);
    let sink: Arc<MemoryTraceSink> = Arc::new(MemoryTraceSink::new());

    let error = router.install_trace_sink(sink).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("router sessions are read-only"),
        "unexpected message: {error}"
    );
}

// ---------------------------------------------------------------------------
// Per-session trace-sink hook
// ---------------------------------------------------------------------------

#[tokio::test]
async fn installed_sink_records_spans_started_afterwards() {
    let workflow = WorkflowSession::new("s1");
    let before = Arc::new(MemoryTraceSink::new());
    workflow.install_trace_sink(before.clone()).unwrap();
    assert!(before.is_empty());

    let sink = Arc::new(MemoryTraceSink::new());
    workflow.install_trace_sink(sink.clone()).unwrap();

    let span = workflow
        .tracer()
        .expect("a workflow session owns a tracer")
        .start_span("after", InvokeType::Prompt);
    span.finish(Value::Null);

    assert_eq!(sink.len(), 1);
    assert!(before.is_empty(), "the replaced sink records nothing");
    assert_eq!(
        workflow.trace().position().map(|p| p.frame_count),
        Some(1),
        "`trace()` reports the installed sink's position"
    );
    let trace_id = workflow.tracer().expect("tracer").trace_id().to_string();
    assert_eq!(
        workflow.trace().position().map(|p| p.trace_id),
        Some(trace_id),
        "the installed sink reports the session's trace id"
    );
}

#[tokio::test]
async fn installed_sink_is_shared_with_children_and_checkpoints() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let node = workflow.create_node_session("node");
    let sub = SubWorkflowSession::new(&node, "inner");
    let sink = Arc::new(MemoryTraceSink::new());

    // Installing through a node or a sub-workflow targets the shared session.
    sub.install_trace_sink(sink.clone()).unwrap();
    let span = workflow
        .tracer()
        .expect("tracer")
        .start_span("through-sub", InvokeType::Prompt);
    span.finish(Value::Null);
    assert_eq!(sink.len(), 1);

    node.install_trace_sink(sink.clone()).unwrap();
    let span = workflow
        .tracer()
        .expect("tracer")
        .start_span("through-node", InvokeType::Prompt);
    span.finish(Value::Null);
    assert_eq!(sink.len(), 2);

    let snapshot = workflow.snapshot().await.unwrap();
    assert_eq!(
        snapshot.trace.map(|p| p.frame_count),
        Some(2),
        "the checkpoint carries the installed sink's position"
    );
}

#[tokio::test]
async fn installing_on_an_agent_session_reroutes_its_spans() {
    let session = openjiuwen_session::sessions::AgentSession::new("a1");
    let sink = Arc::new(MemoryTraceSink::new());
    session.install_trace_sink(sink.clone()).unwrap();
    let span = session
        .tracer()
        .expect("an agent session owns a tracer")
        .start_span("agent", InvokeType::Prompt);
    span.finish(Value::Null);
    assert_eq!(sink.len(), 1);
}

#[test]
fn swappable_sink_defaults_to_a_noop_and_delegates_positions() {
    let wrapper = SwappableTraceSink::default();
    assert!(wrapper.position().is_none());
    assert_eq!(
        format!("{:?}", wrapper.current()),
        format!(
            "{:?}",
            Arc::new(openjiuwen_session::tracer::NoopTraceSink::new()) as Arc<dyn TraceSink>
        )
    );

    let memory = Arc::new(MemoryTraceSink::new());
    wrapper.install(memory.clone());
    wrapper.set_position(TracePosition {
        trace_id: "trace-x".to_string(),
        last_invoke_id: Some("inv-1".to_string()),
        frame_count: 3,
    });
    assert_eq!(
        wrapper.position(),
        Some(TracePosition {
            trace_id: "trace-x".to_string(),
            last_invoke_id: Some("inv-1".to_string()),
            frame_count: 3,
        })
    );
    assert_eq!(memory.position().map(|p| p.frame_count), Some(3));
}

/// The `otel` helper is a documented no-op without the feature: it returns
/// `Ok(())` and leaves the session's sink untouched.
#[cfg(not(feature = "otel"))]
#[tokio::test]
async fn otel_helper_is_a_noop_without_the_feature() {
    let workflow = WorkflowSession::new("s1");
    let sink = Arc::new(MemoryTraceSink::new());

    otel::install_trace_sink(&workflow, sink.clone()).unwrap();
    let span = workflow
        .tracer()
        .expect("tracer")
        .start_span("ignored", InvokeType::Prompt);
    span.finish(Value::Null);

    assert!(
        sink.is_empty(),
        "without the `otel` feature the sink is not installed"
    );
    assert_eq!(
        workflow.trace().position().map(|p| p.frame_count),
        Some(1),
        "the session keeps its own stream sink, which observed the span"
    );
}

/// With the `otel` feature the same helper forwards to the session hook.
#[cfg(feature = "otel")]
#[tokio::test]
async fn otel_helper_installs_the_sink_with_the_feature() {
    let workflow = WorkflowSession::new("s1");
    let sink = Arc::new(MemoryTraceSink::new());

    otel::install_trace_sink(&workflow, sink.clone()).unwrap();
    let span = workflow
        .tracer()
        .expect("tracer")
        .start_span("recorded", InvokeType::Prompt);
    span.finish(Value::Null);

    assert_eq!(sink.len(), 1);
    assert_eq!(workflow.trace().position().map(|p| p.frame_count), Some(1));
}

/// With the `otel` feature a read-only session still rejects installation.
#[cfg(feature = "otel")]
#[tokio::test]
async fn otel_helper_propagates_router_rejection() {
    let workflow = Arc::new(WorkflowSession::new("s1"));
    let router = RouterSession::new(Arc::clone(&workflow) as Arc<dyn Session>);
    let error = otel::install_trace_sink(&router, Arc::new(MemoryTraceSink::new())).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

/// The `Session` default leaves minimal sessions unchanged: a session that does
/// not own a swappable sink reports the documented error.
#[tokio::test]
async fn default_session_install_reports_unsupported() {
    /// A session implementing only the required methods.
    #[derive(Debug)]
    struct Minimal {
        sink: MemoryTraceSink,
        global: parking_lot::Mutex<Value>,
    }

    #[async_trait::async_trait]
    impl Session for Minimal {
        #[allow(clippy::unnecessary_literal_bound)] // signature fixed by the trait
        fn id(&self) -> &str {
            "minimal"
        }

        fn parent_id(&self) -> Option<&str> {
            None
        }

        fn session_type(&self) -> SessionType {
            SessionType::Workflow
        }

        async fn get_state(&self, key: &str) -> openjiuwen_common::error::Result<Option<Value>> {
            let _ = key;
            Ok(Some(self.global.lock().clone()))
        }

        async fn update_state(&self, patch: Value) -> openjiuwen_common::error::Result<()> {
            self.global.lock().clone_from(&patch);
            Ok(())
        }

        async fn get_component_state(
            &self,
            component_id: &str,
        ) -> openjiuwen_common::error::Result<Option<Value>> {
            let _ = component_id;
            Ok(None)
        }

        async fn update_component_state(
            &self,
            component_id: &str,
            patch: Value,
        ) -> openjiuwen_common::error::Result<()> {
            let _ = (component_id, patch);
            Ok(())
        }

        async fn write_stream(&self, item: StreamItem) -> openjiuwen_common::error::Result<()> {
            let _ = item;
            Ok(())
        }

        async fn request_interaction(
            &self,
            request: InteractionRequest,
        ) -> openjiuwen_common::error::Result<openjiuwen_session::interaction::InteractionResponse>
        {
            let _ = request;
            Err(openjiuwen_common::error::Error::unsupported(
                "no interactions",
            ))
        }

        fn trace(&self) -> &dyn TraceSink {
            &self.sink
        }
    }

    let session = Minimal {
        sink: MemoryTraceSink::new(),
        global: parking_lot::Mutex::new(json!({})),
    };
    let error = session
        .install_trace_sink(Arc::new(MemoryTraceSink::new()))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error
            .to_string()
            .contains("session does not support installing a trace sink"),
        "unexpected message: {error}"
    );
}
