//! Unit tests for the workflow engine: compilation, run wiring, streaming,
//! interrupts, resume, and the `CompiledWorkflow` facade.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::stream::StreamMode;
use openjiuwen_session::{
    InteractiveInput, MemoryCheckpointer, Session, WorkflowSession, WorkflowSessionOptions,
};
use serde_json::json;

use super::test_support::{CountingComponent, TestComponent};
use super::*;
use crate::components::{BranchRouterAdapter, EndComponent, StartComponent};
use crate::condition::{Branch, BranchRouter, Condition};
use crate::output::{WorkflowExecutionState, END_NODE_STREAM, WORKFLOW_FINAL};

fn session(name: &str) -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new(name))
}

/// A session checkpointed into a fresh in-memory store, plus the store itself.
fn checkpointed_session(name: &str) -> (Arc<dyn Session>, MemoryCheckpointer) {
    let checkpointer = MemoryCheckpointer::new();
    let session: Arc<dyn Session> =
        Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
            session_id: Some(name.to_owned()),
            checkpointer: Some(Arc::new(checkpointer.clone())),
            ..WorkflowSessionOptions::default()
        }));
    (session, checkpointer)
}

fn echo() -> TestComponent {
    TestComponent::new(vec![ComponentAbility::Invoke])
}

/// Records the conventional `start`/`end` component ids.
fn with_endpoints(mut definition: WorkflowDefinition) -> WorkflowDefinition {
    definition.start = Some("start".to_owned());
    definition.end = Some("end".to_owned());
    definition
}

/// `start -> mid -> end`, where `mid` is the component under test.
fn linear(mid: TestComponent, end: NodeConfig) -> WorkflowDefinition {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component("mid", mid.shared(), NodeConfig::default())
        .insert_component("end", Arc::new(EndComponent::default()), end)
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["mid".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );
    with_endpoints(definition)
}

/// An end node whose declared ability is `Stream`.
fn streaming_end() -> NodeConfig {
    NodeConfig {
        abilities: Some(vec![ComponentAbility::Stream]),
        ..NodeConfig::default()
    }
}

#[test]
fn build_reports_validation_problems() {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition.insert_component(
        "start",
        Arc::new(StartComponent::new()),
        NodeConfig::default(),
    );
    let error = CompiledWorkflow::build(definition).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(!error.message().is_empty());
}

#[test]
fn build_exposes_the_card_the_spec_and_the_abilities() {
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();

    assert_eq!(workflow.card().id, "wf");
    assert_eq!(workflow.start_comp_id(), "start");
    assert_eq!(workflow.end_comp_id(), "end");
    assert_eq!(workflow.spec().start_nodes, vec!["start".to_owned()]);
    assert!(workflow.spec().comp_configs.contains_key("mid"));
    assert!(!workflow.is_streaming());
    assert_eq!(workflow.ability("mid"), Some(ComponentAbility::Invoke));
    assert_eq!(workflow.ability("missing"), None);
    assert_eq!(workflow.config().card.id, "wf");
    assert_eq!(workflow.config().spec.start_nodes, vec!["start".to_owned()]);
    assert_eq!(
        workflow.definition().component_ids(),
        vec!["start", "mid", "end"]
    );
    assert!(format!("{workflow:?}").contains("CompiledWorkflow"));
}

#[test]
fn a_streaming_end_makes_the_workflow_streaming() {
    let workflow = CompiledWorkflow::build(linear(echo(), streaming_end())).unwrap();
    assert!(workflow.is_streaming());
    assert_eq!(workflow.ability("end"), Some(ComponentAbility::Stream));
}

#[tokio::test]
async fn invoke_runs_a_linear_workflow() {
    let session = session("linear");
    let end = NodeConfig {
        inputs_schema: Some(json!({"answer": "${mid.echo.query}"})),
        ..NodeConfig::default()
    };
    let workflow = CompiledWorkflow::build(linear(echo(), end)).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "hi"}}));
    // Python's `commit_user_inputs`: the inputs landed in the global lane.
    assert_eq!(session.get_state("query").await.unwrap(), Some(json!("hi")));
}

#[tokio::test]
async fn root_and_node_references_both_resolve() {
    let session = session("references");
    let mid = TestComponent {
        abilities: vec![ComponentAbility::Invoke],
        ..TestComponent::new(vec![ComponentAbility::Invoke])
    };
    let mut definition = linear(mid, NodeConfig::default());
    definition.insert_component(
        "mid",
        echo().shared(),
        NodeConfig {
            inputs_schema: Some(json!({
                "root": "${query}",
                "node": "${start.query}",
                "missing": "${nope.x}",
            })),
            ..NodeConfig::default()
        },
    );
    let workflow = CompiledWorkflow::build(definition).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();

    let node = output
        .result
        .get("output")
        .and_then(|state| state.get("mid"))
        .unwrap();
    assert_eq!(
        node,
        &json!({"echo": {"root": "hi", "node": "hi", "missing": null}})
    );
}

#[tokio::test]
async fn non_object_inputs_are_not_merged_into_the_session() {
    let session = session("scalars");
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
    let output = workflow
        .invoke(
            json!("plain"),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(session.get_state("query").await.unwrap(), None);
    assert_eq!(
        output.result.get("output").unwrap().get("inputs"),
        Some(&json!("plain"))
    );
}

#[tokio::test]
async fn an_interrupt_reports_input_required_with_the_wires() {
    let session = session("interrupt");
    let mid = echo().with_question(json!({"id": "ask", "question": "go?"}));
    let workflow = CompiledWorkflow::build(linear(mid, NodeConfig::default())).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        output.result,
        json!([{
            "type": "__interaction__",
            "index": 0,
            "payload": {"id": "ask", "index": 0, "question": {"id": "ask", "question": "go?"}}
        }])
    );
}

#[tokio::test]
async fn invoke_reports_cancellation_and_timeouts() {
    let session = session("timeouts");
    let workflow = CompiledWorkflow::build(linear(
        echo().with_delay(Duration::from_secs(30)),
        NodeConfig::default(),
    ))
    .unwrap();

    let token = CancellationToken::new();
    token.cancel();
    let cancelled = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default().with_cancellation(token),
        )
        .await
        .unwrap_err();
    assert_eq!(cancelled.kind(), ErrorKind::Cancelled);

    let timed_out = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default().with_timeout(Duration::from_millis(10)),
        )
        .await
        .unwrap_err();
    assert_eq!(timed_out.kind(), ErrorKind::Timeout);
    assert!(timed_out.to_string().contains("execution timed out"));
}

#[tokio::test]
async fn resume_requires_a_checkpointer() {
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
    let error = workflow
        .resume(
            InteractiveInput::new(),
            session("no-checkpointer"),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.message(),
        "interactive resume requested without a checkpointer"
    );
}

#[tokio::test]
async fn stream_terminates_when_the_run_produces_no_end_frames() {
    // Regression: the stream generator used to keep the run scope (and with it the
    // only `UnboundedSender` clone) alive across its yields, so the `recv` loop
    // waited forever for a sender that could not drop. The timeout turns a
    // regression into a failure instead of a hang.
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
    let mut stream = workflow.stream(
        json!({"query": "hi"}),
        session("terminating"),
        WorkflowRunOptions::default(),
    );

    let outcome = tokio::time::timeout(Duration::from_secs(5), async move {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }
        chunks
    })
    .await;
    assert!(outcome.is_ok(), "the stream must terminate");
    let mut chunks = outcome.unwrap();

    assert_eq!(chunks.len(), 1);
    let chunk = chunks.pop().unwrap().unwrap();
    assert_eq!(chunk.item_type, WORKFLOW_FINAL);
}

#[tokio::test]
async fn a_streaming_producer_stores_raw_frames_and_the_end_wires() {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component(
            "mid",
            TestComponent::new(vec![ComponentAbility::Stream])
                .with_frames(vec![json!(1), json!(2)])
                .shared(),
            NodeConfig {
                abilities: Some(vec![ComponentAbility::Stream]),
                ..NodeConfig::default()
            },
        )
        .insert_component(
            "end",
            TestComponent::new(vec![ComponentAbility::Stream])
                .with_frames(vec![json!("done")])
                .shared(),
            NodeConfig {
                abilities: Some(vec![ComponentAbility::Stream]),
                outputs_schema: Some(json!({"producer": "${mid}", "end": "${output}"})),
                ..NodeConfig::default()
            },
        )
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["mid".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );
    let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            session("buffered-stream"),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();

    // A non-end producer stores the raw payloads; only the end node builds wires.
    assert_eq!(output.result["producer"], json!([1, 2]));
    assert_eq!(
        output.result["end"],
        json!([{"type": END_NODE_STREAM, "index": 0, "payload": "done"}])
    );
}

#[tokio::test]
async fn a_completed_run_deletes_the_snapshot_and_an_interrupt_keeps_it() {
    let (session, checkpointer) = checkpointed_session("checkpoint-lifecycle");
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        checkpointer.len(session.id()),
        0,
        "a completed run releases its graph snapshot"
    );

    let interrupted = CompiledWorkflow::build(linear(
        echo().with_question(json!({"id": "ask", "question": "go?"})),
        NodeConfig::default(),
    ))
    .unwrap();
    let output = interrupted
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        checkpointer.len(session.id()),
        1,
        "an interrupted run keeps its pre-superstep snapshot"
    );
}

#[tokio::test]
async fn an_interrupt_resumes_the_interrupted_superstep_with_the_seeded_answer() {
    let (session, _checkpointer) = checkpointed_session("resume-interrupt");
    let workflow = CompiledWorkflow::build(linear(
        echo().with_question(json!({"id": "ask", "question": "go?"})),
        NodeConfig::default(),
    ))
    .unwrap();

    let first = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);

    let mut interactive = InteractiveInput::new();
    interactive.answer("ask", json!("yes"));

    let resumed = workflow
        .resume(
            interactive,
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(resumed.result["output"]["mid"], json!({"answer": "yes"}));
}

#[tokio::test]
async fn a_resumed_run_does_not_re_execute_completed_nodes() {
    let (session, checkpointer) = checkpointed_session("resume-failure");
    let mid_calls = Arc::new(AtomicUsize::new(0));
    let boom_calls = Arc::new(AtomicUsize::new(0));

    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component(
            "mid",
            CountingComponent::new(Arc::clone(&mid_calls), 0).shared(),
            NodeConfig::default(),
        )
        .insert_component(
            "boom",
            CountingComponent::new(Arc::clone(&boom_calls), 1).shared(),
            NodeConfig::default(),
        )
        .insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig {
                inputs_schema: Some(json!({"seen": "${boom.call}"})),
                ..NodeConfig::default()
            },
        )
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["mid".to_owned()],
            "boom".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["boom".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );
    let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

    let failure = workflow
        .invoke(
            json!({"query": "hi"}),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(
        failure.to_string().contains("attempt 1 failed"),
        "{failure}"
    );
    assert_eq!(mid_calls.load(Ordering::SeqCst), 1);
    assert_eq!(boom_calls.load(Ordering::SeqCst), 1);
    assert_eq!(checkpointer.len(session.id()), 1);

    let resumed = workflow
        .resume(
            InteractiveInput::new(),
            Arc::clone(&session),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(resumed.result["output"], json!({"seen": 2}));
    assert_eq!(
        mid_calls.load(Ordering::SeqCst),
        1,
        "the resumed run must not re-execute completed nodes"
    );
    assert_eq!(boom_calls.load(Ordering::SeqCst), 2);
    assert_eq!(checkpointer.len(session.id()), 0);
}

#[tokio::test]
async fn stream_yields_end_frames_then_the_final_output() {
    let workflow = CompiledWorkflow::build(linear(echo(), streaming_end())).unwrap();
    let mut stream = workflow.stream(
        json!({"query": "hi"}),
        session("stream"),
        WorkflowRunOptions::default(),
    );

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }
    assert!(!chunks.is_empty());

    let final_chunk = chunks.pop().unwrap();
    assert_eq!(final_chunk.item_type, WORKFLOW_FINAL);
    assert_eq!(final_chunk.mode, StreamMode::Output);
    assert_eq!(final_chunk.index, Some(0));

    assert!(!chunks.is_empty());
    for (index, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.item_type, END_NODE_STREAM);
        assert_eq!(chunk.mode, StreamMode::Output);
        assert_eq!(chunk.index, Some(i64::try_from(index).unwrap()));
    }
    // The final payload is the collected `end node stream` wire list.
    assert_eq!(final_chunk.payload.as_array().unwrap().len(), chunks.len());
    assert_eq!(final_chunk.payload[0]["type"], json!(END_NODE_STREAM));
    assert_eq!(final_chunk.payload[0]["index"], json!(0));
}

#[tokio::test]
async fn stream_forwards_custom_frames_and_filters_modes() {
    let workflow = CompiledWorkflow::build(linear(
        echo().with_custom(json!({"progress": 1})),
        NodeConfig::default(),
    ))
    .unwrap();

    let mut stream = workflow.stream(
        json!({"query": "hi"}),
        session("custom"),
        WorkflowRunOptions::default().with_stream_modes(vec![StreamMode::Custom]),
    );
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].mode, StreamMode::Custom);
    assert_eq!(chunks[0].payload, json!({"progress": 1}));
}

#[tokio::test]
async fn stream_yields_exactly_one_error() {
    let workflow =
        CompiledWorkflow::build(linear(echo().with_failure("kaboom"), NodeConfig::default()))
            .unwrap();
    let mut stream = workflow.stream(
        json!({"query": "hi"}),
        session("failure"),
        WorkflowRunOptions::default(),
    );

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk);
    }
    assert_eq!(chunks.len(), 1);
    let error = chunks.pop().unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.to_string().contains("kaboom"), "{error}");
}

#[tokio::test]
async fn stream_ends_without_a_final_frame_when_the_run_interrupts() {
    let workflow = CompiledWorkflow::build(linear(
        echo().with_question(json!({"id": "ask", "question": "go?"})),
        NodeConfig::default(),
    ))
    .unwrap();
    let mut stream = workflow.stream(
        json!({"query": "hi"}),
        session("stream-interrupt"),
        WorkflowRunOptions::default(),
    );

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].item_type, "__interaction__");
    assert_eq!(chunks[0].payload["id"], json!("ask"));
}

#[tokio::test]
async fn conditional_routers_select_targets_and_union_their_results() {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component("mid", echo().shared(), NodeConfig::default())
        .insert_component("a", echo().shared(), NodeConfig::default())
        .insert_component("b", echo().shared(), NodeConfig::default())
        .insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        )
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["a".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );

    let mut router = BranchRouter::new();
    let _ = router.add_branch(Branch::new(
        Condition::new("${query} == 'hi'").unwrap(),
        vec!["a".to_owned()],
    ));
    definition.add_conditional(
        "mid",
        "a",
        Arc::new(BranchRouterAdapter::new(Arc::new(router))),
    );
    let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            session("routing"),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    let state = output.result.get("output").unwrap();
    assert!(state.get("a").is_some(), "{state}");
    assert!(state.get("b").is_none(), "{state}");
}

#[tokio::test]
async fn an_undeclared_router_target_is_a_run_error() {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component("mid", echo().shared(), NodeConfig::default())
        .insert_component("a", echo().shared(), NodeConfig::default())
        .insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        )
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["a".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );

    let mut router = BranchRouter::new();
    let _ = router.add_branch(Branch::new(
        Condition::new("${query} == 'hi'").unwrap(),
        vec!["b".to_owned()],
    ));
    definition.add_conditional(
        "mid",
        "a",
        Arc::new(BranchRouterAdapter::new(Arc::new(router))),
    );
    let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

    let error = workflow
        .invoke(
            json!({"query": "hi"}),
            session("undeclared"),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("undeclared target 'b'"),
        "{error}"
    );
}

#[tokio::test]
async fn component_routing_short_circuits_the_routers() {
    let mut definition = WorkflowDefinition::new(test_support::card());
    definition
        .insert_component(
            "start",
            Arc::new(StartComponent::new()),
            NodeConfig::default(),
        )
        .insert_component(
            "mid",
            echo().with_targets(vec!["b".to_owned()]).shared(),
            NodeConfig::default(),
        )
        .insert_component("a", echo().shared(), NodeConfig::default())
        .insert_component("b", echo().shared(), NodeConfig::default())
        .insert_component(
            "end",
            Arc::new(EndComponent::default()),
            NodeConfig::default(),
        )
        .add_edge(
            vec!["start".to_owned()],
            "mid".to_owned(),
            DefinitionEdgeKind::Regular,
        )
        .add_edge(
            vec!["b".to_owned()],
            "end".to_owned(),
            DefinitionEdgeKind::Regular,
        );
    // The router below would pick `a`; the component's own `route` wins and picks
    // `b`, which is the only branch that reaches `end`.
    let mut router = BranchRouter::new();
    let _ = router.add_branch(Branch::new(
        Condition::new("${query} == 'hi'").unwrap(),
        vec!["a".to_owned()],
    ));
    definition
        .add_conditional(
            "mid",
            "a",
            Arc::new(BranchRouterAdapter::new(Arc::new(router))),
        )
        .add_conditional("mid", "b", test_support::router());
    let workflow = CompiledWorkflow::build(with_endpoints(definition)).unwrap();

    let output = workflow
        .invoke(
            json!({"query": "hi"}),
            session("component-route"),
            WorkflowRunOptions::default(),
        )
        .await
        .unwrap();
    let state = output.result.get("output").unwrap();
    assert!(state.get("b").is_some(), "{state}");
    assert!(state.get("a").is_none(), "{state}");
}

#[test]
fn to_mermaid_renders_nodes_and_edges() {
    let workflow = CompiledWorkflow::build(linear(echo(), NodeConfig::default())).unwrap();
    let rendered = workflow.to_mermaid().unwrap();
    assert!(rendered.contains("start"), "{rendered}");
    assert!(rendered.contains("mid"), "{rendered}");
    assert!(rendered.contains("end"), "{rendered}");
}

#[test]
fn rendering_an_uncompilable_definition_reports_the_compile_error() {
    // `order` names a component that was never inserted, so validation passes but the graph
    // cannot be compiled. `CompiledWorkflow` is immutable, which is why the test shapes the
    // definition before building it.
    let mut definition = linear(echo(), NodeConfig::default());
    definition.order.push("ghost".to_owned());
    let workflow = CompiledWorkflow::build(definition).unwrap();

    let mermaid = workflow.to_mermaid().unwrap_err();
    assert_eq!(mermaid.kind(), ErrorKind::Workflow);
    assert!(mermaid.message().contains("ghost"), "{mermaid}");
    assert!(
        !mermaid.to_string().contains("could not be compiled"),
        "no placeholder comment is produced: {mermaid}"
    );
    assert!(
        !mermaid.to_string().contains("%%"),
        "no Mermaid text leaks into the error: {mermaid}"
    );

    let svg = workflow.to_svg().unwrap_err();
    assert_eq!(svg.kind(), ErrorKind::Workflow);
    assert!(svg.message().contains("ghost"), "{svg}");
    assert!(
        !svg.to_string().contains("<?xml"),
        "no SVG document is fabricated: {svg}"
    );
    assert!(
        !svg.to_string().contains("could not be compiled"),
        "no error document is fabricated: {svg}"
    );
}

#[test]
fn run_options_are_chainable_and_permissive() {
    let options = WorkflowRunOptions::new();
    assert!(!options.resume);
    assert!(options.timeout.is_none());
    assert!(options.interactive.is_none());
    assert!(options.stream_modes.is_none());
    assert!(options.max_supersteps.is_none());
    assert!(!options.cancellation.is_cancelled());
    assert_eq!(options.stream_delivery, StreamDelivery::Buffered);

    let token = CancellationToken::new();
    token.cancel();
    let mut interactive = InteractiveInput::new();
    interactive.answer("ask", json!("yes"));
    let configured = WorkflowRunOptions::default()
        .with_cancellation(token.clone())
        .with_timeout(Duration::from_secs(1))
        .with_interactive(interactive)
        .with_resume(true)
        .with_stream_modes(vec![StreamMode::Trace])
        .with_max_supersteps(3)
        .with_stream_delivery(StreamDelivery::Interleaved);
    assert!(configured.cancellation.is_cancelled());
    assert_eq!(configured.timeout, Some(Duration::from_secs(1)));
    assert!(configured.interactive.is_some());
    assert!(configured.resume);
    assert_eq!(configured.stream_modes, Some(vec![StreamMode::Trace]));
    assert_eq!(configured.max_supersteps, Some(3));
    assert_eq!(
        configured.stream_delivery,
        StreamDelivery::Interleaved,
        "the delivery mode is recorded verbatim for the run funnel to check"
    );
}
