//! Event streaming, both in-process and through the session stream hub.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::json;

use openjiuwen_graph::{ExecutionOptions, GraphBuilder, GraphEventKind};
use openjiuwen_session::{Session, StreamEvent};

fn two_step_graph() -> openjiuwen_graph::CompiledGraph {
    GraphBuilder::new("stream")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .edge("a", "b")
        .exit("b")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn events_arrive_in_execution_order() {
    let events: Vec<_> = two_step_graph()
        .stream(json!({}), common::session())
        .map(|item| item.unwrap())
        .collect()
        .await;

    let shape: Vec<(GraphEventKind, Option<String>, u32)> = events
        .iter()
        .map(|event| (event.kind, event.node_id.clone(), event.superstep))
        .collect();
    let none = None;
    let expected = vec![
        (GraphEventKind::SuperstepStarted, none.clone(), 1),
        (GraphEventKind::NodeStarted, Some("a".to_string()), 1),
        (GraphEventKind::NodeCompleted, Some("a".to_string()), 1),
        (GraphEventKind::EdgeTraversed, Some("a".to_string()), 1),
        (GraphEventKind::SuperstepCompleted, none.clone(), 1),
        (GraphEventKind::SuperstepStarted, none.clone(), 2),
        (GraphEventKind::NodeStarted, Some("b".to_string()), 2),
        (GraphEventKind::NodeCompleted, Some("b".to_string()), 2),
        (GraphEventKind::SuperstepCompleted, none.clone(), 2),
        (GraphEventKind::GraphCompleted, none, 2),
    ];
    assert_eq!(shape, expected);
}

#[tokio::test]
async fn stream_wire_payloads_are_objects() {
    let events: Vec<_> = two_step_graph()
        .stream(json!({}), common::session())
        .map(|item| item.unwrap())
        .collect()
        .await;
    for event in &events {
        let wire = event.to_wire();
        assert!(wire.is_object());
        assert!(wire["payload"].is_object());
        assert_eq!(wire["superstep"], json!(event.superstep));
    }
}

async fn trace_payloads(session: &Arc<dyn Session>) -> Vec<serde_json::Value> {
    let mut subscription = session.subscribe();
    let mut payloads = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_millis(200), subscription.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                assert_eq!(item.item_type, "graph_event");
                assert!(item.payload.is_object());
                payloads.push(item.payload);
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("stream error: {error}"),
        }
    }
    payloads
}

#[tokio::test]
async fn session_trace_mirrors_every_event() {
    let session = common::session();
    let mut subscription = session.subscribe();
    let graph = two_step_graph();
    graph
        .run(json!({}), Arc::clone(&session), ExecutionOptions::default())
        .await
        .unwrap();

    let mut payloads = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_millis(200), subscription.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                assert_eq!(item.item_type, "graph_event");
                assert!(item.payload.is_object());
                assert_eq!(item.mode, openjiuwen_session::StreamMode::Trace);
                payloads.push(item.payload);
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("stream error: {error}"),
        }
    }
    assert_eq!(payloads.len(), 10);
    assert_eq!(payloads[0]["kind"], json!("superstep_started"));
    assert_eq!(payloads[1]["kind"], json!("node_started"));
    assert_eq!(payloads[1]["node_id"], json!("a"));
    assert_eq!(payloads[9]["kind"], json!("graph_completed"));
}

#[tokio::test]
async fn trace_mirroring_can_be_disabled() {
    let session = common::session();
    let subscription = session.subscribe();
    two_step_graph()
        .run(
            json!({}),
            Arc::clone(&session),
            ExecutionOptions {
                write_session_trace: false,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(trace_payloads(&session).await.is_empty());
    drop(subscription);
}

#[tokio::test]
async fn execution_options_sender_receives_events() {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    two_step_graph()
        .run(
            json!({}),
            common::session(),
            ExecutionOptions {
                events: Some(sender),
                write_session_trace: false,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap();
    let mut kinds = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        kinds.push(event.kind);
    }
    assert_eq!(kinds.len(), 10);
    assert_eq!(kinds[0], GraphEventKind::SuperstepStarted);
    assert_eq!(kinds[9], GraphEventKind::GraphCompleted);
}
