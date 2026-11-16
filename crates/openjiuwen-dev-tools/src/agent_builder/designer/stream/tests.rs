//! Tests for the streamed three-pass designer pipeline.
//!
//! Every case runs offline: either the deterministic designer or a
//! `ScriptedModel`, never a network call.

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

use super::*;

/// A valid final-design answer for a model-assisted pipeline.
const FINAL_FENCE: &str = r#"```json
{"name": "model-final", "description": "the final design",
 "nodes": [
   {"id": "start", "type": "start", "title": "Start", "description": "entry"},
   {"id": "work", "type": "llm", "title": "Do the work"},
   {"id": "end", "type": "end", "title": "End", "description": "exit"}],
 "edges": [{"source": "start", "target": "work"}, {"source": "work", "target": "end"}]}
```"#;

/// Drains a stream into the order it yielded.
async fn drain(mut stream: BoxStream<'static, Result<DesignEvent>>) -> Vec<Result<DesignEvent>> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// Coerces a scripted model into the trait object the designer takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

#[tokio::test]
async fn stream_reports_every_pass_in_order() {
    let designer = WorkflowDesigner::new();
    let events = drain(designer.stream_design("Collect the order. Then ship it.")).await;

    assert_eq!(events.len(), 7);
    let events: Vec<DesignEvent> = events
        .into_iter()
        .map(|event| event.expect("event"))
        .collect();
    assert_eq!(events[0], DesignEvent::PassStarted(DesignPassKind::Basic));
    assert!(matches!(
        events[1],
        DesignEvent::PassCompleted(DesignPass {
            kind: DesignPassKind::Basic,
            ..
        })
    ));
    assert_eq!(events[2], DesignEvent::PassStarted(DesignPassKind::Branch));
    assert!(matches!(
        events[3],
        DesignEvent::PassCompleted(DesignPass {
            kind: DesignPassKind::Branch,
            ..
        })
    ));
    assert_eq!(
        events[4],
        DesignEvent::PassStarted(DesignPassKind::Reflection)
    );
    assert!(matches!(
        events[5],
        DesignEvent::PassCompleted(DesignPass {
            kind: DesignPassKind::Reflection,
            ..
        })
    ));
    match &events[6] {
        DesignEvent::Completed(result) => {
            assert_eq!(result.passes.len(), 3);
            assert_eq!(result.provenance.len(), 3);
            result.design.validate().expect("design is valid");
        }
        other => panic!("expected the completed event, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_matches_the_three_pass_result() {
    let designer = WorkflowDesigner::new();
    let query = "Check the order. If it is valid, confirm it.";
    let expected = designer.design_three_pass(query).await.expect("three pass");
    let events = drain(designer.stream_design(query)).await;

    match events.last() {
        Some(Ok(DesignEvent::Completed(result))) => assert_eq!(result, &expected),
        other => panic!("expected the completed event, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_is_send_and_static() {
    let designer = WorkflowDesigner::new();
    let stream = designer.stream_design("Collect the order.");
    let handle = tokio::spawn(async move {
        let events = drain(stream).await;
        events.len()
    });
    assert_eq!(handle.await.expect("join"), 7);
}

#[tokio::test]
async fn stream_rejects_a_blank_query() {
    let designer = WorkflowDesigner::new();
    let events = drain(designer.stream_design("  \n ")).await;

    assert_eq!(events.len(), 1);
    let error = events[0].as_ref().expect_err("blank query");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("query must not be blank"));
}

#[tokio::test]
async fn stream_records_model_provenance() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text(FINAL_FENCE),
        ScriptedTurn::text(FINAL_FENCE),
        ScriptedTurn::text(FINAL_FENCE),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let events = drain(designer.stream_design("Collect the order.")).await;

    match events.last() {
        Some(Ok(DesignEvent::Completed(result))) => {
            assert_eq!(result.provenance.len(), 3);
            assert!(!result.provenance.has_deviation());
            assert_eq!(result.design.name, "model-final");
        }
        other => panic!("expected the completed event, got {other:?}"),
    }
    assert_eq!(model.call_count(), 3);
}

#[tokio::test]
async fn stream_reports_a_fallback_per_pass() {
    let model = Arc::new(ScriptedModel::new([
        ScriptedTurn::text("no fence"),
        ScriptedTurn::text("no fence"),
        ScriptedTurn::text("no fence"),
    ]));
    let designer = WorkflowDesigner::with_model(dynamic(&model));
    let events = drain(designer.stream_design("Collect the order.")).await;

    match events.last() {
        Some(Ok(DesignEvent::Completed(result))) => {
            assert_eq!(result.provenance.fallback_count(), 3);
            assert!(result.design.validate().is_ok());
        }
        other => panic!("expected the completed event, got {other:?}"),
    }
    assert_eq!(model.call_count(), 3);
}

#[test]
fn design_event_serializes_externally_tagged() {
    let started = serde_json::to_value(DesignEvent::PassStarted(DesignPassKind::Reflection))
        .expect("serialize");
    assert_eq!(started, serde_json::json!({"pass_started": "reflection"}));

    let completed = DesignEvent::Completed(ThreePassDesign::default());
    let value = serde_json::to_value(&completed).expect("serialize");
    assert_eq!(
        value,
        serde_json::json!({"completed": {
            "design": {"name": "", "description": "", "nodes": [], "edges": []},
            "passes": [],
            "provenance": {"stages": []},
            "notes": []
        }})
    );
    assert_eq!(
        serde_json::from_value::<DesignEvent>(value).expect("parse"),
        completed
    );
}
