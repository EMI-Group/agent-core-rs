//! Integration tests for harness-side optimisation progress events.
//!
//! Python's `openjiuwen.rsi.harness_rsi.progress` is a downstream-facing
//! contract: consumers depend on the event's serialised shape and on the
//! callback firing exactly once per event. These tests cover both, plus the
//! `SeedOptimizationDecision` conversion used by seed-acceptance callbacks.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_rsi::harness_rsi::progress::{
    emit_progress, OptimizationProgressEvent, ProgressCallback, SeedOptimizationDecision,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[test]
fn progress_events_round_trip_through_json() {
    let mut metrics = BTreeMap::new();
    metrics.insert("tokens".to_owned(), json!(128));
    let mut artifacts = BTreeMap::new();
    artifacts.insert("patch".to_owned(), json!({"lines": 4}));
    let event = OptimizationProgressEvent {
        phase: "optimize".to_owned(),
        epoch: 2,
        batch_index: 3,
        stage: "evaluate".to_owned(),
        score: Some(0.875),
        improved: Some(true),
        message: "batch 3 passed".to_owned(),
        metrics,
        artifacts,
    };

    let value: Value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["phase"], "optimize");
    assert_eq!(value["epoch"], 2);
    assert_eq!(value["batch_index"], 3);
    assert_eq!(value["score"], 0.875);
    assert_eq!(value["improved"], true);
    assert_eq!(value["metrics"]["tokens"], 128);

    let round_trip: OptimizationProgressEvent = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, event);
}

#[test]
fn progress_events_default_missing_and_ignore_unknown_fields() {
    let event: OptimizationProgressEvent =
        serde_json::from_value(json!({"phase": "plan", "future_field": [1, 2]})).unwrap();
    assert_eq!(event, OptimizationProgressEvent::new("plan"));
    assert_eq!(event.epoch, 0);
    assert_eq!(event.batch_index, 0);
    assert_eq!(event.stage, "");
    assert!(event.score.is_none());
    assert!(event.improved.is_none());
    assert_eq!(event.message, "");
    assert_eq!(event.metrics.len(), 0);
    assert_eq!(event.artifacts.len(), 0);

    let default_value = serde_json::to_value(OptimizationProgressEvent::default()).unwrap();
    assert_eq!(default_value["phase"], "");
}

#[test]
fn builders_populate_the_optional_fields() {
    let event = OptimizationProgressEvent::new("learn")
        .with_score(0.5)
        .with_improved(false)
        .with_message("flat");
    assert_eq!(event.phase, "learn");
    assert_eq!(event.score, Some(0.5));
    assert_eq!(event.improved, Some(false));
    assert_eq!(event.message, "flat");
}

#[test]
fn seed_decisions_convert_to_and_from_booleans() {
    assert_eq!(
        SeedOptimizationDecision::from_bool(true),
        SeedOptimizationDecision::Accept
    );
    assert_eq!(
        SeedOptimizationDecision::from_bool(false),
        SeedOptimizationDecision::Reject
    );
    assert_eq!(
        SeedOptimizationDecision::from(true),
        SeedOptimizationDecision::Accept
    );
    assert!(SeedOptimizationDecision::Accept.as_bool());
    assert!(!SeedOptimizationDecision::Reject.as_bool());
    assert_eq!(SeedOptimizationDecision::Accept.as_str(), "accept");
    assert_eq!(SeedOptimizationDecision::Reject.as_str(), "reject");
    assert_eq!(SeedOptimizationDecision::Reject.to_string(), "reject");
    assert_eq!(
        SeedOptimizationDecision::default(),
        SeedOptimizationDecision::Accept
    );
    let value = serde_json::to_value(SeedOptimizationDecision::Reject).unwrap();
    assert_eq!(value, json!("reject"));
    let round_trip: SeedOptimizationDecision = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, SeedOptimizationDecision::Reject);
}

#[test]
fn emit_progress_invokes_the_callback_once_per_event() {
    let sink: Arc<Mutex<Vec<OptimizationProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&sink);
    let callback: ProgressCallback = Arc::new(move |event| {
        captured.lock().unwrap().push(event.clone());
    });

    let first = OptimizationProgressEvent::new("plan").with_score(0.1);
    let second = OptimizationProgressEvent::new("evaluate").with_score(0.2);
    emit_progress(Some(&callback), &first);
    emit_progress(Some(&callback), &second);

    let received = sink.lock().unwrap();
    assert_eq!(received.len(), 2);
    assert_eq!(received[0], first);
    assert_eq!(received[1], second);
}

#[test]
fn emit_progress_without_a_callback_is_a_no_op() {
    let event = OptimizationProgressEvent::new("plan");
    emit_progress(None, &event);
}
