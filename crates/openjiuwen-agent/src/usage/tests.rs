//! Unit tests of the usage seam: identity helpers and the telemetry buffer.
#![allow(clippy::assert_is_empty)]

use super::*;
use openjiuwen_context::ContextWindowUsage;
use serde_json::json;

/// A snapshot distinguishable by its window total and request id.
fn snapshot(tokens: u64, request_id: &str) -> ContextUsageSnapshot {
    ContextUsageSnapshot {
        request_id: request_id.to_owned(),
        context_window: ContextWindowUsage {
            tokens,
            ..ContextWindowUsage::default()
        },
        ..ContextUsageSnapshot::default()
    }
}

#[test]
fn main_context_id_matches_the_loop_context() {
    assert_eq!(MAIN_CONTEXT_ID, "main");
}

#[test]
fn default_analyzer_is_the_approximate_counter() {
    let analyzer = default_usage_analyzer();
    assert_eq!(analyzer.counter_identity(), &CounterIdentity::approx());
    assert_eq!(analyzer.model(), "");
    assert_eq!(analyzer.context_window_limit(), None);
}

#[test]
fn cache_scope_key_uses_the_run_session_and_the_request_model() {
    let request = ModelRequestConfig::new("gpt-4o");
    let scope = cache_scope_key("s1", "openai", &request);
    assert_eq!(scope.session_id, "s1");
    assert_eq!(scope.provider, "openai");
    assert_eq!(scope.model, "gpt-4o");
    // The remaining identity fields keep their documented defaults.
    assert_eq!(scope.deployment, "");
    assert_eq!(scope.cache_mode, "provider");
    assert_eq!(scope.cache_scope, "session");
    assert_eq!(scope.agent_id, None);
}

#[test]
fn request_id_is_deterministic_and_names_the_iteration() {
    assert_eq!(request_id("s1", 1), "s1#1");
    assert_eq!(request_id("s1", 12), "s1#12");
    // The same call always maps to the same id, so a retry does not double count.
    assert_eq!(request_id("s1", 2), request_id("s1", 2));
    assert_ne!(request_id("s1", 1), request_id("s2", 1));
}

#[test]
fn provider_usage_value_serializes_the_wire_shape() {
    let usage = Usage::new(11, 7);
    let value = provider_usage_value(&usage);
    assert_eq!(value["input_tokens"], 11);
    assert_eq!(value["output_tokens"], 7);
    assert_eq!(value["total_tokens"], 18);
    // The engine's parser reads exactly this object back.
    let parsed = openjiuwen_context::request_usage_from_metadata(&value);
    assert_eq!(parsed.input_tokens, Some(11));
}

#[test]
fn telemetry_records_lists_and_drains() {
    let telemetry = UsageTelemetry::new();
    assert!(telemetry.is_empty());
    assert_eq!(telemetry.last(), None);

    telemetry.record(snapshot(10, "s1#1"));
    telemetry.record(snapshot(20, "s1#2"));
    assert_eq!(telemetry.len(), 2);
    assert_eq!(telemetry.recorded_total(), 2);
    assert_eq!(
        telemetry.last().map(|event| event.request_id),
        Some("s1#2".to_owned())
    );

    let drained = telemetry.drain();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].context_window.tokens, 10);
    assert_eq!(drained[1].context_window.tokens, 20);
    // Draining empties the buffer but keeps the lifetime counter.
    assert!(telemetry.is_empty());
    assert_eq!(telemetry.last(), None);
    assert_eq!(telemetry.recorded_total(), 2);
    assert!(telemetry.drain().is_empty());
}

#[test]
fn telemetry_clones_share_one_buffer() {
    let telemetry = UsageTelemetry::new();
    let handle = telemetry.clone();
    handle.record(snapshot(5, "s1#1"));
    assert_eq!(telemetry.len(), 1);
    assert_eq!(telemetry.drain().len(), 1);
    assert!(handle.is_empty());
}

#[test]
fn telemetry_is_capped_and_drops_the_oldest_event() {
    let telemetry = UsageTelemetry::new();
    for tokens in 0..(MAX_TELEMETRY_EVENTS as u64 + 2) {
        telemetry.record(snapshot(tokens, "s1#1"));
    }
    assert_eq!(telemetry.len(), MAX_TELEMETRY_EVENTS);
    // Records 0 and 1 fell off the front; the last one is still the newest.
    let drained = telemetry.drain();
    assert_eq!(
        drained.first().map(|event| event.context_window.tokens),
        Some(2)
    );
    assert_eq!(
        drained.last().map(|event| event.context_window.tokens),
        Some(MAX_TELEMETRY_EVENTS as u64 + 1)
    );
    assert_eq!(telemetry.recorded_total(), MAX_TELEMETRY_EVENTS as u64 + 2);
}

#[test]
fn telemetry_clear_drops_without_returning() {
    let telemetry = UsageTelemetry::new();
    telemetry.record(snapshot(1, "s1#1"));
    telemetry.clear();
    assert!(telemetry.is_empty());
    assert_eq!(telemetry.recorded_total(), 1);
}

#[test]
fn snapshots_serialize_with_the_python_payload_shape() {
    // The frame payload carries the snapshot verbatim, so the seam must not
    // reshape it.
    let event = snapshot(42, "s1#1");
    let value = serde_json::to_value(&event).expect("a snapshot serializes");
    assert_eq!(value["type"], json!("context.usage"));
    assert_eq!(value["context_window"]["tokens"], 42);
}
