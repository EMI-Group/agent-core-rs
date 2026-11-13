//! Integration tests for the persisted run ledger.
//!
//! These cover the behaviours Python encodes in `openjiuwen.rsi.usage`
//! (token alias normalisation, JSONL replay, `call_id` deduplication, the
//! `task_id` resume guard, pending-call delivery, and cumulative aggregation)
//! through the crate's public API only. Everything here is offline and
//! deterministic: temporary directories, no network, no model, no clock
//! assertions beyond the shape of a timestamp.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_rsi::ledger::{
    aggregate, tokens_from_usage, usage_snapshot, FileLedgerStore, InMemoryLedgerStore,
    LedgerStore, ModelCall, ModelCallStatus, RunEvent, RunLedger, StageRecord, UsageRecord,
    UsageTokens, UsageTotals,
};
use openjiuwen_rsi::ErrorKind;
use serde_json::{json, Value};
use std::sync::Arc;

/// Approximate floating-point equality, keeping assertions exact in intent.
fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

/// Builds a token triple.
fn tokens(input: Option<u64>, output: Option<u64>, cache_hit: Option<u64>) -> UsageTokens {
    UsageTokens::new(input, output, cache_hit)
}

/// Builds a usage record for `task` with a one-call model record.
fn usage_record(task_id: &str, call_id: &str, model: &str, tokens: UsageTokens) -> UsageRecord {
    UsageRecord::new(task_id, call_id, ModelCall::new(model).with_tokens(tokens))
}

/// Builds a stage record.
fn stage_record(node_ref: &str, note: &str) -> StageRecord {
    StageRecord::new(node_ref, json!({"stage": note})).with_note(note)
}

// ---------------------------------------------------------------------------
// Token normalisation (Python `usage.usage_tokens`)
// ---------------------------------------------------------------------------

#[test]
fn tokens_prefer_the_python_alias_order() {
    let raw = json!({
        "input_tokens": 11,
        "prompt_tokens": 22,
        "input": 33,
        "output_tokens": 44,
        "completion_tokens": 55,
        "output": 66,
        "cache_read_tokens": 7,
        "prompt_cache_hit_tokens": 8,
        "cache_hit": 9,
        "prompt_tokens_details": {"cached_tokens": 10},
    });
    assert_eq!(tokens_from_usage(&raw), tokens(Some(11), Some(44), Some(7)));
}

#[test]
fn tokens_accept_each_alias_alone() {
    for (key, expected) in [("input_tokens", 1_u64), ("prompt_tokens", 2), ("input", 3)] {
        assert_eq!(
            tokens_from_usage(&json!({ key: expected })).input,
            Some(expected),
            "input alias {key}"
        );
    }
    for (key, expected) in [
        ("output_tokens", 1_u64),
        ("completion_tokens", 2),
        ("output", 3),
    ] {
        assert_eq!(
            tokens_from_usage(&json!({ key: expected })).output,
            Some(expected),
            "output alias {key}"
        );
    }
    for (key, expected) in [
        ("cache_read_tokens", 1_u64),
        ("prompt_cache_hit_tokens", 2),
        ("cache_hit", 3),
    ] {
        assert_eq!(
            tokens_from_usage(&json!({ key: expected })).cache_hit,
            Some(expected),
            "cache alias {key}"
        );
    }
    assert_eq!(
        tokens_from_usage(&json!({"prompt_tokens_details": {"cached_tokens": 9}})).cache_hit,
        Some(9)
    );
}

#[test]
fn tokens_reject_invalid_counters_and_keep_falling_back() {
    let negative = json!({"input_tokens": -5, "prompt_tokens": 7});
    assert_eq!(tokens_from_usage(&negative).input, Some(7));

    let invalid = json!({"input_tokens": "12", "prompt_tokens": true});
    assert_eq!(tokens_from_usage(&invalid).input, None);

    assert_eq!(tokens_from_usage(&json!({"input_tokens": 1.5})).input, None);
    assert_eq!(
        tokens_from_usage(&json!({"input_tokens": 12.0})).input,
        Some(12)
    );
    assert_eq!(tokens_from_usage(&json!({"output": 3.0})).output, Some(3));

    assert!(tokens_from_usage(&json!({})).is_empty());
    assert!(tokens_from_usage(&json!(null)).is_empty());
}

#[test]
fn usage_snapshot_requires_token_or_cost_information() {
    assert!(usage_snapshot(&json!({})).is_none());
    assert!(usage_snapshot(&json!({"call_count": 3})).is_none());
    assert!(usage_snapshot(&json!(null)).is_none());

    let snapshot = usage_snapshot(&json!({
        "tokens": {"input_tokens": 5, "output_tokens": 2},
        "cost_estimate": 0.25,
        "call_count": 2,
    }))
    .unwrap();
    assert_eq!(snapshot.tokens.input, Some(5));
    assert_eq!(snapshot.tokens.output, Some(2));
    assert_eq!(snapshot.call_count, Some(2));
    assert!(close(snapshot.cost_estimate.unwrap(), 0.25));

    let aliases = usage_snapshot(&json!({"cost_usd": 1, "calls": 4})).unwrap();
    assert_eq!(aliases.tokens, UsageTokens::default());
    assert_eq!(aliases.call_count, Some(4));
    assert!(close(aliases.cost_estimate.unwrap(), 1.0));
}

// ---------------------------------------------------------------------------
// JSONL persistence and replay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn file_store_round_trips_and_replays_totals() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FileLedgerStore::new(dir.path()));

    let mut writer = RunLedger::new("task-1").with_store(store.clone());
    writer.bind().await.unwrap();
    let first = usage_record(
        "task-1",
        "call-1",
        "model-a",
        tokens(Some(10), Some(5), Some(2)),
    )
    .with_stage_ref("plan")
    .with_cost_usd(0.5);
    let second = usage_record("task-1", "call-2", "model-b", tokens(Some(7), None, None));
    assert!(writer.record_usage(first).await.unwrap());
    assert!(writer.record_usage(second).await.unwrap());
    writer
        .record_stage(stage_record("h0", "build"))
        .await
        .unwrap();

    // One flat JSON object per line, Python-shaped.
    let text = tokio::fs::read_to_string(store.path_for("task-1"))
        .await
        .unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    let first_line: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first_line["event_type"], "progress.usage");
    assert_eq!(first_line["family"], "progress");
    assert_eq!(first_line["call_id"], "call-1");
    let stage_line: Value = serde_json::from_str(lines[2]).unwrap();
    assert_eq!(stage_line["event_type"], "node.stage");
    assert_eq!(stage_line["node_ref"], "h0");

    // A fresh ledger replays the persisted events and the same totals.
    let mut resumed = RunLedger::new("task-1").with_store(store.clone());
    resumed.bind().await.unwrap();
    assert_eq!(resumed.records().len(), 3);
    assert_eq!(resumed.totals(), writer.totals());
    assert_eq!(resumed.totals().calls, 2);
    assert_eq!(resumed.totals().input_tokens, 17);
    assert_eq!(resumed.totals().output_tokens, 5);
    assert_eq!(resumed.totals().cache_hit_tokens, 2);
    assert!(close(resumed.totals().cost_usd, 0.5));
    assert_eq!(
        resumed.records()[2],
        RunEvent::Stage(stage_record("h0", "build"))
    );
}

#[tokio::test]
async fn file_store_reports_malformed_lines() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileLedgerStore::new(dir.path());
    tokio::fs::write(store.path_for("task-1"), "not json\n")
        .await
        .unwrap();

    let mut ledger = RunLedger::new("task-1").with_store(Arc::new(store));
    let error = ledger.bind().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("run ledger line 1"), "{error}");
}

#[tokio::test]
async fn file_store_skips_blank_lines_but_not_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileLedgerStore::new(dir.path());
    let event = RunEvent::Usage(usage_record(
        "task-1",
        "call-1",
        "model-a",
        tokens(Some(1), None, None),
    ));
    let line = serde_json::to_string(&event).unwrap();
    let payload = format!("{line}\n\n   \n{line}\n");
    tokio::fs::write(store.path_for("task-1"), payload)
        .await
        .unwrap();
    let loaded = store.load("task-1").await.unwrap();
    assert_eq!(loaded.len(), 2);
}

#[tokio::test]
async fn clear_removes_the_persisted_history() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FileLedgerStore::new(dir.path()));
    let mut ledger = RunLedger::new("task-1").with_store(store.clone());
    ledger.bind().await.unwrap();
    ledger
        .record_usage(usage_record(
            "task-1",
            "call-1",
            "model-a",
            tokens(Some(1), None, None),
        ))
        .await
        .unwrap();
    assert!(store.path_for("task-1").exists());

    ledger.clear().await.unwrap();
    assert!(!store.path_for("task-1").exists());
    assert_eq!(ledger.records().len(), 0);
    assert_eq!(ledger.totals(), UsageTotals::default());
    assert_eq!(store.load("task-1").await.unwrap().len(), 0);
}

#[tokio::test]
async fn remove_ignores_a_missing_history() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileLedgerStore::new(dir.path());
    store.remove("never-written").await.unwrap();
    assert_eq!(store.load("never-written").await.unwrap().len(), 0);
}

#[tokio::test]
async fn file_store_keeps_unsafe_task_ids_inside_the_root() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileLedgerStore::new(dir.path());
    let path = store.path_for("../escape");
    assert_eq!(path.parent(), Some(dir.path()));
    assert_eq!(path.file_name().unwrap(), "___escape.jsonl");
}

// ---------------------------------------------------------------------------
// Deduplication and resume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn duplicate_call_ids_are_recorded_once() {
    let store: Arc<dyn LedgerStore> = Arc::new(InMemoryLedgerStore::new());
    let mut ledger = RunLedger::new("task-1").with_store(store);
    ledger.bind().await.unwrap();

    let first = usage_record(
        "task-1",
        "call-1",
        "model-a",
        tokens(Some(10), Some(5), None),
    );
    let duplicate = usage_record(
        "task-1",
        "call-1",
        "model-a",
        tokens(Some(10), Some(5), None),
    );
    assert!(ledger.record_usage(first).await.unwrap());
    assert!(!ledger.record_usage(duplicate).await.unwrap());
    assert_eq!(ledger.totals().calls, 1);
    assert_eq!(ledger.totals().input_tokens, 10);
    assert_eq!(ledger.records().len(), 1);
}

#[tokio::test]
async fn replay_never_double_counts_and_bind_is_idempotent() {
    let store = Arc::new(InMemoryLedgerStore::new());
    let event = RunEvent::Usage(usage_record(
        "task-1",
        "call-1",
        "model-a",
        tokens(Some(4), Some(2), None),
    ));
    store.append("task-1", &event).await.unwrap();
    store.append("task-1", &event).await.unwrap();

    let mut ledger = RunLedger::new("task-1").with_store(store);
    ledger.bind().await.unwrap();
    assert_eq!(ledger.totals().calls, 1);
    assert_eq!(ledger.totals().input_tokens, 4);
    assert_eq!(ledger.records().len(), 2);

    ledger.bind().await.unwrap();
    assert_eq!(ledger.totals().calls, 1);
    assert_eq!(ledger.records().len(), 2);
}

#[tokio::test]
async fn bind_rejects_a_foreign_task_id() {
    let store: Arc<dyn LedgerStore> = Arc::new(InMemoryLedgerStore::new());
    let foreign = RunEvent::Usage(usage_record(
        "task-2",
        "call-1",
        "model-a",
        tokens(Some(1), None, None),
    ));
    store.append("task-1", &foreign).await.unwrap();

    let mut ledger = RunLedger::new("task-1").with_store(store);
    let error = ledger.bind().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("task_id does not match this run"),
        "{error}"
    );
}

#[tokio::test]
async fn record_usage_rejects_a_foreign_task_id() {
    let mut ledger = RunLedger::new("task-1");
    let error = ledger
        .record_usage(usage_record(
            "task-2",
            "call-1",
            "model-a",
            UsageTokens::default(),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(ledger.records().len(), 0);
}

// ---------------------------------------------------------------------------
// Delivery and snapshots
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pending_calls_fail_delivery_until_finished() {
    let mut ledger = RunLedger::new("task-1");
    ledger.mark_pending("call-9");
    let error = ledger.check_delivery().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(error.to_string().contains("delivery failed"), "{error}");

    ledger.finish_pending(ModelCallStatus::Incomplete);
    assert!(ledger.check_delivery().is_ok());
}

#[tokio::test]
async fn finishing_a_pending_call_rewrites_its_persisted_record() {
    let mut ledger = RunLedger::new("task-1");
    ledger
        .record_usage(usage_record(
            "task-1",
            "call-1",
            "model-a",
            UsageTokens::default(),
        ))
        .await
        .unwrap();
    ledger.mark_pending("call-1");
    ledger.finish_pending(ModelCallStatus::Incomplete);

    let record = ledger.records()[0].as_usage().unwrap();
    assert_eq!(record.model_call.status, ModelCallStatus::Incomplete);
    assert!(ledger.check_delivery().is_ok());
}

#[tokio::test]
async fn recording_a_pending_call_completes_it() {
    let mut ledger = RunLedger::new("task-1");
    ledger.mark_pending("call-1");
    ledger
        .record_usage(usage_record(
            "task-1",
            "call-1",
            "model-a",
            UsageTokens::default(),
        ))
        .await
        .unwrap();
    assert!(ledger.check_delivery().is_ok());
}

#[tokio::test]
async fn snapshots_follow_the_recorded_totals() {
    let mut ledger = RunLedger::new("task-1");
    let empty = ledger.snapshot();
    assert_eq!(empty.call_count, Some(0));
    assert!(empty.tokens.input.is_none());
    assert!(empty.cost_estimate.is_none());

    ledger
        .record_usage(
            usage_record(
                "task-1",
                "call-1",
                "model-a",
                tokens(Some(8), Some(3), Some(1)),
            )
            .with_cost_usd(0.25),
        )
        .await
        .unwrap();
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.call_count, Some(1));
    assert_eq!(snapshot.tokens, tokens(Some(8), Some(3), Some(1)));
    assert!(close(snapshot.cost_estimate.unwrap(), 0.25));
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn aggregate_buckets_by_model_and_stage() {
    let mut first = usage_record(
        "task-1",
        "c1",
        "model-a",
        tokens(Some(10), Some(5), Some(2)),
    )
    .with_stage_ref("plan")
    .with_cost_usd(0.5);
    first.model_call.call_count = 2;
    let mut second = usage_record("task-1", "c2", "model-a", tokens(Some(3), None, None));
    second.model_call.status = ModelCallStatus::Failed;
    let mut third =
        usage_record("task-1", "c3", "model-b", UsageTokens::default()).with_stage_ref("plan");
    third.model_call.status = ModelCallStatus::Incomplete;

    let events = vec![
        RunEvent::Usage(first),
        RunEvent::Usage(second),
        RunEvent::Usage(third),
        RunEvent::Stage(stage_record("h0", "build")),
    ];
    let metrics = aggregate(&events);

    assert_eq!(metrics.totals.calls, 4);
    assert_eq!(metrics.totals.input_tokens, 13);
    assert_eq!(metrics.totals.output_tokens, 5);
    assert_eq!(metrics.totals.cache_hit_tokens, 2);
    assert!(close(metrics.totals.cost_usd, 0.5));
    assert_eq!(metrics.failed_calls, 1);
    assert_eq!(metrics.incomplete_calls, 1);

    let model_a = metrics.by_model.get("model-a").unwrap();
    assert_eq!(model_a.calls, 3);
    assert_eq!(model_a.input_tokens, 13);
    let model_b = metrics.by_model.get("model-b").unwrap();
    assert_eq!(model_b.calls, 1);
    assert_eq!(model_b.input_tokens, 0);

    let plan = metrics.by_stage.get("plan").unwrap();
    assert_eq!(plan.calls, 3);
    let unattributed = metrics.by_stage.get("unattributed").unwrap();
    assert_eq!(unattributed.calls, 1);
    assert_eq!(metrics.by_stage.len(), 2);

    // The ledger's own metrics agree with the pure aggregator.
    let mut ledger = RunLedger::new("task-1");
    for event in &events {
        match event {
            RunEvent::Usage(record) => {
                ledger.record_usage(record.clone()).await.unwrap();
            }
            RunEvent::Stage(record) => {
                ledger.record_stage(record.clone()).await.unwrap();
            }
        }
    }
    assert_eq!(ledger.metrics(), metrics);
}

// ---------------------------------------------------------------------------
// Trait-object usability
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stores_are_usable_behind_the_trait_object() {
    let memory = Arc::new(InMemoryLedgerStore::new());
    let store: Arc<dyn LedgerStore> = memory.clone();
    let mut ledger = RunLedger::new("task-1").with_store(store);
    ledger.bind().await.unwrap();
    ledger
        .record_usage(usage_record(
            "task-1",
            "call-1",
            "model-a",
            tokens(Some(1), Some(1), None),
        ))
        .await
        .unwrap();

    assert_eq!(memory.tasks(), vec!["task-1".to_owned()]);
    let events = memory.events("task-1");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].as_usage().unwrap().call_id, "call-1");

    let event = events[0].clone();
    assert!(matches!(event, RunEvent::Usage(_)));
    assert_eq!(event.event_type(), "progress.usage");
    assert!(event.as_stage().is_none());
}
