#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for compression telemetry.
//!
//! The suite pins the [`ContextCompressionState`] wire contract (field names,
//! `type` tag, status and phase spellings, round-trip and unknown-field
//! tolerance), the recorder's in-place rewrite diffing, its bounded history, the
//! sink and session-stream fan-out (including swallowed failures), and the
//! task-local compaction correlation scope.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, no model is configured, and the session is an
//! in-process `AgentSession` whose stream is read through a bounded await.
//!
//! [`ContextCompressionState`]: openjiuwen_context::telemetry::ContextCompressionState
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use openjiuwen_context::telemetry::{
    current_compression_operation, current_compression_operation_id, current_request_purpose,
    format_timestamp, modified_messages, stamp_current_operation, with_operation,
    with_operation_scope, CompressionOperation, CompressionStatePhase, CompressionStateSink,
    CompressionStateStatus, ContextCompressionMetric, ContextCompressionModifiedMessage,
    ContextCompressionState, ContextCompressionUsage, ContextProcessorStateInput,
    ContextProcessorStateRecorder, RecordingSink, COMPACTION_PURPOSE,
    CONTEXT_COMPRESSION_STATE_TYPE, CONTEXT_OPERATION_ID_KEY, REQUEST_PURPOSE_KEY,
};
use openjiuwen_context::CONTEXT_MESSAGE_ID_KEY;
use openjiuwen_foundation::llm::ModelRequestConfig;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Usage};
use openjiuwen_session::{AgentSession, Session, StreamEvent};

/// How long a stream read may take before the test fails.
const BOUND: Duration = Duration::from_secs(5);

fn stamped(mut message: BaseMessage, id: &str) -> BaseMessage {
    message.metadata.insert(
        CONTEXT_MESSAGE_ID_KEY.to_owned(),
        serde_json::Value::String(id.to_owned()),
    );
    message
}

fn recorder() -> ContextProcessorStateRecorder {
    ContextProcessorStateRecorder::new("session-1", "main")
        .with_token_counter(Arc::new(ApproxTokenCounter::with_chars_per_token(1)))
}

fn timing() -> (SystemTime, SystemTime) {
    let started = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    (started, started + Duration::from_millis(250))
}

/// Four messages, 46 tokens with the one-character-per-token counter.
fn before_messages() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        stamped(BaseMessage::user("hi"), "m1"),
        stamped(BaseMessage::assistant("hello"), "m2"),
        stamped(BaseMessage::tool("call_1", "0123456789abcdef"), "m3"),
    ]
}

/// Three messages, 42 tokens: `m1` is dropped, `m0` is new, `m2` was rewritten.
fn after_messages() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        stamped(BaseMessage::user("[CONTEXT_SUMMARY]"), "m0"),
        stamped(BaseMessage::assistant("hello!"), "m2"),
    ]
}

fn completed_input<'a>(
    before: &'a [BaseMessage],
    after: &'a [BaseMessage],
) -> ContextProcessorStateInput<'a> {
    let (started, ended) = timing();
    ContextProcessorStateInput::new(
        "op-1",
        CompressionStateStatus::Completed,
        CompressionStatePhase::ActiveCompress,
        "dialogue",
        before,
    )
    .with_after(after)
    .with_model("gpt-4o")
    .with_context_max(1_000)
    .with_timing(started, Some(ended))
    .with_usage(ContextCompressionUsage::from_usage(
        &Usage::new(10, 5),
        "gpt-4o",
    ))
    .with_compact_summary("short")
}

// ---------------------------------------------------------------------------
// The wire contract
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_record_uses_the_python_field_names() {
    let before = before_messages();
    let after = after_messages();
    let state = recorder().emit(completed_input(&before, &after)).await;
    let value = serde_json::to_value(&state).unwrap();

    assert_eq!(value["type"], CONTEXT_COMPRESSION_STATE_TYPE);
    assert_eq!(value["operation_id"], "op-1");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["phase"], "active_compress");
    assert_eq!(value["processor"], "dialogue");
    assert_eq!(value["model"], "gpt-4o");
    assert_eq!(value["before"]["messages"], 4);
    assert!(value["before"]["time"].is_string());
    assert_eq!(value["after"]["messages"], 3);
    assert!(value["after"]["tokens"].is_u64());
    assert_eq!(value["statistic"]["total_messages"], 3);
    assert!(value["saved"]["percent"].is_f64());
    assert_eq!(
        value["modified_messages"],
        serde_json::json!([{
            "message_id": "m2",
            "role": "assistant",
            "tool_call_id": null,
            "offload_handle": null,
            "offload_type": null,
        }])
    );
    assert_eq!(value["compression_usage"]["calls"], 1);
    assert_eq!(value["compression_usage"]["model_name"], "gpt-4o");
    assert_eq!(value["duration_ms"], 250);
    assert_eq!(value["context_max"], 1_000);
    assert_eq!(value["compact_summary"], "short");
    assert_eq!(value["error"], serde_json::Value::Null);
    assert_eq!(value["reason"], "");
    assert!(value["summary"]
        .as_str()
        .unwrap()
        .starts_with("Compressed 4 -> 3 messages"));
}

#[test]
fn every_status_and_phase_serializes_to_its_python_spelling() {
    let statuses = [
        (CompressionStateStatus::Started, "\"started\""),
        (CompressionStateStatus::Completed, "\"completed\""),
        (CompressionStateStatus::Noop, "\"noop\""),
        (CompressionStateStatus::Skipped, "\"skipped\""),
        (CompressionStateStatus::Failed, "\"failed\""),
        (CompressionStateStatus::Paused, "\"paused\""),
        (CompressionStateStatus::Cancelled, "\"cancelled\""),
    ];
    for (status, wire) in statuses {
        assert_eq!(serde_json::to_string(&status).unwrap(), wire);
    }

    let phases = [
        (CompressionStatePhase::AddMessages, "\"add_messages\""),
        (
            CompressionStatePhase::GetContextWindow,
            "\"get_context_window\"",
        ),
        (CompressionStatePhase::ActiveCompress, "\"active_compress\""),
    ];
    for (phase, wire) in phases {
        assert_eq!(serde_json::to_string(&phase).unwrap(), wire);
    }
}

#[tokio::test]
async fn a_record_round_trips_and_tolerates_unknown_or_missing_fields() {
    let before = before_messages();
    let after = after_messages();
    let state = recorder().emit(completed_input(&before, &after)).await;

    let json = serde_json::to_string(&state).unwrap();
    assert_eq!(
        serde_json::from_str::<ContextCompressionState>(&json).unwrap(),
        state
    );

    let loose: ContextCompressionState =
        serde_json::from_str(r#"{"operation_id": "op-9", "status": "paused", "future_field": 3}"#)
            .unwrap();
    assert_eq!(loose.operation_id, "op-9");
    assert_eq!(loose.status, CompressionStateStatus::Paused);
    assert_eq!(loose.state_type, CONTEXT_COMPRESSION_STATE_TYPE);
    assert_eq!(loose.before, ContextCompressionMetric::default());
    assert_eq!(loose.after, None);
    assert_eq!(loose.saved, None);
    assert_eq!(loose.modified_messages.len(), 0);

    let empty: ContextCompressionState = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, ContextCompressionState::default());
    assert_eq!(empty.status, CompressionStateStatus::Started);
    assert_eq!(empty.phase, CompressionStatePhase::AddMessages);
}

#[test]
fn the_convenience_constructors_set_the_expected_status() {
    let phase = CompressionStatePhase::GetContextWindow;
    assert_eq!(
        ContextCompressionState::started("op", phase).status,
        CompressionStateStatus::Started
    );
    assert_eq!(
        ContextCompressionState::completed("op", phase).status,
        CompressionStateStatus::Completed
    );
    assert_eq!(
        ContextCompressionState::noop("op", phase).status,
        CompressionStateStatus::Noop
    );
    let failed = ContextCompressionState::failed("op", phase, "boom");
    assert_eq!(failed.status, CompressionStateStatus::Failed);
    assert_eq!(failed.error.as_deref(), Some("boom"));
    assert!(failed.is_terminal());
    assert!(!ContextCompressionState::started("op", phase).is_terminal());
}

// ---------------------------------------------------------------------------
// In-place rewrite diffing
// ---------------------------------------------------------------------------

#[test]
fn a_changed_message_is_reported_even_when_the_length_is_unchanged() {
    let before = vec![stamped(BaseMessage::assistant("abcdef"), "m1")];
    let after = vec![stamped(BaseMessage::assistant("abcXef"), "m1")];
    let modified = modified_messages(&before, &after);
    assert_eq!(modified.len(), 1);
    assert_eq!(modified[0].message_id, "m1");
    assert_eq!(modified[0].role, "assistant");
    assert_eq!(modified[0].offload_handle, None);
}

#[test]
fn unchanged_new_and_removed_messages_are_not_rewrites() {
    let before = vec![
        stamped(BaseMessage::user("same"), "a"),
        stamped(BaseMessage::user("gone"), "b"),
        BaseMessage::user("no id"),
    ];
    let after = vec![
        stamped(BaseMessage::user("same"), "a"),
        stamped(BaseMessage::user("brand new"), "c"),
        BaseMessage::user("no id"),
    ];
    assert_eq!(modified_messages(&before, &after), Vec::new());
}

#[test]
fn a_rewrite_that_offloads_propagates_the_handle_and_type() {
    let before = vec![stamped(
        BaseMessage::tool("call_1", "very long output"),
        "t1",
    )];
    let after = vec![stamped(
        BaseMessage::tool("call_1", "[[OFFLOAD: handle=h1, type=filesystem]]"),
        "t1",
    )];
    assert_eq!(
        modified_messages(&before, &after),
        vec![ContextCompressionModifiedMessage {
            message_id: "t1".to_owned(),
            role: "tool".to_owned(),
            tool_call_id: Some("call_1".to_owned()),
            offload_handle: Some("h1".to_owned()),
            offload_type: Some("filesystem".to_owned()),
        }]
    );
}

#[test]
fn an_unchanged_message_is_not_reported_even_with_an_offload_marker() {
    let message = stamped(
        BaseMessage::tool("call_1", "[[OFFLOAD: handle=h1, type=in_memory]]"),
        "t1",
    );
    assert_eq!(
        modified_messages(
            std::slice::from_ref(&message),
            std::slice::from_ref(&message)
        ),
        Vec::new()
    );
}

// ---------------------------------------------------------------------------
// The recorder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_recorder_measures_both_sides_and_the_savings() {
    let before = before_messages();
    let after = after_messages();
    let state = recorder().emit(completed_input(&before, &after)).await;

    assert_eq!(state.before.messages, 4);
    assert_eq!(state.before.tokens, 46);
    assert_eq!(state.before.context_percent, Some(5));
    assert_eq!(
        state.before.time.as_deref(),
        Some("2023-11-14T22:13:20.000Z")
    );
    let after_metric = state.after.unwrap();
    assert_eq!(after_metric.messages, 3);
    assert_eq!(after_metric.tokens, 42);
    assert_eq!(
        after_metric.time.as_deref(),
        Some("2023-11-14T22:13:20.250Z")
    );
    assert_eq!(state.duration_ms, Some(250));
    assert_eq!(state.context_max, Some(1_000));

    let saved = state.saved.unwrap();
    assert_eq!(saved.messages, 1);
    assert_eq!(saved.tokens, 4);
    assert!((saved.percent - 8.7).abs() < 1e-9);

    assert_eq!(state.statistic.total_messages, 3);
    assert_eq!(state.statistic.system_messages, 1);
    assert_eq!(state.compression_usage.unwrap().calls, 1);
    assert_eq!(state.modified_messages.len(), 1);
}

#[tokio::test]
async fn a_started_record_has_no_after_metric_savings_or_diff() {
    let before = before_messages();
    let state = recorder()
        .emit(ContextProcessorStateInput::new(
            "op-2",
            CompressionStateStatus::Started,
            CompressionStatePhase::GetContextWindow,
            "truncation",
            &before,
        ))
        .await;
    assert_eq!(state.after, None);
    assert_eq!(state.saved, None);
    assert_eq!(state.modified_messages.len(), 0);
    assert_eq!(state.duration_ms, None);
    assert_eq!(state.summary, "Compressing 4 messages, ~46 tokens");
}

#[tokio::test]
async fn a_failed_record_keeps_the_error_and_the_before_size() {
    let before = before_messages();
    let state = recorder()
        .emit(
            ContextProcessorStateInput::new(
                "op-3",
                CompressionStateStatus::Failed,
                CompressionStatePhase::ActiveCompress,
                "dialogue",
                &before,
            )
            .with_error("model rejected the request"),
        )
        .await;
    assert_eq!(state.error.as_deref(), Some("model rejected the request"));
    assert!(state.summary.contains("Context processor failed"));
}

#[tokio::test]
async fn supplier_statistics_win_and_are_otherwise_computed() {
    let before = before_messages();
    let supplied =
        openjiuwen_context::compute_stats(&before, &ApproxTokenCounter::with_chars_per_token(1));
    let state = recorder()
        .emit(
            ContextProcessorStateInput::new(
                "op-4",
                CompressionStateStatus::Noop,
                CompressionStatePhase::AddMessages,
                "truncation",
                &before,
            )
            .with_stats(supplied.clone(), None),
        )
        .await;
    assert_eq!(state.statistic, supplied);

    let computed = recorder()
        .emit(ContextProcessorStateInput::new(
            "op-5",
            CompressionStateStatus::Noop,
            CompressionStatePhase::AddMessages,
            "truncation",
            &before,
        ))
        .await;
    assert_eq!(computed.statistic.total_messages, 4);
    assert_eq!(computed.statistic.total_tokens, 46);
    // The trailing tool result starts an incomplete round, as in Python.
    assert_eq!(computed.statistic.total_dialogues, 2);
}

#[tokio::test]
async fn the_history_keeps_only_the_newest_records() {
    let before = before_messages();
    let recorder = ContextProcessorStateRecorder::new("session-1", "main").with_history_limit(2);
    for index in 0..5 {
        recorder
            .emit(ContextProcessorStateInput::new(
                format!("op-{index}"),
                CompressionStateStatus::Noop,
                CompressionStatePhase::AddMessages,
                "truncation",
                &before,
            ))
            .await;
    }
    let history = recorder.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].operation_id, "op-3");
    assert_eq!(history[1].operation_id, "op-4");
    assert_eq!(recorder.last().unwrap().operation_id, "op-4");
    assert_eq!(recorder.history_limit(), 2);

    recorder.load_history(history);
    assert_eq!(recorder.history().len(), 2);
    recorder.clear_history();
    assert_eq!(recorder.history(), Vec::new());
    assert_eq!(recorder.last(), None);
}

// ---------------------------------------------------------------------------
// Sink fan-out
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_sink_sees_started_then_the_terminal_states_in_order() {
    let sink = Arc::new(RecordingSink::new());
    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_sink(Arc::clone(&sink) as Arc<dyn CompressionStateSink>);
    let before = before_messages();
    let after = after_messages();

    let statuses = [
        CompressionStateStatus::Started,
        CompressionStateStatus::Completed,
        CompressionStateStatus::Noop,
        CompressionStateStatus::Failed,
    ];
    for status in statuses {
        recorder
            .emit(ContextProcessorStateInput::new(
                "op-1",
                status,
                CompressionStatePhase::ActiveCompress,
                "dialogue",
                &before,
            ))
            .await;
    }
    recorder.emit(completed_input(&before, &after)).await;

    assert_eq!(
        sink.statuses(),
        vec![
            CompressionStateStatus::Started,
            CompressionStateStatus::Completed,
            CompressionStateStatus::Noop,
            CompressionStateStatus::Failed,
            CompressionStateStatus::Completed,
        ]
    );
    assert_eq!(sink.len(), 5);
    assert!(!sink.is_empty());
    assert_eq!(sink.last().unwrap().modified_messages.len(), 1);
    assert_eq!(recorder.history().len(), 5);
}

#[tokio::test]
async fn a_failing_sink_does_not_fail_emit() {
    let sink = Arc::new(RecordingSink::failing());
    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_sink(Arc::clone(&sink) as Arc<dyn CompressionStateSink>);
    let before = before_messages();
    let state = recorder
        .emit(ContextProcessorStateInput::new(
            "op-1",
            CompressionStateStatus::Failed,
            CompressionStatePhase::ActiveCompress,
            "dialogue",
            &before,
        ))
        .await;
    assert_eq!(state.operation_id, "op-1");
    assert_eq!(sink.len(), 0);
    assert_eq!(recorder.history().len(), 1);
}

// ---------------------------------------------------------------------------
// Session stream emission
// ---------------------------------------------------------------------------

#[tokio::test]
async fn emit_writes_a_compression_state_frame_to_the_session_stream() {
    let session = Arc::new(AgentSession::new("session-1"));
    let mut frames = session.subscribe();
    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_session(Arc::clone(&session) as Arc<dyn Session>);
    let before = before_messages();
    let after = after_messages();

    recorder.emit(completed_input(&before, &after)).await;

    let event = tokio::time::timeout(BOUND, frames.next())
        .await
        .expect("a frame arrives within the bound")
        .expect("the stream stays open")
        .expect("the stream yields no error");
    let frame = match event {
        StreamEvent::Item(item) => item,
        StreamEvent::End => panic!("the stream ended before the frame"),
    };
    assert_eq!(frame.item_type, CONTEXT_COMPRESSION_STATE_TYPE);
    assert_eq!(frame.index, Some(0));
    assert_eq!(frame.payload["type"], CONTEXT_COMPRESSION_STATE_TYPE);
    assert_eq!(frame.payload["status"], "completed");
    assert_eq!(frame.payload["operation_id"], "op-1");
    assert_eq!(frame.payload["modified_messages"][0]["message_id"], "m2");
}

#[tokio::test]
async fn a_recorder_without_a_session_still_records() {
    let before = before_messages();
    let recorder = ContextProcessorStateRecorder::new("session-1", "main");
    recorder
        .emit(ContextProcessorStateInput::new(
            "op-1",
            CompressionStateStatus::Noop,
            CompressionStatePhase::AddMessages,
            "truncation",
            &before,
        ))
        .await;
    assert_eq!(recorder.history().len(), 1);
    let debugged = format!("{recorder:?}");
    assert!(debugged.contains("has_session: false"));
    assert!(debugged.contains("history_len: 1"));
}

// ---------------------------------------------------------------------------
// Timestamps and summary text
// ---------------------------------------------------------------------------

#[test]
fn timestamps_are_iso_8601_utc_with_milliseconds() {
    assert_eq!(format_timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
    let later = UNIX_EPOCH + Duration::from_millis(1_704_164_645_006);
    assert_eq!(format_timestamp(later), "2024-01-02T03:04:05.006Z");
}

#[tokio::test]
async fn a_noop_pass_reports_the_unchanged_size() {
    let before = before_messages();
    let state = recorder()
        .emit(
            ContextProcessorStateInput::new(
                "op-1",
                CompressionStateStatus::Noop,
                CompressionStatePhase::ActiveCompress,
                "dialogue",
                &before,
            )
            .with_after(&before),
        )
        .await;
    assert_eq!(
        state.summary,
        "Context unchanged at ~46 tokens (saved 0.0%)"
    );
    let saved = state.saved.unwrap();
    assert_eq!(saved.messages, 0);
    assert_eq!(saved.tokens, 0);
}

// ---------------------------------------------------------------------------
// Compaction correlation scope
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_operation_id_is_visible_only_inside_its_scope() {
    assert_eq!(current_compression_operation_id(), None);
    assert_eq!(current_compression_operation(), None);
    assert_eq!(current_request_purpose(), None);

    let seen = with_operation("op-1", async {
        assert_eq!(current_compression_operation_id().as_deref(), Some("op-1"));
        assert_eq!(
            current_compression_operation().unwrap().purpose,
            COMPACTION_PURPOSE
        );
        assert_eq!(
            current_request_purpose().as_deref(),
            Some(COMPACTION_PURPOSE)
        );
        current_compression_operation_id()
    })
    .await;
    assert_eq!(seen.as_deref(), Some("op-1"));
    assert_eq!(current_compression_operation_id(), None);
}

#[tokio::test]
async fn nested_scopes_restore_the_outer_operation() {
    with_operation("outer", async {
        assert_eq!(current_compression_operation_id().as_deref(), Some("outer"));
        with_operation("inner", async {
            assert_eq!(current_compression_operation_id().as_deref(), Some("inner"));
        })
        .await;
        assert_eq!(current_compression_operation_id().as_deref(), Some("outer"));
    })
    .await;
    assert_eq!(current_compression_operation_id(), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_tasks_see_their_own_operation() {
    let mut handles = Vec::new();
    for index in 0..8 {
        let operation_id = format!("op-{index}");
        handles.push(tokio::spawn(with_operation(
            operation_id.clone(),
            async move {
                tokio::task::yield_now().await;
                assert_eq!(
                    current_compression_operation_id().as_deref(),
                    Some(operation_id.as_str())
                );
                current_compression_operation_id()
            },
        )));
    }
    for (index, handle) in handles.into_iter().enumerate() {
        assert_eq!(
            handle.await.unwrap(),
            Some(format!("op-{index}")),
            "each task sees its own operation id"
        );
    }
    assert_eq!(current_compression_operation_id(), None);
}

#[tokio::test]
async fn a_custom_purpose_is_carried_by_the_operation() {
    with_operation_scope(
        CompressionOperation::with_purpose("op-2", "summarisation"),
        async {
            assert_eq!(current_request_purpose().as_deref(), Some("summarisation"));
            assert_eq!(current_compression_operation_id().as_deref(), Some("op-2"));
        },
    )
    .await;
}

#[tokio::test]
async fn stamping_annotates_an_outbound_request() {
    let mut request = ModelRequestConfig::new("gpt-4o");
    with_operation("op-7", async {
        stamp_current_operation(&mut request);
    })
    .await;
    assert_eq!(request.extra[REQUEST_PURPOSE_KEY], COMPACTION_PURPOSE);
    assert_eq!(request.extra[CONTEXT_OPERATION_ID_KEY], "op-7");

    // Outside a scope nothing is stamped.
    let mut untouched = ModelRequestConfig::new("gpt-4o");
    stamp_current_operation(&mut untouched);
    assert_eq!(untouched.extra, serde_json::Value::Null);

    // An existing extra map keeps its own keys.
    let mut seeded = ModelRequestConfig::new("gpt-4o").with_extra(serde_json::json!({"seed": 1}));
    CompressionOperation::new("op-8").apply_to(&mut seeded);
    assert_eq!(seeded.extra["seed"], 1);
    assert_eq!(seeded.extra[CONTEXT_OPERATION_ID_KEY], "op-8");
    assert_eq!(
        CompressionOperation::new("op-8").model_kwargs()[REQUEST_PURPOSE_KEY],
        COMPACTION_PURPOSE
    );
    assert_eq!(CompressionOperation::default().model_kwargs().len(), 0);
}
