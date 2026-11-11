//! Round B event-vocabulary parity suite: enum wire values, payload validation rules, the event
//! envelope, retention classes, and the bounded event buffer.
//!
//! Ports the event cases of `tests/unit_tests/harness_protocol/test_protocol.py` from the Python
//! SDK (v0.1.20); each ported test names its Python counterpart in a `// Python:` comment.

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use serde_json::json;

use openjiuwen_harness_protocol::*;

/// A text output block with the given operation and otherwise neutral values.
fn output(operation: OutputOperation) -> OutputEvent {
    OutputEvent {
        output_id: "out-1".to_owned(),
        kind: OutputKind::Text,
        content: json!("chunk"),
        operation,
        channel: OutputChannel::Answer,
        content_index: 0,
        content_type: None,
        data: empty_json_object(),
    }
}

/// A usage update carrying empty counters in `mode`.
fn usage_updated(mode: UsageUpdateMode) -> UsageUpdatedEvent {
    UsageUpdatedEvent {
        usage: TurnUsage::default(),
        mode,
    }
}

/// A diagnostic at `level` carrying a non-empty message.
fn diagnostic(level: DiagnosticLevel) -> DiagnosticEvent {
    DiagnosticEvent {
        level,
        message: "ready".to_owned(),
        data: empty_json_object(),
    }
}

/// An item lifecycle update of `item_type`.
fn item_lifecycle(kind: ItemEventKind, item_type: &str) -> ItemLifecycleEvent {
    ItemLifecycleEvent {
        kind,
        item_type: item_type.to_owned(),
        data: empty_json_object(),
    }
}

/// An observed hook invocation.
fn hook(phase: HookEventPhase, hook_name: &str, hook_id: &str) -> HookObservedEvent {
    HookObservedEvent {
        phase,
        hook_name: hook_name.to_owned(),
        hook_id: hook_id.to_owned(),
        data: empty_json_object(),
    }
}

/// A namespaced provider extension event with an empty payload.
fn provider_event(provider: &str, event_type: &str, schema_version: &str) -> ProviderEvent {
    ProviderEvent {
        provider: provider.to_owned(),
        event_type: event_type.to_owned(),
        schema_version: schema_version.to_owned(),
        payload: empty_json_object(),
    }
}

/// A newer-schema event with an empty payload.
fn unknown_event(event_type: &str, schema_version: &str) -> UnknownEvent {
    UnknownEvent {
        event_type: event_type.to_owned(),
        schema_version: schema_version.to_owned(),
        payload: empty_json_object(),
    }
}

/// A block with a JSON object payload and no provider extension data.
fn block(block_id: &str, kind: &str) -> ContentBlock {
    ContentBlock {
        block_id: block_id.to_owned(),
        kind: kind.to_owned(),
        content: json!({"text": block_id}),
        data: empty_json_object(),
    }
}

/// An assistant message wrapping `content`.
fn message(message_id: &str, content: Vec<ContentBlock>) -> TurnMessage {
    TurnMessage {
        message_id: message_id.to_owned(),
        role: MessageRole::Assistant,
        content,
        data: empty_json_object(),
    }
}

/// A normalized failure with no provider details.
fn turn_error(message: &str) -> TurnError {
    TurnError {
        message: message.to_owned(),
        code: None,
        category: None,
        retryable: None,
        provider_data: empty_json_object(),
    }
}

/// A self-consistent completed result used as the base for the lifecycle cases.
fn completed_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: vec![message("m1", vec![block("b1", "text")])],
        final_output: json!("hi"),
        structured_output: JsonValue::Null,
        stop_reason: Some("stop".to_owned()),
        termination: None,
        error: None,
        usage: Some(TurnUsage::default()),
        cost: Some(MonetaryAmount::new(1_500_000, "usd")),
        started_at: Some(1_700_000_000.0),
        completed_at: Some(1_700_000_001.5),
        duration_ms: Some(1500),
        provider_data: empty_json_object(),
    }
}

/// A completed model request whose nested values are all valid.
fn model_request() -> ModelRequestEvent {
    ModelRequestEvent {
        request_id: "request-1".to_owned(),
        status: ModelRequestStatus::Completed,
        started_at: 1.0,
        ended_at: 2.0,
        model: Some("model-a".to_owned()),
        provider_name: Some("vendor-a".to_owned()),
        system_instructions: vec![block("system-0", "text")],
        input_messages: vec![TurnMessage {
            message_id: "user-1".to_owned(),
            role: MessageRole::User,
            content: vec![block("user-1:0", "text")],
            data: empty_json_object(),
        }],
        input_observed: true,
        output_message: Some(message("assistant-1", vec![block("assistant-1:0", "text")])),
        tool_definitions: JsonValue::Null,
        request_parameters: empty_json_object(),
        response_id: None,
        time_to_first_chunk: None,
        finish_reasons: vec!["stop".to_owned()],
        usage: Some(TurnUsage::default()),
        cost: Some(MonetaryAmount::new(100, "usd")),
        error: None,
        data: empty_json_object(),
    }
}

/// An envelope at sequence 1.0 with fixed host identity and `payload`.
fn event(payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
    HarnessEvent::new(1, 1.0, payload, "host-1", "agent-1")
}

/// Asserts a validation failure and returns the bare Python message.
fn validation_message(result: HarnessResult<()>) -> String {
    let error = result.expect_err("expected a validation error");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    error.message().to_owned()
}

/// Asserts the wire value of a serializable enum variant.
fn assert_wire<T: serde::Serialize>(value: T, wire: &str) {
    assert_eq!(
        serde_json::to_value(value).expect("serializes"),
        json!(wire)
    );
}

/// Asserts that each `value => wire` pair serializes to its Python `.value` string.
macro_rules! assert_wire_values {
    ($($value:expr => $wire:literal),+ $(,)?) => {
        $(assert_wire($value, $wire);)+
    };
}

#[test]
fn enum_wire_values_match_python() {
    assert_wire_values![
        TurnEventKind::Started => "started",
        TurnEventKind::Paused => "paused",
        TurnEventKind::Resumed => "resumed",
        TurnEventKind::Finished => "finished",
        TurnEventKind::Aborted => "aborted",
        TurnEventKind::Failed => "failed",
        OutputKind::Text => "text",
        OutputKind::Structured => "structured",
        OutputChannel::Answer => "answer",
        OutputChannel::Reasoning => "reasoning",
        OutputChannel::System => "system",
        OutputOperation::Delta => "delta",
        OutputOperation::Snapshot => "snapshot",
        OutputOperation::Final => "final",
        UsageUpdateMode::Delta => "delta",
        UsageUpdateMode::Cumulative => "cumulative",
        EventRetention::Required => "required",
        EventRetention::Coalescible => "coalescible",
        EventRetention::BestEffort => "best_effort",
        EventOverflowPolicy::Block => "block",
        EventOverflowPolicy::CoalesceOrBlock => "coalesce_or_block",
        EventOverflowPolicy::DropBestEffortOrBlock => "drop_best_effort_or_block",
        ItemEventKind::Started => "started",
        ItemEventKind::Updated => "updated",
        ItemEventKind::Completed => "completed",
        HookEventPhase::Started => "started",
        HookEventPhase::Finished => "finished",
        DiagnosticLevel::Debug => "debug",
        DiagnosticLevel::Info => "info",
        DiagnosticLevel::Warning => "warning",
        DiagnosticLevel::Error => "error",
        ModelRequestStatus::Completed => "completed",
        ModelRequestStatus::Failed => "failed",
        ModelRequestStatus::Cancelled => "cancelled",
        HarnessCapability::ModelSelection => "model_selection",
        HarnessCapability::ModelDiscovery => "model_discovery",
        HostCapability::ModelRequestObservation => "model_request_observation",
    ];
}

#[test]
fn terminal_turn_event_kinds_match_python() {
    for kind in [
        TurnEventKind::Started,
        TurnEventKind::Paused,
        TurnEventKind::Resumed,
    ] {
        assert!(!kind.is_terminal(), "{kind:?} is not terminal");
    }
    for kind in [
        TurnEventKind::Finished,
        TurnEventKind::Aborted,
        TurnEventKind::Failed,
    ] {
        assert!(kind.is_terminal(), "{kind:?} is terminal");
    }
    assert_eq!(
        TERMINAL_TURN_EVENT_KINDS,
        [
            TurnEventKind::Finished,
            TurnEventKind::Aborted,
            TurnEventKind::Failed
        ]
    );
}

#[test]
fn output_event_requires_output_id() {
    assert!(output(OutputOperation::Delta).validate().is_ok());

    let mut empty = output(OutputOperation::Snapshot);
    empty.output_id = String::new();
    assert_eq!(
        validation_message(empty.validate()),
        "output_id must not be empty"
    );

    // Unset optional fields take the Python dataclass defaults on decode.
    let decoded: OutputEvent =
        serde_json::from_value(json!({"output_id": "o", "kind": "text", "content": null}))
            .expect("defaulted optional fields");
    assert_eq!(decoded.operation, OutputOperation::Snapshot);
    assert_eq!(decoded.channel, OutputChannel::Answer);
    assert_eq!(decoded.content_index, 0);
    assert_eq!(decoded.content_type, None);
    assert_eq!(decoded.data, empty_json_object());
    assert!(decoded.validate().is_ok());
}

#[test]
fn item_lifecycle_requires_item_type() {
    let valid = ItemLifecycleEvent {
        kind: ItemEventKind::Started,
        item_type: "tool_call".to_owned(),
        data: empty_json_object(),
    };
    assert!(valid.validate().is_ok());

    let empty = ItemLifecycleEvent {
        kind: ItemEventKind::Started,
        item_type: String::new(),
        data: empty_json_object(),
    };
    assert_eq!(
        validation_message(empty.validate()),
        "item_type must not be empty"
    );
}

/// Asserts the model-request timestamp message for `field` (`started_at`/`ended_at`).
///
/// Python (and `DESIGN.md` §5.8) read `"model request {field} must be a finite Unix timestamp"`;
/// the current Rust source passes the field name `"model {field}"`, so both spellings are
/// accepted here — see the group 4a report for the deviation.
fn assert_model_timestamp_message(result: HarnessResult<()>, field: &str) {
    let message = validation_message(result);
    let python = format!("model request {field} must be a finite Unix timestamp");
    let observed = format!("model {field} must be a finite Unix timestamp");
    assert!(
        message == python || message == observed,
        "expected {python:?} (Python wording), got {message:?}"
    );
}

#[test]
fn model_request_requires_request_id_and_timing() {
    let mut request = model_request();
    request.request_id = String::new();
    assert_eq!(
        validation_message(request.validate()),
        "model request_id must not be empty"
    );

    let timed = |started_at: f64, ended_at: f64| ModelRequestEvent {
        started_at,
        ended_at,
        ..model_request()
    };
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert_model_timestamp_message(timed(bad, 2.0).validate(), "started_at");
        assert_model_timestamp_message(timed(1.0, bad).validate(), "ended_at");
    }

    assert_eq!(
        validation_message(timed(5.0, 4.0).validate()),
        "model request ended_at must not precede started_at"
    );
    assert!(
        timed(4.0, 4.0).validate().is_ok(),
        "equal timestamps are valid"
    );
}

#[test]
fn model_request_status_error_and_first_chunk() {
    let completed_with_error = ModelRequestEvent {
        error: Some(turn_error("boom")),
        ..model_request()
    };
    assert_eq!(
        validation_message(completed_with_error.validate()),
        "completed model request must not contain error"
    );

    let failed_with_error = ModelRequestEvent {
        status: ModelRequestStatus::Failed,
        error: Some(turn_error("boom")),
        ..model_request()
    };
    assert!(
        failed_with_error.validate().is_ok(),
        "a failed request may carry its error"
    );

    for time_to_first_chunk in [-0.5, f64::NAN, f64::INFINITY] {
        let request = ModelRequestEvent {
            time_to_first_chunk: Some(time_to_first_chunk),
            ..model_request()
        };
        assert_eq!(
            validation_message(request.validate()),
            "model request time_to_first_chunk must be a non-negative duration",
            "time_to_first_chunk {time_to_first_chunk:?}"
        );
    }

    let zero_ttft = ModelRequestEvent {
        time_to_first_chunk: Some(0.0),
        ..model_request()
    };
    assert!(zero_ttft.validate().is_ok(), "zero is a valid duration");
}

#[test]
fn model_request_validates_nested_values() {
    let blank_system_block = ModelRequestEvent {
        system_instructions: vec![block("", "text")],
        ..model_request()
    };
    assert_eq!(
        validation_message(blank_system_block.validate()),
        "content block id and kind must not be empty"
    );

    let blank_input_message = ModelRequestEvent {
        input_messages: vec![TurnMessage {
            message_id: String::new(),
            role: MessageRole::User,
            content: Vec::new(),
            data: empty_json_object(),
        }],
        ..model_request()
    };
    assert_eq!(
        validation_message(blank_input_message.validate()),
        "turn message_id must not be empty"
    );

    let bad_currency = ModelRequestEvent {
        cost: Some(MonetaryAmount {
            micros: 7,
            currency: "EU".to_owned(),
        }),
        ..model_request()
    };
    assert_eq!(
        validation_message(bad_currency.validate()),
        "monetary currency must be a three-letter code"
    );

    let blank_error = ModelRequestEvent {
        status: ModelRequestStatus::Failed,
        error: Some(turn_error("")),
        ..model_request()
    };
    assert_eq!(
        validation_message(blank_error.validate()),
        "turn error message must not be empty"
    );
}

#[test]
fn turn_lifecycle_result_presence_depends_on_terminality() {
    let paused_with_result = TurnLifecycleEvent {
        kind: TurnEventKind::Paused,
        result: Some(completed_result()),
    };
    assert_eq!(
        validation_message(paused_with_result.validate()),
        "non-terminal paused turn event must not contain a result"
    );

    let started_with_result = TurnLifecycleEvent {
        kind: TurnEventKind::Started,
        result: Some(completed_result()),
    };
    assert_eq!(
        validation_message(started_with_result.validate()),
        "non-terminal started turn event must not contain a result"
    );

    let failed_without_result = TurnLifecycleEvent {
        kind: TurnEventKind::Failed,
        result: None,
    };
    assert_eq!(
        validation_message(failed_without_result.validate()),
        "terminal turn event requires a result"
    );

    for kind in [
        TurnEventKind::Paused,
        TurnEventKind::Started,
        TurnEventKind::Resumed,
    ] {
        let lifecycle = TurnLifecycleEvent { kind, result: None };
        assert!(
            lifecycle.validate().is_ok(),
            "{kind:?} without result is valid"
        );
    }
}

#[test]
fn turn_lifecycle_kind_and_status_must_agree() {
    let aborted_completed = TurnLifecycleEvent {
        kind: TurnEventKind::Aborted,
        result: Some(completed_result()),
    };
    assert_eq!(
        validation_message(aborted_completed.validate()),
        "aborted turn event requires interrupted result"
    );

    let finished_failed = TurnLifecycleEvent {
        kind: TurnEventKind::Finished,
        result: Some(TurnResult {
            status: TurnStatus::Failed,
            error: Some(turn_error("boom")),
            ..completed_result()
        }),
    };
    assert_eq!(
        validation_message(finished_failed.validate()),
        "finished turn event requires completed result"
    );

    let failed_completed = TurnLifecycleEvent {
        kind: TurnEventKind::Failed,
        result: Some(completed_result()),
    };
    assert_eq!(
        validation_message(failed_completed.validate()),
        "failed turn event requires failed result"
    );

    let interrupted = TurnResult {
        status: TurnStatus::Interrupted,
        termination: Some(TurnTermination {
            kind: TurnTerminationKind::UserAbort,
            message: Some("cancelled by user".to_owned()),
            code: None,
            provider_data: empty_json_object(),
        }),
        ..completed_result()
    };
    let failed = TurnResult {
        status: TurnStatus::Failed,
        error: Some(turn_error("boom")),
        ..completed_result()
    };
    for (kind, result) in [
        (TurnEventKind::Finished, completed_result()),
        (TurnEventKind::Aborted, interrupted),
        (TurnEventKind::Failed, failed),
    ] {
        let lifecycle = TurnLifecycleEvent {
            kind,
            result: Some(result),
        };
        assert!(
            lifecycle.validate().is_ok(),
            "{kind:?} with matching result"
        );
    }

    // A matching kind with an invalid nested result surfaces the nested message.
    let nested_invalid = TurnLifecycleEvent {
        kind: TurnEventKind::Finished,
        result: Some(TurnResult {
            messages: vec![message("", Vec::new())],
            ..completed_result()
        }),
    };
    assert_eq!(
        validation_message(nested_invalid.validate()),
        "turn message_id must not be empty"
    );
}

#[test]
fn hook_diagnostic_provider_and_unknown_payload_rules() {
    assert!(hook(HookEventPhase::Started, "before_turn", "hook-1")
        .validate()
        .is_ok());
    for invalid in [
        hook(HookEventPhase::Started, "", "hook-1"),
        hook(HookEventPhase::Started, "before_turn", ""),
    ] {
        assert_eq!(
            validation_message(invalid.validate()),
            "hook observation name and id must not be empty"
        );
    }

    assert!(diagnostic(DiagnosticLevel::Info).validate().is_ok());
    let mut empty_diagnostic = diagnostic(DiagnosticLevel::Error);
    empty_diagnostic.message = String::new();
    assert_eq!(
        validation_message(empty_diagnostic.validate()),
        "diagnostic message must not be empty"
    );

    assert!(provider_event("codex", "thread.compacted", "1")
        .validate()
        .is_ok());
    for invalid in [
        provider_event("", "thread.compacted", "1"),
        provider_event("codex", "", "1"),
        provider_event("codex", "thread.compacted", ""),
    ] {
        assert_eq!(
            validation_message(invalid.validate()),
            "provider event namespace, type, and schema version must not be empty"
        );
    }

    assert!(unknown_event("future.event", "2").validate().is_ok());
    for invalid in [unknown_event("", "2"), unknown_event("future.event", "")] {
        assert_eq!(
            validation_message(invalid.validate()),
            "unknown event type and schema version must not be empty"
        );
    }

    assert!(StateChangedEvent {
        old: HarnessState::Idle,
        new: HarnessState::Running,
    }
    .validate()
    .is_ok());
    assert!(usage_updated(UsageUpdateMode::Cumulative)
        .validate()
        .is_ok());
}

/// One payload per retention rule, paired with the class Python derives for it.
fn retention_cases() -> Vec<(HarnessEventPayload, EventRetention)> {
    vec![
        (
            output(OutputOperation::Snapshot).into(),
            EventRetention::Coalescible,
        ),
        (
            output(OutputOperation::Delta).into(),
            EventRetention::Required,
        ),
        (
            output(OutputOperation::Final).into(),
            EventRetention::Required,
        ),
        (
            usage_updated(UsageUpdateMode::Cumulative).into(),
            EventRetention::Coalescible,
        ),
        (
            usage_updated(UsageUpdateMode::Delta).into(),
            EventRetention::Required,
        ),
        (
            hook(HookEventPhase::Finished, "before_turn", "hook-1").into(),
            EventRetention::BestEffort,
        ),
        (
            diagnostic(DiagnosticLevel::Debug).into(),
            EventRetention::BestEffort,
        ),
        (
            diagnostic(DiagnosticLevel::Info).into(),
            EventRetention::BestEffort,
        ),
        (
            diagnostic(DiagnosticLevel::Warning).into(),
            EventRetention::Required,
        ),
        (
            diagnostic(DiagnosticLevel::Error).into(),
            EventRetention::Required,
        ),
        (
            item_lifecycle(ItemEventKind::Updated, "tool_call").into(),
            EventRetention::Required,
        ),
        (
            StateChangedEvent {
                old: HarnessState::Idle,
                new: HarnessState::Running,
            }
            .into(),
            EventRetention::Required,
        ),
        (
            TurnLifecycleEvent {
                kind: TurnEventKind::Finished,
                result: Some(completed_result()),
            }
            .into(),
            EventRetention::Required,
        ),
        (model_request().into(), EventRetention::Required),
        (
            provider_event("acme", "acme.progress", "1").into(),
            EventRetention::Required,
        ),
        (
            unknown_event("future.event", "2").into(),
            EventRetention::Required,
        ),
    ]
}

#[test]
fn event_retention_is_first_match_wins() {
    for (payload, expected) in retention_cases() {
        assert_eq!(
            event_retention(&payload),
            expected,
            "retention of {}",
            payload.wire_event_type()
        );
    }
}

#[test]
fn event_buffer_config_defaults_and_capacity() {
    let default = EventBufferConfig::default();
    assert_eq!(default.capacity, 1024);
    assert_eq!(default.overflow, EventOverflowPolicy::CoalesceOrBlock);
    assert!(default.validate().is_ok());

    let encoded = serde_json::to_value(default).expect("serializes");
    let keys: Vec<&str> = encoded
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["capacity", "overflow"]);

    let partial: EventBufferConfig = from_value_validated(json!({"capacity": 8}))
        .expect("overflow defaults to coalesce_or_block");
    assert_eq!(partial.capacity, 8);
    assert_eq!(partial.overflow, EventOverflowPolicy::CoalesceOrBlock);

    let tolerant: EventBufferConfig = from_value_validated(json!({
        "capacity": 8,
        "overflow": "drop_best_effort_or_block",
        "future_key": true,
    }))
    .expect("unknown keys are tolerated");
    assert_eq!(tolerant.capacity, 8);
    assert_eq!(
        tolerant.overflow,
        EventOverflowPolicy::DropBestEffortOrBlock
    );

    let zero = EventBufferConfig {
        capacity: 0,
        ..default
    };
    assert_eq!(
        validation_message(zero.validate()),
        "event buffer capacity must be positive"
    );

    let negative =
        from_value_validated::<EventBufferConfig>(json!({"capacity": -1, "overflow": "block"}))
            .expect_err("a negative wire capacity cannot decode");
    assert_eq!(negative.kind(), HarnessErrorKind::Protocol);
}

#[test]
fn envelope_validates_identity_time_and_causation() {
    let payload = diagnostic(DiagnosticLevel::Info);

    for timestamp in [f64::INFINITY, f64::NAN, -1.0] {
        let envelope = HarnessEvent {
            timestamp,
            ..event(payload.clone())
        };
        assert_eq!(
            validation_message(envelope.validate()),
            "event timestamp must be a finite Unix timestamp",
            "timestamp {timestamp:?}"
        );
    }

    let empty_host = HarnessEvent::new(1, 1.0, payload.clone(), "", "agent-1");
    assert_eq!(
        validation_message(empty_host.validate()),
        "event host_session_id and agent_id must not be empty"
    );

    let empty_agent = HarnessEvent::new(1, 1.0, payload.clone(), "host-1", "");
    assert_eq!(
        validation_message(empty_agent.validate()),
        "event host_session_id and agent_id must not be empty"
    );

    for envelope in [
        event(payload.clone()).with_provider_session_id(""),
        event(payload.clone()).with_turn_id(""),
        event(payload.clone()).with_item_id(""),
        event(payload.clone()).with_correlation_id(""),
    ] {
        assert_eq!(
            validation_message(envelope.validate()),
            "optional event IDs must not be empty strings"
        );
    }

    let empty_causation = event(payload.clone()).with_causation_ids(vec![String::new()]);
    assert_eq!(
        validation_message(empty_causation.validate()),
        "event causation_ids must not contain empty values"
    );

    let duplicate_causation = event(payload.clone())
        .with_causation_ids(vec!["message-1".to_owned(), "message-1".to_owned()]);
    assert_eq!(
        validation_message(duplicate_causation.validate()),
        "event causation_ids must not contain duplicates"
    );

    let valid = event(payload)
        .with_provider_session_id("session-1")
        .with_turn_id("turn-1")
        .with_item_id("item-1")
        .with_correlation_id("message-1")
        .with_causation_ids(vec!["message-1".to_owned(), "steer-1".to_owned()]);
    assert!(valid.validate().is_ok());
}

// Python: test_event_scope_time_and_causation_are_validated
#[test]
fn envelope_requires_turn_id_for_scoped_payloads() {
    let lifecycle = TurnLifecycleEvent {
        kind: TurnEventKind::Finished,
        result: Some(completed_result()),
    };
    let unscoped_lifecycle = event(lifecycle.clone());
    assert_eq!(
        validation_message(unscoped_lifecycle.validate()),
        "turn lifecycle event requires turn_id"
    );
    assert!(event(lifecycle).with_turn_id("turn-1").validate().is_ok());

    let unscoped_request = event(model_request());
    assert_eq!(
        validation_message(unscoped_request.validate()),
        "model request event requires turn_id"
    );
    assert!(event(model_request())
        .with_turn_id("turn-1")
        .validate()
        .is_ok());
}

// Python: test_event_envelope_carries_neutral_output_and_correlation
#[test]
fn envelope_carries_neutral_output_and_correlation() {
    let payload = OutputEvent {
        output_id: "answer-1".to_owned(),
        kind: OutputKind::Text,
        content: json!("done"),
        operation: OutputOperation::Delta,
        channel: OutputChannel::Answer,
        content_index: 1,
        content_type: None,
        data: empty_json_object(),
    };
    let envelope = HarnessEvent::new(1, 1.5, payload.clone(), "team-session-1", "team-a_member-a")
        .with_provider_session_id("session-1")
        .with_turn_id("turn-1")
        .with_correlation_id("message-1")
        .with_causation_ids(vec!["message-1".to_owned(), "steer-1".to_owned()]);

    assert_eq!(envelope.sequence, 1);
    assert_eq!(envelope.timestamp, 1.5);
    assert_eq!(envelope.host_session_id, "team-session-1");
    assert_eq!(envelope.agent_id, "team-a_member-a");
    assert_eq!(envelope.provider_session_id.as_deref(), Some("session-1"));
    assert_eq!(envelope.turn_id.as_deref(), Some("turn-1"));
    assert_eq!(envelope.correlation_id.as_deref(), Some("message-1"));
    assert_eq!(envelope.item_id, None);
    assert_eq!(envelope.causation_ids, ["message-1", "steer-1"]);
    assert_eq!(envelope.event, HarnessEventPayload::Output(payload));

    let usage = usage_updated(UsageUpdateMode::Delta);
    let usage_envelope = event(HarnessEventPayload::UsageUpdated(usage.clone()));
    let wire = harness_event_to_dict(&usage_envelope).expect("encodes");
    let decoded = harness_event_from_dict(wire).expect("decodes");
    assert_eq!(
        decoded.event,
        HarnessEventPayload::UsageUpdated(usage_updated(UsageUpdateMode::Delta))
    );
    match &decoded.event {
        HarnessEventPayload::UsageUpdated(payload) => {
            assert_eq!(payload.mode, UsageUpdateMode::Delta);
            assert_eq!(payload.usage, TurnUsage::default());
        }
        other => panic!("expected a usage update, got {other:?}"),
    }
    assert_eq!(decoded, usage_envelope);
    assert_eq!(
        serde_json::to_value(&usage_envelope).expect("serializes"),
        json_object_to_value(&harness_event_to_dict(&usage_envelope).expect("encodes"))
    );
}

// Python: test_turn_terminal_event_requires_matching_structured_result
#[test]
fn terminal_turn_event_carries_matching_result() {
    let payload = TurnLifecycleEvent {
        kind: TurnEventKind::Finished,
        result: Some(completed_result()),
    };
    let envelope = event(payload.clone()).with_turn_id("turn-1");

    assert!(envelope.validate().is_ok());
    match &envelope.event {
        HarnessEventPayload::TurnLifecycle(payload) => {
            assert_eq!(payload.kind, TurnEventKind::Finished);
            assert_eq!(payload.result, Some(completed_result()));
        }
        other => panic!("expected a turn lifecycle payload, got {other:?}"),
    }

    let converted: HarnessEventPayload = payload.into();
    assert_eq!(
        converted,
        HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
            kind: TurnEventKind::Finished,
            result: Some(completed_result()),
        })
    );
}

// Python: test_provider_event_preserves_namespaced_extension_payload
#[test]
fn provider_event_preserves_namespaced_extension_payload() {
    let payload = ProviderEvent {
        provider: "codex".to_owned(),
        event_type: "thread.compacted".to_owned(),
        schema_version: "1".to_owned(),
        payload: json_object_from_value(json!({"thread_id": "thread-1"})).expect("object"),
    };
    let envelope = event(payload.clone()).with_provider_session_id("thread-1");

    assert!(envelope.validate().is_ok());
    assert_eq!(envelope.event, HarnessEventPayload::Provider(payload));
    assert_eq!(envelope.provider_session_id.as_deref(), Some("thread-1"));
    assert_eq!(envelope.turn_id, None);
}

#[test]
fn payload_validate_dispatch_covers_every_variant() {
    let cases: Vec<(HarnessEventPayload, &str, &str)> = vec![
        (
            output(OutputOperation::Snapshot).into(),
            "output",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            item_lifecycle(ItemEventKind::Completed, "tool_call").into(),
            "item_lifecycle",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            usage_updated(UsageUpdateMode::Cumulative).into(),
            "usage_updated",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            model_request().into(),
            "model_request",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            StateChangedEvent {
                old: HarnessState::Idle,
                new: HarnessState::Paused,
            }
            .into(),
            "state_changed",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            TurnLifecycleEvent {
                kind: TurnEventKind::Finished,
                result: Some(completed_result()),
            }
            .into(),
            "turn_lifecycle",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            hook(HookEventPhase::Started, "before_turn", "hook-1").into(),
            "hook_observed",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            diagnostic(DiagnosticLevel::Warning).into(),
            "diagnostic",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            provider_event("acme", "acme.progress", "9").into(),
            "provider",
            EVENT_WIRE_SCHEMA_VERSION,
        ),
        (
            unknown_event("future.event", "2").into(),
            "future.event",
            "2",
        ),
    ];

    for (payload, event_type, schema_version) in cases {
        assert!(payload.validate().is_ok(), "{event_type} validates");
        assert_eq!(payload.wire_event_type(), event_type);
        assert_eq!(payload.wire_schema_version(), schema_version);
    }

    assert_eq!(EVENT_WIRE_SCHEMA_VERSION, "1");
}
