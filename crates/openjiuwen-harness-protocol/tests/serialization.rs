//! Round B event wire-codec parity suite: the twelve-key envelope, per-variant payload key sets,
//! round-trip equality, the schema-version gate, and the Python-style decode errors.
//!
//! Python counterparts live in `tests/unit_tests/harness_protocol/test_protocol.py` of the Python
//! SDK (v0.1.20); each ported case names its counterpart in a `// Python:` comment. The normative
//! contract for every shape asserted here is `DESIGN.md` §6 (wire codec) with §5.8 (payloads),
//! §5.9 (envelope), and §10 (test plan).

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use serde_json::json;

use openjiuwen_harness_protocol::*;

/// The twelve envelope keys in DESIGN §6 order.
const ENVELOPE_KEYS: [&str; 12] = [
    "schema_version",
    "event_type",
    "sequence",
    "timestamp",
    "host_session_id",
    "agent_id",
    "provider_session_id",
    "turn_id",
    "item_id",
    "correlation_id",
    "causation_ids",
    "payload",
];

/// The nineteen `model_request` payload keys in DESIGN §6 order.
const MODEL_REQUEST_PAYLOAD_KEYS: [&str; 19] = [
    "request_id",
    "status",
    "started_at",
    "ended_at",
    "model",
    "provider_name",
    "system_instructions",
    "input_messages",
    "input_observed",
    "output_message",
    "tool_definitions",
    "request_parameters",
    "response_id",
    "time_to_first_chunk",
    "finish_reasons",
    "usage",
    "cost",
    "error",
    "data",
];

/// The six `TurnUsage` keys in DESIGN §6 order.
const USAGE_KEYS: [&str; 6] = [
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "provider_data",
];

/// Converts a JSON value into the crate's deterministic object type.
fn wire(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the value is a JSON object")
}

/// Encodes `event` into the wire JSON value (object keys sorted by `BTreeMap`).
///
/// Order-sensitive assertions must go through [`harness_event_to_json_string`] instead, because
/// [`JsonObject`] is a sorted map.
fn to_wire(event: &HarnessEvent) -> JsonValue {
    json_object_to_value(&harness_event_to_dict(event).expect("the event encodes to a wire object"))
}

/// Returns the payload key order of `event`, read from the JSON text (insertion order preserved).
fn payload_keys(event: &HarnessEvent) -> Vec<String> {
    let text = harness_event_to_json_string(event).expect("the event encodes as JSON text");
    let parsed: JsonValue = serde_json::from_str(&text).expect("the encoded event is valid JSON");
    parsed["payload"]
        .as_object()
        .expect("the payload is a JSON object")
        .keys()
        .cloned()
        .collect()
}

/// Decodes a wire value through its JSON text form, exercising the public text entry point.
fn decode_wire(value: &JsonValue) -> HarnessResult<HarnessEvent> {
    harness_event_from_json_str(&value.to_string())
}

/// Returns the bare message of an error result.
fn error_message<T: std::fmt::Debug>(result: HarnessResult<T>) -> String {
    result.expect_err("expected an error").message().to_owned()
}

/// Asserts a [`HarnessErrorKind::Protocol`] error whose message contains `detail`.
fn protocol_detail<T: std::fmt::Debug>(result: HarnessResult<T>, detail: &str) {
    let error = result.expect_err("expected a protocol error");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error.message().contains(detail),
        "expected detail {detail:?} in {:?}",
        error.message()
    );
}

/// Asserts a [`HarnessErrorKind::Protocol`] error and returns its exact bare message.
fn protocol_message<T: std::fmt::Debug>(result: HarnessResult<T>) -> String {
    let error = result.expect_err("expected a protocol error");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    error.message().to_owned()
}

/// Asserts a [`HarnessErrorKind::Validation`] error and returns its exact bare message.
fn validation_message<T: std::fmt::Debug>(result: HarnessResult<T>) -> String {
    let error = result.expect_err("expected a validation error");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    error.message().to_owned()
}

/// A plain final/structured output event with no optional envelope ids.
fn output_event() -> HarnessEvent {
    HarnessEvent::new(
        1,
        1.0,
        OutputEvent {
            output_id: "answer-1".to_owned(),
            kind: OutputKind::Structured,
            content: json!({"answer": [1, 2]}),
            operation: OutputOperation::Final,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: None,
            data: empty_json_object(),
        },
        "host-1",
        "agent-1",
    )
}

/// The Python `_model_request_event()` fixture, field for field.
fn model_request_event() -> ModelRequestEvent {
    ModelRequestEvent {
        request_id: "request-1".to_owned(),
        status: ModelRequestStatus::Completed,
        started_at: 10.0,
        ended_at: 12.5,
        model: Some("model-a".to_owned()),
        provider_name: Some("vendor-a".to_owned()),
        system_instructions: vec![ContentBlock {
            block_id: "system-0".to_owned(),
            kind: "text".to_owned(),
            content: json!("be brief"),
            data: empty_json_object(),
        }],
        input_messages: vec![TurnMessage {
            message_id: "user-1".to_owned(),
            role: MessageRole::User,
            content: vec![ContentBlock {
                block_id: "user-1:0".to_owned(),
                kind: "text".to_owned(),
                content: json!("list files"),
                data: empty_json_object(),
            }],
            data: wire(json!({"origin": "external_user"})),
        }],
        input_observed: true,
        output_message: Some(TurnMessage {
            message_id: "assistant-1".to_owned(),
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock {
                    block_id: "assistant-1:0".to_owned(),
                    kind: "reasoning".to_owned(),
                    content: json!("look first"),
                    data: empty_json_object(),
                },
                ContentBlock {
                    block_id: "assistant-1:1".to_owned(),
                    kind: "tool_call".to_owned(),
                    content: json!({"id": "call-1", "name": "ls", "arguments": {"path": "."}}),
                    data: empty_json_object(),
                },
            ],
            data: empty_json_object(),
        }),
        tool_definitions: json!([{"name": "ls", "input_schema": {"type": "object"}}]),
        request_parameters: empty_json_object(),
        response_id: None,
        time_to_first_chunk: None,
        finish_reasons: Vec::new(),
        usage: Some(TurnUsage {
            input_tokens: Some(30),
            output_tokens: Some(5),
            cached_input_tokens: Some(10),
            ..TurnUsage::default()
        }),
        cost: None,
        error: None,
        data: wire(json!({"vendor-a": {"ttft_ms": 120}})),
    }
}

/// The Python model-request event envelope (sequence 3, timestamp 12.5, turn `turn-1`).
fn model_request_envelope() -> HarnessEvent {
    HarnessEvent::new(3, 12.5, model_request_event(), "team-session-1", "member-a")
        .with_turn_id("turn-1")
}

/// A completed turn result with one assistant text block, usage, cost, and timings.
fn completed_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: vec![TurnMessage {
            message_id: "assistant-1".to_owned(),
            role: MessageRole::Assistant,
            content: vec![ContentBlock {
                block_id: "assistant-1:0".to_owned(),
                kind: "text".to_owned(),
                content: json!("done"),
                data: empty_json_object(),
            }],
            data: empty_json_object(),
        }],
        final_output: json!("done"),
        structured_output: json!({"answer": 42}),
        stop_reason: Some("stop".to_owned()),
        termination: None,
        error: None,
        usage: Some(TurnUsage {
            input_tokens: Some(12),
            output_tokens: Some(4),
            ..TurnUsage::default()
        }),
        cost: Some(MonetaryAmount::new(1_500_000, "usd")),
        started_at: Some(10.0),
        completed_at: Some(12.5),
        duration_ms: Some(2_500),
        provider_data: wire(json!({"vendor-a": {"phase": "done"}})),
    }
}

/// An interrupted turn result carrying the mandatory termination cause.
fn interrupted_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Interrupted,
        termination: Some(TurnTermination {
            kind: TurnTerminationKind::UserAbort,
            message: Some("cancelled by user".to_owned()),
            code: None,
            provider_data: empty_json_object(),
        }),
        ..completed_result()
    }
}

/// A finished turn lifecycle event with a completed result and a `turn_id`.
fn turn_lifecycle_event() -> HarnessEvent {
    HarnessEvent::new(
        2,
        2.0,
        TurnLifecycleEvent {
            kind: TurnEventKind::Finished,
            result: Some(completed_result()),
        },
        "team-session-1",
        "team-a_member-a",
    )
    .with_turn_id("turn-1")
}

/// The Python provider-extension fixture (`codex` / `thread.compacted` / schema `"2"`).
fn provider_event() -> HarnessEvent {
    HarnessEvent::new(
        4,
        4.0,
        ProviderEvent {
            provider: "codex".to_owned(),
            event_type: "thread.compacted".to_owned(),
            schema_version: "2".to_owned(),
            payload: wire(json!({"thread_id": "thread-1"})),
        },
        "team-session-1",
        "team-a_member-a",
    )
    .with_provider_session_id("thread-1")
}

/// A verbatim unknown event from a newer schema.
fn unknown_event() -> HarnessEvent {
    HarnessEvent::new(
        9,
        9.0,
        UnknownEvent {
            event_type: "vendor.thing".to_owned(),
            schema_version: "3".to_owned(),
            payload: wire(json!({"a": 1})),
        },
        "team-session-1",
        "member-a",
    )
}

#[test]
fn output_event_round_trips_and_preserves_unknown_events() {
    // Python: test_event_json_codec_round_trips_and_preserves_unknown_events
    let event = HarnessEvent::new(
        7,
        7.5,
        OutputEvent {
            output_id: "answer-1".to_owned(),
            kind: OutputKind::Structured,
            content: json!({"answer": [1, 2]}),
            operation: OutputOperation::Final,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: None,
            data: empty_json_object(),
        },
        "team-session-1",
        "team-a_member-a",
    )
    .with_provider_session_id("provider-session-1")
    .with_turn_id("turn-1")
    .with_correlation_id("trace-1")
    .with_causation_ids(vec!["message-1".to_owned()]);

    let dict = harness_event_to_dict(&event).expect("the event encodes to a wire object");
    let text = serde_json::to_string(&dict).expect("the wire object is JSON serializable");
    assert!(
        text.starts_with('{'),
        "json.dumps-equivalent text is an object"
    );
    assert_eq!(dict["host_session_id"], json!("team-session-1"));
    assert_eq!(dict["agent_id"], json!("team-a_member-a"));
    assert_eq!(dict["provider_session_id"], json!("provider-session-1"));
    assert_eq!(dict["event_type"], json!("output"));
    assert_eq!(dict["schema_version"], json!("1"));
    for legacy in ["team_session_id", "member_agent_id", "session_id"] {
        assert!(
            !dict.contains_key(legacy),
            "legacy key {legacy} must be absent"
        );
    }
    assert_eq!(
        harness_event_from_dict(dict.clone()).expect("the wire object round trips"),
        event
    );

    // An unknown event type under schema "1" becomes an UnknownEvent that keeps the payload.
    let mut future = dict;
    future.insert("event_type".to_owned(), json!("future_event"));
    future.insert("payload".to_owned(), json!({"new_field": [1, 2, 3]}));
    let decoded = harness_event_from_dict(future).expect("an unknown event type decodes");
    let HarnessEventPayload::Unknown(unknown) = &decoded.event else {
        panic!("an unknown event_type must decode as UnknownEvent");
    };
    assert_eq!(unknown.event_type, "future_event");
    assert_eq!(unknown.schema_version, EVENT_WIRE_SCHEMA_VERSION);
    assert_eq!(unknown.payload, wire(json!({"new_field": [1, 2, 3]})));
    let re_encoded = harness_event_to_dict(&decoded).expect("the unknown event re-encodes");
    assert_eq!(re_encoded["event_type"], json!("future_event"));
    assert_eq!(re_encoded["payload"], json!({"new_field": [1, 2, 3]}));

    // A delta text output also survives the dict round trip unchanged.
    let delta = HarnessEvent::new(
        8,
        8.0,
        OutputEvent {
            output_id: "answer-1".to_owned(),
            kind: OutputKind::Text,
            content: json!("more"),
            operation: OutputOperation::Delta,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: Some("text/plain".to_owned()),
            data: empty_json_object(),
        },
        "team-session-1",
        "team-a_member-a",
    )
    .with_turn_id("turn-1");
    assert_eq!(
        harness_event_from_dict(harness_event_to_dict(&delta).expect("encodes")).expect("decodes"),
        delta
    );
}

#[test]
fn model_request_event_round_trips_like_the_python_fixture() {
    // Python: test_model_request_event_round_trips_and_is_required_retention (fixture `_model_request_event`)
    let event = model_request_envelope();

    let text = serde_json::to_string(&to_wire(&event)).expect("the wire value serializes");
    assert!(text.starts_with('{'));
    assert_eq!(
        harness_event_from_dict(harness_event_to_dict(&event).expect("encodes")).expect("decodes"),
        event
    );
    assert_eq!(event_retention(&event.event), EventRetention::Required);
    assert_eq!(
        HostCapability::ModelRequestObservation.as_str(),
        "model_request_observation"
    );

    let encoded: JsonValue =
        serde_json::from_str(&harness_event_to_json_string(&event).expect("encodes as text"))
            .expect("valid JSON");
    assert_eq!(encoded["event_type"], json!("model_request"));
    let payload = encoded["payload"]
        .as_object()
        .expect("payload is an object");
    let keys: Vec<&str> = payload.keys().map(String::as_str).collect();
    assert_eq!(keys, MODEL_REQUEST_PAYLOAD_KEYS);
    assert_eq!(payload["request_parameters"], json!({}));
    assert_eq!(payload["response_id"], JsonValue::Null);
    assert_eq!(payload["time_to_first_chunk"], JsonValue::Null);
    assert_eq!(payload["finish_reasons"], json!([]));
    assert_eq!(payload["cost"], JsonValue::Null);
    assert_eq!(payload["error"], JsonValue::Null);
    assert_eq!(
        payload["input_messages"][0]["data"],
        json!({"origin": "external_user"})
    );
    assert_eq!(
        payload["usage"],
        json!({
            "input_tokens": 30,
            "output_tokens": 5,
            "cached_input_tokens": 10,
            "reasoning_output_tokens": null,
            "total_tokens": null,
            "provider_data": {},
        })
    );
}

#[test]
fn usage_item_state_hook_and_diagnostic_payload_keys() {
    // Python: test_event_envelope_carries_neutral_output_and_correlation (usage mode) plus DESIGN §6 key table
    let item = HarnessEvent::new(
        1,
        1.0,
        ItemLifecycleEvent {
            kind: ItemEventKind::Updated,
            item_type: "tool_call".to_owned(),
            data: wire(json!({"index": 0})),
        },
        "host-1",
        "agent-1",
    );
    assert_eq!(payload_keys(&item), ["kind", "item_type", "data"]);

    let usage = HarnessEvent::new(
        2,
        2.0,
        UsageUpdatedEvent {
            usage: TurnUsage {
                input_tokens: Some(4),
                ..TurnUsage::default()
            },
            mode: UsageUpdateMode::Delta,
        },
        "host-1",
        "agent-1",
    );
    assert_eq!(payload_keys(&usage), ["usage", "mode"]);
    let usage_text = harness_event_to_json_string(&usage).expect("encodes as text");
    let usage_parsed: JsonValue = serde_json::from_str(&usage_text).expect("valid JSON");
    let usage_keys: Vec<&str> = usage_parsed["payload"]["usage"]
        .as_object()
        .expect("usage is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(usage_keys, USAGE_KEYS);

    let state = HarnessEvent::new(
        3,
        3.0,
        StateChangedEvent {
            old: HarnessState::Idle,
            new: HarnessState::Running,
        },
        "host-1",
        "agent-1",
    );
    assert_eq!(payload_keys(&state), ["old", "new"]);

    let hook = HarnessEvent::new(
        4,
        4.0,
        HookObservedEvent {
            phase: HookEventPhase::Started,
            hook_name: "before_tool".to_owned(),
            hook_id: "hook-1".to_owned(),
            data: wire(json!({"tool": "ls"})),
        },
        "host-1",
        "agent-1",
    );
    assert_eq!(
        payload_keys(&hook),
        ["phase", "hook_name", "hook_id", "data"]
    );

    let diagnostic = HarnessEvent::new(
        5,
        5.0,
        DiagnosticEvent {
            level: DiagnosticLevel::Warning,
            message: "slow provider".to_owned(),
            data: empty_json_object(),
        },
        "host-1",
        "agent-1",
    );
    assert_eq!(payload_keys(&diagnostic), ["level", "message", "data"]);

    assert_eq!(
        payload_keys(&provider_event()),
        ["provider", "event_type", "schema_version", "payload"]
    );
}

#[test]
fn turn_lifecycle_event_round_trips_a_full_result() {
    // Python: test_turn_result_preserves_messages_termination_and_exact_cost (wire form)
    let event = turn_lifecycle_event();
    assert_eq!(
        harness_event_from_dict(harness_event_to_dict(&event).expect("encodes")).expect("decodes"),
        event
    );

    assert_eq!(payload_keys(&event), ["kind", "result"]);
    let encoded: JsonValue =
        serde_json::from_str(&harness_event_to_json_string(&event).expect("encodes as text"))
            .expect("valid JSON");
    let payload = encoded["payload"]
        .as_object()
        .expect("payload is an object");
    assert_eq!(payload["kind"], json!("finished"));
    let result_keys: Vec<&str> = payload["result"]
        .as_object()
        .expect("result is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        result_keys,
        [
            "status",
            "messages",
            "final_output",
            "structured_output",
            "stop_reason",
            "termination",
            "error",
            "usage",
            "cost",
            "started_at",
            "completed_at",
            "duration_ms",
            "provider_data",
        ]
    );
    assert_eq!(
        payload["result"]["cost"],
        json!({"micros": 1_500_000, "currency": "USD"})
    );

    // An aborted turn carries an interrupted result with its termination cause.
    let aborted = HarnessEvent::new(
        6,
        6.0,
        TurnLifecycleEvent {
            kind: TurnEventKind::Aborted,
            result: Some(interrupted_result()),
        },
        "team-session-1",
        "team-a_member-a",
    )
    .with_turn_id("turn-1");
    let decoded = harness_event_from_dict(harness_event_to_dict(&aborted).expect("encodes"))
        .expect("decodes");
    assert_eq!(decoded, aborted);
    let HarnessEventPayload::TurnLifecycle(lifecycle) = &decoded.event else {
        panic!("turn_lifecycle must dispatch to TurnLifecycleEvent");
    };
    let Some(result) = &lifecycle.result else {
        panic!("an aborted turn result must survive the round trip");
    };
    assert_eq!(result.status, TurnStatus::Interrupted);
    assert_eq!(
        result.termination.as_ref().expect("termination").kind,
        TurnTerminationKind::UserAbort
    );
}

#[test]
fn provider_event_round_trips() {
    // Python: test_provider_event_preserves_namespaced_extension_payload
    let event = provider_event();
    let wire_value = to_wire(&event);
    assert_eq!(wire_value["event_type"], json!("provider"));
    assert_eq!(wire_value["schema_version"], json!("1"));
    assert_eq!(
        payload_keys(&event),
        ["provider", "event_type", "schema_version", "payload"]
    );
    assert_eq!(
        harness_event_from_dict(harness_event_to_dict(&event).expect("encodes")).expect("decodes"),
        event
    );
    assert_eq!(wire_value["payload"]["provider"], json!("codex"));
    assert_eq!(
        wire_value["payload"]["event_type"],
        json!("thread.compacted")
    );
}

#[test]
fn unknown_event_round_trips() {
    // Python: test_event_json_codec_round_trips_and_preserves_unknown_events (unknown branch)
    let event = unknown_event();
    let dict = harness_event_to_dict(&event).expect("encodes");
    assert_eq!(dict["event_type"], json!("vendor.thing"));
    assert_eq!(dict["schema_version"], json!("3"));

    let decoded = harness_event_from_dict(dict).expect("decodes");
    assert_eq!(decoded, event);
    let HarnessEventPayload::Unknown(unknown) = &decoded.event else {
        panic!("an UnknownEvent payload must stay unknown");
    };
    assert_eq!(unknown.event_type, "vendor.thing");
    assert_eq!(unknown.schema_version, "3");
    assert_eq!(unknown.payload, wire(json!({"a": 1})));

    let re_encoded = to_wire(&decoded);
    assert_eq!(re_encoded["event_type"], json!("vendor.thing"));
    assert_eq!(re_encoded["schema_version"], json!("3"));
    assert_eq!(re_encoded["payload"], json!({"a": 1}));
}

#[test]
fn every_payload_variant_encodes_the_twelve_envelope_keys() {
    // Python: test_event_json_codec_round_trips_and_preserves_unknown_events (envelope shape)
    let events = [
        output_event(),
        HarnessEvent::new(
            2,
            2.0,
            ItemLifecycleEvent {
                kind: ItemEventKind::Started,
                item_type: "tool_call".to_owned(),
                data: empty_json_object(),
            },
            "host-1",
            "agent-1",
        ),
        HarnessEvent::new(
            3,
            3.0,
            UsageUpdatedEvent {
                usage: TurnUsage::default(),
                mode: UsageUpdateMode::Cumulative,
            },
            "host-1",
            "agent-1",
        ),
        model_request_envelope(),
        HarnessEvent::new(
            4,
            4.0,
            StateChangedEvent {
                old: HarnessState::Idle,
                new: HarnessState::Running,
            },
            "host-1",
            "agent-1",
        ),
        turn_lifecycle_event(),
        HarnessEvent::new(
            5,
            5.0,
            HookObservedEvent {
                phase: HookEventPhase::Finished,
                hook_name: "after_tool".to_owned(),
                hook_id: "hook-2".to_owned(),
                data: empty_json_object(),
            },
            "host-1",
            "agent-1",
        ),
        HarnessEvent::new(
            6,
            6.0,
            DiagnosticEvent {
                level: DiagnosticLevel::Info,
                message: "ready".to_owned(),
                data: empty_json_object(),
            },
            "host-1",
            "agent-1",
        ),
        HarnessEvent::new(
            7,
            7.0,
            ProviderEvent {
                provider: "codex".to_owned(),
                event_type: "thread.compacted".to_owned(),
                schema_version: "2".to_owned(),
                payload: empty_json_object(),
            },
            "host-1",
            "agent-1",
        ),
        unknown_event(),
    ];

    for event in &events {
        let text = harness_event_to_json_string(event).expect("encodes as text");
        let parsed: JsonValue = serde_json::from_str(&text).expect("valid JSON");
        let object = parsed.as_object().expect("the envelope is an object");
        let keys: Vec<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(keys, ENVELOPE_KEYS);
        for key in ENVELOPE_KEYS {
            assert!(
                object.contains_key(key),
                "envelope key {key} is always present"
            );
        }
        for (key, absent) in [
            ("provider_session_id", event.provider_session_id.is_none()),
            ("item_id", event.item_id.is_none()),
            ("correlation_id", event.correlation_id.is_none()),
            ("turn_id", event.turn_id.is_none()),
        ] {
            if absent {
                assert_eq!(object[key], JsonValue::Null, "absent {key} is null");
            }
        }
        assert_eq!(object["causation_ids"], json!([]));
        assert!(object["sequence"].is_number(), "sequence is a JSON number");
    }
}

#[test]
fn dict_and_serde_forms_are_equivalent() {
    // Python: test_event_json_codec_round_trips_and_preserves_unknown_events (serializable smoke check)
    let events = [
        output_event(),
        turn_lifecycle_event(),
        model_request_envelope(),
        provider_event(),
        unknown_event(),
    ];

    for event in &events {
        let dict = harness_event_to_dict(event).expect("encodes");
        assert_eq!(
            json_object_to_value(&dict),
            serde_json::to_value(event).expect("the manual Serialize delegates to the codec")
        );
        let from_dict = harness_event_from_dict(dict.clone()).expect("decodes");
        let from_serde = serde_json::from_value::<HarnessEvent>(json_object_to_value(&dict))
            .expect("the manual Deserialize delegates to the codec");
        assert_eq!(from_dict, from_serde);
        assert_eq!(from_dict, *event);

        let text = harness_event_to_json_string(event).expect("encodes as text");
        assert_eq!(harness_event_from_json_str(&text).expect("decodes"), *event);
    }
}

#[test]
fn schema_version_two_becomes_unknown_event_and_re_encodes_losslessly() {
    // Python: DESIGN §6 step 2 (schema gate) — the Python suite has no direct fixture
    let event = output_event();
    let original = to_wire(&event);
    let original_payload = original["payload"].clone();
    let mut newer = original;
    newer["schema_version"] = json!("2");

    let decoded = decode_wire(&newer).expect("a newer schema decodes as an unknown event");
    let HarnessEventPayload::Unknown(unknown) = &decoded.event else {
        panic!("a newer schema must never be partially parsed");
    };
    assert_eq!(unknown.event_type, "output");
    assert_eq!(unknown.schema_version, "2");
    assert_eq!(unknown.payload, wire(original_payload.clone()));

    let re_encoded = to_wire(&decoded);
    assert_eq!(re_encoded["schema_version"], json!("2"));
    assert_eq!(re_encoded["event_type"], json!("output"));
    assert_eq!(re_encoded["payload"], original_payload);

    let decoded_again = harness_event_from_dict(harness_event_to_dict(&decoded).expect("encodes"))
        .expect("decodes");
    assert_eq!(decoded_again, decoded);
}

#[test]
fn unknown_event_type_with_schema_one_preserves_payload() {
    // Python: test_event_json_codec_round_trips_and_preserves_unknown_events (future_event branch)
    let mut value = to_wire(&output_event());
    value["event_type"] = json!("future_event");
    value["payload"] = json!({"new_field": [1, 2, 3]});

    let decoded = decode_wire(&value).expect("an unknown event type decodes");
    let HarnessEventPayload::Unknown(unknown) = &decoded.event else {
        panic!("an unknown event type must decode as UnknownEvent");
    };
    assert_eq!(unknown.event_type, "future_event");
    assert_eq!(unknown.schema_version, "1");
    assert_eq!(unknown.payload, wire(json!({"new_field": [1, 2, 3]})));

    let re_encoded = to_wire(&decoded);
    assert_eq!(re_encoded["event_type"], json!("future_event"));
    assert_eq!(re_encoded["schema_version"], json!("1"));
    assert_eq!(re_encoded["payload"], json!({"new_field": [1, 2, 3]}));
}

#[test]
fn known_payload_tolerates_extra_unknown_keys() {
    // Python: DESIGN §6 "extra unknown keys inside a known payload are ignored"
    let event = output_event();
    let mut value = to_wire(&event);
    value["payload"]["future_field"] = json!(1);
    value["envelope_future"] = json!(true);

    let decoded = decode_wire(&value).expect("unknown extra keys are tolerated");
    assert_eq!(decoded, event);
    let HarnessEventPayload::Output(output) = &decoded.event else {
        panic!("output must dispatch to OutputEvent");
    };
    assert_eq!(output.output_id, "answer-1");
    assert_eq!(output.operation, OutputOperation::Final);
    assert_eq!(output.content, json!({"answer": [1, 2]}));

    let re_encoded = to_wire(&decoded);
    assert!(
        re_encoded["payload"].get("future_field").is_none(),
        "unknown payload keys are dropped on re-encode"
    );
    assert!(
        re_encoded.get("envelope_future").is_none(),
        "unknown envelope keys are dropped on re-encode"
    );
}

#[test]
#[allow(clippy::too_many_lines)] // One case per Python error message is clearer than a helper table.
fn decode_reports_python_style_envelope_errors() {
    // Python: test_event_scope_time_and_causation_are_validated plus DESIGN §6 error wording
    let base = to_wire(&output_event());

    // Step 1: the discriminators and the payload object raise their message unwrapped.
    let mut without_event_type = base.clone();
    without_event_type
        .as_object_mut()
        .expect("object")
        .remove("event_type");
    assert_eq!(
        protocol_message(decode_wire(&without_event_type)),
        "event_type must be a non-empty string"
    );

    let mut empty_schema = base.clone();
    empty_schema["schema_version"] = json!("");
    assert_eq!(
        protocol_message(decode_wire(&empty_schema)),
        "schema_version must be a non-empty string"
    );

    let mut without_payload = base.clone();
    without_payload
        .as_object_mut()
        .expect("object")
        .remove("payload");
    assert_eq!(
        protocol_message(decode_wire(&without_payload)),
        "payload must be a JSON object"
    );

    // Every other envelope-reader failure is wrapped in the single protocol message.
    for bad_sequence in [json!("x"), json!(1.5), json!(null)] {
        let mut value = base.clone();
        value["sequence"] = bad_sequence;
        protocol_detail(
            decode_wire(&value),
            "invalid harness event wire object: sequence must be an integer",
        );
    }

    let mut bad_timestamp = base.clone();
    bad_timestamp["timestamp"] = json!("x");
    protocol_detail(
        decode_wire(&bad_timestamp),
        "invalid harness event wire object: timestamp must be a number",
    );

    let mut empty_host = base.clone();
    empty_host["host_session_id"] = json!("");
    protocol_detail(
        decode_wire(&empty_host),
        "invalid harness event wire object: host_session_id must be a non-empty string",
    );

    let mut bad_provider_session = base.clone();
    bad_provider_session["provider_session_id"] = json!(5);
    protocol_detail(
        decode_wire(&bad_provider_session),
        "invalid harness event wire object: provider_session_id must be a non-empty string",
    );

    let mut empty_agent = base.clone();
    empty_agent["agent_id"] = json!("");
    protocol_detail(
        decode_wire(&empty_agent),
        "invalid harness event wire object: agent_id must be a non-empty string",
    );

    let mut bad_causation = base.clone();
    bad_causation["causation_ids"] = json!("x");
    protocol_detail(
        decode_wire(&bad_causation),
        "causation_ids must be a JSON array",
    );

    let mut empty_causation = base.clone();
    empty_causation["causation_ids"] = json!([""]);
    protocol_detail(
        decode_wire(&empty_causation),
        "causation_id must be a non-empty string",
    );

    // Envelope validation errors propagate unwrapped with the Python text.
    let mut negative_timestamp = base.clone();
    negative_timestamp["timestamp"] = json!(-1.0);
    assert_eq!(
        validation_message(decode_wire(&negative_timestamp)),
        "event timestamp must be a finite Unix timestamp"
    );

    let mut duplicate_causation = base;
    duplicate_causation["causation_ids"] = json!(["message-1", "message-1"]);
    assert_eq!(
        validation_message(decode_wire(&duplicate_causation)),
        "event causation_ids must not contain duplicates"
    );

    let mut lifecycle = to_wire(&HarnessEvent::new(
        2,
        2.0,
        TurnLifecycleEvent {
            kind: TurnEventKind::Started,
            result: None,
        },
        "host-1",
        "agent-1",
    ));
    lifecycle.as_object_mut().expect("object").remove("turn_id");
    assert_eq!(
        validation_message(decode_wire(&lifecycle)),
        "turn lifecycle event requires turn_id"
    );

    let request = HarnessEvent::new(3, 12.5, model_request_event(), "host-1", "agent-1");
    assert_eq!(
        validation_message(decode_wire(&to_wire(&request))),
        "model request event requires turn_id"
    );

    // A known payload missing a required key surfaces the wrapped serde detail.
    let mut missing_output_id = to_wire(&output_event());
    missing_output_id["payload"]
        .as_object_mut()
        .expect("payload object")
        .remove("output_id");
    protocol_detail(
        decode_wire(&missing_output_id),
        "invalid harness event wire object:",
    );
}

#[test]
fn decode_rejects_non_object_json_text() {
    // Python: the Python API only accepted dicts, so this covers the Rust text entry point.
    let error = harness_event_from_json_str("[1,2]").expect_err("an array is not an envelope");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(error.message(), "harness event JSON must be a JSON object");

    let result = harness_event_from_json_str("{not json");
    assert_eq!(
        result
            .as_ref()
            .expect_err("malformed JSON is rejected")
            .kind(),
        HarnessErrorKind::Protocol
    );
    assert!(
        error_message(result).contains("invalid harness JSON"),
        "serde failures stay protocol errors"
    );

    // `harness_event_from_dict` takes a `JsonObject`, so a non-object top-level value is not
    // expressible through that typed entry point and needs no test.
}
