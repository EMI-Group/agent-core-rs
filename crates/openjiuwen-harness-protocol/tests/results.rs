//! Round A turn-result parity suite: content blocks, messages, termination, cost, usage, error,
//! and the cross-field rules of the terminal `TurnResult`.
//!
//! Where a Python counterpart exists it is named in a `// Python:` comment and lives in
//! `tests/unit_tests/harness_protocol/` of the Python SDK (v0.1.20).

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use serde_json::json;

use openjiuwen_harness_protocol::*;

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

/// A self-consistent completed result used as the base for the cross-field cases.
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

/// Asserts a validation failure and returns the bare Python message.
fn validation_message(result: HarnessResult<()>) -> String {
    let error = result.expect_err("expected a validation error");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    error.message().to_owned()
}

#[test]
fn content_block_validates_id_and_kind_and_round_trips() {
    assert!(block("b1", "text").validate().is_ok());

    for invalid in [block("", "text"), block("b1", "")] {
        assert_eq!(
            validation_message(invalid.validate()),
            "content block id and kind must not be empty"
        );
    }
    assert_eq!(
        block("", "text").validate().unwrap_err().to_string(),
        "invalid harness value: content block id and kind must not be empty"
    );

    let decoded: ContentBlock = from_value_validated(
        json!({"block_id": "b2", "kind": "tool_use", "content": {"name": "ls"}}),
    )
    .expect("data defaults to an empty object");
    assert_eq!(decoded.data, empty_json_object());

    let populated = ContentBlock {
        block_id: "b3".to_owned(),
        kind: "reasoning".to_owned(),
        content: json!({"parts": ["one", {"value": 2}]}),
        data: json_object_from_value(json!({"tokens": 12, "cached": true}))
            .expect("data is an object"),
    };
    let encoded = serde_json::to_value(&populated).expect("serializes");
    assert_eq!(encoded["content"], json!({"parts": ["one", {"value": 2}]}));
    assert_eq!(encoded["data"], json!({"cached": true, "tokens": 12}));

    let round_tripped: ContentBlock = from_value_validated(encoded).expect("round trip");
    assert_eq!(round_tripped, populated);
}

#[test]
fn turn_message_validates_id_blocks_and_role_wire_values() {
    assert!(message("m1", vec![block("b1", "text")]).validate().is_ok());
    assert!(message("m1", Vec::new()).validate().is_ok());

    assert_eq!(
        validation_message(message("", Vec::new()).validate()),
        "turn message_id must not be empty"
    );

    // A nested invalid block surfaces the block's own message, not the message's.
    assert_eq!(
        validation_message(message("m1", vec![block("b1", "text"), block("", "text")]).validate()),
        "content block id and kind must not be empty"
    );

    for (role, wire) in [
        (MessageRole::User, "user"),
        (MessageRole::Assistant, "assistant"),
        (MessageRole::System, "system"),
        (MessageRole::Tool, "tool"),
    ] {
        assert_eq!(serde_json::to_value(role).expect("serializes"), json!(wire));
        let decoded: MessageRole = serde_json::from_value(json!(wire)).expect("deserializes");
        assert_eq!(decoded, role);
    }

    let decoded: TurnMessage = from_value_validated(json!({
        "message_id": "m2",
        "role": "tool",
        "content": [{"block_id": "b1", "kind": "text", "content": "ok"}],
    }))
    .expect("valid message with defaulted data");
    assert_eq!(decoded.data, empty_json_object());
    assert_eq!(decoded.content[0].data, empty_json_object());
}

#[test]
fn turn_termination_has_no_rules_and_uses_python_wire_values() {
    let termination = TurnTermination {
        kind: TurnTerminationKind::Policy,
        message: Some(String::new()),
        code: Some(String::new()),
        provider_data: empty_json_object(),
    };
    assert!(termination.validate().is_ok());

    for (kind, wire) in [
        (TurnTerminationKind::UserAbort, "user_abort"),
        (TurnTerminationKind::Timeout, "timeout"),
        (TurnTerminationKind::Policy, "policy"),
        (TurnTerminationKind::Provider, "provider"),
        (TurnTerminationKind::HarnessStop, "harness_stop"),
    ] {
        assert_eq!(serde_json::to_value(kind).expect("serializes"), json!(wire));
        let decoded: TurnTerminationKind =
            serde_json::from_value(json!(wire)).expect("deserializes");
        assert_eq!(decoded, kind);
    }

    let decoded: TurnTermination =
        from_value_validated(json!({"kind": "timeout"})).expect("defaults for the optional fields");
    assert_eq!(decoded.kind, TurnTerminationKind::Timeout);
    assert_eq!(decoded.message, None);
    assert_eq!(decoded.code, None);
    assert_eq!(decoded.provider_data, empty_json_object());
}

#[test]
fn monetary_amount_defaults_normalises_and_checks_three_letters() {
    let amount = MonetaryAmount::new(125_000, "usd");
    assert_eq!(
        amount,
        MonetaryAmount {
            micros: 125_000,
            currency: "USD".to_owned(),
        }
    );
    assert!(amount.validate().is_ok());

    let defaulted: MonetaryAmount =
        from_value_validated(json!({"micros": 125_000})).expect("currency defaults to USD");
    assert_eq!(defaulted, amount);
    assert_eq!(
        serde_json::to_value(&defaulted).expect("serializes"),
        json!({"micros": 125_000, "currency": "USD"})
    );

    // Validation uppercases first, so a lowercase in-memory currency is accepted.
    let lowercase = MonetaryAmount {
        micros: 1,
        currency: "eur".to_owned(),
    };
    assert!(lowercase.validate().is_ok());
    assert!(
        from_value_validated::<MonetaryAmount>(json!({"micros": 1, "currency": "eur"})).is_ok()
    );

    for bad in ["US", "USDD", "U1D"] {
        let amount = MonetaryAmount {
            micros: 1,
            currency: bad.to_owned(),
        };
        assert_eq!(
            validation_message(amount.validate()),
            "monetary currency must be a three-letter code",
            "currency {bad:?}"
        );
    }

    let error = from_value_validated::<MonetaryAmount>(json!({"micros": -1}))
        .expect_err("negative micros is rejected by the unsigned field");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
}

#[test]
fn turn_usage_defaults_round_trips_and_rejects_negative_counters() {
    let usage = TurnUsage::default();
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.reasoning_output_tokens, None);
    assert_eq!(usage.total_tokens, None);
    assert_eq!(usage.provider_data, empty_json_object());
    assert!(usage.validate().is_ok());

    let populated = TurnUsage {
        input_tokens: Some(10),
        output_tokens: Some(20),
        cached_input_tokens: Some(3),
        reasoning_output_tokens: Some(4),
        total_tokens: Some(33),
        provider_data: json_object_from_value(json!({"provider": "fake"})).expect("object"),
    };
    assert!(populated.validate().is_ok());

    let encoded = serde_json::to_value(&populated).expect("serializes");
    assert_eq!(
        encoded,
        json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "cached_input_tokens": 3,
            "reasoning_output_tokens": 4,
            "total_tokens": 33,
            "provider_data": {"provider": "fake"},
        })
    );
    let decoded: TurnUsage = from_value_validated(encoded).expect("round trip");
    assert_eq!(decoded, populated);

    let error = from_value_validated::<TurnUsage>(json!({"total_tokens": -5}))
        .expect_err("negative counter is rejected by the unsigned field");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
}

#[test]
fn turn_error_requires_message_and_defaults_optional_fields() {
    let empty = TurnError {
        message: String::new(),
        code: None,
        category: None,
        retryable: None,
        provider_data: empty_json_object(),
    };
    assert_eq!(
        validation_message(empty.validate()),
        "turn error message must not be empty"
    );

    let decoded: TurnError =
        from_value_validated(json!({"message": "provider failed", "retryable": true}))
            .expect("valid error");
    assert_eq!(decoded.code, None);
    assert_eq!(decoded.category, None);
    assert_eq!(decoded.retryable, Some(true));
    assert_eq!(decoded.provider_data, empty_json_object());
}

#[test]
fn turn_result_couples_status_and_error() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_failed_result_requires_structured_error
    let failed_without_error = TurnResult {
        status: TurnStatus::Failed,
        ..completed_result()
    };
    assert_eq!(
        validation_message(failed_without_error.validate()),
        "failed turn result requires error"
    );

    let error = TurnError {
        message: "provider failed".to_owned(),
        code: Some("provider_error".to_owned()),
        category: None,
        retryable: Some(true),
        provider_data: empty_json_object(),
    };
    let completed_with_error = TurnResult {
        error: Some(error.clone()),
        ..completed_result()
    };
    assert_eq!(
        validation_message(completed_with_error.validate()),
        "only failed turn result may contain error"
    );

    let failed_with_error = TurnResult {
        status: TurnStatus::Failed,
        error: Some(error),
        ..completed_result()
    };
    assert!(failed_with_error.validate().is_ok());
}

#[test]
fn turn_result_couples_status_and_termination() {
    let interrupted_without_termination = TurnResult {
        status: TurnStatus::Interrupted,
        ..completed_result()
    };
    assert_eq!(
        validation_message(interrupted_without_termination.validate()),
        "interrupted turn result requires termination"
    );

    let termination = TurnTermination {
        kind: TurnTerminationKind::UserAbort,
        message: Some("cancelled by user".to_owned()),
        code: None,
        provider_data: empty_json_object(),
    };
    let interrupted_with_termination = TurnResult {
        status: TurnStatus::Interrupted,
        termination: Some(termination.clone()),
        ..completed_result()
    };
    assert!(interrupted_with_termination.validate().is_ok());

    let completed_with_termination = TurnResult {
        termination: Some(termination),
        ..completed_result()
    };
    assert_eq!(
        validation_message(completed_with_termination.validate()),
        "only interrupted turn result may contain termination"
    );
}

#[test]
fn turn_result_timestamps_and_duration() {
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let result = TurnResult {
            started_at: Some(bad),
            ..completed_result()
        };
        assert_eq!(
            validation_message(result.validate()),
            "turn started_at must be a finite Unix timestamp"
        );

        let result = TurnResult {
            started_at: None,
            completed_at: Some(bad),
            ..completed_result()
        };
        assert_eq!(
            validation_message(result.validate()),
            "turn completed_at must be a finite Unix timestamp"
        );
    }

    let reversed = TurnResult {
        started_at: Some(10.0),
        completed_at: Some(9.999),
        ..completed_result()
    };
    assert_eq!(
        validation_message(reversed.validate()),
        "turn completed_at must not precede started_at"
    );

    let equal = TurnResult {
        started_at: Some(10.0),
        completed_at: Some(10.0),
        ..completed_result()
    };
    assert!(equal.validate().is_ok());
    let increasing = TurnResult {
        started_at: Some(10.0),
        completed_at: Some(10.5),
        ..completed_result()
    };
    assert!(increasing.validate().is_ok());

    let duration = TurnResult {
        duration_ms: Some(1500),
        ..completed_result()
    };
    let encoded = serde_json::to_value(&duration).expect("serializes");
    assert_eq!(encoded["duration_ms"], json!(1500));
    let decoded: TurnResult = from_value_validated(encoded).expect("round trip");
    assert_eq!(decoded.duration_ms, Some(1500));

    // The unsigned field subsumes Python's negative-duration rule: the wire value is rejected.
    let error =
        from_value_validated::<TurnResult>(json!({"status": "completed", "duration_ms": -1}))
            .expect_err("negative duration is rejected by the unsigned field");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
}

#[test]
fn turn_result_validates_nested_values() {
    let bad_message = TurnResult {
        messages: vec![message("m1", vec![block("", "text")])],
        ..completed_result()
    };
    assert_eq!(
        validation_message(bad_message.validate()),
        "content block id and kind must not be empty"
    );

    // The termination rule precedes nested validation for interrupted turns.
    let interrupted_without_termination = TurnResult {
        status: TurnStatus::Interrupted,
        termination: None,
        ..completed_result()
    };
    assert_eq!(
        validation_message(interrupted_without_termination.validate()),
        "interrupted turn result requires termination"
    );

    let bad_error = TurnResult {
        status: TurnStatus::Failed,
        error: Some(TurnError {
            message: String::new(),
            code: None,
            category: None,
            retryable: None,
            provider_data: empty_json_object(),
        }),
        ..completed_result()
    };
    assert_eq!(
        validation_message(bad_error.validate()),
        "turn error message must not be empty"
    );

    // `Completed` permits `cost`, so the nested amount currency is reached.
    let bad_cost = TurnResult {
        cost: Some(MonetaryAmount {
            micros: 1,
            currency: "EU".to_owned(),
        }),
        ..completed_result()
    };
    assert_eq!(
        validation_message(bad_cost.validate()),
        "monetary currency must be a three-letter code"
    );

    // An invalid usage counter can only arrive through the wire, where it fails to decode.
    let error = from_value_validated::<TurnResult>(json!({
        "status": "completed",
        "usage": {"input_tokens": -3},
    }))
    .expect_err("negative usage counter is rejected by the unsigned field");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
}

#[test]
fn turn_result_round_trips_with_every_key() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_turn_result_preserves_messages_termination_and_exact_cost
    let result = TurnResult {
        status: TurnStatus::Completed,
        messages: vec![message(
            "message-1",
            vec![
                block("block-1", "text"),
                ContentBlock {
                    block_id: "block-2".to_owned(),
                    kind: "tool_use".to_owned(),
                    content: json!({"name": "pwd", "arguments": {}}),
                    data: json_object_from_value(json!({"index": 1})).expect("object"),
                },
            ],
        )],
        final_output: json!("partial"),
        structured_output: json!({"answer": 42}),
        stop_reason: Some("stop".to_owned()),
        termination: None,
        error: None,
        usage: Some(TurnUsage {
            input_tokens: Some(11),
            output_tokens: Some(7),
            cached_input_tokens: Some(0),
            reasoning_output_tokens: Some(2),
            total_tokens: Some(18),
            provider_data: json_object_from_value(json!({"provider": "fake"})).expect("object"),
        }),
        cost: Some(MonetaryAmount::new(125_000, "usd")),
        started_at: Some(1_700_000_000.0),
        completed_at: Some(1_700_000_001.25),
        duration_ms: Some(1250),
        provider_data: json_object_from_value(json!({"provider": "fake", "trace": "t1"}))
            .expect("object"),
    };
    assert!(result.validate().is_ok());

    let encoded = serde_json::to_value(&result).expect("serializes");
    let keys: Vec<&str> = encoded
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
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
    assert_eq!(encoded["status"], json!("completed"));
    assert_eq!(encoded["termination"], JsonValue::Null);
    assert_eq!(encoded["error"], JsonValue::Null);
    assert_eq!(
        encoded["cost"],
        json!({"micros": 125_000, "currency": "USD"})
    );
    assert_eq!(
        encoded["messages"][0]["content"][1]["kind"],
        json!("tool_use")
    );

    let decoded: TurnResult = from_value_validated(encoded).expect("round trip");
    assert_eq!(decoded, result);
}

#[test]
fn turn_result_tolerates_unknown_keys() {
    let decoded: TurnResult = from_value_validated(json!({
        "status": "failed",
        "future_top_level_key": {"nested": [1, 2, 3]},
        "error": {"message": "provider failed", "future_error_key": true},
        "cost": {"micros": 5, "currency": "usd", "future_cost_key": "ignored"},
        "usage": {"input_tokens": 1, "future_usage_key": 9},
    }))
    .expect("unknown keys are tolerated everywhere");

    assert_eq!(decoded.status, TurnStatus::Failed);
    assert_eq!(
        decoded.error.as_ref().map(|error| error.message.as_str()),
        Some("provider failed")
    );
    assert_eq!(decoded.cost.as_ref().map(|cost| cost.micros), Some(5));
    assert_eq!(
        decoded.usage.as_ref().and_then(|usage| usage.input_tokens),
        Some(1)
    );
    assert_eq!(decoded.messages, Vec::new());
}
