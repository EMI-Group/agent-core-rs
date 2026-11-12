//! Unit tests of the LLM payload normalization and GenAI attribute helpers.
//!
//! Port of the GenAI attribute assertions of
//! `tests/unit_tests/extensions/tracer_otel/test_handler.py`:
//! `TestGenAiSemconvAttrs`, the `_standard_messages` cases, the retrieval
//! projection cases, and the `usage_metadata` latency cases.
// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use opentelemetry::trace::Tracer as _;
use opentelemetry::{Context, Value as SpanValue};
use opentelemetry_sdk::trace::SpanData;
use serde_json::json;

use crate::otel::config::{ExporterType, OtelTracerConfig};
use crate::otel::handle::{init_otel_tracer, TracerHandle};

/// Name of the span every export helper opens.
const SPAN_NAME: &str = "chat Model";

/// A configuration with redaction off: attribute text stays readable.
fn plain_config() -> OtelTracerConfig {
    OtelTracerConfig {
        redaction_enabled: false,
        ..OtelTracerConfig::default()
    }
}

/// Runs `fill` on a fresh span, ends it, flushes, and returns its export.
fn export(fill: impl FnOnce(&mut Span)) -> SpanData {
    let handle: TracerHandle =
        init_otel_tracer(OtelTracerConfig::default().with_exporter_type(ExporterType::Memory))
            .expect("the memory exporter always initializes");
    let mut span = handle
        .tracer()
        .expect("the memory handle is active")
        .start_with_context(SPAN_NAME, &Context::new());
    fill(&mut span);
    span.end();
    handle.force_flush();
    handle
        .capture()
        .expect("the memory handle captures")
        .span_named(SPAN_NAME)
        .expect("the span was exported")
}

/// The export of one LLM request span.
fn request_span(inputs: &Value, info: &Value, config: &OtelTracerConfig) -> SpanData {
    export(|span| set_llm_request_attributes(span, inputs, info, config))
}

/// The export of one LLM response span.
fn response_span(outputs: &Value) -> SpanData {
    export(|span| set_llm_response_attributes(span, outputs))
}

/// The typed OTel value of one attribute, when present.
fn value_of<'a>(data: &'a SpanData, key: &str) -> Option<&'a SpanValue> {
    data.attributes
        .iter()
        .find(|pair| pair.key.as_str() == key)
        .map(|pair| &pair.value)
}

/// The string value of one attribute, when present.
fn attribute(data: &SpanData, key: &str) -> Option<String> {
    value_of(data, key).map(|value| value.as_str().into_owned())
}

/// The numeric value of one attribute, whether stored as int or float.
// Widening an `i64` attribute (token counts) into an `f64` is exact here.
#[allow(clippy::cast_precision_loss)]
fn number(data: &SpanData, key: &str) -> Option<f64> {
    match value_of(data, key)? {
        SpanValue::F64(value) => Some(*value),
        SpanValue::I64(value) => Some(*value as f64),
        _ => None,
    }
}

/// The integer value of one attribute, when stored as an `i64`.
fn integer(data: &SpanData, key: &str) -> Option<i64> {
    match value_of(data, key)? {
        SpanValue::I64(value) => Some(*value),
        _ => None,
    }
}

/// The string-array value of one attribute, when present.
fn strings(data: &SpanData, key: &str) -> Option<Vec<String>> {
    match value_of(data, key)? {
        SpanValue::Array(Array::String(values)) => Some(
            values
                .iter()
                .map(|value| value.as_str().to_owned())
                .collect(),
        ),
        _ => None,
    }
}

/// Parses a JSON-encoded attribute payload.
fn json_attribute(data: &SpanData, key: &str) -> Value {
    let text = attribute(data, key).expect("the attribute is written");
    serde_json::from_str(&text).expect("the attribute holds JSON")
}

#[test]
fn request_parameters_keep_their_json_types() {
    // test_handler.py::test_llm_start_sets_request_params_and_message_count
    let info = json!({
        "class_name": "TestModel",
        "request_params": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 4096,
            "stop_sequences": ["END"], "reasoning_level": "high",
        },
        "message_count": 3,
    });
    let data = request_span(&Value::Null, &info, &plain_config());
    let temperature = value_of(&data, GEN_AI_REQUEST_TEMPERATURE);
    assert!(matches!(temperature, Some(SpanValue::F64(_))));
    assert_eq!(number(&data, GEN_AI_REQUEST_TEMPERATURE), Some(0.7));
    assert_eq!(number(&data, GEN_AI_REQUEST_TOP_P), Some(0.9));
    // An integer knob widens to the float attribute the schema expects.
    assert_eq!(number(&data, GEN_AI_REQUEST_TOP_K), Some(40.0));
    assert_eq!(integer(&data, GEN_AI_REQUEST_MAX_TOKENS), Some(4096));
    assert_eq!(integer(&data, OJ_LLM_PREV_MESSAGE_COUNT), Some(3));
    let stops = strings(&data, GEN_AI_REQUEST_STOP_SEQUENCES);
    assert_eq!(stops, Some(vec!["END".to_owned()]));
    let model = attribute(&data, GEN_AI_REQUEST_MODEL);
    assert_eq!(model.as_deref(), Some("TestModel"));
    let level = attribute(&data, GEN_AI_REQUEST_REASONING_LEVEL);
    assert_eq!(level.as_deref(), Some("high"));
    let trace_name = attribute(&data, OJ_GEN_AI_TRACE_NAME);
    assert_eq!(trace_name.as_deref(), Some("TestModel"));
    let operation = attribute(&data, GEN_AI_OPERATION_NAME);
    assert_eq!(operation.as_deref(), Some(GEN_AI_OPERATION_CHAT));
}

#[test]
fn request_parameters_skip_absent_and_malformed_values() {
    // test_handler.py::test_llm_start_without_params_sets_no_request_attrs
    let info = json!({
        "class_name": "Model",
        "request_params": {
            "temperature": "not-a-number", "max_tokens": ["bad"],
            "stop_sequences": "END", "reasoning_level": 5,
        },
    });
    let data = request_span(&Value::Null, &info, &plain_config());
    for key in [
        GEN_AI_REQUEST_TEMPERATURE,
        GEN_AI_REQUEST_MAX_TOKENS,
        GEN_AI_REQUEST_STOP_SEQUENCES,
        GEN_AI_REQUEST_REASONING_LEVEL,
        OJ_LLM_PREV_MESSAGE_COUNT,
    ] {
        assert!(value_of(&data, key).is_none(), "{key} must be skipped");
    }
    // A non-object instance_info behaves like an empty one.
    let data = request_span(&Value::Null, &json!("nope"), &plain_config());
    assert_eq!(attribute(&data, GEN_AI_REQUEST_MODEL).as_deref(), Some(""));
}

#[test]
fn identity_attributes_follow_the_common_info() {
    // test_handler.py::test_llm_start_sets_agent_identity_attrs
    let info = json!({
        "class_name": "TestModel", "agent_id": "card-1", "agent_name": "HelperAgent",
        "agent_description": "A helper", "conversation_id": "sess-42", "user_id": "u-9",
        "metadata": {"user_id": "u-9", "channel": "web"},
    });
    let data = request_span(&Value::Null, &info, &plain_config());
    for (key, expected) in [
        (GEN_AI_AGENT_ID, "card-1"),
        (GEN_AI_AGENT_NAME, "HelperAgent"),
        (GEN_AI_AGENT_DESCRIPTION, "A helper"),
        (GEN_AI_CONVERSATION_ID, "sess-42"),
        (OJ_GEN_AI_USER_ID, "u-9"),
    ] {
        assert_eq!(attribute(&data, key).as_deref(), Some(expected), "{key}");
    }
    let metadata = json_attribute(&data, OJ_GEN_AI_METADATA);
    assert_eq!(metadata, json!({"user_id": "u-9", "channel": "web"}));
    // The executing agent names the trace, not the span entity.
    let trace_name = attribute(&data, OJ_GEN_AI_TRACE_NAME);
    assert_eq!(trace_name.as_deref(), Some("HelperAgent"));
    // Absent facts (and empty metadata) skip their attribute.
    let info = json!({"class_name": "Model", "metadata": {}});
    let data = request_span(&Value::Null, &info, &plain_config());
    for key in [
        GEN_AI_AGENT_ID,
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_DESCRIPTION,
        GEN_AI_CONVERSATION_ID,
        OJ_GEN_AI_USER_ID,
        OJ_GEN_AI_METADATA,
    ] {
        assert!(value_of(&data, key).is_none(), "{key} must be absent");
    }
}

#[test]
fn inputs_split_into_system_instructions_and_user_messages() {
    // test_handler.py::test_agent_llm_inputs_normalized_to_dict
    let inputs = json!({"inputs": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
    ]});
    let data = request_span(&inputs, &json!({}), &plain_config());
    let instructions = json_attribute(&data, GEN_AI_SYSTEM_INSTRUCTIONS);
    assert_eq!(
        instructions,
        json!([{"type": "text", "content": "you are helpful"}])
    );
    let messages = json!([{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]);
    assert_eq!(json_attribute(&data, GEN_AI_INPUT_MESSAGES), messages);
    // A null payload writes neither attribute.
    let data = request_span(&Value::Null, &json!({}), &plain_config());
    assert!(value_of(&data, GEN_AI_INPUT_MESSAGES).is_none());
    assert!(value_of(&data, GEN_AI_SYSTEM_INSTRUCTIONS).is_none());
}

#[test]
fn standard_messages_covers_every_payload_shape() {
    // ::test_agent_llm_inputs_normalized_to_dict and the tool-call path.
    let (instructions, messages) = standard_messages(&json!("hi"), "user");
    assert_eq!(instructions.len(), 0);
    let expected = vec![json!({"role": "user", "parts": [{"type": "text", "content": "hi"}]})];
    assert_eq!(messages, expected);
    // An array of scalars: each item becomes a content-only message.
    let (_, messages) = standard_messages(&json!(["a", "b"]), "assistant");
    let expected = vec![
        json!({"role": "assistant", "parts": [{"type": "text", "content": "a"}]}),
        json!({"role": "assistant", "parts": [{"type": "text", "content": "b"}]}),
    ];
    assert_eq!(messages, expected);
    // `messages` wins over `inputs`, a blank role takes the default, and a
    // `response`/`output` key stands in for content.
    let payload = json!({"messages": [{"role": "", "response": "x"}], "inputs": ["no"]});
    let (_, messages) = standard_messages(&payload, "user");
    let expected = vec![json!({"role": "user", "parts": [{"type": "text", "content": "x"}]})];
    assert_eq!(messages, expected);
    // A message with no content keeps an empty parts list; null yields nothing.
    let (_, messages) = standard_messages(&json!({"role": "user"}), "user");
    assert_eq!(messages, vec![json!({"role": "user", "parts": []})]);
    let (instructions, messages) = standard_messages(&Value::Null, "user");
    assert_eq!((instructions.len(), messages.len()), (0, 0));
    // A tool call copies its own keys, falls back to `function`, skips a
    // null/absent field, and ignores non-object calls.
    let inputs = json!([{
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {"id": "call-1", "function": {"name": "weather", "arguments": "{\"city\":\"北京\"}"}},
            {"id": null, "name": "echo", "arguments": {"raw": true}},
            "not-an-object",
        ],
    }]);
    let (instructions, messages) = standard_messages(&inputs, "user");
    assert_eq!(instructions.len(), 0);
    assert_eq!(
        messages,
        vec![json!({
            "role": "assistant",
            "parts": [
                {"type": "tool_call", "id": "call-1", "name": "weather",
                 "arguments": "{\"city\":\"北京\"}"},
                {"type": "tool_call", "name": "echo", "arguments": {"raw": true}},
            ],
        })]
    );
}

#[test]
fn prompt_redaction_hashes_content_and_honours_the_field_override() {
    // test_handler.py::test_agent_llm_prompt_completion_split_redaction
    let config = OtelTracerConfig {
        redaction_enabled: true,
        ..OtelTracerConfig::default()
    };
    let data = request_span(&json!("secret data"), &json!({}), &config);
    let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
    let content = messages[0]["parts"][0]["content"].as_str().unwrap();
    assert!(content.starts_with("sha256:"), "unexpected {content}");
    // `redacted_json` is the same walk; hashing replaces the raw text.
    let redactor = Redactor::new(&config);
    let hashed = redacted_json(&messages, &redactor, RedactionField::Prompts, false);
    assert!(hashed.contains("sha256:") && !hashed.contains("secret data"));
    // `redact_prompts = false` overrides `redaction_enabled = true`.
    let config = OtelTracerConfig {
        redaction_enabled: true,
        redact_prompts: Some(false),
        ..OtelTracerConfig::default()
    };
    let data = request_span(&json!("visible prompt"), &json!({}), &config);
    let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
    assert_eq!(messages[0]["parts"][0]["content"], json!("visible prompt"));
}

#[test]
fn system_instructions_bypass_the_attribute_length_cap() {
    // test_handler.py::test_agent_llm_system_instructions_bypass_attribute_length_cap
    let config = OtelTracerConfig {
        redaction_enabled: false,
        max_attr_length: 8,
        ..OtelTracerConfig::default()
    };
    let system_prompt = "stable-system-prompt-".repeat(20);
    let inputs = json!({"inputs": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "long-user-message"},
    ]});
    let data = request_span(&inputs, &json!({}), &config);
    let instructions = json_attribute(&data, GEN_AI_SYSTEM_INSTRUCTIONS);
    assert_eq!(instructions[0]["content"], json!(system_prompt));
    let messages = json_attribute(&data, GEN_AI_INPUT_MESSAGES);
    let truncated = messages[0]["parts"][0]["content"].as_str().unwrap();
    assert!(
        truncated.contains("OTel attribute truncated"),
        "{truncated}"
    );
}

#[test]
fn response_finish_reason_and_container_unwrapping() {
    // test_handler.py::test_llm_end_unwraps_outputs_container_into_content
    for container in ["outputs", "messages", "inputs"] {
        let payload = json!({
            container: {"finish_reason": "stop", "usage_metadata": {"input_tokens": 4}},
        });
        let data = response_span(&payload);
        let reasons = strings(&data, GEN_AI_RESPONSE_FINISH_REASONS);
        assert_eq!(reasons, Some(vec!["stop".to_owned()]), "{container}");
        assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(4));
    }
    // The schema default "null", a null, a `false`, and an absent value all
    // write nothing; the usage block is unaffected.
    for reason in [
        json!({"finish_reason": "null", "usage_metadata": {"input_tokens": 1}}),
        json!({"finish_reason": null, "usage_metadata": {"input_tokens": 1}}),
        json!({"finish_reason": false, "usage_metadata": {"input_tokens": 1}}),
        json!({"usage_metadata": {"input_tokens": 1}}),
    ] {
        let data = response_span(&reason);
        let key = GEN_AI_RESPONSE_FINISH_REASONS;
        assert!(value_of(&data, key).is_none(), "{reason}");
        assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(1));
    }
    // A scalar payload, a container holding one, a non-object, or a missing
    // `usage_metadata` object writes no usage attributes at all.
    for payload in [
        json!("plain"),
        json!({"outputs": "plain"}),
        json!([1]),
        json!({}),
    ] {
        let data = response_span(&payload);
        let model = value_of(&data, GEN_AI_RESPONSE_MODEL);
        assert!(model.is_none(), "{payload}");
        assert!(value_of(&data, GEN_AI_USAGE_INPUT_TOKENS).is_none());
    }
}

#[test]
fn response_usage_cost_and_cache_attributes_are_recorded() {
    // test_handler.py::test_llm_end_extracts_response_usage_and_cost_attrs
    let payload = json!({
        "finish_reason": "stop",
        "usage_metadata": {
            "model_name": "claude-test", "input_tokens": 11, "output_tokens": 22,
            "cache_read_tokens": 5, "cache_creation_input_tokens": 7,
            "reasoning_tokens": 3, "input_cost": 0.001, "output_cost": 0.002,
            "total_cost": 0.003,
        },
    });
    let data = response_span(&payload);
    let model = attribute(&data, GEN_AI_RESPONSE_MODEL);
    assert_eq!(model.as_deref(), Some("claude-test"));
    for (key, expected) in [
        (GEN_AI_USAGE_INPUT_TOKENS, 11),
        (GEN_AI_USAGE_OUTPUT_TOKENS, 22),
        (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, 5),
        (GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, 7),
        (GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, 3),
    ] {
        assert_eq!(integer(&data, key), Some(expected), "{key}");
    }
    for (key, expected) in [
        (OJ_GEN_AI_USAGE_INPUT_COST, 0.001),
        (OJ_GEN_AI_USAGE_OUTPUT_COST, 0.002),
        (OJ_GEN_AI_USAGE_TOTAL_COST, 0.003),
    ] {
        assert_eq!(number(&data, key), Some(expected), "{key}");
    }
    // The legacy cache keys stand in when the explicit splits are absent.
    let payload = json!({"usage_metadata": {
        "input_tokens": 1, "cache_tokens": 9, "cache_creation_input_tokens": 6,
    }});
    let data = response_span(&payload);
    assert_eq!(
        integer(&data, GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS),
        Some(9)
    );
    assert_eq!(
        integer(&data, GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS),
        Some(6)
    );
}

#[test]
fn response_usage_zeroes_are_recorded_as_facts() {
    // test_handler.py::test_llm_end_sets_zero_usage_values
    let data = response_span(&json!({"usage_metadata": {}}));
    assert_eq!(attribute(&data, GEN_AI_RESPONSE_MODEL).as_deref(), Some(""));
    for key in [
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    ] {
        assert_eq!(integer(&data, key), Some(0), "{key}");
    }
    for key in [
        OJ_GEN_AI_USAGE_TOTAL_COST,
        OJ_GEN_AI_USAGE_INPUT_COST,
        OJ_GEN_AI_USAGE_OUTPUT_COST,
    ] {
        assert_eq!(number(&data, key), Some(0.0), "{key}");
    }
    // Latencies have no default: absent stays absent.
    for key in [
        GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS,
        OJ_GEN_AI_REASONING_DURATION_MS,
    ] {
        assert!(value_of(&data, key).is_none(), "{key}");
    }
}

#[test]
fn time_to_first_chunk_table() {
    // test_handler.py::test_llm_end_reads_ttft_from_the_explicit_ttft_seconds_field,
    // ::test_llm_end_prefers_ttft_seconds_over_legacy_first_token_time, and
    // ::test_llm_end_malformed_ttft_seconds_skips_attr_not_finalization
    for (usage, expected) in [
        (json!({"ttft_seconds": 0.42}), Some(0.42)),
        (json!({"ttft_seconds": 0.0}), Some(0.0)),
        (
            json!({"ttft_seconds": 0.35, "first_token_time": "9.99"}),
            Some(0.35),
        ),
        // The legacy fallback applies only when the explicit key is absent.
        (json!({"first_token_time": "0.42"}), Some(0.42)),
        (json!({"first_token_time": ""}), None),
        (
            json!({"ttft_seconds": null, "first_token_time": "0.5"}),
            Some(0.5),
        ),
        // A malformed explicit value skips instead of falling back.
        (
            json!({"ttft_seconds": "bad", "first_token_time": "9.99"}),
            None,
        ),
        (json!({"ttft_seconds": ["bad"]}), None),
    ] {
        let data = response_span(&json!({"usage_metadata": usage}));
        let ttfc = number(&data, GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK);
        assert_eq!(ttfc, expected, "{usage} is wrong");
    }
}

#[test]
fn latency_attributes_are_written_only_when_parseable() {
    // test_handler.py::test_llm_end_sets_streaming_and_reasoning_latency and
    // ::test_llm_end_malformed_latency_skips_attrs_not_finalization
    let payload = json!({"usage_metadata": {
        "inter_token_latency_ms": 12.5, "reasoning_duration_ms": "300.0",
    }});
    let data = response_span(&payload);
    let inter = number(&data, OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS);
    assert_eq!(inter, Some(12.5));
    assert_eq!(number(&data, OJ_GEN_AI_REASONING_DURATION_MS), Some(300.0));
    let malformed = json!({"usage_metadata": {
        "input_tokens": 1, "inter_token_latency_ms": "bad",
        "reasoning_duration_ms": ["bad"],
    }});
    let data = response_span(&malformed);
    assert!(value_of(&data, OJ_GEN_AI_RESPONSE_INTER_TOKEN_LATENCY_MS).is_none());
    assert!(value_of(&data, OJ_GEN_AI_REASONING_DURATION_MS).is_none());
    // Finalization survived: the other usage facts are still written.
    assert_eq!(integer(&data, GEN_AI_USAGE_INPUT_TOKENS), Some(1));
}

#[test]
fn retrieval_query_and_documents_extraction_table() {
    // test_handler.py::test_workflow_retrieval_component_tagged_as_retrieval
    let query = extract_retrieval_query(&json!("什么是向量检索"));
    assert_eq!(query, Some("什么是向量检索".to_owned()));
    let query = extract_retrieval_query(&json!({"query": "hello"}));
    assert_eq!(query, Some("hello".to_owned()));
    for absent in [
        json!(""),
        json!({"query": ""}),
        json!({"query": 7}),
        json!([]),
    ] {
        assert_eq!(extract_retrieval_query(&absent), None, "{absent}");
    }
    assert_eq!(extract_retrieval_query(&Value::Null), None);
    let documents = extract_retrieval_documents(&json!({"results": ["doc-1 内容", "x"]}));
    let expected = vec!["doc-1 内容".to_owned(), "x".to_owned()];
    assert_eq!(documents, Some(expected));
    let documents = extract_retrieval_documents(&json!({"results": [1, {"id": 2}]}));
    assert_eq!(
        documents,
        Some(vec!["1".to_owned(), r#"{"id":2}"#.to_owned()])
    );
    for absent in [json!({"results": []}), json!({"results": "doc"}), json!({})] {
        assert_eq!(extract_retrieval_documents(&absent), None, "{absent}");
    }
    assert_eq!(extract_retrieval_documents(&Value::Null), None);
}
