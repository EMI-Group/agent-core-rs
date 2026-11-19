//! Unit tests for the mock server's wire projection and state behavior.
//!
//! No sockets are bound here: the transport, SSE decoder, and error mapping are
//! driven end to end by `tests/mock_server.rs`, so these tests stay pure and
//! fast. Python parity notes cite the fixture they mirror, for example
//! `tests/unit_tests/fixtures/mock_llm.py`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{header, HeaderMap, HeaderValue, Method, Uri};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::llm::{FinishReason, StreamChunk};
use openjiuwen_foundation::prompt::Usage;
use serde_json::{json, Value};

use crate::responses::{text_chunks, tool_call_chunks, MockCompletion, MockToolCall};

use super::routes;
use super::state::{RecordedRequest, ScriptedResponse, ServerConfig, ServerState};
use super::wire::{self, MockHttpError};

/// Builds a state with the default configuration (empty script, default fallback).
fn default_state() -> ServerState {
    ServerState::new(ServerConfig::default())
}

/// Parses the JSON payload of one `data: <json>` SSE frame.
fn payload(frame: &str) -> Value {
    let prefix = "data: ";
    assert!(
        frame.starts_with(prefix),
        "frame is not an SSE data frame: {frame}"
    );
    assert!(frame.ends_with("\n\n"), "frame is not terminated: {frame}");
    serde_json::from_str(frame.trim_start_matches(prefix).trim_end())
        .expect("the frame payload is JSON")
}

/// The text a [`ScriptedResponse`] carries, for fallback assertions.
fn scripted_text(response: &ScriptedResponse) -> Option<&str> {
    match response {
        ScriptedResponse::Completion(completion) => completion.content.as_deref(),
        ScriptedResponse::Chunks(_) | ScriptedResponse::Error(_) => None,
    }
}

// --- non-streaming body ----------------------------------------------------

#[test]
fn completion_body_matches_the_openai_shape() {
    let completion = MockCompletion::text("hello").with_usage(Usage::new(3, 2));
    let body = wire::completion_body("chatcmpl-mock-0001", "mock-model", &completion);

    assert_eq!(body["id"], json!("chatcmpl-mock-0001"));
    assert_eq!(body["object"], json!("chat.completion"));
    assert_eq!(body["created"], json!(0));
    assert_eq!(body["model"], json!("mock-model"));
    assert_eq!(body["choices"][0]["index"], json!(0));
    assert_eq!(body["choices"][0]["message"]["role"], json!("assistant"));
    assert_eq!(body["choices"][0]["message"]["content"], json!("hello"));
    assert!(body["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
    assert!(body["choices"][0]["message"].get("tool_calls").is_none());
    assert_eq!(body["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(
        body["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    );
}

#[test]
fn completion_body_nulls_missing_content_and_keeps_usage() {
    let completion = MockCompletion::tool_call("add", json!({"a": 1}));
    let body = wire::completion_body("chatcmpl-mock-0001", "mock-model", &completion);

    assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"],
        json!([{
            "id": "call_add",
            "type": "function",
            "function": {"name": "add", "arguments": "{\"a\":1}"},
        }])
    );
    assert_eq!(body["choices"][0]["finish_reason"], json!("tool_calls"));
    // An all-zero usage is still emitted: the client always reads the object.
    assert_eq!(
        body["usage"],
        json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    );
}

#[test]
fn completion_body_carries_reasoning_and_the_scripted_model() {
    let completion = MockCompletion::reasoning("thinking", "answer").with_model("gpt-4o");
    let body = wire::completion_body("id", "gpt-4o", &completion);

    assert_eq!(body["model"], json!("gpt-4o"));
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        json!("thinking")
    );
    assert_eq!(body["choices"][0]["message"]["content"], json!("answer"));
}

#[test]
fn completion_body_reports_every_finish_reason() {
    for (reason, wire_reason) in [
        (FinishReason::Stop, "stop"),
        (FinishReason::Length, "length"),
        (FinishReason::ContentFilter, "content_filter"),
    ] {
        let completion = MockCompletion::text("x").with_finish_reason(reason);
        let body = wire::completion_body("id", "m", &completion);
        assert_eq!(body["choices"][0]["finish_reason"], json!(wire_reason));
    }
}

#[test]
fn chunks_body_folds_frames_into_the_same_shape() {
    let body = wire::chunks_body("id", "mock-model", &text_chunks("hello"));
    assert_eq!(body["choices"][0]["message"]["content"], json!("hello"));
    assert_eq!(body["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(
        body["usage"],
        json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    );

    let tool_calls = tool_call_chunks(vec![MockToolCall::new("add", json!({"a": 1}))]);
    let folded = wire::chunks_body("id", "mock-model", &tool_calls);
    assert_eq!(folded["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(folded["choices"][0]["finish_reason"], json!("tool_calls"));
    assert_eq!(
        folded["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        json!("add")
    );
}

#[test]
fn chunks_body_defaults_an_empty_script_to_a_stopped_null_message() {
    let body = wire::chunks_body("id", "mock-model", &[]);
    assert_eq!(body["choices"][0]["message"]["content"], Value::Null);
    assert_eq!(body["choices"][0]["finish_reason"], json!("stop"));
}

// --- SSE frames ------------------------------------------------------------

#[test]
fn sse_frames_of_a_tool_call_completion_keep_the_provider_order() {
    let completion = MockCompletion::tool_calls(vec![MockToolCall::new("add", json!({"a": 1}))])
        .with_usage(Usage::new(3, 2));
    let frames = wire::sse_frames("chatcmpl-mock-0001", "mock-model", &completion.to_chunks());

    // role, tool call, usage, finish, [DONE]
    assert_eq!(frames.len(), 5);
    assert_eq!(frames.last().map(String::as_str), Some("data: [DONE]\n\n"));

    let role = payload(&frames[0]);
    assert_eq!(role["object"], json!("chat.completion.chunk"));
    assert_eq!(role["created"], json!(0));
    assert_eq!(role["model"], json!("mock-model"));
    assert_eq!(role["choices"][0]["delta"]["role"], json!("assistant"));
    assert_eq!(role["choices"][0]["finish_reason"], Value::Null);

    let call = payload(&frames[1]);
    assert_eq!(
        call["choices"][0]["delta"]["tool_calls"],
        json!([{
            "index": 0,
            "id": "call_add",
            "type": "function",
            "function": {"name": "add", "arguments": "{\"a\":1}"},
        }])
    );
    assert_eq!(call["choices"][0]["finish_reason"], Value::Null);

    let usage = payload(&frames[2]);
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(
        usage["usage"],
        json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    );

    let finish = payload(&frames[3]);
    assert_eq!(finish["choices"][0]["delta"], json!({}));
    assert_eq!(finish["choices"][0]["finish_reason"], json!("tool_calls"));
}

#[test]
fn sse_frames_of_a_text_completion_skip_the_usage_frame_when_empty() {
    let frames = wire::sse_frames("id", "m", &text_chunks("hello"));
    // role, content, finish, [DONE]
    assert_eq!(frames.len(), 4);
    assert_eq!(
        payload(&frames[1])["choices"][0]["delta"]["content"],
        json!("hello")
    );
    assert_eq!(
        payload(&frames[2])["choices"][0]["finish_reason"],
        json!("stop")
    );
}

#[test]
fn sse_frames_stream_reasoning_next_to_the_content() {
    let completion = MockCompletion::reasoning("why", "because");
    let frames = wire::sse_frames("id", "m", &completion.to_chunks());
    let delta = &payload(&frames[1])["choices"][0]["delta"];
    assert_eq!(delta["reasoning_content"], json!("why"));
    assert_eq!(delta["content"], json!("because"));
}

#[test]
fn a_done_chunk_never_becomes_a_json_frame() {
    assert_eq!(
        wire::sse_frames("id", "m", &[StreamChunk::Done]),
        vec!["data: [DONE]\n\n".to_owned()]
    );
    assert_eq!(
        wire::sse_frames("id", "m", &[]),
        vec!["data: [DONE]\n\n".to_owned()]
    );

    // A Done in the middle still yields one sentinel at the end.
    let mut chunks = text_chunks("hi");
    chunks.insert(1, StreamChunk::Done);
    let frames = wire::sse_frames("id", "m", &chunks);
    assert_eq!(frames.last().map(String::as_str), Some("data: [DONE]\n\n"));
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame.as_str() == "data: [DONE]\n\n")
            .count(),
        1
    );
}

#[test]
fn a_partial_tool_call_delta_omits_absent_function_fields() {
    let chunks = vec![StreamChunk::Delta {
        index: 0,
        role: None,
        content: None,
        reasoning: None,
        tool_call: Some(openjiuwen_foundation::prompt::ToolCallDelta {
            index: Some(2),
            id: None,
            name: Some("add".to_owned()),
            arguments: None,
        }),
        finish_reason: None,
    }];
    let frames = wire::sse_frames("id", "m", &chunks);
    assert_eq!(
        payload(&frames[0])["choices"][0]["delta"]["tool_calls"],
        json!([{"index": 2, "type": "function", "function": {"name": "add"}}])
    );
}

// --- usage -----------------------------------------------------------------

#[test]
fn usage_json_emits_the_three_mandatory_counters() {
    assert_eq!(
        wire::usage_json(&Usage::new(10, 4)),
        json!({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
    );
    assert_eq!(
        wire::usage_json(&Usage::default()),
        json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    );
}

#[test]
fn usage_json_adds_details_only_for_non_zero_counters_and_merges_extra() {
    let mut usage = Usage::new(10, 4);
    usage.cache_tokens = 3;
    usage.reasoning_tokens = 2;
    usage.extra.insert("vendor_score".to_owned(), json!(0.5));

    assert_eq!(
        wire::usage_json(&usage),
        json!({
            "vendor_score": 0.5,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        })
    );

    let details_only = Usage {
        cache_tokens: 1,
        ..Usage::default()
    };
    let value = wire::usage_json(&details_only);
    assert_eq!(value["prompt_tokens_details"], json!({"cached_tokens": 1}));
    assert!(value.get("completion_tokens_details").is_none());
}

// --- error and health bodies ----------------------------------------------

/// Python parity: the error payloads mirror `httpx.Response` fakes in
/// `tests/unit_tests/fixtures/mock_llm.py`.
#[test]
fn error_constructors_build_the_openai_envelope() {
    let rate_limited = MockHttpError::rate_limited();
    assert_eq!(rate_limited.status, 429);
    assert_eq!(
        rate_limited.body,
        json!({"error": {
            "message": "rate limit exceeded",
            "type": "rate_limit_error",
            "param": null,
            "code": null,
        }})
    );

    let unauthorized = MockHttpError::unauthorized();
    assert_eq!(unauthorized.status, 401);
    assert_eq!(
        unauthorized.body["error"]["type"],
        json!("authentication_error")
    );

    let server_error = MockHttpError::server_error("boom");
    assert_eq!(server_error.status, 500);
    assert_eq!(server_error.body["error"]["message"], json!("boom"));
    assert_eq!(server_error.body["error"]["type"], json!("server_error"));

    let not_found = MockHttpError::not_found();
    assert_eq!(not_found.status, 404);
    assert_eq!(not_found.body["error"]["message"], json!("not found"));

    let custom = MockHttpError::with_body(418, json!({"detail": "teapot"}));
    assert_eq!(custom.status, 418);
    assert_eq!(custom.body, json!({"detail": "teapot"}));

    let scripted = ScriptedResponse::error(503, "unavailable");
    let ScriptedResponse::Error(error) = scripted else {
        panic!("expected an error script");
    };
    assert_eq!(error.status, 503);
    assert_eq!(error.body["error"]["message"], json!("unavailable"));
}

#[test]
fn router_bodies_are_json_envelopes() {
    assert_eq!(wire::health_body(), json!({"status": "ok"}));
    assert_eq!(
        wire::not_found_body(),
        json!({"error": {
            "message": "not found",
            "type": "invalid_request_error",
            "param": null,
            "code": null,
        }})
    );

    let malformed = wire::bad_request_body("expected a JSON object");
    assert_eq!(malformed["error"]["type"], json!("invalid_request_error"));
    assert_eq!(
        malformed["error"]["message"],
        json!("invalid request body: expected a JSON object")
    );
}

// --- request parsing -------------------------------------------------------

#[test]
fn request_parsing_accepts_unknown_fields_and_defaults() {
    let request = wire::parse_request(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "tools": [{"type": "function"}],
        "tool_choice": "auto",
        "vendor_extension": {"x": 1},
    }))
    .expect("an object parses");

    assert_eq!(request.model.as_deref(), Some("gpt-4o"));
    assert!(request.stream);
    assert_eq!(request.messages.len(), 1);

    let empty = wire::parse_request(&json!({})).expect("an empty object parses");
    assert_eq!(empty.model, None);
    assert!(!empty.stream);
    assert_eq!(empty.messages, Vec::<Value>::new());

    // A wrongly typed field degrades to its default instead of failing the call.
    let odd = wire::parse_request(&json!({"model": 7, "stream": "yes"}))
        .expect("an object with odd field types still parses");
    assert_eq!(odd.model, None);
    assert!(!odd.stream);
}

#[test]
fn request_parsing_rejects_non_objects() {
    for value in [Value::Null, json!([1, 2]), json!("text"), json!(3)] {
        assert!(wire::parse_request(&value).is_none(), "value: {value}");
    }
}

// --- queue, ids, and fallback ---------------------------------------------

#[test]
fn the_queue_pops_one_entry_per_request_and_only_pops_bump_the_sequence() {
    let state = default_state();
    assert_eq!(state.queued(), 0);

    state.push(ScriptedResponse::text("first"));
    state.push_many([
        ScriptedResponse::text("second"),
        ScriptedResponse::chunks(Vec::new()),
    ]);
    assert_eq!(state.queued(), 3);

    let first = state.pop();
    assert_eq!(first.seq, 1);
    assert_eq!(scripted_text(&first.response), Some("first"));
    assert_eq!(state.queued(), 2);

    state.clear_script();
    assert_eq!(state.queued(), 0);

    // The fallback is served without consuming anything and keeps the counter.
    let fallback = state.pop();
    assert_eq!(fallback.seq, 1);
    assert_eq!(
        scripted_text(&fallback.response),
        Some("Default mock response")
    );
    assert_eq!(state.queued(), 0);

    let again = state.pop();
    assert_eq!(again.seq, 1);
    assert_eq!(
        scripted_text(&again.response),
        Some("Default mock response")
    );
}

#[test]
fn the_fallback_can_be_replaced_and_is_reused() {
    let state = default_state();
    state.set_fallback(ScriptedResponse::text("custom fallback"));

    for _ in 0..3 {
        let popped = state.pop();
        assert_eq!(scripted_text(&popped.response), Some("custom fallback"));
    }
    assert_eq!(state.queued(), 0);
}

#[test]
fn response_identity_uses_the_deterministic_id_and_the_default_model() {
    let state = default_state();
    state.push(ScriptedResponse::text("a"));
    let identity = routes::response_identity(&state, &state.pop());
    assert_eq!(
        identity,
        (
            Some("chatcmpl-mock-0001".to_owned()),
            "mock-model".to_owned()
        )
    );

    state.push(ScriptedResponse::text("b"));
    let identity = routes::response_identity(&state, &state.pop());
    assert_eq!(
        identity,
        (
            Some("chatcmpl-mock-0002".to_owned()),
            "mock-model".to_owned()
        )
    );
}

#[test]
fn response_identity_prefers_a_scripted_id_and_model() {
    let state = default_state();
    state.push(ScriptedResponse::completion(
        MockCompletion::text("a")
            .with_id("chatcmpl-custom")
            .with_model("gpt-4o"),
    ));
    let identity = routes::response_identity(&state, &state.pop());
    assert_eq!(
        identity,
        (Some("chatcmpl-custom".to_owned()), "gpt-4o".to_owned())
    );
}

#[test]
fn response_identity_uses_the_configured_prefix_and_model() {
    let state = ServerState::new(ServerConfig {
        id_prefix: "test-".to_owned(),
        default_model: "small-model".to_owned(),
        ..ServerConfig::default()
    });
    state.push(ScriptedResponse::text("a"));
    assert_eq!(
        routes::response_identity(&state, &state.pop()),
        (
            Some("chatcmpl-test-0001".to_owned()),
            "small-model".to_owned()
        )
    );

    // A frame script gets the same deterministic id and the default model.
    state.push(ScriptedResponse::chunks(Vec::new()));
    assert_eq!(
        routes::response_identity(&state, &state.pop()),
        (
            Some("chatcmpl-test-0002".to_owned()),
            "small-model".to_owned()
        )
    );

    // The fallback reuses the current counter value without bumping it.
    assert_eq!(
        routes::response_identity(&state, &state.pop()),
        (
            Some("chatcmpl-test-0002".to_owned()),
            "small-model".to_owned()
        )
    );

    state.push(ScriptedResponse::error(500, "boom"));
    assert_eq!(routes::response_identity(&state, &state.pop()).0, None);
}

#[test]
fn the_chunk_delay_defaults_to_zero() {
    assert!(default_state().chunk_delay().is_zero());
    let delayed = ServerState::new(ServerConfig {
        chunk_delay: Duration::from_millis(5),
        ..ServerConfig::default()
    });
    assert_eq!(delayed.chunk_delay(), Duration::from_millis(5));
}

#[test]
fn scripted_response_constructors_cover_every_variant() {
    assert!(matches!(
        ScriptedResponse::from(MockCompletion::text("x")),
        ScriptedResponse::Completion(_)
    ));
    assert!(matches!(
        ScriptedResponse::completion(MockCompletion::text("x")),
        ScriptedResponse::Completion(_)
    ));
    assert!(matches!(
        ScriptedResponse::chunks(Vec::new()),
        ScriptedResponse::Chunks(_)
    ));
    assert!(matches!(
        ScriptedResponse::text("x"),
        ScriptedResponse::Completion(_)
    ));
    assert!(matches!(
        ScriptedResponse::error(500, "x"),
        ScriptedResponse::Error(_)
    ));
}

// --- recordings ------------------------------------------------------------

#[test]
fn recording_lowercases_headers_and_joins_repeated_values() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer key"),
    );
    headers.append(header::ACCEPT, HeaderValue::from_static("a"));
    headers.append(header::ACCEPT, HeaderValue::from_static("b"));

    let uri: Uri = "/v1/chat/completions?trace=1".parse().expect("a URI");
    let recorded = routes::recording(
        &Method::POST,
        &uri,
        &headers,
        json!({"messages": [{"role": "user", "content": "hi"}]}),
        "{\"messages\":[]}",
        true,
        Some("gpt-4o".to_owned()),
        Some("chatcmpl-mock-0001".to_owned()),
    );

    assert_eq!(recorded.method, "POST");
    assert_eq!(recorded.path, "/v1/chat/completions");
    assert_eq!(recorded.query.as_deref(), Some("trace=1"));
    assert_eq!(recorded.header("Authorization"), Some("Bearer key"));
    assert_eq!(recorded.header("AUTHORIZATION"), Some("Bearer key"));
    assert_eq!(recorded.header("accept"), Some("a, b"));
    assert_eq!(recorded.header("missing"), None);
    assert_eq!(recorded.raw_body, "{\"messages\":[]}");
    assert!(recorded.stream);
    assert_eq!(recorded.model.as_deref(), Some("gpt-4o"));
    assert_eq!(recorded.response_id.as_deref(), Some("chatcmpl-mock-0001"));
}

#[test]
fn recorded_request_accessors_read_the_body() {
    let recorded = RecordedRequest {
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        query: None,
        headers: BTreeMap::new(),
        body: json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "http://example.test/i.png"}},
                {"type": "text", "text": "b"},
            ]},
        ]}),
        raw_body: String::new(),
        stream: false,
        model: None,
        response_id: None,
    };

    assert_eq!(recorded.messages().len(), 3);
    assert_eq!(
        recorded.texts_of_role("user"),
        vec!["hi".to_owned(), "ab".to_owned()]
    );
    assert_eq!(
        recorded.user_texts(),
        vec!["hi".to_owned(), "ab".to_owned()]
    );
    assert_eq!(
        recorded.texts_of_role("assistant"),
        vec!["hello".to_owned()]
    );
    assert_eq!(recorded.texts_of_role("system"), Vec::<String>::new());

    let empty = RecordedRequest {
        body: Value::Null,
        ..recorded
    };
    assert_eq!(empty.messages(), Vec::<Value>::new());
    assert_eq!(empty.user_texts(), Vec::<String>::new());
}

// --- waiting ----------------------------------------------------------------

/// A minimal recording: enough to bump the counter and wake a waiter.
fn recorded(model: Option<&str>) -> RecordedRequest {
    RecordedRequest {
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        query: None,
        headers: BTreeMap::new(),
        body: json!({"messages": []}),
        raw_body: String::new(),
        stream: false,
        model: model.map(str::to_owned),
        response_id: None,
    }
}

/// Ports the waiter contract of `MockLLMServer.wait_for_request`: a waiter
/// registered before the recording must still observe it, and a count that is
/// already satisfied returns without awaiting.
#[tokio::test]
async fn wait_for_requests_registers_before_rechecking_the_count() {
    let state = Arc::new(default_state());

    tokio::time::timeout(Duration::from_millis(50), state.wait_for_requests_until(0))
        .await
        .expect("count 0 returns immediately");

    let recorder = Arc::clone(&state);
    let record_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        recorder.record(recorded(Some("gpt-4o")));
    });

    tokio::time::timeout(Duration::from_secs(1), state.wait_for_requests_until(1))
        .await
        .expect("the waiter wakes on the first recording");
    record_task.await.expect("the recorder task finishes");

    assert_eq!(state.request_count(), 1);
    assert_eq!(
        state
            .last_request()
            .and_then(|request| request.model)
            .as_deref(),
        Some("gpt-4o")
    );

    // No second request ever arrives, so waiting for two must expire.
    let error = state
        .wait_for_requests_timeout(2, Duration::from_millis(20))
        .await
        .expect_err("a second request never arrives");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.message().contains("wait_for_requests(2)"),
        "{}",
        error.message()
    );
}
