//! End-to-end tests for the loopback mock LLM server.
//!
//! Unlike the unit tests beside `src/mock_llm/`, these tests bind a real socket
//! and drive it through the real [`OpenAiCompatibleClient`], so the transport,
//! request encoder, SSE decoder, finish-reason parsing, and error mapping are all
//! exercised against the wire format a provider would send.
//!
//! They mirror, end to end, the two Python patterns of
//! `tests/unit_tests/fixtures/mock_llm.py`: the ordered response queue of
//! `MockLLMModel` (no prompt matching, a reusable fallback once the queue is
//! empty) and the `httpx.MockTransport` request fakes (which recorded requests so
//! a test could assert on the outgoing payload).
//!
//! Every test owns its server, its script, and its assertions, stays on loopback,
//! waits at most a second at a time, and ends with an explicit
//! [`MockLlmServer::shutdown`] instead of relying on `Drop`.
//!
//! [`OpenAiCompatibleClient`]: openjiuwen_foundation::llm::OpenAiCompatibleClient

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::llm::{
    FinishReason, Model, ModelExt, ModelRequestConfig, OpenAiCompatibleClient, StreamAccumulator,
    StreamChunk,
};
use openjiuwen_foundation::prompt::{BaseMessage, ToolCall, Usage};
use openjiuwen_test_support::{
    eventually, MockCompletion, MockHttpError, MockLlmServer, ScriptedResponse,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Model name every test asks for; the server answers with its own default model
/// unless a scripted turn names one.
const MODEL: &str = "gpt-4o";

/// Builds a client pointed at `server`.
fn client_for(server: &MockLlmServer) -> OpenAiCompatibleClient {
    server
        .openai_client(MODEL)
        .expect("the mock server yields a usable client")
}

/// The per-call request configuration every test sends.
fn request_for(server: &MockLlmServer) -> ModelRequestConfig {
    server.request_config(MODEL)
}

/// The single-user-message conversation every test sends.
fn conversation(text: &str) -> Vec<BaseMessage> {
    vec![BaseMessage::user(text)]
}

/// Collects a whole [`Model::stream`] into chunk results.
async fn collect_stream(
    client: &OpenAiCompatibleClient,
    messages: Vec<BaseMessage>,
    request: ModelRequestConfig,
) -> Vec<Result<StreamChunk, Error>> {
    client.stream(messages, request).collect().await
}

/// Concatenates the content deltas of a successfully decoded stream.
fn streamed_text(chunks: &[Result<StreamChunk, Error>]) -> String {
    chunks
        .iter()
        .map(|chunk| chunk.as_ref().expect("no stream errors"))
        .filter_map(StreamChunk::content)
        .collect()
}

// --- plain text ------------------------------------------------------------

#[tokio::test]
async fn non_streaming_text_completion_carries_text_usage_id_and_model() {
    let server = MockLlmServer::with_script([ScriptedResponse::completion(
        MockCompletion::text("hello").with_usage(Usage::new(3, 2)),
    )])
    .await;
    let client = client_for(&server);

    let response = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");

    assert_eq!(response.id, "chatcmpl-mock-0001");
    assert_eq!(response.model, "mock-model");
    let message = response.message().expect("one choice");
    assert_eq!(message.text().as_ref(), "hello");
    assert_eq!(response.usage, Usage::new(3, 2));
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));

    server.shutdown().await;
}

#[tokio::test]
async fn streaming_text_deltas_concatenate_and_usage_arrives_on_its_own_frame() {
    let server = MockLlmServer::with_script([ScriptedResponse::completion(
        MockCompletion::text("hello").with_usage(Usage::new(3, 2)),
    )])
    .await;
    let client = client_for(&server);

    let chunks = collect_stream(
        &client,
        conversation("hi"),
        request_for(&server).with_stream(true),
    )
    .await;

    assert_eq!(streamed_text(&chunks), "hello");

    let chunks: Vec<StreamChunk> = chunks.into_iter().map(|chunk| chunk.unwrap()).collect();
    let done_count = chunks
        .iter()
        .filter(|chunk| matches!(chunk, StreamChunk::Done))
        .count();
    assert_eq!(done_count, 1, "the stream ends with exactly one Done");
    assert_eq!(
        chunks.last(),
        Some(&StreamChunk::Done),
        "Done is the final item"
    );
    let usage = chunks
        .iter()
        .find_map(|chunk| match chunk {
            StreamChunk::Usage(usage) => Some(usage.clone()),
            _ => None,
        })
        .expect("the scripted usage is streamed");
    assert_eq!(usage, Usage::new(3, 2));

    server.shutdown().await;
}

#[tokio::test]
async fn complete_accumulates_the_streamed_assistant_turn() {
    let server = MockLlmServer::with_script([ScriptedResponse::completion(
        MockCompletion::text("hello").with_usage(Usage::new(4, 1)),
    )])
    .await;
    let client = client_for(&server);

    // `complete` forces `stream = true`, so the SSE decoder and the accumulator
    // are both in the path.
    let turn = client
        .complete(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");

    assert_eq!(turn.message.text().as_ref(), "hello");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    assert_eq!(turn.usage, Usage::new(4, 1));

    server.shutdown().await;
}

// --- tool calls ------------------------------------------------------------

#[tokio::test]
async fn non_streaming_tool_call_round_trips_id_name_arguments_and_finish_reason() {
    let server = MockLlmServer::with_script([ScriptedResponse::completion(
        MockCompletion::tool_call("get_weather", json!({"city": "Paris"})),
    )])
    .await;
    let client = client_for(&server);

    let response = client
        .invoke(&conversation("weather?"), &request_for(&server))
        .await
        .expect("the mock server answers");

    assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    let message = response.message().expect("one choice");
    assert_eq!(message.text().as_ref(), "", "a tool call has no text");
    assert_eq!(message.tool_calls.len(), 1);
    let call: &ToolCall = &message.tool_calls[0];
    assert_eq!(call.id.as_deref(), Some("call_get_weather"));
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, r#"{"city":"Paris"}"#);

    server.shutdown().await;
}

#[tokio::test]
async fn streaming_tool_call_fragments_merge_into_one_call() {
    let server = MockLlmServer::with_script([ScriptedResponse::completion(
        MockCompletion::tool_call("get_weather", json!({"city": "Paris"})),
    )])
    .await;
    let client = client_for(&server);

    let chunks = collect_stream(
        &client,
        conversation("weather?"),
        request_for(&server).with_stream(true),
    )
    .await;

    let mut accumulator = StreamAccumulator::new();
    let mut finish_reasons = Vec::new();
    for chunk in &chunks {
        let chunk = chunk.as_ref().expect("no stream errors");
        accumulator.push(chunk);
        if let StreamChunk::Delta {
            finish_reason: Some(reason),
            ..
        } = chunk
        {
            finish_reasons.push(reason.clone());
        }
    }
    let turn = accumulator.finish();

    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "a delta reports the terminal reason"
    );
    assert_eq!(
        turn.message.tool_calls.len(),
        1,
        "fragments merge per index"
    );
    let call = &turn.message.tool_calls[0];
    assert_eq!(call.id.as_deref(), Some("call_get_weather"));
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
    assert_eq!(turn.finish_reason, Some(FinishReason::ToolCalls));

    server.shutdown().await;
}

// --- reasoning -------------------------------------------------------------

#[tokio::test]
async fn reasoning_content_survives_both_projections() {
    let server = MockLlmServer::with_script([
        ScriptedResponse::completion(MockCompletion::reasoning("thinking", "answer")),
        ScriptedResponse::completion(MockCompletion::reasoning("pondering", "reply")),
    ])
    .await;
    let client = client_for(&server);

    let response = client
        .invoke(&conversation("why?"), &request_for(&server))
        .await
        .expect("the mock server answers");
    let message = response.message().expect("one choice");
    assert_eq!(message.reasoning_content.as_deref(), Some("thinking"));
    assert_eq!(message.text().as_ref(), "answer");

    let chunks = collect_stream(
        &client,
        conversation("why?"),
        request_for(&server).with_stream(true),
    )
    .await;
    let mut accumulator = StreamAccumulator::new();
    for chunk in &chunks {
        accumulator.push(chunk.as_ref().expect("no stream errors"));
    }
    let turn = accumulator.finish();
    assert_eq!(turn.message.reasoning_content.as_deref(), Some("pondering"));
    assert_eq!(turn.message.text().as_ref(), "reply");
    assert!(
        chunks.iter().any(|chunk| matches!(
            chunk,
            Ok(StreamChunk::Delta {
                reasoning: Some(reasoning),
                ..
            }) if reasoning == "pondering"
        )),
        "a delta carries the reasoning fragment"
    );

    server.shutdown().await;
}

// --- errors ----------------------------------------------------------------

#[tokio::test]
async fn http_errors_map_to_the_client_error_kinds_without_retrying() {
    let server = MockLlmServer::with_script([
        ScriptedResponse::Error(MockHttpError::rate_limited()),
        ScriptedResponse::Error(MockHttpError::server_error("boom")),
        ScriptedResponse::Error(MockHttpError::unauthorized()),
        ScriptedResponse::Error(MockHttpError::server_error("stream boom")),
    ])
    .await;
    let client = client_for(&server);

    let limited = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect_err("429 is an error");
    assert_eq!(limited.kind(), ErrorKind::Llm);
    assert_eq!(server.request_count(), 1);

    let broken = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect_err("500 is an error");
    assert_eq!(broken.kind(), ErrorKind::Llm);
    assert_eq!(
        server.request_count(),
        2,
        "max_retries(0) means one attempt"
    );

    let unauthorized = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect_err("401 is an error");
    assert_eq!(unauthorized.kind(), ErrorKind::Config);
    assert_eq!(server.request_count(), 3);

    // A streamed call surfaces the error as the single item of the stream.
    let chunks = collect_stream(
        &client,
        conversation("hi"),
        request_for(&server).with_stream(true),
    )
    .await;
    assert_eq!(chunks.len(), 1, "an HTTP error ends the stream immediately");
    assert_eq!(
        chunks[0].as_ref().expect_err("the item is an error").kind(),
        ErrorKind::Llm
    );
    assert_eq!(server.request_count(), 4);

    server.shutdown().await;
}

#[tokio::test]
async fn complete_surfaces_a_streamed_http_error() {
    let server =
        MockLlmServer::with_script([ScriptedResponse::Error(MockHttpError::unauthorized())]).await;
    let client = client_for(&server);

    let error = client
        .complete(&conversation("hi"), &request_for(&server))
        .await
        .expect_err("401 is an error");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(server.request_count(), 1);

    server.shutdown().await;
}

// --- script queue and fallback --------------------------------------------

#[tokio::test]
async fn queue_exhaustion_serves_the_untouched_fallback() {
    let server =
        MockLlmServer::with_script([ScriptedResponse::text("one"), ScriptedResponse::text("two")])
            .await;
    let client = client_for(&server);

    let first = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("first scripted turn");
    assert_eq!(first.message().expect("one choice").text().as_ref(), "one");
    assert_eq!(server.queued(), 1);

    let second = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("second scripted turn");
    assert_eq!(second.message().expect("one choice").text().as_ref(), "two");
    assert_eq!(server.queued(), 0);

    // Python parity: `"Default mock response"`, served without popping anything.
    let fallback = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the fallback answers");
    assert_eq!(
        fallback.message().expect("one choice").text().as_ref(),
        "Default mock response"
    );
    assert_eq!(server.queued(), 0, "the fallback is never consumed");

    // The queue still works after a fallback answer.
    server.push(ScriptedResponse::text("three"));
    let third = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("third scripted turn");
    assert_eq!(
        third.message().expect("one choice").text().as_ref(),
        "three"
    );

    server.set_fallback(ScriptedResponse::text("custom fallback"));
    let custom = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the replaced fallback answers");
    assert_eq!(
        custom.message().expect("one choice").text().as_ref(),
        "custom fallback"
    );
    assert_eq!(server.queued(), 0);

    server.shutdown().await;
}

#[tokio::test]
async fn fallback_answers_reuse_the_current_sequence_id() {
    // The documented id rule: `seq` counts popped script entries from 1, and an
    // answer served from the fallback reuses the current counter value without
    // advancing it. The counter therefore starts at 0, so a fresh server's first
    // fallback answer is `0000` and stays there until a script entry is popped.
    let server = MockLlmServer::start().await;
    let client = client_for(&server);

    let first = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the fallback answers");
    assert_eq!(first.id, "chatcmpl-mock-0000");
    assert_eq!(
        first.message().expect("one choice").text().as_ref(),
        "Default mock response"
    );
    assert_eq!(server.queued(), 0);

    // One popped entry advances the counter to 1 for its own answer...
    server.push(ScriptedResponse::text("one"));
    let second = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the scripted turn answers");
    assert_eq!(second.id, "chatcmpl-mock-0001");
    assert_eq!(second.message().expect("one choice").text().as_ref(), "one");
    assert_eq!(server.queued(), 0);

    // ...and the fallback answer that follows reuses it instead of advancing it.
    let third = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the fallback answers again");
    assert_eq!(third.id, "chatcmpl-mock-0001");
    assert_eq!(
        third.message().expect("one choice").text().as_ref(),
        "Default mock response"
    );
    assert_eq!(server.queued(), 0, "the fallback is never consumed");

    server.shutdown().await;
}

// --- recordings and addressing --------------------------------------------

#[tokio::test]
async fn requests_are_recorded_with_their_wire_shape() {
    let server = MockLlmServer::with_script([
        ScriptedResponse::text("hello"),
        ScriptedResponse::text("streamed"),
    ])
    .await;
    let client = client_for(&server);

    client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");
    server
        .wait_for_requests(1)
        .await
        .expect("the request is recorded");

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let recorded = &requests[0];
    assert_eq!(recorded.path, "/v1/chat/completions");
    assert_eq!(recorded.method, "POST");
    assert!(!recorded.stream);
    assert_eq!(
        recorded.header("AUTHORIZATION"),
        Some("Bearer test-api-key"),
        "header lookup is case-insensitive"
    );
    assert_eq!(recorded.model.as_deref(), Some(MODEL));
    assert_eq!(recorded.response_id.as_deref(), Some("chatcmpl-mock-0001"));
    assert_eq!(recorded.user_texts(), vec!["hi".to_owned()]);

    let chunks = collect_stream(
        &client,
        conversation("hi"),
        request_for(&server).with_stream(true),
    )
    .await;
    assert_eq!(streamed_text(&chunks), "streamed");
    server
        .wait_for_requests(2)
        .await
        .expect("both requests are recorded");

    let requests = server.requests();
    let streamed = requests.last().expect("the streaming request is recorded");
    assert!(streamed.stream, "the recording flags the stream request");
    assert_eq!(streamed.response_id.as_deref(), Some("chatcmpl-mock-0002"));
    assert_eq!(
        server
            .last_request()
            .as_ref()
            .map(|last| last.path.as_str()),
        Some("/v1/chat/completions")
    );

    let base_url = server.base_url();
    assert!(
        base_url.starts_with("http://127.0.0.1:"),
        "loopback base url: {base_url}"
    );
    assert!(base_url.ends_with("/v1"), "versioned base url: {base_url}");
    assert_eq!(server.endpoint(), format!("{base_url}/chat/completions"));
    assert!(server.addr().ip().is_loopback());

    server.shutdown().await;
}

#[tokio::test]
async fn wait_for_requests_timeout_reports_a_timeout_until_a_call_lands() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("hello")]).await;

    let early = server
        .wait_for_requests_timeout(1, Duration::from_millis(20))
        .await
        .expect_err("no request has been made yet");
    assert_eq!(early.kind(), ErrorKind::Timeout);

    let client = client_for(&server);
    client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");

    server
        .wait_for_requests_timeout(1, Duration::from_millis(500))
        .await
        .expect("the request is observed");
    assert_eq!(server.request_count(), 1);

    server.shutdown().await;
}

#[tokio::test]
async fn wait_for_requests_wakes_when_the_request_arrives_later() {
    // Every other test waits *after* the request completed. Here the waiter is
    // already parked when the request lands, so only the recording's notification
    // can wake it. The outer bound turns a lost wake-up into a failure instead of
    // a hung test.
    let completed = tokio::time::timeout(Duration::from_secs(2), async {
        let server = MockLlmServer::with_script([ScriptedResponse::text("late")]).await;
        let client = client_for(&server);
        let request = request_for(&server);

        let responder = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            client
                .invoke(&conversation("late?"), &request)
                .await
                .expect("the mock server answers the late request")
        });

        server
            .wait_for_requests(1)
            .await
            .expect("the late request is recorded");
        assert_eq!(server.request_count(), 1);
        assert_eq!(
            server.last_request().expect("one recording").user_texts(),
            vec!["late?".to_owned()]
        );

        responder.await.expect("the responder task completes");
        server.shutdown().await;
    })
    .await;
    completed.expect("wait_for_requests woke up within the bound");
}

// --- builder ---------------------------------------------------------------

#[tokio::test]
async fn builder_configures_ids_model_and_chunk_delay() {
    let server = MockLlmServer::builder()
        .id_prefix("test-")
        .default_model("gpt-test")
        .item(ScriptedResponse::text("ok"))
        .chunk_delay(Duration::from_millis(5))
        .start()
        .await;
    let client = client_for(&server);

    let response = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");
    assert_eq!(response.id, "chatcmpl-test-0001");
    assert_eq!(response.model, "gpt-test");
    assert_eq!(
        response.message().expect("one choice").text().as_ref(),
        "ok"
    );

    // The per-frame delay must not corrupt the SSE decoding of `complete`.
    server.push(ScriptedResponse::text("delayed"));
    let turn = client
        .complete(&conversation("hi"), &request_for(&server))
        .await
        .expect("delayed frames decode");
    assert_eq!(turn.message.text().as_ref(), "delayed");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));

    server.shutdown().await;
}

#[tokio::test]
async fn builder_script_replaces_the_queue_and_item_appends_to_it() {
    // `MockLlmServerBuilder::script` REPLACES the script, while `item` appends to
    // it. Calling `script` after an `item` therefore discards that item, and any
    // `item` after `script` is appended behind it — the order below pins that.
    let server = MockLlmServer::builder()
        .item(ScriptedResponse::text("dropped"))
        .script([
            ScriptedResponse::text("first"),
            ScriptedResponse::text("second"),
        ])
        .item(ScriptedResponse::text("third"))
        .start()
        .await;
    let client = client_for(&server);
    assert_eq!(
        server.queued(),
        3,
        "script replaced the earlier item; item appended after it"
    );

    for (expected, expected_id) in [
        ("first", "chatcmpl-mock-0001"),
        ("second", "chatcmpl-mock-0002"),
        ("third", "chatcmpl-mock-0003"),
    ] {
        let response = client
            .invoke(&conversation("hi"), &request_for(&server))
            .await
            .expect("the mock server answers");
        assert_eq!(response.id, expected_id);
        assert_eq!(
            response.message().expect("one choice").text().as_ref(),
            expected
        );
    }
    assert_eq!(server.queued(), 0);

    server.shutdown().await;
}

#[tokio::test]
async fn script_helpers_push_append_clear_and_try_start() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("scripted")]).await;
    assert_eq!(server.queued(), 1);
    let client = client_for(&server);

    let response = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the mock server answers");
    assert_eq!(
        response.message().expect("one choice").text().as_ref(),
        "scripted"
    );
    assert_eq!(server.queued(), 0);

    server.push(ScriptedResponse::text("pushed"));
    server.push_many([
        ScriptedResponse::text("first"),
        ScriptedResponse::text("second"),
    ]);
    assert_eq!(server.queued(), 3);

    server.clear_script();
    assert_eq!(server.queued(), 0, "clear_script drops every queued answer");
    let fallback = client
        .invoke(&conversation("hi"), &request_for(&server))
        .await
        .expect("the fallback answers");
    assert_eq!(
        fallback.message().expect("one choice").text().as_ref(),
        "Default mock response"
    );

    let another = MockLlmServer::try_start()
        .await
        .expect("loopback binding succeeds");
    assert!(another.addr().ip().is_loopback());
    assert_ne!(
        another.addr(),
        server.addr(),
        "each server binds its own port"
    );

    another.shutdown().await;
    server.shutdown().await;
}

// --- lifecycle -------------------------------------------------------------

#[tokio::test]
async fn several_servers_are_alive_without_cross_talk() {
    let first = MockLlmServer::with_script([ScriptedResponse::text("from first")]).await;
    let second = MockLlmServer::with_script([
        ScriptedResponse::text("from second 1"),
        ScriptedResponse::text("from second 2"),
    ])
    .await;
    assert_ne!(first.addr(), second.addr());

    let first_client = client_for(&first);
    let second_client = client_for(&second);

    let from_first = first_client
        .invoke(&conversation("first?"), &request_for(&first))
        .await
        .expect("the first server answers");
    assert_eq!(
        from_first.message().expect("one choice").text().as_ref(),
        "from first"
    );

    let from_second = second_client
        .invoke(&conversation("second?"), &request_for(&second))
        .await
        .expect("the second server answers");
    assert_eq!(
        from_second.message().expect("one choice").text().as_ref(),
        "from second 1"
    );

    // The first server's queue is empty and its scripts are untouched by the
    // second server's traffic.
    let first_fallback = first_client
        .invoke(&conversation("again?"), &request_for(&first))
        .await
        .expect("the first server falls back");
    assert_eq!(
        first_fallback
            .message()
            .expect("one choice")
            .text()
            .as_ref(),
        "Default mock response"
    );

    assert_eq!(first.request_count(), 2);
    assert_eq!(second.request_count(), 1);
    assert_eq!(second.queued(), 1);
    assert_eq!(first.requests()[0].user_texts(), vec!["first?".to_owned()]);
    assert_eq!(
        second.requests()[0].user_texts(),
        vec!["second?".to_owned()]
    );

    first.shutdown().await;
    second.shutdown().await;
}

#[tokio::test]
async fn wait_for_requests_fails_within_its_bound_instead_of_hanging() {
    // No request is ever made, so the waiter can never be satisfied: it must
    // report the expected and observed counts inside its default bound instead of
    // parking the suite until the CI timeout. The outer bound fails the test if
    // the helper really waits forever.
    let server = MockLlmServer::with_script([ScriptedResponse::text("unused")]).await;
    let error = tokio::time::timeout(Duration::from_secs(5), server.wait_for_requests(1))
        .await
        .expect("wait_for_requests is bounded by DEFAULT_EVENTUALLY_TIMEOUT")
        .expect_err("no request was ever made");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.message().contains("wait_for_requests(1)"),
        "{}",
        error.message()
    );
    assert!(
        error.message().contains("saw 0 request(s)"),
        "{}",
        error.message()
    );
    server.shutdown().await;
}

#[tokio::test]
async fn a_dropped_server_leaves_no_connection_task_running() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("unused")]).await;
    let connections = server.connections();
    assert_eq!(
        connections.live(),
        0,
        "a fresh server has no connection task"
    );
    // A connection that never finishes its request headers keeps one task parked
    // inside the HTTP server: exactly the task an aborted `axum::serve` used to
    // leave running after the test.
    let mut stalled = TcpStream::connect(server.addr()).await.expect("connect");
    stalled
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: loopback\r\n")
        .await
        .expect("write a partial request");
    eventually(Duration::from_secs(2), || {
        let connections = connections.clone();
        async move { connections.live() == 1 }
    })
    .await
    .expect("the server spawned one connection task");
    drop(server);
    eventually(Duration::from_secs(2), || {
        let connections = connections.clone();
        async move { connections.live() == 0 }
    })
    .await
    .expect("dropping the server aborts every connection task");
}

#[tokio::test]
async fn shutdown_completes_gracefully_after_a_finished_stream() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("streamed")]).await;
    let client = client_for(&server);

    let chunks = collect_stream(
        &client,
        conversation("hi"),
        request_for(&server).with_stream(true),
    )
    .await;
    assert_eq!(streamed_text(&chunks), "streamed", "the stream completed");

    // A drained connection winds the server down through the graceful signal,
    // which is far inside the 5 s abort bound `shutdown` uses as a fallback; a
    // `timeout` here therefore fails if shutdown ever falls back to aborting.
    let stopped = tokio::time::timeout(Duration::from_secs(2), server.shutdown()).await;
    assert!(stopped.is_ok(), "shutdown drained within the bound");
}

// --- raw HTTP surface ------------------------------------------------------

/// Sends one raw HTTP/1.1 request and returns the whole response text.
async fn raw_http(addr: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(addr)
        .await
        .expect("the mock server accepts loopback connections");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("the request is written");
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(1), stream.read_to_end(&mut response))
        .await
        .expect("the mock server answers and closes the connection")
        .expect("the response is readable");
    String::from_utf8(response).expect("the response is UTF-8")
}

/// A `GET` that asks the server to close the connection after the response.
fn get_request(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
}

/// A JSON `POST` that asks the server to close the connection after the response.
fn post_request(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Splits a raw HTTP response into its status line and its body.
fn split_response(response: &str) -> (&str, &str) {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .expect("a response separates headers from body");
    (
        head.lines().next().expect("a response has a status line"),
        body,
    )
}

#[tokio::test]
async fn raw_http_surface_answers_health_404_and_a_malformed_body() {
    let server = MockLlmServer::start().await;
    assert_eq!(server.request_count(), 0);

    let health = raw_http(server.addr(), &get_request("/health")).await;
    assert!(health.starts_with("HTTP/1.1 200"), "health: {health}");
    assert!(health.contains(r#"{"status":"ok"}"#), "health: {health}");

    let unknown = raw_http(server.addr(), &get_request("/nope")).await;
    assert!(unknown.starts_with("HTTP/1.1 404"), "unknown: {unknown}");
    assert!(
        unknown.contains(r#""type":"invalid_request_error""#),
        "unknown: {unknown}"
    );
    assert_eq!(
        server.request_count(),
        0,
        "health probes and 404s are not model calls"
    );

    let body = "{not-json";
    let malformed = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let rejected = raw_http(server.addr(), &malformed).await;
    assert!(rejected.starts_with("HTTP/1.1 400"), "rejected: {rejected}");
    assert!(
        rejected.contains("invalid request body"),
        "rejected: {rejected}"
    );
    assert_eq!(server.request_count(), 1, "the bad request is recorded");
    let recorded = server.last_request().expect("the bad request is recorded");
    assert!(
        recorded.response_id.is_none(),
        "a 400 stamps no response id"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn chat_completions_alias_answers_exactly_like_the_versioned_path() {
    // The script is empty, so both requests are answered by the same fallback and
    // reuse the same id: the two bodies must therefore be byte-identical, which
    // proves the alias is a real route rather than a differently-shaped reply.
    let server = MockLlmServer::builder()
        .fallback(ScriptedResponse::text("aliased"))
        .start()
        .await;
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    })
    .to_string();

    let versioned = raw_http(server.addr(), &post_request("/v1/chat/completions", &body)).await;
    let alias = raw_http(server.addr(), &post_request("/chat/completions", &body)).await;

    let (versioned_status, versioned_body) = split_response(&versioned);
    let (alias_status, alias_body) = split_response(&alias);
    assert!(versioned_status.starts_with("HTTP/1.1 200"), "{versioned}");
    assert!(alias_status.starts_with("HTTP/1.1 200"), "{alias}");
    assert!(
        alias_body.contains(r#""content":"aliased""#),
        "the alias serves the scripted content: {alias}"
    );
    assert_eq!(
        alias_body, versioned_body,
        "both paths serialize the same chat-completion JSON"
    );

    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].path, "/v1/chat/completions");
    assert_eq!(requests[1].path, "/chat/completions");
    assert_eq!(requests[1].method, "POST");

    server.shutdown().await;
}

#[tokio::test]
async fn a_wrong_method_is_a_json_404_that_consumes_nothing() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("unused")]).await;
    let response = raw_http(server.addr(), &get_request("/v1/chat/completions")).await;
    let (status, body) = split_response(&response);

    assert!(status.starts_with("HTTP/1.1 404"), "{response}");
    let body: Value = serde_json::from_str(body).expect("the 404 body is JSON");
    assert!(
        body["error"]["message"].is_string(),
        "the envelope carries a message: {body}"
    );
    assert_eq!(body["error"]["type"], "invalid_request_error");

    // A rejected method is not a model call: no script entry was popped and no
    // recording was appended.
    assert_eq!(server.queued(), 1);
    assert_eq!(server.request_count(), 0);
    assert!(server.last_request().is_none());

    server.shutdown().await;
}
