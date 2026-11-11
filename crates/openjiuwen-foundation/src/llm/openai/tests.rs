//! End-to-end tests of the OpenAI-compatible client against a local mock server.
//!
//! Every test talks to a [`wiremock::MockServer`] on loopback: no real network
//! access and no real credential is ever needed.

use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::http::{HttpClient, HttpClientConfig, RetryPolicy};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{convert_tool_choice, convert_tools, normalize_api_base, OpenAiCompatibleClient};
use crate::llm::config::{ModelClientConfig, ModelRequestConfig};
use crate::llm::model::{FinishReason, Model, ModelExt, StreamAccumulator, StreamChunk};
use crate::prompt::{BaseMessage, Usage};

/// The model every test asks for.
const MODEL: &str = "gpt-4o";

/// Builds a client pointed at the mock server's `/v1` root.
fn client(server: &MockServer) -> OpenAiCompatibleClient {
    OpenAiCompatibleClient::new(
        ModelClientConfig::new("openai")
            .with_api_key("test-key")
            .with_api_base(format!("{}/v1", server.uri())),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the test client builds")
}

/// The request payload every non-streaming test sends.
fn request() -> ModelRequestConfig {
    ModelRequestConfig::new(MODEL)
}

/// A canned non-streaming response.
fn completion_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    })
}

#[tokio::test]
async fn invoke_posts_a_completion_and_parses_the_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let response = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect("the call succeeds");

    assert_eq!(response.id, "chatcmpl-1");
    assert_eq!(response.model, MODEL);
    assert_eq!(response.message().expect("one choice").text(), "hi there");
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    assert_eq!(response.usage, Usage::new(5, 2));
    server.verify().await;
}

#[tokio::test]
async fn invoke_reports_the_client_identity() {
    let server = MockServer::start().await;
    let client = client(&server);

    assert_eq!(client.provider(), "openai");
    assert_eq!(client.model_name(), MODEL);
    assert_eq!(client.base_url(), format!("{}/v1", server.uri()));
    assert_eq!(client.request_config().model_name, MODEL);
    assert_eq!(
        client
            .client_config()
            .api_key
            .as_ref()
            .map(openjiuwen_common::security::SecretString::expose_secret),
        Some("test-key")
    );
}

#[tokio::test]
async fn stream_assembles_an_event_stream_into_a_turn() {
    let server = MockServer::start().await;
    let events = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(events.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("every chunk decodes");
        if matches!(chunk, StreamChunk::Done) {
            chunks.push(chunk);
            // The stream must not produce a second `Done`.
            assert!(stream.next().await.is_none(), "the stream ends after Done");
            break;
        }
        chunks.push(chunk);
    }

    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| matches!(chunk, StreamChunk::Done))
            .count(),
        1
    );
    let usage = chunks
        .iter()
        .find_map(|chunk| match chunk {
            StreamChunk::Usage(usage) => Some(usage.clone()),
            _ => None,
        })
        .expect("usage arrives on the stream");
    assert_eq!(usage, Usage::new(5, 2));

    let mut accumulator = StreamAccumulator::new();
    for chunk in &chunks {
        accumulator.push(chunk);
    }
    let turn = accumulator.finish();
    assert_eq!(turn.message.text(), "Hello");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    assert_eq!(turn.usage, Usage::new(5, 2));
    server.verify().await;
}

#[tokio::test]
async fn complete_drives_the_stream_to_a_turn() {
    let server = MockServer::start().await;
    let events = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(events.as_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = client(&server);
    let turn = client
        .complete(&[BaseMessage::user("hello")], &request())
        .await
        .expect("the stream completes");

    assert_eq!(turn.message.text(), "ok");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn a_transient_500_is_retried_and_then_succeeds() {
    assert!(
        RetryPolicy::default().is_retryable_status(500),
        "500 must be retryable by default"
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream hiccup"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .mount(&server)
        .await;

    let transport = HttpClient::new(
        HttpClientConfig::default()
            .with_max_retries(1)
            .with_retry_backoff(Duration::from_millis(1), Duration::from_millis(1)),
    )
    .expect("the transport builds");
    let client = OpenAiCompatibleClient::new(
        ModelClientConfig::new("openai")
            .with_api_key("test-key")
            .with_api_base(format!("{}/v1", server.uri()))
            .with_max_retries(1),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the client builds")
    .with_http_client(transport);

    let response = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect("the retry succeeds");
    assert_eq!(response.message().expect("one choice").text(), "hi there");

    let received = server.received_requests().await.expect("recorded requests");
    assert_eq!(received.len(), 2, "the 500 was retried exactly once");
}

#[tokio::test]
async fn an_authentication_failure_is_a_config_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(json!({"error": {"message": "invalid api key"}})),
        )
        .mount(&server)
        .await;

    let error = client(&server)
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("401 must fail");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("authentication failed"),
        "{error}"
    );
    assert!(!error.to_string().contains("invalid api key"), "{error}");
}

#[tokio::test]
async fn a_malformed_success_body_is_an_llm_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("this is not json"))
        .mount(&server)
        .await;

    let error = client(&server)
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("malformed JSON must fail");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("decoding"), "{error}");
}

#[tokio::test]
async fn a_non_success_stream_response_is_reported_as_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("no key"))
        .mount(&server)
        .await;

    let client = client(&server);
    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let error = stream
        .next()
        .await
        .expect("one error item")
        .expect_err("401 must fail the stream");

    assert_eq!(error.kind(), ErrorKind::Config);
}

/// Starts a loopback server that answers a POST with SSE headers, writes
/// `initial_body` (if any), and then leaves the connection open without sending
/// anything else.
///
/// This is the only way to exercise the stream budgets offline: `wiremock` always
/// writes a complete body, while these tests need a body that stalls mid-stream.
fn stall_after(initial_body: Option<&'static str>) -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("binds a loopback port");
    let address = listener.local_addr().expect("the port is bound");
    listener
        .set_nonblocking(true)
        .expect("the listener becomes non-blocking");
    let listener = tokio::net::TcpListener::from_std(listener).expect("tokio takes the listener");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one connection");
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                  connection: close\r\n\r\n",
            )
            .await
            .expect("writes the headers");
        if let Some(body) = initial_body {
            socket
                .write_all(body.as_bytes())
                .await
                .expect("writes the first event");
        }
        // Keep the connection open, without a body, until the runtime is dropped.
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    address
}

fn stalling_client(
    address: std::net::SocketAddr,
    first: Duration,
    idle: Duration,
) -> OpenAiCompatibleClient {
    OpenAiCompatibleClient::new(
        ModelClientConfig::new("openai")
            .with_api_key("test-key")
            .with_api_base(format!("http://{address}/v1"))
            .with_stream_timeouts(Some(first), Some(idle)),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the client builds")
}

#[tokio::test]
async fn a_stream_that_never_produces_content_times_out() {
    let address = stall_after(None);
    let client = stalling_client(address, Duration::from_millis(200), Duration::from_secs(10));

    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let error = stream
        .next()
        .await
        .expect("one error item")
        .expect_err("the first content delta never arrives");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("no content"), "{error}");
}

#[tokio::test]
async fn a_stream_that_stalls_after_the_first_delta_times_out() {
    let address = stall_after(Some(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
    ));
    let client = stalling_client(address, Duration::from_secs(10), Duration::from_millis(200));

    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let first = stream
        .next()
        .await
        .expect("one delta")
        .expect("the first delta decodes");
    assert_eq!(first.content(), Some("hi"));

    let error = stream
        .next()
        .await
        .expect("one error item")
        .expect_err("the stream went idle");

    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("idle"), "{error}");
}

#[tokio::test]
async fn the_stream_stops_reading_at_done() {
    let address = stall_after(Some("data: [DONE]\n\n"));
    let client = stalling_client(address, Duration::from_secs(10), Duration::from_secs(10));

    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let done = stream.next().await.expect("the sentinel arrives");
    assert_eq!(done.expect("the sentinel decodes"), StreamChunk::Done);
    assert!(
        stream.next().await.is_none(),
        "a connection that stays open after [DONE] must not block the caller"
    );
}

#[tokio::test]
async fn a_blank_provider_resolves_to_openai() {
    let server = MockServer::start().await;
    let client = OpenAiCompatibleClient::new(
        ModelClientConfig::new("")
            .with_api_key("test-key")
            .with_api_base(format!("{}/v1", server.uri())),
        ModelRequestConfig::new(MODEL),
    )
    .expect("a blank provider means openai");

    assert_eq!(client.provider(), "openai");
}

#[tokio::test]
async fn a_provider_on_another_protocol_is_rejected() {
    let error = OpenAiCompatibleClient::new(
        ModelClientConfig::new("anthropic").with_api_key("test-key"),
        ModelRequestConfig::new("claude-3-5-sonnet"),
    )
    .expect_err("anthropic is not served by this client");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("Anthropic"), "{error}");
}

#[tokio::test]
async fn an_unknown_provider_is_rejected() {
    let error = OpenAiCompatibleClient::new(
        ModelClientConfig::new("acme").with_api_key("test-key"),
        ModelRequestConfig::new(MODEL),
    )
    .expect_err("acme is not built in");

    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn an_api_base_that_is_a_full_endpoint_is_normalized() {
    let server = MockServer::start().await;
    let client = OpenAiCompatibleClient::new(
        ModelClientConfig::new("openai")
            .with_api_key("test-key")
            .with_api_base(format!("{}/v1/chat/completions", server.uri())),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the client builds");

    assert_eq!(client.base_url(), format!("{}/v1", server.uri()));
    assert_eq!(
        normalize_api_base(&format!("{}/v1/chat/completions", server.uri())),
        format!("{}/v1", server.uri())
    );
}

#[tokio::test]
async fn tools_and_tool_choices_reach_the_wire_verbatim() {
    let server = MockServer::start().await;
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Reads the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    })];
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": false,
            "tools": convert_tools(&tools),
            "tool_choice": "auto"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
        .expect(1)
        .mount(&server)
        .await;

    let request = request().with_tools(tools.clone());
    assert_eq!(convert_tool_choice(&json!("auto")), json!("auto"));

    client(&server)
        .invoke(&[BaseMessage::user("weather?")], &request)
        .await
        .expect("the call succeeds");
    server.verify().await;
}
