//! Wire-level tests for [`AnthropicClient`].
//!
//! Every HTTP call is served by `wiremock`; the tests assert the request path,
//! the wire headers (`x-api-key`, `anthropic-version`, never `Authorization`),
//! the projected body, and the mapping of provider failures onto the shared
//! error kinds. No test needs a network, a credential, or a Claude model.

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;

use super::*;
use crate::llm::config::{ModelClientConfig, ModelRequestConfig};
use crate::llm::model::{FinishReason, Model, ModelExt, StreamChunk};
use crate::prompt::{BaseMessage, Usage};

const MODEL: &str = "claude-sonnet-4";

/// Builds a client pointed at the mock server with a fixed test key.
fn build_client(server: &MockServer, request: ModelRequestConfig) -> AnthropicClient {
    let config = ModelClientConfig::new("anthropic")
        .with_api_key("test-key")
        .with_api_base(server.uri());
    AnthropicClient::new(config, request).expect("the test client builds")
}

/// A canned Messages response with text, a tool call, and cache-aware usage.
fn canned_response() -> serde_json::Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 2,
            "output_tokens": 4,
        },
    })
}

/// An Anthropic event stream for the text `hello world`.
fn canned_event_stream() -> String {
    concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello \"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    )
    .to_owned()
}

#[tokio::test]
async fn invoke_posts_the_projected_body_with_the_anthropic_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32768,
            "stream": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    assert_eq!(client.base_url(), server.uri());

    let request = ModelRequestConfig::new(MODEL);
    let response = client
        .invoke(&[BaseMessage::user("hello")], &request)
        .await
        .expect("the response parses");

    assert_eq!(response.id, "msg_1");
    assert_eq!(response.model, MODEL);
    let message = response.message().expect("one choice");
    assert_eq!(message.text(), "checking");
    assert_eq!(message.tool_calls.len(), 1);
    assert_eq!(message.tool_calls[0].name, "get_weather");
    assert_eq!(message.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
    assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    assert_eq!(response.usage.input_tokens, 12);
    assert_eq!(response.usage.output_tokens, 4);
    assert_eq!(response.usage.total_tokens, 16);
    assert_eq!(response.usage.cache_tokens, 2);
}

#[tokio::test]
async fn a_messages_suffix_in_the_api_base_is_not_duplicated() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let config = ModelClientConfig::new("anthropic")
        .with_api_key("test-key")
        .with_api_base(format!("{}/v1/messages", server.uri()));
    let client = AnthropicClient::new(config, ModelRequestConfig::new(MODEL)).expect("client");
    assert_eq!(client.base_url(), server.uri());

    let response = client
        .invoke(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect("the request succeeds");
    assert_eq!(response.message().expect("one choice").text(), "ok");
}

#[tokio::test]
async fn stream_assembles_the_event_stream_into_chunks() {
    let server = MockServer::start().await;
    let body = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello \"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        ModelRequestConfig::new(MODEL),
    );
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("every chunk decodes"));
    }

    assert_eq!(chunks.len(), 7, "chunks: {chunks:?}");
    assert_eq!(
        chunks[0],
        StreamChunk::Delta {
            index: 0,
            role: Some(crate::prompt::Role::Assistant),
            content: None,
            reasoning: None,
            tool_call: None,
            finish_reason: None,
        }
    );
    match &chunks[1] {
        StreamChunk::Usage(usage) => {
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 0);
        }
        other => panic!("expected the prompt usage, got {other:?}"),
    }
    assert_eq!(chunks[2].content(), Some("hello "));
    assert_eq!(chunks[3].content(), Some("world"));

    let finish = match &chunks[4] {
        StreamChunk::Delta { finish_reason, .. } => finish_reason.clone(),
        other => panic!("expected a finish delta, got {other:?}"),
    };
    assert_eq!(finish, Some(FinishReason::Stop));

    match &chunks[5] {
        StreamChunk::Usage(usage) => {
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 4);
            assert_eq!(usage.total_tokens, 14);
        }
        other => panic!("expected a usage chunk, got {other:?}"),
    }
    assert_eq!(chunks[6], StreamChunk::Done);
    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| matches!(chunk, StreamChunk::Done))
            .count(),
        1
    );
}

#[tokio::test]
async fn complete_returns_the_accumulated_turn_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(canned_event_stream()),
        )
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let turn = client
        .complete(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect("the stream completes");

    assert_eq!(turn.message.text(), "hello world");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    assert_eq!(turn.usage, Usage::new(10, 4));
}

#[tokio::test]
async fn unauthorized_responses_are_config_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("{\"error\":\"invalid key\"}"))
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let error = client
        .invoke(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect_err("a 401 is an authentication failure");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("authentication failed"));
}

#[tokio::test]
async fn malformed_success_bodies_are_llm_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let error = client
        .invoke(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect_err("the body cannot be decoded");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error
        .to_string()
        .contains("decoding the Anthropic response"));
}

#[tokio::test]
async fn a_non_object_success_body_is_an_llm_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!(["not", "an", "object"])))
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let error = client
        .invoke(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect_err("the payload cannot be a response");

    assert_eq!(error.kind(), ErrorKind::Llm);
}

#[tokio::test]
async fn a_stream_error_event_is_an_llm_error() {
    let server = MockServer::start().await;
    let body =
        "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try later\"}}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = build_client(&server, ModelRequestConfig::new(MODEL));
    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        ModelRequestConfig::new(MODEL),
    );
    let mut error = None;
    while let Some(item) = stream.next().await {
        if let Err(failure) = item {
            error = Some(failure);
            break;
        }
    }

    let error = error.expect("the stream must surface the provider error");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.to_string().contains("overloaded_error"));
}

#[test]
fn a_provider_mismatch_is_a_config_error() {
    let error = AnthropicClient::new(
        ModelClientConfig::new("openai").with_api_key("sk-test"),
        ModelRequestConfig::new("gpt-4o"),
    )
    .expect_err("openai is not an Anthropic endpoint");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("Anthropic Messages API"));
}

#[test]
fn an_empty_provider_defaults_to_anthropic() {
    let config = ModelClientConfig {
        provider: String::new(),
        api_key: Some(SecretString::new("test-key")),
        ..ModelClientConfig::default()
    };

    let client = AnthropicClient::new(config, ModelRequestConfig::new(MODEL)).expect("client");
    assert_eq!(client.provider(), "anthropic");
    assert_eq!(client.base_url(), "https://api.anthropic.com");
    assert_eq!(client.client_config().provider, "anthropic");
    assert_eq!(client.request_config().model_name, MODEL);
    assert_eq!(client.model_name(), MODEL);
}

#[test]
fn the_accessors_expose_the_resolved_configuration() {
    let mut config = ModelClientConfig::new("Claude").with_api_key("test-key");
    config.api_base = Some("https://example.test/v1/".to_owned());
    let client = AnthropicClient::new(config, ModelRequestConfig::new(MODEL)).expect("client");

    assert_eq!(client.client_config().provider, "anthropic");
    assert_eq!(client.base_url(), "https://example.test");
    assert_eq!(client.request_config().model_name, MODEL);
}
