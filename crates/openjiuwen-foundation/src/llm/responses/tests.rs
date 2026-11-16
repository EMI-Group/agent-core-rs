//! End-to-end tests of the `OpenAI` Responses client against a local mock server.
//!
//! Every HTTP test talks to a [`wiremock::MockServer`] on loopback: no real network
//! access and no real credential is ever needed. The provider-selection tests call
//! the real factory, so they also prove the `openai` → Responses dispatch and the
//! `openai_account` profile.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{
    build_responses_body, InMemoryResponsesSessionStore, OpenAiResponsesClient,
    ResponsesSessionStore,
};
use crate::llm::config::{ApiMode, AuthMode, ModelClientConfig, ModelRequestConfig};
use crate::llm::factory::init_model;
use crate::llm::model::{FinishReason, Model, StreamAccumulator, StreamChunk};
use crate::prompt::{BaseMessage, ToolCall, Usage};

/// The model every test asks for.
const MODEL: &str = "gpt-5";

/// Builds a client pointed at the mock server root.
fn client(server: &MockServer) -> OpenAiResponsesClient {
    OpenAiResponsesClient::new(
        ModelClientConfig::new("openai_account")
            .with_api_key("test-key")
            .with_api_base(server.uri()),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the test client builds")
}

/// The per-call request every test sends.
fn request() -> ModelRequestConfig {
    ModelRequestConfig::new(MODEL)
}

/// A minimal event stream that answers with `text`.
fn text_stream(text: &str) -> String {
    format!(
        concat!(
            "event: response.created\n",
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\"}}}}\n\n",
            "event: response.output_text.delta\n",
            "data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{text}\"}}\n\n",
            "event: response.completed\n",
            "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",",
            "\"model\":\"{model}\",\"usage\":{{\"input_tokens\":5,\"output_tokens\":2,",
            "\"total_tokens\":7}}}}}}\n\n",
        ),
        text = text,
        model = MODEL,
    )
}

/// The body a simple user message produces, including the forced streaming flag.
fn simple_body() -> serde_json::Value {
    body_for("hello")
}

/// The body one user message with `text` produces, including the streaming flag.
fn body_for(text: &str) -> serde_json::Value {
    json!({
        "model": MODEL,
        "instructions": "",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "store": false,
        "stream": true,
    })
}

/// A mock that answers one `POST /responses` call with `text_stream(text)`.
async fn mount_text_stream(server: &MockServer, text: &str) {
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream(text).as_bytes(), "text/event-stream"),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn invoke_posts_a_responses_request_and_aggregates_the_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("accept", "application/json"))
        .and(header("content-type", "application/json"))
        .and(body_json(simple_body()))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("hi there").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let response = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect("the call succeeds");

    assert_eq!(response.model, MODEL);
    assert_eq!(response.message().expect("one choice").text(), "hi there");
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    assert_eq!(response.usage, Usage::new(5, 2));
    server.verify().await;
}

#[tokio::test]
async fn the_client_reports_its_identity_and_endpoint() {
    let server = MockServer::start().await;
    let client = client(&server);

    assert_eq!(client.provider(), "openai_account");
    assert_eq!(client.model_name(), MODEL);
    assert_eq!(client.base_url(), server.uri());
    assert_eq!(client.chat_path(), "/responses");
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
async fn a_pasted_responses_suffix_is_stripped_from_the_base() {
    let server = MockServer::start().await;
    let client = OpenAiResponsesClient::new(
        ModelClientConfig::new("openai_account")
            .with_api_key("test-key")
            .with_api_base(format!("{}/responses/", server.uri())),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the client builds");

    assert_eq!(client.base_url(), server.uri());
}

#[tokio::test]
async fn session_ids_become_request_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("session_id", "sess-42"))
        .and(header("x-client-request-id", "sess-42"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("ok").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let request = request().with_extra(json!({"session_id": "sess-42"}));
    let response = client
        .invoke(&[BaseMessage::user("hello")], &request)
        .await
        .expect("the call succeeds");
    assert_eq!(response.message().expect("one choice").text(), "ok");
    server.verify().await;
}

#[tokio::test]
async fn stream_assembles_text_reasoning_and_usage() {
    let server = MockServer::start().await;
    let events = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\"}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"thinking\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
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
        let done = matches!(chunk, StreamChunk::Done);
        chunks.push(chunk);
        if done {
            assert!(stream.next().await.is_none(), "the stream ends after Done");
            break;
        }
    }

    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| matches!(chunk, StreamChunk::Done))
            .count(),
        1
    );
    let mut accumulator = StreamAccumulator::new();
    for chunk in &chunks {
        accumulator.push(chunk);
    }
    let turn = accumulator.finish();
    assert_eq!(turn.message.text(), "Hello");
    assert_eq!(turn.message.reasoning_content.as_deref(), Some("thinking"));
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    assert_eq!(turn.usage, Usage::new(5, 2));
    server.verify().await;
}

#[tokio::test]
async fn a_streamed_function_call_becomes_one_tool_call() {
    let server = MockServer::start().await;
    let events = concat!(
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"city\\\":\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"\\\"Paris\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":9,\"output_tokens\":4}}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(events.as_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = client(&server);
    let response = client
        .invoke(&[BaseMessage::user("weather?")], &request())
        .await
        .expect("the call succeeds");

    let message = response.message().expect("one choice");
    assert_eq!(message.tool_calls.len(), 1);
    assert_eq!(message.tool_calls[0].id.as_deref(), Some("call_1"));
    assert_eq!(message.tool_calls[0].name, "get_weather");
    assert_eq!(message.tool_calls[0].arguments, "{\"city\":\"Paris\"}");
    assert_eq!(response.finish_reason(), Some(&FinishReason::ToolCalls));
    assert_eq!(response.usage, Usage::new(9, 4));
}

#[tokio::test]
async fn a_failed_event_fails_the_stream() {
    let server = MockServer::start().await;
    let events =
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"boom\"}}}\n\n";
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(events.as_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = client(&server);
    let error = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("a failed response is an error");
    assert_eq!(error.kind(), ErrorKind::Llm);
}

#[tokio::test]
async fn a_malformed_frame_is_skipped_and_the_stream_still_completes() {
    let server = MockServer::start().await;
    let events = concat!(
        "data: {not json\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(events.as_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = client(&server);
    let response = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect("the malformed frame is skipped");
    assert_eq!(response.message().expect("one choice").text(), "ok");
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
}

#[tokio::test]
async fn unauthorized_and_server_errors_map_to_config_and_llm() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": {"message": "invalid token"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let unauthorized = client(&server);
    let error = unauthorized
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("401 is an error");
    assert_eq!(error.kind(), ErrorKind::Config);
    server.verify().await;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"detail": "exploded"})))
        .expect(1)
        .mount(&server)
        .await;

    let failing = client(&server);
    let error = failing
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("500 is an error");
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(
        !error.to_string().contains("exploded"),
        "provider error bodies are redacted"
    );
    server.verify().await;
}

#[tokio::test]
async fn a_timeout_status_maps_to_the_timeout_kind() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(504))
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let error = client
        .invoke(&[BaseMessage::user("hello")], &request())
        .await
        .expect_err("504 is an error");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("failed with status 504"));
}

#[tokio::test]
async fn a_history_with_tool_calls_is_projected_into_input_items() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(json!({
            "model": MODEL,
            "instructions": "be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "id": "call_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                {"role": "user", "content": [{"type": "input_text", "text": "thanks"}]},
            ],
            "store": false,
            "stream": true,
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("done").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let messages = [
        BaseMessage::system("be brief"),
        BaseMessage::user("weather?"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall {
            id: Some("call_1".to_owned()),
            name: "get_weather".to_owned(),
            arguments: "{\"city\":\"Paris\"}".to_owned(),
            ..ToolCall::default()
        }]),
        BaseMessage::tool("call_1", "sunny"),
        BaseMessage::user("thanks"),
    ];
    let client = client(&server);
    client
        .invoke(&messages, &request())
        .await
        .expect("the call succeeds");
    server.verify().await;
}

#[tokio::test]
async fn init_model_dispatches_openai_with_api_mode_responses() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("routed").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = ModelClientConfig::new("openai")
        .with_api_key("test-key")
        .with_api_base(format!("{}/v1", server.uri()))
        .with_api_mode(ApiMode::Responses);
    let model = init_model(client, ModelRequestConfig::new(MODEL))
        .await
        .expect("the factory builds a Responses client");
    assert_eq!(model.provider(), "openai");

    let response = model
        .invoke(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(MODEL),
        )
        .await
        .expect("the routed call succeeds");
    assert_eq!(response.message().expect("one choice").text(), "routed");
    server.verify().await;
}

#[tokio::test]
async fn the_openai_account_provider_uses_the_default_responses_base() {
    let _guard = env_lock();
    let _restore = EnvRestore::clear(&[
        "API_BASE",
        "OPENAI_BASE_URL",
        "API_KEY",
        "OPENAI_API_KEY",
        "OPENJIUWEN_API_KEY",
    ]);

    let client = OpenAiResponsesClient::new(
        ModelClientConfig::new("openai_account").with_auth_mode(AuthMode::OpenAiAccountOAuth),
        ModelRequestConfig::new(MODEL),
    )
    .expect("the default base resolves");
    assert_eq!(client.provider(), "openai_account");
    assert_eq!(client.base_url(), "https://chatgpt.com/backend-api/codex");
    assert_eq!(client.chat_path(), "/responses");
    assert_eq!(
        client.client_config().provider,
        "openai_account",
        "the alias normalizes to the canonical name"
    );
}

#[tokio::test]
async fn a_chat_completions_provider_cannot_build_a_responses_client() {
    let error = OpenAiResponsesClient::new(
        ModelClientConfig::new("qwen")
            .with_api_key("test-key")
            .with_api_base("https://example.test/v1"),
        ModelRequestConfig::new(MODEL),
    )
    .expect_err("qwen does not speak Responses");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("Responses"));
}

#[tokio::test]
async fn max_tool_calls_is_forwarded_to_the_body() {
    let server = MockServer::start().await;
    let mut expected = simple_body();
    expected["max_tool_calls"] = json!(3);
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(expected))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("ok").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server);
    let response = client
        .invoke(
            &[BaseMessage::user("hello")],
            &request().with_extra(json!({"max_tool_calls": 3})),
        )
        .await
        .expect("the call succeeds");
    assert_eq!(response.message().expect("one choice").text(), "ok");
    server.verify().await;
}

#[tokio::test]
async fn an_invalid_max_tool_calls_is_a_validation_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("never").as_bytes(), "text/event-stream"),
        )
        .expect(0)
        .mount(&server)
        .await;

    let client = client(&server);
    for value in [json!(0), json!(-2), json!(2.5), json!("many"), json!(true)] {
        let error = client
            .invoke(
                &[BaseMessage::user("hello")],
                &request().with_extra(json!({"max_tool_calls": value})),
            )
            .await
            .expect_err("an invalid limit fails before the request");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("max_tool_calls"),
            "the error names the field: {error}"
        );
    }
    server.verify().await;
}

#[tokio::test]
async fn a_follow_up_call_sends_the_stored_previous_response_id() {
    let server = MockServer::start().await;
    let request = request().with_extra(json!({"conversation_id": "conv-7"}));
    let client = client(&server);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(simple_body()))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("first").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let first = client
        .invoke(&[BaseMessage::user("hello")], &request)
        .await
        .expect("the first call succeeds");
    assert_eq!(first.id, "resp_1", "the response id is captured");

    let mut expected = body_for("again");
    expected["previous_response_id"] = json!("resp_1");
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(expected))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("second").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let second = client
        .invoke(&[BaseMessage::user("again")], &request)
        .await
        .expect("the follow-up call succeeds");
    assert_eq!(second.message().expect("one choice").text(), "second");
    server.verify().await;
}

#[tokio::test]
async fn an_explicit_previous_response_id_overrides_the_store() {
    let server = MockServer::start().await;
    let store = Arc::new(InMemoryResponsesSessionStore::new());
    store.remember_response_id("conv-9", "resp_stored");

    let mut expected = simple_body();
    expected["previous_response_id"] = json!("resp_explicit");
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_json(expected))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(text_stream("ok").as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client(&server).with_session_store(store);
    client
        .invoke(
            &[BaseMessage::user("hello")],
            &request().with_extra(json!({
                "conversation_id": "conv-9",
                "previous_response_id": "resp_explicit",
            })),
        )
        .await
        .expect("the explicit id wins over the stored one");
    server.verify().await;
}

#[tokio::test]
async fn a_streamed_response_records_its_id_for_the_configured_key() {
    let server = MockServer::start().await;
    mount_text_stream(&server, "streamed").await;

    let client = client(&server).with_conversation_key("conv-stream");
    assert_eq!(client.conversation_key(), Some("conv-stream"));

    let mut stream = client.stream(
        vec![BaseMessage::user("hello")],
        request().with_stream(true),
    );
    let mut done = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("every chunk decodes");
        if matches!(chunk, StreamChunk::Done) {
            done = true;
            assert!(
                stream.next().await.is_none(),
                "the stream ends after recording the conversation state"
            );
            break;
        }
    }
    assert!(done, "the stream always terminates");
    assert_eq!(
        client
            .session_store()
            .last_response_id("conv-stream")
            .as_deref(),
        Some("resp_1"),
        "the streamed path records the answered response id"
    );
}

#[test]
fn the_client_side_hints_never_reach_the_body() {
    let request = request().with_extra(json!({
        "conversation_id": "conv-1",
        "max_tool_calls": 4,
        "previous_response_id": "resp_1",
    }));
    let body = build_responses_body(&[BaseMessage::user("hello")], &request, true)
        .expect("the body builds");

    assert!(
        body.get("conversation_id").is_none(),
        "the conversation key is a client-side hint: {body}"
    );
    assert_eq!(body["max_tool_calls"], json!(4));
    assert_eq!(body["previous_response_id"], json!("resp_1"));
}

#[test]
fn the_body_omits_the_conversation_fields_by_default() {
    let body = build_responses_body(&[BaseMessage::user("hello")], &request(), true)
        .expect("the body builds");
    for key in ["max_tool_calls", "previous_response_id", "conversation_id"] {
        assert!(
            body.get(key).is_none(),
            "{key} is present only when the request configures it"
        );
    }
}

#[test]
fn a_hint_nested_in_extra_body_is_not_forwarded_either() {
    let request = request().with_extra(json!({
        "extra_body": {"conversation_id": "conv-1", "safety_identifier": "user-7"},
    }));
    let body = build_responses_body(&[BaseMessage::user("hello")], &request, true)
        .expect("the body builds");

    assert!(body.get("conversation_id").is_none());
    assert_eq!(body["safety_identifier"], json!("user-7"));
}

#[test]
fn background_mode_and_hosted_tools_are_typed_unsupported() {
    let background = request().with_extra(json!({"background": true}));
    let error = build_responses_body(&[BaseMessage::user("hello")], &background, true)
        .expect_err("background mode needs the live service");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("live OpenAI Responses service"));

    let hosted = request().with_tools(vec![json!({"type": "web_search_preview"})]);
    let error = build_responses_body(&[BaseMessage::user("hello")], &hosted, true)
        .expect_err("hosted tools need the live service");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("web_search_preview"));
    assert!(error.to_string().contains("live OpenAI Responses service"));

    let disabled = request().with_extra(json!({"background": false}));
    let body = build_responses_body(&[BaseMessage::user("hello")], &disabled, true)
        .expect("`background: false` is not a request for background mode");
    assert_eq!(body["background"], json!(false));
}

/// Serializes tests that mutate the process environment.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Removes environment variables for the guard's lifetime and restores them.
struct EnvRestore {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvRestore {
    fn clear(names: &[&'static str]) -> Self {
        let saved = names
            .iter()
            .map(|name| {
                let previous = std::env::var(name).ok();
                std::env::remove_var(name);
                (*name, previous)
            })
            .collect();
        Self { saved }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, previous) in &self.saved {
            match previous {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}
