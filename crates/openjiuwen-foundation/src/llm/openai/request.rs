//! OpenAI-compatible request projection: base URL, messages, tools, and body.
//!
//! Ports `_normalize_openai_base_url`, the message dictionary conversion of
//! `base_model_client.py`, and the `DeepSeek` reasoning replay of
//! `utils/endpoint_profiles.py`. Everything here is pure, so the exact JSON a
//! provider would receive is asserted in tests without a server.

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Map, Value};

use crate::prompt::tool_call::ToolCall;
use crate::prompt::BaseMessage;

use super::super::config::ModelRequestConfig;
use super::super::provider::normalize_provider_name;

/// Suffix a caller may have left on `api_base` by pasting a full endpoint.
const CHAT_COMPLETIONS_SUFFIX: &str = "/chat/completions";

/// The model name DeepSeek-like endpoints require on every assistant turn.
const DEEPSEEK_MARKER: &str = "deepseek";

/// Removes artifacts from a configured `api_base` so endpoint paths can be
/// appended to it.
///
/// Whitespace and trailing slashes are trimmed, and a value that already ends in
/// `/chat/completions` loses that suffix, so a caller who pasted the full
/// endpoint does not produce `.../chat/completions/chat/completions`. No `/v1`
/// is ever invented: self-hosted gateways frequently live under another prefix,
/// and the provider table is responsible for the default base URL.
///
/// ```
/// use openjiuwen_foundation::llm::openai::normalize_api_base;
///
/// assert_eq!(
///     normalize_api_base(" https://api.openai.com/v1/ "),
///     "https://api.openai.com/v1"
/// );
/// assert_eq!(
///     normalize_api_base("https://api.openai.com/v1/chat/completions"),
///     "https://api.openai.com/v1"
/// );
/// assert_eq!(normalize_api_base("   "), "");
/// ```
#[must_use]
pub fn normalize_api_base(api_base: &str) -> String {
    let base = api_base.trim().trim_end_matches('/');
    match base.strip_suffix(CHAT_COMPLETIONS_SUFFIX) {
        Some(stripped) => stripped.trim_end_matches('/').to_owned(),
        None => base.to_owned(),
    }
}

/// Whether assistant turns must carry `reasoning_content` for this endpoint.
///
/// `DeepSeek` requires the field to be present — with an empty string when the
/// turn has no reasoning — on every replayed assistant message; every other
/// OpenAI-compatible provider rejects the field or ignores it, so it is omitted.
/// A provider is DeepSeek-like when its normalized name is `deepseek` or when the
/// model name mentions `deepseek` (case-insensitive), matching the Python
/// `model_requires_reasoning_content` predicate.
///
/// ```
/// use openjiuwen_foundation::llm::openai::replays_reasoning_content;
///
/// assert!(replays_reasoning_content("deepseek", "deepseek-chat"));
/// assert!(replays_reasoning_content("openai", "DeepSeek-R1"));
/// assert!(!replays_reasoning_content("openai", "gpt-4o"));
/// ```
#[must_use]
pub fn replays_reasoning_content(provider: &str, model_name: &str) -> bool {
    normalize_provider_name(provider) == DEEPSEEK_MARKER
        || model_name.to_ascii_lowercase().contains(DEEPSEEK_MARKER)
}

/// Converts one message into the `OpenAI` chat-completions wire shape.
///
/// The result always carries `role` and `content`; `name`, `tool_calls`,
/// `tool_call_id`, and (for DeepSeek-like endpoints) `reasoning_content` are
/// added only when relevant. Text content becomes a JSON string and a part list
/// becomes an array of provider-shaped objects, so multimodal payloads pass
/// through unchanged.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the content cannot be serialized, which
/// cannot happen for the content types this crate produces.
pub fn convert_message(provider: &str, model_name: &str, message: &BaseMessage) -> Result<Value> {
    let mut object = Map::new();
    object.insert(
        "role".to_owned(),
        Value::String(message.role.as_str().to_owned()),
    );
    object.insert(
        "content".to_owned(),
        serde_json::to_value(&message.content)
            .map_err(|error| Error::validation(format!("serializing message content: {error}")))?,
    );
    if let Some(name) = &message.name {
        object.insert("name".to_owned(), Value::String(name.clone()));
    }
    if !message.tool_calls.is_empty() {
        let calls = message
            .tool_calls
            .iter()
            .map(ToolCall::to_openai)
            .collect::<Vec<_>>();
        object.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        object.insert(
            "tool_call_id".to_owned(),
            Value::String(tool_call_id.clone()),
        );
    }
    if message.role.is_assistant() && replays_reasoning_content(provider, model_name) {
        let reasoning = message.reasoning_content.clone().unwrap_or_default();
        object.insert("reasoning_content".to_owned(), Value::String(reasoning));
    }
    Ok(Value::Object(object))
}

/// Converts a whole conversation into the `OpenAI` chat-completions wire shape.
///
/// # Errors
///
/// Returns the error of [`convert_message`] for the first message it rejects.
pub fn convert_messages(
    provider: &str,
    model_name: &str,
    messages: &[BaseMessage],
) -> Result<Vec<Value>> {
    messages
        .iter()
        .map(|message| convert_message(provider, model_name, message))
        .collect()
}

/// Passes tool definitions through unchanged.
///
/// Tools reach this crate already in the `OpenAI` `{"type":"function","function":
/// {...}}` shape, so re-encoding them could only lose vendor extensions.
#[must_use]
pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    tools.to_vec()
}

/// Passes a `tool_choice` value through unchanged.
///
/// Both spellings are valid on the wire: the string form (`"auto"`, `"none"`,
/// `"required"`) and the object form (`{"type":"function","function":{"name":N}}`).
#[must_use]
pub fn convert_tool_choice(choice: &Value) -> Value {
    choice.clone()
}

/// Builds the `POST /chat/completions` request body for one call.
///
/// The body always contains `model`, `messages`, and `stream`. Optional fields
/// are present only when configured: sampling parameters when set, `stop` when
/// non-empty, and `tools`/`tool_choice` when at least one tool is configured
/// (`tool_choice` defaults to `"auto"`). `stream: true` adds
/// `stream_options: {"include_usage": true}` so usage arrives on the stream.
/// [`ModelRequestConfig::extra`] is merged last at the top level, so a caller can
/// override or extend any computed key.
///
/// # Errors
///
/// Returns [`Error::Config`] when [`ModelRequestConfig::model_name`] is empty,
/// and [`Error::Validation`] when a message cannot be converted.
pub fn build_chat_completion_body(
    provider: &str,
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
    stream: bool,
) -> Result<Value> {
    let model_name = request.model_name.trim();
    if model_name.is_empty() {
        return Err(Error::config(
            "model_name must not be empty: set ModelRequestConfig::model_name \
             (or the MODEL_NAME environment variable)",
        ));
    }

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model_name.to_owned()));
    body.insert(
        "messages".to_owned(),
        Value::Array(convert_messages(provider, model_name, messages)?),
    );
    body.insert("stream".to_owned(), Value::Bool(stream));

    if let Some(temperature) = request.temperature.filter(|value| value.is_finite()) {
        body.insert("temperature".to_owned(), sampling_value(temperature));
    }
    if let Some(top_p) = request.top_p.filter(|value| value.is_finite()) {
        body.insert("top_p".to_owned(), sampling_value(top_p));
    }
    if let Some(max_tokens) = request.max_tokens {
        body.insert("max_tokens".to_owned(), Value::from(max_tokens));
    }
    if !request.stop.is_empty() {
        body.insert("stop".to_owned(), Value::from(request.stop.clone()));
    }

    let tools = request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty());
    if let Some(tools) = tools {
        body.insert("tools".to_owned(), Value::Array(convert_tools(tools)));
        let choice = request
            .tool_choice
            .clone()
            .unwrap_or_else(|| Value::String("auto".to_owned()));
        body.insert("tool_choice".to_owned(), convert_tool_choice(&choice));
    }

    if stream {
        body.insert(
            "stream_options".to_owned(),
            json!({ "include_usage": true }),
        );
    }

    if let Some(extra) = request.extra.as_object() {
        for (key, value) in extra {
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Renders an `f32` sampling parameter as the decimal the caller wrote.
///
/// [`Value::from(f32)`] widens through `f64`, so `0.7f32` would reach the wire as
/// `0.699999988079071`. Sampling parameters are rounded to six decimal places,
/// which is far more precision than any provider honours and keeps `0.7` on the
/// wire as `0.7`. A non-finite value renders as [`Value::Null`]; callers filter
/// those out before calling this.
fn sampling_value(value: f32) -> Value {
    let rounded = (f64::from(value) * 1_000_000.0).round() / 1_000_000.0;
    serde_json::Number::from_f64(rounded).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::prompt::content::{ContentPart, ImageUrl, MessageContent};
    use crate::prompt::ToolCall;

    fn request() -> ModelRequestConfig {
        ModelRequestConfig::new("gpt-4o")
    }

    fn body(messages: &[BaseMessage], request: &ModelRequestConfig) -> Value {
        build_chat_completion_body("openai", messages, request, false)
            .expect("the body builds for a non-empty model")
    }

    #[test]
    fn normalize_api_base_strips_whitespace_and_trailing_slashes() {
        assert_eq!(
            normalize_api_base("  https://api.openai.com/v1/  "),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_api_base("https://api.example.com/v1///"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            normalize_api_base("http://localhost:11434/v1"),
            "http://localhost:11434/v1"
        );
    }

    #[test]
    fn normalize_api_base_strips_a_pasted_chat_completions_path() {
        assert_eq!(
            normalize_api_base("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_api_base("https://api.openai.com/v1/chat/completions/"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_api_base("https://host.example/chat/completions"),
            "https://host.example"
        );
    }

    #[test]
    fn normalize_api_base_never_invents_a_version_segment() {
        assert_eq!(
            normalize_api_base("https://host.example"),
            "https://host.example"
        );
        assert_eq!(
            normalize_api_base("https://host.example/v2"),
            "https://host.example/v2"
        );
        assert_eq!(normalize_api_base("   "), "");
        assert_eq!(normalize_api_base(""), "");
    }

    #[test]
    fn reason_content_replay_is_limited_to_deepseek_endpoints() {
        for (provider, model) in [
            ("deepseek", "deepseek-chat"),
            ("DeepSeek", "whatever"),
            ("deep-seek", "deepseek-reasoner"),
            ("openai", "deepseek-chat"),
            ("openai", "DeepSeek-R1"),
            ("qwen", "some-deepseek-distill"),
        ] {
            assert!(
                replays_reasoning_content(provider, model),
                "{provider}/{model} replays reasoning"
            );
        }

        for (provider, model) in [
            ("openai", "gpt-4o"),
            ("qwen", "qwen-max"),
            ("anthropic", "claude-3-5-sonnet"),
            ("", ""),
        ] {
            assert!(
                !replays_reasoning_content(provider, model),
                "{provider}/{model} keeps reasoning out of the body"
            );
        }
    }

    #[test]
    fn default_body_omits_every_optional_field() {
        let body = body(&[BaseMessage::user("hello")], &request());
        assert_eq!(
            body,
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false,
            })
        );
    }

    #[test]
    fn sampling_fields_are_emitted_only_when_set() {
        let request = request()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_max_tokens(256)
            .with_stop(["STOP".to_owned(), "END".to_owned()]);
        let body = body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["max_tokens"], json!(256));
        assert_eq!(body["stop"], json!(["STOP", "END"]));
    }

    #[test]
    fn an_empty_stop_list_is_omitted() {
        let request = request().with_stop(Vec::<String>::new());
        assert!(body(&[BaseMessage::user("hi")], &request)
            .get("stop")
            .is_none());
    }

    #[test]
    fn tools_default_the_tool_choice_to_auto() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Reads the weather",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let request = request().with_tools(tools.clone());
        let body = body(&[BaseMessage::user("hi")], &request);

        assert_eq!(body["tools"], json!(tools));
        assert_eq!(body["tool_choice"], json!("auto"));
    }

    #[test]
    fn an_explicit_tool_choice_is_passed_through() {
        let request = request()
            .with_tools(vec![
                json!({"type": "function", "function": {"name": "ping"}}),
            ])
            .with_tool_choice(json!({"type": "function", "function": {"name": "ping"}}));
        let body = body(&[BaseMessage::user("hi")], &request);

        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "function": {"name": "ping"}})
        );
    }

    #[test]
    fn a_tool_choice_without_tools_is_omitted() {
        let request = request().with_tool_choice(json!("required"));
        let body = body(&[BaseMessage::user("hi")], &request);

        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn an_empty_tool_list_is_omitted() {
        let request = request().with_tools(Vec::new());
        let body = body(&[BaseMessage::user("hi")], &request);

        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn streaming_adds_include_usage() {
        let body =
            build_chat_completion_body("openai", &[BaseMessage::user("hi")], &request(), true)
                .expect("body builds");

        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn non_streaming_omits_stream_options() {
        let body = body(&[BaseMessage::user("hi")], &request());
        assert_eq!(body["stream"], json!(false));
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn extra_is_merged_last_and_wins_over_computed_keys() {
        let request = request().with_temperature(0.7).with_extra(json!({
            "temperature": 0.1,
            "stream_options": {"include_usage": false},
            "chat_template_kwargs": {"enable_thinking": true}
        }));
        let body = build_chat_completion_body("openai", &[BaseMessage::user("hi")], &request, true)
            .expect("body builds");

        assert_eq!(
            body["temperature"],
            json!(0.1),
            "extra overrides a computed key"
        );
        assert_eq!(
            body["stream_options"],
            json!({"include_usage": false}),
            "extra overrides the stream options"
        );
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": true}),
            "unrelated extra keys survive"
        );
        assert_eq!(body["model"], json!("gpt-4o"));
    }

    #[test]
    fn text_content_is_a_json_string() {
        let message = BaseMessage::user("plain text");
        assert_eq!(
            convert_message("openai", "gpt-4o", &message).expect("converts"),
            json!({"role": "user", "content": "plain text"})
        );
    }

    #[test]
    fn multimodal_parts_pass_through_verbatim() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what is this?".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/cat.png".to_owned(),
                    detail: Some("high".to_owned()),
                },
            },
            ContentPart::Other(json!({"type": "video_url", "video_url": {"url": "v"}})),
        ]);
        let message = BaseMessage::user(content);
        let converted = convert_message("openai", "gpt-4o", &message).expect("converts");

        assert_eq!(
            converted,
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {
                        "url": "https://example.com/cat.png",
                        "detail": "high"
                    }},
                    {"type": "video_url", "video_url": {"url": "v"}},
                ],
            })
        );
    }

    #[test]
    fn tool_calls_use_the_nested_function_shape() {
        let call = ToolCall::new("get_weather")
            .with_id("call_1")
            .with_arguments(json!({"city": "Paris"}));
        let message = BaseMessage::assistant("").with_tool_calls(vec![call]);
        let converted = convert_message("openai", "gpt-4o", &message).expect("converts");

        assert_eq!(
            converted["tool_calls"],
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }])
        );
    }

    #[test]
    fn tool_turns_carry_the_tool_call_id() {
        let message = BaseMessage::tool("call_1", "18 degrees");
        let converted = convert_message("openai", "gpt-4o", &message).expect("converts");

        assert_eq!(
            converted,
            json!({"role": "tool", "content": "18 degrees", "tool_call_id": "call_1"})
        );
    }

    #[test]
    fn a_message_name_is_forwarded() {
        let message = BaseMessage::user("hi").with_name("alice");
        let converted = convert_message("openai", "gpt-4o", &message).expect("converts");

        assert_eq!(converted["name"], json!("alice"));
    }

    #[test]
    fn deepseek_replays_reasoning_content_including_an_empty_string() {
        let with_reasoning = BaseMessage::assistant("answer").with_reasoning("thinking");
        let without_reasoning = BaseMessage::assistant("answer");

        let converted =
            convert_message("deepseek", "deepseek-reasoner", &with_reasoning).expect("converts");
        assert_eq!(converted["reasoning_content"], json!("thinking"));

        let converted =
            convert_message("deepseek", "deepseek-reasoner", &without_reasoning).expect("converts");
        assert_eq!(converted["reasoning_content"], json!(""));

        let converted =
            convert_message("openai", "deepseek-chat", &without_reasoning).expect("converts");
        assert_eq!(converted["reasoning_content"], json!(""));
    }

    #[test]
    fn other_providers_never_emit_reasoning_content() {
        let message = BaseMessage::assistant("answer").with_reasoning("thinking");
        let converted = convert_message("openai", "gpt-4o", &message).expect("converts");

        assert!(converted.get("reasoning_content").is_none());
    }

    #[test]
    fn reasoning_content_is_not_added_to_non_assistant_turns() {
        let message = BaseMessage::user("hi").with_reasoning("thinking");
        let converted = convert_message("deepseek", "deepseek-chat", &message).expect("converts");

        assert!(converted.get("reasoning_content").is_none());
    }

    #[test]
    fn an_empty_model_name_is_a_config_error() {
        let error = build_chat_completion_body(
            "openai",
            &[BaseMessage::user("hi")],
            &ModelRequestConfig::default(),
            false,
        )
        .expect_err("an empty model name must fail");

        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        let message = error.to_string();
        assert!(message.contains("model_name"), "message: {message}");
        assert!(message.contains("MODEL_NAME"), "message: {message}");
    }

    #[test]
    fn a_whitespace_model_name_is_rejected() {
        let error =
            build_chat_completion_body("openai", &[], &ModelRequestConfig::new("   "), false)
                .expect_err("a blank model name must fail");

        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    }

    #[test]
    fn the_model_name_is_trimmed() {
        let request = ModelRequestConfig::new("  gpt-4o-mini  ");
        let body = body(&[BaseMessage::user("hi")], &request);
        assert_eq!(body["model"], json!("gpt-4o-mini"));
    }

    #[test]
    fn convert_tools_and_tool_choice_are_identity_maps() {
        let tools = vec![json!({"type": "function", "function": {"name": "ping"}})];
        assert_eq!(convert_tools(&tools), tools);
        assert_eq!(convert_tool_choice(&json!("none")), json!("none"));
    }

    #[test]
    fn messages_are_converted_in_order() {
        let messages = [
            BaseMessage::system("be brief"),
            BaseMessage::user("hi"),
            BaseMessage::assistant("hello"),
        ];
        let converted = convert_messages("openai", "gpt-4o", &messages).expect("converts");

        assert_eq!(
            converted,
            vec![
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": "hi"}),
                json!({"role": "assistant", "content": "hello"}),
            ]
        );
    }

    #[test]
    fn sampling_values_avoid_f32_widening_artifacts() {
        assert_eq!(sampling_value(0.7), json!(0.7));
        assert_eq!(sampling_value(0.2), json!(0.2));
        assert_eq!(sampling_value(1.0), json!(1.0));
        assert_eq!(sampling_value(1.5), json!(1.5));
        assert_eq!(sampling_value(f32::NAN), Value::Null);
    }
}
