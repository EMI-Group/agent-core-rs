//! Unit tests for the two model-config types: transport resolution, the Python
//! decoding rules, and builder coverage.

use super::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Every variable the resolvers in this module may read.
const ENV_VARS: &[&str] = &[
    env::OPENJIUWEN_API_KEY,
    env::API_KEY,
    env::OPENAI_API_KEY,
    env::API_BASE,
    env::OPENAI_BASE_URL,
    env::MODEL_PROVIDER,
    env::MODEL_NAME,
];

/// Serializes tests that mutate the process environment.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Clears every variable this module reads and restores it on drop.
struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn clear() -> Self {
        let lock = env_lock();
        let saved: Vec<(&'static str, Option<String>)> = ENV_VARS
            .iter()
            .map(|name| (*name, std::env::var(name).ok()))
            .collect();
        for name in ENV_VARS {
            std::env::remove_var(name);
        }
        Self { _lock: lock, saved }
    }

    #[allow(clippy::unused_self)]
    fn set(&self, name: &str, value: &str) {
        std::env::set_var(name, value);
    }

    #[allow(clippy::unused_self)]
    fn remove(&self, name: &str) {
        std::env::remove_var(name);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

#[test]
fn defaults_match_the_documented_values() {
    let config = ModelClientConfig::default();

    assert!(uuid::Uuid::parse_str(&config.client_id).is_ok());
    assert_eq!(config.provider, "");
    assert_eq!(config.api_key, None);
    assert_eq!(config.api_base, None);
    assert!(config.verify_ssl);
    assert_eq!(config.timeout, Some(Duration::from_secs(360)));
    assert_eq!(
        config.stream_first_chunk_timeout,
        Some(Duration::from_secs(300))
    );
    assert_eq!(config.stream_idle_timeout, Some(Duration::from_secs(60)));
    assert_eq!(config.max_retries, 0);
    assert!(config.headers.is_empty());
    assert_eq!(config.api_mode, None);
    assert_eq!(config.auth_mode, AuthMode::ApiKey);
    assert_eq!(config.endpoint_profile, None);
    assert_eq!(config.extra, Value::Null);
}

#[test]
fn client_ids_are_unique_per_default() {
    assert_ne!(
        ModelClientConfig::default().client_id,
        ModelClientConfig::default().client_id
    );
}

#[test]
fn json_round_trip_preserves_every_field() {
    let config = ModelClientConfig::new("anthropic")
        .with_client_id("client-1")
        .with_api_key("sk-ant-test")
        .with_api_base("https://proxy.example/v1")
        .with_timeout(Duration::from_secs(12))
        .with_stream_timeouts(Some(Duration::from_secs(7)), None)
        .with_max_retries(4)
        .with_header("X-Org", "research")
        .with_verify_ssl(false)
        .with_extra(json!({"api_mode": "chat_completions"}));

    let encoded = serde_json::to_string(&config).expect("serializes");
    assert!(encoded.contains("\"provider\":\"anthropic\""));

    let restored: ModelClientConfig = serde_json::from_str(&encoded).expect("round-trips");
    assert_eq!(restored, config);
    assert_eq!(
        restored.api_key.as_ref().map(SecretString::expose_secret),
        Some("sk-ant-test")
    );
}

#[test]
fn deserialization_accepts_python_aliases_and_unknown_fields() {
    let config: ModelClientConfig = serde_json::from_value(json!({
        "client_provider": "OpenAI",
        "api_key": "sk-python",
        "timeout": {"secs": 9, "nanos": 0},
        "custom_headers": {"ignored": "value"},
    }))
    .expect("python config deserializes");

    assert_eq!(config.provider, "OpenAI");
    assert_eq!(config.timeout, Some(Duration::from_secs(9)));
    assert!(uuid::Uuid::parse_str(&config.client_id).is_ok());
    assert_eq!(
        config.headers.get("ignored").map(String::as_str),
        Some("value"),
        "`custom_headers` is the Python spelling of `headers`"
    );

    let request: ModelRequestConfig =
        serde_json::from_value(json!({"model": "gpt-4o-mini", "unknown": true}))
            .expect("model alias deserializes");
    assert_eq!(request.model_name, "gpt-4o-mini");
    assert!(!request.stream);
    assert_eq!(request.stop, Vec::<String>::new());
}

#[test]
fn python_wire_modes_decode() {
    let config: ModelClientConfig = serde_json::from_value(json!({
        "client_provider": "openai",
        "api_mode": "responses",
        "auth_mode": "openai_account_oauth",
        "endpoint_profile": "codex",
        "custom_headers": {"X-Tenant": "acme"},
        "unknown_field": {"kept": false},
    }))
    .expect("a Python-serialized config decodes");

    assert_eq!(config.api_mode, Some(ApiMode::Responses));
    assert_eq!(config.auth_mode, AuthMode::OpenAiAccountOAuth);
    assert_eq!(config.endpoint_profile.as_deref(), Some("codex"));
    assert_eq!(
        config.headers.get("X-Tenant").map(String::as_str),
        Some("acme")
    );
    assert_eq!(config.api_key, None, "an omitted key stays missing");
}

#[test]
fn api_and_auth_modes_use_the_python_wire_values() {
    let decode = |value: &str| serde_json::from_str::<ApiMode>(value).expect("api mode");
    assert_eq!(decode("\"chat_completions\""), ApiMode::ChatCompletions);
    assert_eq!(decode("\"responses\""), ApiMode::Responses);
    assert_eq!(decode("\"anthropic_messages\""), ApiMode::AnthropicMessages);

    let decode = |value: &str| serde_json::from_str::<AuthMode>(value).expect("auth mode");
    assert_eq!(decode("\"api_key\""), AuthMode::ApiKey);
    assert_eq!(decode("\"none\""), AuthMode::NoneAuth);
    assert_eq!(decode("\"custom_headers\""), AuthMode::CustomHeaders);
    assert_eq!(
        decode("\"openai_account_oauth\""),
        AuthMode::OpenAiAccountOAuth
    );
    assert_eq!(AuthMode::default(), AuthMode::ApiKey);

    assert_eq!(
        serde_json::to_value(ApiMode::Responses).expect("serializes"),
        json!("responses")
    );
    assert_eq!(
        serde_json::to_value(AuthMode::NoneAuth).expect("serializes"),
        json!("none")
    );
}

#[test]
fn non_api_key_auth_modes_do_not_require_a_credential() {
    for auth_mode in [
        AuthMode::NoneAuth,
        AuthMode::CustomHeaders,
        AuthMode::OpenAiAccountOAuth,
    ] {
        let config = ModelClientConfig::new("openai_account")
            .with_auth_mode(auth_mode)
            .with_api_base("https://example.test/backend-api/codex");
        let resolved = config
            .resolve_for("openai_account")
            .unwrap_or_else(|error| panic!("{auth_mode:?} must not need a key: {error}"));
        assert_eq!(resolved.auth_mode, auth_mode);
    }
}

#[test]
fn new_builders_set_the_protocol_fields() {
    let config = ModelClientConfig::new("openai")
        .with_api_mode(ApiMode::Responses)
        .with_auth_mode(AuthMode::CustomHeaders)
        .with_endpoint_profile("codex");
    assert_eq!(config.api_mode, Some(ApiMode::Responses));
    assert_eq!(config.auth_mode, AuthMode::CustomHeaders);
    assert_eq!(config.endpoint_profile.as_deref(), Some("codex"));

    let restored: ModelClientConfig =
        serde_json::from_value(serde_json::to_value(&config).expect("serializes"))
            .expect("round-trips");
    assert_eq!(restored, config);
}

#[test]
fn builders_set_every_field() {
    let secret = SecretString::new("sk-secret");
    let config = ModelClientConfig::new("qwen")
        .with_client_id("c-2")
        .with_secret(secret.clone())
        .with_api_base("https://example.test/v1")
        .with_timeout(Duration::from_secs(5))
        .with_stream_timeouts(Some(Duration::from_secs(2)), Some(Duration::from_secs(1)))
        .with_max_retries(2)
        .with_headers(vec![("X-A".to_owned(), "1".to_owned())])
        .with_header("X-B", "2")
        .with_verify_ssl(false)
        .with_extra(json!({"request_extra_body": {"seed": 1}}));

    assert_eq!(config.client_id, "c-2");
    assert_eq!(config.provider, "qwen");
    assert_eq!(config.api_key, Some(secret));
    assert_eq!(config.api_base.as_deref(), Some("https://example.test/v1"));
    assert_eq!(config.timeout, Some(Duration::from_secs(5)));
    assert_eq!(
        config.stream_first_chunk_timeout,
        Some(Duration::from_secs(2))
    );
    assert_eq!(config.stream_idle_timeout, Some(Duration::from_secs(1)));
    assert_eq!(config.max_retries, 2);
    assert_eq!(config.headers.len(), 2);
    assert_eq!(config.headers.get("X-B").map(String::as_str), Some("2"));
    assert!(!config.verify_ssl);
    assert_eq!(config.extra, json!({"request_extra_body": {"seed": 1}}));
}

#[test]
fn request_builders_set_every_field() {
    let request = ModelRequestConfig::new("gpt-4o")
        .with_model_name("gpt-4o-2024")
        .with_temperature(0.1)
        .with_top_p(0.9)
        .with_max_tokens(512)
        .with_tools(vec![json!({"type": "function"})])
        .with_tool_choice(json!("auto"))
        .with_stop(vec!["STOP".to_owned()])
        .with_stream(true)
        .with_extra(json!({"reasoning": {"effort": "low"}}));

    assert_eq!(request.model_name, "gpt-4o-2024");
    assert_eq!(request.temperature, Some(0.1));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.max_tokens, Some(512));
    assert_eq!(request.tools, Some(json!([{"type": "function"}])));
    assert_eq!(request.tool_choice, Some(json!("auto")));
    assert_eq!(request.stop, vec!["STOP".to_owned()]);
    assert!(request.stream);
    assert_eq!(request.extra, json!({"reasoning": {"effort": "low"}}));
}

#[test]
fn empty_tool_list_clears_the_field() {
    let request = ModelRequestConfig::default().with_tools(Vec::new());
    assert_eq!(request.tools, None);
}

#[test]
fn http_client_config_maps_transport_settings() {
    let config = ModelClientConfig::new("openai")
        .with_timeout(Duration::from_secs(42))
        .with_max_retries(3)
        .with_verify_ssl(false)
        .with_header("X-Custom", "value")
        .with_header("Authorization", "Bearer leaked");

    let http = config.http_client_config();
    assert_eq!(http.timeout, Duration::from_secs(42));
    assert_eq!(http.max_retries, 3);
    assert!(!http.verify_ssl);
    assert_eq!(http.headers.len(), 1);
    assert_eq!(
        http.headers.get("X-Custom").map(String::as_str),
        Some("value")
    );
    assert!(!http.headers.contains_key("Authorization"));
}

#[test]
fn http_client_config_falls_back_to_the_default_timeout() {
    let mut config = ModelClientConfig::new("openai");
    config.timeout = None;
    assert_eq!(
        config.http_client_config().timeout,
        Duration::from_secs(360)
    );
}

#[test]
fn sanitized_headers_drop_protected_names_and_blank_values() {
    let mut config = ModelClientConfig::new("openai");
    for (name, value) in [
        ("Authorization", "Bearer secret"),
        ("Host", "api.example"),
        ("Content-Length", "12"),
        ("Transfer-Encoding", "chunked"),
        ("Connection", "keep-alive"),
        ("X-Blank", "   "),
        ("X-Kept", "kept"),
    ] {
        config.headers.insert(name.to_owned(), value.to_owned());
    }

    let sanitized = config.sanitized_headers();
    assert_eq!(sanitized.len(), 1);
    assert_eq!(sanitized.get("X-Kept").map(String::as_str), Some("kept"));
}

#[test]
fn resolve_for_applies_the_provider_default_base_and_key() {
    let _env = EnvGuard::clear();
    let original = ModelClientConfig::new("OpenAI")
        .with_api_key("sk-explicit")
        .with_timeout(Duration::from_secs(11))
        .with_max_retries(2)
        .with_verify_ssl(false);
    let resolved = original
        .clone()
        .resolve_for("OpenAI")
        .expect("openai resolves");

    assert_eq!(resolved.provider, "openai");
    assert_eq!(
        resolved.api_base.as_deref(),
        Some("https://api.openai.com/v1")
    );
    assert_eq!(
        resolved.api_key.as_ref().map(SecretString::expose_secret),
        Some("sk-explicit")
    );
    assert_eq!(resolved.timeout, Some(Duration::from_secs(11)));
    assert_eq!(
        resolved.stream_first_chunk_timeout,
        Some(Duration::from_secs(300))
    );
    assert_eq!(resolved.stream_idle_timeout, Some(Duration::from_secs(60)));
    assert_eq!(resolved.max_retries, 2);
    assert!(!resolved.verify_ssl);
    assert_eq!(resolved.client_id, original.client_id);
}

#[test]
fn resolve_for_prefers_explicit_base_then_env_then_provider_default() {
    let env = EnvGuard::clear();

    let explicit = ModelClientConfig::new("openai")
        .with_api_key("sk")
        .with_api_base("https://explicit.example/v1")
        .resolve_for("openai")
        .expect("resolves");
    assert_eq!(
        explicit.api_base.as_deref(),
        Some("https://explicit.example/v1")
    );

    env.set(env::API_BASE, "https://env.example/v1");
    let from_env = ModelClientConfig::new("openai")
        .with_api_key("sk")
        .resolve_for("openai")
        .expect("resolves");
    assert_eq!(from_env.api_base.as_deref(), Some("https://env.example/v1"));

    env.remove(env::API_BASE);
    env.remove(env::OPENAI_BASE_URL);
    let from_default = ModelClientConfig::new("deepseek")
        .with_api_key("sk")
        .resolve_for("deepseek")
        .expect("resolves");
    assert_eq!(
        from_default.api_base.as_deref(),
        Some("https://api.deepseek.com/v1")
    );
}

#[test]
fn resolve_for_prefers_the_explicit_key_then_the_environment() {
    let env = EnvGuard::clear();
    env.set(env::OPENJIUWEN_API_KEY, "env-key");

    let explicit = ModelClientConfig::new("openai")
        .with_api_key("explicit-key")
        .resolve_for("openai")
        .expect("resolves");
    assert_eq!(
        explicit.api_key.as_ref().map(SecretString::expose_secret),
        Some("explicit-key")
    );

    let inherited = ModelClientConfig::new("openai")
        .resolve_for("openai")
        .expect("resolves");
    assert_eq!(
        inherited.api_key.as_ref().map(SecretString::expose_secret),
        Some("env-key")
    );
}

#[test]
fn blank_explicit_key_is_treated_as_missing() {
    let env = EnvGuard::clear();
    env.set(env::OPENAI_API_KEY, "env-openai-key");

    let resolved = ModelClientConfig::new("openai")
        .with_api_key("   ")
        .resolve_for("openai")
        .expect("resolves");
    assert_eq!(
        resolved.api_key.as_ref().map(SecretString::expose_secret),
        Some("env-openai-key")
    );
}

#[test]
fn resolve_for_normalizes_vendor_aliases() {
    let _env = EnvGuard::clear();
    let resolved = ModelClientConfig::new("DashScope")
        .with_api_key("sk")
        .resolve_for("DashScope")
        .expect("dashscope resolves");

    assert_eq!(resolved.provider, "qwen");
    assert_eq!(
        resolved.api_base.as_deref(),
        Some("https://dashscope.aliyuncs.com/compatible-mode/v1")
    );
}

#[test]
fn local_providers_do_not_require_a_key() {
    let _env = EnvGuard::clear();
    for (name, base) in [
        ("ollama", "http://localhost:11434/v1"),
        ("lmstudio", "http://localhost:1234/v1"),
    ] {
        let resolved = ModelClientConfig::new(name)
            .resolve_for(name)
            .unwrap_or_else(|error| panic!("{name} must not need a key: {error}"));
        assert_eq!(resolved.provider, name);
        assert_eq!(resolved.api_base.as_deref(), Some(base));
        assert_eq!(resolved.api_key, None);
    }
}

#[test]
fn pangu_without_an_api_base_is_a_config_error() {
    let _env = EnvGuard::clear();
    let error = ModelClientConfig::new("pangu")
        .with_api_key("sk")
        .resolve_for("pangu")
        .expect_err("pangu has no default base");

    assert_eq!(error.kind(), ErrorKind::Config);
    let message = error.to_string();
    assert!(message.contains("pangu"), "message: {message}");
    assert!(message.contains(env::API_BASE), "message: {message}");
}

#[test]
fn missing_api_key_names_the_environment_variables_without_a_secret() {
    let _env = EnvGuard::clear();
    let error = ModelClientConfig::new("openai")
        .resolve_for("openai")
        .expect_err("openai requires a key");

    assert_eq!(error.kind(), ErrorKind::Config);
    let message = error.to_string();
    assert!(message.contains(env::API_KEY), "message: {message}");
    assert!(message.contains(env::OPENAI_API_KEY), "message: {message}");
    assert!(
        message.contains(env::OPENJIUWEN_API_KEY),
        "message: {message}"
    );
    assert!(!message.contains("sk-"), "message: {message}");
}

#[test]
fn unknown_provider_is_unsupported_and_lists_the_builtins() {
    let _env = EnvGuard::clear();
    let error = ModelClientConfig::new("acme")
        .with_api_key("sk")
        .resolve_for("acme")
        .expect_err("acme is not built in");

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("acme"), "message: {message}");
    assert!(message.contains("openai"), "message: {message}");
    assert!(message.contains("anthropic"), "message: {message}");
}

#[test]
fn from_env_reads_provider_key_base_and_model_name() {
    let env = EnvGuard::clear();
    env.set(env::MODEL_PROVIDER, "deepseek");
    env.set(env::API_KEY, "env-key");
    env.set(env::OPENAI_BASE_URL, "https://env.example/v1");
    env.set(env::MODEL_NAME, "deepseek-chat");

    let client = ModelClientConfig::from_env();
    assert_eq!(client.provider, "deepseek");
    assert_eq!(
        client.api_key.as_ref().map(SecretString::expose_secret),
        Some("env-key")
    );
    assert_eq!(client.api_base.as_deref(), Some("https://env.example/v1"));
    assert!(uuid::Uuid::parse_str(&client.client_id).is_ok());

    assert_eq!(ModelRequestConfig::from_env().model_name, "deepseek-chat");
}

#[test]
fn from_env_defaults_the_provider_to_openai() {
    let _env = EnvGuard::clear();
    let client = ModelClientConfig::from_env();
    assert_eq!(client.provider, "openai");
    assert_eq!(client.api_key, None);
    assert_eq!(client.api_base, None);
    assert_eq!(ModelRequestConfig::from_env().model_name, "");
}

#[test]
fn debug_output_never_contains_the_api_key() {
    let config = ModelClientConfig::new("openai").with_api_key("sk-super-secret");
    let rendered = format!("{config:?}");
    assert!(!rendered.contains("sk-super-secret"), "debug: {rendered}");
}

// --- Python decoding rules --------------------------------------------------
#[test]
fn python_float_second_timeouts_decode() {
    // `ModelClientConfig(timeout=360.0, stream_first_chunk_timeout=None, ...)`
    // dumped with `model_dump(mode="json")` writes plain float seconds.
    let config: ModelClientConfig = serde_json::from_value(json!({
        "client_provider": "openai",
        "timeout": 360.0,
        "stream_first_chunk_timeout": 300,
        "stream_idle_timeout": 60.5,
    }))
    .expect("a Python-serialized client decodes");

    assert_eq!(config.timeout, Some(Duration::from_secs(360)));
    assert_eq!(
        config.stream_first_chunk_timeout,
        Some(Duration::from_secs(300))
    );
    assert_eq!(
        config.stream_idle_timeout,
        Some(Duration::from_millis(60_500))
    );
}

#[test]
fn timeouts_accept_numeric_strings_and_null() {
    let config: ModelClientConfig = serde_json::from_value(json!({
        "client_provider": "openai",
        "timeout": "45",
        "stream_first_chunk_timeout": " 12.5 ",
        "stream_idle_timeout": null,
    }))
    .expect("numeric strings decode");

    assert_eq!(config.timeout, Some(Duration::from_secs(45)));
    assert_eq!(
        config.stream_first_chunk_timeout,
        Some(Duration::from_millis(12_500))
    );
    assert_eq!(config.stream_idle_timeout, None);
}

#[test]
fn non_positive_and_unparsable_timeouts_are_errors() {
    for value in [json!(-5), json!(0), json!(0.0), json!("soon"), json!(true)] {
        let decoded = serde_json::from_value::<ModelClientConfig>(json!({
            "client_provider": "openai",
            "timeout": value,
        }));
        assert!(decoded.is_err(), "{value} must not decode");
    }
}

#[test]
fn timeouts_serialize_as_seconds_and_round_trip() {
    let config = ModelClientConfig::new("openai")
        .with_timeout(Duration::from_secs(360))
        .with_stream_timeouts(Some(Duration::from_millis(1500)), None);
    let encoded = serde_json::to_value(&config).expect("serializes");

    assert_eq!(encoded["timeout"], json!(360.0));
    assert_eq!(encoded["stream_first_chunk_timeout"], json!(1.5));
    assert_eq!(encoded["stream_idle_timeout"], json!(null));

    let restored: ModelClientConfig =
        serde_json::from_value(encoded).expect("round-trips through seconds");
    assert_eq!(restored, config);
}

#[test]
fn single_string_stop_and_null_decode() {
    let single: ModelRequestConfig =
        serde_json::from_value(json!({"model": "gpt-4o", "stop": "\n"}))
            .expect("Python's single string decodes");
    assert_eq!(single.stop, vec!["\n".to_owned()]);

    let none: ModelRequestConfig =
        serde_json::from_value(json!({"model": "gpt-4o", "stop": null})).expect("null decodes");
    assert_eq!(none.stop, Vec::<String>::new());

    let array: ModelRequestConfig =
        serde_json::from_value(json!({"model": "gpt-4o", "stop": ["END", "STOP"]}))
            .expect("the array form decodes");
    assert_eq!(array.stop, vec!["END".to_owned(), "STOP".to_owned()]);

    assert!(
        serde_json::from_value::<ModelRequestConfig>(json!({"model": "gpt-4o", "stop": 3}))
            .is_err(),
        "an unknown type is an error"
    );
}

#[test]
fn stop_serializes_as_an_array() {
    let request = ModelRequestConfig::new("gpt-4o").with_stop(vec!["\n".to_owned()]);
    let encoded = serde_json::to_value(&request).expect("serializes");
    assert_eq!(encoded["stop"], json!(["\n"]));
}

#[test]
fn custom_headers_coerce_python_any_values() {
    let config: ModelClientConfig = serde_json::from_value(json!({
        "client_provider": "openai",
        "custom_headers": {
            "X-Text": "value",
            "X-Rate": 10,
            "X-Ratio": 1.5,
            "X-Debug": true,
            "X-Dropped": null,
        },
    }))
    .expect("`dict[str, Any]` headers decode");

    assert_eq!(
        config.headers.get("X-Text").map(String::as_str),
        Some("value")
    );
    assert_eq!(config.headers.get("X-Rate").map(String::as_str), Some("10"));
    assert_eq!(
        config.headers.get("X-Ratio").map(String::as_str),
        Some("1.5")
    );
    assert_eq!(
        config.headers.get("X-Debug").map(String::as_str),
        Some("true")
    );
    assert!(!config.headers.contains_key("X-Dropped"));

    let encoded = serde_json::to_value(&config).expect("serializes");
    assert_eq!(encoded["headers"]["X-Rate"], json!("10"));

    assert!(
        serde_json::from_value::<ModelClientConfig>(json!({
            "client_provider": "openai",
            "custom_headers": {"X-Bad": ["a"]},
        }))
        .is_err(),
        "an array header value is an error"
    );
}

#[test]
fn typed_reasoning_and_context_window_decode_and_round_trip() {
    let request: ModelRequestConfig = serde_json::from_value(json!({
        "model": "claude-sonnet-4",
        "reasoning": {"mode": "enabled", "effort": "high", "budget_tokens": 2048},
        "context_window": 200_000,
        "unknown_python_key": {"kept": false},
    }))
    .expect("the Python request config decodes");

    assert_eq!(
        request.reasoning,
        Some(json!({"mode": "enabled", "effort": "high", "budget_tokens": 2048}))
    );
    assert_eq!(request.context_window, Some(200_000));
    assert_eq!(request.extra, Value::Null, "unknown keys are not forwarded");

    let encoded = serde_json::to_value(&request).expect("serializes");
    assert_eq!(encoded["context_window"], json!(200_000));
    assert_eq!(
        encoded["reasoning"],
        json!({"mode": "enabled", "effort": "high", "budget_tokens": 2048})
    );

    let restored: ModelRequestConfig = serde_json::from_value(encoded).expect("round-trips");
    assert_eq!(restored, request);
}

#[test]
fn absent_reasoning_and_context_window_are_not_serialized() {
    let encoded = serde_json::to_value(ModelRequestConfig::new("gpt-4o")).expect("serializes");
    assert!(!encoded
        .as_object()
        .expect("object")
        .contains_key("reasoning"));
    assert!(!encoded
        .as_object()
        .expect("object")
        .contains_key("context_window"));
}

#[test]
fn new_request_builders_set_the_metadata_fields() {
    let request = ModelRequestConfig::new("gpt-4o")
        .with_context_window(128_000)
        .with_reasoning(json!({"mode": "disabled"}));

    assert_eq!(request.context_window, Some(128_000));
    assert_eq!(request.reasoning, Some(json!({"mode": "disabled"})));
}
