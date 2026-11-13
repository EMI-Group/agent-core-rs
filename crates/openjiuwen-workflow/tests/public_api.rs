#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::assert_is_empty)]
#![allow(dead_code)]
//! The public config surface at the crate root and at the Python-shaped module paths.
//!
//! Python's `openjiuwen/core/workflow/__init__.py` re-exports every built-in component *and*
//! the whole `Http*` configuration tree at the package root, so
//! `from openjiuwen.core.workflow import HttpAuthConfig` works. This suite pins the Rust
//! equivalent: every item listed in the crate-root re-export block is named here, so removing
//! a re-export breaks the build.
//!
//! Three kinds of proof live here:
//!
//! - **Import proof (compile time).** The `use openjiuwen_workflow::{…}` block below names each
//!   newly exposed item; an unexported name fails to compile.
//! - **Type identity.** The root path and the module path annotate the same type, so a future
//!   duplicate definition would break the ascription instead of silently shadowing.
//! - **Wire behaviour.** A Python-shaped `HttpComponentConfig` document (all seven nested
//!   sections plus unknown keys) deserializes, re-serializes, and round-trips through the
//!   root-exported types, and every wire enum serializes to its Python string.
//!
//! Everything is offline and deterministic: no fixtures, no sockets, no model.

use std::collections::BTreeMap;

use serde_json::json;

// Every newly exposed crate-root item, imported by its root path.
use openjiuwen_workflow::{
    frame_to_wire, FrameCollector, HttpAdvancedOptionsConfig, HttpAuthConfig, HttpAuthIn,
    HttpAuthType, HttpBackoffType, HttpComponentConfig, HttpContentType, HttpHeadersConfig,
    HttpRateLimitConfig, HttpRateLimitUnit, HttpRequestBodyConfig, HttpRequestParamConfig,
    HttpResponseFormat, HttpResponseHandlingConfig, HttpResponseMode, HttpRetryConfig,
    LLMCompConfig, LLMComponent, OutputFrame, ReActAgentCompConfig,
    COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID,
    COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE,
    COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID, COMPONENT_INTENT_DETECTION_CONFIG_ERROR,
    COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID, COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR,
    COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR, COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED,
    COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN,
    COMPONENT_INTENT_DETECTION_KEYWORD_INVALID, COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED,
    COMPONENT_INTENT_DETECTION_PROMPT_INVALID, COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR,
    COMPONENT_QUESTIONER_CONFIG_ERROR, COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR,
    COMPONENT_QUESTIONER_INPUT_INVALID, COMPONENT_QUESTIONER_INPUT_PARAM_ERROR,
    COMPONENT_QUESTIONER_INVOKE_CALL_FAILED, COMPONENT_QUESTIONER_RUNTIME_ERROR,
    COMPONENT_QUESTIONER_STATE_INIT_FAILED, CUSTOM_FRAME_TYPE, INVALID_ID_CHARS_REASON,
    MAX_COMPONENT_ID_LEN, ON_MAX_ITERATIONS_ERROR, ON_MAX_ITERATIONS_FINAL_ANSWER,
    ON_MAX_ITERATIONS_RETURN_OBSERVATIONS, TRACE_FRAME_TYPE,
};

/// A Python-shaped `HttpComponentConfig` document with every nested section and two keys the
/// Rust type does not know (`component_type`, `future_field`).
fn python_http_config() -> serde_json::Value {
    json!({
        "request_params": {
            "url": "https://example.com/api",
            "method": "post",
            "headers": {"X-Key": "{{key}}"},
            "query_parameters": {"q": "rust"},
            "body": {
                "content_type": "form",
                "form_data": {"a": "1"}
            },
            "authentication": {
                "type": "bearer",
                "token": "secret"
            },
            "response_handling": {
                "response_format": "json",
                "response_mode": "on-success",
                "response_code_success_codes": [200, 204]
            },
            "advanced_options": {
                "follow_redirect": false,
                "proxy": "http://proxy.invalid"
            },
            "retry_config": {
                "enabled": true,
                "max_retries": 5,
                "backoff_type": "linear"
            },
            "rate_limit_config": {
                "enabled": true,
                "requests_per_unit": 10,
                "unit": "minute"
            }
        },
        "metadata": {"owner": "public-api"},
        "component_type": "http_request",
        "future_field": {"ignored": true}
    })
}

/// The Python-name aliases are the canonical Rust types, in both directions.
fn aliases_are_the_canonical_types(
    config: LLMCompConfig,
    component: LLMComponent,
    react: ReActAgentCompConfig,
) -> (
    openjiuwen_workflow::LlmCompConfig,
    openjiuwen_workflow::LlmComponent,
    openjiuwen_workflow::ReActAgentConfig,
) {
    (config, component, react)
}

#[test]
fn root_and_module_paths_name_the_same_types() {
    // Every HTTP config type: root path on the left, module path on the right (or the
    // reverse), so a second definition at either path fails to compile.
    let _: openjiuwen_workflow::components::http_request::HttpAuthType = HttpAuthType::None;
    let _: HttpContentType = openjiuwen_workflow::components::http_request::HttpContentType::Json;
    let _: openjiuwen_workflow::HttpAuthIn = HttpAuthIn::Query;
    let _: openjiuwen_workflow::HttpBackoffType = HttpBackoffType::Fixed;
    let _: openjiuwen_workflow::HttpResponseMode = HttpResponseMode::OnError;
    let _: openjiuwen_workflow::HttpRateLimitUnit = HttpRateLimitUnit::Hour;
    let _: openjiuwen_workflow::HttpResponseFormat = HttpResponseFormat::Buffer;
    let _: openjiuwen_workflow::HttpHeadersConfig = HttpHeadersConfig::Map(BTreeMap::new());
    let _: openjiuwen_workflow::HttpRequestBodyConfig = HttpRequestBodyConfig::default();
    let _: openjiuwen_workflow::HttpAuthConfig = HttpAuthConfig::default();
    let _: openjiuwen_workflow::HttpResponseHandlingConfig = HttpResponseHandlingConfig::default();
    let _: openjiuwen_workflow::HttpAdvancedOptionsConfig = HttpAdvancedOptionsConfig::default();
    let _: openjiuwen_workflow::HttpRetryConfig = HttpRetryConfig::default();
    let _: openjiuwen_workflow::HttpRateLimitConfig = HttpRateLimitConfig::default();
    let _: openjiuwen_workflow::HttpRequestParamConfig = HttpRequestParamConfig::default();

    // The umbrella component config: the crate root, `components`, and the Python-shaped
    // `components::http_request` path all name the same struct.
    let config: openjiuwen_workflow::components::HttpComponentConfig =
        HttpComponentConfig::default();
    let _: openjiuwen_workflow::components::http_request::HttpComponentConfig = config;

    // The Python-name aliases sit beside their canonical types.
    let _: openjiuwen_workflow::components::llm::LLMCompConfig = LLMCompConfig::default();
    let _: Option<openjiuwen_workflow::components::llm::LLMComponent> =
        Option::<openjiuwen_workflow::LlmComponent>::None;
    let _: openjiuwen_workflow::components::react_agent::ReActAgentCompConfig =
        ReActAgentCompConfig::default();
    let _: openjiuwen_workflow::components::LLMCompConfig = LLMCompConfig::default();
    let _: openjiuwen_workflow::components::ReActAgentCompConfig = ReActAgentCompConfig::default();
}

#[test]
fn python_name_aliases_are_usable() {
    // `LLMCompConfig`: the alias is a value with the canonical builders.
    let config = LLMCompConfig::default().with_model_id("m");
    assert_eq!(config.model_id.as_deref(), Some("m"));

    // `LLMComponent`: the alias has the inherent constructor and *is* `LlmComponent`.
    assert!(LLMComponent::new(LLMCompConfig::default()).is_err());
    let component: openjiuwen_workflow::LlmComponent =
        LLMComponent::new(LLMCompConfig::default().with_model(std::sync::Arc::new(
            openjiuwen_test_support::ScriptedModel::text("answer"),
        )))
        .unwrap();
    let _: &dyn openjiuwen_workflow::Component = &component;

    // `ReActAgentCompConfig`: the alias has the canonical builders.
    let react = ReActAgentCompConfig::default().with_max_iterations(2);
    assert_eq!(react.max_iterations, 2);
    assert!(react.validate().is_ok());
}

#[test]
fn python_shaped_http_config_deserializes_through_the_root_types() {
    let config: HttpComponentConfig = serde_json::from_value(python_http_config()).unwrap();
    let params: &HttpRequestParamConfig = &config.request_params;

    assert_eq!(params.url, "https://example.com/api");
    assert_eq!(params.method, "post");
    assert_eq!(
        params.headers,
        HttpHeadersConfig::Map(BTreeMap::from([("X-Key".to_owned(), "{{key}}".to_owned())]))
    );
    assert_eq!(params.query_parameters["q"], json!("rust"));

    let body: &HttpRequestBodyConfig = params.body.as_ref().unwrap();
    assert_eq!(body.content_type, HttpContentType::Form);
    assert_eq!(body.form_data.as_ref().unwrap()["a"], json!("1"));
    assert_eq!(body.json_data, None);

    let auth: &HttpAuthConfig = params.authentication.as_ref().unwrap();
    assert_eq!(auth.auth_type, HttpAuthType::Bearer);
    assert_eq!(auth.token.as_deref(), Some("secret"));
    // Omitted auth fields keep Python's defaults.
    assert_eq!(auth.name, "Authorization");
    assert_eq!(auth.in_location, HttpAuthIn::Header);

    let handling: &HttpResponseHandlingConfig = &params.response_handling;
    assert_eq!(handling.response_format, HttpResponseFormat::Json);
    assert_eq!(handling.response_mode, HttpResponseMode::OnSuccess);
    assert_eq!(handling.response_code_success_codes, vec![200, 204]);
    // An omitted `response_code_failure_codes` falls back to Python's empty list.
    assert!(handling.response_code_failure_codes.is_empty());

    let advanced: &HttpAdvancedOptionsConfig = &params.advanced_options;
    assert!(!advanced.follow_redirect);
    assert_eq!(advanced.proxy.as_deref(), Some("http://proxy.invalid"));
    assert_eq!(advanced.max_body_length, 1_048_576);

    let retry: &HttpRetryConfig = &params.retry_config;
    assert!(retry.enabled);
    assert_eq!(retry.max_retries, 5);
    assert_eq!(retry.backoff_type, HttpBackoffType::Linear);
    assert_eq!(retry.retry_on_status_codes, vec![429, 500, 502, 503, 504]);

    let rate_limit: &HttpRateLimitConfig = &params.rate_limit_config;
    assert!(rate_limit.enabled);
    assert_eq!(rate_limit.requests_per_unit, 10);
    assert_eq!(rate_limit.unit, HttpRateLimitUnit::Minute);

    assert_eq!(
        config.metadata.as_ref().unwrap()["owner"],
        json!("public-api")
    );
}

#[test]
fn python_shaped_http_config_re_serializes_and_round_trips() {
    let config: HttpComponentConfig = serde_json::from_value(python_http_config()).unwrap();
    let wire = serde_json::to_value(&config).unwrap();
    let params = &wire["request_params"];

    // Nested enums keep Python's wire strings, including the two `rename`d variants.
    assert_eq!(params["body"]["content_type"], json!("form"));
    assert_eq!(params["authentication"]["type"], json!("bearer"));
    assert_eq!(params["authentication"]["in_location"], json!("header"));
    assert_eq!(
        params["response_handling"]["response_format"],
        json!("json")
    );
    assert_eq!(
        params["response_handling"]["response_mode"],
        json!("on-success")
    );
    assert_eq!(params["retry_config"]["backoff_type"], json!("linear"));
    assert_eq!(params["rate_limit_config"]["unit"], json!("minute"));
    assert_eq!(params["headers"], json!({"X-Key": "{{key}}"}));

    // Unknown keys of the Python document are dropped, not echoed.
    assert!(wire.get("component_type").is_none());
    assert!(wire.get("future_field").is_none());

    // Round-trip: the re-serialized document parses back to an equal value.
    let round_tripped: HttpComponentConfig = serde_json::from_value(wire).unwrap();
    assert_eq!(round_tripped, config);
}

#[test]
fn wire_enums_match_python_strings() {
    for (value, expected) in [
        (HttpContentType::Json, "json"),
        (HttpContentType::Form, "form"),
        (HttpContentType::MultipartForm, "multipart_form"),
        (HttpContentType::Binary, "binary"),
        (HttpContentType::Text, "text"),
        (HttpContentType::Auto, "auto"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpAuthType::None, "none"),
        (HttpAuthType::Basic, "basic"),
        (HttpAuthType::Bearer, "bearer"),
        (HttpAuthType::ApiKey, "api_key"),
        (HttpAuthType::Digest, "digest"),
        (HttpAuthType::Aws, "aws"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpAuthIn::Header, "header"),
        (HttpAuthIn::Query, "query"),
        (HttpAuthIn::Body, "body"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpResponseFormat::Autodetect, "autodetect"),
        (HttpResponseFormat::Json, "json"),
        (HttpResponseFormat::Text, "text"),
        (HttpResponseFormat::Binary, "binary"),
        (HttpResponseFormat::Buffer, "buffer"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpResponseMode::Full, "full"),
        (HttpResponseMode::OnSuccess, "on-success"),
        (HttpResponseMode::OnError, "on-error"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpBackoffType::Fixed, "fixed"),
        (HttpBackoffType::Exponential, "exponential"),
        (HttpBackoffType::Linear, "linear"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
    for (value, expected) in [
        (HttpRateLimitUnit::Second, "second"),
        (HttpRateLimitUnit::Minute, "minute"),
        (HttpRateLimitUnit::Hour, "hour"),
    ] {
        assert_eq!(serde_json::to_value(value).unwrap(), json!(expected));
    }
}

#[test]
fn react_agent_policies_and_component_status_codes_are_exported() {
    assert_eq!(ON_MAX_ITERATIONS_ERROR, "error");
    assert_eq!(ON_MAX_ITERATIONS_FINAL_ANSWER, "final_answer");
    assert_eq!(ON_MAX_ITERATIONS_RETURN_OBSERVATIONS, "return_observations");

    // The questioner block (101070..=101076) is both at the root and under its module.
    assert_eq!(COMPONENT_QUESTIONER_INPUT_PARAM_ERROR, 101_070);
    assert_eq!(COMPONENT_QUESTIONER_CONFIG_ERROR, 101_071);
    assert_eq!(COMPONENT_QUESTIONER_INPUT_INVALID, 101_072);
    assert_eq!(COMPONENT_QUESTIONER_STATE_INIT_FAILED, 101_073);
    assert_eq!(COMPONENT_QUESTIONER_RUNTIME_ERROR, 101_074);
    assert_eq!(COMPONENT_QUESTIONER_INVOKE_CALL_FAILED, 101_075);
    assert_eq!(COMPONENT_QUESTIONER_EXECUTION_PROCESS_ERROR, 101_076);
    assert_eq!(
        openjiuwen_workflow::components::questioner::COMPONENT_QUESTIONER_RUNTIME_ERROR,
        COMPONENT_QUESTIONER_RUNTIME_ERROR
    );

    // The intent-detection block (101050..=101062), same rule.
    assert_eq!(COMPONENT_INTENT_DETECTION_INPUT_PARAM_ERROR, 101_050);
    assert_eq!(COMPONENT_INTENT_DETECTION_LLM_INIT_FAILED, 101_051);
    assert_eq!(COMPONENT_INTENT_DETECTION_INVOKE_CALL_FAILED, 101_052);
    assert_eq!(COMPONENT_INTENT_DETECTION_CATEGORY_NAME_INVALID, 101_053);
    assert_eq!(COMPONENT_INTENT_DETECTION_CATEGORY_NAME_DUPLICATE, 101_054);
    assert_eq!(COMPONENT_INTENT_DETECTION_ACCEPT_LANGUAGE_INVALID, 101_055);
    assert_eq!(COMPONENT_INTENT_DETECTION_KEYWORD_CATEGORY_UNKNOWN, 101_056);
    assert_eq!(COMPONENT_INTENT_DETECTION_KEYWORD_INVALID, 101_057);
    assert_eq!(COMPONENT_INTENT_DETECTION_EXAMPLE_INVALID, 101_058);
    assert_eq!(COMPONENT_INTENT_DETECTION_PROMPT_INVALID, 101_059);
    assert_eq!(COMPONENT_INTENT_DETECTION_CONFIG_ERROR, 101_060);
    assert_eq!(COMPONENT_INTENT_DETECTION_HISTORY_CONFIG_ERROR, 101_061);
    assert_eq!(COMPONENT_INTENT_DETECTION_ROUTER_CONFIG_ERROR, 101_062);
    assert_eq!(
        openjiuwen_workflow::components::intent_detection::COMPONENT_INTENT_DETECTION_CONFIG_ERROR,
        COMPONENT_INTENT_DETECTION_CONFIG_ERROR
    );
}

#[test]
fn id_limits_and_frame_helpers_are_exported() {
    assert_eq!(MAX_COMPONENT_ID_LEN, 100);
    assert!(!INVALID_ID_CHARS_REASON.is_empty());

    assert_eq!(CUSTOM_FRAME_TYPE, "custom");
    assert_eq!(TRACE_FRAME_TYPE, "trace");

    let collector: FrameCollector = FrameCollector::default();
    assert!(collector.lock().is_empty());

    let frame = OutputFrame {
        item_type: CUSTOM_FRAME_TYPE.to_owned(),
        index: 3,
        payload: json!({"a": 1}),
    };
    assert_eq!(
        frame_to_wire(&frame),
        json!({"type": "custom", "index": 3, "payload": {"a": 1}})
    );
    assert_eq!(
        frame_to_wire(&frame),
        openjiuwen_workflow::output::frame_to_wire(&frame)
    );
}
