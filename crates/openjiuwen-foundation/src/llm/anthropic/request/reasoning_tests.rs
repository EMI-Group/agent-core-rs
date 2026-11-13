//! Reasoning-plan wiring tests for the Anthropic request projection.
//!
//! These assert the exact body fields the provider-neutral reasoning intent
//! (`ModelRequestConfig::reasoning`, or the legacy `extra["reasoning"]`)
//! produces for the `anthropic_messages` protocol: the
//! `thinking` / `output_config` shapes, the legacy-control precedence, the fact
//! that the intent key never reaches the wire, and that a malformed intent is a
//! validation error instead of a silently dropped request.
use super::*;
use serde_json::json;

/// A model that has an explicit Anthropic budget-only profile.
const SONNET_45: &str = "claude-sonnet-4-5-20250929";

/// A model no reasoning profile lists, so the protocol default applies.
const UNPROFILED: &str = "claude-opus-4-1-20250805";

/// Builds a body for the `anthropic` provider with no API-base hint.
fn body_for(model: &str, extra: Value) -> Value {
    build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new(model).with_extra(extra),
        false,
        "anthropic",
        None,
    )
    .expect("the body builds")
}

#[test]
fn no_reasoning_intent_adds_no_reasoning_fields() {
    let body = body_for(SONNET_45, json!({}));
    assert!(body.get("thinking").is_none());
    assert!(body.get("output_config").is_none());
    assert!(body.get("reasoning_effort").is_none());
    assert!(body.get("enable_thinking").is_none());

    // The wrapper is exactly the provider-aware builder with `anthropic`/`None`,
    // so bodies built without an intent stay byte-identical.
    let wrapped = build_messages_body(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new(SONNET_45),
        false,
    )
    .expect("the body builds");
    let explicit = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new(SONNET_45),
        false,
        "anthropic",
        None,
    )
    .expect("the body builds");
    assert_eq!(wrapped, explicit);
}

#[test]
fn a_null_reasoning_value_is_no_intent_and_is_consumed() {
    let body = body_for(SONNET_45, json!({"reasoning": null}));
    assert!(
        body.get("reasoning").is_none(),
        "the intent key is consumed"
    );
    assert!(body.get("thinking").is_none());
}

#[test]
fn a_disabled_intent_switches_thinking_off_for_the_budget_profile() {
    let body = body_for(SONNET_45, json!({"reasoning": {"mode": "disabled"}}));
    assert_eq!(body.get("thinking"), Some(&json!({"type": "disabled"})));
    assert!(body.get("output_config").is_none());
    assert!(body.get("reasoning").is_none());
}

#[test]
fn a_false_boolean_shorthand_disables_thinking() {
    let body = body_for(SONNET_45, json!({"reasoning": false}));
    assert_eq!(body.get("thinking"), Some(&json!({"type": "disabled"})));
}

#[test]
fn an_effort_intent_maps_onto_the_sonnet_45_budget() {
    let body = body_for(
        SONNET_45,
        json!({"reasoning": {"mode": "enabled", "effort": "medium"}}),
    );
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 8192})),
        "the anthropic_manual wire maps medium onto 8192 budget tokens"
    );
    assert!(body.get("output_config").is_none());
}

#[test]
fn an_explicit_budget_works_for_an_unprofiled_model() {
    let body = body_for(
        UNPROFILED,
        json!({"reasoning": {"mode": "enabled", "budget_tokens": 2048}}),
    );
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 2048})),
        "an unlisted model falls back to the standard Anthropic encoder"
    );
    assert!(body.get("output_config").is_none());
}

#[test]
fn an_effort_without_a_budget_uses_the_adaptive_shape_for_unprofiled_models() {
    let body = body_for(
        UNPROFILED,
        json!({"reasoning": {"mode": "enabled", "effort": "high"}}),
    );
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "adaptive"})),
        "adaptive thinking carries no budget"
    );
    assert_eq!(body.get("output_config"), Some(&json!({"effort": "high"})));
}

#[test]
fn a_caller_thinking_object_wins_over_the_reasoning_intent() {
    let body = body_for(
        SONNET_45,
        json!({
            "reasoning": {"mode": "enabled", "effort": "high"},
            "thinking": {"type": "disabled"},
        }),
    );
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "disabled"})),
        "the explicit legacy control is forwarded verbatim"
    );
    assert!(
        body.get("output_config").is_none(),
        "no plan keys are added when a legacy control is present"
    );
    assert!(body.get("reasoning").is_none());
}

#[test]
fn every_legacy_control_key_at_the_top_level_suppresses_the_plan() {
    for key in LEGACY_REASONING_KEYS {
        let body = body_for(
            SONNET_45,
            json!({
                "reasoning": {"mode": "enabled", "effort": "high"},
                key: {"caller": true},
            }),
        );
        assert_eq!(
            body.get(key),
            Some(&json!({"caller": true})),
            "`{key}` is forwarded unchanged"
        );
        if key != "thinking" {
            assert!(
                body.get("thinking").is_none(),
                "`{key}` at the top level suppresses the plan"
            );
        }
        assert!(body.get("output_config").is_none());
    }
}

#[test]
fn a_legacy_control_inside_a_nested_extra_body_suppresses_the_plan() {
    let body = body_for(
        SONNET_45,
        json!({
            "reasoning": {"mode": "enabled", "effort": "high"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }),
    );
    assert_eq!(
        body.get("extra_body"),
        Some(&json!({"thinking": {"type": "disabled"}})),
        "the nested object keeps its existing forwarding behaviour"
    );
    assert!(
        body.get("thinking").is_none(),
        "a nested legacy control also suppresses the plan"
    );
}

#[test]
fn the_intent_key_is_never_forwarded_even_when_the_plan_is_skipped() {
    let body = body_for(
        SONNET_45,
        json!({
            "reasoning": {"mode": "enabled", "effort": "high"},
            "thinking": {"type": "disabled"},
        }),
    );
    assert!(body.get("reasoning").is_none());
}

#[test]
fn the_cache_control_knobs_stay_consumed_alongside_a_reasoning_intent() {
    let body = body_for(
        SONNET_45,
        json!({
            "reasoning": {"mode": "enabled", "effort": "medium"},
            "disable_cache_control": true,
        }),
    );
    assert!(body.get("disable_cache_control").is_none());
    assert!(body.get("reasoning").is_none());
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 8192}))
    );
    let rendered = body.to_string();
    assert!(
        !rendered.contains("cache_control"),
        "the opt-out still removes every anchor: {rendered}"
    );
}

#[test]
fn a_malformed_reasoning_value_is_a_validation_error() {
    for malformed in [
        json!(5),
        json!([1, 2]),
        json!({"thinking": {"type": "enabled"}}),
    ] {
        let error = build_messages_body_with_provider(
            &[BaseMessage::user("hello")],
            &ModelRequestConfig::new(SONNET_45).with_extra(json!({"reasoning": malformed})),
            false,
            "anthropic",
            None,
        )
        .expect_err("a malformed intent is rejected");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation,
            "`{malformed}` must be a validation error"
        );
        assert!(error.to_string().contains("reasoning"));
    }
}

#[test]
fn an_invalid_mode_is_a_validation_error() {
    let error = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new(SONNET_45).with_extra(json!({"reasoning": {"mode": "turbo"}})),
        false,
        "anthropic",
        None,
    )
    .expect_err("an unknown mode is rejected");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
}

#[test]
fn the_api_base_host_hint_selects_the_vendor_profile() {
    // A generic provider name defers to the host, so a DashScope-hosted Claude
    // endpoint resolves the DashScope Anthropic wire (native `output_config`)
    // instead of the Anthropic default.
    let body = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new("qwen3.8-max")
            .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "low"}})),
        false,
        "openai",
        Some("https://dashscope.aliyuncs.com/api/v2/apps/claude-code-proxy"),
    )
    .expect("the body builds");
    assert_eq!(body.get("thinking"), Some(&json!({"type": "enabled"})));
    assert_eq!(body.get("output_config"), Some(&json!({"effort": "low"})));
}

#[test]
fn an_explicit_provider_wins_over_the_api_base_host_hint() {
    // The vendor-specific name is authoritative, so the same host does not
    // change the Anthropic budget encoder for a Sonnet 4.5 request.
    let body = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &ModelRequestConfig::new(SONNET_45)
            .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "low"}})),
        false,
        "anthropic",
        Some("https://dashscope.aliyuncs.com/api/v2/apps/claude-code-proxy"),
    )
    .expect("the body builds");
    assert_eq!(
        body.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 2048}))
    );
    assert!(body.get("output_config").is_none());
}
#[test]
fn a_vllm_endpoint_profile_encodes_enable_thinking_for_the_messages_api() {
    let request = ModelRequestConfig::new("glm-5")
        .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "medium"}}));
    let body = build_messages_body_with_endpoint(
        &[BaseMessage::user("hello")],
        &request,
        false,
        "openai",
        Some("vllm"),
        Some("http://localhost:8000/v1"),
    )
    .expect("the body builds");
    assert_eq!(body.get("enable_thinking"), Some(&json!(true)));
    assert_eq!(
        body.get("chat_template_kwargs"),
        Some(&json!({"enable_thinking": true}))
    );
    assert_eq!(body.get("thinking_budget"), Some(&json!(4096)));
    assert!(
        body.get("thinking").is_none(),
        "a self-hosted gateway ignores the official thinking object: {body}"
    );
    assert!(body.get("output_config").is_none());
    assert!(body.get("reasoning_effort").is_none());
}

#[test]
fn a_vllm_endpoint_profile_disables_thinking_in_both_keys() {
    let request =
        ModelRequestConfig::new("glm-5").with_extra(json!({"reasoning": {"mode": "disabled"}}));
    let body = build_messages_body_with_endpoint(
        &[BaseMessage::user("hello")],
        &request,
        false,
        "anthropic",
        Some("vllm"),
        None,
    )
    .expect("the body builds");
    assert_eq!(body.get("enable_thinking"), Some(&json!(false)));
    assert_eq!(
        body.get("chat_template_kwargs"),
        Some(&json!({"enable_thinking": false}))
    );
    assert!(body.get("thinking").is_none());
}

#[test]
fn a_non_vllm_endpoint_profile_keeps_the_profile_table_encoder() {
    let request = ModelRequestConfig::new(SONNET_45)
        .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "low"}}));
    let with_profile = build_messages_body_with_endpoint(
        &[BaseMessage::user("hello")],
        &request,
        false,
        "anthropic",
        Some("anthropic"),
        Some("https://api.anthropic.com"),
    )
    .expect("the body builds");
    let with_provider = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &request,
        false,
        "anthropic",
        None,
    )
    .expect("the body builds");
    assert_eq!(with_profile, with_provider);
    assert_eq!(
        with_profile.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 2048}))
    );
    assert!(with_profile.get("enable_thinking").is_none());
}

/// Builds a body for a full request config, so the typed fields are exercised.
fn body_for_request(request: &ModelRequestConfig) -> Value {
    build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        request,
        false,
        "anthropic",
        None,
    )
    .expect("the body builds")
}

#[test]
fn a_typed_reasoning_intent_matches_the_extra_spelling() {
    let typed = ModelRequestConfig::new(SONNET_45)
        .with_reasoning(json!({"mode": "enabled", "effort": "medium"}));
    let legacy = ModelRequestConfig::new(SONNET_45)
        .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "medium"}}));
    let typed_body = body_for_request(&typed);
    let legacy_body = body_for_request(&legacy);

    assert_eq!(typed_body, legacy_body);
    assert_eq!(
        typed_body.get("thinking"),
        Some(&json!({"type": "enabled", "budget_tokens": 8192}))
    );
    assert!(typed_body.get("reasoning").is_none());
}

#[test]
fn the_typed_reasoning_field_wins_over_the_extra_spelling() {
    let request = ModelRequestConfig::new(UNPROFILED)
        .with_reasoning(json!({"mode": "enabled", "effort": "low"}))
        .with_extra(json!({"reasoning": {"mode": "enabled", "effort": "high"}}));
    let body = body_for_request(&request);

    assert_eq!(body.get("output_config"), Some(&json!({"effort": "low"})));
    assert!(body.get("reasoning").is_none());
}

#[test]
fn a_typed_null_reasoning_value_is_no_intent() {
    let request = ModelRequestConfig::new(SONNET_45).with_reasoning(Value::Null);
    let body = body_for_request(&request);

    assert!(body.get("thinking").is_none());
    assert!(body.get("output_config").is_none());
}

#[test]
fn a_typed_non_intent_payload_is_a_validation_error() {
    let request =
        ModelRequestConfig::new(SONNET_45).with_reasoning(json!({"thinking": {"type": "enabled"}}));
    let error = build_messages_body_with_provider(
        &[BaseMessage::user("hello")],
        &request,
        false,
        "anthropic",
        None,
    )
    .expect_err("the Anthropic transport has no raw `reasoning` body key");
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
}

#[test]
fn context_window_never_reaches_the_messages_body() {
    let request = ModelRequestConfig::new(SONNET_45).with_context_window(200_000);
    let body = body_for_request(&request);

    assert!(body.get("context_window").is_none());
    let rendered = body.to_string();
    assert!(!rendered.contains("context_window"), "body: {rendered}");
}
