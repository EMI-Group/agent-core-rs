//! Cross-cutting reasoning tests: the profile table, resolution, and application
//! used together, plus invariants that keep the table honest.
use super::profiles::{find_reasoning_profile, MODEL_REASONING_FALLBACKS, REASONING_PROFILES};
use super::{
    apply_reasoning_plan, apply_reasoning_plan_with, matches_pattern, resolve_reasoning_plan,
    resolve_reasoning_plan_for, ReasoningConfig, ReasoningMode, ReasoningPlan, ReasoningProtocol,
};
use serde_json::{json, Map, Value};
/// Every encoder name the resolver understands.
const KNOWN_WIRES: [&str; 25] = [
    "always_on",
    "anthropic_manual",
    "anthropic_toggle",
    "dashscope_budget",
    "dashscope_glm_anthropic",
    "deepseek",
    "deepseek_anthropic",
    "enable_thinking",
    "glm",
    "glm_anthropic",
    "kimi_k3",
    "minimax_m2",
    "minimax_m3",
    "minimax_m3_anthropic",
    "modelarts_deepseek",
    "modelarts_deepseek_anthropic",
    "openrouter",
    "qianfan_toggle",
    "qwen38",
    "qwen38_anthropic",
    "standard_anthropic",
    "standard_openai",
    "thinking_toggle",
    "unsupported",
    "volcengine_seed",
];
/// Turns a glob into a concrete model id so every table entry can be exercised.
fn sample_model(pattern: &str) -> String {
    pattern.replace('*', "sample").replace('?', "1")
}
#[test]
fn every_profile_uses_a_known_encoder() {
    for profile in REASONING_PROFILES.iter().chain(MODEL_REASONING_FALLBACKS) {
        assert!(
            KNOWN_WIRES.contains(&profile.wire),
            "{} uses unknown wire {}",
            profile.provider,
            profile.wire
        );
        assert!(
            KNOWN_WIRES.contains(&profile.anthropic_wire),
            "{} uses unknown anthropic wire {}",
            profile.provider,
            profile.anthropic_wire
        );
    }
}
#[test]
fn every_capability_is_lowercase_unique_and_self_consistent() {
    for profile in REASONING_PROFILES.iter().chain(MODEL_REASONING_FALLBACKS) {
        let capability = profile.capability.to_capability();
        for option in &capability.options {
            assert_eq!(*option, option.to_lowercase(), "{option} is not lower case");
        }
        let mut seen = capability.options.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            capability.options.len(),
            "{} has duplicate options",
            profile.provider
        );
        if let Some(recommended) = &capability.recommended {
            assert!(
                capability.supports(recommended),
                "{} recommends unsupported {recommended}",
                profile.provider
            );
        }
        assert_ne!(
            profile.model_patterns,
            &[] as &[&str],
            "{} has no model patterns",
            profile.provider
        );
        for pattern in profile.model_patterns {
            assert_ne!(*pattern, "", "{} has an empty pattern", profile.provider);
        }
    }
}
#[test]
fn every_table_entry_is_reachable_by_a_matching_model_id() {
    for profile in REASONING_PROFILES {
        let model = sample_model(profile.model_patterns[0]);
        let found = find_reasoning_profile(profile.provider, &model)
            .unwrap_or_else(|| panic!("{} {model} did not resolve", profile.provider));
        assert!(
            found
                .model_patterns
                .iter()
                .any(|pattern| matches_pattern(pattern, &model)),
            "{} {model} resolved to another profile",
            profile.provider
        );
    }
    for profile in MODEL_REASONING_FALLBACKS {
        let model = sample_model(profile.model_patterns[0]);
        assert!(
            find_reasoning_profile("custom-gateway", &model).is_some(),
            "fallback {} {model} did not resolve",
            profile.provider
        );
    }
}
#[test]
fn resolution_never_panics_for_any_table_entry() {
    let intents = [
        ReasoningConfig::new(),
        ReasoningConfig::new().with_mode(ReasoningMode::Enabled),
        ReasoningConfig::disabled(),
        ReasoningConfig::new().with_effort("high"),
        ReasoningConfig::new()
            .with_mode(ReasoningMode::Enabled)
            .with_effort("medium")
            .with_budget_tokens(2048),
    ];
    for profile in REASONING_PROFILES.iter().chain(MODEL_REASONING_FALLBACKS) {
        let model = sample_model(profile.model_patterns[0]);
        for intent in &intents {
            for protocol in [
                ReasoningProtocol::ChatCompletions,
                ReasoningProtocol::Responses,
                ReasoningProtocol::AnthropicMessages,
            ] {
                let plan =
                    resolve_reasoning_plan_for(protocol, profile.provider, None, &model, intent);
                assert!(
                    plan.warnings.len() <= 1,
                    "{} {model} produced warning spam: {:?}",
                    profile.provider,
                    plan.warnings
                );
                if intent.is_no_intent() {
                    assert!(
                        plan.is_empty(),
                        "{} {model} sent {}",
                        profile.provider,
                        plan.warnings.len()
                    );
                }
            }
        }
    }
}
#[test]
fn shorthand_json_intents_resolve_end_to_end() {
    let intent = ReasoningConfig::from_value(&json!("high"))
        .expect("a string shorthand parses")
        .expect("a string is an intent");
    let plan = resolve_reasoning_plan("openai", None, "gpt-5.1", &intent);
    let mut body = json!({"model": "gpt-5.1", "messages": []});
    apply_reasoning_plan(&mut body, &plan);
    assert_eq!(body["reasoning_effort"], json!("high"));

    let intent = ReasoningConfig::from_value(&json!({"mode": "disabled"}))
        .expect("an intent object parses")
        .expect("an intent object is an intent");
    let plan = resolve_reasoning_plan("deepseek", None, "deepseek-v4-pro", &intent);
    let mut body = json!({"model": "deepseek-v4-pro"});
    apply_reasoning_plan(&mut body, &plan);
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
}
#[test]
fn empty_plan_keeps_a_body_byte_identical() {
    let before = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
        "temperature": 0.7,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto"
    });
    let mut body = before.clone();
    let plan = resolve_reasoning_plan("openai", None, "gpt-4o", &ReasoningConfig::new());
    assert!(plan.is_empty());
    apply_reasoning_plan(&mut body, &plan);
    assert_eq!(
        serde_json::to_string(&body).unwrap(),
        serde_json::to_string(&before).unwrap()
    );
    let mut overridden = before.clone();
    apply_reasoning_plan_with(&mut overridden, &plan, true);
    assert_eq!(
        serde_json::to_string(&overridden).unwrap(),
        serde_json::to_string(&before).unwrap()
    );
}
#[test]
fn a_plan_with_only_warnings_still_never_touches_the_body() {
    let intent = ReasoningConfig::new()
        .with_mode(ReasoningMode::Enabled)
        .with_effort("max");
    let plan = resolve_reasoning_plan("zhipu", None, "codegeex-4", &intent);
    assert!(plan.is_empty());
    assert_eq!(plan.warnings.len(), 1);
    let mut body = json!({"model": "codegeex-4"});
    let before = body.clone();
    apply_reasoning_plan(&mut body, &plan);
    assert_eq!(body, before);
}
#[test]
fn caller_extra_can_override_a_plan_key() {
    let intent = ReasoningConfig::new()
        .with_mode(ReasoningMode::Enabled)
        .with_effort("low");
    let plan = resolve_reasoning_plan("openai", None, "gpt-5.1", &intent);
    let mut body = Value::Object(Map::from_iter([("model".to_owned(), json!("gpt-5.1"))]));
    apply_reasoning_plan(&mut body, &plan);
    // The request builder merges `extra` after the plan, so a caller override
    // wins even though `apply_reasoning_plan` filled the key first.
    if let Some(object) = body.as_object_mut() {
        object.insert("reasoning_effort".to_owned(), json!("high"));
    }
    assert_eq!(body["reasoning_effort"], json!("high"));
}
#[test]
fn tables_have_no_duplicate_provider_pattern_pairs() {
    for table in [REASONING_PROFILES, MODEL_REASONING_FALLBACKS] {
        let mut pairs: Vec<(&str, &str)> = table
            .iter()
            .flat_map(|profile| {
                profile
                    .model_patterns
                    .iter()
                    .map(move |pattern| (profile.provider, *pattern))
            })
            .collect();
        let before = pairs.len();
        pairs.sort_unstable();
        pairs.dedup();
        assert_eq!(before, pairs.len(), "duplicate provider/pattern entry");
    }
}
#[test]
fn plans_are_serializable_and_round_trip() {
    let intent = ReasoningConfig::new()
        .with_mode(ReasoningMode::Enabled)
        .with_budget_tokens(4096);
    let plan = resolve_reasoning_plan_for(
        ReasoningProtocol::AnthropicMessages,
        "anthropic",
        None,
        "claude-sonnet-4-5",
        &intent,
    );
    let value = serde_json::to_value(&plan).expect("plans serialize");
    let parsed: ReasoningPlan = serde_json::from_value(value).expect("plans deserialize");
    assert_eq!(parsed, plan);
    assert_eq!(parsed.sdk_params["thinking"]["budget_tokens"], json!(4096));
}
