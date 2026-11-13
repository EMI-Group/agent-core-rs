use super::*;
use serde_json::json;
fn chat(provider: &str, model: &str, intent: ReasoningConfig) -> ReasoningPlan {
    resolve_reasoning_plan(provider, None, model, &intent)
}
fn anthropic(provider: &str, model: &str, intent: ReasoningConfig) -> ReasoningPlan {
    resolve_reasoning_plan_for(
        ReasoningProtocol::AnthropicMessages,
        provider,
        None,
        model,
        &intent,
    )
}
fn enabled() -> ReasoningConfig {
    ReasoningConfig::new().with_mode(ReasoningMode::Enabled)
}
fn disabled() -> ReasoningConfig {
    ReasoningConfig::disabled()
}
fn effort(value: &str) -> ReasoningConfig {
    ReasoningConfig::new().with_effort(value)
}
fn enabled_effort(value: &str) -> ReasoningConfig {
    enabled().with_effort(value)
}
#[test]
fn no_intent_resolves_to_an_empty_plan_for_every_family() {
    for (provider, model) in [
        ("openai", "gpt-5.1"),
        ("deepseek", "deepseek-v4-pro"),
        ("dashscope", "qwen3-max"),
        ("zhipu", "glm-5.2"),
        ("minimax", "minimax-m2.5"),
        ("ollama", "llama3.1"),
    ] {
        let plan = chat(provider, model, ReasoningConfig::new());
        assert!(plan.is_empty(), "{provider}/{model}: {plan:?}");
        assert!(!plan.has_warnings(), "{provider}/{model}: {plan:?}");
    }
}
#[test]
fn standard_openai_encodes_effort_and_disable() {
    let plan = chat("openai", "gpt-5.1", enabled_effort("high"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("high"));
    assert!(plan.extra_body.is_empty());
    let plan = chat("openai", "gpt-5.1", disabled());
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("none"));
    let plan = chat("openai", "gpt-5.1", effort("low"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("low"));
    assert!(chat("openai", "gpt-5.1", enabled()).is_empty());
}
#[test]
fn unknown_models_get_the_generic_effort_encoding() {
    let plan = chat("ollama", "llama3.1", enabled_effort("low"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("low"));
    assert!(
        plan.warnings.is_empty(),
        "no warning spam for unknown models"
    );
    let plan = chat("ollama", "llama3.1", disabled());
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("none"));
    assert!(chat("ollama", "llama3.1", enabled()).is_empty());
}
#[test]
fn standard_anthropic_encodes_adaptive_thinking_budget_and_effort() {
    let plan = anthropic("openai", "unlisted-model", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "adaptive"}));
    assert!(chat("openai", "unlisted-model", enabled()).is_empty());

    let plan = anthropic(
        "openai",
        "unlisted-model",
        enabled().with_budget_tokens(4096),
    );
    assert_eq!(
        plan.sdk_params["thinking"],
        json!({"type": "enabled", "budget_tokens": 4096})
    );

    let plan = anthropic("openai", "unlisted-model", enabled_effort("high"));
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "adaptive"}));
    assert_eq!(plan.sdk_params["output_config"], json!({"effort": "high"}));

    let plan = anthropic("openai", "unlisted-model", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
}
#[test]
fn anthropic_manual_maps_efforts_onto_budgets() {
    let plan = anthropic("anthropic", "claude-sonnet-4-5", enabled());
    assert_eq!(
        plan.sdk_params["thinking"],
        json!({"type": "enabled", "budget_tokens": 8192})
    );
    for (effort_value, budget) in [("low", 2048), ("medium", 8192), ("high", 16_384)] {
        let plan = anthropic(
            "anthropic",
            "claude-sonnet-4-5-20250929",
            enabled_effort(effort_value),
        );
        assert_eq!(plan.sdk_params["thinking"]["budget_tokens"], json!(budget));
    }
    let plan = anthropic(
        "anthropic",
        "claude-sonnet-4-5",
        enabled().with_budget_tokens(1500),
    );
    assert_eq!(plan.sdk_params["thinking"]["budget_tokens"], json!(1500));
    let plan = anthropic("anthropic", "claude-sonnet-4-5", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
}
#[test]
fn dashscope_budget_keeps_the_budget_optional() {
    let plan = anthropic("dashscope", "qwen3-max", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    let plan = anthropic("dashscope", "qwen3-max", enabled().with_budget_tokens(2048));
    assert_eq!(
        plan.sdk_params["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );
    let plan = anthropic("dashscope", "qwen3-max", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
}
#[test]
fn output_config_effort_prefers_a_budget_over_an_effort() {
    let plan = anthropic("dashscope", "glm-5.2", enabled_effort("max"));
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.sdk_params["output_config"], json!({"effort": "max"}));

    let plan = anthropic(
        "dashscope",
        "glm-5.2",
        enabled_effort("max").with_budget_tokens(1024),
    );
    assert_eq!(
        plan.sdk_params["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
    assert!(!plan.sdk_params.contains_key("output_config"));

    let plan = anthropic("dashscope", "glm-5.2", effort("off"));
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));

    let plan = anthropic("dashscope", "qwen3.8-max", enabled_effort("medium"));
    assert_eq!(
        plan.sdk_params["output_config"],
        json!({"effort": "medium"})
    );
}
#[test]
fn enable_thinking_switch_and_budget() {
    let plan = chat("dashscope", "qwen3-max", enabled());
    assert_eq!(plan.extra_body["enable_thinking"], json!(true));
    let plan = chat("dashscope", "qwen3-max", disabled());
    assert_eq!(plan.extra_body["enable_thinking"], json!(false));
    let plan = chat("dashscope", "qwen3-max", enabled_effort("off"));
    assert_eq!(plan.extra_body["enable_thinking"], json!(false));
    // A level the toggle cannot express is reported, but the budget still
    // reaches the wire.
    let plan = chat(
        "dashscope",
        "qwen-plus",
        effort("high").with_budget_tokens(2048),
    );
    assert_eq!(plan.extra_body["thinking_budget"], json!(2048));
    assert!(!plan.extra_body.contains_key("enable_thinking"));
    assert_eq!(plan.warnings.len(), 1);
}
#[test]
fn qwen38_uses_enable_thinking_plus_a_labeled_effort() {
    let plan = chat("dashscope", "qwen3.8-max", enabled_effort("xhigh"));
    assert_eq!(plan.extra_body["enable_thinking"], json!(true));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("xhigh"));
    let plan = chat("dashscope", "qwen3.8-max", effort("low"));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("low"));
    assert!(!plan.extra_body.contains_key("enable_thinking"));
    let plan = chat("dashscope", "qwen3.8-max", effort("off"));
    assert_eq!(
        plan.extra_body,
        Map::from_iter([("enable_thinking".to_owned(), json!(false))])
    );
    let plan = chat(
        "dashscope",
        "qwen3.8-max",
        enabled().with_budget_tokens(3000),
    );
    assert_eq!(plan.extra_body["thinking_budget"], json!(3000));
    assert!(!plan.extra_body.contains_key("reasoning_effort"));
}
#[test]
fn deepseek_normalizes_effort_and_places_thinking_in_the_extension_body() {
    let plan = chat("deepseek", "deepseek-v4-pro", enabled_effort("high"));
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("high"));
    let plan = chat("deepseek", "deepseek-v4-pro", enabled_effort("max"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("max"));
    let plan = chat("deepseek", "deepseek-v4-pro", effort("low"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("low"));
    assert!(!plan.extra_body.contains_key("thinking"));
    let plan = chat("deepseek", "deepseek-v4-pro", disabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
    assert!(plan.sdk_params.is_empty());
    // A generic OpenAI-compatible provider pointed at a `DeepSeek` host still
    // gets the `DeepSeek` encoding.
    let plan = resolve_reasoning_plan(
        "openai",
        Some("https://api.deepseek.com/v1"),
        "deepseek-v4-pro",
        &disabled(),
    );
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
}
#[test]
fn deepseek_anthropic_uses_top_level_thinking_and_output_config() {
    let plan = anthropic("deepseek", "deepseek-v4-pro", enabled_effort("low"));
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.sdk_params["output_config"], json!({"effort": "low"}));
    let plan = anthropic("deepseek", "deepseek-v4-pro", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    let plan = anthropic("deepseek", "deepseek-v4-pro", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
}
#[test]
fn glm_sends_thinking_and_a_normalized_effort() {
    let plan = chat("zhipu", "glm-5.2", enabled_effort("high"));
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("high"));
    let plan = chat("zhipu", "glm-5.2", enabled_effort("max"));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("max"));
    let plan = chat("zhipu", "glm-5.2", enabled_effort("off"));
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
    let plan = chat("zhipu", "glm-5.2", disabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
    // glm-5 is a plain toggle: the label is rejected by the capability check.
    let plan = chat("zhipu", "glm-5", enabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
}
#[test]
fn glm_anthropic_moves_the_effort_into_the_extension_body() {
    let plan = anthropic("zhipu", "glm-5.2", enabled_effort("max"));
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("max"));
    let plan = anthropic("zhipu", "glm-5.2", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
}
#[test]
fn kimi_k3_uses_labeled_efforts_only() {
    let plan = chat("moonshot", "kimi-k3", enabled_effort("high"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("high"));
    let plan = chat("moonshot", "kimi-k3-preview", enabled_effort("max"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("max"));
    assert!(chat("moonshot", "kimi-k3", disabled()).is_empty());
    let plan = anthropic("moonshot", "kimi-k3", enabled_effort("low"));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("low"));
}
#[test]
fn kimi_k2_families_are_plain_toggles() {
    let plan = chat("moonshot", "kimi-k2.6", enabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    let plan = chat("moonshot", "kimi-k2.5", disabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
    let plan = anthropic("moonshot", "kimi-k2.6", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    let plan = anthropic("moonshot", "kimi-k2.6", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
    assert!(anthropic("moonshot", "kimi-k2.6", effort("high")).is_empty());
}
#[test]
fn minimax_m3_is_adaptive_and_m2_forces_reasoning_split() {
    let plan = chat("minimax", "minimax-m3", enabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "adaptive"}));
    let plan = chat("minimax", "minimax-m3", disabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
    let plan = chat("minimax", "minimax-m2.5", enabled());
    assert_eq!(plan.extra_body["reasoning_split"], json!(true));
    // Always-on families keep the split flag even when asked to disable.
    let plan = chat("minimax", "minimax-m2.5", disabled());
    assert_eq!(plan.extra_body["reasoning_split"], json!(true));
    let plan = anthropic("minimax", "minimax-m3", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "adaptive"}));
    let plan = anthropic("minimax", "minimax-m3", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
    assert!(anthropic("minimax", "minimax-m2.5", enabled()).is_empty());
}
#[test]
fn volcengine_seed_uses_reasoning_effort_with_minimal_as_the_disable_value() {
    let plan = chat("volcengine", "seed-2.0-pro", enabled_effort("medium"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("medium"));
    let plan = chat("volcengine", "seed-2.0-pro", disabled());
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("minimal"));
    let plan = chat("volcengine", "doubao-seed-2-0-pro", effort("off"));
    assert_eq!(plan.sdk_params["reasoning_effort"], json!("minimal"));
    assert!(chat("volcengine", "seed-2.0-pro", enabled()).is_empty());
}
#[test]
fn qianfan_toggle_reuses_the_enable_thinking_encoder() {
    let plan = chat("qianfan", "ernie-5.0", enabled());
    assert_eq!(plan.extra_body["enable_thinking"], json!(true));
    let plan = chat("qianfan", "emie-5.0", disabled());
    assert_eq!(plan.extra_body["enable_thinking"], json!(false));
}
#[test]
fn modelarts_deepseek_is_a_toggle_only_family() {
    let plan = chat("modelarts", "deepseek-v4-pro", enabled());
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    let plan = anthropic("modelarts", "deepseek-v4-pro", enabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "enabled"}));
    let plan = anthropic("modelarts", "deepseek-v4-pro", disabled());
    assert_eq!(plan.sdk_params["thinking"], json!({"type": "disabled"}));
    // The family exposes off/on only, so a labeled level is reported.
    let plan = chat("modelarts", "deepseek-v4-pro", enabled_effort("low"));
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.warnings.len(), 1);
}
#[test]
fn openrouter_encodes_one_normalized_reasoning_object() {
    let plan = chat(
        "openrouter",
        "openai/gpt-5.6-preview",
        enabled_effort("high"),
    );
    assert_eq!(
        plan.extra_body["reasoning"],
        json!({"enabled": true, "effort": "high"})
    );
    let plan = chat("openrouter", "openai/gpt-5.6-preview", disabled());
    assert_eq!(plan.extra_body["reasoning"], json!({"effort": "none"}));
    let plan = chat(
        "openrouter",
        "openai/gpt-5.6-preview",
        enabled().with_budget_tokens(2048),
    );
    assert_eq!(
        plan.extra_body["reasoning"],
        json!({"enabled": true, "max_tokens": 2048})
    );
    // Unknown vendor prefixes are left alone.
    assert!(chat("openrouter", "mistralai/mixtral-8x7b", enabled()).is_empty());
    assert!(chat(
        "openrouter",
        "openai/gpt-5.6-preview",
        ReasoningConfig::new()
    )
    .is_empty());
}
#[test]
fn unsupported_and_always_on_families_send_nothing() {
    assert!(chat("zhipu", "codegeex-4", enabled()).is_empty());
    assert!(chat("zhipu", "codegeex-4", disabled()).is_empty());
    assert!(chat("moonshot", "kimi-k2-thinking", enabled()).is_empty());
    assert!(chat("volcengine", "doubao-1.5-pro", enabled()).is_empty());
    // The family has no control, so an explicit effort is rejected with a
    // warning instead of being silently dropped.
    let plan = chat("zhipu", "codegeex-4", enabled_effort("high"));
    assert!(plan.is_empty());
    assert_eq!(plan.warnings.len(), 1);
    assert!(
        plan.warnings[0].contains("codegeex-4"),
        "{:?}",
        plan.warnings
    );
    assert!(plan.warnings[0].contains("no selectable reasoning control"));
    let plan = chat(
        "moonshot",
        "kimi-k2-thinking",
        ReasoningConfig::new().with_budget_tokens(1024),
    );
    assert!(plan.is_empty());
    assert!(plan.warnings[0].contains("budget_tokens"));
}
#[test]
fn capability_rejection_names_the_model_and_supported_values() {
    // gpt-5 accepts minimal/low/medium/high but not max.
    let plan = chat("openai", "gpt-5", enabled_effort("max"));
    assert!(plan.is_empty());
    assert_eq!(plan.warnings.len(), 1);
    let warning = &plan.warnings[0];
    assert!(warning.contains("\"max\""), "{warning}");
    assert!(warning.contains("gpt-5"), "{warning}");
    assert!(warning.contains("minimal, low, medium, high"), "{warning}");
    // A toggle-only family cannot express a labeled level: the level is
    // dropped and reported, but the explicit `enabled` mode is still encoded.
    let plan = chat("zhipu", "glm-5", enabled_effort("medium"));
    assert_eq!(plan.extra_body["thinking"], json!({"type": "enabled"}));
    assert_eq!(plan.warnings.len(), 1);
    assert!(plan.warnings[0].contains("off, on"), "{:?}", plan.warnings);
    // Supported levels never warn.
    assert!(!chat("openai", "gpt-5", enabled_effort("high")).has_warnings());
}
#[test]
fn responses_protocol_encodes_a_nested_reasoning_object() {
    let plan = resolve_reasoning_plan_for(
        ReasoningProtocol::Responses,
        "openai",
        None,
        "gpt-5.1",
        &enabled_effort("low"),
    );
    assert_eq!(plan.sdk_params["reasoning"], json!({"effort": "low"}));
    let plan = resolve_reasoning_plan_for(
        ReasoningProtocol::Responses,
        "openai",
        None,
        "gpt-5.1",
        &disabled(),
    );
    assert_eq!(plan.sdk_params["reasoning"], Value::Null);
    let plan = resolve_reasoning_plan_for(
        ReasoningProtocol::Responses,
        "openai",
        None,
        "gpt-5.1",
        &enabled(),
    );
    assert!(plan.is_empty());
}
#[test]
fn provider_resolution_accepts_vendor_spellings_and_host_hints() {
    // `qwen` is this crate's canonical name for `DashScope`.
    let plan = chat("qwen", "qwen3.8-max", enabled_effort("low"));
    assert_eq!(plan.extra_body["reasoning_effort"], json!("low"));
    // A bare host hint reaches the same profile through a generic provider.
    let plan = resolve_reasoning_plan(
        "openai",
        Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "qwen3.8-max",
        &enabled_effort("low"),
    );
    assert_eq!(plan.extra_body["reasoning_effort"], json!("low"));
}
#[test]
fn normalizers_match_the_python_tables() {
    assert_eq!(normalize_deepseek_effort(Some("minimal"), ""), Some("low"));
    assert_eq!(
        normalize_deepseek_effort(Some(" medium "), ""),
        Some("high")
    );
    assert_eq!(
        normalize_deepseek_effort(Some("xhigh"), "deepseek-v4-pro"),
        Some("max")
    );
    assert_eq!(
        normalize_deepseek_effort(Some("xhigh"), "deepseek-v4"),
        Some("high")
    );
    assert_eq!(normalize_deepseek_effort(Some("banana"), ""), None);
    assert_eq!(normalize_deepseek_effort(None, ""), None);

    assert_eq!(normalize_glm_effort(Some("minimal")), Some("none"));
    assert_eq!(normalize_glm_effort(Some("medium")), Some("high"));
    assert_eq!(normalize_glm_effort(Some("max")), Some("max"));
    assert_eq!(normalize_glm_effort(Some("banana")), None);

    assert_eq!(
        normalize_volcengine_seed_effort(Some("off")),
        Some("minimal")
    );
    assert_eq!(
        normalize_volcengine_seed_effort(Some("xhigh")),
        Some("high")
    );

    assert_eq!(normalize_kimi_k3_effort(Some("ultra")), Some("max"));
    assert_eq!(normalize_kimi_k3_effort(Some("light")), Some("low"));
    assert_eq!(normalize_kimi_k3_effort(Some("high")), Some("high"));

    assert_eq!(normalize_qwen38_effort(Some("off")), Some("none"));
    assert_eq!(normalize_qwen38_effort(Some("minimal")), Some("low"));
    assert_eq!(normalize_qwen38_effort(Some("max")), Some("xhigh"));
    assert_eq!(normalize_qwen38_effort(Some("  ")), None);

    assert_eq!(normalize_modelarts_effort(Some("low")), Some("high"));
    assert_eq!(normalize_modelarts_effort(None), None);
}
#[test]
fn moonshot_families_are_classified_from_the_last_segment() {
    assert_eq!(moonshot_family("moonshotai/kimi-k3"), MoonshotFamily::K3);
    assert_eq!(moonshot_family("kimi-k2.7-code"), MoonshotFamily::K27Code);
    assert_eq!(moonshot_family("Kimi-K2.6"), MoonshotFamily::K2Toggle);
    assert_eq!(moonshot_family("kimi-k1"), MoonshotFamily::Unknown);
}
/// Wires the resolver answers before dispatching to an encoder: `profile_plan`
/// short-circuits them, so they deliberately have no `encode_wire` arm.
const GUARD_WIRES: [&str; 3] = ["always_on", "minimax_m2", "unsupported"];
#[test]
fn every_table_wire_dispatches_to_a_real_encoder() {
    use super::super::profiles::{MODEL_REASONING_FALLBACKS, REASONING_PROFILES};
    for profile in REASONING_PROFILES.iter().chain(MODEL_REASONING_FALLBACKS) {
        for (protocol, wire) in [
            (ReasoningProtocol::ChatCompletions, profile.wire),
            (ReasoningProtocol::AnthropicMessages, profile.anthropic_wire),
        ] {
            if GUARD_WIRES.contains(&wire) {
                continue;
            }
            let context = format!("{}/{wire}", profile.provider);
            let capability = profile.capability_spec_for(protocol).to_capability();
            let effort = capability
                .recommended
                .clone()
                .or_else(|| {
                    capability
                        .options
                        .iter()
                        .find(|option| option.as_str() != "off")
                        .cloned()
                })
                .unwrap_or_else(|| panic!("{context}: row exposes no selectable level"));
            let model = profile.model_patterns[0]
                .replace('*', "sample")
                .replace('?', "1");
            let model_key = model.to_lowercase();
            let intent = enabled_effort(&effort);
            let plan = encode_wire(wire, protocol, &model, &model_key, &intent);
            assert!(
                !plan.is_empty(),
                "{context}: no encoder arm produced a plan for {model}"
            );
        }
    }
}
