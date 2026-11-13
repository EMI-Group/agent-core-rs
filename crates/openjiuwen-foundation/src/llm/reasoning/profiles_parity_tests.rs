//! Python-parity pins for the reasoning profile tables.
//!
//! [`provider_table_matches_the_python_reference_in_order`] and
//! [`fallback_table_matches_the_python_reference_in_order`] compare the Rust table
//! against the hard-coded transcription in `profiles_python_rows.rs`; the other
//! tests exercise the public lookup API for the audited families and prove that a
//! provider-specific row always wins over a `"*"` fallback.
use super::super::{
    reasoning_capability, reasoning_capability_for, ReasoningProfile, ReasoningProtocol,
};
use super::{
    find_reasoning_profile, CapabilitySpec, MODEL_REASONING_FALLBACKS, REASONING_PROFILES,
};
use rows::{Cap, Row, AUDITED_FAMILIES, EXPECTED_FALLBACKS, EXPECTED_PROFILES, EXPECTED_PROVIDERS};
#[path = "profiles_python_rows.rs"]
mod rows;
/// Asserts one capability spec against its Python counterpart.
fn assert_capability(actual: CapabilitySpec, expected: Cap, context: &str) {
    let (options, recommended) = expected;
    assert_eq!(actual.options, options, "{context}: capability options");
    assert_eq!(
        actual.recommended, recommended,
        "{context}: recommended level"
    );
}
/// Asserts one table row against its Python counterpart.
fn assert_row(expected: &Row, actual: &ReasoningProfile, context: &str) {
    let (provider, patterns, capability, wire, anthropic_wire) = *expected;
    assert_eq!(actual.provider, provider, "{context}: provider");
    assert_eq!(actual.model_patterns, patterns, "{context}: model patterns");
    assert_capability(actual.capability, capability, context);
    assert!(
        actual.anthropic_capability.is_none(),
        "{context}: the Python table sets no anthropic capability"
    );
    assert_eq!(actual.wire, wire, "{context}: wire");
    assert_eq!(
        actual.anthropic_wire, anthropic_wire,
        "{context}: anthropic wire"
    );
}
#[test]
fn provider_table_matches_the_python_reference_in_order() {
    assert_eq!(
        REASONING_PROFILES.len(),
        EXPECTED_PROFILES.len(),
        "row count drifted from the Python reference"
    );
    for (index, (expected, actual)) in EXPECTED_PROFILES.iter().zip(REASONING_PROFILES).enumerate()
    {
        assert_row(expected, actual, &format!("REASONING_PROFILES[{index}]"));
    }
}
#[test]
fn fallback_table_matches_the_python_reference_in_order() {
    assert_eq!(
        MODEL_REASONING_FALLBACKS.len(),
        EXPECTED_FALLBACKS.len(),
        "row count drifted from the Python reference"
    );
    for (index, (expected, actual)) in EXPECTED_FALLBACKS
        .iter()
        .zip(MODEL_REASONING_FALLBACKS)
        .enumerate()
    {
        assert_row(
            expected,
            actual,
            &format!("MODEL_REASONING_FALLBACKS[{index}]"),
        );
    }
}
#[test]
fn providers_appear_in_the_python_reference_order() {
    let mut distinct: Vec<&str> = Vec::new();
    for profile in REASONING_PROFILES {
        if distinct.last() != Some(&profile.provider) {
            distinct.push(profile.provider);
        }
    }
    assert_eq!(distinct, EXPECTED_PROVIDERS);
}
#[test]
fn every_python_provider_pattern_pair_exists_in_the_tables() {
    let actual: Vec<(&str, &str)> = REASONING_PROFILES
        .iter()
        .chain(MODEL_REASONING_FALLBACKS)
        .flat_map(|profile| {
            profile
                .model_patterns
                .iter()
                .map(move |pattern| (profile.provider, *pattern))
        })
        .collect();
    assert_eq!(
        actual.len(),
        EXPECTED_PROFILES
            .iter()
            .chain(EXPECTED_FALLBACKS.iter())
            .map(|(_, patterns, ..)| patterns.len())
            .sum::<usize>()
    );
    for (provider, patterns, ..) in EXPECTED_PROFILES.iter().chain(EXPECTED_FALLBACKS.iter()) {
        for pattern in *patterns {
            assert!(
                actual.contains(&(*provider, *pattern)),
                "{provider}/{pattern} is missing from the Rust tables"
            );
        }
    }
}
#[test]
fn audited_families_resolve_to_their_capability_and_wire() {
    for &(provider, model, provider_key, wire, anthropic_wire, capability) in AUDITED_FAMILIES {
        let context = format!("{provider}/{model}");
        let profile = find_reasoning_profile(provider, model)
            .unwrap_or_else(|| panic!("{context} resolved to no profile"));
        assert_eq!(profile.provider, provider_key, "{context}: provider key");
        assert_eq!(profile.wire, wire, "{context}: wire");
        assert_eq!(
            profile.anthropic_wire, anthropic_wire,
            "{context}: anthropic wire"
        );
        assert_capability(profile.capability, capability, &context);
        // The public capability API resolves the same row for both protocols,
        // because the Python table sets no Anthropic override.
        let chat = reasoning_capability(provider, model);
        let options: Vec<&str> = chat.options.iter().map(String::as_str).collect();
        assert_eq!(
            options.as_slice(),
            capability.0,
            "{context}: chat capability"
        );
        assert_eq!(
            chat.recommended.as_deref(),
            capability.1,
            "{context}: chat recommendation"
        );
        let anthropic =
            reasoning_capability_for(ReasoningProtocol::AnthropicMessages, provider, model);
        let options: Vec<&str> = anthropic.options.iter().map(String::as_str).collect();
        assert_eq!(
            options.as_slice(),
            capability.0,
            "{context}: anthropic capability"
        );
        assert_eq!(
            anthropic.recommended.as_deref(),
            capability.1,
            "{context}: anthropic recommendation"
        );
    }
}
#[test]
fn provider_specific_rows_win_over_the_fallback_table() {
    // `zhipu`/`glm-5.2` is profiled with the GLM encoder; the same model on an
    // unprofiled gateway only reaches the `"*"` row.
    let zhipu = find_reasoning_profile("zhipu", "glm-5.2").expect("zhipu profiles glm-5.2");
    assert_eq!(zhipu.provider, "zhipu");
    assert_eq!(zhipu.wire, "glm");
    let gateway = find_reasoning_profile("custom-gateway", "glm-5.2").expect("fallback");
    assert_eq!(gateway.provider, "*");

    // `OpenRouter` deliberately exposes off/high/xhigh for DeepSeek V4 while the
    // cross-provider fallback keeps the official off/low/high/max levels.
    let openrouter =
        find_reasoning_profile("openrouter", "deepseek/deepseek-v4-pro").expect("openrouter row");
    assert_eq!(openrouter.provider, "openrouter");
    assert_eq!(openrouter.capability.options, &["off", "high", "xhigh"][..]);
    let fallback = find_reasoning_profile("custom-gateway", "deepseek/deepseek-v4-pro")
        .expect("fallback via the last model segment");
    assert_eq!(fallback.provider, "*");
    assert_eq!(
        fallback.capability.options,
        &["off", "low", "high", "max"][..]
    );
}
#[test]
fn fallback_rows_resolve_through_the_last_model_segment() {
    for (_, patterns, capability, wire, _) in EXPECTED_FALLBACKS {
        let family = patterns[0].replace('*', "").replace('?', "1");
        let model = format!("vendor/{family}");
        let profile = find_reasoning_profile("custom-gateway", &model)
            .unwrap_or_else(|| panic!("{model} resolved to no fallback"));
        assert_eq!(profile.provider, "*", "{model}: provider key");
        assert_eq!(profile.wire, wire, "{model}: wire");
        assert_capability(profile.capability, capability, &model);
    }
}
