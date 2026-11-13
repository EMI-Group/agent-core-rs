//! Self-hosted `vLLM` gateway encoding (Python's `_vllm_enable_thinking_plan`).
//!
//! A self-hosted OpenAI-compatible gateway serving `GLM`/`DeepSeek`/`Qwen` honors
//! the `DashScope`-style `enable_thinking` switch (in the body and again inside
//! `chat_template_kwargs`) and ignores the official `thinking.type` /
//! `reasoning_effort` spellings, so intensity is encoded as `thinking_budget`.
//! Callers opt in with `endpoint_profile = "vllm"`; no API-base host is
//! hardcoded here, so a deployment under any host can select this encoder.
//!
//! `reasoning_effort` is deliberately never sent: the gateway ignores it, and
//! official `DashScope` rejects it together with `thinking_budget`.

use serde_json::{json, Map, Value};

use super::{ReasoningConfig, ReasoningMode, ReasoningPlan};

/// Labelled effort → `thinking_budget`, the same scale `DashScope` `Qwen` uses.
const BUDGET_BY_EFFORT: [(&str, u32); 6] = [
    ("minimal", 1024),
    ("low", 1024),
    ("medium", 4096),
    ("high", 8192),
    ("xhigh", 16384),
    ("max", 32768),
];

/// Effort spellings that switch thinking off, including the `"disabled"` word.
const DISABLED_EFFORTS: [&str; 3] = ["off", "none", "disabled"];

/// Encodes one intent for a self-hosted `vLLM` gateway.
///
/// - A disabled mode or a disabled effort writes the switch off in both places.
/// - `auto` with no effort and no budget sends nothing at all.
/// - Otherwise the switch is written on, plus a `thinking_budget` taken from an
///   explicit `budget_tokens` (which wins) or from [`BUDGET_BY_EFFORT`]; an
///   unrecognized effort sends the switch without an intensity.
#[must_use]
pub(super) fn vllm_enable_thinking_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.mode == ReasoningMode::Disabled || disabled_effort(intent.effort.as_deref()) {
        return switch_plan(false, None);
    }
    let budget = intent.budget_tokens.or_else(|| {
        labeled_effort(intent.effort.as_deref()).and_then(|effort| {
            BUDGET_BY_EFFORT
                .iter()
                .find_map(|(label, budget)| (*label == effort).then_some(*budget))
        })
    });
    if intent.mode == ReasoningMode::Auto && budget.is_none() && !has_effort(intent) {
        return ReasoningPlan::new();
    }
    switch_plan(true, budget)
}

/// Builds a plan carrying the switch pair and an optional intensity.
fn switch_plan(enabled: bool, budget: Option<u32>) -> ReasoningPlan {
    let mut extra_body = Map::new();
    extra_body.insert("enable_thinking".to_owned(), Value::Bool(enabled));
    extra_body.insert(
        "chat_template_kwargs".to_owned(),
        json!({ "enable_thinking": enabled }),
    );
    if let Some(budget) = budget {
        extra_body.insert("thinking_budget".to_owned(), json!(budget));
    }
    ReasoningPlan {
        sdk_params: Map::new(),
        extra_body,
        warnings: Vec::new(),
    }
}

/// Whether the intent names an effort at all (even one with no known budget).
fn has_effort(intent: &ReasoningConfig) -> bool {
    labeled_effort(intent.effort.as_deref()).is_some() || disabled_effort(intent.effort.as_deref())
}

/// The trimmed, lowercased effort, when it is not a disable marker.
fn labeled_effort(effort: Option<&str>) -> Option<String> {
    let value = effort?.trim().to_lowercase();
    (!value.is_empty() && !disabled_effort(Some(&value))).then_some(value)
}

/// Whether `effort` is one of the spellings that switch thinking off.
fn disabled_effort(effort: Option<&str>) -> bool {
    effort
        .map(|effort| effort.trim().to_lowercase())
        .is_some_and(|effort| DISABLED_EFFORTS.contains(&effort.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_mode_switches_both_keys_off() {
        let plan = vllm_enable_thinking_plan(&ReasoningConfig::disabled());
        assert_eq!(plan.extra_body["enable_thinking"], json!(false));
        assert_eq!(
            plan.extra_body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert!(plan.sdk_params.is_empty());
    }

    #[test]
    fn auto_without_effort_or_budget_sends_nothing() {
        let plan = vllm_enable_thinking_plan(&ReasoningConfig::new());
        assert!(plan.is_empty());
    }

    #[test]
    fn enabled_mode_switches_thinking_on_without_a_budget() {
        let plan =
            vllm_enable_thinking_plan(&ReasoningConfig::new().with_mode(ReasoningMode::Enabled));
        assert_eq!(plan.extra_body["enable_thinking"], json!(true));
        assert_eq!(
            plan.extra_body["chat_template_kwargs"],
            json!({"enable_thinking": true})
        );
        assert!(plan.extra_body.get("thinking_budget").is_none());
    }

    #[test]
    fn budget_wins_over_the_labelled_effort() {
        let plan = vllm_enable_thinking_plan(
            &ReasoningConfig::new()
                .with_mode(ReasoningMode::Enabled)
                .with_effort("max")
                .with_budget_tokens(2048),
        );
        assert_eq!(plan.extra_body["thinking_budget"], json!(2048));
    }

    #[test]
    fn unknown_effort_sends_the_switch_without_an_intensity() {
        let plan = vllm_enable_thinking_plan(
            &ReasoningConfig::new()
                .with_mode(ReasoningMode::Auto)
                .with_effort("turbo"),
        );
        assert_eq!(plan.extra_body["enable_thinking"], json!(true));
        assert!(plan.extra_body.get("thinking_budget").is_none());
    }

    #[test]
    fn every_labelled_effort_maps_onto_its_budget() {
        for (effort, budget) in BUDGET_BY_EFFORT {
            let plan = vllm_enable_thinking_plan(
                &ReasoningConfig::new()
                    .with_mode(ReasoningMode::Enabled)
                    .with_effort(effort),
            );
            assert_eq!(
                plan.extra_body["thinking_budget"],
                json!(budget),
                "{effort}"
            );
        }
    }

    #[test]
    fn disabled_effort_spellings_switch_thinking_off() {
        for effort in ["off", "none", "disabled", " OFF ", "None"] {
            let plan = vllm_enable_thinking_plan(
                &ReasoningConfig::new()
                    .with_mode(ReasoningMode::Auto)
                    .with_effort(effort),
            );
            assert_eq!(plan.extra_body["enable_thinking"], json!(false), "{effort}");
        }
    }

    #[test]
    fn reasoning_effort_is_never_emitted() {
        let plan = vllm_enable_thinking_plan(
            &ReasoningConfig::new()
                .with_mode(ReasoningMode::Enabled)
                .with_effort("high"),
        );
        assert!(plan.sdk_params.get("reasoning_effort").is_none());
        assert!(!plan.extra_body.contains_key("reasoning_effort"));
    }
}
