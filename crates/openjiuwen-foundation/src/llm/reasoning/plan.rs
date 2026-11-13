//! Intent resolution: from a neutral [`ReasoningConfig`] to provider wire
//! parameters.
//!
//! Ports `openjiuwen/core/foundation/llm/reasoning.py`. Resolution is pure and
//! side-effect free, so a request body never depends on hidden state:
//!
//! ```text
//! intent (mode/effort/budget_tokens)
//!   └─ no intent?            → empty plan
//!   └─ protocol             → chat_completions | responses | anthropic_messages
//!        └─ provider profile → capability check, then wire encoder
//!        └─ no profile       → protocol default encoder
//! ```
//!
//! Every encoder is a private function named after the Python one it mirrors, so
//! a vendor change is a table edit plus (at most) one encoder.
use super::profiles::{find_reasoning_profile, provider_key_for};
use super::{
    CapabilitySpec, ReasoningConfig, ReasoningMode, ReasoningPlan, ReasoningProfile,
    ReasoningProtocol,
};
use serde_json::{json, Map, Value};
/// Resolves a reasoning intent for OpenAI-compatible chat completions.
///
/// `provider` is the provider name the client was configured with (canonical or
/// vendor spelling); `api_base` is an optional host hint used only when the
/// provider name carries no vendor information; `model` is the provider-facing
/// model id; `intent` is the caller's request.
///
/// An intent that asks for nothing (`mode: auto`, no `effort`, no
/// `budget_tokens`) resolves to an empty plan, so a body built without an intent
/// is byte-identical to one built before this layer existed.
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::{
///     resolve_reasoning_plan, ReasoningConfig, ReasoningMode,
/// };
/// use serde_json::json;
///
/// let intent = ReasoningConfig::new().with_mode(ReasoningMode::Disabled);
/// let plan = resolve_reasoning_plan("deepseek", None, "deepseek-v4-pro", &intent);
/// assert_eq!(plan.extra_body["thinking"], json!({"type": "disabled"}));
///
/// assert!(resolve_reasoning_plan("openai", None, "gpt-5.1", &ReasoningConfig::new()).is_empty());
/// ```
#[must_use]
pub fn resolve_reasoning_plan(
    provider: &str,
    api_base: Option<&str>,
    model: &str,
    intent: &ReasoningConfig,
) -> ReasoningPlan {
    resolve_reasoning_plan_for(
        ReasoningProtocol::ChatCompletions,
        provider,
        api_base,
        model,
        intent,
    )
}
/// Resolves a reasoning intent for an explicit wire protocol.
///
/// [`resolve_reasoning_plan`] is the OpenAI-compatible chat-completions
/// convenience; call this when the client speaks the Responses API or the
/// Anthropic Messages API so the right encoder is chosen.
#[must_use]
pub fn resolve_reasoning_plan_for(
    protocol: ReasoningProtocol,
    provider: &str,
    api_base: Option<&str>,
    model: &str,
    intent: &ReasoningConfig,
) -> ReasoningPlan {
    if intent.is_no_intent() {
        return ReasoningPlan::new();
    }
    let provider_key = provider_key_for(provider, api_base);
    match protocol {
        ReasoningProtocol::Responses => responses_plan(intent),
        ReasoningProtocol::AnthropicMessages => anthropic_plan(&provider_key, model, intent),
        ReasoningProtocol::ChatCompletions => {
            let provider_key = if is_deepseek(provider, api_base, &provider_key) {
                "deepseek".to_owned()
            } else {
                provider_key
            };
            match find_reasoning_profile(&provider_key, model) {
                Some(profile) => profile_plan(profile, protocol, model, intent),
                None if provider_key == "openrouter" => openrouter_plan(model, intent),
                None => reasoning_effort_plan(intent, "none"),
            }
        }
    }
}
/// Whether this endpoint is `DeepSeek`, whichever name or host it was configured
/// under (Python's `_is_deepseek`).
fn is_deepseek(provider: &str, api_base: Option<&str>, provider_key: &str) -> bool {
    provider_key == "deepseek"
        || provider.trim().to_lowercase().replace('_', "-") == "deepseek"
        || api_base.is_some_and(|base| base.to_lowercase().contains("deepseek.com"))
}
/// Anthropic Messages resolution: a matching profile wins, otherwise the
/// protocol default.
fn anthropic_plan(provider_key: &str, model: &str, intent: &ReasoningConfig) -> ReasoningPlan {
    match find_reasoning_profile(provider_key, model) {
        Some(profile) => profile_plan(profile, ReasoningProtocol::AnthropicMessages, model, intent),
        None => standard_anthropic_plan(intent),
    }
}
/// Encodes one profile with the encoder named by the profile for `protocol`.
///
/// The order of the guards mirrors Python's `_profile_reasoning_plan`: the
/// always-on `MiniMax` M2 split flag first, then the "cannot switch off" case, then
/// the families with no encoder at all.
fn profile_plan(
    profile: &'static ReasoningProfile,
    protocol: ReasoningProtocol,
    model: &str,
    intent: &ReasoningConfig,
) -> ReasoningPlan {
    let wire = profile.wire_for(protocol);
    let capability = profile.capability_spec_for(protocol);
    let model_key = model.trim().to_lowercase();
    let (intent, warnings) = sanitize_intent(capability, model, intent);
    let disabled = intent.is_disabled();

    let mut plan = if wire == "minimax_m2" {
        plan_with_extra("reasoning_split", json!(true))
    } else if (disabled && !capability.supports("off"))
        || wire == "unsupported"
        || wire == "always_on"
    {
        ReasoningPlan::new()
    } else {
        encode_wire(wire, protocol, model, &model_key, &intent)
    };
    plan.warnings = warnings;
    plan
}
/// Dispatches one wire name onto its encoder.
fn encode_wire(
    wire: &str,
    protocol: ReasoningProtocol,
    model: &str,
    model_key: &str,
    intent: &ReasoningConfig,
) -> ReasoningPlan {
    match wire {
        "standard_openai" => reasoning_effort_plan(intent, "none"),
        "standard_anthropic" => standard_anthropic_plan(intent),
        "anthropic_manual" => anthropic_manual_budget_plan(intent),
        "openrouter" => openrouter_plan(model, intent),
        "qwen38" => qwen38_plan(intent),
        "qwen38_anthropic" => output_config_effort_plan(intent, normalize_qwen38_effort),
        "enable_thinking" | "qianfan_toggle" => enable_thinking_plan(intent),
        "dashscope_budget" => dashscope_budget_plan(intent),
        "dashscope_glm_anthropic" => output_config_effort_plan(intent, normalize_glm_effort),
        "thinking_toggle" => thinking_toggle_plan(intent, "enabled"),
        "anthropic_toggle" => anthropic_toggle_plan(intent),
        "deepseek" => deepseek_chat_plan(intent, model_key),
        "deepseek_anthropic" => deepseek_anthropic_plan(intent, model_key),
        "modelarts_deepseek" => thinking_with_effort_plan(intent, true, normalize_modelarts_effort),
        "modelarts_deepseek_anthropic" => {
            anthropic_thinking_with_extra_effort_plan(intent, normalize_modelarts_effort)
        }
        "glm" => thinking_with_effort_plan(intent, true, normalize_glm_effort),
        "glm_anthropic" => anthropic_thinking_with_extra_effort_plan(intent, normalize_glm_effort),
        "kimi_k3" => moonshot_plan(protocol, model, intent),
        "minimax_m3" => minimax_chat_plan(model_key, intent),
        "minimax_m3_anthropic" => minimax_anthropic_plan(model, intent),
        "volcengine_seed" => volcengine_seed_plan(intent),
        _ => ReasoningPlan::new(),
    }
}
/// Drops intent a model family cannot express and explains why.
///
/// The Python resolver silently ignores an unsupported labelled level. This port
/// keeps the resulting wire parameters but reports what was dropped, with at most
/// one warning per plan:
///
/// - A family with **no selectable control** (`unsupported`, `always_on`) loses
///   any explicit `effort`/`budget_tokens`; the plan ends up empty and carries one
///   warning naming the model.
/// - A family that **does** expose controls but not the requested level loses only
///   that level: the warning names the model and the supported values, while an
///   explicit `mode` (and any budget) is still encoded. Dropping the whole intent
///   would otherwise silently discard a caller's `enabled` request.
fn sanitize_intent(
    capability: CapabilitySpec,
    model: &str,
    intent: &ReasoningConfig,
) -> (ReasoningConfig, Vec<String>) {
    let mut sanitized = intent.clone();
    if capability.is_empty() {
        let mut dropped: Vec<&str> = Vec::new();
        if intent.effective_effort().is_some() {
            dropped.push("effort");
        }
        if intent.budget_tokens.is_some() {
            dropped.push("budget_tokens");
        }
        if dropped.is_empty() {
            return (sanitized, Vec::new());
        }
        sanitized.effort = None;
        sanitized.budget_tokens = None;
        let warning = format!(
            "reasoning {} was ignored: model {model} exposes no selectable reasoning control",
            dropped.join(" and ")
        );
        return (sanitized, vec![warning]);
    }
    match intent.effective_effort() {
        Some(effort) if !capability.supports(effort) => {
            sanitized.effort = None;
            let warning = format!(
                "reasoning effort \"{effort}\" was ignored: model {model} supports {}",
                capability.options_list()
            );
            (sanitized, vec![warning])
        }
        _ => (sanitized, Vec::new()),
    }
}
/// `OpenAI` Responses API encoding (Python's `_openai_responses_reasoning_plan`).
///
/// Only an explicit `disabled` mode clears the field (sending JSON `null`); a
/// labelled effort goes into the nested `reasoning` object. Python tests the mode
/// alone here, so an `"off"` effort without `mode: disabled` is passed through
/// verbatim — kept for parity.
fn responses_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.mode == ReasoningMode::Disabled {
        return plan_with_sdk("reasoning", Value::Null);
    }
    match explicit_effort(intent) {
        Some(effort) => plan_with_sdk("reasoning", json!({ "effort": effort })),
        None => ReasoningPlan::new(),
    }
}
/// Generic OpenAI-compatible effort encoding (Python's `_reasoning_effort_plan`).
fn reasoning_effort_plan(intent: &ReasoningConfig, disabled_value: &str) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_sdk("reasoning_effort", json!(disabled_value));
    }
    match explicit_effort(intent) {
        Some(effort) => plan_with_sdk("reasoning_effort", json!(effort)),
        None => ReasoningPlan::new(),
    }
}
/// Anthropic Messages default: adaptive thinking, plus `output_config.effort` when
/// an effort was requested, or an explicit budget when one was given.
fn standard_anthropic_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    let effort = explicit_effort(intent);
    let budget = intent.budget_tokens;
    if intent.mode == ReasoningMode::Auto && effort.is_none() && budget.is_none() {
        return ReasoningPlan::new();
    }
    let mut thinking = anthropic_thinking(budget.is_some());
    if let Some(budget) = budget {
        insert(&mut thinking, "budget_tokens", json!(budget));
    }
    let mut plan = plan_with_sdk("thinking", Value::Object(thinking));
    if let Some(effort) = effort {
        insert(
            &mut plan.sdk_params,
            "output_config",
            json!({ "effort": effort }),
        );
    }
    plan
}
/// Budget-only Anthropic thinking (Claude Sonnet 4.5 and earlier).
///
/// The API requires `budget_tokens >= 1024` whenever thinking is enabled, so a
/// labelled effort maps onto a budget and plain `enabled` uses the `medium` budget
/// rather than the weakest supported one.
fn anthropic_manual_budget_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Auto
        && explicit_effort(intent).is_none()
        && intent.budget_tokens.is_none()
    {
        return ReasoningPlan::new();
    }
    let budget = intent
        .budget_tokens
        .unwrap_or_else(|| anthropic_manual_budget(intent.effort.as_deref()));
    plan_with_sdk(
        "thinking",
        json!({ "type": "enabled", "budget_tokens": budget }),
    )
}
/// Labeled effort → `thinking.budget_tokens` for budget-only Anthropic endpoints.
fn anthropic_manual_budget(effort: Option<&str>) -> u32 {
    let effort = effort
        .map(str::trim)
        .map(str::to_lowercase)
        .unwrap_or_default();
    match effort.as_str() {
        "minimal" => 1024,
        "low" => 2048,
        "high" | "xhigh" | "max" => 16_384,
        _ => 8192,
    }
}
/// Anthropic-compatible toggle for `DashScope` models without an effort field.
///
/// `budget_tokens` stays optional on this endpoint (the vendor default is the
/// model's maximum chain-of-thought length), so plain `enabled` deliberately sends
/// no budget instead of pinning the old 1024 API minimum.
fn dashscope_budget_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Auto
        && explicit_effort(intent).is_none()
        && intent.budget_tokens.is_none()
    {
        return ReasoningPlan::new();
    }
    // Python always sends `enabled` here, with the budget optional.
    let mut thinking = anthropic_thinking(true);
    if let Some(budget) = intent.budget_tokens {
        insert(&mut thinking, "budget_tokens", json!(budget));
    }
    plan_with_sdk("thinking", Value::Object(thinking))
}
/// Anthropic-compatible thinking with vendor-native `output_config.effort`.
///
/// Used by Model Studio (`DashScope`) Anthropic endpoints where `budget_tokens` is
/// deprecated. An explicit numeric budget still wins and is sent alone, because
/// `DashScope` rejects budget and effort together.
fn output_config_effort_plan(
    intent: &ReasoningConfig,
    normalizer: fn(Option<&str>) -> Option<&'static str>,
) -> ReasoningPlan {
    let effort = normalizer(intent.effort.as_deref());
    if intent.is_disabled() || effort == Some("none") {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Auto
        && explicit_effort(intent).is_none()
        && intent.budget_tokens.is_none()
    {
        return ReasoningPlan::new();
    }
    // Python always sends `enabled` here; intensity comes from `output_config`.
    let mut thinking = anthropic_thinking(true);
    if let Some(budget) = intent.budget_tokens {
        insert(&mut thinking, "budget_tokens", json!(budget));
        return plan_with_sdk("thinking", Value::Object(thinking));
    }
    let mut plan = plan_with_sdk("thinking", Value::Object(thinking));
    if let Some(effort) = effort {
        insert(
            &mut plan.sdk_params,
            "output_config",
            json!({ "effort": effort }),
        );
    }
    plan
}
/// `DashScope`-style `enable_thinking` switch plus optional `thinking_budget`
/// (Python's `_qwen_enable_thinking_plan`, also used by `qianfan_toggle`).
fn enable_thinking_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    let mut extra = Map::new();
    if intent.mode == ReasoningMode::Enabled && !intent.has_disabled_effort() {
        insert(&mut extra, "enable_thinking", json!(true));
    } else if intent.is_disabled() {
        insert(&mut extra, "enable_thinking", json!(false));
    }
    if let Some(budget) = intent.budget_tokens {
        insert(&mut extra, "thinking_budget", json!(budget));
    }
    ReasoningPlan {
        extra_body: extra,
        ..ReasoningPlan::new()
    }
}
/// `DashScope` Qwen 3.8 Max chat encoding (`enable_thinking` plus a labeled effort).
fn qwen38_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    let effort = normalize_qwen38_effort(intent.effort.as_deref());
    if intent.is_disabled() || effort == Some("none") {
        return plan_with_extra("enable_thinking", json!(false));
    }
    let mut extra = Map::new();
    if intent.mode == ReasoningMode::Enabled {
        insert(&mut extra, "enable_thinking", json!(true));
    }
    if let Some(budget) = intent.budget_tokens {
        insert(&mut extra, "thinking_budget", json!(budget));
    } else if let Some(effort) = effort {
        insert(&mut extra, "reasoning_effort", json!(effort));
    }
    ReasoningPlan {
        extra_body: extra,
        ..ReasoningPlan::new()
    }
}
/// `{"thinking": {"type": "enabled" | "disabled" | "adaptive"}}` in the vendor
/// extension body (Python's `_thinking_type_toggle_plan`).
fn thinking_toggle_plan(intent: &ReasoningConfig, enabled_type: &str) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_extra("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Enabled {
        return plan_with_extra("thinking", json!({ "type": enabled_type }));
    }
    ReasoningPlan::new()
}
/// `thinking` as a top-level SDK parameter (the Anthropic Messages toggle).
fn anthropic_toggle_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Enabled {
        return plan_with_sdk("thinking", json!({ "type": "enabled" }));
    }
    ReasoningPlan::new()
}
/// Toggle plus an effort, with the effort placed where the vendor reads it.
fn thinking_with_effort_plan(
    intent: &ReasoningConfig,
    effort_in_extra_body: bool,
    normalizer: fn(Option<&str>) -> Option<&'static str>,
) -> ReasoningPlan {
    let effort = normalizer(intent.effort.as_deref());
    if intent.is_disabled() || effort == Some("none") {
        return plan_with_extra("thinking", json!({ "type": "disabled" }));
    }
    let mut plan = thinking_toggle_plan(intent, "enabled");
    if let Some(effort) = effort {
        let target = if effort_in_extra_body {
            &mut plan.extra_body
        } else {
            &mut plan.sdk_params
        };
        insert(target, "reasoning_effort", json!(effort));
    }
    plan
}
/// Anthropic toggle plus an effort that travels in the vendor extension body.
fn anthropic_thinking_with_extra_effort_plan(
    intent: &ReasoningConfig,
    normalizer: fn(Option<&str>) -> Option<&'static str>,
) -> ReasoningPlan {
    let effort = normalizer(intent.effort.as_deref());
    if intent.is_disabled() || effort == Some("none") {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    let mut plan = anthropic_toggle_plan(intent);
    if let Some(effort) = effort {
        insert(&mut plan.extra_body, "reasoning_effort", json!(effort));
    }
    plan
}
/// `DeepSeek` chat encoding: `thinking` in the extension body, normalized effort at
/// the top level.
fn deepseek_chat_plan(intent: &ReasoningConfig, model_key: &str) -> ReasoningPlan {
    if intent.is_disabled() {
        return plan_with_extra("thinking", json!({ "type": "disabled" }));
    }
    let effort = normalize_deepseek_effort(intent.effort.as_deref(), model_key);
    let mut plan = ReasoningPlan::new();
    if intent.mode == ReasoningMode::Enabled {
        insert(
            &mut plan.extra_body,
            "thinking",
            json!({ "type": "enabled" }),
        );
    }
    if let Some(effort) = effort {
        insert(&mut plan.sdk_params, "reasoning_effort", json!(effort));
    }
    plan
}
/// `DeepSeek` over the Anthropic Messages wire: `thinking` and `output_config`
/// travel as top-level parameters.
fn deepseek_anthropic_plan(intent: &ReasoningConfig, model_key: &str) -> ReasoningPlan {
    let effort = normalize_deepseek_effort(intent.effort.as_deref(), model_key);
    if intent.is_disabled() {
        return plan_with_sdk("thinking", json!({ "type": "disabled" }));
    }
    let mut plan = ReasoningPlan::new();
    match effort {
        Some(effort) => {
            insert(
                &mut plan.sdk_params,
                "thinking",
                json!({ "type": "enabled" }),
            );
            insert(
                &mut plan.sdk_params,
                "output_config",
                json!({ "effort": effort }),
            );
        }
        None if intent.mode == ReasoningMode::Enabled => {
            insert(
                &mut plan.sdk_params,
                "thinking",
                json!({ "type": "enabled" }),
            );
        }
        None => {}
    }
    plan
}
/// Kimi K3 and the K2 toggle families (Python's `_moonshot_*_reasoning_plan`).
fn moonshot_plan(
    protocol: ReasoningProtocol,
    model: &str,
    intent: &ReasoningConfig,
) -> ReasoningPlan {
    let anthropic = protocol == ReasoningProtocol::AnthropicMessages;
    let family = moonshot_family(model);
    if family == MoonshotFamily::K3 {
        if intent.is_disabled() {
            return ReasoningPlan::new();
        }
        return match normalize_kimi_k3_effort(intent.effort.as_deref()) {
            Some(effort) if anthropic => plan_with_extra("reasoning_effort", json!(effort)),
            Some(effort) => plan_with_sdk("reasoning_effort", json!(effort)),
            None => ReasoningPlan::new(),
        };
    }
    if family != MoonshotFamily::K2Toggle {
        return ReasoningPlan::new();
    }
    if anthropic {
        if intent.is_disabled() {
            return plan_with_sdk("thinking", json!({ "type": "disabled" }));
        }
        if intent.mode == ReasoningMode::Auto {
            return ReasoningPlan::new();
        }
        return plan_with_sdk("thinking", json!({ "type": "enabled" }));
    }
    if intent.is_disabled() {
        return plan_with_extra("thinking", json!({ "type": "disabled" }));
    }
    if intent.mode == ReasoningMode::Enabled {
        return plan_with_extra("thinking", json!({ "type": "enabled" }));
    }
    ReasoningPlan::new()
}
/// Kimi family recognized from the model id's last `/` segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoonshotFamily {
    /// Kimi K3: labelled efforts only.
    K3,
    /// `kimi-k2.7-code*`: always-on reasoning.
    K27Code,
    /// `kimi-k2.6*`/`kimi-k2.5*`: on/off toggle.
    K2Toggle,
    /// Anything else in the family.
    Unknown,
}
/// Classifies a Kimi model id (Python's `_moonshot_model_family`).
fn moonshot_family(model: &str) -> MoonshotFamily {
    let model_key = model.trim().to_lowercase();
    let model_key = model_key.rsplit('/').next().unwrap_or(&model_key);
    if model_key.starts_with("kimi-k3") {
        MoonshotFamily::K3
    } else if model_key.contains("k2.7-code") {
        MoonshotFamily::K27Code
    } else if model_key.starts_with("kimi-k2.6") || model_key.starts_with("kimi-k2.5") {
        MoonshotFamily::K2Toggle
    } else {
        MoonshotFamily::Unknown
    }
}
/// `MiniMax` M3 chat encoding (Python's `_minimax_chat_reasoning_plan`).
fn minimax_chat_plan(model_key: &str, intent: &ReasoningConfig) -> ReasoningPlan {
    if model_key.starts_with("minimax-m3") {
        return thinking_toggle_plan(intent, "adaptive");
    }
    if model_key.starts_with("minimax-m2") {
        return plan_with_extra("reasoning_split", json!(true));
    }
    ReasoningPlan::new()
}
/// `MiniMax` M3 over the Anthropic Messages wire.
fn minimax_anthropic_plan(model: &str, intent: &ReasoningConfig) -> ReasoningPlan {
    let model_key = model.trim().to_lowercase();
    if model_key.starts_with("minimax-m3") {
        if intent.is_disabled() {
            return plan_with_sdk("thinking", json!({ "type": "disabled" }));
        }
        if intent.mode == ReasoningMode::Enabled {
            return plan_with_sdk("thinking", json!({ "type": "adaptive" }));
        }
    }
    ReasoningPlan::new()
}
/// Volcano Engine Seed 2.0: a top-level `reasoning_effort`, with `minimal` as the
/// disable encoding.
fn volcengine_seed_plan(intent: &ReasoningConfig) -> ReasoningPlan {
    let effort = normalize_volcengine_seed_effort(intent.effort.as_deref());
    if intent.mode == ReasoningMode::Disabled {
        return plan_with_sdk("reasoning_effort", json!("minimal"));
    }
    match effort {
        Some(effort) => plan_with_sdk("reasoning_effort", json!(effort)),
        None => ReasoningPlan::new(),
    }
}
/// `OpenRouter`'s normalized `reasoning` object (Python's
/// `_openrouter_reasoning_plan`).
///
/// `OpenRouter` exposes one object for both API shapes; a model id from a vendor
/// family with no documented reasoning control is left alone.
fn openrouter_plan(model: &str, intent: &ReasoningConfig) -> ReasoningPlan {
    let model_key = model.trim().to_lowercase();
    let prefix = model_key.split_once('/').map_or("", |(prefix, _)| prefix);
    const KNOWN_PREFIXES: [&str; 9] = [
        "anthropic",
        "deepseek",
        "google",
        "minimax",
        "moonshotai",
        "openai",
        "poolside",
        "qwen",
        "z-ai",
    ];
    if !prefix.is_empty() && !KNOWN_PREFIXES.contains(&prefix) {
        return ReasoningPlan::new();
    }
    let mut payload = Map::new();
    if intent.is_disabled() {
        insert(&mut payload, "effort", json!("none"));
    } else if intent.mode == ReasoningMode::Enabled {
        insert(&mut payload, "enabled", json!(true));
    }
    if let Some(budget) = intent.budget_tokens {
        insert(&mut payload, "max_tokens", json!(budget));
    } else if let Some(effort) = explicit_effort(intent) {
        insert(&mut payload, "effort", json!(effort));
    }
    if payload.is_empty() {
        ReasoningPlan::new()
    } else {
        plan_with_extra("reasoning", Value::Object(payload))
    }
}
/// The effort the caller wrote, trimmed and non-empty.
fn explicit_effort(intent: &ReasoningConfig) -> Option<&str> {
    intent
        .effort
        .as_deref()
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
}
/// `DeepSeek` labelled levels; `xhigh` becomes `max` only on `pro` variants.
fn normalize_deepseek_effort(effort: Option<&str>, model_key: &str) -> Option<&'static str> {
    match normalized(effort)?.as_str() {
        "minimal" | "low" => Some("low"),
        "medium" | "high" => Some("high"),
        "xhigh" => Some(if model_key.contains("pro") {
            "max"
        } else {
            "high"
        }),
        "max" => Some("max"),
        _ => None,
    }
}
/// GLM labelled levels: off → `none`, low/medium/high → `high`, xhigh/max → `max`.
fn normalize_glm_effort(effort: Option<&str>) -> Option<&'static str> {
    match normalized(effort)?.as_str() {
        "none" | "off" | "minimal" => Some("none"),
        "low" | "medium" | "high" => Some("high"),
        "xhigh" | "max" => Some("max"),
        _ => None,
    }
}
/// Volcano Engine Seed levels: off → `minimal`, xhigh/max → `high`.
fn normalize_volcengine_seed_effort(effort: Option<&str>) -> Option<&'static str> {
    match normalized(effort)?.as_str() {
        "none" | "off" | "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" => Some("high"),
        _ => None,
    }
}
/// Kimi K3 levels: ultra/xhigh/max → `max`, medium/high → `high`, the rest → `low`.
fn normalize_kimi_k3_effort(effort: Option<&str>) -> Option<&'static str> {
    match normalized(effort)?.as_str() {
        "ultra" | "xhigh" | "max" => Some("max"),
        "medium" | "high" => Some("high"),
        "minimum" | "minimal" | "light" | "low" => Some("low"),
        _ => None,
    }
}
/// Qwen 3.8 Max levels: off → `none`, minimal/low → `low`, high/xhigh/max → `xhigh`.
fn normalize_qwen38_effort(effort: Option<&str>) -> Option<&'static str> {
    match normalized(effort)?.as_str() {
        "none" | "off" => Some("none"),
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" => Some("xhigh"),
        _ => None,
    }
}
/// `ModelArts` `DeepSeek` exposes one intensity only, so any effort maps onto `high`.
fn normalize_modelarts_effort(effort: Option<&str>) -> Option<&'static str> {
    normalized(effort).map(|_| "high")
}
/// Trims and lowercases an effort, treating an empty string as absent.
fn normalized(effort: Option<&str>) -> Option<String> {
    let value = effort?.trim().to_lowercase();
    (!value.is_empty()).then_some(value)
}
/// Builds the Anthropic `thinking` object with the given enablement style.
fn anthropic_thinking(enabled: bool) -> Map<String, Value> {
    let mut thinking = Map::new();
    insert(
        &mut thinking,
        "type",
        json!(if enabled { "enabled" } else { "adaptive" }),
    );
    thinking
}
/// Inserts a parameter into a map.
fn insert(map: &mut Map<String, Value>, key: &str, value: Value) {
    map.insert(key.to_owned(), value);
}
/// A plan carrying one top-level parameter.
fn plan_with_sdk(key: &str, value: Value) -> ReasoningPlan {
    let mut plan = ReasoningPlan::new();
    insert(&mut plan.sdk_params, key, value);
    plan
}
/// A plan carrying one vendor extension parameter.
fn plan_with_extra(key: &str, value: Value) -> ReasoningPlan {
    let mut plan = ReasoningPlan::new();
    insert(&mut plan.extra_body, key, value);
    plan
}
#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
