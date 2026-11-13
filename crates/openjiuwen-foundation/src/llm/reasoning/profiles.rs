//! Static reasoning capability table, provider-key resolution, and model matching.
//!
//! Ports `openjiuwen/core/foundation/llm/reasoning_profiles.py`. Everything here
//! is data: one [`ReasoningProfile`] per provider-scoped model family, ordered
//! from specific to broad, plus a cross-provider fallback table consulted only
//! when the provider-specific entries miss.
use super::super::provider::normalize_provider_name;
use super::{CapabilitySpec, ReasoningCapability, ReasoningProfile, ReasoningProtocol};
/// Reasoning cannot be switched off and has no effort control.
const ALWAYS_ON: CapabilitySpec = CapabilitySpec::new(&[], None);
/// Reasoning cannot be switched off but accepts a thinking budget.
const ALWAYS_ON_WITH_BUDGET: CapabilitySpec = ALWAYS_ON;
/// The family exposes no user-selectable reasoning control at all.
const UNSUPPORTED: CapabilitySpec = CapabilitySpec::new(&[], None);
/// An on/off switch, recommended on.
const TOGGLE: CapabilitySpec = CapabilitySpec::new(&["off", "on"], Some("on"));
/// An on/off switch that also accepts a thinking budget.
const TOGGLE_WITH_BUDGET: CapabilitySpec = TOGGLE;
/// Budget-only Anthropic thinking (Claude Sonnet 4.5 and earlier): no effort
/// field on the wire, so labelled levels become `budget_tokens` (low 2048,
/// medium 8192, high 16384; the API minimum is 1024).
const ANTHROPIC_MANUAL_BUDGET: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "medium", "high"], Some("medium"));
/// `DeepSeek` V4 families.
const DEEPSEEK_V4: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "high", "max"], Some("high"));
/// GLM-5.2 accept `off`/`high`/`max`, recommended max.
const GLM_52: CapabilitySpec = CapabilitySpec::new(&["off", "high", "max"], Some("max"));
/// Qwen 3.8 Max levels.
const QWEN_38: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "medium", "xhigh"], Some("xhigh"));
/// Kimi K3 levels.
const KIMI_K3: CapabilitySpec = CapabilitySpec::new(&["low", "high", "max"], Some("high"));
/// Volcano Engine Seed 2.0 levels.
const VOLCENGINE_SEED_20: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "medium", "high"], Some("medium"));
/// `OpenAI` GPT-5.6 levels.
const OPENAI_GPT_56: CapabilitySpec = CapabilitySpec::new(
    &["off", "low", "medium", "high", "xhigh", "max"],
    Some("medium"),
);
/// `OpenAI` GPT-5.5 levels.
const OPENAI_GPT_55: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "medium", "high", "xhigh"], Some("medium"));
/// `OpenAI` GPT-5.1 – 5.4 levels, recommended off.
const OPENAI_GPT_51_TO_54: CapabilitySpec =
    CapabilitySpec::new(&["off", "low", "medium", "high", "xhigh"], Some("off"));
/// `OpenAI` GPT-5 levels (no off switch, `minimal` instead).
const OPENAI_GPT_5: CapabilitySpec =
    CapabilitySpec::new(&["minimal", "low", "medium", "high"], Some("medium"));
/// Provider-specific profiles, ordered from exact/specific families to broad ones.
///
/// The same model name can expose different controls through a different gateway
/// or subscription endpoint, so the table is provider-scoped; the first entry
/// whose provider matches and whose patterns match the model id wins.
pub const REASONING_PROFILES: &[ReasoningProfile] = &[
    // First-party protocol models with narrower behavior than the generic
    // protocol defaults.
    ReasoningProfile::new(
        "openai",
        &["gpt-5.6*"],
        OPENAI_GPT_56,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "openai",
        &["gpt-5.5*"],
        OPENAI_GPT_55,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "openai",
        &["gpt-5.4*", "gpt-5.3*", "gpt-5.2*", "gpt-5.1*"],
        OPENAI_GPT_51_TO_54,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "openai",
        &["gpt-5*"],
        OPENAI_GPT_5,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "anthropic",
        &["claude-sonnet-4-5*", "claude-sonnet-4.5*"],
        ANTHROPIC_MANUAL_BUDGET,
        "standard_openai",
        "anthropic_manual",
    ),
    // `OpenRouter` exposes one normalized reasoning object for both API shapes.
    ReasoningProfile::new(
        "openrouter",
        &["openai/gpt-5.6-*"],
        OPENAI_GPT_56,
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["qwen/qwen3.8-max*"],
        QWEN_38,
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["z-ai/glm-5.2*"],
        CapabilitySpec::new(&["off", "high", "xhigh"], Some("high")),
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["deepseek/deepseek-v4-pro*"],
        CapabilitySpec::new(&["off", "high", "xhigh"], Some("high")),
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["moonshotai/kimi-k3*"],
        KIMI_K3,
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["minimax/minimax-m3*"],
        TOGGLE,
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["anthropic/claude-sonnet-4.5*"],
        TOGGLE,
        "openrouter",
        "openrouter",
    ),
    ReasoningProfile::new(
        "openrouter",
        &["google/gemini-2.5-flash*", "poolside/laguna-s-2.1*"],
        TOGGLE_WITH_BUDGET,
        "openrouter",
        "openrouter",
    ),
    // Alibaba Model Studio / `DashScope`. The Anthropic endpoint supports
    // output_config.effort for qwen3.8-max (low/medium/xhigh, default xhigh) and
    // deprecates budget_tokens, so both protocols expose the same levels.
    ReasoningProfile::new(
        "dashscope",
        &["qwen3.8-max*"],
        QWEN_38,
        "qwen38",
        "qwen38_anthropic",
    ),
    ReasoningProfile::new(
        "dashscope",
        &["qwen3-vl-*-thinking*"],
        ALWAYS_ON_WITH_BUDGET,
        "always_on",
        "always_on",
    ),
    ReasoningProfile::new(
        "dashscope",
        &["qwen3*", "qwen-plus*", "qwen-max*", "qwen-flash*"],
        TOGGLE_WITH_BUDGET,
        "enable_thinking",
        "dashscope_budget",
    ),
    ReasoningProfile::new(
        "dashscope",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    // Same Model Studio endpoint: output_config.effort supports high/max for
    // glm-5.2 (default max), so the Anthropic protocol keeps the GLM levels.
    ReasoningProfile::new(
        "dashscope",
        &["glm-5.2*"],
        GLM_52,
        "glm",
        "dashscope_glm_anthropic",
    ),
    ReasoningProfile::new(
        "dashscope",
        &["glm-5*", "glm-4.7*", "kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE_WITH_BUDGET,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new("dashscope", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    // `DeepSeek` official.
    ReasoningProfile::new(
        "deepseek",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    // Moonshot/Kimi official and Coding Plan.
    ReasoningProfile::new("moonshot", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    ReasoningProfile::new(
        "moonshot",
        &["kimi-k2.7-code*", "kimi-k2-thinking*"],
        ALWAYS_ON,
        "always_on",
        "always_on",
    ),
    ReasoningProfile::new(
        "moonshot",
        &["kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    // `MiniMax` M3 is hybrid; M2/M1 families are native reasoning models without a
    // documented disable or effort control.
    ReasoningProfile::new(
        "minimax",
        &["minimax-m3*"],
        TOGGLE,
        "minimax_m3",
        "minimax_m3_anthropic",
    ),
    ReasoningProfile::new(
        "minimax",
        &["minimax-m2*", "minimax-m1*"],
        ALWAYS_ON,
        "minimax_m2",
        "always_on",
    ),
    // Zhipu official and Coding Plan.
    ReasoningProfile::new("zhipu", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    ReasoningProfile::new(
        "zhipu",
        &["glm-5*", "glm-4.7*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "zhipu",
        &["codegeex-4*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    // Huawei Cloud `ModelArts` MaaS. The former pangu-ultra/large/small aliases are
    // not MaaS V2 model IDs and are deliberately not profiled.
    ReasoningProfile::new(
        "modelarts",
        &[
            "openpangu-2.0-*",
            "glm-5.1*",
            "kimi-k2.6*",
            "qwen3-32b*",
            "qwen3-30b-a3b*",
        ],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new("modelarts", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    ReasoningProfile::new(
        "modelarts",
        &["deepseek-v4-*"],
        TOGGLE,
        "modelarts_deepseek",
        "modelarts_deepseek_anthropic",
    ),
    // Volcano Engine Ark.
    ReasoningProfile::new(
        "volcengine",
        &[
            "seed-2.0-*",
            "seed-2-0-*",
            "doubao-seed-2-0-*",
            "doubao-seed-2.0-*",
        ],
        VOLCENGINE_SEED_20,
        "volcengine_seed",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "volcengine",
        &["seed-1.6*", "doubao-seed-1-6*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "volcengine",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    ReasoningProfile::new(
        "volcengine",
        &["doubao-1.5-pro*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    // Baidu Qianfan, including models proxied from other vendors.
    ReasoningProfile::new(
        "qianfan",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    ReasoningProfile::new(
        "qianfan",
        &["glm-5.2*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "qianfan",
        &["glm-5*", "kimi-k2.6*", "kimi-k2.5*", "mimo-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    // "emie-5.*" is intentional typo tolerance: some Qianfan catalog listings
    // expose the misspelled ID alongside the canonical "ernie" name.
    ReasoningProfile::new(
        "qianfan",
        &["ernie-5.*", "emie-5.*"],
        TOGGLE,
        "qianfan_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "qianfan",
        &["qianfan-ocr-*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    // Xiaomi `MiMo` official and Token Plan use the same request format.
    // "mino-v2.5*" is intentional typo tolerance for misspelled catalog IDs.
    ReasoningProfile::new(
        "mimo",
        &["mimo-v2.5*", "mino-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
];
/// Cross-provider model-name fallbacks, consulted after every provider-specific
/// entry missed.
///
/// These cover custom OpenAI/Anthropic-compatible endpoints whose host does not
/// reveal the upstream vendor. They describe the model family's usual wire
/// format; unknown models still fall through to the protocol defaults.
pub const MODEL_REASONING_FALLBACKS: &[ReasoningProfile] = &[
    ReasoningProfile::new(
        "*",
        &["gpt-5.6*"],
        OPENAI_GPT_56,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["gpt-5.5*"],
        OPENAI_GPT_55,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["gpt-5.4*", "gpt-5.3*", "gpt-5.2*", "gpt-5.1*"],
        OPENAI_GPT_51_TO_54,
        "standard_openai",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["gpt-5*"],
        OPENAI_GPT_5,
        "standard_openai",
        "standard_anthropic",
    ),
    // Custom Anthropic-compatible endpoints must not fall through to the protocol
    // default (which includes max for adaptive Claude); Sonnet 4.5 is budget-only.
    ReasoningProfile::new(
        "*",
        &["claude-sonnet-4-5*", "claude-sonnet-4.5*"],
        ANTHROPIC_MANUAL_BUDGET,
        "standard_openai",
        "anthropic_manual",
    ),
    ReasoningProfile::new(
        "*",
        &["qwen3.8-max*"],
        QWEN_38,
        "qwen38",
        "qwen38_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["qwen3-vl-*-thinking*"],
        ALWAYS_ON_WITH_BUDGET,
        "always_on",
        "always_on",
    ),
    ReasoningProfile::new(
        "*",
        &["qwen3*", "qwen-plus*", "qwen-max*", "qwen-flash*"],
        TOGGLE_WITH_BUDGET,
        "enable_thinking",
        "dashscope_budget",
    ),
    ReasoningProfile::new(
        "*",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    ReasoningProfile::new("*", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    ReasoningProfile::new(
        "*",
        &["kimi-k2.7-code*", "kimi-k2-thinking*"],
        ALWAYS_ON,
        "always_on",
        "always_on",
    ),
    ReasoningProfile::new(
        "*",
        &["kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "*",
        &["minimax-m3*"],
        TOGGLE,
        "minimax_m3",
        "minimax_m3_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["minimax-m2*", "minimax-m1*"],
        ALWAYS_ON,
        "minimax_m2",
        "always_on",
    ),
    ReasoningProfile::new("*", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    ReasoningProfile::new(
        "*",
        &["glm-5*", "glm-4.7*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "*",
        &["codegeex-4*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    ReasoningProfile::new(
        "*",
        &["openpangu-2.0-*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "*",
        &[
            "seed-2.0-*",
            "seed-2-0-*",
            "doubao-seed-2-0-*",
            "doubao-seed-2.0-*",
        ],
        VOLCENGINE_SEED_20,
        "volcengine_seed",
        "standard_anthropic",
    ),
    ReasoningProfile::new(
        "*",
        &["seed-1.6*", "doubao-seed-1-6*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "*",
        &["doubao-1.5-pro*", "qianfan-ocr-*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    ReasoningProfile::new(
        "*",
        &["ernie-5.*", "emie-5.*"],
        TOGGLE,
        "qianfan_toggle",
        "anthropic_toggle",
    ),
    ReasoningProfile::new(
        "*",
        &["mimo-v2.5*", "mino-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
];
/// Provider keys treated as "no vendor information": an API-base host hint may
/// override them.
const GENERIC_PROVIDER_KEYS: [&str; 3] = ["", "openai", "openaicompatible"];
/// Endpoint profiles treated as "no vendor information" (Python's
/// `_GENERIC_ENDPOINT_PROFILES`, after `_` → `-` normalization).
const GENERIC_ENDPOINT_PROFILES: [&str; 3] = ["", "openai", "openai-compatible"];
/// API-base hosts that identify the upstream vendor, longest/most specific first.
const API_BASE_PROVIDER_HINTS: &[(&str, &str)] = &[
    ("openrouter.ai", "openrouter"),
    ("dashscope-intl.aliyuncs.com", "dashscope"),
    ("dashscope.aliyuncs.com", "dashscope"),
    ("maas.aliyuncs.com", "dashscope"),
    ("api.deepseek.com", "deepseek"),
    ("api.moonshot.cn", "moonshot"),
    ("api.moonshot.ai", "moonshot"),
    ("api.kimi.com", "moonshot"),
    ("api.minimax.io", "minimax"),
    ("api.minimax.chat", "minimax"),
    ("api.minimaxi.com", "minimax"),
    ("api-ap-southeast-1.modelarts-maas.com", "modelarts"),
    ("api.modelarts-maas.com", "modelarts"),
    ("qianfan.baidubce.com", "qianfan"),
    ("open.bigmodel.cn", "zhipu"),
    ("api.z.ai", "zhipu"),
    ("ark.cn-beijing.volces.com", "volcengine"),
    ("ark.volces.com", "volcengine"),
    ("api.xiaomimimo.com", "mimo"),
    ("token-plan-cn.xiaomimimo.com", "mimo"),
    ("token-plan-sgp.xiaomimimo.com", "mimo"),
    ("token-plan-ams.xiaomimimo.com", "mimo"),
];
/// Resolves the provider key the reasoning table is scoped by.
///
/// The provider name is normalized the same way
/// [`normalize_provider_name`](super::super::provider::normalize_provider_name) does
/// (lowercased, separators dropped, vendor aliases resolved), then mapped onto
/// the reasoning table's vocabulary — most importantly `qwen` (this crate's
/// canonical name for `DashScope`) becomes `dashscope`.
///
/// An `api_base` host hint is used **only** for a generic provider name (`""`,
/// `openai`, `openai-compatible`), so an explicit, vendor-specific provider name
/// always wins over the host. This is slightly stricter than the Python resolver,
/// which lets the host override `client_provider` whenever `endpoint_profile` is
/// generic; in this crate the provider name *is* the endpoint profile, so an
/// explicit non-generic name is authoritative.
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::provider_key_for;
///
/// assert_eq!(provider_key_for("DashScope", None), "dashscope");
/// assert_eq!(provider_key_for("kimi", None), "moonshot");
/// // A generic provider defers to the host:
/// assert_eq!(
///     provider_key_for("openai", Some("https://openrouter.ai/api/v1")),
///     "openrouter"
/// );
/// // An explicit provider wins over the host:
/// assert_eq!(
///     provider_key_for("deepseek", Some("https://openrouter.ai/api/v1")),
///     "deepseek"
/// );
/// ```
#[must_use]
pub fn provider_key_for(provider: &str, api_base: Option<&str>) -> String {
    provider_key_for_endpoint(provider, None, api_base)
}
/// Resolves the provider key from the provider name, its `endpoint_profile`, and
/// an optional API-base host hint.
///
/// The endpoint profile is trimmed, lowercased, and has `_` folded onto `-`. A
/// non-generic profile (`"vllm"`, `"dashscope"`, …) names the wire dialect the
/// gateway speaks and therefore wins over both the provider name and the host.
/// A generic profile (`""`, `"openai"`, `"openai-compatible"`) falls back to the
/// host hint, then to the profile as written, then to the normalized provider —
/// the Python `inferred or profile or provider_key` order. An empty or absent
/// profile reduces to [`provider_key_for`] exactly, so existing call sites keep
/// their resolution; a written generic profile additionally lets the host hint
/// override a non-generic provider name, because the caller has just declared the
/// endpoint's dialect generic.
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::provider_key_for_endpoint;
///
/// // A non-generic profile names the dialect:
/// assert_eq!(
///     provider_key_for_endpoint("openai", Some("vllm"), Some("http://localhost:8000/v1")),
///     "vllm"
/// );
/// // A generic profile defers to the host hint:
/// assert_eq!(
///     provider_key_for_endpoint("openai", Some("openai-compatible"), Some("https://openrouter.ai/api/v1")),
///     "openrouter"
/// );
/// // `_` is the same generic profile, which defers to the host hint:
/// assert_eq!(
///     provider_key_for_endpoint("dashscope", Some("openai_compatible"), Some("openrouter.ai")),
///     "openrouter"
/// );
/// // A generic profile without a host hint is returned as written, exactly like
/// // the Python resolver, so blank the profile to keep the provider's dialect:
/// assert_eq!(
///     provider_key_for_endpoint("dashscope", Some("openai"), None),
///     "openai"
/// );
/// assert_eq!(provider_key_for_endpoint("dashscope", None, None), "dashscope");
/// ```
#[must_use]
pub fn provider_key_for_endpoint(
    provider: &str,
    endpoint_profile: Option<&str>,
    api_base: Option<&str>,
) -> String {
    let profile = normalized_endpoint_profile(endpoint_profile);
    if !GENERIC_ENDPOINT_PROFILES.contains(&profile.as_str()) {
        return profile;
    }
    let canonical = reasoning_provider_key(provider);
    // An explicit generic profile says "this is a generic OpenAI-compatible
    // gateway", so the host is the only vendor evidence there is; with no profile
    // at all the provider name stays authoritative for a non-generic name, which
    // is `provider_key_for`'s documented rule.
    let hint_applies = !profile.is_empty() || GENERIC_PROVIDER_KEYS.contains(&canonical.as_str());
    if hint_applies {
        if let Some(hint) = api_base.and_then(provider_from_api_base) {
            return hint.to_owned();
        }
    }
    if profile.is_empty() {
        canonical
    } else {
        profile
    }
}
/// Normalizes an endpoint profile to its comparison form.
fn normalized_endpoint_profile(endpoint_profile: Option<&str>) -> String {
    endpoint_profile
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .replace('_', "-")
}
/// Maps a provider name onto the reasoning table's provider vocabulary.
fn reasoning_provider_key(provider: &str) -> String {
    let canonical = normalize_provider_name(provider);
    match canonical.as_str() {
        "qwen" => "dashscope".to_owned(),
        _ => canonical,
    }
}
/// Infers the upstream vendor from an API base URL.
///
/// Accepts a full URL or a bare host, ignores userinfo and the port, and matches
/// the host itself or any subdomain of a known vendor host.
#[must_use]
fn provider_from_api_base(api_base: &str) -> Option<&'static str> {
    let host = host_of(api_base)?.to_lowercase();
    for (suffix, provider) in API_BASE_PROVIDER_HINTS {
        if host == *suffix || host.ends_with(&format!(".{suffix}")) {
            return Some(provider);
        }
    }
    // Volcano Engine Ark deployments use region-specific `ark.<region>.volces.com`
    // hosts, so the same vendor is recognized by shape rather than by a full list.
    if host.starts_with("ark.") && host.ends_with(".volces.com") {
        return Some("volcengine");
    }
    None
}
/// Extracts the lowercase-able host from an API base URL or bare host.
fn host_of(api_base: &str) -> Option<&str> {
    let raw = api_base.trim();
    if raw.is_empty() {
        return None;
    }
    let rest = match raw.find("://") {
        Some(index) => &raw[index + 3..],
        None => raw,
    };
    let authority = rest.split(['/', '?', '#']).next()?;
    let host = authority.rsplit('@').next()?.split(':').next()?;
    (!host.is_empty()).then_some(host)
}
/// Case-insensitive glob matching with `*` (any run of characters) and `?` (one
/// character).
///
/// No other character is special, so model ids keep their `.`, `-`, and `/`
/// semantics. The matcher is iterative and allocation-light, using the standard
/// star-backtracking algorithm.
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::matches_pattern;
///
/// assert!(matches_pattern("gpt-5*", "GPT-5.1-mini"));
/// assert!(matches_pattern("qwen3-vl-*-thinking*", "qwen3-vl-32b-thinking-2025"));
/// assert!(matches_pattern("glm-?.2*", "glm-5.2"));
/// assert!(!matches_pattern("gpt-5*", "o3"));
/// ```
#[must_use]
pub fn matches_pattern(pattern: &str, model: &str) -> bool {
    let pattern: Vec<char> = pattern.to_lowercase().chars().collect();
    let model: Vec<char> = model.to_lowercase().chars().collect();
    let (mut pattern_index, mut model_index) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut resume = 0usize;
    while model_index < model.len() {
        match pattern.get(pattern_index) {
            Some(character) if *character == '?' || *character == model[model_index] => {
                pattern_index += 1;
                model_index += 1;
            }
            Some('*') => {
                star = Some(pattern_index);
                pattern_index += 1;
                resume = model_index;
            }
            _ => match star {
                Some(star_index) => {
                    pattern_index = star_index + 1;
                    resume += 1;
                    model_index = resume;
                }
                None => return false,
            },
        }
    }
    pattern[pattern_index..]
        .iter()
        .all(|character| *character == '*')
}
/// Finds the profile for `provider`/`model`: first match wins.
///
/// Provider-scoped profiles are tried in table order against the full model id.
/// Then the cross-provider [`MODEL_REASONING_FALLBACKS`] are tried against both
/// the full model id and its last `/` segment, so a gateway-qualified id such as
/// `openrouter/anthropic/claude-sonnet-4-5` still matches `claude-sonnet-4-5*`.
///
/// `provider` is normalized with [`provider_key_for`]'s alias mapping, so callers
/// may pass either the canonical crate name (`qwen`) or the vendor name the table
/// uses (`dashscope`).
///
/// ```
/// use openjiuwen_foundation::llm::reasoning::find_reasoning_profile;
///
/// let profile = find_reasoning_profile("OpenAI", "gpt-5.1").expect("gpt-5.1 is profiled");
/// assert_eq!(profile.wire, "standard_openai");
/// assert!(find_reasoning_profile("openai", "gpt-4o").is_none());
/// ```
#[must_use]
pub fn find_reasoning_profile(provider: &str, model: &str) -> Option<&'static ReasoningProfile> {
    let provider_key = reasoning_provider_key(provider);
    let model_key = model.trim().to_lowercase();
    for profile in REASONING_PROFILES {
        if profile.provider != provider_key {
            continue;
        }
        if profile
            .model_patterns
            .iter()
            .any(|pattern| matches_pattern(pattern, &model_key))
        {
            return Some(profile);
        }
    }
    let last_segment = model_key.rsplit('/').next().unwrap_or(&model_key);
    MODEL_REASONING_FALLBACKS.iter().find(|profile| {
        profile.model_patterns.iter().any(|pattern| {
            matches_pattern(pattern, &model_key) || matches_pattern(pattern, last_segment)
        })
    })
}
/// The capability offered by `provider`/`model` over `protocol`.
///
/// Falls back to the protocol default ([`standard_reasoning_capability`]) when no
/// profile matches, so a UI always has a usable option list.
#[must_use]
pub fn reasoning_capability_for(
    protocol: ReasoningProtocol,
    provider: &str,
    model: &str,
) -> ReasoningCapability {
    capability_for_key(protocol, &reasoning_provider_key(provider), model)
}
/// The capability offered by `provider`/`model` over OpenAI-compatible chat
/// completions, resolving an `endpoint_profile` first.
///
/// An `endpoint_profile` only changes which profile the *resolver* consults; the
/// capability is advisory (it drives a UI's option list) and keeps using the
/// model-name fallback table, so a gateway profile such as `"vllm"` reports the
/// capability of the model family it serves rather than nothing.
#[must_use]
pub fn reasoning_capability_for_endpoint(
    provider: &str,
    endpoint_profile: Option<&str>,
    api_base: Option<&str>,
    model: &str,
) -> ReasoningCapability {
    capability_for_key(
        ReasoningProtocol::ChatCompletions,
        &provider_key_for_endpoint(provider, endpoint_profile, api_base),
        model,
    )
}
/// Looks a capability up by an already-resolved provider key.
fn capability_for_key(
    protocol: ReasoningProtocol,
    provider_key: &str,
    model: &str,
) -> ReasoningCapability {
    match find_reasoning_profile(provider_key, model) {
        Some(profile) => profile.capability_for(protocol),
        None => standard_reasoning_capability(protocol),
    }
}
/// The capability offered by `provider`/`model` over OpenAI-compatible chat
/// completions.
#[must_use]
pub fn reasoning_capability(provider: &str, model: &str) -> ReasoningCapability {
    reasoning_capability_for(ReasoningProtocol::ChatCompletions, provider, model)
}
/// The conservative default capability of one protocol, used for unlisted models.
///
/// OpenAI-compatible: `off`/`low`/`medium`/`high`. Anthropic Messages:
/// `off`/`low`/`medium`/`high`/`max` (adaptive thinking supports `max`). Neither
/// carries a recommendation, because the protocol documents none.
#[must_use]
pub fn standard_reasoning_capability(protocol: ReasoningProtocol) -> ReasoningCapability {
    ReasoningCapability::standard(protocol)
}
#[cfg(test)]
#[path = "profiles_parity_tests.rs"]
mod parity_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glob_matches_prefix_suffix_and_infix_stars() {
        assert!(matches_pattern("gpt-5*", "gpt-5"));
        assert!(matches_pattern("gpt-5*", "gpt-5.1-mini"));
        assert!(matches_pattern("*code*", "kimi-k2.7-code-preview"));
        assert!(matches_pattern("deepseek-v4-*", "deepseek-v4-pro"));
        assert!(matches_pattern("*", "anything"));
        assert!(matches_pattern("*", ""));
        assert!(matches_pattern("seed-2.0-*", "seed-2.0-pro"));
    }
    #[test]
    fn glob_matches_single_character_wildcards() {
        assert!(matches_pattern("glm-?.2*", "glm-5.2"));
        assert!(matches_pattern("glm-5.??", "glm-5.22"));
        assert!(!matches_pattern("glm-?.2", "glm-5.22"));
        assert!(!matches_pattern("glm-?", "glm-"));
    }
    #[test]
    fn glob_is_case_insensitive() {
        assert!(matches_pattern("GPT-5*", "gpt-5.1"));
        assert!(matches_pattern("gpt-5*", "GPT-5.1"));
        assert!(matches_pattern("Qwen3.8-Max*", "qwen3.8-max-preview"));
    }
    #[test]
    fn glob_rejects_non_matches() {
        assert!(!matches_pattern("gpt-5*", "o3"));
        assert!(!matches_pattern("gpt-5", "gpt-5.1"));
        assert!(!matches_pattern("qwen3*", "qw En3"));
        assert!(!matches_pattern("", "gpt-5"));
        assert!(matches_pattern("", ""));
    }
    #[test]
    fn provider_keys_normalize_aliases_onto_table_vocabulary() {
        assert_eq!(provider_key_for("DashScope", None), "dashscope");
        assert_eq!(provider_key_for("qwen", None), "dashscope");
        assert_eq!(provider_key_for("aliyun", None), "dashscope");
        assert_eq!(provider_key_for("kimi", None), "moonshot");
        assert_eq!(provider_key_for("MoonshotAI", None), "moonshot");
        assert_eq!(provider_key_for("glm", None), "zhipu");
        assert_eq!(provider_key_for("Claude", None), "anthropic");
        assert_eq!(provider_key_for(" OpenRouter ", None), "openrouter");
        assert_eq!(provider_key_for("modelarts", None), "modelarts");
        assert_eq!(provider_key_for("unknown-vendor", None), "unknownvendor");
        assert_eq!(provider_key_for("", None), "");
    }
    #[test]
    fn provider_keys_use_the_host_hint_only_for_generic_providers() {
        assert_eq!(
            provider_key_for("openai", Some("https://openrouter.ai/api/v1")),
            "openrouter"
        );
        assert_eq!(
            provider_key_for(
                "",
                Some("https://dashscope.aliyuncs.com/compatible-mode/v1")
            ),
            "dashscope"
        );
        assert_eq!(
            provider_key_for("openai-compatible", Some("api.deepseek.com")),
            "deepseek"
        );
        assert_eq!(
            provider_key_for("deepseek", Some("https://openrouter.ai/api/v1")),
            "deepseek"
        );
        assert_eq!(
            provider_key_for("siliconflow", Some("https://openrouter.ai/api/v1")),
            "siliconflow"
        );
        assert_eq!(
            provider_key_for("openai", Some("https://api.openai.com/v1")),
            "openai"
        );
    }
    #[test]
    fn host_hints_cover_documented_vendors() {
        let cases = [
            ("https://openrouter.ai/api/v1", "openrouter"),
            (
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "dashscope",
            ),
            ("https://maas.aliyuncs.com/v1", "dashscope"),
            ("https://api.deepseek.com/v1", "deepseek"),
            ("https://api.moonshot.ai/v1", "moonshot"),
            ("https://api.kimi.com/coding-plan", "moonshot"),
            ("https://api.minimax.io/v1", "minimax"),
            ("https://api.minimax.chat/v1", "minimax"),
            (
                "https://api-ap-southeast-1.modelarts-maas.com/v1",
                "modelarts",
            ),
            ("https://qianfan.baidubce.com/v2", "qianfan"),
            ("https://open.bigmodel.cn/api/paas/v4", "zhipu"),
            ("https://api.z.ai/api/paas/v4", "zhipu"),
            ("https://ark.cn-beijing.volces.com/api/v3", "volcengine"),
            ("https://ark.ap-southeast-1.volces.com/api/v3", "volcengine"),
            ("https://token-plan-cn.xiaomimimo.com/v1", "mimo"),
            ("https://api.xiaomimimo.com/v1", "mimo"),
        ];
        for (api_base, expected) in cases {
            assert_eq!(
                provider_key_for("openai", Some(api_base)),
                expected,
                "{api_base}"
            );
        }
    }
    #[test]
    fn host_hints_ignore_userinfo_ports_and_garbage() {
        assert_eq!(
            provider_key_for("", Some("https://user:secret@api.deepseek.com:8443/v1")),
            "deepseek"
        );
        assert_eq!(provider_key_for("", Some("   ")), "");
        assert_eq!(provider_key_for("", Some("https://example.com/v1")), "");
    }
    #[test]
    fn specific_profiles_win_over_broader_ones() {
        let profile = find_reasoning_profile("openai", "gpt-5.6-preview").expect("profiled");
        assert_eq!(profile.capability.to_capability().options.len(), 6);
        let profile = find_reasoning_profile("openai", "gpt-5.5").expect("profiled");
        assert_eq!(profile.capability.to_capability().options.len(), 5);
        assert_eq!(profile.capability.recommended, Some("medium"));
        let profile = find_reasoning_profile("openai", "gpt-5.1").expect("profiled");
        assert_eq!(profile.capability.recommended, Some("off"));
        let profile = find_reasoning_profile("openai", "gpt-5").expect("profiled");
        assert_eq!(profile.capability.recommended, Some("medium"));
        assert!(profile.capability.supports("minimal"));
    }
    #[test]
    fn provider_scoped_profiles_are_preferred_over_model_fallbacks() {
        // The same model id has a different wire on `DashScope` and on a custom
        // OpenAI-compatible endpoint.
        let dashscope = find_reasoning_profile("qwen", "deepseek-v4-pro").expect("profiled");
        assert_eq!(dashscope.provider, "dashscope");
        assert_eq!(dashscope.wire, "deepseek");
        let custom = find_reasoning_profile("siliconflow", "deepseek-v4-pro").expect("fallback");
        assert_eq!(custom.provider, "*");
        assert_eq!(custom.wire, "deepseek");
    }
    #[test]
    fn fallbacks_match_the_last_model_segment() {
        let profile =
            find_reasoning_profile("openrouter", "anthropic/claude-sonnet-4-5").expect("fallback");
        assert_eq!(profile.provider, "*");
        assert_eq!(profile.anthropic_wire, "anthropic_manual");
        let profile =
            find_reasoning_profile("some-gateway", "vendor/qwen3-max-preview").expect("fallback");
        assert_eq!(profile.wire, "enable_thinking");
    }
    #[test]
    fn lookup_normalizes_the_provider_argument() {
        let by_canonical = find_reasoning_profile("qwen", "qwen3.8-max").expect("profiled");
        let by_alias = find_reasoning_profile("DashScope", "qwen3.8-max").expect("profiled");
        assert_eq!(by_canonical.provider, "dashscope");
        assert_eq!(by_alias.provider, "dashscope");
        assert_eq!(by_alias.wire, "qwen38");
    }
    #[test]
    fn models_without_a_profile_return_none() {
        assert!(find_reasoning_profile("openai", "gpt-4o").is_none());
        assert!(find_reasoning_profile("ollama", "llama3.1").is_none());
        assert!(find_reasoning_profile("openai", "").is_none());
    }
    #[test]
    fn capability_falls_back_to_the_protocol_standard() {
        let capability = reasoning_capability("openai", "gpt-4o");
        assert_eq!(capability.options, ["off", "low", "medium", "high"]);
        assert_eq!(capability.recommended, None);
        let anthropic =
            reasoning_capability_for(ReasoningProtocol::AnthropicMessages, "openai", "gpt-4o");
        assert_eq!(anthropic.options, ["off", "low", "medium", "high", "max"]);
        assert_eq!(
            standard_reasoning_capability(ReasoningProtocol::ChatCompletions).options,
            ["off", "low", "medium", "high"]
        );
    }
    #[test]
    fn capability_reports_profile_values() {
        let capability = reasoning_capability("deepseek", "deepseek-v4-pro");
        assert_eq!(capability.options, ["off", "low", "high", "max"]);
        assert_eq!(capability.recommended.as_deref(), Some("high"));
        let capability = reasoning_capability("dashscope", "qwen3.8-max");
        assert_eq!(capability.options, ["off", "low", "medium", "xhigh"]);
        assert_eq!(capability.recommended.as_deref(), Some("xhigh"));
    }
    #[test]
    fn capability_is_shared_across_protocols_without_an_anthropic_override() {
        let profile = find_reasoning_profile("openai", "gpt-5.1").expect("profiled");
        assert_eq!(
            profile.capability_spec_for(ReasoningProtocol::ChatCompletions),
            profile.capability_spec_for(ReasoningProtocol::Responses)
        );
        assert_eq!(
            profile.capability_spec_for(ReasoningProtocol::AnthropicMessages),
            profile.capability
        );
    }
    #[test]
    fn endpoint_profile_wins_only_when_it_is_not_generic() {
        assert_eq!(
            provider_key_for_endpoint("openai", Some("vllm"), None),
            "vllm"
        );
        assert_eq!(
            provider_key_for_endpoint("openai", Some(" VLLM "), Some("https://api.openai.com")),
            "vllm"
        );
        assert_eq!(
            provider_key_for_endpoint("anthropic", Some("dashscope"), None),
            "dashscope"
        );
        for generic in ["openai", "openai-compatible", "openai_compatible"] {
            // A generic profile without a hint is returned as written.
            assert_eq!(
                provider_key_for_endpoint("deepseek", Some(generic), None),
                generic.replace('_', "-"),
                "{generic} must be returned as written"
            );
            assert_eq!(
                provider_key_for_endpoint(
                    "openai",
                    Some(generic),
                    Some("https://openrouter.ai/api/v1")
                ),
                "openrouter",
                "{generic} must defer to the host hint"
            );
        }
        assert_eq!(
            provider_key_for_endpoint("openai", Some(""), None),
            "openai"
        );
        assert_eq!(provider_key_for_endpoint("qwen", None, None), "dashscope");
        // No profile at all is exactly `provider_key_for`.
        for (provider, api_base) in [
            ("openai", None),
            ("openai", Some("https://api.deepseek.com/v1")),
            ("kimi", None),
            ("", Some("https://example.com")),
        ] {
            assert_eq!(
                provider_key_for_endpoint(provider, None, api_base),
                provider_key_for(provider, api_base)
            );
        }
    }
    #[test]
    fn endpoint_capability_reports_the_model_family_for_a_gateway_profile() {
        // The gateway profile only forces the wire; the advisory capability still
        // comes from the model-name fallback table.
        assert_eq!(
            reasoning_capability_for_endpoint("openai", Some("vllm"), None, "glm-5.2"),
            reasoning_capability("zhipu", "glm-5.2")
        );
        assert_eq!(
            reasoning_capability_for_endpoint("openai", None, None, "gpt-4o"),
            standard_reasoning_capability(ReasoningProtocol::ChatCompletions)
        );
    }
    #[test]
    fn table_order_places_first_party_families_first() {
        assert_eq!(REASONING_PROFILES[0].provider, "openai");
        assert!(REASONING_PROFILES
            .iter()
            .all(|profile| profile.provider != "*"));
        assert!(MODEL_REASONING_FALLBACKS
            .iter()
            .all(|profile| profile.provider == "*"));
        assert_eq!(REASONING_PROFILES.len(), 42);
        assert_eq!(MODEL_REASONING_FALLBACKS.len(), 23);
    }
}
