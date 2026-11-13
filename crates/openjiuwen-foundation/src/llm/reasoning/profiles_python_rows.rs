//! Hard-coded transcription of the Python reasoning-profile tables.
//!
//! One source of truth for the parity pins: [`EXPECTED_PROFILES`] and
//! [`EXPECTED_FALLBACKS`] mirror `REASONING_PROFILES` and
//! `MODEL_REASONING_FALLBACKS` from
//! `openjiuwen/core/foundation/llm/reasoning_profiles.py` (reference v0.1.20) row
//! for row and in Python order, and [`AUDITED_FAMILIES`] lists the families that
//! were reported missing during review plus their neighbours. Python is never
//! read at test time.
/// The `(options, recommended)` pair of one capability in the Python table.
pub(super) type Cap = (&'static [&'static str], Option<&'static str>);
/// One Python table row: `(provider, model patterns, capability, wire, anthropic wire)`.
pub(super) type Row = (
    &'static str,
    &'static [&'static str],
    Cap,
    &'static str,
    &'static str,
);
/// An on/off switch, recommended on (`TOGGLE`).
pub(super) const TOGGLE: Cap = (&["off", "on"], Some("on"));
/// An on/off switch that also accepts a thinking budget (`TOGGLE_WITH_BUDGET`).
pub(super) const TOGGLE_WITH_BUDGET: Cap = TOGGLE;
/// Reasoning cannot be switched off and has no effort control (`ALWAYS_ON`).
pub(super) const ALWAYS_ON: Cap = (&[], None);
/// Always-on reasoning that accepts a budget (`ALWAYS_ON_WITH_BUDGET`).
pub(super) const ALWAYS_ON_WITH_BUDGET: Cap = ALWAYS_ON;
/// The family exposes no user-selectable control (`UNSUPPORTED`).
pub(super) const UNSUPPORTED: Cap = (&[], None);
/// Budget-only Anthropic thinking (`ANTHROPIC_MANUAL_BUDGET`).
pub(super) const ANTHROPIC_MANUAL_BUDGET: Cap = (&["off", "low", "medium", "high"], Some("medium"));
/// `DEEPSEEK_V4`.
pub(super) const DEEPSEEK_V4: Cap = (&["off", "low", "high", "max"], Some("high"));
/// `GLM_52`.
pub(super) const GLM_52: Cap = (&["off", "high", "max"], Some("max"));
/// `QWEN_38`.
pub(super) const QWEN_38: Cap = (&["off", "low", "medium", "xhigh"], Some("xhigh"));
/// `KIMI_K3`.
pub(super) const KIMI_K3: Cap = (&["low", "high", "max"], Some("high"));
/// `VOLCENGINE_SEED_20`.
pub(super) const VOLCENGINE_SEED_20: Cap = (&["off", "low", "medium", "high"], Some("medium"));
/// `OPENAI_GPT_56`.
pub(super) const OPENAI_GPT_56: Cap = (
    &["off", "low", "medium", "high", "xhigh", "max"],
    Some("medium"),
);
/// `OPENAI_GPT_55`.
pub(super) const OPENAI_GPT_55: Cap = (&["off", "low", "medium", "high", "xhigh"], Some("medium"));
/// `OPENAI_GPT_51_TO_54`.
pub(super) const OPENAI_GPT_51_TO_54: Cap =
    (&["off", "low", "medium", "high", "xhigh"], Some("off"));
/// `OPENAI_GPT_5`.
pub(super) const OPENAI_GPT_5: Cap = (&["minimal", "low", "medium", "high"], Some("medium"));
/// The inline capability of the two `OpenRouter` rows.
pub(super) const OPENROUTER_OFF_HIGH_XHIGH: Cap = (&["off", "high", "xhigh"], Some("high"));
/// `REASONING_PROFILES` exactly as the Python reference declares it.
pub(super) const EXPECTED_PROFILES: [Row; 42] = [
    (
        "openai",
        &["gpt-5.6*"],
        OPENAI_GPT_56,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "openai",
        &["gpt-5.5*"],
        OPENAI_GPT_55,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "openai",
        &["gpt-5.4*", "gpt-5.3*", "gpt-5.2*", "gpt-5.1*"],
        OPENAI_GPT_51_TO_54,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "openai",
        &["gpt-5*"],
        OPENAI_GPT_5,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "anthropic",
        &["claude-sonnet-4-5*", "claude-sonnet-4.5*"],
        ANTHROPIC_MANUAL_BUDGET,
        "standard_openai",
        "anthropic_manual",
    ),
    (
        "openrouter",
        &["openai/gpt-5.6-*"],
        OPENAI_GPT_56,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["qwen/qwen3.8-max*"],
        QWEN_38,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["z-ai/glm-5.2*"],
        OPENROUTER_OFF_HIGH_XHIGH,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["deepseek/deepseek-v4-pro*"],
        OPENROUTER_OFF_HIGH_XHIGH,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["moonshotai/kimi-k3*"],
        KIMI_K3,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["minimax/minimax-m3*"],
        TOGGLE,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["anthropic/claude-sonnet-4.5*"],
        TOGGLE,
        "openrouter",
        "openrouter",
    ),
    (
        "openrouter",
        &["google/gemini-2.5-flash*", "poolside/laguna-s-2.1*"],
        TOGGLE_WITH_BUDGET,
        "openrouter",
        "openrouter",
    ),
    (
        "dashscope",
        &["qwen3.8-max*"],
        QWEN_38,
        "qwen38",
        "qwen38_anthropic",
    ),
    (
        "dashscope",
        &["qwen3-vl-*-thinking*"],
        ALWAYS_ON_WITH_BUDGET,
        "always_on",
        "always_on",
    ),
    (
        "dashscope",
        &["qwen3*", "qwen-plus*", "qwen-max*", "qwen-flash*"],
        TOGGLE_WITH_BUDGET,
        "enable_thinking",
        "dashscope_budget",
    ),
    (
        "dashscope",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    (
        "dashscope",
        &["glm-5.2*"],
        GLM_52,
        "glm",
        "dashscope_glm_anthropic",
    ),
    (
        "dashscope",
        &["glm-5*", "glm-4.7*", "kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE_WITH_BUDGET,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    ("dashscope", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    (
        "deepseek",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    ("moonshot", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    (
        "moonshot",
        &["kimi-k2.7-code*", "kimi-k2-thinking*"],
        ALWAYS_ON,
        "always_on",
        "always_on",
    ),
    (
        "moonshot",
        &["kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "minimax",
        &["minimax-m3*"],
        TOGGLE,
        "minimax_m3",
        "minimax_m3_anthropic",
    ),
    (
        "minimax",
        &["minimax-m2*", "minimax-m1*"],
        ALWAYS_ON,
        "minimax_m2",
        "always_on",
    ),
    ("zhipu", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    (
        "zhipu",
        &["glm-5*", "glm-4.7*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "zhipu",
        &["codegeex-4*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    (
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
    ("modelarts", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    (
        "modelarts",
        &["deepseek-v4-*"],
        TOGGLE,
        "modelarts_deepseek",
        "modelarts_deepseek_anthropic",
    ),
    (
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
    (
        "volcengine",
        &["seed-1.6*", "doubao-seed-1-6*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "volcengine",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    (
        "volcengine",
        &["doubao-1.5-pro*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    (
        "qianfan",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    (
        "qianfan",
        &["glm-5.2*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "qianfan",
        &["glm-5*", "kimi-k2.6*", "kimi-k2.5*", "mimo-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "qianfan",
        &["ernie-5.*", "emie-5.*"],
        TOGGLE,
        "qianfan_toggle",
        "anthropic_toggle",
    ),
    (
        "qianfan",
        &["qianfan-ocr-*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    (
        "mimo",
        &["mimo-v2.5*", "mino-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
];
/// `MODEL_REASONING_FALLBACKS` exactly as the Python reference declares it.
pub(super) const EXPECTED_FALLBACKS: [Row; 23] = [
    (
        "*",
        &["gpt-5.6*"],
        OPENAI_GPT_56,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "*",
        &["gpt-5.5*"],
        OPENAI_GPT_55,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "*",
        &["gpt-5.4*", "gpt-5.3*", "gpt-5.2*", "gpt-5.1*"],
        OPENAI_GPT_51_TO_54,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "*",
        &["gpt-5*"],
        OPENAI_GPT_5,
        "standard_openai",
        "standard_anthropic",
    ),
    (
        "*",
        &["claude-sonnet-4-5*", "claude-sonnet-4.5*"],
        ANTHROPIC_MANUAL_BUDGET,
        "standard_openai",
        "anthropic_manual",
    ),
    (
        "*",
        &["qwen3.8-max*"],
        QWEN_38,
        "qwen38",
        "qwen38_anthropic",
    ),
    (
        "*",
        &["qwen3-vl-*-thinking*"],
        ALWAYS_ON_WITH_BUDGET,
        "always_on",
        "always_on",
    ),
    (
        "*",
        &["qwen3*", "qwen-plus*", "qwen-max*", "qwen-flash*"],
        TOGGLE_WITH_BUDGET,
        "enable_thinking",
        "dashscope_budget",
    ),
    (
        "*",
        &["deepseek-v4-*"],
        DEEPSEEK_V4,
        "deepseek",
        "deepseek_anthropic",
    ),
    ("*", &["kimi-k3*"], KIMI_K3, "kimi_k3", "kimi_k3"),
    (
        "*",
        &["kimi-k2.7-code*", "kimi-k2-thinking*"],
        ALWAYS_ON,
        "always_on",
        "always_on",
    ),
    (
        "*",
        &["kimi-k2.6*", "kimi-k2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "*",
        &["minimax-m3*"],
        TOGGLE,
        "minimax_m3",
        "minimax_m3_anthropic",
    ),
    (
        "*",
        &["minimax-m2*", "minimax-m1*"],
        ALWAYS_ON,
        "minimax_m2",
        "always_on",
    ),
    ("*", &["glm-5.2*"], GLM_52, "glm", "glm_anthropic"),
    (
        "*",
        &["glm-5*", "glm-4.7*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "*",
        &["codegeex-4*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    (
        "*",
        &["openpangu-2.0-*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
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
    (
        "*",
        &["seed-1.6*", "doubao-seed-1-6*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
    (
        "*",
        &["doubao-1.5-pro*", "qianfan-ocr-*"],
        UNSUPPORTED,
        "unsupported",
        "unsupported",
    ),
    (
        "*",
        &["ernie-5.*", "emie-5.*"],
        TOGGLE,
        "qianfan_toggle",
        "anthropic_toggle",
    ),
    (
        "*",
        &["mimo-v2.5*", "mino-v2.5*"],
        TOGGLE,
        "thinking_toggle",
        "anthropic_toggle",
    ),
];
/// The provider keys of `REASONING_PROFILES`, in Python order.
pub(super) const EXPECTED_PROVIDERS: [&str; 12] = [
    "openai",
    "anthropic",
    "openrouter",
    "dashscope",
    "deepseek",
    "moonshot",
    "minimax",
    "zhipu",
    "modelarts",
    "volcengine",
    "qianfan",
    "mimo",
];
/// One audited family: `(provider, model id, provider key, wire, anthropic wire, capability)`.
pub(super) type Family = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    Cap,
);
/// Families reported missing during review, plus their neighbours.
pub(super) const AUDITED_FAMILIES: &[Family] = &[
    // `OpenRouter` rows.
    (
        "openrouter",
        "minimax/minimax-m3",
        "openrouter",
        "openrouter",
        "openrouter",
        TOGGLE,
    ),
    (
        "openrouter",
        "anthropic/claude-sonnet-4.5",
        "openrouter",
        "openrouter",
        "openrouter",
        TOGGLE,
    ),
    (
        "openrouter",
        "google/gemini-2.5-flash-preview",
        "openrouter",
        "openrouter",
        "openrouter",
        TOGGLE_WITH_BUDGET,
    ),
    (
        "openrouter",
        "poolside/laguna-s-2.1",
        "openrouter",
        "openrouter",
        "openrouter",
        TOGGLE_WITH_BUDGET,
    ),
    // `DashScope` rows.
    (
        "dashscope",
        "qwen3-vl-32b-thinking-2025",
        "dashscope",
        "always_on",
        "always_on",
        ALWAYS_ON_WITH_BUDGET,
    ),
    (
        "dashscope",
        "glm-5",
        "dashscope",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE_WITH_BUDGET,
    ),
    (
        "dashscope",
        "glm-4.7-flash",
        "dashscope",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE_WITH_BUDGET,
    ),
    (
        "dashscope",
        "kimi-k2.6",
        "dashscope",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE_WITH_BUDGET,
    ),
    (
        "dashscope",
        "kimi-k2.5",
        "dashscope",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE_WITH_BUDGET,
    ),
    // Cross-provider fallbacks.
    (
        "custom-gateway",
        "qwen3.8-max",
        "*",
        "qwen38",
        "qwen38_anthropic",
        QWEN_38,
    ),
    (
        "custom-gateway",
        "qwen3-vl-8b-thinking",
        "*",
        "always_on",
        "always_on",
        ALWAYS_ON_WITH_BUDGET,
    ),
    (
        "custom-gateway",
        "qwen3-30b-a3b",
        "*",
        "enable_thinking",
        "dashscope_budget",
        TOGGLE_WITH_BUDGET,
    ),
    (
        "custom-gateway",
        "deepseek-v4-flash",
        "*",
        "deepseek",
        "deepseek_anthropic",
        DEEPSEEK_V4,
    ),
    (
        "custom-gateway",
        "kimi-k3",
        "*",
        "kimi_k3",
        "kimi_k3",
        KIMI_K3,
    ),
    (
        "custom-gateway",
        "kimi-k2.7-code-preview",
        "*",
        "always_on",
        "always_on",
        ALWAYS_ON,
    ),
    (
        "custom-gateway",
        "kimi-k2-thinking",
        "*",
        "always_on",
        "always_on",
        ALWAYS_ON,
    ),
    (
        "custom-gateway",
        "kimi-k2.6",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "glm-5.2",
        "*",
        "glm",
        "glm_anthropic",
        GLM_52,
    ),
    (
        "custom-gateway",
        "glm-5",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "minimax-m3",
        "*",
        "minimax_m3",
        "minimax_m3_anthropic",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "minimax-m2.5",
        "*",
        "minimax_m2",
        "always_on",
        ALWAYS_ON,
    ),
    (
        "custom-gateway",
        "claude-sonnet-4.5",
        "*",
        "standard_openai",
        "anthropic_manual",
        ANTHROPIC_MANUAL_BUDGET,
    ),
    (
        "custom-gateway",
        "mimo-v2.5",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "mino-v2.5",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "ernie-5.0",
        "*",
        "qianfan_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "seed-1.6-flash",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "openpangu-2.0-flash",
        "*",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "custom-gateway",
        "doubao-1.5-pro-32k",
        "*",
        "unsupported",
        "unsupported",
        UNSUPPORTED,
    ),
    (
        "custom-gateway",
        "qianfan-ocr-vl",
        "*",
        "unsupported",
        "unsupported",
        UNSUPPORTED,
    ),
    (
        "custom-gateway",
        "codegeex-4",
        "*",
        "unsupported",
        "unsupported",
        UNSUPPORTED,
    ),
    // Provider-specific neighbours of the audited rows.
    (
        "zhipu",
        "glm-4.7",
        "zhipu",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "moonshot",
        "kimi-k2.5",
        "moonshot",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "qianfan",
        "mimo-v2.5",
        "qianfan",
        "thinking_toggle",
        "anthropic_toggle",
        TOGGLE,
    ),
    (
        "minimax",
        "minimax-m1",
        "minimax",
        "minimax_m2",
        "always_on",
        ALWAYS_ON,
    ),
];
