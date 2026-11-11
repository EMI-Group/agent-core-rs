//! Provider table: canonical names, vendor aliases, and endpoint profiles.
//!
//! Ports the provider metadata of `openjiuwen/core/foundation/llm/schema/config.py`
//! (`ProviderType`) and `.../utils/endpoint_profiles.py`, but keeps it as data:
//! every built-in provider is one [`ProviderProfile`] entry, so adding a vendor
//! is a table edit rather than a new code path.
//!
//! Only vendors with a documented, stable endpoint are listed. A vendor such as
//! `pangu` that lives on air-gapped deployments still has a profile, but its
//! [`ProviderProfile::default_api_base`] is `None`, so callers must supply an
//! `api_base` themselves.
//!
//! ```
//! use openjiuwen_foundation::llm::provider::{
//!     normalize_provider_name, provider_profile, ProviderFlavor,
//! };
//!
//! assert_eq!(normalize_provider_name("DashScope"), "qwen");
//! assert_eq!(normalize_provider_name("claude"), "anthropic");
//! let profile = provider_profile("kimi").expect("moonshot is built in");
//! assert_eq!(profile.flavor, ProviderFlavor::OpenAiCompatible);
//! assert_eq!(profile.default_api_base, Some("https://api.moonshot.cn/v1"));
//! ```
use serde::{Deserialize, Serialize};

/// The `anthropic-version` header value required by the Anthropic Messages API.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Path of the OpenAI-compatible chat-completions endpoint.
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";

/// Path of the Anthropic Messages endpoint.
const MESSAGES_PATH: &str = "/v1/messages";

/// Wire protocol spoken by a built-in provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderFlavor {
    /// OpenAI-compatible `POST {base}/chat/completions`.
    OpenAiCompatible,
    /// Anthropic Messages API, `POST {base}/v1/messages`.
    Anthropic,
}

/// Static description of one built-in provider.
///
/// The transport details are data, so clients read them from here instead of
/// branching on the provider name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderProfile {
    /// Canonical provider name, as returned by [`normalize_provider_name`].
    pub name: &'static str,
    /// Wire protocol implemented by the endpoint.
    pub flavor: ProviderFlavor,
    /// Default API base, or `None` when the deployment must supply one.
    pub default_api_base: Option<&'static str>,
    /// Path appended to the base URL for chat requests.
    pub chat_path: &'static str,
    /// Extra static headers the endpoint expects, beyond auth and protocol ones.
    pub extra_headers: &'static [(&'static str, &'static str)],
}

/// Every built-in provider, in [`provider_names`] order.
///
/// All OpenAI-compatible profiles share [`CHAT_COMPLETIONS_PATH`] and define no
/// extra headers; the Anthropic profile uses [`MESSAGES_PATH`].
const PROFILES: &[ProviderProfile] = &[
    ProviderProfile {
        name: "openai",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://api.openai.com/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "qwen",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "deepseek",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://api.deepseek.com/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "moonshot",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://api.moonshot.cn/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "openrouter",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://openrouter.ai/api/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "siliconflow",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://api.siliconflow.cn/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "zhipu",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://open.bigmodel.cn/api/paas/v4"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "minimax",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("https://api.minimax.chat/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "ollama",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("http://localhost:11434/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "lmstudio",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: Some("http://localhost:1234/v1"),
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "anthropic",
        flavor: ProviderFlavor::Anthropic,
        default_api_base: Some("https://api.anthropic.com"),
        chat_path: MESSAGES_PATH,
        extra_headers: &[],
    },
    ProviderProfile {
        name: "pangu",
        flavor: ProviderFlavor::OpenAiCompatible,
        default_api_base: None,
        chat_path: CHAT_COMPLETIONS_PATH,
        extra_headers: &[],
    },
];

/// Canonical provider names, in the same order as the profile table.
const PROVIDER_NAMES: &[&str] = &[
    "openai",
    "qwen",
    "deepseek",
    "moonshot",
    "openrouter",
    "siliconflow",
    "zhipu",
    "minimax",
    "ollama",
    "lmstudio",
    "anthropic",
    "pangu",
];

/// Vendor aliases mapped to a canonical provider name.
///
/// Keys are already normalized (lowercased, no `-`, `_`, or whitespace), so
/// spellings such as `silicon_flow` or `lm-studio` reach the canonical name
/// through normalization alone and need no entry here.
const ALIASES: &[(&str, &str)] = &[
    ("dashscope", "qwen"),
    ("aliyun", "qwen"),
    ("alibaba", "qwen"),
    ("claude", "anthropic"),
    ("kimi", "moonshot"),
    ("moonshotai", "moonshot"),
    ("glm", "zhipu"),
    ("bigmodel", "zhipu"),
    ("zhipuai", "zhipu"),
    ("silicon", "siliconflow"),
    ("siliconcloud", "siliconflow"),
];

/// Canonical names of every built-in provider, in profile-table order.
#[must_use]
pub fn provider_names() -> &'static [&'static str] {
    PROVIDER_NAMES
}

/// Normalizes a provider name to its canonical spelling.
///
/// Lowercases the input, removes `-`, `_`, and whitespace, then maps vendor
/// aliases to canonical names: `dashscope`/`aliyun`/`alibaba` → `qwen`,
/// `claude` → `anthropic`, `kimi`/`moonshotai` → `moonshot`,
/// `glm`/`bigmodel`/`zhipuai` → `zhipu`, and `silicon`/`siliconcloud` →
/// `siliconflow`.
///
/// Unknown names are returned as the lowercased, separator-free input instead of
/// failing; use [`provider_profile`] to test whether a name is built in.
#[must_use]
pub fn normalize_provider_name(name: &str) -> String {
    let mut stripped = String::with_capacity(name.len());
    for character in name.chars() {
        if character == '-' || character == '_' || character.is_whitespace() {
            continue;
        }
        stripped.extend(character.to_lowercase());
    }
    let alias = ALIASES
        .iter()
        .find_map(|(alias, canonical)| (*alias == stripped).then_some(*canonical));
    alias.map_or(stripped, str::to_string)
}

/// Looks up the built-in profile for `name`, applying [`normalize_provider_name`].
///
/// Returns `Some` for every canonical name and alias in the table, and `None` for
/// vendors that are not built in (their name still normalizes, so callers can
/// report it or register a custom [`super::registry::ModelProvider`]).
#[must_use]
pub fn provider_profile(name: &str) -> Option<ProviderProfile> {
    let canonical = normalize_provider_name(name);
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.name == canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_match_the_profile_table() {
        let names: Vec<&str> = PROFILES.iter().map(|profile| profile.name).collect();
        assert_eq!(provider_names(), names.as_slice());
        assert_eq!(names.len(), 12);
    }

    #[test]
    fn canonical_names_are_their_own_normalization() {
        for name in provider_names() {
            assert_eq!(normalize_provider_name(name), *name);
            let profile = provider_profile(name).expect("canonical name resolves");
            assert_eq!(profile.name, *name);
        }
    }

    #[test]
    fn documented_base_urls_and_flavors() {
        let expected = [
            (
                "openai",
                "https://api.openai.com/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "qwen",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "deepseek",
                "https://api.deepseek.com/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "moonshot",
                "https://api.moonshot.cn/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "openrouter",
                "https://openrouter.ai/api/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "siliconflow",
                "https://api.siliconflow.cn/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "zhipu",
                "https://open.bigmodel.cn/api/paas/v4",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "minimax",
                "https://api.minimax.chat/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "ollama",
                "http://localhost:11434/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "lmstudio",
                "http://localhost:1234/v1",
                ProviderFlavor::OpenAiCompatible,
            ),
            (
                "anthropic",
                "https://api.anthropic.com",
                ProviderFlavor::Anthropic,
            ),
        ];
        for (name, base, flavor) in expected {
            let profile = provider_profile(name).expect("documented provider resolves");
            assert_eq!(profile.default_api_base, Some(base), "base of {name}");
            assert_eq!(profile.flavor, flavor, "flavor of {name}");
        }
    }

    #[test]
    fn anthropic_uses_the_messages_path_and_version_constant() {
        let profile = provider_profile("anthropic").expect("anthropic resolves");
        assert_eq!(profile.chat_path, "/v1/messages");
        assert_eq!(profile.flavor, ProviderFlavor::Anthropic);
        assert_eq!(ANTHROPIC_VERSION, "2023-06-01");
    }

    #[test]
    fn openai_compatible_profiles_use_chat_completions() {
        for profile in PROFILES {
            if profile.flavor == ProviderFlavor::OpenAiCompatible {
                assert_eq!(
                    profile.chat_path, "/chat/completions",
                    "path of {}",
                    profile.name
                );
            }
        }
    }

    #[test]
    fn profiles_carry_no_extra_headers() {
        for profile in PROFILES {
            assert!(
                profile.extra_headers.is_empty(),
                "unexpected extra headers for {}",
                profile.name
            );
        }
    }

    #[test]
    fn pangu_is_openai_compatible_but_has_no_default_base() {
        let profile = provider_profile("pangu").expect("pangu resolves");
        assert_eq!(profile.flavor, ProviderFlavor::OpenAiCompatible);
        assert_eq!(profile.default_api_base, None);
        assert_eq!(profile.chat_path, "/chat/completions");
    }

    #[test]
    fn aliases_resolve_to_canonical_names() {
        let cases = [
            ("dashscope", "qwen"),
            ("dash_scope", "qwen"),
            ("DashScope", "qwen"),
            ("aliyun", "qwen"),
            ("alibaba", "qwen"),
            ("claude", "anthropic"),
            ("Claude", "anthropic"),
            ("kimi", "moonshot"),
            ("moonshotai", "moonshot"),
            ("glm", "zhipu"),
            ("bigmodel", "zhipu"),
            ("zhipuai", "zhipu"),
            ("silicon", "siliconflow"),
            ("siliconcloud", "siliconflow"),
            ("silicon_flow", "siliconflow"),
            ("silicon-flow", "siliconflow"),
            ("Silicon Flow", "siliconflow"),
            ("lm_studio", "lmstudio"),
            ("lm-studio", "lmstudio"),
            ("Open AI", "openai"),
            ("open-ai", "openai"),
            ("OPENAI", "openai"),
            ("deep-seek", "deepseek"),
            ("open_router", "openrouter"),
            ("mini-max", "minimax"),
        ];
        for (input, canonical) in cases {
            assert_eq!(
                normalize_provider_name(input),
                canonical,
                "normalizing {input}"
            );
        }
    }

    #[test]
    fn alias_lookups_return_the_canonical_profile() {
        assert_eq!(
            provider_profile("dashscope"),
            provider_profile("qwen"),
            "dashscope is qwen"
        );
        assert_eq!(
            provider_profile("claude"),
            provider_profile("anthropic"),
            "claude is anthropic"
        );
        assert_eq!(
            provider_profile("kimi").map(|profile| profile.name),
            Some("moonshot")
        );
    }

    #[test]
    fn every_alias_target_is_a_builtin_profile() {
        for (alias, canonical) in ALIASES {
            assert_eq!(normalize_provider_name(canonical), *canonical);
            assert!(
                provider_profile(canonical).is_some(),
                "alias {alias} points at unknown provider {canonical}"
            );
            assert_eq!(normalize_provider_name(alias), *canonical);
        }
    }

    #[test]
    fn unknown_provider_is_none() {
        assert!(provider_profile("acme").is_none());
        assert!(provider_profile("my-custom-provider").is_none());
        assert!(provider_profile("").is_none());
    }

    #[test]
    fn unknown_names_keep_the_stripped_lowercase_form() {
        assert_eq!(
            normalize_provider_name("My-Custom_Provider "),
            "mycustomprovider"
        );
        assert_eq!(normalize_provider_name("  ACME  AI "), "acmeai");
        assert_eq!(normalize_provider_name(""), "");
    }

    #[test]
    fn profile_lookup_normalizes_case_and_separators() {
        assert_eq!(provider_profile("  Open-AI "), provider_profile("openai"));
        assert_eq!(
            provider_profile("LM_Studio").map(|profile| profile.name),
            Some("lmstudio")
        );
    }

    #[test]
    fn flavor_serializes_as_snake_case() {
        let encoded = serde_json::to_string(&ProviderFlavor::Anthropic).expect("serializes");
        assert_eq!(encoded, "\"anthropic\"");
        let decoded: ProviderFlavor = serde_json::from_str("\"open_ai_compatible\"")
            .expect("deserializes openai-compatible flavor");
        assert_eq!(decoded, ProviderFlavor::OpenAiCompatible);
    }
}
