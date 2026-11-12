//! The prompt language selector.
//!
//! Python's prompt builder keys its template map by a
//! `Literal["zh-CN", "en-US"]` value and resolves it in `select_template`, which
//! silently falls back to `zh-CN` for an unknown tag. Rust models the value as
//! [`Language`], a copyable enum whose JSON representation is the same tag, so
//! configuration files stay interchangeable between the two SDKs.
//!
//! Parsing is stricter than Python on purpose: [`Language::from_str`] accepts the
//! canonical tags (case-insensitively) and a few community spellings, and rejects
//! anything else with an [`openjiuwen_common::error::Error::Validation`] naming
//! the offending value. Defaulting until the input matches would hide a typo in
//! a configuration file and silently emit Chinese prompts, so it is a documented
//! deviation (see [`crate::prompt_builder`]).

use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The language of the bundled prompt templates.
///
/// One variant per template bank in [`crate::prompt_builder::templates`].
/// [`Language::ZhCn`] is the default, matching Python's `select_template`
/// fallback and the primary language of the Python SDK's prompt texts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// Simplified Chinese, serialized as `"zh-CN"`.
    #[default]
    #[serde(rename = "zh-CN")]
    ZhCn,
    /// American English, serialized as `"en-US"`.
    #[serde(rename = "en-US")]
    EnUs,
}

impl Language {
    /// The canonical BCP-47 tag: `"zh-CN"` or `"en-US"`.
    ///
    /// The same text [`Language`] serializes to, so it can be handed straight to
    /// a provider or written into a configuration file.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ZhCn => "zh-CN",
            Self::EnUs => "en-US",
        }
    }
}

impl fmt::Display for Language {
    /// Writes the canonical tag, e.g. `zh-CN`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Language {
    type Err = Error;

    /// Parses a language tag.
    ///
    /// The input is trimmed and compared case-insensitively against the
    /// canonical tags and their aliases:
    ///
    /// - `zh-CN`, `zh_cn`, `zhcn`, `zh` → [`Language::ZhCn`]
    /// - `en-US`, `en_us`, `enus`, `en` → [`Language::EnUs`]
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for the empty string or any other tag; the
    /// message names the rejected value. Unknown input is never coerced to a
    /// default.
    fn from_str(value: &str) -> Result<Self> {
        let normalized = value.trim().to_lowercase();
        match normalized.as_str() {
            "zh-cn" | "zh_cn" | "zhcn" | "zh" => Ok(Self::ZhCn),
            "en-us" | "en_us" | "enus" | "en" => Ok(Self::EnUs),
            _ => Err(Error::validation(format!(
                "unsupported prompt language {value:?}: expected \"zh-CN\" or \"en-US\""
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn default_is_zh_cn() {
        assert_eq!(Language::default(), Language::ZhCn);
    }

    #[test]
    fn display_and_as_str_use_the_canonical_tags() {
        assert_eq!(Language::ZhCn.as_str(), "zh-CN");
        assert_eq!(Language::EnUs.as_str(), "en-US");
        assert_eq!(Language::ZhCn.to_string(), "zh-CN");
        assert_eq!(Language::EnUs.to_string(), "en-US");
    }

    #[test]
    fn serde_uses_the_python_literal_tags() {
        assert_eq!(serde_json::to_string(&Language::ZhCn).unwrap(), "\"zh-CN\"");
        assert_eq!(serde_json::to_string(&Language::EnUs).unwrap(), "\"en-US\"");

        let zh: Language = serde_json::from_str("\"zh-CN\"").unwrap();
        let en: Language = serde_json::from_str("\"en-US\"").unwrap();
        assert_eq!(zh, Language::ZhCn);
        assert_eq!(en, Language::EnUs);

        // Round-trip through the same document shape a config file uses.
        for language in [Language::ZhCn, Language::EnUs] {
            let json = serde_json::to_string(&language).unwrap();
            let parsed: Language = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, language);
            // Serde and `Display` agree on the tag.
            assert_eq!(json, format!("\"{language}\""));
        }
    }

    #[test]
    fn from_str_accepts_canonical_tags_case_insensitively() {
        for value in ["zh-CN", "ZH-CN", "zh-cn", "Zh-Cn", "  zh-CN  "] {
            assert_eq!(
                value.parse::<Language>().unwrap(),
                Language::ZhCn,
                "{value}"
            );
        }
        for value in ["en-US", "EN-US", "en-us", "En-Us", "\ten-US\n"] {
            assert_eq!(
                value.parse::<Language>().unwrap(),
                Language::EnUs,
                "{value}"
            );
        }
    }

    #[test]
    fn from_str_accepts_the_aliases() {
        for value in ["zh", "zh_cn", "zhcn", "ZH", "Zhcn"] {
            assert_eq!(
                value.parse::<Language>().unwrap(),
                Language::ZhCn,
                "{value}"
            );
        }
        for value in ["en", "en_us", "enus", "EN", "Enus"] {
            assert_eq!(
                value.parse::<Language>().unwrap(),
                Language::EnUs,
                "{value}"
            );
        }
    }

    #[test]
    fn from_str_rejects_unknown_and_empty_tags() {
        for value in ["", "   ", "fr-FR", "zh CN", "zh-CN-x", "english"] {
            let error = value.parse::<Language>().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{value:?}");
            assert!(
                error.to_string().contains(&format!("{value:?}")),
                "the error must name the rejected value: {error}"
            );
        }
    }
}
