//! Run-mode detection shared by every example.
//!
//! The examples are offline by default: they answer from a scripted model or a
//! loopback mock server and need neither a network nor a credential. Live mode
//! is opt-in and defensive — it is selected only when the user asks for it *and*
//! the configuration it needs is present, so a half-configured environment
//! degrades to offline with an explanation instead of failing.

use openjiuwen::core::common::config::env;

/// Environment variable that opts an example into live mode: `OPENJIUWEN_LIVE`.
///
/// Truthy values are `1`, `true`, `yes`, and `on` (ASCII case-insensitive, with
/// surrounding whitespace ignored); every other value — including `0`, `false`,
/// and the empty string — leaves the example offline.
pub const LIVE_FLAG: &str = "OPENJIUWEN_LIVE";

/// Model-name variables, in precedence order.
const MODEL_VARIABLES: &[&str] = &["MODEL_NAME", "OPENJIUWEN_MODEL_NAME", "OPENAI_MODEL"];

/// Credential variables, in precedence order.
const CREDENTIAL_VARIABLES: &[&str] = &["OPENJIUWEN_API_KEY", "API_KEY", "OPENAI_API_KEY"];

/// Provider variables, in precedence order.
const PROVIDER_VARIABLES: &[&str] = &["MODEL_PROVIDER", "OPENJIUWEN_MODEL_PROVIDER"];

/// Provider assumed when no provider variable is set.
const DEFAULT_PROVIDER: &str = "openai";

/// How an example talks to a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunMode {
    /// Scripted or loopback fakes only; no network, no credentials.
    Offline,
    /// A real provider, reached with the configured credential.
    Live {
        /// Provider name, for example `openai` or `anthropic`.
        provider: String,
        /// Model name, for example `gpt-4o-mini`.
        model: String,
    },
}

impl RunMode {
    /// Returns `true` for [`RunMode::Live`].
    #[must_use]
    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live { .. })
    }

    /// A one-line description for the banner: `offline (scripted model)` or
    /// `live (<provider>/<model>)`.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Offline => "offline (scripted model)".to_owned(),
            Self::Live { provider, model } => format!("live ({provider}/{model})"),
        }
    }

    /// The provider name, or `None` in offline mode.
    #[must_use]
    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::Offline => None,
            Self::Live { provider, .. } => Some(provider),
        }
    }

    /// The model name, or `None` in offline mode.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Offline => None,
            Self::Live { model, .. } => Some(model),
        }
    }
}

/// The detected mode plus the configuration that kept it offline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    /// Mode the example should run in.
    pub mode: RunMode,
    /// Variables that are unset although the live flag asked for them, in the
    /// order model name then credential; empty when nothing is missing.
    pub missing: Vec<&'static str>,
}

/// Detects the run mode from the process environment.
///
/// Equivalent to [`detect_with`] with a lookup that reads `std::env`.
#[must_use]
pub fn detect() -> Evaluation {
    detect_with(|name| std::env::var(name).ok())
}

/// Detects the run mode from an injected lookup, so tests never touch the
/// process environment.
///
/// Live mode requires the [`LIVE_FLAG`] to be truthy **and** a model name
/// (first non-blank of `MODEL_NAME`, `OPENJIUWEN_MODEL_NAME`, `OPENAI_MODEL`)
/// **and** a credential (first non-blank of `OPENJIUWEN_API_KEY`, `API_KEY`,
/// `OPENAI_API_KEY`). The provider is the first non-blank of `MODEL_PROVIDER`,
/// `OPENJIUWEN_MODEL_PROVIDER`, defaulting to `openai`.
///
/// A truthy flag with a prerequisite missing yields [`RunMode::Offline`] with
/// the absent variables recorded in [`Evaluation::missing`]; an absent or falsy
/// flag yields [`RunMode::Offline`] with an empty `missing`. The function never
/// panics and never fails: a typo in the environment must not stop an example
/// from running offline.
#[must_use]
pub fn detect_with(lookup: impl Fn(&str) -> Option<String>) -> Evaluation {
    if !is_live_requested(&lookup) {
        return Evaluation {
            mode: RunMode::Offline,
            missing: Vec::new(),
        };
    }

    let model = first_non_blank(&lookup, MODEL_VARIABLES);
    let credential = first_non_blank(&lookup, CREDENTIAL_VARIABLES);

    let mut missing = Vec::new();
    if model.is_none() {
        missing.push(MODEL_VARIABLES[0]);
    }
    if credential.is_none() {
        missing.push(CREDENTIAL_VARIABLES[0]);
    }

    match (model, credential) {
        (Some(model), Some(_credential)) => Evaluation {
            mode: RunMode::Live {
                provider: first_non_blank(&lookup, PROVIDER_VARIABLES)
                    .unwrap_or_else(|| DEFAULT_PROVIDER.to_owned()),
                model,
            },
            missing,
        },
        _ => Evaluation {
            mode: RunMode::Offline,
            missing,
        },
    }
}

/// Whether [`LIVE_FLAG`] is set to a truthy value.
fn is_live_requested(lookup: &impl Fn(&str) -> Option<String>) -> bool {
    lookup(LIVE_FLAG).is_some_and(|value| env::truthy(&value))
}

/// The first value among `names` that is not blank after trimming.
fn first_non_blank(
    lookup: &impl Fn(&str) -> Option<String>,
    names: &[&'static str],
) -> Option<String> {
    names.iter().find_map(|name| {
        lookup(name)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a lookup over `pairs`, blank values included verbatim.
    fn env_of<'a>(
        pairs: &'a [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            pairs
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn nothing_set_is_offline() {
        let evaluation = detect_with(env_of(&[]));
        assert_eq!(evaluation.mode, RunMode::Offline);
        assert_eq!(evaluation.missing, Vec::<&str>::new());
        assert!(!evaluation.mode.is_live());
    }

    #[test]
    fn flag_only_degrades_to_offline() {
        let evaluation = detect_with(env_of(&[(LIVE_FLAG, "1")]));
        assert_eq!(evaluation.mode, RunMode::Offline);
        assert_eq!(evaluation.missing, vec!["MODEL_NAME", "OPENJIUWEN_API_KEY"]);
    }

    #[test]
    fn flag_and_model_degrades_with_credential_missing() {
        let evaluation = detect_with(env_of(&[(LIVE_FLAG, "1"), ("MODEL_NAME", "gpt-4o-mini")]));
        assert_eq!(evaluation.mode, RunMode::Offline);
        assert_eq!(evaluation.missing, vec!["OPENJIUWEN_API_KEY"]);
    }

    #[test]
    fn flag_model_and_credential_is_live_with_default_provider() {
        let evaluation = detect_with(env_of(&[
            (LIVE_FLAG, "1"),
            ("MODEL_NAME", "gpt-4o-mini"),
            ("OPENAI_API_KEY", "sk-test"),
        ]));
        assert_eq!(
            evaluation.mode,
            RunMode::Live {
                provider: "openai".to_owned(),
                model: "gpt-4o-mini".to_owned(),
            }
        );
        assert_eq!(evaluation.missing, Vec::<&str>::new());
        assert!(evaluation.mode.is_live());
        assert_eq!(evaluation.mode.provider(), Some("openai"));
        assert_eq!(evaluation.mode.model(), Some("gpt-4o-mini"));
    }

    #[test]
    fn provider_is_read_from_the_environment() {
        let evaluation = detect_with(env_of(&[
            (LIVE_FLAG, "true"),
            ("MODEL_PROVIDER", "anthropic"),
            ("MODEL_NAME", "claude-sonnet-4"),
            ("API_KEY", "sk-test"),
        ]));
        assert_eq!(
            evaluation.mode.describe(),
            "live (anthropic/claude-sonnet-4)"
        );
    }

    #[test]
    fn openjiuwen_variables_win_over_generic_ones() {
        let evaluation = detect_with(env_of(&[
            (LIVE_FLAG, "on"),
            ("MODEL_PROVIDER", " "),
            ("OPENJIUWEN_MODEL_PROVIDER", "dashscope"),
            ("OPENAI_MODEL", "generic-model"),
            ("OPENJIUWEN_MODEL_NAME", "qwen-max"),
            ("API_KEY", "generic-key"),
            ("OPENJIUWEN_API_KEY", "primary-key"),
        ]));
        assert_eq!(evaluation.mode.describe(), "live (dashscope/qwen-max)");
    }

    #[test]
    fn falsy_flags_stay_offline_without_missing_entries() {
        for value in ["0", "false", "FALSE", "", "  ", "no", "off", "enabled"] {
            let evaluation = detect_with(env_of(&[(LIVE_FLAG, value)]));
            assert_eq!(
                evaluation.mode,
                RunMode::Offline,
                "flag value `{value}` must stay offline"
            );
            assert!(
                evaluation.missing.is_empty(),
                "flag value `{value}` must not report missing variables"
            );
        }
    }

    #[test]
    fn truthy_flag_ignores_case_and_surrounding_whitespace() {
        for value in [" 1 ", "TRUE", "Yes", " oN "] {
            let evaluation = detect_with(env_of(&[
                (LIVE_FLAG, value),
                ("MODEL_NAME", "gpt-4o-mini"),
                ("OPENAI_API_KEY", "sk-test"),
            ]));
            assert!(
                evaluation.mode.is_live(),
                "flag value `{value}` must enable live mode"
            );
        }
    }

    #[test]
    fn blank_values_count_as_absent() {
        let evaluation = detect_with(env_of(&[
            (LIVE_FLAG, "1"),
            ("MODEL_NAME", "   "),
            ("OPENJIUWEN_MODEL_NAME", ""),
            ("API_KEY", "\t"),
        ]));
        assert_eq!(evaluation.mode, RunMode::Offline);
        assert_eq!(evaluation.missing, vec!["MODEL_NAME", "OPENJIUWEN_API_KEY"]);
    }

    #[test]
    fn values_are_trimmed_before_use() {
        let evaluation = detect_with(env_of(&[
            (LIVE_FLAG, "1"),
            ("MODEL_NAME", "  gpt-4o-mini  "),
            ("MODEL_PROVIDER", " openai "),
            ("OPENAI_API_KEY", " sk-test "),
        ]));
        assert_eq!(evaluation.mode.describe(), "live (openai/gpt-4o-mini)");
    }
}
