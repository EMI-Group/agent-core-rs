//! Agent execution mode and system-prompt verbosity mode.
//!
//! Ports `openjiuwen/harness/schema/agent_mode.py` (`AgentMode`) and the mode
//! vocabulary of `openjiuwen/harness/prompts` (`PromptMode`, resolved by
//! Python's `prompts.resolve_mode`).
//!
//! Both types are leaf vocabulary with no dependencies on other harness
//! modules: [`AgentMode`] is re-exported by [`crate::schema`], [`PromptMode`]
//! by [`crate::prompts`].
//!
//! Wire form is `snake_case` and round-trips through serde; [`AgentMode::parse`]
//! and [`PromptMode::parse`] only accept the exact wire names.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Execution mode of the `DeepAgent`.
///
/// `Normal` runs the task directly; `Plan` restricts the agent to planning
/// (typically exposing plan-specific tools) until it is switched back.
/// Python parity: `AgentMode(str, Enum)`; default [`Self::Normal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Direct task execution; wire name `normal`.
    #[default]
    Normal,
    /// Plan-first execution; wire name `plan`.
    Plan,
}

impl AgentMode {
    /// Returns the serde wire name of the mode (`"normal"` or `"plan"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Plan => "plan",
        }
    }

    /// Parses an exact wire name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "normal" => Some(Self::Normal),
            "plan" => Some(Self::Plan),
            _ => Option::None,
        }
    }
}

/// Verbosity of the generated system prompt.
///
/// Python parity: the prompt mode resolved by `prompts.resolve_mode`.
/// The variant names and wire forms match Python exactly, including
/// [`Self::None`]. Default is [`Self::Full`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PromptMode {
    /// Emit the full system prompt; wire name `full`.
    #[default]
    Full,
    /// Emit a minimal system prompt; wire name `minimal`.
    Minimal,
    /// Emit no system prompt at all; wire name `none`.
    None,
}

impl PromptMode {
    /// Returns the serde wire name of the mode (`"full"`, `"minimal"` or `"none"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Minimal => "minimal",
            Self::None => "none",
        }
    }

    /// Parses an exact wire name; any other string yields `None`.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "full" => Some(Self::Full),
            "minimal" => Some(Self::Minimal),
            "none" => Some(Self::None),
            _ => Option::None,
        }
    }

    /// Resolves an optional configuration value to a mode.
    ///
    /// Mirrors Python `prompts.resolve_mode`: an absent or unrecognised value
    /// falls back to [`Self::Full`], never to "no prompt".
    #[must_use]
    pub fn resolve(raw: Option<&str>) -> Self {
        raw.and_then(Self::parse).unwrap_or(Self::Full)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{AgentMode, PromptMode};

    #[test]
    fn defaults_are_normal_and_full() {
        assert_eq!(AgentMode::default(), AgentMode::Normal);
        assert_eq!(PromptMode::default(), PromptMode::Full);
    }

    #[test]
    fn agent_mode_wire_names_round_trip() {
        for mode in [AgentMode::Normal, AgentMode::Plan] {
            let json = serde_json::to_string(&mode).expect("serialize agent mode");
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            let back: AgentMode = serde_json::from_str(&json).expect("deserialize agent mode");
            assert_eq!(back, mode);
        }
        assert_eq!(
            serde_json::to_string(&AgentMode::Normal).expect("serialize"),
            "\"normal\""
        );
        assert_eq!(
            serde_json::to_string(&AgentMode::Plan).expect("serialize"),
            "\"plan\""
        );
    }

    #[test]
    fn prompt_mode_wire_names_round_trip() {
        for mode in [PromptMode::Full, PromptMode::Minimal, PromptMode::None] {
            let json = serde_json::to_string(&mode).expect("serialize prompt mode");
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            let back: PromptMode = serde_json::from_str(&json).expect("deserialize prompt mode");
            assert_eq!(back, mode);
        }
        assert_eq!(
            serde_json::to_string(&PromptMode::Full).expect("serialize"),
            "\"full\""
        );
        assert_eq!(
            serde_json::to_string(&PromptMode::Minimal).expect("serialize"),
            "\"minimal\""
        );
        assert_eq!(
            serde_json::to_string(&PromptMode::None).expect("serialize"),
            "\"none\""
        );
    }

    #[test]
    fn agent_mode_parse_is_exact_match() {
        assert_eq!(AgentMode::parse("normal"), Some(AgentMode::Normal));
        assert_eq!(AgentMode::parse("plan"), Some(AgentMode::Plan));
        assert_eq!(AgentMode::parse("Normal"), None);
        assert_eq!(AgentMode::parse("PLAN"), None);
        assert_eq!(AgentMode::parse(" plan"), None);
        assert_eq!(AgentMode::parse("plan "), None);
        assert_eq!(AgentMode::parse("bogus"), None);
        assert_eq!(AgentMode::parse(""), None);
    }

    #[test]
    fn prompt_mode_parse_is_exact_match() {
        assert_eq!(PromptMode::parse("full"), Some(PromptMode::Full));
        assert_eq!(PromptMode::parse("minimal"), Some(PromptMode::Minimal));
        assert_eq!(PromptMode::parse("none"), Some(PromptMode::None));
        assert_eq!(PromptMode::parse("Full"), None);
        assert_eq!(PromptMode::parse("NONE"), None);
        assert_eq!(PromptMode::parse("bogus"), None);
        assert_eq!(PromptMode::parse(""), None);
    }

    #[test]
    fn prompt_mode_resolve_falls_back_to_full() {
        assert_eq!(PromptMode::resolve(Some("minimal")), PromptMode::Minimal);
        assert_eq!(PromptMode::resolve(Some("none")), PromptMode::None);
        assert_eq!(PromptMode::resolve(Some("full")), PromptMode::Full);
        assert_eq!(PromptMode::resolve(Option::None), PromptMode::Full);
        assert_eq!(PromptMode::resolve(Some("bogus")), PromptMode::Full);
        assert_eq!(PromptMode::resolve(Some("")), PromptMode::Full);
    }

    #[test]
    fn as_str_round_trips_with_parse() {
        for mode in [AgentMode::Normal, AgentMode::Plan] {
            assert_eq!(AgentMode::parse(mode.as_str()), Some(mode));
        }
        for mode in [PromptMode::Full, PromptMode::Minimal, PromptMode::None] {
            assert_eq!(PromptMode::parse(mode.as_str()), Some(mode));
        }
    }
}
