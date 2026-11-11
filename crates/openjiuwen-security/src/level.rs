//! Risk levels and content types shared by every guardrail.
//!
//! Ports `openjiuwen/core/security/guardrail/enums.py`: [`RiskLevel`] keeps the
//! Python string values (`"safe"` … `"critical"`) and their severity order, and
//! [`GuardrailContentType`] names the payload shape a guardrail inspects.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Severity of a guardrail detection.
///
/// The variants are ordered by severity (`Safe < Low < Medium < High <
/// Critical`), so [`RiskLevel::at_least`] and plain comparisons answer "is this
/// at least as bad as the blocking threshold?".
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    /// Nothing suspicious was found.
    #[default]
    Safe,
    /// A minor style or policy concern.
    Low,
    /// A policy violation that is usually tolerated.
    Medium,
    /// A serious violation worth surfacing as a finding.
    High,
    /// A violation that aborts the guarded call.
    Critical,
}

impl RiskLevel {
    /// The lowercase Python value of the level (`"safe"` … `"critical"`).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// The uppercase name of the level (`"SAFE"` … `"CRITICAL"`).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Safe => "SAFE",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
            Self::Critical => "CRITICAL",
        }
    }

    /// Whether the level denotes an actual risk, i.e. anything but [`Self::Safe`].
    #[must_use]
    pub fn is_risky(&self) -> bool {
        *self != Self::Safe
    }

    /// Whether `self` is at least as severe as `other`.
    #[must_use]
    pub fn at_least(&self, other: Self) -> bool {
        *self >= other
    }

    /// Parses a level from its Python value, ignoring surrounding whitespace and
    /// ASCII case; unknown values yield `None`.
    #[must_use]
    pub fn from_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "safe" => Some(Self::Safe),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "critical" => Some(Self::Critical),
            _ => None,
        }
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Shape of the payload carried by a [`crate::GuardrailContext`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailContentType {
    /// A single plain-text string.
    #[default]
    Text,
    /// A serialized chat-message array.
    Messages,
    /// A tool-call name plus its JSON arguments.
    ToolCall,
    /// An opaque JSON value.
    Raw,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn as_str_matches_python_values() {
        assert_eq!(RiskLevel::Safe.as_str(), "safe");
        assert_eq!(RiskLevel::Low.as_str(), "low");
        assert_eq!(RiskLevel::Medium.as_str(), "medium");
        assert_eq!(RiskLevel::High.as_str(), "high");
        assert_eq!(RiskLevel::Critical.as_str(), "critical");
    }

    #[test]
    fn name_is_uppercase() {
        assert_eq!(RiskLevel::Safe.name(), "SAFE");
        assert_eq!(RiskLevel::Low.name(), "LOW");
        assert_eq!(RiskLevel::Medium.name(), "MEDIUM");
        assert_eq!(RiskLevel::High.name(), "HIGH");
        assert_eq!(RiskLevel::Critical.name(), "CRITICAL");
        assert_eq!(RiskLevel::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn severity_order_is_safe_to_critical() {
        let ordered = [
            RiskLevel::Safe,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ];
        for window in ordered.windows(2) {
            assert!(window[0] < window[1], "{} < {}", window[0], window[1]);
        }
        assert!(RiskLevel::High.at_least(RiskLevel::Medium));
        assert!(RiskLevel::High.at_least(RiskLevel::High));
        assert!(!RiskLevel::Medium.at_least(RiskLevel::High));
        assert_eq!(RiskLevel::default(), RiskLevel::Safe);
    }

    #[test]
    fn is_risky_only_for_non_safe() {
        assert!(!RiskLevel::Safe.is_risky());
        assert!(RiskLevel::Low.is_risky());
        assert!(RiskLevel::Critical.is_risky());
    }

    #[test]
    fn serializes_as_python_string() {
        assert_eq!(
            serde_json::to_value(RiskLevel::High).unwrap(),
            json!("high")
        );
        assert_eq!(
            serde_json::to_value(RiskLevel::Critical).unwrap(),
            json!("critical")
        );
        let level: RiskLevel = serde_json::from_value(json!("medium")).unwrap();
        assert_eq!(level, RiskLevel::Medium);
    }

    #[test]
    fn from_value_round_trips_and_rejects_unknown() {
        for level in [
            RiskLevel::Safe,
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ] {
            assert_eq!(RiskLevel::from_value(level.as_str()), Some(level));
            assert_eq!(
                RiskLevel::from_value(&format!("  {} ", level.as_str())),
                Some(level)
            );
        }
        assert_eq!(RiskLevel::from_value("CRITICAL"), Some(RiskLevel::Critical));
        assert_eq!(RiskLevel::from_value("severe"), None);
        assert_eq!(RiskLevel::from_value(""), None);
    }

    #[test]
    fn content_type_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_value(GuardrailContentType::Text).unwrap(),
            json!("text")
        );
        assert_eq!(
            serde_json::to_value(GuardrailContentType::ToolCall).unwrap(),
            json!("tool_call")
        );
        let content_type: GuardrailContentType = serde_json::from_value(json!("messages")).unwrap();
        assert_eq!(content_type, GuardrailContentType::Messages);
        assert_eq!(GuardrailContentType::default(), GuardrailContentType::Text);
    }
}
