//! [`MemoryType`]: the memory kinds of the Python `openjiuwen.core.memory` model.
//!
//! Serialized values are exactly the Python enum values (`"user_profile"`,
//! `"semantic_memory"`, ...). Deserialization is tolerant: an unrecognized string
//! becomes [`MemoryType::Unknown`] instead of failing, matching Python's
//! `MemoryType(value)` fallback path when reading old records.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::str::FromStr;
/// The kind of a memory record.
///
/// [`MemoryType::Variable`] and [`MemoryType::Unknown`] are never returned by recall;
/// [`MemoryType::Variable`] is never accepted as a write input either (variables are
/// stored through the facade's variable API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Durable facts about the user (preferences, identity, constraints).
    UserProfile,
    /// General facts and knowledge extracted from conversations.
    SemanticMemory,
    /// Time-stamped events and experiences.
    EpisodicMemory,
    /// A named variable owned by the variable store (never a searchable record).
    Variable,
    /// A condensed summary of older conversation.
    Summary,
    /// A coding-memory fact (`lite` module).
    Coding,
    /// An unrecognized or absent kind; also the serde default.
    #[default]
    Unknown,
}
impl MemoryType {
    /// Every kind, in declaration order.
    pub const ALL: [MemoryType; 7] = [
        MemoryType::UserProfile,
        MemoryType::SemanticMemory,
        MemoryType::EpisodicMemory,
        MemoryType::Variable,
        MemoryType::Summary,
        MemoryType::Coding,
        MemoryType::Unknown,
    ];
    /// The kinds recall may return.
    pub const SEARCHABLE: [MemoryType; 5] = [
        MemoryType::UserProfile,
        MemoryType::SemanticMemory,
        MemoryType::EpisodicMemory,
        MemoryType::Summary,
        MemoryType::Coding,
    ];
    /// The Python enum value of this kind, used in JSON and storage keys.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserProfile => "user_profile",
            Self::SemanticMemory => "semantic_memory",
            Self::EpisodicMemory => "episodic_memory",
            Self::Variable => "variable",
            Self::Summary => "summary",
            Self::Coding => "coding",
            Self::Unknown => "unknown",
        }
    }
    /// Returns `true` when recall may return this kind.
    #[must_use]
    pub const fn is_searchable(self) -> bool {
        matches!(
            self,
            Self::UserProfile
                | Self::SemanticMemory
                | Self::EpisodicMemory
                | Self::Summary
                | Self::Coding
        )
    }
    /// Parses a kind name tolerantly.
    ///
    /// The comparison is case-insensitive and treats `-` and spaces as `_`, so
    /// `"SemanticMemory"`, `"semantic-memory"`, and `"semantic_memory"` are equal.
    /// Common short Python spellings are accepted too; anything unrecognized maps to
    /// [`MemoryType::Unknown`], so parsing never fails.
    fn parse(raw: &str) -> Self {
        let normalized = raw.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        match normalized.as_str() {
            "user_profile" | "userprofile" | "profile" => Self::UserProfile,
            "semantic_memory" | "semantic" | "semanticmemory" => Self::SemanticMemory,
            "episodic_memory" | "episodic" | "episodicmemory" => Self::EpisodicMemory,
            "variable" | "var" => Self::Variable,
            "summary" | "summaries" => Self::Summary,
            "coding" | "code" => Self::Coding,
            _ => Self::Unknown,
        }
    }
}
impl FromStr for MemoryType {
    /// Unused: [`MemoryType::from_str`] is total and maps unknown names to
    /// [`MemoryType::Unknown`]. The associated type exists for interface symmetry.
    type Err = Error;
    /// Parses a kind name; unknown names yield [`MemoryType::Unknown`], never an error.
    fn from_str(raw: &str) -> Result<Self> {
        Ok(Self::parse(raw))
    }
}
impl fmt::Display for MemoryType {
    /// Writes the Python enum value (see [`MemoryType::as_str`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for MemoryType {
    /// Deserializes from a kind string; unknown strings become [`MemoryType::Unknown`].
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn serde_uses_python_values() {
        for kind in MemoryType::ALL {
            let json = serde_json::to_value(kind).unwrap();
            assert_eq!(json, serde_json::Value::String(kind.as_str().to_owned()));
            let restored: MemoryType = serde_json::from_value(json).unwrap();
            assert_eq!(restored, kind);
        }
        assert_eq!(
            serde_json::to_value(MemoryType::SemanticMemory).unwrap(),
            serde_json::json!("semantic_memory")
        );
    }
    #[test]
    fn deserialize_maps_unknown_strings_to_unknown() {
        let kind: MemoryType = serde_json::from_value(serde_json::json!("not_a_kind")).unwrap();
        assert_eq!(kind, MemoryType::Unknown);
        let kind: MemoryType = serde_json::from_value(serde_json::json!("")).unwrap();
        assert_eq!(kind, MemoryType::Unknown);
        assert!(serde_json::from_value::<MemoryType>(serde_json::json!(42)).is_err());
    }
    #[test]
    fn from_str_is_tolerant() {
        assert_eq!(
            "user_profile".parse::<MemoryType>().unwrap(),
            MemoryType::UserProfile
        );
        assert_eq!(
            "SemanticMemory".parse::<MemoryType>().unwrap(),
            MemoryType::SemanticMemory
        );
        assert_eq!(
            "semantic-memory".parse::<MemoryType>().unwrap(),
            MemoryType::SemanticMemory
        );
        assert_eq!(
            " Summary ".parse::<MemoryType>().unwrap(),
            MemoryType::Summary
        );
        assert_eq!("bogus".parse::<MemoryType>().unwrap(), MemoryType::Unknown);
        for kind in MemoryType::ALL {
            assert_eq!(kind.as_str().parse::<MemoryType>().unwrap(), kind);
        }
    }
    #[test]
    fn default_is_unknown_and_display_matches_as_str() {
        assert_eq!(MemoryType::default(), MemoryType::Unknown);
        assert_eq!(MemoryType::Coding.to_string(), "coding");
    }
    #[test]
    fn searchable_kinds_exclude_variable_and_unknown() {
        assert!(!MemoryType::Variable.is_searchable());
        assert!(!MemoryType::Unknown.is_searchable());
        assert_eq!(MemoryType::SEARCHABLE.len(), 5);
        for kind in MemoryType::SEARCHABLE {
            assert!(kind.is_searchable());
        }
        assert_eq!(
            MemoryType::SEARCHABLE,
            [
                MemoryType::UserProfile,
                MemoryType::SemanticMemory,
                MemoryType::EpisodicMemory,
                MemoryType::Summary,
                MemoryType::Coding,
            ]
        );
        assert_eq!(MemoryType::ALL.len(), 7);
    }
}
