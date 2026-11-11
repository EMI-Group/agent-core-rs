//! Conversation roles.
//!
//! Ports the `role` field of `openjiuwen/core/foundation/llm/schema/message.py`.
//! Python models the role as a plain `str`; Rust keeps the four known roles as
//! variants and preserves anything else verbatim in [`Role::Other`], so vendor
//! extensions (`"developer"`, `"function"`, ...) round-trip instead of failing.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

/// The role of a chat message.
///
/// Serializes as the lowercase role string (`"system"`, `"user"`, `"assistant"`,
/// `"tool"`) and deserializes from any string, mapping unrecognized values to
/// [`Role::Other`] so provider-specific roles survive a round-trip.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Role {
    /// The system prompt that frames the conversation.
    System,
    /// Input authored by the end user.
    User,
    /// Output authored by the model.
    Assistant,
    /// The result of a tool call.
    Tool,
    /// Any role this crate does not know; the payload is the original string.
    Other(String),
}

impl Role {
    /// The role as a wire string (`Role::Other(s)` yields `s`).
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(other) => other.as_str(),
        }
    }

    /// Whether this is [`Role::System`].
    #[must_use]
    pub const fn is_system(&self) -> bool {
        matches!(self, Self::System)
    }

    /// Whether this is [`Role::User`].
    #[must_use]
    pub const fn is_user(&self) -> bool {
        matches!(self, Self::User)
    }

    /// Whether this is [`Role::Assistant`].
    #[must_use]
    pub const fn is_assistant(&self) -> bool {
        matches!(self, Self::Assistant)
    }

    /// Whether this is [`Role::Tool`].
    #[must_use]
    pub const fn is_tool(&self) -> bool {
        matches!(self, Self::Tool)
    }
}

impl Default for Role {
    /// The default role is [`Role::User`].
    fn default() -> Self {
        Self::User
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Role {
    type Err = std::convert::Infallible;

    /// Never fails: unknown strings become [`Role::Other`].
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::from(s))
    }
}

impl From<&str> for Role {
    fn from(value: &str) -> Self {
        match value {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl From<String> for Role {
    fn from(value: String) -> Self {
        match value.as_str() {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            _ => Self::Other(value),
        }
    }
}

impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(role) => Ok(Self::from(role)),
            other => Err(de::Error::custom(format!(
                "expected a role string such as \"user\", found {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_roles_serialize_as_plain_strings() {
        assert_eq!(serde_json::to_value(Role::System).unwrap(), "system");
        assert_eq!(serde_json::to_value(Role::User).unwrap(), "user");
        assert_eq!(serde_json::to_value(Role::Assistant).unwrap(), "assistant");
        assert_eq!(serde_json::to_value(Role::Tool).unwrap(), "tool");
    }

    #[test]
    fn known_roles_round_trip_through_serde() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let json = serde_json::to_string(&role).unwrap();
            let parsed: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, role);
        }
    }

    #[test]
    fn unknown_role_is_preserved_verbatim() {
        // ports the tolerant role handling of `foundation/llm/schema/message.py`.
        let role: Role = serde_json::from_str("\"developer\"").unwrap();
        assert_eq!(role, Role::Other("developer".to_owned()));
        assert_eq!(serde_json::to_string(&role).unwrap(), "\"developer\"");
        assert_eq!(role.as_str(), "developer");
        assert_eq!(role.to_string(), "developer");
    }

    #[test]
    fn non_string_roles_are_errors() {
        for raw in ["42", "null", "true", "{}", "[]", "1.5"] {
            let parsed = serde_json::from_str::<Role>(raw);
            assert!(parsed.is_err(), "`{raw}` must not deserialize into a role");
        }
    }

    #[test]
    fn from_str_and_display_round_trip() {
        for raw in ["system", "user", "assistant", "tool", "human", ""] {
            let role: Role = raw.parse().unwrap();
            assert_eq!(role.to_string(), raw);
            assert_eq!(role.as_str(), raw);
        }
        assert_eq!("system".parse::<Role>().unwrap(), Role::System);
        assert_eq!(
            "not-a-role".parse::<Role>().unwrap(),
            Role::Other("not-a-role".to_owned())
        );
    }

    #[test]
    fn default_is_user() {
        assert_eq!(Role::default(), Role::User);
        assert!(Role::default().is_user());
    }

    #[test]
    fn from_conversions_match_known_roles() {
        assert_eq!(Role::from("system"), Role::System);
        assert_eq!(Role::from("tool".to_owned()), Role::Tool);
        assert_eq!(Role::from("function"), Role::Other("function".to_owned()));
        assert_eq!(
            Role::from(String::from("native")),
            Role::Other("native".to_owned())
        );
    }

    #[test]
    fn predicate_helpers_classify_each_role() {
        assert!(Role::System.is_system());
        assert!(Role::User.is_user());
        assert!(Role::Assistant.is_assistant());
        assert!(Role::Tool.is_tool());
        assert!(!Role::System.is_user());
        assert!(!Role::Other("system".to_owned()).is_system());
    }

    #[test]
    fn equality_and_hashing_follow_the_payload() {
        assert_eq!(Role::Other("x".to_owned()), Role::Other("x".to_owned()));
        assert_ne!(Role::Other("x".to_owned()), Role::Other("y".to_owned()));
        assert_ne!(Role::System, Role::User);

        let mut set = std::collections::HashSet::new();
        set.insert(Role::System);
        set.insert(Role::System);
        set.insert(Role::Other("x".to_owned()));
        set.insert(Role::Other("x".to_owned()));
        set.insert(Role::User);
        assert_eq!(set.len(), 3);
    }
}
