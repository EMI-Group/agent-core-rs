//! Session scopes and subjects: the isolation boundary a disk session belongs to.
//!
//! Ports `openjiuwen/core/session/session_controller/scope.py` (v0.1.20). A
//! [`SessionScope`] is a [`Scope`] plus an optional [`Subject`], and a
//! [`SessionScopeKey`] prefixes it with the owning agent id; the string forms
//! are the on-disk index keys (`agent:<agent_id>:<scope>[:<subject>]`).
//!
//! ```
//! use openjiuwen_session::SessionScope;
//!
//! let scope = SessionScope::group_user("team-1", "ada");
//! assert_eq!(scope.to_string(), "main:group:team-1:user:ada");
//! ```
use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The basic isolation boundary of a session.
///
/// [`Scope::Main`] is the built-in default (`"main"`); [`Scope::Custom`] holds
/// any other scope name, so applications can add their own namespaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Scope {
    /// The built-in default scope, serialized as `"main"`.
    Main,
    /// An application-defined scope name.
    Custom(String),
}

impl Scope {
    /// Returns the scope name used in the string form.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Main => "main",
            Self::Custom(name) => name,
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Scope {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        if value == "main" {
            return Ok(Self::Main);
        }
        if value.is_empty() {
            return Err(Error::validation("scope must not be empty"));
        }
        if value.contains(':') {
            return Err(Error::validation(format!(
                "scope must not contain ':': {value:?}"
            )));
        }
        Ok(Self::Custom(value.to_string()))
    }
}

impl From<Scope> for String {
    fn from(scope: Scope) -> Self {
        scope.to_string()
    }
}

impl TryFrom<String> for Scope {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

/// The participant a session is isolated for within a [`Scope`].
///
/// The wire forms mirror Python: `direct:<user>`, `group:<group>`, and
/// `group:<group>:user:<user>`. Group-user ids that themselves contain `:` are
/// written with the `group-user-v2:` prefix and percent-escaped, while plain
/// ids keep the legacy form so persisted keys stay stable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Subject {
    /// A one-to-one conversation with one user (`direct:<user>`).
    Direct {
        /// The user identifier.
        user_id: String,
    },
    /// A group conversation shared by its members (`group:<group>`).
    Group {
        /// The group identifier.
        group_id: String,
    },
    /// One user's isolated perspective inside a group
    /// (`group:<group>:user:<user>`).
    GroupUser {
        /// The group identifier.
        group_id: String,
        /// The user identifier.
        user_id: String,
    },
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct { user_id } => write!(f, "direct:{user_id}"),
            Self::Group { group_id } => write!(f, "group:{group_id}"),
            Self::GroupUser { group_id, user_id } => {
                if group_id.contains(':') || user_id.contains(':') {
                    write!(
                        f,
                        "group-user-v2:{}:user:{}",
                        escape_id(group_id),
                        escape_id(user_id)
                    )
                } else {
                    write!(f, "group:{group_id}:user:{user_id}")
                }
            }
        }
    }
}

impl FromStr for Subject {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        if let Some(rest) = value.strip_prefix("group-user-v2:") {
            return parse_group_user(rest, "group-user-v2", true);
        }
        if let Some(user_id) = value.strip_prefix("direct:") {
            if user_id.is_empty() {
                return Err(Error::validation(
                    "direct subject user_id must not be empty",
                ));
            }
            return Ok(Self::Direct {
                user_id: user_id.to_string(),
            });
        }
        if let Some(rest) = value.strip_prefix("group:") {
            if rest.contains(":user:") {
                return parse_group_user(rest, "group", false);
            }
            if rest.is_empty() {
                return Err(Error::validation(
                    "group subject group_id must not be empty",
                ));
            }
            return Ok(Self::Group {
                group_id: rest.to_string(),
            });
        }
        Err(Error::validation(format!(
            "unknown subject format: {value:?}"
        )))
    }
}

impl From<Subject> for String {
    fn from(subject: Subject) -> Self {
        subject.to_string()
    }
}

impl TryFrom<String> for Subject {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

/// Parses `<group>:user:<user>` with an exact four-part split.
fn parse_group_user(value: &str, prefix: &str, decode: bool) -> Result<Subject> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 || parts[1] != "user" {
        return Err(Error::validation(format!(
            "group-user subject must have format \
             '{prefix}:{{group_id}}:user:{{user_id}}', got {value:?}"
        )));
    }
    let (mut group_id, mut user_id) = (parts[0].to_string(), parts[2].to_string());
    if decode {
        group_id = percent_decode(&group_id);
        user_id = percent_decode(&user_id);
    }
    if group_id.is_empty() || user_id.is_empty() {
        return Err(Error::validation(
            "group-user subject group_id and user_id must not be empty",
        ));
    }
    Ok(Subject::GroupUser { group_id, user_id })
}

/// Percent-escapes `%` and `:` in an id for the `group-user-v2` form.
fn escape_id(value: &str) -> String {
    value.replace('%', "%25").replace(':', "%3A")
}

/// Decodes `%XX` escapes; malformed sequences are preserved literally.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if let Some(hex) = value.get(index + 1..index + 3) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    index += 3;
                    continue;
                }
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A session isolation boundary: a [`Scope`] and an optional [`Subject`].
///
/// The string form is `"{scope}"` or `"{scope}:{subject}"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionScope {
    /// The basic isolation boundary.
    pub scope: Scope,
    /// The participant the session is isolated for, when any.
    pub subject: Option<Subject>,
}

impl SessionScope {
    /// The built-in main scope with no subject.
    #[must_use]
    pub fn main() -> Self {
        Self {
            scope: Scope::Main,
            subject: None,
        }
    }

    /// The main scope isolated to one direct-chat user.
    #[must_use]
    pub fn direct(user_id: impl Into<String>) -> Self {
        Self {
            scope: Scope::Main,
            subject: Some(Subject::Direct {
                user_id: user_id.into(),
            }),
        }
    }

    /// The main scope shared by a group.
    #[must_use]
    pub fn group(group_id: impl Into<String>) -> Self {
        Self {
            scope: Scope::Main,
            subject: Some(Subject::Group {
                group_id: group_id.into(),
            }),
        }
    }

    /// The main scope for one user's perspective inside a group.
    #[must_use]
    pub fn group_user(group_id: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            scope: Scope::Main,
            subject: Some(Subject::GroupUser {
                group_id: group_id.into(),
                user_id: user_id.into(),
            }),
        }
    }

    /// A custom scope name with no subject.
    #[must_use]
    pub fn custom(scope_name: impl Into<String>) -> Self {
        Self {
            scope: Scope::Custom(scope_name.into()),
            subject: None,
        }
    }
}

impl fmt::Display for SessionScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.subject {
            Some(subject) => write!(f, "{}:{subject}", self.scope),
            None => write!(f, "{}", self.scope),
        }
    }
}

impl FromStr for SessionScope {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let (scope_str, subject_str) = match value.split_once(':') {
            Some((scope, rest)) => (scope, Some(rest)),
            None => (value, None),
        };
        let scope: Scope = scope_str.parse()?;
        let subject = match subject_str {
            Some(subject) if !subject.is_empty() => Some(subject.parse()?),
            _ => None,
        };
        Ok(Self { scope, subject })
    }
}

impl From<SessionScope> for String {
    fn from(scope: SessionScope) -> Self {
        scope.to_string()
    }
}

impl TryFrom<String> for SessionScope {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

/// The globally unique index key of a session collection:
/// `agent:{agent_id}:{scope}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionScopeKey {
    /// The owning agent id.
    pub agent_id: String,
    /// The isolation boundary.
    pub scope: SessionScope,
}

impl fmt::Display for SessionScopeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "agent:{}:{}", self.agent_id, self.scope)
    }
}

impl FromStr for SessionScopeKey {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let rest = value.strip_prefix("agent:").ok_or_else(|| {
            Error::validation(format!(
                "session scope key must start with 'agent:': {value:?}"
            ))
        })?;
        let (agent_id, scope_str) = rest.split_once(':').ok_or_else(|| {
            Error::validation(format!("session scope key is missing a scope: {value:?}"))
        })?;
        if agent_id.is_empty() {
            return Err(Error::validation(
                "session scope key agent_id must not be empty",
            ));
        }
        Ok(Self {
            agent_id: agent_id.to_string(),
            scope: scope_str.parse()?,
        })
    }
}

impl From<SessionScopeKey> for String {
    fn from(key: SessionScopeKey) -> Self {
        key.to_string()
    }
}

impl TryFrom<String> for SessionScopeKey {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{Scope, SessionScope, SessionScopeKey, Subject};

    #[test]
    fn scope_round_trips_main_and_custom() {
        assert_eq!(Scope::Main.to_string(), "main");
        assert_eq!("main".parse::<Scope>().unwrap(), Scope::Main);
        assert_eq!(
            "tenant-a".parse::<Scope>().unwrap(),
            Scope::Custom("tenant-a".to_string())
        );
        assert_eq!(
            Scope::Custom("tenant-a".to_string()).to_string(),
            "tenant-a"
        );
    }

    #[test]
    fn subject_round_trips_all_three_forms() {
        for text in ["direct:u1", "group:g1", "group:g1:user:u1"] {
            let subject: Subject = text.parse().unwrap();
            assert_eq!(subject.to_string(), text);
        }
    }

    #[test]
    fn group_user_escapes_ids_containing_colons() {
        let subject = Subject::GroupUser {
            group_id: "g:1".to_string(),
            user_id: "u:1".to_string(),
        };
        let text = subject.to_string();
        assert_eq!(text, "group-user-v2:g%3A1:user:u%3A1");
        assert_eq!(text.parse::<Subject>().unwrap(), subject);
    }

    #[test]
    fn legacy_percent_sequences_stay_literal() {
        let subject: Subject = "group:g%3A1:user:u1".parse().unwrap();
        assert_eq!(
            subject,
            Subject::GroupUser {
                group_id: "g%3A1".to_string(),
                user_id: "u1".to_string(),
            }
        );
        // Plain ids keep the legacy form, so the percent stays literal.
        assert_eq!(subject.to_string(), "group:g%3A1:user:u1");
    }

    #[test]
    fn session_scope_round_trips() {
        let cases = [
            ("main", SessionScope::main()),
            ("main:direct:u1", SessionScope::direct("u1")),
            ("main:group:g1", SessionScope::group("g1")),
            (
                "main:group:g1:user:u1",
                SessionScope::group_user("g1", "u1"),
            ),
            ("tenant-a", SessionScope::custom("tenant-a")),
        ];
        for (text, expected) in cases {
            assert_eq!(text.parse::<SessionScope>().unwrap(), expected);
            assert_eq!(expected.to_string(), text);
        }
    }

    #[test]
    fn scope_key_round_trips() {
        let key = SessionScopeKey {
            agent_id: "agent-1".to_string(),
            scope: SessionScope::direct("u1"),
        };
        assert_eq!(key.to_string(), "agent:agent-1:main:direct:u1");
        assert_eq!(key.to_string().parse::<SessionScopeKey>().unwrap(), key);
    }

    #[test]
    fn malformed_values_are_validation_errors() {
        assert!("".parse::<Scope>().is_err());
        assert!("a:b".parse::<Scope>().is_err());
        assert!("unknown".parse::<Subject>().is_err());
        assert!("direct:".parse::<Subject>().is_err());
        assert!("group:g1:user:".parse::<Subject>().is_err());
        assert!("group:g1:user:u1:x".parse::<Subject>().is_err());
        assert!("nope:main".parse::<SessionScopeKey>().is_err());
        assert!("agent::main".parse::<SessionScopeKey>().is_err());
    }
}
