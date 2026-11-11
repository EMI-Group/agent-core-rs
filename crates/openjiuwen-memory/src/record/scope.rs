//! [`MemoryScope`]: the user / scope / session / agent hierarchy that owns a record.
//!
//! Ports the scope handling of the Python `openjiuwen.core.memory` package: the
//! `scope_id` rules follow `LongTermMemory._validate_id` (non-empty, no `/`, at most
//! 128 characters) and a missing id falls back to [`DEFAULT_ID`] (`"__default__"`).
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
/// Placeholder id used when a caller does not supply a `user_id` or `scope_id`.
pub const DEFAULT_ID: &str = "__default__";
/// Maximum accepted `scope_id` length, in characters (Python `_validate_id` parity).
const MAX_SCOPE_ID_CHARS: usize = 128;
/// The ownership tuple of a memory record.
///
/// `user_id` and `scope_id` always identify the tenant and therefore participate in
/// every storage key; `session_id` and `agent_id` narrow the scope further and are
/// only present when the caller binds memory to a session or an agent.
///
/// JSON is flat and `snake_case`; `session_id`/`agent_id` are omitted when absent and
/// a missing `user_id`/`scope_id` defaults to [`DEFAULT_ID`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemoryScope {
    /// Owning user id; defaults to [`DEFAULT_ID`].
    pub user_id: String,
    /// Owning scope id (workspace, tenant, or project); defaults to [`DEFAULT_ID`].
    pub scope_id: String,
    /// Session the memory belongs to, when the scope is session-bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Agent the memory belongs to, when the scope is agent-bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}
impl Default for MemoryScope {
    /// Both required ids are [`DEFAULT_ID`]; optional ids are absent.
    fn default() -> Self {
        Self {
            user_id: DEFAULT_ID.to_owned(),
            scope_id: DEFAULT_ID.to_owned(),
            session_id: None,
            agent_id: None,
        }
    }
}
impl MemoryScope {
    /// Builds a scope from the required ids; optional ids start absent.
    #[must_use]
    pub fn new(user_id: impl Into<String>, scope_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            scope_id: scope_id.into(),
            ..Self::default()
        }
    }
    /// Builds a user-wide scope (`scope_id = DEFAULT_ID`).
    #[must_use]
    pub fn user(user_id: impl Into<String>) -> Self {
        Self::new(user_id, DEFAULT_ID)
    }
    /// Returns the scope bound to `session_id`.
    #[must_use]
    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Returns the scope bound to `agent_id`.
    #[must_use]
    pub fn agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }
    /// Validates every present id, returning `Error::Validation` naming the field.
    ///
    /// `user_id` and `scope_id` must be non-empty and free of NUL; `scope_id` also
    /// rejects `/` and `\` (it must stay a single storage-key segment) and is limited
    /// to 128 characters. Optional ids must be non-empty and free of NUL when present;
    /// a `session_id` additionally rejects `/`.
    pub fn validate(&self) -> Result<()> {
        validate_component("user_id", &self.user_id, false, false, None)?;
        validate_component(
            "scope_id",
            &self.scope_id,
            true,
            true,
            Some(MAX_SCOPE_ID_CHARS),
        )?;
        if let Some(session_id) = &self.session_id {
            validate_component("session_id", session_id, true, false, None)?;
        }
        if let Some(agent_id) = &self.agent_id {
            validate_component("agent_id", agent_id, false, false, None)?;
        }
        Ok(())
    }
    /// Returns `true` when `other` falls inside this scope.
    ///
    /// `user_id` and `scope_id` must always match; `session_id`/`agent_id` only
    /// constrain the match when they are set on `self`, so an unbound scope selects
    /// every session and agent of the same user and scope.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.user_id == other.user_id
            && self.scope_id == other.scope_id
            && self
                .session_id
                .as_ref()
                .is_none_or(|id| other.session_id.as_ref() == Some(id))
            && self
                .agent_id
                .as_ref()
                .is_none_or(|id| other.agent_id.as_ref() == Some(id))
    }
}
/// Validates one scope id component.
///
/// `reject_slash`/`reject_backslash` forbid the storage-key separators;
/// `max_chars` bounds the character count.
fn validate_component(
    field: &str,
    value: &str,
    reject_slash: bool,
    reject_backslash: bool,
    max_chars: Option<usize>,
) -> Result<()> {
    if value.is_empty() {
        return Err(Error::validation(format!(
            "memory scope `{field}` must not be empty"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "memory scope `{field}` must not contain NUL"
        )));
    }
    if reject_backslash && value.contains('\\') {
        return Err(Error::validation(format!(
            "memory scope `{field}` must not contain a backslash"
        )));
    }
    if reject_slash && value.contains('/') {
        return Err(Error::validation(format!(
            "memory scope `{field}` must not contain '/'"
        )));
    }
    if let Some(limit) = max_chars {
        let length = value.chars().count();
        if length > limit {
            return Err(Error::validation(format!(
                "memory scope `{field}` must be at most {limit} characters (got {length})"
            )));
        }
    }
    Ok(())
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn default_uses_default_id_for_both_required_ids() {
        let scope = MemoryScope::default();
        assert_eq!(scope.user_id, DEFAULT_ID);
        assert_eq!(scope.scope_id, DEFAULT_ID);
        assert_eq!(scope.session_id, None);
        assert_eq!(scope.agent_id, None);
        assert!(scope.validate().is_ok());
    }
    #[test]
    fn new_user_and_builders() {
        let scope = MemoryScope::user("alice");
        assert_eq!(scope.user_id, "alice");
        assert_eq!(scope.scope_id, DEFAULT_ID);
        let scope = MemoryScope::new("alice", "ws").session("s1").agent("a1");
        assert_eq!(scope.session_id.as_deref(), Some("s1"));
        assert_eq!(scope.agent_id.as_deref(), Some("a1"));
        assert!(scope.validate().is_ok());
    }
    #[test]
    fn validate_rejects_bad_user_id() {
        let scope = MemoryScope::new("", "ws");
        let err = scope.validate().unwrap_err();
        assert!(err.to_string().contains("user_id"));
        assert!(matches!(err, Error::Validation(_)));
        let scope = MemoryScope::new("ali\0ce", "ws");
        assert!(scope.validate().unwrap_err().to_string().contains("NUL"));
    }
    #[test]
    fn validate_rejects_bad_scope_id() {
        assert!(MemoryScope::new("alice", "")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("scope_id"));
        assert!(MemoryScope::new("alice", "with/slash")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("scope_id"));
        assert!(MemoryScope::new("alice", "with\\backslash")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("scope_id"));
        assert!(MemoryScope::new("alice", "nul\0id")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("NUL"));
        let long = "x".repeat(MAX_SCOPE_ID_CHARS + 1);
        assert!(MemoryScope::new("alice", &long)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most 128"));
        let at_limit = "x".repeat(MAX_SCOPE_ID_CHARS);
        assert!(MemoryScope::new("alice", &at_limit).validate().is_ok());
    }
    #[test]
    fn validate_rejects_bad_session_and_agent() {
        assert!(MemoryScope::new("alice", "ws")
            .session("")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id"));
        assert!(MemoryScope::new("alice", "ws")
            .session("s/1")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("session_id"));
        assert!(MemoryScope::new("alice", "ws")
            .agent("")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("agent_id"));
        assert!(MemoryScope::new("alice", "ws")
            .agent("a\0b")
            .validate()
            .unwrap_err()
            .to_string()
            .contains("NUL"));
    }
    #[test]
    fn matches_honours_optional_bound_ids() {
        let base = MemoryScope::new("alice", "ws");
        let with_session = MemoryScope::new("alice", "ws").session("s1");
        let other_session = MemoryScope::new("alice", "ws").session("s2");
        assert!(base.matches(&with_session));
        assert!(base.matches(&other_session));
        assert!(with_session.matches(&with_session));
        assert!(!with_session.matches(&other_session));
        assert!(!with_session.matches(&base));
        let with_agent = MemoryScope::new("alice", "ws").agent("a1");
        assert!(base.matches(&with_agent));
        assert!(with_agent.matches(&with_agent));
        assert!(!with_agent.matches(&base));
        assert!(!base.matches(&MemoryScope::new("alice", "other")));
        assert!(!base.matches(&MemoryScope::new("bob", "ws")));
    }
    #[test]
    fn serde_omits_absent_optionals_and_defaults_missing_ids() {
        let scope = MemoryScope::new("alice", "ws");
        assert_eq!(
            serde_json::to_value(&scope).unwrap(),
            serde_json::json!({"user_id": "alice", "scope_id": "ws"})
        );
        let restored: MemoryScope = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(restored, MemoryScope::default());
        let restored: MemoryScope =
            serde_json::from_value(serde_json::json!({"user_id": "alice"})).unwrap();
        assert_eq!(restored.user_id, "alice");
        assert_eq!(restored.scope_id, DEFAULT_ID);
        let restored: MemoryScope = serde_json::from_value(serde_json::json!({
            "user_id": "alice", "scope_id": "ws", "session_id": "s1", "agent_id": "a1"
        }))
        .unwrap();
        assert_eq!(
            restored,
            MemoryScope::new("alice", "ws").session("s1").agent("a1")
        );
    }
}
