//! Execution and session identities for persistent agents.
//!
//! An [`ExecutionIdentity`] names one execution of one agent session; its fields are stable
//! across retries and are the input to every derived id in this module (event ids, operation ids
//! and directory names). A [`SessionBinding`] names the narrower `(agent, session)` pair used by
//! the persisted session record and its archive keys.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::util::{digest_hex, is_valid_identifier, percent_encode_identifier};

/// Prefix every persisted session key starts with.
pub const SESSION_KEY_PREFIX: &str = "pa:";

/// How an execution relates to its predecessor.
///
/// The relations are ordered from the first execution of a task to a plain continuation, so a
/// caller can compare them (a resume sorts before a continuation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkRelation {
    /// The first execution of a task; it has no predecessor.
    First,
    /// A resume of an execution that stopped to await the initiator's confirmation.
    ConfirmationResume,
    /// A continuation requested by the initiator through a recovery reference.
    Continuation,
}

impl LinkRelation {
    /// Returns the persisted snake-case spelling of the relation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::ConfirmationResume => "confirmation_resume",
            Self::Continuation => "continuation",
        }
    }
}

/// One execution of one agent session.
///
/// `source_ref`, `request_id` and `execution_id` distinguish executions of the same session;
/// `namespace_id` scopes them to a host namespace. All six values are identifiers (see
/// [`is_valid_identifier`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionIdentity {
    /// Host namespace that owns the execution.
    pub namespace_id: String,
    /// Agent that runs the execution.
    pub agent_id: String,
    /// Session the execution belongs to.
    pub session_id: String,
    /// Request that started the execution.
    pub request_id: String,
    /// Unique id of this execution.
    pub execution_id: String,
    /// Host correlation reference of the source message or trigger.
    pub source_ref: String,
}

impl ExecutionIdentity {
    /// Creates and validates an execution identity.
    ///
    /// # Errors
    /// `Error::Validation` when any of the six arguments is not a valid identifier.
    pub fn new(
        namespace_id: impl Into<String>,
        agent_id: impl Into<String>,
        session_id: impl Into<String>,
        request_id: impl Into<String>,
        execution_id: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Result<Self> {
        let identity = Self {
            namespace_id: namespace_id.into(),
            agent_id: agent_id.into(),
            session_id: session_id.into(),
            request_id: request_id.into(),
            execution_id: execution_id.into(),
            source_ref: source_ref.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Validates every identifier of the identity.
    ///
    /// # Errors
    /// `Error::Validation("<field> must be an identifier")` naming the first invalid field.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("namespace_id", &self.namespace_id),
            ("agent_id", &self.agent_id),
            ("session_id", &self.session_id),
            ("request_id", &self.request_id),
            ("execution_id", &self.execution_id),
            ("source_ref", &self.source_ref),
        ] {
            if !is_valid_identifier(value) {
                return Err(Error::validation(format!("{field} must be an identifier")));
            }
        }
        Ok(())
    }

    /// Returns the persisted session binding of this execution.
    #[must_use]
    pub fn session_binding(&self) -> SessionBinding {
        SessionBinding {
            agent_id: self.agent_id.clone(),
            session_id: self.session_id.clone(),
        }
    }

    /// Returns the directory name of this execution's journal.
    ///
    /// The name is the percent-encoded execution id, so it is a single safe path component.
    #[must_use]
    pub fn directory_name(&self) -> String {
        percent_encode_identifier(&self.execution_id)
    }

    /// Returns the stable event id of local sequence number `local_seq`.
    ///
    /// The id is `"evt_"` followed by the SHA-256 of the canonical JSON of the six identity
    /// fields plus `local_seq`; it is stable across retries and differs for every sequence
    /// number.
    #[must_use]
    pub fn event_id(&self, local_seq: u64) -> String {
        let identity = json!({
            "namespace_id": self.namespace_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "source_ref": self.source_ref,
            "local_seq": local_seq,
        });
        format!("evt_{}", digest_hex(&identity))
    }
}

/// The `(agent, session)` pair a persisted session record belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionBinding {
    /// Agent that owns the session.
    pub agent_id: String,
    /// Session the record belongs to.
    pub session_id: String,
}

impl SessionBinding {
    /// Creates and validates a session binding.
    ///
    /// # Errors
    /// `Error::Validation` when either argument is not a valid identifier.
    pub fn new(agent_id: impl Into<String>, session_id: impl Into<String>) -> Result<Self> {
        let binding = Self {
            agent_id: agent_id.into(),
            session_id: session_id.into(),
        };
        if !is_valid_identifier(&binding.agent_id) {
            return Err(Error::validation("agent_id must be an identifier"));
        }
        if !is_valid_identifier(&binding.session_id) {
            return Err(Error::validation("session_id must be an identifier"));
        }
        Ok(binding)
    }

    /// Returns the key prefix of this session's records.
    ///
    /// Both ids are percent-encoded on their own, so no two identity pairs share a prefix.
    #[must_use]
    pub fn key_prefix(&self) -> String {
        format!(
            "{SESSION_KEY_PREFIX}{}:{}:",
            percent_encode_identifier(&self.agent_id),
            percent_encode_identifier(&self.session_id)
        )
    }

    /// Returns the key of the persisted session state record.
    #[must_use]
    pub fn state_key(&self) -> String {
        format!("{}state", self.key_prefix())
    }

    /// Returns the key of the archive of the old window with sequence number `window_seq`.
    ///
    /// # Errors
    /// `Error::Validation` when `window_seq` is zero.
    pub fn archive_key(&self, window_seq: u64) -> Result<String> {
        if window_seq == 0 {
            return Err(Error::validation(
                "window sequence must be a positive integer",
            ));
        }
        Ok(format!("{}archive:{window_seq}", self.key_prefix()))
    }

    /// Returns the key of the rebuildable archive index.
    #[must_use]
    pub fn archive_index_key(&self) -> String {
        format!("{}archive_index", self.key_prefix())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{ExecutionIdentity, LinkRelation, SessionBinding};

    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    #[test]
    fn session_keys_match_the_python_scheme() {
        let binding = SessionBinding::new("agent", "s/1").unwrap();
        assert_eq!(binding.key_prefix(), "pa:agent:s%2F1:");
        assert_eq!(binding.state_key(), "pa:agent:s%2F1:state");
        assert_eq!(binding.archive_key(2).unwrap(), "pa:agent:s%2F1:archive:2");
        assert_eq!(binding.archive_index_key(), "pa:agent:s%2F1:archive_index");
    }

    #[test]
    fn session_key_prefix_encodes_both_ids_independently() {
        let binding = SessionBinding::new("a:b", "c:d").unwrap();
        assert_eq!(binding.key_prefix(), "pa:a%3Ab:c%3Ad:");
        let other = SessionBinding::new("a", "b:c:d").unwrap();
        assert_ne!(binding.key_prefix(), other.key_prefix());
    }

    #[test]
    fn archive_key_rejects_zero() {
        let binding = SessionBinding::new("agent", "session").unwrap();
        let error = binding.archive_key(0).unwrap_err();
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
        assert!(error.to_string().contains("positive integer"));
    }

    #[test]
    fn invalid_identifiers_are_rejected() {
        assert!(ExecutionIdentity::new("", "a", "s", "r", "e", "src").is_err());
        assert!(ExecutionIdentity::new("ns", "a b", "s", "r", "e", "src").is_err());
        let error = ExecutionIdentity::new("ns", "a", "s\n", "r", "e", "src").unwrap_err();
        assert!(error
            .to_string()
            .contains("session_id must be an identifier"));
        assert!(SessionBinding::new("agent", "").is_err());
        assert!(SessionBinding::new("", "session").is_err());
    }

    #[test]
    fn directory_name_percent_encodes_the_execution() {
        let identity =
            ExecutionIdentity::new("ns", "agent", "session", "request", "exec/1", "source")
                .unwrap();
        assert_eq!(identity.directory_name(), "exec%2F1");
    }

    #[test]
    fn event_ids_are_stable_and_sequence_specific() {
        let identity = identity();
        let first = identity.event_id(1);
        assert_eq!(first, identity.event_id(1));
        assert!(first.starts_with("evt_"));
        assert_eq!(first.len(), 4 + 64);
        assert!(first[4..].chars().all(|ch| ch.is_ascii_hexdigit()));
        assert_ne!(first, identity.event_id(2));
        let other =
            ExecutionIdentity::new("ns2", "agent", "session", "request", "execution", "source")
                .unwrap();
        assert_ne!(first, other.event_id(1));
    }

    #[test]
    fn identities_round_trip_and_tolerate_unknown_fields() {
        let identity = identity();
        let encoded = serde_json::to_value(&identity).unwrap();
        let decoded: ExecutionIdentity = serde_json::from_value(encoded).unwrap();
        assert_eq!(identity, decoded);
        let tolerant: ExecutionIdentity = serde_json::from_value(serde_json::json!({
            "namespace_id": "ns",
            "agent_id": "agent",
            "session_id": "session",
            "request_id": "request",
            "execution_id": "execution",
            "source_ref": "source",
            "future_field": true,
        }))
        .unwrap();
        assert_eq!(tolerant, identity);
    }

    #[test]
    fn session_bindings_round_trip_and_hash() {
        let binding = SessionBinding::new("agent", "session").unwrap();
        assert_eq!(identity().session_binding(), binding);
        let decoded: SessionBinding =
            serde_json::from_str(&serde_json::to_string(&binding).unwrap()).unwrap();
        assert_eq!(binding, decoded);
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(binding));
    }

    #[test]
    fn link_relations_use_snake_case_and_order() {
        assert_eq!(
            serde_json::to_string(&LinkRelation::First).unwrap(),
            "\"first\""
        );
        assert_eq!(
            serde_json::to_string(&LinkRelation::ConfirmationResume).unwrap(),
            "\"confirmation_resume\""
        );
        assert_eq!(
            serde_json::to_string(&LinkRelation::Continuation).unwrap(),
            "\"continuation\""
        );
        let decoded: LinkRelation = serde_json::from_str("\"continuation\"").unwrap();
        assert_eq!(decoded, LinkRelation::Continuation);
        assert!(LinkRelation::First < LinkRelation::ConfirmationResume);
        assert_eq!(LinkRelation::First.as_str(), "first");
    }
}
