//! The team join descriptor: how an external process is told which team to attach to.
//!
//! Python parity source: `openjiuwen/agent_teams/external/descriptor.py`. The descriptor is the
//! only connection source the `team-member` CLI has: it carries the team and member identity, the
//! client [`ClientScope`] (a first-class member vs. an outside operator), and the exact team
//! settings (dispatch mode, teammate mode, messaging and subtask limits) the process must honour so
//! its behaviour matches the in-process member it stands in for.
//!
//! Three equivalent carriers exist and all end in the same JSON shape:
//!
//! * the [`TEAM_JOIN_ENV`] environment variable (used by a spawned member),
//! * a descriptor file, and
//! * a raw JSON string (used by scripts).
//!
//! Deserialization is tolerant by construction: every field is `#[serde(default)]`, so a partial
//! descriptor keeps the Python defaults and unknown fields written by a newer runtime are ignored.

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::constants::DEFAULT_LANGUAGE;
pub use crate::constants::TEAM_JOIN_ENV;
use crate::schema::{DispatchMode, MemberMode, TeamRole};

/// Which surface of a team the joining client drives.
///
/// Wire values are the Python strings `operator` and `member`; [`ClientScope::Operator`] is the
/// default, because an external caller that does not say otherwise is treated as a team operator
/// rather than as a first-class member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientScope {
    /// Outside team-control surface over the task board, messaging and roster.
    #[default]
    Operator,
    /// A spawned CLI acting as a first-class member, driven through the teammate tool set.
    Member,
}

impl ClientScope {
    /// Returns the wire spelling of this scope (`operator` or `member`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::Member => "member",
        }
    }
}

impl FromStr for ClientScope {
    type Err = Error;

    /// Parses a wire spelling (`operator` or `member`), ignoring surrounding whitespace.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the string is neither of the two wire spellings.
    fn from_str(raw: &str) -> Result<Self> {
        match raw.trim() {
            "operator" => Ok(Self::Operator),
            "member" => Ok(Self::Member),
            other => Err(Error::validation(format!(
                "unknown client scope '{other}'; expected 'operator' or 'member'"
            ))),
        }
    }
}

/// Everything an external process needs in order to join and drive one team.
///
/// The field set is the Python `TeamJoinDescriptor`; the JSON names are `snake_case` and every
/// field has a default, so a descriptor written by a partial producer still loads. `Default` is
/// manual because the Python default role is [`TeamRole::Teammate`] while [`TeamRole::default`] is
/// [`TeamRole::Leader`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamJoinDescriptor {
    /// Session the joining process belongs to; empty when the runtime did not pin one.
    pub session_id: String,
    /// Team to attach to.
    pub team_name: String,
    /// Member name the process acts as (its addressing key inside the team).
    pub member_name: String,
    /// Role the process plays in the team; defaults to [`TeamRole::Teammate`].
    pub role: TeamRole,
    /// Client surface the process drives; defaults to [`ClientScope::Operator`].
    pub scope: ClientScope,
    /// Language code used for prompts and rendered text (`cn` by default, `cf`/`en` accepted).
    pub language: String,
    /// Dispatch mode of the team; the tool set and the claim loop follow it.
    pub dispatch_mode: DispatchMode,
    /// Whether members may message each other directly under scheduled dispatch.
    pub enable_member_messaging: bool,
    /// Maximum subtask nesting a member may create; `0` disables member task creation.
    pub max_subtask_depth: usize,
    /// Teammate mode of the team (`build_mode` or `plan_mode`).
    pub teammate_mode: MemberMode,
    /// Storage configuration passed through verbatim to the team store factory.
    pub db_config: Value,
    /// Transport configuration passed through verbatim to the messager factory.
    pub transport_config: Value,
    /// Workspace configuration passed through verbatim to the workspace provider.
    pub workspace_config: Value,
    /// Absolute path of the member workspace, when the spawner pinned one.
    pub workspace_path: Option<String>,
}

impl Default for TeamJoinDescriptor {
    /// Returns the Python descriptor defaults (see the field documentation).
    fn default() -> Self {
        Self {
            session_id: String::new(),
            team_name: String::new(),
            member_name: String::new(),
            role: TeamRole::Teammate,
            scope: ClientScope::Operator,
            language: DEFAULT_LANGUAGE.to_string(),
            dispatch_mode: DispatchMode::Autonomous,
            enable_member_messaging: false,
            max_subtask_depth: 0,
            teammate_mode: MemberMode::BuildMode,
            db_config: Value::Object(Map::new()),
            transport_config: Value::Object(Map::new()),
            workspace_config: Value::Object(Map::new()),
            workspace_path: None,
        }
    }
}

impl TeamJoinDescriptor {
    /// Loads the descriptor from the [`TEAM_JOIN_ENV`] environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the variable is unset or empty, and [`Error::Validation`]
    /// when its value is not a valid descriptor.
    pub fn from_env() -> Result<Self> {
        Self::from_env_map(|key| std::env::var(key).ok())
    }

    /// Loads the descriptor through an injectable environment lookup.
    ///
    /// This is the form used by [`Self::from_env`]; tests call it with a map instead of mutating
    /// the process environment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the lookup yields `None` or a blank value, and
    /// [`Error::Validation`] when the value is not a valid descriptor.
    pub fn from_env_map(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        match get(TEAM_JOIN_ENV) {
            Some(raw) if !raw.trim().is_empty() => Self::from_json(&raw),
            _ => Err(Error::config(format!(
                "environment variable {TEAM_JOIN_ENV} is not set; cannot join team"
            ))),
        }
    }

    /// Parses a descriptor from its JSON representation.
    ///
    /// Missing fields keep their defaults and unknown fields are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `raw` is not a JSON object shaped like the descriptor.
    pub fn from_json(raw: &str) -> Result<Self> {
        serde_json::from_str(raw)
            .map_err(|err| Error::validation(format!("malformed team join descriptor: {err}")))
    }

    /// Serializes the descriptor to compact JSON, the form written to [`TEAM_JOIN_ENV`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if the descriptor cannot be serialized.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|err| {
            Error::validation(format!("failed to serialize team join descriptor: {err}"))
        })
    }

    /// Reads and parses a descriptor file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the file cannot be read, and [`Error::Validation`] when its
    /// contents are not a valid descriptor.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|err| {
            Error::config(format!(
                "failed to read team join descriptor '{}': {err}",
                path.display()
            ))
        })?;
        Self::from_json(&raw)
    }

    /// Returns the environment overlay that hands this descriptor to a child process.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] if the descriptor cannot be serialized.
    pub fn to_env(&self) -> Result<BTreeMap<String, String>> {
        Ok(BTreeMap::from([(
            TEAM_JOIN_ENV.to_string(),
            self.to_json()?,
        )]))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn defaults_match_the_python_descriptor() {
        let descriptor = TeamJoinDescriptor::default();
        assert_eq!(descriptor.session_id, "");
        assert_eq!(descriptor.team_name, "");
        assert_eq!(descriptor.member_name, "");
        assert_eq!(descriptor.role, TeamRole::Teammate);
        assert_eq!(descriptor.scope, ClientScope::Operator);
        assert_eq!(descriptor.language, "cn");
        assert_eq!(descriptor.dispatch_mode, DispatchMode::Autonomous);
        assert!(!descriptor.enable_member_messaging);
        assert_eq!(descriptor.max_subtask_depth, 0);
        assert_eq!(descriptor.teammate_mode, MemberMode::BuildMode);
        assert_eq!(descriptor.db_config, Value::Object(Map::new()));
        assert_eq!(descriptor.transport_config, Value::Object(Map::new()));
        assert_eq!(descriptor.workspace_config, Value::Object(Map::new()));
        assert_eq!(descriptor.workspace_path, None);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let descriptor = TeamJoinDescriptor::from_json(
            r#"{"team_name": "alpha", "member_name": "dev-1", "future_knob": {"a": 1}}"#,
        )
        .unwrap();
        assert_eq!(descriptor.team_name, "alpha");
        assert_eq!(descriptor.member_name, "dev-1");
        assert_eq!(descriptor.role, TeamRole::Teammate);
    }

    #[test]
    fn partial_json_keeps_the_defaults() {
        let descriptor = TeamJoinDescriptor::from_json(r#"{"team_name": "alpha"}"#).unwrap();
        assert_eq!(
            descriptor,
            TeamJoinDescriptor {
                team_name: "alpha".to_string(),
                ..TeamJoinDescriptor::default()
            }
        );
    }

    #[test]
    fn wire_spellings_round_trip() {
        let descriptor = TeamJoinDescriptor {
            scope: ClientScope::Member,
            role: TeamRole::Teammate,
            dispatch_mode: DispatchMode::Scheduled,
            teammate_mode: MemberMode::PlanMode,
            ..TeamJoinDescriptor::default()
        };
        let value: Value = serde_json::from_str(&descriptor.to_json().unwrap()).unwrap();
        assert_eq!(value["scope"], "member");
        assert_eq!(value["role"], "teammate");
        assert_eq!(value["dispatch_mode"], "scheduled");
        assert_eq!(value["teammate_mode"], "plan_mode");
        assert_eq!(
            TeamJoinDescriptor::from_json(&descriptor.to_json().unwrap()).unwrap(),
            descriptor
        );
    }

    #[test]
    fn client_scope_parses_its_wire_spellings() {
        assert_eq!(ClientScope::default(), ClientScope::Operator);
        assert_eq!(ClientScope::Operator.as_str(), "operator");
        assert_eq!(ClientScope::Member.as_str(), "member");
        assert_eq!(
            " member ".parse::<ClientScope>().unwrap(),
            ClientScope::Member
        );
        let err = "observer".parse::<ClientScope>().unwrap_err();
        assert!(err.to_string().contains("unknown client scope 'observer'"));
    }

    #[test]
    fn missing_environment_variable_is_a_config_error() {
        let err = TeamJoinDescriptor::from_env_map(|_| None).unwrap_err();
        assert!(
            err.to_string()
                .contains("environment variable OPENJIUWEN_TEAM_JOIN is not set; cannot join team"),
            "unexpected message: {err}"
        );
        let blank = TeamJoinDescriptor::from_env_map(|_| Some("   ".to_string())).unwrap_err();
        assert!(blank.to_string().contains("is not set"));
    }

    #[test]
    fn environment_map_is_parsed() {
        let descriptor = TeamJoinDescriptor::from_env_map(|key| {
            (key == TEAM_JOIN_ENV)
                .then(|| r#"{"team_name": "alpha", "scope": "member"}"#.to_string())
        })
        .unwrap();
        assert_eq!(descriptor.team_name, "alpha");
        assert_eq!(descriptor.scope, ClientScope::Member);
    }

    #[test]
    fn malformed_json_reports_the_serde_reason() {
        let err = TeamJoinDescriptor::from_json("{not json").unwrap_err();
        let message = err.to_string();
        assert!(
            message.starts_with("validation: malformed team join descriptor: "),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn file_round_trip() {
        let descriptor = TeamJoinDescriptor {
            team_name: "alpha".to_string(),
            member_name: "dev-1".to_string(),
            scope: ClientScope::Member,
            workspace_path: Some("/tmp/ws".to_string()),
            ..TeamJoinDescriptor::default()
        };
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(descriptor.to_json().unwrap().as_bytes())
            .unwrap();
        file.flush().unwrap();
        assert_eq!(
            TeamJoinDescriptor::from_file(file.path()).unwrap(),
            descriptor
        );
    }

    #[test]
    fn missing_file_is_a_config_error() {
        let path = std::path::Path::new("does-not-exist/team-join.json");
        let err = TeamJoinDescriptor::from_file(path).unwrap_err();
        assert!(
            err.to_string().starts_with(
                "config: failed to read team join descriptor 'does-not-exist/team-join.json': "
            ),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn env_overlay_serializes_the_descriptor() {
        let descriptor = TeamJoinDescriptor {
            team_name: "alpha".to_string(),
            ..TeamJoinDescriptor::default()
        };
        let env = descriptor.to_env().unwrap();
        assert_eq!(env.len(), 1);
        assert_eq!(
            TeamJoinDescriptor::from_json(&env[TEAM_JOIN_ENV]).unwrap(),
            descriptor
        );
    }
}
