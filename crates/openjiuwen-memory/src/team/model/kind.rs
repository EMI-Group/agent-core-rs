//! [`TeamMemoryKind`] and [`TeamVisibility`]: what a team memory is and who may read it.
//!
//! Kinds serialize as plain `snake_case` strings so an unknown kind from a newer
//! writer round-trips as [`TeamMemoryKind::Custom`] instead of failing. Visibility is
//! externally tagged: `"team"`, `"private"`, or `{"members": ["alice", "bob"]}`.
use super::validate::{normalize_members, validate_component};
use super::{MAX_SCOPE_ID_CHARS, MAX_VISIBILITY_MEMBERS};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;
/// The kind of a team memory record.
///
/// The six builtin kinds mirror the Python `TEAM_MEMORY.md` extractor sections; any
/// other value is preserved verbatim as [`TeamMemoryKind::Custom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum TeamMemoryKind {
    /// A fact shared by the whole team; also the serde default.
    #[default]
    SharedFact,
    /// Durable profile information about one member.
    MemberProfile,
    /// A decision the team made, including its rationale.
    Decision,
    /// An open or completed work item.
    Task,
    /// A produced artifact (path, URL, or summary).
    Artifact,
    /// A handoff note or lesson learned between members.
    HandoffNote,
    /// A caller-defined kind, preserved verbatim.
    Custom(String),
}
impl TeamMemoryKind {
    /// Every builtin kind, in declaration order.
    pub const BUILTIN: [TeamMemoryKind; 6] = [
        TeamMemoryKind::SharedFact,
        TeamMemoryKind::MemberProfile,
        TeamMemoryKind::Decision,
        TeamMemoryKind::Task,
        TeamMemoryKind::Artifact,
        TeamMemoryKind::HandoffNote,
    ];
    /// The `snake_case` string of this kind, used in JSON and stats keys.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::SharedFact => "shared_fact",
            Self::MemberProfile => "member_profile",
            Self::Decision => "decision",
            Self::Task => "task",
            Self::Artifact => "artifact",
            Self::HandoffNote => "handoff_note",
            Self::Custom(kind) => kind,
        }
    }
    /// Returns `true` when this is one of the six builtin kinds.
    #[must_use]
    pub fn is_builtin(&self) -> bool {
        Self::BUILTIN.contains(self)
    }
    /// The Python `TEAM_MEMORY.md` extractor section this kind feeds.
    ///
    /// `Decision` maps to `decision`, `MemberProfile` to `member`, `HandoffNote` to
    /// `lesson`, and every other kind to `context`.
    #[must_use]
    pub const fn extractor_tag(&self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::MemberProfile => "member",
            Self::HandoffNote => "lesson",
            _ => "context",
        }
    }
    /// Parses a kind tolerantly.
    ///
    /// The comparison is case-insensitive and treats `-` and spaces as `_`; common
    /// short aliases are accepted. An empty string yields
    /// [`TeamMemoryKind::SharedFact`], and anything unrecognized is preserved verbatim
    /// as [`TeamMemoryKind::Custom`], so parsing never fails.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Self::SharedFact;
        }
        let normalized = trimmed.to_ascii_lowercase().replace(['-', ' '], "_");
        match normalized.as_str() {
            "shared_fact" | "shared" | "fact" | "context" => Self::SharedFact,
            "member_profile" | "member" | "profile" => Self::MemberProfile,
            "decision" | "decisions" => Self::Decision,
            "task" | "tasks" | "todo" => Self::Task,
            "artifact" | "artifacts" => Self::Artifact,
            "handoff_note" | "handoff" | "lesson" | "lessons" => Self::HandoffNote,
            _ => Self::Custom(trimmed.to_owned()),
        }
    }
}
impl fmt::Display for TeamMemoryKind {
    /// Writes [`TeamMemoryKind::as_str`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl FromStr for TeamMemoryKind {
    /// Unused: [`TeamMemoryKind::from_str`] is total and maps unknown names to
    /// [`TeamMemoryKind::Custom`]. The associated type exists for interface symmetry.
    type Err = Error;
    /// Parses a kind name; unknown names become [`TeamMemoryKind::Custom`], never an
    /// error.
    fn from_str(raw: &str) -> Result<Self> {
        Ok(Self::parse(raw))
    }
}
impl Serialize for TeamMemoryKind {
    /// Serializes [`TeamMemoryKind::as_str`] as a plain string, so an unknown kind
    /// round-trips through [`TeamMemoryKind::Custom`].
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for TeamMemoryKind {
    /// Deserializes from a kind string through [`TeamMemoryKind::parse`].
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}
/// Who may read a team memory record.
///
/// [`TeamVisibility::Team`] (the default) is readable by every member,
/// [`TeamVisibility::Private`] only by its author, and [`TeamVisibility::Members`] by
/// the listed members (and its author). A team-wide/leader scope ignores visibility.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamVisibility {
    /// Readable by every member of the team; the serde default.
    #[default]
    Team,
    /// Readable only by the author.
    Private,
    /// Readable by the listed members and the author; JSON `{"members": [...]}`.
    Members(Vec<String>),
}
impl TeamVisibility {
    /// Builds a member list from unnormalized ids (trimmed, empties dropped,
    /// duplicates removed).
    #[must_use]
    pub fn members<I, S>(members: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Members(normalize_members(
            members.into_iter().map(Into::into).collect(),
        ))
    }
    /// The member list of a [`TeamVisibility::Members`] visibility; empty for the
    /// other variants.
    #[must_use]
    pub fn member_list(&self) -> &[String] {
        match self {
            Self::Members(members) => members,
            Self::Team | Self::Private => &[],
        }
    }
    /// Validates a [`TeamVisibility::Members`] list: non-empty, at most
    /// [`MAX_VISIBILITY_MEMBERS`] members, each a valid id.
    pub fn validate(&self) -> Result<()> {
        if let Self::Members(members) = self {
            if members.is_empty() {
                return Err(Error::validation(
                    "team memory `visibility` must name at least one member",
                ));
            }
            if members.len() > MAX_VISIBILITY_MEMBERS {
                return Err(Error::validation(format!(
                    "team memory `visibility` must name at most {MAX_VISIBILITY_MEMBERS} members (got {})",
                    members.len()
                )));
            }
            for member in members {
                validate_component("visibility member", member, MAX_SCOPE_ID_CHARS)?;
            }
        }
        Ok(())
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn kind_serializes_as_a_plain_string() {
        assert_eq!(
            serde_json::to_value(TeamMemoryKind::SharedFact).unwrap(),
            serde_json::json!("shared_fact")
        );
        assert_eq!(
            serde_json::to_value(TeamMemoryKind::HandoffNote).unwrap(),
            serde_json::json!("handoff_note")
        );
        assert_eq!(
            serde_json::to_value(TeamMemoryKind::Custom("review_rule".to_owned())).unwrap(),
            serde_json::json!("review_rule")
        );
        for kind in TeamMemoryKind::BUILTIN {
            let json = serde_json::to_value(&kind).unwrap();
            let restored: TeamMemoryKind = serde_json::from_value(json).unwrap();
            assert_eq!(restored, kind);
        }
    }
    #[test]
    fn unknown_kind_strings_become_custom() {
        let kind: TeamMemoryKind = serde_json::from_value(serde_json::json!("novel_kind")).unwrap();
        assert_eq!(kind, TeamMemoryKind::Custom("novel_kind".to_owned()));
        let kind: TeamMemoryKind =
            serde_json::from_value(serde_json::json!(" review-rule ")).unwrap();
        assert_eq!(kind, TeamMemoryKind::Custom("review-rule".to_owned()));
        assert!(serde_json::from_value::<TeamMemoryKind>(serde_json::json!(7)).is_err());
    }
    #[test]
    fn parse_is_tolerant_and_total() {
        assert_eq!(TeamMemoryKind::parse(""), TeamMemoryKind::SharedFact);
        assert_eq!(TeamMemoryKind::parse("  "), TeamMemoryKind::SharedFact);
        assert_eq!(TeamMemoryKind::parse("shared"), TeamMemoryKind::SharedFact);
        assert_eq!(TeamMemoryKind::parse("context"), TeamMemoryKind::SharedFact);
        assert_eq!(
            TeamMemoryKind::parse("Member-Profile"),
            TeamMemoryKind::MemberProfile
        );
        assert_eq!(
            TeamMemoryKind::parse("profile"),
            TeamMemoryKind::MemberProfile
        );
        assert_eq!(TeamMemoryKind::parse("TODO"), TeamMemoryKind::Task);
        assert_eq!(
            TeamMemoryKind::parse("handoff"),
            TeamMemoryKind::HandoffNote
        );
        assert_eq!(TeamMemoryKind::parse("lesson"), TeamMemoryKind::HandoffNote);
        assert_eq!(
            TeamMemoryKind::parse("Something Else"),
            TeamMemoryKind::Custom("Something Else".to_owned())
        );
    }
    #[test]
    fn from_str_and_display_round_trip() {
        for kind in TeamMemoryKind::BUILTIN {
            assert_eq!(kind.as_str().parse::<TeamMemoryKind>().unwrap(), kind);
            assert_eq!(kind.to_string(), kind.as_str());
        }
        let custom = TeamMemoryKind::Custom("review_rule".to_owned());
        assert_eq!(custom.to_string(), "review_rule");
        assert_eq!("review_rule".parse::<TeamMemoryKind>().unwrap(), custom);
        assert_eq!(
            "bogus".parse::<TeamMemoryKind>().unwrap(),
            TeamMemoryKind::Custom("bogus".to_owned())
        );
    }
    #[test]
    fn builtin_membership_and_extractor_tags() {
        assert_eq!(TeamMemoryKind::BUILTIN.len(), 6);
        for kind in TeamMemoryKind::BUILTIN {
            assert!(kind.is_builtin());
        }
        let custom = TeamMemoryKind::Custom("review_rule".to_owned());
        assert!(!custom.is_builtin());
        assert_eq!(TeamMemoryKind::default(), TeamMemoryKind::SharedFact);
        assert_eq!(TeamMemoryKind::Decision.extractor_tag(), "decision");
        assert_eq!(TeamMemoryKind::MemberProfile.extractor_tag(), "member");
        assert_eq!(TeamMemoryKind::HandoffNote.extractor_tag(), "lesson");
        assert_eq!(TeamMemoryKind::SharedFact.extractor_tag(), "context");
        assert_eq!(TeamMemoryKind::Task.extractor_tag(), "context");
        assert_eq!(TeamMemoryKind::Artifact.extractor_tag(), "context");
        assert_eq!(custom.extractor_tag(), "context");
    }
    #[test]
    fn visibility_json_shape_and_round_trip() {
        assert_eq!(
            serde_json::to_value(TeamVisibility::Team).unwrap(),
            serde_json::json!("team")
        );
        assert_eq!(
            serde_json::to_value(TeamVisibility::Private).unwrap(),
            serde_json::json!("private")
        );
        let members = TeamVisibility::members(["alice", "bob"]);
        assert_eq!(
            serde_json::to_value(&members).unwrap(),
            serde_json::json!({"members": ["alice", "bob"]})
        );
        let restored: TeamVisibility = serde_json::from_value(serde_json::json!({
            "members": ["alice", "bob"]
        }))
        .unwrap();
        assert_eq!(restored, members);
        assert_eq!(TeamVisibility::default(), TeamVisibility::Team);
        assert!(serde_json::from_value::<TeamVisibility>(serde_json::json!("shared")).is_err());
    }
    #[test]
    fn visibility_members_are_normalized() {
        let members = TeamVisibility::members([" alice ", "", "alice", "bob"]);
        assert_eq!(
            members,
            TeamVisibility::Members(vec!["alice".to_owned(), "bob".to_owned()])
        );
        assert_eq!(members.member_list(), ["alice", "bob"]);
        assert_eq!(TeamVisibility::Team.member_list().len(), 0);
        assert_eq!(TeamVisibility::Private.member_list().len(), 0);
    }
    #[test]
    fn visibility_validation() {
        assert!(TeamVisibility::Team.validate().is_ok());
        assert!(TeamVisibility::Private.validate().is_ok());
        assert!(TeamVisibility::Members(Vec::new())
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one member"));
        assert!(TeamVisibility::members(["alice"]).validate().is_ok());
        let too_many = vec!["alice".to_owned(); MAX_VISIBILITY_MEMBERS + 1];
        assert!(TeamVisibility::Members(too_many)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at most"));
        assert!(TeamVisibility::Members(vec!["a/b".to_owned()])
            .validate()
            .unwrap_err()
            .to_string()
            .contains("visibility member"));
    }
}
