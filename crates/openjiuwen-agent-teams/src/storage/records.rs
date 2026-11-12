//! Persisted row types of the team store.
//!
//! These are the shapes a [`crate::storage::TeamStore`] persists: the team row, the roster row, the
//! roster patch and the mailbox row. The task rows (`TaskRecord`, `TaskPatch`, `TaskFilter`,
//! `ReviewVote`) live in [`crate::schema::task`] because the task board codes against them; the
//! statuses live in [`crate::schema::status`].
//!
//! Python parity source: `openjiuwen/agent_teams/tools/database/{team_dao,member_dao,message_dao}.py`
//! (`TeamInfo`, `TeamMember`, `TeamMessage`). Every type is `serde` `snake_case`, tolerates unknown
//! fields and defaults every collection, so a row written by an older build still decodes.

use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_LEADER_MEMBER_NAME;
use crate::schema::{DispatchMode, ExecutionStatus, MemberMode, MemberStatus, TeamRole};

/// Wire protocol name of a plain-text team message; the Python `TeamMessage.protocol` default.
pub const PLAIN_MESSAGE_PROTOCOL: &str = "plain";

/// One row of the team table.
///
/// Python parity: `TeamInfo`. It carries the leader-facing description and prompt the leader
/// context is rebuilt from on `resume`, so it is persisted rather than derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamInfoRecord {
    /// Team identity; the key of the row and of every member/task/message row.
    pub team_name: String,
    /// Human-facing team name; falls back to [`TeamInfoRecord::team_name`] at render time.
    pub display_name: String,
    /// Member name of the leader; default `team_leader`.
    pub leader_member_name: String,
    /// Team description shown in the leader prompt.
    pub desc: String,
    /// Leader prompt persisted with the team.
    pub prompt: String,
    /// Who hands work to the members; default `autonomous`.
    pub dispatch_mode: DispatchMode,
    /// Whether an assigned task must pass the review gate before it completes.
    pub enable_task_verification: bool,
    /// Millisecond wall-clock creation stamp; `0` until the store stamps it.
    pub created: u64,
    /// Millisecond wall-clock stamp of the last write; `0` until the store stamps it.
    pub updated_at: u64,
}

impl Default for TeamInfoRecord {
    fn default() -> Self {
        Self {
            team_name: String::new(),
            display_name: String::new(),
            leader_member_name: DEFAULT_LEADER_MEMBER_NAME.to_owned(),
            desc: String::new(),
            prompt: String::new(),
            dispatch_mode: DispatchMode::default(),
            enable_task_verification: false,
            created: 0,
            updated_at: 0,
        }
    }
}

impl TeamInfoRecord {
    /// Builds a team row for `team_name` with the Python defaults.
    #[must_use]
    pub fn new(team_name: impl Into<String>) -> Self {
        Self {
            team_name: team_name.into(),
            ..Self::default()
        }
    }

    /// Returns the display name to render: [`Self::display_name`], else [`Self::team_name`].
    #[must_use]
    pub fn display_name_or_default(&self) -> &str {
        if self.display_name.is_empty() {
            &self.team_name
        } else {
            &self.display_name
        }
    }
}

/// One row of the roster table.
///
/// Python parity: `TeamMember`. It holds both the lifecycle status
/// ([`MemberRecord::status`]) and the per-task execution status
/// ([`MemberRecord::execution`]) because the leader renders both in one row.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberRecord {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member identity, unique within the team.
    pub member_name: String,
    /// Human-facing member name.
    pub display_name: String,
    /// Member description shown in the leader prompt.
    pub desc: String,
    /// Roster role; default `leader`.
    pub role: TeamRole,
    /// Lifecycle status; default `unstarted`.
    pub status: MemberStatus,
    /// Detail status of the member's current task execution; default `idle`.
    pub execution: ExecutionStatus,
    /// Tool and prompt set of a non-leader member; default `build_mode`.
    pub mode: MemberMode,
    /// System prompt the member harness runs with.
    pub prompt: String,
    /// Model reference pinned for this member, when the team allocated one from the pool.
    pub model_ref: Option<String>,
    /// Member workspace path or worktree id, when a workspace was allocated.
    pub worktree: Option<String>,
    /// Millisecond wall-clock stamp of the last write; `0` until the store stamps it.
    pub updated_at: u64,
}

impl MemberRecord {
    /// Builds an `unstarted`, `idle` member row for `member_name` in `team_name`.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        member_name: impl Into<String>,
        role: TeamRole,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            member_name: member_name.into(),
            role,
            ..Self::default()
        }
    }

    /// Returns the display name to render: [`Self::display_name`], else [`Self::member_name`].
    #[must_use]
    pub fn display_name_or_default(&self) -> &str {
        if self.display_name.is_empty() {
            &self.member_name
        } else {
            &self.display_name
        }
    }
}

/// A partial update of a [`MemberRecord`].
///
/// Every field is optional and `None` means "leave unchanged". Releasing `model_ref`/`worktree`
/// cannot be expressed by the `Option` alone, so it takes the explicit `clear_*` flags; the flags
/// win over the corresponding setter when both are present, because clearing is the deliberate
/// action. Builders consume and return the patch so a call site reads as one expression.
///
/// A patch never validates status legality: only
/// [`TeamStore::transition_member`](crate::storage::TeamStore::transition_member) does, because it
/// is the compare-and-set path every lifecycle change goes through.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberPatch {
    /// New display name.
    pub display_name: Option<String>,
    /// New description.
    pub desc: Option<String>,
    /// New role.
    pub role: Option<TeamRole>,
    /// New lifecycle status, applied verbatim.
    pub status: Option<MemberStatus>,
    /// New execution status.
    pub execution: Option<ExecutionStatus>,
    /// New tool/prompt mode.
    pub mode: Option<MemberMode>,
    /// New system prompt.
    pub prompt: Option<String>,
    /// New model reference; use [`MemberPatch::clearing_model`] to release it instead.
    pub model_ref: Option<String>,
    /// When `true`, the stored model reference is set to `null`.
    pub clear_model: bool,
    /// New workspace path or worktree id; use [`MemberPatch::clearing_worktree`] instead.
    pub worktree: Option<String>,
    /// When `true`, the stored workspace reference is set to `null`.
    pub clear_worktree: bool,
}

impl MemberPatch {
    /// Returns an empty patch: nothing is changed.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            display_name: None,
            desc: None,
            role: None,
            status: None,
            execution: None,
            mode: None,
            prompt: None,
            model_ref: None,
            clear_model: false,
            worktree: None,
            clear_worktree: false,
        }
    }

    /// Returns `true` when the patch sets nothing, so it can be skipped entirely.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.display_name.is_none()
            && self.desc.is_none()
            && self.role.is_none()
            && self.status.is_none()
            && self.execution.is_none()
            && self.mode.is_none()
            && self.prompt.is_none()
            && self.model_ref.is_none()
            && !self.clear_model
            && self.worktree.is_none()
            && !self.clear_worktree
    }

    /// Sets the new display name.
    #[must_use]
    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    /// Sets the new description.
    #[must_use]
    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.desc = Some(desc.into());
        self
    }

    /// Sets the new role.
    #[must_use]
    pub const fn with_role(mut self, role: TeamRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Sets the new lifecycle status.
    #[must_use]
    pub const fn with_status(mut self, status: MemberStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Sets the new execution status.
    #[must_use]
    pub const fn with_execution(mut self, execution: ExecutionStatus) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Sets the new tool/prompt mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: MemberMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the new system prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the new model reference.
    #[must_use]
    pub fn with_model_ref(mut self, model_ref: impl Into<String>) -> Self {
        self.model_ref = Some(model_ref.into());
        self
    }

    /// Releases the stored model reference.
    #[must_use]
    pub const fn clearing_model(mut self) -> Self {
        self.clear_model = true;
        self
    }

    /// Sets the new workspace reference.
    #[must_use]
    pub fn with_worktree(mut self, worktree: impl Into<String>) -> Self {
        self.worktree = Some(worktree.into());
        self
    }

    /// Releases the stored workspace reference.
    #[must_use]
    pub const fn clearing_worktree(mut self) -> Self {
        self.clear_worktree = true;
        self
    }

    /// Applies the patch to `record` and returns `true` when the row actually changed.
    ///
    /// The `clear_*` flags win over their setter; every other field is applied only when present.
    pub fn apply(&self, record: &mut MemberRecord) -> bool {
        let before = record.clone();
        if let Some(display_name) = &self.display_name {
            record.display_name.clone_from(display_name);
        }
        if let Some(desc) = &self.desc {
            record.desc.clone_from(desc);
        }
        if let Some(role) = self.role {
            record.role = role;
        }
        if let Some(status) = self.status {
            record.status = status;
        }
        if let Some(execution) = self.execution {
            record.execution = execution;
        }
        if let Some(mode) = self.mode {
            record.mode = mode;
        }
        if let Some(prompt) = &self.prompt {
            record.prompt.clone_from(prompt);
        }
        if self.clear_model {
            record.model_ref = None;
        } else if let Some(model_ref) = &self.model_ref {
            record.model_ref = Some(model_ref.clone());
        }
        if self.clear_worktree {
            record.worktree = None;
        } else if let Some(worktree) = &self.worktree {
            record.worktree = Some(worktree.clone());
        }
        *record != before
    }
}

/// One row of the mailbox table.
///
/// Python parity: `TeamMessage`. A row is either a direct message (one recipient, `is_read` is the
/// per-row flag) or a broadcast (`broadcast == true`, `to_member_name` unset, and read state is the
/// per-member watermark kept by the store — never a row flag).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MessageRecord {
    /// Team the message belongs to.
    pub team_name: String,
    /// Message identity, unique within the team.
    pub message_id: String,
    /// Sender member name.
    pub from_member_name: String,
    /// Recipient member name; `None` for a broadcast.
    pub to_member_name: Option<String>,
    /// Message body.
    pub content: String,
    /// Millisecond wall-clock send stamp, used both for ordering and as the broadcast read cursor.
    pub timestamp: u64,
    /// `true` when the message addresses every member rather than [`Self::to_member_name`].
    pub broadcast: bool,
    /// Content protocol; default `plain`.
    pub protocol: String,
    /// Per-row read flag of a direct message; always `false` for a broadcast row.
    pub is_read: bool,
}

impl Default for MessageRecord {
    fn default() -> Self {
        Self {
            team_name: String::new(),
            message_id: String::new(),
            from_member_name: String::new(),
            to_member_name: None,
            content: String::new(),
            timestamp: 0,
            broadcast: false,
            protocol: PLAIN_MESSAGE_PROTOCOL.to_owned(),
            is_read: false,
        }
    }
}

impl MessageRecord {
    /// Builds an unread direct message.
    #[must_use]
    pub fn direct(
        team_name: impl Into<String>,
        message_id: impl Into<String>,
        from_member_name: impl Into<String>,
        to_member_name: impl Into<String>,
        content: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            message_id: message_id.into(),
            from_member_name: from_member_name.into(),
            to_member_name: Some(to_member_name.into()),
            content: content.into(),
            timestamp,
            ..Self::default()
        }
    }

    /// Builds an unread broadcast message.
    #[must_use]
    pub fn broadcast(
        team_name: impl Into<String>,
        message_id: impl Into<String>,
        from_member_name: impl Into<String>,
        content: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            message_id: message_id.into(),
            from_member_name: from_member_name.into(),
            content: content.into(),
            timestamp,
            broadcast: true,
            ..Self::default()
        }
    }

    /// Returns `true` when this row is a broadcast (read state comes from the watermark).
    #[must_use]
    pub const fn is_broadcast(&self) -> bool {
        self.broadcast
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn team_record_wire_shape_and_defaults() {
        let record = TeamInfoRecord::new("alpha");
        assert_eq!(record.leader_member_name, "team_leader");
        assert_eq!(record.dispatch_mode, DispatchMode::Autonomous);
        assert!(!record.enable_task_verification);
        assert_eq!(record.display_name_or_default(), "alpha");

        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["team_name"], "alpha");
        assert_eq!(json["leader_member_name"], "team_leader");
        assert_eq!(json["dispatch_mode"], "autonomous");
        assert_eq!(json["enable_task_verification"], false);
        assert_eq!(json["created"], 0);
        assert_eq!(json["updated_at"], 0);
    }

    #[test]
    fn team_record_tolerates_unknown_fields_and_missing_fields() {
        let decoded: TeamInfoRecord =
            serde_json::from_str(r#"{"team_name": "t", "unknown": 1}"#).unwrap();
        assert_eq!(decoded.team_name, "t");
        assert_eq!(decoded.leader_member_name, "team_leader");
        assert_eq!(decoded.desc, "");

        let decoded: TeamInfoRecord = serde_json::from_str("{}").unwrap();
        assert_eq!(decoded, TeamInfoRecord::default());
        assert_eq!(decoded.leader_member_name, "team_leader");
    }

    #[test]
    fn member_record_wire_shape_and_defaults() {
        let record = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
        assert_eq!(record.status, MemberStatus::Unstarted);
        assert_eq!(record.execution, ExecutionStatus::Idle);
        assert_eq!(record.mode, MemberMode::BuildMode);
        assert_eq!(record.model_ref, None);
        assert_eq!(record.worktree, None);
        assert_eq!(record.display_name_or_default(), "dev-1");

        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["role"], "teammate");
        assert_eq!(json["status"], "unstarted");
        assert_eq!(json["execution"], "idle");
        assert_eq!(json["mode"], "build_mode");
        assert_eq!(json["model_ref"], serde_json::Value::Null);
        assert_eq!(json["worktree"], serde_json::Value::Null);
    }

    #[test]
    fn member_record_tolerates_unknown_fields() {
        let decoded: MemberRecord = serde_json::from_str(
            r#"{"team_name":"t","member_name":"m","status":"busy","legacy_field":[1,2]}"#,
        )
        .unwrap();
        assert_eq!(decoded.member_name, "m");
        assert_eq!(decoded.status, MemberStatus::Busy);
        assert_eq!(decoded.role, TeamRole::Leader);
    }

    #[test]
    fn member_patch_merges_option_fields() {
        let mut record = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
        record.model_ref = Some("model-a".to_owned());
        record.worktree = Some("/ws/dev-1".to_owned());

        let patch = MemberPatch::new()
            .with_status(MemberStatus::Ready)
            .with_execution(ExecutionStatus::Running)
            .with_model_ref("model-b");
        assert!(patch.apply(&mut record));
        assert_eq!(record.status, MemberStatus::Ready);
        assert_eq!(record.execution, ExecutionStatus::Running);
        assert_eq!(record.model_ref.as_deref(), Some("model-b"));
        // Untouched fields survive.
        assert_eq!(record.worktree.as_deref(), Some("/ws/dev-1"));
        assert_eq!(record.role, TeamRole::Teammate);

        // An identical patch is a no-op.
        assert!(!patch.apply(&mut record));
    }

    #[test]
    fn member_patch_clears_beat_setters_and_empty_patch_is_empty() {
        let mut record = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
        record.model_ref = Some("model-a".to_owned());
        record.worktree = Some("/ws/dev-1".to_owned());

        let patch = MemberPatch::new()
            .with_model_ref("model-b")
            .clearing_model()
            .with_worktree("/ws/other")
            .clearing_worktree();
        assert!(patch.apply(&mut record));
        assert_eq!(record.model_ref, None);
        assert_eq!(record.worktree, None);

        assert!(MemberPatch::new().is_empty());
        assert!(!MemberPatch::new().clearing_model().is_empty());
        assert!(!MemberPatch::default().with_prompt("p").is_empty());
    }

    #[test]
    fn member_patch_wire_shape() {
        let json = serde_json::to_value(MemberPatch::default()).unwrap();
        assert_eq!(json["clear_model"], false);
        assert_eq!(json["clear_worktree"], false);
        assert_eq!(json["display_name"], serde_json::Value::Null);

        let decoded: MemberPatch =
            serde_json::from_str(r#"{"status":"paused","clear_worktree":true,"junk":1}"#).unwrap();
        assert_eq!(decoded.status, Some(MemberStatus::Paused));
        assert!(decoded.clear_worktree);
        assert_eq!(decoded.mode, None);
    }

    #[test]
    fn message_record_wire_shape_and_constructors() {
        let direct = MessageRecord::direct("alpha", "m1", "leader", "dev-1", "do it", 7);
        assert_eq!(direct.to_member_name.as_deref(), Some("dev-1"));
        assert!(!direct.is_broadcast());
        assert_eq!(direct.protocol, "plain");
        assert!(!direct.is_read);

        let bcast = MessageRecord::broadcast("alpha", "m2", "leader", "standup in 5", 8);
        assert!(bcast.is_broadcast());
        assert_eq!(bcast.to_member_name, None);

        let json = serde_json::to_value(&direct).unwrap();
        assert_eq!(json["message_id"], "m1");
        assert_eq!(json["to_member_name"], "dev-1");
        assert_eq!(json["broadcast"], false);
        assert_eq!(json["protocol"], "plain");
        assert_eq!(json["is_read"], false);
    }

    #[test]
    fn message_record_defaults_protocol_and_tolerates_unknown_fields() {
        let decoded: MessageRecord = serde_json::from_str(
            r#"{"team_name":"t","message_id":"m","from_member_name":"a","timestamp":3}"#,
        )
        .unwrap();
        assert_eq!(decoded.protocol, "plain");
        assert!(!decoded.broadcast);
        assert!(!decoded.is_read);
        assert_eq!(
            decoded,
            MessageRecord {
                team_name: "t".to_owned(),
                message_id: "m".to_owned(),
                from_member_name: "a".to_owned(),
                timestamp: 3,
                ..MessageRecord::default()
            }
        );
    }
}
