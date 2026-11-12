//! Team-scoped runtime types: the member roles a team addresses and the runtime mirrors of a team
//! spec.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/team.py` (`TeamRole`, `TeamSpec`,
//! `TeamRuntimeContext`); [`TeamSnapshot`] is the Rust-only projection of the same runtime state.
//!
//! [`TeamRole`] is the vocabulary shared by the roster specs in [`super::member_spec`] and by
//! runtime dispatch. [`TeamSpec`] is the runtime mirror of [`TeamAgentSpec`] that every
//! [`TeamRuntimeContext`] carries, so a path which only sees a member's context (external CLI
//! spawn, the skill CLI, the MCP server) resolves the same dispatch knobs, model pool and
//! workspace shape as an in-process member. [`TeamSnapshot`] is the Rust-only projection the
//! runtime manager returns when a caller asks for the current state of a team.
//!
//! Deliberate deviation from Python (documented in `src/schema/CONTEXT.md`):
//! [`TeamSpec::evolution_enabled`] defaults to `false`, because this port ships no self-evolution
//! coverage.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use openjiuwen_harness_protocol::HarnessState;

use super::deep_agent::WorkspaceSpec;
use super::member_spec::ModelPoolEntry;
use super::spec::{ConversationMode, DispatchMode, MemberMode, ModelPoolStrategy, TeamAgentSpec};
use super::status::{ExecutionStatus, MemberStatus};
use super::task::TaskSummary;
use crate::constants::{DEFAULT_LANGUAGE, DEFAULT_TEAM_NAME};

/// The role a team member plays.
///
/// Wire values are the Python `TeamRole` strings, so a role persisted by the Python runtime loads
/// unchanged. [`TeamRole::Leader`] is the default (Python's `TeamRuntimeContext` defaults to the
/// leader role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamRole {
    /// Drives the team: dispatches tasks, reviews submissions and talks to the user.
    Leader,
    /// Autonomous member that polls the mailbox and claims tasks from the board.
    Teammate,
    /// Avatar channel for a real human joining the team.
    HumanAgent,
    /// Directed human who is addressed but never polls the board.
    PassiveHuman,
    /// Local teammate paired with a remote agent reachable over a text protocol.
    BridgeAgent,
    /// Organisation-layer worker.
    Worker,
    /// Member whose work is performed by an external CLI process.
    ExternalCli,
}

impl Default for TeamRole {
    /// Python default: [`TeamRole::Leader`].
    fn default() -> Self {
        Self::Leader
    }
}

impl TeamRole {
    /// Returns the Python wire string of this role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Leader => "leader",
            Self::Teammate => "teammate",
            Self::HumanAgent => "human_agent",
            Self::PassiveHuman => "passive_human",
            Self::BridgeAgent => "bridge_agent",
            Self::Worker => "worker",
            Self::ExternalCli => "external_cli",
        }
    }

    /// Returns `true` for roles that participate in the coordination loop as autonomous members.
    ///
    /// Python: `TeamRole.is_coordinated_member` — teammates and external CLI members both poll the
    /// mailbox, claim tasks and self-nudge, so gates should ask this instead of comparing against
    /// [`TeamRole::Teammate`].
    #[must_use]
    pub const fn is_coordinated_member(self) -> bool {
        matches!(self, Self::Teammate | Self::ExternalCli)
    }
}

/// The runtime mirror of [`TeamAgentSpec`] that a [`TeamRuntimeContext`] carries.
///
/// Python parity: `TeamSpec`. It repeats the knobs a member must resolve without seeing the
/// blueprint — dispatch mode, peer messaging, subtask depth, teammate mode, the model pool and the
/// shared workspace — because external CLI members and MCP tool sets are assembled from the
/// runtime context alone. [`TeamSpec::from_spec`] materializes it.
///
/// [`Self::evolution_enabled`] is a Rust deviation: Python defaults it to `true`, this port
/// defaults it to `false` because no self-evolution coverage is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamSpec {
    /// Team name, copied from [`TeamAgentSpec::team_name`].
    pub team_name: String,
    /// Human-readable team label: [`TeamAgentSpec::team_display_name`], else the team name.
    pub display_name: String,
    /// How members exchange messages.
    pub conversation_mode: ConversationMode,
    /// Member name of the leader, when the runtime knows it.
    pub leader_member_name: Option<String>,
    /// Language of prompts and generated text; `None` falls back to the default in
    /// [`TeamRuntimeContext::new`].
    pub language: Option<String>,
    /// Who hands work to members — mirrors [`TeamAgentSpec::dispatch_mode`].
    pub dispatch_mode: DispatchMode,
    /// Whether scheduled-dispatch members may message each other — mirrors
    /// [`TeamAgentSpec::enable_member_messaging`].
    pub enable_member_messaging: bool,
    /// Member subtask ceiling — mirrors [`TeamAgentSpec::max_subtask_depth`].
    pub max_subtask_depth: usize,
    /// Tool and prompt set non-leader members run with — mirrors
    /// [`TeamAgentSpec::teammate_mode`].
    pub teammate_mode: MemberMode,
    /// LLM endpoints members may be allocated from — mirrors [`TeamAgentSpec::model_pool`].
    pub model_pool: Vec<ModelPoolEntry>,
    /// How members are assigned an endpoint from [`Self::model_pool`].
    pub model_pool_strategy: ModelPoolStrategy,
    /// Shared workspace of the team, materialized only when the spec pins a root path.
    pub workspace: Option<WorkspaceSpec>,
    /// Self-evolution coverage switch; `false` in this port.
    pub evolution_enabled: bool,
    /// Whether member directories are prefixed by the member name; `true` by default.
    pub member_workspace_prefix: bool,
    /// Free-form extension payload, never inspected by this crate.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for TeamSpec {
    /// Defaults of a built team that names nothing: a temporary, leader-mediated autonomous team
    /// named `agent_team`, with an empty pool and no shared workspace.
    fn default() -> Self {
        Self {
            team_name: DEFAULT_TEAM_NAME.to_owned(),
            // Python requires `display_name`; `build()` passes the team name, and so does this
            // default so a defaulted spec never renders an empty label.
            display_name: DEFAULT_TEAM_NAME.to_owned(),
            conversation_mode: ConversationMode::default(),
            leader_member_name: None,
            language: None,
            dispatch_mode: DispatchMode::default(),
            enable_member_messaging: false,
            max_subtask_depth: 0,
            teammate_mode: MemberMode::default(),
            model_pool: Vec::new(),
            model_pool_strategy: ModelPoolStrategy::default(),
            workspace: None,
            evolution_enabled: false,
            member_workspace_prefix: true,
            metadata: BTreeMap::new(),
        }
    }
}

impl TeamSpec {
    /// Materializes the runtime mirror of a team blueprint.
    ///
    /// Python parity: the `TeamSpec(...)` construction inside `TeamAgentSpec.build()`. The
    /// display label is the spec's `team_display_name` when it has one and the team name
    /// otherwise; the shared workspace is mirrored only when the spec pins
    /// [`TeamWorkspaceSpec::root_path`](super::spec::TeamWorkspaceSpec::root_path), and then it
    /// inherits the spec's resolved language with `stable_base: false`.
    ///
    /// Unlike Python's `build()`, this takes `&self` and never mutates the spec.
    #[must_use]
    pub fn from_spec(spec: &TeamAgentSpec) -> Self {
        Self {
            team_name: spec.team_name.clone(),
            display_name: spec
                .team_display_name
                .clone()
                .unwrap_or_else(|| spec.team_name.clone()),
            conversation_mode: spec.conversation_mode,
            leader_member_name: Some(spec.leader_member_name().to_owned()),
            language: spec.language.clone(),
            dispatch_mode: spec.dispatch_mode,
            enable_member_messaging: spec.enable_member_messaging,
            max_subtask_depth: spec.max_subtask_depth,
            teammate_mode: spec.teammate_mode,
            model_pool: spec.model_pool.clone(),
            model_pool_strategy: spec.model_pool_strategy,
            workspace: spec
                .workspace
                .root_path
                .clone()
                .map(|root_path| WorkspaceSpec {
                    root_path,
                    language: spec.resolved_language().to_owned(),
                    stable_base: false,
                }),
            evolution_enabled: false,
            member_workspace_prefix: true,
            metadata: spec.metadata.clone(),
        }
    }
}

/// The runtime context of one team member.
///
/// Python parity: `TeamRuntimeContext`. Every identity field is stored directly (no nested
/// roster row) so the spawn path can render a member's prompt, roster and workspace from one
/// value: `display_name`/`desc` are the public label and description peers see, while `prompt` is
/// the private working agreement delivered only to this member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamRuntimeContext {
    /// Role this member plays in the team.
    pub role: TeamRole,
    /// Wire name of the member; `None` before the member is registered.
    pub member_name: Option<String>,
    /// Human-readable member label (`team_member.display_name`).
    pub display_name: String,
    /// Public member description (`team_member.desc`), shared into peers' rosters.
    pub desc: String,
    /// Private, member-only working agreement (`team_member.prompt`).
    pub prompt: String,
    /// Runtime mirror of the team the member belongs to.
    pub team_spec: TeamSpec,
    /// Root directory the member's workspace is allocated under.
    pub workspace_root: String,
    /// Model allocated to this member, when an allocator picked one.
    pub member_model: Option<String>,
    /// Resolved language of prompts and generated text.
    pub language: String,
    /// Agent-group template this member was composed from, when it came from a package.
    pub agent_template_spec: Option<Value>,
}

impl Default for TeamRuntimeContext {
    /// Defaults of a leader context with no identity yet: the default team spec and language.
    fn default() -> Self {
        Self {
            role: TeamRole::default(),
            member_name: None,
            display_name: String::new(),
            desc: String::new(),
            prompt: String::new(),
            team_spec: TeamSpec::default(),
            workspace_root: String::new(),
            member_model: None,
            language: DEFAULT_LANGUAGE.to_owned(),
            agent_template_spec: None,
        }
    }
}

impl TeamRuntimeContext {
    /// Builds a context for `role` in `team_spec`, resolving the language from the team.
    ///
    /// Python parity: `TeamRuntimeContext(role=..., team_spec=...)`. The language is the team's
    /// when it declares one and [`DEFAULT_LANGUAGE`] otherwise; identity, workspace, model and
    /// template fields stay empty until the runtime fills them.
    #[must_use]
    pub fn new(role: TeamRole, team_spec: TeamSpec) -> Self {
        let language = team_spec
            .language
            .clone()
            .unwrap_or_else(|| DEFAULT_LANGUAGE.to_owned());
        Self {
            role,
            team_spec,
            language,
            ..Self::default()
        }
    }
}

/// A point-in-time projection of a team: its state, roster and task board.
///
/// Python has no single equivalent; this is the Rust view the runtime manager and the skill CLI
/// return when a caller asks what the team currently looks like. Both collections are sorted by
/// [`TeamSnapshot::sort`] before they are handed out, so a snapshot diff is stable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamSnapshot {
    /// Name of the team the snapshot was taken from.
    pub team_name: String,
    /// Lifecycle state of the team at that moment.
    pub state: HarnessState,
    /// Roster, sorted by [`MemberSnapshot::member_name`] once sorted.
    pub members: Vec<MemberSnapshot>,
    /// Task board, sorted by [`TaskSummary::task_id`] once sorted.
    pub tasks: Vec<TaskSummary>,
    /// Millisecond wall-clock timestamp of the snapshot.
    pub updated_at: u64,
}

impl Default for TeamSnapshot {
    /// An empty snapshot of an unnamed, idle team at timestamp `0`.
    fn default() -> Self {
        Self {
            team_name: String::new(),
            state: HarnessState::Idle,
            members: Vec::new(),
            tasks: Vec::new(),
            updated_at: 0,
        }
    }
}

impl TeamSnapshot {
    /// Creates an empty snapshot of `team_name` in `state`.
    #[must_use]
    pub fn new(team_name: impl Into<String>, state: HarnessState) -> Self {
        Self {
            team_name: team_name.into(),
            state,
            ..Self::default()
        }
    }

    /// Sorts the roster by member name and the task board by task id.
    ///
    /// The store and the event bus do not guarantee an order, so the projection is canonicalized
    /// here instead of at every call site.
    pub fn sort(&mut self) {
        self.members
            .sort_by(|left, right| left.member_name.cmp(&right.member_name));
        self.tasks
            .sort_by(|left, right| left.task_id.cmp(&right.task_id));
    }
}

/// One roster row of a [`TeamSnapshot`].
///
/// Python parity: the leader's `list_members` projection — the identity a peer sees plus the
/// member's lifecycle and execution status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSnapshot {
    /// Wire name of the member.
    pub member_name: String,
    /// Human-readable label peers see.
    pub display_name: String,
    /// Role the member plays.
    pub role: TeamRole,
    /// Lifecycle status of the member.
    pub status: MemberStatus,
    /// What the member is currently executing.
    pub execution: ExecutionStatus,
}

impl Default for MemberSnapshot {
    /// Defaults of an unstarted leader row with no identity yet.
    fn default() -> Self {
        Self {
            member_name: String::new(),
            display_name: String::new(),
            role: TeamRole::default(),
            status: MemberStatus::default(),
            execution: ExecutionStatus::default(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Every role paired with its pinned wire string.
    const ROLE_WIRE_VALUES: [(TeamRole, &str); 7] = [
        (TeamRole::Leader, "leader"),
        (TeamRole::Teammate, "teammate"),
        (TeamRole::HumanAgent, "human_agent"),
        (TeamRole::PassiveHuman, "passive_human"),
        (TeamRole::BridgeAgent, "bridge_agent"),
        (TeamRole::Worker, "worker"),
        (TeamRole::ExternalCli, "external_cli"),
    ];

    #[test]
    fn team_role_wire_values_match_python() {
        for (role, wire) in ROLE_WIRE_VALUES {
            assert_eq!(role.as_str(), wire, "{role:?}");
            assert_eq!(serde_json::to_value(role).unwrap(), json!(wire), "{role:?}");
            assert_eq!(
                serde_json::from_value::<TeamRole>(json!(wire)).unwrap(),
                role,
                "{wire}"
            );
        }
    }

    #[test]
    fn team_role_default_is_leader() {
        assert_eq!(TeamRole::default(), TeamRole::Leader);
        let parsed: TeamRole = serde_json::from_value(json!("leader")).unwrap();
        assert_eq!(parsed, TeamRole::default());
    }

    #[test]
    fn coordinated_members_are_teammates_and_external_cli() {
        for (role, _) in ROLE_WIRE_VALUES {
            assert_eq!(
                role.is_coordinated_member(),
                matches!(role, TeamRole::Teammate | TeamRole::ExternalCli),
                "{role:?}"
            );
        }
    }

    #[test]
    fn unknown_role_is_rejected() {
        assert!(serde_json::from_value::<TeamRole>(json!("supervisor")).is_err());
    }

    /// A blueprint with every field [`TeamSpec::from_spec`] maps explicitly set.
    fn full_blueprint() -> TeamAgentSpec {
        TeamAgentSpec {
            team_name: "review_team".to_owned(),
            team_display_name: Some("Review Team".to_owned()),
            language: Some("en".to_owned()),
            dispatch_mode: DispatchMode::Scheduled,
            enable_member_messaging: true,
            max_subtask_depth: 2,
            teammate_mode: MemberMode::PlanMode,
            workspace: crate::schema::spec::TeamWorkspaceSpec {
                root_path: Some("/tmp/review_team".to_owned()),
                ..crate::schema::spec::TeamWorkspaceSpec::default()
            },
            metadata: BTreeMap::from([("owner".to_owned(), json!("ops"))]),
            ..TeamAgentSpec::default()
        }
    }

    #[test]
    fn team_spec_json_defaults_match_python() {
        let value = serde_json::to_value(TeamSpec::default()).unwrap();
        assert_eq!(value["team_name"], json!("agent_team"));
        assert_eq!(value["display_name"], json!("agent_team"));
        assert_eq!(value["conversation_mode"], json!("team"));
        assert_eq!(value["leader_member_name"], json!(null));
        assert_eq!(value["language"], json!(null));
        assert_eq!(value["dispatch_mode"], json!("autonomous"));
        assert_eq!(value["enable_member_messaging"], json!(false));
        assert_eq!(value["max_subtask_depth"], json!(0));
        assert_eq!(value["teammate_mode"], json!("build_mode"));
        assert_eq!(value["model_pool"], json!([]));
        assert_eq!(value["model_pool_strategy"], json!("round_robin"));
        assert_eq!(value["workspace"], json!(null));
        assert_eq!(value["evolution_enabled"], json!(false));
        assert_eq!(value["member_workspace_prefix"], json!(true));
        assert_eq!(value["metadata"], json!({}));

        // Missing keys take the defaults above; unknown keys are tolerated.
        let parsed: TeamSpec = serde_json::from_value(json!({
            "team_name": "alpha",
            "display_name": "Alpha",
            "future_knob": {"nested": true},
        }))
        .unwrap();
        assert_eq!(parsed.team_name, "alpha");
        assert_eq!(parsed.display_name, "Alpha");
        assert!(!parsed.evolution_enabled);
        assert!(parsed.member_workspace_prefix);
        assert_eq!(parsed.model_pool_strategy, ModelPoolStrategy::RoundRobin);
        assert_eq!(parsed.conversation_mode, ConversationMode::Team);
        assert!(parsed.leader_member_name.is_none());
        assert!(parsed.workspace.is_none());
        assert_eq!(parsed.model_pool, Vec::new());
        assert!(parsed.metadata.is_empty());
    }

    #[test]
    fn team_spec_from_spec_mirrors_the_blueprint() {
        let mirror = TeamSpec::from_spec(&full_blueprint());
        assert_eq!(mirror.team_name, "review_team");
        assert_eq!(mirror.display_name, "Review Team");
        assert_eq!(mirror.conversation_mode, ConversationMode::Team);
        assert_eq!(mirror.leader_member_name.as_deref(), Some("team_leader"));
        assert_eq!(mirror.language.as_deref(), Some("en"));
        assert_eq!(mirror.dispatch_mode, DispatchMode::Scheduled);
        assert!(mirror.enable_member_messaging);
        assert_eq!(mirror.max_subtask_depth, 2);
        assert_eq!(mirror.teammate_mode, MemberMode::PlanMode);
        assert_eq!(mirror.model_pool, Vec::new());
        assert_eq!(mirror.model_pool_strategy, ModelPoolStrategy::RoundRobin);
        assert_eq!(
            mirror.workspace,
            Some(WorkspaceSpec {
                root_path: "/tmp/review_team".to_owned(),
                language: "en".to_owned(),
                stable_base: false,
            })
        );
        assert!(!mirror.evolution_enabled);
        assert!(mirror.member_workspace_prefix);
        assert_eq!(mirror.metadata.get("owner"), Some(&json!("ops")));
    }

    #[test]
    fn team_spec_from_spec_falls_back_to_the_team_name_and_default_language() {
        let mirror = TeamSpec::from_spec(&TeamAgentSpec::default());
        assert_eq!(mirror.team_name, "agent_team");
        assert_eq!(mirror.display_name, "agent_team");
        assert_eq!(mirror.leader_member_name.as_deref(), Some("team_leader"));
        assert!(mirror.language.is_none());
        assert!(mirror.workspace.is_none());
        assert_eq!(mirror.model_pool, Vec::new());
        assert!(mirror.metadata.is_empty());

        // A workspace node is materialized only when the blueprint pins a root path, and it then
        // inherits the *resolved* language rather than the raw `Option`.
        let spec = TeamAgentSpec {
            team_display_name: Some(String::new()),
            workspace: crate::schema::spec::TeamWorkspaceSpec {
                root_path: Some("./team".to_owned()),
                ..crate::schema::spec::TeamWorkspaceSpec::default()
            },
            leader: crate::schema::member_spec::LeaderSpec {
                member_name: "boss".to_owned(),
                ..crate::schema::member_spec::LeaderSpec::default()
            },
            ..TeamAgentSpec::default()
        };
        let mirror = TeamSpec::from_spec(&spec);
        assert_eq!(mirror.display_name, "");
        assert_eq!(mirror.leader_member_name.as_deref(), Some("boss"));
        assert_eq!(
            mirror.workspace,
            Some(WorkspaceSpec {
                root_path: "./team".to_owned(),
                language: "cn".to_owned(),
                stable_base: false,
            })
        );
    }

    #[test]
    fn team_runtime_context_language_falls_back_to_the_default() {
        let team_spec = TeamSpec {
            team_name: "alpha".to_owned(),
            language: Some("en".to_owned()),
            ..TeamSpec::default()
        };
        let context = TeamRuntimeContext::new(TeamRole::Teammate, team_spec);
        assert_eq!(context.role, TeamRole::Teammate);
        assert_eq!(context.language, "en");
        assert_eq!(context.team_spec.team_name, "alpha");
        assert!(context.member_name.is_none());
        assert!(context.member_model.is_none());
        assert!(context.agent_template_spec.is_none());
        assert_eq!(context.display_name, "");
        assert_eq!(context.desc, "");
        assert_eq!(context.prompt, "");
        assert_eq!(context.workspace_root, "");

        let context = TeamRuntimeContext::new(TeamRole::Leader, TeamSpec::default());
        assert_eq!(context.role, TeamRole::Leader);
        assert_eq!(context.language, crate::constants::DEFAULT_LANGUAGE);
        assert_eq!(context.language, "cn");

        let parsed: TeamRuntimeContext =
            serde_json::from_value(json!({"role": "teammate", "language": "en", "future": 1}))
                .unwrap();
        assert_eq!(parsed.role, TeamRole::Teammate);
        assert_eq!(parsed.language, "en");
        assert_eq!(parsed.team_spec.team_name, "agent_team");
    }

    /// One roster row carrying only its member name, for the sorting tests.
    fn member(name: &str) -> MemberSnapshot {
        MemberSnapshot {
            member_name: name.to_owned(),
            display_name: name.to_owned(),
            ..MemberSnapshot::default()
        }
    }

    /// One task summary carrying only its id, for the sorting tests.
    fn task(id: &str) -> TaskSummary {
        TaskSummary {
            task_id: id.to_owned(),
            ..TaskSummary::default()
        }
    }

    #[test]
    fn team_snapshot_sort_orders_members_and_tasks() {
        let mut snapshot = TeamSnapshot::new("alpha", HarnessState::Running);
        assert_eq!(snapshot.team_name, "alpha");
        assert_eq!(snapshot.state, HarnessState::Running);
        assert_eq!(snapshot.members, Vec::new());
        assert_eq!(snapshot.tasks, Vec::new());
        assert_eq!(snapshot.updated_at, 0);

        snapshot.members = vec![member("dev-2"), member("team_leader"), member("dev-1")];
        snapshot.tasks = vec![task("t-3"), task("t-1"), task("t-2")];
        snapshot.sort();

        let names: Vec<&str> = snapshot
            .members
            .iter()
            .map(|member| member.member_name.as_str())
            .collect();
        assert_eq!(names, ["dev-1", "dev-2", "team_leader"]);
        let ids: Vec<&str> = snapshot
            .tasks
            .iter()
            .map(|task| task.task_id.as_str())
            .collect();
        assert_eq!(ids, ["t-1", "t-2", "t-3"]);

        // Sorting is idempotent.
        let before = snapshot.clone();
        snapshot.sort();
        assert_eq!(snapshot, before);
    }

    #[test]
    fn team_snapshot_json_round_trip() {
        let mut snapshot = TeamSnapshot::new("alpha", HarnessState::Running);
        snapshot.members.push(MemberSnapshot {
            member_name: "dev-1".to_owned(),
            display_name: "Dev One".to_owned(),
            role: TeamRole::Teammate,
            status: MemberStatus::Busy,
            execution: ExecutionStatus::Running,
        });
        snapshot.tasks.push(TaskSummary {
            task_id: "t-1".to_owned(),
            title: "wire the bus".to_owned(),
            status: crate::schema::status::TaskStatus::InProgress,
            assignee: Some("dev-1".to_owned()),
            ..TaskSummary::default()
        });
        snapshot.updated_at = 1_700_000_000_000;

        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value["team_name"], json!("alpha"));
        assert_eq!(value["state"], json!("running"));
        assert_eq!(value["members"][0]["member_name"], json!("dev-1"));
        assert_eq!(value["members"][0]["role"], json!("teammate"));
        assert_eq!(value["members"][0]["status"], json!("busy"));
        assert_eq!(value["members"][0]["execution"], json!("running"));
        assert_eq!(value["tasks"][0]["task_id"], json!("t-1"));
        assert_eq!(value["tasks"][0]["status"], json!("in_progress"));
        assert_eq!(value["updated_at"], json!(1_700_000_000_000_u64));

        let parsed: TeamSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, snapshot);

        // A sparse snapshot takes the defaults and tolerates unknown keys.
        let sparse: TeamSnapshot =
            serde_json::from_value(json!({"team_name": "beta", "future": 1})).unwrap();
        assert_eq!(
            sparse,
            TeamSnapshot {
                team_name: "beta".to_owned(),
                ..TeamSnapshot::default()
            }
        );
        assert_eq!(sparse.state, HarnessState::Idle);
    }
}
