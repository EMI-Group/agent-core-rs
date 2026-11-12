//! The team blueprint: [`TeamAgentSpec`] and the spec-level enums and wrappers it is built from.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/blueprint.py` (the `TeamAgentSpec` pydantic
//! model plus `TransportSpec`, `StorageSpec` and `TeamWorkspaceConfig`).
//!
//! [`TeamAgentSpec`] is the entry point of the crate: a caller describes the team (which agent
//! blueprints exist, who leads it, and every knob the runtime reads) and `spec.build()` /
//! `build_with` materializes it. That makes the defaults here a seam — every other node inherits
//! them, so they are pinned against literal JSON rather than derived again downstream.
//!
//! Deliberate deviations from Python (documented in `src/schema/CONTEXT.md`): [`SpawnMode`]
//! defaults to `inprocess` because this port has no OS-process members, and [`TeamAgentSpec`]'s
//! validation additionally rejects a predefined member name outside `^[a-z][a-z0-9_-]*$`.

// NOTE: this file is intentionally long: it holds the whole team blueprint — every spec enum, the
// member-facing leaf types' wiring and the validation rules pinned to Python messages.
use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::deep_agent::DeepAgentSpec;
use super::member_spec::{
    LeaderSpec, MemberSpecBase, ModelPoolEntry, PredefinedMemberSpec, TeamMemberSpec,
};
use super::team::TeamRole;
use crate::constants::{
    is_reserved_name, DEFAULT_LANGUAGE, DEFAULT_LEADER_MEMBER_NAME, DEFAULT_TEAM_NAME,
    RESERVED_MEMBER_NAMES,
};

/// Key of the leader's agent blueprint inside [`TeamAgentSpec::agents`] (Python `agents["leader"]`).
const LEADER_AGENT_KEY: &str = "leader";

/// Key of the fallback teammate blueprint inside [`TeamAgentSpec::agents`].
const TEAMMATE_AGENT_KEY: &str = "teammate";

/// Transport kind used when a spec does not name one (Python `TransportSpec(type="inprocess")`).
const DEFAULT_TRANSPORT_KIND: &str = "inprocess";

/// Storage kind used when a spec does not name one (Python `StorageSpec(type="memory")`).
const DEFAULT_STORAGE_KIND: &str = "memory";

/// Conflict strategy of a version-controlled team workspace (Python default `lock`).
const DEFAULT_CONFLICT_STRATEGY: &str = "lock";

/// Artifact directories every team workspace reserves (Python `TeamWorkspaceConfig`).
const DEFAULT_ARTIFACT_DIRS: [&str; 4] = [
    "artifacts/code",
    "artifacts/docs",
    "artifacts/reports",
    "trajectories",
];

/// Default per-task review-round ceiling (Python `default_max_review_rounds = 3`).
const DEFAULT_MAX_REVIEW_ROUNDS: usize = 3;

/// Default seconds before a stalled review round escalates (Python `review_stall_timeout = 2400`).
const DEFAULT_REVIEW_STALL_TIMEOUT: u64 = 2_400;

/// Default idle window before the stall sweep nudges a member (Python `600`).
const DEFAULT_STALE_IDLE_TIMEOUT: u64 = 600;

/// Default steering batch size of a non-leader member (Python `steer_batch_size = 2`).
const DEFAULT_STEER_BATCH_SIZE: usize = 2;

/// How long a team's rows survive (Python `TeamLifecycle`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamLifecycle {
    /// Rows are removed when the team stops; the Python default.
    #[default]
    Temporary,
    /// Rows are kept so the team can be resumed later.
    Persistent,
}

/// How members talk to each other (Python `conversation_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationMode {
    /// Leader-mediated: members exchange messages through the team channel; the Python default.
    #[default]
    Team,
    /// Free-for-all group chat in which every member may address every other member.
    GroupChat,
}

/// Who hands work to the members (Python `dispatch_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchMode {
    /// Members poll the mailbox and claim work from the board themselves; the Python default.
    #[default]
    Autonomous,
    /// The leader-side scheduler assigns tasks and starts members.
    Scheduled,
}

impl DispatchMode {
    /// Returns the Python wire string of this mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Autonomous => "autonomous",
            Self::Scheduled => "scheduled",
        }
    }
}

/// Which tool and prompt set a non-leader member runs with (Python `teammate_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberMode {
    /// Members implement directly (`member_complete_task`, no `claim_task`); the Python default.
    #[default]
    BuildMode,
    /// Members plan first and hand the plan to the leader for approval.
    PlanMode,
}

/// Where teammates run (Python `spawn_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawnMode {
    /// Teammates run inside the leader's process; wire value `inprocess`, the Python default.
    #[default]
    #[serde(rename = "inprocess")]
    InProcess,
    /// Teammates run as OS processes (deferred in this port); wire value `process`.
    Process,
}

/// Who materializes the team row (Python `bootstrap_mode`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapMode {
    /// The leader calls the `build_team` tool; the Python default.
    #[default]
    Leader,
    /// The framework builds the team on `start()`; requires scheduled dispatch.
    Framework,
}

/// How a member is assigned a model from [`TeamAgentSpec::model_pool`] (Python
/// `model_pool_strategy`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPoolStrategy {
    /// Hand out endpoints in rotation, one per member; the Python default.
    #[default]
    RoundRobin,
    /// Pick the endpoint whose `model_name` the member asked for.
    ByModelName,
}

/// Language of prompts, rail messages and generated text (Python `language`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    /// Chinese; the Python default.
    #[default]
    Cn,
    /// Traditional Chinese.
    Cf,
    /// English.
    En,
}

impl Language {
    /// Returns the Python wire string of this language.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cn => "cn",
            Self::Cf => "cf",
            Self::En => "en",
        }
    }

    /// Resolves a declared language to a wire value, falling back to [`DEFAULT_LANGUAGE`].
    ///
    /// Python's `resolve_language` treats an unknown or absent language as the default rather than
    /// failing, so a spec written against a newer catalogue still loads.
    #[must_use]
    pub fn resolve(value: Option<&str>) -> &'static str {
        match value {
            Some("cf") => Self::Cf.as_str(),
            Some("en") => Self::En.as_str(),
            _ => DEFAULT_LANGUAGE,
        }
    }
}

/// Pluggable transport of a team (Python `TransportSpec`).
///
/// The kind round-trips under the Python key `"type"`; `"kind"` is accepted as an alias because the
/// Rust field carries that name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TransportSpec {
    /// Backend key; defaults to `inprocess`. Serialized as `"type"`.
    #[serde(rename = "type", alias = "kind")]
    pub kind: String,
    /// Backend-specific parameters, never inspected by this crate.
    pub params: BTreeMap<String, Value>,
}

impl Default for TransportSpec {
    fn default() -> Self {
        Self::inprocess()
    }
}

impl TransportSpec {
    /// Builds the in-process (default) transport spec.
    #[must_use]
    pub fn inprocess() -> Self {
        Self {
            kind: DEFAULT_TRANSPORT_KIND.to_owned(),
            params: BTreeMap::new(),
        }
    }

    /// Returns `true` when this spec selects the in-process messager.
    #[must_use]
    pub fn is_inprocess(&self) -> bool {
        self.kind == DEFAULT_TRANSPORT_KIND
    }
}

/// Persistence backend of a team (Python `StorageSpec`).
///
/// The kind round-trips under the Python key `"type"`; `"kind"` is accepted as an alias because the
/// Rust field carries that name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StorageSpec {
    /// Backend key; defaults to `memory`. Serialized as `"type"`.
    #[serde(rename = "type", alias = "kind")]
    pub kind: String,
    /// Backend-specific parameters, never inspected by this crate.
    pub params: BTreeMap<String, Value>,
}

impl Default for StorageSpec {
    fn default() -> Self {
        Self::in_memory()
    }
}

impl StorageSpec {
    /// Builds the in-memory (default) storage spec.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            kind: DEFAULT_STORAGE_KIND.to_owned(),
            params: BTreeMap::new(),
        }
    }

    /// Returns `true` when this spec selects the in-memory store.
    #[must_use]
    pub fn is_in_memory(&self) -> bool {
        self.kind == DEFAULT_STORAGE_KIND
    }
}

/// Shared workspace of a team (Python `TeamWorkspaceConfig`).
///
/// `enabled: false` keeps every member on its own member workspace; enabling it allocates the
/// shared directory tree with the reserved [`Self::artifact_dirs`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamWorkspaceSpec {
    /// Whether members share one workspace; defaults to `false`.
    pub enabled: bool,
    /// Root of the shared workspace; `None` uses the default team home.
    pub root_path: Option<String>,
    /// Artifact directories reserved inside the workspace.
    pub artifact_dirs: Vec<String>,
    /// Whether the workspace tree is version controlled; defaults to `true`.
    pub version_control: bool,
    /// How concurrent writes are arbitrated; defaults to `lock`.
    pub conflict_strategy: String,
    /// Remote URL of the workspace repository, when it has one.
    pub remote_url: Option<String>,
}

impl Default for TeamWorkspaceSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            root_path: None,
            artifact_dirs: DEFAULT_ARTIFACT_DIRS
                .iter()
                .map(ToString::to_string)
                .collect(),
            version_control: true,
            conflict_strategy: DEFAULT_CONFLICT_STRATEGY.to_owned(),
            remote_url: None,
        }
    }
}

/// The blueprint of a team: who is in it and how it is run.
///
/// Python parity: `TeamAgentSpec`. A caller fills the fields, calls
/// [`TeamAgentSpec::validate`] and hands the result to `spec.build()` (or, in this port,
/// `build_with`). Field names are their JSON names, so a Python-dumped spec loads unchanged;
/// unknown fields are ignored.
///
/// Only the knobs this port implements are modelled — organisation-layer, swarmflow and
/// model-router configuration are absent rather than carried as dead fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamAgentSpec {
    /// Team name written to the team row; defaults to `agent_team`.
    pub team_name: String,
    /// Display label of the team; `None` falls back to the generic default.
    pub team_display_name: Option<String>,
    /// Team description written to the team row; `None` falls back to the generic default.
    pub team_desc: Option<String>,
    /// Agent blueprints by name; must contain a `leader` entry.
    pub agents: BTreeMap<String, DeepAgentSpec>,
    /// Identity of the team leader.
    pub leader: LeaderSpec,
    /// Members declared up front, materialized when the team is built.
    pub predefined_members: Vec<PredefinedMemberSpec>,
    /// Agent-group package to compose the roster from, when the team uses one.
    pub agent_group_name: Option<String>,
    /// Whether team rows are discarded or kept when the team stops.
    pub lifecycle: TeamLifecycle,
    /// How members exchange messages.
    pub conversation_mode: ConversationMode,
    /// Who hands work to members.
    pub dispatch_mode: DispatchMode,
    /// Tool and prompt set non-leader members run with.
    pub teammate_mode: MemberMode,
    /// Where teammates run; `inprocess` in this port.
    pub spawn_mode: SpawnMode,
    /// Who materializes the team row.
    pub bootstrap_mode: BootstrapMode,
    /// Whether scheduled-dispatch members may message each other directly.
    pub enable_member_messaging: bool,
    /// Whether the leader is asked to assign reviewers to tasks.
    pub enable_task_verification: bool,
    /// Subtask nesting depth allowed to members; `0` keeps `create_task` leader-only.
    pub max_subtask_depth: usize,
    /// Capability ceiling for human-in-the-team members.
    pub enable_hitt: bool,
    /// Language of prompts and generated text; `None` resolves to the default.
    pub language: Option<String>,
    /// LLM endpoints members may be allocated from.
    pub model_pool: Vec<ModelPoolEntry>,
    /// How members are assigned an endpoint from [`Self::model_pool`].
    pub model_pool_strategy: ModelPoolStrategy,
    /// Shared workspace configuration.
    pub workspace: TeamWorkspaceSpec,
    /// Transport of the team.
    pub transport: TransportSpec,
    /// Persistence backend of the team.
    pub storage: StorageSpec,
    /// Per-task review-round ceiling used when a task sets none; defaults to 3.
    pub default_max_review_rounds: usize,
    /// Seconds before a stalled review round escalates; defaults to 2400.
    pub review_stall_timeout: u64,
    /// Seconds an autonomous member may hold active work while idle; defaults to 600.
    pub stale_claim_idle_timeout: u64,
    /// Seconds the leader may sit idle on unclaimed work; defaults to 600.
    pub stale_pending_idle_timeout: u64,
    /// Queued steering messages a member takes per model call; defaults to 2.
    pub steer_batch_size: usize,
    /// Free-form extension payload, never inspected by this crate.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for TeamAgentSpec {
    /// Python defaults: a temporary in-process team named `agent_team`, led by `team_leader`, with
    /// an empty roster and the review/stall knobs at their Python values.
    fn default() -> Self {
        Self {
            team_name: DEFAULT_TEAM_NAME.to_owned(),
            team_display_name: None,
            team_desc: None,
            agents: BTreeMap::new(),
            leader: LeaderSpec::default(),
            predefined_members: Vec::new(),
            agent_group_name: None,
            lifecycle: TeamLifecycle::default(),
            conversation_mode: ConversationMode::default(),
            dispatch_mode: DispatchMode::default(),
            teammate_mode: MemberMode::default(),
            spawn_mode: SpawnMode::default(),
            bootstrap_mode: BootstrapMode::default(),
            enable_member_messaging: false,
            enable_task_verification: false,
            max_subtask_depth: 0,
            enable_hitt: false,
            language: None,
            model_pool: Vec::new(),
            model_pool_strategy: ModelPoolStrategy::default(),
            workspace: TeamWorkspaceSpec::default(),
            transport: TransportSpec::default(),
            storage: StorageSpec::default(),
            default_max_review_rounds: DEFAULT_MAX_REVIEW_ROUNDS,
            review_stall_timeout: DEFAULT_REVIEW_STALL_TIMEOUT,
            stale_claim_idle_timeout: DEFAULT_STALE_IDLE_TIMEOUT,
            stale_pending_idle_timeout: DEFAULT_STALE_IDLE_TIMEOUT,
            steer_batch_size: DEFAULT_STEER_BATCH_SIZE,
            metadata: BTreeMap::new(),
        }
    }
}

impl TeamAgentSpec {
    /// Checks the spec before it is built.
    ///
    /// Python parity: the `_validate_*` model validators of `TeamAgentSpec`, in their Python order:
    /// the leader blueprint and reserved names first, then the bootstrap/dispatch pairing, then the
    /// numeric knobs the scheduler and stall sweep consume without further guarding.
    ///
    /// Reserved-name rules follow Python: the leader may keep the default `team_leader` (but not
    /// `user` / `human_agent`), and a predefined `human_agent` entry is exempt because the
    /// framework injects one under that reserved name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with the Python message text for a missing leader blueprint, a
    /// reserved or malformed member name, a framework bootstrap outside scheduled dispatch, and an
    /// out-of-range review/stall/steering knob.
    pub fn validate(&self) -> Result<()> {
        if !self.agents.contains_key(LEADER_AGENT_KEY) {
            return Err(Error::validation("agents dict must contain a 'leader' key"));
        }

        let leader_name = self.leader_member_name();
        if leader_name != DEFAULT_LEADER_MEMBER_NAME && is_reserved_name(leader_name) {
            return Err(Error::validation(format!(
                "LeaderSpec.member_name '{leader_name}' is reserved; pick a different name"
            )));
        }

        for member in &self.predefined_members {
            let name = member.member_name();
            if member.role_type() == TeamRole::HumanAgent {
                continue;
            }
            if is_reserved_name(name) {
                let mut reserved = RESERVED_MEMBER_NAMES;
                reserved.sort_unstable();
                return Err(Error::validation(format!(
                    "predefined member '{name}' uses a reserved name (reserved: {reserved:?})"
                )));
            }
            if !Self::is_member_name_valid(name) {
                return Err(Error::validation(format!(
                    "predefined member '{name}' is not a valid member name \
                     (expected ^[a-z][a-z0-9_-]*$)"
                )));
            }
        }

        if self.bootstrap_mode == BootstrapMode::Framework
            && self.dispatch_mode != DispatchMode::Scheduled
        {
            return Err(Error::validation(format!(
                "bootstrap_mode='framework' requires dispatch_mode='scheduled'; \
                 got dispatch_mode='{}'",
                self.dispatch_mode.as_str()
            )));
        }

        if self.default_max_review_rounds < 1 {
            return Err(Error::validation(format!(
                "default_max_review_rounds must be >= 1, got {}",
                self.default_max_review_rounds
            )));
        }
        if self.review_stall_timeout == 0 {
            return Err(Error::validation(format!(
                "review_stall_timeout must be > 0 seconds, got {}",
                self.review_stall_timeout
            )));
        }
        if self.stale_claim_idle_timeout == 0 {
            return Err(Error::validation(format!(
                "stale_claim_idle_timeout must be > 0 seconds, got {}",
                self.stale_claim_idle_timeout
            )));
        }
        if self.stale_pending_idle_timeout == 0 {
            return Err(Error::validation(format!(
                "stale_pending_idle_timeout must be > 0 seconds, got {}",
                self.stale_pending_idle_timeout
            )));
        }
        if self.steer_batch_size == 0 {
            return Err(Error::validation(format!(
                "steer_batch_size must be > 0 messages, got {}",
                self.steer_batch_size
            )));
        }

        Ok(())
    }

    /// Returns the member name of the leader, falling back to `team_leader`.
    #[must_use]
    pub fn leader_member_name(&self) -> &str {
        if self.leader.member_name.is_empty() {
            DEFAULT_LEADER_MEMBER_NAME
        } else {
            &self.leader.member_name
        }
    }

    /// Returns the resolved language wire value, falling back to [`DEFAULT_LANGUAGE`].
    #[must_use]
    pub fn resolved_language(&self) -> &str {
        Language::resolve(self.language.as_deref())
    }

    /// Returns the default teammate blueprint (`agents["teammate"]`), when the spec has one.
    #[must_use]
    pub fn teammate_defaults(&self) -> Option<&DeepAgentSpec> {
        self.agents.get(TEAMMATE_AGENT_KEY)
    }

    /// Returns the member spec a name resolves to, whether or not it is spawned.
    ///
    /// Resolution order follows the contract: a matching [`Self::predefined_members`] entry wins,
    /// then the agent blueprint registered under `name`, then the shared `agents["teammate"]`
    /// blueprint (with `name` as its member name). Anything else is `None`.
    ///
    /// The derived spec takes its display name and public description from the blueprint's
    /// [`AgentCard`](openjiuwen_agent::AgentCard) (an empty card name falls back to `name`, and an
    /// absent card to `name` / an empty description), its private prompt from
    /// [`DeepAgentSpec::system_prompt`] and its model from [`DeepAgentSpec::model_name`]. A name
    /// equal to [`Self::leader_member_name`] resolves to [`TeamRole::Leader`].
    #[must_use]
    pub fn member_spec(&self, name: &str) -> Option<TeamMemberSpec> {
        if let Some(predefined) = self
            .predefined_members
            .iter()
            .find(|member| member.member_name() == name)
        {
            return Some(predefined.clone().into_member_spec());
        }

        let blueprint = self
            .agents
            .get(name)
            .or_else(|| self.agents.get(TEAMMATE_AGENT_KEY))?;
        let card = blueprint.card.as_ref();
        let display_name = card
            .map(|card| card.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| name.to_owned());
        let role_type = if name == self.leader_member_name() {
            TeamRole::Leader
        } else {
            TeamRole::Teammate
        };

        Some(TeamMemberSpec {
            base: MemberSpecBase {
                member_name: name.to_owned(),
                display_name,
                desc: card
                    .map(|card| card.description.clone())
                    .unwrap_or_default(),
                prompt: blueprint.system_prompt.clone().unwrap_or_default(),
                model_name: blueprint.model_name().map(str::to_owned),
            },
            role_type,
            goal: None,
        })
    }

    /// Returns `true` when `name` matches the member-name grammar `^[a-z][a-z0-9_-]*$`.
    ///
    /// The check is a hand-written character scan: no regex engine and no global state is involved,
    /// and the empty name is invalid.
    #[must_use]
    pub fn is_member_name_valid(name: &str) -> bool {
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        if !first.is_ascii_lowercase() {
            return false;
        }
        chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_agent::AgentCard;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::super::deep_agent::ModelRef;
    use super::*;

    /// A blueprint-bearing agent, used to check derived member specs.
    fn agent_spec(name: &str, description: &str, prompt: &str, model: &str) -> DeepAgentSpec {
        DeepAgentSpec {
            card: Some(AgentCard {
                name: name.to_owned(),
                description: description.to_owned(),
                ..AgentCard::default()
            }),
            system_prompt: Some(prompt.to_owned()),
            model: Some(ModelRef {
                model_name: Some(model.to_owned()),
                ..ModelRef::default()
            }),
            ..DeepAgentSpec::default()
        }
    }

    /// A knob mutation paired with the [`TeamAgentSpec::validate`] message it must produce.
    type Knob = fn(&mut TeamAgentSpec);

    /// A spec that passes [`TeamAgentSpec::validate`].
    fn minimal_spec() -> TeamAgentSpec {
        let mut spec = TeamAgentSpec::default();
        spec.agents
            .insert(LEADER_AGENT_KEY.to_owned(), DeepAgentSpec::default());
        spec
    }

    /// A spec whose single predefined teammate is named `name`.
    fn spec_with_predefined(name: &str) -> TeamAgentSpec {
        let mut spec = minimal_spec();
        spec.predefined_members
            .push(PredefinedMemberSpec::Teammate(MemberSpecBase {
                member_name: name.to_owned(),
                ..MemberSpecBase::default()
            }));
        spec
    }

    /// Pins one enum variant's wire value in both directions.
    fn round_trip<T>(value: &T, wire: &str)
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        assert_eq!(
            serde_json::to_value(value).unwrap(),
            json!(wire),
            "{value:?}"
        );
        let parsed: T = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(&parsed, value, "{wire}");
    }

    #[test]
    fn default_spec_matches_the_pinned_wire_shape() {
        assert_eq!(
            serde_json::to_value(TeamAgentSpec::default()).unwrap(),
            json!({
                "team_name": "agent_team",
                "team_display_name": null,
                "team_desc": null,
                "agents": {},
                "leader": {
                    "member_name": "team_leader",
                    "display_name": "Team Leader",
                    "desc": "",
                    "goal": null,
                },
                "predefined_members": [],
                "agent_group_name": null,
                "lifecycle": "temporary",
                "conversation_mode": "team",
                "dispatch_mode": "autonomous",
                "teammate_mode": "build_mode",
                "spawn_mode": "inprocess",
                "bootstrap_mode": "leader",
                "enable_member_messaging": false,
                "enable_task_verification": false,
                "max_subtask_depth": 0,
                "enable_hitt": false,
                "language": null,
                "model_pool": [],
                "model_pool_strategy": "round_robin",
                "workspace": {
                    "enabled": false,
                    "root_path": null,
                    "artifact_dirs": [
                        "artifacts/code",
                        "artifacts/docs",
                        "artifacts/reports",
                        "trajectories",
                    ],
                    "version_control": true,
                    "conflict_strategy": "lock",
                    "remote_url": null,
                },
                "transport": {"type": "inprocess", "params": {}},
                "storage": {"type": "memory", "params": {}},
                "default_max_review_rounds": 3,
                "review_stall_timeout": 2400,
                "stale_claim_idle_timeout": 600,
                "stale_pending_idle_timeout": 600,
                "steer_batch_size": 2,
                "metadata": {},
            })
        );
    }

    #[test]
    fn spec_parses_partial_json_and_tolerates_unknown_fields() {
        let spec: TeamAgentSpec = serde_json::from_value(json!({
            "team_name": "demo",
            "language": "en",
            "dispatch_mode": "scheduled",
            "teammate_mode": "plan_mode",
            "enable_hitt": true,
            "unknown_python_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(spec.team_name, "demo");
        assert_eq!(spec.language.as_deref(), Some("en"));
        assert_eq!(spec.dispatch_mode, DispatchMode::Scheduled);
        assert_eq!(spec.teammate_mode, MemberMode::PlanMode);
        assert!(spec.enable_hitt);
        assert_eq!(spec.default_max_review_rounds, DEFAULT_MAX_REVIEW_ROUNDS);
        assert!(spec.transport.is_inprocess());
        assert!(spec.storage.is_in_memory());

        let reserialized = serde_json::to_value(&spec).unwrap();
        assert!(reserialized.get("unknown_python_field").is_none());
        let reparsed: TeamAgentSpec = serde_json::from_value(reserialized).unwrap();
        assert_eq!(reparsed, spec);
    }

    #[test]
    fn spec_enums_use_the_python_wire_values() {
        round_trip(&TeamLifecycle::Temporary, "temporary");
        round_trip(&TeamLifecycle::Persistent, "persistent");
        round_trip(&ConversationMode::Team, "team");
        round_trip(&ConversationMode::GroupChat, "group_chat");
        round_trip(&DispatchMode::Autonomous, "autonomous");
        round_trip(&DispatchMode::Scheduled, "scheduled");
        round_trip(&MemberMode::BuildMode, "build_mode");
        round_trip(&MemberMode::PlanMode, "plan_mode");
        round_trip(&SpawnMode::InProcess, "inprocess");
        round_trip(&SpawnMode::Process, "process");
        round_trip(&BootstrapMode::Leader, "leader");
        round_trip(&BootstrapMode::Framework, "framework");
        round_trip(&ModelPoolStrategy::RoundRobin, "round_robin");
        round_trip(&ModelPoolStrategy::ByModelName, "by_model_name");
        round_trip(&Language::Cn, "cn");
        round_trip(&Language::Cf, "cf");
        round_trip(&Language::En, "en");
        assert_eq!(DispatchMode::Scheduled.as_str(), "scheduled");
        assert_eq!(Language::En.as_str(), "en");
        // Python writes `inprocess`, never the `snake_case` form of the variant name.
        assert!(serde_json::from_value::<SpawnMode>(json!("in_process")).is_err());
    }

    #[test]
    fn spec_enum_defaults_match_python() {
        assert_eq!(TeamLifecycle::default(), TeamLifecycle::Temporary);
        assert_eq!(ConversationMode::default(), ConversationMode::Team);
        assert_eq!(DispatchMode::default(), DispatchMode::Autonomous);
        assert_eq!(MemberMode::default(), MemberMode::BuildMode);
        assert_eq!(SpawnMode::default(), SpawnMode::InProcess);
        assert_eq!(BootstrapMode::default(), BootstrapMode::Leader);
        assert_eq!(ModelPoolStrategy::default(), ModelPoolStrategy::RoundRobin);
        assert_eq!(Language::default(), Language::Cn);
    }

    #[test]
    fn wrapper_specs_default_and_accept_both_keys() {
        let transport: TransportSpec = serde_json::from_value(json!({"type": "pyzmq"})).unwrap();
        assert_eq!(transport.kind, "pyzmq");
        assert!(!transport.is_inprocess());
        let aliased: TransportSpec = serde_json::from_value(json!({"kind": "pyzmq"})).unwrap();
        assert_eq!(aliased, transport);

        let storage: StorageSpec =
            serde_json::from_value(json!({"kind": "sqlite", "params": {"path": "team.db"}}))
                .unwrap();
        assert_eq!(storage.params.get("path"), Some(&json!("team.db")));
        assert!(!storage.is_in_memory());

        // The Python `type` key is what a dump carries.
        assert_eq!(
            serde_json::to_value(TransportSpec::inprocess()).unwrap(),
            json!({"type": "inprocess", "params": {}})
        );
        assert_eq!(
            serde_json::to_value(StorageSpec::in_memory()).unwrap(),
            json!({"type": "memory", "params": {}})
        );
        assert!(TransportSpec::default().is_inprocess());
        assert!(StorageSpec::default().is_in_memory());

        let workspace = TeamWorkspaceSpec::default();
        assert!(!workspace.enabled);
        assert_eq!(workspace.root_path, None);
        assert_eq!(
            workspace.artifact_dirs,
            [
                "artifacts/code",
                "artifacts/docs",
                "artifacts/reports",
                "trajectories"
            ]
        );
        assert!(workspace.version_control);
        assert_eq!(workspace.conflict_strategy, "lock");
        assert_eq!(workspace.remote_url, None);

        let parsed: TeamWorkspaceSpec =
            serde_json::from_value(json!({"unknown_python_field": 1})).unwrap();
        assert_eq!(parsed, workspace);
    }

    #[test]
    fn language_and_spec_getters_fall_back_to_their_defaults() {
        for (declared, resolved) in [
            (None, DEFAULT_LANGUAGE),
            (Some("cn"), "cn"),
            (Some("cf"), "cf"),
            (Some("en"), "en"),
            (Some("fr"), DEFAULT_LANGUAGE),
            (Some(""), DEFAULT_LANGUAGE),
        ] {
            assert_eq!(Language::resolve(declared), resolved, "{declared:?}");
        }

        let mut spec = TeamAgentSpec::default();
        assert_eq!(spec.resolved_language(), DEFAULT_LANGUAGE);
        assert_eq!(spec.leader_member_name(), DEFAULT_LEADER_MEMBER_NAME);
        spec.language = Some("en".to_owned());
        spec.leader.member_name = "chief".to_owned();
        assert_eq!(spec.resolved_language(), "en");
        assert_eq!(spec.leader_member_name(), "chief");
        spec.language = Some("klingon".to_owned());
        assert_eq!(spec.resolved_language(), DEFAULT_LANGUAGE);
    }

    #[test]
    fn validate_accepts_a_minimal_spec_but_requires_a_leader_agent() {
        assert!(minimal_spec().validate().is_ok());
        // The leader may keep the reserved-by-default `team_leader` name.
        let mut spec = minimal_spec();
        spec.leader.member_name = DEFAULT_LEADER_MEMBER_NAME.to_owned();
        assert!(spec.validate().is_ok());

        let err = TeamAgentSpec::default().validate().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert_eq!(err.message(), "agents dict must contain a 'leader' key");
    }

    #[test]
    fn validate_rejects_a_reserved_leader_name() {
        for reserved in ["user", "human_agent"] {
            let mut spec = minimal_spec();
            spec.leader.member_name = reserved.to_owned();
            assert_eq!(
                spec.validate().unwrap_err().message(),
                format!("LeaderSpec.member_name '{reserved}' is reserved; pick a different name")
            );
        }
    }

    #[test]
    fn validate_rejects_reserved_and_invalid_predefined_names() {
        assert_eq!(
            spec_with_predefined("user")
                .validate()
                .unwrap_err()
                .message(),
            "predefined member 'user' uses a reserved name \
             (reserved: [\"human_agent\", \"team_leader\", \"user\"])"
        );
        for invalid in ["Dev 1", "1dev", "", "-dev", "dev.1", "Dev"] {
            assert_eq!(
                spec_with_predefined(invalid)
                    .validate()
                    .unwrap_err()
                    .message(),
                format!(
                    "predefined member '{invalid}' is not a valid member name \
                     (expected ^[a-z][a-z0-9_-]*$)"
                )
            );
        }

        // A predefined human agent is the one legal reserved name (Python skips it).
        let mut spec = minimal_spec();
        spec.predefined_members
            .push(PredefinedMemberSpec::HumanAgent(MemberSpecBase {
                member_name: "human_agent".to_owned(),
                ..MemberSpecBase::default()
            }));
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn validate_rejects_framework_bootstrap_under_autonomous_dispatch() {
        let mut spec = minimal_spec();
        spec.bootstrap_mode = BootstrapMode::Framework;
        assert_eq!(
            spec.validate().unwrap_err().message(),
            "bootstrap_mode='framework' requires dispatch_mode='scheduled'; \
             got dispatch_mode='autonomous'"
        );
        spec.dispatch_mode = DispatchMode::Scheduled;
        assert!(spec.validate().is_ok());
        assert!(
            minimal_spec().validate().is_ok(),
            "leader bootstrap is unrestricted"
        );
    }

    #[test]
    fn validate_range_checks_the_review_and_stall_knobs() {
        let knobs: [(Knob, &str); 5] = [
            (
                |spec| spec.default_max_review_rounds = 0,
                "default_max_review_rounds must be >= 1, got 0",
            ),
            (
                |spec| spec.review_stall_timeout = 0,
                "review_stall_timeout must be > 0 seconds, got 0",
            ),
            (
                |spec| spec.stale_claim_idle_timeout = 0,
                "stale_claim_idle_timeout must be > 0 seconds, got 0",
            ),
            (
                |spec| spec.stale_pending_idle_timeout = 0,
                "stale_pending_idle_timeout must be > 0 seconds, got 0",
            ),
            (
                |spec| spec.steer_batch_size = 0,
                "steer_batch_size must be > 0 messages, got 0",
            ),
        ];
        for (zero, message) in knobs {
            let mut spec = minimal_spec();
            zero(&mut spec);
            assert_eq!(spec.validate().unwrap_err().message(), message);
        }

        let mut spec = minimal_spec();
        spec.default_max_review_rounds = 1;
        spec.review_stall_timeout = 1;
        spec.stale_claim_idle_timeout = 1;
        spec.stale_pending_idle_timeout = 1;
        spec.steer_batch_size = 1;
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn member_spec_prefers_a_predefined_member() {
        let base = MemberSpecBase {
            member_name: "dev".to_owned(),
            display_name: "Developer".to_owned(),
            desc: "from predefined".to_owned(),
            prompt: "predefined prompt".to_owned(),
            model_name: Some("m2".to_owned()),
        };
        let mut spec = minimal_spec();
        spec.agents.insert(
            "dev".to_owned(),
            agent_spec("Dev", "from agents", "agents prompt", "m1"),
        );
        spec.predefined_members
            .push(PredefinedMemberSpec::Teammate(base.clone()));
        let member = spec.member_spec("dev").unwrap();
        assert_eq!(member.base, base);
        assert_eq!(member.role_type, TeamRole::Teammate);
        assert_eq!(member.goal, None);
    }

    #[test]
    fn member_spec_derives_from_the_named_agent_blueprint() {
        let mut spec = minimal_spec();
        spec.agents.insert(
            "dev".to_owned(),
            agent_spec("Developer", "builds things", "private agreement", "gpt-4o"),
        );

        let member = spec.member_spec("dev").unwrap();
        assert_eq!(member.base.member_name, "dev");
        assert_eq!(member.base.display_name, "Developer");
        assert_eq!(member.base.desc, "builds things");
        assert_eq!(member.base.prompt, "private agreement");
        assert_eq!(member.base.model_name.as_deref(), Some("gpt-4o"));
        assert_eq!(member.role_type, TeamRole::Teammate);
    }

    #[test]
    fn member_spec_derivation_handles_a_missing_or_empty_card() {
        let mut spec = minimal_spec();
        spec.agents
            .insert("bare".to_owned(), DeepAgentSpec::default());
        let member = spec.member_spec("bare").unwrap();
        assert_eq!(member.base.display_name, "bare");
        assert_eq!(member.base.desc, "");
        assert_eq!(member.base.prompt, "");
        assert_eq!(member.base.model_name, None);

        let mut spec = minimal_spec();
        spec.agents.insert(
            "nameless".to_owned(),
            agent_spec("", "no name", "prompt", "m"),
        );
        let member = spec.member_spec("nameless").unwrap();
        assert_eq!(member.base.display_name, "nameless");
        assert_eq!(member.base.desc, "no name");
    }

    #[test]
    fn member_spec_falls_back_to_the_teammate_blueprint() {
        let mut spec = minimal_spec();
        spec.agents.insert(
            TEAMMATE_AGENT_KEY.to_owned(),
            agent_spec("Teammate", "shared", "shared prompt", "gpt-4o-mini"),
        );

        let member = spec.member_spec("dev-1").unwrap();
        assert_eq!(member.base.member_name, "dev-1");
        assert_eq!(member.base.display_name, "Teammate");
        assert_eq!(member.base.desc, "shared");
        assert_eq!(member.base.prompt, "shared prompt");
        assert_eq!(member.base.model_name.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(member.role_type, TeamRole::Teammate);
        assert!(spec.teammate_defaults().is_some());
    }

    #[test]
    fn member_spec_returns_the_leader_role_for_the_leader_name() {
        let mut spec = minimal_spec();
        spec.agents.insert(
            DEFAULT_LEADER_MEMBER_NAME.to_owned(),
            agent_spec("Chief", "leads", "leader prompt", "gpt-4o"),
        );

        let leader = spec.member_spec(DEFAULT_LEADER_MEMBER_NAME).unwrap();
        assert_eq!(leader.role_type, TeamRole::Leader);
        assert_eq!(leader.base.display_name, "Chief");
        assert_eq!(leader.base.prompt, "leader prompt");
    }

    #[test]
    fn member_spec_is_none_for_an_unknown_name() {
        assert!(minimal_spec().member_spec("ghost").is_none());
        assert_eq!(minimal_spec().teammate_defaults(), None);
    }

    #[test]
    fn member_name_pattern_matches_the_contract() {
        for valid in ["dev", "dev-1", "dev_1", "a", "a1", "team2-member_3"] {
            assert!(TeamAgentSpec::is_member_name_valid(valid), "{valid}");
        }
        for invalid in [
            "", "Dev", "1dev", "-dev", "_dev", "dev 1", "dev.1", "dev/1", "dév",
        ] {
            assert!(!TeamAgentSpec::is_member_name_valid(invalid), "{invalid}");
        }
        // The pattern is syntactic; reserved names are rejected by `validate` instead.
        for reserved in RESERVED_MEMBER_NAMES {
            assert!(TeamAgentSpec::is_member_name_valid(reserved), "{reserved}");
        }
    }
}
