//! Roster entries: leader, member and model-pool specs.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/blueprint.py` (`LeaderSpec`,
//! `TinyAgentSpec`), `openjiuwen/agent_teams/schema/team.py` (`TeamGoalSpec`, `MemberSpecBase`,
//! `TeamMemberSpec`, `BridgeMemberSpec`, `ExternalCliMemberSpec`) and
//! `openjiuwen/agent_teams/models/pool.py` (`ModelPoolEntry`).
//!
//! These are cold, serializable specifications: a blueprint names its members and the model
//! endpoints they may draw on, and the runtime materializes them later. Wire shapes matter —
//! [`MemberSpecBase`] is flattened into [`TeamMemberSpec`] and into the bridge/external payloads,
//! and [`PredefinedMemberSpec`] is internally tagged by `role_type`.
//!
//! Defaults follow Python except where `src/schema/CONTEXT.md` documents a deliberate deviation:
//! [`LeaderSpec::desc`] ships empty (Python injects an i18n string at construction), so callers
//! that need a human-facing default use [`LeaderSpec::desc_or_default`].

use std::collections::BTreeMap;

use openjiuwen_common::security::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::team::TeamRole;
use crate::constants::{DEFAULT_LEADER_DISPLAY_NAME, DEFAULT_LEADER_MEMBER_NAME};

/// Human-facing leader description used when [`LeaderSpec::desc`] is empty.
///
/// Python injects the same text from its i18n catalogue (`blueprint.default_desc`); the Rust port
/// ships the English string and never reads an i18n table from this module.
pub const LEADER_DEFAULT_DESC: &str = "Team leader";

/// Bridge mailbox wrapping mode used when a bridge spec does not name one
/// (Python `BridgeMailboxInjectMode.PASSTHROUGH`).
const DEFAULT_MAILBOX_INJECT_MODE: &str = "passthrough";

/// Default name of a team-scoped tiny agent.
const DEFAULT_TINY_AGENT_NAME: &str = "tiny";

/// Default iteration budget of a tiny agent.
const DEFAULT_TINY_MAX_ITERATIONS: usize = 6;

/// Default self-driven continuation budget of a goal (Python `TeamGoalSpec.max_attempts = 10`).
const DEFAULT_GOAL_MAX_ATTEMPTS: u32 = 10;

/// Identity of the team leader.
///
/// Python parity: `LeaderSpec`. Python inherits `prompt`/`model_name` from `MemberSpecBase`; the
/// Rust port keeps only the leader-facing identity and reads the prompt from
/// `agents["leader"]` instead (see `src/schema/CONTEXT.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LeaderSpec {
    /// Member name the team addresses the leader by; defaults to `team_leader`.
    pub member_name: String,
    /// Display name shown in the roster; defaults to `Team Leader`.
    pub display_name: String,
    /// Public blurb shown in other members' roster. Defaults to `""`; see
    /// [`LeaderSpec::desc_or_default`].
    pub desc: String,
    /// Goal mode for the leader; `None` keeps it off.
    pub goal: Option<TeamGoalSpec>,
}

impl Default for LeaderSpec {
    fn default() -> Self {
        Self {
            member_name: DEFAULT_LEADER_MEMBER_NAME.to_owned(),
            display_name: DEFAULT_LEADER_DISPLAY_NAME.to_owned(),
            desc: String::new(),
            goal: None,
        }
    }
}

impl LeaderSpec {
    /// Returns [`LeaderSpec::desc`], or [`LEADER_DEFAULT_DESC`] when it is empty.
    #[must_use]
    pub fn desc_or_default(&self) -> &str {
        if self.desc.is_empty() {
            LEADER_DEFAULT_DESC
        } else {
            &self.desc
        }
    }
}

/// How a goal-mode member's completion claim is judged (Python
/// `TeamGoalSpec.strategy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalStrategy {
    /// Trust the member's own report; the submission check is off.
    AgentReport,
    /// Always check a claim against the transcript with an isolated model call.
    Transcript,
    /// Trust "still working" reports; verify "done"/"blocked" claims (default).
    Hybrid,
}

impl Default for GoalStrategy {
    /// Python default: [`GoalStrategy::Hybrid`].
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Goal-mode settings for a leader or for the team's members.
///
/// Python parity: `TeamGoalSpec`. The limits bound how long a member may keep working on a task
/// (or a leader on the user's request) before the goal blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamGoalSpec {
    /// How completion is judged.
    pub strategy: GoalStrategy,
    /// Self-driven continuation rounds allowed before the goal blocks; `None` removes the limit.
    pub max_attempts: Option<u32>,
    /// Total tokens goal rounds may spend before the goal blocks; `None` means no budget.
    pub token_budget: Option<u64>,
}

impl Default for TeamGoalSpec {
    fn default() -> Self {
        Self {
            strategy: GoalStrategy::Hybrid,
            max_attempts: Some(DEFAULT_GOAL_MAX_ATTEMPTS),
            token_budget: None,
        }
    }
}

/// Shared identity surface of every team member (leader, teammate, bridge).
///
/// Python parity: `MemberSpecBase`. The two description fields have disjoint destinations:
/// [`MemberSpecBase::desc`] is public (roster, `list_members`) and
/// [`MemberSpecBase::prompt`] is private (only this member's own system prompt).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSpecBase {
    /// Member name the team addresses this member by.
    pub member_name: String,
    /// Display name shown in the roster.
    pub display_name: String,
    /// Public blurb shown to every other member.
    pub desc: String,
    /// Private working agreement injected only into this member's own prompt.
    pub prompt: String,
    /// Optional pool `model_name` to allocate from when the team pool is configured.
    pub model_name: Option<String>,
}

impl MemberSpecBase {
    /// Returns [`MemberSpecBase::display_name`], or [`MemberSpecBase::member_name`] when it is
    /// empty.
    #[must_use]
    pub fn display_name_or_default(&self) -> &str {
        if self.display_name.is_empty() {
            &self.member_name
        } else {
            &self.display_name
        }
    }
}

/// Declarative input for pre-defining a team member.
///
/// Python parity: `TeamMemberSpec`. Only meaningful for `predefined_members` at team creation
/// time; spawn and restart paths read the member row instead. [`MemberSpecBase`] is flattened, so
/// its fields sit at the top level of the JSON object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamMemberSpec {
    /// Shared identity fields, flattened into this object's JSON.
    #[serde(flatten)]
    pub base: MemberSpecBase,
    /// Role this entry declares; defaults to [`TeamRole::Teammate`].
    pub role_type: TeamRole,
    /// Per-member goal-mode switch; `None` follows the team setting.
    pub goal: Option<bool>,
}

impl Default for TeamMemberSpec {
    /// Python default: `role_type = teammate` (note this differs from [`TeamRole`]'s own default,
    /// which is `leader`).
    fn default() -> Self {
        Self {
            base: MemberSpecBase::default(),
            role_type: TeamRole::Teammate,
            goal: None,
        }
    }
}

/// A pre-defined roster entry, discriminated by its `role_type` tag.
///
/// Python parity: the `PredefinedMemberSpec` discriminated union of `TeamMemberSpec` /
/// `BridgeMemberSpec` / `ExternalCliMemberSpec`. The tag values are the [`TeamRole`] wire strings,
/// so a missing `role_type` is a deserialization error (Python's documented `teammate` default is
/// not emulated).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role_type", rename_all = "snake_case")]
pub enum PredefinedMemberSpec {
    /// An ordinary teammate.
    Teammate(MemberSpecBase),
    /// The avatar channel of a real human joining the team.
    HumanAgent(MemberSpecBase),
    /// A directed human that is addressed but never polls the board.
    PassiveHuman(MemberSpecBase),
    /// A local teammate paired with a remote agent behind a text protocol.
    BridgeAgent(MemberSpecBaseBridge),
    /// A member whose work is performed by an external CLI process.
    ExternalCli(MemberSpecBaseExternal),
}

impl PredefinedMemberSpec {
    /// Returns the shared identity fields of this entry.
    #[must_use]
    pub fn base(&self) -> &MemberSpecBase {
        match self {
            Self::Teammate(base) | Self::HumanAgent(base) | Self::PassiveHuman(base) => base,
            Self::BridgeAgent(bridge) => &bridge.base,
            Self::ExternalCli(external) => &external.base,
        }
    }

    /// Returns the member name of this entry.
    #[must_use]
    pub fn member_name(&self) -> &str {
        &self.base().member_name
    }

    /// Returns the role this entry declares.
    #[must_use]
    pub const fn role_type(&self) -> TeamRole {
        match self {
            Self::Teammate(_) => TeamRole::Teammate,
            Self::HumanAgent(_) => TeamRole::HumanAgent,
            Self::PassiveHuman(_) => TeamRole::PassiveHuman,
            Self::BridgeAgent(_) => TeamRole::BridgeAgent,
            Self::ExternalCli(_) => TeamRole::ExternalCli,
        }
    }

    /// Converts this entry into a plain [`TeamMemberSpec`].
    ///
    /// Bridge and external-CLI entries collapse to a teammate: their role-specific payload is
    /// data-only in this port, so it is dropped here exactly as Python's
    /// `TeamMemberSpec`-shaped view does.
    #[must_use]
    pub fn into_member_spec(self) -> TeamMemberSpec {
        match self {
            Self::Teammate(base) => TeamMemberSpec {
                base,
                role_type: TeamRole::Teammate,
                goal: None,
            },
            Self::HumanAgent(base) => TeamMemberSpec {
                base,
                role_type: TeamRole::HumanAgent,
                goal: None,
            },
            Self::PassiveHuman(base) => TeamMemberSpec {
                base,
                role_type: TeamRole::PassiveHuman,
                goal: None,
            },
            Self::BridgeAgent(bridge) => TeamMemberSpec {
                base: bridge.base,
                role_type: TeamRole::Teammate,
                goal: None,
            },
            Self::ExternalCli(external) => TeamMemberSpec {
                base: external.base,
                role_type: TeamRole::Teammate,
                goal: None,
            },
        }
    }
}

impl From<TeamMemberSpec> for PredefinedMemberSpec {
    /// Projects a member spec onto the closest [`PredefinedMemberSpec`] variant.
    ///
    /// Python drops the leader/bridge/worker/external-CLI distinction here, keeping only the
    /// human roles and folding everything else into a teammate.
    fn from(spec: TeamMemberSpec) -> Self {
        match spec.role_type {
            TeamRole::HumanAgent => Self::HumanAgent(spec.base),
            TeamRole::PassiveHuman => Self::PassiveHuman(spec.base),
            TeamRole::Leader
            | TeamRole::Teammate
            | TeamRole::BridgeAgent
            | TeamRole::Worker
            | TeamRole::ExternalCli => Self::Teammate(spec.base),
        }
    }
}

/// Predefined-member payload for a bridge agent.
///
/// Python parity: `BridgeMemberSpec`. A bridge agent is a full local teammate paired with a remote
/// independent agent reachable over a pure-text protocol; this port carries the fields so a plan
/// written elsewhere round-trips, but no adapter is wired yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSpecBaseBridge {
    /// Shared identity fields, flattened into this object's JSON.
    #[serde(flatten)]
    pub base: MemberSpecBase,
    /// How a team-side message is wrapped before being relayed; defaults to `passthrough`.
    pub mailbox_inject_mode: String,
    /// Protocol identifier (`a2a`, `acp`, `claudecode`, …); empty means no adapter is wired.
    pub protocol: String,
    /// Free-form adapter parameters (endpoint, auth handle, relay timeout, …), never inspected.
    pub adapter_config: BTreeMap<String, Value>,
}

impl Default for MemberSpecBaseBridge {
    fn default() -> Self {
        Self {
            base: MemberSpecBase::default(),
            mailbox_inject_mode: DEFAULT_MAILBOX_INJECT_MODE.to_owned(),
            protocol: String::new(),
            adapter_config: BTreeMap::new(),
        }
    }
}

/// Predefined-member payload for an external-CLI member.
///
/// Python parity: `ExternalCliMemberSpec`; the Rust port keeps the launch subset (which CLI agent,
/// where its binary lives, the command line and the environment) and not the full
/// `ExternalCliAgentSpec` model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSpecBaseExternal {
    /// Shared identity fields, flattened into this object's JSON.
    #[serde(flatten)]
    pub base: MemberSpecBase,
    /// External CLI agent kind (`claude`, `codex`, …); empty when the spec names none.
    pub cli_agent: String,
    /// Explicit path to the CLI binary, when it is not resolved from `PATH`.
    pub cli_path: Option<String>,
    /// Extra command-line arguments appended to the CLI invocation.
    pub command: Vec<String>,
    /// Environment variables injected into the external process.
    pub env: BTreeMap<String, String>,
}

/// Team-scoped tiny-agent specification.
///
/// Python parity: `TinyAgentSpec` — a tiny agent carries only a system prompt, a model and an
/// optional structured-output schema; it has no tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TinyAgentSpec {
    /// System prompt handed to the tiny agent.
    pub system_prompt: String,
    /// Model name resolved against the team model pool.
    pub model_name: String,
    /// Registry name of this tiny agent; defaults to `tiny`.
    pub name: String,
    /// Iteration budget of one tiny-agent call; defaults to 6.
    pub max_iterations: usize,
    /// Whether the tiny agent runs behind the security rail; defaults to `false`.
    pub enable_security_rail: bool,
    /// Optional default JSON Schema applied to structured output on every call.
    pub default_schema: Option<Value>,
}

impl Default for TinyAgentSpec {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model_name: String::new(),
            name: DEFAULT_TINY_AGENT_NAME.to_owned(),
            max_iterations: DEFAULT_TINY_MAX_ITERATIONS,
            enable_security_rail: false,
            default_schema: None,
        }
    }
}

/// One LLM endpoint of a team's allocation pool.
///
/// Python parity: `ModelPoolEntry`. [`ModelPoolEntry::model_id`] is a process-local client
/// identity: it is regenerated every time a pool is loaded from a spec and never persisted, while
/// `(model_name, group_index)` is the semantic identity stored with a member.
/// [`ModelPoolEntry::api_key`] is a `SecretString`, so it round-trips as plaintext JSON but never
/// leaks through `Debug`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelPoolEntry {
    /// Model name served by this endpoint.
    pub model_name: String,
    /// Credential used to reach the endpoint.
    pub api_key: SecretString,
    /// Base URL of the endpoint.
    pub api_base_url: String,
    /// Provider key (`openai`, `anthropic`, …).
    pub api_provider: String,
    /// Optional human-readable description of the endpoint.
    pub description: Option<String>,
    /// Process-local client identity; a fresh 32-character lowercase hex id by default.
    pub model_id: String,
    /// Free-form extension payload (client/request overrides and allocator policy hints).
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ModelPoolEntry {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            api_key: SecretString::default(),
            api_base_url: String::new(),
            api_provider: String::new(),
            description: None,
            model_id: new_model_id(),
            metadata: BTreeMap::new(),
        }
    }
}

impl ModelPoolEntry {
    /// Builds an entry for one endpoint, generating a fresh [`ModelPoolEntry::model_id`].
    #[must_use]
    pub fn new(
        model_name: impl Into<String>,
        api_provider: impl Into<String>,
        api_base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            api_key: SecretString::new(api_key),
            api_base_url: api_base_url.into(),
            api_provider: api_provider.into(),
            description: None,
            model_id: new_model_id(),
            metadata: BTreeMap::new(),
        }
    }
}

/// Generates a process-local model id: 32 lowercase hex characters, like Python's `uuid4().hex`.
fn new_model_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Returns a member base filled with every field, for round-trip tests.
    fn full_base() -> MemberSpecBase {
        MemberSpecBase {
            member_name: "dev-1".to_owned(),
            display_name: "Developer One".to_owned(),
            desc: "builds things".to_owned(),
            prompt: "private agreement".to_owned(),
            model_name: Some("gpt-4o".to_owned()),
        }
    }

    #[test]
    fn goal_strategy_wire_values_match_python() {
        let cases = [
            (GoalStrategy::AgentReport, "agent_report"),
            (GoalStrategy::Transcript, "transcript"),
            (GoalStrategy::Hybrid, "hybrid"),
        ];
        for (strategy, wire) in cases {
            assert_eq!(serde_json::to_value(strategy).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<GoalStrategy>(json!(wire)).unwrap(),
                strategy
            );
        }
        assert_eq!(GoalStrategy::default(), GoalStrategy::Hybrid);
    }

    #[test]
    fn leader_spec_json_defaults_match_python() {
        let leader = LeaderSpec::default();
        assert_eq!(leader.member_name, "team_leader");
        assert_eq!(leader.display_name, "Team Leader");
        assert_eq!(leader.desc, "");
        assert_eq!(leader.goal, None);
        assert_eq!(
            serde_json::to_value(&leader).unwrap(),
            json!({
                "member_name": "team_leader",
                "display_name": "Team Leader",
                "desc": "",
                "goal": null,
            })
        );
        assert_eq!(
            serde_json::from_value::<LeaderSpec>(json!({})).unwrap(),
            leader
        );
    }

    #[test]
    fn leader_desc_or_default_falls_back() {
        assert_eq!(LeaderSpec::default().desc_or_default(), LEADER_DEFAULT_DESC);
        assert_eq!(LEADER_DEFAULT_DESC, "Team leader");
        let leader = LeaderSpec {
            desc: "keeps the team honest".to_owned(),
            ..LeaderSpec::default()
        };
        assert_eq!(leader.desc_or_default(), "keeps the team honest");
    }

    #[test]
    fn team_goal_spec_json_defaults_match_python() {
        let goal = TeamGoalSpec::default();
        assert_eq!(goal.strategy, GoalStrategy::Hybrid);
        assert_eq!(goal.max_attempts, Some(10));
        assert_eq!(goal.token_budget, None);
        assert_eq!(
            serde_json::to_value(&goal).unwrap(),
            json!({"strategy": "hybrid", "max_attempts": 10, "token_budget": null})
        );
        assert_eq!(
            serde_json::from_value::<TeamGoalSpec>(json!({})).unwrap(),
            goal
        );

        let unbounded: TeamGoalSpec =
            serde_json::from_value(json!({"max_attempts": null, "token_budget": 5000})).unwrap();
        assert_eq!(unbounded.max_attempts, None);
        assert_eq!(unbounded.token_budget, Some(5000));
    }

    #[test]
    fn member_spec_base_defaults_and_display_name_fallback() {
        let base = MemberSpecBase::default();
        assert_eq!(
            serde_json::to_value(&base).unwrap(),
            json!({
                "member_name": "",
                "display_name": "",
                "desc": "",
                "prompt": "",
                "model_name": null,
            })
        );
        assert_eq!(base.display_name_or_default(), "");
        let named = MemberSpecBase {
            member_name: "dev-1".to_owned(),
            ..MemberSpecBase::default()
        };
        assert_eq!(named.display_name_or_default(), "dev-1");
        assert_eq!(full_base().display_name_or_default(), "Developer One");
    }

    #[test]
    fn team_member_spec_flattens_base_and_round_trips() {
        let spec = TeamMemberSpec {
            base: full_base(),
            role_type: TeamRole::HumanAgent,
            goal: Some(false),
        };
        assert_eq!(
            serde_json::to_value(&spec).unwrap(),
            json!({
                "member_name": "dev-1",
                "display_name": "Developer One",
                "desc": "builds things",
                "prompt": "private agreement",
                "model_name": "gpt-4o",
                "role_type": "human_agent",
                "goal": false,
            })
        );
        let reparsed: TeamMemberSpec = serde_json::from_value(json!({
            "member_name": "dev-1",
            "display_name": "Developer One",
            "desc": "builds things",
            "prompt": "private agreement",
            "model_name": "gpt-4o",
            "role_type": "human_agent",
            "goal": false,
        }))
        .unwrap();
        assert_eq!(reparsed, spec);

        // Defaults: `role_type` is `teammate`, everything else empty.
        let bare: TeamMemberSpec = serde_json::from_value(json!({"member_name": "dev-1"})).unwrap();
        assert_eq!(bare.role_type, TeamRole::Teammate);
        assert_eq!(bare.goal, None);
        assert_eq!(bare.base.member_name, "dev-1");
        assert_eq!(bare.base.display_name, "");
        assert_eq!(TeamMemberSpec::default().role_type, TeamRole::Teammate);
    }

    #[test]
    fn team_member_spec_projects_onto_predefined_roles() {
        let cases = [
            (TeamRole::Teammate, TeamRole::Teammate),
            (TeamRole::HumanAgent, TeamRole::HumanAgent),
            (TeamRole::PassiveHuman, TeamRole::PassiveHuman),
            (TeamRole::Leader, TeamRole::Teammate),
            (TeamRole::Worker, TeamRole::Teammate),
            (TeamRole::BridgeAgent, TeamRole::Teammate),
            (TeamRole::ExternalCli, TeamRole::Teammate),
        ];
        for (role_type, expected) in cases {
            let spec = TeamMemberSpec {
                base: full_base(),
                role_type,
                goal: None,
            };
            let predefined = PredefinedMemberSpec::from(spec);
            assert_eq!(predefined.role_type(), expected, "{role_type:?}");
            assert_eq!(predefined.base(), &full_base(), "{role_type:?}");
        }
    }

    #[test]
    fn predefined_member_spec_round_trips_every_variant() {
        let variants = [
            (
                PredefinedMemberSpec::Teammate(full_base()),
                json!({
                    "role_type": "teammate",
                    "member_name": "dev-1",
                    "display_name": "Developer One",
                    "desc": "builds things",
                    "prompt": "private agreement",
                    "model_name": "gpt-4o",
                }),
            ),
            (
                PredefinedMemberSpec::HumanAgent(full_base()),
                json!({
                    "role_type": "human_agent",
                    "member_name": "dev-1",
                    "display_name": "Developer One",
                    "desc": "builds things",
                    "prompt": "private agreement",
                    "model_name": "gpt-4o",
                }),
            ),
            (
                PredefinedMemberSpec::PassiveHuman(full_base()),
                json!({
                    "role_type": "passive_human",
                    "member_name": "dev-1",
                    "display_name": "Developer One",
                    "desc": "builds things",
                    "prompt": "private agreement",
                    "model_name": "gpt-4o",
                }),
            ),
            (
                PredefinedMemberSpec::BridgeAgent(MemberSpecBaseBridge {
                    base: full_base(),
                    mailbox_inject_mode: "rephrase".to_owned(),
                    protocol: "a2a".to_owned(),
                    adapter_config: BTreeMap::from([("endpoint".to_owned(), json!("http://x"))]),
                }),
                json!({
                    "role_type": "bridge_agent",
                    "member_name": "dev-1",
                    "display_name": "Developer One",
                    "desc": "builds things",
                    "prompt": "private agreement",
                    "model_name": "gpt-4o",
                    "mailbox_inject_mode": "rephrase",
                    "protocol": "a2a",
                    "adapter_config": {"endpoint": "http://x"},
                }),
            ),
            (
                PredefinedMemberSpec::ExternalCli(MemberSpecBaseExternal {
                    base: full_base(),
                    cli_agent: "claude".to_owned(),
                    cli_path: Some("/usr/local/bin/claude".to_owned()),
                    command: vec!["--print".to_owned()],
                    env: BTreeMap::from([("API_KEY".to_owned(), "k".to_owned())]),
                }),
                json!({
                    "role_type": "external_cli",
                    "member_name": "dev-1",
                    "display_name": "Developer One",
                    "desc": "builds things",
                    "prompt": "private agreement",
                    "model_name": "gpt-4o",
                    "cli_agent": "claude",
                    "cli_path": "/usr/local/bin/claude",
                    "command": ["--print"],
                    "env": {"API_KEY": "k"},
                }),
            ),
        ];

        for (spec, wire) in variants {
            assert_eq!(serde_json::to_value(&spec).unwrap(), wire, "{spec:?}");
            let reparsed: PredefinedMemberSpec = serde_json::from_value(wire).unwrap();
            assert_eq!(reparsed, spec);
        }
    }

    #[test]
    fn predefined_member_spec_helpers_and_conversion() {
        let bridge = PredefinedMemberSpec::BridgeAgent(MemberSpecBaseBridge {
            base: full_base(),
            ..MemberSpecBaseBridge::default()
        });
        assert_eq!(
            bridge.base(),
            &full_base(),
            "bridge base is reached through the flattened payload"
        );
        assert_eq!(bridge.member_name(), "dev-1");
        assert_eq!(bridge.role_type(), TeamRole::BridgeAgent);
        let teammate = bridge.into_member_spec();
        assert_eq!(teammate.role_type, TeamRole::Teammate);
        assert_eq!(teammate.base, full_base());
        assert_eq!(teammate.goal, None);

        let human = PredefinedMemberSpec::HumanAgent(full_base()).into_member_spec();
        assert_eq!(human.role_type, TeamRole::HumanAgent);

        let external = PredefinedMemberSpec::ExternalCli(MemberSpecBaseExternal {
            base: full_base(),
            ..MemberSpecBaseExternal::default()
        });
        assert_eq!(external.role_type(), TeamRole::ExternalCli);
        assert_eq!(external.into_member_spec().role_type, TeamRole::Teammate);
    }

    #[test]
    fn predefined_member_spec_bridge_and_external_defaults() {
        let bridge = MemberSpecBaseBridge::default();
        assert_eq!(bridge.mailbox_inject_mode, "passthrough");
        assert_eq!(bridge.protocol, "");
        assert_eq!(bridge.adapter_config, BTreeMap::new());
        assert_eq!(
            serde_json::to_value(&bridge).unwrap(),
            json!({
                "member_name": "",
                "display_name": "",
                "desc": "",
                "prompt": "",
                "model_name": null,
                "mailbox_inject_mode": "passthrough",
                "protocol": "",
                "adapter_config": {},
            })
        );

        let external = MemberSpecBaseExternal::default();
        assert_eq!(external.cli_agent, "");
        assert_eq!(external.cli_path, None);
        assert_eq!(external.command, Vec::<String>::new());
        assert_eq!(external.env, BTreeMap::new());
        assert_eq!(
            serde_json::to_value(&external).unwrap(),
            json!({
                "member_name": "",
                "display_name": "",
                "desc": "",
                "prompt": "",
                "model_name": null,
                "cli_agent": "",
                "cli_path": null,
                "command": [],
                "env": {},
            })
        );
    }

    #[test]
    fn predefined_member_spec_requires_role_type_tag() {
        let error = serde_json::from_value::<PredefinedMemberSpec>(json!({"member_name": "dev-1"}));
        assert!(error.is_err(), "a missing role_type tag must not default");
        assert!(
            serde_json::from_value::<PredefinedMemberSpec>(json!({"role_type": "worker"})).is_err(),
            "worker has no predefined payload"
        );
    }

    #[test]
    fn tiny_agent_spec_json_defaults_match_python() {
        let tiny = TinyAgentSpec::default();
        assert_eq!(tiny.name, "tiny");
        assert_eq!(tiny.max_iterations, 6);
        assert!(!tiny.enable_security_rail);
        assert_eq!(tiny.default_schema, None);
        assert_eq!(
            serde_json::to_value(&tiny).unwrap(),
            json!({
                "system_prompt": "",
                "model_name": "",
                "name": "tiny",
                "max_iterations": 6,
                "enable_security_rail": false,
                "default_schema": null,
            })
        );

        let parsed: TinyAgentSpec = serde_json::from_value(json!({
            "system_prompt": "judge this",
            "model_name": "gpt-4o-mini",
            "default_schema": {"type": "object"},
        }))
        .unwrap();
        assert_eq!(parsed.name, "tiny");
        assert_eq!(parsed.max_iterations, 6);
        assert_eq!(parsed.default_schema, Some(json!({"type": "object"})));
    }

    #[test]
    fn model_pool_entry_new_and_default() {
        let entry = ModelPoolEntry::new("gpt-4o", "openai", "https://api.openai.com/v1", "sk-1");
        assert_eq!(entry.model_name, "gpt-4o");
        assert_eq!(entry.api_provider, "openai");
        assert_eq!(entry.api_base_url, "https://api.openai.com/v1");
        assert_eq!(entry.api_key.expose_secret(), "sk-1");
        assert_eq!(entry.description, None);
        assert_eq!(entry.metadata, BTreeMap::new());
        assert_eq!(entry.model_id.len(), 32);
        assert!(
            entry
                .model_id
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "model_id must be lowercase hex: {}",
            entry.model_id
        );

        let default = ModelPoolEntry::default();
        assert_eq!(default.model_id.len(), 32);
        assert_ne!(default.model_id, entry.model_id, "ids are fresh per entry");
        assert_eq!(default.api_key.expose_secret(), "");
    }

    #[test]
    fn model_pool_entry_debug_redacts_the_api_key() {
        let entry = ModelPoolEntry::new("gpt-4o", "openai", "https://api.openai.com/v1", "sk-1");
        let rendered = format!("{entry:?}");
        assert!(!rendered.contains("sk-1"), "{rendered}");
        assert!(rendered.contains("<REDACTED>"), "{rendered}");
    }

    #[test]
    fn model_pool_entry_round_trips_and_tolerates_unknown_fields() {
        let wire = json!({
            "model_name": "gpt-4o",
            "api_key": "sk-1",
            "api_base_url": "https://api.openai.com/v1",
            "api_provider": "openai",
            "description": "primary",
            "model_id": "0123456789abcdef0123456789abcdef",
            "metadata": {"client": {"timeout": 30}},
            "unknown_python_field": true,
        });
        let entry: ModelPoolEntry = serde_json::from_value(wire).unwrap();
        assert_eq!(entry.description.as_deref(), Some("primary"));
        assert_eq!(entry.model_id, "0123456789abcdef0123456789abcdef");
        assert_eq!(entry.metadata.get("client"), Some(&json!({"timeout": 30})));
        let reserialized = serde_json::to_value(&entry).unwrap();
        assert_eq!(reserialized["api_key"], json!("sk-1"));
        assert!(reserialized.get("unknown_python_field").is_none());
        let reparsed: ModelPoolEntry = serde_json::from_value(reserialized).unwrap();
        assert_eq!(reparsed, entry);
    }

    #[test]
    fn every_struct_tolerates_unknown_fields() {
        let unknown = json!({"unknown_python_field": 1});
        assert_eq!(
            serde_json::from_value::<LeaderSpec>(unknown.clone()).unwrap(),
            LeaderSpec::default()
        );
        assert_eq!(
            serde_json::from_value::<TeamGoalSpec>(unknown.clone()).unwrap(),
            TeamGoalSpec::default()
        );
        assert_eq!(
            serde_json::from_value::<MemberSpecBase>(unknown.clone()).unwrap(),
            MemberSpecBase::default()
        );
        assert_eq!(
            serde_json::from_value::<MemberSpecBaseBridge>(unknown.clone()).unwrap(),
            MemberSpecBaseBridge::default()
        );
        assert_eq!(
            serde_json::from_value::<MemberSpecBaseExternal>(unknown.clone()).unwrap(),
            MemberSpecBaseExternal::default()
        );
        assert_eq!(
            serde_json::from_value::<TinyAgentSpec>(unknown.clone())
                .unwrap()
                .name,
            TinyAgentSpec::default().name
        );
        let entry: ModelPoolEntry = serde_json::from_value(unknown).unwrap();
        assert_eq!(entry.model_name, "");
    }
}
