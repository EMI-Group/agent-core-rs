# openjiuwen-agent-teams/src/schema

## Intent

The data model of the crate: every type that is serialized, persisted, put on the event bus or streamed to a caller.
Python parity source: `openjiuwen/agent_teams/schema/` (`blueprint.py`, `team.py`, `events.py`, `status.py`, `task.py`, `stream.py`, `deep_agent_spec.py`) plus `harness/schema/deep_agent_spec.py`.
Every other module of the crate codes against these types, so the shapes here are a seam: changing one requires updating `../CONTEXT.md` and every dependent node.
Serialized JSON names, defaults, and status values follow Python where the Rust port implements the capability; Rust-specific fields are listed under Notes for Agents.

## API Surface

All types derive `Debug, Clone, PartialEq, Serialize, Deserialize`, carry `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default collections to empty.
Enums that are persisted are unit-only, serialize to lower_snake_case strings, and implement `Default` for their Python default (required by the struct-level `default`).
Every public item carries a doc comment (`missing_docs` is a warning and clippy runs with `-D warnings`).

### `status.rs` — lifecycle vocabulary

```rust
pub enum MemberStatus { Unstarted, Starting, Ready, Busy, Paused, Stopped, Restarting,
                        ShutdownRequested, ShutDown, Dormant, Error }   // ShutDown => "shut_down"
pub enum ExecutionStatus { Idle, Starting, Running, CancelRequested, Cancelling, Cancelled,
                           Completing, Completed, Failed, TimedOut }
pub enum TaskStatus { Pending, Blocked, Planning, InProgress, InReview, Completed, Cancelled }
pub enum MemberLifetime { Shutdown, Dormant }        // Python Literal["shutdown","dormant"]
pub enum RoundOutcome { Started, Finished, Aborted, Failed }  // "started"|"finished"|"aborted"|"failed"

impl MemberStatus {
    pub const fn as_str(self) -> &'static str;
    pub const fn is_departed(self) -> bool;      // {shutdown_requested, shut_down}
    pub const fn is_unreachable(self) -> bool;   // {shut_down}
    pub const fn is_settled(self) -> bool;       // {ready, paused, stopped, shut_down, dormant}
    pub const fn is_quiescent(self) -> bool;     // settled + {unstarted, error}
    pub const fn can_transition_to(self, to: Self) -> bool;
    pub fn is_valid_transition(from: Self, to: Self) -> bool;   // associated, same table
}
impl ExecutionStatus { pub const fn as_str(self) -> &'static str; pub const fn can_transition_to(self, to: Self) -> bool;
                       pub fn is_valid_transition(from: Self, to: Self) -> bool; }
impl TaskStatus {
    pub const fn as_str(self) -> &'static str;
    pub const fn is_terminal(self) -> bool;        // completed | cancelled
    pub const fn is_claimable(self) -> bool;       // pending
    pub const fn is_reassignable(self) -> bool;    // pending | blocked | in_progress
    pub const fn can_transition_to(self, to: Self) -> bool;
    pub fn is_valid_transition(from: Self, to: Self) -> bool;
}
pub static MEMBER_TRANSITIONS: &[(MemberStatus, &[MemberStatus])];
pub static EXECUTION_TRANSITIONS: &[(ExecutionStatus, &[ExecutionStatus])];
pub static TASK_TRANSITIONS: &[(TaskStatus, &[TaskStatus])];
pub const MEMBER_DEPARTED_STATUSES: [MemberStatus; 2];
pub const MEMBER_UNREACHABLE_STATUSES: [MemberStatus; 1];
pub const MEMBER_SETTLED_STATUSES: [MemberStatus; 5];
pub const MEMBER_QUIESCENT_STATUSES: [MemberStatus; 7];
pub const TASK_REASSIGNABLE_STATUSES: [TaskStatus; 3];
pub const TASK_TERMINAL_STATUSES: [TaskStatus; 2];
pub fn is_valid_member_transition(from: MemberStatus, to: MemberStatus) -> bool;
pub fn retire_target(lifetime: Option<MemberLifetime>, force: bool) -> MemberStatus;
pub fn round_transitions(outcome: RoundOutcome) -> Vec<ExecutionStatus>;
```

Transition tables are the data the helpers read; the helpers must never be a second copy of the rules.

- `MEMBER_TRANSITIONS` (Python `MEMBER_TRANSITIONS`): `unstarted -> {starting, ready, shut_down, error}`; `starting -> {ready, unstarted, shut_down, error}`; `ready -> {ready, busy, paused, stopped, shutdown_requested, shut_down, error}`; `busy -> {ready, paused, stopped, shutdown_requested, error}`; `paused -> {ready, restarting, stopped, shutdown_requested, shut_down, error}`; `stopped -> {ready, restarting, shutdown_requested, shut_down, error}`; `restarting -> {ready, stopped, error, shut_down}`; `shutdown_requested -> {shut_down, dormant, error}`; `shut_down -> {restarting}`; `dormant -> {restarting, shut_down}`; `error -> {restarting, ready, stopped, shutdown_requested, shut_down}`.
- `EXECUTION_TRANSITIONS` (Python `EXECUTION_TRANSITIONS`): `idle -> {starting}`; `starting -> {running, cancel_requested, cancelling, failed, timed_out}`; `running -> {cancel_requested, cancelling, completing, failed, timed_out}`; `cancel_requested -> {cancelling, cancelled, failed, timed_out}`; `cancelling -> {cancelled, failed, timed_out}`; `cancelled -> {idle}`; `completing -> {completed, failed, timed_out}`; `completed -> {idle}`; `failed -> {idle}`; `timed_out -> {idle}`.
- `TASK_TRANSITIONS` (Rust contract, narrower than Python — see Notes): `pending -> {blocked, planning, in_progress, cancelled}`; `blocked -> {pending, cancelled}`; `planning -> {in_progress, cancelled}`; `in_progress -> {in_review, completed, cancelled, planning}`; `in_review -> {completed, in_progress, cancelled}`; `completed -> {}`; `cancelled -> {}`.
- `retire_target(Some(Dormant), false) == Dormant`; every other input returns `ShutDown` (including `Some(Dormant)` with `force == true`).
- `round_transitions`: `Started -> [Starting, Running]`; `Finished -> [Completing, Completed, Idle]`; `Aborted -> [CancelRequested, Cancelling, Cancelled, Idle]`; `Failed -> [Failed, Idle]`.

### `deep_agent.rs` — member blueprint (Python `harness/schema/deep_agent_spec.py`)

```rust
pub struct ModelRef { pub name: Option<String>, pub model_name: Option<String>,
                      pub provider: Option<String>, pub params: BTreeMap<String, Value> }
impl ModelRef { pub fn resolved_name(&self) -> Option<&str>;   // model_name, else name }

pub struct DeepAgentSpec {
    pub model: Option<ModelRef>, pub card: Option<AgentCard>, pub system_prompt: Option<String>,
    pub tools: Vec<ToolSpec>, pub mcps: Vec<McpServerSpec>, pub subagents: Vec<SubAgentSpec>,
    pub rails: Vec<RailSpec>, pub enable_task_loop: bool /* true */, pub max_iterations: Option<usize>,
    pub workspace: Option<WorkspaceSpec>, pub cwd: Option<String>, pub project_root: Option<String>,
    pub skills: Option<Vec<String>>, pub language: Option<String>,
    pub approval_required_tools: Vec<String>, pub completion_timeout: f64 /* 600.0 */,
    pub enable_security_rail: bool /* true */, pub prompt_mode: Option<String>,
    #[serde(flatten)] pub extra: BTreeMap<String, Value>,
}
impl DeepAgentSpec {
    pub fn model_name(&self) -> Option<&str>;                          // model.resolved_name()
    pub fn with_language(self, language: impl Into<String>) -> Self;
}
// `Default` is hand-written: enable_task_loop = true, completion_timeout = 600.0,
// enable_security_rail = true, collections empty.

pub struct ToolSpec { pub name: String, pub description: String, pub input_schema: Value }
// Custom `Deserialize`: accepts a bare string, a builtin object `{"type","params"}` (name = type,
// description = params.description, input_schema = params.input_schema | params.parameters | {}),
// or a ToolCard-shaped object (name/description/input_schema | input_params). Unknown keys ignored.
// `Serialize` emits `{"name", "description", "input_schema"}`.

pub struct RailSpec { #[serde(rename = "type")] pub kind: String, pub params: BTreeMap<String, Value> }
pub struct McpServerSpec { pub name: String, pub transport: String /* "stdio" */,
                           pub command: Vec<String>, pub url: Option<String>,
                           pub env: BTreeMap<String, String> }
pub struct SubAgentSpec { pub agent_card: Option<AgentCard>, pub system_prompt: String,
                          pub tools: Vec<ToolSpec>, pub mcps: Vec<McpServerSpec>, pub name: Option<String> }
pub struct WorkspaceSpec { pub root_path: String /* "./" */, pub language: String /* DEFAULT_LANGUAGE */,
                           pub stable_base: bool /* false */ }
```

### `team.rs` — runtime mirror

```rust
pub enum TeamRole { Leader, Teammate, HumanAgent, PassiveHuman, BridgeAgent, Worker, ExternalCli }
impl TeamRole { pub const fn as_str(self) -> &'static str; pub const fn is_coordinated_member(self) -> bool; }
// Default = Leader; is_coordinated_member = teammate | external_cli.

pub struct TeamSpec { pub team_name: String, pub display_name: String,
    pub conversation_mode: ConversationMode, pub leader_member_name: Option<String>,
    pub language: Option<String>, pub dispatch_mode: DispatchMode,
    pub enable_member_messaging: bool, pub max_subtask_depth: usize,
    pub teammate_mode: MemberMode, pub model_pool: Vec<ModelPoolEntry>,
    pub model_pool_strategy: ModelPoolStrategy, pub workspace: Option<WorkspaceSpec>,
    pub evolution_enabled: bool /* false */, pub member_workspace_prefix: bool /* true */,
    pub metadata: BTreeMap<String, Value> }
impl TeamSpec { pub fn from_spec(spec: &TeamAgentSpec) -> Self; }   // materializes the mirror

pub struct TeamRuntimeContext { pub role: TeamRole, pub member_name: Option<String>,
    pub display_name: String, pub desc: String, pub prompt: String, pub team_spec: TeamSpec,
    pub workspace_root: String, pub member_model: Option<String>, pub language: String,
    pub agent_template_spec: Option<Value> }
impl TeamRuntimeContext { pub fn new(role: TeamRole, team_spec: TeamSpec) -> Self; }  // language from team_spec

pub struct TeamSnapshot { pub team_name: String, pub state: HarnessState,
    pub members: Vec<MemberSnapshot>, pub tasks: Vec<TaskSummary>, pub updated_at: u64 }
impl TeamSnapshot {
    pub fn new(team_name: impl Into<String>, state: HarnessState) -> Self;
    pub fn sort(&mut self);     // members by member_name, tasks by task_id
}
pub struct MemberSnapshot { pub member_name: String, pub display_name: String,
    pub role: TeamRole, pub status: MemberStatus, pub execution: ExecutionStatus }
```

`HarnessState` is `openjiuwen_harness_protocol::HarnessState`, re-exported by this module — never define a second enum.

### `member_spec.rs` — roster entries (Python `blueprint.py` LeaderSpec/TinyAgentSpec, `team.py` member specs, `models/pool.py`)

```rust
pub const LEADER_DEFAULT_DESC: &str = "Team leader";   // Python's i18n default, shipped as English

pub struct LeaderSpec { pub member_name: String /* DEFAULT_LEADER_MEMBER_NAME */,
                        pub display_name: String /* DEFAULT_LEADER_DISPLAY_NAME */, pub desc: String,
                        pub goal: Option<TeamGoalSpec> }
impl LeaderSpec { pub fn desc_or_default(&self) -> &str; }   // desc, else LEADER_DEFAULT_DESC

pub enum GoalStrategy { AgentReport, Transcript, Hybrid }   // default Hybrid
pub struct TeamGoalSpec { pub strategy: GoalStrategy, pub max_attempts: Option<u32> /* Some(10) */,
                          pub token_budget: Option<u64> }

pub struct MemberSpecBase { pub member_name: String, pub display_name: String, pub desc: String,
                            pub prompt: String, pub model_name: Option<String> }
impl MemberSpecBase { pub fn display_name_or_default(&self) -> &str;  // display_name, else member_name }

pub struct TeamMemberSpec { #[serde(flatten)] pub base: MemberSpecBase,
                            pub role_type: TeamRole /* Teammate */, pub goal: Option<bool> }
impl From<TeamMemberSpec> for PredefinedMemberSpec;   // leader/teammate => Teammate

pub enum PredefinedMemberSpec {            // #[serde(tag = "role_type")]
    Teammate(MemberSpecBase), HumanAgent(MemberSpecBase), PassiveHuman(MemberSpecBase),
    BridgeAgent(MemberSpecBaseBridge), ExternalCli(MemberSpecBaseExternal),
}
impl PredefinedMemberSpec {
    pub fn base(&self) -> &MemberSpecBase;
    pub fn member_name(&self) -> &str;
    pub fn role_type(&self) -> TeamRole;
    pub fn into_member_spec(self) -> TeamMemberSpec;
}
pub struct MemberSpecBaseBridge { #[serde(flatten)] pub base: MemberSpecBase,
    pub mailbox_inject_mode: String /* "passthrough" */, pub protocol: String,
    pub adapter_config: BTreeMap<String, Value> }
pub struct MemberSpecBaseExternal { #[serde(flatten)] pub base: MemberSpecBase,
    pub cli_agent: String, pub cli_path: Option<String>, pub command: Vec<String>,
    pub env: BTreeMap<String, String> }

pub struct TinyAgentSpec { pub system_prompt: String, pub model_name: String,
    pub name: String /* "tiny" */, pub max_iterations: usize /* 6 */,
    pub enable_security_rail: bool /* false */, pub default_schema: Option<Value> }

pub struct ModelPoolEntry { pub model_name: String, pub api_key: SecretString,
    pub api_base_url: String, pub api_provider: String, pub description: Option<String>,
    pub model_id: String /* fresh uuid hex */, pub metadata: BTreeMap<String, Value> }
impl ModelPoolEntry { pub fn new(model_name: impl Into<String>, api_provider: impl Into<String>,
                                 api_base_url: impl Into<String>, api_key: impl Into<String>) -> Self; }
```

`LeaderSpec`, `MemberSpecBase`, `MemberSpecBaseBridge`, `MemberSpecBaseExternal` and `TeamGoalSpec` all need `Default` (the goal defaults: `Hybrid`, `Some(10)`, `None`).

### `spec.rs` — team blueprint (Python `blueprint.py` enums, `TransportSpec`/`StorageSpec`/`TeamWorkspaceSpec`)

```rust
pub enum TeamLifecycle { Temporary, Persistent }            // default Temporary
pub enum ConversationMode { Team, GroupChat }               // default Team
pub enum DispatchMode { Autonomous, Scheduled }             // default Autonomous
pub enum MemberMode { BuildMode, PlanMode }                 // default BuildMode
pub enum SpawnMode { InProcess, Process }                   // "inprocess" | "process"; default InProcess
pub enum BootstrapMode { Leader, Framework }                // default Leader
pub enum ModelPoolStrategy { RoundRobin, ByModelName }      // default RoundRobin
pub enum Language { Cn, Cf, En }                            // "cn"|"cf"|"en"; default Cn
impl Language { pub const fn as_str(self) -> &'static str;
                pub fn resolve(value: Option<&str>) -> &'static str; }   // None/unknown => DEFAULT_LANGUAGE

pub struct TransportSpec { #[serde(rename = "type", alias = "kind")] pub kind: String /* "inprocess" */,
                           pub params: BTreeMap<String, Value> }
impl TransportSpec { pub fn inprocess() -> Self; pub fn is_inprocess(&self) -> bool; }
pub struct StorageSpec { #[serde(rename = "type", alias = "kind")] pub kind: String /* "memory" */,
                         pub params: BTreeMap<String, Value> }
impl StorageSpec { pub fn in_memory() -> Self; pub fn is_in_memory(&self) -> bool; }

pub struct TeamWorkspaceSpec { pub enabled: bool /* false */, pub root_path: Option<String>,
    pub artifact_dirs: Vec<String> /* artifacts/code, artifacts/docs, artifacts/reports, trajectories */,
    pub version_control: bool /* true */, pub conflict_strategy: String /* "lock" */,
    pub remote_url: Option<String> }

pub struct TeamAgentSpec {                      // field name == JSON name for every field
    pub team_name: String /* "agent_team" */, pub team_display_name: Option<String>,
    pub team_desc: Option<String>, pub agents: BTreeMap<String, DeepAgentSpec> /* required */,
    pub leader: LeaderSpec, pub predefined_members: Vec<PredefinedMemberSpec>,
    pub agent_group_name: Option<String>, pub lifecycle: TeamLifecycle,
    pub conversation_mode: ConversationMode, pub dispatch_mode: DispatchMode,
    pub teammate_mode: MemberMode, pub spawn_mode: SpawnMode, pub bootstrap_mode: BootstrapMode,
    pub enable_member_messaging: bool /* false */, pub enable_task_verification: bool /* false */,
    pub max_subtask_depth: usize /* 0 */, pub enable_hitt: bool /* false */,
    pub language: Option<String>, pub model_pool: Vec<ModelPoolEntry>,
    pub model_pool_strategy: ModelPoolStrategy, pub workspace: TeamWorkspaceSpec,
    pub transport: TransportSpec, pub storage: StorageSpec,
    pub default_max_review_rounds: usize /* 3 */, pub review_stall_timeout: u64 /* 2400 */,
    pub stale_claim_idle_timeout: u64 /* 600 */, pub stale_pending_idle_timeout: u64 /* 600 */,
    pub steer_batch_size: usize /* 2 */, pub metadata: BTreeMap<String, Value> }
impl TeamAgentSpec {
    pub fn validate(&self) -> Result<()>;
    pub fn leader_member_name(&self) -> &str;                       // leader.member_name, else DEFAULT_LEADER_MEMBER_NAME
    pub fn resolved_language(&self) -> &str;                        // language, else DEFAULT_LANGUAGE
    pub fn teammate_defaults(&self) -> Option<&DeepAgentSpec>;      // agents["teammate"]
    pub fn member_spec(&self, name: &str) -> Option<TeamMemberSpec>;
    pub fn is_member_name_valid(name: &str) -> bool;                // ^[a-z][a-z0-9_-]*$
}
```

`validate` returns `Error::validation` with these messages (Python text kept verbatim where quoted):
- `agents dict must contain a 'leader' key`
- `LeaderSpec.member_name '{name}' is reserved; pick a different name`
- `predefined member '{name}' uses a reserved name (reserved: ...)`
- `predefined member '{name}' is not a valid member name (expected ^[a-z][a-z0-9_-]*$)`
- `bootstrap_mode='framework' requires dispatch_mode='scheduled'; got dispatch_mode='{value}'`
- `default_max_review_rounds must be >= 1, got {n}`
- `review_stall_timeout must be > 0 seconds, got {n}`
- `stale_claim_idle_timeout must be > 0 seconds, got {n}`
- `stale_pending_idle_timeout must be > 0 seconds, got {n}`
- `steer_batch_size must be > 0 messages, got {n}`
Reserved-name checks use `crate::constants::is_reserved_name`; the name pattern is a hand-written char check (no regex, no global state).

`member_spec(name)`: a `predefined_members` entry whose `member_name` matches wins; else derive a teammate from `agents[name]`; else derive from `agents["teammate"]`; else `None`.
Derivation fills `display_name`/`desc` from `DeepAgentSpec.card` (falling back to the member name / empty), `prompt` from `system_prompt`, `model_name` from `DeepAgentSpec::model_name()`, and sets `role_type = Leader` for the leader's name.

### `task.rs` — task board records

```rust
pub enum Verdict { Pass, Fail }            // "pass" | "fail"; Default = Pass

pub struct TaskRecord { pub team_name: String, pub task_id: String, pub title: String,
    pub content: String, pub status: TaskStatus, pub assignee: Option<String>,
    pub reviewers: Vec<String>, pub review_round: u32, pub max_review_rounds: u32 /* 3 */,
    pub summary: Option<String>, pub note: Option<String>, pub parent_task_id: Option<String>,
    pub parent_revision: Option<u32>, pub created_by: Option<String>, pub revision: u32,
    pub observed_revision: u32, pub change_notes: Vec<String>, pub updated_at: u64 }
impl TaskRecord {
    pub fn new(team_name: impl Into<String>, task_id: impl Into<String>,
               title: impl Into<String>, content: impl Into<String>) -> Self;
    pub fn is_terminal(&self) -> bool;
    pub fn push_change_note(&mut self, note: impl Into<String>);
}

pub struct TaskSummary { pub task_id: String, pub title: String, pub status: TaskStatus,
    pub assignee: Option<String>, pub assignee_status: Option<MemberStatus>,
    pub blocked_by: Vec<String>, pub parent_task_id: Option<String>,
    pub stale_against: Option<String>, pub updated_at: u64 }
impl TaskSummary { pub fn from_record(record: &TaskRecord) -> Self; }

pub struct TaskDetail { pub task_id: String, pub title: String, pub content: String,
    pub status: TaskStatus, pub assignee: Option<String>, pub assignee_status: Option<MemberStatus>,
    pub reviewers: Vec<String>, pub review_round: u32, pub max_review_rounds: u32,
    pub summary: Option<String>, pub note: Option<String>, pub blocked_by: Vec<String>,
    pub blocks: Vec<String>, pub parent_task_id: Option<String>, pub subtasks: Vec<String>,
    pub revision: u32, pub updated_at: u64 }
impl TaskDetail { pub fn from_record(record: &TaskRecord) -> Self; }

pub struct TaskGraphSpec { pub title: String, pub content: String, pub task_id: Option<String>,
    pub depends_on: Vec<String>, pub depended_by: Vec<String>, pub assignee: Option<String>,
    pub reviewers: Vec<String>, pub max_review_rounds: Option<u32>,
    pub parent_task_id: Option<String>, pub created_by: Option<String>,
    pub parent_revision: Option<u32> }

pub struct ReviewVote { pub team_name: String, pub task_id: String, pub round: u32,
                        pub reviewer: String, pub verdict: Verdict }
pub struct TaskRevision { pub revision: u32, pub note: String, pub by: Option<String> }
impl TaskRevision { pub fn to_note(&self) -> String; }

pub struct TaskFilter { pub status: Option<TaskStatus>, pub assignee: Option<String>,
                        pub parent: Option<String>, pub limit: Option<usize> }

pub struct TaskPatch {      // every field Option; `None` means "leave unchanged"
    pub status: Option<TaskStatus>, pub title: Option<String>, pub content: Option<String>,
    pub assignee: Option<String>, pub clear_assignee: bool, pub reviewers: Option<Vec<String>>,
    pub review_round: Option<u32>, pub max_review_rounds: Option<u32>, pub summary: Option<String>,
    pub note: Option<String>, pub parent_task_id: Option<String>, pub revision: Option<u32>,
    pub observed_revision: Option<u32>, pub updated_at: Option<u64> }
impl TaskPatch {
    pub fn is_empty(&self) -> bool;
    pub fn new() -> Self;
    pub fn with_status(self, status: TaskStatus) -> Self;
    pub fn with_assignee(self, assignee: impl Into<String>) -> Self;
    pub fn clearing_assignee(self) -> Self;
    pub fn with_reviewers(self, reviewers: Vec<String>) -> Self;
    pub fn with_review_round(self, round: u32) -> Self;
    pub fn with_max_review_rounds(self, rounds: u32) -> Self;
    pub fn with_summary(self, summary: impl Into<String>) -> Self;
    pub fn with_note(self, note: impl Into<String>) -> Self;
    pub fn with_revision(self, revision: u32) -> Self;
    pub fn with_observed_revision(self, revision: u32) -> Self;
    pub fn with_updated_at(self, updated_at: u64) -> Self;
}
```

### `events.rs` — bus envelope, topics, payloads

```rust
pub struct EventMessage { pub event_type: String, pub payload: Value, pub sender_id: String }
impl EventMessage {
    pub fn new(event_type: impl Into<String>, payload: Value) -> Self;
    pub fn with_sender(self, sender_id: impl Into<String>) -> Self;
    pub fn to_json(&self) -> Result<String>;
    pub fn from_json(data: &str) -> Result<Self>;
}

pub enum TeamTopic { Team, Task, Message }        // "team" | "task" | "message"
impl TeamTopic {
    pub const fn as_str(self) -> &'static str;
    pub fn topic(self, session_id: &str, team_name: &str) -> String;
    pub fn parse(topic: &str) -> Option<(String, String, TeamTopic)>;
}
pub fn topic_part(value: &str) -> String;          // "%" -> "%25", then ":" -> "%3A"
```

`topic` emits `session:<sid>:team:<team>:<value>` with `topic_part` applied to each component; `parse` splits the five segments, requires the `session` / `team` literals, and un-escapes the two components.
`TeamTopic::parse` returns `None` for a malformed topic or an unknown last segment.

```rust
pub struct MessageEvent { pub message_id: String, pub from_member_name: String,
                          pub to_member_name: Option<String> }
pub struct BroadcastEvent { pub message_id: String, pub from_member_name: String }
pub struct MemberStatusChangedEvent { pub team_name: String, pub member_name: String,
                                      pub status: MemberStatus, pub previous: Option<MemberStatus> }
pub struct MemberExecutionChangedEvent { pub team_name: String, pub member_name: String,
                                         pub status: ExecutionStatus }
pub struct TaskCreatedEvent { pub team_name: String, pub task_id: String, pub title: String,
                              pub assignee: Option<String> }
pub struct TaskStatusChangedEvent { pub team_name: String, pub task_id: String,
                                    pub status: TaskStatus, pub assignee: Option<String> }
pub struct TaskVerifiedEvent { pub team_name: String, pub task_id: String, pub member_name: String,
                               pub verdict: Verdict }
pub struct TaskReviewVoteEvent { pub team_name: String, pub task_id: String, pub round: u32,
                                 pub reviewer: String, pub verdict: Verdict }
pub struct ToolApprovalResultEvent { pub team_name: String, pub member_name: String,
                                     pub tool_call_id: String, pub approved: bool }
pub struct TeamCompletedEvent { pub team_name: String, pub reason: Option<String> }

pub enum EventPayload { Message(MessageEvent), Broadcast(BroadcastEvent),
    MemberStatusChanged(MemberStatusChangedEvent), MemberExecutionChanged(MemberExecutionChangedEvent),
    TaskCreated(TaskCreatedEvent), TaskStatusChanged(TaskStatusChangedEvent),
    TaskVerified(TaskVerifiedEvent), TaskReviewVote(TaskReviewVoteEvent),
    ToolApprovalResult(ToolApprovalResultEvent), TeamCompleted(TeamCompletedEvent),
    Unknown { event_type: String, payload: Value } }
impl EventPayload {
    pub fn event_type(&self) -> &str;                       // Python TeamEvent strings
    pub fn payload_value(&self) -> Result<Value>;           // serialized payload object
    pub fn from_value(event_type: &str, payload: &Value) -> Result<Self>;   // unknown => Unknown
    pub fn to_message(&self, sender_id: impl Into<String>) -> Result<EventMessage>;
    pub fn from_message(message: &EventMessage) -> Result<Self>;
}
```

Event type strings (Python `TeamEvent` constants): `message`, `broadcast`, `member_status_changed`, `member_execution_changed`, `task_created`, `task_status_changed`, `task_verified`, `task_review_vote`, `tool_approval_result`, `team_completed`.

### `stream.rs` — stream frames

```rust
pub struct TeamOutputSchema { pub source_member: String, pub role: TeamRole, pub chunk: Value }
pub const TEAM_IDLE: &str = "team.idle";
pub const TEAM_COMPLETED: &str = "team.completed";
pub const TEAM_ERROR: &str = "team.error";
pub const RETRYING: &str = "retrying";
pub fn team_idle_frame() -> Value;         // {"event_type":"team.idle","member_count":0,"members":{}}
pub fn team_completed_frame() -> Value;    // {"event_type":"team.completed","member_count":0,"task_count":0}
pub fn tag_chunk(source_member: impl Into<String>, role: TeamRole, chunk: Value) -> Value;
pub fn is_team_event_marker(chunk: &Value) -> bool;   // chunk.payload.event_type starts with "team."
```

### `mod.rs`

Module docs, `pub mod` declarations for all eight files, `pub use` flat re-exports of every public item, and `pub use openjiuwen_harness_protocol::HarnessState;`.

## Constraints

- Only this directory is edited; `../constants.rs` and `../lib.rs` belong to the crate root.
- `openjiuwen_common::error::{Error, Result}` only; user input never panics; no `unwrap`/`expect` outside `#[cfg(test)]`.
- No `unsafe`, no global mutable state, no I/O, no clock reads, no logging in this module: it is pure data plus pure helpers.
- `tracing` only if something genuinely needs a warning (there is no such case here).
- Dependencies: `serde`, `serde_json`, `openjiuwen-common` (`SecretString`, `Error`), `openjiuwen-agent` (`AgentCard`), `openjiuwen-harness-protocol` (`HarnessState`). Nothing else.
- Every file stays well under 1000 lines; the split is by responsibility, not size.

## Routing Table

- `./mod.rs` → module docs, submodule declarations, flat re-exports, `HarnessState` re-export
- `./status.rs` → `MemberStatus`, `ExecutionStatus`, `TaskStatus`, `MemberLifetime`, `RoundOutcome`, transition tables, retire/round helpers
- `./deep_agent.rs` → `DeepAgentSpec`, `ModelRef`, `ToolSpec`, `RailSpec`, `McpServerSpec`, `SubAgentSpec`, `WorkspaceSpec`
- `./team.rs` → `TeamRole`, `TeamSpec`, `TeamRuntimeContext`, `TeamSnapshot`, `MemberSnapshot`
- `./member_spec.rs` → `LeaderSpec`, `TeamGoalSpec`, `MemberSpecBase`, `TeamMemberSpec`, `PredefinedMemberSpec`, bridge/external member specs, `TinyAgentSpec`, `ModelPoolEntry`
- `./spec.rs` → `TeamAgentSpec` (+ `validate`, `member_spec`), `TransportSpec`, `StorageSpec`, `TeamWorkspaceSpec`, the spec enums
- `./task.rs` → `TaskRecord`, `TaskSummary`, `TaskDetail`, `TaskGraphSpec`, `ReviewVote`, `Verdict`, `TaskFilter`, `TaskPatch`, `TaskRevision`
- `./events.rs` → `EventMessage`, `TeamTopic`, `topic_part`, event payload structs, `EventPayload`
- `./stream.rs` → `TeamOutputSchema`, stream frame constants and helpers

## Test Strategy

Unit tests live in `#[cfg(test)] mod tests` beside the definitions (`#[allow(clippy::unwrap_used, clippy::expect_used)]`), and pin wire behaviour rather than Rust details:
- serde round-trip for every public type, including unknown-field tolerance (`{"unknown": 1}` still parses) and empty-default behaviour.
- exact wire strings for all enums (`"shut_down"`, `"shutdown_requested"`, `"in_progress"`, `"inprocess"`, `"round_robin"`, `"pass"`, …) and exact JSON defaults of `TeamAgentSpec`, `DeepAgentSpec`, `LeaderSpec`, `TaskRecord`.
- the pinned transition facts: legal `shutdown_requested -> dormant`, `dormant -> restarting`, `dormant -> shut_down`; illegal `ready -> dormant`; every member/execution/task table entry; `retire_target` and `round_transitions` vectors.
- status sets: departed/unreachable/settled/quiescent membership for all eleven member statuses.
- topic building + escaping (`%` → `%25`, `:` → `%3A`) and `parse` round-trip, including an escaped component.
- `ToolSpec` acceptance of all three input shapes and its canonical output shape.
- `EventPayload` ↔ `EventMessage` conversion for every known event type plus `Unknown`.
- `TeamAgentSpec::validate` acceptance/rejection cases with the pinned message fragments; `member_spec` derivation from `predefined_members`, `agents[name]`, `agents["teammate"]`.
`status.rs` and `spec.rs` keep their production code under the 1000-line baseline (about 610 and 600 lines respectively); their totals exceed it only because of the inline exhaustive test suites, and each file carries a top-of-file note recording that.
Definition of done for this node: `cargo test -p openjiuwen-agent-teams --lib` (116 tests), `cargo test -p openjiuwen-agent-teams --no-default-features --lib`, no clippy diagnostics pointing at `src/schema/`, and `rustfmt --check` clean on every `src/schema/*.rs` file.
Crate-wide `cargo fmt`/`-D warnings` also cover `constants.rs` and the still-empty sibling modules; their diagnostics belong to the crate root node, not here.

## Notes for Agents

- Rust deviations from Python, all deliberate: `spawn_mode` defaults to `inprocess` (no OS-process members), `TeamSpec.evolution_enabled` defaults to `false`, `TaskStatus` is `str`-valued, and `ToolSpec` replaces Python's undiscriminated `ToolCard | BuiltinToolSpec` union with one normalized shape.
- `TASK_TRANSITIONS` is narrower than Python's: Rust rejects `planning -> planning|pending|blocked` and `in_progress -> pending|blocked` and `in_review -> pending`. Do not widen it here; the task board codes against this table.
- Python `MemberStatus` has no methods and `RoundOutcome`/`round_transitions` do not exist in Python (`RoundOutcome` there is a dataclass). The Rust helpers are the crate contract and are the only sanctioned way to ask transition questions.
- `LeaderSpec` keeps only `member_name`/`display_name`/`desc`/`goal`; Python's inherited `prompt`/`model_name` are read from `agents["leader"]` instead.
- `LeaderSpec.desc` defaults to `""`; Python injects an i18n string at construction. Use `desc_or_default()` when a human-facing default is needed.
- `TransportSpec`/`StorageSpec` serialize their kind as `"type"` (Python key) while the Rust field is `kind`; both `"type"` and `"kind"` are accepted on input.
- `ModelPoolEntry.api_key` is a `SecretString`: it round-trips as plaintext JSON but never leaks through `Debug`.
- `TeamMemberSpec` flattens `MemberSpecBase`, so its JSON is `{"member_name":…, "display_name":…, "role_type":…}`.
- `PredefinedMemberSpec` is internally tagged by `role_type`; a missing tag is a deserialization error (documented Python-default of `teammate` is not emulated).
- Payload field names follow this crate's contract, not Python's: for example Python `MemberStatusChangedEvent` carries `old_status`/`new_status` while Rust carries `status`/`previous`.
- `change_notes` holds rendered note strings (`TaskRevision::to_note`); `TaskRevision` is the structured form used at call sites.
- `validate` exempts `PredefinedMemberSpec::HumanAgent` entries from the reserved-name and pattern checks (Python `_validate_reserved_names` does the same, because the framework injects the `human_agent` avatar under that reserved name).
- `member_spec(name)` only derives a `TeamRole::Leader` spec when the leader is keyed in `agents` under its member name; the normal blueprint keys it as `agents["leader"]`, so the runtime builds the leader context from `LeaderSpec` plus `agents["leader"]` instead.
- `TeamRole::default()` is `Leader` while `TeamMemberSpec::default().role_type` is `Teammate` (Python's member default); `TeamMemberSpec` therefore has a hand-written `Default`.
- `TeamSpec::default().display_name` is `DEFAULT_TEAM_NAME`, and `TeamSpec::from_spec` mirrors `workspace` only when `workspace.root_path` is set (`stable_base: false`, language from `resolved_language()`).
- `TaskSummary`/`TaskDetail` only fill `max_review_rounds` from the record in `from_record`; a bare JSON `TaskDetail` decodes it as `0` (Python's view model defaults to `None`).
- `RailSpec` accepts Python's `"type"` key only; unlike `TransportSpec`/`StorageSpec` there is no `"kind"` alias.
- `EventMessage` derives `Deserialize` with struct defaults, so serde's sequence form (`"[]"`) yields the all-default envelope; malformed-event tests must use a non-deserializable shape.
- `is_team_event_marker` expects the frame produced by `tag_chunk` (payload nested under `chunk`); the bare `team_idle_frame()`/`team_completed_frame()` payloads are not markers on their own.
