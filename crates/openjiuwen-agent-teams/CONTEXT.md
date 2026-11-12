# openjiuwen-agent-teams

## Intent

Leader/teammate multi-agent runtime: the Rust port of the Python package `openjiuwen.agent_teams` (v0.1.20, ~108k lines / 336 files).
Scope is the **in-process core**: team specs and expert catalog, the leader/teammate runtime (organisation dispatch, task board, mailbox, member lifecycle, local workspace isolation), HITL forwarding, the harness-protocol adapter, the `team-member` skill CLI, and the `openjiuwen-team-mcp` MCP server.
Distributed/process modes (ZeroMQ, PostgreSQL, Pulsar, external CLI members) exist only as documented trait seams returning `Error::Unsupported`.
Parity rule: serialized JSON names, defaults, status values, tool/command names and observable lifecycle behaviour follow Python; internals are idiomatic Rust.

## API Surface

Crate root (`lib.rs`) re-exports the public surface flat, mirroring `agent_teams/__init__.py` plus the seams listed below.

```rust
// identity / addressing
pub const DEFAULT_LEADER_MEMBER_NAME: &str = "team_leader";
pub const HUMAN_AGENT_MEMBER_NAME: &str = "human_agent";
pub const USER_PSEUDO_MEMBER_NAME: &str = "user";
pub const RESERVED_MEMBER_NAMES: [&str; N];
pub fn is_reserved_name(name: &str) -> bool;
pub fn parse_mention(text: &str) -> Option<(&str, &str)>;   // "^@(\S+)\s+([\s\S]+)$"

// spec → runtime
#[async_trait] pub trait SpecBuild { async fn build(&self) -> Result<Arc<TeamAgent>>; async fn build_with(&self, deps: TeamBuildDeps) -> Result<Arc<TeamAgent>>; }
pub struct TeamAgentSpec { /* Python blueprint.py: agents, leader, predefined_members, dispatch_mode, … */ }
pub struct TeamAgent;            // impl openjiuwen_agent::Agent + openjiuwen_multiagent::Team
pub struct TeamRuntimeManager;   // activate / live / interact / finalize_member / stop_team / snapshot
pub struct TeamRuntimePool;

// protocol
pub use openjiuwen_harness_protocol::HarnessState;   // re-export, never redefine
pub struct HarnessMemberRuntime;                     // impl MemberRuntime over Arc<dyn HarnessProtocol>
pub trait MemberHarnessFactory;                      // builds one Arc<dyn HarnessProtocol> per member
pub struct NativeMemberFactory;                      // DeepAgentSpec → DeepAgent → NativeHarness

// experts
pub trait ExpertCatalog;  pub struct ExpertRef;  pub struct PackageLocation;
pub fn set_process_expert_catalog(c: Arc<dyn ExpertCatalog>);  pub fn get_process_expert_catalog() -> Option<Arc<dyn ExpertCatalog>>;

// CLI / MCP
pub trait TeamClient;  pub struct TeamJoinDescriptor;  pub fn run_skill_cli(argv: Vec<String>) -> i32;
pub fn connect_live_client(d: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>;  // in-process attach: global registry
pub struct SkillCli;   pub enum CliCommand { Member(MemberCommand), Operator(OperatorCommand) }  // clap tree, publicly matchable
pub fn run_skill_cli(argv: Vec<String>) -> i32;      // exit 0 success/--help, 1 refused or unhosted, 2 CLI-side, 130 cancelled
pub struct TeamMcpServer;  pub fn build_mcp_server(client: Arc<dyn TeamClient>) -> TeamMcpServer;
```

Binaries (mirroring the Python console script and MCP entry point): `team-member` → `run_skill_cli`;
`openjiuwen-team-mcp` (`mcp` feature) → `mcp::run_mcp_server_from_env` over stdio.

Modules are documented by their own `CONTEXT.md`; the list is in the Routing Table.

## Cross-module seams (do not change without updating every dependent node)

Everything below is the contract child nodes code against. Types are `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default collections to empty.

### `schema` → everyone

```rust
pub struct TeamAgentSpec {                      // Python blueprint.py
    pub team_name: String,                      // "agent_team"
    pub team_display_name: Option<String>, pub team_desc: Option<String>,
    pub agents: BTreeMap<String, DeepAgentSpec>,// required; must contain "leader"
    pub leader: LeaderSpec,
    pub predefined_members: Vec<PredefinedMemberSpec>,
    pub agent_group_name: Option<String>,
    pub lifecycle: TeamLifecycle,               // temporary | persistent
    pub conversation_mode: ConversationMode,    // team | group_chat
    pub dispatch_mode: DispatchMode,            // autonomous | scheduled
    pub teammate_mode: MemberMode,              // build_mode | plan_mode
    pub spawn_mode: SpawnMode,                  // inprocess | process
    pub bootstrap_mode: BootstrapMode,          // leader | framework
    pub enable_member_messaging: bool,          // false
    pub enable_task_verification: bool,         // false
    pub max_subtask_depth: usize,               // 0
    pub enable_hitt: bool,                      // false
    pub language: Option<String>,               // cn default; cf|en
    pub model_pool: Vec<ModelPoolEntry>, pub model_pool_strategy: ModelPoolStrategy, // round_robin
    pub workspace: TeamWorkspaceSpec, pub transport: TransportSpec, pub storage: StorageSpec,
    pub default_max_review_rounds: usize,       // 3
    pub review_stall_timeout: u64,              // 2400 s
    pub stale_claim_idle_timeout: u64,          // 600 s
    pub stale_pending_idle_timeout: u64,        // 600 s
    pub steer_batch_size: usize,                // 2
    pub metadata: BTreeMap<String, Value>,
}
impl TeamAgentSpec {
    pub fn validate(&self) -> Result<()>;              // leader present, reserved names, dispatch/teammate knobs, bootstrap rules
    pub fn leader_member_name(&self) -> &str;          // leader.member_name, default "team_leader"
    pub fn member_spec(&self, name: &str) -> Option<TeamMemberSpec>;  // predefined, else derived from agents[name]/agents["teammate"]
    pub fn teammate_defaults(&self) -> Option<&DeepAgentSpec>;
}

pub struct DeepAgentSpec { /* subset of Python harness/schema/deep_agent_spec.py */ }
pub enum PredefinedMemberSpec { Teammate { .. }, HumanAgent { .. }, PassiveHuman { .. }, BridgeAgent { .. }, ExternalCli { .. } } // #[serde(tag = "role_type")]
pub struct TeamMemberSpec { pub base: MemberSpecBase, pub role_type: TeamRole, pub goal: Option<bool> }
pub struct MemberSpecBase { pub member_name: String, pub display_name: String, pub desc: String, pub prompt: String, pub model_name: Option<String> }
pub struct LeaderSpec { pub member_name: String /* "team_leader" */, pub display_name: String /* "Team Leader" */, pub desc: String, pub goal: Option<TeamGoalSpec> }
pub struct TeamGoalSpec { pub strategy: GoalStrategy /* hybrid */, pub max_attempts: Option<u32> /* 10 */, pub token_budget: Option<u64> }
pub struct TeamSpec { /* runtime mirror: team_name, display_name, conversation_mode, leader_member_name, language, dispatch_mode, enable_member_messaging, max_subtask_depth, teammate_mode, model_pool, model_pool_strategy, workspace, metadata */ }
pub struct TeamRuntimeContext { /* role, member_name, display_name, desc, prompt, team_spec, workspace_root, member_model, language */ }
pub struct TransportSpec { pub kind: String /* inprocess */, pub params: BTreeMap<String, Value> }
pub struct StorageSpec  { pub kind: String /* memory */,    pub params: BTreeMap<String, Value> }
pub struct TinyAgentSpec;  pub struct ModelPoolEntry;
```

`events` seam: `EventMessage { event_type: String, payload: Value, sender_id: String }`, `TeamTopic` (`Team|Task|Message`) with Python topic strings, and typed payload structs (`MessageEvent`, `BroadcastEvent`, `MemberStatusChangedEvent`, `TaskCreatedEvent`, `TaskVerifiedEvent`, `ToolApprovalResultEvent`, …) serialized into `payload`.

`status` seam: `MemberStatus`, `ExecutionStatus`, `TaskStatus`, `is_valid_member_transition`, `retire_target`, `round_transitions(RoundOutcome) -> Vec<ExecutionStatus>`, `TASK_TRANSITIONS`.
`task` seam: `TaskRecord`, `TaskSummary`, `TaskDetail`, `TaskGraphSpec`, `ReviewVote`, `TaskFilter`, `TaskPatch`.

### `storage` → `taskboard`, `member`, `runtime`, `skill`, `mcp`

```rust
#[async_trait] pub trait TeamStore: Send + Sync {
    async fn upsert_team(&self, rec: TeamInfoRecord) -> Result<()>;
    async fn team(&self, team_name: &str) -> Result<Option<TeamInfoRecord>>;
    async fn insert_member(&self, rec: MemberRecord) -> Result<()>;
    async fn member(&self, team: &str, name: &str) -> Result<Option<MemberRecord>>;
    async fn list_members(&self, team: &str) -> Result<Vec<MemberRecord>>;
    async fn update_member(&self, team: &str, name: &str, patch: MemberPatch) -> Result<bool>;
    async fn transition_member(&self, team: &str, name: &str, from: MemberStatus, to: MemberStatus) -> Result<bool>; // CAS
    async fn insert_task(&self, rec: TaskRecord) -> Result<()>;
    async fn task(&self, team: &str, task_id: &str) -> Result<Option<TaskRecord>>;
    async fn list_tasks(&self, team: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>>;
    async fn update_task(&self, team: &str, task_id: &str, patch: TaskPatch) -> Result<bool>;
    async fn transition_task(&self, team: &str, task_id: &str, from: TaskStatus, to: TaskStatus) -> Result<bool>;
    async fn add_dependency(&self, team: &str, task_id: &str, depends_on: &str) -> Result<()>;
    async fn dependencies(&self, team: &str, task_id: &str) -> Result<Vec<String>>;
    async fn dependents(&self, team: &str, task_id: &str) -> Result<Vec<String>>;
    async fn insert_vote(&self, rec: ReviewVote) -> Result<()>;
    async fn votes(&self, team: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>>;
    async fn insert_message(&self, rec: MessageRecord) -> Result<()>;
    async fn unread_messages(&self, team: &str, member: &str) -> Result<Vec<MessageRecord>>;
    async fn mark_read(&self, team: &str, member: &str, ids: &[String]) -> Result<()>;
    async fn read_watermark(&self, team: &str, member: &str) -> Result<Option<u64>>;
    async fn set_read_watermark(&self, team: &str, member: &str, ts: u64) -> Result<()>;
}
pub struct InMemoryTeamStore;    // `StorageSpec::in("memory")`; deterministic, no I/O
pub fn open_store(spec: &StorageSpec) -> Result<Arc<dyn TeamStore>>;  // non-`memory` → Error::Unsupported
```

### `messager` → `runtime`, `member`

```rust
pub type EventSink = Arc<dyn Fn(EventMessage) + Send + Sync>;
#[async_trait] pub trait Messager: Send + Sync {
    fn name(&self) -> &str;                                          // "inprocess"
    fn subscribe(&self, member: &str, topics: &[TeamTopic], sink: EventSink) -> Result<()>;
    fn unsubscribe(&self, member: &str) -> Result<()>;
    async fn publish(&self, topic: TeamTopic, event: EventMessage) -> Result<()>;
    async fn send(&self, target: &str, event: EventMessage) -> Result<()>;
    async fn subscribe_transport(&self, team_name: &str) -> Result<()>;
}
pub fn create_messager(cfg: &TransportSpec) -> Result<Arc<dyn Messager>>;   // inprocess only; pyzmq/hybrid → Unsupported
```
A sink failure is logged, never propagated; `sender_id == ""` is stamped with the sending member name.

### `member` → `runtime`

```rust
pub trait MemberRuntime: Send + Sync {
    fn session_id(&self) -> &str;
    fn state(&self) -> HarnessState;
    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()>;
    async fn stop(&self) -> Result<()>;
    async fn send(&self, content: String, immediate: bool) -> Result<()>;
    async fn abort(&self, immediate: bool) -> Result<()>;
    async fn pause(&self) -> Result<()>;
    async fn resume(&self, query: Option<String>) -> Result<()>;
    fn subscribe(&self, observer: Arc<dyn MemberObserver>);
}
pub trait MemberObserver: Send + Sync { fn on_state(&self, s: HarnessState); fn on_round(&self, r: RoundOutcome); fn on_chunk(&self, chunk: Value); }
pub struct TeamMember;            // status CAS + lifecycle; owns Arc<dyn MemberRuntime>
pub struct InProcessSpawnHandle;  // owns the spawned task + cancellation token
pub struct MemberActivityRegistry;
```

### `runtime` entry points consumed by CLI/MCP/umbrella

```rust
impl TeamAgent {
    pub fn card(&self) -> &AgentCard;
    pub async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    pub fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn start(&self) -> Result<()>;   pub async fn stop(&self) -> Result<()>;
    pub fn state(&self) -> HarnessState;
    pub async fn interact(&self, payload: InteractPayload) -> DeliverResult;
    pub async fn snapshot(&self) -> TeamSnapshot;      // members + tasks, sorted
    pub fn backend(&self) -> Arc<TeamBackend>;         // org facade used by tools/CLI
}
pub struct TeamBuildDeps { pub harness_factory: Arc<dyn MemberHarnessFactory>, pub store: Arc<dyn TeamStore>,
                           pub workspace: Arc<dyn WorkspaceProvider>, pub language: String }
```

## Constraints

- Only `crates/openjiuwen-agent-teams/**` is edited; root manifest, sibling crates and `docs/` are out of scope.
- Dependency direction: this crate depends on `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-core`/`openjiuwen-multiagent`/`openjiuwen-agent`/`openjiuwen-session`, `openjiuwen-common`, `openjiuwen-foundation`; nothing may depend on this crate. No cycles, no `openjiuwen-cli`/`openjiuwen` imports.
- Async-first tokio; `#[async_trait]` for object-safe async seams; locks never held across `.await` (`parking_lot` for short sections, `tokio::sync` for awaitable coordination); cancellation via `tokio_util::sync::CancellationToken`.
- Errors: `openjiuwen_common::error::{Error, Result}` only. User input never panics. Deferred capability → `Error::Unsupported` with the Python capability name.
- Serde on every public data type (snake_case, unknown fields tolerated). No `JsonSchema` needed; Cards crossing the wire are `openjiuwen_agent::AgentCard`.
- `tracing` only — no `println!`/`eprintln!` in library code; the CLI writes to stdout through its own writer.
- No `unsafe`. No undocumented global mutable state: the process-global expert catalog and the process-global `TeamRegistry` (team_name → runtime manager) are the two documented singletons; tests construct local instances.
- Files ≤1000 lines; split by responsibility; a legitimately long file carries a top-of-file note and is listed in this file's Test Strategy/Notes.
- `--no-default-features` must build: `mcp` gates `rmcp`, `distributed` gates transport stubs.

## Behaviour parity notes (wire-visible)

- `MemberStatus` wire values: `unstarted, starting, ready, busy, paused, stopped, restarting, shutdown_requested, shut_down, dormant, error`.
- `TaskStatus`: `pending, blocked, planning, in_progress, in_review, completed, cancelled`; terminal = `completed|cancelled`; claimable = `pending`.
- `ExecutionStatus`: `idle, starting, running, cancel_requested, cancelling, cancelled, completing, completed, failed, timed_out`.
- Topics: `session:<sid>:team:<team>:{team|task|message}` with `%`→`%25` and `:`→`%3A` escaping.
- Addressing: `"*"` = broadcast, `targets=[…]` = multicast, `"leader"`/`"user"` are role placeholders resolved at delivery; `parse_mention` grammar `@name body`.
- Member names match `^[a-z][a-z0-9_-]*$` and may not collide with reserved names (`user`, `team_leader`, `human_agent`).
- Review gate: author completes → `in_review` when reviewers exist (autonomous: first verdict; scheduled: vote tally per round); otherwise `completed` and dependent edges unblock (`blocked` → `pending`).
- Broadcast read state is a single per-member watermark, never a row flag.

## Deferred (documented, `Error::Unsupported` through seams)

- Transports: `pyzmq`, `hybrid` (WebSocket), Pulsar — `Messager` seam + `create_messager` rejection.
- Storage: `sqlite`, `postgresql`, `mysql` — `TeamStore` seam; Python's SQL schema is not ported.
- Spawn: OS-process members (`spawn_mode="process"`), external CLI members (claude/codex), SSH transport, external leader.
- Workspace: git worktree backend (`isolation="worktree"`), remote worktree backend, team-workspace artifact versioning/linking.
- Organisation layer (cross-team org task pool, summary teams, org messaging), swarmflow, fork/compact, KV-cache affinity, OTel observability, reliability rail, group-chat policy, monitor/TUI.
- Model router / IntelliRouter / builtin-model selection (round-robin and by-name allocator only).
- Skill visibility library, `skill/rail_spec` rails, prompt packs (`prompts/{cn,en}` are not shipped; built-in English prompts only).

## Routing Table

- `./src/schema/` → spec/blueprint types, team runtime types, statuses/transitions, task records, event payloads, stream frames (`src/schema/CONTEXT.md`)
- `./src/experts/` → `ExpertCatalog`, `ExpertRef`, agent-group packages, template→member composition (`src/experts/CONTEXT.md`)
- `./src/storage/` → `TeamStore` trait, records/patches, in-memory store, `open_store` (`src/storage/CONTEXT.md`)
- `./src/taskboard/` → task tree: dependency graph, claims, completion, reviews/votes, revisions, subtasks (`src/taskboard/CONTEXT.md`)
- `./src/messager/` → `Messager` seam, in-process bus, transport factories (`src/messager/CONTEXT.md`)
- `./src/workspace/` → member workspace allocation, local directory layout, worktree seam (`src/workspace/CONTEXT.md`)
- `./src/member/` → `MemberRuntime`/`MemberObserver`, `TeamMember` status machine, in-process spawn handles (`src/member/CONTEXT.md`)
- `./src/interaction/` → HITL payloads, routing grammar, `InteractGate`, human inbox, tool-approval bridge (`src/interaction/CONTEXT.md`)
- `./src/runtime/` → `TeamAgent` (leader/teammate loop), `TeamBackend`, dispatch decision, runtime manager/pool, model allocator, `Team` adapter (`src/runtime/CONTEXT.md`)
- `./src/harness/` → harness-protocol adapter: `MemberHarnessFactory`, `HarnessMemberRuntime`, native factory, checkpoints (`src/harness/CONTEXT.md`)
- `./src/skill/` → `team-member` clap CLI, join descriptor, `TeamClient` implementations, live-team attach via the runtime registry (`src/skill/CONTEXT.md`)
- `./src/bin/` → the two binaries: `team-member` (skill CLI, exit `0/1/2/130`) and `openjiuwen-team-mcp` (stdio MCP server, `mcp` feature)
- `./src/mcp/` → `openjiuwen-team-mcp` server: tool catalogue + dispatch, stdio serve (`src/mcp/CONTEXT.md`)
- `./src/constants.rs` → reserved names, defaults, mention parsing
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests live beside their module (`#[cfg(test)]` in each file), ~676 of them, and cover the internals of `schema`, `storage`, `taskboard`, `messager`, `workspace`, `member`, `interaction`, `runtime`, `harness`, `skill` and `mcp`.
- Integration suites in `./tests/` drive the public API only and pin cross-module behaviour (187 tests): `spec_build.rs` (JSON parse/defaults/unknown fields, `validate`, `build_with`, activate admission), `team_run.rs` (full in-process run, single `team.completed` marker, open-task run, `interact` outcomes, snapshot/stop), `member_and_mail.rs` (spawn, factory failure, shutdown matrix, workspace isolation, direct/broadcast/multicast mail, leader-only tasks), `task_flow.rs` (dependency graph, cycles, claims, review gate, votes, subtasks), `hitl.rs` (mention grammar, targets, gate, approval bridge), `protocol_adapter.rs` (command forwarding, observer order, mapping tables, checkpoint round-trip), `cli.rs` (full clap tree, output lines, exit codes 0/1/2), `mcp.rs` (tool tables per scope, exact `call_tool` text), `expert_catalog.rs` (agent-group packages, template composition), and `bins.rs` (the two binaries' process contract through `CARGO_BIN_EXE_*`: help, missing descriptor, unhosted team, MCP stderr diagnostic).
- Fakes live in `./tests/support/` (scripted harness factory, recording workspace provider, in-memory store, bounded `wait_until`) and in `openjiuwen-test-support` as a dev-dependency. Everything offline and deterministic: no network, no `#[ignore]`, no sleeps beyond short bounded waits.
- Definition of done (all four green): `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings`.

## Notes for Agents

- Python `TeamAgent` is one class for both roles. Keep that: `TeamAgent` carries `TeamRole`; the leader owns dispatch, teammates poll the board.
- Python `spec.build()` mutates the spec (fills `language`); the Rust `build_with` takes `&self` and does not mutate — documented deviation.
- `HarnessState` comes from `openjiuwen_harness_protocol`; re-export it, never define a second enum.
- The Python `_Bus` (in-process messager), `TeamDatabase` (sqlite) and `asyncio.Task` spawn handle map to `messager::InProcessMessager`, `storage::InMemoryTeamStore` and `member::InProcessSpawnHandle`.
- `openjiuwen_multiagent` is used only for the `Team`/`AgentTeamSession` seam so `Runner::run_agent_team` can drive a `TeamAgent`; its message bus is not used (Python `agent_teams` has its own envelope).
- Prefer `Error::validation` for user-supplied names/dispatch knobs, `Error::config` for missing build dependencies (model, factory), `Error::not_found` for unknown members/tasks.
- Cross-node deviations from Python, each owned and explained by its child node: `SpecBuild` is `#[async_trait]` and `build_with` takes `&self` without mutating the spec (`./src/runtime/`); the worktree name suffix is a FNV-1a digest instead of `sha256(..)[:10]` and member state lives at `<base>/<team>/sessions/<sid>/members/<member>` (`./src/workspace/`); the multiagent `AgentTeamSession` is ignored by the adapter, `Team::add_agent` writes its row on the current runtime, and one run at a time per team (`./src/runtime/`); `HarnessMemberRuntime::send` drops transport-level `SendReceipt` (`./src/member/` Known gaps); `TaskStatus` transitions are narrower than Python's (no `in_progress→pending`, `planning→pending`) (`./src/taskboard/`).
- Live attach is in-process only: `connect_live_client` reads the process-global runtime registry, so the CLI/MCP server can act on a team this process hosts; a team this process does not host is exit `1` (`Error::config`/CLI `1`), and cross-process attach via a shared store or distributed transport is deferred.
- The `openjiuwen` umbrella re-exports this crate as `openjiuwen::agent_teams` (and forwards the `live-tests` feature).
- The `live-tests` feature is declared in `Cargo.toml` but no test or module in this crate reads it: real-provider runs are not gated here, so nothing in the crate needs the network.
- Models reach a member only through `MemberHarnessFactory::create(&DeepAgentSpec, &MemberHarnessContext)`: the model name comes from `DeepAgentSpec::model` (else the resolver's default), and credentials/provider come from the factory's `ModelResolver` (`NativeMemberFactory::from_env` → `EnvModelResolver` reads `MODEL_PROVIDER`/`MODEL_NAME`/`API_KEY`/`API_BASE`).
- `NativeMemberFactory` writes the effective model identity onto the `DeepAgentConfig` it builds as well as onto the builder model: `model_name`/`model_provider` take the spec's pinned `ModelRef` values, else the resolved model's own name/provider (blanks skipped).
  `DeepAgentBuilder` copies that identity into the inner `ReActAgentConfig`, and the agent loop fills a blank request name from it — leaving it unset makes every provider request and `list_models` run with an empty `model_name` (`./src/harness/`).
- `TeamAgentSpec::model_pool` allocation is recorded on `MemberRecord::model_ref` only; it is never passed into the harness factory, so a pool entry currently selects no endpoint.
- Offline runs need either an injected factory (the test pattern) or `NativeMemberFactory::new(resolver)` with a resolver returning a scripted `openjiuwen_foundation::llm::Model`; `TeamAgentSpec::build()` over `TeamBuildDeps::default()` fails with `ErrorKind::Config` when no provider/model is configured.
