# src/runtime — team agent, leader loop, dispatch, runtime manager

## Intent

The coordination half of `openjiuwen-agent-teams`: the `TeamAgent` (one type for both roles), its build
path from a `TeamAgentSpec`, the leader's run loop, the organisation facade over the store/task board/bus,
the pool and process-global runtime manager used by the CLI/MCP, and the model-pool allocator.
Python parity sources: `openjiuwen/agent_teams/agent/team_agent.py`, `agent/agent_configurator.py`,
`runtime/{manager,dispatch,pool,gate}.py`, `models/allocator.py`.

The module owns *policy and orchestration*: who runs, what a run means, when a team is done, how an
external payload reaches a member. It owns no state of its own beyond the live-runtime maps: rows live in
[`crate::storage`], task decisions in [`crate::taskboard`], member status in [`crate::member`], transports in
[`crate::messager`], routing grammar in [`crate::interaction`].

`TeamAgent` carries a [`TeamRole`]: a leader dispatches and steers, a teammate polls the board and answers
mail. Both roles run the same type, per Python.

## API Surface

```rust
// deps.rs — the injected build inputs
pub struct TeamBuildDeps { pub harness_factory: Arc<dyn MemberHarnessFactory>, pub store: Arc<dyn TeamStore>,
                           pub workspace: Arc<dyn WorkspaceProvider>, pub language: Option<String> }  // Clone
impl TeamBuildDeps { pub fn new() -> Self; pub fn with_harness_factory(..) -> Self;
                     pub fn with_store(..) -> Self; pub fn with_workspace(..) -> Self;
                     pub fn with_language(..) -> Self; pub fn resolved_language(&self) -> &'static str; }
// Default: NativeMemberFactory::from_env(), InMemoryTeamStore, LocalWorkspaceProvider rooted at
// std::env::temp_dir().join("openjiuwen-agent-teams"), language None.

// allocator.rs — Python models/allocator.py
pub struct TeamModelAllocation { pub model_name: String, pub index: usize }     // serde
pub struct ModelAllocator;                                                       // Clone
impl ModelAllocator {
    pub fn new(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Self;
    pub fn is_empty(&self) -> bool;   pub fn len(&self) -> usize;
    pub fn strategy(&self) -> ModelPoolStrategy;
    pub fn allocate(&self, model_name: Option<&str>) -> Result<TeamModelAllocation>;
    pub fn state_dict(&self) -> Value;                    // {"strategy","next_index"}
    pub fn from_state_dict(value: &Value) -> Result<Self>;   // restores the cursor, empty pool
}

// dispatch.rs — run admission for one (team, session)
pub enum RunActionKind { CreateTeam, AttachExisting, ResumeExisting, ResumePaused,
                         RejectCrossSession, RejectStopped, RejectLifecycle }
pub struct RunAction { pub kind: RunActionKind, pub team_name: String, pub session_id: String,
                       pub reason: Option<String> }
impl RunActionKind { pub const fn as_str(self) -> &'static str; pub const fn is_reject(self) -> bool; }
pub struct LiveTeamState { pub team_name: String, pub session_id: String,
                           pub lifecycle: TeamLifecycle, pub paused: bool, pub state: HarnessState }
pub fn decide_run_action(live: Option<&LiveTeamState>, requested_session: &str,
                         lifecycle: TeamLifecycle) -> RunAction;

// initial_input.rs — caller request → leader query + pre-routed deliveries
pub struct InitialInput { pub leader_query: Option<String>, pub deliveries: Vec<InteractPayload> }
pub fn parse_initial_input(input: &Value, roster: &[String]) -> InitialInput;

// backend.rs — the organisation facade exposed as TeamAgent::backend()
pub struct TeamBackend;
impl TeamBackend {
    pub fn new(team_spec: TeamSpec, store: Arc<dyn TeamStore>, taskboard: Arc<TaskBoard>,
               messager: Arc<dyn Messager>, spawner: Arc<dyn MemberSpawner>,
               workspace: Arc<dyn WorkspaceProvider>, activity: Arc<MemberActivityRegistry>) -> Self;
    pub fn team_name(&self) -> &str;        pub fn spec(&self) -> &TeamSpec;
    pub fn store(&self) -> Arc<dyn TeamStore>;   pub fn tasks(&self) -> &Arc<TaskBoard>;
    pub fn messager(&self) -> &Arc<dyn Messager>; pub fn spawner(&self) -> &Arc<dyn MemberSpawner>;
    pub fn workspace(&self) -> &Arc<dyn WorkspaceProvider>;
    pub fn activity(&self) -> &Arc<MemberActivityRegistry>;
    pub fn members(&self) -> Vec<Arc<TeamMember>>;              // sorted by member_name
    pub fn member(&self, name: &str) -> Option<Arc<TeamMember>>;
    pub fn member_names(&self) -> Vec<String>;                  // sorted
    pub fn add_member(&self, member: Arc<TeamMember>);           // registers row + activity
    pub fn remove_member(&self, name: &str) -> Option<Arc<TeamMember>>;
    pub async fn upsert_team_info(&self, info: TeamInfoRecord) -> Result<()>;
    pub async fn send_message(&self, from: &str, to: Option<&str>, content: &str)
        -> Result<MessageRecord>;                               // first persisted row
    pub async fn unread(&self, name: &str) -> Result<Vec<MessageRecord>>;
    pub async fn mark_read(&self, name: &str, ids: &[String]) -> Result<()>;
    pub async fn post_event(&self, topic: TeamTopic, payload: EventPayload) -> Result<String>;  // sender team_name
    pub async fn publish_mutation(&self, mutation: &TaskMutation) -> Result<()>;
    pub async fn list_tasks(&self, filter: &TaskFilter) -> Result<Vec<TaskSummary>>;
    pub fn snapshot(&self) -> TeamSnapshot;                     // members + no tasks (sync view)
    pub async fn snapshot_with_tasks(&self) -> Result<TeamSnapshot>;
}

// spawner.rs — MemberSpawner over the harness factory and the spec roster
pub struct TeamMemberSpawner;
impl TeamMemberSpawner { pub fn new(spec: TeamAgentSpec, deps: TeamBuildDeps, session_id: impl Into<String>) -> Arc<Self>; }
#[async_trait] impl MemberSpawner for TeamMemberSpawner { /* spawn / despawn */ }

// team_agent.rs + build.rs — the agent itself
#[async_trait] pub trait SpecBuild { async fn build(&self) -> Result<Arc<TeamAgent>>;
                                     async fn build_with(&self, deps: TeamBuildDeps) -> Result<Arc<TeamAgent>>; }
impl SpecBuild for TeamAgentSpec;                 // the impl lives in build.rs
pub struct TeamAgent;       // Clone: cheap handle over one Arc'd interior state
impl TeamAgent {
    pub fn card(&self) -> &AgentCard;      pub fn team_name(&self) -> &str;
    pub fn role(&self) -> TeamRole;        pub fn session_id(&self) -> &str;
    pub fn spec(&self) -> &TeamSpec;       pub fn leader_member_name(&self) -> &str;
    pub fn state(&self) -> HarnessState;   // leader runtime state, else Idle
    pub fn backend(&self) -> Arc<TeamBackend>;
    pub fn gate(&self) -> Arc<InteractGate>;
    pub fn activity(&self) -> Arc<MemberActivityRegistry>;
    pub fn member(&self) -> Arc<TeamMember>;  pub fn runtime(&self) -> Arc<dyn MemberRuntime>;
    pub fn is_started(&self) -> bool;
    pub async fn start(&self) -> Result<()>;  pub async fn stop(&self) -> Result<()>;
    pub async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    pub fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn interact(&self, payload: InteractPayload) -> DeliverResult;
    pub async fn deliver(&self, payload: InteractPayload) -> DeliverResult;
    pub async fn snapshot(&self) -> TeamSnapshot;
    pub async fn spawn_member(&self, name: &str) -> Result<()>;
    pub async fn shutdown_member(&self, name: &str, lifetime: MemberLifetime, force: bool) -> Result<MemberStatus>;
    pub async fn auto_start_all(&self) -> Result<()>;
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskRecord>;
    pub async fn send_message(&self, to: Option<&str>, content: &str) -> Result<MessageRecord>;
    pub async fn members(&self) -> Vec<Arc<TeamMember>>;
}

// leader.rs — the coordination loop (`pub(crate)` surface)
pub const SETTLE_POLL_INTERVAL: Duration;
pub(crate) struct RunOutcome { pub output: AgentOutput, pub frames: Vec<AgentChunk> }
pub(crate) async fn run(agent: TeamAgent, query: Option<String>, session: Option<Arc<dyn Session>>,
                        deliver: Vec<InteractPayload>, sink: Option<ChunkSink>) -> Result<RunOutcome>;

// pool.rs — session → team runtime map
pub struct TeamRuntimePool;
impl TeamRuntimePool {
    pub fn new() -> Self;   pub fn insert(&self, session_id: &str, team: Arc<TeamAgent>) -> Option<Arc<TeamAgent>>;
    pub fn get(&self, session_id: &str, team_name: &str) -> Option<Arc<TeamAgent>>;
    pub fn get_for_session(&self, session_id: &str) -> BTreeMap<String, Arc<TeamAgent>>;
    pub fn remove(&self, session_id: &str, team_name: &str) -> Option<Arc<TeamAgent>>;
    pub fn remove_session(&self, session_id: &str) -> Vec<Arc<TeamAgent>>;
    pub fn find(&self, team_name: &str) -> Option<(String, Arc<TeamAgent>)>;   // includes the session id
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn snapshot(&self) -> BTreeMap<String, Vec<String>>;   // session → sorted team names
    pub fn teams_of_session(&self, session_id: &str) -> Vec<String>;  // sorted
}

// manager.rs — one process-wide facade, also usable as a local instance
pub struct TeamRuntimeManager;
impl TeamRuntimeManager {
    pub fn new(deps: TeamBuildDeps) -> Self;   pub fn with_defaults() -> Self;
    pub fn deps(&self) -> &TeamBuildDeps;      pub fn pool(&self) -> Arc<TeamRuntimePool>;
    pub async fn activate(&self, spec: &TeamAgentSpec, session_id: &str, inputs: &Value) -> Result<Arc<TeamAgent>>;
    pub fn live(&self, team_name: &str) -> Option<Arc<TeamAgent>>;
    pub fn live_state(&self, team_name: &str) -> Option<LiveTeamState>;
    pub async fn interact(&self, team_name: &str, payload: InteractPayload) -> DeliverResult;
    pub async fn stop_team(&self, team_name: &str, session_id: &str) -> Result<bool>;
    pub async fn finalize_member(&self, team_name: &str, member_name: &str) -> Result<()>;
    pub async fn remove(&self, team_name: &str) -> Result<bool>;
    pub async fn shutdown(&self) -> Result<()>;
    pub async fn snapshot(&self, team_name: &str) -> Option<TeamSnapshot>;
}
pub fn global_registry() -> Arc<TeamRuntimeManager>;
pub fn set_global_registry(manager: Arc<TeamRuntimeManager>);
pub fn clear_global_registry();

// adapter.rs — the multiagent seams over the team
impl openjiuwen_agent::Agent for TeamAgent;          // card / invoke / stream
impl openjiuwen_multiagent::Team for TeamAgent;      // card / kind / runtime / membership / invoke
```

## Build path (`TeamAgentSpec::build_with`)

1. `spec.validate()` — reserved names, dispatch knob rules (the spec's own messages).
2. `TeamSpec::from_spec(spec)` — the runtime mirror; language from `deps.language` else `spec.language`.
3. `opens_store` is injected: `deps.store` (never a second store).
4. One `InProcessMessager`-style transport per team: `deps` has no messager, so the build creates
   `create_messager(&spec.transport)` (in-process) and shares it with the backend and every member.
5. `TaskBoard::new(store, team_name, TaskBoardLimits::new().with_default_max_review_rounds(..)
   .with_max_subtask_depth(..))`.
6. `TeamMemberSpawner::new(spec.clone(), deps.clone(), session_id)` — the spawn seam used by
   `spawn_member` and stored in the backend.
7. `TeamMember` rows: the leader row (`TeamRole::Leader`, `Ready`-capable) plus one `Unstarted` row per
   teammate in the roster (`predefined_members` ∪ `agents` keys except `leader`/leader name). Rows are
   inserted through `TeamStore::insert_member` and wrapped in `TeamMember`.
8. The leader harness: `factory.create(&leader_spec, &MemberHarnessContext{..})` →
   `HarnessMemberRuntime::new(harness, session_id)` → `TeamMember::attach_runtime`.
9. The leader card is `AgentCard` named `<team>_<leader>` (Python's `<team>_<leader_member_name>`),
   description from `team_desc` else the leader `desc`.
10. One messager subscription for the leader (`Team|Task|Message`) whose sink pushes into an unbounded
    channel drained by the run loop.
11. A `MemberObserver` registered on the leader runtime forwards `state`/`round` into the activity
    registry and chunks into the run loop's chunk sinks.
12. `InteractGate::new()`, `MemberActivityRegistry::new()`, `HumanAgentInbox::new()` (disabled until
    `enable_hitt`).

## Run flow (`invoke` / `stream`)

1. `parse_initial_input(inputs, roster)` → `leader_query` + `deliveries`.
2. Each delivery is handed to `deliver` without touching the leader runtime: `HumanAgent` → inbox push +
   persisted message row + `message` event; `Operator`/`GroupChat`/`GodView` → gate + leader mailbox;
   `ExternalTeam` → `DeliverResult::unsupported`.
3. `start()` (idempotent): the leader `MemberRuntime::start(None)`; the gate opens.
4. The leader query is sent with `immediate = false` (a follow-up turn).
5. Loop on `tokio::select!`: member chunks, bus events, caller cancellation, and a settle tick.
   - teammate/leader chunks are forwarded tagged with `TeamOutputSchema{source_member, role, chunk}`;
   - mailbox/task events are translated into one leader `send` batched by `steer_batch_size`;
   - the loop ends when the leader runtime is `Idle` **and** `taskboard.is_settled()`, when the caller
     cancels, or when `stop()` fires.
6. The `team.completed` frame (`schema::stream::team_completed_frame`) is emitted last; the gate closes.
7. `invoke` returns `AgentOutput::answer(<concatenated leader answer>)`; `stream` yields the frames.
   A leader answer is accumulated from `llm_output` chunks (`payload.content`, else `payload.text`).

## Dispatch truth table (`decide_run_action`)

| live | session | lifecycle | paused / state | action |
| --- | --- | --- | --- | --- |
| none | any | any | — | `CreateTeam` |
| some | same | temporary | not paused | `AttachExisting` |
| some | same | temporary | paused | `ResumePaused` |
| some | same | persistent | not paused | `ResumeExisting` |
| some | same | persistent | paused | `ResumePaused` |
| some | different | temporary | any | `RejectCrossSession` |
| some | different | persistent | any | `ResumeExisting` |
| some | any | any | `state == Terminated` | `RejectStopped` |
| some | any | any | `lifecycle != requested` (persistent live, temporary requested) | `RejectLifecycle` |

Documented parity invariant: a temporary team never serves two sessions; a persistent team is resumed by
a new session instead of being rebuilt.

## Constraints

- Only `src/runtime/**` (plus `tests/runtime.rs`) is edited; `src/lib.rs`, `Cargo.toml`, `src/schema/`,
  `src/member/`, `src/harness/` and the sibling modules are read-only — a missing seam is adapted here and
  reported, never widened.
- `openjiuwen_common::error::{Error, Result}` only: `validation` for user input/bad names, `config` for a
  missing build dependency or an empty model pool, `not_found` for unknown members/tasks, `unsupported`
  for every deferred capability.
- Async-first tokio; `#[async_trait]` for object-safe async seams; `parking_lot` locks guard maps for one
  operation and are **never** held across `.await`; every spawned task is owned and cancellable
  (`tokio_util::sync::CancellationToken`).
- `tracing` only; no `println!`/`eprintln!` in this module.
- No `unsafe`, no `unwrap`/`expect` outside tests.
- Process-global state: exactly one new singleton here — the team runtime registry
  (`global_registry`), the crate's second documented singleton after the expert catalog.
- Files ≤1000 lines; split by responsibility (`leader.rs` holds the loop so `team_agent.rs` stays a
  surface).
- `--no-default-features` must build (no feature gates in this module).

## Routing Table

- `./mod.rs` → module map, flat re-exports
- `./deps.rs` → `TeamBuildDeps` and its defaults/builders
- `./allocator.rs` → `ModelAllocator`, `TeamModelAllocation`
- `./dispatch.rs` → `RunActionKind`, `RunAction`, `LiveTeamState`, `decide_run_action`
- `./initial_input.rs` → `InitialInput`, `parse_initial_input`
- `./backend.rs` → `TeamBackend` (members, mail, events, tasks, snapshot)
- `./build.rs` → the `SpecBuild` implementation and every build-only helper (leader record, harness, observer, roster rows, bus wiring)
- `./spawner.rs` → `TeamMemberSpawner` (`MemberSpawner` over the harness factory)
- `./team_agent.rs` → `TeamAgent`, `SpecBuild`, lifecycle, run entry points, `#[cfg(test)] test_support`
- `./leader.rs` → the coordination loop and its settle/batching rules
- `./pool.rs` → `TeamRuntimePool`
- `./manager.rs` → `TeamRuntimeManager`, `global_registry`/`set_global_registry`/`clear_global_registry`
- `./adapter.rs` → `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` implementations
- `./tests/` → crossed-seam integration suite (`support.rs`, `end_to_end.rs`, `members.rs`, `mail.rs`)

## Test Strategy

Unit tests beside each file plus `tests/runtime.rs` for the crossed seams; all offline and deterministic
(a `FakeHarness`/`FakeMemberHarnessFactory` with scripted answers, no sleeps beyond bounded polls).

- `deps.rs`: defaults are wired (native factory name, in-memory store, temp-rooted workspace), builders
  override one field each, `resolved_language` honours `Language::resolve`.
- `allocator.rs`: round-robin cycles in pool order and persists its cursor; `by_name` picks the named
  entry and falls back to the first; empty pool + `Some(name)` echoes the name with `index = 0`; empty pool
  + `None` is `Error::config`; `state_dict` round-trip.
- `dispatch.rs`: every row of the truth table.
- `initial_input.rs`: `$human @ai body` → one `HumanAgent` delivery and no leader query; `@dev-1 body` →
  delivery with the `user` sender; plain text, `#text`, `@all text` → leader query verbatim; nested
  `{"input":{"query":…}}` and plain-string inputs.
- `backend.rs`: mail routing (direct one row + one `message` event, broadcast one row, multicast N rows),
  member bookkeeping, task listing, snapshot ordering, `post_event` sender stamping.
- `team_agent.rs`: build (leader card `<team>_<leader>`, missing `agents["leader"]` → validation), unknown
  member → `not_found`, spawn → `ready`, factory error → `unstarted`, shutdown per lifetime,
  `auto_start_all`, `create_task` leader-only, workspace isolation per member.
- `leader.rs` / `tests/runtime.rs`: invoke with an immediate fake answer and an empty board → exactly one
  `team.completed` marker last and the leader answer in `AgentOutput.output`; an open task keeps the stream
  open until the board settles; cancellation ends the run.
- `manager.rs`: activate → live, session-mismatch rejection for a temporary team, `interact` unknown
  (`not_active`) and after `stop_team` (`gate_closed`/`not_active`), `finalize_member`, `shutdown`.
- `adapter.rs`: `team_id() == team_name`, `agent_count()`, `invoke` delegation, `kind() == "agent_teams"`.

Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`,
`cargo test -p openjiuwen-agent-teams --no-default-features` (549 crate lib tests green in both),
`cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green for `src/runtime/**`
(crate-wide diagnostics belong to the crate root node; the `mcp`/`skill` modules are implemented).

## Notes for Agents / Deviations from Python

- Python's `CoordinationKernel` (a large event-driven supervisor) is reimplemented as one documented loop:
  a `select!` over chunks, bus events, cancellation and a settle tick. Batching, idle debounce and
  escalation heuristics are simplified; the observable contract (one completed marker, deliveries never
  reaching the leader, settle = empty board) is preserved.
- Team members are spawned through the injected `MemberHarnessFactory`, so nothing in this module builds a
  DeepAgent directly; the multiagent `TeamRuntime` is used only as a session/registry seam and its message
  bus is not used (Python `agent_teams` has its own envelope).
- `TeamSpec` (not `TeamAgentSpec`) is what the running team carries, mirroring Python's runtime mirror.
- A teammate runs the same loop but creates no tasks and never dispatches: `create_task` on a teammate is
  `Error::validation("only the leader can create tasks")`.
- The team roster is materialized as rows at build time (`unstarted` teammates); Python creates rows lazily
  on bootstrap. This makes `snapshot()` answer before the first spawn.
- `spawn_mode = process`, external CLI/bridge members, distributed transports, the worktree workspace
  backend and the model router stay deferred through the injected seams (`Error::Unsupported` with the
  Python capability name).
- `Team::invoke` in `openjiuwen-multiagent` carries an `AgentTeamSession`, while `Agent::invoke` takes
  `Option<Arc<dyn Session>>`; the adapter ignores the multiagent session (documented; `openjiuwen-session`'s
  `AgentTeamSession` is not a `dyn Session`).
- `Team::add_agent` accepts a card + `AgentProvider`; the provider is ignored — a member's brain always
  comes from the harness factory, and only an `unstarted` roster row is created. Because that seam is
  synchronous while `TeamStore::insert_member` is async, the row write is spawned on the current tokio
  runtime (a missing runtime is `Error::config`).
- `SpecBuild` is `#[async_trait]` (async `build`/`build_with`): the pinned build order awaits the store,
  the workspace provider, the harness factory and the team-row upsert, so a synchronous trait cannot hold
  it. The crate-level document still sketches synchronous signatures.
- `leader::run` returns the `AgentOutput` directly; frames flow through a `ChunkSink`, so `stream` yields
  them live and `invoke` ignores them.
- One run at a time per team: the bus receiver is single-consumer (`TeamAgent::take_events`), so a second
  concurrent `invoke`/`stream` on the same handle is unsupported (documented, not enforced).
- Settle is two consecutive 25 ms ticks (`SETTLE_POLL_INTERVAL`, `REQUIRED_SETTLE_TICKS = 2`) rather than
  Python's 2 s idle debounce: a run that answers immediately still emits exactly one completed marker.
- `build` always produces the leader handle. Teammates are not a second public constructor: they are
  `TeamMember` rows whose harness is materialized by `spawn_member` through the same factory seam.
- `TeamRuntimeManager` keys a persistent team under the session that first activated it, even when a later
  session resumes it; `shutdown()` never clears the global registry (the caller does).
- `team_agent.rs` is 852 production lines plus its `test_support`/unit tests (the sanctioned long file of
  this node); the build path is split into `build.rs`.
