# src/runtime — team agent, leader loop, dispatch, runtime manager, wiring

## Intent

The coordination half of `openjiuwen-agent-teams`: the `TeamAgent` (one type for both roles), its build
path from a `TeamAgentSpec`, the leader's run loop, the organisation facade over the store/task board/bus,
the pool and process-global runtime manager used by the CLI/MCP, and the model-pool router.
Python parity sources: `openjiuwen/agent_teams/agent/team_agent.py`, `agent/agent_configurator.py`,
`runtime/{manager,dispatch,pool,gate}.py`, `models/allocator.py`.
This node also owns the eight **metadata-driven wiring points**: member spawn kind, fork, workspace isolation, the organisation layer, KV-cache affinity, attach hosting, per-member checkpoints and team memory.
Model routing, observability signals and the monitor stream are wired at build time as well (they are not metadata keys).

The module owns *policy and orchestration*: who runs, what a run means, when a team is done, how an
external payload reaches a member. It owns no state of its own beyond the live-runtime maps: rows live in
`crate::storage`, task decisions in `crate::taskboard`, member status in `crate::member`, transports in
`crate::messager`, routing grammar in `crate::interaction`.

`TeamAgent` carries a `TeamRole`: a leader dispatches and steers, a teammate polls the board and answers
mail. Both roles run the same type, per Python.

## API Surface

```rust
// deps.rs — the injected build inputs (Clone)
pub struct TeamBuildDeps {
    pub harness_factory: Arc<dyn MemberHarnessFactory>,   // required
    pub store: Arc<dyn TeamStore>,                        // required
    pub workspace: Arc<dyn WorkspaceProvider>,            // required
    pub language: Option<String>,
    pub observability_hooks: Vec<Arc<dyn ObservabilityHooks>>,   // empty = no-ops
    pub kv_cache: Option<Arc<TeamKvCacheAffinity>>,       // None = built from metadata
    pub routing: Option<Arc<TeamModelRouter>>,            // None = built from model_pool
    pub checkpoints: Option<Arc<dyn MemberCheckpointStore>>,   // None = resolved from metadata
    pub memory: Option<Arc<TeamMemoryWiring>>,                 // None = resolved from metadata
    pub store_injected: bool,                             // set by `with_store`, cleared by `default()`
}
impl TeamBuildDeps {
    pub fn new() -> Self;                                  // == Default
    pub fn with_harness_factory(self, Arc<dyn MemberHarnessFactory>) -> Self;
    pub fn with_store(self, Arc<dyn TeamStore>) -> Self;   // also sets `store_injected`
    pub fn has_injected_store(&self) -> bool;              // the store-resolution switch
    pub fn with_workspace(self, Arc<dyn WorkspaceProvider>) -> Self;
    pub fn with_language(self, impl Into<String>) -> Self;
    pub fn with_observability_hook(self, Arc<dyn ObservabilityHooks>) -> Self;  // pushes
    pub fn with_kv_cache(self, Arc<TeamKvCacheAffinity>) -> Self;
    pub fn with_routing(self, Arc<TeamModelRouter>) -> Self;
    pub fn with_checkpoint_store(self, Arc<dyn MemberCheckpointStore>) -> Self;
    pub fn has_injected_checkpoints(&self) -> bool;        // `checkpoints.is_some()`
    pub fn with_memory(self, Arc<TeamMemoryWiring>) -> Self;
    pub fn has_injected_memory(&self) -> bool;             // `memory.is_some()`
    pub fn resolved_language(&self) -> &'static str;       // unknown/None -> "cn"
}
// Default: NativeMemberFactory::from_env(), InMemoryTeamStore, LocalWorkspaceProvider rooted at
// std::env::temp_dir().join("openjiuwen-agent-teams"), language None, no hooks/kv/routing/checkpoints/memory.

// spec_meta.rs — the metadata parsers (all tolerant; an absent key keeps the default)
pub const SPAWN_METADATA_KEY: &str = "spawn";        pub const FORK_METADATA_KEY: &str = "fork";
pub const WORKSPACE_METADATA_KEY: &str = "workspace"; pub const ORGANIZATION_METADATA_KEY: &str = "organization";
pub const KV_CACHE_METADATA_KEY: &str = "kv_cache";   pub const ATTACH_METADATA_KEY: &str = "attach";
pub const CHECKPOINT_METADATA_KEY: &str = "checkpoint";
pub const ATTACH_SOCKET_ENV: &str = "OPENJIUWEN_TEAM_SOCKET";
pub struct ForkDirective { pub parent: String, pub task: Option<ForkTaskOverride>, pub tools: ForkToolPolicy }
pub struct OrganizationBlock { pub spec: OrganizationSpec, pub rules: DelegationRules }
pub struct AttachConfig { pub socket: PathBuf, pub required: bool }
pub struct CheckpointConfig { pub policy: MemberCheckpointPolicy, pub store: CheckpointStoreConfig }
pub enum CheckpointStoreConfig { Memory, File { dir: Option<PathBuf> } }   // Default = Memory
pub fn spawn_options(&TeamAgentSpec) -> MemberSpawnOptions;              // bare or {"options":..}
pub fn fork_directive(&TeamAgentSpec) -> Option<ForkDirective>;          // spawn.fork, else fork
pub fn isolation_mode(&TeamAgentSpec) -> IsolationMode;                  // "worktree" | {"isolation":..}
pub fn organization_block(&TeamAgentSpec) -> Result<Option<OrganizationBlock>>;  // Validation on a bad chart
pub fn kv_cache_config(&TeamAgentSpec) -> KvCacheConfig;                 // enabled=true by default
pub fn attach_config(&TeamAgentSpec) -> Option<AttachConfig>;            // metadata, else env socket
pub fn checkpoint_config(&TeamAgentSpec) -> CheckpointConfig;            // bare policy or {"policy","store","dir"}
// checkpoints.rs — per-member checkpoint identity and store resolution (`pub(crate)`)
pub const CHECKPOINT_DIR: &str = ".openjiuwen/checkpoints";
pub(crate) fn member_card_id(&DeepAgentSpec, member_name: &str) -> String;         // card id, else member name
pub(crate) fn team_checkpoint_store(&CheckpointConfig, Option<&Arc<dyn MemberCheckpointStore>>) -> Option<Arc<dyn MemberCheckpointStore>>;
pub(crate) fn member_checkpoint_store(&CheckpointConfig, Option<&Arc<dyn MemberCheckpointStore>>, &Path) -> Option<Arc<dyn MemberCheckpointStore>>;
pub(crate) fn wire_runtime(HarnessMemberRuntime, card_id: &str, &CheckpointConfig,
                           Option<&Arc<dyn MemberCheckpointStore>>, &Path) -> HarnessMemberRuntime;

// routing.rs — model pool → one endpoint per member (`Arc`-shared, not `Clone`)
pub struct ModelSelection { pub allocation: TeamModelAllocation, pub entry: Option<ModelPoolEntry> }
impl ModelSelection { pub fn model_name(&self) -> &str; }
pub struct TeamModelRouter;
impl TeamModelRouter {
    pub fn new(entries: &[ModelPoolEntry], strategy: ModelPoolStrategy) -> Result<Self>;
    pub fn empty() -> Self;   pub fn for_spec(&TeamAgentSpec) -> Result<Self>;
    pub fn is_empty(&self) -> bool;   pub fn strategy(&self) -> ModelPoolStrategy;
    pub fn allocator(&self) -> &ModelAllocator;   pub fn pool(&self) -> &ModelPool;
    pub fn select(&self, Option<&str>, TeamRole) -> Result<ModelSelection>;              // advances
    pub fn select_for(&self, name: &str, Option<&str>, TeamRole) -> Result<ModelSelection>;  // memoized
    pub fn model_ref(&self, Option<&str>, TeamRole) -> Option<String>;                    // advances
    pub fn model_ref_for(&self, name: &str, Option<&str>, TeamRole) -> Option<String>;    // memoized
    pub fn effective_spec(&self, &DeepAgentSpec, Option<&str>, TeamRole) -> Result<(DeepAgentSpec, ModelSelection)>;
    pub fn effective_spec_for(&self, name: &str, &DeepAgentSpec, Option<&str>, TeamRole) -> Result<(DeepAgentSpec, ModelSelection)>;
}

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

// observability_bind.rs — signals + monitor emitter over the observability hooks
pub struct TeamSignals;                                  // Clone
impl TeamSignals {
    pub fn new(team_name: impl Into<String>, hooks: Vec<Arc<dyn ObservabilityHooks>>) -> Self;
    pub fn observability(&self) -> Arc<TeamObservability>;   pub fn team_name(&self) -> &str;
    pub fn install_monitor(&self, MonitorEmitter);   pub fn monitor_emitter(&self) -> Option<MonitorEmitter>;
    pub fn monitor_stream(&self, usize, BackpressurePolicy) -> MonitorReceiver;   // replaces the emitter
    pub fn emit(&self, MonitorEvent);
    pub fn team_started(&self, Option<&str>);   pub fn team_finished(&self);
    pub fn member_spawned(&self, &str, TeamRole);   pub fn member_status(&self, &str, TeamRole, MemberStatus);
    pub fn member_stopped(&self, &str, TeamRole, &str);   pub fn round_started(&self, &str, u32);
    pub fn round_finished(&self, &str, u32, RoundOutcome, u64);
    pub fn task_created(&self, &TaskCreatedEvent);   pub fn task_status(&self, &TaskStatusChangedEvent, actor: &str);
    pub fn delegation(&self, from: &str, to: &str, task_id: &str, reason: &str);
}

// org_bind.rs — the organisation layer over the chart, router, task pool and mailbox
pub const ORG_TASK_EVENT: &str = "org_task";
pub struct OrgBinding;
impl OrgBinding {
    pub fn build(&OrganizationBlock, Arc<dyn TeamStore>, Option<OrgTaskEventSink>) -> Result<Self>;
    pub fn organization_id(&self) -> &str;   pub fn chart(&self) -> &Arc<OrgChart>;   pub fn spec(&self) -> &OrganizationSpec;
    pub fn rules(&self) -> &DelegationRules;   pub fn router(&self) -> &Arc<DelegationRouter>;
    pub fn pool(&self) -> &Arc<OrgTaskPool>;   pub fn mailbox(&self) -> &Arc<OrgMailbox>;
    pub fn teams(&self) -> Vec<(String, OrgRole)>;
    pub fn check_delegation(&self, &DelegationRequest) -> Result<()>;          // Validation when refused
    pub async fn create_task(&self, OrgTaskSpec) -> Result<OrgTaskMutation>;
    pub async fn task(&self, &str) -> Result<Option<OrgTask>>;
    pub async fn delegate(&self, task_id: &str, from_team: &str, from_member: &str,
                          to_team: &str, to_member: Option<&str>) -> Result<OrgTaskOpResult>;
    pub async fn unread(&self, team: &str) -> Result<Vec<OrgMessage>>;
    pub async fn acknowledge(&self, team: &str, ids: &[String]) -> Result<usize>;
    pub async fn send_mail(&self, from_team: &str, to_team: &str, subject: &str, body: &str) -> Result<OrgMessage>;
    pub async fn broadcast_mail(&self, from_team: &str, subject: &str, body: &str) -> Result<OrgMessage>;
}
pub fn org_task_sink(Arc<TeamBackend>) -> OrgTaskEventSink;   // publishes on TeamTopic::Task

// attach_host.rs — the Unix-socket host the CLI/MCP can dial
pub struct AttachHost;
impl AttachHost { pub async fn start(&AttachConfig, Arc<dyn AttachHandler>) -> Result<Self>;
                  pub fn path(&self) -> &Path;   pub async fn shutdown(self); }
pub struct BackendAttachHandler;
impl BackendAttachHandler { pub fn new(Arc<TeamBackend>, leader: &str, CancellationToken) -> Self; }  // impl AttachHandler
fn map(error: &Error) -> AttachError;

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

// backend.rs — the facade exposed as TeamAgent::backend(); also the signals/org/monitor carrier
impl TeamBackend {
    pub fn new(team_spec: TeamSpec, store: Arc<dyn TeamStore>, taskboard: Arc<TaskBoard>,
               messager: Arc<dyn Messager>, spawner: Arc<dyn MemberSpawner>,
               workspace: Arc<dyn WorkspaceProvider>, activity: Arc<MemberActivityRegistry>) -> Self;
    pub fn with_signals(self, TeamSignals) -> Self;      pub fn signals(&self) -> Option<TeamSignals>;
    pub fn monitor_stream(&self, usize, BackpressurePolicy) -> Option<MonitorReceiver>;
    pub fn install_org(&self, Arc<OrgBinding>);          pub fn org(&self) -> Option<Arc<OrgBinding>>;
    pub fn team_name(&self) -> &str;        pub fn spec(&self) -> &TeamSpec;
    pub fn store(&self) -> Arc<dyn TeamStore>;   pub fn tasks(&self) -> &Arc<TaskBoard>;
    pub fn messager(&self) -> &Arc<dyn Messager>; pub fn spawner(&self) -> &Arc<dyn MemberSpawner>;
    pub fn workspace(&self) -> &Arc<dyn WorkspaceProvider>;
    pub fn activity(&self) -> &Arc<MemberActivityRegistry>;
    pub fn members(&self) -> Vec<Arc<TeamMember>>;              // sorted by member_name
    pub fn member(&self, name: &str) -> Option<Arc<TeamMember>>;   pub fn member_names(&self) -> Vec<String>;
    pub fn add_member(&self, member: Arc<TeamMember>);           pub fn remove_member(&self, name: &str) -> Option<Arc<TeamMember>>;
    pub async fn upsert_team_info(&self, info: TeamInfoRecord) -> Result<()>;
    pub async fn send_message(&self, from: &str, to: Option<&str>, content: &str) -> Result<MessageRecord>;
    pub async fn unread(&self, name: &str) -> Result<Vec<MessageRecord>>;
    pub async fn mark_read(&self, name: &str, ids: &[String]) -> Result<()>;
    pub async fn post_event(&self, topic: TeamTopic, payload: EventPayload) -> Result<String>;
    pub async fn publish_mutation(&self, mutation: &TaskMutation) -> Result<()>;   // emits task signals
    pub async fn list_tasks(&self, filter: &TaskFilter) -> Result<Vec<TaskSummary>>;
    pub fn snapshot(&self) -> TeamSnapshot;   pub async fn snapshot_with_tasks(&self) -> Result<TeamSnapshot>;
}

// spawner.rs — MemberSpawner over the harness factory, the roster, the spawn options and the fork
impl TeamMemberSpawner {
    pub fn new(TeamAgentSpec, TeamBuildDeps, impl Into<String>) -> Arc<Self>;
    pub fn with_parts(TeamAgentSpec, TeamBuildDeps, String, MemberSpawnOptions,
                      Option<ForkDirective>, Arc<TeamModelRouter>) -> Arc<Self>;
    pub fn spec(&self) -> &TeamAgentSpec;   pub fn deps(&self) -> &TeamBuildDeps;
    pub fn session_id(&self) -> &str;       pub fn team_name(&self) -> &str;
    pub fn options(&self) -> &MemberSpawnOptions;  pub fn fork(&self) -> Option<&ForkDirective>;
    pub fn routing(&self) -> &Arc<TeamModelRouter>;
    pub fn memory(&self) -> Option<Arc<TeamMemoryWiring>>;   // teammates mount their tools from it
    pub fn install_memory(&self, Arc<TeamMemoryWiring>);     // the build path installs the resolved wiring
    pub fn spawn_kind(&self, &str) -> Option<MemberSpawnKind>;
    pub fn forked_member(&self, &str) -> Option<ForkedMemberSpec>;
    pub fn forked_context(&self, &str) -> Result<Option<ForkedMemberSpec>>;
    pub fn deep_agent_spec(&self, &str) -> Option<&DeepAgentSpec>;
    pub fn member_spec(&self, &str) -> Option<TeamMemberSpec>;
    pub fn member_names(&self) -> Vec<String>;                 // predefined ∪ agents keys − leader
}

// team_agent/ + build.rs — the agent itself
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
    pub fn spawner(&self) -> Arc<TeamMemberSpawner>;
    pub fn kv_cache(&self) -> Arc<TeamKvCacheAffinity>;  pub fn kv_env(&self, &str) -> Option<BTreeMap<String,String>>;
    pub fn memory(&self) -> Option<Arc<TeamMemoryWiring>>;   // the team's wiring, `None` when disabled
    pub fn signals(&self) -> TeamSignals;  pub fn monitor_stream(&self, usize, BackpressurePolicy) -> Option<MonitorReceiver>;
    pub fn fork_directive(&self) -> Option<&ForkDirective>;  pub fn org(&self) -> Option<Arc<OrgBinding>>;
    pub fn attach_socket(&self) -> Option<PathBuf>;
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
    pub async fn create_task(&self, spec: TaskGraphSpec) -> Result<TaskRecord>;   // leader only
    pub async fn create_org_task(&self, OrgTaskSpec) -> Result<OrgTaskMutation>;
    pub async fn delegate_org_task(&self, task_id: &str, to_team: &str, to_member: Option<&str>) -> Result<OrgTaskOpResult>;
    pub async fn send_message(&self, to: Option<&str>, content: &str) -> Result<MessageRecord>;
    pub async fn members(&self) -> Vec<Arc<TeamMember>>;
    pub fn roster(&self) -> Vec<String>;
}

// leader.rs — the coordination loop (`pub(crate)` surface)
pub const SETTLE_POLL_INTERVAL: Duration;
pub(crate) type ChunkSink = UnboundedSender<Result<AgentChunk>>;
pub(crate) async fn run(agent: TeamAgent, query: Option<String>, session: Option<Arc<dyn Session>>,
                        deliver: Vec<InteractPayload>, sink: Option<ChunkSink>) -> Result<AgentOutput>;

// health.rs — the sync, read-only liveness projection (no new state, no probe)
pub struct MemberHealth { pub name: String, pub role: TeamRole, pub status: MemberStatus,
                          pub execution: ExecutionStatus, pub alive: bool }        // serde snake_case
pub struct TeamHealth { pub team_name: String, pub started: bool,
                        pub leader: Option<MemberHealth>, pub members: Vec<MemberHealth> }
impl TeamBackend { pub fn health(&self) -> TeamHealth; }   // always `started: false`
impl TeamAgent { pub fn health(&self) -> TeamHealth; }     // `started` from the handle

// pool.rs / manager.rs / adapter.rs — listed in the routing table below
```

`observer.rs`: `MemberStreamObserver::new(activity, Weak<TeamMember>, role, member_name, TeamSignals, chunk_sinks)`
— it turns harness `state`/`round` callbacks into `MemberActivityRegistry` updates, `member_status`/`round_*`
signals and forwarded chunks, so every member (not only the leader) feeds the same sinks.

## Metadata-driven wiring

Eight optional keys under `TeamAgentSpec::metadata` tune the runtime; `spec_meta.rs` parses seven of them and
every parser except `organization` degrades to its default with a `warn!` on a malformed value.
`memory` is parsed by `crate::memory::memory_config` (memory node) and resolved by `memory_bind.rs`.

| key | accepted shapes | effect |
| --- | --- | --- |
| `spawn` | a bare `MemberSpawnOptions` (`{"kind": "in_process"\|"process"\|"external_cli"\|"ssh", "process": {…}, "external_cli": {…}, "ssh": {…}}`) or a wrapper `{"options": …, "fork": …}` | which `MemberSpawner` materializes a member; the typed `spawn_mode: process` is honoured when the metadata names no kind; malformed ⇒ warn + in-process |
| `fork` | `{"parent": "dev-1", "task": {…}\|null, "tools": {…}}`, also read from `spawn.fork`; a blank parent disables it | the member is forked from `parent`: its blueprint becomes `org::fork_member_context(..)` (`ForkedMemberSpec`, cached in the spawner and readable through `forked_member`), and its KV identity is registered with `register_child` |
| `workspace` | `"worktree"` or `{"isolation": "worktree"}`; anything else logs and stays local; absent ⇒ `IsolationMode::None` | `IsolationMode` handed to `WorkspaceProvider::allocate` for the leader and each spawned member (backend in `src/workspace/`) |
| `organization` | `{"spec": OrganizationSpec, "rules": DelegationRules}` | installs an `OrgBinding` on the backend (chart, delegation router, org task pool, org mailbox); a configured-but-broken chart fails `build_with` with `Validation` |
| `kv_cache` | `{"enabled": bool, "max_entries": n}`; absent ⇒ `enabled: true` | builds the `TeamKvCacheAffinity` used for register → env → release |
| `attach` | `"/path/team.sock"` or `{"socket"\|"path": "…", "required": bool}`; blank ⇒ unset; `OPENJIUWEN_TEAM_SOCKET` is the fallback and an explicit metadata socket wins | `start()` binds an `AttachHost` that serves `AttachOp`s over a Unix socket |

| `checkpoint` | `CheckpointConfig`: `enabled`, `resume`, `policy` and `store` (`"memory"` or `{"store": "file", "dir": "…"}`); absent ⇒ enabled, in-memory, resume-if-available | the store, policy and card id every member runtime is wired with (`./checkpoints.rs`); the build path resolves the team-shared store once, and `file` roots each member under its own workspace unless `dir` is named |
| `memory` | `MemoryConfig`: `enabled`, `store`, `extractor`, `export_on_stop`, `import_on_build`, `hooks`; absent or `null` ⇒ **no team memory**, `{"enabled": false}` ⇒ no wiring either | the team-memory wiring (`./memory_bind.rs`): one per team, its `TEAM_MEMORY.md` at the leader's workspace root, its toolkit on every member context and its turn buffer fed by `TeamBackend::send_message` |
Per-kind launch shapes (`process`, `external_cli`, `ssh`) live in `src/member/CONTEXT.md`; the org chart
shapes (`OrganizationSpec`, `DelegationRules`, `OrgTaskSpec`) live in `src/org/CONTEXT.md`.
`TeamBuildDeps` adds only *injection* hooks: `with_observability_hook`, `with_kv_cache`, `with_routing`, `with_checkpoint_store`, `with_memory`.
It has no monitor/org/attach builder — those come from the metadata; `monitor_stream` is the monitor knob.

### Checkpoint wiring

`metadata["checkpoint"]` is parsed by `checkpoint_config`; `build_with` resolves the team-shared store once (`checkpoints::team_checkpoint_store`: an injected store wins, `memory` mints one `InMemoryMemberCheckpointStore`, `file` pins none) and every member runtime goes through `checkpoints::wire_runtime` with the card id, the policy, `cwd = workspace root` and that store.
Identity is `(member session id, card id)`: the member session is `"<team session>:<member>"` — both the runtime session and the harness `host_session_id`, because `start(None)` makes the derived id the host session — and the card id is the effective spec's card id, else the member name.
`start` restores a compatible record's provider envelope (a payload-only, mismatched or unreadable record starts cold), a finished turn and `stop` capture one record for the identity, and a disabled policy neither resolves a store nor attaches a sink.
A retired (shut-down) roster row cannot be re-spawned — `MemberStatus` has no `shut_down → unstarted` edge — so a member restart is a fresh row (`TeamBackend::remove_member` + `TeamAgent::spawn_member`).
### Memory wiring
`memory_bind::resolve_memory` runs **once per team**, right after the leader's workspace exists (the workspace root anchors the document), and answers the one `Arc<TeamMemoryWiring>` the whole team shares.
Resolution order: an injected `deps.memory` wins (a configured `metadata["memory"]` block is then `warn!`-logged as ignored, like an injected store) → a present non-null `metadata["memory"]` block opens one through `TeamMemoryWiring::open` (which also performs the `import_on_build` read of an existing `TEAM_MEMORY.md`) → no block means no wiring.
A block that parses to `enabled: false` allocates no wiring, so no hook runs, no turn is buffered and nothing is written.
`open` failing fails the build (`Error::Validation` / the store's error) and hands the freshly allocated leader workspace back; every *hook* failure is advisory and never fails the operation that triggered it.
`build_with` installs the wiring on the backend (`install_memory`, the turn seam) and the spawner (`install_memory`, the teammate toolkits) and mounts the leader's `MemberMemoryToolkit` on its own context.

| hook | call site | failure rule |
| --- | --- | --- |
| `on_session_start(team session)` | `TeamAgent::run_start`, after the transport starts and before the leader runtime does | `warn!`, `start` continues |
| `on_member_join(member)` | `TeamAgent::spawn_member`, after the runtime start succeeded and the KV identity registered | `warn!`, the spawn still succeeds |
| `on_member_leave(member)` | `TeamAgent::shutdown_member` (only when the pre-shutdown row was live) and `TeamAgent::stop` for every row still live | `warn!`, the retirement continues |
| `flush()` then `export()` | end of `TeamAgent::stop`, after the workspaces are released | each logged, the teardown still completes |
| `export()` | `TeamAgent::snapshot`, after the roster/board read | logged; the snapshot stays infallible and its shape unchanged |
| `on_turn_message(from, to, content, now)` | `TeamBackend::direct_row` / `broadcast_row`, after the row is persisted and published | `warn!`, the send still succeeds |

Live means `memory_bind::is_live`: a row that started at least once and has not departed — so a failed spawn leaves no join record and an unstarted row gets no handoff note, keeping the join/leave pair symmetric.
The turn seam is the mailbox, not the leader loop: every coordination message (member send, human-agent delivery, leader-bound operator/group-chat/god-view payload) ends in exactly one `direct_row` (one addressee, one turn) or one `broadcast_row` (`to: None`), so relayed content is never buffered twice and the memory tools' own `remember`/`forget` writes buffer nothing.
`stop` closes memory out last: handoff notes for live members, then `flush`, then `export` when `MemoryConfig::export_on_stop` is set (the export gates on that flag itself); a repeated `stop`/`export` flushes nothing new and stores no new record, though the rendered document's `generated_at` header moves.
The toolkit reaches a member through `MemberHarnessContext::with_memory_toolkit(wiring.toolkit(member, Some(team session)))`: the build path does it for the leader, `TeamMemberSpawner::build_harness_runtime` for a teammate, and `NativeMemberFactory` turns it into the four `team_memory_*` tools in `src/harness/`.
A teammate's toolkit lookup happens before the factory call, so an unusable member/session id fails that member's spawn with the same `Validation` error kind as the surrounding build.

### Model routing

`TeamModelRouter` = `ModelAllocator` + `ModelPool` + a per-member selection memo.
Invariant: **one allocation per member**, reused by the roster row (`MemberRecord::model_ref`), the
`DeepAgentSpec` handed to `MemberHarnessFactory` and the KV identity.
`round_robin` hand outs pool entries in roster order (leader first, then sorted teammates) and advances on every member; it ignores a requested model name (Python allocator semantics).
`by_model_name` honours the requested name and falls back to the first entry.
An empty pool leaves the member's own spec untouched and returns `entry: None`; `effective_spec_for` is then a no-op.
`build_with` always builds the router from `model_pool` — an injected `deps.routing` is replaced there and only survives in a directly constructed `TeamMemberSpawner`.
Per-member memoization also means a re-spawn keeps the member's endpoint and the cursor does not drift.

### KV-cache affinity

Registration is register → env → release: the build registers the leader under the team session,
`spawn_member` registers a teammate **after** its start succeeds (a failed start leaves no entry),
`shutdown_member` releases one member and `stop()` releases the whole team.
`kv_env(member)` renders the advisory env bag, which `TeamMemberSpawner` copies onto child-process commands
(`process` and `external_cli` only; the SSH transport carries no env bag and an in-process member carries the
identity through its session).
The identity is `MemberCacheKey::new(team, session, member)`; a fork child is registered with `register_child(parent_key, child_key, model)` so it inherits the parent's cache lineage.

### Workspace lifecycle
`build_with` allocates the leader's workspace and `spawn_member` allocates one per teammate, both through `deps.workspace`.
Every allocated `MemberWorkspace` is retained on the agent handle, keyed by member name (leader included), for the team's lifetime.
`shutdown_member` releases that member's workspace after its runtime is stopped and its child despawned; `stop()` drains and releases every workspace still retained (leader plus members whose row is already shut down) before `team_finished`.
A release failure is `warn!`-logged and never propagated, and both teardown paths are idempotent: each workspace is released exactly once.
A failed spawn hands its freshly allocated workspace straight back instead of retaining it for a member that never got a runtime.
### Observability and monitor

`TeamSignals` fans one event out to a `TeamObservability` built from the `ObservabilityHooks` in
`TeamBuildDeps`, and to the optional `MonitorEmitter`.
Signals fired by this node: `team_started` (root span opened, `start`), `team_finished` (root span closed,
`stop`), `member_spawned`/`member_status`/`member_stopped`, `round_started`/`round_finished`,
`task_created`/`task_status` (through `TeamBackend::publish_mutation`), and `delegation`.
Monitor events are emitted synchronously on the calling task, so a reader must install `monitor_stream`
before the event it wants; a later `monitor_stream` replaces the emitter, so events reach the newest reader only
(bounded queue + `BackpressurePolicy`).
Hooks must not block — they run inline on the leader/observer task.

### Organisation layer

When `metadata["organization"]` is configured, `build_with` installs an `OrgBinding` on the backend:
`check_delegation` runs the `DelegationRouter` over the chart (depth 0 coerces to 1; a refused request is
`Validation` and leaves the board untouched), `delegate` re-checks the router against the target's live open-task
count and then publishes the hop, `TeamAgent::delegate_org_task` records the delegation signal, and an accepted
delegation emits an `org_task` payload on `TeamTopic::Task` through the installed sink.
Mail is acknowledged only after the leader flush succeeded: `flush` restores the batch on a send error, so the
message ids stay unread and are retried on the next run/settle.
A team with no organisation block keeps the flat board (`org()` is `None`, `create_org_task` and `delegate_org_task` are `Error::Config`, while a hop the router refuses inside a configured organisation is `Validation`).

### Attach policy

`start()` binds the host after the messager and the leader runtime started, and only once.
`AttachHost` serves one request/response `AttachOp` per connection through `BackendAttachHandler`, which maps
each op onto the backend: `list_members`, `member_status`, `read_inbox`, `send_member`, `broadcast`,
`list_tasks`, `claimable_tasks`, `task_detail`, `create_task`, `update_task`, `claim_task`, `complete_task`,
`invoke_tool`, `snapshot`, `stop_team` (cancels the run token and answers `{"stopping": true}`).
A bind failure is **not fatal**: it is logged (`warn!`) and the team runs without the socket, unless the config
sets `required: true`, in which case `start()` returns the error (a `SysOp` error when the socket directory
cannot be created) and the started flag is rolled back.
`stop()` shuts the host down and removes the socket file; `TeamRuntimeManager::stop_team`/`remove`/`shutdown`
reach the same path through `TeamAgent::stop`, so a stopped team never leaves a live socket behind.
Hosting is `#[cfg(unix)]`; elsewhere `crate::attach::AttachServer::bind` returns `Unsupported` and a non-required
config therefore just logs.

## Build path (`TeamAgentSpec::build_with`)

1. `spec.validate()` — reserved names, dispatch knob rules (the spec's own messages).
2. `TeamSpec::from_spec(spec)` — the runtime mirror; language from `deps.language` else `spec.language`.
3. `create_messager(&spec.transport)` — one in-process transport shared by the backend and every member.
   The store comes from `resolve_store(spec, &deps)?` (in `build.rs`), never a second one:
   an injected `deps.store` (`TeamBuildDeps::has_injected_store`, set by `with_store`) always wins — a
   non-memory `spec.storage` kind is only `warn!`-logged as ignored, and `open_store` is never called;
   with no injection, a blank or `memory` kind uses `deps.store` (the `InMemoryTeamStore` default);
   with no injection and a real kind, `crate::storage::open_store(&spec.storage)?` opens it, propagating
   `Error::Unsupported` for `postgresql`/`mysql`/unknown kinds. Before a `sqlite` open,
   `ensure_sqlite_parent` best-effort creates the parent directory of a path-shaped target
   (`params` keys `path`/`url`/`database`; `:memory:`, `mode=memory` and `sqlite:` URLs are left alone),
   logging and continuing on failure — so the first write cannot fail on a missing directory.
4. Wiring inputs: `TeamModelRouter::for_spec` (or the injected router for a direct spawner), the
   `TeamKvCacheAffinity` (injected, else `kv_cache_config`), `TeamSignals::new(team, hooks)`, and the five
   metadata parsers (`spawn_options`, `fork_directive`, `isolation_mode`, `attach_config`, `checkpoint_config`).
   `checkpoint_config` is resolved into the team store before the spawner is built
   (`team_checkpoint_store`: the injected store wins, `memory` mints one `InMemoryMemberCheckpointStore`,
   `file` pins none) and written back into `deps.checkpoints`, so the leader and every teammate share it.
   Team memory is *not* resolved yet: its document root is the leader's workspace, allocated in step 9.
5. `TaskBoard::new(store, team_name, TaskBoardLimits::new().with_default_max_review_rounds(..)
   .with_max_subtask_depth(..))`, then `TeamBackend::new(..).with_signals(signals)`.
6. `organization_block(spec)?` → `OrgBinding::build(block, store, Some(org_task_sink(backend)))` →
   `backend.install_org(binding)`; a broken chart aborts here.
7. `TeamMemberSpawner::with_parts(spec, deps, session_id, options, fork, router)`.
8. Team row upsert, then the leader row (`TeamRole::Leader`), then the leader's effective spec through
   `routing.effective_spec_for(leader, ..)` (memoizes the leader's allocation).
9. Workspace allocation (`WorkspaceRequest` with `isolation`, retained keyed by the leader's name); the
   leader's identity is derived from it (`member_card_id(effective, leader)` and
   `member_session_id(session_id, leader)`), the factory receives that context with `.with_card_id(..)`, and
   `wire_runtime(HarnessMemberRuntime::new(harness, member_session), ..)` attaches the policy, the store and
   `cwd = workspace root` before `TeamMember::attach_runtime`.
   The same workspace root anchors team memory: `memory_bind::resolve_memory(spec, deps, team, root)?` (injected
   wiring → non-null `metadata["memory"]` → none) runs here, `install_memory` puts it on the backend (the turn
   seam) and the spawner (the teammate toolkits), and the leader's context gains
   `.with_memory_toolkit(wiring.toolkit(leader, Some(team session)))`; a resolution failure fails the build and
   releases the leader workspace.
10. `kv_cache.register_member(leader_key, model)` — the root allocation.
11. `leader_member.observe(MemberStreamObserver::new(activity, weak leader, Leader, signals, chunk_sinks))`.
12. `insert_teammate_rows` — one `Unstarted` row per roster teammate (`predefined_members` ∪ `agents` keys
    except `leader`/leader name), each row's `model_ref` from `routing.model_ref_for(name, ..)`.
13. Leader card `<team>_<leader>`, messager subscription (`Team|Task|Message`), `InteractGate`,
    `HumanAgentInbox` (disabled until `enable_hitt`), `TeamRuntime`.

## Run flow (`invoke` / `stream`)

1. `parse_initial_input(inputs, roster)` → `leader_query` + `deliveries`.
2. Each delivery is handed to `deliver` without touching the leader runtime: `HumanAgent` → inbox push +
   persisted message row + `message` event; `Operator`/`GroupChat`/`GodView` → gate + leader mailbox;
   `ExternalTeam` → `DeliverResult::unsupported`.
3. `start()` (idempotent): messager start, the team-memory session install (`on_session_start`, `warn!`-and-
   continue), leader `MemberRuntime::start(None)` — which restores the stored checkpoint envelope for the
   leader's `(derived session, card id)` when one exists — attach bind (see policy), `signals.team_started`.
   A teammate's `spawn_member` start does the same restore before its pump and records the memory join after it.
4. The leader query is sent with `immediate = false` (a follow-up turn).
5. Loop on `tokio::select!`: member chunks, bus events, caller cancellation, and a settle tick.
   - teammate/leader chunks are forwarded tagged with `TeamOutputSchema{source_member, role, chunk}`;
   - mailbox/task/org events are translated into one leader `send` batched by `steer_batch_size`;
   - the loop ends when the leader runtime is `Idle` **and** `taskboard.is_settled()`, when the caller
     cancels, or when `stop()` fires.
6. The org mailbox is acknowledged only for the ids whose flush succeeded; a failed flush keeps them unread.
7. The `team.completed` frame (`schema::stream::team_completed_frame`) is emitted last and the gate is
   already closed; a settled run then publishes **one** `EventPayload::TeamCompleted(TeamCompletedEvent
   { team_name, reason: Some("settled") })` on `TeamTopic::Team` through `TeamBackend::post_event`, so a
   caller that holds no stream still sees the completion. A publish failure is `warn!`-logged and never
   fails the run, and a cancelled run (`RunExit::Cancelled`) publishes nothing — the bus event is
   exactly-once per settled run.
8. `invoke` returns `AgentOutput::answer(<concatenated leader answer>)`; `stream` yields the frames.
   A leader answer is accumulated from `llm_output` chunks (`payload.content`, else `payload.text`).

## Health

`TeamHealth` / `MemberHealth` (`health.rs`) are the read-only liveness projection: sync, no `.await`, no
new state, no probe task.
`TeamBackend::health()` projects the roster it already holds (`members()`, sorted by name; `leader` is a
copy of the leader's entry when that row exists) and always reports `started: false`, because the backend
does not track the run lifecycle; `TeamAgent::health()` is the caller-facing one and fills `started` from
`is_started()`.
`alive` means "a runtime is attached **and** that runtime's phase is not `Terminated` **and** the row is
not still `Unstarted`" — the last clause keeps a built-but-never-started team reporting nothing alive,
even though `build_with` already attached the leader's harness.
`status`/`execution` are copied from the row's cached projection (`TeamMember::status`/`execution`),
never re-read from the store.

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
- `openjiuwen_common::error::{Error, Result}` only: `validation` for user input/bad names/broken org charts,
  `config` for a missing build dependency or an empty model pool, `not_found` for unknown members/tasks/fork
  parents, `unsupported` for every deferred capability.
- Async-first tokio; `#[async_trait]` for object-safe async seams; `parking_lot` locks guard maps for one
  operation and are **never** held across `.await`; every spawned task is owned and cancellable
  (`tokio_util::sync::CancellationToken`).
- `tracing` only; no `println!`/`eprintln!` in this module.
- No `unsafe`, no `unwrap`/`expect` outside tests.
- Process-global state: exactly one new singleton here — the team runtime registry
  (`global_registry`), the crate's second documented singleton after the expert catalog.
- Files ≤1000 lines; split by responsibility (`leader.rs` holds the loop while `team_agent/` is one
  directory per concern — `mod.rs` is the handle, `lifecycle.rs` the run entry points, `members.rs` the
  member lifecycle, `attach.rs`/`signals.rs`/`org_ops.rs` their own surface — and `routing.rs`, `spec_meta.rs`,
  `observability_bind.rs`, `org_bind.rs`, `attach_host.rs` and `observer.rs` each own one concern).
- Inside `team_agent/`, method privacy is module-of-declaration based: a method one child module calls on
  another's behalf is `pub(super)` (`start_attach` in `attach.rs`, `release_all_workspaces` in `members.rs`),
  every other method stays private, and `ensure_leader_role` in `mod.rs` is reachable as `super::`.
- `--no-default-features` must build (no feature gates in this module).

## Routing Table

- `./mod.rs` → module map, flat re-exports
- `./deps.rs` → `TeamBuildDeps` and its defaults/builders
- `./allocator.rs` → `ModelAllocator`, `TeamModelAllocation`
- `./routing.rs` → `TeamModelRouter`, `ModelSelection` (pool + allocator + per-member memo → effective member spec)
- `./spec_meta.rs` → metadata keys and parsers (`spawn_options`, `fork_directive`, `isolation_mode`, `organization_block`, `kv_cache_config`, `attach_config`)
- `./observer.rs` → `MemberStreamObserver` — activity registry, status/round signals, chunk fan-out
- `./observability_bind.rs` → `TeamSignals` (observability hooks + optional monitor emitter)
- `./org_bind.rs` → `OrgBinding` (chart, delegation router, org task pool, org mailbox, `org_task` events)
- `./attach_host.rs` → `AttachHost` / `BackendAttachHandler` (Unix-socket attach server, `AttachHost::start(&AttachConfig, handler)`)
- `./wiring_tests.rs` → wiring test root (`Wiring` harness, `with_metadata`, `wait_until`, `take_monitor`, `monitor_kinds`)
- `./wiring_tests/` → the eight wiring suites (spawn_modes, routing, kv_affinity, signals, organization, workspace, attach, checkpoints)
- `./checkpoints.rs` → per-member checkpoint identity and store resolution (`member_card_id`, `team_checkpoint_store`, `member_checkpoint_store`, `wire_runtime`, `CHECKPOINT_DIR`)
- `./dispatch.rs` → `RunActionKind`, `RunAction`, `LiveTeamState`, `decide_run_action`
- `./initial_input.rs` → `InitialInput`, `parse_initial_input`
- `./backend.rs` → `TeamBackend` (members, mail, events, tasks, snapshot, signals, org handle)
- `./health.rs` → `TeamHealth` / `MemberHealth`, the `alive` rule and the roster projection
- `./build.rs` → the `SpecBuild` implementation and every build-only helper (store resolution, leader record, harness, observer, roster rows, bus wiring, org install)
- `./spawner.rs` → `TeamMemberSpawner` (`MemberSpawner` over the harness factory, spawn options, fork, routing, child spawners)
- `./team_agent/` → the `TeamAgent` handle directory: the struct and its accessors in `mod.rs`, one file per concern below
- `./team_agent/mod.rs` → `TeamAgent`, `TeamAgentInner`, `SpecBuild`, `ensure_leader_role`, the cheap accessors and the `pub(crate)` seams (`steer_batch_size`, `cancellation_token`, `take_events`)
- `./team_agent/lifecycle.rs` → `start`, `run_start`, `stop`, `invoke`, `stream`, `interact`, `deliver` and `snapshot`
- `./team_agent/members.rs` → `spawn_member`, `shutdown_member`, `auto_start_all`, `create_task`, `send_message`, `members`, `subscribe_chunks`, KV registration and the workspace bookkeeping
- `./team_agent/attach.rs` → `start_attach` (the binding `run_start` calls) and `attach_socket`
- `./team_agent/org_ops.rs` → `org`, `create_org_task`, `delegate_org_task`
- `./team_agent/signals.rs` → `signals`, `monitor_stream`, `kv_cache`, `kv_env`
- `./team_agent/test_support.rs` → the `#[cfg(test)]` offline kit (scripted factory/harness, recording workspace provider, `blueprint`/`deps`/`built`)
- `./team_agent/tests.rs` → the handle's unit tests (build, spawn/shutdown, workspace lifecycle, delivery, gate)
- `./leader.rs` → the coordination loop and its settle/batching/flush acknowledgement rules
- `./pool.rs` → `TeamRuntimePool`
- `./manager.rs` → `TeamRuntimeManager`, `global_registry`/`set_global_registry`/`clear_global_registry`
- `./adapter.rs` → `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` implementations
- `./tests/` → crossed-seam integration suite (`support.rs`, `end_to_end.rs`, `members.rs`, `mail.rs`)

## Test Strategy

Unit tests beside each file, the `./wiring_tests/` suites for the metadata-driven wiring, and the
`./tests/` crossed-seam suite; all offline and deterministic (scripted `HarnessProtocol` fakes from
`team_agent::test_support`, tempdirs, fake `sh` scripts, real Unix sockets under a tempdir, bounded waits
of 2 s polling every 5 ms, no `#[ignore]`).

- `deps.rs`: defaults are wired (native factory name, in-memory store, temp-rooted workspace), builders
  override one field each, `resolved_language` honours `Language::resolve`, and `store_injected` is false
  by default / true after `with_store` (it survives the other overrides).
- `build.rs`: store resolution — a memory spec reuses the dependency store, an injected store wins over a
  `postgresql` spec (as `Arc::ptr_eq`, proving `open_store` was skipped), a `sqlite` spec
  (`#[cfg(feature = "sqlite")]`) opens its own backend and creates the nested parent directory, and
  `postgresql`/`mysql`/`oracle` surface `ErrorKind::Unsupported` naming the backend or kind.
- `health.rs`: `alive` needs an attached runtime whose phase is not `Terminated` on a row that is not
  `Unstarted`; `from_roster` sorts by name and copies the leader; missing/unknown fields and the
  snake_case wire shape (`team_name`, `started`, `leader`, `members`; `name`, `role`, `status`,
  `execution`, `alive`).
- `allocator.rs`: round-robin cycles in pool order and persists its cursor; `by_name` picks the named
  entry and falls back to the first; empty pool + `Some(name)` echoes the name with `index = 0`; empty pool
  + `None` is `Error::config`; `state_dict` round-trip.
- `routing.rs`: pool parsing from the spec, empty-pool no-op, one allocation per member (memo), rotation
  order, `by_name` honouring a requested name, `effective_spec` preserving params, `apply_selection` model ref shape.
- `spec_meta.rs`: every documented shape plus the malformed fallbacks (in-process, no isolation, no fork,
  kv enabled, no attach) and the env fallback for `attach`.
- `observability_bind.rs`: signal → hooks + monitor fan-out, span open/close, newest-reader emitter policy.
- `org_bind.rs`: chart validation, delegation accept/refuse, org task pool, mailbox drain/ack, `org_task` payloads.
- `attach_host.rs`: op → backend mapping for every op, error mapping (`not_found`/`validation`/`unsupported`), shutdown.
- `dispatch.rs`: every row of the truth table.
- `memory_bind.rs`: the resolution order (unconfigured, configured, `{"enabled": false}`, null, injected), the
  document path at the workspace root, and the `is_live` rule for handoff notes.
- `initial_input.rs`: `$human @ai body` → one `HumanAgent` delivery and no leader query; `@dev-1 body` →
  delivery with the `user` sender; plain text, `#text`, `@all text` → leader query verbatim; nested
  `{"input":{"query":…}}` and plain-string inputs.
- `backend.rs`: mail routing (direct one row + one `message` event, broadcast one row, multicast N rows),
  member bookkeeping, task listing, snapshot ordering, `post_event` sender stamping, task signals.
- `team_agent/tests.rs`: build (leader card `<team>_<leader>`, missing `agents["leader"]` → validation), unknown
  member → `not_found`, spawn → `ready`, factory error → `unstarted`, shutdown per lifetime,
  `auto_start_all`, `create_task` leader-only, workspace isolation per member, attach bind policy, and the
  workspace lifecycle (`stop`/`shutdown_member` release exactly once, a failed release is logged).
- `leader.rs` / `./tests/`: invoke with an immediate fake answer and an empty board → exactly one
  `team.completed` marker last, exactly one `team_completed` bus event carrying the team name and
  `reason: "settled"`, and the leader answer in `AgentOutput.output`; an open task keeps the stream
  open until the board settles; cancellation ends the run and publishes no completion event.
- `manager.rs`: activate → live, session-mismatch rejection for a temporary team, `interact` unknown
  (`not_active`) and after `stop_team` (`gate_closed`/`not_active`), `finalize_member`, `shutdown`,
  `stop_team` tearing the attach socket down.
- `adapter.rs`: `team_id() == team_name`, `agent_count()`, `invoke` delegation, `kind() == "agent_teams"`.

`./wiring_tests/` (80 tests, one module per wiring point; the shared `Wiring` harness builds a team over a
`ScriptedFactory` + `RecordingWorkspaceProvider` and returns the agent, the factory and the workspace):

- `spawn_modes.rs` (9): the default blueprint stays in-process; `metadata["spawn"]` selects
  Process/ExternalCli and spawns+reaps a real child through a fake `sh` script; the metadata wins over the
  typed `spawn_mode`; the `{"options", "fork"}` wrapper parses both; malformed metadata falls back to
  in-process; a bare process/SSH request without its launch config fails with `Config`.
- `routing.rs` (11): one allocation per member reused by the harness spec and the roster row (leader `m0`,
  `dev-1` `m1`, `dev-2` `m0` with two entries); rotation on every `select`; `by_name` honouring a name;
  empty pool untouched; an injected router only reaches a directly built spawner; re-spawn keeps the endpoint.
- `kv_affinity.rs` (6): a fresh build registers the leader; a spawn registers the teammate; a fork registers
  via `register_child`; `shutdown_member` releases one member and `stop` releases the team; `enabled: false`
  registers nothing; an injected registry is reused.
- `signals.rs` (6): team start/finish move the root span; spawn/status/stop reach the hooks and the monitor;
  rounds reach both sinks once; task transitions are recorded; delegation carries its task; a single monitor
  stream sees the lifecycle in emission order.
- `organization.rs` (8): a configured chart is installed and an unconfigured team stays flat; an invalid chart
  fails the build; allowed delegation records the hook and publishes `org_task` on `TeamTopic::Task`; a refused
  delegation leaves the task untouched; the mailbox is drained and acknowledged after a successful flush; a
  failed leader flush leaves the mail unacknowledged (a `FailingHarness` whose `send` errors).
- `workspace.rs` (6): `IsolationMode` reads a bare string and the object shape; the default is `None` for the
  leader and teammates; a worktree team still builds and spawns (both members request `Worktree`).
- `attach.rs` (10, `#![cfg(unix)]`): no configuration means no socket; metadata and
  `OPENJIUWEN_TEAM_SOCKET` both parse (metadata wins); a real bind serves every `AttachOp` through
  `BackendAttachHandler`; `stop_team` acknowledges and cancels the run token; an unbindable socket is not
  fatal while `required: true` fails the start; `stop` and the manager's `stop_team` remove the file and
  later calls fail.
- `checkpoints.rs` (11): the default build pins one `"memory"` team store while a `file` config pins none;
  the leader and a spawned teammate restore under `(derived member session, card id)` with the blueprint card
  id and fall back to the member name for a card-less blueprint; a finished turn and a `stop` each reach the
  store once; a restarted member `start`s with the stored provider envelope, `ResumeIfAvailable` and
  `host_session_id == "<team session>:<member>"`; `{"enabled": false}` attaches no sink, restores nothing and
  captures nothing; an explicit `dir` receives one `checkpoint.json` per member.

Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams` (1411 lib
tests in the default build, 1430 with `--features sqlite`, 1439 with `--all-features`, plus 219 integration
tests over 15 binaries),
`cargo test -p openjiuwen-agent-teams --no-default-features`,
`cargo test -p openjiuwen-agent-teams --features sqlite`, `cargo clippy -p openjiuwen-agent-teams
--all-targets -- -D warnings` and `cargo doc -p openjiuwen-agent-teams --no-deps` all green offline.

## Notes for Agents / Deviations from Python

- Python's `CoordinationKernel` (a large event-driven supervisor) is reimplemented as one documented loop:
  a `select!` over chunks, bus events, cancellation and a settle tick. Batching, idle debounce and
  escalation heuristics are simplified; the observable contract (one completed marker, deliveries never
  reaching the leader, settle = empty board) is preserved.
- Team members are spawned through the injected `MemberHarnessFactory` for the in-process path and through
  the `src/member/` child-process spawners for `process`/`external_cli`/`ssh`; nothing in this module builds a
  DeepAgent directly.
- `TeamSpec` (not `TeamAgentSpec`) is what the running team carries, mirroring Python's runtime mirror.
- `TeamAgent` has no direct `store()` accessor: a test or host reaches the store through `TeamAgent::backend().store()` (`backend.rs:177`), reads mail through `TeamBackend::unread`/`mark_read` and the board through `TeamBackend::tasks()`/`list_tasks`.
- `TeamBuildDeps.store` is an `Arc<dyn TeamStore>` field (`deps.rs:49`), set by `with_store`; the default is `InMemoryTeamStore` (`deps.rs:150`) and `--features sqlite` users inject `SqliteTeamStore` instead of a `StorageSpec`-like value.
- A teammate runs the same loop but creates no tasks and never dispatches: `create_task` on a teammate is
  `Error::validation("only the leader can create tasks")`.
- The team roster is materialized as rows at build time (`unstarted` teammates); Python creates rows lazily
  on bootstrap. This makes `snapshot()` answer before the first spawn.
- `metadata` is the only place a host tunes the runtime; `TeamBuildDeps` only injects hooks, a KV registry, a
  checkpoint store and (for direct spawner construction) a router. Adding a new knob means a parser in `spec_meta.rs` plus a field
  on `TeamAgentInner`, never a new constructor argument.
- Formatting/typing invariant: a member's `DeepAgentSpec.model` and its `MemberRecord::model_ref` come from the
  same `ModelSelection`, so a pool entry selects a real endpoint instead of only labelling the row.
- `kv_cache_config` defaults to `enabled: true` when the metadata key is absent, so a freshly built team
  already answers `TeamAgent::kv_env(member)` for its leader and every spawned member.
- Distributing a team across processes, vendor/remote worktree backends and the cross-team org
  federation stay deferred behind the member/workspace/attach seams (`Error::Unsupported` with the Python
  capability name, or a logged warning for the attach socket).
- `Team::invoke` in `openjiuwen-multiagent` carries an `AgentTeamSession`, while `Agent::invoke` takes
  `Option<Arc<dyn Session>>`; the adapter ignores the multiagent session (documented; `openjiuwen-session`'s
  `AgentTeamSession` is not a `dyn Session`).
- `Team::add_agent` accepts a card + `AgentProvider`; the provider is ignored — a member's brain always
  comes from the harness factory, and only an `unstarted` roster row is created. Because that seam is
  synchronous while `TeamStore::insert_member` is async, the row write is spawned on the current tokio
  runtime (a missing runtime is `Error::config`).
- `SpecBuild` is `#[async_trait]` (async `build`/`build_with`): the pinned build order awaits the store,
  the workspace provider, the harness factory and the team-row upsert, so a synchronous trait cannot hold
  it.
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
- `member::test_child` (`FakeChild`, `#[cfg(test)]`) is private to `src/member`, so a runtime test that
  exercises a real process member writes its own `sh` script or asserts the missing-program `Error::Config`;
  attach hosting is `#[cfg(unix)]` (`AttachServer::bind`/`serve` return `Unsupported` elsewhere) and the
  worktree backend needs a real `git` binary plus a repository.
- `TeamHealth`/`MemberHealth` are exported from `runtime` (`runtime::mod` re-export) but **not** from the crate root: `src/lib.rs` enumerates the runtime items by name and belongs to another node, so a crate-root import needs that file updated elsewhere. `openjiuwen_agent_teams::runtime::{TeamHealth, MemberHealth}` works today.
- `alive` is deliberately narrower than "has a harness": `build_with` already attaches the leader's harness, so an `Unstarted` row would otherwise read as alive. The predicate is `status != Unstarted && runtime attached && runtime state != Terminated`.
- `deps.store_injected` is the only store-resolution switch: `with_store` sets it, `default()`/`new()` clear it. Adding a second way to supply a store would silently diverge from `resolve_store`.
- `crates/openjiuwen-e2e/tests/team_backend.rs` still asserts that completion is stream-only (`all(|event| event.event_type != "team_completed")`); a settled run now also publishes `team_completed` on `TeamTopic::Team`, so that assertion belongs to the e2e owner and must be inverted there — this crate cannot edit it.
- `src/schema/CONTEXT.md` (schema node) still says the leader loop emits no completion bus event; that node owns the correction.
- A settled run publishes the bus event after the stream marker is sent, so a caller that reads the bus alone sees completion strictly after the frame; a publish failure is logged and never fails the run. `blueprint`,
  `deps`, `built`, `ScriptedFactory` (`failing_for`/`hanging_for`/`calls`/`harness`), `ScriptedHarness`
  (`calls`/`sends`, answers `done: <member>` by default), `RecordingWorkspaceProvider` (`requests`, `releases`/`released_names`, `fail_releases`).
- `team_agent/` is split by concern (`mod.rs` handle, `lifecycle.rs`, `members.rs` plus the small
  `attach.rs`/`signals.rs`/`org_ops.rs` surfaces), so no file carries both the handle surface and the member
  lifecycle; `backend.rs` (1040 lines) remains the one long file of this node because it owns a single
  contract. Extract before adding another responsibility.

## Not wired (documented gaps)

- Model routing is decided per member: there is no per-request routing, `IntelliRouter` or builtin-model
  selection.
  The router's cursor and per-member memo are not checkpointed, so a rebuilt team re-allocates from the
  pool while an in-process resume keeps its memoized endpoints.
- KV-cache affinity records identity only: no cache-hit telemetry, no eviction policy beyond `max_entries`,
  no provider-side reuse, and the SSH transport carries no env bag.
- Observability hooks are in-process: no OTel exporter (that lives in `openjiuwen-extensions`) and no
  per-tool spans. The monitor stream has no TUI/consumer here.
- The organisation layer is in-process only: no cross-process org messaging, summary teams, runtime
  swarmflow lanes or org-level observability; the chart shapes and their semantics belong to `src/org/`.
- Team memory is in-process and per-team: no cross-team/hub sync, no LLM-backed extractor here (an
  `extractor: "llm"` block degrades to the rules — drive `open_with_model` yourself for the LLM path), no
  distributed store, and the document is the only durable artefact (this crate does not enable the
  `openjiuwen-memory` `sqlite` feature). The hooks are advisory by design, so a broken memory store never
  fails a start, spawn, shutdown or stop.
- A roster removal that bypasses `shutdown_member` (`TeamBackend::remove_member`) records no handoff note: the
  hook is async and the removal seam is sync, and the documented restart path is
  `shutdown_member` → `remove_member` → `spawn_member`, so the leave hook already ran.
- Child-process members receive the leader's `metadata["checkpoint"]` block through
  `MemberSpawnRequest::with_checkpoint_config` (the two `OPENJIUWEN_MEMBER_CHECKPOINT*` env vars); team memory
  is *not* transported to a child, which mounts no toolkit of its own.
- Team memory is in-process and per-team: no cross-team/hub sync, no LLM-backed extractor on the build path
  (an `extractor: "llm"` block degrades to the rules — `TeamMemoryWiring::open_with_model` is the seam), no
  distributed store, and the `TEAM_MEMORY.md` document is the only durable artefact (this crate does not enable
  the `openjiuwen-memory` `sqlite` feature). The hooks are advisory by design, so a broken memory store never
  fails a start, spawn, shutdown or stop.
- A roster removal that bypasses `shutdown_member` (`TeamBackend::remove_member`) records no handoff note: the
  hook is async and the removal seam is sync, and the documented restart path is
  `shutdown_member` → `remove_member` → `spawn_member`, so the leave hook already ran.
- Child-process members receive the leader's `metadata["checkpoint"]` block through
  `MemberSpawnRequest::with_checkpoint_config` (the two `OPENJIUWEN_MEMBER_CHECKPOINT*` env vars); team memory
  is *not* transported to a child, which mounts no toolkit of its own.
- Checkpointing has no team-level resume policy, no retention/pruning or TTL, and no leader restart: only
  `TeamAgentSpec::metadata["checkpoint"]` + `TeamBuildDeps::with_checkpoint_store` decide, and a retired
  member row cannot be revived in place (`TeamBackend::remove_member` + `spawn_member` is the restart path).
- Known cross-node issue: `crates/openjiuwen-agent-teams/tests/member_and_mail.rs`
  (`the_roster_accessors_agree_and_are_sorted`, lines ~689 and ~702) still asserts that a runtime session
  equals the team session; the leader and every teammate now run under `"<team session>:<member>"`, so the
  assertion belongs to the crate-root node and needs `format!("{}:{}", agent.session_id(), agent.member_name())`.
- Attach hosting serves one request/response op per connection with no subscriptions, authentication,
  streaming or non-Unix transport; a CLI or MCP process reaches a hosted team through that socket, and only
  a shared-store or brokered attach remains deferred.
- Spawn backends beyond in-process are the `src/member/` transports; this node only selects among them and
  forwards the KV env (vendor CLI protocols, remote SSH daemons and OS-process pooling stay there).
- Worktree isolation is delegated to `src/workspace/`; the remote worktree backend and team-workspace
  artifact versioning are deferred there.
