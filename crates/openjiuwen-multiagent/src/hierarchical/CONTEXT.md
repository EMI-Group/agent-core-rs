# openjiuwen-multiagent :: hierarchical

## Intent

Hierarchical teams place one supervisor (or root) agent above a flat member list.
Python parity: `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/` and `openjiuwen/core/multi_agent/teams/hierarchical_tools/` (v0.1.20).
Both topologies are `TeamStrategy` objects attached to the crate's single `BaseTeam`: the team owns the card, the limits, the member registry, and the runtime, while the strategy decides how one invocation reaches the supervisor.
A message-bus team registers its supervisor as an ordinary member and dispatches every invocation as one P2P request over the bus.
An agents-as-tools team mounts every sub-agent on its root as an `AgentTool` and streams the root in-process, bypassing the bus.
Everything is in-process and tokio-native; members are resolved from the team runtime only.

## API Surface

### Supervisor over the message bus (`msgbus.rs`)

```rust
pub struct HierarchicalMsgbusConfig {
    #[serde(flatten)] pub team: TeamConfig,     // TeamConfig::new(): 10 / 100 / 30.0
    pub supervisor_agent: AgentCard,            // AgentCard::default()
    pub timeout: Option<f64>,                   // Some(1800.0)
}
impl HierarchicalMsgbusConfig {
    pub fn new(supervisor_agent: AgentCard) -> Self;   // Python defaults plus the supervisor
}
pub struct HierarchicalMsgbusStrategy { /* supervisor_id: String, timeout: Option<Duration> */ }
impl HierarchicalMsgbusStrategy {
    pub fn new(config: HierarchicalMsgbusConfig) -> Arc<Self>;
}
pub fn hierarchical_msgbus_team(card: TeamCard, config: HierarchicalMsgbusConfig) -> BaseTeam;
```

`kind()` is `"hierarchical_msgbus"`; `hierarchical_msgbus_team` is `BaseTeam::with_strategy(card, config.team.clone(), HierarchicalMsgbusStrategy::new(config))`.
The configured `timeout` (seconds) becomes one `Duration` at construction; every other value means "no per-invocation deadline" and defers to the runtime's `p2p_timeout`.

| `timeout` | deadline passed to `TeamRuntime::send` |
| --- | --- |
| `None` | `None` |
| `Some(0.0)` | `None` |
| `Some(negative)` | `None` |
| `Some(NaN)` / `Some(±inf)` | `None` |
| `Some(positive finite)` | `Some(Duration::from_secs_f64(seconds))` |

`on_member_added` adopts the deadline onto the runtime (`set_p2p_timeout`) when the card id matches the supervisor; a `None` deadline leaves the runtime timeout untouched.
`check_ready` requires a non-empty supervisor id and a registered supervisor, returning `ErrorKind::Config` with these exact texts:

- `No supervisor configured in HierarchicalTeamConfig.`
- `Supervisor '<id>' is not registered in runtime. Call add_agent(supervisor_card, supervisor_provider) before invoke()/stream().`

`invoke` checks readiness, acquires a `SessionLease`, sends the input verbatim to the supervisor with sender `ctx.card.id`, finishes the lease, and returns the supervisor's value.
`stream` sends the same request inside `crate::session::team_stream` and writes exactly one frame, `{"type":"message","index":0,"payload":{"output": <result>}}`, when the supervisor result is not JSON null.
A null reply never reaches that guard as a success: `TeamRuntime::send` rejects it as `TeamError::EmptyResponse`, so the guard is defensive and a null reply surfaces as a stream error.
`stream` checks readiness before the first frame, so an unregistered or blank supervisor yields exactly one `Err` item and then the stream ends (mirroring Python raising on the first `__anext__`); on the successful path it writes exactly one `{"type":"message","index":0,"payload":{"output": <result>}}` frame.

### Supervisor factory (`supervisor.rs`)

```rust
pub struct SupervisorConfig {              // process-local: no serde, no JsonSchema
    pub card: AgentCard,
    pub model: Arc<dyn Model>,
    pub system_prompt: String,
    pub max_iterations: usize,             // 5
    pub max_parallel_sub_agents: usize,    // 10
}
impl SupervisorConfig {
    pub fn new(card: AgentCard, model: Arc<dyn Model>, system_prompt: impl Into<String>) -> Self;
}
pub struct SupervisorAgent;
impl SupervisorAgent {
    pub fn create(agents: Vec<AgentCard>,
                  config: SupervisorConfig) -> Result<(AgentCard, AgentProvider)>;
}
pub fn create_supervisor(agents: Vec<AgentCard>,
                         config: SupervisorConfig) -> Result<(AgentCard, AgentProvider)>;
```

`create` validates the sub-agent list before building anything and returns `ErrorKind::Validation` with these exact texts:

- `[SupervisorAgent.create] agents list must not be empty`
- `[SupervisorAgent.create] agent card '<name>' has a blank agent id`
- `[SupervisorAgent.create] agent card '<id>' has a blank agent name`
- `[SupervisorAgent.create] duplicate agent id '<id>'`
- `[SupervisorAgent.create] duplicate agent name '<name>'`

The returned provider builds a fresh `ReActAgent` per resolution, configures `max_iterations`, one system message carrying `system_prompt`, and the model, then returns a `TeamAgent` with the agent as its tool host and a bind hook that mounts the sub-agent tools.
`SupervisorConfig`'s `Debug` prints the card id and `model_name()` instead of the model object.

### Agents as tools (`tools.rs`)

```rust
pub struct HierarchicalToolsConfig {
    #[serde(flatten)] pub team: TeamConfig,    // TeamConfig::new(): 10 / 100 / 30.0
    pub root_agent: AgentCard,                 // AgentCard::default()
}
impl HierarchicalToolsConfig {
    pub fn new(root_agent: AgentCard) -> Self; // Python defaults plus the root
}
pub struct HierarchicalToolsStrategy { /* root_id, pending_children, handles, setup_lock, wired */ }
impl HierarchicalToolsStrategy {
    pub fn new(config: HierarchicalToolsConfig) -> Arc<Self>;
}
pub fn hierarchical_tools_team(card: TeamCard, config: HierarchicalToolsConfig) -> BaseTeam;
```

`HierarchicalToolsConfig` derives `Default`; its `new` keeps those defaults and stores the root card.
`kind()` is `"hierarchical_tools"`; `hierarchical_tools_team` is `BaseTeam::with_strategy(card, config.team.clone(), HierarchicalToolsStrategy::new(config))`.
`check_ready` requires a non-empty root id and a registered root, returning `ErrorKind::Config` with these exact texts:

- `No root agent configured in HierarchicalTeamConfig.`
- `Root agent '<id>' is not registered in runtime. Call add_agent(root_card, root_provider) before invoke()/stream().`

`on_member_added` only queues a `(parent, child)` pair when `MemberOptions::parent_agent_id` is `Some` and non-empty; `setup` mounts the queued tools and pins the instance.
`invoke` checks readiness, wires the hierarchy, acquires a `SessionLease`, sends the input verbatim to the root over the bus with no explicit deadline (so the runtime `p2p_timeout` applies), and finishes the lease.
`stream` checks readiness first (a failure becomes a single error item) and then streams the root in-process, shaping the input:

- an object input keeps its keys and gets `conversation_id` (team session id) and `sender` (team card id) overwritten;
- any other input becomes `{"query": <input>, "conversation_id": <session id>, "sender": <team card id>}`.

Each agent chunk is forwarded as a team frame `{"type": <AgentChunk.kind>, "index": <AgentChunk.index>, "payload": <AgentChunk.payload>}`.
An agent stream failure is swallowed into one `{"type":"message","index":0,"payload":{"output": <error message>, "result_type":"error"}}` frame and the run returns `Ok(())`.

## Constraints

- The crate rules apply unchanged: no `openjiuwen-runner` dependency, `openjiuwen_common::error::{Error, Result}` everywhere, serde-serializable public data, no `unsafe`, no stdout logging.
- Locks are never held across `.await`: `parking_lot::RwLock` guards the short pending/handle maps, `setup_lock` is a `tokio::sync::Mutex`, and `wired` is an `AtomicBool`.
- `SupervisorConfig` is the documented exception to serde: it holds `Arc<dyn Model>` and stays process-local, so it derives neither `Serialize`/`Deserialize` nor `JsonSchema`.
- Every public item and field is documented; the crate runs clippy with `-D warnings` including `missing_docs`.
- `tools.rs` stays under ~1000 lines by keeping every unit test in its own in-module `tests` block; the strategy, shaping helper, and wiring live in the same file.
- Tests are offline and deterministic: scripted models, in-process fake agents, and only explicit short deadlines.

## Design Decisions

- **Sub-agents are ordinary `AgentTool`s mounted by a bind hook**: `SupervisorAgent::create` replaces Python's `P2PAbilityManager` subclassing/injection attribute; the tools dispatch through `TeamRuntime::invoke_agent`, so there is no bus hop and no topic re-entrancy, and one shared `Semaphore` (clamped to `max(1, max_parallel_sub_agents)`) bounds concurrent dispatches.
- **The bind hook is the only point where the runtime exists**: it mounts at most once per bound provider call (a per-provider-call `AtomicBool`), and the sub-agent card set is the snapshot taken at `create` time (Python parity: later members are invisible to the supervisor).
- **`SupervisorAgent::create` takes an `Arc<dyn Model>`** instead of client/request configs: model construction stays with the caller (`init_model`), the crate stays provider-agnostic, and tests pass a `ScriptedModel`.
- **The provider builds a fresh `ReActAgent` per resolution**, so each member instance owns its own mounted tools, semaphore, and mount latch, and two concurrent team invocations never share a parallel-dispatch budget.
- **`hierarchical_tools` wires lazily on the first `invoke`/`stream`** (`setup`, double-checked lock) and pins the wired parent with `runtime.pin_agent`, so later bus sends reach the instance that owns the tools; Python re-invokes the provider and loses the wiring (documented divergence).
- **The strategy keeps a private `handles` cache of the instances it wired**, because `TeamRuntime` exposes no pinned-instance getter; the bus path uses the pin, the bus-free stream path uses the cache, and both refer to the same instance.
- **A later membership event rewires**: `setup` re-runs whenever pending children exist, and `resolve_handle` prefers the cached instance, so new children mount on the already-wired parent instead of a fresh one.
- **The agents-as-tools stream bypasses the bus entirely** (in-process `Agent::stream`), forwards raw `AgentChunk` frames, and swallows an agent stream error into one error frame (Python parity); a readiness or root-resolution failure is an error item, never a panic.
- **Wiring misses are logged, never raised**: a missing parent, a parent without a `ToolHost`, and a rejected mount behave like Python's failed attribute probe.

## Routing Table

- `./mod.rs` → module docs (the two topologies, the `TeamStrategy` seam), the three submodule declarations, and the re-exports.
- `./msgbus.rs` → `HierarchicalMsgbusConfig` (+ `new`/serde defaults), the timeout-to-deadline conversion, `HierarchicalMsgbusStrategy` (readiness before the first frame, deadline adoption, `invoke`, one-frame `stream`), `hierarchical_msgbus_team`, and its 11 unit tests.
- `./supervisor.rs` → `SupervisorConfig` (+ `new`, `Debug`), `SupervisorAgent::create`, `create_supervisor`, sub-agent validation, the provider/bind hook, and its 8 unit tests.
- `./tools.rs` → `HierarchicalToolsConfig` (+ `new`), `shape_input`, `HierarchicalToolsStrategy` (queueing, lazy wiring/pinning, readiness, `invoke`, bus-free `stream`), `hierarchical_tools_team`, and its 20 unit tests.

## Test Strategy

- `msgbus.rs` in-module (11): config serde defaults, flatten, and unknown-field tolerance (3), the deadline conversion table (1), `kind()` (1), the three readiness outcomes with exact texts (3), deadline adoption/keep on `on_member_added` (2), and the team builder's kind/config (1).
- `supervisor.rs` in-module (8): `SupervisorConfig::new` defaults (1), the validation messages for empty/blank/duplicate cards (3), the returned card's serde round-trip (1), a fresh tool-hosting instance per provider call (1), a bind that registers nothing and is idempotent (1), and the `max_parallel_sub_agents = 0` clamp (1).
- `tools.rs` in-module (20): config defaults/`new`/serde (3), `kind()` (1), queueing and removal bookkeeping (2), readiness (3), `shape_input` for objects/scalars/null (2), setup wiring, pinning, reuse, and the two skip paths (4), `invoke` to the root (1), frame forwarding, error-frame swallow, and readiness-as-error-item (3), and the team builder (1).
- `cargo test -p openjiuwen-multiagent --offline --lib hierarchical` runs these 39 unit tests; they need no runtime feature beyond tokio's test runtime.
- `../../tests/hierarchical.rs` holds 23 integration tests in one shared config test plus three groups: msgbus (9), agents as tools (8), supervisor factory (5).
- The msgbus group pins both readiness errors verbatim (as `invoke` errors and as the stream's only error item), the verbatim input and child `parent_id()` of the single P2P hop, deadline adoption and the no-deadline pass-through, the exactly-one `{"type":"message","index":0,"payload":{"output":..}}` stream frame, and the per-invoke `ErrorKind::Timeout`.
- The agents-as-tools group pins `RecordingToolHost` mounts in member order, one provider resolution across two invocations (pinning), mounted-tool dispatch through the runtime, the shaped root input for objects and scalars, both root readiness errors, and the single error frame after an agent failure.
- The supervisor group pins the empty-agent validation, the returned card round-trip, one end-to-end tool-call run over the bus, the model-visible tool cards (`function.name`/`description`/`parameters` plus the empty-object fallback), and that parallel budgets `0` and `1` both dispatch.

## Notes for Agents

- A default config is not "unconfigured": `AgentCard::default()` mints a 32-character uuid hex id, so `No supervisor/root agent configured` is only reachable with an explicitly blank id (`AgentCard::new("").with_id("")`).
- `on_member_added` adopts the configured deadline only for the supervisor's card id; every other registration is a no-op and leaves `runtime.p2p_timeout()` alone.
- Children are queued only when `MemberOptions::parent_agent_id` is `Some` and non-empty; an empty parent id is dropped, and queuing has no registration side effect.
- A parent without a `ToolHost` is skipped with a warning, as is an unregistered parent and a mount the host rejects; the queued list is cleared either way, so a skipped pairing is never retried.
- `on_member_removed` evicts the removed agent's own queued children, removes it from every other parent's child list, and drops its wired handle.
- `is_wired` is `wired && pending_children.is_empty()`, so a membership event after the first run makes `setup` run again and mount the new children on the cached instance.
- The supervisor provider mounts its tools through the bind hook, not in `create`; resolving a member without binding it (for example with `provider()` directly) yields a `ReActAgent` with no sub-agent tools.
- The message-bus stream guards its frame with `!result.is_null()`, but `TeamRuntime::send` already rejects a null reply as `TeamError::EmptyResponse`, so a null supervisor answer fails both `invoke` and `stream` instead of yielding an empty frame.
- The agents-as-tools `invoke` path still travels over the bus to the pinned root; only `stream` bypasses it, which is why a mounted-tool dispatch and a root stream reach the same instance through different paths.
