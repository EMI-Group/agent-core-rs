# openjiuwen-multiagent

## Intent

Teams of agents: team cards/configs, an in-process message bus and runtime, and the three strategies the Python SDK ships (`handoff`, supervisor-over-message-bus, agents-as-tools).
Ports Python `openjiuwen.core.multi_agent` (v0.1.20, 31 files / ~4.1k lines) plus the team-session helpers from `core.session.agent_team`; distributed transports and the product-level `agent_teams` package (leader/teammate, worktrees) are out of scope (see `docs/PARITY.md`).
Everything is in-process and tokio-native: the Python `MessageQueueInMemory` transport becomes a per-topic FIFO worker with oneshot request/response correlation.
The crate never depends on `openjiuwen-runner` (runner depends on it): members are resolved from the team's own runtime registry, and the `Team` trait is the seam the runner drives.

## API Surface

### Crate root (`lib.rs`)

Re-exports: `TeamCard`, `EventDrivenTeamCard`, `TeamConfig`, `TeamError`; `Team`, `BaseTeam`, `TeamStrategy`, `StrategyContext`, `MemberOptions`; `TeamAgent`, `AgentProvider`, `ToolHost`, `BindHook`; `MessageEnvelope`, `MessageBus`, `MessageBusConfig`, `MessageHandler`, `SubscriptionManager`; `TeamRuntime`, `RuntimeConfig`, `SessionLease`, `make_team_session`; `AgentTool`.
Strategy trees: `handoff::*` (`HandoffTeamConfig`, `HandoffConfig`, `HandoffRoute`, `HandoffSignal`, `HandoffRequest`, `HandoffHistoryEntry`, `HandoffTool`, `HandoffOrchestrator`, `ContainerAgent`, `TeamInterruptSignal`, `extract_interrupt_signal`, `flush_team_session`, `handoff_team`, `HandoffStrategy`, the `HANDOFF_*` constants) and `hierarchical::*` (`HierarchicalMsgbusConfig`, `hierarchical_msgbus_team`, `SupervisorAgent`, `SupervisorConfig`, `create_supervisor`, `HierarchicalToolsConfig`, `hierarchical_tools_team`).

### Cards and configs (`card.rs`, `config.rs`)

```rust
pub struct TeamCard {                       // serde snake_case, unknown fields tolerated, every field defaulted
    pub id: String,                          // uuid hex
    pub name: String,                        // ""
    pub description: String,                 // ""
    pub agent_cards: Vec<AgentCard>,         // [] — the member list
    pub topic: String,                       // ""
    pub version: String,                     // "1.0.0"
    pub tags: Vec<String>,                   // []
}
impl TeamCard { pub fn new(name: impl Into<String>) -> Self; /* with_id/name/description/agent_cards/topic/version/tags */ }
impl openjiuwen_common::card::Card for TeamCard { .. }
pub struct EventDrivenTeamCard { #[serde(flatten)] pub card: TeamCard,
                                 pub subscriptions: BTreeMap<String, Vec<String>> }
pub struct TeamConfig {                      // Python TeamConfig, extra fields tolerated
    pub max_agents: usize,                   // 10
    pub max_concurrent_messages: usize,      // 100 — message-queue capacity
    pub message_timeout: f64,                // 30.0 — seconds, seeds the bus process timeout and the handoff chain deadline
}
impl TeamConfig { pub fn configure_max_agents(self, n: usize) -> Self;
                  pub fn configure_concurrency(self, n: usize) -> Self;
                  pub fn configure_timeout(self, seconds: f64) -> Self; }
```

### Members (`member.rs`)

```rust
pub type AgentProvider = Arc<dyn Fn() -> TeamAgent + Send + Sync>;   // invoked per resolution (Python parity)
pub type BindHook = Arc<dyn Fn(&Arc<TeamRuntime>, &str) + Send + Sync>;
pub struct TeamAgent { /* agent, optional tool host, optional bind hook */ }
impl TeamAgent {
    pub fn new(agent: Arc<dyn Agent>) -> Self;
    pub fn with_tool_host(agent: Arc<dyn Agent>, host: Arc<dyn ToolHost>) -> Self;
    pub fn with_bind(self, hook: impl Fn(&Arc<TeamRuntime>, &str) + Send + Sync + 'static) -> Self;
    pub fn react(agent: Arc<openjiuwen_agent::ReActAgent>) -> Self;      // agent + host + no hook
    pub fn agent(&self) -> Arc<dyn Agent>;
    pub fn tool_host(&self) -> Option<Arc<dyn ToolHost>>;
    pub fn bind(&self, runtime: &Arc<TeamRuntime>, agent_id: &str);      // no-op without a hook
}
impl From<Arc<dyn Agent>> for TeamAgent;
pub trait ToolHost: Send + Sync { fn mount_tool(&self, tool: Arc<dyn Tool>) -> bool; }  // true when added/refreshed
impl ToolHost for openjiuwen_agent::ReActAgent;   // delegates to ReActAgent::add_tool
impl ToolHost for openjiuwen_agent::LlmAgent;     // delegates to react().add_tool
```

### Envelope and bus (`envelope.rs`, `subscription.rs`, `message_bus.rs`)

```rust
pub struct MessageEnvelope {                 // Python frozen dataclass; Rust exposes immutable-by-value fields
    pub message_id: String,                  // uuid v4 (hyphenated, Python parity)
    pub message: Value,
    pub sender: Option<String>,
    pub recipient: Option<String>,           // Some => P2P
    pub topic_id: Option<String>,            // Some => pub/sub
    pub session_id: Option<String>,
    pub metadata: BTreeMap<String, Value>,
}
impl MessageEnvelope { pub fn new(message: Value) -> Self;
                       pub fn p2p(message: Value, sender, recipient) -> Self;
                       pub fn pubsub(message: Value, sender, topic_id) -> Self;
                       pub fn is_p2p(&self) -> bool; pub fn is_pubsub(&self) -> bool;
                       pub fn with_session(mut self, id) -> Self; pub fn with_metadata(..) -> Self; }
pub struct SubscriptionManager { /* topic pattern -> BTreeSet<agent id>, bidirectional index */ }
impl SubscriptionManager {
    pub fn new() -> Self;
    pub fn subscribe(&self, agent_id: &str, topic: &str);
    pub fn unsubscribe(&self, agent_id: &str, topic: &str);
    pub fn subscribers(&self, topic_id: &str) -> Vec<String>;        // sorted (deterministic; Python returns a set)
    pub fn topics_of(&self, agent_id: &str) -> Vec<String>;
    pub fn agents(&self) -> Vec<String>;
    pub fn subscription_count(&self) -> usize;
    pub fn remove_agent(&self, agent_id: &str);
}
pub fn topic_matches(pattern: &str, topic: &str) -> bool;             // exact, else fnmatch `*`/`?` only when the pattern contains one

pub struct MessageBusConfig { pub max_queue_size: usize /*1000*/, pub process_timeout: Option<f64> /*1800.0*/,
                              pub team_id: Option<String> /*"default"*/ }
#[async_trait] pub trait MessageHandler: Send + Sync {
    async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value>;
    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()>;
}
pub struct MessageBus { .. }
impl MessageBus {
    pub fn new(config: MessageBusConfig, handler: Arc<dyn MessageHandler>) -> Self;
    pub async fn start(&self) -> Result<()>;                     // idempotent
    pub async fn stop(&self) -> Result<()>;                      // idempotent, joins workers
    pub fn is_started(&self) -> bool;
    pub async fn subscribe(&self, topic: &str) -> Result<()>;    // transport topic; duplicate is an error like Python
    pub async fn unsubscribe(&self, topic: &str) -> Result<()>;
    pub fn add_subscription(&self, agent_id: &str, topic_id: &str);
    pub fn remove_subscription(&self, agent_id: &str, topic_id: &str);
    pub fn subscribers(&self, topic_id: &str) -> Vec<String>;
    pub async fn send(&self, envelope: MessageEnvelope, timeout: Option<Duration>) -> Result<Value>;
    pub async fn publish(&self, envelope: MessageEnvelope) -> Result<()>;   // returns after fan-out completes
    pub async fn cleanup_session(&self, session_id: &str);       // unsubscribes `{team}_[sid]__p2p__` and `__pubsub__`
}
```
Transport topics: P2P `"{team_id}__p2p__"` without a session, `"{team_id}_{session_id}__p2p__"` with one; pub/sub the same with `__pubsub__`. Each topic owns a bounded FIFO (`max_queue_size`) worker task, so deliveries on one topic are processed in enqueue order; different topics run concurrently. P2P requests carry a oneshot reply; a `send` deadline is the caller timeout, else `process_timeout`, else unbounded — on timeout the handler keeps running (documented Python parity). Pub/sub fan-out is concurrent over the sorted subscriber set, waits for all subscribers before returning, and isolates failures (a failed subscriber is logged and never fails the publisher). `stop` unsubscribes every topic, signals workers, and joins them; a duplicate `subscribe` is an error, matching Python's `Topic '...' is already subscribed.`

### Runtime (`runtime.rs`)

```rust
pub struct RuntimeConfig { pub team_id: String /*"default"*/, pub message_bus: Option<MessageBusConfig>,
                           pub p2p_timeout: f64 /*1800.0*/ }
pub struct TeamRuntime { /* Arc<RuntimeState> + Arc<MessageBus> */ }
impl TeamRuntime {
    pub fn new(config: RuntimeConfig) -> Arc<Self>;
    pub fn team_id(&self) -> &str;
    pub fn config(&self) -> &RuntimeConfig;
    pub fn has_agent(&self, agent_id: &str) -> bool;
    pub fn agent_card(&self, agent_id: &str) -> Option<AgentCard>;
    pub fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider>;
    pub fn agent_ids(&self) -> Vec<String>;                       // registration order
    pub fn agent_count(&self) -> usize;
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider);
    pub fn unregister_agent(&self, agent_id: &str) -> Option<AgentCard>;   // also removes subscriptions
    pub fn pin_agent(&self, agent_id: &str, agent: TeamAgent);    // reuse one instance instead of re-invoking the provider
    pub fn bind_team_session(&self, session: Arc<AgentTeamSession>);
    pub fn unbind_team_session(&self, session_id: &str);
    pub fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>>;
    pub fn p2p_timeout(&self) -> Duration;
    pub fn set_p2p_timeout(&self, timeout: Duration);
    pub async fn start(&self) -> Result<()>;
    pub async fn stop(&self) -> Result<()>;
    pub async fn send(&self, message: Value, recipient: &str, sender: &str, session_id: Option<&str>,
                      timeout: Option<Duration>) -> Result<Value>;
    pub async fn publish(&self, message: Value, topic_id: &str, sender: &str, session_id: Option<&str>) -> Result<()>;
    pub async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()>;
    pub async fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<()>;
    pub async fn cleanup_session(&self, session_id: &str);
    pub async fn invoke_agent(&self, agent_id: &str, inputs: Value,
                              session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
}
```
`send` validates a non-empty sender (`"sender is required for message tracing"`), a non-empty recipient (`"recipient is required"`), and registration (`"Recipient '<id>' not registered in runtime"`); an empty P2P response is an error (`TeamError::EmptyResponse`). The P2P handler builds a child `AgentSession` from the bound team session (`create_agent_session(recipient)`), else an `AgentSession` keyed by `session_id`, else `None`; pub/sub builds the same and discards the output.
`invoke_agent` resolves the pinned instance or invokes the provider, calls the member's bind hook, and invokes `Agent::invoke`.

### Strategy seam and team (`strategy.rs`, `team.rs`)

```rust
pub struct MemberOptions { pub parent_agent_id: Option<String> }
#[derive(Clone)] pub struct StrategyContext { pub card: TeamCard, pub config: TeamConfig,
                                              pub members: Vec<AgentCard>, pub runtime: Arc<TeamRuntime> }
#[async_trait]
pub trait TeamStrategy: Send + Sync {
    fn kind(&self) -> &str;                                     // "handoff" | "hierarchical_msgbus" | "hierarchical_tools"
    fn on_member_added(&self, ctx: &StrategyContext, card: &AgentCard, options: &MemberOptions) {}
    fn on_member_removed(&self, ctx: &StrategyContext, agent_id: &str) {}
    fn check_ready(&self, ctx: &StrategyContext) -> Result<()> { Ok(()) }
    async fn setup(&self, ctx: &StrategyContext) -> Result<()> { Ok(()) }           // lazy internal agents / hierarchy wiring
    async fn invoke(&self, ctx: &StrategyContext, input: Value,
                    session: Option<Arc<AgentTeamSession>>) -> Result<Value>;
    fn stream(&self, ctx: &StrategyContext, input: Value,
              session: Option<Arc<AgentTeamSession>>) -> BoxStream<'static, Result<Value>>;
}

#[async_trait]
pub trait Team: Send + Sync {
    fn card(&self) -> TeamCard;                                 // snapshot (member list mutates)
    fn config(&self) -> TeamConfig;
    fn team_id(&self) -> &str;                                  // Python parity: card.name
    fn kind(&self) -> &str;
    fn runtime(&self) -> Arc<TeamRuntime>;
    fn add_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()>;          // duplicate: warning + Ok(())
    fn add_member(&self, card: AgentCard, provider: AgentProvider, options: MemberOptions) -> Result<()>;
    fn remove_agent(&self, agent: &str) -> Result<()>;          // id or card id; unknown is a no-op
    fn get_agent_card(&self, agent_id: &str) -> Option<AgentCard>;
    fn agent_count(&self) -> usize;
    fn list_agents(&self) -> Vec<String>;
    fn check_ready(&self) -> Result<()>;
    async fn send(&self, message: Value, recipient: &str, sender: &str, session_id: Option<&str>,
                  timeout: Option<Duration>) -> Result<Value>;
    async fn publish(&self, message: Value, topic_id: &str, sender: &str, session_id: Option<&str>) -> Result<()>;
    async fn subscribe(&self, agent_id: &str, topic: &str) -> Result<()>;
    async fn unsubscribe(&self, agent_id: &str, topic: &str) -> Result<()>;
    async fn start(&self) -> Result<()>;                         // idempotent
    async fn stop(&self) -> Result<()>;                          // idempotent
    async fn invoke(&self, input: Value, session: Option<Arc<AgentTeamSession>>) -> Result<Value>;
    fn stream(&self, input: Value, session: Option<Arc<AgentTeamSession>>) -> BoxStream<'static, Result<Value>>;
}
pub struct BaseTeam { .. }
impl BaseTeam {
    pub fn new(card: TeamCard, config: TeamConfig) -> Self;      // strategy-free: invoke/stream return Unsupported
    pub fn with_strategy(card: TeamCard, config: TeamConfig, strategy: Arc<dyn TeamStrategy>) -> Self;
    pub fn with_runtime(card: TeamCard, config: TeamConfig, runtime: Arc<TeamRuntime>,
                        strategy: Arc<dyn TeamStrategy>) -> Self;
    pub fn strategy(&self) -> Arc<dyn TeamStrategy>;
}
impl Team for BaseTeam { .. }
```
`add_agent` enforces `max_agents` with `"Agent count exceeds max_agents ({n})"`, registers the provider in the runtime, appends the card to `card.agent_cards`, and calls `TeamStrategy::on_member_added`. `send`/`publish` validate registration like Python (`"Sender '<id>' not found in team '<team_id>'"`, `"Recipient '<id>' not found in team '<team_id>'"`).

### Sessions and streaming (`session.rs`)

```rust
pub fn make_team_session(card: &TeamCard, message: &Value) -> Arc<AgentTeamSession>;   // session_id from `conversation_id`, else uuid v4
pub struct SessionLease { .. }                                   // owns or borrows a team session bound to the runtime
impl SessionLease {
    pub async fn acquire(runtime: &Arc<TeamRuntime>, card: &TeamCard, message: &Value,
                         session: Option<Arc<AgentTeamSession>>) -> Self;
    pub fn session(&self) -> &Arc<AgentTeamSession>;
    pub fn session_id(&self) -> &str;
    pub fn is_standalone(&self) -> bool;
    pub async fn finish(self) -> Result<()>;                     // standalone: unbind -> cleanup_session -> end_stream -> commit
}
pub fn team_stream<F>(runtime: Arc<TeamRuntime>, card: TeamCard, message: Value,
                      session: Option<Arc<AgentTeamSession>>, run: F) -> BoxStream<'static, Result<Value>>
where F: FnOnce(Arc<AgentTeamSession>) -> BoxFuture<'static, Result<()>> + Send + 'static;
```
`team_stream` subscribes before spawning the run task, yields every `StreamItem::to_wire()` (`{type, index?, payload}`) in order, ends when the session stream closes, then joins the task and surfaces a run error after the last frame (never cancels the task — Python parity). A caller-provided session is only `end_stream`ed on completion; a standalone session gets the full teardown. `invoke` uses `SessionLease::acquire`; strategies decide when to finish (handoff holds the lease until the chain resolves).

### Agent tool (`agent_tool.rs`)

```rust
pub struct AgentTool { .. }                                      // an `AgentCard` exposed to a model as a `Tool`
impl AgentTool {
    pub fn new(card: &AgentCard, runtime: Weak<TeamRuntime>) -> Self;
    pub fn with_limiter(self, limiter: Arc<Semaphore>) -> Self;  // caps concurrent sub-agent dispatch
    pub fn card(&self) -> &ToolCard;
}
impl Tool for AgentTool { .. }
```
Card: `id = name = AgentCard::name`, `description = AgentCard::description`, `input_params = AgentCard::input_params` or `{"type":"object","properties":{}}`, `parallel_safe = true`. On invoke it derives `conversation_id = "{parent_sid}:{tool_call_id}"` (the tool-call id from `ToolContext::metadata["tool_call_id"]`, falling back to `invocation_id`), creates a child `AgentSession`, invokes the member through the runtime, and returns the `AgentOutput` value; failures become a tool error string (`"Agent execution error: {e}"`).

### Handoff strategy (`handoff/`)

```rust
pub struct HandoffRoute { pub source: String, pub target: String }
pub type TerminationCondition = Arc<dyn Fn(HandoffSnapshot) -> BoxFuture<'static, Result<bool>> + Send + Sync>;
pub struct HandoffSnapshot { pub current_agent_id: String, pub handoff_count: usize }
pub struct HandoffConfig { pub start_agent: Option<AgentCard>, pub max_handoffs: usize /*10*/,
                           pub routes: Vec<HandoffRoute>, #[serde(skip)] pub termination_condition: Option<TerminationCondition> }
pub struct HandoffTeamConfig { #[serde(flatten)] pub team: TeamConfig, pub handoff: HandoffConfig }
pub fn handoff_team(card: TeamCard, config: HandoffTeamConfig) -> BaseTeam;
pub struct HandoffStrategy { .. }
impl HandoffStrategy { pub fn new(config: HandoffTeamConfig) -> Arc<Self>; }
impl TeamStrategy for HandoffStrategy { .. }

pub const HANDOFF_TARGET_KEY: &str = "__handoff_to__";
pub const HANDOFF_MESSAGE_KEY: &str = "__handoff_message__";
pub const HANDOFF_REASON_KEY: &str = "__handoff_reason__";
pub const COORDINATOR_STATE_KEY: &str = "__handoff_coordinator__";
pub const HISTORY_STATE_KEY: &str = "__handoff_history__";
pub const CONTEXT_HISTORY_STATE_KEY: &str = "__handoff_ctx_history__";
pub struct HandoffSignal { pub target: String, pub message: Option<String>, pub reason: Option<String> }
pub fn extract_handoff_signal(result: &Value) -> Option<HandoffSignal>;                       // top-level > output > result > content
pub fn extract_handoff_signal_from_observations(observations: &[Observation]) -> Option<HandoffSignal>;  // newest first
pub async fn extract_handoff_signal_from_session(result: &Value, session: &Arc<dyn Session>) -> Option<HandoffSignal>;
pub struct HandoffTool { .. }                                                                 // `transfer_to_<id>`, never errors
impl HandoffTool { pub fn new(target_id: impl Into<String>, target_description: &str) -> Self; pub fn name(&self) -> &str; }
pub struct HandoffHistoryEntry { pub agent: String, pub output: Value }
pub struct HandoffRequest { pub input_message: Value, pub history: Vec<HandoffHistoryEntry> }
pub struct HandoffOrchestrator { .. }
impl HandoffOrchestrator {
    pub fn new(start_agent_id: impl Into<String>, members: &[String], config: &HandoffConfig) -> Self;
    pub fn build_route_graph(members: &[String], routes: &[HandoffRoute]) -> BTreeMap<String, BTreeSet<String>>;
    pub fn current_agent_id(&self) -> String;
    pub fn handoff_count(&self) -> usize;
    pub async fn request_handoff(&self, target_id: &str, reason: Option<&str>) -> Result<bool>;
    pub async fn complete(&self, result: Value) -> bool;          // false when already resolved
    pub async fn error(&self, error: Error) -> bool;
    pub async fn wait(&self, timeout: Option<Duration>) -> Result<Value>;
    pub async fn save_to_session(&self, session: &Arc<AgentTeamSession>) -> Result<()>;
    pub fn restore_from_session(snapshot: Option<&Value>, start_agent_id: impl Into<String>,
                                members: &[String], config: &HandoffConfig) -> Result<Self>;
}
pub struct ContainerAgent { .. }                                  // the per-hop endpoint agent
impl Agent for ContainerAgent { .. }
pub struct TeamInterruptSignal { pub result: Value, pub message: Option<String> }
pub fn extract_interrupt_signal(result: &Value) -> Option<TeamInterruptSignal>;
pub async fn flush_team_session(session: Option<&Arc<AgentTeamSession>>);
```
Handoff flow: `setup` registers one endpoint per member (`AgentCard{id: name: "__handoff_ep_{team_id}_{agent_id}"}`) whose provider returns a `ContainerAgent`, and subscribes the endpoint to `container_{agent_id}`. `invoke` restores the coordinator from `__handoff_coordinator__`, publishes a `HandoffRequest` to `container_{current}`, and awaits the coordinator with `message_timeout` when non-zero; teardown drops the coordinator and calls `cleanup_session`. The container resolves its target through the provider, mounts one `transfer_to_<target>` tool per allowed target on the target's `ToolHost` (silent skip when absent, flag set first — Python parity), invokes the target in a team child session, appends `{"agent", "output"}` to history, then either completes, publishes the next hop, or forwards an interrupt (`save_to_session` + `__handoff_history__` + `flush_team_session` + complete). A blocked hop completes with the current result; a failing member reports `coordinator.error` and returns `{}` (never raises).

### Hierarchical strategies (`hierarchical/`)

```rust
pub struct HierarchicalMsgbusConfig { #[serde(flatten)] pub team: TeamConfig,
                                     pub supervisor_agent: AgentCard, pub timeout: Option<f64> /*1800.0*/ }
pub fn hierarchical_msgbus_team(card: TeamCard, config: HierarchicalMsgbusConfig) -> BaseTeam;
pub struct SupervisorConfig { pub card: AgentCard, pub model: Arc<dyn Model>, pub system_prompt: String,
                              pub max_iterations: usize /*5*/, pub max_parallel_sub_agents: usize /*10*/ }
pub struct SupervisorAgent;
impl SupervisorAgent {
    pub fn create(agents: Vec<AgentCard>, config: SupervisorConfig) -> Result<(AgentCard, AgentProvider)>;
}
pub fn create_supervisor(agents: Vec<AgentCard>, config: SupervisorConfig) -> Result<(AgentCard, AgentProvider)>;
pub struct HierarchicalToolsConfig { #[serde(flatten)] pub team: TeamConfig, pub root_agent: AgentCard }
pub fn hierarchical_tools_team(card: TeamCard, config: HierarchicalToolsConfig) -> BaseTeam;
```
`hierarchical_msgbus_team` sends the input to the supervisor with one P2P `send` (timeout = config timeout, overridable per invoke), and streams exactly one `{"output": <result>}` frame. `add_agent` for the supervisor id sets the runtime P2P timeout. `SupervisorAgent::create` builds a `ReActAgent` (model, one system prompt message, `max_iterations`) with one `AgentTool` per sub-agent card sharing a `Semaphore(max_parallel_sub_agents)`; results stay in tool-call order.
`hierarchical_tools_team` wires pending `(parent, child)` pairs by mounting an `AgentTool` for each child on the parent's `ToolHost`, pins the parent instance so later sends reach the wired instance, then sends the input verbatim to the root. Its stream bypasses the bus: it calls the root agent's `stream` with `{conversation_id, sender}` merged into the input, forwards each `AgentChunk::to_value()` as a team frame, and turns a failure into a single `{"output": <message>, "result_type": "error"}` frame.

### Errors

`openjiuwen_common::error::{Error, Result}` everywhere. Empty sender/recipient/subscription args and `max_agents` violations are `Validation`; unknown members are `NotFound`; deadlines are `Timeout`; missing strategy configuration (supervisor/root not registered) is `Config`; a strategy-free team, deferred transport, or unimplemented strategy is `Unsupported`; an empty P2P response is `Error::External(TeamError::EmptyResponse)`.
Members' errors propagate unchanged over P2P; pub/sub delivery failures and handoff member failures are logged and never abort the publisher.

## Constraints

- Follows `docs/ARCHITECTURE.md`: async-first tokio, `openjiuwen_common::error` results, serde-serializable public data, no `unsafe`, `tracing` logging (no stdout), no blocking calls on async tasks.
- No dependency on `openjiuwen-runner`; only `common`, `foundation`, `session`, `agent` and third-party crates from the workspace manifest.
- Public data types are `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default collections to empty; process-local types (models, tools, runtimes, sessions) are not serializable.
- Files stay around ≤1000 lines, split by responsibility (card/config, envelope/subscription/bus, runtime, team/strategy, per-strategy modules).
- Locks are never held across `.await`; interior mutability uses `parking_lot::RwLock`/`Mutex` for short sections and `tokio::sync` primitives for awaitable coordination.
- Every spawned task is owned: bus workers are joined by `stop`, the streaming run task is joined after the stream drains; a `send` timeout leaves the handler running (documented Python parity) but the worker is always tracked by its topic.
- No global mutable state: a `TeamRuntime` is per team, constructed by the team and shared as `Arc`.

## Design Decisions

- **One team type plus strategy objects** instead of Python's abstract `BaseTeam` + subclassing: `BaseTeam` owns card/config/runtime/member registry and implements `Team`; `HandoffStrategy`/`HierarchicalMsgbusStrategy`/`HierarchicalToolsStrategy` implement `TeamStrategy` with `on_member_added`/`setup`/`invoke`/`stream`. Custom strategies are `BaseTeam::with_strategy(..)`.
- **`Team` is the runner seam**: runner calls `team.invoke(input, Some(team_session))`, `team.stream(..)`, and `team.runtime().bind_team_session(session)` / `unbind_team_session(id)`. The session parameter is `Option<Arc<AgentTeamSession>>` because strategies need `create_agent_session`/`commit`, which `dyn Session` does not expose.
- **Member providers return `TeamAgent`**, which pairs the `Arc<dyn Agent>` with optional capabilities (tool host, bind hook). This replaces Python attribute probing (`agent.ability_manager`) and late `Runner` binding without changing observable strategy behaviour.
- **The bus is a transport, the runtime is the dispatcher**: `MessageBus` owns topics, FIFO workers, timeouts, and subscriber patterns; `TeamRuntime` implements the handler (agent resolution, session construction, invocation). No `Arc` cycle: the bus holds a handler over `Weak<TeamRuntime>`, and the runtime state never references the bus.
- **Per-topic FIFO workers** give per-sender ordering and per-topic serialization, stronger and more deterministic than Python's fire-and-forget handler tasks; fan-out across subscribers is concurrent and waits for completion, like Python's `gather`.
- **Subscriber ordering is sorted** (`BTreeSet`), where Python returns an unordered set: deterministic tests and reproducible delivery.
- **Session identity comes from the bus envelope**, not the message payload: `HandoffRequest` no longer carries a session object; the container looks the team session up via `runtime.team_session(envelope.session_id)` (Rust adaptation of Python's object-carrying dataclass).
- **Signal extraction also reads `AgentOutput.observations`** (newest first) before falling back to the session `context` tool messages; this removes the dependency on a specific context-engine state layout while preserving the JSON payload contract.
- **Pinned instances for wired hierarchies**: `hierarchical_tools` mounts child tools on a parent instance obtained from the provider and pins it in the runtime, so the later `send` reaches the wired instance (fixes the Python fresh-provider trap; documented divergence).
- **`SupervisorAgent::create` takes an `Arc<dyn Model>`** instead of client/request configs: model construction stays with the caller (`init_model`), which keeps this crate provider-agnostic and lets tests pass a scripted model.
- **Streams yield wire frames** (`StreamItem::to_wire()`), matching Python consumers that see `OutputSchema` objects, with `type="message"` as the default team frame kind.

## Routing Table

- `./src/card.rs` → `TeamCard`, `EventDrivenTeamCard`, member list
- `./src/config.rs` → `TeamConfig` defaults, builders, timeout semantics
- `./src/member.rs` → `TeamAgent`, `AgentProvider`, `ToolHost`, bind hook
- `./src/envelope.rs` → `MessageEnvelope` and P2P/pub-sub classification
- `./src/subscription.rs` → topic-pattern registry (`*`/`?`), per-agent index
- `./src/message_bus.rs` → topics, FIFO workers, timeouts, fan-out, lifecycle, `MessageHandler`
- `./src/runtime.rs` → `TeamRuntime`, registration, session binding, P2P/public dispatch, pinned instances
- `./src/session.rs` → team session construction, `SessionLease`, stream driver
- `./src/strategy.rs` → `TeamStrategy`, `StrategyContext`, `MemberOptions`
- `./src/team.rs` → `Team` trait, `BaseTeam` machinery and validation
- `./src/agent_tool.rs` → `AgentTool` (agent as a `Tool`)
- `./src/handoff/` → handoff config/signal/tool/orchestrator/container/interrupt and `HandoffStrategy` (`handoff/CONTEXT.md`)
- `./src/hierarchical/` → msgbus supervisor team, supervisor factory, agents-as-tools team (`hierarchical/CONTEXT.md`)
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests beside each module: card/config serde defaults and unknown-field tolerance; envelope P2P/pub-sub classification; wildcard matching and subscription indexing; bus duplicate-subscribe, cleanup/re-subscribe, timeout, fan-out failure isolation, start/stop idempotence; runtime registration/pinning/unregistration; team validation messages and `max_agents`; orchestrator route graph/cap/termination/first-wins; signal precedence; handoff tool card shape and never-raise; `AgentTool` card and dispatch.
- `tests/team_card.rs` pins the wire contract: JSON field names, `version: "1.0.0"`, `agent_cards`, `subscriptions`, `TeamConfig` defaults 10/100/30.0.
- `tests/message_bus.rs` drives the bus directly with a recording handler: P2P ordering per sender, concurrent fan-out to sorted subscribers, missing-subscriber publish no-op, timeout error text, session cleanup, graceful shutdown.
- `tests/runtime.rs` covers registration, provider binding, P2P session construction from a bound team session, empty-response error, and P2P timeout override.
- `tests/team.rs` covers `add_agent`/`add_member`/`remove_agent`, duplicate no-op, `max_agents`, queries, send/publish validation, start/stop idempotence, a custom strategy, and the unsupported-strategy error.
- `tests/handoff.rs` covers the full chain with scripted member agents: direct handoff, routed handoff, blocked hop completes, max-handoffs, timeout, interrupt forwarding, history resume, and streaming frames.
- `tests/hierarchical.rs` covers supervisor-over-bus single hop + single stream frame, sub-agent tool cards and ordered parallel dispatch with the limiter, agents-as-tools wiring (`ToolHost` mount + pinned instance), and the error chunk.
- `tests/streaming.rs` covers `team_stream` frame order, caller-owned vs standalone teardown, and producer-error-after-drain.
- `tests/api_surface.rs` pins the public contract at compile time: every crate-root re-export, inherent constructor and free-function signature, trait-object coercion (`Team`, `TeamStrategy`, `MessageHandler`, `ToolHost`, `Tool`), `Team` method reachability, the `Card` impl, serde round-trips and `schemars` property names, and the `TeamError` → `ErrorKind::External` mapping.
- All tests are offline and deterministic: fake agents are local closures, models come from `openjiuwen-test-support::ScriptedModel`, timeouts use short durations only.

## Deferred Python Behavior

- Distributed transports (Pulsar/Redis message queues, `drunner` factories) — in-process only; `MessageBus` is the seam a distributed backend would implement.
- `AgentTeamEvents` callbacks (`agent_p2p_received`, `agent_pubsub_received`) — no callback framework in this crate.
- `EventDrivenTeamCard.subscriptions` is data only; event-driven routing via `BaseTeam.subscribe` is supported, card-driven auto-subscription is not wired.
- Python's `Runner.resource_mgr` global registry path: strategies resolve members from the team runtime only (the runner registers its members with the team).
- Per-member checkpoint/cache identity (`team:{team}:member:{id}`, shared session ids) — Rust uses parented child `AgentSession`s.
- Python-repr fallback (`ast.literal_eval`) in signal extraction — JSON only.
- `TeamConfig` re-wiring after `configure` (Python only applies it at default-runtime construction) — the config snapshot at construction is authoritative.
- `P2PAbilityManager` subclassing — replaced by mounted `AgentTool`s with a shared semaphore.
- `SupervisorAgent`'s exact ReAct config surface (prompt template names, context engine wiring) — only card, model, system prompt, max iterations, parallel cap.
- No round-robin/selector/scheduled strategies exist in Python's `core.multi_agent`; custom strategies plug in through `TeamStrategy`.

## Notes for Agents

- `team_id` is `card.name`; topics, sessions, and registries key on `card.id` (Python parity — keep both).
- `message_timeout` (30 s) is the bus process timeout and the handoff chain deadline; `p2p_timeout` (1800 s) is the supervisor dispatch deadline. Keep the three deadlines distinct.
- A duplicate `add_agent` logs a warning and returns `Ok(())`; `max_agents` is checked only for new members.
- `HandoffRequest::history` is appended before interrupt detection, so an interrupted hop is recorded then dropped on resume.
- `ContainerAgent::invoke` returns `{}` on every path; results reach consumers only through the shared team stream.
- `flush_team_session` = `end_stream()` then `commit()`; a commit failure is logged, never raised.
