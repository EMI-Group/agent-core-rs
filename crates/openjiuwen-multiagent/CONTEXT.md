# openjiuwen-multiagent

## Intent

Teams of agents: team cards/configs, an in-process message bus and runtime, and the three strategies the Python SDK ships (`handoff`, supervisor-over-message-bus, agents-as-tools).
Ports Python `openjiuwen.core.multi_agent` (v0.1.20, 31 files / ~4.1k lines) plus the team-session helpers from `core.session.agent_team`; the product-level `agent_teams` package (leader/teammate, worktrees) is out of scope (see `docs/PARITY.md`), and of the distributed transports only Redis is implemented (`Pulsar` is a typed `Unsupported`).
tokio-native throughout: the Python `MessageQueueInMemory` transport becomes a per-topic FIFO worker with oneshot request/response correlation inside `InProcessTransport`, and `TeamTransport` is the seam a Redis-backed deployment swaps in.
Team events, card-driven auto-subscription, a resource-manager seam, per-member checkpoint identity, `P2PAbilityManager` and Python-literal rendering are implemented here rather than left to the runner.
The crate never depends on `openjiuwen-runner` (runner depends on it): members are resolved from the team's own runtime registry, and the `Team` trait is the seam the runner drives.

## API Surface

### Crate root (`lib.rs`)

Re-exports: `TeamCard`, `EventDrivenTeamCard`, `TeamConfig`, `TeamError`; `Team`, `BaseTeam`, `TeamStrategy`, `StrategyContext`, `MemberOptions`; `TeamAgent`, `AgentProvider`, `ToolHost`, `BindHook`; `MessageEnvelope`, `MessageBus`, `MessageBusConfig`, `MessageHandler`, `SubscriptionManager`; `TeamRuntime`, `RuntimeConfig`, `SessionLease`, `make_team_session`; `AgentTool`.
Extended re-exports: `TeamEvent`, `TeamEventBus`, `TeamEventFilter`, `TeamEventKind`, `TeamEventListener`, the `TEAM_EVENT_SCOPE`/`AGENT_P2P_RECEIVED_EVENT`/`AGENT_PUBSUB_RECEIVED_EVENT` constants; `AutoSubscriptionMode`, `AutoSubscriptionPlan`, `derive`, `derive_with_overrides`, `derive_from_event_card`, `derived_for_member`, `apply`, `apply_agent`, `SUBSCRIBE_TAG_PREFIX`; `parse_python_literal`, `literal_or_json`, `render_python_value`, `MAX_LITERAL_DEPTH`; `TeamResourceMgr`, `LocalResourceMgr`; `MemberCheckpointer`, `member_checkpoint_key`, `member_checkpoint_id`, `MEMBER_CHECKPOINT_NAMESPACE`; `TeamTransport`, `TransportKind`, `TransportConfig`, `RedisTransportConfig`, `InProcessTransport`, `RedisTransport`, `build_transport`; `AbilityCall`, `AbilityRoute`, `AbilityResult`, `AbilityExecutor`, `P2PAbilityManager`.
Each extended subsystem is also a public module (`openjiuwen_multiagent::events`, `::autosubscribe`, `::pyrepr`, `::resources`, `::member_checkpoint`, `::transport`, `::ability`); the module path and the root re-export name the same item.
Strategy trees: `pub mod handoff` (private module, re-exported at the root: `HandoffTeamConfig`, `HandoffConfig`, `HandoffRoute`, `HandoffSignal`, `HandoffRequest`, `HandoffHistoryEntry`, `HandoffTool`, `HandoffOrchestrator`, `ContainerAgent`, `TeamInterruptSignal`, `extract_interrupt_signal`, `flush_team_session`, `handoff_team`, `HandoffStrategy`, the `HANDOFF_*` constants) and `pub mod hierarchical` (`HierarchicalMsgbusConfig`, `HierarchicalMsgbusStrategy`, `hierarchical_msgbus_team`, `SupervisorAgent`, `SupervisorConfig`, `create_supervisor`, `HierarchicalToolsConfig`, `HierarchicalToolsStrategy`, `hierarchical_tools_team`, plus the Python-shaped path `hierarchical::P2PAbilityManager`/`AbilityCall`/`AbilityResult`/`AbilityRoute`).
Both hierarchical configs expose `new(entry_agent_card)` and both strategies expose `new(config) -> Arc<Self>`.

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
pub struct TeamRuntime { /* Arc<RuntimeState> + Arc<MessageBus> + Arc<dyn TeamTransport> */ }
impl TeamRuntime {
    pub fn new(config: RuntimeConfig) -> Arc<Self>;
    pub fn new_with_resource_mgr(config: RuntimeConfig, resource_mgr: Arc<dyn TeamResourceMgr>) -> Arc<Self>;
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
    // resource seam (Python `Runner.resource_mgr` shape) — src/runtime/seams.rs
    pub fn set_resource_mgr(&self, resource_mgr: Arc<dyn TeamResourceMgr>);
    pub fn resource_mgr(&self) -> Option<Arc<dyn TeamResourceMgr>>;
    pub fn resolve_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>>;          // local instance first, then the manager
    pub fn resolve_tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn resolve_model(&self, model_id: &str) -> Option<Arc<dyn Model>>;
    pub fn resolve_team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>>;
    pub fn resolve_agent_card(&self, agent_id: &str) -> Option<AgentCard>;
    pub fn resolve_agent_provider(&self, agent_id: &str) -> Option<AgentProvider>;
    // events / member checkpoints / transport
    pub fn events(&self) -> Arc<TeamEventBus>;
    pub fn set_member_checkpointer(&self, checkpointer: Arc<MemberCheckpointer>);
    pub fn member_checkpointer(&self) -> Option<Arc<MemberCheckpointer>>;
    pub fn transport(&self) -> Arc<dyn TeamTransport>;
    pub fn set_transport(&self, transport: Arc<dyn TeamTransport>) -> Result<()>;  // Validation on team-id mismatch
    pub async fn start(&self) -> Result<()>;                      // idempotent; delegates to the transport
    pub async fn stop(&self) -> Result<()>;                       // idempotent; delegates to the transport
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
`invoke_agent` resolves the pinned instance or invokes the provider, calls the member's bind hook, restores that member's checkpoint when a `MemberCheckpointer` is configured (before the call), invokes `Agent::invoke`, then saves the member's state after success; a member error propagates unsaved.
Lookups fall back to the resource manager: `agent_card`/`agent_provider` consult the runtime registry first, then `TeamResourceMgr`; `resolve_team_session` consults the bound team session first, then the manager; `resolve_tool`/`resolve_tool_by_name`/`resolve_model` consult the manager only (the runtime keeps no tool or model store of its own), so without a manager they return `None`. `start`/`stop` delegate to `transport()` (`InProcessTransport` over the message bus by default).
Events are emitted from the runtime itself: `register_agent` → `member_joined` (repeat registration carries `detail.replaced = true`), `pin_agent` of an unknown id → `member_joined`, `unregister_agent` of a known id → `member_left`, `send`/`publish` → `message_sent` (with `topic_id` for pub/sub), `dispatch_p2p`/`dispatch_pubsub` → `message_received`/`pubsub_received` before invoke, and every error path → `error` with `detail.operation` (`"send"`, `"publish"`, `"p2p_dispatch"`, `"pubsub_dispatch"`).

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
    fn remove_member(&self, agent: &str) -> Result<()>;         // alias of remove_agent (id or card id; unknown is a no-op)
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
    // card-driven auto-subscription
    pub fn with_auto_subscriptions(self, mode: AutoSubscriptionMode) -> Self;     // default Merge
    pub fn auto_subscriptions(&self) -> AutoSubscriptionMode;
    pub fn set_auto_subscriptions(&self, mode: AutoSubscriptionMode);
    pub fn with_subscription_overrides(self, overrides: BTreeMap<String, Vec<String>>) -> Self;
    pub fn subscription_overrides(&self) -> BTreeMap<String, Vec<String>>;
    pub fn subscription_plan(&self) -> AutoSubscriptionPlan;     // derived from the card + overrides
    pub fn apply_subscriptions(&self) -> usize;                  // whole plan; returns patterns applied
    pub fn apply_member_subscriptions(&self, agent_id: &str) -> usize;   // just one member
}
impl Team for BaseTeam { .. }
```
`add_agent` enforces `max_agents` with `"Agent count exceeds max_agents ({n})"`, registers the provider in the runtime, appends the card to `card.agent_cards`, and calls `TeamStrategy::on_member_added`. `send`/`publish` validate registration like Python (`"Sender '<id>' not found in team '<team_id>'"`, `"Recipient '<id>' not found in team '<team_id>'"`).
Auto-subscription runs in `Merge` mode at member insert (`add_agent`, `add_member`, and the duplicate-add no-op path), at `Team::start` for the whole plan, and in reverse on `remove_agent`/`remove_member` (that member's derived patterns are unsubscribed). `Include`-style explicit overrides (`with_subscription_overrides`) replace a member's derived pattern set inside `subscription_plan`; `AutoSubscriptionMode::Disabled` leaves the logical registry untouched.

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
Card: `id = name = AgentCard::name`, `description = AgentCard::description`, `input_params = AgentCard::input_params` or `{"type":"object","properties":{}}`, `parallel_safe = true`. On invoke it derives `conversation_id = "{parent_sid}:{tool_call_id}"` (the tool-call id from `ToolContext::metadata["tool_call_id"]`, falling back to `invocation_id`), creates a child `AgentSession`, invokes the member through the runtime, and returns the `AgentOutput` value; failures become a tool error string (`"Agent execution error: {e}"`). Every dispatch emits `task_assigned` before the member runs and `task_completed` after it returns, or `error` with `detail.operation = "agent_tool"` when it fails.
`Tool::render_for_llm` mirrors Python's `str(result)` conversion: a bare string passes through unchanged, an object whose only meaningful key is `content` renders that string, `Null` keeps the trait's default empty text, and any other JSON value renders through `pyrepr::render_python_value` (single-quoted dict/list/string repr rather than JSON).

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
Signal extraction accepts Python literals as well as JSON: `extract_handoff_signal*` run the content through `pyrepr::literal_or_json`, so a member that returns `{'__handoff_to__': 'b'}` (single quotes, `True`/`None`) still routes.
The container emits team events: `handoff_started` per hop (`detail.hop`, 1-based; `peer_id` = the member about to run), `handoff_completed` once on the winning completion (`detail.handoffs`, `detail.interrupted = true` for an interrupt forward), and `error` with `detail.operation = "handoff"` when a hop fails. A dropped runtime emits nothing: the hop cannot locate the team session/coordinator any more, so the caller surfaces the pre-existing `External` error.

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
`hierarchical_msgbus_team` sends the input to the supervisor with one P2P `send` (timeout = config timeout, overridable per invoke), and streams exactly one frame `{"type": "message", "index": 0, "payload": {"output": <result>}}`. `add_agent` for the supervisor id sets the runtime P2P timeout. `SupervisorAgent::create` builds a `ReActAgent` (model, one system prompt message, `max_iterations`) with one `AgentTool` per sub-agent card sharing a `Semaphore(max(1, max_parallel_sub_agents))`; results stay in tool-call order.
`hierarchical_tools_team` wires pending `(parent, child)` pairs by mounting an `AgentTool` for each child on the parent's `ToolHost`, pins the parent instance so later sends reach the wired instance, then sends the input verbatim to the root.
Its stream bypasses the bus: it calls the root agent's `stream` with `{conversation_id, sender}` merged into the input, forwards each `AgentChunk::to_stream_item()` as a team frame, and turns an agent stream failure into a single `{"type": "message", "index": 0, "payload": {"output": <message>, "result_type": "error"}}` frame.

### Extended subsystems

The event bus, auto-subscription, Python literals, resource seam, member checkpoints, transport seam and `P2PAbilityManager` are documented at module depth in `./src/CONTEXT.md`.
In one line each: `events` delivers typed `TeamEvent`s in sequence order to filtered, unsubscribable listeners; `autosubscribe` derives deterministic per-member patterns from the card (broadcast topic, `{topic}.*`, directed `{topic}.{member}`, `subscribe:` tags) and applies them in `Merge`/`Replace`/`Disabled` mode with explicit per-agent overrides; `pyrepr` is the `ast.literal_eval`/`str()` fallback used by signal extraction and tool-result rendering; `resources` is the all-defaulted `TeamResourceMgr` seam `resolve_*` falls back to; `member_checkpoint` scopes one shared checkpointer per `(team_session_id, agent_id)` under the `agent` namespace; `transport` is the `TeamTransport` seam with in-process and Redis backends (`Pulsar` = `Unsupported`); `ability` dispatches a supervisor's calls as P2P agent calls or local executor calls, in input order, bounded by a semaphore.

### Rust extensions beyond the Python surface (`routing.rs`, `inbox.rs`, `communicable.rs`)

```rust
pub const GROUP_METADATA_KEY: &str = "group";
pub struct GroupRegistry { .. }
impl GroupRegistry {
    pub fn new() -> Self;
    pub fn register_group(&self, group_id: impl Into<String>, members: impl IntoIterator<Item = String>) -> Result<usize>;
    pub fn add_member(&self, group_id: &str, agent_id: &str) -> Result<bool>;   // true when newly added
    pub fn remove_member(&self, group_id: &str, agent_id: &str) -> bool;
    pub fn unregister_group(&self, group_id: &str) -> bool;
    pub fn group_members(&self, group_id: &str) -> Vec<String>;                 // sorted
    pub fn groups(&self) -> Vec<String>;  pub fn groups_of(&self, agent_id: &str) -> Vec<String>;
    pub fn group_count(&self) -> usize;   pub fn member_count(&self) -> usize;  pub fn clear(&self) -> usize;
}
pub fn broadcast_targets(members: impl IntoIterator<Item = String>, sender: Option<&str>,
                         include_sender: bool) -> Vec<String>;                  // sorted, deduped, sender excluded by default
pub struct GroupDelivery { pub agent_id: String, pub outcome: Result<Value> }   // not serializable
pub struct GroupBroadcast { pub group: String, pub recipients: Vec<String>, pub skipped: Vec<String>,
                            pub deliveries: Vec<GroupDelivery> }
impl GroupBroadcast { pub fn succeeded(&self) -> usize; pub fn failed(&self) -> usize;
                      pub fn value_of(&self, agent_id: &str) -> Option<&Value>;
                      pub fn into_results(self) -> BTreeMap<String, Result<Value>>; }
pub struct RoutingTable { pub team_id: String, pub groups: BTreeMap<String, Vec<String>>,
                          pub topics: BTreeMap<String, Vec<String>>, pub agents: BTreeMap<String, Vec<String>> }
impl RoutingTable { pub fn route(&self, destination: &str) -> Vec<String>;      // group first, then topic patterns
                    pub fn destinations(&self, agent_id: &str) -> Vec<String>; }
pub struct InboxFilter { .. }        // sender allow/deny, topic patterns, required metadata, require_session
impl InboxFilter { pub fn new() -> Self; pub fn allow_senders(..) -> Self; pub fn deny_senders(..) -> Self;
                   pub fn allow_topics(..) -> Self; pub fn require_metadata(key, value) -> Self;
                   pub fn require_session(self) -> Self; pub fn accepts(&self, envelope: &MessageEnvelope) -> bool; }
pub struct InboxFilterRegistry { .. } // set/remove/get/accepts/agents/len/clear; an agent without a filter accepts everything
pub struct CommunicableAgent { .. }   // Python mixin counterpart: bind_runtime/is_bound/runtime/agent_id/unbind
                                      // plus send/publish/subscribe/unsubscribe (unbound => Validation)
```
`TeamRuntime` also gained group/inbox forwarders (`groups`, `inbox_filters`, `register_group`, `unregister_group`, `group_members`, `groups_of`, `set_inbox_filter`, `remove_inbox_filter`, `inbox_filter`, `routing_table`), `broadcast_to_group(message, group_id, sender, session_id, timeout)`, and the Python-parity introspection `is_running`, `list_subscriptions`, `list_agent_subscriptions`, `get_subscription_count`.
`MessageBus` gained `remove_all_subscriptions`, `list_subscriptions`, `list_agent_subscriptions`, `get_subscription_count`; `TeamTransport` gained three defaulted introspection methods (`list_subscriptions`, `agent_subscriptions`, `subscription_count`) overridden by both backends.
Group broadcast is one P2P `send` per member (concurrently, failures isolated) with `metadata["group"]` stamped on every envelope; the sender is excluded unless the caller opts in, and unregistered members are skipped and reported.
An inbox filter is enforced in `TeamRuntime::dispatch_p2p` (rejection is a `Validation` error at the sender) and `dispatch_pubsub` (rejection is logged and the delivery skipped).

### Errors

`openjiuwen_common::error::{Error, Result}` everywhere. Empty sender/recipient/subscription args and `max_agents` violations are `Validation`; unknown members are `NotFound`; deadlines are `Timeout`; missing strategy configuration (supervisor/root not registered) is `Config`; a strategy-free team, a `Pulsar` transport, or an unimplemented strategy is `Unsupported`; an empty P2P response is `Error::External(TeamError::EmptyResponse)`.
Transport selection adds `Error::Config` for `TransportKind::Redis` without `TransportConfig::redis`, `Error::Validation` for `set_transport` with a foreign team id or an empty `RedisTransport::watch_agent` id, and `Error::Timeout` for a backend deadline (`"Message from '<sender>' to '<recipient>' timed out after <t>s"`). An ability-call failure never escapes `execute` as `Err`; it becomes an `AbilityResult` with `AbilityRoute::Error` and the message in `error`.
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
- **Session identity comes from the invoked child session**, not the message payload: `HandoffRequest` carries no session object; the container resolves the team session from the child session's parent id (`runtime.team_session(session.parent_id()?)`), the Rust adaptation of Python's object-carrying dataclass.
- **Signal extraction also reads `AgentOutput.observations`** (newest first) before falling back to the session `context` tool messages; this removes the dependency on a specific context-engine state layout while preserving the JSON payload contract.
- **Pinned instances for wired hierarchies**: `hierarchical_tools` mounts child tools on a parent instance obtained from the provider and pins it in the runtime, so the later `send` reaches the wired instance (fixes the Python fresh-provider trap; documented divergence).
- **`SupervisorAgent::create` takes an `Arc<dyn Model>`** instead of client/request configs: model construction stays with the caller (`init_model`), which keeps this crate provider-agnostic and lets tests pass a scripted model.
- **Streams yield wire frames** (`StreamItem::to_wire()`), matching Python consumers that see `OutputSchema` objects, with `type="message"` as the default team frame kind.
- **Events are a typed bus, not Python's callback names**: `TeamEventBus` carries `TeamEventKind` values, an ordering `sequence`, and data (`agent_id`, `peer_id`, `detail`); `TeamEvent::python_event_name` maps the two Python callback events onto their scope names so a bridge can register under `_framework:agent_p2p_received`. `emit` is sync so any runtime method can publish without holding a lock across an await; listeners are addressed by subscription id (unsubscribe).
- **Card-driven subscription is a pure derivation plus an apply step**: `autosubscribe::derive*` computes the plan from `TeamCard` data with no side effects (deterministic BTreeMap, sorted patterns), and `apply` mutates the runtime's logical registry under a mode (`Merge` default, `Replace`, `Disabled`). Explicit overrides live next to the mode on `BaseTeam`, so the override is visible at the wiring point rather than buried in the card.
- **The resource manager is a trait with all-default methods**: `openjiuwen-runner` depends on this crate, so the crate cannot name the runner's concrete manager. `TeamResourceMgr` is implemented by the runner; `LocalResourceMgr` is the standalone instance-scoped implementation; `resolve_*` is local-first, manager-second, which keeps every existing in-process code path intact.
- **Member checkpoint identity is `(team_session_id, agent_id)` at the `agent` namespace**: the team session id comes from the child session's parent, the agent id from its metadata, so two members sharing one `MemoryCheckpointer` cannot overwrite each other and a resumed team run restores each member's own state. Checkpoint ids are append-only (`{key}:{sequence:06}`), so history is total-ordered per member.
- **Transport is the seam, the message bus is one implementation**: `TeamTransport` abstracts `send`/`publish`/subscriptions so `InProcessTransport` (message-bus FIFO workers) and `RedisTransport` (RESP2 Streams consumer groups with polled readers) are interchangeable; `Pulsar` is a typed `Unsupported`. The runtime delegates and never depends on a backend detail.
- **Python-repr rendering is opt-in per call site**: `pyrepr` parses and renders Python literals only where Python itself does `ast.literal_eval`/`str()` (handoff signal extraction, tool-result rendering), so ordinary JSON handling stays serde-based and lossless.
- **Redis delivery is Streams consumer groups, not lists**: `RPOP` was destructive and per-instance, so two processes could not share a team and a crash lost a request; `XADD`/`XREADGROUP`/`XACK` gives a durable per-agent inbox with at-least-once delivery, and `XAUTOCLAIM` is the reconnect path (a worker acks only after its reply write succeeded).
- **Cross-process subscriptions are a Redis SET family** (`subs`, `subs:{pattern}`) that each instance writes through on `add_subscription`/`remove_subscription` and refreshes on an interval into `remote ∪ owned`, because `TeamTransport::subscribers` is synchronous: the local registry must answer from a snapshot, never from a network call, and a local write must never be dropped by a racing refresh.
- **Group routing is a registry plus a snapshot view**: `GroupRegistry` holds membership only, `RoutingTable` is an assembled read model over groups plus the transport's topic registry, and `broadcast_to_group` is a fan-out of ordinary P2P sends (so it reuses validation, deadlines, events and per-member failure isolation instead of adding a second delivery path).
- **Inbox filters live in the runtime dispatcher, not in a handler wrapper**: the transport already delivers to `RuntimeHandler`, so filtering there covers both backends, keeps the `MessageHandler` trait unchanged, and lets a rejection surface as a `Validation` error at the sender for P2P while pub/sub stays fire-and-forget.

## Routing Table

- `./src/CONTEXT.md` → module-level detail for the event bus, auto-subscription, Python literals, resource seam, member checkpoints, transport seam and `P2PAbilityManager`, plus the per-file map
- `./src/card.rs` → `TeamCard`, `EventDrivenTeamCard`, member list
- `./src/config.rs` → `TeamConfig` defaults, builders, timeout semantics
- `./src/member.rs` → `TeamAgent`, `AgentProvider`, `ToolHost`, bind hook
- `./src/envelope.rs` → `MessageEnvelope` and P2P/pub-sub classification
- `./src/subscription.rs` → topic-pattern registry (`*`/`?`), per-agent index
- `./src/message_bus.rs` → topics, FIFO workers, timeouts, fan-out, lifecycle, `MessageHandler`
- `./src/runtime.rs` → `TeamRuntime`, registration, session binding, P2P/public dispatch, pinned instances
- `./src/runtime/seams.rs` → resource-manager seam, member-checkpointer attachment, transport field, `start`/`stop` delegation
- `./src/session.rs` → team session construction, `SessionLease`, stream driver
- `./src/strategy.rs` → `TeamStrategy`, `StrategyContext`, `MemberOptions`
- `./src/team.rs` → `Team` trait, `BaseTeam` machinery and validation
- `./src/agent_tool.rs` → `AgentTool` (agent as a `Tool`), tool-result rendering
- `./src/events.rs` → `TeamEvent`, `TeamEventKind`, `TeamEventFilter`, `TeamEventBus`, Python callback names
- `./src/autosubscribe.rs` → card/tag/event-card subscription derivation, modes, overrides, apply to the runtime
- `./src/pyrepr.rs` → Python literal parsing and `str()`-style rendering (the `ast.literal_eval` fallback)
- `./src/resources.rs` → `TeamResourceMgr` trait and `LocalResourceMgr`
- `./src/member_checkpoint.rs` → per-member checkpoint identity over one shared checkpointer
- `./src/ability.rs` → `P2PAbilityManager`, `AbilityCall`, `AbilityResult`, `AbilityExecutor`
- `./src/handoff/` → handoff config/signal/tool/orchestrator/container/interrupt and `HandoffStrategy` (`handoff/CONTEXT.md`)
- `./src/hierarchical/` → msgbus supervisor team, supervisor factory, agents-as-tools team, `P2PAbilityManager` (`hierarchical/CONTEXT.md`)
- `./src/routing.rs` → `GroupRegistry`, `broadcast_targets`, `GroupBroadcast`/`GroupDelivery`, `RoutingTable`, `GROUP_METADATA_KEY`
- `./src/inbox.rs` → `InboxFilter`, `InboxFilterRegistry` (per-agent acceptance rules)
- `./src/communicable.rs` → `CommunicableAgent`, the runtime-bound send/publish/subscribe facade
- `./src/runtime/routing.rs` → the runtime's group/inbox forwarders, `broadcast_to_group`, `routing_table`
- `./src/transport/` → `TeamTransport` seam, `TransportKind`/`TransportConfig`, in-process and Redis backends (`redis/` split into config, streams commands, worker, registry, backend), `Pulsar` `Unsupported`
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests beside each module: card/config serde defaults and unknown-field tolerance; envelope P2P/pub-sub classification; wildcard matching and subscription indexing; bus duplicate-subscribe, cleanup/re-subscribe, timeout, fan-out failure isolation, start/stop idempotence; runtime registration/pinning/unregistration; team validation messages and `max_agents`; orchestrator route graph/cap/termination/first-wins; signal precedence; handoff tool card shape and never-raise; `AgentTool` card and dispatch; event filter combination and sequence stamping; subscription derivation rules; Python-literal parsing and rendering; resource-manager duplicates; member checkpoint key/id layout; transport construction and key naming; ability routing.
- `tests/team_card.rs` pins the wire contract: JSON field names, `version: "1.0.0"`, `agent_cards`, `subscriptions`, `TeamConfig` defaults 10/100/30.0.
- `tests/message_bus.rs` drives the bus directly with a recording handler: P2P ordering per sender, concurrent fan-out to sorted subscribers, missing-subscriber publish no-op, timeout error text, session cleanup, graceful shutdown.
- `tests/runtime.rs` covers registration, provider binding, P2P session construction from a bound team session, empty-response error, and P2P timeout override.
- `tests/team.rs` covers `add_agent`/`add_member`/`remove_agent`/`remove_member`, duplicate no-op, `max_agents`, queries, send/publish validation, start/stop idempotence, a custom strategy, and the unsupported-strategy error.
- `tests/handoff.rs` covers the full chain with scripted member agents: direct handoff, routed handoff, blocked hop completes, max-handoffs, timeout, interrupt forwarding, history resume, and streaming frames.
- `tests/hierarchical.rs` covers supervisor-over-bus single hop + single stream frame, sub-agent tool cards and ordered parallel dispatch with the limiter, agents-as-tools wiring (`ToolHost` mount + pinned instance), and the error chunk.
- `tests/streaming.rs` covers `team_stream` frame order, caller-owned vs standalone teardown, and producer-error-after-drain.
- `tests/agent_tool.rs` covers the tool card shape, dispatch through the runtime (`tool_call_id` child session), the dropped-runtime and agent-failure tool errors, the `invocation_id` fallback, the concurrency limiter, tool-result rendering, and the task events.
- `tests/events.rs` covers wire names and the Python parity constants, filtered synchronous delivery, unsubscribe, bounded history, serde/schema contracts, and concurrent emission from several tasks.
- `tests/autosubscribe.rs` pins the derivation rules (broadcast, directed topics, tag-carried capability topics, overrides), the sorted/deduped plan shape, and the three reconciliation modes against a real runtime registry.
- `tests/autosubscribe_team.rs` pins the `BaseTeam` wiring points: tags applied on `add_agent`/`add_member`, per-agent overrides, `Merge`/`Replace`/`Disabled`, removal cleanup, whole-plan application on `start`, and one end-to-end pub/sub delivery to an auto-subscribed member.
- `tests/pyrepr.rs` covers the two call sites: recovering a handoff payload from a `str()`-rendered tool message (`json.loads` first, `ast.literal_eval` second) and rendering a tool result for a model prompt.
- `tests/resources.rs` covers `LocalResourceMgr` and the `TeamResourceMgr` trait object: first-registration-wins, sorted id listings, registration-ordered agent ids, the revision counter, and `clear`.
- `tests/member_checkpoint.rs` is the headline identity scenario: two members of one team sharing a single `MemoryCheckpointer`, each saving different state, then a second team session object with the same id and fresh children restoring exactly its own state.
- `tests/transport.rs` covers the in-process backend over a recording handler (ordering, fan-out, subscribers, deadlines, introspection) plus backend selection (`Pulsar` `Unsupported`, Redis without settings is `Config`).
- `tests/transport_redis.rs` runs the Redis backend over a loopback fake RESP server (`TcpListener` on `127.0.0.1:0`) with a stream-aware state machine (entries, per-group cursors, pending lists, SET registry): cross-instance P2P round-trip, one delivery per group, socket-drop reconnect that claims and acks a pending entry, poison-entry ack, pub/sub fan-out to two transports, registry refresh convergence both ways, `-ERR` resilience, `stop` cleanup, `Cluster`/`Sentinel` `Unsupported`, timeout text, and an assertion that no `RPUSH`/`RPOP`/`LLEN` is ever issued.
- `tests/routing.rs` covers `GroupRegistry` semantics and validation, `broadcast_targets` sender exclusion, `broadcast_to_group` end-to-end (metadata stamp, skipped members, isolated failures, one group event, `NotFound`/`Validation`), and `RoutingTable` resolution plus serde round-trip.
- `tests/inbox.rs` covers every `InboxFilter` rule, registry default-accept, and runtime enforcement (P2P rejection at the sender, pub/sub drop, delivery restored after removal).
- `tests/communicable.rs` covers the unbound validation message, bind/rebind/unbind, and an in-process `send`/`publish`/`subscribe` round trip.
- `tests/ability.rs` covers agent-tool routing over P2P, local executor routing, input-order results, the parallel cap, the error results (`"P2P dispatch to '<name>' failed: ..."`, missing executor), and the empty batch.
- `tests/runtime_integration.rs` covers the runtime seams together: event emission, resource-manager fallback, member-scoped checkpoints across a resume, the outbound transport, and `AgentTool` task events.
- `tests/handoff_events.rs` covers one `handoff_started` per hop, `handoff_completed` with `interrupted`, the `handoff` error event, and silence when the runtime was dropped.
- `tests/api_surface.rs` pins the public contract at compile time: every crate-root re-export, inherent constructor and free-function signature, trait-object coercion (`Team`, `TeamStrategy`, `MessageHandler`, `ToolHost`, `Tool`), `Team` method reachability, the `Card` impl, serde round-trips and `schemars` property names, and the `TeamError` → `ErrorKind::External` mapping.
- All tests are offline and deterministic: fake agents are local closures, models come from `openjiuwen-test-support::ScriptedModel`, timeouts use short durations only.

## Deferred Python Behavior

- `Pulsar` transport — `build_transport` returns `Error::Unsupported` naming Pulsar and the missing protobuf-framed client; no Pulsar client exists in the workspace and none may be added offline.
- Redis cluster/Sentinel topologies — `RedisTopology::{Cluster, Sentinel}` is a typed `Error::Unsupported` (`build_transport` and `start`), because the workspace has no cluster/Sentinel-capable RESP client; `RedisTransportConfig::topology` defaults to `Single`.
- Redis delivery is polled (`XREADGROUP` without `BLOCK`): an idle inbox costs one round trip per `poll_interval_ms`, and a reply costs one `XREAD` per `poll_interval_ms` until it arrives.
- Reclaiming pending entries needs Redis ≥ 6.2 (`XAUTOCLAIM`); on an older server a request left pending by a crashed consumer is never retried (no `XPENDING`/`XCLAIM` fallback).
- No Streams housekeeping (`XTRIM`/`MAXLEN`): acked entries stay in their stream until an operator trims, so a long-lived team grows its inbox keys.
- Cross-process routing is eventually consistent: `publish` fans out from the instance's local registry, refreshed from the Redis `subs` sets every `registry_interval_ms`, so a publish immediately after a remote `add_subscription` can miss that subscriber for one interval.
- A crashed process leaves its registry entries behind (harmless: pub/sub `XADD`s land in a stream nobody consumes); only a clean `stop` removes them.
- Python's `drunner` distributed factories, reply-topic routing and `RemoteAgent` — `TeamTransport` is the seam a distributed backend implements.
- The team event bus is in-process: events are delivered to local listeners and recorded in local history, not published over a transport.
- Python's process-global `Runner.resource_mgr` lookup — the seam is the `TeamResourceMgr` trait the runner implements and injects; this crate never reads a global registry, and an unconfigured runtime resolves from its own instance-scoped stores.
- Python has no group, broadcast or per-agent inbox-filter layer in `core.multi_agent`; `routing` and `inbox` are additive Rust extensions modelled on the `agent_teams` broadcast contract (sender excluded, sorted recipients, per-member results), not ports.
- Python has no `agent_reasoner` behaviour to port: the reference file is empty and the area belongs to `openjiuwen-agent` (deferred there).
- `TeamConfig` re-wiring after `configure` (Python only applies it at default-runtime construction) — the config snapshot at construction is authoritative.
- `SupervisorAgent`'s exact ReAct config surface (prompt template names, context engine wiring) — only card, model, system prompt, max iterations, parallel cap.
- No round-robin/selector/scheduled strategies exist in Python's `core.multi_agent`; custom strategies plug in through `TeamStrategy`.
- `P2PAbilityManager` subclassing — replaced by a shared-semaphore dispatcher with an injectable `AbilityExecutor`.

## Known Issues

- `TeamRuntime::subscribe`/`unsubscribe` (`src/runtime.rs`) and card-driven auto-subscription (`src/autosubscribe.rs`, `src/team.rs`) write through `self.bus`, while `publish`/`list_subscriptions` go through `self.transport()`; with a `RedisTransport` installed the transport's own registry is never populated, so `Team::subscribe` has no effect and `publish` silently fans out to nobody.
- `MessageBus::send`/`publish` await a bounded per-topic channel *before* their deadline is applied (`src/message_bus.rs`), so a full FIFO can block a call past its `timeout`/`process_timeout` with no bound.
- `MemberCheckpointer::restore` (`src/member_checkpoint.rs`) lists every checkpoint of the team session and restores before every member invocation, and each save appends a new checkpoint, so the per-message cost grows with the number of prior invocations.

## Notes for Agents

- `team_id` is `card.name`; topics, sessions, and registries key on `card.id` (Python parity — keep both).
- `message_timeout` (30 s) is the bus process timeout and the handoff chain deadline; `p2p_timeout` (1800 s) is the supervisor dispatch deadline. Keep the three deadlines distinct.
- A duplicate `add_agent` logs a warning and returns `Ok(())`; `max_agents` is checked only for new members.
- `HandoffRequest::history` is appended before interrupt detection, so an interrupted hop is recorded then dropped on resume.
- `ContainerAgent::invoke` returns `{}` on every path; results reach consumers only through the shared team stream.
- `flush_team_session` = `end_stream()` then `commit()`; a commit failure is logged, never raised.
- Additive public helpers beyond the list above: `HandoffSnapshot`, `TerminationCondition` and `HandoffRegistry` (keyed by team session id, the rendezvous map `ContainerAgent` and `HandoffStrategy` share), `extract_handoff_signal`, `extract_handoff_signal_from_observations`, `extract_handoff_signal_from_session`, `team_stream` and `topic_matches` (also re-exported at the crate root, consumed by the strategy modules), `MessageBus::{effective_team_id, p2p_topic, pubsub_topic, agent_topics}`, `TeamRuntime::message_bus`, `SubscriptionManager::{patterns, agents_for_exact}`, `MessageEnvelope::with_metadata_map`, `AgentTool::agent_id`, `TeamError::{empty_response, execution}` plus `From<TeamError> for Error`, and `with_*` builders on `RuntimeConfig`/`MessageBusConfig`.
- `TeamRuntime::new` builds the bus with `Arc::new_cyclic` (handler holds only `Weak<TeamRuntime>`, so there is no `Arc` cycle) and stamps `RuntimeConfig::team_id` onto the bus config, so topic names never fall back to `default` for a team with its own id.
- `SessionLease::acquire` is `async` to mirror Python's acquire point even though the current body only takes short sync locks (clippy `unused_async` is allowed on it).
- Every public data type derives `schemars::JsonSchema`; strategy configs with non-schema fields (for example `HandoffConfig::termination_condition`) need `#[schemars(skip)]` when they land.
- `pin_agent` on an unregistered id inserts a registry entry whose card id and name are the agent id, so later lookups and error messages stay consistent.
- `dispatch_pubsub` reports an unregistered subscriber with the same `not_found` wording as P2P (`"Recipient '<id>' not registered in runtime"`).
- A member invoked over P2P receives a child `AgentSession` whose `parent_id()` is the bound team session id, so a container agent resolves the team session with `runtime.team_session(session.parent_id()?)` rather than from the envelope (the `Agent` seam has no envelope parameter).
- Re-entrancy: a member runs inline in its topic's FIFO worker, so a handler must never await a `publish`/`send` on the same session's transport topic. The handoff container therefore spawns its next-hop publish (failures are logged; the chain still ends via `HandoffOrchestrator::wait`'s deadline) — the first hop publish is awaited normally from the caller's task.
- Bus transport topics are created lazily by `send`/`publish`, so `TeamRuntime::subscribe` only updates the logical pattern registry; a strategy registers every endpoint agent before sending to it, because `send` validates registration up front.
- `TeamEventBus::emit` is synchronous and runs listeners inline on the emitting task: keep listeners short and never re-enter a runtime method that emits (a listener that calls `send` would emit from inside an emission). Emission sites take no runtime lock across the call, but a listener that locks the registry while a runtime method holds it would deadlock, so listeners should only inspect the event.
- `emit` stamps `sequence` itself; construct events with the `TeamEventKind` constructors and let the bus assign ordering. History is bounded (default 256, `with_capacity(0)` disables it) and ordered oldest-first.
- `autosubscribe::apply` in `Merge` mode is additive and tolerant of re-application; `Replace` clears the agent's current patterns first, so it is the mode to use when a card changed. `Disabled` is the switch that reproduces the pre-auto-subscription behaviour.
- Overrides are keyed by `agent_id` (card id), not by name; a `TeamCard` with one member listed twice derives the same plan for both entries because the map is keyed by id.
- `resolve_agent_card`/`resolve_agent_provider`/`resolve_team_session` are local-first (a manager entry can never shadow an instance registered or bound on the runtime); `resolve_tool`/`resolve_tool_by_name`/`resolve_model` are manager-only and return `None` when no `TeamResourceMgr` is configured.
- `set_member_checkpointer` is optional; without it `invoke_agent` neither saves nor restores. `MemberCheckpointer::save` needs the member session to have a team parent (`parent_id()`) and a resolvable agent id (`metadata().entity_id`, then `agent_id`), which the runtime's child sessions always provide.
- `set_transport` requires a transport whose `team_id()` equals the runtime's; the default is the in-process transport wrapping the runtime's own `MessageBus`, so `MessageBus` and transport are one object until a caller swaps it in.
- `RedisTransport` keys embed the team id (`{prefix}:{team}:...`) but not the session id; a shared prefix across teams is safe, a shared prefix across *deployments* is not.
- `pyrepr` deviates from CPython only in cosmetics: strings always render single-quoted and floats without exponent padding; it parses literals, it never evaluates names or calls.
- Redis delivery is at-least-once: a worker acks a stream entry only after its reply `XADD` succeeded, so a reply-write failure re-delivers the request. There is no dedupe; a member invoked by such a retry must tolerate it.
- A Redis worker creates its group with `XGROUP CREATE ... MKSTREAM` before its first `XREADGROUP`; a `BUSYGROUP` error is expected and ignored. `XAUTOCLAIM` (Redis ≥ 6.2) runs on worker start and after a read-error recovery, so pending entries of a dead consumer are redelivered.
- `RedisTransport::stop` clears this instance's registry entries and `start` re-publishes them: a caller swapping transports must `start` after `stop`, or cross-process routes stay invisible.
- `RedisTransport::refresh_subscriptions` (public, and run every `registry_interval_ms` while started) is the deterministic hook tests use; a `publish` immediately after a remote `add_subscription` can miss it until a refresh elapsed.
- Group broadcast excludes the sender unless `broadcast_targets(.., include_sender = true)` is used, stamps `metadata["group"]`, and reports unregistered members in `GroupBroadcast::skipped`; a member failure stays inside `deliveries` and never aborts the batch.
- `RoutingTable::route` resolves a group id first, then matches topic patterns through `subscription::topic_matches`; it is a snapshot of one moment, not a live view.
- An inbox-filter rejection is a `Validation` error at the P2P sender and a logged drop for pub/sub; `dispatch_p2p` emits no `message_received` for a rejected envelope, so a listener sees only deliveries a member actually received.
- `src/runtime.rs` sits at the edge of the size baseline (~980 lines); new runtime behaviour belongs in a `src/runtime/<area>.rs` submodule (as `seams.rs` and `routing.rs` do), not in the core file.
