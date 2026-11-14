# openjiuwen-multiagent — src

## Intent

Module-level reference for the `openjiuwen-multiagent` subsystems the crate root summarises: typed team events, card-driven auto-subscription, Python literals, the resource-manager seam, per-member checkpoints, the transport seam and `P2PAbilityManager`.
Each section documents one module's contract and the invariants a caller or a refactorer must keep; the crate `CONTEXT.md` keeps the one-line summary and routes here.

## Ability (`ability.rs`)

`P2PAbilityManager` (Python `teams/hierarchical_msgbus/p2p_ability_manager.py`) is reachable as `openjiuwen_multiagent::ability::P2PAbilityManager` and as the Python-shaped `hierarchical::P2PAbilityManager` (which also re-exports `AbilityCall`, `AbilityResult`, `AbilityRoute`).

```rust
pub const DEFAULT_MAX_PARALLEL_SUB_AGENTS: usize = 10;
pub struct AbilityCall { pub call_id: String /* uuid v4 */, pub name: String, pub arguments: Value }
pub struct AbilityResult { pub call_id: String, pub name: String, pub route: AbilityRoute,
                           pub output: Value, pub error: Option<String> }
pub enum AbilityRoute { P2p, Local, Error }
#[async_trait] pub trait AbilityExecutor: Send + Sync {
    async fn execute(&self, call: &AbilityCall, session: Option<Arc<dyn Session>>) -> Result<Value>;
}
pub struct P2PAbilityManager { .. }
impl P2PAbilityManager {
    pub fn new(runtime: Arc<TeamRuntime>, supervisor_id: impl Into<String>) -> Arc<Self>;
    pub fn with_max_parallel(self: Arc<Self>, max_parallel: usize) -> Arc<Self>;
    pub fn with_executor(self: Arc<Self>, executor: Arc<dyn AbilityExecutor>) -> Arc<Self>;
    pub fn max_parallel(&self) -> usize;
    pub fn supervisor_id(&self) -> &str;
    pub fn runtime(&self) -> Arc<TeamRuntime>;
    pub fn register_agent_tool(&self, card: &AgentCard);
    pub fn agent_tool_ids(&self) -> Vec<String>;                 // sorted
    pub fn is_agent_tool(&self, name: &str) -> bool;
    pub fn agent_card(&self, name: &str) -> Option<AgentCard>;
    pub async fn execute(&self, calls: Vec<AbilityCall>, session: Option<Arc<dyn Session>>)
        -> Result<Vec<AbilityResult>>;
}
```
A call whose `name` is a registered agent tool dispatches over P2P through the runtime (bounded by one shared `Semaphore(max(1, max_parallel))`), anything else goes to the configured `AbilityExecutor` (`AbilityRoute::Local`); a batch is dispatched concurrently but results come back in input order, an empty batch yields an empty vector, and a failed call becomes `AbilityRoute::Error` with the message (`"P2P dispatch to '<name>' failed: <e>"`, or `"no ability executor configured"` when no executor is set) without aborting the batch.

## Team events (`events.rs`)

```rust
pub const TEAM_EVENT_SCOPE: &str = "agent_team";
pub const AGENT_P2P_RECEIVED_EVENT: &str = "_framework:agent_p2p_received";     // Python name, kept verbatim
pub const AGENT_PUBSUB_RECEIVED_EVENT: &str = "_framework:agent_pubsub_received";
pub enum TeamEventKind { MemberJoined, MemberLeft, MessageSent, MessageReceived,
                         HandoffStarted, HandoffCompleted, TaskAssigned, TaskCompleted, Error }
impl TeamEventKind { pub const ALL: [Self; 9]; pub const fn as_str(&self) -> &'static str;
                     pub fn event_name(&self) -> String;                 // "agent_team:member_joined"
                     pub const fn python_event_name(&self) -> Option<&'static str>; }
pub struct TeamEvent { pub event_id: String, pub sequence: u64, pub kind: TeamEventKind,
                       pub team_id: String, pub session_id: Option<String>,
                       pub agent_id: Option<String>, pub peer_id: Option<String>,
                       pub topic_id: Option<String>, pub message_id: Option<String>,
                       pub detail: BTreeMap<String, Value> }
impl TeamEvent { /* constructors member_joined/member_left/message_sent/message_received/
                    pubsub_received/handoff_started/handoff_completed/task_assigned/task_completed/error,
                    builders with_session/with_agent/with_peer/with_topic/with_message_id/with_detail,
                    python_event_name() -> Option<&'static str> */ }
pub struct TeamEventFilter { .. }
impl TeamEventFilter { pub fn new() -> Self; pub fn all() -> Self;   // new matches nothing, all matches everything
                       pub fn kind(self, kind: TeamEventKind) -> Self;
                       pub fn kinds(self, kinds: impl IntoIterator<Item = TeamEventKind>) -> Self;
                       pub fn team(self, team_id: impl Into<String>) -> Self;
                       pub fn session(self, session_id: impl Into<String>) -> Self;
                       pub fn agent(self, agent_id: impl Into<String>) -> Self;
                       pub fn matches(&self, event: &TeamEvent) -> bool; }
pub type TeamEventListener = Arc<dyn Fn(&TeamEvent) + Send + Sync>;
pub struct TeamEventBus { .. }                                        // non-serializable, process-local
impl TeamEventBus {
    pub fn new() -> Arc<Self>;                                        // history capacity 256
    pub fn with_capacity(history_capacity: usize) -> Arc<Self>;       // 0 disables history
    pub fn subscribe(&self, filter: TeamEventFilter, listener: TeamEventListener) -> u64;
    pub fn unsubscribe(&self, subscription_id: u64) -> bool;
    pub fn subscriber_count(&self) -> usize;
    pub fn emit(&self, event: TeamEvent) -> TeamEvent;                // stamps sequence; never fails
    pub fn history(&self) -> Vec<TeamEvent>;                          // oldest first, bounded
    pub fn history_of(&self, kind: TeamEventKind) -> Vec<TeamEvent>;
    pub fn last_sequence(&self) -> u64;
}
```
`emit` is synchronous and infallible: it stamps `sequence` from an `AtomicU64` starting at 1, delivers to matching listeners in subscription-id order, then appends to the bounded history. A listener that panics is not caught — listeners are expected to be pure observers; this is why every in-crate emission site is on a non-error path or inside an error mapping. Filters combine with AND; `TeamEventFilter::new()` matches nothing so a listener must opt in (a bare `subscribe(TeamEventFilter::all(), ..)` is the Python callback equivalent). `python_event_name` maps `MessageReceived` → `AGENT_P2P_RECEIVED_EVENT` and `pubsub_received` → `AGENT_PUBSUB_RECEIVED_EVENT` and returns `None` for the other seven kinds, so a callback adapter can register under the Python scope name.

## Card-driven auto-subscription (`autosubscribe.rs`)

```rust
pub const SUBSCRIBE_TAG_PREFIX: &str = "subscribe:";
pub enum AutoSubscriptionMode { Merge, Replace, Disabled }            // Default = Merge
pub struct AutoSubscriptionPlan { /* BTreeMap<String, Vec<String>> agent -> sorted patterns */ }
impl AutoSubscriptionPlan { pub fn new() -> Self; pub fn is_empty(&self) -> bool; pub fn len(&self) -> usize;
                            pub fn get(&self, agent_id: &str) -> Option<&[String]>; pub fn agents(&self) -> Vec<String>;
                            pub fn patterns(&self) -> &BTreeMap<String, Vec<String>>;
                            pub fn insert(&mut self, agent_id: impl Into<String>, pattern: impl Into<String>);
                            pub fn entry(&mut self, agent_id: &str) -> Vec<String>;   // owned slot for mutation
                            pub fn merge(&mut self, other: &Self); pub fn subscription_count(&self) -> usize; }
pub fn derive(card: &TeamCard) -> AutoSubscriptionPlan;
pub fn derive_with_overrides(card: &TeamCard, overrides: &BTreeMap<String, Vec<String>>) -> AutoSubscriptionPlan;
pub fn derive_from_event_card(card: &EventDrivenTeamCard) -> AutoSubscriptionPlan;
pub fn derived_for_member(card: &TeamCard, agent_id: &str) -> Vec<String>;
pub fn apply(runtime: &Arc<TeamRuntime>, plan: &AutoSubscriptionPlan, mode: AutoSubscriptionMode) -> usize;
pub fn apply_agent(runtime: &Arc<TeamRuntime>, agent_id: &str, patterns: &[String],
                   mode: AutoSubscriptionMode) -> usize;
```
Derivation rules (deterministic, sorted, deduplicated): every member subscribes to the team broadcast topic `card.topic` and to `"{topic}.*"`; every member additionally subscribes to its own directed topic `"{topic}.{member.name}"`; `TeamCard.tags` entries of the form `subscribe:<pattern>` are applied to every member, `subscribe:<agent_id>:<pattern>` only to the named member. `derive_from_event_card` merges `EventDrivenTeamCard.subscriptions` on top (explicit per-agent lists win for that agent). `derive_with_overrides` replaces an agent's derived list with the override when one exists — that is the explicit override path used by `BaseTeam::with_subscription_overrides`. `Replace` mode unsubscribes the agent's current patterns before applying the plan, `Merge` only adds, and `Disabled` applies nothing (returns 0). Patterns are logical (`TeamRuntime::subscribe`), not transport topics; the bus creates transport topics lazily.

## Python literals (`pyrepr.rs`)

```rust
pub const MAX_LITERAL_DEPTH: usize = 64;
pub fn parse_python_literal(text: &str) -> Option<Value>;   // None when the text is not a Python literal
pub fn literal_or_json(text: &str) -> Option<Value>;        // serde_json first, then parse_python_literal
pub fn render_python_value(value: &Value) -> String;        // Python str() style
```
Handles dicts, lists, tuples (rendered as lists), `True`/`False`/`None`, single- and double-quoted strings with escapes, ints, floats and nesting up to `MAX_LITERAL_DEPTH`; function calls, names, operators and comments are rejected (no `ast` evaluation in Rust). `literal_or_json` is what handoff signal extraction and the tool-result renderer use, so a member returning `{'__handoff_to__': 'b'}` still routes. Documented CPython deviations: string quotes are always single, and floats render without exponent padding (`1e-5`, not `1e-05`).

## Resource manager seam (`resources.rs`)

```rust
pub trait TeamResourceMgr: Send + Sync + fmt::Debug {
    fn agent_card(&self, agent_id: &str) -> Option<AgentCard> { None }
    fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider> { None }
    fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> bool { false }
    fn tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> { None }
    fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> { None }
    fn model(&self, model_id: &str) -> Option<Arc<dyn Model>> { None }
    fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> { None }
    fn register_team_session(&self, session: Arc<AgentTeamSession>) -> bool { false }
    fn remove_team_session(&self, session_id: &str) -> bool { false }
}
pub struct LocalResourceMgr { .. }
impl LocalResourceMgr {
    pub fn new() -> Arc<Self>;
    pub fn register_tool(&self, tool: Arc<dyn Tool>) -> bool;         // keyed by the tool card id
    pub fn register_tool_with(&self, tool_id: impl Into<String>, tool: Arc<dyn Tool>) -> bool;
    pub fn tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>>;
    pub fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>>;
    pub fn tool_card(&self, tool_id: &str) -> Option<ToolCard>;
    pub fn tool_ids(&self) -> Vec<String>;  pub fn remove_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>>;
    pub fn tool_count(&self) -> usize;
    pub fn register_model(&self, model_id: impl Into<String>, model: Arc<dyn Model>) -> bool;
    pub fn model(&self, model_id: &str) -> Option<Arc<dyn Model>>;
    pub fn model_ids(&self) -> Vec<String>;  pub fn remove_model(&self, model_id: &str) -> Option<Arc<dyn Model>>;
    pub fn model_count(&self) -> usize;
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> bool;
    pub fn agent_card(&self, agent_id: &str) -> Option<AgentCard>;
    pub fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider>;
    pub fn agent_ids(&self) -> Vec<String>;  pub fn remove_agent(&self, agent_id: &str) -> Option<AgentCard>;
    pub fn agent_count(&self) -> usize;
    pub fn register_team_session(&self, session: Arc<AgentTeamSession>) -> bool;
    pub fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>>;
    pub fn remove_team_session(&self, session_id: &str) -> bool;
    pub fn session_ids(&self) -> Vec<String>;  pub fn session_count(&self) -> usize;
    pub fn revision(&self) -> u64;           // bumps on every successful mutation
    pub fn clear(&self) -> usize;            // returns the number of removed entries
}
```
`LocalResourceMgr` is first-registration-wins (`register_*` returns `false` for a duplicate id and leaves the existing entry). Every trait method has a `None`/`false` default, so the runner's manager implements only what it owns and a team needs no manager at all. `TeamRuntime::new_with_resource_mgr` / `set_resource_mgr` attach one; `resolve_*` prefers the runtime's own instance-scoped registry and falls back to the manager, so the existing in-process path is unchanged.

## Member checkpoints (`member_checkpoint.rs`)

```rust
pub const MEMBER_CHECKPOINT_NAMESPACE: CheckpointNamespace = CheckpointNamespace::Agent;
pub fn member_checkpoint_key(team_session_id: &str, agent_id: &str) -> String;        // "{team_session_id}:agent:{agent_id}"
pub fn member_checkpoint_id(team_session_id: &str, agent_id: &str, sequence: u64) -> String;  // "...:{sequence:06}"
pub struct MemberCheckpointer { .. }
impl MemberCheckpointer {
    pub fn new(checkpointer: Arc<dyn Checkpointer>) -> Self;         // sequence starts at 0
    pub fn checkpointer(&self) -> Arc<dyn Checkpointer>;
    pub async fn save(&self, member: &dyn Session) -> Result<Option<String>>;
    pub async fn save_in(&self, team_session_id: &str, agent_id: &str, member: &dyn Session)
        -> Result<String>;
    pub async fn load(&self, team_session_id: &str, agent_id: &str) -> Result<Option<Checkpoint>>;
    pub async fn restore(&self, member: &dyn Session) -> Result<bool>;   // false when nothing was saved
    pub async fn save_all(&self, team_session_id: &str, members: &[Arc<dyn Session>]) -> Result<Vec<String>>;
    pub async fn restore_all(&self, team_session_id: &str, members: &[Arc<dyn Session>]) -> Result<usize>;
    pub fn saved_members(&self, team_session_id: &str) -> Result<Vec<String>>;
}
```
Identity: the team session id comes from `member.parent_id()` and the agent id from `member.metadata().entity_id` (falling back to the `agent_id` metadata entry). A member session with no team parent yields `Ok(None)` from `save` (nothing to scope it to); a member with no resolvable agent id is `Error::validation`, never a silent fallback to the session id. The persisted `Checkpoint` carries `session_id = team_session_id`, `namespace = agent`, `entity_id = agent_id`, plus `agent_session_id` in metadata for diagnostics — so two members of one team sharing one `MemoryCheckpointer` restore independently. `save_all`/`restore_all` iterate members in slice order and report how many were restored.

## Transport seam (`transport/`)

```rust
pub const DEFAULT_KEY_PREFIX: &str = "openjiuwen:team";
pub enum TransportKind { InProcess /*default*/, Redis, Pulsar }
impl TransportKind { pub const fn as_str(self) -> &'static str; }
pub struct RedisTransportConfig { pub addr: String /*"127.0.0.1:6379"*/, pub username: Option<String>,
                                  pub password: Option<SecretString>, pub db: Option<u32>,
                                  pub poll_interval_ms: u64 /*10*/, pub consumer_group: Option<String>,
                                  pub claim_idle_ms: u64 /*30_000*/, pub registry_interval_ms: u64 /*250*/,
                                  pub topology: RedisTopology /*Single*/ }
pub struct TransportConfig { pub kind: TransportKind, pub key_prefix: String /*"openjiuwen:team"*/,
                             pub redis: Option<RedisTransportConfig> }
impl TransportConfig { pub fn new() -> Self; pub const fn with_kind(self, kind: TransportKind) -> Self;
                       pub fn with_key_prefix(self, key_prefix: impl Into<String>) -> Self;
                       pub fn with_redis(self, redis: RedisTransportConfig) -> Self; }
#[async_trait] pub trait TeamTransport: Send + Sync + fmt::Debug {
    fn kind(&self) -> TransportKind;
    fn team_id(&self) -> &str;
    async fn start(&self) -> Result<()>;                          // idempotent
    async fn stop(&self) -> Result<()>;                           // idempotent, joins workers
    async fn send(&self, envelope: MessageEnvelope, timeout: Option<Duration>) -> Result<Value>;
    async fn publish(&self, envelope: MessageEnvelope) -> Result<()>;
    fn add_subscription(&self, agent_id: &str, topic: &str);
    fn remove_subscription(&self, agent_id: &str, topic: &str);
    fn subscribers(&self, topic_id: &str) -> Vec<String>;
    // introspection, defaulted to an empty registry and overridden by both backends
    fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>>;
    fn agent_subscriptions(&self, agent_id: &str) -> Vec<String>;
    fn subscription_count(&self) -> usize;
}
pub fn build_transport(config: &TransportConfig, team_id: impl Into<String>, bus_config: MessageBusConfig,
                       handler: Arc<dyn MessageHandler>) -> Result<Arc<dyn TeamTransport>>;
pub struct InProcessTransport { .. }
impl InProcessTransport { pub fn new(team_id: impl Into<String>, bus_config: MessageBusConfig,
                                     handler: Arc<dyn MessageHandler>) -> Self;
                          pub fn from_bus(bus: Arc<MessageBus>) -> Self;
                          pub fn bus(&self) -> Arc<MessageBus>; }
pub struct RedisTransport { .. }
impl RedisTransport { pub fn new(config: RedisTransportConfig, team_id: impl Into<String>,
                                 key_prefix: impl Into<String>, handler: Arc<dyn MessageHandler>) -> Arc<Self>;
                      pub fn config(&self) -> &RedisTransportConfig;
                      pub fn key_prefix(&self) -> &str;
                      pub fn p2p_queue(&self, agent_id: &str) -> String;
                      pub fn reply_queue(&self, message_id: &str) -> String;
                      pub fn pubsub_queue(&self, subscriber: &str) -> String;
                      pub fn is_started(&self) -> bool;
                      pub fn watch_agent(&self, agent_id: &str) -> Result<()>; }   // starts the agent's two inbox pollers
```
`build_transport` returns `InProcessTransport` for `InProcess`, `RedisTransport` for `Redis` (`Error::Config` when `TransportConfig::redis` is unset), and `Error::Unsupported` for `Pulsar` with a message naming Pulsar as not implemented. The Redis backend speaks RESP2 through `openjiuwen_foundation::store::external::redis` (`RedisConnection`), so no vendor SDK is needed: P2P messages go to `{prefix}:{team}:p2p:{recipient}` as a JSON list push, each request carries its `reply_queue` (`{prefix}:{team}:reply:{message_id}`) for the oneshot correlation, and pub/sub fan-out pushes one JSON envelope per subscriber onto `{prefix}:{team}:pubsub:{subscriber}`. Dispatch is poll-based (`poll_interval_ms`), keys never collide with the message-bus in-process topic names, and a `send` deadline produces `"Message from '<sender>' to '<recipient>' timed out after <t>s"` (`Error::Timeout`).
`TeamRuntime` owns one `Arc<dyn TeamTransport>` (in-process by default) and `send`/`publish`/`subscribe`/`start`/`stop` delegate to it; `set_transport` rejects a transport whose `team_id` differs from the runtime's (`Error::Validation`).

## Constraints

- Everything here follows the crate's constraints: async-first tokio, `openjiuwen_common::error` results, serde-serializable public data, no `unsafe`, `tracing` logging, files around ≤1000 lines.
- No module in this node depends upward: `events`, `autosubscribe`, `pyrepr`, `resources`, `member_checkpoint`, `transport` and `ability` all sit on `common`/`foundation`/`session`/`agent` plus this crate's own `runtime`.
- Process-local types (event buses, transports, resource managers, checkpoint wrappers, ability managers) are not serializable; only the data they carry is.

## Routing Table

- `./events.rs` → `TeamEvent`, `TeamEventKind`, `TeamEventFilter`, `TeamEventBus`, the Python callback constants
- `./autosubscribe.rs` → card/tag/event-card subscription derivation, reconciliation modes, overrides
- `./pyrepr.rs` → Python literal parsing and `str()`-style rendering
- `./resources.rs` → `TeamResourceMgr` trait, `LocalResourceMgr`
- `./member_checkpoint.rs` → per-member checkpoint identity over one shared checkpointer
- `./transport/mod.rs` → `TeamTransport`, `TransportKind`, `TransportConfig`, `build_transport`
- `./transport/inproc.rs` → `InProcessTransport` over the message bus
- `./transport/redis.rs` → `RedisTransport` (RESP2 lists, pollers, `watch_agent`)
- `./ability.rs` → `P2PAbilityManager`, `AbilityCall`, `AbilityResult`, `AbilityExecutor`
- `./runtime/` → the runtime these seams hang off (`seams.rs` attaches them)
- `./handoff/` → handoff strategy (`handoff/CONTEXT.md`)
- `./hierarchical/` → hierarchical strategies; re-exports `P2PAbilityManager` under its Python path (`hierarchical/CONTEXT.md`)
