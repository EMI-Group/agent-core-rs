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
pub fn build_transport(config: &TransportConfig, team_id: &str, bus_config: MessageBusConfig,
                       handler: Arc<dyn MessageHandler>) -> Result<Arc<dyn TeamTransport>>;
pub struct InProcessTransport { .. }
impl InProcessTransport { pub fn new(team_id: impl Into<String>, bus_config: MessageBusConfig,
                                     handler: Arc<dyn MessageHandler>) -> Self;
                          pub fn from_bus(bus: Arc<MessageBus>) -> Self;
                          pub fn bus(&self) -> Arc<MessageBus>; }
pub struct RedisTransport { .. }
impl RedisTransport { pub fn new(config: RedisTransportConfig, team_id: impl Into<String>,
                                 key_prefix: impl Into<String>, handler: Arc<dyn MessageHandler>) -> Arc<Self>;
                      pub fn try_new(config: RedisTransportConfig, team_id: impl Into<String>,
                                     key_prefix: impl Into<String>, handler: Arc<dyn MessageHandler>)
                          -> Result<Arc<Self>>;                       // rejects a non-single topology
                      pub fn config(&self) -> &RedisTransportConfig;
                      pub fn key_prefix(&self) -> &str;
                      pub fn consumer_name(&self) -> &str;                // uuid v4, one per instance
                      pub fn p2p_consumer_group(&self) -> String;
                      pub fn pubsub_consumer_group(&self) -> String;
                      pub fn p2p_queue(&self, agent_id: &str) -> String;
                      pub fn reply_queue(&self, message_id: &str) -> String;
                      pub fn pubsub_queue(&self, subscriber: &str) -> String;
                      pub fn is_started(&self) -> bool;
                      pub fn watch_agent(&self, agent_id: &str) -> Result<()>;   // starts the agent's two inbox pollers
                      pub async fn refresh_subscriptions(&self) -> Result<()>; }  // remote ∪ owned, on demand
```
`build_transport` selects the backend: `InProcess` → `InProcessTransport`, `Redis` → `RedisTransport::try_new` (`Error::Config` when `TransportConfig::redis` is unset), `Pulsar` → `Error::Unsupported` with a message naming Pulsar and the missing protobuf-framed client, and a Redis topology of `Cluster`/`Sentinel` → `Error::Unsupported` from both `build_transport` and `start` (the workspace has no cluster- or Sentinel-aware RESP client). The Redis backend speaks RESP2 through `openjiuwen_foundation::store::external::redis` (`RedisConnection`), so no vendor SDK is needed. Three delivery key families: `{prefix}:{team}:p2p:{recipient}` (P2P inbox stream; `send` appends `envelope ... reply <reply-key>` with `XADD`), `{prefix}:{team}:reply:{message_id}` (the reply stream of one request; the recipient's worker `XADD`s one `frame` entry — `{"value": <json>}` or `{"error": "<text>"}` — and the waiting `send` `XREAD`s it, then `DEL`s the key so a later request cannot read a stale frame), and `{prefix}:{team}:pubsub:{subscriber}` (one `XADD` copy per sorted subscriber). Two registry key families mirror subscriptions across processes: `{prefix}:{team}:subs` (every registered pattern) and `{prefix}:{team}:subs:{pattern}` (the agent set of that pattern).
Each inbox kind has one consumer group — `{prefix}:{team}:p2p:cg` and `{prefix}:{team}:pubsub:cg`, overridable via `RedisTransportConfig::consumer_group` — created idempotently with `XGROUP CREATE ... 0 MKSTREAM` (a `BUSYGROUP` reply is expected and ignored). Every transport instance joins with its own `consumer_name()` (uuid v4), so processes may start in any order and the same agent may be watched by several of them without double delivery. Workers poll their stream (`XREADGROUP ... >`, no `BLOCK`), dispatch through the `MessageHandler`, and `XACK` only after the reply `XADD` succeeded, so delivery is at-least-once (a reply-write failure re-delivers the request; there is no dedupe). A claim pass (`XAUTOCLAIM` with `claim_idle_ms`, Redis ≥ 6.2) runs on worker start and after every read-error recovery, so entries a crashed or reconnected consumer left pending are re-dispatched. Only agents registered through `watch_agent` get inbox workers; `start` spawns the ones registered while the transport was stopped.
Subscriptions are write-through: `add_subscription`/`remove_subscription` update the local `SubscriptionManager` immediately and mirror the pair into the Redis registry on a detached best-effort task (`SADD`, then `SREM`/`SCARD` and an index `SREM` when a pattern set empties), so the synchronous calls never wait for the network. A refresher task started by `start` re-reads the registry every `registry_interval_ms` and replaces the local view with `remote ∪ owned` (owned = pairs this instance added itself), so a process publishes to subscribers registered elsewhere while a not-yet-flushed local pair can never be dropped; `refresh_subscriptions()` runs the same refresh on demand. A clean `stop` removes this instance's owned entries; a crashed process may leave some behind, which is harmless (a publisher `XADD`s into a stream no worker consumes).
Reads are polled (`poll_interval_ms`, no `BLOCK`), so an idle inbox costs one round trip per interval and a reply one `XREAD` per interval until it arrives. Keys never collide with the message-bus in-process topic names, and a `send` deadline produces `"Message from '<sender>' to '<recipient>' timed out after <t>s"` (`Error::Timeout`).
`TeamRuntime` owns one `Arc<dyn TeamTransport>` (in-process by default) and `send`/`publish`/`subscribe`/`start`/`stop` delegate to it; `set_transport` rejects a transport whose `team_id` differs from the runtime's (`Error::Validation`).
The factory `build_transport` hands back a bare `Arc<dyn TeamTransport>`, so a caller that needs an inbox worker must build the concrete backend (`RedisTransport::new`/`try_new`, which return `Arc<Self>`): `watch_agent`, `refresh_subscriptions`, `is_started`, `consumer_name`, `config`, `key_prefix` and the `p2p_queue`/`reply_queue`/`pubsub_queue`/`p2p_consumer_group`/`pubsub_consumer_group` accessors are inherent, not on the trait.
Nothing wires a transport's inbound workers to a `TeamRuntime`: `RuntimeHandler` is private, `TeamRuntime::dispatch_p2p`/`dispatch_pubsub` are private, and the only public handle is `set_transport`, which replaces the outbound backend alone — a Redis transport serving a team runtime therefore needs the caller to supply its own `MessageHandler`.

## Group routing (`routing.rs`)

Additive Rust extension, not a port: Python's `core.multi_agent` has no group, broadcast or inbox layer (its `MessageRouter` only classifies messages by transport-topic suffix); the contract is modelled on the `agent_teams` product, where a broadcast excludes the sender. `GroupRegistry` holds named `group id -> sorted member ids` sets: `register_group(group_id, members)` replaces an existing group and returns its member count (`Validation` for an empty id), `add_member` creates a group implicitly and reports whether the pair is new, `remove_member`/`unregister_group` report whether they removed anything, and `group_members`/`groups`/`groups_of` are sorted, with `group_count`, `member_count` and `clear()` (returning how many groups it removed). `broadcast_targets(members, sender, include_sender)` is a pure helper returning a sorted, deduplicated recipient set with the sender excluded unless `include_sender` is set (and nothing excluded when `sender` is `None`).
`TeamRuntime::broadcast_to_group(message, group_id, sender, session_id, timeout)` rejects an empty sender (`Validation`) and an unknown group (`NotFound`), then sends one ordinary P2P request per recipient concurrently through the transport; each envelope carries `metadata["group"] = group_id`, a group member not registered with the runtime is logged and reported in `GroupBroadcast::skipped`, a member failure stays in its `GroupDelivery` and never aborts the batch, and one `message_sent` event is emitted before dispatch with `detail.kind = "group"`, `detail.group` and `detail.recipients` (a count). `GroupBroadcast::{succeeded, failed, value_of, into_results}` summarise the deliveries. `RoutingTable` (`team_id`, `groups`, `topics`, `agents`; serde `snake_case` + `default` + `JsonSchema`) is a plain snapshot assembled by `TeamRuntime::routing_table()`: `route(destination)` resolves a group id first and otherwise matches the destination against every topic pattern via `subscription::topic_matches`, and `destinations(agent_id)` returns that agent's topics plus groups. It is a moment-in-time view, not a live one.

## Inbox filters (`inbox.rs`)

Another additive Rust extension: Python's bus delivers every envelope to its registered member. `InboxFilter` is pure data that serializes and defaults to accepting everything; rules apply in order — a sender in `deny_senders` is rejected (deny wins), a set `allow_senders` requires a matching sender (an envelope without a sender is rejected), a non-empty `topics` pattern list requires a matching `topic_id`, every `metadata` pair must be present with that exact string value, and `require_session` requires a session id. The builders (`allow_senders`, `deny_senders`, `allow_topics`, `require_metadata(key, value)`, `require_session`) accumulate. `InboxFilterRegistry` maps an agent id to its filter (`set`/`remove`/`get`/`accepts`/`agents`/`len`/`is_empty`/`clear`); an agent with no entry accepts everything, so an empty registry reproduces the unfiltered behaviour.
Enforcement lives in the runtime dispatcher, not in a handler wrapper, so it covers every transport: `dispatch_p2p` rejects a filtered envelope with `Error::validation("inbox of '<recipient>' rejected the message from '<sender>'")` and emits no `message_received`, while `dispatch_pubsub` logs at debug and skips that subscriber, so pub/sub stays fire-and-forget and never fails the publisher.

## Communicable facade (`communicable.rs`)

`CommunicableAgent` is the Rust counterpart of Python's `CommunicableAgent` mixin (`team_runtime/communicable_agent.py`): Rust cannot mix the messaging methods into an agent class, so one value owns the binding and exposes the same calls. `new(agent_id)` starts unbound but with a known identity; `bind_runtime(runtime, agent_id)` binds it (the same `(runtime, agent_id)` pair, compared by `Arc` pointer, is a silent no-op, a different pair logs a warning and rebinds), `is_bound`/`runtime`/`agent_id` report the state (the id survives `unbind`), and `unbind` drops only the runtime binding. `send`/`publish`/`subscribe`/`unsubscribe` forward to the runtime with this agent as the sender and the same arguments (no explicit deadline on `send`); while unbound each returns `Error::validation("agent '<id>' is not bound to a team runtime")`, Python's "not bound to a `TeamRuntime`" status error. The binding sits behind a short `parking_lot::RwLock` released before any `.await`, so a bound agent can be shared across tasks.

## Constraints

- Everything here follows the crate's constraints: async-first tokio, `openjiuwen_common::error` results, serde-serializable public data, no `unsafe`, `tracing` logging, files around ≤1000 lines.
- No module in this node depends upward: `events`, `autosubscribe`, `pyrepr`, `resources`, `member_checkpoint`, `routing`, `inbox`, `communicable`, `transport` and `ability` all sit on `common`/`foundation`/`session`/`agent` plus this crate's own `runtime`.
- Process-local types (event buses, transports, resource managers, checkpoint wrappers, ability managers, group and inbox-filter registries, communicable agents, `GroupDelivery`/`GroupBroadcast`) are not serializable; the serializable data types here are `InboxFilter` and `RoutingTable`.

## Routing Table

- `./events.rs` → `TeamEvent`, `TeamEventKind`, `TeamEventFilter`, `TeamEventBus`, the Python callback constants
- `./autosubscribe.rs` → card/tag/event-card subscription derivation, reconciliation modes, overrides
- `./pyrepr.rs` → Python literal parsing and `str()`-style rendering
- `./resources.rs` → `TeamResourceMgr` trait, `LocalResourceMgr`
- `./member_checkpoint.rs` → per-member checkpoint identity over one shared checkpointer
- `./transport/mod.rs` → `TeamTransport`, `TransportKind`, `TransportConfig`, `build_transport`
- `./transport/inproc.rs` → `InProcessTransport` over the message bus
- `./transport/redis/` → `RedisTransport`, split into `config.rs` (settings and topology), `streams.rs` (stream/SET commands and codecs), `worker.rs` (consumer-group delivery loop), `registry.rs` (cross-process subscription registry) and `backend.rs` (the `TeamTransport` impl)
- `./ability.rs` → `P2PAbilityManager`, `AbilityCall`, `AbilityResult`, `AbilityExecutor`
- `./routing.rs` → `GroupRegistry`, `broadcast_targets`, `GroupBroadcast`/`GroupDelivery`, `RoutingTable`, `GROUP_METADATA_KEY`
- `./inbox.rs` → `InboxFilter` and `InboxFilterRegistry` (per-agent acceptance rules)
- `./communicable.rs` → `CommunicableAgent`, the runtime-bound send/publish/subscribe facade
- `./runtime/` → the runtime these seams hang off (`seams.rs` attaches them)
- `./runtime/routing.rs` → the runtime's group/inbox forwarders, `broadcast_to_group` and `routing_table`
- `./handoff/` → handoff strategy (`handoff/CONTEXT.md`)
- `./hierarchical/` → hierarchical strategies; re-exports `P2PAbilityManager` under its Python path (`hierarchical/CONTEXT.md`)
