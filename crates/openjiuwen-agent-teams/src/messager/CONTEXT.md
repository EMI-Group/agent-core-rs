# openjiuwen-agent-teams/src/messager

## Intent

The transport seam for team events: topic-based publish/subscribe plus point-to-point delivery by member name.
Python parity source: `openjiuwen/agent_teams/messager/` (`messager.py` ABC, `base.py` config + factory, `inprocess.py` handler bus, `pyzmq_backend.py`, `hybrid.py`).
Every event that crosses a member boundary goes through a [`Messager`]; no other module of this crate opens a socket, queue or channel for team messaging.
Only the in-process transport is implemented; `pyzmq` / `hybrid` stay documented seams that reject construction (see `../../../docs/PARITY.md` and the crate `CONTEXT.md` "Deferred" list).

## API Surface

```rust
pub type EventSink = Arc<dyn Fn(EventMessage) + Send + Sync>;

#[async_trait]
pub trait Messager: Send + Sync {
    fn name(&self) -> &'static str;                              // "inprocess"
    fn subscribe(&self, member: &str, topics: &[TeamTopic], sink: EventSink) -> Result<()>;
    fn unsubscribe(&self, member: &str) -> Result<()>;
    async fn publish(&self, topic: TeamTopic, event: EventMessage) -> Result<()>;
    async fn send(&self, target: &str, event: EventMessage) -> Result<()>;
    async fn subscribe_transport(&self, team_name: &str) -> Result<()>;
    fn subscribers(&self, topic: TeamTopic) -> Vec<String>;      // sorted member names
    fn is_started(&self) -> bool;
    async fn start(&self) -> Result<()>;                         // idempotent
    async fn stop(&self) -> Result<()>;                          // idempotent
}

pub struct InProcessMessager;
impl InProcessMessager {
    pub fn new() -> Self;                                        // sender "" and live
    pub fn with_sender(sender: impl Into<String>) -> Self;       // stamps this member name
    pub fn from_config(config: &InProcessMessagerConfig) -> Self;
    pub fn sender(&self) -> &str;
    pub fn team_subscriptions(&self) -> Vec<String>;             // sorted
}
impl Default for InProcessMessager;
impl Messager for InProcessMessager;

pub struct MessagerTransportConfig { pub kind: String, pub params: BTreeMap<String, Value> }
impl MessagerTransportConfig {
    pub fn new(kind: impl Into<String>) -> Self;
    pub fn inprocess() -> Self;
    pub fn from_spec(spec: &TransportSpec) -> Self;
}
pub struct MessagerPeerConfig { pub member_name: String, pub address: String }
pub struct InProcessMessagerConfig { pub sender: String, pub peers: Vec<MessagerPeerConfig> }
pub struct PyZmqMessagerConfig {
    pub team_name: String, pub node_id: Option<String>,
    pub direct_addr: Option<String>,
    pub pubsub_publish_addr: Option<String>, pub pubsub_subscribe_addr: Option<String>,
    pub request_timeout_secs: f64, pub peers: Vec<MessagerPeerConfig>,
}

pub fn create_messager(spec: &TransportSpec) -> Result<Arc<dyn Messager>>;

#[cfg(feature = "distributed")]
pub struct PyZmqMessager;
#[cfg(feature = "distributed")]
impl PyZmqMessager { pub fn new(config: PyZmqMessagerConfig) -> Result<Self>; }
```

`MessagerTransportConfig`/`MessagerPeerConfig`/`InProcessMessagerConfig`/`PyZmqMessagerConfig` derive `Debug, Clone, Default, PartialEq, Serialize, Deserialize` with `#[serde(rename_all = "snake_case", default)]` and tolerate unknown fields.
`MessagerTransportConfig.kind` also accepts the Python keys `"type"` and `"backend"` as serde aliases; it serializes as `"kind"`.

### Delivery semantics (pinned)

- `subscribe` is idempotent per `(member, topic)`: re-subscribing replaces that member's sink for the topic, so duplicate subscription still invokes one sink per publish.
  A member may hold several topics; one `subscribe` call with two topics registers the same sink for both.
  A non-empty `topics` also registers the sink as the member's direct (point-to-point) handler.
- `unsubscribe(member)` removes the member from every topic and from the direct registry; unknown members are a no-op (`Ok(())`).
- `publish(topic, event)` clones the subscriber list under a short read lock, releases the lock, then invokes each sink in **sorted member order**.
  A panicking sink is caught with `catch_unwind` and logged via `tracing::warn`; other sinks still run and `publish` returns `Ok(())`.
- `send(target, event)` delivers to exactly one member's direct handler; an unknown target is a no-op returning `Ok(())` (Python parity: mailbox persistence, not transport, decides reachability).
  A panicking handler is caught and logged exactly like `publish`.
- Sender stamping: both `publish` and `send` set `event.sender_id` to the messager's configured sender when the event's `sender_id` is empty; an explicit non-empty `sender_id` is preserved verbatim.
  A messager with the default empty sender leaves an empty `sender_id` empty.
- Lifecycle: a freshly constructed messager is live (`is_started() == true`) so it is usable without a start call.
  `start()` / `stop()` are idempotent; `stop()` does not drop subscriptions, so a later `start()` resumes delivery.
  While stopped, `publish` and `send` drop the event with a `tracing::warn` and return `Ok(())`.
- `subscribe_transport(team_name)` records the team name in a sorted, de-duplicated set (the Python leader hook); it has no delivery effect.
- Topics are `TeamTopic` enum values, not opaque strings: the crate's runtime never formats a topic string for transport.
  The Python `TeamTopic` wire strings remain available via `TeamTopic::as_str`/`topic` for logs and tests.

### Factory (`create_messager`)

| `spec.kind` | Result |
| --- | --- |
| `"inprocess"` | `Arc<InProcessMessager>` (default sender) |
| `"pyzmq"` | `Error::unsupported("transport 'pyzmq' is not implemented; use 'inprocess'")` |
| `"hybrid"` | `Error::validation("transport 'hybrid' is not supported by create_messager")` |
| anything else | `Error::validation(format!("unknown transport '{kind}'"))` |

With `--features distributed`, `PyZmqMessager::new(config)` is visible in rustdoc and returns `Error::unsupported` naming the deferred `pyzmq` capability; `create_messager` keeps the same rejection (no constructible stub instance).

## Constraints

- Files under `src/messager/` only: `mod.rs` (trait, `EventSink`, re-exports), `inprocess.rs` (`InProcessMessager` + unit tests), `factory.rs` (config carriers + `create_messager`), `distributed.rs` (feature-gated `PyZmqMessager` stub).
- `openjiuwen_common::error::{Error, Result}` only; `tracing` only (no `println!`/`eprintln!`); no `unsafe`; no `unwrap`/`expect` outside tests.
- `parking_lot::RwLock` guards the registry. Never invoke a sink while a lock is held: clone the sink list first. That also prevents re-entrant deadlocks when a sink subscribes or unsubscribes.
- `EventSink` is synchronous; a transport that needs async handlers adapts outside this seam.
- The registry is per-instance, not process-global: two `InProcessMessager` values do not see each other's subscriptions. The runtime creates one messager per team and shares the `Arc`.
- `--no-default-features` must build; `distributed` is the only feature this module reads.

## Routing Table

Leaf node: no child directories.

## Test Strategy

Unit tests live beside the implementation (`#[cfg(test)] mod tests` in `inprocess.rs`, `factory.rs`, and feature-gated in `distributed.rs`) and are deterministic: no sleeps, no I/O, no tokio runtime needed.
Coverage contract:
- fan-out order is sorted by member name and stable across repeats;
- a subscriber with two topics receives once per matching publish;
- empty `sender_id` is stamped from the messager sender; an explicit `sender_id` is preserved;
- `send` reaches only the target member; unknown target is `Ok(())`;
- `unsubscribe` stops delivery and is idempotent; duplicate `subscribe` invokes one sink;
- a panicking sink does not block other sinks and does not fail `publish`/`send`;
- `start`/`stop` are idempotent and post-stop publishes/sends are dropped;
- `create_messager` for `inprocess`, `pyzmq`, `hybrid`, and an unknown kind;
- `subscribe_transport` records sorted, de-duplicated teams;
- `TeamTopic::topic(..)` strings round-trip (`TeamTopic::parse`) and equal the Python wire strings;
- config carriers round-trip through JSON and `from_spec` preserves kind/params (aliases `"type"`/`"backend"` accepted).

Definition of done for this node: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --lib`, `cargo test -p openjiuwen-agent-teams --no-default-features --lib`, `cargo test -p openjiuwen-agent-teams --features distributed --lib`, and `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` all green for `src/messager/`.

## Notes for Agents

- Python's `_Bus` is a process-global singleton. This port deliberately makes the bus per-`InProcessMessager`; the runtime shares one `Arc<dyn Messager>` per team instead of relying on global state (the crate allows exactly two documented process-global singletons, and the messager is not one of them).
- Python stamps `sender_id` from `MessagerTransportConfig.node_id`; Rust exposes `InProcessMessager::with_sender`, and callers that already know the publisher use `EventMessage::with_sender` before publishing (then stamping is a no-op).
- Python's `subscribe(topic_id, handler)` takes one topic string; the Rust contract takes `TeamTopic` values so a typo cannot create a dead channel. This is a deliberate API deviation, documented here.
- Python separates `subscribe(topic, handler)` from `register_direct_message_handler(handler)`; Rust folds both into `subscribe` (a non-empty `topics` slice installs the member's direct handler) because the runtime always registers them together. `subscribe(member, &[], sink)` installs no direct handler.
- Python's `MessagerTransportConfig` types every backend field (`backend`, `team_name`, `node_id`, addresses, peer lists, `request_timeout`); the Rust carrier keeps only `kind` plus an opaque `params` map so `MessagerTransportConfig::from_spec` copies any `TransportSpec` losslessly. Only the deferred `PyZmqMessagerConfig` types a subset of those fields.
- Python keeps one p2p handler per `agent_id`; the Rust direct registry mirrors that (latest `subscribe` for a member wins). Publish fan-out uses the separate per-topic registry and is unaffected.
- Self-published events are delivered, not filtered: subscribers filter their own echoes by `sender_id` (Python `CoordinationKernel._filter_self`). Do not add filtering here.
- Python's `inprocess` passes the `EventMessage` object through untouched; because Rust stamps a copy, sinks must treat the received event as read-only.
- The `hybrid` transport in Python is a publish-only WebSocket relay built only by `external/client.py`; `create_messager` always rejected it. Keep the validation error.
- Deferred capabilities keep their Python names in the `Error::Unsupported` message so the parity table can match them.
