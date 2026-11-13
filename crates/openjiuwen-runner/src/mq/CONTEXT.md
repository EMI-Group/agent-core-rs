# mq/ — the in-process message queue

## Intent

A generic, runner-owned topic queue: publish/subscribe plus request/reply, with per-topic FIFO delivery, idempotent start/stop, and explicit error propagation.
Ports Python `openjiuwen/core/runner/message_queue_base.py` (`QueueMessage`, `MessageQueueBase`, `SubscriptionBase`, `AsyncMessageHandler`) and the behavioural core of `message_queue_inmemory.py` / `dunner/dmessage_queue/message_queue_fake.py`, without the distributed wire (`DmqRequestMessage`/`DmqResponseMessage`, reply topics).
The `MessageQueue` trait is the seam a broker backend implements: `openjiuwen-extensions` will supply Pulsar/ZeroMQ; `create_message_queue` answers `Error::Unsupported` for both today.
`build_message_queue` is its infallible twin: a deferred kind yields `DeferredMessageQueue`, a typed placeholder that refuses every operation with that same `Error::Unsupported`, so the runner never serves an in-memory queue under another backend's name.

## API Surface

```rust
// message.rs
pub const SUCCESS: i32 = 0;
pub struct QueueMessage { pub message_id: String, pub payload: Value, pub error_code: i32, pub error_msg: String }
impl QueueMessage {
    pub fn new(payload: Value) -> Self;                 // message_id = uuid v4
    pub fn with_id(self, id: impl Into<String>) -> Self;
    pub fn with_error(self, code: i32, message: impl Into<String>) -> Self;
    pub fn is_error(&self) -> bool;
}
impl Default for QueueMessage                      // empty id, null payload, SUCCESS
#[async_trait] pub trait MessageHandler: Send + Sync {
    /// Handles one delivery and returns the request/reply response (`Value::Null` for pub/sub).
    async fn handle(&self, message: QueueMessage) -> Result<Value>;
}

// queue.rs
#[async_trait] pub trait MessageQueue: Send + Sync {
    async fn start(&self) -> Result<()>;               // idempotent
    async fn stop(&self) -> Result<()>;                // idempotent; joins topic workers
    fn is_started(&self) -> bool;
    async fn subscribe(&self, topic: &str) -> Result<()>;     // create the topic worker; duplicate = Error::Validation
    async fn unsubscribe(&self, topic: &str) -> Result<()>;   // unknown topic = Ok(())
    async fn add_handler(&self, topic: &str, handler: Arc<dyn MessageHandler>) -> Result<()>;  // auto-subscribes
    async fn publish(&self, topic: &str, message: QueueMessage) -> Result<()>;
    async fn request(&self, topic: &str, message: QueueMessage, timeout: Option<Duration>) -> Result<Value>;
    fn topics(&self) -> Vec<String>;                   // sorted
    fn has_subscriber(&self, topic: &str) -> bool;     // a handler is installed
}
pub struct InMemoryMessageQueue { .. }
impl InMemoryMessageQueue {
    pub fn new(config: MessageQueueConfig) -> Arc<Self>;   // uses capacity + default request timeout
    pub fn with_capacity(capacity: usize) -> Arc<Self>;
    pub fn deliveries(&self) -> u64;                       // counter for tests/observability
}
pub async fn create_message_queue(config: &MessageQueueConfig) -> Result<Arc<dyn MessageQueue>>;  // in_memory only; Error::Unsupported otherwise
pub fn build_message_queue(config: &MessageQueueConfig) -> Arc<dyn MessageQueue>;                 // infallible: deferred kind → DeferredMessageQueue

#[derive(Debug)] pub struct DeferredMessageQueue { .. }   // typed stand-in for pulsar/zeromq
impl DeferredMessageQueue {
    pub fn new(kind: MessageQueueKind) -> Arc<Self>;
    pub fn kind(&self) -> MessageQueueKind;                // the configured transport
}
// impl MessageQueue for DeferredMessageQueue: is_started() == false, topics() == [], has_subscriber() == false,
// and start/stop/subscribe/unsubscribe/add_handler/publish/request all return Error::Unsupported.

// mod.rs re-exports: QueueMessage, MessageHandler, MessageQueue, InMemoryMessageQueue, DeferredMessageQueue,
//                    create_message_queue, build_message_queue, MessageQueueKind, MessageQueueConfig (from config)
```

## Semantics (the contract)

- **Topics are created by `subscribe`** (or implicitly by `add_handler`); each topic owns one bounded `mpsc` channel and one tokio worker task. `publish`/`request` on a topic without a worker drop the message (debug log, `Ok(())`) for `publish`, and fail with `Error::NotFound("message queue topic '<topic>' is not subscribed")` for `request` — the fix for Python's "in-flight response orphaned forever" bug.
- **A topic without a handler** behaves the same: `publish` is a logged no-op, `request` is `Error::NotFound("... has no handler")`.
- **Per-topic FIFO**: the worker awaits each handler before dequeuing the next delivery, so deliveries on one topic are strictly ordered and a handler that calls `request` on **another** topic works. A handler that requests its own topic deadlocks; documented and deliberately not worked around (Python spawned a task per delivery and lost ordering — this port keeps ordering).
- **`publish` returns once the delivery is enqueued**, not once it is handled; a full queue awaits capacity (backpressure). Handling errors on the publish path are logged at `warn` (a publisher cannot observe them).
- **`request` waits for the handler's `Result<Value>`**: a handler `Err` is returned to the caller unchanged; a caller timeout is `Error::Timeout("message queue request on '<topic>' timed out after <ms>ms")` and the handler keeps running. `timeout = None` falls back to `MessageQueueConfig::request_timeout()`; both `None` means wait forever.
- **Lifecycle**: `publish`/`request`/`subscribe`/`add_handler` before `start` fail with `Error::Config("message queue is not started")`. `start` after `stop` re-enables the queue but does not resurrect old topics (topics are emptied by `stop`). `stop` unsubscribes every topic (closing channels and joining workers) and is idempotent.
- **Backends**: `create_message_queue` builds `InMemoryMessageQueue` for `MessageQueueKind::InMemory` and returns `Error::Unsupported` naming the deferred backend otherwise (`"<kind> message queue is not supported: use `in_memory`"`); `MessageQueueConfig::url` is carried untouched for that future implementation.
- **A deferred kind is never misrepresented**: `build_message_queue` (used by `Runner::with_config`, which cannot fail) yields `DeferredMessageQueue` for `pulsar`/`zeromq`. Its `kind()` and `Debug` report the configured transport, `is_started()` is `false`, `topics()` is empty, and every fallible operation returns the same typed `Error::Unsupported` naming the transport — so `Runner::message_queue()` fails loudly before `start`, and `Runner::start` returns that same error. There is no silent downgrade to in-memory.

## Constraints

- No `unsafe`, no `std` blocking, no `parking_lot` guard across `.await`; the topic table is a `tokio::sync::Mutex` or a `parking_lot::RwLock` held only for map operations (subscribe/unsubscribe return an owned `JoinHandle` to await outside the lock).
- Every worker is owned: `stop`/`unsubscribe` close the channel, await the `JoinHandle`, and remove the entry. A worker exits when its channel closes.
- Handler panics are contained by `tokio::spawn` semantics: the worker task ends, `request` observes a closed reply channel as `Error::Transport("... worker stopped")`. Never `unwrap` on user data.
- `tracing` target `openjiuwen_runner::mq`; never log payload contents above `debug`.
- Tests: inline unit tests for per-topic ordering, no-handler behavior, timeouts and lifecycle, plus `../tests/message_queue.rs`.

## Notes for Agents

- A handler panic ends that topic's worker (contained by `tokio::spawn`) while the topic stays in the table: later `publish`es become logged no-ops, later `request`s fail with `Error::Transport`, and `unsubscribe`/`subscribe` on the topic is the way back.
- `publish` to a subscribed topic with no handler returns before enqueueing, so it consumes no capacity in the bounded channel (same logged-no-op contract).
- `deliveries()` counts the deliveries a worker actually started (handler present), including ones whose handler returned `Err`; requests rejected before enqueueing are not counted.
- A handler that calls `request` on its own topic deadlocks (the worker awaits that handler); this is the price of strict per-topic FIFO and is not worked around.
- `DeferredMessageQueue` is a placeholder, not a backend: even `stop` returns `Error::Unsupported` (a host that wants a working queue uses `in_memory`). `Runner::start` also installs it in the queue slot when the configured kind is deferred, so `message_queue()` keeps reporting the configured transport after a `set_config`.

## Routing Table

- `./src/mq/message.rs` → `QueueMessage`, `MessageHandler`, `SUCCESS`
- `./src/mq/queue.rs` → `MessageQueue` trait, `create_message_queue`, `build_message_queue`, `DeferredMessageQueue`
- `./src/mq/memory.rs` → `InMemoryMessageQueue` and its topic workers
- `./src/mq/mod.rs` → re-exports
- `../tests/message_queue.rs` → behavior suite (sibling — read-only, escalate writes to the crate root)
