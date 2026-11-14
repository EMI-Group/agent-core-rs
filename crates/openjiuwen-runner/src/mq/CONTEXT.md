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
pub struct QueueStats {                             // the report a backend can honestly give
    pub started: bool,                              // between start and stop
    pub topics: usize,                              // topics the backend owns
    pub handlers: usize,                            // topics with a handler installed
    pub deliveries: u64,                            // deliveries a worker started
    pub dropped_no_worker: u64,                     // no topic worker (unsubscribed, or its worker stopped)
    pub dropped_no_handler: u64,                    // no handler installed
    pub handler_errors: u64,                        // handler Err on the publish or request path
    pub rejected_not_started: u64,                  // operations refused on a stopped queue
}                                                   // derives Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize (snake_case, default)
impl QueueStats {
    pub fn is_empty(&self) -> bool;                 // equals QueueStats::default()
    pub fn dropped(&self) -> u64;                   // dropped_no_worker + dropped_no_handler
}
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
    fn stats(&self) -> QueueStats { QueueStats::default() }   // defaulted, synchronous, object-safe
}
pub struct InMemoryMessageQueue { .. }
impl InMemoryMessageQueue {
    pub fn new(config: MessageQueueConfig) -> Arc<Self>;   // uses capacity + default request timeout
    pub fn with_capacity(capacity: usize) -> Arc<Self>;
    pub fn deliveries(&self) -> u64;                       // delegates to stats().deliveries
}
pub async fn create_message_queue(config: &MessageQueueConfig) -> Result<Arc<dyn MessageQueue>>;  // in_memory only; Error::Unsupported otherwise
pub fn build_message_queue(config: &MessageQueueConfig) -> Arc<dyn MessageQueue>;                 // infallible: deferred kind → DeferredMessageQueue

#[derive(Debug)] pub struct DeferredMessageQueue { .. }   // typed stand-in for pulsar/zeromq
impl DeferredMessageQueue {
    pub fn new(kind: MessageQueueKind) -> Arc<Self>;
    pub fn kind(&self) -> MessageQueueKind;                // the configured transport
}
// impl MessageQueue for DeferredMessageQueue: is_started() == false, topics() == [], has_subscriber() == false,
// stats() == QueueStats::default(), and start/stop/subscribe/unsubscribe/add_handler/publish/request
// all return Error::Unsupported.

// mod.rs re-exports: QueueMessage, MessageHandler, MessageQueue, QueueStats, InMemoryMessageQueue,
//                    DeferredMessageQueue, create_message_queue, build_message_queue,
//                    MessageQueueKind, MessageQueueConfig (from config)
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
- **Statistics go through the trait, synchronously**: `MessageQueue::stats()` returns a `QueueStats` report and is defaulted to `QueueStats::default()`, so a third-party backend keeps compiling and object safety survives (`Arc<dyn MessageQueue>` samples the report without an `.await` or a downcast). A backend that cannot report statistics answers the empty report; nothing fabricates a counter.
- **`InMemoryMessageQueue` reports real counters**: `started`/`topics`/`handlers` are read from the topic table under one short lock, and the monotonic totals (`deliveries`, `dropped_no_worker`, `dropped_no_handler`, `handler_errors`, `rejected_not_started`) are `AtomicU64`s shared with the topic workers. `deliveries()` delegates to `stats().deliveries`, so there is one counter.
- **The counters move where the code logs**: a publish to an unsubscribed topic (or to a channel whose worker stopped) is `dropped_no_worker`; a delivery with no handler installed is `dropped_no_handler`; a handler returning `Err` on either path is `handler_errors`; any `publish`/`request`/`subscribe`/`add_handler` refused on a stopped queue is `rejected_not_started`. A drop is not a delivery, and `QueueStats::dropped()` sums only the two drop counters.
- **The report outlives the topics**: `stop` empties the topic table (`topics`/`handlers` return to `0`) but keeps the monotonic totals, so a host can still read what a queue did after it was stopped.

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
- `stats()` is the full report: use it instead of `deliveries()` when a host needs drops, handler errors, or refusals. A `request` whose handler was removed after the delivery was queued answers `Error::NotFound` and counts as `dropped_no_handler`.
- A handler that calls `request` on its own topic deadlocks (the worker awaits that handler); this is the price of strict per-topic FIFO and is not worked around.
- `DeferredMessageQueue` is a placeholder, not a backend: even `stop` returns `Error::Unsupported` (a host that wants a working queue uses `in_memory`). `Runner::start` also installs it in the queue slot when the configured kind is deferred, so `message_queue()` keeps reporting the configured transport after a `set_config`.

## Routing Table

- `./src/mq/message.rs` → `QueueMessage`, `MessageHandler`, `SUCCESS`
- `./src/mq/queue.rs` → `QueueStats`, `MessageQueue` trait, `create_message_queue`, `build_message_queue`, `DeferredMessageQueue`
- `./src/mq/memory.rs` → `InMemoryMessageQueue` and its topic workers
- `./src/mq/counters.rs` → the private atomic counters the in-memory queue reports
- `./src/mq/mod.rs` → re-exports
- `../tests/message_queue.rs` → behavior suite (sibling — read-only, escalate writes to the crate root)
