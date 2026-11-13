# message_queue/ — queue factory, JSON bridge, deferred Pulsar backend

## Intent

The kernel owns the runner message queue (`openjiuwen_runner::mq`: `MessageQueue`, `QueueMessage`, `MessageHandler`, `InMemoryMessageQueue`); Python's `extensions/message_queue/` only contributes the Pulsar transport.
Rust keeps the seam: a config-driven factory over the kernel queue, a JSON bridge that adapts any `MessageQueue` to `serde_json::Value` payloads and topic streams, and a Pulsar surface that is deferred with `Error::Unsupported`.
Pick this node for "build a queue from config", "bridge a queue to JSON", or "Pulsar".

## API Surface

```rust
pub use openjiuwen_core::runner::mq::{
    InMemoryMessageQueue, MessageHandler, MessageQueue, QueueMessage, SUCCESS,
};

pub enum QueueKind { InMemory, Pulsar }
pub struct QueueConfig {
    pub kind: QueueKind,
    pub capacity: usize,                 // kernel default when None
    pub request_timeout: Option<Duration>,
    pub url: Option<String>,
    pub max_workers: usize,
    pub wire_version: WireVersion,       // Pulsar only, default V1
    pub legacy_key_encoding: KeyEncoding,// Pulsar only, default Literal
}
impl QueueConfig { pub fn in_memory() -> Self; pub fn pulsar(url: impl Into<String>) -> Self; }

/// Builds and starts nothing: the returned queue is stopped, exactly like `Runner` expects.
pub async fn create_queue(config: &QueueConfig) -> Result<Arc<dyn MessageQueue>>;
// InMemory -> kernel `InMemoryMessageQueue` (capacity + request timeout applied)
// Pulsar   -> Error::Unsupported("pulsar message queue backend is deferred; use the in-process queue")

pub struct QueueBridge { .. }
impl QueueBridge {
    pub fn new(queue: Arc<dyn MessageQueue>) -> Self;
    pub async fn publish_json(&self, topic: &str, payload: Value) -> Result<()>;
    pub async fn request_json(&self, topic: &str, payload: Value, timeout: Option<Duration>) -> Result<Value>;
    pub async fn subscribe_json(&self, topic: &str, handler: Arc<dyn JsonMessageHandler>) -> Result<()>;
    pub fn topic_stream(&self, topic: &str, capacity: usize) -> Result<BoxStream<'static, Result<Value>>>;
    pub fn queue(&self) -> &Arc<dyn MessageQueue>;
}
#[async_trait] pub trait JsonMessageHandler: Send + Sync {
    async fn handle_json(&self, payload: Value) -> Result<Value>;
}

pub enum WireVersion { V1, V2 }        // Pulsar wire formats
pub enum KeyEncoding { Literal, Escaped }

#[cfg(feature = "pulsar")]
pub mod pulsar {
    pub struct PulsarConfig { pub url: Option<String>, pub max_workers: usize }  // default workers 8
    pub struct PulsarMessageQueue { .. }
    impl PulsarMessageQueue {
        pub fn new(config: PulsarConfig) -> Result<Self>;                       // validates max_workers > 0
    }
    /// The one deferral text: names `pulsar`, the protobuf-framed
    /// `pulsar-client` protocol + missing `prost`/`tonic` codegen, the absent
    /// broker, and the in-process alternative.
    pub const PULSAR_DEFERRED_REASON: &str;
    // Module-level functions (Python's provider entry points): validate, then defer.
    pub async fn connect(config: PulsarConfig) -> Result<Arc<dyn MessageQueue>>; // Error::Unsupported(PULSAR_DEFERRED_REASON)
    pub async fn start_with_retry(config: PulsarConfig, max_retries: u32, retry_delay: Duration)
        -> Result<Arc<dyn MessageQueue>>;                                        // Error::Unsupported(PULSAR_DEFERRED_REASON)
}
```

## Ported Behavior

- `create_queue` maps the Python `MessageQueueConfig.type` discriminators: `fake`/`in_memory` → the in-process queue; `pulsar` → deferred.
- The bridge keeps the kernel request/reply contract: `publish_json` returns once enqueued, `request_json` returns the handler's JSON response, handler errors propagate unchanged, and timeouts surface as `Error::Timeout` from the kernel.
- `topic_stream` installs a handler that forwards each delivery into a bounded `mpsc` channel (backpressure by capacity) and yields it as a `BoxStream`; a `QueueMessage` with `error_code != SUCCESS` becomes `Error::Transport(error_msg)` on the stream.
- Pulsar surface preserves Python names/constants: `PulsarConfig{url, max_workers}`, `MAX_PRODUCERS = 10_000`, `DEFAULT_SUBSCRIPTION_NAME = "default"`, `DEFAULT_MAX_RETRIES = 3`, `DEFAULT_RETRY_DELAY_SECS = 1.0`, `WireVersion` (`V1` default) and `KeyEncoding` (`Literal` default).
- Deferral message: `Error::Unsupported(PULSAR_DEFERRED_REASON)` naming `pulsar`, the missing transport (protobuf-framed `pulsar-client` protocol on `pulsar://` 6650 / `pulsar+ssl://` 6651 plus an HTTP admin API, no `prost`/`tonic` codegen offline, no test broker) and the in-process alternative (`openjiuwen_runner::mq::InMemoryMessageQueue`, i.e. `create_queue(&QueueConfig::in_memory())`); no silent fallback to the in-memory queue (Python's `fallback_to_fake` is documented, not ported), no retry/sleep, and `PulsarMessageQueue` is an introspection handle with no `start`/`send`/`subscribe`.

## Constraints

- The in-memory queue is always the kernel implementation — this crate adds no transport.
- No broker client dependency; the `pulsar` feature is API-only — `pulsar.rs` compiles no transport, and `PULSAR_DEFERRED_REASON` is the single deferral text for both its entry points (the `create_queue` factory in `mod.rs` keeps its own shorter `PULSAR_DEFERRED`).
- The bridge never holds a lock across `.await`; topics are created via `add_handler`/`subscribe`.
- `topic_stream` keeps its synchronous signature: it spawns the `add_handler` registration on the current tokio runtime and the stream yields a registration failure as its first (and only) item, so it must be called inside a runtime.
- Errors: `Error::Config`/`Error::Validation` (config), `Error::Unsupported` (deferred backend), otherwise the kernel's error unchanged.
- Files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`/`bridge.rs`: `QueueConfig::in_memory` builds a started queue, publish/request round-trips JSON, handler errors propagate, `topic_stream` yields published payloads in order and maps error codes, `pulsar` config → `ErrorKind::Unsupported`.
- `pulsar` tests: `PulsarConfig` defaults/serde round-trip and unknown-field tolerance, `connect`/`start_with_retry` returning `ErrorKind::Unsupported` with exactly `PULSAR_DEFERRED_REASON` (substrings: `pulsar`, `in-process`, `protobuf`, `pulsar-client`/`broker`, `InMemoryMessageQueue`, `QueueConfig::in_memory()`, ports 6650/6651, `admin API`, `prost`), and config validation winning over the deferral (`max_workers = 0` → `ErrorKind::Validation`); the Pulsar wire enums (`WireVersion` V1 / `KeyEncoding` Literal defaults) are covered by `mod.rs` tests.
- Python counterparts named in comments: `tests/unit_tests/extensions/message_queue/test_start_with_retry.py`, `test_wire_formats.py`, `test_unsubscribe_all.py`.

## Routing Table

- `./mod.rs` → `QueueKind`, `QueueConfig`, `create_queue`, wire enums, tests
- `./bridge.rs` → `QueueBridge`, `JsonMessageHandler`, topic stream
- `./pulsar.rs` → `pulsar` feature surface (`PulsarConfig`, `PulsarMessageQueue`)
