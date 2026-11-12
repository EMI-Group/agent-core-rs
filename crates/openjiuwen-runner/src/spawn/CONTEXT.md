# spawn/ — the subprocess spawn seam

## Intent

Pin the contract a future subprocess teammate/agent backend implements, without spawning anything today.
Ports the *shape* of Python `openjiuwen/core/runner/spawn/` (`agent_config.py`'s `SpawnAgentConfig`/`ClassAgentSpawnConfig`/`SpawnAgentKind`, `protocol.py`'s `Message`/`MessageType`, `process_manager.py`'s handle surface) and defers `child_process.py` and the real `python -m` child (see the crate-level `## Deferred Python Behavior`).

## API Surface

```rust
// mod.rs
pub enum SpawnAgentKind { ClassAgent, TeamAgent }          // Python SpawnAgentKind
pub struct SpawnSpec {                                     // serde, snake_case, unknown fields tolerated
    pub agent_kind: SpawnAgentKind,
    pub agent_module: Option<String>,                      // ClassAgentSpawnConfig.agent_module
    pub agent_class: Option<String>,                       // ClassAgentSpawnConfig.agent_class
    pub init_kwargs: Value,                                // ClassAgentSpawnConfig.init_kwargs (default {})
    pub payload: Value,                                    // TeamAgentSpawnConfig payload
    pub session_id: Option<String>,
    pub health_check_interval_secs: f64,                   // 5.0
    pub shutdown_timeout_secs: f64,                        // 10.0
    pub envs: BTreeMap<String, String>,
}
impl SpawnSpec { pub fn class_agent(module: impl Into<String>, class: impl Into<String>) -> Self;
                 pub fn team_agent(payload: Value) -> Self;
                 pub fn with_session_id / with_init_kwargs / with_env }
pub trait SpawnedAgent: Send + Sync {
    fn id(&self) -> &str;
    fn is_alive(&self) -> bool;
    fn pid(&self) -> Option<u32>;
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self, timeout: Duration) -> Result<bool>;   // false when it had to be killed
    async fn force_kill(&self) -> Result<()>;
}
#[async_trait] pub trait AgentSpawner: Send + Sync {
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
pub struct UnsupportedSpawner;                             // every spawn -> Error::Unsupported(<message>)

// protocol.rs — the NDJSON wire a real implementation uses (Python spawn/protocol.py)
pub enum SpawnMessageKind { Input, Output, HealthCheck, HealthCheckResponse, Shutdown, ShutdownAck, Error, StreamChunk, Done }
pub struct SpawnMessage { pub kind: SpawnMessageKind, pub payload: Value, pub timestamp: f64, pub message_id: String }
impl SpawnMessage {
    pub fn new(kind: SpawnMessageKind, payload: Value) -> Self;   // message_id = uuid v4, timestamp = unix secs
    pub fn to_line(&self) -> Result<String>;                      // JSON + "\n", never contains a raw newline
    pub fn from_line(line: &str) -> Result<Self>;                 // trims, rejects blank/non-JSON
}
```
`Runner::set_spawner` installs an `Arc<dyn AgentSpawner>`; `Runner::spawn_agent(spec)` delegates to it and returns `Error::Unsupported` when none is installed (the default `UnsupportedSpawner`).

## Constraints

- Nothing here starts a process, reads stdin, or writes stdout; this module is data + traits only.
- `SpawnSpec`/`SpawnMessage` field names and defaults mirror Python so a future backend can accept Python-shaped configuration; newline framing is `\n`-delimited UTF-8 JSON (Python NDJSON) and `to_line` must reject payloads whose serialization would break the framing.
- Errors are `Error::Unsupported` (no spawner / deferred feature) or `Error::Validation` (malformed protocol line).
- Tests: inline unit tests for spec defaults, line round-trips, and the unsupported spawner message; no process spawning.

## Routing Table

- `./src/spawn/mod.rs` → `SpawnAgentKind`, `SpawnSpec`, `SpawnedAgent`, `AgentSpawner`, `UnsupportedSpawner`
- `./src/spawn/protocol.rs` → `SpawnMessage`, `SpawnMessageKind`, NDJSON framing
