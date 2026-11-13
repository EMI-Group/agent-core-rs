# spawn/ — the runner's subprocess and remote spawn seam

## Intent

The seam through which a `Runner` drives an agent that lives in another process or behind an HTTP endpoint.
Ports the surface of Python `openjiuwen/core/runner/spawn/` (`agent_config.py`, `protocol.py`, `process_manager.py`) plus the HTTP client surface of `openjiuwen/core/runner/drunner/remote_client/` (`remote_client.py`, `remote_client_config.py`, `remote_agent.py`).
The transport-neutral seam is `AgentSpawner` → `SpawnedAgent` over a JSON-safe `SpawnSpec`; two backends implement it.
`SubprocessSpawner` launches a real child with `tokio::process` and drives it over its stdio pipes.
`RemoteSpawner` / `RemoteAgent` address an agent served over HTTP with `reqwest`.
Both speak the NDJSON frame vocabulary of `protocol` over the bounded line framing of `frames`.
`UnsupportedSpawner` remains the documented default while no backend is installed on a runner.
The deferred transports are typed, not silent: Pulsar/ZeroMQ reply-topic routing (`RemoteProtocol::Mq`), the A2A remote transport (`RemoteProtocol::A2a`), Python's `python -m` child, its `process_manager` pool, and the drunner reply-topic plumbing all report `Error::Unsupported` with a message naming the seam.

## API Surface

### Seam — `mod.rs`

```rust
pub enum SpawnAgentKind { ClassAgent, TeamAgent }          // serde snake_case
pub struct SpawnSpec {                                     // serde snake_case, default, unknown fields tolerated
    pub agent_kind: SpawnAgentKind,                        // ClassAgent
    pub agent_module: Option<String>,
    pub agent_class: Option<String>,
    pub init_kwargs: Value,                                // {}
    pub payload: Value,                                    // null
    pub session_id: Option<String>,
    pub health_check_interval_secs: f64,                   // 5.0
    pub shutdown_timeout_secs: f64,                        // 10.0
    pub envs: BTreeMap<String, String>,
}
impl SpawnSpec { pub fn class_agent(module, class) -> Self; pub fn team_agent(payload) -> Self;
                 pub fn with_session_id / with_init_kwargs / with_env }
#[async_trait] pub trait SpawnedAgent: Send + Sync {
    fn id(&self) -> &str; fn is_alive(&self) -> bool; fn pid(&self) -> Option<u32>;
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self, timeout: Duration) -> Result<bool>;   // false when it had to be killed
    async fn force_kill(&self) -> Result<()>;
}
#[async_trait] pub trait AgentSpawner: Send + Sync { async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>; }
pub struct UnsupportedSpawner;                             // every spawn -> Error::Unsupported(<message naming Runner::set_spawner>)
```

### Wire — `protocol.rs`

```rust
pub enum SpawnMessageKind { Input, Output, HealthCheck, HealthCheckResponse, Shutdown, ShutdownAck, Error, StreamChunk, Done }
pub struct SpawnMessage { pub kind: SpawnMessageKind, pub payload: Value, pub timestamp: f64, pub message_id: String }
impl SpawnMessage {
    pub fn new(kind, payload) -> Self;      // message_id = uuid v4, timestamp = unix secs
    pub fn to_line(&self) -> Result<String>;   // JSON + "\n"; Validation on unserializable or raw newline
    pub fn from_line(line: &str) -> Result<Self>;  // trims; Validation on blank / non-JSON / missing field / unknown kind
}
```

`timestamp` is `f64` unix seconds; compare round-tripped frames with a tolerance (~1e-6) because `serde_json`'s float parser is not correctly rounded.
Wire names are `snake_case` (`health_check_response`), not Python's upper-case `MessageType` values.

### Framing — `frames.rs`

`DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024`.
`read_frame(reader, max_frame_bytes) -> Result<Option<SpawnMessage>>` — skips non-JSON noise, returns `Ok(None)` at EOF, parses an unterminated trailing line.
`parse_frame_line(bytes, max_frame_bytes) -> Result<Option<SpawnMessage>>` — `Ok(None)` for blank or non-JSON lines, `Validation` for over-cap / non-UTF-8 / JSON-that-is-not-a-frame.
`frame_bytes(&SpawnMessage) -> Result<Vec<u8>>` — `to_line()` as bytes.
The cap is enforced through `fill_buf`/`consume`, so only consumed bytes are accumulated and a runaway line fails instead of allocating.

### Subprocess backend — `subprocess.rs`

```rust
pub struct SubprocessConfig {                              // serde snake_case, default
    pub program: String,                                      // "" — empty is Error::Config at spawn
    pub args: Vec<String>,                                    // []
    pub cwd: Option<PathBuf>,                                 // None = inherit the parent's
    pub env: BTreeMap<String, String>,                        // {} added on top of the parent env
    pub request_timeout_secs: f64,                            // 30.0
    pub shutdown_timeout_secs: f64,                           // 10.0
    pub health_check_interval_secs: f64,                      // 5.0, 0.0 disables the background task
    pub health_check_timeout_secs: f64,                       // 5.0
    pub max_health_failures: u32,                             // 3, 0 never kills on health
    pub max_frame_bytes: usize,                               // 8 MiB
}
impl SubprocessConfig { pub fn new(program); with_arg / with_args / with_cwd / with_env /
                        with_request_timeout / with_shutdown_timeout / with_health_check / with_max_frame_bytes }
pub struct SubprocessSpawner;                              // new(config), config(), spawn_process(spec) -> Result<Arc<SpawnedProcess>>
pub struct SpawnedProcess;                                  // id, pid, is_alive, is_healthy, health, send, stream,
                                                            // shutdown, force_kill, into_agent(self: Arc<Self>) -> Arc<dyn SpawnedAgent>
```

`SubprocessSpawner::spawn_process` returns the concrete `Arc<SpawnedProcess>` so hosts can call `health`/`pid`; the `AgentSpawner::spawn` impl coerces it to `Arc<dyn SpawnedAgent>`.
`SpawnedProcess` implements `SpawnedAgent` directly, so both paths expose the same behaviour.
`is_healthy` is the last observed probe result (fresh child = `true`); `health()` runs one probe synchronously.
Configured timeouts are clamped to `[0, 86_400]` seconds, so `Instant + Duration` cannot overflow and nonsense configuration means "no deadline".
`SubprocessConfig` env wins over the inherited environment and `SpawnSpec::envs` wins over `SubprocessConfig` env.

### Remote backend — `remote.rs`

```rust
pub const DEFAULT_INVOKE_PATH: &str = "agents/{agent_id}/invoke";
pub const DEFAULT_STREAM_PATH: &str = "agents/{agent_id}/stream";
pub const DEFAULT_CANCEL_PATH: &str = "agents/{agent_id}/cancel";
pub const DEFAULT_REMOTE_TIMEOUT_SECS: f64 = 60.0;

pub enum RemoteProtocol { Http, Mq, A2a }                  // serde snake_case; as_str(), client(config)
pub struct RemoteAgentConfig {                             // serde snake_case, default
    pub agent_id: String,                                     // ""
    pub base_url: String,                                     // "" — empty is Error::Validation
    pub invoke_path: String,                                  // "agents/{agent_id}/invoke"
    pub stream_path: String,                                  // "agents/{agent_id}/stream"
    pub cancel_path: Option<String>,                          // Some("agents/{agent_id}/cancel")
    pub session_id: Option<String>,                           // None
    pub timeout_secs: f64,                                    // 60.0
    pub headers: BTreeMap<String, String>,                    // {}
}
impl RemoteAgentConfig { pub fn new(agent_id, base_url); with_paths / with_cancel_path / with_session_id / with_timeout / with_header;
                         pub fn effective_timeout(&self) -> Duration; // 60 s when zero / negative / non-finite
                         pub fn invoke_url / stream_url / cancel_url(&self) -> Result<...> }
pub fn create_remote_client(protocol, config) -> Result<Arc<dyn RemoteClient>>;
#[async_trait] pub trait RemoteClient: Send + Sync {
    async fn start(&self) -> Result<()>; async fn stop(&self) -> Result<()>;
    fn is_started(&self) -> bool; fn is_stopped(&self) -> bool;
    async fn invoke(&self, inputs: Value, timeout: Option<Duration>) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value, timeout: Option<Duration>) -> BoxStream<'static, Result<AgentChunk>>;
    async fn cancel_task(&self, task_id: &str) -> Result<()>;
}
pub struct RemoteAgent;                                    // new(config), with_protocol(config, protocol), config(), agent_id(), protocol()
pub struct RemoteSpawner;                                  // new(base_url), with_timeout, with_header, base_url(), config_for(agent_id)
```

`RemoteAgent` implements both `RemoteClient` and `SpawnedAgent`.
`RemoteSpawner::spawn` derives the agent id from `agent_class`, then `agent_module`, then a non-empty `init_kwargs["agent_id"]` (`Validation` when none is usable), copies `SpawnSpec::session_id` into the config, and validates `base_url` at spawn time rather than at first invoke.

URL layout: the endpoint is `{base_url}/{path}` with `{agent_id}` substituted into the path; extra slashes are collapsed outside the scheme and a missing or extra slash between base and path is tolerated, so `("http://host:8080/", "/agents/{agent_id}/invoke")` and `("http://host:8080", "agents/{agent_id}/invoke")` build the same URL.
A path that contains `{agent_id}` requires a non-empty agent id (`Validation`), and an empty `base_url` is `Validation`.

## Wire Contract

Frames travel one JSON object per line.

| Direction | Kind | Payload |
| --- | --- | --- |
| parent → child, first | `input` | `{"spawn_spec": <SpawnSpec>}` (bootstrap) |
| parent → child, per invocation | `input` | `{"inputs": <inputs>}` |
| child → parent | `output` | an `AgentOutput` object; ends a `send` |
| child → parent | `stream_chunk` | an `AgentChunk` object |
| child → parent | `done` | ends a stream; a non-null payload is also accepted as `send`'s terminal output |
| either | `error` | `{"message": …}` or a bare string (anything else is rendered as JSON) → `Error::Transport` naming the child |
| parent → child | `health_check` | `null`; answered by `health_check_response`, whose payload counts as healthy for `null`/`true`/no `healthy` field |
| parent → child | `shutdown` | `null`; answered by `shutdown_ack` (EOF before the ack is acceptable) |

Stdout/body handling, identical on both backends: blank and non-JSON lines are skipped at `trace` (a child's stray logging); a line that is JSON but not a frame, not valid UTF-8, or longer than `max_frame_bytes` is `Error::Validation`; an unterminated trailing line is still parsed.
Any other frame arriving in the wrong direction is ignored at `trace` (Python's reader tolerates stray frames).

HTTP: `POST {base_url}/{invoke_path}` and `POST {base_url}/{stream_path}` with body `{"session_id": <session_id|null>, "inputs": <inputs>}`; `POST {base_url}/{cancel_path}` with `{"task_id": <id>}`.
The stream endpoint answers one frame per line, parsed by the table above.
Body lines that are not JSON are skipped; a `stream_chunk` that is not an `AgentChunk` is `Validation` and ends the stream; an `error` frame ends the stream after one `Err`.
Status mapping: `404` → `Error::NotFound`, any other non-2xx → `Error::Transport` with the status and a whitespace-collapsed body snippet (≤ 256 chars), a `reqwest` timeout → `Error::Timeout`, a non-JSON success body or an invalid `AgentOutput` → `Error::Validation`, an invalid configured header → `Error::Config`.

## Constraints

- Errors are the crate vocabulary from `crate::error` (`transport`, `unsupported`, `run_timeout`, `not_started`) plus `Error::{Config, Validation, NotFound, Timeout, Unsupported}`; never `External`, never a panic on user input.
- No `unsafe`, no `println!`/`eprintln!`; logging is `tracing` with target `openjiuwen_runner::spawn`.
- `frames.rs` owns no pipe and `protocol.rs` no I/O: both are pure functions over `AsyncBufRead`/bytes.
- Files stay ≤ ~1000 lines; the oversized inline unit-test modules live in `subprocess_tests.rs` and `remote_tests.rs`, declared with `#[cfg(test)] #[path = "….rs"] mod tests;`.
- Tests are offline: unit tests inline (moved as above), the integration suites in `tests/` use a re-executed fake child and a loopback `wiremock` server.

## Design Decisions

- **One exchange at a time per handle.** The protocol carries no request correlation id (`message_id` is generated per frame but nothing correlates on it), so a handle serialises exchanges with one fair `tokio::sync::Mutex` request guard; callers queue rather than interleave frames.
- **The desync rule.** An exchange that ends without consuming its terminator — an abandoned stream, a request deadline, or a health probe that hits its deadline — leaves frames unread, so the handle is marked out of sync *before* the lock is released and every later `send`/`stream`/`health` fails with `Error::Validation` naming the child; the caller respawns.
- **Kill on drop.** `tokio::process::Child::drop` does not kill and can leak a zombie, so dropping the last handle reference (its `Drop`) arms `start_kill` plus a detached reap, with `kill_on_drop(true)` as the final net when no runtime is available.
- **stderr is always drained.** A child that fills its stderr pipe would block, so a task drains it into the debug log (first 512 chars per line) and is aborted with the handle.
- **The health task holds a `Weak`.** It upgrades per tick, so it can never outlive or resurrect the handle; it is aborted on drop and on shutdown, and an interval of `0` disables it entirely (`health()` still works).
- **Health escalation.** `max_health_failures` consecutive failed probes kill the child and mark it unhealthy; `0` disables killing; a healthy probe resets the run.
- **A failed bootstrap is fatal.** When the bootstrap frame cannot be written, the child is killed and the error returned instead of handing back a broken handle.
- **Remote liveness is client state.** `RemoteAgent::pid()` is always `None`, `is_alive` is "the client is started", and `shutdown`/`force_kill` are just `stop()` (a remote agent always reports a graceful exit, `true`).
- **Auto-start.** The `reqwest` client is built lazily on first `invoke`/`stream` (or by `start`), so a caller need not wire a lifecycle; the built client is cloned out of the lock before any await, so no lock is held across a request.
- **Per-call timeouts.** `invoke`/`stream` take an optional `Duration` that overrides the configured deadline for that request only.
- **No cancel endpoint, no cancel.** `cancel_task` without a configured `cancel_path` is `Error::Unsupported`, never a silent no-op.
- **Backends are state-free spawners.** `SubprocessSpawner` and `RemoteSpawner` hold configuration only; every `spawn` builds an independent child or client.

## Deferred

- **Pulsar/ZeroMQ reply-topic routing** — `RemoteProtocol::Mq` (Python's `drunner` MQ transport); `create_remote_client` returns `Error::Unsupported` naming it.
- **A2A remote transport** — `RemoteProtocol::A2a`; `Error::Unsupported` pointing at `openjiuwen-extensions`' `a2a` feature.
- **Python's `python -m …` child** (`child_process.py`), its `process_manager` pool and health-check escalation machinery, and the drunner `mq_remote_client`/`RemoteAgent` reply-topic plumbing — not ported; the subprocess backend here launches whatever `SubprocessConfig::program` is and supervises only the child it started.
- The crate-level `## Deferred Python Behavior` in `crates/openjiuwen-runner/CONTEXT.md` still lists subprocess spawn and distributed reply-topic routing as deferred and is reconciled by the parent node.

## Routing Table

- `./src/spawn/mod.rs` → `SpawnAgentKind`, `SpawnSpec`, `SpawnedAgent`, `AgentSpawner`, `UnsupportedSpawner`, module re-exports, seam docs
- `./src/spawn/protocol.rs` → `SpawnMessage`, `SpawnMessageKind`, NDJSON line serialization
- `./src/spawn/frames.rs` → bounded line framing, `read_frame`, `parse_frame_line`, `frame_bytes`, `DEFAULT_MAX_FRAME_BYTES`
- `./src/spawn/subprocess.rs` → `SubprocessSpawner`, `SpawnedProcess`, request guard/desync, health task, stderr drain, kill-on-drop, wire helpers
- `./src/spawn/subprocess_config.rs` → `SubprocessConfig` (serde config, builders, clamped deadlines), private to the module and re-exported as `spawn::subprocess::SubprocessConfig`
- `./src/spawn/subprocess_tests.rs` → inline unit tests of `subprocess.rs` (loaded via `#[path]`)
- `./src/spawn/remote.rs` → `RemoteAgentConfig`, `RemoteProtocol`, `RemoteClient`, `RemoteAgent`, `create_remote_client`, `RemoteSpawner`, URL building, stream parsing, status mapping
- `./src/spawn/remote_tests.rs` → inline unit tests of `remote.rs` (loaded via `#[path]`)

## Notes for Agents

- Install a backend with `Runner::set_spawner`; `UnsupportedSpawner` stays the default, so an unconfigured runner fails a spawn with `Error::Unsupported`.
- The crate-level flat re-exports in `src/lib.rs` are maintained outside this node; this module's own `pub use` group is the source of truth for names.
- Integration suites: `tests/spawn_subprocess.rs` re-executes the test binary as the fake child (`FAKE_CHILD_MODE` selects echo/stream/crash/silent/malformed/oversized/health-failure/shutdown modes), and `tests/spawn_remote.rs` drives a loopback `wiremock` server.
- The crate-level `## Deferred Python Behavior` and `tests/CONTEXT.md` still describe the pre-backend deferral and are reconciled by the parent node.
- A handle is single-use once desynchronized: respawn rather than retry, and never assume a timed-out exchange can be recovered.
- There is no child-side serve helper (no `serve_child_stdio`): a host's child program must read/write NDJSON itself — read one `input` bootstrap frame (`{"spawn_spec": …}`), then answer `input` (`{"inputs": …}`) frames with `output`/`stream_chunk`/`done`/`error` and reply to `health_check`/`shutdown`; `SpawnMessage::from_line`/`to_line` and `read_frame`/`frame_bytes` are reusable on that side.
- The reference child implementation is `tests/spawn_subprocess.rs`'s `#[ignore]`d `fake_child` libtest case re-executed with `fake_child --exact --ignored --test-threads=1 --nocapture` (`--nocapture` is mandatory or libtest redirects the child's stdout and no frame reaches the pipe).
- `SubprocessConfig::program` empty is `Error::Config` at spawn, not at construction, so a host can build a config incrementally.
