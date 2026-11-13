# spawn/ — the runner's subprocess and remote spawn seam

## Intent

The seam through which a `Runner` drives an agent that lives in another process or behind an HTTP endpoint.
Ports the surface of Python `openjiuwen/core/runner/spawn/` (`agent_config.py`, `protocol.py`, `process_manager.py`) plus the HTTP client surface of `openjiuwen/core/runner/drunner/remote_client/` (`remote_client.py`, `remote_client_config.py`, `remote_agent.py`).
The transport-neutral seam is `AgentSpawner` → `SpawnedAgent` over a JSON-safe `SpawnSpec`; four backends implement it, each naming itself with a `SpawnBackend`.
`SubprocessSpawner` launches a real child with `tokio::process` and drives it over its stdio pipes.
`ExternalCommandSpawner` resolves a host-configured program on `PATH` and delegates to that same subprocess backend, labelling its children `SpawnBackend::ExternalCommand`.
`InProcessSpawner` wraps a host-bound `InProcessHandler`, so an agent living in this process looks like any other handle.
`RemoteSpawner` / `RemoteAgent` address an agent served over HTTP with `reqwest`.
Every backend speaks the NDJSON frame vocabulary of `protocol` over the bounded line framing of `frames`.
`SpawnerRegistry` hosts several backends side by side and routes a `SpawnSpec` to one of them; `DeferredSpawner` answers for every backend that is not registered.
`MonitoredSpawnedAgent` decorates any handle so one spawned child's lifecycle is reported into the runner's `RunMonitor` as a `RunKind::Spawn` run.
`UnsupportedSpawner` remains the documented default while no backend is installed on a runner.
The deferred transports are typed, not silent: Pulsar/ZeroMQ reply-topic routing (`RemoteProtocol::Mq`), the A2A remote transport (`RemoteProtocol::A2a`), Python's `python -m` child, its `process_manager` pool, and the drunner reply-topic plumbing all report `Error::Unsupported` with a message naming the seam.

## API Surface

### Seam — `mod.rs`

```rust
pub enum SpawnAgentKind { ClassAgent, TeamAgent }          // serde snake_case
pub enum SpawnBackend {                                    // serde snake_case; Copy/Ord/Hash, as_str(), is_implemented()
    Unknown,          // default; a handle or spawner that names no backend
    Subprocess, ExternalCommand, InProcess, RemoteHttp,    // implemented
    RemoteMq,         // Python's drunner Pulsar/ZeroMQ reply-topic transport (deferred)
    RemoteA2a,        // the A2A remote transport, which lives in `openjiuwen-extensions` (deferred)
}                                                          // Display == as_str()
pub struct SpawnHealth {                                   // serde snake_case, default, unknown fields tolerated
    pub backend: SpawnBackend,                             // Unknown
    pub alive: bool,                                       // false
    pub healthy: bool,                                     // false
    pub pid: Option<u32>,                                  // None
    pub last_error: Option<String>,                        // None
}                                                          // + healthy(backend, pid) -> Self
pub struct SpawnSpec {                                     // serde snake_case, default, unknown fields tolerated
    pub agent_kind: SpawnAgentKind,                        // ClassAgent
    pub agent_module: Option<String>,
    pub agent_class: Option<String>,
    pub init_kwargs: Value,                                // {}
    pub payload: Value,                                    // null
    pub backend: Option<SpawnBackend>,                     // None; omitted from the wire object when unset
    pub session_id: Option<String>,
    pub health_check_interval_secs: f64,                   // 5.0
    pub shutdown_timeout_secs: f64,                        // 10.0
    pub envs: BTreeMap<String, String>,
}
impl SpawnSpec { pub fn class_agent(module, class) -> Self; pub fn team_agent(payload) -> Self;
                 pub fn with_session_id / with_init_kwargs / with_env / with_backend }
#[async_trait] pub trait SpawnedAgent: Send + Sync {
    // required
    fn id(&self) -> &str; fn is_alive(&self) -> bool; fn pid(&self) -> Option<u32>;
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self, timeout: Duration) -> Result<bool>;   // false when it had to be killed
    async fn force_kill(&self) -> Result<()>;
    // defaulted (object-safe, so no implementor breaks)
    fn backend(&self) -> SpawnBackend;                     // Unknown
    fn is_healthy(&self) -> bool;                          // == is_alive(): no probe ⇒ liveness is the answer
    fn health_snapshot(&self) -> SpawnHealth;              // backend/is_alive/is_healthy/pid, last_error: None
    async fn health(&self) -> Result<bool>;                // Unsupported("agent health probe", <backend> hint)
    async fn cancel_task(&self, task_id: &str) -> Result<()>;  // Unsupported("spawned task cancellation", <backend> hint)
    async fn send_with_timeout(&self, inputs, timeout: Option<Duration>) -> Result<AgentOutput>;   // default: send(inputs)
    fn stream_with_timeout(&self, inputs, timeout: Option<Duration>) -> BoxStream<...>;            // default: stream(inputs)
}
#[async_trait] pub trait AgentSpawner: Send + Sync {
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
    fn backend(&self) -> SpawnBackend;                     // Unknown
}
pub struct UnsupportedSpawner;                             // every spawn -> Error::Unsupported(<message naming Runner::set_spawner>)
```

`SpawnSpec::backend` is a routing hint for `SpawnerRegistry`, not a wire change: it is additive metadata a Python child ignores (unknown fields are tolerated) and it never alters the frames.
A minimal `SpawnedAgent` (only the required methods) reports `SpawnBackend::Unknown`, `is_healthy() == is_alive()`, an `Unsupported` `health`/`cancel_task` whose message names its backend, and ignores the `send_with_timeout`/`stream_with_timeout` deadline hint (they call `send`/`stream`).
The deadline on the bounded variants is advisory and never reaches the wire: `None` means "the backend's own default", `SpawnedProcess` and `RemoteAgent` honour a `Some(..)`, and `InProcessAgent` keeps the defaults.

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
pub struct SubprocessSpawner;                              // new(config) / with_backend(config, backend), config(), backend(),
                                                            // spawn_process(spec) -> Result<Arc<SpawnedProcess>>
pub struct SpawnedProcess;                                  // id, backend, pid, is_alive, is_healthy, health, send, stream,
                                                            // send_with_timeout, stream_with_timeout, shutdown, force_kill,
                                                            // into_agent(self: Arc<Self>) -> Arc<dyn SpawnedAgent>
```

`SubprocessSpawner::spawn_process` returns the concrete `Arc<SpawnedProcess>` so hosts can call `health`/`pid`; the `AgentSpawner::spawn` impl coerces it to `Arc<dyn SpawnedAgent>`.
`SubprocessSpawner::new(config)` labels its children `SpawnBackend::Subprocess`, while `SubprocessSpawner::with_backend(config, backend)` is the wrapper form: a backend that is this protocol plus a step of its own labels its children with its own name, and both `Spawner::backend()` (the spawner's own `backend` field) and every child's handle report that label.
`SpawnedProcess` implements `SpawnedAgent` directly, so both paths expose the same behaviour, including the trait's health accessors: `backend()` is the spawner's label (not always `subprocess`), `is_healthy()` reads the same atomic the background task updates, `health()` runs one probe, and `health_snapshot()` carries the same label and leaves `last_error` empty (a failed probe is an `Err` from `health`, not a fabricated snapshot field).
The handle's `Debug` output shows that label beside its id, pid, liveness, and health.
`cancel_task` keeps the unsupported default: the NDJSON protocol has no task handle to address, so a caller stops a child with `shutdown`/`force_kill`.
`send_with_timeout(inputs, timeout)`/`stream_with_timeout(inputs, timeout)` bound one exchange: `Some(..)` is that exchange's deadline and `None` falls back to the configured `request_timeout_secs` (`send`/`stream` are the `None` form), so a host holding `Arc<dyn SpawnedAgent>` can bound a single invocation.
`shutdown(timeout)` treats `Duration::ZERO` as "the configured `shutdown_timeout_secs`" (read through the clamped `SubprocessConfig::shutdown_timeout()`); a non-zero caller timeout wins over the configuration, and a child that does not exit within it is killed and reported `false`.
`is_healthy` is the last observed probe result (fresh child = `true`); `health()` runs one probe synchronously.
Configured timeouts are clamped to `[0, 86_400]` seconds, so `Instant + Duration` cannot overflow and nonsense configuration means "no deadline".
`SubprocessConfig` env wins over the inherited environment and `SpawnSpec::envs` wins over `SubprocessConfig` env.

### External-command backend — `external.rs`

```rust
pub struct ExternalCommandSpawner { command: String, config: SubprocessConfig, search_path: Option<PathBuf> }
impl ExternalCommandSpawner {
    pub fn new(command) -> Self; pub fn command(&self) -> &str; pub fn config(&self) -> &SubprocessConfig;
    pub fn with_args(Vec<String>) / with_arg / with_env / with_cwd / with_search_path(PathBuf) /
           with_request_timeout / with_shutdown_timeout / with_health_check(interval, timeout, max_failures) /
           with_max_frame_bytes;
    pub fn resolve(&self) -> Result<PathBuf>;
}
```

`resolve` is the whole backend: an empty (or whitespace-only) command is `Error::Config` naming `ExternalCommandSpawner::command`; a command containing a path separator or an absolute path must be an existing regular file and, on unix, carry an executable bit, else `Error::NotFound` naming the command and the reason; a bare name is looked up in `with_search_path` (used instead of `PATH`, unsplit, for hermetic hosts) or in `PATH` split with `std::env::split_paths`, and the first candidate passing the same checks wins.
A bare name that matches nothing is `Error::NotFound` naming the command and how many directories were searched (or that none is configured).
`spawn` resolves, clones the config with `program` set to the resolved path, and delegates to `SubprocessSpawner::with_backend(config, SpawnBackend::ExternalCommand).spawn(spec)`: process handling, framing, timeouts and the health task are the subprocess backend's, and the label makes it the external-command backend — so `Spawner::backend()`, the child's `backend()`/`health_snapshot()`, and a runner's monitor detail all report `external_command`.

### In-process backend — `in_process.rs`

```rust
#[async_trait] pub trait InProcessHandler: Send + Sync {
    fn is_alive(&self) -> bool;                            // true
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self) -> Result<()>;
}
pub type InProcessFactory = Arc<dyn Fn(&SpawnSpec) -> Result<Arc<dyn InProcessHandler>> + Send + Sync>;
pub struct InProcessSpawner;                               // new(factory), from_handler(Arc<dyn InProcessHandler>)
pub struct InProcessAgent;                                 // id, spec(), handler(), the trait methods
```

`new` calls the factory once per spawn, so a spec can select the bound agent; `from_handler` shares one already-built handler across every spawn.
Each spawn wraps its handler in a fresh `InProcessAgent` with the id `in-process-<uuid v4>`.
`is_alive` is this handle's alive flag *and* `handler.is_alive()`, `pid()` is always `None`, `backend()` is `InProcess`, `is_healthy()`/`health()` report liveness (there is no probe in-process), and `send`/`stream` delegate to the handler.
`shutdown` calls `handler.shutdown()` then marks the handle dead and returns `Ok(true)`; `force_kill` does the same because an in-process handler is stopped, not signalled — there is no OS process — and both are idempotent because the handler's `shutdown` must be.
A factory `Err` propagates from `spawn` and launches nothing.

### Registry — `registry.rs`

```rust
pub struct DeferredSpawner;                                // for_backend(backend), new(backend, reason, hint), backend_name()
pub struct SpawnerRegistry;                                // Debug lists backends + default
impl SpawnerRegistry {
    pub fn new() -> Self; pub fn with_deferred() -> Self;   // pre-registers RemoteMq + RemoteA2a
    pub fn register(backend, Arc<dyn AgentSpawner>) -> Option<Arc<dyn AgentSpawner>>;  // returns the replaced one
    pub fn register_subprocess(SubprocessConfig) / register_external_command(ExternalCommandSpawner) /
           register_in_process(InProcessSpawner) / register_remote(RemoteSpawner);
    pub fn unregister(backend) -> Option<Arc<dyn AgentSpawner>>;  // also clears a default that pointed at it
    pub fn get(backend) -> Option<Arc<dyn AgentSpawner>>; pub fn contains(backend) -> bool;
    pub fn len() -> usize; pub fn is_empty() -> bool; pub fn backends() -> Vec<SpawnBackend>;  // sorted
    pub fn set_default(backend) -> Result<()>;             // Validation when it is not registered
    pub fn default_backend() -> Option<SpawnBackend>;
    pub fn spawner_for(&SpawnSpec) -> Arc<dyn AgentSpawner>;   // never None
    pub async fn spawn(SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
    pub async fn spawn_with(SpawnBackend, SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
```

`DeferredSpawner::for_backend` is the wording table: `RemoteMq` → `"Pulsar/ZeroMQ spawn transport"` with a hint naming drunner reply-topic routing; `RemoteA2a` → `"a2a spawn transport"` with a hint naming `openjiuwen-extensions`' `a2a` feature; an implemented-but-unregistered backend → `"no <name> spawner registered"` with a hint naming the matching `register_*` method; anything else → ``"spawn backend `<name>`"``.
Every `spawn` on one is `Error::Unsupported` (`error::unsupported(reason, hint)`).

Routing precedence in `spawner_for`: `spec.backend` when set (registered spawner, else the matching `DeferredSpawner` — never a fallback to another backend) → the explicit default → the sole registered spawner → a `DeferredSpawner` with reason `"no default spawn backend"` (empty) or `"ambiguous spawn backend"` (more than one), whose hint names `set_default` and `SpawnSpec::backend`.
`spawn_with` resolves exactly the named backend and answers with `DeferredSpawner::for_backend` when it is not registered.
The registry holds one `parking_lot::RwLock` over `{spawners: BTreeMap<SpawnBackend, Arc<dyn AgentSpawner>>, default: Option<SpawnBackend>}`; the spawner `Arc` is cloned out before any `.await`, so no guard is held across one.

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
Through the seam it reports `backend() == RemoteHttp`, `pid() == None`, `is_healthy()` = the client's started state, `health()` = `Ok(is_started)` (the HTTP wire contract defines only invoke/stream/cancel, so there is no probe endpoint), and `cancel_task` forwards to `RemoteClient::cancel_task` (the one place a remote backend owns a real task handle).
`send_with_timeout`/`stream_with_timeout` forward the deadline to `RemoteClient::invoke`/`RemoteClient::stream`, so a bounded invocation is real on this backend while `send`/`stream` keep the configured `timeout_secs`.
`RemoteAgent::new` stays lazy (the `reqwest` client is built on first `invoke`/`stream` or by `start`, so a directly constructed handle reports itself not alive until then), but `RemoteSpawner::spawn` calls `RemoteClient::start` before returning: a spawned remote child is started, so `is_alive()`/`is_healthy()`/`health()` are `true` immediately with no request sent (Python's `remote_agent.py` starts the child too), and a client that cannot be built fails the spawn.
`RemoteSpawner::backend()` is `RemoteHttp`.
`RemoteSpawner::spawn` derives the agent id from `agent_class`, then `agent_module`, then a non-empty `init_kwargs["agent_id"]` (`Validation` when none is usable), copies `SpawnSpec::session_id` into the config, and validates the endpoint at spawn time rather than at first invoke: an empty `base_url` and a non-empty one that is still not an absolute URL both fail there (`Validation`), as does a client that cannot be built (an invalid configured header).

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
- Files stay ≤ ~1000 lines; the oversized inline unit-test modules live in `mod_tests.rs`, `subprocess_tests.rs` and `remote_tests.rs`, declared with `#[cfg(test)] #[path = "….rs"] mod tests;`.
- Tests are offline: unit tests inline (moved as above), the integration suites in `tests/` use a re-executed fake child and a loopback `wiremock` server.

## Design Decisions

- **One exchange at a time per handle.** The protocol carries no request correlation id (`message_id` is generated per frame but nothing correlates on it), so a handle serialises exchanges with one fair `tokio::sync::Mutex` request guard; callers queue rather than interleave frames.
- **The desync rule.** An exchange that ends without consuming its terminator — an abandoned stream, a request deadline, or a health probe that hits its deadline — leaves frames unread, so the handle is marked out of sync *before* the lock is released and every later `send`/`stream`/`health` fails with `Error::Validation` naming the child; the caller respawns.
- **Kill on drop.** `tokio::process::Child::drop` does not kill and can leak a zombie, so dropping the last handle reference (its `Drop`) arms `start_kill` plus a detached reap, with `kill_on_drop(true)` as the final net when no runtime is available.
- **stderr is always drained.** A child that fills its stderr pipe would block, so a task drains it into the debug log (first 512 chars per line) and is aborted with the handle.
- **The health task holds a `Weak`.** It upgrades per tick, so it can never outlive or resurrect the handle; it is aborted on drop and on shutdown, and an interval of `0` disables it entirely (`health()` still works).
- **Health escalation.** `max_health_failures` consecutive failed probes kill the child and mark it unhealthy; `0` disables killing; a healthy probe resets the run.
- **A failed bootstrap is fatal.** When the bootstrap frame cannot be written, the child is killed and the error returned instead of handing back a broken handle.
- **Caller deadlines are hints that two backends honour.** `send_with_timeout`/`stream_with_timeout` are defaulted on the trait (the default is plain `send`/`stream`), so adding the capability breaks no implementor; the subprocess backend falls back to `request_timeout_secs` for `None`, the remote backend to `timeout_secs`, and both are bounded per exchange rather than per handle.
- **A zero shutdown timeout means the configuration.** `SpawnedProcess::shutdown(Duration::ZERO)` uses the clamped `SubprocessConfig::shutdown_timeout()` (the `shutdown_timeout_secs` builder finally has an effect); a non-zero caller timeout wins. `ExternalCommandSpawner` inherits the rule because it delegates, and its docs say so.
- **Remote liveness is client state.** `RemoteAgent::pid()` is always `None`, `is_alive` is "the client is started", and `shutdown`/`force_kill` are just `stop()` (a remote agent always reports a graceful exit, `true`).
- **Auto-start, and start-at-spawn.** The `reqwest` client is built lazily on first `invoke`/`stream` (or by `start`), so a directly constructed `RemoteAgent` needs no lifecycle wiring; `RemoteSpawner::spawn` starts it immediately, so a host polling readiness through `Arc<dyn SpawnedAgent>` never sees a freshly spawned remote child as dead, and a client-construction failure surfaces at spawn. The built client is cloned out of the lock before any await, so no lock is held across a request.
- **Per-call timeouts.** `invoke`/`stream` take an optional `Duration` that overrides the configured deadline for that request only, and the seam's `send_with_timeout`/`stream_with_timeout` forward it; `None` everywhere means "the configured default".
- **No cancel endpoint, no cancel.** `cancel_task` without a configured `cancel_path` is `Error::Unsupported`, never a silent no-op.
- **Health is a seam-level accessor, not a downcast.** `SpawnedAgent` carries `backend`, `is_healthy`, `health_snapshot` and `health` beside `is_alive`, so a caller holding `Arc<dyn SpawnedAgent>` needs no knowledge of `SpawnedProcess`; every one of them has a default (`Unknown`, `is_alive`, the composition, and an `Unsupported` probe naming the backend), which keeps the trait object-safe and every existing implementor compiling.
- **Only liveness-backed health is promised by default.** "No probe ⇒ liveness is the answer" is the documented fallback; a backend that can probe (`SpawnedProcess`, `RemoteAgent`) overrides `health`/`is_healthy`, and one that cannot (`InProcessAgent` has nothing to ask) answers from its alive flag rather than inventing a probe.
- **Cancellation is per-backend, never simulated.** The NDJSON protocol has no request correlation id, so a subprocess handle keeps the `Unsupported` default for `cancel_task` and stops its child with `shutdown`/`force_kill`; only the HTTP client, which owns a real `cancel_path`, forwards it.
- **A deferred backend is a registered answer.** `SpawnerRegistry::with_deferred` stores `DeferredSpawner`s for `RemoteMq`/`RemoteA2a`, so routing is uniform: the missing transport is named in the `Unsupported` message (`Pulsar`/`a2a`) instead of surfacing as "nothing registered".
- **Routing never falls back silently.** A `SpawnSpec::backend` that is not registered is a deferred failure naming that backend, `spawn_with` resolves exactly what it was asked for, and only an *unspecified* backend reaches the default / sole-spawner / ambiguity rules.
- **Observation is a decorator, not a backend.** `MonitoredSpawnedAgent` wraps whatever handle a backend returned, so every backend — including a host's own `AgentSpawner` — reports the same `RunKind::Spawn` lifecycle without knowing the monitor exists; `is_healthy` stays a pure read and only `health` records, which keeps the poll path free of monitor churn.
- **Runner routing is registry-first and last-install-wins.** A `Runner` holds one spawner and an optional registry beside it: a registry routes `spawn_agent` (its `spawner_for` precedence), `spawn_agent_with` requires the registry and refuses to fall back, and `set_spawner` clears the registry while `set_spawner_registry` leaves the single spawner installed for `spawn_backend`/`spawn_agent`-without-registry reporting.
- **An external command is a resolved path, not a shell.** `ExternalCommandSpawner` refuses an empty command, a missing file, a directory, and (on unix) a file without an executable bit, then hands the absolute path to the subprocess backend; `PATH` is only a lookup, never an execution vehicle.
- **An in-process agent is stopped, not signalled.** `shutdown` and `force_kill` both call the handler's own `shutdown`, `pid()` is `None`, and liveness is the handle's flag *and* the handler's, so a stopped handler cannot be reported alive.
- **Backends are state-free spawners.** `SubprocessSpawner`, `ExternalCommandSpawner`, `InProcessSpawner` and `RemoteSpawner` hold configuration only; every `spawn` builds an independent child, client, or handle.

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
- `./src/spawn/external.rs` → `ExternalCommandSpawner` (command resolution on a search path / `PATH`, executable checks, delegation to the subprocess backend)
- `./src/spawn/in_process.rs` → `InProcessHandler`, `InProcessFactory`, `InProcessSpawner`, `InProcessAgent` (host-bound agent, no child)
- `./src/spawn/registry.rs` → `SpawnerRegistry` (multi-backend routing), `DeferredSpawner` (the wording table for undeferrable/undeferred backends)
- `./src/spawn/monitored.rs` → `MonitoredSpawnedAgent` (lifecycle into `RunMonitor` as `RunKind::Spawn`: admission, health observation, shutdown/kill/drop) and its unit tests
- `./src/spawn/mod_tests.rs` → inline unit tests of `mod.rs` (loaded via `#[path]`): wire names, `SpawnHealth`, `SpawnSpec::backend`, the defaulted accessors
- `./src/spawn/subprocess_tests.rs` → inline unit tests of `subprocess.rs` (loaded via `#[path]`)
- `./src/spawn/remote.rs` → `RemoteAgentConfig`, `RemoteProtocol`, `RemoteClient`, `RemoteAgent`, `create_remote_client`, `RemoteSpawner`, URL building, stream parsing, status mapping
- `./src/spawn/remote_tests.rs` → inline unit tests of `remote.rs` (loaded via `#[path]`)

## Notes for Agents

- Install a backend with `Runner::set_spawner`, or several with `Runner::set_spawner_registry`; `UnsupportedSpawner` stays the default, so an unconfigured runner fails a spawn with `Error::Unsupported`.
- Every successful `Runner::spawn_agent`/`Runner::spawn_agent_with` is wrapped in `MonitoredSpawnedAgent::track`, so `Runner::active_spawns` lists the live ones and a failed spawn creates no monitor entry.
- This module's own `pub use` group is the source of truth for names; `src/lib.rs` re-exports them flat (`SpawnBackend`, `SpawnHealth`, `SpawnerRegistry`, `DeferredSpawner`, the four backends, and `MonitoredSpawnedAgent`).
- Integration suites: `tests/spawn_subprocess.rs` re-executes the test binary as the fake child (`FAKE_CHILD_MODE` selects echo/stream/crash/silent/malformed/oversized/health-failure/shutdown modes), and `tests/spawn_remote.rs` drives a loopback `wiremock` server.
- The crate-level `## Deferred Python Behavior` and `tests/CONTEXT.md` still describe the pre-backend deferral and are reconciled by the parent node.
- A handle is single-use once desynchronized: respawn rather than retry, and never assume a timed-out exchange can be recovered.
- `send_with_timeout`/`stream_with_timeout` are the only way to bound *one* exchange: `send`/`stream` always use the backend's configured deadline, and the trait defaults ignore the hint (so an in-process handle or a host's own backend simply behaves like `send`/`stream`).
- `shutdown(Duration::ZERO)` is not "kill now": it means "use the configured `shutdown_timeout_secs`". Pass a non-zero duration to override the configuration for one shutdown.
- `RemoteSpawner::spawn` returns a *started* `RemoteAgent` (readiness is `true` without a request); `RemoteAgent::new` is lazy, so a directly constructed handle reports not-alive until its first request or an explicit `RemoteClient::start`.
- There is no child-side serve helper (no `serve_child_stdio`): a host's child program must read/write NDJSON itself — read one `input` bootstrap frame (`{"spawn_spec": …}`), then answer `input` (`{"inputs": …}`) frames with `output`/`stream_chunk`/`done`/`error` and reply to `health_check`/`shutdown`; `SpawnMessage::from_line`/`to_line` and `read_frame`/`frame_bytes` are reusable on that side.
- The reference child implementation is `tests/spawn_subprocess.rs`'s `#[ignore]`d `fake_child` libtest case re-executed with `fake_child --exact --ignored --test-threads=1 --nocapture` (`--nocapture` is mandatory or libtest redirects the child's stdout and no frame reaches the pipe).
- `SubprocessConfig::program` empty is `Error::Config` at spawn, not at construction, so a host can build a config incrementally.
- `SubprocessSpawner::with_backend(config, backend)` is how a wrapper backend labels its children; a `SpawnedProcess` reports its spawner's label, so `backend() == Subprocess` only for a plain `SubprocessSpawner::new`/`default`.
- The lib unit tests re-execute the lib test binary as the hermetic child: `subprocess_tests.rs`'s `#[ignore]`d `fake_child` is selected by `--exact` with the filter `<module_path minus the crate>::fake_child`; the child must close libtest's partial `test … ` line with a leading newline or the parent skips its first frame as noise.
- `ExternalCommandSpawner::with_search_path` is the hermetic form: it replaces `PATH` with one directory, so a test never depends on the machine's installed programs.
- `SpawnerRegistry` is the multi-backend host; the runner-level routing lives in `src/runner_spawn.rs` (`Runner::spawner`, `spawn_backend`, `spawn_backends`, `set_spawner`, `set_spawner_registry`, `spawner_registry`, `spawn_agent`, `spawn_agent_with`, `active_spawns`).
