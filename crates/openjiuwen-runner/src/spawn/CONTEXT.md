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
    async fn set_health_interval(&self, interval: Duration) -> Result<()>;  // Unsupported("runtime health-interval re-tuning", <backend> hint)
    async fn cancel_task(&self, task_id: &str) -> Result<()>;  // Unsupported("spawned task cancellation", <backend> hint)
    async fn cancel_task_with_request_id(&self, request_id: &str, reason: Option<&str>) -> Result<()>;  // default: cancel_task(request_id)
    async fn send_with_timeout(&self, inputs, timeout: Option<Duration>) -> Result<AgentOutput>;   // default: send(inputs)
    fn stream_with_timeout(&self, inputs, timeout: Option<Duration>) -> BoxStream<...>;            // default: stream(inputs)
}
#[async_trait] pub trait AgentSpawner: Send + Sync {
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
    fn backend(&self) -> SpawnBackend;                     // Unknown
    fn describe(&self) -> Value;                           // Value::Null; JSON-safe, secret-free summary
}
pub struct UnsupportedSpawner;                             // every spawn -> Error::Unsupported(<message naming Runner::set_spawner>)
                                                            // describe() -> Value::Null
```

`describe()` is the configuration introspection a host holding only `Arc<dyn AgentSpawner>` (from `Runner::spawner()` or `SpawnerRegistry::get`) can rely on: it is object-safe, defaulted to `Value::Null`, and JSON-safe for logging/diagnostics.
Its per-backend shapes are `{"backend": "subprocess", "program", "args", "cwd", "env_names", "request_timeout_secs", "shutdown_timeout_secs", "health_check_interval_secs", "health_check_timeout_secs", "max_health_failures", "max_frame_bytes"}`,
`{"backend": "external_command", "command", "search_path", "launch": <subprocess summary>}`,
`{"backend": "in_process", "hosted": true}` (a marker: a factory has no configuration),
`{"backend": "remote_http", "base_url", "timeout_secs", "header_names"}`, and
`{"backend": <deferred backend>, "deferred": true, "reason", "hint"}` for a `DeferredSpawner`.
No secret is ever reported: the environment and the HTTP headers appear as **names only** (`env_names`, `header_names`), so a configured `Authorization: Bearer <token>` or `API_KEY=<value>` never reaches the summary.

`SpawnSpec::backend` is a routing hint for `SpawnerRegistry`, not a wire change: it is additive metadata a Python child ignores (unknown fields are tolerated) and it never alters the frames.
A minimal `SpawnedAgent` (only the required methods) reports `SpawnBackend::Unknown`, `is_healthy() == is_alive()`, an `Unsupported` `health`/`set_health_interval`/`cancel_task`/`cancel_task_with_request_id` whose message names its backend, and ignores the `send_with_timeout`/`stream_with_timeout` deadline hint (they call `send`/`stream`).
Cancellation has two seam methods: `cancel_task(task_id)` is the convenience form (no reason) and `cancel_task_with_request_id(request_id, reason)` is the correlated one, defaulted to `cancel_task(request_id)` — so a backend whose wire carries only an id still correlates by id, and a backend with no cancellation propagates that method's `Unsupported` instead of succeeding, while a blank id is `Validation` and an id that matches nothing in flight is `NotFound` from the backend that owns the requests.
`set_health_interval` is the one seam method that re-configures a running backend instead of exchanging a message: only the process backends run a background health loop, so `InProcessAgent`, `RemoteAgent` and every host backend keep the default and report the gap by name — while a `MonitoredSpawnedAgent` over such a handle arms its own supervised loop, so the cadence is honoured where a host actually observes the spawn.
The deadline on the bounded variants is advisory and never reaches the wire: `None` means "the backend's own default", `SpawnedProcess` and `RemoteAgent` honour a `Some(..)`, and `InProcessAgent` keeps the defaults.

### Wire — `protocol.rs`

```rust
pub enum SpawnMessageKind { Input, Output, HealthCheck, HealthCheckResponse, Shutdown, ShutdownAck, Error, StreamChunk, Done, Cancel, CancelAck }
impl SpawnMessageKind { pub const fn as_str(self) -> &'static str }   // "cancel_ack" …
pub struct SpawnMessage { pub kind: SpawnMessageKind, pub payload: Value, pub timestamp: f64, pub message_id: String }
impl SpawnMessage {
    pub fn new(kind, payload) -> Self;      // message_id = uuid v4, timestamp = unix secs
    pub fn cancel(request_id: &str, reason: Option<&str>) -> Result<Self>;  // the Cancel frame
    pub fn to_line(&self) -> Result<String>;   // JSON + "\n"; Validation on unserializable or raw newline
    pub fn from_line(line: &str) -> Result<Self>;  // trims; Validation on blank / non-JSON / missing field / unknown kind
}
pub struct CancelRequest { pub request_id: String, pub reason: Option<String> }   // serde snake_case
impl CancelRequest {
    pub fn new(request_id, reason: Option<&str>) -> Result<Self>;  // Validation on a blank id
    pub fn to_frame(&self) -> SpawnMessage;                        // {"request_id": …, "reason": … | null}
    pub fn from_frame(frame: &SpawnMessage) -> Result<Self>;       // Validation on a non-cancel frame / malformed payload / blank id
}
```

`timestamp` is `f64` unix seconds; compare round-tripped frames with a tolerance (~1e-6) because `serde_json`'s float parser is not correctly rounded.
Wire names are `snake_case` (`health_check_response`, `cancel_ack`), not Python's upper-case `MessageType` values, and `SpawnMessageKind::as_str` is the same table for messages.
`cancel`/`cancel_ack` are the request-correlation pair: the parent names the in-flight request it wants cancelled (plus an optional reason) and the child echoes the same id back, so "received" is distinguishable from "never arrived". A blank id is `Validation` at both ends of the wire; an id that matches no request is conveyed faithfully (only the backend that owns the requests reports `NotFound`).

### Framing — `frames.rs`

`DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024`.
`read_frame(reader, max_frame_bytes) -> Result<Option<SpawnMessage>>` — skips non-JSON noise, returns `Ok(None)` at EOF, parses an unterminated trailing line.
`parse_frame_line(bytes, max_frame_bytes) -> Result<Option<SpawnMessage>>` — `Ok(None)` for blank or non-JSON lines, `Validation` for over-cap / non-UTF-8 / JSON-that-is-not-a-frame.
`frame_bytes(&SpawnMessage) -> Result<Vec<u8>>` — `to_line()` as bytes.
The cap is enforced through `fill_buf`/`consume`, so only consumed bytes are accumulated and a runaway line fails instead of allocating.

### Subprocess backend — `subprocess.rs` + `subprocess_io.rs`

```rust
pub struct SubprocessConfig {                              // serde snake_case, default
    pub program: String,                                      // "" — empty is Error::Config at spawn
    pub args: Vec<String>,                                    // []
    pub cwd: Option<PathBuf>,                                 // None = inherit the parent's
    pub env: BTreeMap<String, String>,                        // {} added on top of the parent env
    pub request_timeout_secs: f64,                            // 30.0
    pub shutdown_timeout_secs: f64,                           // 10.0
    pub health_check_interval_secs: f64,                      // 5.0 initial value; 0.0 starts with no background task
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
                                                            // health_interval() -> Duration,
                                                            // set_health_interval(interval: Duration) -> Result<()>,
                                                            // into_agent(self: Arc<Self>) -> Arc<dyn SpawnedAgent>
```

`SubprocessSpawner::spawn_process` returns the concrete `Arc<SpawnedProcess>` so hosts can call `health`/`pid`; the `AgentSpawner::spawn` impl coerces it to `Arc<dyn SpawnedAgent>`.
`describe()` reports the launch settings through the `describe_settings` helper the external backend reuses: `program`, `args`, `cwd`, `env_names`, and the configured deadlines/caps, with the spawner's own `backend` label.
`SubprocessSpawner::new(config)` labels its children `SpawnBackend::Subprocess`, while `SubprocessSpawner::with_backend(config, backend)` is the wrapper form: a backend that is this protocol plus a step of its own labels its children with its own name, and both `Spawner::backend()` (the spawner's own `backend` field) and every child's handle report that label.
`SpawnedProcess` implements `SpawnedAgent` directly, so both paths expose the same behaviour, including the trait's health accessors: `backend()` is the spawner's label (not always `subprocess`), `is_healthy()` reads the same atomic the background task updates, `health()` runs one probe, and `health_snapshot()` carries the same label and leaves `last_error` empty (a failed probe is an `Err` from `health`, not a fabricated snapshot field).
The handle's `Debug` output shows that label beside its id, pid, liveness, and health.
`cancel_task`/`cancel_task_with_request_id` write one `cancel` frame carrying the id (and the reason) down the same serialized stdin writer — deliberately *without* taking the exchange lock, because a cancel has to reach a child that is mid-exchange — and the child answers with a `cancel_ack` the in-flight exchange ignores as a stray frame; a blank id is `Validation`, a desynchronized handle reports the same desync error an exchange would, and a child that already exited is a transport failure. A child that ignores the frame is not killed: stop it with `shutdown`/`force_kill`.
`send_with_timeout(inputs, timeout)`/`stream_with_timeout(inputs, timeout)` bound one exchange: `Some(..)` is that exchange's deadline and `None` falls back to the configured `request_timeout_secs` (`send`/`stream` are the `None` form), so a host holding `Arc<dyn SpawnedAgent>` can bound a single invocation.
`shutdown(timeout)` treats `Duration::ZERO` as "the configured `shutdown_timeout_secs`" (read through the clamped `SubprocessConfig::shutdown_timeout()`); a non-zero caller timeout wins over the configuration, and a child that does not exit within it is killed and reported `false`.
`is_healthy` is the last observed probe result (fresh child = `true`); `health()` runs one probe synchronously.
The background health cadence is per handle: it starts as `SubprocessConfig::health_check_interval_secs` (held in the child state behind a short synchronous `parking_lot` lock, never across an `.await`) and `set_health_interval` re-tunes it.
A re-tune stores the value, stops the old loop cooperatively (its stop token cancels only the wait between probes, so an exchange in flight consumes its reply) and starts a fresh loop that reads the stored value; `Duration::ZERO` leaves no loop running and explicit `health()` probes keep working.
A re-tune never exchanges a frame, so it never desynchronises the handle, and it leaves the `healthy` flag and the consecutive-failure count untouched; repeating it is idempotent and a stopped child (graceful shutdown, kill, or an exited child) is `Error::Validation` naming the child — a dead handle supervises nothing, so respawn instead.
`describe()` keeps reporting the configured initial values; a re-tune changes the handle, never the spawner's configuration.
Configured timeouts are clamped to `[0, 86_400]` seconds, so `Instant + Duration` cannot overflow and nonsense configuration means "no deadline".
`SubprocessConfig` env wins over the inherited environment and `SpawnSpec::envs` wins over `SubprocessConfig` env.
The handle, its state, and the launch/bootstrap path are in `subprocess.rs`; the exchange and supervision layer is the private `subprocess_io` child module declared there — the request guard and its desync rule, `send`/`stream`/`health`/`shutdown`/`force_kill`, the frame I/O, the health loop, the stderr drain, and the kill escalation.
That child module's helpers are `pub(super)` where `SpawnedProcess` calls them, so the public surface is unchanged.

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
`describe()` reports `command`, `search_path` (`null` when unset) and the inherited subprocess summary under `launch`, so a host sees the configuration this backend adds on top of the protocol.
`spawn` resolves, clones the config with `program` set to the resolved path, and delegates to `SubprocessSpawner::with_backend(config, SpawnBackend::ExternalCommand).spawn(spec)`: process handling, framing, timeouts and the health task are the subprocess backend's, and the label makes it the external-command backend — so `Spawner::backend()`, the child's `backend()`/`health_snapshot()`, and a runner's monitor detail all report `external_command`.

### In-process backend — `in_process.rs`

```rust
#[async_trait] pub trait InProcessHandler: Send + Sync {
    fn is_alive(&self) -> bool;                            // true
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    async fn send_with_request_id(&self, request_id: &str, inputs: Value) -> Result<AgentOutput>;  // default: send(inputs)
    fn cancel_request(&self, request_id: &str, reason: Option<&str>) -> bool;       // default: false
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self) -> Result<()>;
}
pub type InProcessFactory = Arc<dyn Fn(&SpawnSpec) -> Result<Arc<dyn InProcessHandler>> + Send + Sync>;
pub struct InProcessSpawner;                               // new(factory), from_handler(Arc<dyn InProcessHandler>)
pub struct InProcessAgent;                                 // id, spec(), handler(), send_with_request_id(id, inputs), the trait methods
```

`new` calls the factory once per spawn, so a spec can select the bound agent; `from_handler` shares one already-built handler across every spawn.
Each spawn wraps its handler in a fresh `InProcessAgent` with the id `in-process-<uuid v4>`.
`is_alive` is this handle's alive flag *and* `handler.is_alive()`, `pid()` is always `None`, `backend()` is `InProcess`, `is_healthy()`/`health()` report liveness (there is no probe in-process), and `send`/`stream` delegate to the handler.
`shutdown` calls `handler.shutdown()` then marks the handle dead and returns `Ok(true)`; `force_kill` does the same because an in-process handler is stopped, not signalled — there is no OS process — and both are idempotent because the handler's `shutdown` must be.
A factory `Err` propagates from `spawn` and launches nothing.
`describe()` is the marker `{"backend": "in_process", "hosted": true}`: a factory (or one shared handler) is not a transport configuration.
Cancellation is correlated inside the hosted agent, because there is no child to send a frame to: `InProcessHandler::send_with_request_id` starts one invocation under a caller-known id and `InProcessHandler::cancel_request` cancels exactly that invocation (a `request_id`-keyed registry of `CancellationToken`s, typically), leaving the handler's other in-flight calls running; a handler that tracks nothing keeps the defaults (the id is accepted and ignored, `cancel_request` answers `false`).
`InProcessAgent::send_with_request_id(request_id, inputs)` is the inherent entry point that pairs with the seam's cancels; `cancel_task(task_id)`/`cancel_task_with_request_id(request_id, reason)` forward to `cancel_request` — the id *is* the request id, so there is no cancel-all form — and answer `Error::NotFound` naming the id when nothing matches (including for a handler that cannot track requests), `Error::Validation` for a blank id.

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
    pub fn spawner_for(&SpawnSpec) -> Arc<dyn AgentSpawner>;   // route(spec).1, never None
    pub fn route(&SpawnSpec) -> (SpawnBackend, Arc<dyn AgentSpawner>);  // resolved key + spawner
    pub async fn spawn(SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
    pub async fn spawn_with(SpawnBackend, SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
```

`DeferredSpawner::for_backend` is the wording table: `RemoteMq` → `"Pulsar/ZeroMQ spawn transport"` with a hint naming drunner reply-topic routing; `RemoteA2a` → `"a2a spawn transport"` with a hint naming `openjiuwen-extensions`' `a2a` feature; an implemented-but-unregistered backend → `"no <name> spawner registered"` with a hint naming the matching `register_*` method; anything else → ``"spawn backend `<name>`"``.
Every `spawn` on one is `Error::Unsupported` (`error::unsupported(reason, hint)`).
`DeferredSpawner::describe()` reports the deferred backend with its `reason` and `hint` and `"deferred": true`, so a diagnostic dump says why a spawn cannot happen; `SpawnerRegistry` is not an `AgentSpawner` and has no `describe` of its own (its registered spawners each do).

Routing precedence in `route`: `spec.backend` when set (registered spawner, else the matching `DeferredSpawner` — never a fallback to another backend) → the explicit default → the sole registered spawner → `(Unknown, DeferredSpawner)` with reason `"no default spawn backend"` (empty) or `"ambiguous spawn backend"` (more than one), whose hint names `set_default` and `SpawnSpec::backend`.
`route` also reports the resolved key, so a spawner registered under a key but reporting `Unknown` still labels its spawn with that key; `spawner_for` is `route(spec).1` with the same precedence.
`spawn_with` resolves exactly the named backend and answers with `DeferredSpawner::for_backend` when it is not registered.
The registry holds one `parking_lot::RwLock` over `{spawners: BTreeMap<SpawnBackend, Arc<dyn AgentSpawner>>, default: Option<SpawnBackend>}`; the spawner `Arc` is cloned out before any `.await`, so no guard is held across one.

### Remote backend — `remote.rs` + `remote_io.rs`

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
    async fn cancel_task_with_request_id(&self, request_id: &str, reason: Option<&str>) -> Result<()>;  // default: cancel_task(request_id)
}
pub struct RemoteAgent;                                    // new(config), with_protocol(config, protocol), config(), agent_id(), protocol()
pub struct RemoteSpawner;                                  // new(base_url), with_timeout, with_header, base_url(), config_for(agent_id)
```

`RemoteAgent` implements both `RemoteClient` and `SpawnedAgent`.
Through the seam it reports `backend() == RemoteHttp`, `pid() == None`, `is_healthy()` = the client's started state, `health()` = `Ok(is_started)` (the HTTP wire contract defines only invoke/stream/cancel, so there is no probe endpoint), and `cancel_task` forwards to `RemoteClient::cancel_task` (the one place a remote backend owns a real task handle) while `cancel_task_with_request_id` forwards to `RemoteClient::cancel_task_with_request_id`, which posts the id and the optional reason.
`send_with_timeout`/`stream_with_timeout` forward the deadline to `RemoteClient::invoke`/`RemoteClient::stream`, so a bounded invocation is real on this backend while `send`/`stream` keep the configured `timeout_secs`.
`RemoteAgent::new` stays lazy (the `reqwest` client is built on first `invoke`/`stream` or by `start`, so a directly constructed handle reports itself not alive until then), but `RemoteSpawner::spawn` calls `RemoteClient::start` before returning: a spawned remote child is started, so `is_alive()`/`is_healthy()`/`health()` are `true` immediately with no request sent (Python's `remote_agent.py` starts the child too), and a client that cannot be built fails the spawn.
`RemoteSpawner::backend()` is `RemoteHttp`.
`RemoteSpawner::describe()` reports `base_url`, `timeout_secs` (`null` when unset), and `header_names` — the header names only, so an `Authorization` bearer token never reaches the summary.
`RemoteSpawner::spawn` derives the agent id from `agent_class`, then `agent_module`, then a non-empty `init_kwargs["agent_id"]` (`Validation` when none is usable), copies `SpawnSpec::session_id` into the config, and validates the endpoint at spawn time rather than at first invoke: an empty `base_url` and a non-empty one that is still not an absolute URL both fail there (`Validation`), as does a client that cannot be built (an invalid configured header).

URL layout: the endpoint is `{base_url}/{path}` with `{agent_id}` substituted into the path; extra slashes are collapsed outside the scheme and a missing or extra slash between base and path is tolerated, so `("http://host:8080/", "/agents/{agent_id}/invoke")` and `("http://host:8080", "agents/{agent_id}/invoke")` build the same URL.
A path that contains `{agent_id}` requires a non-empty agent id (`Validation`), and an empty `base_url` is `Validation`.
The configuration, the URL layout, `RemoteProtocol`, `RemoteClient`, the `RemoteAgent` lifecycle (`ensure_started`/`http_client`/`start`/`stop`), and `RemoteSpawner` are in `remote.rs`; client construction, request execution (`invoke`/`stream`/`cancel_task`), status mapping, and the NDJSON stream parser are in the private `remote_io` child module declared there.
That child module's helpers are `pub(super)` where `RemoteAgent` calls them, so the public surface is unchanged.

### Monitored decorator — `monitored.rs`

```rust
pub struct MonitoredSpawnedAgent;                           // new(inner, monitor, run_id),
                                                            // track(inner, monitor, session_id, detail),
                                                            // inner(), run_id(), is_finished()
```

`MonitoredSpawnedAgent` implements `SpawnedAgent` and forwards the whole seam surface to the wrapped handle: `id`, `is_alive`, `pid`, `backend`, `is_healthy`, `health_snapshot`, `send`, `stream`, `shutdown`, `force_kill`, `cancel_task`, and the bounded `send_with_timeout`/`stream_with_timeout` (the caller's `Option<Duration>` deadline reaches the inner handle verbatim, so a bounded invocation stays bounded through the decorator instead of falling back to the trait's `send`/`stream` default).
Only its own observation differs: `track`/`new` admit the run, `health` records the probe, `shutdown`/`force_kill` finish it (`Completed`/`Cancelled`), `Drop` finishes an unfinished run as `Cancelled` with `{"reason": "handle_dropped"}` — an abandoned spawn is never reported as a success — and an error from `send`, `stream`, `send_with_timeout` or `stream_with_timeout` finishes it as failed when the inner handle reports `is_alive() == false` (the shared `SpawnWatch`/`finish_if_child_exited` latch, never a second latch path); a failure against a live child — a caller deadline that elapsed, for instance — leaves the run `Running`.

**Supervised health cadence.** `set_health_interval` forwards to the inner handle first and reads the answer.
An `Ok` means the backend owns the cadence: any supervised loop of the decorator's is stopped, because a second loop would report every observation twice, and nothing is recorded, because a cadence change is not a lifecycle transition.
`Error::Unsupported` starts — or re-tunes — the decorator's own supervised loop (`HealthSupervisor`, one slot holding one `SupervisedLoop` per armed cadence) over the wrapped handle and returns `Ok(())`; `Duration::ZERO` stops it (the zero is still forwarded to the backend), and any other error is the backend's own and is returned unchanged with no loop armed.
Each supervised tick runs `SpawnWatch::record_probe(ProbeMode::Tick)`: the same probe, the same `{"backend", "alive", "healthy", "pid"}` success detail, the same `{"healthy": false, "error"}` failure detail and the same dead-child rule as `health`, with every monitor call discarded by `let _ =` so an observation never fails the spawn.
The one deliberate difference is an unimplemented probe (`Error::Unsupported`, the shape of the trait default for a bare host handle): the tick records `{"backend", "alive", "healthy", "pid", "probe": "liveness_only"}`, because liveness is real data while a fabricated probe result would not be, while the explicit `health` call keeps recording that error as the failed observation it is.
The loop is one owned task that holds only a `SpawnWatch` and a `CancellationToken` (never the supervisor, so it cannot keep its own owner alive): a re-tune cancels and joins the previous task before arming the next, `shutdown`/`force_kill` cancel and join it before their exchange, `Drop` aborts it, and it also exits on the shared `closed` latch, so a terminal finish (a dead child, a stream error, a kill, a dropped handle) ends it within one interval.

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
| parent → child | `cancel` | `{"request_id": …, "reason": … \| null}`; refused (`Validation`) when the id is blank |
| child → parent | `cancel_ack` | echoes `{"request_id": …}` (plus whatever the child adds); the in-flight exchange reads it as a stray frame |

Stdout/body handling, identical on both backends: blank and non-JSON lines are skipped at `trace` (a child's stray logging); a line that is JSON but not a frame, not valid UTF-8, or longer than `max_frame_bytes` is `Error::Validation`; an unterminated trailing line is still parsed.
Any other frame arriving in the wrong direction is ignored at `trace` (Python's reader tolerates stray frames).

HTTP: `POST {base_url}/{invoke_path}` and `POST {base_url}/{stream_path}` with body `{"session_id": <session_id|null>, "inputs": <inputs>}`; `POST {base_url}/{cancel_path}` with `{"task_id": <id>, "request_id": <id>, "reason": <reason|null>}` — the id under both its original key and its correlation name, so an endpoint written against the first layout keeps working while a reason-carrying cancel loses nothing on the way out.
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

- **One exchange at a time per handle.** The protocol correlates a *cancellation* (the `cancel` frame names the request id the caller and the child agreed on) but not an exchange (`message_id` is generated per frame but nothing correlates on it, and an `input` frame carries no id), so a handle serialises exchanges with one fair `tokio::sync::Mutex` request guard; callers queue rather than interleave frames. A `cancel` frame is written outside that guard on purpose: it must reach a child that is in the middle of an exchange, and the ordered stdin pipe is the only serialization it needs.
- **The desync rule.** An exchange that ends without consuming its terminator — an abandoned stream, a request deadline, or a health probe that hits its deadline — leaves frames unread, so the handle is marked out of sync *before* the lock is released and every later `send`/`stream`/`health` fails with `Error::Validation` naming the child; the caller respawns.
- **Kill on drop.** `tokio::process::Child::drop` does not kill and can leak a zombie, so dropping the last handle reference (its `Drop`) arms `start_kill` plus a detached reap, with `kill_on_drop(true)` as the final net when no runtime is available.
- **stderr is always drained, one bounded line at a time.** A child that fills its stderr pipe would block, so a task drains it into the debug log; a line is read through `frames::read_line_capped` under an 8 KiB cap, the retained prefix is logged as a 512-char preview with `truncated` on the event, everything past the cap is drained and discarded, and the task is aborted with the handle.
- **The health task holds a `Weak`.** It upgrades per tick, so it can never outlive or resurrect the handle; it is aborted on drop and on shutdown, and an interval of `0` disables it entirely (`health()` still works).
- **Re-tuning replaces the loop, never the child.** `set_health_interval` stores the value in the handle's state and restarts the loop around it: the old loop is stopped by a `CancellationToken` that only fires between probes (a probe in flight finishes, so no reply is stranded and the handle stays in sync), and it is aborted only if it cannot stop within 5 s because it is queued behind a caller's exchange. The child, its `healthy` flag and its consecutive-failure count are untouched, so a cadence change can never look like a health observation; a stopped child refuses the call with `Error::Validation` instead of starting a loop on a dead process.
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
- **Cancellation is carried, never simulated.** A process backend writes a `cancel` frame with the caller's request id (and reason) and the child decides what to do; the in-process backend cancels the matching in-flight call through its handler; the HTTP client posts the id to its `cancel_path`. No backend fabricates a cancellation, and a backend with no cancellation keeps the trait's typed `Unsupported` (`cancel_task_with_request_id` defaults to `cancel_task`, so it reports the same absence). An id that matches nothing is conveyed faithfully by the wire and reported `NotFound` by whoever owns the in-flight requests.
- **A deferred backend is a registered answer.** `SpawnerRegistry::with_deferred` stores `DeferredSpawner`s for `RemoteMq`/`RemoteA2a`, so routing is uniform: the missing transport is named in the `Unsupported` message (`Pulsar`/`a2a`) instead of surfacing as "nothing registered".
- **Routing never falls back silently.** A `SpawnSpec::backend` that is not registered is a deferred failure naming that backend, `spawn_with` resolves exactly what it was asked for, and only an *unspecified* backend reaches the default / sole-spawner / ambiguity rules.
- **A hint a runner cannot honour is a configuration error.** With only a single spawner installed, a `SpawnSpec::backend` that contradicts the spawner's own `backend()` is `Error::Config` naming both sides and both remedies (`Runner::set_spawner_registry`, `Runner::spawn_agent_with`); an unlabelled `Unknown` spawner cannot contradict the hint and keeps it as the run's label.
- **Observation is a decorator, not a backend.** `MonitoredSpawnedAgent` wraps whatever handle a backend returned, so every backend — including a host's own `AgentSpawner` — reports the same `RunKind::Spawn` lifecycle without knowing the monitor exists; `is_healthy` stays a pure read and only `health` records, which keeps the poll path free of monitor churn.
- **A health cadence is configuration; the decorator supervises it only when the backend cannot.** `set_health_interval` forwards first and lets the backend's own loop win, so a `SpawnedProcess` keeps exactly one loop; only `Error::Unsupported` makes the decorator arm its own single supervised loop (`HealthSupervisor`), which is why a monitored in-process or host handle honours a cadence instead of silently doing nothing — and why the trait default stays an honest `Unsupported` rather than growing a no-op loop in every backend.
- **A dead child is terminal.** After a `health` probe, and after a `send`/`stream` (or a bounded `send_with_timeout`/`stream_with_timeout`) that returned an error, a child reporting `is_alive() == false` finishes the run as failed with `{"reason": "child_exited", "backend": …}` (latched once), while the same failure against a live child only observes — so `Runner::active_spawns` stops listing an exited child once something probes or calls it, and never mistakes a desync or an elapsed caller deadline for an exit.
- **Runner routing is registry-first and last-install-wins.** A `Runner` holds one spawner and an optional registry beside it: a registry routes `spawn_agent` through its `route` (resolved key + spawner), `spawn_agent_with` requires the registry and refuses to fall back, and `set_spawner` clears the registry while `set_spawner_registry` leaves the single spawner installed for `spawn_backend`/`spawn_agent`-without-registry reporting.
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
- `./src/spawn/subprocess.rs` → `SubprocessSpawner`, `SpawnedProcess`, `ProcessState` (handle, state, construction/bootstrap, kill-on-drop), `bootstrap_frame`, `merged_env`, and the `SubprocessConfig` re-export
- `./src/spawn/subprocess_io.rs` → the exchange and supervision layer of that backend: request guard/desync, send/stream/health/shutdown/force-kill, frame I/O, health loop, stderr drain, kill escalation
- `./src/spawn/subprocess_config.rs` → `SubprocessConfig` (serde config, builders, clamped deadlines), private to the module and re-exported as `spawn::subprocess::SubprocessConfig`
- `./src/spawn/external.rs` → `ExternalCommandSpawner` (command resolution on a search path / `PATH`, executable checks, delegation to the subprocess backend)
- `./src/spawn/in_process.rs` → `InProcessHandler`, `InProcessFactory`, `InProcessSpawner`, `InProcessAgent` (host-bound agent, no child)
- `./src/spawn/registry.rs` → `SpawnerRegistry` (multi-backend routing), `DeferredSpawner` (the wording table for undeferrable/undeferred backends)
- `./src/spawn/monitored.rs` → `MonitoredSpawnedAgent` (lifecycle into `RunMonitor` as `RunKind::Spawn`: admission, health observation, dead-child terminal rule, shutdown/kill/drop) plus the supervision it owns (`SpawnWatch`, `ProbeMode`, `HealthSupervisor`, `SupervisedLoop`, `supervised_health_loop`, `observed_stream`)
- `./src/spawn/mod_tests.rs` → inline unit tests of `mod.rs` (loaded via `#[path]`): wire names, `SpawnHealth`, `SpawnSpec::backend`, the defaulted accessors
- `./src/spawn/subprocess_tests.rs` → inline unit tests of `subprocess.rs` (loaded via `#[path]`)
- `./src/spawn/monitored_tests.rs` → inline unit tests of `monitored.rs` (loaded via `#[path]`): admission/observation/terminal transitions, the fully forwarded seam surface and its bounded deadlines, and the supervised health cadence (re-tune, zero, stop paths)
- `./src/spawn/remote.rs` → `RemoteAgentConfig`, `RemoteProtocol`, `RemoteClient`, the `RemoteAgent` lifecycle, `create_remote_client`, `RemoteSpawner`, URL building, spawn-time validation
- `./src/spawn/remote_io.rs` → the HTTP transport of that backend: client construction, request body, invoke/stream/cancel execution, status mapping, NDJSON stream parsing
- `./src/spawn/remote_tests.rs` → inline unit tests of `remote.rs` (loaded via `#[path]`)

## Notes for Agents

- Install a backend with `Runner::set_spawner`, or several with `Runner::set_spawner_registry`; `UnsupportedSpawner` stays the default, so an unconfigured runner fails a spawn with `Error::Unsupported`.
- Cancelling through a handle from `Runner::spawn_agent` uses `cancel_task(task_id)` (no reason) or `cancel_task_with_request_id(request_id, reason)` (both); `MonitoredSpawnedAgent` forwards both unchanged — cancellation is not a lifecycle transition, so the run keeps its status until the cancelled request actually ends.
- Every successful `Runner::spawn_agent`/`Runner::spawn_agent_with` is wrapped in `MonitoredSpawnedAgent::track`, so `Runner::active_spawns` lists the spawns not yet reported finished (a probe seeing a dead child, `shutdown`, `force_kill`, or `Drop`) and a failed spawn creates no monitor entry.
- This module's own `pub use` group is the source of truth for names; `src/lib.rs` re-exports them flat (`SpawnBackend`, `SpawnHealth`, `SpawnerRegistry`, `DeferredSpawner`, the four backends, and `MonitoredSpawnedAgent`).
- Integration suites: `tests/spawn_subprocess.rs` re-executes the test binary as the fake child (`FAKE_CHILD_MODE` selects echo/stream/crash/silent/malformed/oversized/health-failure/shutdown modes), and `tests/spawn_remote.rs` drives a loopback `wiremock` server.
- The crate-level `## Deferred Python Behavior` and `tests/CONTEXT.md` still describe the pre-backend deferral and are reconciled by the parent node.
- A handle is single-use once desynchronized: respawn rather than retry, and never assume a timed-out exchange can be recovered.
- `send_with_timeout`/`stream_with_timeout` are the only way to bound *one* exchange: `send`/`stream` always use the backend's configured deadline, and the trait defaults ignore the hint (so an in-process handle or a host's own backend simply behaves like `send`/`stream`). `MonitoredSpawnedAgent` forwards both bounded variants, so a handle from `Runner::spawn_agent` keeps a caller's per-request deadline.
- `shutdown(Duration::ZERO)` is not "kill now": it means "use the configured `shutdown_timeout_secs`". Pass a non-zero duration to override the configuration for one shutdown.
- `SpawnedProcess::health_interval()` is the interval *this handle* currently supervises at: the configured `health_check_interval_secs` until `set_health_interval` changes it, and `Duration::ZERO` means "no background task". A re-tune is per handle, so two children of one spawner can probe at different rates; it is idempotent, and it is `Error::Validation` once the child has stopped (shutdown, kill, or an exit of its own) — respawn instead.
- `set_health_interval` reaches a handle through the seam too, where `InProcessAgent`, `RemoteAgent` and any host backend keep the default and answer `Error::Unsupported` naming the backend and the missing background loop; through `MonitoredSpawnedAgent` (the handle `Runner::spawn_agent` returns) that answer is not a dead end: the decorator supervises the cadence itself and returns `Ok(())`, so a monitored in-process agent really does get background observations, and `Duration::ZERO` stops them.
- `RemoteSpawner::spawn` returns a *started* `RemoteAgent` (readiness is `true` without a request); `RemoteAgent::new` is lazy, so a directly constructed handle reports not-alive until its first request or an explicit `RemoteClient::start`.
- There is no child-side serve helper (no `serve_child_stdio`): a host's child program must read/write NDJSON itself — read one `input` bootstrap frame (`{"spawn_spec": …}`), then answer `input` (`{"inputs": …}`) frames with `output`/`stream_chunk`/`done`/`error`, reply to `health_check`/`shutdown`, and answer a `cancel` frame with a `cancel_ack` echoing `{"request_id": …}` (the fake child's `cancel_echo` mode in `subprocess_tests.rs` is the worked example); `SpawnMessage::from_line`/`to_line` and `read_frame`/`frame_bytes` are reusable on that side.
- The reference child implementation is `tests/spawn_subprocess.rs`'s `#[ignore]`d `fake_child` libtest case re-executed with `fake_child --exact --ignored --test-threads=1 --nocapture` (`--nocapture` is mandatory or libtest redirects the child's stdout and no frame reaches the pipe).
- `SubprocessConfig::program` empty is `Error::Config` at spawn, not at construction, so a host can build a config incrementally.
- `SubprocessSpawner::with_backend(config, backend)` is how a wrapper backend labels its children; a `SpawnedProcess` reports its spawner's label, so `backend() == Subprocess` only for a plain `SubprocessSpawner::new`/`default`.
- The lib unit tests re-execute the lib test binary as the hermetic child: `subprocess_tests.rs`'s `#[ignore]`d `fake_child` is selected by `--exact` with the filter `<module_path minus the crate>::fake_child`; the child must close libtest's partial `test … ` line with a leading newline or the parent skips its first frame as noise.
- `ExternalCommandSpawner::with_search_path` is the hermetic form: it replaces `PATH` with one directory, so a test never depends on the machine's installed programs.
- `SpawnerRegistry` is the multi-backend host; the runner-level routing lives in `src/runner_spawn.rs` (`Runner::spawner`, `spawn_backend`, `spawn_backends`, `set_spawner`, `set_spawner_registry`, `spawner_registry`, `spawn_agent`, `spawn_agent_with`, `active_spawns`).
