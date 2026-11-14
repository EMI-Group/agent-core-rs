# openjiuwen-runner

## Intent

The process-level orchestrator of the kernel: one `Runner` owns the resource registries, the callback framework, the message queue, the run monitor, the HITL broker, the spawn backends, and the team runtime pool.
Ports Python `openjiuwen/core/runner/` (v0.1.20, ~13.9k lines, 53 files) with real in-process behaviour kept and only network/broker/process-pool machinery deferred:
`runner.py` + `runner_config.py` (facade, config, lifecycle), `resources_manager/` (registries plus the tag/model/prompt/sys-operation/MCP managers), `callback/` (framework, filters, chains, decorators, guardrail binding), `message_queue_base.py` (in-process queue), `spawn/` (real `tokio::process` child + NDJSON protocol), `drunner/remote_client/`'s HTTP path (`RemoteAgent`), `team_runner.py`'s in-process team path with its pool, and the HITL/monitor surface (`interact_agent_team`, `pause`/`stop`, `list_active_teams`, `get_agent_team_monitor`).
Deferred in `## Deferred Python Behavior` below: Pulsar/ZeroMQ/A2A transports, Python's `python -m` child bootstrap/pool nuances, live in-loop `InteractGate` enforcement, the org layer, cross-process attach, OS-process members, and monitor persistence/export.

The crate sits at layer 4 (`../docs/ARCHITECTURE.md` §1): it depends on `common`, `foundation`, `session`, `workflow`, `agent`, `multiagent`, `security`, and `sysop`, and is consumed by `openjiuwen-core` (kernel umbrella), `openjiuwen-harness`, `openjiuwen-agent-teams`, and `openjiuwen-cli`.
No crate below depends on it: the runner drives lower layers through their public seams (`Agent`, `Workflow`, `Team`, `Session`), never by reaching into internals.

## API Surface

Everything below is re-exported flat at the crate root; the module paths stay available for hosts that prefer them.

### Facade — `src/runner.rs` + `src/runner_spawn.rs` + `src/runner_seams.rs`
```rust
pub struct Runner { /* config, ResourceMgr, CallbackFramework, MessageQueue, checkpointer, guardrails, spawner + spawner registry, RunMonitor, InteractionHub, TeamPool */ }
impl Runner {
    pub fn new() -> Arc<Self>;                                  // default RunnerConfig
    pub fn with_config(config: RunnerConfig) -> Arc<Self>;
    pub fn global() -> Arc<Self>;                               // documented process singleton
    pub fn reset_global() -> Arc<Self>;                         // test-only: installs a fresh singleton
    pub fn runner_id(&self) -> &str;
    pub fn config(&self) -> RunnerConfig;
    pub fn set_config(&self, config: RunnerConfig);             // takes effect on the next start
    pub fn resource_mgr(&self) -> &Arc<ResourceMgr>;
    pub fn callbacks(&self) -> &Arc<CallbackFramework>;
    pub fn message_queue(&self) -> Arc<dyn MessageQueue>;
    pub fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>>;
    pub fn monitor(&self) -> Arc<RunMonitor>;                   // cheap snapshot of the installed seam
    pub fn set_monitor(&self, monitor: Arc<RunMonitor>) -> Arc<RunMonitor>;        // installs for later runs; returns the previous one
    pub fn hitl(&self) -> Arc<InteractionHub>;                  // cheap snapshot of the installed hub
    pub fn set_hitl(&self, hub: Arc<InteractionHub>) -> Arc<InteractionHub>;       // installs as given; returns the previous hub
    pub fn set_message_queue(&self, queue: Arc<dyn MessageQueue>) -> Arc<dyn MessageQueue>;  // until the next `start`; returns the previous queue
    pub fn queue_stats(&self) -> QueueStats;                    // snapshot of the live backend (`MessageQueue::stats`)
    pub fn team_pool(&self) -> TeamPool;                        // cheap clone of the pool handle
    pub fn is_started(&self) -> bool;
    pub async fn start(&self) -> Result<()>;                    // idempotent; queue + checkpointer + fresh pool; refuses distributed/deferred transports
    pub async fn stop(&self) -> Result<()>;                     // idempotent; sweeps the pool, releases resources
    pub async fn release(&self, session_id: &str) -> Result<()>;
    // execution (see ./src/run/CONTEXT.md)
    pub async fn run_agent(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
    pub fn run_agent_streaming(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn run_workflow(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> Result<WorkflowOutput>;
    pub fn run_workflow_streaming(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn run_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<Value>;  // direct, non-pooled
    pub fn run_agent_team_streaming(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<Value>>;
    // team pool (see ./src/pool/CONTEXT.md)
    pub async fn start_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<PoolRun>;
    pub async fn try_start_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<PoolRun>;      // refuses at capacity instead of queueing
    pub async fn acquire_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<PoolLease>;      // starts or joins (attach)
    pub fn take_team_payloads(&self, team_name: &str, session_id: &str) -> Vec<InteractPayload>;
    pub fn list_active_teams(&self) -> Vec<ActiveTeamInfo>;
    pub async fn stop_agent_team(&self, team_name: &str, session_id: &str) -> Result<bool>;
    pub async fn pause_agent_team(&self, team_name: &str, session_id: &str) -> Result<()>;
    pub async fn resume_agent_team(&self, team_name: &str, session_id: &str) -> Result<()>;
    pub fn interact_agent_team(&self, team_name: &str, session_id: &str, payload: InteractPayload) -> DeliverResult;
    pub async fn answer_interaction(&self, response: InteractionResponse) -> Result<()>;
    pub fn agent_team_status(&self, team_name: &str, session_id: &str) -> Option<RunStatus>;
    // security (see ./src/callback/CONTEXT.md §Security binding)
    pub fn set_guardrails(&self, pipeline: Arc<GuardrailPipeline>) -> Vec<CallbackId>;
    pub fn guardrails(&self) -> Option<Arc<GuardrailPipeline>>;
    pub async fn scan(&self, ctx: GuardrailContext) -> Result<GuardrailOutcome>;
    // spawn (see ./src/spawn/CONTEXT.md)
    pub fn set_spawner(&self, spawner: Arc<dyn AgentSpawner>);  // clears the registry: last install wins
    pub fn spawner(&self) -> Arc<dyn AgentSpawner>;             // the installed single backend
    pub fn spawn_backend(&self) -> SpawnBackend;
    pub fn set_spawner_registry(&self, registry: Arc<SpawnerRegistry>);   // takes precedence over the single spawner
    pub fn spawner_registry(&self) -> Option<Arc<SpawnerRegistry>>;
    pub fn spawn_backends(&self) -> Vec<SpawnBackend>;          // registry backends, else [spawn_backend()]
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;              // registry-first, else the single spawner; observed as RunKind::Spawn
    pub async fn spawn_agent_with(&self, backend: SpawnBackend, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;  // needs a registry; never falls back
    pub fn active_spawns(&self) -> Vec<RunStatus>;              // spawns not yet reported finished
}
```
`global()` is the ONE documented process-global: a `OnceLock<RwLock<Option<Arc<Runner>>>>` slot, created on first use with `RunnerConfig::default()`, replaced wholesale by `reset_global()`.
Everything else is per-instance and `Runner::new()` is what tests use.
The monitor, the hub, and the pool are built per instance; `services()` attaches the monitor and the hub to every run, so all six execution entry points are observable without extra wiring.
The monitor, the hub, and the queue are replaceable after construction (`set_monitor`, `set_hitl`, `set_message_queue`), each returning the instance it displaced.
The observation seams live behind a `parking_lot::RwLock`, so the accessors hand out owned `Arc` clones rather than references.
The spawn facade lives in `src/runner_spawn.rs` and the seam setters live in `src/runner_seams.rs` (both `impl Runner` in private modules) to keep `runner.rs` under the size budget.

### Configuration — `src/config.rs`
`RunnerConfig` (`runner_id`, `distributed_mode=false`, `distributed`, `message_queue`, `team_pool`, `default_session_id="default_session"`, `checkpointer: Option<CheckpointerConfig>`, `default_run_timeout_secs`, `shutdown_timeout_secs=5.0`), `TeamPoolConfig` (`max_concurrent_runs`, `idle_timeout_secs`, `run_timeout_secs`, `payload_capacity`, `retain_finished`, `frame_capacity`, `reaper_interval_secs`, `stop_timeout_secs`), `MessageQueueConfig`, `MessageQueueKind { InMemory = "in_memory", Pulsar, ZeroMq }`, `DistributedConfig` (data only), and the `DEFAULT_*` constants (including the `DEFAULT_TEAM_POOL_*` family).
All of it is serde (`snake_case`, `default`, unknown fields tolerated) so a host can persist its runner configuration.
`distributed_mode` and `DistributedConfig` are accepted, persisted, and inspectable, but execution is refused at `start()` with a typed `Unsupported` naming the missing transports.

### `src/resources/` — registries and managers
`ResourceMgr` owns the four typed registries (tools, agents, workflows, teams) and five sub-managers: `tags() -> &TagMgr`, `models() -> &ModelMgr`, `prompts() -> &PromptMgr`, `sysops() -> &SysOperationMgr`, `mcp_servers() -> &McpServerMgr`.
Registries: `register_*` (duplicate = `Error::Validation`), `register_*_with(policy)`, `*_replacing`, `get`, `card`, `by_name`, `remove`, `cards`, `ids`, `len`, `is_empty`, `clear_*`, `revision`, `release()`; providers are `Arc<dyn Fn() -> T + Send + Sync>` (a fresh instance per lookup).
Tag manager: `Tag`, `ResourceKind`, `TagMatchStrategy {Any, All}`, `TagUpdateStrategy {Merge, Replace}`, `GLOBAL_TAG`, bi-directional indexes, and tag garbage collection on unregister.
Manager kinds: `ModelCard`/`ModelProvider`/`ModelMgr`, `PromptCard`/`PromptMgr` (Python's overwrite-on-add), `SysOperationProvider`/`SysOperationMgr`, `McpServerInfo`/`McpServerState`/`McpConnector`/`DefaultMcpConnector`/`McpServerMgr` (register → connect → load tools → disconnect lifecycle).
See `./src/resources/CONTEXT.md`.

### `src/callback/` — event framework, filters, chains, decorators
Framework: `CallbackEvent`, `CallbackContext`, `CallbackHandler`, `CallbackVerdict`, `CallbackReport`, `CallbackFailure`, `CallbackId`, `CallbackRegistration`, `CallbackInfo`, `CallbackFramework`.
Filters: `EventFilter`, `FilterDecision {Allow, Skip, Reject}`, `FilterRunOutcome`, `FilterChain`, `FilterInput`, `FilterFeedback`, `Clock`/`ManualClock`, composition combinators, and the built-ins `NameFilter`, `KindFilter`, `SessionFilter`, `RateLimitFilter`, `CircuitBreakerFilter`, `ValidationFilter`, `LoggingFilter`, `AuthFilter`, `ParamModifyFilter`, `ConditionalFilter`, `PredicateFilter`, `ContextModifier`.
Chains: `CallbackChain` (ordered entries, names/ids, enable/disable, per-entry filters, `Abort` short-circuit, `Stop`, isolated-vs-fatal errors), `ChainEntry`, `ChainEntryId`, `ChainReport`, `EntryOutcome`, `EntryRecord`.
Decorators: `HandlerBuilder`/`BuiltHandler`/`on`, `emit_before`/`emit_after`/`emit_around`, `transform_io`/`TransformHandler`/`PayloadTransform`, `wrap`/`on_wrap`/`WrappedCall`/`WrapRegistry`, plus typed payload accessors (`input_as`, `output_as`, `require_input`, `require_output`, `set_input`, `set_output`).
Free combinator functions stay under `openjiuwen_runner::callback::{decorator, filters}` (their names are too generic for a flat prelude); the types are re-exported flat.
Security binding: `bind_guardrails`, `GuardrailHandler`, `scan_stage_for`, `event_for_stage`.

### `src/mq/` — the message queue
`QueueMessage`, `MessageHandler`, `MessageQueue` trait, `InMemoryMessageQueue`, `DeferredMessageQueue`, `create_message_queue` (fallible), `build_message_queue` (infallible: in-memory, or a deferred placeholder for a transport this build lacks), `MessageQueueKind`.
`MessageQueue::stats() -> QueueStats` is synchronous, object-safe, and defaulted to the empty report, so a host samples an `Arc<dyn MessageQueue>` without an `.await` or a downcast; `Runner::queue_stats()` is the facade's snapshot of whichever backend is installed.
The counters are `started`, `topics`, `handlers`, `deliveries`, `dropped_no_worker`, `dropped_no_handler`, `handler_errors`, and `rejected_not_started`, with the `is_empty()`/`dropped()` helpers; the report is `snake_case` JSON tolerant of unknown fields.
The configured kind is never misrepresented: a `pulsar`/`zeromq` configuration yields `DeferredMessageQueue`, whose every operation is `Error::Unsupported` naming the transport and pointing at `in_memory` — and whose `stats()` is the empty report, because it performs no work.

### `src/run/` — execution APIs
`RunOptions` (session, timeout, cancellation, envs, metadata, `team_session`), `AgentRef`, `WorkflowRef`, `TeamRef`, `RunServices` (registries, callbacks, config, checkpointer, guardrails, optional `monitor`/`hitl`), session resolution, per-run cancellation/timeout (`run_with_limits`, `next_stream_step`), teardown, and tracing spans.

### `src/spawn/` — the spawn seam and its four backends
Seam: `SpawnAgentKind`, `SpawnSpec` (incl. `backend: Option<SpawnBackend>` + `with_backend`), `SpawnedAgent`, `AgentSpawner`, `SpawnBackend`, `SpawnHealth`, `UnsupportedSpawner` (the documented default).
`SpawnedAgent` exposes `id`, `backend`, `is_alive`, `is_healthy`, `pid`, `health_snapshot`, `health`, `set_health_interval`, `send`/`send_with_timeout`, `stream`/`stream_with_timeout`, `cancel_task`, `shutdown`, `force_kill`; every accessor added beyond the original six is defaulted, so a minimal handle still implements the trait.
`AgentSpawner` exposes `backend()` and `describe()` (a JSON-safe, secret-free configuration summary: env/header names only).
Backends: `SubprocessConfig`/`SubprocessSpawner`/`SpawnedProcess` (real `tokio::process` child over NDJSON, env/cwd merge, per-request and shutdown deadlines, background health checks with escalation and a per-handle re-tunable interval, kill-on-drop); `ExternalCommandSpawner` (the same child protocol plus `PATH`/search-path command resolution, children labelled `ExternalCommand`); `InProcessSpawner`/`InProcessAgent` (a host-bound `InProcessHandler` in the same process, no pid); `RemoteAgentConfig`/`RemoteClient`/`RemoteAgent`/`RemoteSpawner`/`RemoteProtocol`/`create_remote_client` (HTTP invoke/stream/cancel over `reqwest`, NDJSON stream, started at spawn); `DeferredSpawner` (typed `Unsupported` naming a missing transport).
`SpawnerRegistry`: `register*`/`unregister`/`get`/`backends`/`set_default`/`route`/`spawn`/`spawn_with`, with routing precedence `SpawnSpec::backend` → explicit default → sole registered spawner → typed failure; `with_deferred()` pre-registers `RemoteMq`/`RemoteA2a` as deferred.
`MonitoredSpawnedAgent`: the decorator `Runner::spawn_agent` returns; forwards the whole seam (including the bounded variants) and records the child's lifecycle into `RunMonitor` under `RunKind::Spawn`.
Frame helpers: `SpawnMessage`, `SpawnMessageKind`, `frame_bytes`, `parse_frame_line`, `read_frame`, `DEFAULT_MAX_FRAME_BYTES`.
See `./src/spawn/CONTEXT.md`.

### `src/monitor/` — run status and events
`RunKind {Agent, Workflow, Team, Spawn}`, `RunPhase {Queued, Running, Paused, Completed, Failed, Cancelled}`, `RunStart`, `RunOutcome`, `RunStatus`, `RunEvent`, `RunMonitor` (`begin`, `track`, `update`, `finish`, `snapshot`, `list_active`, `history`, `subscribe`, `clear`), and `EVENT_HISTORY_LIMIT`/`LIVE_RUN_LIMIT`/`EVENT_CHANNEL_CAPACITY`.
`RunKind::Spawn` tracks a spawned child through the same bounded history and broadcast channel as a run.
See `./src/monitor/CONTEXT.md`.

### `src/hitl/` — live-run interaction broker
`InteractKind`, `InteractPayload`, `RunTarget`, `DeliverResult {Accepted, MissingTarget, NoInbox, Closed}`, `ActiveRunSpec`, `ActiveRunInfo`, `RunHandle`, `InteractionHub` (`register`, `unregister`, `attach_session`, `deliver`, `take_inbox`, `answer`, `pending`, `pause`, `resume`, `cancel`, `list_active`, `close_interactions`), `INBOX_CAPACITY`.
See `./src/hitl/CONTEXT.md`.

### `src/pool/` — team runtime pool
`TeamPool` (`new`, `with_config`, `from_runner_config`, `config`, `capacity`, `available_permits`, `start`, `try_start`, `acquire`, `get`, `list_active`, `stop`, `stop_all`, `pause`, `resume`, `take_payloads`, `phases`, `len`, `is_empty`, `is_closed`), `PoolRequest`, `PoolKey`, `PoolRun`, `PoolLease`, `PoolFrame`, `PoolError`, `PoolShutdown`, `ActiveTeamInfo`.
The facade routes `start`, `try_start`, `acquire`, and `take_payloads` so a host can queue, refuse, attach, and drain without rebuilding `RunServices`.
See `./src/pool/CONTEXT.md`.

### `src/error.rs`
Thin constructors for the runner's error vocabulary (`not_found`, `invalid_session`, `timeout`, `run_timeout`, `unsupported`, `not_started`) over `openjiuwen_common::error::{Error, Result}`; no parallel error enum.

## Pluggable Seams

The seam inventory a host reads before wiring a process is the `# Pluggable seams` section of `src/runner.rs`'s module docs; `tests/seams.rs` pins the same list at compile time.
Plugged in through a setter: the message queue (`MessageQueue` + `Runner::set_message_queue`), the spawn backends (`AgentSpawner` + `Runner::set_spawner`, `SpawnerRegistry` + `Runner::set_spawner_registry`), the run monitor (`Runner::set_monitor`), the HITL hub (`Runner::set_hitl`), and the guardrails (`Runner::set_guardrails`).
Plugged in through a registration: resource providers (`ResourceMgr::register_*` takes a `Provider<T>` closure called per lookup), callback handlers/filters/chains (`CallbackFramework::register`, `EventFilter`, `CallbackChain`).
Configured rather than plugged in: the checkpointer (`RunnerConfig::checkpointer`, built by `Runner::start`).
Refused with `Error::Unsupported` naming the transport: `MessageQueueKind::{Pulsar, ZeroMq}` (→ `DeferredMessageQueue`), `SpawnerRegistry::with_deferred()`'s `RemoteMq`/`RemoteA2a` (→ `DeferredSpawner`), `RemoteProtocol::{Mq, A2a}`, and `RunnerConfig::distributed_mode` (refused by `start`).
Team transports are not the runner's: they belong to `openjiuwen-multiagent` (`TeamTransport`/`TransportKind`/`build_transport`, where `Pulsar` is `Error::Unsupported`) and are reached only through the `Team` seam.

## Constraints

- Follows `../docs/ARCHITECTURE.md`: async-first tokio, `openjiuwen_common::error` results, serde-serializable public data, no `unsafe`, `tracing` only (target `openjiuwen_runner::<module>`), no stdout.
- Locks are `parking_lot` for short synchronous sections and `tokio::sync` for awaitable coordination; a lock is never held across `.await`.
- Every spawned task has an owner: queue topic workers are joined by `stop`, pool runs and the reaper are joined by `stop_all`, health-check tasks are joined by the process handle, and run teardown tasks are bounded by their session's lifetime.
- No panics on user input: every missing id, bad session type, duplicate registration, closed queue, deadline, cancellation, full inbox, dead child, and non-2xx response is an `Error`.
- Files stay ≤ ~1000 lines; split by responsibility (the callback, spawn, and subprocess supervision/exchange areas are split for that reason).
- Two global-state rules: the `Runner` singleton is documented and replaceable (`reset_global`); everything else hangs off a runner instance, never off module statics.
- Tests are offline and deterministic (`openjiuwen_test_support::{ScriptedModel, MockLlmServer}`, `wiremock` on loopback, re-exec fake child processes, `tempfile`); timeouts in tests are short and only used to prove a deadline, never to synchronize.
- The foreign repository is read-only.

## Design Decisions

- **One `Runner` type, instance-first.** Python's `_RunnerImpl` + classmethod facade collapses to one struct: `Runner::new()` is the isolated instance, `Runner::global()` the parity singleton.
- **Registries store cards + providers, not live state.** Python's `ResourceMgr` keys on `card.id` and calls the provider on every lookup; the Rust port keeps that contract and adds `card`/`by_name` lookups, per-registry revision counters, and an explicit `RegisterPolicy { Fail, Keep, Replace }`.
- **Managers live under `resources/`.** Tags, models, prompts, sys-operations, and MCP servers are `ResourceMgr` sub-managers (not a separate top-level area) so every resource kind has one lifecycle: register → look up → tag → release.
- **Deduplication of transports.** `openjiuwen-multiagent` already owns a team-scoped request/reply bus; the runner's `MessageQueue` is the generic topic queue, a plain trait `openjiuwen-extensions` can back with a broker.
- **Callbacks carry a typed context, not `**kwargs`.** Result semantics are `CallbackVerdict::{Continue, Abort, Stop}`; error isolation is explicit rather than accidental.
- **Event names are an enum with the exact Python strings** (`CallbackEvent::as_str()`), so a harness subscribing by string and a guardrail binding by stage agree without a translation table; `Custom(String)` keeps unknown events expressible.
- **Security wiring is a binding, not a special case.** `bind_guardrails` registers one handler per guarded stage at Python's priority 100; a blocked verdict is `Ok(CallbackVerdict::Abort(Error::Security))`, non-blocking findings stay in the report.
- **Filters fail closed.** A filter returning `Err` becomes `FilterRunOutcome::Reject`; Python's skip-on-error is opt-in via `lenient(...)`. Rate-limit and circuit-breaker filters take an injectable `Clock` so tests drive time instead of sleeping.
- **Chains record outcomes.** Python logs chain actions; Rust records `EntryOutcome`/`EntryRecord` per entry, with `fatal` entries escalating an `Err` to an abort and non-fatal entries collecting into the report.
- **Cancellation and deadlines are run-scoped and uniform.** `RunOptions` carries a `CancellationToken` and an optional timeout; agent, team, and their streaming variants race the invocation (`run_with_limits`, `next_stream_step`), returning `Error::Cancelled` / `Error::Timeout` and recording `RunPhase::Cancelled`/`Failed`. Workflow runs pass their limits to the engine, which owns its own deadline.
- **Session ownership is explicit.** A caller-provided session is used and left alone; a session the runner creates is committed/closed by the runner; teardown runs on stream completion or on drop.
- **Subprocess spawning is a real backend, not a fork/exec transliteration.** The child is a program the host configures; the wire is the NDJSON `SpawnMessage` protocol, one reader task owns stdout, stdin writes are serialized, every frame is size-capped, and the handle kills the child on drop. Python's `python -m <module> <class>` bootstrap is a host concern (pass it as `program`/`args`).
- **The HTTP remote client defines its own wire.** Python's `RemoteClient` ABC is preserved as the `RemoteClient` trait, but the transport is an explicit JSON layout (`/agents/{id}/invoke|stream|cancel`) rather than an undocumented A2A/MQ envelope; `Mq`/`A2a` stay typed `Unsupported` until a reply-topic transport lands.
- **The spawn seam is one trait for every backend shape.** Process, external command, in-process, and HTTP handles all answer the same liveness/health/probe/deadline surface, with defaults (`Unknown` backend, liveness-as-health, `Unsupported` probe/cancel) so a backend that cannot answer says so instead of pretending.
- **Spawn routing is explicit and observable.** A `SpawnerRegistry` routes by `SpawnSpec::backend` → default → sole backend; a contradiction between the hint and the installed single spawner is an error rather than a silent misroute; every successful spawn is wrapped so its lifecycle (probe, death, shutdown, kill, drop) lands in the run monitor under `RunKind::Spawn`.
- **The pool owns run lifecycle, not strategy state.** One live run per `(team_name, session_id)`; `acquire` refcounts leases so two hosts can join the same entry; a semaphore bounds concurrency (`start` queues, `try_start` refuses); a reaper evicts idle entries on `tokio::time`; frames go through a broadcast channel so a slow consumer can never block a run; the drain of the HITL inbox is buffered for host replay because an in-process strategy has no mid-flight input gate.
- **HITL answers are session-backed.** `InteractionHub::answer` routes to the live session's `resolve_interaction`, so a run that published an interaction request is answered end-to-end; pause/resume are status/gate signals, and `cancel` trips the run's own `CancellationToken`.
- **Monitoring is bounded and observable.** One event per transition on a broadcast channel, a bounded per-run history (128) and a bounded retention for finished runs (64), with `RecvError::Lagged` surfaced rather than hidden.
- **No hidden retries or fallbacks.** A missing registry id is `Error::NotFound` with the id in the message, an unfinished `start()` is `Error::Config`, an MCP server that cannot connect is recorded `Failed { reason }`, a delivery with no consumer is `DeliverResult::NoInbox`, a deferred backend is `Error::Unsupported` naming the seam, a deferred queue kind is a `DeferredMessageQueue` rather than a working in-memory stand-in, and `distributed_mode` refuses to start instead of running in-process silently.

## Routing Table

- `./src/runner.rs` → `Runner` facade, lifecycle (`start`/`stop`/`release`), global singleton, monitor/hub/pool accessors, team + HITL facade methods, security wiring
- `./src/runner_seams.rs` → the replaceable-seam half of the facade: `set_monitor`, `set_hitl`, `set_message_queue` and what a replacement does not migrate
- `./src/runner_spawn.rs` → the spawn half of the facade: single spawner and registry installation, backend introspection, registry-first routing, monitored spawn observation, `active_spawns`
- `./src/config.rs` → `RunnerConfig`, `TeamPoolConfig`, `MessageQueueConfig`, `MessageQueueKind`, `DistributedConfig`, defaults
- `./src/error.rs` → error constructors and message wording
- `./src/resources/` → `ResourceMgr`, four registries, tag/model/prompt/sysop/MCP managers (`resources/CONTEXT.md`)
- `./src/callback/` → events, framework, guardrail binding, filters, chains, decorators (`callback/CONTEXT.md`)
- `./src/mq/` → `QueueMessage`, `MessageQueue`, `InMemoryMessageQueue`, `DeferredMessageQueue`, backend factories (`mq/CONTEXT.md`)
- `./src/run/` → `RunOptions`, refs, `RunServices`, session resolution, cancellation/timeouts, teardown, spans (`run/CONTEXT.md`)
- `./src/spawn/` → spawn seam, health/liveness accessors, registry, monitored decorator, subprocess/external-command/in-process/remote backends (`spawn/CONTEXT.md`)
- `./src/monitor/` → run status, events, `RunKind::Spawn`, bounded history/retention (`monitor/CONTEXT.md`)
- `./src/hitl/` → live-run registry, payload delivery, interaction answering (`hitl/CONTEXT.md`)
- `./src/pool/` → team runtime pool, leases, concurrency, reaper (`pool/CONTEXT.md`)
- `./tests/` → integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests beside every module: config defaults/serde, error wording, registry insert/lookup/policy/revision, managers (tags/models/prompts/sysops/MCP), event-name mapping, framework ordering/isolation/verdicts, filters/chains/decorators, queue delivery ordering/errors/timeouts plus the deferred-kind placeholder, spawn frame framing and protocol round-trips, spawn backend labels/health/snapshots/deadlines/`describe`, registry routing, monitored lifecycle, subprocess and remote backends, monitor transitions/retention/lag, hub delivery/answering/cancel, pool admission/leases/eviction, session-id resolution, ref parsing.
- Integration suites in `./tests/`: `resources.rs`, `managers.rs`, `managers_mcp.rs`, `callbacks.rs`, `callback_filters.rs`, `callback_chains.rs`, `callback_decorators.rs`, `security_binding.rs`, `message_queue.rs`, `runner_lifecycle.rs`, `runner_facade.rs`, `run_agent.rs`, `run_workflow.rs`, `run_team.rs`, `spawn_subprocess.rs`, `spawn_remote.rs`, `spawn_seam.rs` (every handle through `Arc<dyn SpawnedAgent>`), `spawn_routing.rs` (registry routing, monitored spawn lifecycle, bounded invocation), `runner_team_attach.rs` (pool try-start/acquire/payload drain), `monitor.rs`, `hitl.rs`, `team_pool.rs`, `team_pool_leases.rs`, `team_pool_lifecycle.rs`, `team_pool_hitl.rs`, `api_surface.rs`.
- Subprocess tests re-exec the test binary as a hermetic fake child (no external script, no network); remote tests use `wiremock` on loopback; pool eviction is proven with short idle timeouts and paused/bounded time, never wall-clock sleeps.
- Behavior ports cite the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/runner/test_resource_manager.py`).
- Gates: `cargo fmt -p openjiuwen-runner`, `cargo test -p openjiuwen-runner --offline`, `cargo clippy -p openjiuwen-runner --all-targets --all-features --offline -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc -p openjiuwen-runner --no-deps --offline`, all clean and offline.

## Deferred Python Behavior

- **Distributed mode**: `drunner/` transports (Pulsar backend, ZeroMQ, `message_queue_fake.py` wire envelopes, reply-topic subscription, `ResponseCollector`, `MqRemoteClient`/`MqServerAdapter`) — `RemoteProtocol::Mq`/`SpawnBackend::RemoteMq` and `MessageQueueKind::{Pulsar, ZeroMq}` are typed `Unsupported` naming the transport, `DeferredMessageQueue` stands in for a deferred queue kind, `RemoteProtocol::A2a`/`SpawnBackend::RemoteA2a` are `Unsupported` (A2A lives in `openjiuwen-extensions`), and `RunnerConfig::distributed_mode` is refused by `start()`.
- **Subprocess spawn extras**: Python's `python -m <module> <class>` bootstrap inside the child, the process pool, Windows stdio emulation, and Python's exact graceful→force escalation timings. The trait, spec, NDJSON protocol, health checks (including per-call health control), deadlines, and kill-on-drop are implemented; the child program is host-configured.
- **Spawn extras**: spawner configuration introspection beyond `AgentSpawner::describe`, downcasting a seam handle to its concrete backend, and re-tuning the *request* deadline or the health-probe deadline of a live child (only the background health *interval* is re-tunable).
- **Team pool extras**: OS-process/external-CLI members and SSH transport, cross-process attach over a network transport (lease joining is available in-process through `Runner::acquire_agent_team`), `member=True` spawn routing, `delete_agent_team`, `register_human_agent_inbound`, `start_agent_team_streaming`'s leader-lifecycle frames, `TeamStreamLogger`/monitor TUI, `run_context`, and the org layer.
- **HITL extras**: Python's in-loop `InteractGate` enforcement (pausing does not suspend an in-flight invocation), auth/ACL on who may answer, and cross-process interaction routing. Payloads delivered to a pooled run are buffered for host replay rather than waking a member mid-turn.
- **Monitor extras**: persistence, MQ/WebSocket/OTLP export, metrics/aggregation, and a cross-process view.
- **Resource-manager extras**: tag `display()` dumps, Python's inert `ALL`/`ACTIVE`/`INACTIVE` constants, MCP `expiry_time`/cached tool ids and `server_name`-keyed lookup/tag filters, prompt-pack filesystem loading, `decorate_model_with_trace`/`get_model(..., session)`, and the sysop tool adapter that binds specs to `Tool` objects (specs are exposed; the caller binds).
- **Callback framework extras**: Python's signature-inspection `*args/**kwargs` narrowing, `trigger_parallel`/`trigger_until`/`trigger_delayed`/`trigger_generator`, metrics/state dumps, and `lazy_callback_framework`. Every event name is preserved so these can land without a wire change.
- **Runner internals**: `ReadWriteLockManager`, root task group/`ContextVar` task manager, `ModelContext`, remote-agent detection, `LegacyBaseAgent` handling, `enable_session_controller`, `enable_a2a`.
- Python quirks deliberately not reproduced: missing ids raising `AttributeError`, non-idempotent distributed `start`, `stop()`'s `finally` skipping teardown, in-flight responses orphaned on queue shutdown, `message_queue_inmemory.py`'s `raise response(...)` bug, and the global config being overwritten by constructing a second runner.

## Notes for Agents

- Use `Runner::new()` in tests; never `Runner::global()`. `reset_global()` exists so a test that must touch the singleton can restore a clean one.
- The runner never reads `OPENJIUWEN_*` itself: configuration is injected through `RunnerConfig`. `.env` loading is a binary concern (`openjiuwen-cli`).
- `run_agent`/`run_workflow`/`run_agent_team` accept either a registry id or a live instance; an id is resolved through `ResourceMgr` and a missing id fails with `Error::NotFound` before any session is created.
- Execution does not require `start()`: the facade builds a `RunServices` snapshot per call (now including the monitor and the hub). `start()` is for the queue, the checkpointer, and rebuilding the pool; `stop()` sweeps the pool *before* its `started` early return and installs a fresh open pool, so a later `start()` can admit again.
- The streaming entry points return immediately, so a resolution or session failure is delivered as the stream's single `Err` frame; per-frame failures terminate the stream after one `Err`, and a cancelled/timed-out stream ends with exactly one terminal `Err` and no frame after it.
- `RunOptions::default()` means: no caller session, no deadline, never cancelled, session id from `inputs["conversation_id"]` else `RunnerConfig::default_session_id`.
- `Runner::run_agent_team` is the direct path; `Runner::start_agent_team` queues through the pool and returns a `PoolRun`, `try_start_agent_team` refuses at capacity, and `acquire_agent_team` starts or joins with a refcounted `PoolLease` (the join ignores the joining request's inputs/options). Use the direct path for one-shot calls and the pool for anything a host wants to list, address, pause, or stop.
- Pool: one live run per `(team_name, session_id)`; a second `start` on a live key is `PoolError::Busy`, while `acquire` joins and refcounts. `stop_all()` closes the pool permanently — the facade replaces it.
- HITL: `deliver` needs a pool-registered run with an open inbox (otherwise `NoInbox`/`MissingTarget`); `answer` needs a run whose session was attached by the run path. Pause/resume are signals, not suspension.
- Monitor: `update` refuses terminal phases (only `finish` ends a run); finished runs are retained up to `LIVE_RUN_LIMIT`, so `history`/`snapshot` still work after completion but not indefinitely.
- `set_monitor` installs a monitor for every run started afterwards and re-points the installed hub at it (`InteractionHub::attach_monitor`); it does not migrate anything: in-flight runs keep the monitor they were admitted with, and the retired monitor keeps its live statuses, its retention history, and its subscribers.
- `set_hitl` installs a hub as given — the caller wires it to the runner's monitor with `InteractionHub::with_monitor`/`attach_monitor` if it wants mirroring — and it does not migrate live registrations, so runs in flight stay addressable on the retired hub only.
- `set_message_queue` installs a host-owned queue (the point being a broker backend) in whatever started state the host left it — the runner never starts it implicitly — and the next `Runner::start` replaces it with the configured backend again.
- Subprocess: a handle desynchronizes after an abandoned stream or a probe that hits its deadline and is then single-use; `shutdown(Duration::ZERO)` means "use `SubprocessConfig::shutdown_timeout_secs`", a non-zero timeout wins; `RemoteAgent::pid()` is always `None`; a remote `shutdown` is `stop()`, and a remote handle spawned through `RemoteSpawner` is started at spawn (a directly constructed `RemoteAgent` is lazy).
- Spawn routing: install one backend with `Runner::set_spawner` or a multi-backend `SpawnerRegistry` with `Runner::set_spawner_registry` (the registry wins; `set_spawner` clears it). With a single spawner installed, a `SpawnSpec::backend` hint that contradicts it is `Error::Config`; an unlabelled (`Unknown`) spawner accepts the hint. `RunOptions` has no spawner field and `ResourceMgr` takes no part.
- Every successful `Runner::spawn_agent*` returns a `MonitoredSpawnedAgent`: a `RunKind::Spawn` status exists while the child is live, `health()` records an observation, and a probe or failed call that sees `is_alive() == false` — or `shutdown`/`force_kill`/drop — finishes it with `Failed`/`Completed`/`Cancelled`. `active_spawns()` therefore lists spawns not yet reported finished, not necessarily processes still running.
- `AgentSpawner::describe()` is the only configuration a host can read through the seam; it never contains env values or header values.
- Callback events the runner itself fires: `AgentInvokeInput`/`AgentInvokeOutput`, `AgentStreamInput`/`AgentStreamOutput`, `WorkflowInvokeInput`/`WorkflowInvokeOutput`, `WorkflowError`, `WorkflowStreamInput`/`WorkflowStreamOutput`, and `AgentSessionCreated` when it creates a session. LLM/tool events are fired by whoever owns the model/tool call.
- Never hold a `parking_lot` guard across `.await`; registries and callback registries guard only synchronous lookups.
- A runner's `runner_id` is fixed at construction: `set_config` keeps the instance id (logging a mismatch) so `runner_id()` can return `&str`.
- `message_queue()` always returns a queue: before `start` it is the configured backend in the stopped state (operations fail with `Error::Config`), not `None`; a configuration naming a deferred transport (`pulsar` / `zeromq`) yields `DeferredMessageQueue`, whose every operation is `Error::Unsupported` naming the transport.
- `start()` refuses `distributed_mode` and deferred queue transports with `Error::Unsupported` and leaves the runner stopped.
- `set_guardrails` unregisters the previous binding first, so calling it twice replaces rather than double-scans.
- `RunBinding::run_id` and `RunStatusGuard` stay `pub(crate)` and carry `#[allow(dead_code)]` for that reason; do not widen them to silence lints.
