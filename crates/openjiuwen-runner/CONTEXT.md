# openjiuwen-runner

## Intent

The process-level orchestrator of the kernel: one `Runner` owns the resource registry, the callback framework, and the message queue, and it is the only sanctioned entry point for running an agent, a workflow, or an agent team.
Ports Python `openjiuwen/core/runner/` (v0.1.20, ~13.9k lines, 53 files) with the essential seams kept and the operational machinery deferred:
`runner.py` + `runner_config.py` (facade, config, lifecycle), `resources_manager/` (the four typed registries), `callback/` (event framework), `message_queue_base.py` + `dunner/dmessage_queue/message_queue_fake.py` (in-process queue), `spawn/` (subprocess protocol), and `team_runner.py`'s **in-process** team path.
Deferred and documented in `## Deferred Python Behavior`: distributed mode (Pulsar/ZeroMQ), reply-topic routing/`RemoteAgent`, real subprocess spawning, team pools/HITL/monitoring, tags-as-registry-index, and MCP/sys-op/model/prompt managers.

The crate sits at layer 4 (`docs/ARCHITECTURE.md` §1): it depends on `common`, `foundation`, `session`, `workflow`, `agent`, `multiagent`, and `security`, and is consumed by `openjiuwen-core` (kernel umbrella), `openjiuwen-harness`, `openjiuwen-agent-teams`, and `openjiuwen-cli`.
No crate below depends on it: the runner drives lower layers through their public seams (`Agent`, `Workflow`, `Team`, `Session`), never by reaching into internals.

## API Surface

Everything below is re-exported flat at the crate root.

### Facade — `src/runner.rs`
```rust
pub struct Runner { /* config, ResourceMgr, CallbackFramework, MessageQueue, checkpointer, spawner, guardrails */ }
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
    pub fn message_queue(&self) -> Arc<dyn MessageQueue>;        // the live queue (rebuilt by set_config+start)
    pub fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>>; // built by start()
    pub fn is_started(&self) -> bool;
    pub async fn start(&self) -> Result<()>;                    // idempotent; starts the queue, builds the checkpointer
    pub async fn stop(&self) -> Result<()>;                     // idempotent; stops the queue, releases resources
    pub async fn release(&self, session_id: &str) -> Result<()>; // checkpointer.release (Python Runner.release)
    // execution (see ./src/run/CONTEXT.md)
    pub async fn run_agent(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
    pub fn run_agent_streaming(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn run_workflow(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> Result<WorkflowOutput>;
    pub fn run_workflow_streaming(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn run_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<Value>;
    pub fn run_agent_team_streaming(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<Value>>;
    // security (see ./src/callback/CONTEXT.md §Security binding)
    pub fn set_guardrails(&self, pipeline: Arc<GuardrailPipeline>) -> Vec<CallbackId>;
    pub fn guardrails(&self) -> Option<Arc<GuardrailPipeline>>;
    pub async fn scan(&self, ctx: GuardrailContext) -> Result<GuardrailOutcome>;
    // subprocess seam (see ./src/spawn/CONTEXT.md)
    pub fn set_spawner(&self, spawner: Arc<dyn AgentSpawner>);
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
```
`global()` is the ONE documented process-global: a `OnceLock<RwLock<Option<Arc<Runner>>>>` slot, created on first use with `RunnerConfig::default()`, replaced wholesale by `reset_global()`.
Everything else is per-instance and `Runner::new()` is what tests use.

### Configuration — `src/config.rs`
`RunnerConfig` (`runner_id`, `distributed_mode=false`, `distributed`, `message_queue`, `default_session_id="default_session"`, `checkpointer: Option<CheckpointerConfig>`, `default_run_timeout_secs`, `shutdown_timeout_secs=5.0`), `MessageQueueConfig` (`kind`, `capacity=1024`, `request_timeout_secs=30.0`, `url`, `max_workers=8`), `MessageQueueKind { InMemory = "in_memory", Pulsar, ZeroMq }`, `DistributedConfig` (data only), and the constants `DEFAULT_SESSION_ID`, `CONVERSATION_ID_KEY`, `DEFAULT_RUNNER_ID`, `DEFAULT_TOPIC_CAPACITY`, `DEFAULT_REQUEST_TIMEOUT_SECS`.
All of it is serde (`snake_case`, `default`, unknown fields tolerated) so a host can persist its runner configuration.

### `src/resources/` — the four registries
`ResourceMgr` (tools, agents, workflows, teams) with `register_*` (duplicate = `Error::Validation`), `register_*_with(policy)`, `*_replacing`, `get`, `card`, `by_name`, `remove`, `cards`, `ids`, `len`, `is_empty`, `clear_*`, `revision`, `release()`.
Providers are `Arc<dyn Fn() -> T + Send + Sync>` (Python parity: a fresh instance per lookup); agents/workflows/teams are card+provider, tools are providers too (built from the tool's own card).
See `./src/resources/CONTEXT.md`.

### `src/callback/` — the event framework
`CallbackEvent` (typed enum over Python's `_framework:<event>` names, plus `Custom(String)`), `CallbackContext`, `CallbackHandler`, `CallbackVerdict`, `CallbackReport`, `CallbackFailure`, `CallbackId`, `CallbackRegistration`, `CallbackInfo`, `CallbackFramework` (`register*`, `unregister*`, `trigger`, `has_subscribers`, `handlers`, `enabled`), and the guardrail binding (`scan_stage_for`, `GuardrailHandler`, `bind_guardrails`).
See `./src/callback/CONTEXT.md`.

### `src/mq/` — the message queue
`QueueMessage`, `MessageHandler`, `MessageQueue` trait (`start`/`stop`/`is_started`/`subscribe`/`unsubscribe`/`add_handler`/`publish`/`request`/`topics`), `InMemoryMessageQueue`, `create_message_queue`, `MessageQueueKind` (re-exported from `config`).
See `./src/mq/CONTEXT.md`.

### `src/run/` — execution APIs
`RunOptions`, `AgentRef`, `WorkflowRef`, `TeamRef`, session resolution, per-run cancellation/timeout, teardown, and tracing spans.
See `./src/run/CONTEXT.md`.

### `src/pool/` — the team runtime pool
`TeamPool` (`new`, `with_config`, `from_runner_config`, `start`/`try_start`/`acquire`, `get`/`list_active`, `stop`/`stop_all`, `pause`/`resume`, `take_payloads`, `capacity`/`available_permits`, `len`/`is_empty`/`is_closed`), `PoolRun`, `PoolLease`, `PoolRequest`, `PoolKey`, `PoolFrame`, `ActiveTeamInfo`, `PoolError`, `PoolShutdown`, and `TeamPoolConfig` (in `src/config.rs`).
See `./src/pool/CONTEXT.md`.

### `src/spawn/` — subprocess seam
`SpawnAgentKind`, `SpawnSpec`, `SpawnedAgent`, `AgentSpawner`, `UnsupportedSpawner`, and the NDJSON protocol types (`SpawnMessage`, `SpawnMessageKind`).
See `./src/spawn/CONTEXT.md`.

### `src/error.rs`
Thin constructors for the runner's error vocabulary (`not_found`, `invalid_session`, `timeout`, `unsupported`, `not_started`) over `openjiuwen_common::error::{Error, Result}`; no parallel error enum.

## Constraints

- Follows `docs/ARCHITECTURE.md`: async-first tokio, `openjiuwen_common::error` results, serde-serializable public data, no `unsafe`, `tracing` only (target `openjiuwen_runner::<module>`), no stdout.
- Locks are `parking_lot` for short synchronous sections and `tokio::sync` for awaitable coordination; a lock is never held across `.await`.
- Every spawned task has an owner: queue topic workers are joined by `stop`, run teardown tasks are bounded by their session's lifetime.
- No panics on user input: every missing id, bad session type, duplicate registration, closed queue, deadline, and cancellation is an `Error`.
- Files stay ≤ ~1000 lines; split by responsibility.
- Two global-state rules: the `Runner` singleton is documented and replaceable (`reset_global`); the callback framework and registries hang off a runner instance, never off module statics.
- Tests are offline and deterministic (`openjiuwen_test_support::{ScriptedModel, MockLlmServer}`, `tempfile`); timeouts in tests are short and only used to prove a deadline, never to synchronize.
- The foreign repository is read-only.

## Design Decisions

- **One `Runner` type, instance-first.** Python's `_RunnerImpl` + classmethod facade collapses to one struct: `Runner::new()` is the isolated instance, `Runner::global()` the parity singleton. This removes Python's "constructing a second runner overwrites the process config" trap.
- **Registries store cards + providers, not live state.** Python's `ResourceMgr` keys on `card.id` and calls the provider on every `get_agent`/`get_workflow`; the Rust port keeps that contract (fresh instance per lookup) so a provider can build per-run state, and adds `card`/`by_name` lookups, per-registry revision counters (cache invalidation), and an explicit `RegisterPolicy { Fail, Keep, Replace }` in place of Python's `refresh`/`skip_if_exists` keyword soup.
- **Deduplication of transports.** `openjiuwen-multiagent` already owns a team-scoped request/reply bus; the runner's `MessageQueue` is the *generic topic queue* (pub/sub plus request/reply) that hosts need for runner-level plumbing. Neither re-implements the other; the runner's queue is a plain trait so `openjiuwen-extensions` can plug a broker in.
- **Callbacks carry a typed context, not `**kwargs`.** Python narrows `*args/**kwargs` per handler with signature inspection; Rust passes one `CallbackContext` and handlers read what they need. Result semantics are encoded in `CallbackVerdict::{Continue, Abort, Stop}` (Python's "swallowed exception" vs `AbortError` vs `AbortError(is_error=False)`), so error isolation is explicit rather than accidental.
- **Event names are an enum with the exact Python strings.** `CallbackEvent::as_str()` returns `"_framework:llm_invoke_input"` etc., so a harness that subscribes by string and a guardrail that binds by `ScanStage::event()` agree without a translation table. `Custom(String)` keeps unknown/custom events expressible.
- **Security wiring is a binding, not a special case.** `GuardrailHandler` implements `CallbackHandler` over a `GuardrailPipeline`; `bind_guardrails` registers one handler per guarded stage at Python's priority 100 in the `guardrail` namespace. A blocked verdict is `Ok(CallbackVerdict::Abort(Error::Security))`, which `trigger` turns into the call's error (Python's `AbortError`). Non-blocking findings stay in the `CallbackReport` (Python swallows them).
- **Cancellation and deadlines are run-scoped.** `RunOptions` carries a `CancellationToken` and an optional timeout; the runner races the invocation against them because the lower agent/workflow seams take a session, not a token. A cancelled run returns `Error::Cancelled`; a timed-out run returns `Error::Timeout`, and both drop the in-flight future (documented: the lower layer does not observe the token itself).
- **Session ownership is explicit.** A caller-provided session is used and left alone; a session the runner creates is committed/closed by the runner (`AgentSession::post_run`, `SessionLease::finish`). Teardown runs on stream completion or on drop (a `Drop` guard spawns it), so an early-dropped stream does not leak an unbound team session.
- **No hidden retries or fallbacks.** A missing registry id is `Error::NotFound` with the id in the message (Python raised `AttributeError` on `None.invoke`), an unfinished `start()` is `Error::Config`, and a deferred backend is `Error::Unsupported` with a message naming the seam.
- **Subprocess spawning is a trait, not a fork/exec today.** `AgentSpawner`/`SpawnedAgent` + the NDJSON `SpawnMessage` protocol pin the contract a future `openjiuwen-agent-teams`/extensions implementation fills; the built-in `UnsupportedSpawner` returns `Error::Unsupported`.

## Routing Table

- `./src/runner.rs` → `Runner` facade, lifecycle (`start`/`stop`/`release`), global singleton, security/spawner wiring
- `./src/config.rs` → `RunnerConfig`, `MessageQueueConfig`, `MessageQueueKind`, `DistributedConfig`, defaults
- `./src/error.rs` → error constructors and message wording
- `./src/resources/` → `ResourceMgr`, `Registry`, providers, register policies, revision counters (`resources/CONTEXT.md`)
- `./src/callback/` → `CallbackEvent`, `CallbackContext`, handlers, `CallbackFramework`, guardrail binding (`callback/CONTEXT.md`)
- `./src/mq/` → `QueueMessage`, `MessageQueue`, `InMemoryMessageQueue`, backend factory (`mq/CONTEXT.md`)
- `./src/run/` → `RunOptions`, `AgentRef`/`WorkflowRef`/`TeamRef`, session resolution, cancellation/timeouts, teardown, spans (`run/CONTEXT.md`)
- `./src/pool/` → `TeamPool` admission, leases, bounded concurrency, idle eviction, HITL payloads (`pool/CONTEXT.md`)
- `./src/monitor/` → `RunMonitor` run phase/outcome mirroring and events (`monitor/CONTEXT.md`)
- `./src/hitl/` → `InteractionHub` live-run registry, inbox delivery, pause/cancel signals (`hitl/CONTEXT.md`)
- `./src/spawn/` → spawn traits, `SpawnSpec`, NDJSON protocol (`spawn/CONTEXT.md`)
- `./tests/` → integration suites (`tests/CONTEXT.md`)

## Test Strategy

- Unit tests beside every module: config defaults/serde, error wording, registry insert/lookup/policy/revision, event-name mapping, framework ordering/isolation/verdicts, guardrail stage mapping, queue delivery ordering/errors/timeouts, spawn protocol round-trip, session-id resolution, ref parsing.
- Integration suites in `./tests/`: `resources.rs`, `callbacks.rs`, `security_binding.rs`, `message_queue.rs`, `runner_lifecycle.rs`, `run_agent.rs`, `run_workflow.rs`, `run_team.rs`, `api_surface.rs`, and the team-pool suites `team_pool.rs`, `team_pool_lifecycle.rs`, `team_pool_leases.rs`, `team_pool_hitl.rs`.
- Behavior ports cite the Python test they mirror in a comment (e.g. `// ports tests/unit_tests/core/runner/test_resource_manager.py`).
- Deterministic and offline: `ScriptedModel` for agents, `MockLlmServer` only when HTTP is under test, no sleeps used for ordering, `tempfile` for file checkpoints.
- Gates: `cargo fmt -p openjiuwen-runner`, `cargo test -p openjiuwen-runner`, `cargo clippy -p openjiuwen-runner --all-targets -- -D warnings`, all clean and offline.

## Deferred Python Behavior

- **Distributed mode**: `drunner/` entirely (Pulsar backend, `FakeMQ` wire envelopes, reply-topic subscription, `ResponseCollector`, `MqRemoteClient`/`MqServerAdapter`, `RemoteAgent`, A2A adapter bootstrap). `RunnerConfig::distributed` is data only; `create_message_queue` returns `Unsupported` for `pulsar`/`zeromq`.
- **Subprocess spawn**: `spawn/process_manager.py` and `child_process.py` (real `python -m ...` child, health checks, graceful→force escalation, Windows stdio emulation, spawn logging config). The trait, spec, and NDJSON protocol are ported; `UnsupportedSpawner` is the built-in.
- **Team product surface**: the in-process pool IS implemented in `./src/pool/` (admission with the reject-kind vocabulary, acquire/release leases, bounded concurrency, idle eviction and orphan reclaim, `list_active`, pause/resume/stop, HITL payload buffering). Still deferred: `delete_agent_team`, `register_human_agent_inbound`, `start_agent_team_streaming`, `TeamStreamLogger`/monitor TUI, `run_context`, the live `InteractGate` wake-up, and `member=True` spawn routing. In-process `Team::invoke`/`Team::stream` over the multiagent seam IS implemented.
- **Resource managers beyond the four registries**: tags (`TagMgr`, `Tag`, `TagMatchStrategy`), MCP servers/tools, sys-operations, models, prompts. Models/prompts are host concerns in Rust (`init_model`, prompt values are passed at call sites).
- **Callback framework extras**: decorators (`emit_before/after/around`, `transform_io`, `wrap`/`on_wrap`), `EventFilter` implementations (rate limit, circuit breaker, validation, auth, conditional, param modify), hooks, `CallbackChain` (retry/rollback), `trigger_parallel`/`trigger_until`/`trigger_delayed`/`trigger_stream`/`trigger_generator`, metrics/history/state dump, `lazy_callback_framework`. Every event name is preserved so these can land without a wire change.
- **Runner internals**: `ReadWriteLockManager`, root task group / `ContextVar` task manager, `ModelContext`, `envs` forwarding (accepted in `RunOptions` and applied to created sessions' config only), remote-agent detection, `LegacyBaseAgent` handling, `enable_session_controller`, `enable_a2a`.
- Python quirks deliberately not reproduced: missing ids raising `AttributeError`, non-idempotent distributed `start`, `stop()`'s `finally` skipping teardown, in-flight responses orphaned on queue shutdown, `message_queue_inmemory.py`'s `raise response(...)` bug, duplicate callback registration firing twice (Rust keeps it legal but offers `CallbackId` for explicit removal — see `callback/CONTEXT.md`), and the global config being overwritten by constructing a second runner.

## Notes for Agents

- Use `Runner::new()` in tests; never `Runner::global()`. `reset_global()` exists so a test that must touch the singleton can restore a clean one.
- The runner never reads `OPENJIUWEN_*` itself: configuration is injected through `RunnerConfig`. `.env` loading is a binary concern (`openjiuwen-cli`).
- `run_agent`/`run_workflow`/`run_agent_team` accept either a registry id or a live instance via `AgentRef`/`WorkflowRef`/`TeamRef`; an id is resolved through `ResourceMgr` and a missing id fails with `Error::NotFound` before any session is created.
- Execution does not require `start()`: the facade builds a `RunServices` snapshot per call and delegates to the `run::*` free functions. `start()` is for the queue and the checkpointer.
- The streaming entry points return immediately (they are not `async`), so a resolution or session failure is delivered as the stream's single `Err` frame; per-frame failures terminate the stream after one `Err`.
- `RunOptions::with_envs` and `with_metadata` are per-entry (`with_envs("K", json!(1))`), matching how hosts build them incrementally; the fields themselves are plain `BTreeMap`s.
- `Runner::stop()` releases the resource registries (Python `ResourceMgr.release`), so a stopped runner is also an emptied one; `set_config` takes effect on the next `start` and preserves `runner_id`.
- `RunOptions::default()` means: no caller session, no deadline, never cancelled, session id from `inputs["conversation_id"]` else `RunnerConfig::default_session_id`.
- Callback events the runner itself fires: `AgentInvokeInput`/`AgentInvokeOutput`, `AgentStreamInput`/`AgentStreamOutput`, `WorkflowInvokeInput`/`WorkflowInvokeOutput`, `WorkflowError`, `WorkflowStreamInput`/`WorkflowStreamOutput`, and `AgentSessionCreated` when it creates a session. LLM/tool events are fired by whoever owns the model/tool call (agent rails or harness code); the runner ships the binding (`set_guardrails`) and the `scan` shortcut, not the invocation sites.
- Never hold a `parking_lot` guard across `.await`; the resource registry and callback registry guard only synchronous lookups.
- A runner's `runner_id` is fixed at construction: `set_config` keeps the instance id (logging a mismatch) so `runner_id()` can return `&str` and identity never changes mid-life.
- `message_queue()` always returns a queue: before `start` it is the configured backend in the stopped state (operations fail with `Error::Config`), not `None`.
- `stop` clears the built checkpointer and calls `resource_mgr().release()`; `release(session_id)` without a checkpointer is a no-op.
- `set_guardrails` unregisters the previous binding first, so calling it twice replaces rather than double-scans.
