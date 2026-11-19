# run/ — execution APIs

## Intent

The runner's six execution entry points and everything they share: reference resolution (id vs instance), session creation/resolution, per-run cancellation and deadlines, callback firing, tracing spans, and teardown.
Ports the behaviour of Python `runner.py::_prepare_agent` / `_prepare_workflow` / `_create_agent_session` / `_create_workflow_session` and `team_runner.py::_run_base_team`, plus the streaming wrappers `run_agent_streaming` / `run_workflow_streaming` / `run_agent_team_streaming`.

## API Surface

```rust
// run/mod.rs
pub enum AgentRef { Id(String), Agent(Arc<dyn Agent>) }
pub enum WorkflowRef { Id(String), Workflow(Arc<Workflow>) }
pub enum TeamRef { Id(String), Team(Arc<dyn Team>) }
// From<&str>, From<String>, From<Arc<dyn T>>, and From<Arc<Concrete>> for ReActAgent/LlmAgent/WorkflowAgent/BaseTeam.
impl AgentRef { pub fn id(id: impl Into<String>) -> Self; pub fn of(agent: Arc<dyn Agent>) -> Self; }

pub struct RunOptions {
    pub session: Option<Arc<dyn Session>>,            // caller-owned session (agent/workflow)
    pub team_session: Option<Arc<AgentTeamSession>>,  // caller-owned team session
    pub session_id: Option<String>,                   // Python `session="id"`
    pub cancellation: CancellationToken,              // default: never cancelled
    pub timeout: Option<Duration>,                    // None -> RunnerConfig::default_run_timeout()
    pub stream_modes: Option<Vec<StreamMode>>,        // workflow streaming only
    pub envs: BTreeMap<String, Value>,                // applied to sessions the runner creates
    pub checkpointer: Option<Arc<dyn Checkpointer>>,  // overrides the runner's checkpointer
    pub metadata: BTreeMap<String, Value>,            // copied into created session metadata
}
impl Default / Clone / Debug for RunOptions
impl RunOptions { with_session, with_team_session, with_session_id, with_cancellation, with_timeout,
                  with_stream_modes, with_envs, with_checkpointer, with_metadata }

// RunServices: the owned, cheap-to-clone execution context the facade hands down
#[derive(Clone)]
pub struct RunServices {
    pub resources: Arc<ResourceMgr>,
    pub callbacks: Arc<CallbackFramework>,
    pub config: RunnerConfig,
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    pub guardrails: Option<Arc<GuardrailPipeline>>,
    pub monitor: Option<Arc<crate::monitor::RunMonitor>>,  // None: status mirroring is inert
    pub hitl: Option<Arc<crate::hitl::InteractionHub>>,    // None: no run is registered
}
impl RunServices { pub fn new(resources, callbacks, config, checkpointer) -> Self; pub fn with_guardrails(self, pipeline) -> Self;
                  pub fn with_monitor(self, monitor) -> Self; pub fn with_hitl(self, hub) -> Self;
                  pub fn require_local(&self) -> Result<()>; }  // delegates to RunnerConfig::require_local

// run/agent.rs — public free functions; `Runner::run_agent*` are one-line delegates
pub async fn invoke_agent(services: RunServices, agent: AgentRef, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
pub fn stream_agent(services: RunServices, agent: AgentRef, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<AgentChunk>>;
// run/workflow.rs
pub async fn invoke_workflow(services: RunServices, workflow: WorkflowRef, inputs: Value, options: RunOptions) -> Result<WorkflowOutput>;
pub fn stream_workflow(services: RunServices, workflow: WorkflowRef, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
// run/team.rs
pub async fn invoke_team(services: RunServices, team: TeamRef, inputs: Value, options: RunOptions) -> Result<Value>;
pub fn stream_team(services: RunServices, team: TeamRef, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<Value>>;
// run/span.rs  (pub(crate))
//   RunSpan: opens a session-tracer Chain span when the session has a tracer, finishes with outputs or failure
// run/teardown.rs (pub(crate))
//   Teardown: an owned teardown action that runs on `finish().await` or is spawned from `Drop`
```

## Behaviour (the contract)

**Distributed-mode guard (first, always).** Every path calls `RunServices::require_local()` before anything else — `Ok(())` for a local runner, and the crate's single `Error::Unsupported` (from `RunnerConfig::require_local` → `error::distributed_mode`) naming distributed runner mode and the reason: this build ships no distributed transport (Pulsar, ZeroMQ, and A2A reply-topic transports are deferred).
For the three `invoke_*` functions the refusal is the awaited `Err`, before a ref is resolved, a session is created, a callback fires, or a monitor/hub status exists.
For the three `stream_*` functions the refusal is the stream's **single terminal `Err` frame** — the guard is the first statement inside the `async_stream::stream!` body, yielding that error and returning — so nothing else happens and no frame follows it.
Pool admission applies the same guard before creating or queueing an entry (see `../pool/CONTEXT.md`).
The spawn seam is deliberately exempt: a spawned child is backend-selected, not mode-selected.

**Reference resolution.** `Id(_)` looks the resource up in `ResourceMgr` and a miss is `Error::NotFound("agent '<id>' is not registered")` (same wording for workflow/team) **before** any session is created. `Agent(..)`/`Workflow(..)`/`Team(..)` are used as-is. Python's `str | instance` union maps to these enums.

**Session id resolution (Python precedence).** `inputs["conversation_id"]` (non-empty string) → `RunOptions::session_id` → `RunnerConfig::default_session_id` for agents/workflows; agents and teams treat an empty/non-string value as absent, and an empty `RunOptions::session_id` counts as absent too (`resolve_session_id`).

**Run status and HITL seam.** Every invocation uses one run id (UUID v4) shared by the monitor status and the hub registration; both seams are inert when `RunServices::monitor`/`RunServices::hitl` is `None`.
The facade builds this snapshot once per call, so a seam installed afterwards (`Runner::set_monitor`/`set_hitl`) governs later runs only; an in-flight run keeps the monitor and hub it was admitted with.
Admission — the monitor's `Queued` begin plus the hub register/attach — is deferred until after every setup step that can fail (reference resolution, session preparation, and the `*InvokeInput`/`*StreamInput` callback), so a failed setup never leaves a run registered or begun.
`Running` is set immediately before the invocation.
Every terminal exit finishes the status as `Completed`, `Failed`, or `Cancelled` and then unregisters the hub entry.
A future or stream dropped mid-flight is finished as `Cancelled` with `{"reason": "run dropped"}` by the status guard, and the binding unregisters on drop.
A caller-provided agent/workflow session (`RunOptions::session`) is never attached to the hub watcher, while a session the runner created is; team runs always attach the lease session.
The detail of every transition is `{"component": <card id>}` — no payloads, no PII, nothing that grows per frame.
Status and mirror failures are logged by the monitor and the hub and never fail a run.

**Agent run.**
1. Resolve the agent; fire `AgentInvokeInput` (`component` = card id, `input` = inputs, `session` = the resolved session when owned). `Abort` from the callback aborts the run with that error; other findings continue.
2. Session: `RunOptions::session` when present (caller-owned, never committed); else create `AgentSession::with_options` with `session_id`, `agent_id` = card id, `agent_name`, `envs`, `metadata`, `checkpointer` (options' or the runner's), and fire `AgentSessionCreated`.
3. Admit the run — `RunStart::new(RunKind::Agent, card id)` on the monitor and the hub registration, then `Running` — see *Run status and HITL seam*; `team_name` is `None` and only an owned session is attached.
4. Invoke inside `RunSpan` + `tokio::select!` against `options.cancellation` and the deadline.
5. Fire `AgentInvokeOutput` with the output value; on error fire nothing extra (Python emits no agent-error event).
6. Teardown for an owned session: `post_run()` (end stream + commit); a teardown failure is logged, never replaces a successful output; on an invocation error the teardown still runs and the invocation error wins.
7. `Error::Cancelled` for cancellation (the in-flight future is dropped); `Error::Timeout("run_agent timed out after <ms>ms")` for the deadline. A runner-level deadline (`RunnerConfig::default_run_timeout`) applies when `options.timeout` is `None`.
8. A session created before an aborting `AgentSessionCreated` handler is torn down (`post_run`) before the abort error propagates, so an owned session never leaks.
9. An aborting `AgentInvokeOutput` handler wins over the successful output (the run reports the abort error) but only after the owned session's teardown has run.
10. The status is finished from that callback verdict (`Completed`, or `Failed` carrying the abort error) and the hub entry unregistered, before the run returns.

**Agent streaming.** Fire `AgentStreamInput` after resolving the session, then wrap `Agent::stream` in a guarded stream: per chunk fire `AgentStreamOutput` (with `index` = frame index) — the chunk is projected with `to_value` only when that event has subscribers, so a run with none pays no serialisation per frame — check cancellation between frames, enforce the deadline as a **total** budget. Termination: cancellation → one `Err(Error::Cancelled)` then end; deadline → one `Err(Error::Timeout(..))` then end; inner error → forwarded then end. The run is admitted after `AgentStreamInput` succeeds (an abort there registers nothing) and its status is finished from the stream's failure before the terminal `Err`; teardown runs after the last frame or on drop.

**Workflow run.** Resolve; session = `RunOptions::session` (caller-owned) else a `WorkflowSession` built with the resolved id, `checkpointer`, and `envs`; fire `WorkflowInvokeInput`, invoke with `WorkflowRunOptions { cancellation, timeout, stream_modes, .. }` where `timeout` is the effective deadline (`options.timeout`, else `RunnerConfig::default_run_timeout()`, the same rule the agent and team paths apply; the engine owns the race), fire `WorkflowInvokeOutput` on success or `WorkflowError` on failure (an aborting `WorkflowError` handler is logged and the engine's error wins). No post-run commit for a workflow session (the engine commits user inputs); a created session is closed via `Session::close` after the run. The run is admitted (`RunKind::Workflow`, caller-owned session never attached) only after `WorkflowInvokeInput` succeeds, and its status is finished from the output callback's verdict or from the engine's error.

**Workflow streaming.** `Workflow::stream_with(inputs, session, WorkflowRunOptions { cancellation, timeout, stream_modes, .. })` with the effective deadline (the engine's `invoke` enforces it; its `stream` path currently applies no timeout wrapper, so the runner passes it through rather than re-timing the call), firing `WorkflowStreamInput` before and `WorkflowStreamOutput` per frame (a frame is projected with `to_wire` only when that event has subscribers); teardown closes a created session. The run is admitted after `WorkflowStreamInput` succeeds and its status is finished from the stream's failure before the terminal `Err`.

**Team run.** Resolve; call `Team::start()` (idempotent); take a `SessionLease` with `RunOptions::team_session` (borrowed) or a session built from the resolved id (owned): an owned lease creates, binds, and later unbinds + cleans topics + ends + commits; a borrowed lease is used as bound by the caller (Python's unconditional bind/unbind is deliberately relaxed to the multiagent contract — document in the deviating method's rustdoc).
`Team::invoke(inputs, Some(session))` is driven through `run_with_limits`, so it is bounded by `RunOptions::cancellation` and by the effective deadline — `options.timeout`, else `RunnerConfig::default_run_timeout()` when the runner configures one, the same rule the agent path applies.
A tripped token returns `Error::Cancelled` and an expired deadline `Error::Timeout("run_agent_team timed out after <ms>ms")`, both after the lease teardown, the span, and the mirrored run are finalised.
The runner can only race the invocation: a cancelled or timed-out call drops the in-flight `Team::invoke` future, because the multiagent seam takes a session, not a token, so a strategy that keeps a task of its own alive goes on in the background.
The result `Value` is returned verbatim.
Callback events: none (Python's `AgentTeamEvents` are p2p/pub-sub, which the multiagent bus owns).
The run is admitted (`RunKind::Team`, `team_name` = card id) once the lease holds the session — the last point on this path that cannot fail — and always attaches that lease session to the hub watcher.

**Team streaming.** `Team::stream(inputs, Some(session))` frames are forwarded verbatim — nothing is re-wrapped, the strategy already yields wire frames — while every step is polled through `next_stream_step` against `RunOptions::cancellation` and the effective deadline (`options.timeout`, else `RunnerConfig::default_run_timeout()`), the same rule `invoke_team` applies.
A tripped token or an expired deadline yields exactly one terminal `Err` — `Error::Cancelled`, or the `run_agent_team` timeout error — and no frame follows it.
Before that frame is yielded, the span, the trace event, the run status, the hub binding, and the lease are finalised by `Teardown` (on completion or drop); a frame error finishes the status as `Failed`, a clean end as `Completed`, and an early-dropped stream as `Cancelled` by the status guard.

**Team teardown outcome.** A lease-teardown failure is logged (`tracing::warn`, target `openjiuwen_runner::run`) and never replaces a successful team result.

**Tracing.** Always emit `tracing` events (`openjiuwen_runner::run`) with `run_id`, `kind`, `component`, `session_id`, and duration on completion; additionally open a `Chain` span on the session tracer when the session exposes one (`RunSpan`). Span failures never affect the run.

## Constraints

- The distributed-mode guard runs first on every path; no session is created before the reference resolves; no callback fires when resolution fails; no stream emits anything but the refusal frame for a distributed runner.
- No session is created before the reference resolves; no callback fires when resolution fails.
- A caller-provided session is never committed, closed, or ended by the runner (documented on each method).
- Teardown is idempotent and never panics; `Drop` spawns it on the current runtime and logs when no runtime exists.
- The monitor and hub seams are synchronous best-effort mirrors: their failures are logged by the monitor/hub and never propagated to a run.
- `tokio::select!` cancellation is biased to the token first so a cancelled run cannot win the race by completing.
- Public streams are `Send + 'static`; built with `async_stream::stream!`.
- Files ≤ ~1000 lines: `mod.rs` (options + refs + session resolution + span + teardown), `agent.rs`, `workflow.rs`, `team.rs`.

## Routing Table

- `./src/run/mod.rs` → `RunServices` (including the `monitor`/`hitl` seams and `require_local`), `RunOptions`, `AgentRef`/`WorkflowRef`/`TeamRef`, session resolution, `RunSpan`, `Teardown`
- `./src/run/agent.rs` → agent invoke + streaming
- `./src/run/workflow.rs` → workflow invoke + streaming
- `./src/run/team.rs` → team invoke + streaming over `SessionLease`
- `../tests/run_agent.rs`, `../tests/run_workflow.rs`, `../tests/run_team.rs` → behavior suites (siblings — read-only, escalate writes to the crate root)
