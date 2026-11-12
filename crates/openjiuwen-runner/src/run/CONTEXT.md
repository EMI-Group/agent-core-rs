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

// run/agent.rs
//   Runner::run_agent / Runner::run_agent_streaming (implemented as `pub(crate)` fns called by the facade)
// run/workflow.rs
//   Runner::run_workflow / Runner::run_workflow_streaming
// run/team.rs
//   Runner::run_agent_team / Runner::run_agent_team_streaming
// run/span.rs  (pub(crate))
//   RunSpan: opens a session-tracer Chain span when the session has a tracer, finishes with outputs or failure
// run/teardown.rs (pub(crate))
//   Teardown: an owned teardown action that runs on `finish().await` or is spawned from `Drop`
```

## Behaviour (the contract)

**Reference resolution.** `Id(_)` looks the resource up in `ResourceMgr` and a miss is `Error::NotFound("agent '<id>' is not registered")` (same wording for workflow/team) **before** any session is created. `Agent(..)`/`Workflow(..)`/`Team(..)` are used as-is. Python's `str | instance` union maps to these enums.

**Session id resolution (Python precedence).** `inputs["conversation_id"]` (non-empty string) → `RunOptions::session_id` → `RunnerConfig::default_session_id` for agents/workflows; agents and teams treat an empty/non-string value as absent.

**Agent run.**
1. Resolve the agent; fire `AgentInvokeInput` (`component` = card id, `input` = inputs, `session` = the resolved session when owned). `Abort` from the callback aborts the run with that error; other findings continue.
2. Session: `RunOptions::session` when present (caller-owned, never committed); else create `AgentSession::with_options` with `session_id`, `agent_id` = card id, `agent_name`, `envs`, `metadata`, `checkpointer` (options' or the runner's), and fire `AgentSessionCreated`.
3. Invoke inside `RunSpan` + `tokio::select!` against `options.cancellation` and the deadline.
4. Fire `AgentInvokeOutput` with the output value; on error fire nothing extra (Python emits no agent-error event).
5. Teardown for an owned session: `post_run()` (end stream + commit); a teardown failure is logged, never replaces a successful output; on an invocation error the teardown still runs and the invocation error wins.
6. `Error::Cancelled` for cancellation (the in-flight future is dropped); `Error::Timeout("run_agent timed out after <ms>ms")` for the deadline. A runner-level deadline (`RunnerConfig::default_run_timeout`) applies when `options.timeout` is `None`.

**Agent streaming.** Fire `AgentStreamInput` after resolving the session, then wrap `Agent::stream` in a guarded stream: per chunk fire `AgentStreamOutput` (with `index` = frame index), check cancellation between frames, enforce the deadline as a **total** budget. Termination: cancellation → one `Err(Error::Cancelled)` then end; deadline → one `Err(Error::Timeout(..))` then end; inner error → forwarded then end. Teardown runs after the last frame or on drop.

**Workflow run.** Resolve; session = `RunOptions::session` (caller-owned) else a `WorkflowSession` built with the resolved id, `checkpointer`, and `envs`; fire `WorkflowInvokeInput`, invoke with `WorkflowRunOptions { cancellation, timeout, stream_modes, .. }` (the workflow engine owns its own deadline/cancellation), fire `WorkflowInvokeOutput` on success or `WorkflowError` on failure. No post-run commit for a workflow session (the engine commits user inputs); a created session is closed via `Session::close` after the run.

**Workflow streaming.** `Workflow::stream_with(inputs, session, WorkflowRunOptions { stream_modes, .. })`, firing `WorkflowStreamInput` before and `WorkflowStreamOutput` per frame; teardown closes a created session.

**Team run.** Resolve; call `Team::start()` (idempotent); take a `SessionLease` with `RunOptions::team_session` (borrowed) or a session built from the resolved id (owned): an owned lease creates, binds, and later unbinds + cleans topics + ends + commits; a borrowed lease is used as bound by the caller (Python's unconditional bind/unbind is deliberately relaxed to the multiagent contract — document in the deviating method's rustdoc). `Team::invoke(inputs, Some(session))`; the result `Value` is returned verbatim. Callback events: none (Python's `AgentTeamEvents` are p2p/pub-sub, which the multiagent bus owns).

**Team streaming.** `Team::stream(inputs, Some(session))` frames forwarded verbatim; the lease is finished by `Teardown` (on completion or drop). Nothing is re-wrapped: the strategy already yields wire frames.

**Tracing.** Always emit `tracing` events (`openjiuwen_runner::run`) with `run_id`, `kind`, `component`, `session_id`, and duration on completion; additionally open a `Chain` span on the session tracer when the session exposes one (`RunSpan`). Span failures never affect the run.

## Constraints

- No session is created before the reference resolves; no callback fires when resolution fails.
- A caller-provided session is never committed, closed, or ended by the runner (documented on each method).
- Teardown is idempotent and never panics; `Drop` spawns it on the current runtime and logs when no runtime exists.
- `tokio::select!` cancellation is biased to the token first so a cancelled run cannot win the race by completing.
- Public streams are `Send + 'static`; built with `async_stream::stream!`.
- Files ≤ ~1000 lines: `mod.rs` (options + refs + session resolution + span + teardown), `agent.rs`, `workflow.rs`, `team.rs`.

## Routing Table

- `./src/run/mod.rs` → `RunOptions`, `AgentRef`/`WorkflowRef`/`TeamRef`, session resolution, `RunSpan`, `Teardown`
- `./src/run/agent.rs` → agent invoke + streaming
- `./src/run/workflow.rs` → workflow invoke + streaming
- `./src/run/team.rs` → team invoke + streaming over `SessionLease`
- `../tests/run_agent.rs`, `../tests/run_workflow.rs`, `../tests/run_team.rs` → behavior suites (siblings — read-only, escalate writes to the crate root)
