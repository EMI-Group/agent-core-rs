# callback/ — the event framework and guardrail binding

## Intent

The runner's publish/subscribe seam for lifecycle events: typed event names matching Python's `_framework:<event>` vocabulary, priority-ordered dispatch with explicit error isolation, and the guardrail binding that turns a `GuardrailPipeline` into event handlers.
Ports Python `openjiuwen/core/runner/callback/` (`events.py`, `framework.py`'s registration/`trigger`, `errors.py`'s `AbortError`, `utils.py`'s `trigger`) with the decorators, filters, hooks, chains, and trigger variants deferred (see the crate-level `## Deferred Python Behavior`).

## API Surface

```rust
// event.rs
pub enum CallbackEvent {
    AgentStarted, AgentInvokeInput, AgentInvokeOutput, AgentStreamInput, AgentStreamOutput,
    AgentP2pReceived, AgentPubsubReceived,
    WorkflowStarted, WorkflowFinished, WorkflowError, WorkflowCancelled,
    WorkflowNodeExecuted, WorkflowNodeError, WorkflowEdgeTraversed,
    WorkflowLoopStarted, WorkflowLoopFinished,
    WorkflowInvokeInput, WorkflowInvokeOutput, WorkflowStreamInput, WorkflowStreamOutput,
    WorkflowComponentBatchInput, WorkflowComponentBatchOutput,
    WorkflowComponentStreamInput, WorkflowComponentStreamOutput,
    LlmCallStarted, LlmCallError, LlmResponseReceived, LlmResponseStarted,
    LlmInvokeInput, LlmInvokeOutput, LlmStreamInput, LlmStreamOutput, LlmStreamCompleted,
    LlmRoute, LlmRouteAttempt, LlmInput, LlmOutput,
    ToolCallStarted, ToolCallFinished, ToolCallError, ToolResultReceived,
    ToolParseStarted, ToolParseFinished,
    ToolInvokeInput, ToolInvokeOutput, ToolStreamInput, ToolStreamOutput, ToolAuth,
    ContextUpdated, ContextOffloaded, ContextRetrieved, ContextCleared, ContextCompressionState,
    SessionCreated, AgentSessionCreated,
    RetrievalStarted,
    MemoryAdded, MemorySearchStarted, MemorySearchFinished, MemoryUpdated, MemoryDeleted,
    TaskCreated, TaskRunning, TaskCompleted, TaskFailed, TaskCancelled, TaskTimeout,
    Custom(String),
}
impl CallbackEvent {
    pub fn name(&self) -> &str;                 // "agent_invoke_input"
    pub fn as_str(&self) -> &str;               // "_framework:agent_invoke_input"  (Python value)
    pub fn from_name(name: &str) -> Option<Self>;   // accepts with or without the "_framework:" prefix
    pub fn domain(&self) -> EventDomain;        // Agent / Team / Workflow / Llm / Tool / Context / Session / Retrieval / Memory / Task / Custom
    pub fn is_llm_or_tool(&self) -> bool;
}
pub enum EventDomain { Agent, Team, Workflow, Llm, Tool, Context, Session, Retrieval, Memory, Task, Custom }
impl CallbackEvent { pub const ALL: &'static [CallbackEvent]; }   // every non-custom variant, for iteration/tests

// context.rs
pub struct CallbackContext {
    pub event: CallbackEvent,
    pub session: Option<Arc<dyn Session>>,
    pub component: String,                    // agent id / workflow id / team id / component id
    pub input: Option<Value>,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub index: Option<i64>,                   // stream frame index
    pub metadata: BTreeMap<String, Value>,
    pub guardrail: Option<GuardrailContext>,   // content to scan when the caller has it
}
impl CallbackContext {
    pub fn new(event: CallbackEvent) -> Self;                  // component "", everything else None/empty
    pub fn with_session(self, session: Arc<dyn Session>) -> Self;
    pub fn with_component(self, component: impl Into<String>) -> Self;
    pub fn with_input(self, input: Value) -> Self;
    pub fn with_output(self, output: Value) -> Self;
    pub fn with_error(self, error: impl Into<String>) -> Self;
    pub fn with_index(self, index: i64) -> Self;
    pub fn with_metadata(self, key: impl Into<String>, value: Value) -> Self;
    pub fn with_guardrail(self, ctx: GuardrailContext) -> Self;
    pub fn session_id(&self) -> Option<&str>;
}
pub enum CallbackVerdict { Continue, Abort(Error), Stop }      // Python: None / AbortError(cause) / AbortError(is_error=False)
pub struct CallbackFailure { pub id: CallbackId, pub handler: String, pub error: Error }
pub struct CallbackReport { pub event: CallbackEvent, pub fired: usize, pub failures: Vec<CallbackFailure>, pub aborted: bool }
impl CallbackReport { pub fn is_clean(&self) -> bool; }

// handler.rs
pub type CallbackResult = Result<CallbackVerdict>;
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    fn name(&self) -> &str { "handler" }
    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult;
}
pub struct SyncHandler<F>(..);   // F: Fn(&CallbackContext) -> CallbackResult + Send + Sync
pub struct AsyncHandler<F>(..);  // F: Fn(CallbackContext) -> BoxFuture<'static, CallbackResult> + Send + Sync
// A handler returning Err(_) is recorded and dispatch continues (Python's swallowed exception).

// framework.rs
pub struct CallbackId(u64);                     // Copy, Eq, Hash, Display
pub struct CallbackInfo { pub id: CallbackId, pub event: CallbackEvent, pub name: String,
                          pub priority: i32, pub namespace: String, pub once: bool,
                          pub enabled: bool, pub tags: BTreeSet<String>, pub sequence: u64 }
pub struct CallbackRegistration { pub event: CallbackEvent, pub handler: Arc<dyn CallbackHandler>,
                                  pub priority: i32, pub namespace: String, pub tags: BTreeSet<String>, pub once: bool }
impl CallbackRegistration { pub fn new(event, handler) -> Self;   // priority 0, namespace "default"
    pub fn with_priority(self, p: i32) -> Self; pub fn with_namespace(..); pub fn with_tag(..); pub fn once(self) -> Self }
pub struct CallbackFramework { .. }
impl CallbackFramework {
    pub fn new() -> Arc<Self>;
    pub fn register(&self, registration: CallbackRegistration) -> CallbackId;
    pub fn register_sync<F>(&self, event: CallbackEvent, priority: i32, handler: F) -> CallbackId;
    pub fn register_async<F>(&self, event: CallbackEvent, priority: i32, handler: F) -> CallbackId;
    pub fn unregister(&self, id: CallbackId) -> bool;
    pub fn unregister_event(&self, event: CallbackEvent) -> usize;
    pub fn unregister_namespace(&self, namespace: &str) -> usize;
    pub fn unregister_tag(&self, tag: &str) -> usize;
    pub fn unregister_all(&self) -> usize;
    pub fn set_enabled(&self, id: CallbackId, enabled: bool) -> bool;
    pub fn has_subscribers(&self, event: CallbackEvent) -> bool;
    pub fn handlers(&self, event: CallbackEvent) -> Vec<CallbackInfo>;
    pub fn events(&self) -> Vec<CallbackEvent>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub async fn trigger(&self, event: CallbackEvent, ctx: &CallbackContext) -> Result<CallbackReport>;
}

// security.rs
pub const GUARDRAIL_PRIORITY: i32 = 100;
pub const GUARDRAIL_NAMESPACE: &str = "guardrail";
pub fn scan_stage_for(event: CallbackEvent) -> Option<ScanStage>;   // LlmInvokeInput/Output, ToolInvokeInput/Output, UserInput
pub fn event_for_stage(stage: &ScanStage) -> CallbackEvent;          // inverse
pub struct GuardrailHandler { .. }                                  // pipeline + stage
impl GuardrailHandler { pub fn new(pipeline: Arc<GuardrailPipeline>, stage: ScanStage) -> Self; }
impl CallbackHandler for GuardrailHandler { .. }
pub fn bind_guardrails(framework: &CallbackFramework, pipeline: Arc<GuardrailPipeline>) -> Vec<CallbackId>;
```

## Dispatch semantics (the contract)

- Handlers run in descending `priority`, ties broken by registration order (`sequence`).
- A handler returning `Ok(Continue)` lets the next handler run; `Ok(Stop)` ends dispatch without an error (`aborted = true` in the report); `Ok(Abort(e))` ends dispatch and `trigger` returns `Err(e)`; `Err(e)` is recorded in `report.failures` and logged at `warn`, and dispatch continues.
- `trigger` snapshots the handler list under a read lock, releases the lock, then awaits each handler in order. No lock is held across `.await`.
- `once = true` removes the handler after its first successful (`Continue`/`Stop`) run. A failed (`Err`) or aborted run does not consume it.
- A disabled handler is skipped without being reported as fired.
- Duplicate registration is legal and dispatches twice (Python parity); `CallbackId` is the removal handle.
- `trigger` with no subscribers returns `Ok(CallbackReport { fired: 0, .. })` without touching the context.
- A guardrail handler builds its `GuardrailContext` from `ctx.guardrail` when present; otherwise it derives one from `ctx.input` (tool/LLM input) or `ctx.output` (tool/LLM output) as raw text and sets the stage. A blocked outcome is `Ok(Abort(error))` where `error` is `outcome.enforce()`'s `Error::Security`; allowed outcomes are `Ok(Continue)` (non-blocking findings are logged and dropped, matching Python's swallowed `GuardrailError`).

## Constraints

- `CallbackEvent::as_str()` values are a wire contract with `openjiuwen-security` (`ScanStage::event()`) and with harness code that subscribes by string; every Python preset name must be representable, and `from_name` must round-trip all of `ALL`.
- Errors are `openjiuwen_common::error::{Error, Result}`; no parallel error type. `CallbackVerdict` carries the abort error instead of a bespoke exception.
- The framework owns no tasks and no globals; it is `Arc`-shared from a `Runner`.
- `tracing` target `openjiuwen_runner::callback`; never log `CallbackContext` payload contents at info level (they may contain prompts).
- Files ≤ ~1000 lines: `event.rs` holds the vocabulary and mapping, `framework.rs` the registry/dispatch.

## Routing Table

- `./src/callback/event.rs` → `CallbackEvent`, `EventDomain`, name mapping, `ALL`
- `./src/callback/context.rs` → `CallbackContext`, `CallbackVerdict`, `CallbackReport`, `CallbackFailure`
- `./src/callback/handler.rs` → `CallbackHandler`, `SyncHandler`, `AsyncHandler`
- `./src/callback/framework.rs` → `CallbackId`, `CallbackInfo`, `CallbackRegistration`, `CallbackFramework`
- `./src/callback/security.rs` → stage mapping, `GuardrailHandler`, `bind_guardrails`
- `./src/callback/mod.rs` → re-exports
- `../tests/callbacks.rs`, `../tests/security_binding.rs` → behavior suites (siblings — read-only, escalate writes to the crate root)
