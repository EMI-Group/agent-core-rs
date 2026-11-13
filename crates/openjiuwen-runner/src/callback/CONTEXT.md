# callback/ — event framework, filters, chains, and handler combinators

## Intent

The runner's publish/subscribe seam for lifecycle events: typed event names matching Python's `_framework:<event>` vocabulary, priority-ordered dispatch with explicit error isolation, and the guardrail binding that turns a `GuardrailPipeline` into event handlers.
Three layers build on that dispatch core: `EventFilter` values that gate a handler before it runs, `CallbackChain` for ordered groups of handlers with per-entry filters and error policy, and the combinators (`on`, `emit_*`, `transform_io`, `wrap`/`on_wrap`) that are the Rust equivalent of Python's callback decorators.
Ports Python `openjiuwen/core/runner/callback/` (`events.py`, `framework.py`, `errors.py`, `utils.py`, `filters.py`, `chain.py`, `decorator.py`).
Retries, rollback handlers, per-entry timeouts, the trigger variants, and the metrics/history extras remain deferred (see the crate-level `## Deferred Python Behavior`).

## API Surface

```rust
// event.rs
pub enum CallbackEvent {
    // agent: Started, InvokeInput, InvokeOutput, StreamInput, StreamOutput, P2pReceived, PubsubReceived
    //        (`Agent*`), plus `AgentSessionCreated`
    // workflow: Started, Finished, Error, Cancelled, NodeExecuted, NodeError, EdgeTraversed, LoopStarted,
    //           LoopFinished, InvokeInput, InvokeOutput, StreamInput, StreamOutput, ComponentBatchInput,
    //           ComponentBatchOutput, ComponentStreamInput, ComponentStreamOutput  (all `Workflow*`)
    // llm: CallStarted, CallError, ResponseReceived, ResponseStarted, InvokeInput, InvokeOutput,
    //      StreamInput, StreamOutput, StreamCompleted, Route, RouteAttempt, Input, Output  (all `Llm*`)
    // tool: CallStarted, CallFinished, CallError, ResultReceived, ParseStarted, ParseFinished,
    //       InvokeInput, InvokeOutput, StreamInput, StreamOutput, Auth  (all `Tool*`)
    // context: Updated, Offloaded, Retrieved, Cleared, CompressionState (`Context*`);
    //          session: `SessionCreated`, `AgentSessionCreated`; retrieval: `RetrievalStarted`
    // memory: Added, SearchStarted, SearchFinished, Updated, Deleted (`Memory*`)
    // task: Created, Running, Completed, Failed, Cancelled, Timeout (`Task*`)
    Custom(String),
}
impl CallbackEvent {
    pub fn name(&self) -> &str;                     // "agent_invoke_input"
    pub fn as_str(&self) -> &str;                   // "_framework:agent_invoke_input"  (Python value)
    pub fn from_name(name: &str) -> Option<Self>;   // accepts with or without the "_framework:" prefix
    pub fn domain(&self) -> EventDomain;
    pub fn is_llm_or_tool(&self) -> bool;
    pub const ALL: &'static [CallbackEvent];        // every non-custom variant, for iteration/tests
}
pub const DEFAULT_SCOPE: &str = "_framework";
pub const SCOPE_PREFIX: &str = "_framework:";
pub enum EventDomain { Agent, Team, Workflow, Llm, Tool, Context, Session, Retrieval, Memory, Task, Custom }

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
    pub fn with_input(self, input: Value) -> Self;   pub fn with_output(self, output: Value) -> Self;
    pub fn with_error(self, error: impl Into<String>) -> Self;   pub fn with_index(self, index: i64) -> Self;
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
pub struct CallbackInfo { pub id, pub event, pub name, pub priority, pub namespace, pub once, pub enabled,
                          pub tags: BTreeSet<String>, pub sequence: u64 }
pub struct CallbackRegistration { pub event, pub handler: Arc<dyn CallbackHandler>, pub priority, pub namespace,
                                  pub tags, pub once }
impl CallbackRegistration { pub fn new(event, handler) -> Self;   // priority 0, namespace "default"
    pub fn with_priority(..); pub fn with_namespace(..); pub fn with_tag(..); pub fn once(..) }
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
    pub fn enabled(&self, id: CallbackId) -> Option<bool>;
    pub fn has_subscribers(&self, event: CallbackEvent) -> bool;
    pub fn handlers(&self, event: CallbackEvent) -> Vec<CallbackInfo>;
    pub fn events(&self) -> Vec<CallbackEvent>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub async fn trigger(&self, event: CallbackEvent, ctx: &CallbackContext) -> Result<CallbackReport>;
}

// filters.rs — vocabulary and composition
pub enum FilterDecision { Allow, Skip { reason: String }, Reject(Error) }
pub struct FilterInput<'a> { pub handler: &'a str, pub namespace: &'a str, pub ctx: &'a mut CallbackContext }
pub trait EventFilter: Send + Sync {
    fn name(&self) -> &str { "filter" }
    fn allows(&self, input: FilterInput<'_>) -> Result<FilterDecision>;
    fn feedback(&self) -> Option<&dyn FilterFeedback> { None }
}
pub trait FilterFeedback: Send + Sync { fn record_success(..); fn record_failure(..); }
pub struct ClosureFilter<F>(..);                 // ClosureFilter::new("name", predicate)
pub enum FilterRunOutcome { Allow, Skip { filter: String, reason: String }, Reject(Error) }
pub fn run_filters(filters: &[Arc<dyn EventFilter>], handler: &str, namespace: &str,
                   ctx: &mut CallbackContext) -> Result<FilterRunOutcome>;
pub fn record_success(filters: &[Arc<dyn EventFilter>], ctx: &CallbackContext, handler: &str);
pub fn record_failure(filters: &[Arc<dyn EventFilter>], ctx: &CallbackContext, handler: &str);
pub struct Clock;        // Clock::system() (Instant::now) / Clock::from_fn(now)
pub struct ManualClock;  // new/at/advance/set/now/clock() -> Clock
pub struct FilterChain;  // new/with/with_arc/push/filters/len/is_empty ("filter_chain")
pub struct OrFilter;     // new(left, right)   ("or")
pub struct NotFilter;    // new(inner)         ("not")
pub struct LenientFilter;// new(inner)         ("lenient")
pub fn and(left, right) -> FilterChain;  pub fn or(left, right) -> OrFilter;
pub fn not(inner) -> NotFilter;          pub fn lenient(inner) -> LenientFilter;
pub trait EventFilterExt { fn and(..) -> FilterChain; fn or(..) -> OrFilter; fn negate(..) -> NotFilter;
                           fn lenient(..) -> LenientFilter; fn into_filter_arc(..) -> Arc<dyn EventFilter>; }
pub use builtin::{AuthFilter, CircuitBreakerFilter, ConditionalFilter, ContextModifier, KindFilter,
                  LoggingFilter, NameFilter, ParamModifyFilter, PredicateFilter, RateLimitFilter,
                  SessionFilter, ValidationFilter};

// filters/builtin.rs — the concrete filters
NameFilter / KindFilter / SessionFilter          // name, name prefix, namespace; event and/or domain; session ids
RateLimitFilter / CircuitBreakerFilter           // sliding call budget; failure threshold plus cooldown
ValidationFilter / LoggingFilter / AuthFilter    // Error::Validation; log-and-allow; ctx.metadata auth
ParamModifyFilter / ConditionalFilter            // in-place rewrite (Python's MODIFY); conditional inner filter
PredicateFilter                                  // allows iff the predicate accepts the context
type ContextModifier = Arc<dyn Fn(&mut CallbackContext) -> Result<()> + Send + Sync>;

// chain.rs
pub struct ChainEntryId(u64);                    // Copy, Hash, Display ("chain-entry-N")
pub struct ChainEntry { .. }                     // name, namespace, priority, enabled, fatal, filters, handler
impl ChainEntry { pub fn new(name, handler) -> Self; pub fn from_handler(handler) -> Self;
    pub fn with_priority(..); pub fn in_namespace(..); pub fn when(filter); pub fn with_filter(arc);
    pub fn fatal(..); pub fn disabled(..); pub fn name(..); pub fn handler(..); pub fn filters(..);
    pub fn is_fatal(..) }
pub struct ChainEntryInfo { pub id, pub name, pub namespace, pub priority, pub enabled, pub fatal,
                            pub handler: String, pub filters: Vec<String> }
pub enum EntryOutcome { Ran, Stopped, Skipped { filter: String, reason: String }, Failed }
pub struct EntryRecord { pub name: String, pub namespace: String, pub outcome: EntryOutcome }
pub struct ChainReport { pub report: CallbackReport, pub entries: Vec<EntryRecord> }
impl ChainReport { pub fn skipped(&self) -> Vec<&EntryRecord>; pub fn ran(&self) -> Vec<&EntryRecord>; }
pub struct CallbackChain { .. }
impl CallbackChain {
    pub fn new(name) -> Self;  pub fn name(&self) -> &str;
    pub fn add(&self, entry: ChainEntry) -> ChainEntryId;
    pub fn add_handler(&self, handler: Arc<dyn CallbackHandler>) -> ChainEntryId;
    pub fn add_named(&self, name, handler: Arc<dyn CallbackHandler>) -> ChainEntryId;
    pub fn prepend(&self, entry: ChainEntry) -> ChainEntryId;
    pub fn remove(&self, name: &str) -> usize;  pub fn remove_id(&self, id: ChainEntryId) -> bool;
    pub fn clear(&self);  pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn names(&self) -> Vec<String>;  pub fn entries(&self) -> Vec<ChainEntryInfo>;
    pub fn set_enabled(&self, name: &str, enabled: bool) -> usize;
    pub fn set_enabled_id(&self, id: ChainEntryId, enabled: bool) -> bool;
    pub async fn trigger(&self, event, ctx: &CallbackContext) -> Result<CallbackReport>;
    pub async fn trigger_mut(&self, event, ctx: &mut CallbackContext) -> Result<CallbackReport>;
    pub async fn trigger_payload<T: Serialize>(&self, event, payload: &T) -> Result<CallbackReport>;
    pub async fn trigger_detailed(&self, event, ctx: &mut CallbackContext) -> Result<ChainReport>;
}
impl CallbackHandler for CallbackChain;          // Continue / Stop / Abort(error)
impl CallbackHandler for Arc<CallbackChain>;

// decorator.rs — typed payload helpers, the builder, and the combinator re-exports
pub fn input_as<T>(ctx) -> Result<Option<T>>;    pub fn output_as<T>(ctx) -> Result<Option<T>>;
pub fn require_input<T>(ctx) -> Result<T>;       pub fn require_output<T>(ctx) -> Result<T>;
pub fn set_input<T: Serialize>(ctx, &T) -> Result<()>;  pub fn set_output<T: Serialize>(ctx, &T) -> Result<()>;
pub fn on(event: CallbackEvent) -> HandlerBuilder;
impl HandlerBuilder { named / with_priority / in_namespace / with_tag / once / fatal /
                      when(filter) / when_arc(arc) / sync(f) / async_fn(f) / handler(h) -> BuiltHandler }
impl BuiltHandler { name/event/priority/namespace/is_fatal/filters/inner_handler/handler() -> Arc<dyn CallbackHandler>;
                    registration() -> CallbackRegistration; register(&framework) -> CallbackId;
                    chain_entry() -> ChainEntry; add_to(&chain) -> ChainEntryId }

// decorator/emit.rs
pub struct EmitHandler { .. }                    // named(..)
pub fn emit_before(framework: Arc<CallbackFramework>, event) -> EmitHandler;
pub fn emit_after (framework: Arc<CallbackFramework>, event) -> EmitHandler;
pub fn emit_around(framework: Arc<CallbackFramework>, before, after) -> EmitHandler;

// decorator/transform.rs
pub type PayloadTransform = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;
pub struct TransformHandler { .. }               // new(name) / with_input(..) / with_output(..)
impl TransformHandler { rewrite(&mut ctx) -> Result<()>; last_input(); last_output();
                        async apply(input, call) -> Result<Value> }
pub fn transform_io(name) -> TransformHandler;

// decorator/wrap.rs
pub type PayloadCall = Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value>> + Send + Sync>;
pub fn payload_call(f) -> PayloadCall;
pub struct NextCall;                             // call(input) -> Result<Value>; called() -> bool
pub trait WrapHandler { fn name(&self) -> &str; async fn handle(&self, ctx, NextCall) -> Result<Value>; }
pub struct ClosureWrapHandler<F>(..);  pub fn wrap_handler(name, f) -> Arc<dyn WrapHandler>;
pub struct CallReport { pub value: Value, pub inner_ran: bool }
pub struct WrappedCall { .. }                    // new / from_registry(registry, event, inner)
impl WrappedCall { with_handler(layer) / with_layer(layer) / event() / layers() / call(input) /
                   call_with_report(input) -> Result<CallReport> }
pub fn wrap(event, inner: PayloadCall) -> WrappedCall;
pub struct WrapId(u64);  pub struct WrapRegistration { .. }   // with_priority / register(&registry)
pub fn on_wrap(event, handler: Arc<dyn WrapHandler>) -> WrapRegistration;
pub struct WrapRegistry { .. }                   // new / register / unregister / unregister_event /
                                                 // set_enabled / handlers(&event) / events / len / is_empty

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
- A guardrail handler builds its `GuardrailContext` from `ctx.guardrail` when present; otherwise it derives one from `ctx.input` (tool/LLM input) or `ctx.output` (tool/LLM output) as raw text and sets the stage.
- A blocked guardrail outcome is `Ok(Abort(error))` where `error` is `outcome.enforce()`'s `Error::Security`; allowed outcomes are `Ok(Continue)` and non-blocking findings are logged and dropped.

## Filter semantics

- Python's `FilterAction::{CONTINUE, SKIP, STOP, MODIFY}` maps to `FilterDecision::{Allow, Skip{reason}, Reject(Error)}`; MODIFY needs no variant because `FilterInput::ctx` is a mutable working context.
- A rewrite a filter makes to `ctx` is visible to later filters and to the guarded handler, and is written back to the caller by a chain's `trigger_mut`.
- `FilterInput { handler, namespace, ctx }` identifies the guarded handler by name and namespace so a filter can route on either.
- `run_filters` stops at the first `Skip` and returns `FilterRunOutcome::Skip { filter, reason }`; a `Reject` becomes `FilterRunOutcome::Reject(error)`; an empty list is `Allow`.
- A filter returning `Err` fails closed: `run_filters` propagates it and the dispatch site turns it into an abort, so a filter that cannot decide never silently allows its handler (Python skipped instead; `lenient(...)` restores the skip).
- `FilterFeedback` lets a stateful filter observe the guarded handler's outcome; the dispatch sites call the free `record_success`/`record_failure` helpers over `&[Arc<dyn EventFilter>]`, which forward to each filter whose `feedback()` is `Some`.
- `CircuitBreakerFilter` is the only built-in feedback sink: it counts failures per event+handler, opens at the threshold, skips while open, and closes on the first call after the cooldown elapses.
- `Clock`/`ManualClock` make the time-dependent filters deterministic: `RateLimitFilter` prunes its sliding window and `CircuitBreakerFilter` evaluates its cooldown against the injected clock, so tests advance the clock instead of sleeping.
- `EventFilter` is implemented for `Arc<dyn EventFilter>` (so shared filters compose without unwrapping) and for any `Fn(FilterInput<'_>) -> Result<FilterDecision>`; `ClosureFilter::new(name, predicate)` gives such a closure a real name in reports.
- `EventFilterExt` gives the fluent form of the same combinators: `and`, `or`, `negate`, `lenient`, `into_filter_arc`.
- `KindFilter` matches event and/or domain sets, `SessionFilter` matches session ids (`allow_absent` waives the requirement), and `LoggingFilter` always allows while logging no payload values.
- `NameFilter` matches an exact name, a name prefix, and/or a namespace, and a dimension it was never configured with does not constrain the match.
- A filter denial is soft by default: `AuthFilter` reads `ctx.metadata` (`user_role`, `auth_token`, or a predicate) and skips unless built with `rejecting()`, which returns `Error::Security`.
- `ConditionalFilter` applies its inner filter only when its condition holds and can invert that into an allow with `allow_on_false`; `PredicateFilter` is the allow-iff-predicate shorthand.
- `FilterChain` is logical AND (`with`/`with_arc`/`push`), `OrFilter` is logical OR where a rejection from either side wins over a skip, `NotFilter` inverts allow/skip but never a rejection, and `LenientFilter` converts an error into a skip.

## Chain semantics

- An entry carries a name, a namespace, a priority, an optional filter list, an `enabled` toggle, and a `fatal` flag; entries run by priority descending with insertion order as the tiebreak, and `prepend` inserts ahead of every equal-or-lower priority entry.
- Filters run against the working context before the handler; a `Skip` records `EntryOutcome::Skipped { filter, reason }` without running the handler, and a `Reject` aborts the chain with that error.
- A `Continue` verdict records `EntryOutcome::Ran` and moves on; `Stop` records `EntryOutcome::Stopped` and ends the chain successfully (`report.aborted = true`); `Abort(e)` ends the chain and becomes the `trigger` error, annotated with the chain and entry names while preserving the error kind.
- A plain handler `Err` is isolated into `report.failures` and recorded as `EntryOutcome::Failed`, so dispatch continues; a `fatal` entry turns the same failure into the `trigger` error and `report.aborted = true`.
- A disabled entry is not considered at all and produces no `EntryRecord`.
- `trigger` dispatches on a clone of the context, so filter rewrites stay inside the call; `trigger_mut` dispatches on the caller's context, so filter rewrites survive; `trigger_payload` serializes a value into a fresh context's `input`.
- `trigger_detailed` returns the `ChainReport` (the framework-shaped `CallbackReport` plus per-entry `EntryRecord`s); `trigger` and `trigger_mut` return the plain `CallbackReport`; `ChainReport::ran`/`skipped` filter the records.
- `CallbackChain` implements `CallbackHandler` (and `Arc<CallbackChain>` does too), reporting `Continue` for a completed run, `Stop` when its dispatch was stopped, and `Abort(error)` when it aborted or a fatal entry failed; that is what lets a chain nest inside another chain or register on a `CallbackFramework` and be recorded as one entry outcome.
- `add`/`add_named`/`add_handler` return a `ChainEntryId`; `prepend` inserts at the front; `remove(name)` drops every entry of that name; `remove_id` drops one; `set_enabled`/`set_enabled_id` toggle without removing; `names`/`entries` report dispatch order.
- Duplicate names are legal, so `ChainEntryId` (not the name) is the precise removal and enable/disable handle.

## Handler combinators (the decorators)

- Rust has no decorators, so each combinator is a constructor that builds a `CallbackHandler` usable with `CallbackFramework::register*` and as a `ChainEntry`.
- `on(event)` starts a `HandlerBuilder` (priority 0, namespace `"default"`); `named`/`with_priority`/`in_namespace`/`with_tag`/`once`/`fatal`/`when`/`when_arc` configure it, and `sync`/`async_fn`/`handler` build a `BuiltHandler`.
- The built handler's name defaults to the adapter's (`"sync_handler"`, `"async_handler"`) or to the supplied handler's own name; `handler()` returns the composed wrapper that normalizes the name, applies the filters, and enforces the fatal policy.
- A composed handler turns a filter skip into `Continue` (the guarded handler does not run), a filter rejection or filter error into `Abort`, and a handler error into `Err` — or into `Abort` when the builder is `fatal`.
- `registration()` embeds the composed wrapper (so filters and policy apply inside a framework dispatch), while `chain_entry()` puts the filters on the entry and keeps the raw handler; `register(&framework)` and `add_to(&chain)` are the shortcuts.
- `emit_before`/`emit_after`/`emit_around` build an `EmitHandler` that fires an auxiliary event on the given `Arc<CallbackFramework>` with a context derived from the source (`aux_context` copies session, component, input, index, error, metadata, and guardrail content, plus the source's output for the after/around form, and stamps `metadata["source_event"]`).
- An auxiliary abort becomes `Abort` on the source event, an auxiliary stop becomes `Stop`, and a clean auxiliary dispatch continues; `emit_around` fires both events back to back inside one handler, so use `wrap` when the "after" event must observe work between them.
- `transform_io(name)` builds a `TransformHandler` with `.with_input`/`.with_output` rewrites: as a `CallbackHandler` it rewrites its own clone of the context and publishes the result through `last_input`/`last_output`; as an `EventFilter` it rewrites the working context in place, so later entries see the transformed payloads; `apply` runs a call between the two rewrites.
- A transform error is always propagated, never swallowed.
- `wrap(event, call)` builds a `WrappedCall`; `with_handler`/`with_layer` append layers outside-in, so the first layer added is the outermost and runs first.
- A `WrapHandler` receives `NextCall`, may rewrite the argument passed to `call`, may rewrite the returned value, and may never call it to short-circuit; `NextCall::called()` makes that observable, and `call_with_report` reports it as `CallReport::inner_ran`.
- `WrapRegistry` is a separate per-event layer store (Python's `__wrap__:<event>` channel) because the framework exposes only registration metadata, never handler values; `handlers(&event)` returns the enabled layers for a call, ordered by priority descending then insertion.
- `on_wrap(event, handler)` returns a `WrapRegistration` (`.with_priority`, `.register(&registry)`), and `WrappedCall::from_registry` resolves layers on every call, so a layer registered later affects the next call.
- Payloads stay `serde_json::Value`; `input_as`/`output_as` deserialize, `require_input`/`require_output` fail on a missing payload, and `set_input`/`set_output` serialize into the context.
- Unknown events keep working through `CallbackEvent::Custom`, whose payloads ride the same `input`/`output` fields.

## Design Decisions

- `ScanStage::UserInput` has no preset callback event (`"user_input"` is unscoped), so `event_for_stage` represents it as `CallbackEvent::Custom("user_input")` and `scan_stage_for` maps that back.
- `bind_guardrails` therefore subscribes only to the four event-backed stages (`GUARDRAIL_STAGES`); user-input scanning is an explicit `Runner::scan`.
- `CallbackId` is the sole removal handle; duplicate registration stays legal so Python-style double-subscription dispatches twice.
- `CallbackContext`'s manual `Debug` prints payload shapes and metadata keys, never values.
- Filters fail closed on `Err` (a filter that cannot decide must not allow its handler), with `lenient(...)` for Python's skip-on-error behaviour.
- `FilterDecision::Reject` carries an `Error` instead of Python's `FilterResult` with an action enum, so a rejection travels as the dispatch error without a translation table.
- MODIFY is in-place mutation of `FilterInput::ctx` rather than a fourth variant, so a payload rewrite has exactly one expression whatever the dispatch site.
- Chain `Abort`/`Stop` are verdicts rather than exceptions with `is_error=False`, so "stop cleanly" and "fail" stay distinguishable in the return type.
- Wrap layers live in an explicit `WrapRegistry` instead of Python's `"__wrap__:<event>"` key prefix, because the framework does not expose handler values to a caller.
- Time-dependent filters take an injectable `Clock` instead of reading `time.monotonic` or sleeping, which keeps dispatch synchronous, single-threaded per call, and testable.
- Per-entry outcomes are recorded in `ChainReport` instead of Python's chain logging, so a caller can assert on what ran, stopped, was skipped, or failed.

## Constraints

- `CallbackEvent::as_str()` values are a wire contract with `openjiuwen-security` (`ScanStage::event()`) and with harness code that subscribes by string; every Python preset name must be representable, and `from_name` must round-trip all of `ALL`.
- Errors are `openjiuwen_common::error::{Error, Result}`; no parallel error type. `CallbackVerdict` carries the abort error instead of a bespoke exception.
- The framework, filters, chains, and combinators own no globals and spawn no tasks; a framework is `Arc`-shared from a `Runner`.
- `tracing` target `openjiuwen_runner::callback` via `LOG_TARGET`; never log `CallbackContext` payload contents at info level (they may contain prompts).
- `event.rs` holds the vocabulary and mapping, `framework.rs` the registry and dispatch; `framework.rs` is already at the size limit (~990 lines), so dispatch extensions live in the filters, chain, and decorator files.
- No `unsafe`, and no panics on user input: a bad builder configuration, a rejected filter, and a missing payload all surface as `Error` values.
- Every dispatch path is deterministic: no sleeps, no timers, no background pruning — the clock is injected and advanced by the caller.
- Files stay around ≤1000 lines; filters (`filters.rs`, `filters/builtin.rs`), the chain, and the decorator family are separate modules with separate suites.

## Test Strategy

- Unit tests sit beside each module: payload helpers and name normalization (`decorator.rs`), clock-driven rate limit and circuit breaker plus filter composition (`filters.rs`, `filters/builtin.rs`), an empty chain and ordering (`chain.rs`), emit payloads and abort propagation (`decorator/emit.rs`), transform rewrites (`decorator/transform.rs`), wrap ordering and short-circuit (`decorator/wrap.rs`).
- `tests/callback_filters.rs` covers name/kind/session filters, the rate-limit window, the circuit breaker's threshold and cooldown recovery (including feedback-driven operation), validation/logging/auth, param-modify visibility, conditional delegation, composition (skip stops the chain, OR keeps rejections), `EventFilterExt`/`Arc` storage, `run_filters` naming the first skipping filter, closure naming, and filters gating chain entries end to end.
- `tests/callback_chains.rs` covers priority descending with insertion tiebreak, prepend, abort/stop/failure/fatal outcomes, filters skipping entries, mutation visibility and `trigger_mut` write-back, nested chains, disable/enable, removal handles, framework registration, `trigger_payload`, and entry metadata.
- `tests/callback_decorators.rs` covers emit before/after/around payloads and abort/stop propagation, `transform_io` rewrites (direct and in-place), wrap outermost-first ordering, short-circuiting, error propagation, registry resolution per call, builder registration and framework interplay, and typed payload helpers on preset and custom events.
- `tests/callbacks.rs` and `tests/security_binding.rs` cover the framework itself: ordering, `once`, enable/disable, isolation, guardrail stage binding, and blocking.
- Everything is offline and deterministic: `ManualClock` for time, `ScriptedModel` where an agent is involved, no sleeps, no wall-clock ordering.

## Routing Table

- `./src/callback/event.rs` → `CallbackEvent`, `EventDomain`, name mapping, `DEFAULT_SCOPE`/`SCOPE_PREFIX`, `ALL`
- `./src/callback/context.rs` → `CallbackContext`, `CallbackVerdict`, `CallbackReport`, `CallbackFailure`
- `./src/callback/handler.rs` → `CallbackHandler`, `SyncHandler`, `AsyncHandler`
- `./src/callback/framework.rs` → `CallbackId`, `CallbackInfo`, `CallbackRegistration`, `CallbackFramework`
- `./src/callback/filters.rs` → filter vocabulary (`FilterDecision`, `FilterInput`, `EventFilter`, `FilterFeedback`, `ClosureFilter`, `FilterRunOutcome`, `run_filters`, `record_success`/`record_failure`, `Clock`, `ManualClock`) plus the composition combinators (`FilterChain`, `OrFilter`, `NotFilter`, `LenientFilter`, `and`/`or`/`not`/`lenient`, `EventFilterExt`); re-exports the built-ins
- `./src/callback/filters/builtin.rs` → `NameFilter`, `KindFilter`, `SessionFilter`, `RateLimitFilter`, `CircuitBreakerFilter`, `ValidationFilter`, `LoggingFilter`, `AuthFilter`, `ParamModifyFilter`, `ContextModifier`, `ConditionalFilter`, `PredicateFilter`
- `./src/callback/chain.rs` → `CallbackChain`, `ChainEntry`, `ChainEntryId`, `ChainEntryInfo`, `EntryOutcome`, `EntryRecord`, `ChainReport`
- `./src/callback/decorator.rs` → typed payload helpers, `on`/`HandlerBuilder`/`BuiltHandler`, re-exports of the wrap/emit/transform combinators
- `./src/callback/decorator/wrap.rs` → `PayloadCall`, `NextCall`, `WrapHandler`, `ClosureWrapHandler`, `wrap_handler`, `CallReport`, `WrappedCall`, `wrap`, `WrapId`, `WrapRegistration`, `on_wrap`, `WrapRegistry`
- `./src/callback/decorator/emit.rs` → `EmitHandler`, `emit_before`, `emit_after`, `emit_around`
- `./src/callback/decorator/transform.rs` → `TransformHandler`, `PayloadTransform`, `transform_io`
- `./src/callback/security.rs` → stage mapping, `GuardrailHandler`, `bind_guardrails`
- `./src/callback/mod.rs` → re-exports and the `LOG_TARGET` constant
- `../tests/callbacks.rs`, `../tests/security_binding.rs`, `../tests/callback_chains.rs`, `../tests/callback_filters.rs`, `../tests/callback_decorators.rs` → behavior suites (siblings — read-only, escalate writes to the crate root)

## Notes for Agents

- A filter rejection is a value (`FilterDecision::Reject`); the dispatch site converts it into `Abort` and the framework or chain surfaces it as the trigger error.
- `run_filters` stops at the first skip, so filter order is significant: put cheap identity filters first and side-effecting modifiers before the handlers that must observe them.
- Dispatch order is priority descending and ties keep insertion order; `CallbackChain` and `CallbackFramework` sort the same way, so the same handlers fire alike in both.
- `CallbackChain::trigger` works on a clone, so filter mutations do not escape; use `trigger_mut` when the caller must observe them.
- A chain handler receives `&CallbackContext`, so it can only report a verdict, never mutate; rewrite payloads with a filter instead.
- `TransformHandler` rewrites a clone inside `CallbackHandler::handle`; read `last_input`/`last_output`, or use it as a filter, if the rewritten values are needed downstream.
- A `WrapHandler` must return `Ok(...)` for a successful call and must not call `next` twice; a layer that only observes can still forward the value unchanged.
- `emit_*` needs an `Arc<CallbackFramework>`, so build the framework first and pass a clone into every combinator.
- A nested chain records its inner run as a single entry outcome in the outer `ChainReport`; reach for `trigger_detailed` on the inner chain when its per-entry detail matters.
- `BuiltHandler::handler()` is the composed wrapper (filters and fatal policy included) while `chain_entry()` carries the filters on the entry and the raw handler inside, so do not apply both to the same dispatch.
- Use `ManualClock::clock()` in tests instead of sleeping; the rate limiter and circuit breaker never read the wall clock themselves.
