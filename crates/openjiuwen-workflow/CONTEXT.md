# openjiuwen-workflow

## Intent

Workflow builder, components, and `${node.field}` data flow.
Ports the Python package `openjiuwen.core.workflow` (v0.1.20: `workflow.py`, `_workflow.py`, `workflow_config.py`, `base.py`, `components/`) onto `openjiuwen-graph`'s superstep executor, `openjiuwen-session`'s state/stream/checkpoint/trace seams, and `openjiuwen-foundation`'s `Model`/`PromptTemplate`/`Tool`.
A workflow is a directed graph of components with one start and one end; the builder records component configs and edges, and compiling maps them onto `GraphBuilder` nodes/edges/routers.
Data flow is textual: a component's `inputs_schema` is a JSON template whose pure `"${node.field}"` strings resolve against workflow state before the component runs; references never fail (a missing path resolves to `null`).
Components declare capabilities (`invoke`, `stream`, `collect`, `transform`) that determine legal edges and how the engine drives them.

## API Surface

### Builder (`src/workflow.rs`)

```rust
pub struct Workflow { /* card + config + components + edges + recorded problems */ }
impl Workflow {
    pub fn new(card: WorkflowCard) -> Self;
    pub fn with_config(config: WorkflowConfig) -> Self;
    pub fn card(&self) -> &WorkflowCard;
    pub fn config(&self) -> &WorkflowConfig;
    pub fn set_start_comp(&mut self, id: &str, comp: impl Component + 'static, options: ComponentOptions) -> &mut Self;
    pub fn add_workflow_comp(&mut self, id: &str, comp: impl Component + 'static, options: ComponentOptions) -> &mut Self;
    pub fn set_end_comp(&mut self, id: &str, comp: impl Component + 'static, options: ComponentOptions) -> &mut Self;
    pub fn add_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn add_connections(&mut self, from: &[&str], to: &str) -> &mut Self;      // barrier fan-in
    pub fn add_stream_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn add_conditional_connection(&mut self, from: &str, to: &str, router: Arc<dyn Router>) -> &mut Self;
    pub fn start_comp_id(&self) -> Option<&str>;
    pub fn end_comp_id(&self) -> Option<&str>;
    pub fn component_ids(&self) -> Vec<&str>;
    pub fn spec(&self) -> WorkflowSpec;                                    // serializable, no runnables
    pub fn build(&self) -> Result<CompiledWorkflow>;                       // validate + compile
    pub fn to_mermaid(&self) -> Result<String>;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub async fn invoke_with(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn stream_with(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
}
```

- Builder methods never return `Err`; they record problems and `build`/`invoke`/`stream` report them (same policy as `openjiuwen-graph`).
- Setter options: `ComponentOptions` (schema mapping, ability override, `wait_for_all`, `max_retries`, `timeout_ms`, `name`) with chainable methods; `ComponentOptions::streaming()` marks a streaming end (`response_mode="streaming"`).
- `invoke`/`stream` compile on the fly; a `Workflow` is immutable during a run, so one instance runs concurrently.

### Compiled engine (`src/engine/`)

```rust
#[derive(Clone)] pub struct CompiledWorkflow { /* Arc inner: graph, card, spec, end id, streaming flag */ }
impl CompiledWorkflow {
    pub fn card(&self) -> &WorkflowCard;
    pub fn spec(&self) -> &WorkflowSpec;
    pub fn end_comp_id(&self) -> &str;
    pub fn is_streaming(&self) -> bool;
    pub fn to_mermaid(&self) -> String;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
}

#[derive(Clone)] pub struct WorkflowRunOptions {
    pub cancellation: CancellationToken,        // default: never cancelled
    pub timeout: Option<Duration>,              // None → SessionConfig::execute_timeout_ms → none
    pub interactive: Option<InteractiveInput>,  // seeded HITL answers
    pub resume: bool,                           // continue from a stored graph snapshot
    pub stream_modes: Option<Vec<StreamMode>>,  // which frame modes `stream` yields (None → all)
    pub max_supersteps: Option<usize>,          // None → graph default
}
```

`Engine` compiles a component graph as follows: every component becomes a `graph::Node` adapter; `add_connection` → `GraphBuilder::edge`, `add_connections` → `barrier_edge`, `add_stream_connection` → `edge` plus stream-ability wiring, conditional connections → `GraphBuilder::conditional_edges_to` with one combined router per source, `start` → `entry`, `end` → `exit`.
`NodeOptions { timeout_ms, max_retries }` carries the per-component limits.

### Outputs and chunks (`src/output.rs`)

```rust
pub struct WorkflowOutput { pub result: Value, pub state: WorkflowExecutionState }
pub enum WorkflowExecutionState { Completed, InputRequired, Error }   // wire "COMPLETED" | "INPUT_REQUIRED" | "ERROR"
pub type WorkflowChunk = StreamItem;                                    // mode + "type" + index + payload
pub struct OutputFrame { /* type, index, payload */ }                   // Python OutputSchema projection
pub const END_NODE_STREAM: &str = "end node stream";
pub const WORKFLOW_FINAL: &str = "workflow_final";
```

- `invoke`: `result` is the end component's output value; when the end is streaming (or the run was interrupted) `result` is the JSON array of collected chunk wires (`{"type","index","payload"}`), matching Python's chunk-list result.
- `stream`: yields frames in production order, then one `workflow_final` output frame carrying the end outputs (omitted when they are null), then ends; a failed run yields exactly one `Err`.
- Frames a component emits with `ctx.write_stream` / `ctx.write_custom_stream` are forwarded verbatim (mode `output` / `custom`); end-node streaming frames use type `end node stream`; HITL frames use `__interaction__`.

### Components (`src/component.rs`, `src/components/`)

```rust
#[async_trait]
pub trait Component: Send + Sync {
    fn card(&self) -> ComponentCard;                      // identity + capability metadata
    fn abilities(&self) -> Vec<ComponentAbility>;         // default: [Invoke]
    fn component_type(&self) -> &str;                     // "start", "end", "llm", …
    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value>;   // default: Error::Unsupported
    fn stream(self: Arc<Self>, inputs: Value, ctx: Arc<ComponentContext>) -> BoxStream<'static, Result<Value>>;  // default: one Unsupported error
    async fn collect(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value>;  // default: Error::Unsupported
    fn transform(self: Arc<Self>, inputs: Value, ctx: Arc<ComponentContext>) -> BoxStream<'static, Result<Value>>;
    fn route(&self, source: &Value) -> Option<Result<Vec<String>>>;                    // branch components only
    fn trace_metadata(&self) -> Option<Value>;
}
impl<T: Component + ?Sized> Component for Arc<T> { /* forwarding */ }

pub struct ComponentCard { pub id, pub name, pub description, pub component_type, pub abilities: Vec<ComponentAbility>, pub input_params: Value, pub output_params: Value }
pub struct ComponentAbility { Invoke, Stream, Collect, Transform }        // wire "invoke" | "stream" | "collect" | "transform"
pub struct ComponentRegistry;   // name → factory(Value) -> Result<Arc<dyn Component>>; `builtin()` registers start/end/branch/loop
```

Built-ins (all offline-testable):

| Component | Module | Behavior |
|---|---|---|
| `StartComponent` | `components/start.rs` | `invoke` returns its resolved inputs unchanged |
| `EndComponent` | `components/end.rs` | default `{"output": <non-null inputs>}`; with `EndConfig::response_template` renders `{{path}}` segments (batch: `{"response": …}`; stream/transform: one `end node stream` frame per segment element) |
| `LlmComponent` | `components/llm.rs` | `PromptTemplate` system/user templates → `Model::invoke`/`stream`; output keys from `output_config` (default `{"result": …}`) |
| `ToolComponent` | `components/tool.rs` | `Arc<dyn Tool>::invoke`; returns `{"error_code","error_message","data"}` |
| `BranchComponent` + `BranchRouter` | `components/branch.rs` | `route` evaluates ordered `Branch`es (condition DSL) to target ids; `invoke` returns `{}` |
| `LoopComponent` | `components/loop.rs` | bounded loop over an inner component list: number / array / condition, capped by `loop_number_max_limit` (1000), returns `{"loop_results": […]}` |
| `KnowledgeRetrievalComponent` | `components/retrieval.rs` | `SimpleKnowledgeBase::retrieve_with_config` → `{"query", "documents": [{text, score, metadata}]}` |
| `MemoryRetrievalComponent`, `MemoryWriteComponent` | `components/memory.rs` | operate over the local `MemoryRecall` / `MemoryRecord` trait seams (adapters land in `openjiuwen-memory`) |

### Component context (`src/context.rs`)

```rust
pub struct ComponentContext { /* session + component/workflow ids + interaction state + cancellation */ }
impl ComponentContext {
    pub fn new(session: Arc<dyn Session>, component_id: impl Into<String>, workflow_id: impl Into<String>) -> Self;
    // chainable for direct component tests and engine wiring:
    pub fn with_interactions(self, Arc<InteractionState>) -> Self;
    pub fn with_writer(self, Arc<FrameWriter>) -> Self;
    pub fn with_cancellation(self, CancellationToken) -> Self;
    pub fn with_metadata(self, BTreeMap<String, Value>) -> Self;
    pub fn with_executable_id(self, impl Into<String>) -> Self;
    pub fn session(&self) -> &Arc<dyn Session>;
    pub fn component_id(&self) -> &str;  pub fn workflow_id(&self) -> &str;  pub fn executable_id(&self) -> &str;
    pub async fn get_state(&self, key: &str) -> Result<Option<Value>>;        // session global lane, `${...}` aware
    pub async fn update_state(&self, patch: Value) -> Result<()>;
    pub async fn get_component_state(&self, key: &str) -> Result<Option<Value>>;
    pub async fn update_component_state(&self, patch: Value) -> Result<()>;
    pub async fn dump_state(&self) -> Result<Value>;
    pub async fn write_stream(&self, payload: Value) -> Result<()>;          // output frame, type "output"
    pub async fn write_custom_stream(&self, payload: Value) -> Result<()>;   // custom frame
    pub async fn trace(&self, data: Value) -> Result<()>;                    // trace frame
    pub async fn interact(&self, question: Value) -> Result<Value>;          // seeded answer, else halts the run
    pub fn cancellation(&self) -> &CancellationToken;
    pub fn check_cancelled(&self) -> Result<()>;
    pub fn seeded_answer(&self, id: &str) -> Option<Value>;
}
```

### Reference resolution (`src/reference.rs`)

```rust
pub fn is_ref_path(s: &str) -> bool;                       // "${…}", length > 3
pub fn extract_origin_key(s: &str) -> Option<&str>;        // "${a.b}" → "a.b"
pub fn split_path(path: &str) -> Vec<PathSegment>;         // a.b[0]['k'].c
pub fn resolve_path(source: &Value, path: &str) -> Option<Value>;
pub fn resolve_schema(schema: &Value, source: &Value) -> Value;   // deep template resolution
pub fn project(value: &Value, path: &str) -> Value;        // alias used by routers
```

Rules: a string that is exactly `${…}` resolves to the referenced value (any JSON type); other strings are literals. Nested paths support dotted keys, `[i]`, `[-1]`, `['k']`; a numeric segment after `.` indexes arrays but falls back to a key. A missing path or a `null` root resolves to `null`; resolution never errors. When an intermediate value is an array and the next segment is a non-numeric field name, the field is projected over the elements (needed for buffered stream composition — see Design Decisions).

### Conditions (`src/condition.rs`)

```rust
pub struct Condition { /* expression source + compiled form */ }
impl Condition {
    pub fn new(expression: impl Into<String>) -> Result<Self>;
    pub fn evaluate(&self, source: &Value) -> Result<bool>;
}
pub struct Branch { pub name: Option<String>, pub targets: Vec<String>, pub condition: Condition }
pub struct BranchRouter { /* ordered branches + optional default */ }
impl BranchRouter {
    pub fn new(); pub fn add_branch(&mut self, branch: Branch) -> &mut Self;
    pub fn default_target(&mut self, targets: Vec<String>) -> &mut Self;
    pub fn targets(&self) -> Vec<String>;
}
```

Grammar: `&&`/`||`/`!` (and `and`/`or`/`not`), comparisons `== != > >= < <=`, `in`/`not_in`, parentheses, numeric/string/bool/null literals, `${path}` references, and the helper functions `len`, `is_empty`, `is_not_empty`. No match in a `BranchRouter` is `Error::workflow` with code `101021` (`COMPONENT_BRANCH_EXECUTION_ERROR`).

### Cards and specs (`src/card.rs`)

```rust
pub struct WorkflowCard { pub id: String, pub name: String, pub description: String, pub version: String, pub input_params: Option<Value> }
pub struct CompIoConfig { pub inputs_schema: Option<Value>, pub outputs_schema: Option<Value> }
pub struct ExceptionConfig { pub handle_type: String /* "interrupt" */, pub default_outputs: Option<Value>, pub metadata: BTreeMap<String, Value> }
pub struct NodeSpec { pub io_configs: Option<CompIoConfig>, pub stream_io_configs: Option<CompIoConfig>, pub abilities: Vec<ComponentAbility>, pub max_retries: u32, pub timeout: f64, pub exception_config: Option<ExceptionConfig>, pub name: Option<String> }
pub struct WorkflowSpec { pub edges: BTreeMap<String, Vec<String>>, pub stream_edges: BTreeMap<String, Vec<String>>, pub comp_configs: BTreeMap<String, NodeSpec>, pub stream_source_groups: BTreeMap<String, Vec<Vec<String>>>, pub start_nodes: Vec<String> }
pub struct WorkflowConfig { pub card: WorkflowCard, pub spec: WorkflowSpec, pub workflow_max_nesting_depth: u32 /* default 5 */ }
```

Field names, defaults, and card keys match Python (`id`, `name`, `description`, `version`, `input_params`; `timeout` in seconds, negative = none). Every type is serde snake_case, tolerates unknown fields, and round-trips through JSON.

### Errors (`src/error.rs`)

`openjiuwen_common::error::{Error, Result}` only; `Error::Workflow` carries Python-style `"[<code>] <message>"` text with the code constants (`WORKFLOW_COMPONENT_ID_INVALID = 100010`, `…_ABILITY_INVALID = 100011`, `…_EDGE_INVALID = 100012`, `…_CONDITION_EDGE_INVALID = 100013`, `…_SCHEMA_INVALID = 100014`, `…_STREAM_EDGE_INVALID = 100015`, `…_COMPILE_ERROR = 100100`, `…_COMPONENT_EXECUTION_ERROR = 100054`, `COMPONENT_BRANCH_EXECUTION_ERROR = 101021`, `COMPONENT_LOOP_EXECUTION_ERROR = 101040`).

## Constraints

- Follows `docs/ARCHITECTURE.md`: async-first tokio, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, `tracing` only (no stdout), files ≤ ~1000 lines.
- The graph engine is consumed as-is (`openjiuwen-graph`); this crate never reimplements supersteps, channels, checkpoints, or Mermaid rendering.
- Only this crate's files and `Cargo.toml` change; `openjiuwen-session`, `openjiuwen-foundation`, `openjiuwen-retrieval` and `openjiuwen-graph` are used through their documented seams.
- Public streams are `Send + 'static`; cancellation flows through `CancellationToken`; locks are never held across `.await`.
- No panics on user input: every builder problem, schema problem, and component failure becomes an `Error`; component panics are not caught (same policy as the rest of the workspace).
- Tests are deterministic and offline: `ScriptedModel`, `HashEmbedding`, `InMemoryVectorStore`, no sleeps for ordering.

## Design Decisions

- **Compiled per call, no runnable state in the builder.** `Workflow` holds `Arc<dyn Component>` handles plus recorded config; `build()` validates once and produces a `CompiledWorkflow` that owns the graph. `stream` clones the compiled graph into the returned stream so the stream is `'static`.
- **Components receive a cloned `GraphState` and write their output under their own id.** This is the graph crate's node-delta model: the engine resolves schemas against state, calls the component, then writes `state[component_id] = output`. References therefore compose without a separate data-bus.
- **Streaming is buffered, not interleaved.** A `Stream`/`Transform` producer accumulates its frames into an array stored under its id instead of feeding live generators downstream; a `Collect`/`Transform` consumer resolves `stream_inputs_schema` (projecting arrays) and runs after its producers. Frames the *end* node emits are still `end node stream` frames in template order, so the observable text and order match Python for finite streams; only live interleaving of two concurrently streaming producers is lost.
- **Array projection is an intentional extension.** `resolve_path` projects a field over an array of objects (`${producer.output}` → `[f1, f2, …]`) so buffered stream composition works; Python would return `null` there. Documented in `docs/PARITY.md`-style notes below.
- **Interrupts are data, not exceptions.** `ComponentContext::interact` returns a seeded answer when one exists; otherwise it records an interaction frame and returns an interrupt error that the node adapter converts into `NodeOutcome::Halt`. The run finishes as `INPUT_REQUIRED`, the graph store keeps the pre-superstep snapshot, and `Workflow::resume` re-runs that superstep with seeded answers (`InteractiveInput`).
- **`wait_for_all` is accepted but cannot change scheduler semantics.** The graph engine's channels are CNF barriers, so every fan-in waits for all predecessors; Python's first-arrival (`wait_for_all=False`) mode has no equivalent and is documented as a deviation.
- **`ComponentAbility` serializes as `"invoke" | "stream" | "collect" | "transform"`.** Python's tuple-valued enum dumps as tuples, which is not a usable JSON contract; the string form is the Rust wire format.
- **Session component state is namespaced by component id via the `Session` trait.** Python's `NodeSession` nesting is approximated with `get_component_state`/`update_component_state`; `${node.field}` resolution uses graph state, not session lanes.
- **`WorkflowInput`/`session` handling is explicit.** `invoke` merges the workflow inputs into the session global lane (Python `commit_user_inputs`) and seeds graph state with them, so root refs (`${query}`) and node refs (`${start.query}`) both resolve.
- **Sub-workflows are deferred.** `SubWorkflowComponent` needs nested-session semantics that `openjiuwen-session` does not expose; the builder keeps a single-level graph.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, crate-root re-exports
- `./src/ability.rs` → `ComponentAbility`
- `./src/card.rs` → `WorkflowCard`, `WorkflowConfig`, `WorkflowSpec`, `NodeSpec`, `CompIoConfig`, `ExceptionConfig`
- `./src/error.rs` → Python error codes and message builders
- `./src/reference.rs` → `${node.field}` parsing, nested paths, schema resolution
- `./src/condition.rs` → condition expression parser/evaluator, `Branch`, `BranchRouter`
- `./src/component.rs` → `Component` trait, `ComponentCard`, ability defaults, boxed helpers
- `./src/context.rs` → `ComponentContext` session facade (state, streams, HITL, cancellation)
- `./src/interrupt.rs` → interaction frames, seeded answers, interrupt error
- `./src/registry.rs` → `ComponentRegistry` and builtin factories
- `./src/output.rs` → `WorkflowOutput`, `WorkflowExecutionState`, chunk/frame helpers
- `./src/components/` → built-in components: start, end, llm, tool, branch, loop, retrieval, memory (`components/CONTEXT.md`)
- `./src/engine/` → `CompiledWorkflow`, compile/validation, run/stream, node adapter (`engine/CONTEXT.md`)
- `./src/workflow.rs` → `Workflow` builder and its invoke/stream/resume entry points
- `./tests/` → behavior tests (see Test Strategy)

## Test Strategy

Unit tests beside modules: `reference` (path grammar, projection, missing paths), `condition` (operator precedence, helpers, errors), `card` (JSON round-trip, defaults), `components::end` (template segmentation/rendering), `components::loop` (bound enforcement).

Integration tests in `./tests/` (all offline, using `openjiuwen_test_support::ScriptedModel` and `openjiuwen_session::WorkflowSession`):

- `common/mod.rs` — shared helpers: session factory, `EchoComponent`/`CounterComponent`/streaming fakes, tool fixture.
- `builder_validation.rs` — missing/duplicate start and end, bad ids, unknown edge endpoints, stream-edge capability mismatch, schema key overlap, `ComponentOptions` defaults.
- `linear.rs` — Start → LLM (`ScriptedModel`) → End, `invoke` result and session global state.
- `references.rs` — `${start.a}`, nested `a.b[0]`, whole-node refs, missing path → null, literals untouched, `outputs_schema` mapping.
- `branch.rs` — `BranchRouter` conditional fan-out, router closure returns subset, no-match error.
- `streaming.rs` — frame order and indices, `workflow_final` payload, custom frames, stream-mode filtering, error yields one `Err`, end-template frames over a stream producer.
- `end_template.rs` — `{{path}}` templates, missing segments, batch vs streaming response.
- `llm_component.rs` — prompt assembly (system/user, `{{var}}` placeholders), output key mapping, streaming deltas.
- `tool_component.rs` — tool invocation, error mapping to `error_code`.
- `retrieval_component.rs` — `SimpleKnowledgeBase` + `HashEmbedding` + `InMemoryVectorStore` query.
- `loop_component.rs` — number/array/condition loops, bound errors.
- `session_state.rs` — component state namespacing, `dump_state`, trace spans emitted, cancellation.
- `checkpoint_resume.rs` — failing node resumes from the graph store without re-running completed nodes; interruption returns `INPUT_REQUIRED` and resumes with seeded answers.
- `card_spec.rs` — `WorkflowCard`/`WorkflowSpec` JSON round-trip, unknown fields tolerated, `to_mermaid` output.

Behavior ports cite the Python test they mirror in a comment (e.g. `// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow`).
Gates: `cargo fmt -p openjiuwen-workflow`, `cargo test -p openjiuwen-workflow`, `cargo clippy -p openjiuwen-workflow --all-targets -- -D warnings`, all clean and offline.

## Notes for Agents

- Use `openjiuwen_graph::{GraphBuilder, ExecutionOptions, Router, RouterContext, Node, NodeOutcome, NodeOptions, GraphState, CheckpointerStore}`; do not add new graph APIs.
- Stage 1 is implemented: `ability`, `card`, `error`, `reference`, `condition`, `component`, `context`, `output`, `interrupt`, `registry`, and all built-in components under `src/components/`.
- `src/engine/` (`CompiledWorkflow`) and `src/workflow.rs` (`Workflow` builder) are still to be implemented; stage 2 appends those modules to `src/lib.rs`.
- The loop component lives in `src/components/loop_.rs` because `loop` is a Rust keyword; it is re-exported as `LoopComponent`.
- The crate-root re-exports every stage-1 public item, so stage 2 can reach them as `crate::{Component, ComponentContext, ...}`.
- `WorkflowSession` is the expected concrete session; anything implementing `Session` must work (the engine only uses the trait).
- Checkpointing is opt-in through `session.checkpointer()`; when present the engine compiles with `CheckpointerStore` (namespace = workflow card id) and deletes the snapshot when a run completes.
- `Session::config()` supplies `execute_timeout_ms`; `WorkflowRunOptions::timeout` overrides it.
- `openjiuwen-retrieval` is a normal dependency; `openjiuwen-memory` is not (it is still a stub), hence the local memory trait seams.

## Known Issues and Deferred Work

- Deferred components: `SubWorkflowComponent`, `QuestionerComponent`, `IntentDetectionComponent`, `ReActAgentComp`, `HTTPRequestComponent`, and the Python streaming actor network (live interleaving of concurrent producers).
- Deferred semantics: Python's `wait_for_all=False` first-arrival fan-in, `draw()` PNG/SVG rendering, per-run `async` graph mutation, `WORKFLOW_DRAWABLE` gating.
- Deviations from Python documented in `src/components/CONTEXT.md`: batch end templates stringify with JSON text instead of `str()`, branch routers support a Rust-only default target, loops return a `loop_results` wrapper, the retrieval component returns `documents` instead of Python's `results`/`context`, and memory components return `memories`/record results instead of Python's fragment/summary split.
- LLM `response_format` instruction injection and `enable_history` history insertion are carried in config but not applied.
- Unknown-component ids and capability mismatches are reported as one compile error rather than Python's split between build-time and run-start checks.
- `ExceptionConfig::default_outputs` recovery is not wired into the graph executor; only `handle_type="interrupt"` semantics (error propagation) are implemented.
