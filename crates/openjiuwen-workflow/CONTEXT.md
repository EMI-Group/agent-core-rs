# openjiuwen-workflow

## Intent

Workflow builder, components, and `${node.field}` data flow.
Ports the Python package `openjiuwen.core.workflow` (v0.1.20: `workflow.py`, `_workflow.py`, `workflow_config.py`, `base.py`, `components/`) onto `openjiuwen-graph`'s superstep executor, `openjiuwen-session`'s state/stream/checkpoint/trace seams, and `openjiuwen-foundation`'s `Model`/`PromptTemplate`/`Tool`.
A workflow is a directed graph of components with one start and one end; the builder records component configs and edges, and compiling maps them onto `GraphBuilder` nodes/edges/routers.
Data flow is textual: a component's `inputs_schema` is a JSON template whose pure `"${node.field}"` strings resolve against workflow state before the component runs; references never fail (a missing path resolves to `null`).
Components declare capabilities (`invoke`, `stream`, `collect`, `transform`) that determine legal edges and how the engine drives them.
Eighteen built-ins cover control flow (including the four loop-coordination components), the leaf components (LLM, tool, retrieval, memory), HTTP requests (multipart, digest and SigV4 included), human-in-the-loop questioning, intent detection, ReAct, and nested workflows.
`execute_single_component` runs one component outside any graph, mirroring Python's `_workflow.py` helper.
A component vets its own configuration through `Component::validate_config`, so `Workflow::build` reports a bad bound — or any other component-local problem — with its Python status code before a run starts.
Workflows render locally, and deterministically, as Mermaid text or SVG; PNG selection is a typed `Error::Unsupported` because Python rasterizes it through the external mermaid.ink service, and a definition that cannot compile reports the coded compile error instead of a diagram.
Stream composition is buffered and template-ordered; live interleaving is selected through `WorkflowRunOptions::stream_delivery` and `StreamDelivery::Interleaved` is a typed `Error::Unsupported` because `openjiuwen-graph` has no `stream_actor` machinery.

## API Surface

### Builder (`src/workflow.rs`)

```rust
pub struct Workflow { /* WorkflowDefinition: card + config + components + edges + recorded problems */ }
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
    pub fn to_svg(&self) -> Result<String>;                                // build + render
    pub fn draw(&self, format: DrawFormat) -> Result<String>;              // Mermaid text or SVG; Png is Error::Unsupported
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub async fn invoke_with(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn stream_with(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
}
```

- Builder methods never return `Err`; they record problems and `build`/`invoke`/`stream` report them (same policy as `openjiuwen-graph`).
- Setter options: `ComponentOptions` (schema mapping, ability override, `wait_for_all`, `max_retries`, `timeout_ms`, `name`, `exception_config`) — public fields plus chainable `with_*` methods and `into_node_config`; `ComponentOptions::streaming()` marks a streaming end (`response_mode="streaming"`: abilities `[Invoke, Stream]`, plus `Transform` when a stream input schema is set, which also forces `wait_for_all = Some(true)`).
- Recorded problems: a duplicate component id (first registration wins), a second start/end (first wins), and a barrier edge without sources; `build`/`invoke`/`stream`/`to_mermaid` report them through `validate`.
- `invoke`/`stream` compile on the fly; a `Workflow` is immutable during a run, so one instance runs concurrently.
- A recorded topology can vary for one run without touching the builder: `CompiledWorkflow::mutated`/`invoke_mutated`/`stream_mutated` clone the definition, apply a `WorkflowDefinition` mutation, revalidate and recompile (see Design Decisions).

### Compiled engine (`src/engine/`)

```rust
#[derive(Clone)] pub struct CompiledWorkflow { /* Arc inner: graph, card, config, spec, abilities, end id, streaming flag */ }
impl CompiledWorkflow {
    pub fn build(definition: WorkflowDefinition) -> Result<Self>;
    pub fn card(&self) -> &WorkflowCard;
    pub fn config(&self) -> &WorkflowConfig;
    pub fn spec(&self) -> &WorkflowSpec;
    pub fn start_comp_id(&self) -> &str;
    pub fn end_comp_id(&self) -> &str;
    pub fn ability(&self, id: &str) -> Option<ComponentAbility>;
    pub fn is_streaming(&self) -> bool;
    pub fn definition(&self) -> &WorkflowDefinition;         // the recorded topology mutation starts from
    pub fn to_mermaid(&self) -> Result<String>;             // compile + render; a compile failure is the coded error
    pub fn to_svg(&self) -> Result<String>;                 // compile + render; a compile failure is the coded error
    pub fn mutated(&self, mutate: impl FnOnce(&mut WorkflowDefinition)) -> Result<Self>;   // clone → mutate → revalidate → recompile
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub async fn invoke_mutated(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions, mutate: impl FnOnce(&mut WorkflowDefinition)) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn stream_mutated(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions, mutate: impl FnOnce(&mut WorkflowDefinition)) -> BoxStream<'static, Result<WorkflowChunk>>;   // an invalid mutation is the single Err
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
}

#[derive(Clone)] pub struct WorkflowRunOptions {
    pub cancellation: CancellationToken,        // default: never cancelled
    pub timeout: Option<Duration>,              // None → SessionConfig::execute_timeout_ms → none
    pub interactive: Option<InteractiveInput>,  // newly supplied HITL answers
    pub answers: BTreeMap<String, Vec<Value>>,  // per-id FIFO answers replayed before `interactive` (nested runs); `with_answers(..)`, empty by default
    pub resume: bool,                           // continue from a stored graph snapshot
    pub stream_modes: Option<Vec<StreamMode>>,  // which frame modes `stream` yields (None → all)
    pub max_supersteps: Option<usize>,          // None → graph default
    pub depth: usize,                           // nesting level; 0 at top level, child run = parent + 1
    pub stream_delivery: StreamDelivery,        // Buffered (default) | Interleaved (typed Unsupported); `with_stream_delivery(..)`
}

pub enum StreamDelivery { Buffered, Interleaved }   // wire "buffered" | "interleaved"; `as_str`/`Display`/`all`/`is_supported`
```

`Engine` compiles a component graph as follows: every component becomes a `graph::Node` adapter; `add_connection` → `GraphBuilder::edge`, `add_connections` → `barrier_edge`, `add_stream_connection` → `edge` plus stream-ability wiring, conditional connections → `GraphBuilder::conditional_edges_to` with one combined router per source, `start` → `entry`, `end` → `exit`.
A node with `wait_for_all = false` and at least two distinct inbound sources compiles to dynamic "star" edges instead (first-arrival fan-in; see Design Decisions).
`NodeOptions { timeout_ms, max_retries }` carries the per-component limits.
`stream_delivery` is checked at the top of every run entry point, before any session mutation or graph compile.
A per-run mutation recompiles a mutated copy of the recorded definition, so `self` keeps its own topology.

### Single-component execution (`src/single.rs`)

```rust
pub struct SingleComponentOptions { pub inputs_schema: Option<Value>, pub outputs_schema: Option<Value>, pub interactive: Option<InteractiveInput> }
impl SingleComponentOptions {
    pub fn new();                                                    // all defaults
    pub fn with_inputs_schema(self, Value) -> Self;
    pub fn with_outputs_schema(self, Value) -> Self;
    pub fn with_interactive(self, InteractiveInput) -> Self;
}
pub async fn execute_single_component(component_id: &str, component: Arc<dyn Component>, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
pub async fn execute_single_component_with(component_id: &str, component: Arc<dyn Component>, inputs: Value, session: Arc<dyn Session>, options: SingleComponentOptions) -> Result<WorkflowOutput>;
```

- Ports Python `_workflow.py::execute_single_component`: `inputs_schema` resolves against the call's inputs, `invoke` runs once against a `ComponentContext` whose component *and* workflow id are `component_id` (Python builds `WorkflowSession(workflow_id=component_id)`), and `outputs_schema` projects the result through the engine's own projection (`${output.field}` and `${<component_id>.field}` both work).
- An unseeded `interact` returns `WorkflowExecutionState::InputRequired` with the same `{"type","index","payload"}` interaction wires an interrupted run reports; a seeded `interactive` answer completes the call instead.
- Deviations: no graph node, superstep, or barrier participation and no checkpointer snapshot (Python runs a one-vertex Pregel graph whose hooks fire), so a resume has nothing to restore; only the batch `invoke` ability is dispatched; outputs are returned rather than committed to session state under the component id; a non-interrupt failure is returned unwrapped instead of `[100054]`; Python's optional `context` (`ModelContext`) argument has no counterpart because `ComponentContext::history` already reads the session `__messages__` lane.

### Outputs and chunks (`src/output.rs`)

```rust
pub struct WorkflowOutput { pub result: Value, pub state: WorkflowExecutionState }
pub enum WorkflowExecutionState { Completed, InputRequired, Error }   // wire "COMPLETED" | "INPUT_REQUIRED" | "ERROR"
pub type WorkflowChunk = StreamItem;                                    // mode + "type" + index + payload
pub struct OutputFrame { /* type, index, payload */ }                   // Python OutputSchema projection
pub enum WorkflowChunkType { Interaction, Output, Error }               // wire "interaction" | "output" | "error"
pub const END_NODE_STREAM: &str = "end node stream";
pub const WORKFLOW_FINAL: &str = "workflow_final";
```

- `WorkflowChunkType` ports Python `base.py`'s coarse chunk category (`as_str()`/`Display`); the frame itself keeps the finer `type`/`index`/`payload` shape.
- `invoke`: `result` is the end component's output value; when the end is streaming (or the run was interrupted) `result` is the JSON array of collected chunk wires (`{"type","index","payload"}`), matching Python's chunk-list result.
- `stream`: yields frames in production order, then one `workflow_final` output frame carrying the end outputs (omitted when they are null), then ends; an interrupted run ends quietly after its interaction frames; any other failure yields exactly one `Err`.
- Frames a component emits with `ctx.write_stream` / `ctx.write_custom_stream` are forwarded verbatim (mode `output` / `custom`); end-node streaming frames use type `end node stream`; HITL frames use `__interaction__`.

### Components (`src/component.rs`, `src/components/`)

```rust
#[async_trait]
pub trait Component: Send + Sync {
    fn card(&self) -> ComponentCard;                      // identity + capability metadata
    fn abilities(&self) -> Vec<ComponentAbility>;         // default: [Invoke]
    fn component_type(&self) -> &str;                     // "start", "end", "llm", …
    fn validate_config(&self) -> Result<()>;              // default: Ok(()); called for every node at build time
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
pub struct ComponentRegistry;   // name → factory(Value) -> Result<Arc<dyn Component>>; `builtin()` registers nine factories
```

Built-ins (all offline-testable):

| Component | Module | Behavior |
|---|---|---|
| `StartComponent` | `components/start.rs` | `invoke` returns its resolved inputs unchanged |
| `EndComponent` | `components/end.rs` | default `{"output": <non-null inputs>}`; with `EndConfig::response_template` renders `{{path}}` segments (batch: `{"response": …}`; stream/transform: one `end node stream` frame per segment element) |
| `LlmComponent` | `components/llm/` | `PromptTemplate` system/user templates → history splicing → response-format prompt rewrite → `Model::invoke`/`stream`; output shaping from `output_config` (default `{"result": …}`) |
| `ToolComponent` | `components/tool.rs` | `Arc<dyn Tool>::invoke`; returns `{"error_code","error_message","data"}` |
| `BranchComponent` + `BranchRouter` | `components/branch.rs` | `route` evaluates ordered `Branch`es (condition DSL) to target ids; `invoke` returns `{}` |
| `LoopComponent` | `components/loop_.rs` | bounded loop over an inner component list: number / array / condition, capped by `loop_number_max_limit` (1000), returns `{"loop_results": […]}`; plan, cap and break folding come from `loop_driver.rs`, and `validate_config` rejects a missing/contradictory bound (`[101030]`) |
| `LoopGroupComponent` | `components/loop_group.rs` | `loop_group`: number / array / condition / `until_break` over a body, loop variables exposed to the body and updated by `loop_set_variable`, returns `{"loop_results","loop_variables","loop_count","broken"}` |
| `AdvancedLoopComponent` | `components/advanced_loop.rs` | `advanced_loop`: re-evaluates every terminator (cap, count, array, condition, body break) before each round, with per-iteration variable updates landing before the next decision |
| `LoopBreakComponent` | `components/loop_control.rs` | `loop_break`: optional condition → `{"__loop_break__": <bool>}` marker the loop containers fold in |
| `LoopSetVariableComponent` | `components/loop_control.rs` | `loop_set_variable`: template-resolved name→value mapping → `{"__loop_set_variables__": {…}}` marker the loop containers fold in |
| `HttpRequestComponent` | `components/http_request/` | Python `HttpComponentConfig` tree → `openjiuwen_common::http::HttpClient`; returns `{statusCode, headers, body, url, ok}`, 4xx/5xx as data unless `throw_on_http_error` (`[102002]`); JSON/text/form/binary/multipart bodies, basic/bearer/api-key/digest/SigV4 auth, `max_body_length` (`[102001]`) and a per-component rate limiter |
| `QuestionerComponent` | `components/questioner/` | HITL `start`/`user_interact`/`end` state machine persisted under `questioner_state`; optional chat history (`with_chat_history`, default off, `chat_history_max_rounds` default 5) read/written in the `__messages__` lane; returns `{user_response, question, <fields>}` |
| `IntentDetectionComponent` + `IntentRouter` | `components/intent_detection.rs` (+`intent_detection/`) | model or keyword classification → `{classification_id, category_name, reason}`; `IntentRouter` picks successors |
| `ReActAgentComponent` + `AgentRunner` | `components/react_agent.rs` | delegates to an injected runner and normalizes to `{output, result_type}` (`answer`/`error`/`interrupt`) |
| `SubWorkflowComponent` | `components/sub_workflow.rs` | runs a compiled child workflow in a child session and forwards resume answers into it (recorded under `sub_workflow_answers`); abilities `[Invoke, Stream]` |
| `KnowledgeRetrievalComponent` | `components/retrieval.rs` | `SimpleKnowledgeBase::retrieve_with_config` → `{"query", "documents": [{text, score, metadata}]}` |
| `MemoryRetrievalComponent`, `MemoryWriteComponent` | `components/memory.rs` | operate over the local `MemoryRecall` / `MemoryRecord` trait seams (adapters land in `openjiuwen-memory`) |

### Component context (`src/context.rs`)

```rust
pub struct ComponentContext { /* session + component/workflow ids + interaction state + cancellation + depth + history */ }
impl ComponentContext {
    pub fn new(session: Arc<dyn Session>, component_id: impl Into<String>, workflow_id: impl Into<String>) -> Self;
    // chainable for direct component tests and engine wiring:
    pub fn with_interactions(self, Arc<InteractionState>) -> Self;
    pub fn with_writer(self, Arc<FrameWriter>) -> Self;
    pub fn with_cancellation(self, CancellationToken) -> Self;
    pub fn with_metadata(self, BTreeMap<String, Value>) -> Self;
    pub fn with_executable_id(self, impl Into<String>) -> Self;
    pub fn with_depth(self, usize) -> Self;                                  // 0 at top level
    pub fn with_history(self, Arc<dyn MessageHistory>) -> Self;
    pub fn session(&self) -> &Arc<dyn Session>;
    pub fn component_id(&self) -> &str;  pub fn workflow_id(&self) -> &str;  pub fn executable_id(&self) -> &str;
    pub fn depth(&self) -> usize;
    pub fn history(&self) -> Arc<dyn MessageHistory>;                        // defaults to SessionMessageHistory over the session
    pub fn history_corruption(&self) -> Option<HistoryCorruption>;           // what the store's last read/write reported
    pub fn history_corrupt_error(&self) -> Option<Error>;                    // that record as the coded `[100060]` error
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
    pub fn seeded_answer(&self, id: &str) -> Option<Value>;                   // consuming pop of the next queued answer
    pub fn queued_answers(&self) -> BTreeMap<String, Vec<Value>>;            // non-consuming; nested components forward these to a child run
}
```

### Conversation history (`src/context/history.rs`)

```rust
pub const MESSAGES_STATE_KEY: &str = "__messages__";
#[async_trait] pub trait MessageHistory: Send + Sync {
    async fn messages(&self) -> Vec<BaseMessage>;
    async fn append(&self, messages: Vec<BaseMessage>);
    async fn clear(&self);
    fn corruption(&self) -> Option<HistoryCorruption> { None }   // defaulted: external stores stay source-compatible
}
pub struct SessionMessageHistory;   // new(Arc<dyn Session>), session() -> &Arc<dyn Session>
pub struct HistoryCorruption { pub reason: HistoryCorruptionReason, pub detail: String }
pub enum HistoryCorruptionReason { NotAnArray, Unreadable, MalformedEntries, WriteFailed }   // wire "not_an_array" | "unreadable" | "malformed_entries" | "write_failed"
pub const WORKFLOW_HISTORY_CORRUPT: u32 = 100_060;
pub fn history_corrupt_error(&HistoryCorruption) -> Error;   // "[100060] message history is corrupt, reason=<reason>, detail='<detail>'"
```

The store is the session global lane `__messages__` (a JSON array of `{"role","content"}`), shared by every component on one session.
Reads and writes never fail a run: a missing lane is clean, empty history; a non-array lane, an unreadable lane, skipped malformed entries, or a failed write is logged with `tracing::warn!` and recorded as a `HistoryCorruption` (each record's `detail` names the lane, the session error, or the skipped-entry count).
`corruption()` always describes the most recent read or write, and a successful `append`/`clear` clears the record because the lane then holds exactly what the store wrote.
`ComponentContext::history()` keeps one `SessionMessageHistory` per context, so `history_corruption`/`history_corrupt_error` observe what a component's store reported; an overridden store reports its own record, and a store that keeps the trait default always reports `None`.

### Rendering (`src/render.rs`)

```rust
pub enum DrawFormat { Mermaid, Svg, Png }       // wire "mermaid" | "svg" | "png"; default Mermaid
pub enum LayoutDirection { LeftRight, TopDown }
pub struct SvgOptions { pub title: Option<String>, pub direction: LayoutDirection, pub end_nodes: Vec<String> }
pub fn to_svg(spec: &WorkflowSpec, card: &WorkflowCard, title: Option<&str>) -> String;
pub fn render_svg(spec: &WorkflowSpec, card: &WorkflowCard, options: &SvgOptions) -> String;
pub fn error_svg(message: &str) -> String;
```

Layered BFS layout over the sorted spec maps, rounded start/end boxes, arrow markers, dashed stream edges, XML-escaped labels, and an optional `<title>`; output is byte-identical for the same spec (no clocks, no unordered iteration).
`Workflow::draw` renders `Mermaid` text and `Svg` here; `Png` returns a typed `Error::Unsupported` whose message names the external mermaid.ink service and points at `DrawFormat::Svg` (no local rasterizer, and the rejection never degrades into another format's output).
`CompiledWorkflow::to_svg`/`to_mermaid` (and `Workflow::to_svg`/`to_mermaid`) return `Result<String>` and report the coded compile error, so an uncompilable definition never becomes a document or a `%%` comment; `error_svg` stays a public utility for callers that explicitly want an error document, and the engine does not use it.

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
pub fn generate_workflow_key(workflow_id: &str, workflow_version: &str) -> String;   // Python base.py: "{id}_{version}", bare concatenation
```

Field names, defaults, and card keys match Python (`id`, `name`, `description`, `version`, `input_params`; `timeout` in seconds, negative = none). Every type is serde snake_case, tolerates unknown fields, and round-trips through JSON.
`WorkflowCard` and `ComponentCard` are plain serde types: neither implements `openjiuwen_common::card::Card`, embeds `CardBase`, nor derives `schemars::JsonSchema`, so no JSON Schema is exposed for workflow cards, and neither type has a field-level serde attribute (`rename_all` is a no-op on the already-snake_case field names).

### Errors (`src/error.rs`)

`openjiuwen_common::error::{Error, Result}` only; `Error::Workflow` carries Python-style `"[<code>] <message>"` text with the code constants (`WORKFLOW_COMPONENT_ID_INVALID = 100010`, `…_ABILITY_INVALID = 100011`, `…_EDGE_INVALID = 100012`, `…_CONDITION_EDGE_INVALID = 100013`, `…_SCHEMA_INVALID = 100014`, `…_STREAM_EDGE_INVALID = 100015`, `…_COMPILE_ERROR = 100100`, `…_COMPONENT_EXECUTION_ERROR = 100054`, `COMPONENT_BRANCH_EXECUTION_ERROR = 101021`, `COMPONENT_LOOP_EXECUTION_ERROR = 101040`, `COMPONENT_LOOP_PARAM_INVALID = 101030`, `COMPONENT_LOOP_SET_VAR_PARAM_INVALID = 101031`).
Component-local codes live beside the component: `intent_detection/error.rs` owns `101050..=101062`, the questioner owns `101070..=101076`, the HTTP component `102000..=102002` (`102002` is its own `throw_on_http_error` failure code, borrowed from the tool-init slot because Python never raises on a status), the LLM component `101002`/`101004`, and the nesting guard `100100`.
`context/history.rs` owns the Rust-only `WORKFLOW_HISTORY_CORRUPT = 100060` and its `history_corrupt_error` builder, because `src/error.rs` is the shared Python code table; `100060` sits in the unused gap between `100054` and `100100`.

## Constraints

- Follows `docs/ARCHITECTURE.md`: async-first tokio, `openjiuwen_common::error` results, serde-serializable public types, no `unsafe`, `tracing` only (no stdout), files ≤ ~1000 lines.
- The graph engine is consumed as-is (`openjiuwen-graph`); this crate never reimplements supersteps, channels, checkpoints, or Mermaid rendering.
- Only this crate's files and `Cargo.toml` change; `openjiuwen-session`, `openjiuwen-foundation`, `openjiuwen-retrieval` and `openjiuwen-graph` are used through their documented seams.
- Public streams are `Send + 'static`; cancellation flows through `CancellationToken`; locks are never held across `.await`.
- No panics on user input: every builder problem, schema problem, and component failure becomes an `Error`; component panics are not caught (same policy as the rest of the workspace).
- Tests are deterministic and offline: `ScriptedModel`, `HashEmbedding`, `InMemoryVectorStore`, loopback `wiremock`, no sleeps for ordering.

## Design Decisions

- **Compiled per call, no runnable state in the builder.** `Workflow` holds `Arc<dyn Component>` handles plus recorded config; `build()` validates once and produces a `CompiledWorkflow` that owns the graph. `stream` clones the compiled graph into the returned stream so the stream is `'static`.
- **Per-run mutation is a recompiled copy.** `CompiledWorkflow::mutated` clones the recorded `WorkflowDefinition`, applies a mutation (`component_mut`, `remove_component`, `remove_edges_to`, `retarget_edge`), and validates/compiles the copy, so `invoke_mutated`/`stream_mutated` run one altered topology while the receiver keeps its own and stays concurrently runnable. The mutation runs before the run starts — never mid-execution — so the immutable-record model above still holds.
- **Components receive a cloned `GraphState` and write their output under their own id.** This is the graph crate's node-delta model: the engine resolves schemas against state, calls the component, then writes `state[component_id] = output`. References therefore compose without a separate data-bus.
- **Streaming is buffered, and the choice is an explicit seam.** A `Stream`/`Transform` producer accumulates its frames into an array stored under its id instead of feeding live generators downstream; a `Collect`/`Transform` consumer resolves `stream_inputs_schema` (projecting arrays) and runs after its producers. Frames the *end* node emits are still `end node stream` frames in template order, so the observable text and order match Python for finite streams; only live interleaving of two concurrently streaming producers is lost. `WorkflowRunOptions::stream_delivery` (`StreamDelivery::Buffered` by default) states the model, and `StreamDelivery::Interleaved` — Python's `stream_actor` behaviour — is rejected by every run entry point with a typed `Error::Unsupported` naming that gap, before any session mutation or graph compile; `stream` delivers the rejection as its single `Err` item.
- **Array projection is an intentional extension.** `resolve_path` projects a field over an array of objects (`${producer.output}` → `[f1, f2, …]`) so buffered stream composition works; Python would return `null` there.
- **Interrupts are data, not exceptions.** `ComponentContext::interact` returns a seeded answer when one exists; otherwise it records an interaction frame and returns an interrupt error that the node adapter propagates, so the graph's failure path saves the *pre-superstep* boundary with the pending channel messages. The run reports `INPUT_REQUIRED` with the interaction wires, and `Workflow::resume` re-runs that superstep with the seeded answers (`InteractiveInput`).
- **`wait_for_all = false` is first-arrival fan-in.** A node with that flag and ≥2 distinct inbound sources compiles its static edges as dynamic router edges, so `openjiuwen-graph` keeps no CNF group and the channel becomes a trigger channel: the node runs on *any* arrival, runs again on a later one, and a not-yet-produced template key resolves to `null`. Single-source nodes, nodes reached only by a declared conditional edge, and `Collect`/`Transform` nodes (validation `[100011]`) stay CNF.
- **`ExceptionConfig::default_outputs` recovery is inline.** After the graph exhausts `max_retries`, an `Invoke`/`Collect` node whose config carries `default_outputs` under a non-`interrupt` `handle_type` writes those defaults (projected through `outputs_schema`) under its id and the run continues; the span notes `error_recovery`. Streaming failures and `handle_type == "interrupt"` still fail with `[100054]`.
- **`ComponentAbility` serializes as `"invoke" | "stream" | "collect" | "transform"`.** Python's tuple-valued enum dumps as tuples, which is not a usable JSON contract; the string form is the Rust wire format.
- **Session component state is namespaced by component id via the `Session` trait.** Python's `NodeSession` nesting is approximated with `get_component_state`/`update_component_state`; `${node.field}` resolution uses graph state, not session lanes.
- **Build-time configuration checking is a component hook.** `Component::validate_config` (default `Ok(())`) is called for every registered node by `engine::validate`, so a component's own configuration problem is reported with its status code before any run and joined into the same `"; "`-separated compile error as the topology checks. The loop family is its first user: a missing, zero or over-cap bound and an unterminatable container are `[101030]`, an invalid variable mapping `[101031]`. A loop *body* is a plain component list, not registered nodes, so a body component's check is not reached by `Workflow::build`.
- **Session-history corruption is signalled, not just logged.** A broken `__messages__` lane still degrades to empty history (never a run failure by itself), but the store records a `HistoryCorruption` with a reason and a detail, `corruption()` reports the most recent read or write, and `ComponentContext::history_corruption`/`history_corrupt_error` let a component either inspect it or fail hard with `[100060]`. Reporting is additive: `corruption()` is defaulted, so the three data methods keep their signatures and external stores stay source-compatible.
- **HTTP multipart, digest and SigV4 are hand-rolled in this crate.** The multipart encoder builds the boundary and CRLF-framed parts itself and sends them through `HttpRequest::bytes`, so retries, cancellation and the size cap keep their single implementation; digest is challenge-driven (a `401` with `WWW-Authenticate: Digest` replays once with the computed header) and SigV4 is signed pre-flight, with a hand-rolled RFC 1321 MD5 and HMAC-SHA256 over `sha2`. Python only declares these configuration branches, so their semantics are this crate's.
- **Conversation history is session-backed.** `enable_history` reads and writes `__messages__` through `MessageHistory`/`SessionMessageHistory`, so history-enabled nodes on one session share it (Python keeps it per `ModelContext`).
- **`response_format` rewrites the prompt, not the request.** Markdown and JSON replace the *last user message* with the verbatim Python instruction templates and shape the answer locally; no provider-side `response_format` parameter is sent.
- **Sub-workflows get a child session, not a nested graph.** `SubWorkflowComponent` runs a `CompiledWorkflow` against a fresh `WorkflowSession` parented to the caller, so child node ids never collide with parent ids; `depth + 1` flows through `WorkflowRunOptions::depth` and the child's `workflow_max_nesting_depth` rejects deeper nesting with `[100100]`.
- **Nested resume replays answers instead of re-running blind.** The child is replayed from a fresh session (its own snapshot is not resumed), so the sub-workflow node reads the answers already forwarded from the parent session lane `sub_workflow_answers` (`NESTED_ANSWERS_KEY`), folds Python's dotted `"<sub node id>.<child id>"` ids onto the bare child id, appends the answers still queued in the parent run (`ComponentContext::queued_answers`), records the merged map back, and passes it as `WorkflowRunOptions::answers`; the child run therefore consumes the same answers in the same order on every replay.
- **Rendering is local and never fabricates a diagram.** `draw(DrawFormat::Svg)`/`to_svg` produce SVG in-process (Python calls the mermaid.ink service), so the same spec always renders the same bytes; a definition that cannot compile returns the coded compile error instead of an error document or a Mermaid comment, and `render::error_svg` remains only for callers that explicitly ask for one.
- **PNG is a typed rejection, not a fallback.** `DrawFormat` keeps Python's third format so a caller selecting it gets an `Error::Unsupported` naming mermaid.ink and pointing at `DrawFormat::Svg`; silently returning SVG or Mermaid text would misreport the requested format.
- **`WorkflowInput`/`session` handling is explicit.** `invoke` merges the workflow inputs into the session global lane (Python `commit_user_inputs`) and seeds graph state with them, so root refs (`${query}`) and node refs (`${start.query}`) both resolve.

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, crate-root re-exports
- `./src/ability.rs` → `ComponentAbility`
- `./src/card.rs` → `WorkflowCard`, `WorkflowConfig`, `WorkflowSpec`, `NodeSpec`, `CompIoConfig`, `ExceptionConfig`
- `./src/error.rs` → Python error codes and message builders
- `./src/reference.rs` → `${node.field}` parsing, nested paths, schema resolution
- `./src/condition.rs` → condition expression parser/evaluator, `Branch`, `BranchRouter`
- `./src/component.rs` → `Component` trait (including `validate_config`), `ComponentCard`, ability defaults, boxed helpers
- `./src/context.rs` → `ComponentContext` session facade (state, streams, HITL, cancellation, depth, history, history corruption)
- `./src/context/history.rs` → `MessageHistory`, `SessionMessageHistory`, `__messages__` lane, `HistoryCorruption`/`HistoryCorruptionReason`, `WORKFLOW_HISTORY_CORRUPT`/`history_corrupt_error`
- `./src/interrupt.rs` → interaction frames, seeded-answer queues (`seed`, `append`, `queued_answers`, `take_answer`), interrupt error
- `./src/registry.rs` → `ComponentRegistry` and the nine builtin factories (`start`, `end`, `branch`, `loop`, `loop_group`, `advanced_loop`, `loop_break`, `loop_set_variable`, `http_request`)
- `./src/output.rs` → `WorkflowOutput`, `WorkflowExecutionState`, chunk/frame helpers
- `./src/render.rs` → offline SVG renderer (`to_svg`, `render_svg`, `error_svg` as a public error-document utility the engine does not use, `SvgOptions`, `DrawFormat`, `LayoutDirection`)
- `./src/single.rs` → `execute_single_component`/`execute_single_component_with`, `SingleComponentOptions` (one component outside a graph)
- `./src/components/` → built-in components, the loop-coordination family and the `http_request/` submodules (`components/CONTEXT.md`, `components/http_request/CONTEXT.md`)
- `./src/engine/` → `CompiledWorkflow`, compile/validation, `StreamDelivery`, run/stream, node adapter, per-run mutation in `mutation.rs` (`engine/CONTEXT.md`)
- `./src/workflow.rs` → `Workflow` builder, `ComponentOptions`, `build`/`to_mermaid`/`to_svg`/`draw` and the `invoke`/`resume`/`stream` entry points
- `./tests/` → behavior tests (see Test Strategy)

## Test Strategy

Unit tests beside modules: `reference` (path grammar, projection, missing paths), `condition` (operator precedence, helpers, errors), `card` (JSON round-trip, defaults, `generate_workflow_key`), `interrupt` (seeding order, `append`/`queued_answers`), `components::end` (template segmentation/rendering), `components::loop_` (bound enforcement, markers, `validate_config`), `components::loop_driver` (bound planning, `validate_bound`, `object_inputs`, variable resolution, iteration inputs/overlay, marker folding, container wrapper), `components::loop_group` and `components::advanced_loop` (each bound kind, variables, break handling, `validate_config`, JSON round-trip), `components::loop_control` (`should_break`, marker shapes, name normalization, `101031`), `components::http_request::{template, body, multipart, auth, digest, sigv4, response, rate_limit}` (templating, body selection, part framing, credentials, RFC 1321/2617 and AWS vectors, response formats, limiter schedule), `components::llm::format` (instruction rewriting, schema generation, output shaping), `components::intent_detection::tests` (category model, model-reply post-processing, keyword rules, `IntentRouter`), `context::history` (lane round-trip, role filtering, every `HistoryCorruptionReason`, JSON round-trip, the `100060` builder), `render` (layout, escaping, determinism), `engine` (stream termination, buffered frames, backpressure with and without a deadline, the streaming deadline/cancellation terminal `Timeout`/`Cancelled` errors, snapshots, resume, `to_mermaid`/`to_svg` compile-error propagation, run-option defaults), plus the inline suites of most component modules (the questioner's behavior is covered at crate level by `tests/questioner_component.rs`).

Integration tests in `./tests/` (all offline, using `openjiuwen_test_support::ScriptedModel`, loopback `wiremock` where HTTP is involved, and `openjiuwen_session::WorkflowSession`):

- `common/mod.rs` — shared helpers: session factory, `EchoComponent`/`CounterComponent`/streaming fakes, tool fixture.
- `builder_validation.rs` — missing/duplicate start and end, bad ids, unknown edge endpoints, stream-edge capability mismatch, schema key overlap, `ComponentOptions` defaults, and the build-time component checks (`[101030]` for a missing bound, an over-cap group and an unterminatable group), while a valid bounded group still builds and runs.
- `linear.rs` — Start → LLM (`ScriptedModel`) → End, `invoke` result and session global state.
- `references.rs` — `${start.a}`, nested `a.b[0]`, whole-node refs, missing path → null, literals untouched, `outputs_schema` mapping.
- `branch.rs` — `BranchRouter` conditional fan-out, router closure returns subset, no-match error.
- `fan_in.rs` — CNF barrier versus first-arrival activation, `null` for a not-yet-produced key, later arrivals re-firing.
- `streaming.rs` — frame order and indices, `workflow_final` payload, custom frames, stream-mode filtering, error yields one `Err`, end-template frames over a stream producer, and the `StreamDelivery` contract (default `Buffered`, wire names, `Interleaved` rejected by `invoke`/`stream`/`resume` before any session mutation).
- `end_template.rs` — `{{path}}` templates, missing segments, batch vs streaming response.
- `llm_component.rs` — prompt assembly (system/user, `{{var}}` placeholders), output key mapping, streaming deltas.
- `llm_semantics.rs` — `enable_history` turn splicing/recording, `response_format` prompt rewrite and `output_config` shaping.
- `tool_component.rs` — tool invocation, error mapping to `error_code`.
- `retrieval_component.rs` — `SimpleKnowledgeBase` + `HashEmbedding` + `InMemoryVectorStore` query.
- `loop_component.rs` — number/array/condition loops, bound errors, `101040` at invoke time, and the build-time `[101030]` rejection of a loop without its bound parameter.
- `loop_coordination.rs` — the four loop-coordination components through the public API and by direct invoke: the container wrapper (`loop_results`/`loop_variables`/`loop_count`/`broken`) for number, array, condition, `until_break` and nested containers, variable accumulation, break handling (conditional, unconditional, inside `AdvancedLoopComponent`), the session iteration cap, the standalone markers and the `[101040]` cap error.
- `history_corruption.rs` — the `MessageHistory` corruption signal: a non-array lane (`NotAnArray`), malformed entries skipped and counted, a clean lane plus `append`/`clear` forgetting the record, handles over one session sharing it, an external implementation inheriting the default `None`, and `ComponentContext` observing (or hiding, when the store is overridden) that record.
- `per_run_mutation.rs` — per-run topology mutation as a recompiled copy: a mutated end template applying to one run only, inserting a node and retargeting an edge, removing a node and rewiring edges, the coded `[100100]` compile error for an invalid mutation, `Interleaved` still rejected, `stream_mutated` frames, the receiver staying runnable concurrently, and the mutation helpers being state-aware.
- `http_component.rs` — request encoding, response projection, `response_data_property`/`response_mode`, 4xx/5xx as data versus `throw_on_http_error` (`[102002]`), retries, `[102000]` transport failures, multipart framing, digest challenge/replay (and a `401` without a challenge), SigV4 signing, the `max_body_length` rejection, the per-component rate limiter (with cancellation and a generous config), and `binary` versus `buffer` bodies.
- `questioner_component.rs` — `start → questioner → end` over a checkpointed session, `INPUT_REQUIRED` plus `resume` per question, and the opt-in chat history (prior turns in `dialogue_history`, rounds trimming, trailing-turn rule, user/assistant recording, structured content).
- `intent_detection.rs` — model path via `ScriptedModel`, offline keyword path, `IntentRouter` successor selection.
- `react_agent.rs` — inputs reach the runner unchanged, `{output, result_type}` normalization, error/interrupt data, `on_max_iterations` policies.
- `sub_workflow.rs` — child output passthrough, isolated child state lanes, nesting limit, forwarded frames, stream cache, relayed HITL, and nested resume (bare and dotted interaction ids answering the child, the recorded `sub_workflow_answers` lane, replay across two resumes).
- `exception_recovery.rs` — `default_outputs` recovery after retries, `outputs_schema` projection, `interrupt`/stream failures still propagate.
- `draw.rs` — `draw` formats, deterministic SVG, XML escaping, rounded start/end, empty/single/disconnected specs, the `Png` wire name and its typed `Unsupported` (naming mermaid.ink and pointing at `Svg`, with no SVG/Mermaid text leaking), and an uncompilable workflow reporting the compile error instead of an error document or a Mermaid `%%` comment.
- `single_component.rs` — `execute_single_component(_with)`: input/output template resolution, passthrough without schemas, `INPUT_REQUIRED` plus a seeded answer, error propagation, `generate_workflow_key`, `WorkflowChunkType` wire values.
- `public_api.rs` — crate-root/module-path parity for the re-exported surface (including the loop-coordination components, `LoopBound`, `StreamDelivery` and the history-corruption items), Python-shaped `HttpComponentConfig` round-trip through the root types, alias usability, and the wire strings of the HTTP enums, component status codes and `on_max_iterations` policy constants.
- `session_state.rs` — component state namespacing, `dump_state`, trace spans emitted, cancellation.
- `checkpoint_resume.rs` — failing node resumes from the graph store without re-running completed nodes; interruption returns `INPUT_REQUIRED` and resumes with seeded answers.
- `card_spec.rs` — `WorkflowCard`/`WorkflowSpec` JSON round-trip, unknown fields tolerated, `to_mermaid` output.

Behavior ports cite the Python test they mirror in a comment (e.g. `// port of tests/unit_tests/core/workflow/test_workflow.py::test_simple_workflow`).
Gates: `cargo fmt -p openjiuwen-workflow`, `cargo test -p openjiuwen-workflow`, `cargo clippy -p openjiuwen-workflow --all-targets -- -D warnings`, all clean and offline.

## Notes for Agents

- Use `openjiuwen_graph::{GraphBuilder, ExecutionOptions, Router, RouterContext, Node, NodeOutcome, NodeOptions, GraphState, CheckpointerStore}`; do not add new graph APIs.
- Every module is implemented: `ability`, `card`, `error`, `reference`, `condition`, `component`, `context` (+`history`), `output`, `interrupt`, `registry`, `render`, `single`, `components/`, `engine/`, and `workflow`.
- Component families that grew beyond one file live in directories: `components/llm/` (`mod.rs`, `format.rs`, `messages.rs`), `components/http_request/` (`mod.rs`, `config.rs`, `template.rs`, `body.rs`, `multipart.rs`, `auth.rs`, `digest.rs`, `sigv4.rs`, `response.rs`, `rate_limit.rs`), `components/questioner/` (`mod.rs`, `config.rs`, `prompts.rs`), and `components/intent_detection/` (`error.rs`, `tests.rs`) beside `components/intent_detection.rs`.
- Local error codes live beside their component (`components/intent_detection/error.rs`, the questioner's `coded` helpers) instead of growing `src/error.rs`.
- `src/workflow.rs` exposes `Workflow` (builder + `build`/`invoke`/`invoke_with`/`resume`/`stream`/`stream_with`/`to_mermaid`/`to_svg`/`draw`) and `ComponentOptions`; both are re-exported from the crate root.
- `stream`/`stream_with` deliver a build error as the stream's single `Err` item instead of returning it.
- Every suite listed under Test Strategy exists in `./tests/`; `tests/common/mod.rs` provides the shared fakes (`session`, `session_with_checkpointer`, `EchoComponent`, `FixedComponent`, `CounterComponent`, `StreamingProducer`, `TransformProducer`, `CollectorComponent`, `InteractingComponent`, `FailingComponent`, `WaitOnCancellationComponent`, `b_of_a_tool`, `llm_component`, `end_with_template`).
- The loop family spans four modules: `src/components/loop_.rs` (`LoopComponent`, renamed because `loop` is a Rust keyword), `loop_group.rs` (`LoopGroupComponent`), `advanced_loop.rs` (`AdvancedLoopComponent`) and `loop_control.rs` (`LoopBreakComponent`, `LoopSetVariableComponent`), all sharing `loop_driver.rs` for bound planning, iteration inputs, marker folding, the container wrapper and build-time bound validation.
- `src/condition.rs` is ~1100 lines on purpose: the lexer, parser, evaluator, `Condition`, `Branch`, `BranchRouter` and their inline tests form one cohesive grammar unit and lose coherence if split (split only when a second expression family appears).
- `src/engine/definition.rs` (~1000 lines) likewise keeps the builder's recorded definition, ability selection, and `WorkflowSpec` projection together so the builder and validator share one source of truth.
- The crate root mirrors Python's `openjiuwen/core/workflow/__init__.py`: every built-in component (the loop-coordination family included), the whole `Http*` configuration tree, the component status codes (`COMPONENT_QUESTIONER_*`, `COMPONENT_INTENT_DETECTION_*`, `COMPONENT_LOOP_PARAM_INVALID`, `COMPONENT_LOOP_SET_VAR_PARAM_INVALID`), the `on_max_iterations` policy constants, `card::{MAX_COMPONENT_ID_LEN, INVALID_ID_CHARS_REASON, generate_workflow_key}`, `output::{CUSTOM_FRAME_TYPE, TRACE_FRAME_TYPE, FrameCollector, frame_to_wire, WorkflowChunkType}`, `single::*`, the engine surface (`CompiledWorkflow`, `StreamDelivery`, `WorkflowRunOptions`, `WorkflowDefinition`, `NodeConfig`, `Edge`, `EdgeKind`, `Entry`, `ConditionalSource`), the history-corruption surface (`HistoryCorruption`, `HistoryCorruptionReason`, `MessageHistory`, `SessionMessageHistory`, `MESSAGES_STATE_KEY`, `history_corrupt_error`, `WORKFLOW_HISTORY_CORRUPT`), and the Python-name aliases `LLMComponent`/`LLMCompConfig` (`components::llm`) and `ReActAgentCompConfig` (`components::react_agent`).
- Module-path-only items (reachable as `openjiuwen_workflow::<module>::…` but not at the crate root): the component error *builder functions* — `components::questioner::{input_param_error, config_error, input_invalid, state_init_failed, runtime_error, invoke_call_failed, execution_process_error}` and the 13 `components::intent_detection::intent_detection_*` builders (their status codes are at the root); `components::sub_workflow::NESTED_ANSWERS_KEY`; `render::error_svg`; and `engine::validate::validate`. The history items are *not* in this group: `history_corrupt_error` and `WORKFLOW_HISTORY_CORRUPT` are re-exported at the root beside `HistoryCorruption`/`HistoryCorruptionReason`.
- `src/single.rs` reuses the engine's `pub(crate)` helpers rather than duplicating them: `engine::run::interaction_wires` for the interrupted result and `engine::node::project_output` for `outputs_schema`, so a single-component call and a workflow run agree on both shapes.
- `Component::route` is consulted before the selected ability and wins outright: a component returning `Some(targets)` produces no value and writes no state (`NodeOutcome::Route`). `IntentDetectionComponent` therefore does not implement `route`; a caller wires an `IntentRouter` through `add_conditional_connection` so the classification is still written to state.
- `AgentRunner` is this crate's seam to the agent runtimes: `openjiuwen-agent`/`openjiuwen-harness` adapt to it in their own crates, and this crate never depends on them.
- HTTP goes through `openjiuwen_common::http::HttpClient`; the crate manifest therefore depends on `base64`, `bytes` and `sha2` (all workspace deps) — `sha2` is used only by the hand-rolled SigV4 signer and digest SHA-256, since no `md-5`/`hmac` crate is available.
- A `CompiledWorkflow` is never edited in place: `definition()` exposes the recorded topology read-only, and per-run topology changes go through `mutated`/`invoke_mutated`/`stream_mutated`, which mutate a clone and revalidate it.
- `WorkflowSession` is the expected concrete session; anything implementing `Session` must work (the engine only uses the trait).
- Checkpointing is opt-in through `session.checkpointer()`; when present the engine compiles with `CheckpointerStore` (namespace = workflow card id) and deletes the snapshot when a run completes.
- `Session::config()` supplies `execute_timeout_ms`; `WorkflowRunOptions::timeout` overrides it.
- `openjiuwen-retrieval` is a normal dependency; `openjiuwen-memory` is not one, although it is fully implemented — the memory components reach it through the local `MemoryRecall`/`MemoryRecord` trait seams, so callers inject adapters instead of this crate depending on the memory engine.
- `ComponentContext::interact` keys an interaction by the question object's `"id"` field when present, otherwise by the component id; a resume answer may be node-addressed (`InteractiveInput::update`) or interrupt-addressed (`InteractiveInput::answer`), and for one id an interrupt-addressed answer in the same seeding call wins over the node-addressed one.
- `Workflow::resume` requires the session to carry a checkpointer (the interrupted superstep is replayed from the stored graph snapshot); `openjiuwen_test_support::workflow_session_with_checkpointer` is the fixture for that.
- Consumers without a direct dependency on this crate (for example `openjiuwen-e2e`) reach every public item through the umbrella as `openjiuwen::core::workflow::…`; no feature flag gates workflow types.

## Known Issues and Deferred Work

- Python's condition classes (`FuncCondition`, `ArrayCondition`, `NumberCondition`, `AlwaysTrue`) map onto `Condition`'s expression DSL plus `BranchRouter`; a closure condition has no serializable equivalent because branches are registry/serde configs.
- Mutually exclusive branch predecessors are not merged into one OR group (that needs graph-level barrier groups), so a join reached only through declared conditional edges stays CNF and waits for every predecessor; workaround: a dedicated successor per alternative, or `wait_for_all = false` on the join (which validation rejects for `Collect`/`Transform`).
- `Workflow::draw(DrawFormat::Png)` is a typed `Error::Unsupported` because Python posts the chart to the external mermaid.ink service and no offline rasterizer reproduces it; Python's `WORKFLOW_DRAWABLE` environment gate is not modelled, so rendering is always available.
- Interaction ids are the bare component id, where Python uses the node's possibly dotted path; nested answer forwarding accepts both spellings.
- Python's per-component config objects are replaced by injected collaborators: `ToolComponentConfig`, `KnowledgeRetrievalCompConfig`/`ComponentKBConfig` and `MemoryRetrievalCompConfig`/`MemoryWriteCompConfig` have no Rust counterpart (the components take the tool/KB/memory seam directly), and `WorkflowCard::tool_info()` has no counterpart because workflow-as-tool registration belongs to the runner/agent layers.
- A workflow is immutable during a run: per-run topology changes are a mutated, revalidated copy (`CompiledWorkflow::mutated`/`invoke_mutated`/`stream_mutated`), never an edit of a running graph.
- `outputs_schema` resolves against workflow state plus the node's own id and an `output` key (`${<id>.field}` / `${output.field}`); there is no `self` shorthand, and a bare `${field}` resolves to `null`.
- `ComponentOptions::streaming()` declares `[Invoke, Stream]` (plus `Transform` when a stream input schema is set), so it cannot be used on a component that implements only `Stream`, and a streaming end is not a valid `add_stream_connection` target unless it declares `Transform`.
- Unknown-component ids and capability mismatches are reported as one compile error rather than Python's split between build-time and run-start checks.
- Recovery is inline in `ComponentNode::run` rather than Python's `component_error_recovery` callback framework; the observable contract (defaults written under the node id, run continues) is the same.
- `enable_history` history is per session (`__messages__`), not per `ModelContext`: two history-enabled nodes on one session see each other's turns, and only plain text survives a round trip.
- Hot-path cost: `ComponentNode::run` deep-clones the whole state (`state.as_value()`) and the resolved inputs (for the trace span, whether or not a tracer exists) and `project_output` clones state plus the raw output twice, so one node execution costs several full-state copies per superstep.
- Hot-path cost: `reference::resolve_schema` re-parses every `${…}` string (`split_path` allocates a `Vec<char>` and a `Vec<PathSegment>`) on every node execution and every output projection; nothing is cached even though a node's schemas are immutable.
- `components/http_request/mod.rs::HttpRequestComponent::invoke` builds a new `HttpClient` per invocation, so connection pooling is lost and each HTTP node pays a client/TLS build.
- `components/http_request/digest.rs::digest_parameters` slices `trimmed[..6]` without a char-boundary check; unreachable through HTTP today only because `HeaderValue::to_str` accepts visible ASCII, but a non-ASCII header value would panic.
- Deviations from Python documented in `src/components/CONTEXT.md`: batch end templates stringify with JSON text instead of `str()`, branch routers support a Rust-only default target, loops return a `loop_results` wrapper, the retrieval component returns `documents` instead of Python's `results`/`context`, memory components return `memories`/record results instead of Python's fragment/summary split, the HTTP component accepts `max_redirects`/`disable_compression`/`use_stream` without acting on them (its multipart, digest and SigV4 branches are implemented here while Python only declares the configuration), the questioner injects its model directly and keeps chat history in the session `__messages__` lane rather than a `ModelContext` window, intent detection adds an offline keyword path, ReAct returns a plain answer/observations value instead of Python's rich result dict and maps interrupts to data, and a sub-workflow's stream cache merges mixed types last-wins instead of raising `TypeError`.
