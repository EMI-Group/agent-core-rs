# openjiuwen-workflow/src/engine

## Intent

The compile-and-run side of a workflow: `WorkflowDefinition` is what the builder records, `validate` turns a
definition into one compile error listing every problem, `ComponentNode` adapts a `Component` to an
`openjiuwen_graph::Node`, and `CompiledWorkflow` maps a validated definition onto a fresh
`openjiuwen_graph::CompiledGraph` per run and projects the result into `WorkflowOutput` / `WorkflowChunk`.
Fan-in follows Python's `wait_for_all`: CNF barriers by default, first-arrival (trigger) channels when a node
declares `wait_for_all = false`.
Everything about supersteps, channels, barriers, checkpoints, and Mermaid rendering belongs to
`openjiuwen-graph`; this module never reimplements it.
SVG rendering is delegated to `crate::render` and only driven from here, and rendering a definition that does
not compile is a typed error rather than a fabricated document.
Per-run topology mutation (`mutation.rs`) compiles a mutated *copy* of the recorded definition for one run.

## API Surface

### `definition.rs`

```rust
pub struct NodeConfig {
    pub inputs_schema: Option<Value>, pub outputs_schema: Option<Value>,
    pub stream_inputs_schema: Option<Value>, pub stream_outputs_schema: Option<Value>,
    pub abilities: Option<Vec<ComponentAbility>>,   // None = infer from topology
    pub wait_for_all: Option<bool>,                 // Some(false) + >=2 sources = first arrival
    pub max_retries: u32, pub timeout_ms: Option<u64>,
    pub exception_config: Option<ExceptionConfig>,  // default_outputs recovery
    pub name: Option<String>,
}
pub enum EdgeKind { Regular, Stream, Barrier }
pub struct Edge { pub sources: Vec<String>, pub target: String, pub kind: EdgeKind }
pub struct ConditionalSource { pub targets: Vec<String>, pub routers: Vec<Arc<dyn openjiuwen_graph::Router>> }
pub struct Entry { pub component: Arc<dyn Component>, pub node: NodeConfig }
pub struct WorkflowDefinition {
    pub card: WorkflowCard, pub config: WorkflowConfig,
    pub components: BTreeMap<String, Entry>,  // lookup; declaration order lives in `order`
    pub order: Vec<String>,
    pub edges: Vec<Edge>,                     // regular + stream + barrier, in insertion order
    pub conditional: BTreeMap<String, ConditionalSource>,
    pub start: Option<String>, pub end: Option<String>,
    pub problems: Vec<String>,                // recorded by the builder, reported by validate
}
```

`WorkflowDefinition` is `Clone` and `Send + Sync`; its `Debug` impl is hand-written (components and routers are
not `Debug`). Helpers: `new`, `with_config`, `insert_component(id, component, node)` (keeps `order`),
`component(id)`, `component_ids()`, `add_edge(sources, target, kind)`, `add_conditional(from, to, router)`,
`record_problem`, `spec() -> WorkflowSpec`, `supported_abilities(id) -> Vec<ComponentAbility>` (declared when
non-empty, else `Component::abilities()`), `selected_ability(id) -> ComponentAbility`.
Mutation helpers (implemented in `mutation.rs`): `component_mut(id) -> Option<&mut Entry>`,
`remove_component(id)` (drops the node, its edges, its conditional references, and a start/end choice naming it),
`remove_edges_to(target)`, and `retarget_edge(source, target, new_target) -> bool` (a miss is reported, not
silently ignored).

`spec()` builds the Python-compatible `WorkflowSpec`: `edges` / `stream_edges` from regular / stream edges
(one entry per source, insertion order), `comp_configs` (`NodeSpec` per node: `io_configs`,
`stream_io_configs`, `abilities`, `max_retries`, `timeout` in seconds — `timeout_ms / 1000.0`, `-1.0` when
unset, `exception_config`), `start_nodes = [start]`, and `stream_source_groups[target] = [[source]]` per
stream edge.

### Ability selection

A node runs exactly one ability. `selected_ability` picks, in order: `Transform`, `Collect`, `Stream`,
`Invoke` — the first one present in the node's declared abilities when they are set; otherwise inferred from
topology over `Component::abilities()`:

- start node (no incoming edge) → `Invoke`
- stream out + regular in → `Stream`
- stream out + stream in → `Transform`
- stream in + regular out → `Collect`
- regular in + regular out → `Invoke`
- otherwise the component's first supported ability, else `Invoke`.

### `validate.rs`

`pub fn validate(def: &WorkflowDefinition) -> Result<()>` collects every problem and returns ONE
`Error::Workflow` whose message joins the coded messages with `"; "` (the graph crate's policy). Each problem
is produced by the matching `error::*` builder so the Python code prefix survives. Rules:

- builder-recorded `def.problems` are included verbatim.
- exactly one start and one end; both registered; they are distinct; the end has no outgoing edge of any kind.
- the start component supports `Invoke`.
- every component id passes `card::validate_component_id`.
- every edge, stream edge, conditional source and conditional target names a registered component; conditional
  sources have at least one router.
- stream edges: source supports `Stream` or `Transform`, target supports `Collect` or `Transform` (100015 /
  100011).
- explicit `wait_for_all == Some(false)` on a node whose selected ability is `Collect`/`Transform` (100011).
- a declared ability the component does not support (100011).
- `inputs_schema` and `stream_inputs_schema` (also `outputs_schema` / `stream_outputs_schema`) sharing a
  flattened key (100014).

### `node.rs`

```rust
pub(crate) struct RunScope {
    pub interactions: Arc<InteractionState>, pub writer: Arc<FrameWriter>,
    pub depth: usize,                       // nesting level of this run
    /* failures: per-node failed-attempt counters */
}
pub(crate) struct NodeRuntime {
    pub id: String, pub component: Arc<dyn Component>, pub node: NodeConfig,
    pub ability: ComponentAbility, pub is_end: bool,
    pub workflow_id: String,
    pub scope: Option<Arc<RunScope>>,   // None only when compiling for `to_mermaid`/`to_svg`
}
pub(crate) struct ComponentNode { /* Arc<NodeRuntime> */ }
impl Node for ComponentNode { /* run(...) */ }
```

`run` semantics:

1. `source = state.as_value()`; the root seed always carries a reserved `"inputs"` key with the run's original
   inputs, so both `${query}` (root) and `${start.query}` resolve.
2. `inputs = resolve_schema(inputs_schema, &source)` when a schema exists, else the whole source object; for
   `Collect` / `Transform`, resolve `stream_inputs_schema` too and merge the objects (`stream` keys win).
3. `ComponentContext::new(session, id, workflow_id)` + `.with_interactions(scope.interactions)` +
   `.with_writer(scope.writer)` + `.with_cancellation(...)` + `.with_depth(scope.depth)`; `run` errors with
   `Error::workflow` when `scope` is `None` (only the diagram compiles omit it).
4. trace, when `session.tracer()` is `Some`: `start_workflow_span(id, name)`, `set_inputs(inputs)`, then
   `finish(output)` / `interrupted(reason)` / `fail(error)`.
5. dispatch: `component.route(&source)` first — `Ok(Some(targets))` returns `NodeOutcome::Route { targets }`
   without invoking; `Ok(None)` falls through. `Invoke` → `invoke`; `Stream` / `Transform` → drain the
   component stream, collecting frames; `Collect` → `collect`.
6. stream/transform frames are stored under the node id as a JSON array. When the node is the workflow end,
   each frame is also written as an `end node stream` output frame with a 0-based index, and the stored array
   holds `{"type": "end node stream", "index", "payload"}` wires so `invoke` can return them verbatim.
7. `outputs_schema` projects the raw output against `state + {"<id>": raw, "output": raw}`;
   `state.set(id, output)`.
8. an error carrying `find_interrupt` keeps the tracing `interrupted(reason)` marker and is propagated
   unchanged (the interaction frame itself was already written by `ComponentContext::interact`), so the graph's
   *failure* path stores the pre-superstep boundary and `resume` re-runs the interrupted superstep.
9. otherwise the failed attempt is counted; after `max_retries` are exhausted an `Invoke`/`Collect` node whose
   `exception_config` has `default_outputs` under a `handle_type` other than `"interrupt"` recovers: the
   defaults are projected through `outputs_schema`, written under the node id, the span records
   `error_recovery`, and the run continues.
10. any other error becomes `error::component_execution_error(component_type, ability, reason, workflow_id)`
    (`[100054]`); streaming failures and `handle_type == "interrupt"` always take this path.

### `mod.rs`

```rust
#[derive(Clone)] pub struct CompiledWorkflow { /* Arc<Inner> */ }
impl CompiledWorkflow {
    pub fn build(def: WorkflowDefinition) -> Result<Self>;
    pub fn card(&self) -> &WorkflowCard;  pub fn config(&self) -> &WorkflowConfig;
    pub fn spec(&self) -> &WorkflowSpec;
    pub fn start_comp_id(&self) -> &str;  pub fn end_comp_id(&self) -> &str;
    pub fn ability(&self, id: &str) -> Option<ComponentAbility>;
    pub fn is_streaming(&self) -> bool;
    pub fn to_mermaid(&self) -> Result<String>;   // compile error when the graph cannot be built
    pub fn to_svg(&self) -> Result<String>;       // same; no error document is fabricated
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
}
pub struct WorkflowRunOptions { pub cancellation, pub timeout: Option<Duration>, pub interactive: Option<InteractiveInput>,
    pub answers: BTreeMap<String, Vec<Value>> /* per-id FIFO answers replayed before `interactive` */,
    pub resume: bool, pub stream_modes: Option<Vec<StreamMode>>, pub max_supersteps: Option<usize>,
    pub depth: usize /* 0 at top level, child run = parent + 1 */,
    pub stream_delivery: StreamDelivery /* Buffered (default) | Interleaved (typed Unsupported) */ }
pub enum StreamDelivery { Buffered, Interleaved }   // wire "buffered" | "interleaved"; `as_str`/`Display`
```

`build` validates, computes `spec`, derives `is_streaming` from the end node's selected ability, and stores the
definition behind an `Arc`. The graph is compiled per run (`GraphBuilder::new(card.id)`, one `node_with` per
component in declaration order, `edge` / `barrier_edge` / `conditional_edges_to`, `entry`, `exit`,
`metadata("workflow_id")`, `metadata("end_comp_id")`) because node instances carry per-run interaction state
and frame writers. Conditional sources get one `CombinedRouter` that runs its routers in registration order,
unions and de-duplicates targets, and rejects a target outside the declared set with `Error::graph`.

First-arrival fan-in: with a `scope` (a real run), a node whose `NodeConfig::wait_for_all` is `Some(false)` and
that has at least two distinct regular/stream/barrier predecessors gets no static channel group. Each such
source is wired with `GraphBuilder::conditional_edges(source, router→[join])`, which contributes no channel
group, so `openjiuwen-graph` treats the join's channel as a trigger channel: it fires on any arrival and
re-fires on a later one. A source that also declares conditional edges into the join merges both decisions
into its one combined router (`conditional_edges`, target `"*"`), so the declared routers still choose. A
declared conditional target that is *not* first-arrival keeps its `conditional_edges_to` (declared-set
validation). Diagram compiles (`to_mermaid`/`to_svg`) skip the rewiring and keep the recorded static edges, so
every predecessor is visible. Single-source nodes, nodes reached only through a declared conditional edge, and
`Collect`/`Transform` nodes stay on CNF barriers.

Run wiring: when `session.checkpointer()` is `Some`, compile with
`CompileOptions { store: Some(Arc::new(CheckpointerStore::new(cp, card.id))), namespace: Some(card.id), checkpoint_every_superstep: true }`
and delete the namespace after a successful completed run. `invoke` merges object inputs into the session
global lane (unless resuming), seeds `InteractionState` from `options.answers` (`append`, so a nested run replays what its child already consumed) before `options.interactive` (`seed`), and runs inside a timeout
(`options.timeout`, else `SessionConfig::execute_timeout_ms` when non-zero); `stream` performs the same merge and two-step seeding but applies no timeout wrapper. An interrupt error drains
`InteractionState::pending` into `{"type", "index", "payload"}` wires and reports `InputRequired`, keeping the
pre-superstep snapshot; a graph that halts via `NodeOutcome::Halt` reports the same state with its halt value;
success → the end output value, or the collected `end node stream` wires when the end streams. `stream` forwards
frames through an unbounded channel, filters by `stream_modes` when set, yields one `workflow_final` frame on
success when the end output is non-null, ends quietly after the interaction frames of an interrupted run, and
yields exactly one `Err` on any other failure; a guard aborts the run task when the stream is dropped.
`resume` requires a checkpointer, sets `resume = true` and seeds `interactive`; continuing from the failure
boundary re-runs the interrupted superstep, so the component's question is answered by the seeded answer
instead of halting again. The stream generator drops its `RunScope` right after compiling the graph (only the
graph's nodes keep a frame-sink clone) and before it starts reading frames; keeping the scope alive would keep
the sink open and stall the frame loop. `to_svg` renders through `crate::render::render_svg` (explicit end node, card name as title) and returns the
compile error when the definition cannot be compiled; `to_mermaid` does the same. Neither fabricates an SVG
document or a Mermaid comment, so a rendered failure can never be mistaken for a diagram. `crate::render::error_svg`
stays a public utility for callers that explicitly want an error document, and the engine no longer uses it.
`stream_delivery` is checked by `check_delivery` at the top of `run_blocking` (the funnel `invoke` and `resume`
share) and of `stream`, i.e. before any session mutation, interaction seeding, or graph compile:
`StreamDelivery::Buffered` is the documented composition model, while `StreamDelivery::Interleaved` (Python's live
`stream_actor`) is a typed `Error::Unsupported` naming that gap and pointing at `Buffered`, which `stream`
delivers as its single `Err` item.

## Constraints

- Only this crate's files change; `openjiuwen-graph`, `openjiuwen-session`, `openjiuwen-foundation` and
  `openjiuwen-retrieval` are consumed through their documented seams.
- No `unsafe`, no `todo!`/`unimplemented!`, no stdout logging, no panics on user input; files ≤ ~1000 lines.
- Locks are never held across `.await` (`SessionConfig` is read into a local before awaiting).
- Public streams and futures are `Send + 'static`; cancellation flows through `CancellationToken`.

## Deviations from Python

- Streaming is buffered: a `Stream` / `Transform` producer's frames are stored under its id rather than fed to
  live consumers, and end-node frames are replayed in template order — observable text and order match for
  finite streams, live interleaving does not. The gap is explicit: `StreamDelivery::Interleaved` is a typed
  `Error::Unsupported` on every run entry point, so a caller asking for Python's `stream_actor` behaviour is told
  what is missing instead of silently receiving composed frames.
- Error recovery is inline: Python only substitutes `default_outputs` through a registered
  `component_error_recovery` callback, while `ComponentNode::run` applies the declared `ExceptionConfig`
  directly once the graph's retries are exhausted.
- A first-arrival node whose only fan-in is a declared conditional edge stays CNF: the graph layer's
  group-less trigger channel needs a dynamic edge per predecessor, so no static edge may target the join.
- Diagrams render the recorded topology, not the first-arrival rewiring; `to_mermaid`/`to_svg` return the compile
  error for a definition that cannot be compiled (no error document, no placeholder).
- Interaction indices and pending frames are tracked by `InteractionState`, not a session broker.
- LLM `response_format` shaping and `enable_history` insertion live in the component
  (`src/components/llm/`), not in the engine; the engine only resolves schemas and writes node state.

## Routing Table

- `./definition.rs` → `WorkflowDefinition`, `NodeConfig`, `Edge`, `ConditionalSource`, ability selection, `spec()`
- `./validate.rs` → compile-time validation and coded problem messages
- `./node.rs` → `RunScope`, `NodeRuntime`, `ComponentNode` graph adapter (schema resolution, dispatch, tracing, interrupt propagation, error recovery); crate-visible only (`pub(crate)`), and its `project_output` helper is reused by `src/single.rs`
- `./mod.rs` → `CompiledWorkflow`, `WorkflowRunOptions`, `StreamDelivery`, `CombinedRouter`, graph compilation (CNF + first-arrival wiring), rendering entry points, `test_support` fakes
- `./mutation.rs` → per-run mutation: `WorkflowDefinition::{component_mut, remove_component, remove_edges_to, retarget_edge}` and `CompiledWorkflow::{mutated, invoke_mutated, stream_mutated}`
- `./run.rs` → `CompiledWorkflow::{invoke, resume, stream}`, `run_blocking`, interaction-wire projection (`pub(crate) interaction_wires`, also used by `src/single.rs`), seed/timeout/depth helpers, `AbortOnDrop`
- `./tests.rs` → engine unit tests (stream termination, buffered frames, snapshot lifecycle, interrupt and failure resume, facade accessors, render-error propagation)

## Test Strategy

Validation is covered through `tests/builder_validation.rs`; run semantics through `tests/linear.rs`,
`references.rs`, `branch.rs`, `fan_in.rs`, `streaming.rs`, `end_template.rs`, `llm_component.rs`,
`llm_semantics.rs`, `tool_component.rs`, `retrieval_component.rs`, `loop_component.rs`, `http_component.rs`,
`questioner_component.rs`, `intent_detection.rs`, `react_agent.rs`, `sub_workflow.rs`, `exception_recovery.rs`,
`draw.rs`, `session_state.rs`, `checkpoint_resume.rs`, `per_run_mutation.rs`, and `card_spec.rs`.
`tests/single_component.rs` exercises the reused `project_output` / `interaction_wires` helpers outside a graph (input/output templates, interrupt wires, seeded answer).
The engine's own unit tests cover termination of a stream whose run emits no frames, buffered producer frames
versus end wires, snapshot lifecycle (deleted on completion, kept on interruption), resume semantics for both an
interrupted superstep and a plain node failure, and the compile-error propagation of `to_mermaid`/`to_svg`
(no SVG document, no Mermaid comment, no `"could not be compiled"` placeholder).
`tests/streaming.rs` pins the `StreamDelivery` contract (default `Buffered`, wire names, `Interleaved` rejected by
`invoke`/`stream`/`resume` with no session mutation first), and `tests/per_run_mutation.rs` pins per-run mutation
(end template swap, node insert/remove with edge retargeting, invalid mutations reporting `[100100]`, refusal of
`Interleaved` on a mutated run, mutated frames from `stream_mutated`, and the original staying runnable).
