# openjiuwen-workflow/src/engine

## Intent

The compile-and-run side of a workflow: `WorkflowDefinition` is what the builder records, `validate` turns a
definition into one compile error listing every problem, `ComponentNode` adapts a `Component` to an
`openjiuwen_graph::Node`, and `CompiledWorkflow` maps a validated definition onto a fresh
`openjiuwen_graph::CompiledGraph` per run and projects the result into `WorkflowOutput` / `WorkflowChunk`.
Everything about supersteps, channels, barriers, checkpoints, and Mermaid rendering belongs to
`openjiuwen-graph`; this module never reimplements it.

## API Surface

### `definition.rs`

```rust
pub struct NodeConfig {
    pub inputs_schema: Option<Value>, pub outputs_schema: Option<Value>,
    pub stream_inputs_schema: Option<Value>, pub stream_outputs_schema: Option<Value>,
    pub abilities: Option<Vec<ComponentAbility>>,   // None = infer from topology
    pub wait_for_all: Option<bool>,                 // accepted, cannot change CNF barriers
    pub max_retries: u32, pub timeout_ms: Option<u64>, pub name: Option<String>,
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

`spec()` builds the Python-compatible `WorkflowSpec`: `edges` / `stream_edges` from regular / stream edges
(one entry per source, insertion order), `comp_configs` (`NodeSpec` per node: `io_configs`,
`stream_io_configs`, `abilities`, `max_retries`, `timeout` in seconds — `timeout_ms / 1000.0`, `-1.0` when
unset), `start_nodes = [start]`, and `stream_source_groups[target] = [[source]]` per stream edge.

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
pub(crate) struct RunScope { pub interactions: Arc<InteractionState>, pub writer: Arc<FrameWriter> }
pub(crate) struct NodeRuntime {
    pub id: String, pub component: Arc<dyn Component>, pub node: NodeConfig,
    pub ability: ComponentAbility, pub is_end: bool,
    pub workflow_id: String,
    pub scope: Option<Arc<RunScope>>,   // None only when compiling for `to_mermaid` (never executed)
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
   `.with_writer(scope.writer)` + `.with_cancellation(ctx.cancellation().clone())`; `run` errors with
   `Error::workflow` when `scope` is `None` (only `to_mermaid` compiles without one).
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
9. any other error becomes `error::component_execution_error(component_type, ability, reason, workflow_id)`.

`wait_for_all` cannot change scheduler semantics: graph channels are CNF barriers, so every fan-in waits.

### `mod.rs`

```rust
#[derive(Clone)] pub struct CompiledWorkflow { /* Arc<Inner> */ }
impl CompiledWorkflow {
    pub fn build(def: WorkflowDefinition) -> Result<Self>;
    pub fn card(&self) -> &WorkflowCard;  pub fn config(&self) -> &WorkflowConfig;
    pub fn spec(&self) -> &WorkflowSpec;  pub fn end_comp_id(&self) -> &str;
    pub fn is_streaming(&self) -> bool;   pub fn to_mermaid(&self) -> String;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>, options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>, options: WorkflowRunOptions) -> Result<WorkflowOutput>;
}
pub struct WorkflowRunOptions { pub cancellation, pub timeout: Option<Duration>, pub interactive: Option<InteractiveInput>,
    pub resume: bool, pub stream_modes: Option<Vec<StreamMode>>, pub max_supersteps: Option<usize> }
```

`build` validates, computes `spec`, derives `is_streaming` from the end node's selected ability, and stores the
definition behind an `Arc`. The graph is compiled per run (`GraphBuilder::new(card.id)`, one `node_with` per
component in declaration order, `edge` / `barrier_edge` / `conditional_edges_to`, `entry`, `exit`,
`metadata("workflow_id")`, `metadata("end_comp_id")`) because node instances carry per-run interaction state
and frame writers. Conditional sources get one `CombinedRouter` that runs its routers in registration order,
unions and de-duplicates targets, and rejects a target outside the declared set with `Error::graph`.

Run wiring: when `session.checkpointer()` is `Some`, compile with
`CompileOptions { store: Some(Arc::new(CheckpointerStore::new(cp, card.id))), namespace: Some(card.id), checkpoint_every_superstep: true }`
and delete the namespace after a successful completed run. `invoke` merges object inputs into the session
global lane (unless resuming), seeds `InteractionState` from `options.interactive`, and runs inside a timeout
(`options.timeout`, else `SessionConfig::execute_timeout_ms` when non-zero). An interrupt error drains
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
the sink open and stall the frame loop.

## Constraints

- Only this crate's files change; `openjiuwen-graph`, `openjiuwen-session`, `openjiuwen-foundation` and
  `openjiuwen-retrieval` are consumed through their documented seams.
- No `unsafe`, no `todo!`/`unimplemented!`, no stdout logging, no panics on user input; files ≤ ~1000 lines.
- Locks are never held across `.await` (`SessionConfig` is read into a local before awaiting).
- Public streams and futures are `Send + 'static`; cancellation flows through `CancellationToken`.

## Deviations from Python

- Streaming is buffered: a `Stream` / `Transform` producer's frames are stored under its id rather than fed to
  live consumers, and end-node frames are replayed in template order — observable text and order match for
  finite streams, live interleaving does not.
- `wait_for_all=False` first-arrival fan-in has no equivalent (CNF barriers only), so alternative branch targets merging into one node wait for every predecessor; wire each alternative path separately.
- Interaction indices and pending frames are tracked by `InteractionState`, not a session broker.
- `response_format` injection and `enable_history` are carried in LLM config but not applied.

## Routing Table

- `./definition.rs` → `WorkflowDefinition`, `NodeConfig`, `Edge`, `ConditionalSource`, ability selection, `spec()`
- `./validate.rs` → compile-time validation and coded problem messages
- `./node.rs` → `RunScope`, `NodeRuntime`, `ComponentNode` graph adapter (schema resolution, dispatch, tracing, interrupt propagation); crate-visible only (`pub(crate)`)
- `./mod.rs` → `CompiledWorkflow`, `WorkflowRunOptions`, `CombinedRouter`, graph compilation, `test_support` fakes
- `./run.rs` → `CompiledWorkflow::{invoke, resume, stream}`, `run_blocking`, interaction-wire projection, seed/timeout helpers, `AbortOnDrop`
- `./tests.rs` → engine unit tests (stream termination, buffered frames, snapshot lifecycle, interrupt and failure resume, facade accessors)

## Test Strategy

Validation is covered through `tests/builder_validation.rs`; run semantics through `tests/linear.rs`,
`references.rs`, `branch.rs`, `streaming.rs`, `end_template.rs`, `llm_component.rs`, `tool_component.rs`,
`retrieval_component.rs`, `loop_component.rs`, `session_state.rs`, `checkpoint_resume.rs`, and `card_spec.rs`.
The engine's own unit tests cover termination of a stream whose run emits no frames, buffered producer frames
versus end wires, snapshot lifecycle (deleted on completion, kept on interruption), and resume semantics for
both an interrupted superstep and a plain node failure.
