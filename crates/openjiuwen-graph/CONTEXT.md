# openjiuwen-graph

## Intent

The Pregel/BSP execution engine that backs every workflow and agent graph in the kernel.
Ports `openjiuwen/core/graph/` (v0.1.20: `base.py`, `graph.py`, `vertex.py`, `executable.py`, `atomic_node.py`, `graph_state.py`, `pregel/`, `store/`, `stream_actor/`, `visualization/`) onto a deterministic, tokio-native executor.
`openjiuwen-workflow` builds on this crate: workflow components become [`Node`]s, workflow edges become graph edges, and the workflow's `${node.field}` resolution reads [`GraphState`].
Python's pickle-persisted `PregelGraph` internals are not reproduced; the observable contract (builder surface, superstep semantics, conditional routing, resume, Mermaid output) is.

## API Surface

Everything below is the stable seam; the executor internals in `src/pregel/` are crate-private.

### Builder (`src/builder.rs`)

```rust
pub struct GraphBuilder { /* … */ }

impl GraphBuilder {
    pub fn new(graph_id: impl Into<String>) -> Self;
    pub fn node(self, id: impl Into<String>, node: impl Node + 'static) -> Self;
    pub fn node_with(self, id: impl Into<String>, node: impl Node + 'static, options: NodeOptions) -> Self;
    pub fn edge(self, from: impl Into<String>, to: impl Into<String>) -> Self;
    pub fn barrier_edge(self, from: impl IntoIterator<Item = impl Into<String>>, to: impl Into<String>) -> Self;
    pub fn conditional_edges(self, from: impl Into<String>, router: impl Router + 'static) -> Self;
    pub fn conditional_edges_to(self, from: impl Into<String>, targets: impl IntoIterator<Item = impl Into<String>>, router: impl Router + 'static) -> Self;
    pub fn entry(self, id: impl Into<String>) -> Self;
    pub fn exit(self, id: impl Into<String>) -> Self;
    pub fn metadata(self, key: impl Into<String>, value: Value) -> Self;
    pub fn spec(&self) -> &GraphSpec;
    pub fn compile(self) -> Result<CompiledGraph>;
    pub fn compile_with(self, options: CompileOptions) -> Result<CompiledGraph>;
}

pub struct CompileOptions {
    pub store: Option<Arc<dyn GraphStore>>,
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    pub namespace: Option<String>,
    pub max_supersteps: Option<usize>,
    pub node_timeout_ms: Option<u64>,
    pub checkpoint_every_superstep: bool, // default true, a no-op without a store
}
```

Builder methods never fail; they record validation problems which `compile`/`compile_with` report as one [`Error::graph`] listing every problem (`; ` separated).
`spec()` exposes the partial spec for inspection and serialization.

### Graph state (`src/state.rs`)

```rust
#[derive(Serialize, Deserialize)] // #[serde(transparent)] over serde_json::Map
pub struct GraphState(pub Map<String, Value>);

impl GraphState {
    pub fn new() -> Self;
    pub fn from_value(value: Value) -> Result<Self>;      // object or null
    pub fn from_map(map: Map<String, Value>) -> Self;
    pub fn as_value(&self) -> Value;
    pub fn into_value(self) -> Value;
    pub fn get(&self, key: &str) -> Option<&Value>;
    pub fn get_path(&self, path: &str) -> Option<&Value>;  // dotted / indexed path
    pub fn set(&mut self, key: impl Into<String>, value: Value);
    pub fn set_path(&mut self, path: &str, value: Value) -> Result<()>;
    pub fn remove(&mut self, key: &str) -> Option<Value>;  // a node's top-level removal propagates to the run state
    pub fn contains_key(&self, key: &str) -> bool;
    pub fn keys(&self) -> impl Iterator<Item = &String>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn merge(&mut self, other: GraphState);            // deep merge, overlay wins
    pub fn merge_node_delta(&mut self, base: &GraphState, node_state: GraphState); // executor per-node delta
}
```

Path access and merging delegate to `openjiuwen_common::utils::json` (`get_path`, `set_path`, `merge`), so graph state has exactly the session state's path and merge semantics.
`merge_node_delta` is what the executor applies per node: changed top-level keys are deep-merged and top-level keys the node removed are deleted; only top-level keys are diffed, so nested removals do not propagate.

### Nodes (`src/node.rs`)

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeOptions { pub timeout_ms: Option<u64>, pub max_retries: u32 }
impl NodeOptions { pub fn timeout(&self) -> Option<Duration>; }

#[async_trait]
pub trait Node: Send + Sync {
    fn id(&self) -> &str { "" }                            // builder id wins
    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]    // {"continue"} | {"route": {"targets": [...]}} | {"halt": {"value": …}}
pub enum NodeOutcome { #[default] Continue, Route { targets: Vec<String> }, Halt { value: Option<Value> } }
impl NodeOutcome { pub fn route(targets: impl IntoIterator<Item = impl Into<String>>) -> Self; pub fn halt(value: Option<Value>) -> Self; }

pub struct GraphContext { /* session, graph_id, node_id, superstep, cancellation, metadata */ }
impl GraphContext {
    pub fn session(&self) -> &Arc<dyn Session>;
    pub fn graph_id(&self) -> &str;
    pub fn node_id(&self) -> &str;
    pub fn superstep(&self) -> u32;
    pub fn cancellation(&self) -> &CancellationToken;
    pub fn is_cancelled(&self) -> bool;
    pub async fn cancelled(&self);
    pub fn metadata(&self) -> &BTreeMap<String, Value>;
}

// closure adapters (boxed futures; see Design Decisions)
pub type BoxNodeFuture<'a> = Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>;
pub struct FnNode<F>;   // F: for<'a> Fn(&'a mut GraphState, &'a GraphContext) -> BoxNodeFuture<'a>
impl<F> FnNode<F> { pub fn new(id: impl Into<String>, f: F) -> Self; }
pub struct SyncNode<F>; // F: Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome> + Send + Sync
impl<F> SyncNode<F> { pub fn new(id: impl Into<String>, f: F) -> Self; }
```

`FnNode` callbacks return `Box::pin(async move { … })`; the boxed future is the stable-Rust spelling of the `AsyncFn` bound (see Design Decisions).
Sync nodes run inline on the executor task, so they must be cheap and non-blocking.

### Routers (`src/router.rs`)

```rust
pub struct RouterContext { source: String, state: GraphState, superstep: u32, session: Arc<dyn Session>, metadata: Arc<BTreeMap<String, Value>> }
impl RouterContext { pub fn source(&self) -> &str; pub fn state(&self) -> &GraphState; pub fn superstep(&self) -> u32; pub fn session(&self) -> &Arc<dyn Session>; pub fn metadata(&self) -> &BTreeMap<String, Value>; }

#[async_trait]
pub trait Router: Send + Sync {
    fn name(&self) -> &str;
    async fn route(&self, ctx: &RouterContext) -> Result<Vec<String>>;   // target node ids
}

pub type BoxRouterFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>>;
pub struct FnRouter<F>; // F: for<'a> Fn(&'a RouterContext) -> BoxRouterFuture<'a>
impl<F> FnRouter<F> { pub fn new(name: impl Into<String>, f: F) -> Self; }
```

A router returns the nodes to activate; an empty vector routes nowhere (the branch ends).
Returned ids must be known nodes, otherwise the run fails with [`Error::graph`].
Static edges are compiled into dedicated static routers; conditional routers are appended for their source, and every dispatch result is unioned.

### Spec (`src/spec.rs`)

```rust
#[derive(Serialize, Deserialize)] pub enum EdgeKind { Static, Barrier, Conditional }
#[derive(Serialize, Deserialize)] pub struct EdgeSpec { pub sources: Vec<String>, pub target: String, pub kind: EdgeKind }
#[derive(Serialize, Deserialize)] pub struct NodeSpec { pub id: String, pub kind: String, pub options: NodeOptions }
#[derive(Serialize, Deserialize)] pub struct GraphSpec {
    pub graph_id: String, pub entry: Vec<String>, pub exit: Vec<String>,
    pub nodes: Vec<NodeSpec>, pub edges: Vec<EdgeSpec>, pub metadata: BTreeMap<String, Value>,
}
impl GraphSpec { pub fn node(&self, id: &str) -> Option<&NodeSpec>; pub fn outgoing(&self, id: &str) -> Vec<&EdgeSpec>; pub fn validate(&self) -> Result<()>; }
pub const DYNAMIC_TARGET: &str = "*";   // target of a router whose targets are only known at run time
```

`GraphSpec` is the serializable description (`snake_case`, tolerates unknown fields, defaults to empty collections); node runnables live only in the compiled graph.
`NodeSpec::kind` is a free-form label (`"node"` by default, `"agent"`/`"llm"`/… for workflow components) used by visualization and spec consumers.

### Compiled graph (`src/compiled.rs`)

```rust
#[derive(Clone)] pub struct CompiledGraph { /* Arc<CompiledInner> */ }

impl CompiledGraph {
    pub fn id(&self) -> &str;
    pub fn spec(&self) -> &GraphSpec;
    pub fn node_ids(&self) -> impl Iterator<Item = &str>;
    pub fn entry_nodes(&self) -> &[String];
    pub fn exit_nodes(&self) -> Vec<&str>;
    pub async fn invoke(&self, input: Value, session: Arc<dyn Session>) -> Result<Value>;
    pub async fn run(&self, input: Value, session: Arc<dyn Session>, options: ExecutionOptions) -> Result<GraphRunResult>;
    pub fn stream(&self, input: Value, session: Arc<dyn Session>) -> BoxStream<'static, Result<GraphEvent>>;
    pub fn stream_with(&self, input: Value, session: Arc<dyn Session>, options: ExecutionOptions) -> BoxStream<'static, Result<GraphEvent>>;
    pub fn to_mermaid(&self) -> String;
    pub fn to_mermaid_with(&self, options: &MermaidOptions) -> String;
}

#[derive(Debug, Clone, Serialize, Deserialize)] pub struct GraphRunResult { pub graph_id: String, pub state: GraphState, pub supersteps: u32, pub visited: Vec<String>, pub halted: Option<HaltInfo>, pub resumed: bool }
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct HaltInfo { pub node_id: String, pub superstep: u32, pub value: Option<Value> }
```

`invoke` returns the final state as a JSON object (`GraphState::into_value`).
`stream` yields [`GraphEvent`]s in execution order and terminates with the run; a failed run yields exactly one `Err` after its `NodeFailed`/`GraphFailed` events.
`stream_with` runs the graph as a spawned task guarded by an `AbortOnDrop`: dropping the stream before it terminates aborts that task, so an abandoned stream cancels the run instead of leaking a background run.

### Execution options (`src/execution.rs`)

```rust
#[derive(Clone)] pub struct ExecutionOptions {
    pub max_supersteps: Option<usize>,          // None → compile option → 10_000
    pub node_timeout_ms: Option<u64>,           // None → compile option → per-node option → none
    pub cancellation: CancellationToken,        // default: never cancelled
    pub resume: bool,                           // load a stored snapshot and continue
    pub events: Option<UnboundedSender<GraphEvent>>,
    pub write_session_trace: bool,              // default true: mirror events as trace frames
}
```

### Events (`src/event.rs`)

```rust
#[derive(Serialize, Deserialize)] pub enum GraphEventKind {
    SuperstepStarted, NodeStarted, NodeCompleted, NodeFailed, EdgeTraversed, SuperstepCompleted, GraphCompleted, GraphHalted, GraphFailed,
}
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct GraphEvent { pub kind: GraphEventKind, pub node_id: Option<String>, pub superstep: u32, pub payload: Value }
impl GraphEvent { pub fn to_wire(&self) -> Value; }   // {"kind", "node_id"?, "superstep", "payload"} — payload is always an object
```

### Store and checkpoints (`src/store.rs`)

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelSnapshot { pub groups: Vec<Vec<String>>, pub fired: Vec<Vec<String>>, pub triggers: Vec<String> }
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct ChannelMessage { pub target: String, pub source: String }
#[derive(Debug, Clone, Serialize, Deserialize)] pub struct GraphSnapshot {
    pub graph_id: String, pub namespace: String, pub superstep: u32, pub state: GraphState,
    pub channels: BTreeMap<String, ChannelSnapshot>, pub buffer: Vec<ChannelMessage>,
}
impl GraphSnapshot { pub fn is_resumable(&self) -> bool; }

#[async_trait]
pub trait GraphStore: Send + Sync + std::fmt::Debug {
    async fn get(&self, session_id: &str, namespace: &str) -> Result<Option<GraphSnapshot>>;
    async fn save(&self, session_id: &str, namespace: &str, snapshot: GraphSnapshot) -> Result<()>;
    async fn delete(&self, session_id: &str, namespace: Option<&str>) -> Result<()>;
}

pub struct InMemoryGraphStore;   // lock-guarded map, clone-on-read
pub struct CheckpointerStore;    // Arc<dyn Checkpointer> adapter, namespace doubles as checkpoint id
impl CheckpointerStore { pub fn new(checkpointer: Arc<dyn Checkpointer>, entity_id: impl Into<String>) -> Self; }
```

`CheckpointerStore` persists each snapshot as a `Checkpoint` in [`CheckpointNamespace::WorkflowGraph`] with `checkpoint_id = namespace`, `entity_id` from the constructor, and the snapshot JSON in `global_state`; `get` loads by `(session_id, namespace)`; `delete(None)` releases the session's snapshots.

### Visualization (`src/visualize.rs`)

```rust
pub struct MermaidOptions { pub title: Option<String>, pub direction: MermaidDirection }
pub enum MermaidDirection { TopDown, LeftRight }
impl MermaidDirection { pub fn as_str(&self) -> &'static str; }
pub fn render_mermaid(spec: &GraphSpec, options: &MermaidOptions) -> String;
```

Python emits Mermaid only (`visualization/Drawable.to_mermaid`); DOT is intentionally omitted and recorded as a deviation.

## Execution semantics

- **Supersteps.** A run executes whole supersteps. Each superstep: compute ready nodes (declaration order), `consume` their channels, run the nodes concurrently, merge results, dispatch routing, `flush` buffered messages, checkpoint.
- **Channels.** Every node owns a CNF barrier channel: `groups: Vec<Set<source>>`, ready when every group contains at least one source that fired. Groups come from incoming edges: a single-source edge contributes `{source}`; a barrier edge `{A,B} → T` contributes `{A}` and `{B}`; a conditional edge contributes `{router source}`. A node with no incoming channel is a trigger channel: ready once any source fires it (entry seeding uses the reserved source `__start__`).
- **Entry.** Explicit `entry(…)` ids are seeded with `__start__` at superstep 1; without explicit entries, nodes with no incoming edges are the entries. A graph with no entry is a compile error.
- **Accept-then-activate barrier.** Routing messages produced in a superstep become visible only in the next superstep (`flush` after all nodes of the superstep complete).
- **Determinism.** Nodes that are ready are executed concurrently (`JoinSet`) but their outcomes are applied in declaration order; state merges, router dispatch, and emitted events therefore follow a fixed order regardless of completion order. Within a superstep, node state is applied as a delta against the pre-superstep state (`GraphState::merge_node_delta`): changed top-level keys are deep-merged in declaration order (later declaration wins on scalar conflicts) and a node's top-level key removals propagate; only top-level keys are diffed, so nested removals do not propagate.
- **Routing.** A node's targets are: `NodeOutcome::Route { targets }` when non-empty (replacing routers), otherwise the union of static-edge targets and every conditional router's result, in declaration order and de-duplicated.
- **Halt.** `NodeOutcome::Halt { value }` ends the run after the current superstep (siblings already running complete, matching Python's interrupt-as-data behavior). The result carries `halted: Some(HaltInfo)`.
- **Limits and failures.** At most `max_supersteps` supersteps (default `10_000`, Python `MAX_RECURSIVE_LIMIT`) — exceeding it is [`Error::graph`]. Node errors propagate the original [`Error`] with the node id and superstep added as context; timeouts are [`ErrorKind::Timeout`]; cancellation is [`ErrorKind::Cancelled`]. Node `max_retries` re-runs an errored node (`max_retries + 1` attempts), never a halt and never a cancellation.
- **Cancellation.** `ExecutionOptions::cancellation` is checked before every superstep and raced against every node attempt; cancelling drops in-flight node futures.
- **Checkpoints.** With a store configured, a snapshot (state + channel fired sets) is saved after every completed superstep (when `checkpoint_every_superstep`), and once more at the last completed boundary when a run fails. `ExecutionOptions::resume` restores the stored snapshot and continues; the input value is ignored on a resume. Resume re-runs the superstep that failed (at-least-once per superstep).
- **Streaming.** Events are delivered through `ExecutionOptions::events` and, when `write_session_trace` is set, mirrored to `Session::write_stream` as `StreamItem::trace("graph_event", …)` frames (payload is always an object, never `null`, because the stream hub rejects null payloads). Multi-subscriber delivery is the session hub's job; the `stream*` methods additionally expose an in-process ordered stream for the caller.
- **Session.** The engine never mutates session state itself; nodes read/write `Session` for their own component state. `GraphState` is engine-owned and returned to the caller.

## Constraints

- `#![forbid(unsafe_code)]`, async-trait, serde; errors are `openjiuwen_common::error::{Error, Result}` (`Error::graph` for graph-specific failures); logging via `tracing` only.
- Files ≤1000 lines; the executor internals stay crate-private so the public surface stays small.
- No panics on user input: every builder/compile problem and every runtime anomaly becomes an `Error`.
- Tests are offline and deterministic: no sleeps for ordering (assert on event order), no network.
- Only this crate's files and `Cargo.toml` may change; the session/common APIs are consumed as-is.

## Design Decisions

- **State is functional per node.** Parallel nodes cannot share `&mut GraphState`, so each node receives a clone and the engine applies the result in declaration order as a delta against the pre-superstep state. Python passed `state=None` and discarded node return values, so this is an extension, not a compatibility break.
- **Node state is merged as a top-level delta.** A node's returned state is compared with the state it was given: changed top-level keys are deep-merged and top-level keys the node dropped are removed, so `GraphState::remove` works from a node. Keys added by an earlier node in the same superstep survive because a later node never saw them; nested removals do not propagate (only top-level keys are diffed).
- **Python's router-shadowing quirks are not reproduced.** Python keys routers by callable `__name__` and silently overwrites same-named routers; Rust stores every router in a `Vec`, so same-named routers all run. Conditional targets are declared explicitly (`conditional_edges_to`) instead of Python's `register_branch_targets` side channel.
- **Cross-branch CNF hoisting is omitted.** Python's `_resolve_barrier_groups` merges groups reachable from one branch into an OR group; the Rust engine uses the literal per-edge groups. This is simpler and predictable; branch fan-in still needs one explicit edge group per incoming path.
- **Validation is stricter than Python.** Python defers unknown endpoints to a `KeyError` at compile time and accepts duplicate node ids silently in places; Rust reports unknown endpoints, duplicate ids, unknown router sources, and a missing entry as one compile error.
- **Recursion limit is strict.** Python allows `limit + 1` steps (`step > max_step`); Rust allows exactly `max_supersteps` and then fails, which makes the guard predictable.
- **Snapshots are JSON, not pickles.** `GraphSnapshot` is a serde model, so `InMemoryGraphStore`, `CheckpointerStore`, and any external backend share one format.
- **Python's `stream_actor` graph-streaming machinery is not ported.** It exists to feed `${...}` stream references into nodes; in this rewrite workflow components consume session streams directly and the graph emits ordered `GraphEvent`s plus session trace frames.
- **Mermaid only.** DOT/PNG/SVG (network round-trip in Python) are omitted.
- **Closure adapters use boxed futures.** A `for<'a> AsyncFn(&'a …) -> …` bound cannot prove the returned future is `Send` on stable Rust (the `async_fn_traits` workaround is nightly-only), so `FnNode` and `FnRouter` take `Fn` callbacks returning [`node::BoxNodeFuture`] / [`router::BoxRouterFuture`]; call sites write `Box::pin(async move { … })`.
- **Dynamic conditional edges name no target.** `conditional_edges` records an edge whose target is [`spec::DYNAMIC_TARGET`] (`*`); it contributes no channel group, and nodes reachable only through such a router still look like implicit entries, so they need an explicit `entry(…)` (or `conditional_edges_to`, which declares the targets for introspection and visualization).
- **Entry seeding is unconditional.** `START_NODE` bypasses a node's CNF groups, so a declared entry runs in superstep 1 even when it also has incoming edges (required for loops).

## Routing Table

- `./src/lib.rs` → crate docs, module declarations, top-level re-exports
- `./src/state.rs` → `GraphState`, merge and path helpers
- `./src/node.rs` → `Node` trait, `NodeOutcome`, `NodeOptions`, `GraphContext`, `FnNode`, `SyncNode`
- `./src/router.rs` → `Router` trait, `RouterContext`, `FnRouter`
- `./src/spec.rs` → `GraphSpec`/`NodeSpec`/`EdgeSpec`/`EdgeKind`, spec validation and serde
- `./src/builder.rs` → `GraphBuilder`, `CompileOptions`, compile-time validation
- `./src/compiled.rs` → `CompiledGraph` facade (`invoke`/`run`/`stream`), `GraphRunResult`, `HaltInfo`
- `./src/execution.rs` → `ExecutionOptions`
- `./src/event.rs` → `GraphEvent`, `GraphEventKind`, session trace projection
- `./src/store.rs` → `GraphStore`, `InMemoryGraphStore`, `CheckpointerStore`, `GraphSnapshot`
- `./src/visualize.rs` → Mermaid rendering
- `./src/pregel/` → crate-private executor: `channel.rs` (CNF channels, buffer/flush/snapshot), `executor.rs` (superstep loop, scheduling, retries, timeouts, cancellation, checkpointing), `dispatch.rs` (router compilation and target resolution)
- `./tests/` → cross-module behavior tests (see below)

## Test Strategy

Unit tests live beside the modules (`state`, `node`, `router`, `spec`, `store`, `visualize`, `pregel::channel`, `pregel::dispatch`).
Integration tests in `./tests/` use `openjiuwen_session::WorkflowSession` as `Arc<dyn Session>` and cover:

- `common/mod.rs` — shared helpers (`session()`, counting/marker/noop nodes).

- `linear.rs` — linear chain, state seeded from input, final value.
- `fan_out_fan_in.rs` — parallel fan-out, barrier fan-in, declaration-order merge determinism, top-level key removal propagation (and re-add by a later node), nested-removal non-propagation, visited order.
- `conditional.rs` — router branching on state, multiple conditionals, routing to a subset, empty route.
- `cycles.rs` — bounded loop with a counter, max-superstep failure.
- `halt.rs` — halt ends the run, siblings complete, `HaltInfo` preserved.
- `errors.rs` — node error carries node id/superstep, retry counts (`max_retries + 1` attempts), timeout is `ErrorKind::Timeout`.
- `cancel.rs` — cancellation before and during a superstep, `ErrorKind::Cancelled`.
- `streaming.rs` — event order (`SuperstepStarted` → `NodeStarted`/`NodeCompleted` → `EdgeTraversed` → … → `GraphCompleted`), superstep numbers, session trace frames observed through `Session::subscribe`/`StreamHub`, and abort-on-drop (an abandoned stream stops the run).
- `checkpoint.rs` — snapshot saved per superstep, failing run resumes without repeating completed supersteps, `InMemoryGraphStore` and `CheckpointerStore` round-trips.
- `spec_serde.rs` — `GraphSpec` JSON round-trip, unknown fields tolerated, defaults.
- `mermaid.rs` — flowchart header, node shapes, conditional edge labels, entry/exit markers.
- `validation.rs` — duplicate id, unknown edge endpoint, unknown router source, missing entry, `NodeOutcome::Route` to an unknown node.

Behavior ports cite the Python test file they mirror, e.g. `// port of tests/unit_tests/core/graph/test_pregel.py`.
`cargo fmt -p openjiuwen-graph`, `cargo test -p openjiuwen-graph`, and `cargo clippy -p openjiuwen-graph --all-targets -- -D warnings` must pass offline.
