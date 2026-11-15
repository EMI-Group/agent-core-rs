# Architecture

This document records the cross-cutting contracts of the Rust rewrite of the openJiuwen Python SDK (v0.1.20).
Per-crate details live in each crate's `CONTEXT.md`; only seams that more than one crate depends on appear here.
Every `rust` excerpt below is a condensation of a current signature — changing one requires updating every dependent crate.

## 1. Crate layering

27 crates sit in layers 0–7, plus three development crates that no library crate depends on, for 30 in total.
Dependencies point downward only, apart from the exceptions listed after the table.

| Layer | Crates | Notes |
|---|---|---|
| 0 | `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-harness-protocol` | No workspace dependencies: macros is a proc-macro crate, the protocol crate is a dependency-free SPI. |
| 1 | `openjiuwen-foundation`, `openjiuwen-session`, `openjiuwen-graph`, `openjiuwen-retrieval`, `openjiuwen-sysop` | Foundation depends on common + macros only; session additionally uses foundation; graph uses session; retrieval and sysop use common. |
| 2 | `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-memory`, `openjiuwen-security` | Workflow composes foundation, session, graph and retrieval; context uses foundation + session; memory uses foundation, retrieval and sysop but not context or session; security uses foundation. |
| 3 | `openjiuwen-agent`, `openjiuwen-multiagent` | Agent composes foundation, session, workflow, context, memory, security and sysop and hosts the `controller`, `kv_cache`, `operator`, `evolve` and `prompt_pack` extension modules; multiagent adds agent. |
| 4 | `openjiuwen-runner`, `openjiuwen-core` | Runner orchestrates the layer 0–3 crates (workflow, agent, multiagent, security, sysop, memory); core re-exports layers 0–4 and defines nothing itself. |
| 5 | `openjiuwen-harness`, `openjiuwen-extensions`, `openjiuwen-dev-tools`, `openjiuwen-symphony` | All four depend on `openjiuwen-core`; harness also drives the kernel crates directly, dev-tools adds agent, symphony adds graph/retrieval/session. |
| 6 | `openjiuwen-harness-providers`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-agent-teams` | Providers/teams sit on harness; rsi and agent-evolving sit on harness + core; auto-harness depends only on rsi and re-exports it. |
| 7 | `openjiuwen`, `openjiuwen-cli` | Umbrella over every library crate plus a curated `prelude`; the binary depends on the umbrella, `openjiuwen-harness` (feature `notify`), `openjiuwen-harness-protocol`, `openjiuwen-core`, `openjiuwen-common` and `openjiuwen-rsi`, and reaches `harness-providers` only transitively through the umbrella. |
| dev | `openjiuwen-test-support`, `openjiuwen-e2e`, `openjiuwen-examples` | `publish = false`, never dependencies of library crates. |

`openjiuwen-core` is 17 modules of pure aliases (`common`, `foundation`, `session`, `graph`, `workflow`, `context_engine`, `memory`, `retrieval`, `sys_operation`, `security`, `single_agent`, `application`, `controller`, `kv_cache`, `operator`, `multi_agent`, `runner`) and defines nothing.
`openjiuwen-macros` is the only kernel-adjacent crate it does not alias.

Rules:
- `openjiuwen-core` and `openjiuwen` are umbrellas; concrete crates depend on the concrete crate, never on an umbrella, except layer 5 and above.
- `openjiuwen-macros` stays unaliased because it is compile-time only; no runtime path exists for it.
- If two crates need the same type it belongs in the lowest common ancestor (`openjiuwen-common` or `openjiuwen-foundation`).
- Python's function-local `core → harness/extensions/agent_teams` imports become traits implemented in the upper crate and registered at startup; cycles are never introduced to mirror them.
- Four edges break strict downward layering: the same-layer `openjiuwen-harness-providers/native-v2` → `openjiuwen-agent-teams` (layer 6) and `openjiuwen-cli` → `openjiuwen` (layer 7), plus the upward `openjiuwen-agent-teams/otlp` → `openjiuwen-extensions` and `openjiuwen-agent-evolving/redis` → `openjiuwen-extensions`.
- The three feature-gated edges are opt-in and off by default; the `openjiuwen-cli` → `openjiuwen` edge is unconditional.

## 2. Shared contracts

These signatures are the public seams.

### 2.1 Cards, configs, and errors (`openjiuwen-common`)

```rust
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Serialized identity + metadata; crosses process boundaries.
pub trait Card: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
}

/// Common card fields reused by all concrete cards.
pub struct CardBase { pub id: String, pub name: String, pub description: Option<String>,
                      pub version: Option<String>, pub metadata: BTreeMap<String, Value> }

pub enum Error { Config(String), Validation(String), Llm(String), Tool(String), Workflow(String),
                 Graph(String), Session(String), Memory(String), Retrieval(String), SysOp(String),
                 Security(String), Mcp(String), Transport(String), Timeout(String),
                 NotFound(String), Unsupported(String), Cancelled, External(Box<dyn Error + Send + Sync>) }
```

`ErrorKind` mirrors those variants so callers can match without owning a message; the Python `Card`/`Config` split is preserved (`*Card` is serializable and registry-safe, `*Config` is process-local, may hold clients or credentials, and is passed to `configure`/`set_config`).
All serialized structs use `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default missing collections to empty.

### 2.2 Messages, prompts, and models (`openjiuwen-foundation`)

```rust
pub struct ModelClientConfig { pub client_id: String, pub provider: String,
                               pub api_key: Option<SecretString>, pub api_base: Option<String>,
                               pub verify_ssl: bool, pub timeout: Option<Duration>,
                               pub stream_first_chunk_timeout: Option<Duration>,
                               pub stream_idle_timeout: Option<Duration>, pub max_retries: u32,
                               pub headers: BTreeMap<String, String>, pub api_mode: Option<ApiMode>,
                               pub auth_mode: AuthMode, pub endpoint_profile: Option<String>,
                               pub extra: Value }

pub struct ModelRequestConfig { pub model_name: String, pub temperature: Option<f32>,
                                pub top_p: Option<f32>, pub max_tokens: Option<u32>,
                                pub stop: Vec<String>, pub context_window: Option<u32>,
                                pub reasoning: Option<Value>, pub purpose: Option<String>,
                                pub metadata: BTreeMap<String, Value>, pub tools: Option<Value>,
                                pub tool_choice: Option<Value>, pub stream: bool, pub extra: Value }

#[async_trait]
pub trait Model: Send + Sync {
    fn provider(&self) -> &str;
    fn model_name(&self) -> &str;
    async fn invoke(&self, messages: &[BaseMessage], request: &ModelRequestConfig) -> Result<ChatResponse>;
    fn stream(&self, messages: Vec<BaseMessage>, request: ModelRequestConfig)
        -> BoxStream<'static, Result<StreamChunk>>;
}

pub async fn init_model(client: ModelClientConfig, request: ModelRequestConfig) -> Result<Arc<dyn Model>>;
```

Messages live in `prompt` (`BaseMessage`, `MessageContent::{Text, Parts}`, `Role`, `ToolCall`), and `llm` re-exports them.
`StreamChunk` is `Delta { index, role, content, reasoning, tool_call, finish_reason }`, `Usage(Usage)` or `Done`.

Three transports are implemented, one per submodule: `llm::openai` (every OpenAI-compatible vendor through `api_base`), `llm::anthropic` (Messages API) and `llm::responses` (`OpenAiResponsesClient`, which drives the SSE parser in both modes because the endpoint has no non-streaming shape).
`ModelRequestConfig::purpose` and `::metadata` are caller-owned correlation fields: no transport reads them, they never enter a request body, and the builders strip the legacy `extra` spellings via `extra_without_internal_keys()`.
Sibling modules carry reasoning profiles (`llm::reasoning::resolve_reasoning_plan`), routing and failover (`llm::routing::RoutingModel`, `compile_route`, `is_retryable_failure`), OAuth device-code login with `TokenStore`/`InMemoryTokenStore`/`FileTokenStore` (`llm::oauth`), the `ExternalProviderRegistry` with the OpenAI-account catalogue and its disk cache (`llm::external_provider`), and DashScope media generation (`llm::dashscope`).
Providers self-register in `ModelProviderRegistry::global()` by implementing `ModelProvider` (`name`, `aliases`, `default_api_base`, `create`); an unknown provider name is `Error::Unsupported`.

### 2.3 Tools (`openjiuwen-foundation`)

```rust
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn card(&self) -> &ToolCard;
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value>;
    fn stream(self: Arc<Self>, inputs: Value, ctx: ToolContext)
        -> BoxStream<'static, Result<Value>>;             // default: one invoke
    fn render_for_llm(&self, output: &Value) -> String;   // default
    fn is_parallel_safe(&self) -> bool;                   // default: card().parallel_safe
}

pub fn new(card: ToolCard, handler: ToolHandler) -> FunctionTool;   // FunctionTool::new
pub fn builder() -> FunctionToolBuilder;                            // FunctionTool::builder
```

`ToolCard` is flat: `id`, `name`, `description`, `input_params` (JSON Schema), `properties`, `exposure`, `execution`, `parallel_safe`, `stateless`, `idempotent`, `version`, `metadata`.
`FunctionTool::new` takes a pre-built card plus a boxed async handler; `FunctionToolBuilder` derives the schema (`input_schema_for::<T: JsonSchema>()`) and accepts `handler`/`typed_handler`.
The `#[tool]` attribute macro in `openjiuwen-macros` emits `FunctionTool::new(card, handler)`, not the builder.
`ToolInfo` carries the nested `{type, function}` wire shape and converts through `flat()`, `from_flat()`, `openai()` and `from_openai()`, with the batch helpers `flat_tools`/`from_wire_tools`.
Other implementations are `RestfulApiTool` and `McpTool` over stdio, SSE, streamable-HTTP and OpenAPI transports; the Playwright transport is a typed `Error::Unsupported`.
`ClientTool` lives in `openjiuwen-agent`, next to the registry that executes it.

### 2.4 Sessions (`openjiuwen-session`)

```rust
#[async_trait]
pub trait Session: Send + Sync {
    fn id(&self) -> &str;
    fn parent_id(&self) -> Option<&str>;
    fn session_type(&self) -> SessionType;
    async fn get_state(&self, key: &str) -> Result<Option<Value>>;
    async fn update_state(&self, patch: Value) -> Result<()>;
    async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>>;
    async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()>;
    async fn write_stream(&self, item: StreamItem) -> Result<()>;
    async fn request_interaction(&self, request: InteractionRequest) -> Result<InteractionResponse>;
    fn trace(&self) -> &dyn TraceSink;
    // defaulted seams: install_trace_sink, metadata, checkpointer, tracer, config, writers,
    // subscribe, resolve_interaction, end_stream, close, snapshot, restore, save_checkpoint, resume
}

#[async_trait]
pub trait Checkpointer: Send + Sync + std::fmt::Debug {
    async fn save(&self, checkpoint: Checkpoint) -> Result<String>;
    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>>;
    async fn latest(&self, session_id: &str) -> Result<Option<Checkpoint>>;   // defaulted
    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>>;
    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()>;
    async fn release(&self, session_id: &str) -> Result<()>;
    async fn session_exists(&self, session_id: &str) -> Result<bool>;
}
```

`WorkflowState` (Python's `CommitState`) stages writes in per-node lanes via `update`, `update_global`, `update_workflow` and `update_component`, then either `commit(node_id: Option<&str>) -> Result<()>` or `rollback(node_id: Option<&str>)`; `AgentState` writes directly with no staging.
`StreamHub` is the bounded multi-subscriber transport for `StreamItem` frames: `send_now` is synchronous, a lagging subscriber loses its oldest frames with a warning, and a hub with no subscribers accepts and drops frames.
`InteractionBroker` pairs `InteractionRequest`s with the responses that answer them (`request`, `resolve`, `seed`, `pending`, `cancel_all`).
`Tracer` emits `TraceFrame`s into a `TraceSink` (`NoopTraceSink`, `MemoryTraceSink`, `StreamTraceSink`); `SwappableTraceSink` is the indirection that lets a host install a real sink after construction, and `install_trace_sink` is the `otel`-gated helper over it.
Checkpointer backends are in-memory, filesystem JSON, SQLite (`sqlite`) and Redis RESP2 — the RESP2 client lives in-crate (`checkpoint::resp`) and is always compiled, and `create_checkpointer(&CheckpointerConfig)` selects one by `CheckpointerKind`.
Concrete sessions are `WorkflowSession`, `AgentSession`, `AgentTeamSession` and `NodeSession`, with the `SubWorkflowSession` and `RouterSession` views; resume is keyed by `session_id`, global state is JSON, and component state is namespaced by component id.

### 2.5 Graph and workflow (`openjiuwen-graph`, `openjiuwen-workflow`)

```rust
// graph
pub struct GraphState(pub Map<String, Value>);
#[async_trait] pub trait Node: Send + Sync { fn id(&self) -> &str;
    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome>; }
pub enum NodeOutcome { Continue, Route { targets: Vec<String> }, Halt { value: Option<Value> } }
pub struct CompiledGraph { /* pregel/BSP executor */ }
impl CompiledGraph {
    pub fn spec(&self) -> &GraphSpec;
    pub async fn invoke(&self, input: Value, session: Arc<dyn Session>) -> Result<Value>;
    pub async fn run(&self, input: Value, session: Arc<dyn Session>,
                     options: ExecutionOptions) -> Result<GraphRunResult>;
    pub fn stream(&self, input: Value, session: Arc<dyn Session>)
        -> BoxStream<'static, Result<GraphEvent>>;
    pub fn stream_with(&self, input: Value, session: Arc<dyn Session>,
                       options: ExecutionOptions) -> BoxStream<'static, Result<GraphEvent>>;
    pub fn to_mermaid_with(&self, options: &MermaidOptions) -> String;
}

// workflow
impl Workflow {
    pub fn set_start_comp(&mut self, id: &str, component: impl Component + 'static,
                          options: ComponentOptions) -> &mut Self;
    pub fn add_workflow_comp(&mut self, id: &str, component: impl Component + 'static,
                             options: ComponentOptions) -> &mut Self;
    pub fn set_end_comp(&mut self, id: &str, component: impl Component + 'static,
                        options: ComponentOptions) -> &mut Self;
    pub fn add_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn add_connections(&mut self, from: &[&str], to: &str) -> &mut Self;
    pub fn build(&self) -> Result<CompiledWorkflow>;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>)
        -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>)
        -> Result<WorkflowOutput>;                        // HITL continuation
    pub fn to_svg(&self) -> Result<String>;
    pub fn draw(&self, format: DrawFormat) -> Result<String>;
}
```

`invoke_with`/`stream_with` take `WorkflowRunOptions` (stream modes, deadline, cancellation).

Superstep scheduling uses two channel kinds: a `NodeChannel` fires on the barrier groups accumulated from `EdgeKind::Barrier` edges, and one with no groups is a trigger channel that fires from a plain edge; both snapshot and restore with the run, so a resumed run re-derives readiness from persisted state rather than re-executing completed nodes.
`NodeOutcome::Route` selects the next nodes, `Halt` ends the run with a value, and a failed run saves a boundary snapshot at its last completed superstep.
Data flow uses textual `${node.field}` references resolved in the engine, with one documented extension: a field name applied to an array projects over its elements; component capabilities (`invoke`, `stream`, `transform`, `collect`) determine legal edge kinds.
Rendering is split: graph emits Mermaid (`CompiledGraph::to_mermaid_with`), and the deterministic SVG renderer lives in `openjiuwen-workflow::render` (`to_svg`, `render_svg`, `error_svg`, `SvgOptions`), where the same spec always produces byte-identical output and `DrawFormat::Png` is a typed `Error::Unsupported`.

### 2.6 Agents (`openjiuwen-agent`) and runner (`openjiuwen-runner`)

```rust
#[async_trait]
pub trait Agent: Send + Sync {
    fn card(&self) -> &AgentCard;
    async fn invoke(&self, inputs: Value, session: Option<Arc<dyn Session>>) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value, session: Option<Arc<dyn Session>>)
        -> BoxStream<'static, Result<AgentChunk>>;
}
```

There is no `BaseAgent`: `Agent` plus `AgentCard` is the whole seam, and `ReActAgent`, `LlmAgent`, `WorkflowAgent` and `ReActAgentEvolve` implement it.
`ControllerAgent` is not an `Agent`: it implements `TaskExecutor` over an `Arc<dyn Agent>`.
`ReActAgent::configure(ReActAgentConfig)`, `WorkflowAgent::add_workflows` and `LlmAgent::new(LlmAgentConfig)` are the configuration entry points.
Rail hooks are one trait, `AgentRail`, with nine callbacks: `before_invoke`, `after_invoke`, `before_model_call`, `after_model_call`, `on_model_exception`, `before_tool_call`, `after_tool_call`, `on_tool_exception`, `after_react_iteration`, plus `name` and `priority`.
Interrupts are typed by `InterruptKind` (`Undeclared`, `Permission`, `Confirm`, `AskUser`, `SkillAuthorization`, `SkillEvolutionApproval`, `TeamToolApproval`, `ClientTool`, `Custom`) and each maps to a stable wire string.
The controller stack (`controller::{BaseController, Controller, GroupRouter, TaskManager, TaskQueue, TaskSchedulerLoop}`) owns task bookkeeping; `TaskManager` is clone-shared and indexes tasks by id, by priority and by parent/child links including the root ids, while session lookup and status filtering are linear scans.
`KvCacheActionRuntime` (`prepare`/`suspend`/`release`) is the KV-cache affinity runtime, installed through `KVCacheModelCallHook::with_action_runtime`.
Operators implement `Operator` (`get_tunables`, `get_state`, `set_parameter`, `load_state`, `apply_update`); staged changes go through `PendingChangeStore` (`stage` → `approve`/`reject`/`retry`), which appends records to a `SkillExperienceStore` and reports a partial commit when the sink fails mid-way.

```rust
pub trait SpawnedAgent: Send + Sync {
    fn id(&self) -> &str;
    fn is_alive(&self) -> bool;
    async fn send(&self, inputs: Value) -> Result<AgentOutput>;
    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>>;
    async fn shutdown(&self) -> Result<bool>;
    async fn health(&self) -> Result<bool>;   // default: Error::Unsupported without a probe
}

impl Runner {
    pub fn new() -> Arc<Self>;
    pub fn global() -> Arc<Self>;             // process-global, documented singleton
    pub fn resource_mgr(&self) -> &Arc<ResourceMgr>;
    pub async fn start(&self) -> Result<()>;  // builds the checkpointer and queue
    pub async fn run_agent(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
    pub async fn run_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<Value>;
    pub async fn start_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<PoolRun>;
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
```

`Runner::run_workflow` mirrors `run_agent` with a `WorkflowRef`, every `run_*` has a `*_streaming` twin, and team runs additionally expose `try_start_agent_team`, `acquire_agent_team`, `pause_/resume_/stop_agent_team`, `interact_agent_team` and `answer_interaction`.
`RunOptions` carries the optional caller session, deadline, cancellation token, session id, envs, checkpointer override, stream modes and metadata; an id passed as `AgentRef`/`WorkflowRef`/`TeamRef` is resolved through `ResourceMgr`.
`ResourceMgr` registers tools, agents, workflows and teams by card, owns the tag, model, prompt, sysop and MCP-server managers, and `TeamPool` keys pooled team runs by `(team name, session id)` behind `start_agent_team`.
`SpawnerRegistry` routes a `SpawnSpec` to a backend (`spec.backend` hint → explicit default → sole spawner → typed failure) and `Runner::spawn_agent` returns a `SpawnedAgent` whose `health` probes the backend.
`DreamService` owns the single dreaming loop task over `openjiuwen_memory::dreaming::DreamOrchestrator` and is installed per `Runner`, so the memory crate stays spawn-free.
Tools are mounted by the host on an agent's `AbilityManager` and agents never read a global registry; tests use local `Runner` instances, while production code may use `Runner::global()` for parity.

### 2.7 Harness protocol (`openjiuwen-harness-protocol`)

Provider-neutral SPI with no workspace dependencies.
`HarnessProtocol` is the async surface (`card`, `state`, `provider_session_id`, `event_buffer_config`, `events`, `turn_events`, `start`, `stop`, `send`, `send_auto`, `abort`, `abort_graceful`, `pause`, `resume`, `export_checkpoint`).
`HarnessProvider` (`card`, `create`) is the synchronous factory half, and `HarnessModelControl` (`list_models`, `set_model`) is the optional model-selection half.
Service seams are `HarnessEventCursor`, `HarnessCheckpointSink`, `HarnessHookDispatcher`, `HarnessInteractionHandler`, `ToolGateway`, `HarnessTelemetry` and `Validate`, plus serde models for cards, events, results, streams and interactions.
The wire is pinned by `PROTOCOL_VERSION` (`"1.0"`) and `EVENT_WIRE_SCHEMA_VERSION` (`"1"`); capability gaps surface as `HarnessError::UnsupportedCapability`, not `Error::Unsupported`.
`openjiuwen-harness` implements the native provider (`NativeHarness`), and `openjiuwen-harness-providers` turns implementations into registrable factory entries (`NativeHarnessProvider`, `NativeV2HarnessProvider` over `NativeV2Driver`, and the three vendor adapters over `VendorDriver`).
`native` is default-on and complete; `native_v2` is opt-in and lacks checkpoint export, MCP tools and model selection; `claudecode`, `codex` and `dsh` are opt-in under `vendor` and drive an operator-installed CLI, with per-provider gaps (`codex` has no model control, `dsh` supports only open/close/run-turn).
`builtin_provider(name)` resolves the compile-time table and `resolve_provider`/`create_harness` fall back to runtime registrations through `register_harness_provider`.

### 2.8 Frame and event streams

Graph runs publish ordered `GraphEvent`s (`kind`, optional `node_id`, 1-based `superstep`, object `payload`): `superstep_started`, `node_started`, `node_completed`, `node_failed`, `edge_traversed`, `superstep_completed`, and one terminal `graph_completed`/`graph_halted`/`graph_failed`.
A consumed graph stream ends with that terminal event, or with exactly one `Err` after the failure's events; dropping the stream aborts the run.
Workflow runs stream `WorkflowChunk` frames in production order, filtered by `WorkflowRunOptions::stream_modes`, ending in one `workflow_final` frame or exactly one `Err`; an interrupted run ends quietly after its interaction frames.
Session frames are one struct, `StreamItem { mode: StreamMode, item_type (wire "type"), index, payload }`, where `mode` is the local routing class (`output`, `trace`, `custom`) and is dropped from the wire projection; `output`, `output_at`, `trace` and `custom` are the constructors.
Harness streams wrap provider events in `HarnessEvent`, whose serde keys are `schema_version` (`"1"`), `event_type`, `sequence`, `timestamp`, `host_session_id`, `agent_id`, `provider_session_id`, `turn_id`, `item_id`, `correlation_id`, `causation_ids` and `payload`; a foreign `schema_version` decodes to `UnknownEvent` and re-encodes unchanged.
Context telemetry adds two frame types of its own: `context.compression_state` (`CONTEXT_COMPRESSION_STATE_TYPE`) around every compression pass and `context.usage` (`CONTEXT_USAGE_EVENT_TYPE`) for the prepared window's usage block; the goal rail publishes `goal.updated` under `GOAL_EVENT_EXTRA_KEY` (`harness.goal_events`) once a goal settles, because rails have no direct event sink.
Subagents stream `SubagentActivityFrame`s through `ActivityThrottle` with `DEFAULT_ACTIVITY_CAPACITY`, `DEFAULT_ACTIVITY_THROTTLE_MS` and `DEFAULT_ACTIVITY_TEXT_MAX_LEN` bounds, and agent/team streams follow the same discipline as graph and workflow: `AgentChunk` frames in order and at most one terminal error, with an owned session committed after the last frame and on early drop.

### 2.9 Context engine (`openjiuwen-context`)

`ContextEngine` is a struct with inherent methods (no trait): `new`, `with_store(s)`, `with_recorder`, `with_usage_analyzer`, `config`, `set_recorder`, `set_usage_analyzer`, `create_context`, `create_context_with`, `get_context`, `remove_context`, `prepare`, `observe`, `observe_tool_result`, `compress`, `recover_from_model_exception`, `is_context_overflow_error`, `resolved_window`, `detect_window_change`, `usage_report`, `usage_snapshot`, `record_provider_usage`, `kv_cache_usage`, `save_contexts`, `load_contexts`.
`prepare` runs the pipeline over a working copy and never mutates stored history; `observe` appends one message and applies the rolling `max_context_messages` limit; `compress` persists the change and reports `noop`, `compressed` or `busy`.
There is no `get_context_window`: `prepare` returning `PreparedContext` is the window/messages getter, `ModelContext::messages()` is the raw reader, and `usage_report`/`usage_snapshot` explain the last prepared window.
Tokenizer selection is `TokenizerRegistry` + `TokenizerSelector`, resolved as explicit spec → native artefact → compatible fallback → registry (`exact`, then unique `family`) → model encoding → estimate, and `TokenizerArtifactManager` owns the on-disk vocabulary cache.
Offload handles are opaque ids scoped by `OffloadScope { session_id, context_id }`, markers keep Python's `[[OFFLOAD: handle=..., type=in_memory|filesystem]]` shape, and full compaction reinjects host-rendered state sections through the `SnapshotRegistry` seam (`PipelineDeps::reinjection`, wired with `with_reinjection`).
Every LLM-backed processor step has a deterministic offline fallback, so compression works with no network and no credentials.

### 2.10 Memory providers (`openjiuwen-memory`)

```rust
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> ProviderCapabilities;
    async fn initialize(&self) -> Result<()>;
    async fn add(&self, scope: &MemoryScope, memories: Vec<NewMemory>, options: &AddOptions) -> Result<AddOutcome>;
    async fn search(&self, scope: &MemoryScope, query: &str, options: &SearchOptions) -> Result<Vec<RecallHit>>;
    async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>>;
    async fn update(&self, scope: &MemoryScope, id: &str, update: MemoryUpdate, options: &CallOptions) -> Result<MemoryRecord>;
    async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool>;
    async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    async fn clear(&self, filter: &MemoryFilter) -> Result<usize>;
    async fn health(&self) -> Result<ProviderHealth>;
    // defaulted rail: prefetch, sync_turn, on_session_end, system_prompt_block,
    // get_tool_schemas, handle_tool_call, is_available, is_initialized, default_scope
}
```

`sync_turn(user_msg, assistant_msg)` and `on_session_end(&[ConversationTurn])` are the turn-lifecycle hooks a host drives; `sync_turn`'s default writes one episodic memory via `add(&self.default_scope(), …)`, `prefetch`'s default recalls and renders snippets, and only `on_session_end` and `system_prompt_block` are inert.
`EngineProvider` adapts the in-crate long-term engine to that SPI, `ProviderRegistry`/`ProviderKind` plus the vendor handles and `RestProvider` expose vendor memories as record-only implementations, and `TeamMemory` + `MemberMemoryToolkit` (`team_memory_remember|search|forget|stats`) cover team scope over `TeamMemoryObserver`.
Dreaming is a repository + orchestrator contract: `DreamRepository::claim_next_leased` takes a `LeaseRequest` and returns a `LeasedJob`, `DreamLeaseGuard` renews or releases it, and `DreamOrchestrator::tick` runs one bounded sweep.

### 2.11 Observability (`openjiuwen-extensions::observability`, OTel sink, team bridge)

`openjiuwen-extensions::observability` is dependency-neutral: only `file_exporter` and `transforming` sit behind `otel`, while `project_semconv`, `redaction`, `content_addressing`, `tool_outcome`, `error_reporting` and `deferred` compile everywhere.
`project_semconv` owns the attribute names and counts (`OJ_TURN_ID`, `AT_TEAM_ID`, `DA_TASK_ITERATION`, `GEN_AI_KEYS`, `OJ_KEYS`, `TRAJECTORY_SPAN_SCHEMA_VERSION`), `redaction` owns `mask_secrets`, `redact_error_summary`/`_stacktrace` and `ERROR_SUMMARY_MAX_LENGTH`, and `content_addressing` owns `content_hash`, `address_payload` and the `@oj-seq` sequence references.
`observability::deferred` is the documented adapter hole: `init_observability`, `publish_span_snapshot`, `publish_stream_frame`, `register_span_consumer` and `is_initialized` (a `const fn` returning `false`).
The real exporter is separate: `otel::OtelTraceSink` implements the kernel `TraceSink` over `opentelemetry` + `tracing-opentelemetry`, with `ExporterType::File` wired to `TraceFileExporter`; the only consumer of both halves is the `otlp`-gated team bridge, where `agent_teams::observability::otlp::OtlpMemoryObserver` implements `TeamMemoryObserver` and emits one `Component` span `memory.{operation}` per memory event.

### 2.12 Resume and journal (`openjiuwen-harness`)

`persistent_agent::ExecutionCatalog` is the on-disk index keyed by execution id: `new(root, capacity)`, `path_for(&self, execution_id: &str) -> PathBuf`, `async fn list(&self) -> Result<Vec<ExecutionSummary>>` and `async fn open(&self, execution_id: &str) -> Result<ExecutionLog>`.
`persistent_agent::resume(store, logs, links, binding: &SessionBinding, request_id, continuation, assembler: &WindowAssembler)` builds a `ResumePlan` — serde-serializable both ways — carrying `record`, `cursor`, `window`, `chain`, `background` and `digest`, with `seed_messages()`/`seed_history()` for replay; the module never appends log entries itself, the rails do.
`ResumePlan` is a value, so a host can persist or forward it; the CLI resumes through `cli::run_with_model_resumed`, the counterpart of `cli::run_with_model`.
The runner has no resume entry point of its own: `Runner` exposes only `run_agent(_streaming)`, `run_workflow(_streaming)` and `run_agent_team(_streaming)`.

## 3. Async and streaming model

- All I/O is async on tokio and CPU-bound work uses `spawn_blocking`; public streaming APIs return `BoxStream<'static, Result<T>>` built with `async_stream::try_stream!` and are always `Send + 'static`.
- Long-running loops check `CancellationToken` at await points; deadlines use `tokio::time::timeout`.
- Session fan-out is `tokio::sync::broadcast` behind `StreamHub`: bounded per subscriber, and a slow consumer drops its oldest frames instead of stalling the producer.
- Harness event buffering advertises `EventBufferConfig { capacity: 1024, overflow: CoalesceOrBlock }`; the policy coalesces progress frames and blocks rather than dropping a terminal frame.
- Checkpoints are taken per superstep in the graph engine (opt-in via `checkpoint_every_superstep`, plus one boundary snapshot on failure) and per turn through `HarnessCheckpointSink`, so a resumed run restarts from a completed boundary rather than re-executing a node.
- Structured concurrency: spawned tasks are joined or explicitly detached with a documented owner, and no detached task may outlive its `Runner` without being tracked.
- Shared mutable state uses `Arc<RwLock<..>>`/`Arc<Mutex<..>>`, locks are never held across `.await`, and the only process-global mutable state is `Runner::global()` plus the TTL-bounded OpenRouter window cache in `openjiuwen-context::policy` and the harness provider registry — all three documented exceptions.

## 4. Configuration and environment

- No central config file, matching Python. Configs are typed structs at call sites; YAML is only for packaged resources (prompts, templates, skills).
- `.env` loading is opt-in (`dotenvy`) at binary entry points only, never in library constructors.
- API keys resolve in the order `OPENJIUWEN_API_KEY` → `API_KEY` → `OPENAI_API_KEY`, trimming blanks; other provider variables have `OPENJIUWEN_*` and bare forms.

| Family | Variables | Effect |
| --- | --- | --- |
| Home and workspace | `OPENJIUWEN_HOME`, `OPENJIUWEN_WORKSPACE` | Root of runtime state (`OpenJiuwenHome::resolve()`, `ensure()` creates it) and the working directory tools operate on. |
| Credentials and model defaults | `OPENJIUWEN_API_KEY`, `OPENJIUWEN_API_BASE`, `OPENJIUWEN_MODEL`, `OPENJIUWEN_MODEL_NAME`, `OPENJIUWEN_MODEL_PROVIDER`, `OPENJIUWEN_PROVIDER`, `OPENJIUWEN_MAX_TOKENS`, `OPENJIUWEN_MAX_ITERATIONS`, `OPENJIUWEN_LLM_TIMEOUT_SECS` | Model selection and per-call defaults. |
| Provider aliases | `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `NO_COLOR` | Shared / OpenAI spellings honoured alongside the `OPENJIUWEN_*` forms; `NO_COLOR` disables CLI colour. |
| CLI, service and logging | `OPENJIUWEN_SERVER_URL`, `OPENJIUWEN_HUB_URL`, `OPENJIUWEN_HUB_API_KEY`, `OPENJIUWEN_AUTH_FILE`, `OPENJIUWEN_NO_ONBOARDING`, `OPENJIUWEN_LIVE`, `OPENJIUWEN_LOG_LEVEL`, `OPENJIUWEN_LOG_FORMAT`, `OPENJIUWEN_OTEL_SERVICE_NAME`, `OPENJIUWEN_OTEL_EXPORTER_OTLP_ENDPOINT`, `OPENJIUWEN_OTEL_EXPORTER_OTLP_HEADERS`, `OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL`, `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` | CLI target, auth-file location, onboarding suppression, the offline/live switch for examples, `tracing` subscriber setup and the OTLP/HTTP export settings. |
| Sessions and teams | `OPENJIUWEN_SESSION_ID`, `OPENJIUWEN_TEAM_NAME`, `OPENJIUWEN_TEAM_JOIN`, `OPENJIUWEN_TEAM_SOCKET`, `OPENJIUWEN_MEMBER_NAME`, `OPENJIUWEN_MEMBER_WORKSPACE`, `OPENJIUWEN_MEMBER_CHECKPOINTS`, `OPENJIUWEN_MEMBER_CHECKPOINT`, `OPENJIUWEN_MEMBER_CHECKPOINT_DIR` | Team identity, join target, the Unix-socket attach path, member workspace, and where member checkpoints are read from. |
| Storage and caches | `OPENJIUWEN_REDIS_URL`, `OPENJIUWEN_REDIS_USERNAME`, `OPENJIUWEN_REDIS_PASSWORD`, `OPENJIUWEN_TOKENIZER_CACHE_DIR`, `OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE`, `OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`, `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR` | Redis connection, tokenizer artefact cache, OpenAI-account catalogue cache and the rule-compression dumps; Python's full-compaction dumps are not ported (see `PARITY.md` §Deferred Features). |
| Object stores | `OPENJIUWEN_S3_{ENDPOINT,BUCKET,REGION,ACCESS_KEY_ID,SECRET_ACCESS_KEY}`, `OPENJIUWEN_GCS_{ENDPOINT,PROJECT,ACCESS_TOKEN}`, `OPENJIUWEN_AZURE_STORAGE_{ACCOUNT,KEY,SAS,ENDPOINT}` | Symphony object-artifact transports. |
| Tools and harness | `OPENJIUWEN_BASH_STRICT`, `OPENJIUWEN_MCP_{CONFIG,SERVER_CMD,RECONNECT_ATTEMPTS}`, `OPENJIUWEN_WEB_SEARCH_ENDPOINT`, `OPENJIUWEN_SUBAGENT_TASK_TIMEOUT_S`, `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY`, `OPENJIUWEN_EXTENSIONS_TEARDOWN_HELPER` | Shell strictness, MCP client wiring, the generic HTTP web-search backend, subagent deadlines, history replay, and the test teardown helper. |
| Vendor adapters | `OPENJIUWEN_CODEX_API_KEY`, `OPENJIUWEN_DASHSCOPE_API_KEY`, `OPENJIUWEN_DASHSCOPE_API_BASE`, `OPENJIUWEN_DASHSCOPE_API_KEY_ENV`, `OPENJIUWEN_DASHSCOPE_API_BASE_ENV` | Vendor CLI credentials and the DashScope names to read keys from. |

- Secrets use a `SecretString` wrapper that redacts in `Debug` and logging; serialization preserves the value so configuration round-trips, and credentials are never logged.

## 5. Feature flags

Default builds need no network and no external service; every feature except the ones marked *default* is additive and opt-in.

| Crate | Features | Default | Effect |
| --- | --- | --- | --- |
| `openjiuwen-foundation` | `mcp`, `tiktoken` | yes | MCP client/server tooling (`rmcp`) and `cl100k_base` counting. |
| `openjiuwen-foundation` | `sqlite` | no | SQLite KV/document store backends. |
| `openjiuwen-session` | `sqlite`, `otel` | no | `SqliteCheckpointer`; `otel` is an empty marker that makes `install_trace_sink` install the supplied sink instead of the documented no-op. |
| `openjiuwen-memory` | `sqlite` | no | SQLite stores, FTS5 coding index, migration runner. |
| `openjiuwen-sysop` | `sandbox` | no | Sandbox HTTP transport (`reqwest`); remote `invoke` still returns `Error::Unsupported`. |
| `openjiuwen-core` | `otel`, `sqlite`, `sandbox`, `live-tests` | no | Forwarders to session/foundation/memory/sysop; they cover the whole gated kernel surface. |
| `openjiuwen-extensions` | `a2a` | no | A2A client/server, wire types, harness bridge. |
| `openjiuwen-extensions` | `otel` | no | Real OTel SDK provider, `OtelTraceSink`, handlers, tracing layer, `observability::{file_exporter, transforming}`. |
| `openjiuwen-extensions` | `sqlite`, `redis`, `pulsar` | no | SQLite checkpointer/store, Redis KV/document/vector stores + checkpointer, Pulsar queue surface (constructs then rejects). |
| `openjiuwen-extensions` | `mcp` | no | `rmcp` dependency only — no source file reads it (dead gate). |
| `openjiuwen-harness` | `native` | yes | Empty marker; no source reads it. |
| `openjiuwen-harness` | `browser`, `cua`, `mobile`, `notify` | no | Documented `Unsupported` subagent presets; `notify` pulls `notify` and gates `watcher::backend`, without which `start` is a typed `Unsupported`. |
| `openjiuwen-harness-providers` | `native` | yes | The in-process DeepAgent harness provider. |
| `openjiuwen-harness-providers` | `vendor`, `native-v2` | no | Registers `claudecode`/`codex`/`dsh` over `vendor_cli`; adds the team-hosted `native_v2` provider (pulls `openjiuwen-agent-teams`). |
| `openjiuwen-agent-teams` | `mcp` | yes | The `openjiuwen-team-mcp` stdio MCP server binary. |
| `openjiuwen-agent-teams` | `sqlite`, `otel`, `distributed`, `otlp` | no | `SqliteTeamStore`; in-memory/tracing OTel seam; distributed `Messager` seams that reject at construction; `OtlpMemoryObserver` over `openjiuwen-extensions/otel`. |
| `openjiuwen-symphony` | `sqlite` | no | `KvAssetStore::sqlite` and `SqliteExperienceStore`. |
| `openjiuwen-agent-evolving` | `redis` | no | `RedisOnlineStore` (cluster/`rediss://` rejected); the HTTP router still hard-wires `MemoryOnlineStore`. |
| `openjiuwen` | `native` | yes | Forwards `harness/native` + `harness-providers/native`. |
| `openjiuwen` | `native-v2`, `vendor`, `browser`, `cua`, `mobile`, `notify`, `sqlite`, `a2a`, `mcp`, `redis`, `pulsar`, `distributed`, `otel`, `otlp`, `sandbox`, `live-tests` | no | One forwarder each to the owning crate; `sqlite` fans out over foundation, session, memory, extensions, symphony and agent-teams, `otlp` forwards only `agent-teams/otlp` (the team-memory bridge), and `live-tests` forwards the foundation/core/harness/providers/teams/extensions gates. |
| `openjiuwen-e2e` | `a2a`, `otel`, `sqlite` | no | Enable the `a2a_roundtrip`, `plugin_observability` and sqlite-gated scenario cases. |
| `openjiuwen-examples` | `a2a`, `mcp` | no | Compile example 11; `mcp` is a marker only (MCP is always compiled through foundation's default). |

- 15 crates declare `default = []` and no other feature: `openjiuwen-agent`, `openjiuwen-auto-harness`, `openjiuwen-common`, `openjiuwen-context`, `openjiuwen-dev-tools`, `openjiuwen-graph`, `openjiuwen-harness-protocol`, `openjiuwen-macros`, `openjiuwen-multiagent`, `openjiuwen-retrieval`, `openjiuwen-rsi`, `openjiuwen-runner`, `openjiuwen-security`, `openjiuwen-test-support`, `openjiuwen-workflow`.
- `live-tests` is declared by foundation, harness, harness-providers, agent-teams, extensions, cli and the umbrella, plus core as a forwarder; all of them are empty gates, and `OPENJIUWEN_LIVE` is what switches the examples between offline and live.
- Suites that need an optional capability are gated with `#![cfg(feature = "…")]` (`a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `vendor`, `native-v2`) or `#![cfg(unix)]`, never by `live-tests`.
- `openjiuwen-harness` and `openjiuwen-harness-providers` are non-optional umbrella dependencies taken with default features, so `native` always applies and nothing enables `vendor`.
- `openjiuwen-cli` depends on `openjiuwen-harness` with `features = ["notify"]` directly, so the file watcher is compiled into the binary regardless of the umbrella's `notify`.

## 6. Testing infrastructure

- `openjiuwen-test-support` is dev-only (`publish = false`, never a library dependency) and holds 152 tests of its own: a loopback `MockLlmServer` (axum, OpenAI-compatible, chat completions + SSE + tool calls + scripted responses), `ScriptedModel`/`MockCompletion` fakes, temporary workspace/home/env/session fixtures, JSON assertions (`assert_json_eq`, `assert_json_subset`, `assert_json_pointer_eq`, `assert_json_roundtrip`, `first_difference`), eventual-consistency polling, and a `wiremock` re-export; there is no card snapshot store.
- Integration tests are one binary per file: 394 top-level `tests/*.rs` files workspace-wide (368 outside `openjiuwen-e2e`); the umbrellas keep only path/feature checks (`openjiuwen/tests/module_paths.rs`, `openjiuwen-core/tests/{module_paths,feature_forwarding}.rs`).
- `openjiuwen-e2e` holds 26 scenario binaries with 125 scenarios, all cross-crate, with bounded helpers (`BOUND` = 10 s, `within`, `collect_bounded`) in `tests/support/mod.rs`.
- `openjiuwen-examples` has 26 example binaries plus `tests/examples_offline.rs`, which runs every example through the offline harness.
- Everything is offline and deterministic by default: scripted models or the loopback server, temp directories, loopback sockets, no credentials, no wall-clock ordering.
- Deferred areas are proven, not just documented: dedicated suites assert `Error::Unsupported` and its message (for example `openjiuwen-dev-tools/tests/deferred_surface.rs`, `openjiuwen-symphony/tests/deferred.rs`, `openjiuwen-agent-evolving/tests/rl.rs`, `openjiuwen-extensions/tests/observability_deferred.rs`, `openjiuwen-rsi/tests/deferred.rs`).
- `default-members` lists the 27 library and binary crates, so a bare `cargo test` skips the three development crates; the Makefile gates pass `--workspace` (or `-p`) and therefore include them.
- Gates: `make test` (`cargo test --workspace`, offline), `make test-all` (the feature-gated suites: e2e `a2a`, extensions `a2a,otel`, harness and agent-teams `--no-default-features`, symphony `sqlite`, sysop `sandbox`, harness-providers `vendor`), and `make verify` (`fmt-check`, `clippy`, `test`, `test-all`, plus the example builds).
- `cargo doc --workspace --no-deps` (and `--all-features`) stays warning-free; link private or feature-gated items as plain code spans instead of widening visibility.

## 7. Porting method

1. Read the Python module and its unit tests through a read-only investigator; preserve serialized names, default values and observable behavior, and never modify the reference repository.
2. Replace Python-specific mechanisms (pydantic, asyncio globals, reflection) with the equivalents in §2; internal structure is idiomatic Rust, not a transliteration.
3. Port representative unit tests and keep each crate's `CONTEXT.md` as the routing table for its directory, with `docs/PARITY.md` as the derived per-package matrix.
4. Prove every deferral with a test that asserts `Error::Unsupported`, so no capability disappears silently.
5. Keep files around ≤1000 lines and split by responsibility; `unsafe` is denied workspace-wide.
6. A fixed gap is deleted from the docs rather than annotated as fixed, so documentation states the current state only.
7. Offline builds are a hard requirement: every dependency is present in the lockfile cache and the gates run without network access.
