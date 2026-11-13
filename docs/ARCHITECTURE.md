# Architecture

This document defines the cross-cutting contracts of the Rust rewrite.
Per-crate details live in each crate's `CONTEXT.md`.

## 1. Crate layering

Dependencies point downward only; a crate may depend on crates in lower layers, never upward.
`openjiuwen-core` is the kernel umbrella and may depend on every kernel crate.
`openjiuwen` is the top-level umbrella and may depend on every library crate (never on the CLI binary).

| Layer | Crates | Notes |
|---|---|---|
| 0 | `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-harness-protocol` | No workspace dependencies (macros is a proc-macro crate; protocol is a pure SPI). |
| 1 | `openjiuwen-foundation`, `openjiuwen-sysop`, `openjiuwen-session`, `openjiuwen-graph`, `openjiuwen-retrieval` | Session may depend on foundation and common; foundation never depends on session. |
| 2 | `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-memory`, `openjiuwen-security` | Workflow sits on graph + session + foundation; memory depends on foundation and retrieval (not context or session). |
| 3 | `openjiuwen-agent`, `openjiuwen-multiagent` | Agent composes foundation, session, workflow, context, memory, security, sysop and hosts the `controller`, `kv_cache`, `operator`, `evolve` and `prompt_pack` extension modules. |
| 4 | `openjiuwen-runner`, `openjiuwen-core` | Runner owns the process-global orchestrator over the kernel crates it drives; core re-exports every kernel crate. |
| 5 | `openjiuwen-extensions`, `openjiuwen-dev-tools`, `openjiuwen-symphony`, `openjiuwen-harness` | Depend on `openjiuwen-core`; harness also drives the kernel crates directly (agent, workflow, graph, session, security, context, memory, retrieval, sysop), dev-tools depends on agent, symphony on graph/retrieval/session. |
| 6 | `openjiuwen-harness-providers`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-agent-teams` | Depend on harness and/or core; auto-harness depends only on rsi and re-exports it. |
| 7 | `openjiuwen` (umbrella), `openjiuwen-cli` | Umbrella over every library crate; the binary composes harness, harness-protocol and core. |
| dev | `openjiuwen-test-support`, `openjiuwen-e2e`, `openjiuwen-examples` | Never depended on by library crates. |

Rules:
- If two crates need the same type, it belongs in the lowest common ancestor (usually `openjiuwen-common` or `openjiuwen-foundation`).
- Do not introduce cycles to mirror Python's function-local imports; use traits and registration instead.
- Python's function-local `core → harness/extensions/agent_teams` imports become extension traits implemented in the upper crate and registered at startup.

## 2. Shared contracts

These signatures are the public seams; changing one requires updating every dependent crate.

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

The Python `Card`/`Config` split is preserved: `*Card` is serializable and registry-safe; `*Config` is process-local, may hold clients/credentials, and is passed to `configure`/`set_config`.

All serialized structs use `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields, and default missing collections to empty.

### 2.2 Messages, prompts, and models (`openjiuwen-foundation`)

```rust
pub enum Role { System, User, Assistant, Tool, Other(String) }
pub enum ContentPart { Text { text: String }, ImageUrl { url: String, detail: Option<String> },
                       Audio { data: String, format: String }, File { ... }, Other(Value) }
// Messages live in the `prompt` module, not `llm`; `llm` re-exports them.
pub struct BaseMessage { pub role: Role, pub content: MessageContent, pub name: Option<String>,
                         pub tool_calls: Vec<ToolCall>, pub tool_call_id: Option<String>,
                         pub reasoning_content: Option<String>, pub metadata: BTreeMap<String, Value> }
pub enum MessageContent { Text(String), Parts(Vec<ContentPart>) } // serde accepts both forms
pub struct PromptTemplate { pub name: String, pub content: TemplateContent,
                            pub placeholder_prefix: String, pub placeholder_suffix: String }

pub struct ModelClientConfig { pub provider: String, pub api_key: Option<SecretString>,
                               pub api_base: Option<String>, pub verify_ssl: bool,
                               pub timeout: Option<Duration>, pub max_retries: u32,
                               pub headers: BTreeMap<String, String>, pub extra: Value }
pub struct ModelRequestConfig { pub model_name: String, pub temperature: Option<f32>,
                                pub top_p: Option<f32>, pub max_tokens: Option<u32>,
                                pub stop: Vec<String>, pub tools: Option<Value>,
                                pub tool_choice: Option<Value>, pub stream: bool, pub extra: Value }

pub struct ChatResponse { pub id: String, pub model: String, pub choices: Vec<Choice>, pub usage: Usage }
pub enum StreamChunk { Delta { index: usize, content: Option<String>, reasoning: Option<String>,
                              tool_call: Option<ToolCallDelta>, role: Option<Role>,
                              finish_reason: Option<FinishReason> },
                       Usage(Usage), Done }

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

Two transports are implemented: the OpenAI-compatible client (every OpenAI-compatible vendor through `api_base`) and the Anthropic Messages client.
Sibling modules carry reasoning profiles (`llm::reasoning`), IntelliRouter-style routing and failover (`llm::routing`), OpenAI OAuth (`llm::oauth`) and DashScope media generation (`llm::dashscope`); the OpenAI Responses API has no transport.
Providers self-register in a `ModelProviderRegistry`; extension providers implement `ModelProvider` in `openjiuwen-extensions`.

### 2.3 Tools (`openjiuwen-foundation`)

```rust
pub struct ToolCard { pub id: String, pub name: String, pub description: String,
                      pub input_params: Value /* JSON Schema */,
                      pub exposure: ToolExposure, pub execution: ToolExecution,
                      pub parallel_safe: bool, pub stateless: bool, pub idempotent: bool,
                      pub version: Option<String>, pub metadata: BTreeMap<String, Value> }
pub struct ToolContext { pub session_id: Option<String>, pub invocation_id: String,
                         pub cancellation: CancellationToken, pub metadata: BTreeMap<String, Value>,
                         pub timeout: Option<Duration> }

#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn card(&self) -> &ToolCard;
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value>;
    fn stream(self: Arc<Self>, inputs: Value, ctx: ToolContext)
        -> BoxStream<'static, Result<Value>> { /* one invoke */ }
    fn render_for_llm(&self, output: &Value) -> String { /* ToolOutput else JSON text */ }
}
```

Implementations are `FunctionTool` (closure/typed handler + JSON Schema), `RestfulApiTool`, and `McpTool` over stdio, SSE, streamable-HTTP and OpenAPI transports; the Playwright transport is a typed `Error::Unsupported`.
`ToolCard.input_params` is generated from `schemars` where a typed handler exists; the Python `@tool` decorator maps to the `#[tool]` attribute macro in `openjiuwen-macros` or the `FunctionTool::builder`.
`ClientTool` lives in `openjiuwen-agent`, next to the client-tool registry that executes it.

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
    async fn request_interaction(&self, request: InteractionRequest)
        -> Result<InteractionResponse>;
    fn trace(&self) -> &dyn TraceSink;
    // checkpointing
    async fn snapshot(&self) -> Result<SessionSnapshot>;
    async fn restore(&self, snapshot: SessionSnapshot) -> Result<()>;
    async fn save_checkpoint(&self, checkpoint_id: Option<&str>) -> Result<String>;
    async fn resume(&self, checkpoint_id: &str) -> Result<bool>;
    // defaulted seams: install_trace_sink, metadata, checkpointer, tracer, config, writers,
    // subscribe, resolve_interaction, end_stream, close
}
```

Concrete sessions: `WorkflowSession`, `AgentSession`, `AgentTeamSession`, `NodeSession`, plus the `SubWorkflowSession` and `RouterSession` views.
Checkpointing is a trait (`Checkpointer`): in-memory, filesystem/JSON, SQLite (`sqlite` feature) and Redis RESP2 (always compiled) backends, with `DiskSessionStore` records and `SessionHistory` versions on top.
Resume semantics are keyed by `session_id` exactly as in Python; global state is JSON, component state is namespaced by component id.

### 2.5 Graph and workflow (`openjiuwen-graph`, `openjiuwen-workflow`)

```rust
// graph
pub struct GraphState(pub Map<String, Value>);
#[async_trait] pub trait Node: Send + Sync { fn id(&self) -> &str;
    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome>; }
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
    pub fn to_mermaid(&self) -> String;
    pub fn to_mermaid_with(&self, options: &MermaidOptions) -> String;
}

// workflow
pub struct Workflow { /* builder + card */ }
impl Workflow {
    pub fn set_start_comp(&mut self, id: &str, comp: impl Component + 'static,
                          options: ComponentOptions) -> &mut Self;
    pub fn add_workflow_comp(&mut self, id: &str, comp: impl Component + 'static,
                             options: ComponentOptions) -> &mut Self;
    pub fn set_end_comp(&mut self, id: &str, comp: impl Component + 'static,
                        options: ComponentOptions) -> &mut Self;
    pub fn add_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn add_connections(&mut self, from: &[&str], to: &str) -> &mut Self;
    pub fn add_conditional_connection(&mut self, from: &str, to: &str,
                                      router: Arc<dyn Router>) -> &mut Self;
    pub fn add_stream_connection(&mut self, from: &str, to: &str) -> &mut Self;
    pub fn build(&self) -> Result<CompiledWorkflow>;
    pub async fn invoke(&self, inputs: Value, session: Arc<dyn Session>) -> Result<WorkflowOutput>;
    pub async fn invoke_with(&self, inputs: Value, session: Arc<dyn Session>,
                             options: WorkflowRunOptions) -> Result<WorkflowOutput>;
    pub async fn resume(&self, interactive: InteractiveInput, session: Arc<dyn Session>)
        -> Result<WorkflowOutput>;                       // HITL continuation
    pub fn stream(&self, inputs: Value, session: Arc<dyn Session>)
        -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn stream_with(&self, inputs: Value, session: Arc<dyn Session>,
                       options: WorkflowRunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub fn to_mermaid(&self) -> Result<String>;
    pub fn to_svg(&self) -> Result<String>;              // local, deterministic
    pub fn draw(&self, format: DrawFormat) -> Result<String>;
}
```

Graph checkpoints round-trip through the serializable `GraphSpec` and the `GraphStore`/`GraphSnapshot` seam.
Data flow uses textual references `${node.field}` resolved in the engine, with one documented extension: a field name applied to an array projects over its elements.
Component capabilities (`invoke`, `stream`, `transform`, `collect`) determine legal edge kinds.

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

`ReActAgent::configure(ReActAgentConfig)`, `WorkflowAgent::add_workflows`, and `LlmAgent::new(LlmAgentConfig)` plus `set_prompt_template` are the configuration entry points.
Tools are mounted by the host on an agent's `AbilityManager`; `Runner::resource_mgr` is the registry the host resolves them from, and the agent never reads a global registry.

```rust
pub struct Runner { /* resources, callbacks, message queue, config, checkpointer, guardrails,
                       spawner, monitor, HITL hub, team pool */ }
impl Runner {
    pub fn new() -> Arc<Self>;
    pub fn with_config(config: RunnerConfig) -> Arc<Self>;
    pub fn global() -> Arc<Self>;             // process-global, documented singleton
    pub fn resource_mgr(&self) -> &Arc<ResourceMgr>;
    pub async fn start(&self) -> Result<()>; // builds the checkpointer and queue
    pub async fn stop(&self) -> Result<()>;
    pub async fn run_agent(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> Result<AgentOutput>;
    pub fn run_agent_streaming(&self, agent: impl Into<AgentRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<AgentChunk>>;
    pub async fn run_workflow(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> Result<WorkflowOutput>;
    pub fn run_workflow_streaming(&self, workflow: impl Into<WorkflowRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<WorkflowChunk>>;
    pub async fn run_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<Value>;
    pub fn run_agent_team_streaming(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> BoxStream<'static, Result<Value>>;
    pub async fn start_agent_team(&self, team: impl Into<TeamRef>, inputs: Value, options: RunOptions) -> Result<PoolRun>;
    pub async fn pause_agent_team(&self, team_name: &str, session_id: &str) -> Result<()>;
    pub fn interact_agent_team(&self, team_name: &str, session_id: &str,
                               payload: InteractPayload) -> DeliverResult;
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>>;
}
```

`RunOptions` carries the optional caller session, deadline, cancellation token, session id, envs, checkpointer override, stream modes, and metadata; an id passed as `AgentRef`/`WorkflowRef`/`TeamRef` is resolved through `ResourceMgr`.
`ResourceMgr` registers tools, workflows, agents, and teams by id; `TeamPool` keys pooled team runs by `(team name, session id)` and backs `start_agent_team`.
Tests use local `Runner` instances; production code may use `Runner::global()` for parity.

### 2.7 Harness protocol (`openjiuwen-harness-protocol`)

Provider-neutral SPI with no workspace dependencies: `HarnessProtocol`, `HarnessProvider`, `HarnessModelControl`, `HarnessEventCursor`, `HarnessCheckpointSink`, `HarnessHookDispatcher`, `HarnessInteractionHandler`, `ToolGateway`, `HarnessTelemetry` and `Validate`, plus serde models for cards, events, results, streams and interactions.
The event wire codec is pinned by `PROTOCOL_VERSION` (`"1.0"`) and `EVENT_WIRE_SCHEMA_VERSION` (`"1"`); capability gaps surface as `HarnessError::UnsupportedCapability`, not `Error::Unsupported`.
`openjiuwen-harness` implements the native provider; `openjiuwen-harness-providers` registers providers (`native` implemented; `native_v2` and `claudecode`, `codex`, `dsh` declared but returning `Error::Unsupported`, the last three behind the `vendor` feature).

### 2.8 Frame and event streams

Graph runs publish ordered `GraphEvent`s (`kind`, optional `node_id`, 1-based `superstep`, object `payload`): `superstep_started`, `node_started`, `node_completed`, `node_failed`, `edge_traversed`, `superstep_completed`, and one terminal `graph_completed`/`graph_halted`/`graph_failed`.
A consumed graph stream ends with that terminal event, or with exactly one `Err` after the failure's events; dropping the stream aborts the run.
Workflow runs stream `WorkflowChunk` frames (the `StreamItem` alias) in production order, filtered by `WorkflowRunOptions::stream_modes`, ending in one `workflow_final` frame or exactly one `Err`, while an interrupted run ends quietly after its interaction frames.
Agent and team streams follow the same discipline: `AgentChunk` frames in order and at most one terminal error, with an owned session committed after the last frame and on early drop.

## 3. Async and streaming model

- All I/O is async on tokio; CPU-bound work uses `spawn_blocking`.
- Public streaming APIs return `BoxStream<'static, Result<T>>` built with `async_stream::try_stream!`.
- Long-running loops check `CancellationToken` at await points; timeouts use `tokio::time::timeout`.
- Structured concurrency: spawned tasks are joined or explicitly detached with a documented owner; no detached task may outlive its `Runner` without being tracked.
- Shared mutable state uses `Arc<RwLock<..>>`/`Arc<Mutex<..>>`; locks are never held across `.await`.

## 4. Configuration and environment

- No central config file, matching Python. Configs are typed structs at call sites; YAML is only for packaged resources (prompts, templates, skills).
- Environment variables: `OPENJIUWEN_HOME`, `OPENJIUWEN_WORKSPACE`, `OPENJIUWEN_API_KEY`, `OPENJIUWEN_API_BASE`, `OPENJIUWEN_MODEL`, `OPENJIUWEN_MODEL_NAME`, `OPENJIUWEN_PROVIDER`, `OPENJIUWEN_SERVER_URL`, `OPENJIUWEN_MAX_TOKENS`, `OPENJIUWEN_MAX_ITERATIONS`, `OPENJIUWEN_LLM_TIMEOUT_SECS`, `OPENJIUWEN_LOG_LEVEL`, `OPENJIUWEN_LOG_FORMAT`, `OPENJIUWEN_BASH_STRICT`, `OPENJIUWEN_MCP_RECONNECT_ATTEMPTS`, `OPENJIUWEN_MCP_CONFIG`, `OPENJIUWEN_MCP_SERVER_CMD`, `OPENJIUWEN_HISTORY`, `OPENJIUWEN_HISTORY_REPLAY`, `OPENJIUWEN_NO_ONBOARDING`, `OPENJIUWEN_AUTH_FILE`, `OPENJIUWEN_LIVE`, the `OPENJIUWEN_OTEL_*` export settings, the team/member variables (`OPENJIUWEN_TEAM_*`, `OPENJIUWEN_MEMBER_*`, `OPENJIUWEN_SESSION_ID`) and the Redis connection variables.
- Provider variables are honoured alongside the `OPENJIUWEN_*` forms: `API_KEY`, `API_BASE`, `MODEL_PROVIDER`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`; the CLI's colour honours `NO_COLOR`.
- `.env` loading is opt-in (`dotenvy`) at binary entry points only, never in library constructors.
- Secrets use a `SecretString` wrapper that redacts in `Debug` and logging; serialization preserves the value so configuration round-trips, and credentials are never logged.

## 5. Feature flags

Default features must compile with no external services.
- `openjiuwen-foundation`: `mcp` (default; stdio/SSE/streamable-HTTP/OpenAPI transports), `tiktoken` (default; `tiktoken-rs` token counting), `sqlite` (SQLite store backends), `live-tests`.
- `openjiuwen-session`: `sqlite` (SQLite checkpointer), `otel` (empty marker that enables the real trace-sink installation path).
- `openjiuwen-memory`: `sqlite` (SQLite store and migrations).
- `openjiuwen-sysop`: `sandbox` (HTTP sandbox gateway plus the launcher lifecycle).
- `openjiuwen-core`: `otel` (forwards `openjiuwen-session/otel`).
- `openjiuwen-symphony`: `sqlite` (SQLite-backed capability-asset store).
- `openjiuwen-extensions`: `a2a`, `otel`, `sqlite`, `redis`, `pulsar`, `mcp`, `live-tests` (all optional; the Elasticsearch vector store and the AIO/JiuwenBox/YuanRong sandbox transports are always compiled in).
- `openjiuwen-harness`: `native` (default), `browser`/`cua`/`mobile` (empty toggles that only compile the `Error::Unsupported` subagent presets), `notify` (optional dependency with no gated code), `live-tests`.
- `openjiuwen-harness-providers`: `native` (default), `vendor` (compiles the vendor config types and registers their `Error::Unsupported` stubs), `live-tests`; all three are empty toggles.
- `openjiuwen-agent-teams`: `mcp` (default), `sqlite` (team store), `otel` (observability hooks over the in-memory seam), `distributed` (transport stubs), `live-tests`.
- `openjiuwen`: `native` (default), `otel`, `a2a`, `sqlite`, `live-tests`.
- `openjiuwen-cli`: `live-tests` only; `openjiuwen-e2e`: `a2a` (A2A round-trip scenario) and `otel` (parity marker); `openjiuwen-examples`: `a2a` (compiles example 11) and `mcp` (empty marker).
- `openjiuwen-common`, `openjiuwen-macros`, `openjiuwen-graph`, `openjiuwen-workflow`, `openjiuwen-context`, `openjiuwen-retrieval`, `openjiuwen-security`, `openjiuwen-agent`, `openjiuwen-multiagent`, `openjiuwen-runner`, `openjiuwen-dev-tools`, `openjiuwen-rsi`, `openjiuwen-agent-evolving`, `openjiuwen-auto-harness`, `openjiuwen-harness-protocol`, `openjiuwen-test-support`: no features.
- 7 manifests declare `live-tests`; it is never enabled by default and no test in the workspace is gated on it (`OPENJIUWEN_LIVE` only switches the examples from offline to live).
- The `native` provider is default-on in `openjiuwen-harness-providers`, and the umbrella depends on that crate non-optionally, so `openjiuwen::harness_providers::create_harness("native", …)` and `resolve_provider("native")` are always available; vendor providers require a direct dependency with `features = ["vendor"]`.

## 6. Testing infrastructure

- `openjiuwen-test-support` is dev-only (`publish = false`, never a library dependency) and provides:
  - `MockLlmServer` — an axum loopback OpenAI-compatible endpoint supporting chat completions, SSE streaming, tool calls, and scripted responses;
  - `ScriptedModel`/`MockCompletion` fakes;
  - temporary workspace, home, env, and session/checkpointer fixtures;
  - JSON assertion helpers (`assert_json_eq`, `assert_json_subset`, `assert_json_pointer_eq`, `assert_json_roundtrip`, `first_difference`), `eventually` polling helpers, and a `wiremock` re-export; there is no card snapshot store.
- `openjiuwen-e2e` holds 8 cross-crate binaries: `kernel_flow` (ReAct tool round, checkpoint round trip, workflow `start → llm → end`, local `Runner`), `deep_agent` (rounds and permissions), `provider_native`, `team_flow`, `a2a_roundtrip` (feature `a2a`), `symphony_plan`, `self_improvement`, and `cli_binary` (the harness CLI layer).
  Long-term memory recall is covered by the `openjiuwen-memory` integration suites, MCP stdio by `openjiuwen-foundation`, and process-level CLI smoke by `openjiuwen-cli/tests/cli_smoke.rs` (the e2e variant is `#[ignore]`d).
- Every e2e scenario runs offline against `ScriptedModel`, a loopback `MockLlmServer`, or loopback sockets, with a per-await bound in `tests/support`.
- Unit tests live next to modules; behavior-port tests reference the corresponding Python test file in a comment.
- 3 tests are `#[ignore]`d and each names its prerequisite: a live sandbox gateway (`openjiuwen-sysop`, reachable with `--features sandbox -- --ignored`), the e2e process-boundary CLI test (Cargo exposes `CARGO_BIN_EXE_*` only inside the binary's own package), and the runner's re-executed stdio child (selected by `FAKE_CHILD_MODE`).
- `openjiuwen-examples` has no test binaries: its examples are validated by compilation, while `src/support/` carries the run-mode, logging and banner unit tests.

## 7. Porting method

1. Read the Python module and its unit tests through a read-only investigator.
2. Preserve serialized names, default values, and observable behavior.
3. Replace Python-specific mechanisms (pydantic, asyncio globals, reflection) with Rust equivalents described above.
4. Port representative unit tests; document gaps in `docs/PARITY.md`.
5. Never modify the Python reference repository.
